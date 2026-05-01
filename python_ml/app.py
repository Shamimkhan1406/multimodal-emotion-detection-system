"""
EmoAware Python ML API.

Loads the 3 ONNX models once at startup and serves predictions:
  POST /predict/text   (form: text=...)
  POST /predict/face   (file: image)
  POST /predict/voice  (file: audio)
  POST /predict/all    (file: image, file: audio, form: text)
  GET  /health

Canonical label order across the API:
  ["joy", "sadness", "anger", "disgust", "fear", "surprise", "neutral"]

Face model was trained on FER2013 in the order
  ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
so its probs are remapped before fusion.
"""
import io
import os
import logging
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import librosa
from transformers import AutoTokenizer, AutoFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("emoaware")

ROOT = Path(__file__).resolve().parent.parent
FACE_MODEL_PATH = ROOT / "emotion_model.onnx"
VOICE_MODEL_PATH = ROOT / "VOICE_emotion_model.onnx"
TEXT_MODEL_PATH = ROOT / "text_emotion_model.onnx"

CANONICAL = ["joy", "sadness", "anger", "disgust", "fear", "surprise", "neutral"]
FACE_NATIVE = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
FACE_TO_CANONICAL = [FACE_NATIVE.index(l) for l in CANONICAL]

TARGET_SR = 16_000
MAX_AUDIO_SAMPLES = TARGET_SR * 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

TEXT_BASE = "SamLowe/roberta-base-go_emotions"
VOICE_BASE = "microsoft/wavlm-base-plus"

face_session: ort.InferenceSession | None = None
voice_session: ort.InferenceSession | None = None
text_session: ort.InferenceSession | None = None
text_tokenizer = None
voice_fe = None


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _probs_to_dict(p: np.ndarray) -> dict:
    return {CANONICAL[i]: float(p[i]) for i in range(len(CANONICAL))}


def _topk(p: np.ndarray):
    idx = int(np.argmax(p))
    return {"label": CANONICAL[idx], "label_id": idx, "confidence": float(p[idx])}


def load_models():
    global face_session, voice_session, text_session, text_tokenizer, voice_fe

    for p in (FACE_MODEL_PATH, VOICE_MODEL_PATH, TEXT_MODEL_PATH):
        if not p.exists():
            raise FileNotFoundError(f"Missing model: {p}")

    providers = ["CPUExecutionProvider"]
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    log.info("Loading FACE model …")
    face_session = ort.InferenceSession(str(FACE_MODEL_PATH), sess_opts, providers=providers)
    log.info("Loading VOICE model …")
    voice_session = ort.InferenceSession(str(VOICE_MODEL_PATH), sess_opts, providers=providers)
    log.info("Loading TEXT model …")
    text_session = ort.InferenceSession(str(TEXT_MODEL_PATH), sess_opts, providers=providers)

    log.info("Loading text tokenizer (%s) …", TEXT_BASE)
    text_tokenizer = AutoTokenizer.from_pretrained(TEXT_BASE)

    log.info("Loading voice feature extractor (%s) …", VOICE_BASE)
    voice_fe = AutoFeatureExtractor.from_pretrained(VOICE_BASE)

    log.info("All models loaded.")


def predict_face(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((48, 48), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)

    logits = face_session.run(None, {"input": arr})[0][0]
    probs_native = _softmax(logits)
    probs = probs_native[FACE_TO_CANONICAL]
    return {**_topk(probs), "probs": _probs_to_dict(probs)}


def _decode_audio_to_16k_mono(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes (WAV, OGG, FLAC, WebM/Opus, MP4/AAC, MP3) to
    a mono float32 array at 16 kHz.

    Tries libsndfile first (fast, handles WAV/FLAC/OGG). Falls back to a bundled
    ffmpeg via imageio-ffmpeg for browser-recorded WebM/Opus or other formats
    libsndfile can't read.
    """
    try:
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != TARGET_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        return wav.astype(np.float32)
    except Exception as e:
        log.info("libsndfile failed (%s); falling back to ffmpeg", type(e).__name__)

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", "pipe:0",
         "-f", "wav", "-acodec", "pcm_s16le",
         "-ar", str(TARGET_SR), "-ac", "1", "pipe:1"],
        input=audio_bytes, capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg decode failed: {proc.stderr.decode(errors='replace')[:500]}")
    wav, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32)


def predict_voice(audio_bytes: bytes) -> dict:
    wav = _decode_audio_to_16k_mono(audio_bytes)
    sr = TARGET_SR
    if wav.size == 0:
        wav = np.zeros(TARGET_SR // 2, dtype=np.float32)
    wav, _ = librosa.effects.trim(wav, top_db=30)
    if wav.size == 0:
        wav = np.zeros(TARGET_SR // 2, dtype=np.float32)
    if wav.size > MAX_AUDIO_SAMPLES:
        start = (wav.size - MAX_AUDIO_SAMPLES) // 2
        wav = wav[start:start + MAX_AUDIO_SAMPLES]
    else:
        wav = np.pad(wav, (0, MAX_AUDIO_SAMPLES - wav.size))

    feats = voice_fe(wav, sampling_rate=TARGET_SR, return_tensors="np")
    input_values = feats["input_values"].astype(np.float32)
    logits = voice_session.run(None, {"input_values": input_values})[0][0]
    probs = _softmax(logits)
    return {**_topk(probs), "probs": _probs_to_dict(probs)}


def predict_text(text: str) -> dict:
    if not text or not text.strip():
        # neutral fallback so fusion still has something to weight
        probs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        return {**_topk(probs), "probs": _probs_to_dict(probs)}

    enc = text_tokenizer(text, truncation=True, max_length=128, return_tensors="np")
    feeds = {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    }
    logits = text_session.run(None, feeds)[0][0]
    probs = _softmax(logits)
    return {**_topk(probs), "probs": _probs_to_dict(probs)}


def fuse(face: dict | None, voice: dict | None, text: dict | None,
         w_face: float = 1.0, w_voice: float = 1.0, w_text: float = 1.2) -> dict:
    """Weighted-average fusion of probability vectors."""
    vecs, weights = [], []
    if face is not None:
        vecs.append(np.array([face["probs"][l] for l in CANONICAL], dtype=np.float32))
        weights.append(w_face)
    if voice is not None:
        vecs.append(np.array([voice["probs"][l] for l in CANONICAL], dtype=np.float32))
        weights.append(w_voice)
    if text is not None:
        vecs.append(np.array([text["probs"][l] for l in CANONICAL], dtype=np.float32))
        weights.append(w_text)
    if not vecs:
        raise ValueError("No modalities to fuse")
    weights = np.array(weights, dtype=np.float32)
    weights /= weights.sum()
    fused = np.zeros(len(CANONICAL), dtype=np.float32)
    for v, w in zip(vecs, weights):
        fused += w * v
    fused /= fused.sum()  # renormalize
    return {**_topk(fused), "probs": _probs_to_dict(fused)}


app = FastAPI(title="EmoAware ML API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup():
    load_models()


@app.get("/health")
def health():
    return {
        "ok": True,
        "labels": CANONICAL,
        "models_loaded": all([face_session, voice_session, text_session]),
    }


@app.post("/predict/text")
async def api_text(text: str = Form(...)):
    try:
        return predict_text(text)
    except Exception as e:
        log.exception("text predict failed")
        raise HTTPException(500, str(e))


@app.post("/predict/face")
async def api_face(image: UploadFile = File(...)):
    try:
        return predict_face(await image.read())
    except Exception as e:
        log.exception("face predict failed")
        raise HTTPException(500, str(e))


@app.post("/predict/voice")
async def api_voice(audio: UploadFile = File(...)):
    try:
        return predict_voice(await audio.read())
    except Exception as e:
        log.exception("voice predict failed")
        raise HTTPException(500, str(e))


@app.post("/predict/all")
async def api_all(
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    text: str | None = Form(None),
):
    face_res = predict_face(await image.read()) if image is not None else None
    voice_res = predict_voice(await audio.read()) if audio is not None else None
    text_res = predict_text(text) if text else None

    if not any([face_res, voice_res, text_res]):
        raise HTTPException(400, "Provide at least one of image, audio, text")

    fused = fuse(face_res, voice_res, text_res)
    return {
        "fused": fused,
        "face": face_res,
        "voice": voice_res,
        "text": text_res,
        "labels": CANONICAL,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
