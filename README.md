# EmoAware

Multimodal emotion classifier with three ONNX models (face / voice / text)
served behind a Node/Express gateway, with a Flutter Web client.

```
[Flutter Web App]                       :5173
     |  multipart (image, audio, text)
     v
[Node.js / Express]   /api/predict      :3000
     |  multipart relay
     v
[Python / FastAPI]    /predict/all      :8000
     |  ONNX Runtime
     v
[ Face | Voice | Text ]  ->  weighted-fusion  ->  fused emotion
```

## Models

| Modality | File | Input | Output |
|----------|------|-------|--------|
| Face  | `emotion_model.onnx`        | (B, 3, 48, 48) f32, ImageNet-norm RGB | (B, 7) logits |
| Voice | `VOICE_emotion_model.onnx`  | (B, T) f32 mono 16 kHz                | (B, 7) logits |
| Text  | `text_emotion_model.onnx`   | input_ids + attention_mask (i64)      | (B, 7) logits |

**Canonical label order** (used everywhere in the API):

```
["joy", "sadness", "anger", "disgust", "fear", "surprise", "neutral"]
```

The face model was trained on FER2013, whose native order is
`["angry","disgust","fear","happy","sad","surprise","neutral"]`. The Python API
remaps face probabilities to the canonical order before fusion.

The text tokenizer (`SamLowe/roberta-base-go_emotions`) and the voice feature
extractor (`microsoft/wavlm-base-plus`) are pulled from Hugging Face on first
launch and cached locally.

## Layout

```
emoaware/
├── emotion_model.onnx
├── VOICE_emotion_model.onnx
├── text_emotion_model.onnx
├── python_ml/
│   ├── app.py             # FastAPI app: /health, /predict/{text,face,voice,all}
│   ├── requirements.txt
│   └── start.sh
├── node_backend/
│   ├── server.js          # Express gateway
│   ├── package.json
│   └── .env.example
└── flutter_web_app/
    ├── lib/main.dart      # Camera + mic + STT + result UI
    ├── pubspec.yaml
    └── web/index.html
```

## Run it (3 terminals)

### 1. Python ML API
```bash
cd python_ml
./start.sh           # creates .venv, installs deps, runs uvicorn on :8000
# or manually:
# python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
# ./.venv/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

### 2. Node/Express gateway
```bash
cd node_backend
cp .env.example .env       # PORT=3000, ML_API_URL=http://127.0.0.1:8000
npm install
npm start                  # listening on :3000
```

### 3. Flutter Web app
```bash
cd flutter_web_app
flutter pub get
flutter run -d chrome --dart-define=BACKEND_URL=http://127.0.0.1:3000
# or release build served statically:
flutter build web --release --dart-define=BACKEND_URL=http://127.0.0.1:3000
cd build/web && python3 -m http.server 5173
```

Open http://127.0.0.1:5173.
Grant camera + mic permissions when the browser asks.

## How the UI flow works

1. App initialises the front-facing camera preview and the browser's Web Speech
   API (used by `speech_to_text`).
2. **Capture** freezes a still JPEG.
3. Tap the **mic** button: starts a `MediaRecorder` (WebM/Opus, 16 kHz mono)
   AND `SpeechRecognition` simultaneously. Tap again to stop both.
4. The transcript appears in an editable text field.
5. **Submit & analyze** posts a multipart form (`image`, `audio`, `text`) to
   `POST /api/predict` on the Node gateway, which relays to
   `POST /predict/all` on the Python service.
6. The response shows the **fused** prediction plus a per-modality breakdown
   with bar-chart probabilities.

## Audio decoding

Browser-recorded audio is WebM/Opus, which `libsndfile` cannot decode. The
Python service:

- Tries `soundfile` first (handles WAV / FLAC / OGG natively).
- Falls back to a bundled `imageio_ffmpeg` binary that decodes the bytes to
  PCM s16le mono 16 kHz on stdout.

Either path produces the float32 array the WavLM-based ONNX model expects.

## Fusion logic

`python_ml/app.py:fuse(...)` does a weighted-average of the per-modality
probability vectors (default weights: face 1.0, voice 1.0, text 1.2 — text is
slightly upweighted because the text model is the most accurate of the three).
The result is renormalised so the probabilities sum to 1.

## API reference

### Python (`http://127.0.0.1:8000`)
| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET  | `/health`         | — | service status |
| POST | `/predict/text`   | form `text`              | `{label, confidence, probs}` |
| POST | `/predict/face`   | file `image`             | `{label, confidence, probs}` |
| POST | `/predict/voice`  | file `audio`             | `{label, confidence, probs}` |
| POST | `/predict/all`    | any combination of above | `{fused, face, voice, text, labels}` |

### Node (`http://127.0.0.1:3000`)
| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET  | `/health`           | — | gateway + ML status |
| POST | `/api/predict`      | multipart (image?, audio?, text?) | full fused payload |
| POST | `/api/predict/text` | form `text` | text-only result |

## Quick curl tests

```bash
curl http://127.0.0.1:3000/health

curl -X POST -F "text=I am thrilled and overjoyed!" \
  http://127.0.0.1:3000/api/predict/text

curl -X POST \
  -F "image=@/path/to/photo.jpg" \
  -F "audio=@/path/to/audio.wav" \
  -F "text=I just got promoted!" \
  http://127.0.0.1:3000/api/predict
```
