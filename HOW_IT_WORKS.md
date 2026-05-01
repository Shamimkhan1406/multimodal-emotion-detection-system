# EmoAware — How it works (plain English)

This document walks through the whole system as if you'd never seen it before.
No jargon you don't need. Every step has a tiny example.

---

## The 30-second version

You take a selfie + record yourself talking. The app guesses your emotion
from **three signals at once** (your face, your voice, your words), then
combines those three guesses into one final answer.

```
   YOU                APP                    BACKEND                  MODELS
   ───                ───                    ───────                  ──────
  📷 photo  ──┐
  🎙️ voice  ──┼─► [Flutter Web] ─► [Node.js] ─► [Python] ─► 🧠 face
  🗣️ words  ──┘                                 │             🧠 voice
                                                │             🧠 text
                                                └─────────► combine 3 guesses
                                                              │
              😊 "joy: 78%"  ◄───────────────────────────────┘
```

---

## The three brains (ML models)

| Brain | What it eats | What it spits out |
|-------|--------------|-------------------|
| 🧠 **Face**  | a photo of your face                  | 7 numbers (one per emotion) |
| 🧠 **Voice** | a few seconds of audio                 | 7 numbers (one per emotion) |
| 🧠 **Text**  | the words you said (transcribed)       | 7 numbers (one per emotion) |

The 7 emotions, **always in the same order**, are:

```
[ joy , sadness , anger , disgust , fear , surprise , neutral ]
```

A "guess" looks like this — these are probabilities that add up to 1.0:

```
joy=0.78  sadness=0.05  anger=0.02  disgust=0.01  fear=0.04  surprise=0.08  neutral=0.02
```

The biggest number wins. Here joy = 78%, so the model says "joy".

---

## The three machines

```
┌──────────────────────────┐    HTTP    ┌────────────────┐    HTTP    ┌─────────────┐
│  Flutter Web App         │  ───────►  │  Node.js       │  ───────►  │  Python     │
│  (runs in your browser)  │            │  Express       │            │  FastAPI    │
│  port 5173               │            │  port 3000     │            │  port 8000  │
│                          │            │  (the doorman) │            │  (the brain)│
└──────────────────────────┘            └────────────────┘            └─────────────┘
   captures camera+mic                    forwards files               loads ONNX
   shows the result                       (does no ML)                  runs models
```

- **Flutter Web** = the screen the user sees. It uses the browser's camera
  and microphone.
- **Node.js Express** = a thin pipe in the middle. It just relays whatever
  the app sends to Python. (You could skip it; it exists because the
  architecture diagram in the requirements asked for it.)
- **Python FastAPI** = the only place that actually runs the ML models.

---

## STEP 1 — User opens the app

Browser fires up the Flutter app. The app immediately:

1. Asks the OS for camera + microphone permission.
2. Shows a **live front-camera preview** in a rectangle.
3. Boots the Web Speech API (the browser's built-in speech-to-text).

Nothing has been sent anywhere yet. Everything is local.

---

## STEP 2 — Take a photo 📷

You click **Capture**. The app grabs a single still frame from the camera
preview and stores it in memory as a JPEG.

```
   camera preview   ──click──►   JPEG bytes  (≈ 50 KB)
                                 ┌────────┐
                                 │FFD8FFE0│  ← these bytes live ONLY in
                                 │... ... │    your browser tab for now
                                 └────────┘
```

The image stays in browser memory. Nothing is uploaded yet.

---

## STEP 3 — Record your voice 🎙️

You tap the big **mic** button. The app starts **two things at the same
time**:

1. **A recorder** that captures audio from your microphone into a WebM/Opus
   blob (a compressed audio file format the browser uses by default).
2. **Speech-to-Text** — the browser listens to your voice and transcribes it
   live into words.

Tap the button again to stop. Now you have two artifacts:

```
   audio blob:        webm/opus, ~30 KB, length ≈ 5 seconds
   transcript text:   "I just got promoted, I can't believe it"
```

The transcript shows up in an editable text field, so you can fix typos.

---

## STEP 4 — Submit (what actually goes over the wire)

You click **Submit & analyze**. The app bundles everything into a single
HTTP request — the same kind of request a web form does when you upload a
file. It looks like this:

```
POST http://127.0.0.1:3000/api/predict
Content-Type: multipart/form-data; boundary=----abc

------abc
Content-Disposition: form-data; name="image"; filename="photo.jpg"
Content-Type: image/jpeg

<…JPEG bytes…>
------abc
Content-Disposition: form-data; name="audio"; filename="audio.webm"
Content-Type: audio/webm

<…WebM bytes…>
------abc
Content-Disposition: form-data; name="text"

I just got promoted, I can't believe it
------abc--
```

Three "fields" in one request: an image, an audio file, and a text string.

---

## STEP 5 — Node gets it and forwards it

The Node Express server on port 3000 has one job here: **don't lose any
bytes, hand them to Python**.

```js
// node_backend/server.js  (simplified)
app.post('/api/predict', upload.fields([{name:'image'},{name:'audio'}]), (req, res) => {
  const form = new FormData();
  form.append('text', req.body.text);
  form.append('image', req.files.image[0].buffer, 'photo.jpg');
  form.append('audio', req.files.audio[0].buffer, 'audio.webm');
  axios.post('http://127.0.0.1:8000/predict/all', form, {headers: form.getHeaders()})
       .then(r => res.json(r.data));
});
```

It re-packages the multipart form and POSTs it to the Python service. No ML
happens here.

---

## STEP 6 — Python preprocesses each modality

Python receives the same three things. It can't feed JPEG/WebM/strings to
the models directly — each model has a very specific input shape. So Python
does **pre-processing**: turning the raw user input into the exact tensor
the model expects.

### 6a. The photo → face tensor

```
JPEG bytes              ┌─ resize to 48 × 48 pixels
   ▼                    ├─ convert each pixel from 0..255 → 0..1
 PIL Image (RGB)        ├─ subtract ImageNet mean, divide by ImageNet std
   ▼                    └─ rearrange axes to (channels, height, width)
 numpy float32 tensor
   shape = (1, 3, 48, 48)
```

Tiny example, looking at one pixel:

```
raw pixel:      R=180, G=150, B=140
÷ 255:          0.706, 0.588, 0.549
- mean:        (0.706-0.485)=0.221,  (0.588-0.456)=0.132,  (0.549-0.406)=0.143
÷ std:          0.221/0.229=0.965,   0.132/0.224=0.589,    0.143/0.225=0.636
```

### 6b. The audio → voice tensor

```
WebM/Opus bytes
    │
    │ libsndfile can't read WebM, so Python falls back to a
    │ bundled ffmpeg binary that decodes it to plain WAV
    ▼
 raw waveform:  -1.0 .. +1.0  floats
    │
    ├─ resample to 16,000 samples per second  (the rate WavLM expects)
    ├─ mono-mix if stereo
    ├─ trim leading/trailing silence
    ├─ centre-crop or zero-pad to exactly 4 seconds = 64,000 samples
    └─ normalize: subtract mean, divide by std-dev
    ▼
 numpy float32 tensor
   shape = (1, 64000)
```

Tiny example: a 4-second audio clip is 64,000 numbers like
`[-0.0011, 0.0034, 0.0789, ..., -0.0021]`.

### 6c. The text → tokens tensor

```
"I just got promoted, I can't believe it"
    │
    │ feed through the RoBERTa tokenizer
    ▼
 token IDs:        [0, 100, 95, 122, 14914, 6, 38, 64, 75, 679, 24, 2]
 attention mask:   [1,   1,  1,   1,     1, 1,  1,  1,  1,   1, 1, 1]
    │
    ▼
 numpy int64 tensors
   input_ids       shape = (1, 12)
   attention_mask  shape = (1, 12)
```

Each word (or word-piece) becomes a number. The attention mask just says
"these positions are real, not padding".

---

## STEP 7 — Run each ONNX model

Now we feed the three tensors to the three models. ONNX Runtime is just a
library that runs `.onnx` model files efficiently on CPU.

```python
face_logits  = face_session.run(None,  {"input": face_tensor})[0]
voice_logits = voice_session.run(None, {"input_values": voice_tensor})[0]
text_logits  = text_session.run(None,  {"input_ids": ids,
                                        "attention_mask": mask})[0]
```

Each call returns 7 raw scores called **logits** — they can be any number
(positive or negative). To turn logits into probabilities we apply
**softmax**:

```
softmax(x_i) = e^x_i  /  Σ e^x_j     (so the result sums to 1)
```

Tiny worked example for the text model on "I just got promoted":

```
raw logits:        [ 4.2,  -1.1,  -0.8,  -0.4,  -0.6,  1.1,  -1.3 ]
                   ─ joy ─ sad ── ang ── dis ── fear ── sur ── neu
e^logits:          [66.7,   0.33,  0.45,  0.67,  0.55,  3.0,   0.27]
sum:                71.97
divide each:       [0.927, 0.005, 0.006, 0.009, 0.008, 0.042, 0.004]
                    ↑ joy = 92.7%
```

Now we have **three sets of probabilities**, one per model:

```
face  → { joy:0.16, sadness:0.11, anger:0.09, disgust:0.23, fear:0.10, surprise:0.18, neutral:0.13 }
voice → { joy:0.01, sadness:0.93, anger:0.01, disgust:0.01, fear:0.02, surprise:0.02, neutral:0.01 }
text  → { joy:0.93, sadness:0.01, anger:0.01, disgust:0.01, fear:0.01, surprise:0.04, neutral:0.01 }
```

### One quirky detail about the face model

It was trained on FER2013, which orders its labels differently:
`[angry, disgust, fear, happy, sad, surprise, neutral]`. Before fusion,
Python re-orders the face probabilities into the canonical order so all
three vectors line up.

---

## STEP 8 — Fuse the three guesses into one

We do a **weighted average** of the three probability vectors:

```
fused[i] = (w_face·face[i] + w_voice·voice[i] + w_text·text[i]) / (w_face+w_voice+w_text)
```

We use weights `face=1.0, voice=1.0, text=1.2` (text is upweighted slightly
because the text model was the most accurate during training). Then we
renormalize so the result still sums to 1.

Worked example for the **joy** slot:

```
joy = (1.0·0.16  +  1.0·0.01  +  1.2·0.93) / (1.0 + 1.0 + 1.2)
    = (0.16 + 0.01 + 1.116) / 3.2
    = 1.286 / 3.2
    = 0.402   →  40.2 %
```

Repeat for the other six slots; pick the biggest. Result:

```
fused → { joy:0.40, sadness:0.33, anger:0.03, disgust:0.08, fear:0.05, surprise:0.07, neutral:0.04 }
top   → joy  (40.2%)
```

Notice how fusion behaves: voice was screaming "sadness", but text was very
confident about joy and the face was uncertain. The combined answer leans
toward joy but isn't overconfident — sadness still gets 33%. That's the
whole point of multimodal fusion: each modality covers the others' blind
spots.

---

## STEP 9 — Send the answer back

Python returns one JSON document containing **everything**:

```json
{
  "fused": {
    "label": "joy",
    "confidence": 0.402,
    "probs": { "joy": 0.402, "sadness": 0.328, "anger": 0.034,
               "disgust": 0.081, "fear": 0.046, "surprise": 0.066, "neutral": 0.043 }
  },
  "face":  { "label": "disgust", "confidence": 0.230, "probs": { ... } },
  "voice": { "label": "sadness", "confidence": 0.932, "probs": { ... } },
  "text":  { "label": "joy",     "confidence": 0.923, "probs": { ... } },
  "labels": ["joy","sadness","anger","disgust","fear","surprise","neutral"]
}
```

Node passes this JSON through unchanged → Flutter receives it.

---

## STEP 10 — The UI shows the result

Flutter parses the JSON and renders two cards:

```
┌──────────────────────────────────┐  ┌──────────────────────────────────┐
│ 1. Take a photo  [📷 Capture]    │  │  Fused emotion                   │
│  ┌────────────────────────────┐  │  │  ┌────────────────────────────┐  │
│  │      [your selfie]         │  │  │  │  JOY            40.2 %     │  │
│  └────────────────────────────┘  │  │  │  joy      ▓▓▓▓▓▓▓░░░  40%  │  │
│                                  │  │  │  sadness  ▓▓▓▓▓░░░░░  33%  │  │
│ 2. Record voice  [🎙️]            │  │  │  anger    ░░░░░░░░░░   3%  │  │
│  ● recording…                    │  │  │  ...                       │  │
│                                  │  │  └────────────────────────────┘  │
│ 3. Transcript                    │  │                                  │
│  [I just got promoted...]        │  │  Per-modality breakdown          │
│                                  │  │  Face   → DISGUST    23%         │
│ [    Submit & analyze    ]       │  │  Voice  → SADNESS    93%         │
└──────────────────────────────────┘  │  Text   → JOY        92%         │
                                      └──────────────────────────────────┘
```

The big card on the right is the fused answer. The small cards show what
each individual model thought, so you can see where the disagreement was.

---

## End-to-end timeline (with timings on a laptop)

```
0 ms     : you click Submit
~10 ms   : multipart leaves the browser
~15 ms   : Node receives, repackages, forwards
~20 ms   : Python receives
~25 ms   : photo decoded + resized + normalised
~70 ms   : ffmpeg decodes WebM → WAV
~90 ms   : audio resampled + trimmed + padded
~95 ms   : tokenizer turns text into IDs
~150 ms  : face ONNX inference
~600 ms  : voice ONNX inference (this one is the heavy lifter)
~250 ms  : text ONNX inference
~  1 ms  : softmax + fusion + JSON serialization
~~~~~~~  : total ≈ 1 second on a typical laptop CPU
```

(GPU would shave off ~70% of that.)

---

## Mental model: one sentence per layer

- **Flutter Web** = "I collect raw signals (photo, audio, words) and draw the result."
- **Node Express** = "I'm a delivery service: I move bytes from A to B."
- **Python FastAPI** = "I'm where the AI lives."
- **ONNX models** = "Tell me what shape the input must be and I'll give you 7 scores."
- **Softmax** = "Turn raw scores into percentages that add up to 100."
- **Fusion** = "Three opinions are better than one."

---

## Why fusion beats any single model

A real failure mode of single-modality classifiers:

> Imagine someone saying "I love you" sarcastically while frowning.
> - 🧠 Text model alone → "joy" (the words say so)
> - 🧠 Face model alone → "anger" or "disgust" (the face says so)
> - 🧠 Voice model alone → "anger" (the tone says so)
> - 🧠 **Fused** → tilts toward anger, *but* keeps non-zero joy because the
>   words really do say "love". Closer to the truth: complicated, leaning negative.

That's why we combine all three.
