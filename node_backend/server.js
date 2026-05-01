// EmoAware Node/Express gateway.
// Receives multipart uploads from the Flutter web app, forwards to the Python
// ML API, and returns the fused emotion result.
import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import cors from 'cors';
import morgan from 'morgan';
import axios from 'axios';
import FormData from 'form-data';

const PORT = parseInt(process.env.PORT || '3000', 10);
const ML_API_URL = process.env.ML_API_URL || 'http://127.0.0.1:8000';

const app = express();
app.use(cors());
app.use(morgan('dev'));
app.use(express.json({ limit: '20mb' }));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

app.get('/health', async (_req, res) => {
  try {
    const ml = await axios.get(`${ML_API_URL}/health`, { timeout: 5000 });
    res.json({ ok: true, ml: ml.data });
  } catch (e) {
    res.status(503).json({ ok: false, error: 'ML API unreachable', detail: String(e.message) });
  }
});

const fields = upload.fields([
  { name: 'image', maxCount: 1 },
  { name: 'audio', maxCount: 1 },
]);

app.post('/api/predict', fields, async (req, res) => {
  try {
    const text = (req.body?.text || '').toString();
    const image = req.files?.image?.[0];
    const audio = req.files?.audio?.[0];

    if (!image && !audio && !text) {
      return res.status(400).json({ error: 'Provide at least one of image, audio, text' });
    }

    const form = new FormData();
    if (text) form.append('text', text);
    if (image) form.append('image', image.buffer, { filename: image.originalname || 'image.jpg', contentType: image.mimetype });
    if (audio) form.append('audio', audio.buffer, { filename: audio.originalname || 'audio.wav', contentType: audio.mimetype });

    const resp = await axios.post(`${ML_API_URL}/predict/all`, form, {
      headers: form.getHeaders(),
      timeout: 60_000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
    });
    res.json(resp.data);
  } catch (e) {
    const status = e.response?.status || 500;
    res.status(status).json({
      error: 'ML API call failed',
      detail: e.response?.data || e.message,
    });
  }
});

app.post('/api/predict/text', express.urlencoded({ extended: true }), async (req, res) => {
  try {
    const text = (req.body?.text || '').toString();
    if (!text) return res.status(400).json({ error: 'text required' });
    const form = new FormData();
    form.append('text', text);
    const resp = await axios.post(`${ML_API_URL}/predict/text`, form, {
      headers: form.getHeaders(),
      timeout: 30_000,
    });
    res.json(resp.data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
});

app.use((err, _req, res, _next) => {
  console.error(err);
  res.status(500).json({ error: err.message });
});

app.listen(PORT, () => {
  console.log(`[emoaware-backend] listening on http://0.0.0.0:${PORT}`);
  console.log(`[emoaware-backend] forwarding to ML API at ${ML_API_URL}`);
});
