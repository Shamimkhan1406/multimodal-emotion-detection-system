import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:record/record.dart';
import 'package:speech_to_text/speech_to_text.dart' as stt;

const String kBackendUrl = String.fromEnvironment(
  'BACKEND_URL',
  defaultValue: 'http://127.0.0.1:3000',
);

void main() {
  runApp(const EmoApp());
}

class EmoApp extends StatelessWidget {
  const EmoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EmoAware',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6750A4)),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  CameraController? _camera;
  bool _cameraReady = false;

  final _recorder = AudioRecorder();
  bool _recording = false;

  final _stt = stt.SpeechToText();
  bool _sttAvailable = false;
  String _transcript = '';
  final _textCtrl = TextEditingController();

  Uint8List? _imageBytes;
  Uint8List? _audioBytes;
  String? _audioMime;

  bool _submitting = false;
  Map<String, dynamic>? _result;
  String? _error;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _initStt();
  }

  Future<void> _initCamera() async {
    try {
      final cams = await availableCameras();
      if (cams.isEmpty) {
        setState(() => _error = 'No camera found');
        return;
      }
      final front = cams.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cams.first,
      );
      final ctrl = CameraController(
        front,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await ctrl.initialize();
      if (!mounted) return;
      setState(() {
        _camera = ctrl;
        _cameraReady = true;
      });
    } catch (e) {
      setState(() => _error = 'Camera init failed: $e');
    }
  }

  Future<void> _initStt() async {
    final ok = await _stt.initialize(onError: (e) => debugPrint('STT err: $e'));
    setState(() => _sttAvailable = ok);
  }

  Future<void> _capturePhoto() async {
    if (_camera == null || !_camera!.value.isInitialized) return;
    try {
      final shot = await _camera!.takePicture();
      final bytes = await shot.readAsBytes();
      setState(() {
        _imageBytes = bytes;
      });
    } catch (e) {
      setState(() => _error = 'Capture failed: $e');
    }
  }

  Future<void> _toggleRecord() async {
    if (_recording) {
      await _stopRecording();
    } else {
      await _startRecording();
    }
  }

  Future<void> _startRecording() async {
    final ok = await _recorder.hasPermission();
    if (!ok) {
      setState(() => _error = 'Microphone permission denied');
      return;
    }

    setState(() {
      _audioBytes = null;
      _transcript = '';
      _recording = true;
      _error = null;
    });

    if (_sttAvailable) {
      await _stt.listen(
        onResult: (r) => setState(() => _transcript = r.recognizedWords),
        listenFor: const Duration(seconds: 30),
        pauseFor: const Duration(seconds: 4),
        listenOptions: stt.SpeechListenOptions(partialResults: true),
      );
    }

    final config = const RecordConfig(
      encoder: AudioEncoder.opus,
      numChannels: 1,
      sampleRate: 16000,
    );
    await _recorder.start(config, path: 'audio.webm');
  }

  Future<void> _stopRecording() async {
    setState(() => _recording = false);

    try {
      if (_sttAvailable && _stt.isListening) await _stt.stop();
    } catch (_) {}

    final path = await _recorder.stop();
    if (path == null) return;

    Uint8List? bytes;
    String mime = 'audio/webm';
    if (kIsWeb) {
      try {
        final r = await http.get(Uri.parse(path));
        bytes = r.bodyBytes;
        if (r.headers['content-type'] != null) {
          mime = r.headers['content-type']!.split(';').first;
        }
      } catch (e) {
        setState(() => _error = 'Audio fetch failed: $e');
        return;
      }
    }

    setState(() {
      _audioBytes = bytes;
      _audioMime = mime;
      if (_textCtrl.text.isEmpty) _textCtrl.text = _transcript;
    });
  }

  Future<void> _submit() async {
    final text = _textCtrl.text.trim().isEmpty ? _transcript : _textCtrl.text.trim();
    if (_imageBytes == null && _audioBytes == null && text.isEmpty) {
      setState(() => _error = 'Capture a photo, record audio, or enter text first');
      return;
    }

    setState(() {
      _submitting = true;
      _error = null;
      _result = null;
    });

    try {
      final uri = Uri.parse('$kBackendUrl/api/predict');
      final req = http.MultipartRequest('POST', uri);
      if (text.isNotEmpty) req.fields['text'] = text;
      if (_imageBytes != null) {
        req.files.add(http.MultipartFile.fromBytes(
          'image', _imageBytes!,
          filename: 'photo.jpg',
        ));
      }
      if (_audioBytes != null) {
        final m = _audioMime ?? 'audio/webm';
        final ext = m.contains('webm') ? 'webm'
                  : m.contains('wav') ? 'wav'
                  : m.contains('ogg') ? 'ogg'
                  : m.contains('mp4') ? 'm4a'
                  : 'bin';
        req.files.add(http.MultipartFile.fromBytes(
          'audio', _audioBytes!,
          filename: 'audio.$ext',
        ));
      }

      final streamed = await req.send().timeout(const Duration(seconds: 90));
      final resp = await http.Response.fromStream(streamed);
      if (resp.statusCode != 200) {
        throw Exception('${resp.statusCode}: ${resp.body}');
      }
      setState(() => _result = json.decode(resp.body) as Map<String, dynamic>);
    } catch (e) {
      setState(() => _error = 'Submit failed: $e');
    } finally {
      if (mounted) setState(() => _submitting = false);
    }
  }

  void _reset() {
    setState(() {
      _imageBytes = null;
      _audioBytes = null;
      _audioMime = null;
      _transcript = '';
      _textCtrl.clear();
      _result = null;
      _error = null;
    });
  }

  @override
  void dispose() {
    _camera?.dispose();
    _recorder.dispose();
    _textCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        title: const Text('EmoAware'),
        backgroundColor: cs.primaryContainer,
      ),
      body: LayoutBuilder(builder: (ctx, c) {
        final wide = c.maxWidth > 900;
        final left = _buildCaptureColumn();
        final right = _buildResultColumn();
        return SingleChildScrollView(
          padding: const EdgeInsets.all(16),
          child: wide
              ? Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(child: left),
                    const SizedBox(width: 16),
                    Expanded(child: right),
                  ],
                )
              : Column(children: [left, const SizedBox(height: 16), right]),
        );
      }),
    );
  }

  Widget _buildCaptureColumn() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('1. Take a photo',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            AspectRatio(
              aspectRatio: 4 / 3,
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: _imageBytes != null
                    ? Image.memory(_imageBytes!, fit: BoxFit.cover)
                    : (_cameraReady && _camera != null
                        ? CameraPreview(_camera!)
                        : Container(
                            color: Colors.black12,
                            child: const Center(child: Text('Camera loading…')),
                          )),
              ),
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: FilledButton.icon(
                    onPressed: _cameraReady ? _capturePhoto : null,
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Capture'),
                  ),
                ),
                const SizedBox(width: 8),
                if (_imageBytes != null)
                  OutlinedButton.icon(
                    onPressed: () => setState(() => _imageBytes = null),
                    icon: const Icon(Icons.refresh),
                    label: const Text('Retake'),
                  ),
              ],
            ),
            const SizedBox(height: 24),
            const Text('2. Record your voice',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            Center(
              child: GestureDetector(
                onTap: _toggleRecord,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 200),
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _recording
                        ? Colors.redAccent
                        : Theme.of(context).colorScheme.primary,
                    boxShadow: const [BoxShadow(blurRadius: 6, color: Colors.black26)],
                  ),
                  child: Icon(
                    _recording ? Icons.stop : Icons.mic,
                    color: Colors.white, size: 40,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 8),
            Center(
              child: Text(
                _recording
                    ? 'Recording… tap to stop'
                    : (_audioBytes == null ? 'Tap to record' : 'Audio captured ✓'),
                style: TextStyle(color: Colors.grey.shade700),
              ),
            ),
            const SizedBox(height: 24),
            const Text('3. Speech transcript (you can edit)',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            TextField(
              controller: _textCtrl,
              maxLines: 3,
              decoration: InputDecoration(
                border: const OutlineInputBorder(),
                hintText: _transcript.isEmpty
                    ? 'What you said will appear here…'
                    : _transcript,
              ),
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _submitting ? null : _submit,
              icon: _submitting
                  ? const SizedBox(
                      width: 18, height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.send),
              label: Text(_submitting ? 'Analyzing…' : 'Submit & analyze'),
            ),
            const SizedBox(height: 8),
            TextButton.icon(
              onPressed: _reset,
              icon: const Icon(Icons.restart_alt),
              label: const Text('Reset'),
            ),
            if (_error != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Text(_error!, style: TextStyle(color: Colors.red.shade900)),
              ),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildResultColumn() {
    if (_result == null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            children: [
              const Icon(Icons.psychology_alt_outlined, size: 64, color: Colors.grey),
              const SizedBox(height: 12),
              Text('Results will appear here',
                  style: TextStyle(color: Colors.grey.shade700, fontSize: 16)),
              const SizedBox(height: 8),
              Text('Backend: $kBackendUrl',
                  style: TextStyle(color: Colors.grey.shade500, fontSize: 12)),
            ],
          ),
        ),
      );
    }

    final fused = (_result!['fused'] as Map?)?.cast<String, dynamic>();
    final face = (_result!['face'] as Map?)?.cast<String, dynamic>();
    final voice = (_result!['voice'] as Map?)?.cast<String, dynamic>();
    final text = (_result!['text'] as Map?)?.cast<String, dynamic>();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text('Fused emotion',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            if (fused != null) _buildResultBlock(fused, highlight: true),
            const SizedBox(height: 24),
            const Text('Per-modality breakdown',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            if (face != null) _modalityCard('Face (image)', Icons.face, face),
            if (voice != null) _modalityCard('Voice (audio)', Icons.graphic_eq, voice),
            if (text != null) _modalityCard('Text (transcript)', Icons.chat_bubble_outline, text),
          ],
        ),
      ),
    );
  }

  Widget _modalityCard(String title, IconData icon, Map<String, dynamic> r) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(children: [Icon(icon, size: 18), const SizedBox(width: 6),
            Text(title, style: const TextStyle(fontWeight: FontWeight.w600))]),
          const SizedBox(height: 4),
          _buildResultBlock(r),
        ],
      ),
    );
  }

  Widget _buildResultBlock(Map<String, dynamic> r, {bool highlight = false}) {
    final label = r['label']?.toString() ?? '?';
    final conf = (r['confidence'] as num?)?.toDouble() ?? 0;
    final probs = (r['probs'] as Map?)?.cast<String, dynamic>() ?? {};
    const order = ['joy', 'sadness', 'anger', 'disgust', 'fear', 'surprise', 'neutral'];

    final accent = _emotionColor(label);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: highlight ? accent.withOpacity(0.1) : Colors.grey.shade50,
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: highlight ? accent : Colors.grey.shade300),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                decoration: BoxDecoration(
                  color: accent,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Text(
                  label.toUpperCase(),
                  style: const TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
                ),
              ),
              const Spacer(),
              Text('${(conf * 100).toStringAsFixed(1)}%',
                  style: TextStyle(fontWeight: FontWeight.bold,
                      fontSize: highlight ? 20 : 14)),
            ],
          ),
          const SizedBox(height: 8),
          ...order.map((k) {
            final p = (probs[k] as num?)?.toDouble() ?? 0;
            return Padding(
              padding: const EdgeInsets.symmetric(vertical: 2),
              child: Row(
                children: [
                  SizedBox(width: 70, child: Text(k, style: const TextStyle(fontSize: 12))),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(3),
                      child: LinearProgressIndicator(
                        value: p,
                        minHeight: 8,
                        backgroundColor: Colors.grey.shade200,
                        valueColor: AlwaysStoppedAnimation(_emotionColor(k)),
                      ),
                    ),
                  ),
                  const SizedBox(width: 6),
                  SizedBox(width: 42,
                      child: Text('${(p * 100).toStringAsFixed(1)}%',
                          style: const TextStyle(fontSize: 12),
                          textAlign: TextAlign.right)),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Color _emotionColor(String l) {
    switch (l) {
      case 'joy':       return const Color(0xFFFFB300);
      case 'sadness':   return const Color(0xFF1E88E5);
      case 'anger':     return const Color(0xFFE53935);
      case 'disgust':   return const Color(0xFF6A1B9A);
      case 'fear':      return const Color(0xFF8E24AA);
      case 'surprise':  return const Color(0xFF00ACC1);
      case 'neutral':   return const Color(0xFF757575);
      default:          return Colors.grey;
    }
  }
}
