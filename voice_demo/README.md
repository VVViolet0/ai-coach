# Standalone Voice Module Demo

This is a local-only voice interaction prototype for testing the audio chain before connecting it to the AI Coach core.

Pipeline:

Browser microphone -> WebSocket PCM streaming -> Python VAD segmentation -> local faster-whisper ASR -> browser transcript display -> browser TTS repeat

## Install

From the repository root:

```powershell
python -m pip install -r voice_demo/requirements.txt
```

`faster-whisper` downloads the selected Whisper model the first time it runs. The ASR then runs locally.

If the machine cannot reach Hugging Face, download a CTranslate2 faster-whisper model on a network that works and copy it into this project, for example:

```text
voice_demo/models/faster-whisper-base/
```

Then start the server with the local model directory:

```powershell
python voice_demo/server.py --host 127.0.0.1 --port 8008 --model voice_demo/models/faster-whisper-base --device cpu --compute-type int8
```

The expected model is the CTranslate2/faster-whisper format, such as `Systran/faster-whisper-base`, not the original OpenAI Whisper PyTorch checkpoint.

## Run

```powershell
python voice_demo/server.py --host 127.0.0.1 --port 8008 --model base
```

Open:

```text
http://127.0.0.1:8008
```

Click `Start listening`, allow microphone access, and speak naturally. The browser sends 16 kHz mono PCM frames to the backend. The backend detects utterance boundaries and transcribes each speech segment locally.

## Useful Options

```powershell
python voice_demo/server.py --model tiny
python voice_demo/server.py --model small
python voice_demo/server.py --language zh
python voice_demo/server.py --device cpu --compute-type int8
```

Recommended first trial:

```powershell
python voice_demo/server.py --model base --device cpu --compute-type int8
```

Use `small` if accuracy matters more than latency.

To test the browser microphone and VAD chain without loading Whisper:

```powershell
python voice_demo/server.py --host 127.0.0.1 --port 8008 --asr-backend mock
```

To isolate faster-whisper model loading/transcription problems:

```powershell
python voice_demo/asr_smoke_test.py --model voice_demo/models/faster-whisper-base --device cpu --compute-type int8
```

## Run the Voice Coach

After the standalone voice chain works, run the browser voice version of the AI Coach:

```powershell
python voice_demo/coach_server.py --host 127.0.0.1 --port 8008 --model "voice_demo/models/faster-whisper-base" --device cpu --compute-type int8 --cpu-threads 1 --num-workers 1 --language zh
```

Open:

```text
http://127.0.0.1:8008
```

The first valid utterance is routed to intent modeling and workout planning. Later utterances are routed to the workout executor feedback loop.

The voice coach page uses a dashboard layout:

- Left side: current exercise, round/set, phase, countdown, tempo, and latest spoken cue.
- Right side: full workout plan and exercise instructions.
- Lower panels: recent voice feedback and real-time adjustments.

Only short, actionable coach prompts are spoken. Full plans, separators, JSON summaries, and log paths are shown or saved without TTS playback.

## VAD Sensitivity

The default voice detection is conservative:

- `--vad-mode 3`
- `--vad-start-trigger-ms 160`
- `--vad-min-speech-ms 600`
- `--vad-silence-ms 800`
- `--vad-energy-threshold 350`

If it still triggers too easily, raise the energy threshold:

```powershell
python voice_demo/coach_server.py --host 127.0.0.1 --port 8008 --model "voice_demo/models/faster-whisper-base" --device cpu --compute-type int8 --cpu-threads 1 --num-workers 1 --language zh --vad-energy-threshold 500 --vad-min-speech-ms 800
```

If it misses your speech, lower the threshold:

```powershell
python voice_demo/coach_server.py --host 127.0.0.1 --port 8008 --model "voice_demo/models/faster-whisper-base" --device cpu --compute-type int8 --cpu-threads 1 --num-workers 1 --language zh --vad-energy-threshold 250
```

## WebSocket Events

Client sends:

- Binary 16 kHz mono signed 16-bit PCM frames, 20 ms per frame.
- JSON control messages: `start`, `stop`, `mute`, `unmute`.

Server sends:

- `{"type": "vad_start"}`
- `{"type": "vad_end", "segment_ms": 1800}`
- `{"type": "transcript_final", "text": "...", "latency_ms": 1234}`
- `{"type": "metrics", "segment_ms": 1800, "asr_ms": 900}`
- `{"type": "error", "message": "..."}`

## Notes

- This module does not import or modify the existing AI Coach core.
- TTS uses browser `speechSynthesis`.
- The browser temporarily mutes upstream audio while TTS is speaking to reduce self-triggering.
- The default VAD settings use 300 ms pre-roll, 600 ms silence to end speech, and 10 seconds max segment length.
