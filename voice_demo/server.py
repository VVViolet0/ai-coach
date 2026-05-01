from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import webrtcvad
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing voice demo dependency. Install with: "
        "python -m pip install -r voice_demo/requirements.txt"
    ) from exc

try:
    from .segmenter import FRAME_BYTES, SAMPLE_RATE, SpeechSegment, VadSegmenter
except ImportError:  # pragma: no cover - supports `python voice_demo/server.py`
    from segmenter import FRAME_BYTES, SAMPLE_RATE, SpeechSegment, VadSegmenter


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
SHORT_CONFIRMATION_WORDS = {
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "no",
    "nope",
    "stop",
    "continue",
    "是",
    "对",
    "好",
    "行",
    "否",
    "不",
    "停",
    "停止",
    "继续",
    "繼續",
}
SHORT_ZH_ASR_CORRECTIONS = {
    "til": "停",
    "till": "停",
    "tin": "停",
    "tion": "停",
    "tio": "停",
}


class WhisperTranscriber:
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "auto",
        language: Optional[str] = None,
        cpu_threads: int = 1,
        num_workers: int = 1,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "faster-whisper is not installed. Run: "
                "python -m pip install -r voice_demo/requirements.txt"
            ) from exc

        self.language = language
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load the local faster-whisper model. "
                "If this is the first run, faster-whisper needs to download the model from Hugging Face. "
                "Your error looks like a network/proxy/model-cache problem. "
                "Either connect Hugging Face successfully, or download a CTranslate2 faster-whisper model "
                "and restart with --model pointing to the local model directory. "
                f"Current --model value: {model_size}. Original error: {exc}"
            ) from exc

    def _transcribe_wav(self, wav_path: Path, language: Optional[str]) -> Dict[str, Any]:
        start = time.perf_counter()
        segments, info = self.model.transcribe(
            str(wav_path),
            language=language,
            vad_filter=False,
            beam_size=1,
            temperature=0.0,
        )
        text = "".join(item.text for item in segments).strip()
        return {
            "text": text,
            "asr_ms": int((time.perf_counter() - start) * 1000),
            "language": getattr(info, "language", None),
            "language_probability": getattr(info, "language_probability", None),
        }

    def _should_use_short_utterance_fallback(self, primary: Dict[str, Any], fallback: Dict[str, Any]) -> bool:
        fallback_text = str(fallback.get("text", "")).strip()
        if not fallback_text:
            return False
        primary_text = str(primary.get("text", "")).strip()
        fallback_lower = fallback_text.lower().strip(" .,!?;:'\"")
        return not primary_text or fallback_lower in SHORT_CONFIRMATION_WORDS

    def _normalize_short_zh_command(self, result: Dict[str, Any]) -> Dict[str, Any]:
        text = str(result.get("text", "")).strip()
        normalized = text.lower().strip(" .,!?;:'\"")
        corrected = SHORT_ZH_ASR_CORRECTIONS.get(normalized)
        if corrected is None:
            return result
        patched = dict(result)
        patched["text"] = corrected
        patched["asr_correction"] = f"{text}->{corrected}"
        return patched

    def transcribe(self, segment: SpeechSegment) -> Dict[str, Any]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            with wave.open(str(tmp_path), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(segment.pcm)

            primary = self._transcribe_wav(tmp_path, self.language)
            if self.language == "zh" and segment.duration_ms <= 2200:
                fallback = self._transcribe_wav(tmp_path, None)
                if self._should_use_short_utterance_fallback(primary, fallback):
                    fallback["asr_ms"] += primary["asr_ms"]
                    fallback["fallback_from_language"] = self.language
                    return self._normalize_short_zh_command(fallback)
                return self._normalize_short_zh_command(primary)
            return primary
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass


class MockTranscriber:
    def transcribe(self, segment: SpeechSegment) -> Dict[str, Any]:
        return {
            "text": f"Mock transcript: detected {segment.duration_ms} ms of speech.",
            "asr_ms": 0,
            "language": "mock",
            "language_probability": 1.0,
        }


class AppState:
    def __init__(self) -> None:
        self.transcriber: Optional[WhisperTranscriber | MockTranscriber] = None
        self.asr_backend = "whisper"
        self.model_size = "base"
        self.device = "auto"
        self.compute_type = "auto"
        self.language: Optional[str] = None
        self.vad_mode = 3
        self.vad_silence_ms = 800
        self.vad_min_speech_ms = 180
        self.vad_start_trigger_ms = 100
        self.vad_energy_threshold = 350.0
        self.cpu_threads = 1
        self.num_workers = 1

    def get_transcriber(self) -> WhisperTranscriber | MockTranscriber:
        if self.transcriber is None:
            if self.asr_backend == "mock":
                self.transcriber = MockTranscriber()
            else:
                self.transcriber = WhisperTranscriber(
                    model_size=self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                    language=self.language,
                    cpu_threads=self.cpu_threads,
                    num_workers=self.num_workers,
                )
        return self.transcriber


state = AppState()
app = FastAPI(title="Standalone Voice Module Demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


async def send_json(websocket: WebSocket, payload: Dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(payload, ensure_ascii=False))


async def transcribe_and_send(websocket: WebSocket, segment: SpeechSegment) -> None:
    total_start = time.perf_counter()
    try:
        result = await asyncio.to_thread(state.get_transcriber().transcribe, segment)
        latency_ms = int((time.perf_counter() - total_start) * 1000)
        await send_json(
            websocket,
            {
                "type": "transcript_final",
                "text": result["text"],
                "latency_ms": latency_ms,
                "asr_ms": result["asr_ms"],
                "segment_ms": segment.duration_ms,
                "language": result["language"],
                "language_probability": result["language_probability"],
            },
        )
        await send_json(
            websocket,
            {
                "type": "metrics",
                "segment_ms": segment.duration_ms,
                "asr_ms": result["asr_ms"],
                "latency_ms": latency_ms,
            },
        )
    except Exception as exc:
        await send_json(websocket, {"type": "error", "message": str(exc)})


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    vad = webrtcvad.Vad(state.vad_mode)
    segmenter = VadSegmenter(
        vad=vad,
        silence_ms=state.vad_silence_ms,
        min_segment_ms=state.vad_min_speech_ms,
        start_trigger_ms=state.vad_start_trigger_ms,
        speech_energy_threshold=state.vad_energy_threshold,
    )
    muted = False
    pending_tasks: set[asyncio.Task] = set()

    await send_json(
        websocket,
        {
            "type": "state_update",
            "status": "connected",
            "sample_rate": SAMPLE_RATE,
            "frame_bytes": FRAME_BYTES,
        },
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            if "bytes" in message and message["bytes"] is not None:
                if muted:
                    continue
                frame = message["bytes"]
                if len(frame) != FRAME_BYTES:
                    await send_json(
                        websocket,
                        {
                            "type": "error",
                            "message": f"Invalid frame size: expected {FRAME_BYTES}, got {len(frame)}",
                        },
                    )
                    continue

                for event_type, segment in segmenter.process_frame(frame):
                    if event_type == "vad_start":
                        await send_json(websocket, {"type": "vad_start"})
                    elif event_type == "vad_end":
                        await send_json(
                            websocket,
                            {
                                "type": "vad_end",
                                "segment_ms": segment.duration_ms if segment is not None else 0,
                                "discarded": segment is None,
                            },
                        )
                        if segment is None:
                            continue
                        task = asyncio.create_task(transcribe_and_send(websocket, segment))
                        pending_tasks.add(task)
                        task.add_done_callback(pending_tasks.discard)

            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_json(websocket, {"type": "error", "message": "Invalid JSON control message"})
                    continue

                command = payload.get("type")
                if command == "mute":
                    muted = True
                    segmenter.flush()
                    await send_json(websocket, {"type": "state_update", "status": "muted"})
                elif command == "unmute":
                    muted = False
                    segmenter.reset()
                    await send_json(websocket, {"type": "state_update", "status": "listening"})
                elif command == "stop":
                    flushed = segmenter.flush()
                    if flushed is not None:
                        task = asyncio.create_task(transcribe_and_send(websocket, flushed))
                        pending_tasks.add(task)
                        task.add_done_callback(pending_tasks.discard)
                    await send_json(websocket, {"type": "state_update", "status": "stopped"})
                elif command == "start":
                    muted = False
                    segmenter.reset()
                    await send_json(websocket, {"type": "state_update", "status": "listening"})
                else:
                    await send_json(websocket, {"type": "error", "message": f"Unknown command: {command}"})
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        if 'Cannot call "receive" once a disconnect message has been received' not in str(exc):
            raise
    finally:
        for task in pending_tasks:
            task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone local voice module demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--model", default="base", help="faster-whisper model size, e.g. tiny, base, small")
    parser.add_argument("--device", default="auto", help="faster-whisper device, e.g. auto, cpu, cuda")
    parser.add_argument("--compute-type", default="auto", help="faster-whisper compute type")
    parser.add_argument("--language", default=None, help="Optional ASR language hint, e.g. zh or en")
    parser.add_argument("--vad-mode", type=int, default=3, choices=[0, 1, 2, 3], help="WebRTC VAD aggressiveness")
    parser.add_argument("--vad-silence-ms", type=int, default=800, help="Silence needed to end a segment")
    parser.add_argument("--vad-min-speech-ms", type=int, default=180, help="Minimum speech duration to keep a segment")
    parser.add_argument("--vad-start-trigger-ms", type=int, default=100, help="Consecutive speech needed to start")
    parser.add_argument("--vad-energy-threshold", type=float, default=350.0, help="Minimum RMS energy for speech frames")
    parser.add_argument("--cpu-threads", type=int, default=1, help="CPU threads used by faster-whisper")
    parser.add_argument("--num-workers", type=int, default=1, help="Worker count used by faster-whisper")
    parser.add_argument(
        "--asr-backend",
        choices=["whisper", "mock"],
        default="whisper",
        help="Use mock to test browser/VAD without loading faster-whisper.",
    )
    args = parser.parse_args()

    state.asr_backend = args.asr_backend
    state.model_size = str(Path(args.model).resolve()) if os.path.isdir(args.model) else args.model
    state.device = args.device
    state.compute_type = args.compute_type
    state.language = args.language
    state.vad_mode = args.vad_mode
    state.vad_silence_ms = args.vad_silence_ms
    state.vad_min_speech_ms = args.vad_min_speech_ms
    state.vad_start_trigger_ms = args.vad_start_trigger_ms
    state.vad_energy_threshold = args.vad_energy_threshold
    state.cpu_threads = args.cpu_threads
    state.num_workers = args.num_workers

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
