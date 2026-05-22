from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

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
    from .server import MockTranscriber, WhisperTranscriber
except ImportError:  # pragma: no cover - supports `python voice_demo/coach_server.py`
    from segmenter import FRAME_BYTES, SAMPLE_RATE, SpeechSegment, VadSegmenter
    from server import MockTranscriber, WhisperTranscriber

from ai_coach_system import AICoachSystem


Publisher = Callable[[Dict[str, Any]], None]
INITIAL_PLANNING_SPEECH = "收到您的训练需求，正在为您生成训练计划。与此同时您可以进行一些热身。"
CHINESE_ORDER_MARKERS = ("一", "二", "三", "四", "五", "六", "七", "八", "九", "十")
FEEDBACK_KEYWORDS = {
    "休息",
    "太长",
    "太短",
    "短一点",
    "长一点",
    "快",
    "慢",
    "累",
    "疲劳",
    "疼",
    "痛",
    "不舒服",
    "难",
    "简单",
    "轻松",
    "动作",
    "换",
    "跳过",
    "不想",
    "停止",
    "结束",
    "继续",
    "stop",
    "pain",
    "hurt",
    "tired",
    "rest",
    "faster",
    "slower",
    "hard",
    "easy",
    "skip",
    "感觉很好",
    "很好",
    "状态很好",
    "可以多练",
    "多练",
    "加练",
    "加组",
    "多一组",
    "再来",
    "再练",
    "more",
    "longer",
}
SAFETY_KEYWORDS = {"疼", "痛", "不舒服", "停止", "结束", "停", "stop", "pain", "hurt", "quit"}
COMMON_ASR_HALLUCINATIONS = {
    "谢谢观看",
    "感谢观看",
    "欢迎收看",
    "我认为你会不会有什么事",
}


def _coach_message(
    *,
    display: bool,
    speak: bool,
    message: str,
    speech_text: str = "",
    category: str,
) -> Dict[str, Any]:
    return {
        "display": display,
        "speak": speak,
        "message": message,
        "speech_text": speech_text,
        "category": category,
    }


def _initial_planning_message() -> Dict[str, Any]:
    return {
        "type": "coach_message",
        **_coach_message(
            display=True,
            speak=True,
            message=INITIAL_PLANNING_SPEECH,
            speech_text=INITIAL_PLANNING_SPEECH,
            category="initial_planning",
        ),
    }


def _strip_coach_prefix(message: str) -> str:
    return message.strip().replace("[Coach]", "").replace("[Adjustment]", "").strip()


def _compact_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip().lower())


def _has_safety_keyword(text: str) -> bool:
    normalized = _compact_text(text)
    return any(keyword in normalized for keyword in SAFETY_KEYWORDS)


def _is_repetitive_asr_text(text: str) -> bool:
    normalized = _compact_text(text)
    if len(normalized) < 10:
        return False
    if len(set(normalized)) / max(1, len(normalized)) < 0.28:
        return True
    for width in range(2, min(9, len(normalized) // 2 + 1)):
        chunks = [normalized[i : i + width] for i in range(0, len(normalized) - width + 1, width)]
        if len(chunks) >= 3 and max(chunks.count(chunk) for chunk in set(chunks)) >= 3:
            return True
    return False


def classify_feedback_candidate(text: str) -> tuple[bool, str]:
    normalized = _compact_text(text)
    if not normalized:
        return False, "empty"
    if _has_safety_keyword(normalized):
        return True, "safety_keyword"
    if any(phrase in normalized for phrase in COMMON_ASR_HALLUCINATIONS):
        return False, "common_asr_hallucination"
    if _is_repetitive_asr_text(normalized):
        return False, "repetitive_asr"
    if any(keyword in normalized for keyword in FEEDBACK_KEYWORDS):
        return True, "feedback_keyword"
    if len(normalized) <= 2:
        return False, "too_short"
    return False, "no_feedback_keyword"


def _describe_adjustment(set_delta: int, rest_multiplier: float) -> str:
    parts: list[str] = []
    if set_delta < 0:
        parts.append(f"减少 {abs(set_delta)} 组动作")
    elif set_delta > 0:
        parts.append(f"增加 {set_delta} 组动作")

    if rest_multiplier > 1.02:
        parts.append("延长休息时间")
    elif rest_multiplier < 0.98:
        parts.append("缩短休息时间")

    if not parts:
        parts.append("训练安排已更新")
    return "，".join(parts)


def _format_instruction_speech(raw: str) -> tuple[str, str]:
    exercise_match = re.search(r"Exercise:\s*(.*?)(?:\s+Instructions:|$)", raw, re.IGNORECASE | re.DOTALL)
    instructions_match = re.search(r"Instructions:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
    exercise_name = exercise_match.group(1).strip() if exercise_match else ""
    instruction_text = instructions_match.group(1).strip() if instructions_match else raw
    steps = [step.strip(" -;；。") for step in re.split(r"\s*[;；]\s*", instruction_text) if step.strip(" -;；。")]

    if steps:
        ordered_steps = []
        for index, step in enumerate(steps):
            marker = CHINESE_ORDER_MARKERS[index] if index < len(CHINESE_ORDER_MARKERS) else str(index + 1)
            ordered_steps.append(f"{marker}、{step}")
        speech_text = "动作要求：" + "；".join(ordered_steps) + "。"
    else:
        speech_text = "请保持安全、受控的动作节奏。"

    if exercise_name:
        speech_text = f"接下来是{exercise_name}。" + speech_text

    return exercise_name, speech_text


def classify_coach_message(message: str) -> Dict[str, Any]:
    raw = message.strip()
    compact = " ".join(raw.split())
    if not compact:
        return _coach_message(display=False, speak=False, message=raw, category="empty")

    if set(compact) <= {"-"} or compact.startswith("====="):
        return _coach_message(display=False, speak=False, message=raw, category="separator")

    if compact.startswith("Session log saved"):
        return _coach_message(display=False, speak=False, message=raw, category="log_saved")

    if "训练计划已生成" in compact and "\n" in raw:
        return _coach_message(
            display=False,
            speak=True,
            message=raw,
            speech_text="训练计划已生成，准备开始。",
            category="plan_ready",
        )

    if "正在为你生成训练计划" in compact:
        return _coach_message(
            display=True,
            speak=False,
            message=_strip_coach_prefix(compact),
            category="planning",
        )

    set_match = re.match(r"Set\s+(\d+)/(\d+)\s+start", compact, re.IGNORECASE)
    if set_match:
        current, total = set_match.groups()
        return _coach_message(
            display=False,
            speak=True,
            message=compact,
            speech_text=f"第 {current} 组开始，共 {total} 组。",
            category="set_start",
        )

    rest_match = re.match(r"Rest for\s+(\d+)\s+seconds", compact, re.IGNORECASE)
    if rest_match:
        seconds = rest_match.group(1)
        return _coach_message(
            display=False,
            speak=True,
            message=compact,
            speech_text=f"休息 {seconds} 秒。",
            category="rest_start",
        )

    if compact == "Set finished.":
        return _coach_message(display=False, speak=True, message=compact, speech_text="本组完成。", category="set_end")

    if compact == "Round finished.":
        return _coach_message(display=False, speak=True, message=compact, speech_text="本轮完成。", category="round_end")

    round_match = re.match(r"Round\s+(\d+)/(\d+)\s+start", compact, re.IGNORECASE)
    if round_match:
        current, total = round_match.groups()
        return _coach_message(
            display=False,
            speak=True,
            message=compact,
            speech_text=f"第 {current} 轮开始，共 {total} 轮。",
            category="round_start",
        )

    if compact.startswith("Rest between rounds for"):
        seconds = re.findall(r"\d+", compact)
        speech = f"轮间休息 {seconds[0]} 秒。" if seconds else "轮间休息。"
        return _coach_message(display=False, speak=True, message=compact, speech_text=speech, category="rest_start")

    if "WORKOUT COMPLETE" in compact.upper():
        return _coach_message(display=True, speak=True, message="训练完成。", speech_text="训练完成。", category="session_end")

    if "WORKOUT STOPPED" in compact.upper():
        return _coach_message(display=True, speak=True, message=compact, speech_text="训练已停止。", category="session_end")

    if compact.startswith("Exercise:") and "Instructions:" in compact:
        exercise_name, speech_text = _format_instruction_speech(raw)
        return _coach_message(
            display=True,
            speak=True,
            message=exercise_name or compact,
            speech_text=speech_text,
            category="exercise_instructions",
        )

    if compact.startswith("Exercise:") or compact.startswith("demo_speed:") or compact.startswith("Instructions:") or compact.startswith("- "):
        return _coach_message(display=False, speak=False, message=compact, category="exercise_detail")

    if compact.startswith("[Adjustment]"):
        return _coach_message(display=True, speak=False, message=compact, category="adjustment")

    cleaned = _strip_coach_prefix(compact)
    return _coach_message(display=True, speak=True, message=cleaned, speech_text=cleaned, category="coach_reply")


class VoiceCoachIO:
    def __init__(self, publish: Publisher) -> None:
        self._publish = publish
        self._messages: "queue.Queue[str]" = queue.Queue()
        self._closed = False
        self._speech_lock = threading.Lock()
        self._speech_counter = 0
        self._speech_events: Dict[str, threading.Event] = {}
        self._last_speech_id_by_category: Dict[str, str] = {}
        self.nonblocking_feedback = True

    def enqueue_user_text(self, text: str, *, safety: bool = False) -> None:
        cleaned = text.strip()
        if not cleaned or self._closed:
            return
        if not safety:
            self._drop_pending_normal_feedback()
        self._messages.put(cleaned)

    def _drop_pending_normal_feedback(self) -> None:
        kept = []
        try:
            while True:
                message = self._messages.get_nowait()
                if _has_safety_keyword(message):
                    kept.append(message)
        except queue.Empty:
            pass
        for message in kept:
            self._messages.put(message)

    def send(self, message: str) -> None:
        if self._closed:
            return
        classified = classify_coach_message(message)
        payload = {"type": "coach_message", **classified}
        if classified.get("speak"):
            speech_id = self._register_speech(str(classified.get("category", "")))
            payload["speech_id"] = speech_id
        self._publish(payload)

    def _register_speech(self, category: str) -> str:
        with self._speech_lock:
            self._speech_counter += 1
            speech_id = f"speech-{self._speech_counter}"
            self._speech_events[speech_id] = threading.Event()
            if category:
                self._last_speech_id_by_category[category] = speech_id
            return speech_id

    def mark_speech_done(self, speech_id: str) -> None:
        with self._speech_lock:
            event = self._speech_events.get(speech_id)
        if event:
            event.set()

    def wait_for_speech(self, category: str, timeout: float = 30.0) -> bool:
        with self._speech_lock:
            speech_id = self._last_speech_id_by_category.get(category)
            event = self._speech_events.get(speech_id) if speech_id else None
        if event is None:
            return True
        completed = event.wait(timeout=timeout)
        with self._speech_lock:
            self._speech_events.pop(speech_id, None)
            if self._last_speech_id_by_category.get(category) == speech_id:
                self._last_speech_id_by_category.pop(category, None)
        return completed

    def send_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self._closed:
            return
        self._publish({"type": "runtime_event", "event_type": event_type, "payload": payload})

    def poll_user_input(self) -> Optional[str]:
        if self._closed:
            return None
        try:
            return self._messages.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True


class CoachSession:
    def __init__(self, websocket: WebSocket, config: "ServerConfig") -> None:
        self.websocket = websocket
        self.config = config
        self.loop = asyncio.get_running_loop()
        self.state = "idle"
        self.io = VoiceCoachIO(self.publish_from_thread)
        self.coach_thread: Optional[threading.Thread] = None
        self.transcriber: Optional[WhisperTranscriber | MockTranscriber] = None
        self._closed = False
        self._last_feedback_text = ""
        self._last_feedback_at = 0.0

    async def publish(self, payload: Dict[str, Any]) -> None:
        await self.websocket.send_text(json.dumps(payload, ensure_ascii=False))

    def publish_from_thread(self, payload: Dict[str, Any]) -> None:
        if self._closed:
            return
        if payload.get("type") == "runtime_event":
            event_type = payload.get("event_type")
            if self.state == "planning" and event_type in {"workout_started", "round_start", "exercise_start", "set_start"}:
                self.set_state_from_thread("workout_running")
        asyncio.run_coroutine_threadsafe(self.publish(payload), self.loop)

    async def set_state(self, state: str) -> None:
        self.state = state
        await self.publish({"type": "session_state", "state": state})

    def set_state_from_thread(self, state: str) -> None:
        self.state = state
        self.publish_from_thread({"type": "session_state", "state": state})

    def get_transcriber(self) -> WhisperTranscriber | MockTranscriber:
        if self.transcriber is None:
            if self.config.asr_backend == "mock":
                self.transcriber = MockTranscriber()
            else:
                self.transcriber = WhisperTranscriber(
                    model_size=self.config.resolved_asr_model(),
                    device=self.config.device,
                    compute_type=self.config.compute_type,
                    language=self.config.language,
                    cpu_threads=self.config.cpu_threads,
                    num_workers=self.config.num_workers,
                )
        return self.transcriber

    async def transcribe_segment(self, segment: SpeechSegment) -> None:
        total_start = time.perf_counter()
        try:
            result = await asyncio.to_thread(self.get_transcriber().transcribe, segment)
            latency_ms = int((time.perf_counter() - total_start) * 1000)
            text = result["text"].strip()
            await self.publish(
                {
                    "type": "transcript_final",
                    "text": text,
                    "latency_ms": latency_ms,
                    "asr_ms": result["asr_ms"],
                    "segment_ms": segment.duration_ms,
                    "language": result["language"],
                    "language_probability": result["language_probability"],
                }
            )
            await self.publish(
                {
                    "type": "metrics",
                    "segment_ms": segment.duration_ms,
                    "asr_ms": result["asr_ms"],
                    "latency_ms": latency_ms,
                }
            )
            if text:
                await self.route_transcript(text)
        except Exception as exc:
            await self.set_state("error")
            await self.publish({"type": "error", "message": str(exc)})

    async def route_transcript(self, text: str) -> None:
        if self.state == "idle":
            await self.publish({"type": "user_transcript", "text": text, "target": "initial_intent"})
            await self.publish(_initial_planning_message())
            await self.set_state("planning")
            self.start_coach(text)
            return

        if self.state in {"planning", "workout_running"}:
            accepted, reason = classify_feedback_candidate(text)
            normalized = _compact_text(text)
            now = time.monotonic()
            if accepted and normalized == self._last_feedback_text and now - self._last_feedback_at < 3.0:
                accepted = False
                reason = "duplicate_recent"
            if accepted:
                safety = reason == "safety_keyword"
                self.io.enqueue_user_text(text, safety=safety)
                self._last_feedback_text = normalized
                self._last_feedback_at = now
                await self.publish({"type": "user_transcript", "text": text, "target": "feedback"})
            else:
                target = "ignored_noise" if reason in {"common_asr_hallucination", "repetitive_asr"} else "ignored_feedback"
                await self.publish({"type": "user_transcript", "text": text, "target": target, "reason": reason})
            return

        await self.publish({"type": "user_transcript", "text": text, "target": "ignored"})

    def start_coach(self, initial_request: str) -> None:
        if self.coach_thread and self.coach_thread.is_alive():
            return
        self.coach_thread = threading.Thread(
            target=self._run_coach_session,
            args=(initial_request,),
            daemon=True,
        )
        self.coach_thread.start()

    def _run_coach_session(self, initial_request: str) -> None:
        try:
            system = AICoachSystem(
                exercise_library_path=self.config.exercise_library,
                output_dir=self.config.output_dir,
                intent_model=self.config.intent_model,
                planner_model=self.config.planner_model,
            )
            result = system.run_session(
                initial_request,
                io=self.io,
                feedback_model=self.config.feedback_model,
                execute_workout=True,
            )
            self.set_state_from_thread("ended")
            self.publish_from_thread({"type": "session_summary", "result": result})
        except Exception as exc:
            self.set_state_from_thread("error")
            self.publish_from_thread({"type": "error", "message": str(exc)})
        finally:
            self.io.close()

    async def close(self) -> None:
        self._closed = True
        self.io.close()


class ServerConfig:
    def __init__(self) -> None:
        self.asr_model = "base"
        self.device = "auto"
        self.compute_type = "auto"
        self.language: Optional[str] = "zh"
        self.cpu_threads = 1
        self.num_workers = 1
        self.asr_backend = "whisper"
        self.exercise_library = "libraries/exercise_library.json"
        self.output_dir = "data"
        self.intent_model = "frob/qwen3.5-instruct:4b"
        self.planner_model = "frob/qwen3.5-instruct:4b"
        self.feedback_model = "frob/qwen3.5-instruct:4b"
        self.vad_mode = 3
        self.vad_silence_ms = 800
        self.vad_min_speech_ms = 500
        self.vad_start_trigger_ms = 220
        self.vad_energy_threshold = 600.0

    def resolved_asr_model(self) -> str:
        return str(Path(self.asr_model).resolve()) if os.path.isdir(self.asr_model) else self.asr_model


config = ServerConfig()
app = FastAPI(title="Voice AI Coach")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "coach.html")


@app.websocket("/ws/audio")
async def audio_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    session = CoachSession(websocket, config)
    vad = webrtcvad.Vad(config.vad_mode)
    segmenter = VadSegmenter(
        vad=vad,
        silence_ms=config.vad_silence_ms,
        min_segment_ms=config.vad_min_speech_ms,
        start_trigger_ms=config.vad_start_trigger_ms,
        speech_energy_threshold=config.vad_energy_threshold,
    )
    muted = False
    pending_tasks: set[asyncio.Task] = set()

    await session.publish(
        {
            "type": "state_update",
            "status": "connected",
            "sample_rate": SAMPLE_RATE,
            "frame_bytes": FRAME_BYTES,
        }
    )
    await session.set_state("idle")

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
                    await session.publish(
                        {
                            "type": "error",
                            "message": f"Invalid frame size: expected {FRAME_BYTES}, got {len(frame)}",
                        }
                    )
                    continue

                for event_type, segment in segmenter.process_frame(frame):
                    if event_type == "vad_start":
                        await session.publish({"type": "vad_start"})
                    elif event_type == "vad_end":
                        await session.publish(
                            {
                                "type": "vad_end",
                                "segment_ms": segment.duration_ms if segment is not None else 0,
                                "discarded": segment is None,
                            }
                        )
                        if segment is None:
                            continue
                        task = asyncio.create_task(session.transcribe_segment(segment))
                        pending_tasks.add(task)
                        task.add_done_callback(pending_tasks.discard)

            elif "text" in message and message["text"] is not None:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await session.publish({"type": "error", "message": "Invalid JSON control message"})
                    continue

                command = payload.get("type")
                if command == "mute":
                    muted = True
                    segmenter.flush()
                    await session.publish({"type": "state_update", "status": "muted"})
                elif command == "unmute":
                    muted = False
                    segmenter.reset()
                    await session.publish({"type": "state_update", "status": "listening"})
                elif command == "stop":
                    muted = True
                    await session.publish({"type": "state_update", "status": "stopped"})
                elif command == "start":
                    muted = False
                    segmenter.reset()
                    await session.publish({"type": "state_update", "status": "listening"})
                elif command == "speech_done":
                    speech_id = str(payload.get("speech_id", ""))
                    if speech_id:
                        session.io.mark_speech_done(speech_id)
                else:
                    await session.publish({"type": "error", "message": f"Unknown command: {command}"})
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        if 'Cannot call "receive" once a disconnect message has been received' not in str(exc):
            raise
    finally:
        await session.close()
        for task in pending_tasks:
            task.cancel()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the browser voice AI Coach.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--model", default="base", help="faster-whisper model size or local model directory")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute-type", default="auto")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--asr-backend", choices=["whisper", "mock"], default="whisper")
    parser.add_argument("--exercise-library", default="libraries/exercise_library.json")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--intent-model", default="frob/qwen3.5-instruct:4b")
    parser.add_argument("--planner-model", default="frob/qwen3.5-instruct:4b")
    parser.add_argument("--feedback-model", default="frob/qwen3.5-instruct:4b")
    parser.add_argument("--vad-mode", type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument("--vad-silence-ms", type=int, default=800)
    parser.add_argument("--vad-min-speech-ms", type=int, default=500)
    parser.add_argument("--vad-start-trigger-ms", type=int, default=220)
    parser.add_argument("--vad-energy-threshold", type=float, default=600.0)
    args = parser.parse_args()

    config.asr_model = args.model
    config.device = args.device
    config.compute_type = args.compute_type
    config.language = args.language
    config.cpu_threads = args.cpu_threads
    config.num_workers = args.num_workers
    config.asr_backend = args.asr_backend
    config.exercise_library = args.exercise_library
    config.output_dir = args.output_dir
    config.intent_model = args.intent_model
    config.planner_model = args.planner_model
    config.feedback_model = args.feedback_model
    config.vad_mode = args.vad_mode
    config.vad_silence_ms = args.vad_silence_ms
    config.vad_min_speech_ms = args.vad_min_speech_ms
    config.vad_start_trigger_ms = args.vad_start_trigger_ms
    config.vad_energy_threshold = args.vad_energy_threshold

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
