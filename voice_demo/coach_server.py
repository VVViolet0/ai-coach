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
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from uuid import uuid4

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
from experiment_runner import CONDITIONS, ExperimentRunner
from feedback_understanding import (
    classify_feedback_candidate as classify_feedback_candidate_with_rules,
    compact_feedback_text,
    is_confirmation_text,
    is_safety_feedback_text,
)


Publisher = Callable[[Dict[str, Any]], None]
INITIAL_PLANNING_SPEECH = "收到您的训练需求，正在为您生成训练计划。与此同时您可以进行一些热身。"
EXPERIMENT_PREPARATION_SPEECH = "已接受到训练需求。与此同时您可以进行一些热身。"
A_CONDITION_SPEECH = (
    "接下来这一轮是固定训练方案，不会根据你刚才填写的个人目标或偏好设计训练内容，"
    "因此训练内容可能不完全符合你的需求。本轮训练过程中，请尽量按照系统提示完成训练；"
    "如果出现明显不适，可以随时告知实验人员并停止训练。"
)
B_CONDITION_SPEECH = (
    "接下来这一轮会根据你的训练目标生成训练计划。本轮训练过程中，请尽量按照系统提示完成训练；"
    "如果出现明显不适，可以随时告知实验人员并停止训练。"
)
INTERACTIVE_CONDITION_SPEECH = (
    "接下来这一轮会根据你的训练目标生成训练计划。本轮训练过程中，你可以通过自然语言表达疲劳、"
    "节奏、不适、想换动作或停止训练等需求，系统会尝试根据你的反馈调整后续训练。"
)
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
CONFIRMATION_KEYWORDS = {
    "是",
    "是的",
    "对",
    "对的",
    "好",
    "好的",
    "可以",
    "行",
    "否",
    "不",
    "不要",
    "不用",
    "继续",
    "停止",
    "结束",
    "停",
    "yes",
    "no",
    "stop",
    "continue",
}
COMMON_ASR_HALLUCINATIONS = {
    "谢谢观看",
    "感谢观看",
    "欢迎收看",
    "我认为你会不会有什么事",
}

FEEDBACK_KEYWORDS.update(
    {
        "休息",
        "太长",
        "太短",
        "短一点",
        "长一点",
        "快",
        "慢",
        "累",
        "疲劳",
        "痛",
        "疼",
        "不舒服",
        "难",
        "简单",
        "轻松",
        "动作",
        "换",
        "跳过",
        "不想",
        "停止",
        "停了",
        "结束",
        "继续",
        "不喜欢",
        "膝盖",
        "状态很好",
        "很好",
        "多练",
        "加练",
        "加组",
        "再来",
    }
)
SAFETY_KEYWORDS.update({"痛", "疼", "不舒服", "停止", "停了", "结束", "停", "可以停了", "不想继续"})
CONFIRMATION_KEYWORDS.update({"是", "是的", "对", "对的", "好", "好的", "可以", "行", "否", "不用", "不要", "继续", "停止", "结束", "停"})


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


def _experiment_preparation_message() -> Dict[str, Any]:
    return {
        "type": "coach_message",
        **_coach_message(
            display=True,
            speak=True,
            message=EXPERIMENT_PREPARATION_SPEECH,
            speech_text=EXPERIMENT_PREPARATION_SPEECH,
            category="experiment_preparation",
        ),
    }


def condition_instruction_for(condition_id: str) -> str:
    if condition_id == "A":
        return A_CONDITION_SPEECH
    if condition_id == "B":
        return B_CONDITION_SPEECH
    if condition_id == "C":
        return INTERACTIVE_CONDITION_SPEECH
    return A_CONDITION_SPEECH


def _strip_coach_prefix(message: str) -> str:
    return message.strip().replace("[Coach]", "").replace("[Adjustment]", "").strip()


def _compact_text(text: str) -> str:
    return compact_feedback_text(text)


def _has_safety_keyword(text: str) -> bool:
    return is_safety_feedback_text(text)


def classify_feedback_candidate(text: str) -> tuple[bool, str]:
    return classify_feedback_candidate_with_rules(text)


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
        self._log_lock = threading.Lock()
        self._transcript_log: list[Dict[str, Any]] = []
        self._feedback_contexts: Dict[str, list[Dict[str, Any]]] = {}
        self._condition_stop_lock = threading.Lock()
        self._condition_stop_requested = False
        self._condition_stop_reason = ""
        self._confirmation_lock = threading.Lock()
        self._pending_confirmation_type = ""
        self.nonblocking_feedback = True

    def enqueue_user_text(self, text: str, *, safety: bool = False, transcript_id: str = "") -> None:
        cleaned = text.strip()
        if not cleaned or self._closed:
            return
        if not safety:
            self._drop_pending_normal_feedback()
        with self._log_lock:
            transcript = next(
                (item for item in reversed(self._transcript_log) if item.get("transcript_id") == transcript_id),
                {},
            )
            self._feedback_contexts.setdefault(cleaned, []).append(
                {
                    "feedback_id": uuid4().hex[:12],
                    "transcript_id": transcript_id,
                    "asr_latency_ms": int(transcript.get("latency_ms", 0)),
                    "input_method": transcript.get("input_method", "asr_whisper"),
                    "submit_method": transcript.get("submit_method", "vad_auto"),
                }
            )
        self._messages.put(cleaned)

    def _drop_pending_normal_feedback(self) -> None:
        kept = []
        try:
            while True:
                message = self._messages.get_nowait()
                if _has_safety_keyword(message):
                    kept.append(message)
                else:
                    self.consume_feedback_context(message)
        except queue.Empty:
            pass
        for message in kept:
            self._messages.put(message)

    def consume_feedback_context(self, text: str) -> Dict[str, Any]:
        with self._log_lock:
            contexts = self._feedback_contexts.get(text, [])
            if not contexts:
                return {}
            context = contexts.pop(0)
            if not contexts:
                self._feedback_contexts.pop(text, None)
            return context

    def log_transcript(self, record: Dict[str, Any]) -> str:
        transcript_id = str(record.get("transcript_id") or uuid4().hex[:12])
        with self._log_lock:
            self._transcript_log.append(
                {
                    "transcript_id": transcript_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **record,
                }
            )
        return transcript_id

    def update_transcript_route(self, transcript_id: str, target: str, reason: str = "") -> None:
        if not transcript_id:
            return
        with self._log_lock:
            for record in reversed(self._transcript_log):
                if record.get("transcript_id") == transcript_id:
                    record["target"] = target
                    record["route_reason"] = reason
                    return

    def get_transcript_log(self) -> list[Dict[str, Any]]:
        with self._log_lock:
            return deepcopy(self._transcript_log)

    def send(self, message: str) -> None:
        if self._closed:
            return
        classified = classify_coach_message(message)
        payload = {"type": "coach_message", **classified}
        if classified.get("speak"):
            speech_id = self._register_speech(str(classified.get("category", "")))
            payload["speech_id"] = speech_id
        self._publish(payload)

    def send_coach_message(
        self,
        message: str,
        *,
        category: str,
        speak: bool = True,
        display: bool = True,
        speech_text: str = "",
    ) -> None:
        if self._closed:
            return
        payload = {
            "type": "coach_message",
            **_coach_message(
                display=display,
                speak=speak,
                message=message,
                speech_text=speech_text or message,
                category=category,
            ),
        }
        if speak:
            payload["speech_id"] = self._register_speech(category)
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

    def request_condition_stop(self, reason: str = "manual_condition_stop") -> None:
        with self._condition_stop_lock:
            self._condition_stop_requested = True
            self._condition_stop_reason = reason
        with self._speech_lock:
            events = list(self._speech_events.values())
        for event in events:
            event.set()

    def consume_condition_stop_request(self) -> str:
        with self._condition_stop_lock:
            if not self._condition_stop_requested:
                return ""
            reason = self._condition_stop_reason or "manual_condition_stop"
            self._condition_stop_requested = False
            self._condition_stop_reason = ""
            return reason

    def clear_condition_stop_request(self) -> None:
        with self._condition_stop_lock:
            self._condition_stop_requested = False
            self._condition_stop_reason = ""

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

    def set_pending_confirmation(self, confirmation_type: str) -> None:
        with self._confirmation_lock:
            self._pending_confirmation_type = confirmation_type

    def clear_pending_confirmation(self) -> None:
        with self._confirmation_lock:
            self._pending_confirmation_type = ""

    def has_pending_confirmation(self) -> bool:
        with self._confirmation_lock:
            return bool(self._pending_confirmation_type)


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
        self.experiment_condition_id: Optional[str] = None
        self.participant_id = "anonymous"
        self.experiment_runner: Optional[ExperimentRunner] = None
        self.prepared_experiment: Optional[Dict[str, Any]] = None

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
            transcript_id = self.io.log_transcript(
                {
                    "text": text,
                    "latency_ms": latency_ms,
                    "asr_ms": result["asr_ms"],
                    "segment_ms": segment.duration_ms,
                    "language": result["language"],
                    "language_probability": result["language_probability"],
                    "target": "pending_route",
                }
            )
            await self.publish(
                {
                    "type": "transcript_final",
                    "transcript_id": transcript_id,
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
                await self.route_transcript(text, transcript_id=transcript_id)
        except Exception as exc:
            await self.set_state("error")
            await self.publish({"type": "error", "message": str(exc)})

    async def submit_text_transcript(
        self,
        text: str,
        *,
        input_method: str = "text_manual",
        submit_method: str = "manual",
    ) -> None:
        cleaned = text.strip()
        if not cleaned:
            await self.publish({"type": "error", "message": "Submitted text is empty."})
            return

        transcript_id = self.io.log_transcript(
            {
                "text": cleaned,
                "latency_ms": 0,
                "asr_ms": 0,
                "segment_ms": 0,
                "language": "zh",
                "language_probability": 1.0,
                "target": "pending_route",
                "input_method": input_method,
                "submit_method": submit_method,
            }
        )
        await self.publish(
            {
                "type": "transcript_final",
                "transcript_id": transcript_id,
                "text": cleaned,
                "latency_ms": 0,
                "asr_ms": 0,
                "segment_ms": 0,
                "language": "zh",
                "language_probability": 1.0,
                "input_method": input_method,
                "submit_method": submit_method,
            }
        )
        await self.publish({"type": "metrics", "segment_ms": 0, "asr_ms": 0, "latency_ms": 0})
        await self.route_transcript(cleaned, transcript_id=transcript_id)

    async def route_transcript(self, text: str, transcript_id: str = "") -> None:
        if self.state == "idle":
            self.io.update_transcript_route(transcript_id, "initial_intent")
            await self.publish({"type": "user_transcript", "text": text, "target": "initial_intent"})
            if self.config.experiment_mode:
                await self.publish(_experiment_preparation_message())
                await self.set_state("preparing_experiment")
                self.start_experiment_prepare(text)
            else:
                await self.publish(_initial_planning_message())
                await self.set_state("planning")
                self.start_coach(text)
            return

        if self.state in {"preparing_experiment", "prepared"}:
            self.io.update_transcript_route(transcript_id, "ignored", "experiment_not_running")
            await self.publish(
                {"type": "user_transcript", "text": text, "target": "ignored", "reason": "experiment_not_running"}
            )
            return

        if self.state in {"planning", "workout_running"}:
            if self.experiment_condition_id in {"A", "B"}:
                self.io.update_transcript_route(transcript_id, "ignored_feedback", "feedback_disabled_for_condition")
                await self.publish(
                    {
                        "type": "user_transcript",
                        "text": text,
                        "target": "ignored_feedback",
                        "reason": "feedback_disabled_for_condition",
                    }
                )
                return
            if self.io.has_pending_confirmation() and is_confirmation_text(text):
                self.io.update_transcript_route(transcript_id, "feedback", "confirmation")
                self.io.enqueue_user_text(text, safety=True, transcript_id=transcript_id)
                await self.publish({"type": "user_transcript", "text": text, "target": "feedback", "reason": "confirmation"})
                return
            accepted, reason = classify_feedback_candidate(text)
            normalized = _compact_text(text)
            now = time.monotonic()
            if accepted and normalized == self._last_feedback_text and now - self._last_feedback_at < 3.0:
                accepted = False
                reason = "duplicate_recent"
            if accepted:
                safety = reason == "safety_keyword"
                self.io.update_transcript_route(transcript_id, "feedback", reason)
                self.io.enqueue_user_text(text, safety=safety, transcript_id=transcript_id)
                self._last_feedback_text = normalized
                self._last_feedback_at = now
                await self.publish({"type": "user_transcript", "text": text, "target": "feedback"})
            else:
                target = "ignored_noise" if reason in {"common_asr_hallucination", "repetitive_asr"} else "ignored_feedback"
                self.io.update_transcript_route(transcript_id, target, reason)
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

    def _build_experiment_runner(self) -> ExperimentRunner:
        return ExperimentRunner(
            exercise_library_path=self.config.exercise_library,
            output_dir=self.config.experiment_output_dir,
            intent_model=self.config.intent_model,
            planner_model=self.config.planner_model,
            feedback_model=self.config.feedback_model,
        )

    def start_experiment_prepare(self, initial_request: str) -> None:
        if self.coach_thread and self.coach_thread.is_alive():
            return
        self.coach_thread = threading.Thread(
            target=self._prepare_experiment,
            args=(initial_request,),
            daemon=True,
        )
        self.coach_thread.start()

    def _prepare_experiment(self, initial_request: str) -> None:
        try:
            self.experiment_runner = self._build_experiment_runner()
            self.prepared_experiment = self.experiment_runner.prepare(self.participant_id, initial_request)
            self.set_state_from_thread("prepared")
            self.publish_from_thread({"type": "experiment_prepared", "result": self.prepared_experiment})
        except Exception as exc:
            self.set_state_from_thread("error")
            self.publish_from_thread({"type": "error", "message": str(exc)})

    def start_experiment_condition(self, condition_id: str) -> None:
        if self.coach_thread and self.coach_thread.is_alive():
            return
        self.io.clear_condition_stop_request()
        self.experiment_condition_id = condition_id
        self.coach_thread = threading.Thread(
            target=self._run_experiment_condition,
            args=(condition_id,),
            daemon=True,
        )
        self.coach_thread.start()

    def _run_experiment_condition(self, condition_id: str) -> None:
        try:
            if not self.prepared_experiment:
                raise RuntimeError("Experiment has not been prepared yet.")
            runner = self.experiment_runner or self._build_experiment_runner()
            instruction = condition_instruction_for(condition_id)
            self.io.send_coach_message(
                instruction,
                category="condition_instruction",
                speak=True,
                display=True,
            )
            self.io.wait_for_speech("condition_instruction")
            result = runner.run_prepared_condition(
                self.prepared_experiment,
                condition_id=condition_id,
                io=self.io,
                condition_instruction=instruction,
                close_io=False,
            )
            self.experiment_condition_id = None
            self.set_state_from_thread("prepared")
            self.publish_from_thread({"type": "condition_summary", "result": result})
        except Exception as exc:
            self.experiment_condition_id = None
            self.set_state_from_thread("error")
            self.publish_from_thread({"type": "error", "message": str(exc)})

    def request_current_condition_stop(self, reason: str = "manual_condition_stop") -> bool:
        if not self.experiment_condition_id or self.state not in {"planning", "workout_running"}:
            return False
        self.io.request_condition_stop(reason)
        return True

    def _run_coach_session(self, initial_request: str) -> None:
        try:
            if self.experiment_condition_id:
                runner = self._build_experiment_runner()
                result = runner.run(
                    condition_id=self.experiment_condition_id,
                    participant_id=self.participant_id,
                    user_request=initial_request,
                    io=self.io,
                )
                self.set_state_from_thread("ended")
                self.publish_from_thread({"type": "session_summary", "result": result})
                return

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
        self.experiment_output_dir = "data/experiments"
        self.experiment_mode = False
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
    page = "experiment.html" if config.experiment_mode else "coach.html"
    return FileResponse(STATIC_DIR / page)


@app.get("/experiment")
async def experiment_index() -> FileResponse:
    return FileResponse(STATIC_DIR / "experiment.html")


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
                    if config.experiment_mode:
                        participant_id = str(payload.get("participant_id", "")).strip()
                        if not participant_id:
                            await session.publish({"type": "error", "message": "Participant ID is required."})
                            continue
                        session.participant_id = participant_id
                    muted = False
                    segmenter.reset()
                    await session.publish({"type": "state_update", "status": "listening"})
                elif command == "submit_intent":
                    if not config.experiment_mode:
                        await session.publish({"type": "error", "message": "submit_intent is only available in experiment mode."})
                        continue
                    if session.state != "idle":
                        await session.publish({"type": "error", "message": "The training request has already been submitted."})
                        continue
                    await session.submit_text_transcript(
                        str(payload.get("text", "")),
                        input_method=str(payload.get("input_method", "text_manual")),
                        submit_method=str(payload.get("submit_method", "manual")),
                    )
                elif command == "submit_feedback":
                    if not config.experiment_mode:
                        await session.publish({"type": "error", "message": "submit_feedback is only available in experiment mode."})
                        continue
                    await session.submit_text_transcript(
                        str(payload.get("text", "")),
                        input_method=str(payload.get("input_method", "text_manual")),
                        submit_method=str(payload.get("submit_method", "manual")),
                    )
                elif command == "run_condition":
                    requested_condition = str(payload.get("condition_id", "")).upper()
                    if not config.experiment_mode:
                        await session.publish({"type": "error", "message": "run_condition is only available in experiment mode."})
                        continue
                    if requested_condition not in CONDITIONS:
                        await session.publish({"type": "error", "message": "Invalid experiment condition."})
                        continue
                    if session.state != "prepared" or not session.prepared_experiment:
                        await session.publish({"type": "error", "message": "Prepare the experiment intent before running a condition."})
                        continue
                    session.experiment_condition_id = requested_condition
                    await session.publish(
                        {
                            "type": "experiment_condition",
                            "condition_id": requested_condition,
                            "participant_id": session.participant_id,
                            "condition": asdict(CONDITIONS[requested_condition]),
                        }
                    )
                    await session.set_state("planning")
                    session.start_experiment_condition(requested_condition)
                elif command == "end_condition":
                    if not config.experiment_mode:
                        await session.publish({"type": "error", "message": "end_condition is only available in experiment mode."})
                        continue
                    if session.request_current_condition_stop():
                        muted = True
                        segmenter.flush()
                        await session.publish({"type": "condition_stop_requested", "reason": "manual_condition_stop"})
                    else:
                        await session.publish({"type": "error", "message": "No experiment condition is currently running."})
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
    parser.add_argument("--experiment-output-dir", default="data/experiments")
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
    config.experiment_output_dir = args.experiment_output_dir
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
