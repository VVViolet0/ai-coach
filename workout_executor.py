import json
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union
from uuid import uuid4

from feedback_understanding import (
    FeedbackUnderstandingFailure,
    FeedbackUnderstandingResult,
    classify_confirmation_text,
    is_safety_feedback_text,
    normalize_confirmation_text,
    understand_feedback_fast_path,
    understand_feedback,
)


MIN_REST_SECONDS = 5
MAX_REST_SECONDS = 180
MIN_SETS = 1
MAX_SETS = 8
PRE_EXERCISE_PREVIEW_WAIT_SECONDS = 3
MIN_REST_MULTIPLIER = 0.5
MAX_REST_MULTIPLIER = 2.0
MIN_BEAT_MULTIPLIER = 0.6
MAX_BEAT_MULTIPLIER = 1.4
DEFAULT_BEAT_HZ = 0.75
MIN_SET_DELTA = -3
MAX_SET_DELTA = 3
LOW_FEEDBACK_CONFIDENCE = 0.4
PENDING_FEEDBACK_TTL_SECONDS = 8.0
NO_STATE_DERIVED_ACTION_INTENTS = {
    "stop",
    "pain",
    "neutral",
    "unknown",
    "pace_up",
    "pace_down",
    "preference_dislike",
    "preference_like",
}
SILENT_NO_ACTION_INTENTS = {"unknown", "neutral"}
NO_ACTIONS = {"ask_clarification", "no_action"}
EXERCISE_DEMO_LIBRARY_PATH = Path("libraries/exercise_demo_library.json")


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExecutorIO(Protocol):
    def send(self, message: str) -> None:
        ...

    def poll_user_input(self) -> Optional[str]:
        ...

    def close(self) -> None:
        ...


class Clock(Protocol):
    def sleep(self, seconds: float) -> None:
        ...

    def now(self) -> float:
        ...


class DecisionEngine(Protocol):
    def decide(self, user_text: str, state: "SessionState") -> "DecisionResult":
        ...


class FeedbackUnderstander(Protocol):
    def __call__(
        self,
        user_text: str,
        state_snapshot: Dict,
        model: str = "frob/qwen3.5-instruct:4b",
    ) -> Union[FeedbackUnderstandingResult, FeedbackUnderstandingFailure, None]:
        ...


@dataclass
class AdjustmentAction:
    action_type: str
    rest_multiplier: float = 1.0
    beat_multiplier: float = 1.0
    set_delta: int = 0
    tempo_cue: Optional[str] = None
    reason: str = ""
    rule_id: str = ""
    state_reason: str = ""
    replacement_exercise: Optional[str] = None


@dataclass
class DecisionResult:
    intent: str
    coach_reply: str
    actions: List[AdjustmentAction] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_type: str = ""


@dataclass
class UserConditionState:
    fatigue_level: str = "medium"  # low / medium / high
    difficulty_level: str = "appropriate"  # easy / appropriate / hard
    preference: str = "neutral"  # like / neutral / dislike
    last_feedback_text: str = ""
    updated_at_phase: str = "idle"

    def snapshot(self) -> Dict[str, str]:
        return {
            "fatigue_level": self.fatigue_level,
            "difficulty_level": self.difficulty_level,
            "preference": self.preference,
            "last_feedback_text": self.last_feedback_text,
            "updated_at_phase": self.updated_at_phase,
        }


@dataclass
class SessionState:
    workout_plan: Dict
    condition_id: str = ""
    experiment_metadata: Dict = field(default_factory=dict)
    feedback_enabled: bool = True
    initial_workout_plan: Dict = field(default_factory=dict)
    initial_intent: Dict = field(default_factory=dict)
    exercise_library: List[Dict] = field(default_factory=list)
    current_round: int = 0
    current_exercise: int = 0
    current_set: int = 0
    phase: str = "idle"
    seconds_remaining: int = 0
    rest_multiplier: float = 1.0
    beat_multiplier: float = 1.0
    set_delta: int = 0
    tempo_cue: str = "normal"
    is_active: bool = True
    end_reason: str = "completed"
    awaiting_pain_confirmation: bool = False
    awaiting_stop_confirmation: bool = False
    user_condition: UserConditionState = field(default_factory=UserConditionState)
    event_log: List[Dict] = field(default_factory=list)
    conversation_log: List[Dict] = field(default_factory=list)
    adjustment_log: List[Dict] = field(default_factory=list)
    condition_log: List[Dict] = field(default_factory=list)
    transcript_log: List[Dict] = field(default_factory=list)
    feedback_trace: List[Dict] = field(default_factory=list)
    active_feedback_id: str = ""
    feedback_sequence: int = 0
    session_started_at: str = ""
    session_ended_at: str = ""
    actual_duration_seconds: float = 0.0

    def log_event(self, event_type: str, detail: Dict) -> None:
        self.event_log.append(
            {"condition_id": self.condition_id, "timestamp": _timestamp(), "type": event_type, "detail": detail}
        )

    def log_conversation(self, record: Dict) -> None:
        self.conversation_log.append({"condition_id": self.condition_id, "timestamp": _timestamp(), **record})

    def log_adjustment(self, record: Dict) -> None:
        self.adjustment_log.append(
            {
                "condition_id": self.condition_id,
                "timestamp": _timestamp(),
                "feedback_id": self.active_feedback_id,
                **record,
            }
        )

    def log_condition(self, record: Dict) -> None:
        self.condition_log.append(
            {
                "condition_id": self.condition_id,
                "timestamp": _timestamp(),
                "feedback_id": self.active_feedback_id,
                **record,
            }
        )


class RealClock:
    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def now(self) -> float:
        return time.time()


class CLIExecutorIO:
    def __init__(self) -> None:
        self._messages: "queue.Queue[str]" = queue.Queue()
        self._alive = True
        self._thread = threading.Thread(target=self._read_stdin, daemon=True)
        self._thread.start()

    def _read_stdin(self) -> None:
        while self._alive:
            try:
                line = input()
            except EOFError:
                return
            except Exception:
                return
            if line is not None:
                self._messages.put(line.strip())

    def send(self, message: str) -> None:
        print(message)

    def poll_user_input(self) -> Optional[str]:
        try:
            return self._messages.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._alive = False


def load_exercise_demo_library(path: str | Path = EXERCISE_DEMO_LIBRARY_PATH) -> Dict[str, Dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Exercise demo library must be a JSON object keyed by exercise name.")
    return payload


def _normalize_beat_hz(value: object, fallback: float = DEFAULT_BEAT_HZ) -> float:
    try:
        beat_hz = float(value)
    except (TypeError, ValueError):
        beat_hz = fallback
    if beat_hz <= 0:
        return 0.0
    return round(max(0.2, min(2.0, beat_hz)), 3)


def translate_plan_for_demo(workout_plan: Dict) -> Dict:
    exercise_demo_library = load_exercise_demo_library()
    demo_plan = {
        "rounds": workout_plan["rounds"],
        "rest_between_rounds": workout_plan["rest_between_rounds"],
        "exercises": [],
    }

    for ex in workout_plan["exercises"]:
        name = ex["exercise"]
        details = exercise_demo_library.get(
            name,
            {
                "exercise_name": name.replace("_", " ").title(),
                "instructions": ["保持动作安全、稳定、受控。"],
            },
        )
        default_beat_hz = _normalize_beat_hz(details.get("default_beat_hz"))
        beat_source = "exercise_demo_library" if "default_beat_hz" in details else "fallback"

        demo_exercise = {
            "exercise_name": name,
            "display_name": details["exercise_name"],
            "instructions": details["instructions"],
            "avg_set_time": ex["avg_set_time"],
            "total_sets": ex["total_sets"],
            "rest_seconds": ex["rest_seconds"],
            "rest_after_exercise_seconds": int(ex.get("rest_after_exercise_seconds", 0)),
            "default_beat_hz": default_beat_hz,
            "beat_source": beat_source,
        }
        demo_plan["exercises"].append(demo_exercise)

    return demo_plan


class RuleFirstDecisionEngine:
    def __init__(
        self,
        fallback_parser: Optional[Callable[[str, SessionState], Optional[DecisionResult]]] = None,
    ) -> None:
        self._fallback_parser = fallback_parser

    def decide(self, user_text: str, state: SessionState) -> DecisionResult:
        if self._fallback_parser:
            fallback_result = self._fallback_parser(user_text, state)
            if fallback_result:
                return fallback_result

        return DecisionResult(
            intent="unknown",
            coach_reply=(
                "我可以调整节奏、降低强度或停止训练。"
                "你可以说“太难了”“快一点”或“停止”。"
            ),
        )


class FeedbackWorker:
    """Runs slow feedback understanding off the timing loop.

    Fast-rule feedback is handled synchronously in submit(); only ambiguous
    messages wait for the worker thread and LLM fallback.
    """

    def __init__(
        self,
        state: SessionState,
        io: ExecutorIO,
        decision_engine: DecisionEngine,
        feedback_understander: FeedbackUnderstander,
        feedback_model: str,
    ) -> None:
        self._state = state
        self._io = io
        self._decision_engine = decision_engine
        self._feedback_understander = feedback_understander
        self._feedback_model = feedback_model
        self._messages: "queue.Queue[Optional[Tuple[str, float, int]]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, message: str) -> None:
        self._drop_pending_normal_feedback()
        self._state.feedback_sequence += 1
        sequence_id = self._state.feedback_sequence
        # Clear, high-priority feedback should affect the live set immediately.
        # Ambiguous feedback is queued for the slower LLM fallback and protected by sequence id.
        if understand_feedback_fast_path(message) is not None:
            _emit_runtime_event(self._io, "feedback_processing_start", {"text": message})
            _handle_user_message(
                message,
                self._state,
                self._io,
                self._decision_engine,
                self._feedback_understander,
                self._feedback_model,
                feedback_sequence_id=sequence_id,
            )
            _emit_runtime_event(self._io, "feedback_processing_end", {"text": message})
            return
        self._messages.put((message, time.time(), sequence_id))

    def _drop_pending_normal_feedback(self) -> None:
        kept: List[Tuple[str, float, int]] = []
        while True:
            try:
                item = self._messages.get_nowait()
            except queue.Empty:
                break
            if item is None:
                kept.append(item)
                continue
            message, submitted_at, sequence_id = item
            if is_safety_feedback_text(message):
                kept.append((message, submitted_at, sequence_id))
        for item in kept:
            self._messages.put(item)

    def close(self) -> None:
        self._messages.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            item = self._messages.get()
            if item is None:
                return
            message, submitted_at, sequence_id = item
            if time.time() - submitted_at > PENDING_FEEDBACK_TTL_SECONDS and not is_safety_feedback_text(message):
                _emit_runtime_event(
                    self._io,
                    "feedback_ignored",
                    {
                        "raw_text": message,
                        "intent": "unknown",
                        "confidence": 0.0,
                        "reason": "stale_pending_feedback",
                    },
                )
                continue
            _emit_runtime_event(self._io, "feedback_processing_start", {"text": message})
            _handle_user_message(
                message,
                self._state,
                self._io,
                self._decision_engine,
                self._feedback_understander,
                self._feedback_model,
                feedback_sequence_id=sequence_id,
            )
            _emit_runtime_event(self._io, "feedback_processing_end", {"text": message})


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _emit_runtime_event(io: ExecutorIO, event_type: str, payload: Dict) -> None:
    send_event = getattr(io, "send_event", None)
    if callable(send_event):
        send_event(event_type, payload)


def _log_feedback_ignored(
    state: SessionState,
    io: ExecutorIO,
    raw_text: str,
    actions: List[str],
    confidence: float,
    reason: str,
) -> None:
    detail = {
        "raw_text": raw_text,
        "actions": actions,
        "confidence": confidence,
        "reason": reason,
    }
    state.log_event("feedback_ignored", detail)
    _emit_runtime_event(io, "feedback_ignored", detail)


def _log_feedback_understanding_failure(
    state: SessionState,
    io: ExecutorIO,
    raw_text: str,
    feedback_model: str,
    failure: Optional[FeedbackUnderstandingFailure],
) -> None:
    detail = {
        "raw_text": raw_text,
        "model": feedback_model,
        "error_summary": failure.error_summary if failure else "understander returned None",
        "tried_channels": failure.tried_channels if failure else [],
    }
    state.log_event("feedback_understanding_failed", detail)
    _emit_runtime_event(io, "feedback_understanding_failed", detail)


def _build_feedback_state_snapshot(state: SessionState) -> Dict:
    current_exercise_name = ""
    if state.workout_plan.get("exercises") and state.current_exercise < len(state.workout_plan["exercises"]):
        current = state.workout_plan["exercises"][state.current_exercise]
        current_exercise_name = current.get("exercise_name") or current.get("exercise", "")

    return {
        "phase": state.phase,
        "current_exercise_name": current_exercise_name,
    }


def _structured_understanding_state(understanding: FeedbackUnderstandingResult) -> Dict:
    return {
        "fatigue_level": understanding.fatigue_level,
        "difficulty_level": understanding.difficulty_level,
        "preference": understanding.preference,
        "safety": understanding.safety,
        "confidence": understanding.confidence,
        "actions": understanding.actions,
    }


def _current_beat_payload(state: SessionState, *, active_phase: Optional[bool] = None) -> Dict[str, float]:
    default_beat_hz = DEFAULT_BEAT_HZ
    exercises = state.workout_plan.get("exercises") or []
    if exercises and 0 <= state.current_exercise < len(exercises):
        default_beat_hz = _normalize_beat_hz(exercises[state.current_exercise].get("default_beat_hz"))
    beat_multiplier = round(max(MIN_BEAT_MULTIPLIER, min(MAX_BEAT_MULTIPLIER, state.beat_multiplier)), 3)
    if active_phase is None:
        active_phase = state.phase == "active_set"
    effective_beat_hz = round(default_beat_hz * beat_multiplier, 3) if active_phase else 0.0
    return {
        "default_beat_hz": default_beat_hz,
        "beat_multiplier": beat_multiplier,
        "effective_beat_hz": effective_beat_hz,
    }


def _send_coach_reply(io: ExecutorIO, state: SessionState, reply: str, *, urgent: bool = False) -> None:
    if not reply:
        return
    io.send(f"[Coach] {reply}")
    if urgent:
        _emit_runtime_event(io, "urgent_coach_reply", {"text": reply})
    state.log_conversation({"role": "coach", "content": reply})


def _handle_stop_confirmation(cleaned: str, state: SessionState, io: ExecutorIO) -> bool:
    normalized = normalize_confirmation_text(cleaned)
    clear_confirmation = getattr(io, "clear_pending_confirmation", None)
    stop_words = {
        "是",
        "是的",
        "对",
        "对的",
        "停止",
        "停",
        "停下",
        "结束",
        "停止训练",
        "结束训练",
        "结束整个训练",
        "可以停了",
        "yes",
        "y",
        "stop",
        "quit",
        "end",
    }
    continue_words = {
        "否",
        "不",
        "不是",
        "不要",
        "不停止",
        "不用",
        "继续",
        "继续训练",
        "no",
        "n",
        "continue",
    }
    if normalized in stop_words:
        if callable(clear_confirmation):
            clear_confirmation()
        state.awaiting_stop_confirmation = False
        state.is_active = False
        state.end_reason = "user_requested_stop"
        io.send("好的，现在停止训练。")
        state.log_event("stop_confirmation", {"decision": "stop"})
        return True
    if normalized in continue_words:
        if callable(clear_confirmation):
            clear_confirmation()
        state.awaiting_stop_confirmation = False
        io.send("好的，我们继续。如果需要，我可以随时降低强度。")
        state.log_event("stop_confirmation", {"decision": "continue"})
        return True

    io.send("请回答“是”来停止，或回答“否”来继续。")
    state.log_event("stop_confirmation", {"decision": "unclear"})
    return True


def _handle_pain_confirmation(cleaned: str, state: SessionState, io: ExecutorIO) -> bool:
    confirmation = classify_confirmation_text(cleaned)
    clear_confirmation = getattr(io, "clear_pending_confirmation", None)
    if confirmation == "yes":
        if callable(clear_confirmation):
            clear_confirmation()
        state.awaiting_pain_confirmation = False
        io.send("好的，我们用更轻、更安全的节奏继续。")
        state.log_event("pain_confirmation", {"decision": "continue"})
        return True
    if confirmation == "no":
        if callable(clear_confirmation):
            clear_confirmation()
        state.awaiting_pain_confirmation = False
        state.is_active = False
        state.end_reason = "user_stopped_after_pain"
        io.send("训练已停止。请先休息，如果需要请及时寻求医疗建议。")
        state.log_event("pain_confirmation", {"decision": "stop"})
        return True

    io.send("请回答“继续”来继续训练，或回答“停止”来结束训练。")
    state.log_event("pain_confirmation", {"decision": "unclear"})
    return True


def _apply_condition_from_understanding(
    understanding: FeedbackUnderstandingResult,
    state: SessionState,
) -> bool:
    before = state.user_condition.snapshot()
    fatigue_level = (
        understanding.fatigue_level
        if understanding.fatigue_level in {"low", "medium", "high"}
        else state.user_condition.fatigue_level
    )
    difficulty_level = (
        understanding.difficulty_level
        if understanding.difficulty_level in {"easy", "appropriate", "hard"}
        else state.user_condition.difficulty_level
    )
    preference = (
        understanding.preference
        if understanding.preference in {"like", "neutral", "dislike"}
        else state.user_condition.preference
    )
    changed = (
        state.user_condition.fatigue_level != fatigue_level
        or state.user_condition.difficulty_level != difficulty_level
        or state.user_condition.preference != preference
    )

    state.user_condition.fatigue_level = fatigue_level
    state.user_condition.difficulty_level = difficulty_level
    state.user_condition.preference = preference
    state.user_condition.last_feedback_text = understanding.raw_text
    state.user_condition.updated_at_phase = state.phase

    condition_record = {
        "before_state": before,
        "after_state": state.user_condition.snapshot(),
        "trigger_text": understanding.raw_text,
        "actions": understanding.actions,
        "safety": understanding.safety,
        "confidence": understanding.confidence,
        "reason": understanding.reason,
        "understanding_channel": understanding.llm_channel,
        "phase": state.phase,
    }
    state.log_condition(condition_record)
    state.log_event("condition_updated", condition_record)
    return changed


def _action_to_adjustment(
    action: str,
    state: SessionState,
    understanding: FeedbackUnderstandingResult,
) -> Optional[AdjustmentAction]:
    # The LLM or fast rule selects action names; the executor owns the concrete
    # workout changes and safety bounds behind those names.
    reason = understanding.reason or action

    if action == "decrease_rest":
        return AdjustmentAction(
            action_type="adjust_intensity",
            rest_multiplier=0.75,
            reason=reason,
            rule_id="action_decrease_rest",
            state_reason="llm_action=decrease_rest",
        )

    if action == "increase_rest":
        return AdjustmentAction(
            action_type="adjust_intensity",
            rest_multiplier=1.25,
            reason=reason,
            rule_id="action_increase_rest",
            state_reason="llm_action=increase_rest",
        )

    if action == "decrease_sets":
        return AdjustmentAction(
            action_type="adjust_intensity",
            set_delta=-1,
            reason=reason,
            rule_id="action_decrease_sets",
            state_reason="llm_action=decrease_sets",
        )

    if action == "increase_sets":
        return AdjustmentAction(
            action_type="adjust_intensity",
            set_delta=1,
            reason=reason,
            rule_id="action_increase_sets",
            state_reason="llm_action=increase_sets",
        )

    if action == "slow_tempo":
        return AdjustmentAction(
            action_type="adjust_intensity",
            beat_multiplier=0.8,
            tempo_cue="slower",
            reason=reason,
            rule_id="action_slow_tempo",
            state_reason="llm_action=slow_tempo",
        )

    if action == "speed_up_tempo":
        return AdjustmentAction(
            action_type="adjust_intensity",
            beat_multiplier=1.2,
            tempo_cue="faster",
            reason=reason,
            rule_id="action_speed_up_tempo",
            state_reason="llm_action=speed_up_tempo",
        )

    if action == "decrease_difficulty":
        return AdjustmentAction(
            action_type="adjust_intensity",
            set_delta=-1,
            rest_multiplier=1.25,
            beat_multiplier=0.8,
            tempo_cue="slower",
            reason=reason,
            rule_id="action_decrease_difficulty",
            state_reason="llm_action=decrease_difficulty",
        )

    if action == "increase_difficulty":
        return AdjustmentAction(
            action_type="adjust_intensity",
            set_delta=1,
            rest_multiplier=0.85,
            beat_multiplier=1.2,
            tempo_cue="faster",
            reason=reason,
            rule_id="action_increase_difficulty",
            state_reason="llm_action=increase_difficulty",
        )

    if action == "skip_current_exercise":
        return AdjustmentAction(
            action_type="skip_current_exercise",
            reason=reason,
            rule_id="action_skip_current_exercise",
            state_reason="llm_action=skip_current_exercise",
        )

    if action == "replace_current_exercise":
        return AdjustmentAction(
            action_type="replace_exercise",
            replacement_exercise=_find_replacement_exercise(state),
            reason=reason,
            rule_id="action_replace_current_exercise",
            state_reason="llm_action=replace_current_exercise",
        )

    if action == "stop_workout":
        return AdjustmentAction(
            action_type="stop",
            reason="user_requested_stop",
            rule_id="action_stop_workout",
            state_reason="llm_action=stop_workout",
        )

    return None


def _build_decision_from_actions(
    understanding: FeedbackUnderstandingResult,
    state: SessionState,
    decision_engine: DecisionEngine,
) -> DecisionResult:
    # Safety decisions are handled before normal action mapping so stop/pain
    # cannot be diluted by lower-priority actions.
    actions = understanding.actions

    if "confirm_stop_workout" in actions:
        return DecisionResult(
            intent="stop",
            coach_reply=understanding.reply or "你是想结束整个训练吗？请回答“是”或“否”。",
            actions=[],
            requires_confirmation=True,
            confirmation_type="stop",
        )

    if understanding.safety == "stop_request" or "stop_workout" in actions:
        return DecisionResult(
            intent="stop",
            coach_reply=understanding.reply or "好的，现在停止训练。",
            actions=[
                AdjustmentAction(
                    action_type="stop",
                    reason="user_requested_stop",
                    rule_id="action_stop_workout",
                    state_reason="safety=stop_request",
                )
            ],
        )

    if understanding.safety == "pain":
        return DecisionResult(
            intent="pain",
            coach_reply=understanding.reply or "我听到你有疼痛或不适。我已经降低强度。你还要继续吗？请回答“继续”或“停止”。",
            actions=[
                AdjustmentAction(
                action_type="adjust_intensity",
                set_delta=-1,
                rest_multiplier=1.25,
                beat_multiplier=0.8,
                tempo_cue="slower",
                reason="pain_auto_downshift",
                    rule_id="pain_safety_downshift",
                    state_reason="safety=pain",
                )
            ],
            requires_confirmation=True,
            confirmation_type="pain",
        )

    if any(action in NO_ACTIONS for action in actions):
        if (
            actions == ["no_action"]
            and decision_engine is not None
            and getattr(decision_engine, "_fallback_parser", None) is not None
        ):
            fallback = decision_engine.decide(understanding.raw_text, state)
            if fallback.intent != "unknown":
                return fallback
        return DecisionResult(
            intent=understanding.intent,
            coach_reply=understanding.reply,
        )

    mapped_actions = []
    for action in actions:
        adjustment = _action_to_adjustment(action, state, understanding)
        if adjustment is not None:
            mapped_actions.append(adjustment)

    if mapped_actions:
        return DecisionResult(
            intent=understanding.intent,
            coach_reply=understanding.reply,
            actions=mapped_actions,
        )

    fallback = decision_engine.decide(understanding.raw_text, state) if decision_engine is not None else DecisionResult("unknown", "")
    if fallback.intent != "unknown":
        return fallback

    return DecisionResult(
        intent="unknown",
        coach_reply="",
    )


def _find_replacement_exercise(state: SessionState) -> Optional[str]:
    if not state.workout_plan["exercises"]:
        return None

    current = state.workout_plan["exercises"][state.current_exercise]
    current_name = current["exercise_name"]
    current_def = next((ex for ex in state.exercise_library if ex.get("name") == current_name), None)
    if not current_def:
        return None

    current_targets = set(current_def.get("target_muscles", []))
    existing = {ex["exercise_name"] for ex in state.workout_plan["exercises"]}

    candidates: List[Tuple[int, str]] = []
    for ex in state.exercise_library:
        candidate_name = ex.get("name")
        if not candidate_name or candidate_name == current_name:
            continue
        if candidate_name in existing:
            continue
        overlap = len(current_targets.intersection(set(ex.get("target_muscles", []))))
        if overlap <= 0:
            continue
        candidates.append((overlap, candidate_name))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _derive_state_actions(state: SessionState) -> List[AdjustmentAction]:
    condition = state.user_condition

    if condition.fatigue_level == "high" or condition.difficulty_level == "hard":
        return [
            AdjustmentAction(
                action_type="adjust_intensity",
                set_delta=-1,
                rest_multiplier=1.2,
                beat_multiplier=0.8,
                tempo_cue="slower",
                reason="state_high_fatigue_or_hard",
                rule_id="state_hard_downshift",
                state_reason=(
                    f"fatigue={condition.fatigue_level}, "
                    f"difficulty={condition.difficulty_level}"
                ),
            )
        ]

    if condition.fatigue_level == "medium":
        return [
            AdjustmentAction(
                action_type="adjust_intensity",
                set_delta=0,
                rest_multiplier=1.1,
                beat_multiplier=0.9,
                tempo_cue="slower" if state.tempo_cue == "normal" else state.tempo_cue,
                reason="state_medium_fatigue_recovery",
                rule_id="state_medium_recovery",
                state_reason="fatigue=medium",
            )
        ]

    if condition.fatigue_level == "low" and condition.difficulty_level == "easy":
        return [
            AdjustmentAction(
                action_type="adjust_intensity",
                set_delta=0,
                rest_multiplier=0.9,
                beat_multiplier=1.1,
                tempo_cue="faster",
                reason="state_low_fatigue_easy_progress",
                rule_id="state_easy_progress",
                state_reason="fatigue=low and difficulty=easy",
            )
        ]

    return []


def _apply_adjustment_action(
    action: AdjustmentAction,
    state: SessionState,
    io: ExecutorIO,
    confidence: float = 0.0,
    source_actions: Optional[List[str]] = None,
) -> None:
    if action.action_type == "stop":
        state.is_active = False
        state.end_reason = action.reason or "user_stop"
        state.log_event("session_stop", {"reason": state.end_reason})
        return

    if action.action_type == "replace_exercise":
        target_name = action.replacement_exercise
        if not target_name:
            return

        replacement_def = next((ex for ex in state.exercise_library if ex.get("name") == target_name), None)
        if not replacement_def:
            return

        exercise = state.workout_plan["exercises"][state.current_exercise]
        before_name = exercise["exercise_name"]
        before_display_name = exercise.get("display_name", before_name)
        exercise_demo_library = load_exercise_demo_library()
        details = exercise_demo_library.get(
            target_name,
            {
                "exercise_name": target_name.replace("_", " ").title(),
                "instructions": ["保持动作安全、稳定、受控。"],
            },
        )
        exercise["exercise_name"] = target_name
        exercise["display_name"] = details["exercise_name"]
        exercise["instructions"] = details["instructions"]
        exercise["avg_set_time"] = int(replacement_def.get("avg_set_time", exercise["avg_set_time"]))
        exercise["default_beat_hz"] = _normalize_beat_hz(details.get("default_beat_hz"))
        exercise["beat_source"] = "exercise_demo_library" if "default_beat_hz" in details else "fallback"

        state.log_adjustment(
            {
                "action_type": action.action_type,
                "reason": action.reason,
                "rule_id": action.rule_id,
                "state_reason": action.state_reason,
                "source_actions": source_actions or [],
                "confidence": confidence,
                "replacement": {"before": before_name, "after": target_name},
            }
        )
        state.log_event(
            "exercise_replaced",
            {
                "before": before_name,
                "after": target_name,
                "rule_id": action.rule_id,
                "state_reason": action.state_reason,
            },
        )
        _emit_runtime_event(
            io,
            "exercise_replaced",
            {
                "before": before_name,
                "after": target_name,
                "before_display_name": before_display_name,
                "exercise_index": state.current_exercise,
                "exercise": exercise,
                "reason": action.reason,
                "rule_id": action.rule_id,
            },
        )
        _show_demo(io, exercise)
        return

    if action.action_type == "skip_current_exercise":
        if not state.workout_plan["exercises"]:
            return
        if state.current_exercise >= len(state.workout_plan["exercises"]):
            return

        exercise = state.workout_plan["exercises"][state.current_exercise]
        before_sets = int(exercise["total_sets"])
        # Only skip the remaining sets of the current exercise.
        # Keep rest_seconds unchanged and do not touch other exercises.
        effective_sets = _clamp_int(max(state.current_set, 1), MIN_SETS, MAX_SETS)
        exercise["total_sets"] = min(before_sets, effective_sets)

        state.log_adjustment(
            {
                "action_type": action.action_type,
                "reason": action.reason,
                "rule_id": action.rule_id,
                "state_reason": action.state_reason,
                "source_actions": source_actions or [],
                "confidence": confidence,
                "exercise_change": {
                    "exercise_index": state.current_exercise,
                    "exercise_name": exercise["exercise_name"],
                    "before": {"total_sets": before_sets, "rest_seconds": exercise["rest_seconds"]},
                    "after": {
                        "total_sets": exercise["total_sets"],
                        "rest_seconds": exercise["rest_seconds"],
                    },
                },
            }
        )
        state.log_event(
            "exercise_skipped_current",
            {
                "exercise_index": state.current_exercise,
                "exercise_name": exercise["exercise_name"],
                "before_total_sets": before_sets,
                "after_total_sets": exercise["total_sets"],
                "rule_id": action.rule_id,
            },
        )
        _emit_runtime_event(
            io,
            "exercise_skipped_current",
            {
                "exercise_index": state.current_exercise,
                "exercise_name": exercise["exercise_name"],
                "before_total_sets": before_sets,
                "after_total_sets": exercise["total_sets"],
                "reason": action.reason,
                "rule_id": action.rule_id,
            },
        )
        return

    if action.action_type != "adjust_intensity":
        return

    # Numeric live-state changes are clamped here, keeping adjustment rules
    # deterministic even when an upstream model returns an extreme request.
    safe_set_delta = _clamp_int(action.set_delta, MIN_SET_DELTA, MAX_SET_DELTA)
    safe_multiplier = max(MIN_REST_MULTIPLIER, min(MAX_REST_MULTIPLIER, action.rest_multiplier))
    safe_beat_multiplier = max(MIN_BEAT_MULTIPLIER, min(MAX_BEAT_MULTIPLIER, action.beat_multiplier))
    before_set_delta = state.set_delta
    before_rest_multiplier = state.rest_multiplier
    before_beat_multiplier = state.beat_multiplier
    before_tempo_cue = state.tempo_cue
    state.set_delta = _clamp_int(state.set_delta + safe_set_delta, MIN_SET_DELTA, MAX_SET_DELTA)
    state.rest_multiplier = max(
        MIN_REST_MULTIPLIER,
        min(MAX_REST_MULTIPLIER, state.rest_multiplier * safe_multiplier),
    )
    state.beat_multiplier = max(
        MIN_BEAT_MULTIPLIER,
        min(MAX_BEAT_MULTIPLIER, state.beat_multiplier * safe_beat_multiplier),
    )
    if action.tempo_cue:
        state.tempo_cue = action.tempo_cue

    exercise_changes = []
    for idx, exercise in enumerate(state.workout_plan["exercises"]):
        if idx < state.current_exercise:
            continue

        before_sets = exercise["total_sets"]
        before_rest = exercise["rest_seconds"]
        adjusted_sets = _clamp_int(exercise["total_sets"] + safe_set_delta, MIN_SETS, MAX_SETS)
        adjusted_rest = _clamp_int(
            int(round(exercise["rest_seconds"] * safe_multiplier)),
            MIN_REST_SECONDS,
            MAX_REST_SECONDS,
        )
        exercise["total_sets"] = adjusted_sets
        exercise["rest_seconds"] = adjusted_rest
        exercise_changes.append(
            {
                "exercise_index": idx,
                "exercise_name": exercise["exercise_name"],
                "before": {"total_sets": before_sets, "rest_seconds": before_rest},
                "after": {"total_sets": adjusted_sets, "rest_seconds": adjusted_rest},
            }
        )

    state.log_adjustment(
        {
            "action_type": action.action_type,
            "set_delta": safe_set_delta,
            "rest_multiplier": safe_multiplier,
            "beat_multiplier": safe_beat_multiplier,
            "tempo_cue": action.tempo_cue,
            "reason": action.reason,
            "rule_id": action.rule_id,
            "state_reason": action.state_reason,
            "source_actions": source_actions or [],
            "confidence": confidence,
            "exercise_changes": exercise_changes,
        }
    )
    state.log_event(
        "adjustment_applied",
        {
            "set_delta": safe_set_delta,
            "rest_multiplier": safe_multiplier,
            "beat_multiplier": safe_beat_multiplier,
            "tempo_cue": action.tempo_cue,
            "phase": state.phase,
        },
    )
    _emit_runtime_event(
        io,
        "adjustment_applied",
        {
            "set_delta": safe_set_delta,
            "rest_multiplier": safe_multiplier,
            "tempo_cue": state.tempo_cue,
            "before_set_delta": before_set_delta,
            "after_set_delta": state.set_delta,
            "before_rest_multiplier": before_rest_multiplier,
            "after_rest_multiplier": state.rest_multiplier,
            "before_beat_multiplier": before_beat_multiplier,
            "after_beat_multiplier": state.beat_multiplier,
            "before_tempo_cue": before_tempo_cue,
            "after_tempo_cue": state.tempo_cue,
            **_current_beat_payload(state, active_phase=state.phase == "active_set"),
            "reason": action.reason,
            "rule_id": action.rule_id,
            "source_actions": source_actions or [],
            "confidence": confidence,
            "exercise_changes": exercise_changes,
        },
    )


def _handle_user_message(
    user_text: str,
    state: SessionState,
    io: ExecutorIO,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
    feedback_sequence_id: Optional[int] = None,
) -> None:
    cleaned = user_text.strip()
    if not cleaned:
        return
    if feedback_sequence_id is None:
        state.feedback_sequence += 1
        feedback_sequence_id = state.feedback_sequence
    consume_feedback_context = getattr(io, "consume_feedback_context", None)
    feedback_context = consume_feedback_context(cleaned) if callable(consume_feedback_context) else {}
    feedback_id = str(feedback_context.get("feedback_id") or uuid4().hex[:12])
    feedback_processing_started_at = time.perf_counter()
    asr_latency_ms = int(feedback_context.get("asr_latency_ms", 0))

    def total_response_time_ms() -> int:
        return asr_latency_ms + int((time.perf_counter() - feedback_processing_started_at) * 1000)

    state.active_feedback_id = feedback_id
    trace = {
        "condition_id": state.condition_id,
        "feedback_id": feedback_id,
        "transcript_id": feedback_context.get("transcript_id", ""),
        "input_method": feedback_context.get("input_method", ""),
        "submit_method": feedback_context.get("submit_method", ""),
        "timestamp": _timestamp(),
        "current_exercise": _build_feedback_state_snapshot(state)["current_exercise_name"],
        "phase": state.phase,
        "user_feedback": cleaned,
        "feedback_sequence_id": feedback_sequence_id,
        "parsed_intent": "",
        "llm_channel": "",
        "stale_reason": "",
        "structured_state": {},
        "system_adjustments": [],
        "asr_latency_ms": asr_latency_ms,
        "feedback_understanding_ms": None,
        "response_time_ms": None,
        "status": "received",
    }
    state.feedback_trace.append(trace)

    # This is the only entry point that converts raw feedback text into state
    # changes. Fast rules, LLM fallback, stale-result protection, and logging
    # all pass through here so the experiment logs stay comparable.
    _emit_runtime_event(
        io,
        "feedback_received",
        {
            "text": cleaned,
            "phase": state.phase,
            "input_method": feedback_context.get("input_method", ""),
            "submit_method": feedback_context.get("submit_method", ""),
        },
    )
    state.log_conversation({"role": "user", "content": cleaned})
    state.log_event("user_message", {"content": cleaned})

    if state.awaiting_stop_confirmation:
        _handle_stop_confirmation(cleaned, state, io)
        trace["parsed_intent"] = "stop_confirmation"
        trace["response_time_ms"] = total_response_time_ms()
        trace["status"] = "confirmation_handled"
        state.active_feedback_id = ""
        return

    if state.awaiting_pain_confirmation:
        _handle_pain_confirmation(cleaned, state, io)
        trace["parsed_intent"] = "pain_confirmation"
        trace["response_time_ms"] = total_response_time_ms()
        trace["status"] = "confirmation_handled"
        state.active_feedback_id = ""
        return

    context_before_understanding = {
        "condition_id": state.condition_id,
        "current_exercise": state.current_exercise,
        "phase": state.phase,
    }
    understanding_started_at = time.perf_counter()
    understanding = understand_feedback_fast_path(cleaned)
    if understanding is None:
        understanding = feedback_understander(cleaned, _build_feedback_state_snapshot(state), model=feedback_model)
    response_time_ms = int((time.perf_counter() - understanding_started_at) * 1000)

    if understanding is None or isinstance(understanding, FeedbackUnderstandingFailure):
        failure = understanding if isinstance(understanding, FeedbackUnderstandingFailure) else None
        _log_feedback_understanding_failure(state, io, cleaned, feedback_model, failure)
        state.log_event("feedback_response_time", {"response_time_ms": response_time_ms, "status": "failed"})
        trace["feedback_understanding_ms"] = response_time_ms
        trace["response_time_ms"] = total_response_time_ms()
        trace["status"] = "understanding_failed"
        state.active_feedback_id = ""
        return

    trace["llm_channel"] = understanding.llm_channel

    if understanding.llm_channel == "llm_fallback":
        stale_reason = ""
        if feedback_sequence_id != state.feedback_sequence:
            stale_reason = "superseded_by_newer_feedback"
        elif not state.is_active:
            stale_reason = "workout_inactive"
        elif state.condition_id != context_before_understanding["condition_id"]:
            stale_reason = "context_changed"
        elif state.current_exercise != context_before_understanding["current_exercise"]:
            stale_reason = "context_changed"
        elif state.phase != context_before_understanding["phase"]:
            stale_reason = "context_changed"

        if stale_reason:
            detail = {
                "raw_text": cleaned,
                "intent": understanding.intent,
                "actions": understanding.actions,
                "confidence": understanding.confidence,
                "reason": "stale_llm_result",
                "stale_reason": stale_reason,
                "feedback_id": feedback_id,
                "feedback_sequence_id": feedback_sequence_id,
                "latest_feedback_sequence_id": state.feedback_sequence,
                "llm_channel": understanding.llm_channel,
            }
            state.log_event("feedback_stale_ignored", detail)
            _emit_runtime_event(io, "feedback_ignored", detail)
            trace["parsed_intent"] = understanding.intent
            trace["structured_state"] = _structured_understanding_state(understanding)
            trace["feedback_understanding_ms"] = response_time_ms
            trace["response_time_ms"] = total_response_time_ms()
            trace["status"] = "stale_ignored"
            trace["stale_reason"] = stale_reason
            state.active_feedback_id = ""
            return

    if understanding.confidence <= LOW_FEEDBACK_CONFIDENCE:
        state.log_event("feedback_response_time", {"response_time_ms": response_time_ms, "status": "ignored"})
        _log_feedback_ignored(
            state,
            io,
            cleaned,
            understanding.actions,
            understanding.confidence,
            "low_confidence",
        )
        trace["parsed_intent"] = understanding.intent
        trace["structured_state"] = _structured_understanding_state(understanding)
        trace["feedback_understanding_ms"] = response_time_ms
        trace["response_time_ms"] = total_response_time_ms()
        trace["status"] = "ignored_low_confidence"
        state.active_feedback_id = ""
        return

    condition_changed = _apply_condition_from_understanding(understanding, state)
    trace["parsed_intent"] = understanding.intent
    trace["structured_state"] = _structured_understanding_state(understanding)
    trace["feedback_understanding_ms"] = response_time_ms
    state.log_event(
        "feedback_response_time",
        {
            "asr_latency_ms": asr_latency_ms,
            "feedback_understanding_ms": response_time_ms,
            "response_time_ms": total_response_time_ms(),
            "status": "understood",
            "llm_channel": understanding.llm_channel,
        },
    )
    _emit_runtime_event(
        io,
        "feedback_understood",
        {
            "fatigue_level": understanding.fatigue_level,
            "difficulty_level": understanding.difficulty_level,
            "preference": understanding.preference,
            "actions": understanding.actions,
            "safety": understanding.safety,
            "confidence": understanding.confidence,
            "reason": understanding.reason,
            "raw_text": understanding.raw_text,
            "condition_changed": condition_changed,
            "llm_channel": understanding.llm_channel,
            "feedback_sequence_id": feedback_sequence_id,
            "feedback_understanding_ms": response_time_ms,
            "response_time_ms": total_response_time_ms(),
        },
    )
    decision = _build_decision_from_actions(understanding, state, decision_engine)
    state.log_event("decision", {"intent": decision.intent})

    all_actions = decision.actions
    if (
        decision.intent in SILENT_NO_ACTION_INTENTS
        and not all_actions
        and not decision.requires_confirmation
        and not decision.coach_reply
    ):
        _log_feedback_ignored(
            state,
            io,
            cleaned,
            understanding.actions,
            understanding.confidence,
            "no_applicable_adjustment",
        )
        trace["status"] = "ignored_no_adjustment"
        trace["response_time_ms"] = total_response_time_ms()
        state.active_feedback_id = ""
        return

    _send_coach_reply(io, state, decision.coach_reply, urgent=decision.requires_confirmation)

    for action in all_actions:
        _apply_adjustment_action(
            action,
            state,
            io,
            confidence=understanding.confidence,
            source_actions=understanding.actions,
        )
    trace["system_adjustments"] = [
        item for item in state.adjustment_log if item.get("feedback_id") == feedback_id
    ]
    trace["status"] = "handled"
    trace["response_time_ms"] = total_response_time_ms()

    if decision.requires_confirmation:
        set_confirmation = getattr(io, "set_pending_confirmation", None)
        if callable(set_confirmation):
            set_confirmation(decision.confirmation_type)
        if decision.confirmation_type == "pain":
            state.awaiting_pain_confirmation = True
        elif decision.confirmation_type == "stop":
            state.awaiting_stop_confirmation = True
    state.active_feedback_id = ""


def _drain_messages(
    state: SessionState,
    io: ExecutorIO,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
    feedback_worker: Optional[FeedbackWorker] = None,
) -> None:
    if _consume_external_stop_request(state, io):
        return
    if not state.feedback_enabled:
        return
    while state.is_active:
        if _consume_external_stop_request(state, io):
            return
        message = io.poll_user_input()
        if message is None:
            return
        if feedback_worker is not None:
            feedback_worker.submit(message)
        else:
            _handle_user_message(message, state, io, decision_engine, feedback_understander, feedback_model)


def _consume_external_stop_request(state: SessionState, io: ExecutorIO) -> bool:
    consume_stop = getattr(io, "consume_condition_stop_request", None)
    if not callable(consume_stop):
        return False
    reason = consume_stop()
    if not reason:
        return False
    state.is_active = False
    state.end_reason = reason
    state.log_event("external_condition_stop", {"reason": reason})
    _emit_runtime_event(io, "condition_stop_requested", {"reason": reason})
    return True


def _run_timed_phase(
    seconds: int,
    phase: str,
    state: SessionState,
    io: ExecutorIO,
    clock: Clock,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
    feedback_worker: Optional[FeedbackWorker] = None,
) -> None:
    state.phase = phase
    state.seconds_remaining = seconds
    _emit_runtime_event(
        io,
        "phase_start",
        {
            "phase": phase,
            "seconds": seconds,
            "current_round": state.current_round + 1,
            "current_exercise": state.current_exercise + 1,
            "current_set": state.current_set,
            "tempo_cue": state.tempo_cue,
            **_current_beat_payload(state),
        },
    )

    while state.is_active and state.seconds_remaining > 0:
        if _consume_external_stop_request(state, io):
            break
        _emit_runtime_event(
            io,
            "phase_tick",
            {
                "phase": phase,
                "seconds_remaining": state.seconds_remaining,
                "current_round": state.current_round + 1,
                "current_exercise": state.current_exercise + 1,
                "current_set": state.current_set,
                "tempo_cue": state.tempo_cue,
                "set_delta": state.set_delta,
                "rest_multiplier": state.rest_multiplier,
                **_current_beat_payload(state, active_phase=phase == "active_set"),
            },
        )
        _drain_messages(state, io, decision_engine, feedback_understander, feedback_model, feedback_worker)
        if not state.is_active:
            break
        clock.sleep(1)
        state.seconds_remaining -= 1

    _drain_messages(state, io, decision_engine, feedback_understander, feedback_model, feedback_worker)


def _run_pre_exercise_preview_wait(
    seconds: int,
    state: SessionState,
    io: ExecutorIO,
    clock: Clock,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
    feedback_worker: Optional[FeedbackWorker] = None,
) -> None:
    if seconds <= 0:
        return

    state.phase = "pre_exercise_preview_wait"
    state.seconds_remaining = seconds
    state.log_event(
        "pre_exercise_preview_wait_start",
        {
            "seconds": seconds,
            "round": state.current_round,
            "exercise": state.current_exercise,
        },
    )

    while state.is_active and state.seconds_remaining > 0:
        if _consume_external_stop_request(state, io):
            break
        _emit_runtime_event(
            io,
            "phase_tick",
            {
                "phase": state.phase,
                "seconds_remaining": state.seconds_remaining,
                "current_round": state.current_round + 1,
                "current_exercise": state.current_exercise + 1,
                "current_set": state.current_set,
                "tempo_cue": state.tempo_cue,
                "set_delta": state.set_delta,
                "rest_multiplier": state.rest_multiplier,
                **_current_beat_payload(state, active_phase=False),
            },
        )
        _drain_messages(state, io, decision_engine, feedback_understander, feedback_model, feedback_worker)
        if not state.is_active:
            break
        clock.sleep(1)
        state.seconds_remaining -= 1

    _drain_messages(state, io, decision_engine, feedback_understander, feedback_model, feedback_worker)
    state.log_event(
        "pre_exercise_preview_wait_end",
        {
            "round": state.current_round,
            "exercise": state.current_exercise,
            "is_active": state.is_active,
        },
    )


def _show_demo(io: ExecutorIO, exercise: Dict) -> None:
    io.send(
        f"Exercise: {exercise['display_name']}\n"
        "Instructions: " + "; ".join(exercise["instructions"])
    )
    wait_for_speech = getattr(io, "wait_for_speech", None)
    if callable(wait_for_speech):
        wait_for_speech("exercise_instructions")


def export_session_log(state: SessionState, path: str) -> None:
    adjustment_reason_trace = []
    for item in state.adjustment_log:
        adjustment_reason_trace.append(
            {
                "condition_id": state.condition_id,
                "action_type": item.get("action_type"),
                "rule_id": item.get("rule_id", ""),
                "state_reason": item.get("state_reason", ""),
                "reason": item.get("reason", ""),
            }
        )

    payload = {
        "condition_id": state.condition_id,
        "experiment_metadata": state.experiment_metadata,
        "summary": {
            "condition_id": state.condition_id,
            "is_active": state.is_active,
            "end_reason": state.end_reason,
            "session_started_at": state.session_started_at,
            "session_ended_at": state.session_ended_at,
            "actual_duration_seconds": state.actual_duration_seconds,
            "current_round": state.current_round,
            "current_exercise": state.current_exercise,
            "current_set": state.current_set,
            "phase": state.phase,
        },
        "initial_workout_plan": state.initial_workout_plan,
        "initial_intent": state.initial_intent,
        "final_workout_plan": state.workout_plan,
        "user_condition_final": state.user_condition.snapshot(),
        "condition_timeline": state.condition_log,
        "adjustments": state.adjustment_log,
        "adjustment_reason_trace": adjustment_reason_trace,
        "feedback_understanding_failures": [
            {"condition_id": state.condition_id, **event["detail"]}
            for event in state.event_log
            if event.get("type") == "feedback_understanding_failed"
        ],
        "transcripts": state.transcript_log,
        "feedback_traces": state.feedback_trace,
        "conversation": state.conversation_log,
        "events": state.event_log,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_adaptive_workout(
    workout_plan: Dict,
    io: Optional[ExecutorIO] = None,
    clock: Optional[Clock] = None,
    decision_engine: Optional[DecisionEngine] = None,
    feedback_understander: Optional[FeedbackUnderstander] = None,
    feedback_model: str = "frob/qwen3.5-instruct:4b",
    log_path: Optional[str] = None,
    initial_intent: Optional[Dict] = None,
    exercise_library: Optional[List[Dict]] = None,
    condition_id: str = "",
    experiment_metadata: Optional[Dict] = None,
    feedback_enabled: bool = True,
    close_io: bool = True,
) -> SessionState:
    runtime_io = io or CLIExecutorIO()
    runtime_clock = clock or RealClock()
    runtime_decision_engine = decision_engine or RuleFirstDecisionEngine()
    runtime_feedback_understander = feedback_understander or understand_feedback
    feedback_worker: Optional[FeedbackWorker] = None

    translated_plan = translate_plan_for_demo(workout_plan)
    session_started_clock = runtime_clock.now()
    state = SessionState(
        workout_plan=translated_plan,
        condition_id=condition_id,
        experiment_metadata=deepcopy(experiment_metadata or {}),
        feedback_enabled=feedback_enabled,
        session_started_at=_timestamp(),
        initial_workout_plan=deepcopy(translated_plan),
        initial_intent=deepcopy(initial_intent or {}),
        exercise_library=deepcopy(exercise_library or []),
    )
    state.log_event("session_start", {"clock_time": session_started_clock})
    _emit_runtime_event(
        runtime_io,
        "workout_started",
        {"workout_plan": deepcopy(state.workout_plan), "initial_intent": deepcopy(state.initial_intent)},
    )
    if feedback_enabled and getattr(runtime_io, "nonblocking_feedback", False):
        feedback_worker = FeedbackWorker(
            state=state,
            io=runtime_io,
            decision_engine=runtime_decision_engine,
            feedback_understander=runtime_feedback_understander,
            feedback_model=feedback_model,
        )
    try:
        rounds = state.workout_plan["rounds"]
        for round_idx in range(rounds):
            if not state.is_active:
                break
            state.current_round = round_idx
            _emit_runtime_event(
                runtime_io,
                "round_start",
                {"current_round": round_idx + 1, "total_rounds": rounds},
            )
            runtime_io.send(f"Round {round_idx + 1}/{rounds} start.")

            for ex_idx, exercise in enumerate(state.workout_plan["exercises"]):
                if not state.is_active:
                    break
                state.current_exercise = ex_idx
                _emit_runtime_event(
                    runtime_io,
                    "exercise_start",
                    {
                        "exercise_index": ex_idx,
                        "current_exercise": ex_idx + 1,
                        "total_exercises": len(state.workout_plan["exercises"]),
                        "exercise": deepcopy(exercise),
                        "current_round": round_idx + 1,
                        "total_rounds": rounds,
                    },
                )
                _show_demo(runtime_io, exercise)
                _run_pre_exercise_preview_wait(
                    seconds=PRE_EXERCISE_PREVIEW_WAIT_SECONDS,
                    state=state,
                    io=runtime_io,
                    clock=runtime_clock,
                    decision_engine=runtime_decision_engine,
                    feedback_understander=runtime_feedback_understander,
                    feedback_model=feedback_model,
                    feedback_worker=feedback_worker,
                )
                if not state.is_active:
                    break

                set_index = 0
                while state.is_active and set_index < exercise["total_sets"]:
                    state.current_set = set_index + 1
                    _emit_runtime_event(
                        runtime_io,
                        "set_start",
                        {
                            "current_set": state.current_set,
                            "total_sets": exercise["total_sets"],
                            "exercise_index": ex_idx,
                            "exercise_name": exercise["exercise_name"],
                            "tempo_cue": state.tempo_cue,
                            **_current_beat_payload(state, active_phase=True),
                            "current_round": round_idx + 1,
                            "total_rounds": rounds,
                        },
                    )
                    runtime_io.send(
                        f"Set {state.current_set}/{exercise['total_sets']} start "
                        f"(tempo: {state.tempo_cue})"
                    )
                    _run_timed_phase(
                        seconds=exercise["avg_set_time"],
                        phase="active_set",
                        state=state,
                        io=runtime_io,
                        clock=runtime_clock,
                        decision_engine=runtime_decision_engine,
                        feedback_understander=runtime_feedback_understander,
                        feedback_model=feedback_model,
                        feedback_worker=feedback_worker,
                    )
                    if not state.is_active:
                        break
                    _emit_runtime_event(
                        runtime_io,
                        "set_end",
                        {
                            "current_set": state.current_set,
                            "total_sets": exercise["total_sets"],
                            "exercise_index": ex_idx,
                            "exercise_name": exercise["exercise_name"],
                        },
                    )
                    runtime_io.send("Set finished.")
                    set_index += 1

                    if set_index < exercise["total_sets"]:
                        _emit_runtime_event(
                            runtime_io,
                            "rest_start",
                            {
                                "rest_type": "between_sets",
                                "seconds": exercise["rest_seconds"],
                                "current_set": set_index,
                                "total_sets": exercise["total_sets"],
                                "exercise_index": ex_idx,
                                "exercise_name": exercise["exercise_name"],
                            },
                        )
                        runtime_io.send(f"Rest for {exercise['rest_seconds']} seconds.")
                        _run_timed_phase(
                            seconds=exercise["rest_seconds"],
                            phase="between_sets_rest",
                            state=state,
                            io=runtime_io,
                            clock=runtime_clock,
                            decision_engine=runtime_decision_engine,
                            feedback_understander=runtime_feedback_understander,
                            feedback_model=feedback_model,
                            feedback_worker=feedback_worker,
                        )

                rest_after_exercise = int(exercise.get("rest_after_exercise_seconds", 0))
                if state.is_active and rest_after_exercise > 0:
                    _emit_runtime_event(
                        runtime_io,
                        "rest_start",
                        {
                            "rest_type": "after_exercise",
                            "seconds": rest_after_exercise,
                            "exercise_index": ex_idx,
                            "exercise_name": exercise["exercise_name"],
                        },
                    )
                    runtime_io.send(f"Rest for {rest_after_exercise} seconds.")
                    _run_timed_phase(
                        seconds=rest_after_exercise,
                        phase="after_exercise_rest",
                        state=state,
                        io=runtime_io,
                        clock=runtime_clock,
                        decision_engine=runtime_decision_engine,
                        feedback_understander=runtime_feedback_understander,
                        feedback_model=feedback_model,
                        feedback_worker=feedback_worker,
                    )

            if state.is_active and round_idx < rounds - 1:
                rest_between_rounds = state.workout_plan["rest_between_rounds"]
                runtime_io.send("Round finished.")
                _emit_runtime_event(
                    runtime_io,
                    "rest_start",
                    {
                        "rest_type": "between_rounds",
                        "seconds": rest_between_rounds,
                        "current_round": round_idx + 1,
                        "total_rounds": rounds,
                    },
                )
                runtime_io.send(f"Rest between rounds for {rest_between_rounds} seconds.")
                _run_timed_phase(
                    seconds=rest_between_rounds,
                    phase="between_rounds_rest",
                    state=state,
                    io=runtime_io,
                    clock=runtime_clock,
                    decision_engine=runtime_decision_engine,
                    feedback_understander=runtime_feedback_understander,
                    feedback_model=feedback_model,
                    feedback_worker=feedback_worker,
                )

        if feedback_worker is not None:
            feedback_worker.close()
            feedback_worker = None

        if state.is_active:
            state.end_reason = "completed"
            _emit_runtime_event(runtime_io, "session_end", {"reason": state.end_reason})
            runtime_io.send("Workout complete.")
        else:
            _emit_runtime_event(runtime_io, "session_end", {"reason": state.end_reason})
            runtime_io.send(f"Workout stopped: {state.end_reason}.")
    finally:
        if feedback_worker is not None:
            feedback_worker.close()
        session_ended_clock = runtime_clock.now()
        state.session_ended_at = _timestamp()
        state.actual_duration_seconds = max(0.0, session_ended_clock - session_started_clock)
        get_transcript_log = getattr(runtime_io, "get_transcript_log", None)
        if callable(get_transcript_log):
            state.transcript_log = get_transcript_log()
        state.log_event("session_end", {"clock_time": session_ended_clock, "reason": state.end_reason})
        if log_path:
            export_session_log(state, log_path)
            _emit_runtime_event(runtime_io, "log_saved", {"path": log_path})
            runtime_io.send(f"Session log saved: {log_path}")
        if close_io:
            runtime_io.close()

    return state

