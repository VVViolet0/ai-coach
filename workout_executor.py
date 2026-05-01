import json
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union

from feedback_understanding import (
    FeedbackUnderstandingFailure,
    FeedbackUnderstandingResult,
    understand_feedback,
)


MIN_REST_SECONDS = 5
MAX_REST_SECONDS = 180
MIN_SETS = 1
MAX_SETS = 8
PRE_EXERCISE_PREVIEW_WAIT_SECONDS = 3
MIN_REST_MULTIPLIER = 0.5
MAX_REST_MULTIPLIER = 2.0
MIN_SET_DELTA = -3
MAX_SET_DELTA = 3
YES_WORDS = {
    "yes",
    "y",
    "continue",
    "go",
    "ok",
    "okay",
    "是",
    "对",
    "继续",
    "可以",
    "好的",
    "好",
    "没问题",
    "沒問題",
    "行",
    "yeah",
    "yep",
    "耶",
}
NO_WORDS = {
    "no",
    "n",
    "stop",
    "quit",
    "end",
    "不",
    "不要",
    "停止",
    "结束",
    "結束",
    "停",
    "停下",
    "不继续",
    "不繼續",
    "不用",
    "nope",
    "否",
}
CONFIRMATION_STRIP_CHARS = " \t\r\n,，.。!！?？;；:："
EXERCISE_DEMO_LIBRARY_PATH = Path("libraries/exercise_demo_library.json")


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
    initial_workout_plan: Dict = field(default_factory=dict)
    initial_intent: Dict = field(default_factory=dict)
    exercise_library: List[Dict] = field(default_factory=list)
    current_round: int = 0
    current_exercise: int = 0
    current_set: int = 0
    phase: str = "idle"
    seconds_remaining: int = 0
    rest_multiplier: float = 1.0
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

    def log_event(self, event_type: str, detail: Dict) -> None:
        self.event_log.append({"type": event_type, "detail": detail})


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
                "instructions": ["Follow safe and controlled movement."],
            },
        )

        demo_exercise = {
            "exercise_name": name,
            "display_name": details["exercise_name"],
            "instructions": details["instructions"],
            "avg_set_time": ex["avg_set_time"],
            "total_sets": ex["total_sets"],
            "rest_seconds": ex["rest_seconds"],
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
                "I can adjust pace, reduce intensity, or stop. "
                "Try messages like 'too hard', 'faster', or 'stop'."
            ),
        )


class FeedbackWorker:
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
        self._messages: "queue.Queue[Optional[str]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, message: str) -> None:
        self._messages.put(message)

    def close(self) -> None:
        self._messages.put(None)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while True:
            message = self._messages.get()
            if message is None:
                return
            _emit_runtime_event(self._io, "feedback_processing_start", {"text": message})
            _handle_user_message(
                message,
                self._state,
                self._io,
                self._decision_engine,
                self._feedback_understander,
                self._feedback_model,
            )
            _emit_runtime_event(self._io, "feedback_processing_end", {"text": message})


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _normalize_confirmation_text(text: str) -> str:
    return text.strip().lower().strip(CONFIRMATION_STRIP_CHARS)


def _is_shorter_rest_request(understanding: FeedbackUnderstandingResult) -> bool:
    text = f"{understanding.raw_text} {understanding.reason}".lower()
    rest_terms = ("休息", "rest", "recovery", "break")
    shorter_terms = (
        "短",
        "少",
        "缩短",
        "減少",
        "减少",
        "快一点",
        "快一點",
        "shorter",
        "less",
        "reduce",
        "decrease",
        "cut",
    )
    return any(term in text for term in rest_terms) and any(term in text for term in shorter_terms)


def _emit_runtime_event(io: ExecutorIO, event_type: str, payload: Dict) -> None:
    send_event = getattr(io, "send_event", None)
    if callable(send_event):
        send_event(event_type, payload)


def _apply_condition_from_understanding(
    understanding: FeedbackUnderstandingResult,
    state: SessionState,
) -> bool:
    before = state.user_condition.snapshot()
    changed = (
        state.user_condition.fatigue_level != understanding.fatigue_level
        or state.user_condition.difficulty_level != understanding.difficulty_level
        or state.user_condition.preference != understanding.preference
    )

    state.user_condition.fatigue_level = understanding.fatigue_level
    state.user_condition.difficulty_level = understanding.difficulty_level
    state.user_condition.preference = understanding.preference
    state.user_condition.last_feedback_text = understanding.raw_text
    state.user_condition.updated_at_phase = state.phase

    condition_record = {
        "before_state": before,
        "after_state": state.user_condition.snapshot(),
        "trigger_text": understanding.raw_text,
        "llm_intent": understanding.intent,
        "confidence": understanding.confidence,
        "reason": understanding.reason,
        "llm_channel": understanding.llm_channel,
        "phase": state.phase,
    }
    state.condition_log.append(condition_record)
    state.log_event("condition_updated", condition_record)
    return changed


def _build_decision_from_understanding(
    understanding: FeedbackUnderstandingResult,
    state: SessionState,
    decision_engine: DecisionEngine,
) -> DecisionResult:
    if understanding.intent == "stop":
        return DecisionResult(
            intent="stop",
            coach_reply=(
                "我听到你可能想停止训练。现在要停止吗？请回答“是”或“否”。"
            ),
            requires_confirmation=True,
            confirmation_type="stop",
        )

    if understanding.intent == "pain":
        return DecisionResult(
            intent="pain",
            coach_reply=(
                "我听到你有疼痛或不适。我已经降低强度。你还要继续吗？请回答“继续”或“停止”。"
            ),
            actions=[
                AdjustmentAction(
                    action_type="adjust_intensity",
                    set_delta=-1,
                    rest_multiplier=1.25,
                    tempo_cue="slower",
                    reason="pain_auto_downshift",
                    rule_id="pain_safety_downshift",
                    state_reason="llm_intent=pain",
                )
            ],
            requires_confirmation=True,
            confirmation_type="pain",
        )

    if understanding.intent == "pace_up":
        return DecisionResult(
            intent="pace_up",
            coach_reply="Great. I will shorten upcoming rest and keep cues brisk.",
            actions=[
                AdjustmentAction(
                    action_type="adjust_intensity",
                    set_delta=0,
                    rest_multiplier=0.85,
                    tempo_cue="faster",
                    reason="pace_up",
                    rule_id="pace_up_direct",
                    state_reason="llm_intent=pace_up",
                )
            ],
        )

    if understanding.intent == "preference_like" and _is_shorter_rest_request(understanding):
        return DecisionResult(
            intent="pace_up",
            coach_reply="Got it. I will shorten upcoming rest a little.",
            actions=[
                AdjustmentAction(
                    action_type="adjust_intensity",
                    set_delta=0,
                    rest_multiplier=0.85,
                    tempo_cue="faster",
                    reason="preference_like_shorter_rest",
                    rule_id="preference_like_shorter_rest",
                    state_reason="preference_like with shorter-rest request",
                )
            ],
        )

    if understanding.intent == "pace_down":
        return DecisionResult(
            intent="pace_down",
            coach_reply="No problem. I will slow the pace and extend recovery.",
            actions=[
                AdjustmentAction(
                    action_type="adjust_intensity",
                    set_delta=0,
                    rest_multiplier=1.15,
                    tempo_cue="slower",
                    reason="pace_down",
                    rule_id="pace_down_direct",
                    state_reason="llm_intent=pace_down",
                )
            ],
        )

    if understanding.intent == "preference_dislike":
        return DecisionResult(
            intent="preference_dislike",
            coach_reply="Got it. I will skip this exercise and move to the next one.",
            actions=[
                AdjustmentAction(
                    action_type="skip_current_exercise",
                    reason="preference_dislike_skip_current",
                    rule_id="preference_dislike_skip_current",
                    state_reason="llm_intent=preference_dislike",
                )
            ],
        )

    if understanding.intent in {"fatigue", "preference_dislike", "preference_like"}:
        return DecisionResult(
            intent=understanding.intent,
            coach_reply="",
        )

    fallback = decision_engine.decide(understanding.raw_text, state)
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
    current_equipment = current_def.get("equipment", "none")
    existing = {ex["exercise_name"] for ex in state.workout_plan["exercises"]}

    candidates: List[Tuple[int, str]] = []
    for ex in state.exercise_library:
        candidate_name = ex.get("name")
        if not candidate_name or candidate_name == current_name:
            continue
        if candidate_name in existing:
            continue
        if ex.get("equipment") != current_equipment:
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
                tempo_cue=state.tempo_cue,
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
    source_intent: str = "",
    confidence: float = 0.0,
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
        exercise_demo_library = load_exercise_demo_library()
        details = exercise_demo_library.get(
            target_name,
            {
                "exercise_name": target_name.replace("_", " ").title(),
                "instructions": ["Follow safe and controlled movement."],
            },
        )
        exercise["exercise_name"] = target_name
        exercise["display_name"] = details["exercise_name"]
        exercise["instructions"] = details["instructions"]
        exercise["avg_set_time"] = int(replacement_def.get("avg_set_time", exercise["avg_set_time"]))

        state.adjustment_log.append(
            {
                "action_type": action.action_type,
                "reason": action.reason,
                "rule_id": action.rule_id,
                "state_reason": action.state_reason,
                "source_intent": source_intent,
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
                "exercise_index": state.current_exercise,
                "exercise": exercise,
                "reason": action.reason,
                "rule_id": action.rule_id,
            },
        )
        io.send(f"[Adjustment] replaced exercise: {before_name} -> {target_name}")
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

        state.adjustment_log.append(
            {
                "action_type": action.action_type,
                "reason": action.reason,
                "rule_id": action.rule_id,
                "state_reason": action.state_reason,
                "source_intent": source_intent,
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
        io.send(f"[Adjustment] skipped remaining sets for current exercise: {exercise['exercise_name']}")
        return

    if action.action_type != "adjust_intensity":
        return

    safe_set_delta = _clamp_int(action.set_delta, MIN_SET_DELTA, MAX_SET_DELTA)
    safe_multiplier = max(MIN_REST_MULTIPLIER, min(MAX_REST_MULTIPLIER, action.rest_multiplier))
    state.set_delta = _clamp_int(state.set_delta + safe_set_delta, MIN_SET_DELTA, MAX_SET_DELTA)
    state.rest_multiplier = max(
        MIN_REST_MULTIPLIER,
        min(MAX_REST_MULTIPLIER, state.rest_multiplier * safe_multiplier),
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

    state.adjustment_log.append(
        {
            "action_type": action.action_type,
            "set_delta": safe_set_delta,
            "rest_multiplier": safe_multiplier,
            "tempo_cue": action.tempo_cue,
            "reason": action.reason,
            "rule_id": action.rule_id,
            "state_reason": action.state_reason,
            "source_intent": source_intent,
            "confidence": confidence,
            "exercise_changes": exercise_changes,
        }
    )
    state.log_event(
        "adjustment_applied",
        {
            "set_delta": safe_set_delta,
            "rest_multiplier": safe_multiplier,
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
            "reason": action.reason,
            "rule_id": action.rule_id,
            "source_intent": source_intent,
            "confidence": confidence,
            "exercise_changes": exercise_changes,
        },
    )
    io.send(
        f"[Adjustment] next blocks updated: set_delta={safe_set_delta}, "
        f"rest_multiplier={safe_multiplier:.2f}, tempo={state.tempo_cue}"
    )


def _handle_user_message(
    user_text: str,
    state: SessionState,
    io: ExecutorIO,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
) -> None:
    cleaned = user_text.strip()
    if not cleaned:
        return

    _emit_runtime_event(
        io,
        "feedback_received",
        {"text": cleaned, "phase": state.phase},
    )
    state.conversation_log.append({"role": "user", "content": cleaned})
    state.log_event("user_message", {"content": cleaned})

    if state.awaiting_stop_confirmation:
        lowered = _normalize_confirmation_text(cleaned)
        if lowered in YES_WORDS:
            state.awaiting_stop_confirmation = False
            state.is_active = False
            state.end_reason = "user_requested_stop"
            io.send("好的，现在停止训练。")
            state.log_event("stop_confirmation", {"decision": "stop"})
            return
        if lowered in NO_WORDS:
            state.awaiting_stop_confirmation = False
            io.send("好的，我们继续。如果需要，我可以随时降低强度。")
            state.log_event("stop_confirmation", {"decision": "continue"})
            return
        io.send("请回答“是”来停止，或回答“否”来继续。")
        state.log_event("stop_confirmation", {"decision": "unclear"})
        return

    if state.awaiting_pain_confirmation:
        lowered = _normalize_confirmation_text(cleaned)
        if lowered in YES_WORDS:
            state.awaiting_pain_confirmation = False
            io.send("好的，我们用更轻、更安全的节奏继续。")
            state.log_event("pain_confirmation", {"decision": "continue"})
            return
        if lowered in NO_WORDS:
            state.awaiting_pain_confirmation = False
            state.is_active = False
            state.end_reason = "user_stopped_after_pain"
            io.send("训练已停止。请先休息，如果需要请及时寻求医疗建议。")
            state.log_event("pain_confirmation", {"decision": "stop"})
            return

        io.send("请回答“继续”来继续训练，或回答“停止”来结束训练。")
        state.log_event("pain_confirmation", {"decision": "unclear"})
        return

    state_snapshot = {
        "phase": state.phase,
        "current_round": state.current_round,
        "current_exercise": state.current_exercise,
        "current_set": state.current_set,
        "user_condition": state.user_condition.snapshot(),
        "tempo_cue": state.tempo_cue,
        "set_delta": state.set_delta,
        "rest_multiplier": state.rest_multiplier,
    }
    understanding = feedback_understander(cleaned, state_snapshot, model=feedback_model)

    if understanding is None or isinstance(understanding, FeedbackUnderstandingFailure):
        failure_detail = {
            "raw_text": cleaned,
            "model": feedback_model,
        }
        if isinstance(understanding, FeedbackUnderstandingFailure):
            failure_detail["error_summary"] = understanding.error_summary
            failure_detail["tried_channels"] = understanding.tried_channels
        else:
            failure_detail["error_summary"] = "understander returned None"
            failure_detail["tried_channels"] = []

        state.log_event("feedback_understanding_failed", failure_detail)
        io.send("[Coach] I could not parse that feedback this time. We'll continue and keep monitoring.")
        state.conversation_log.append(
            {"role": "coach", "content": "I could not parse that feedback this time."}
        )
        return

    condition_changed = _apply_condition_from_understanding(understanding, state)
    _emit_runtime_event(
        io,
        "feedback_understood",
        {
            "intent": understanding.intent,
            "fatigue_level": understanding.fatigue_level,
            "difficulty_level": understanding.difficulty_level,
            "preference": understanding.preference,
            "confidence": understanding.confidence,
            "reason": understanding.reason,
            "raw_text": understanding.raw_text,
            "condition_changed": condition_changed,
        },
    )
    decision = _build_decision_from_understanding(understanding, state, decision_engine)
    state.log_event("decision", {"intent": decision.intent})
    if decision.coach_reply:
        io.send(f"[Coach] {decision.coach_reply}")
        state.conversation_log.append({"role": "coach", "content": decision.coach_reply})

    state_actions: List[AdjustmentAction] = []
    if condition_changed and decision.intent not in {
        "stop",
        "pain",
        "neutral",
        "unknown",
        "pace_up",
        "pace_down",
        "preference_dislike",
        "preference_like",
    }:
        state_actions = _derive_state_actions(state)

    for action in decision.actions + state_actions:
        _apply_adjustment_action(
            action,
            state,
            io,
            source_intent=understanding.intent,
            confidence=understanding.confidence,
        )

    if decision.requires_confirmation:
        if decision.confirmation_type == "pain":
            state.awaiting_pain_confirmation = True
        elif decision.confirmation_type == "stop":
            state.awaiting_stop_confirmation = True


def _drain_messages(
    state: SessionState,
    io: ExecutorIO,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
    feedback_worker: Optional[FeedbackWorker] = None,
) -> None:
    while state.is_active:
        message = io.poll_user_input()
        if message is None:
            return
        if feedback_worker is not None:
            feedback_worker.submit(message)
        else:
            _handle_user_message(message, state, io, decision_engine, feedback_understander, feedback_model)


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
        },
    )

    while state.is_active and state.seconds_remaining > 0:
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
    io.send(f"Exercise: {exercise['display_name']}")
    io.send("Instructions: " + "; ".join(exercise["instructions"]))


def export_session_log(state: SessionState, path: str) -> None:
    adjustment_reason_trace = []
    for item in state.adjustment_log:
        adjustment_reason_trace.append(
            {
                "action_type": item.get("action_type"),
                "rule_id": item.get("rule_id", ""),
                "state_reason": item.get("state_reason", ""),
                "reason": item.get("reason", ""),
            }
        )

    payload = {
        "summary": {
            "is_active": state.is_active,
            "end_reason": state.end_reason,
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
            event["detail"]
            for event in state.event_log
            if event.get("type") == "feedback_understanding_failed"
        ],
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
) -> SessionState:
    runtime_io = io or CLIExecutorIO()
    runtime_clock = clock or RealClock()
    runtime_decision_engine = decision_engine or RuleFirstDecisionEngine()
    runtime_feedback_understander = feedback_understander or understand_feedback
    feedback_worker: Optional[FeedbackWorker] = None

    translated_plan = translate_plan_for_demo(workout_plan)
    state = SessionState(
        workout_plan=translated_plan,
        initial_workout_plan=deepcopy(translated_plan),
        initial_intent=deepcopy(initial_intent or {}),
        exercise_library=deepcopy(exercise_library or []),
    )
    state.log_event("session_start", {"timestamp": runtime_clock.now()})
    _emit_runtime_event(
        runtime_io,
        "workout_started",
        {"workout_plan": deepcopy(state.workout_plan), "initial_intent": deepcopy(state.initial_intent)},
    )
    if getattr(runtime_io, "nonblocking_feedback", False):
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
        state.log_event("session_end", {"timestamp": runtime_clock.now(), "reason": state.end_reason})
        if log_path:
            export_session_log(state, log_path)
            _emit_runtime_event(runtime_io, "log_saved", {"path": log_path})
            runtime_io.send(f"Session log saved: {log_path}")
        runtime_io.close()

    return state

