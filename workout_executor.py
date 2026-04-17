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
MIN_DEMO_SPEED = 0.75
MAX_DEMO_SPEED = 1.25
EXERCISE_DEMO_LIBRARY_PATH = Path('libraries/exercise_demo_library.json')
YES_WORDS = {"yes", "y", "continue", "go", "ok", "okay"}
NO_WORDS = {"no", "n", "stop", "quit", "end"}


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
    demo_speed: float = 1.0
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


def _resolve_demo_speed(tempo_cue: str) -> float:
    speed_map = {
        "normal": 1.0,
        "slower": 0.9,
        "faster": 1.1,
    }
    return speed_map.get(tempo_cue, 1.0)


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


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, value))


def _clamp_float(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
                "I heard you may want to stop. Do you want to stop the workout now? (yes/no)"
            ),
            requires_confirmation=True,
            confirmation_type="stop",
        )

    if understanding.intent == "pain":
        return DecisionResult(
            intent="pain",
            coach_reply=(
                "I heard pain/discomfort. I reduced intensity now. "
                "Do you want to continue? (yes/no)"
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

    if understanding.intent in {"fatigue", "preference_dislike", "preference_like"}:
        return DecisionResult(
            intent=understanding.intent,
            coach_reply="Thanks, I captured your feedback and updated upcoming blocks.",
        )

    fallback = decision_engine.decide(understanding.raw_text, state)
    if fallback.intent != "unknown":
        return fallback

    return DecisionResult(
        intent="unknown",
        coach_reply=(
            "I understood your message and updated your condition state. "
            "I will keep guiding and adjusting as we continue."
        ),
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

    if condition.preference == "dislike":
        replacement = _find_replacement_exercise(state)
        if replacement:
            return [
                AdjustmentAction(
                    action_type="replace_exercise",
                    replacement_exercise=replacement,
                    reason="preference_dislike_replace",
                    rule_id="preference_dislike_replace",
                    state_reason="preference=dislike",
                )
            ]

        return [
            AdjustmentAction(
                action_type="adjust_intensity",
                set_delta=-1,
                rest_multiplier=1.2,
                tempo_cue="slower",
                reason="preference_dislike_fallback_downshift",
                rule_id="preference_dislike_fallback",
                state_reason="preference=dislike but no replacement available",
            )
        ]

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
) -> Optional[str]:
    if action.action_type == "stop":
        state.is_active = False
        state.end_reason = action.reason or "user_stop"
        state.log_event("session_stop", {"reason": state.end_reason})
        return None

    if action.action_type == "replace_exercise":
        target_name = action.replacement_exercise
        if not target_name:
            return None

        replacement_def = next((ex for ex in state.exercise_library if ex.get("name") == target_name), None)
        if not replacement_def:
            return None

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
        io.send(f"[Adjustment] replaced exercise: {before_name} -> {target_name}")
        return f"I replaced the current exercise from {before_name} to {target_name}."

    if action.action_type != "adjust_intensity":
        return None

    safe_set_delta = _clamp_int(action.set_delta, MIN_SET_DELTA, MAX_SET_DELTA)
    safe_multiplier = max(MIN_REST_MULTIPLIER, min(MAX_REST_MULTIPLIER, action.rest_multiplier))
    state.set_delta = _clamp_int(state.set_delta + safe_set_delta, MIN_SET_DELTA, MAX_SET_DELTA)
    state.rest_multiplier = max(
        MIN_REST_MULTIPLIER,
        min(MAX_REST_MULTIPLIER, state.rest_multiplier * safe_multiplier),
    )
    if action.tempo_cue:
        state.tempo_cue = action.tempo_cue
    state.demo_speed = _clamp_float(_resolve_demo_speed(state.tempo_cue), MIN_DEMO_SPEED, MAX_DEMO_SPEED)

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
    io.send(
        f"[Adjustment] next blocks updated: set_delta={safe_set_delta}, "
        f"rest_multiplier={safe_multiplier:.2f}, tempo={state.tempo_cue}"
    )
    changes: List[str] = []
    if safe_set_delta < 0:
        changes.append(f"reduced upcoming sets by {abs(safe_set_delta)}")
    elif safe_set_delta > 0:
        changes.append(f"increased upcoming sets by {safe_set_delta}")

    if safe_multiplier > 1.0:
        changes.append(f"increased upcoming rest by {int(round((safe_multiplier - 1.0) * 100))}%")
    elif safe_multiplier < 1.0:
        changes.append(f"reduced upcoming rest by {int(round((1.0 - safe_multiplier) * 100))}%")

    if action.tempo_cue == "slower":
        changes.append("slowed the demo pace")
    elif action.tempo_cue == "faster":
        changes.append("sped up the demo pace")

    if not changes:
        return "I kept the plan structure but refreshed the upcoming pacing settings."

    return "I " + ", ".join(changes) + "."


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

    state.conversation_log.append({"role": "user", "content": cleaned})
    state.log_event("user_message", {"content": cleaned})

    if state.awaiting_stop_confirmation:
        lowered = cleaned.lower()
        if lowered in YES_WORDS:
            state.awaiting_stop_confirmation = False
            state.is_active = False
            state.end_reason = "user_requested_stop"
            io.send("Stopping the workout now.")
            state.log_event("stop_confirmation", {"decision": "stop"})
            return
        if lowered in NO_WORDS:
            state.awaiting_stop_confirmation = False
            io.send("Great, we continue. I can still lower intensity anytime.")
            state.log_event("stop_confirmation", {"decision": "continue"})
            return
        io.send("Please reply with 'yes' to stop or 'no' to continue.")
        state.log_event("stop_confirmation", {"decision": "unclear"})
        return

    if state.awaiting_pain_confirmation:
        lowered = cleaned.lower()
        if lowered in YES_WORDS:
            state.awaiting_pain_confirmation = False
            io.send("Great. We continue with a lighter safer pace.")
            state.log_event("pain_confirmation", {"decision": "continue"})
            return
        if lowered in NO_WORDS:
            state.awaiting_pain_confirmation = False
            state.is_active = False
            state.end_reason = "user_stopped_after_pain"
            io.send("Session stopped. Please rest and seek medical advice if needed.")
            state.log_event("pain_confirmation", {"decision": "stop"})
            return

        io.send("Please reply with 'yes' to continue or 'no' to stop.")
        state.log_event("pain_confirmation", {"decision": "unclear"})
        return

    state_snapshot = {
        "phase": state.phase,
        "current_round": state.current_round,
        "current_exercise": state.current_exercise,
        "current_set": state.current_set,
        "user_condition": state.user_condition.snapshot(),
        "tempo_cue": state.tempo_cue,
        "demo_speed": state.demo_speed,
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
    decision = _build_decision_from_understanding(understanding, state, decision_engine)
    state.log_event("decision", {"intent": decision.intent})
    io.send(f"[Coach] {decision.coach_reply}")
    state.conversation_log.append({"role": "coach", "content": decision.coach_reply})

    state_actions: List[AdjustmentAction] = []
    if condition_changed and decision.intent not in {"stop", "pain", "neutral", "unknown", "pace_up", "pace_down"}:
        state_actions = _derive_state_actions(state)

    coach_adjustment_summaries: List[str] = []
    for action in decision.actions + state_actions:
        summary = _apply_adjustment_action(
            action,
            state,
            io,
            source_intent=understanding.intent,
            confidence=understanding.confidence,
        )
        if summary:
            coach_adjustment_summaries.append(summary)

    if coach_adjustment_summaries:
        combined_summary = " ".join(coach_adjustment_summaries)
        io.send(f"[Coach] {combined_summary}")
        state.conversation_log.append({"role": "coach", "content": combined_summary})

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
) -> None:
    while state.is_active:
        message = io.poll_user_input()
        if message is None:
            return
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
) -> None:
    state.phase = phase
    state.seconds_remaining = seconds

    while state.is_active and state.seconds_remaining > 0:
        _drain_messages(state, io, decision_engine, feedback_understander, feedback_model)
        if not state.is_active:
            break
        clock.sleep(1)
        state.seconds_remaining -= 1

    _drain_messages(state, io, decision_engine, feedback_understander, feedback_model)


def _run_pre_exercise_preview_wait(
    seconds: int,
    state: SessionState,
    io: ExecutorIO,
    clock: Clock,
    decision_engine: DecisionEngine,
    feedback_understander: FeedbackUnderstander,
    feedback_model: str,
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
        _drain_messages(state, io, decision_engine, feedback_understander, feedback_model)
        if not state.is_active:
            break
        clock.sleep(1)
        state.seconds_remaining -= 1

    _drain_messages(state, io, decision_engine, feedback_understander, feedback_model)
    state.log_event(
        "pre_exercise_preview_wait_end",
        {
            "round": state.current_round,
            "exercise": state.current_exercise,
            "is_active": state.is_active,
        },
    )


def _show_demo(io: ExecutorIO, exercise: Dict) -> None:
    io.send("---------------------------")
    io.send(f"Exercise: {exercise['display_name']}")
    io.send("Instructions:")
    for step in exercise["instructions"]:
        io.send(f" - {step}")
    io.send("---------------------------")


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
            "tempo_cue": state.tempo_cue,
            "demo_speed": state.demo_speed,
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

    translated_plan = translate_plan_for_demo(workout_plan)
    state = SessionState(
        workout_plan=translated_plan,
        initial_workout_plan=deepcopy(translated_plan),
        initial_intent=deepcopy(initial_intent or {}),
        exercise_library=deepcopy(exercise_library or []),
        demo_speed=_resolve_demo_speed("normal"),
    )
    state.log_event("session_start", {"timestamp": runtime_clock.now()})
    try:
        rounds = state.workout_plan["rounds"]
        for round_idx in range(rounds):
            if not state.is_active:
                break
            state.current_round = round_idx
            runtime_io.send(f"\n===== ROUND {round_idx + 1}/{rounds} =====")

            for ex_idx, exercise in enumerate(state.workout_plan["exercises"]):
                if not state.is_active:
                    break
                state.current_exercise = ex_idx
                state.demo_speed = _clamp_float(_resolve_demo_speed(state.tempo_cue), MIN_DEMO_SPEED, MAX_DEMO_SPEED)
                _show_demo(runtime_io, exercise)
                _run_pre_exercise_preview_wait(
                    seconds=PRE_EXERCISE_PREVIEW_WAIT_SECONDS,
                    state=state,
                    io=runtime_io,
                    clock=runtime_clock,
                    decision_engine=runtime_decision_engine,
                    feedback_understander=runtime_feedback_understander,
                    feedback_model=feedback_model,
                )
                if not state.is_active:
                    break

                set_index = 0
                while state.is_active and set_index < exercise["total_sets"]:
                    state.current_set = set_index + 1
                    runtime_io.send(
                        f"Set {state.current_set}/{exercise['total_sets']} start "
                        f"(demo_speed: {state.demo_speed:.2f})"
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
                    )
                    if not state.is_active:
                        break
                    runtime_io.send("Set finished.")
                    set_index += 1

                    if set_index < exercise["total_sets"]:
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
                )

        if state.is_active:
            state.end_reason = "completed"
            runtime_io.send("===== WORKOUT COMPLETE =====")
        else:
            runtime_io.send(f"===== WORKOUT STOPPED: {state.end_reason} =====")
    finally:
        state.log_event("session_end", {"timestamp": runtime_clock.now(), "reason": state.end_reason})
        if log_path:
            export_session_log(state, log_path)
            runtime_io.send(f"Session log saved to: {log_path}")
        runtime_io.close()

    return state


