import json
import queue
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol


EXERCISE_DB = {
    "push_up": {
        "exercise_name": "Push Up",
        "instructions": [
            "Place hands slightly wider than shoulders",
            "Keep body straight",
            "Lower chest toward the floor",
            "Push back up",
        ],
        "demo_speed": 1.0,
    },
    "bodyweight_squat": {
        "exercise_name": "Bodyweight Squat",
        "instructions": [
            "Stand with feet shoulder-width apart",
            "Push hips back",
            "Lower until thighs parallel",
            "Drive through heels to stand",
        ],
        "demo_speed": 1.0,
    },
}

MIN_REST_SECONDS = 5
MAX_REST_SECONDS = 180
MIN_SETS = 1
MAX_SETS = 8
MIN_REST_MULTIPLIER = 0.5
MAX_REST_MULTIPLIER = 2.0
MIN_SET_DELTA = -3
MAX_SET_DELTA = 3
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


@dataclass
class AdjustmentAction:
    action_type: str
    rest_multiplier: float = 1.0
    set_delta: int = 0
    tempo_cue: Optional[str] = None
    reason: str = ""


@dataclass
class DecisionResult:
    intent: str
    coach_reply: str
    actions: List[AdjustmentAction] = field(default_factory=list)
    requires_confirmation: bool = False


@dataclass
class SessionState:
    workout_plan: Dict
    initial_workout_plan: Dict = field(default_factory=dict)
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
    event_log: List[Dict] = field(default_factory=list)
    conversation_log: List[Dict] = field(default_factory=list)
    adjustment_log: List[Dict] = field(default_factory=list)

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


def translate_plan_for_demo(workout_plan: Dict) -> Dict:
    demo_plan = {
        "rounds": workout_plan["rounds"],
        "rest_between_rounds": workout_plan["rest_between_rounds"],
        "exercises": [],
    }

    for ex in workout_plan["exercises"]:
        name = ex["exercise"]
        details = EXERCISE_DB.get(
            name,
            {
                "exercise_name": name.replace("_", " ").title(),
                "instructions": ["Follow safe and controlled movement."],
                "demo_speed": 1.0,
            },
        )

        demo_exercise = {
            "exercise_name": name,
            "display_name": details["exercise_name"],
            "instructions": details["instructions"],
            "demo_speed": details["demo_speed"],
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
        text = user_text.lower().strip()

        if any(token in text for token in ("stop", "quit", "end workout", "cancel")):
            return DecisionResult(
                intent="stop",
                coach_reply="Stopping the session now.",
                actions=[AdjustmentAction(action_type="stop", reason="user_requested_stop")],
            )

        if any(token in text for token in ("pain", "hurt", "injury", "ache", "discomfort")):
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
                    )
                ],
                requires_confirmation=True,
            )

        if any(token in text for token in ("tired", "too hard", "fatigue", "exhausted")):
            return DecisionResult(
                intent="fatigue",
                coach_reply="Understood. I lowered upcoming intensity and added more recovery.",
                actions=[
                    AdjustmentAction(
                        action_type="adjust_intensity",
                        set_delta=-1,
                        rest_multiplier=1.2,
                        tempo_cue="slower",
                        reason="fatigue_downshift",
                    )
                ],
            )

        if any(token in text for token in ("faster", "speed up", "quicker", "less rest")):
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
                    )
                ],
            )

        if any(token in text for token in ("slower", "slow down", "more rest", "too fast")):
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
                    )
                ],
            )

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


def _apply_adjustment_action(
    action: AdjustmentAction,
    state: SessionState,
    io: ExecutorIO,
) -> None:
    if action.action_type == "stop":
        state.is_active = False
        state.end_reason = action.reason or "user_stop"
        state.log_event("session_stop", {"reason": state.end_reason})
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


def _handle_user_message(
    user_text: str,
    state: SessionState,
    io: ExecutorIO,
    decision_engine: DecisionEngine,
) -> None:
    cleaned = user_text.strip()
    if not cleaned:
        return

    state.conversation_log.append({"role": "user", "content": cleaned})
    state.log_event("user_message", {"content": cleaned})

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

    decision = decision_engine.decide(cleaned, state)
    state.log_event("decision", {"intent": decision.intent})
    io.send(f"[Coach] {decision.coach_reply}")
    state.conversation_log.append({"role": "coach", "content": decision.coach_reply})

    for action in decision.actions:
        _apply_adjustment_action(action, state, io)

    if decision.requires_confirmation:
        state.awaiting_pain_confirmation = True


def _drain_messages(
    state: SessionState,
    io: ExecutorIO,
    decision_engine: DecisionEngine,
) -> None:
    while state.is_active:
        message = io.poll_user_input()
        if message is None:
            return
        _handle_user_message(message, state, io, decision_engine)


def _run_timed_phase(
    seconds: int,
    phase: str,
    state: SessionState,
    io: ExecutorIO,
    clock: Clock,
    decision_engine: DecisionEngine,
) -> None:
    state.phase = phase
    state.seconds_remaining = seconds

    while state.is_active and state.seconds_remaining > 0:
        _drain_messages(state, io, decision_engine)
        if not state.is_active:
            break
        clock.sleep(1)
        state.seconds_remaining -= 1

    _drain_messages(state, io, decision_engine)


def _show_demo(io: ExecutorIO, exercise: Dict) -> None:
    io.send("---------------------------")
    io.send(f"Exercise: {exercise['display_name']}")
    io.send(f"demo_speed: {exercise['demo_speed']}")
    io.send("Instructions:")
    for step in exercise["instructions"]:
        io.send(f" - {step}")
    io.send("---------------------------")


def export_session_log(state: SessionState, path: str) -> None:
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
        "final_workout_plan": state.workout_plan,
        "adjustments": state.adjustment_log,
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
    log_path: Optional[str] = None,
) -> SessionState:
    runtime_io = io or CLIExecutorIO()
    runtime_clock = clock or RealClock()
    runtime_decision_engine = decision_engine or RuleFirstDecisionEngine()

    translated_plan = translate_plan_for_demo(workout_plan)
    state = SessionState(
        workout_plan=translated_plan,
        initial_workout_plan=deepcopy(translated_plan),
    )
    state.log_event("session_start", {"timestamp": runtime_clock.now()})
    runtime_io.send("===== ADAPTIVE WORKOUT START =====")
    runtime_io.send("Send messages anytime: 'too hard', 'pain', 'faster', 'slower', or 'stop'.")

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
                _show_demo(runtime_io, exercise)

                set_index = 0
                while state.is_active and set_index < exercise["total_sets"]:
                    state.current_set = set_index + 1
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
