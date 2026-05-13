import unittest

from feedback_understanding import FeedbackUnderstandingFailure, FeedbackUnderstandingResult, _normalize_result
from workout_executor import (
    AdjustmentAction,
    DecisionResult,
    RuleFirstDecisionEngine,
    run_adaptive_workout,
)


class FakeClock:
    def __init__(self):
        self.current = 0.0
        self.sleep_calls = 0

    def sleep(self, seconds: float) -> None:
        self.sleep_calls += 1
        self.current += seconds

    def now(self) -> float:
        return self.current


class ScriptedIO:
    def __init__(self, script):
        self.script = script
        self.outputs = []
        self.inbox = []
        self._fired = set()
        self.closed = False

    def send(self, message: str) -> None:
        self.outputs.append(message)
        for idx, rule in enumerate(self.script):
            if idx in self._fired:
                continue
            trigger = rule["when_contains"]
            if trigger in message:
                self.inbox.append(rule["message"])
                self._fired.add(idx)

    def poll_user_input(self):
        if not self.inbox:
            return None
        return self.inbox.pop(0)

    def close(self):
        self.closed = True


def build_plan(total_sets=3, rest_seconds=10):
    return {
        "rounds": 1,
        "rest_between_rounds": 10,
        "exercises": [
            {
                "exercise": "push_up",
                "avg_set_time": 1,
                "total_sets": total_sets,
                "rest_seconds": rest_seconds,
            },
            {
                "exercise": "bodyweight_squat",
                "avg_set_time": 1,
                "total_sets": total_sets,
                "rest_seconds": rest_seconds,
            },
        ],
    }


def build_library():
    return [
        {
            "name": "push_up",
            "target_muscles": ["chest", "arms"],
            "equipment": "none",
            "avg_set_time": 1,
        },
        {
            "name": "bodyweight_squat",
            "target_muscles": ["legs"],
            "equipment": "none",
            "avg_set_time": 1,
        },
        {
            "name": "plank",
            "target_muscles": ["chest", "core"],
            "equipment": "none",
            "avg_set_time": 1,
        },
    ]


def make_understander(mapping, default=None):
    def _understander(user_text, _snapshot, model="frob/qwen3.5-instruct:4b"):
        _ = model
        for token, result in mapping.items():
            if token in user_text.lower():
                return result
        return default

    return _understander


def result(
    intent,
    fatigue="medium",
    difficulty="appropriate",
    preference="neutral",
    confidence=0.9,
    reason=None,
    raw_text="",
    actions=None,
    safety="none",
    reply="",
):
    return FeedbackUnderstandingResult(
        intent=intent,
        fatigue_level=fatigue,
        difficulty_level=difficulty,
        preference=preference,
        confidence=confidence,
        reason=reason or f"intent={intent}",
        raw_text=raw_text,
        llm_channel="sdk",
        actions=actions or ["no_action"],
        safety=safety,
        reply=reply,
    )


class TestAdaptiveWorkoutExecutor(unittest.TestCase):
    def test_exercise_preview_wait_happens_without_countdown_text(self):
        io = ScriptedIO([])
        clock = FakeClock()
        understander = make_understander({}, default=result("neutral", confidence=0.2))

        state = run_adaptive_workout(
            build_plan(total_sets=1),
            io=io,
            clock=clock,
            feedback_understander=understander,
        )

        self.assertEqual("completed", state.end_reason)
        self.assertFalse(any("Get ready..." in msg for msg in io.outputs))
        self.assertTrue(any(e["type"] == "pre_exercise_preview_wait_start" for e in state.event_log))
        self.assertTrue(any(e["type"] == "pre_exercise_preview_wait_end" for e in state.event_log))

    def test_fatigue_during_rest_reduces_upcoming_intensity_only(self):
        io = ScriptedIO([
            {"when_contains": "Rest for 10 seconds.", "message": "too hard"},
        ])
        clock = FakeClock()
        understander = make_understander({
            "too hard": result(
                "fatigue",
                fatigue="high",
                difficulty="hard",
                actions=["decrease_difficulty"],
            ),
        })

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock, feedback_understander=understander)

        self.assertEqual("completed", state.end_reason)
        self.assertEqual(2, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual(2, state.workout_plan["exercises"][1]["total_sets"])
        self.assertTrue(any("set_delta=-1" in line for line in io.outputs))
        self.assertTrue(any("rest_multiplier=1.25" in line for line in io.outputs))

    def test_pain_triggers_auto_downshift_and_confirmation(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/3 start", "message": "I feel pain in my knee"},
            {"when_contains": "你还要继续吗", "message": "yes"},
        ])
        clock = FakeClock()
        understander = make_understander({
            "pain": result(
                "pain",
                fatigue="high",
                difficulty="hard",
                actions=["decrease_difficulty"],
                safety="pain",
            ),
        })

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock, feedback_understander=understander)

        self.assertEqual("completed", state.end_reason)
        self.assertFalse(state.awaiting_pain_confirmation)
        self.assertEqual(2, state.workout_plan["exercises"][0]["total_sets"])
        self.assertTrue(any("疼痛或不适" in line for line in io.outputs))

    def test_pace_up_shortens_rest_within_bounds(self):
        io = ScriptedIO([
            {"when_contains": "Round 1/1", "message": "faster please"},
        ])
        clock = FakeClock()
        understander = make_understander({
            "faster": result(
                "pace_up",
                fatigue="low",
                difficulty="easy",
                actions=["decrease_rest", "speed_up_tempo"],
            ),
        })

        state = run_adaptive_workout(
            build_plan(total_sets=2, rest_seconds=10),
            io=io,
            clock=clock,
            feedback_understander=understander,
        )

        self.assertEqual("completed", state.end_reason)
        self.assertEqual("faster", state.tempo_cue)
        self.assertEqual(8, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertEqual(8, state.workout_plan["exercises"][1]["rest_seconds"])
        self.assertTrue(any("Set 1/2 start (tempo: faster)" in line for line in io.outputs))

    def test_ambiguous_message_uses_fallback_and_clamps_values(self):
        def fallback_parser(_user_text, _state):
            return DecisionResult(
                intent="fallback_adjust",
                coach_reply="Fallback handled this message.",
                actions=[
                    AdjustmentAction(
                        action_type="adjust_intensity",
                        set_delta=-100,
                        rest_multiplier=0.01,
                        tempo_cue="faster",
                        reason="fallback_extreme",
                    )
                ],
            )

        io = ScriptedIO([
            {"when_contains": "Round 1/1", "message": "do your magic"},
        ])
        clock = FakeClock()
        engine = RuleFirstDecisionEngine(fallback_parser=fallback_parser)
        understander = make_understander({"do your magic": result("unknown")}, default=result("unknown", confidence=0.3))

        state = run_adaptive_workout(
            build_plan(total_sets=2, rest_seconds=10),
            io=io,
            clock=clock,
            decision_engine=engine,
            feedback_understander=understander,
        )

        self.assertEqual("completed", state.end_reason)
        self.assertEqual(1, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual(5, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertTrue(any(log["set_delta"] == -3 for log in state.adjustment_log))
        self.assertTrue(any(log["rest_multiplier"] == 0.5 for log in state.adjustment_log))

    def test_stop_intent_requires_confirmation_then_stops(self):
        io = ScriptedIO([
            {"when_contains": "Instructions:", "message": "I want to stop"},
            {"when_contains": "现在要停止吗", "message": "yes"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {"stop": result("stop", confidence=0.7, actions=["stop_workout"], safety="stop_request")}
        )

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock, feedback_understander=understander)

        self.assertFalse(state.is_active)
        self.assertEqual("user_requested_stop", state.end_reason)

    def test_uses_injectable_clock_for_deterministic_timing(self):
        io = ScriptedIO([])
        clock = FakeClock()
        understander = make_understander({}, default=result("neutral", confidence=0.2))

        state = run_adaptive_workout(build_plan(total_sets=1, rest_seconds=10), io=io, clock=clock, feedback_understander=understander)

        self.assertEqual("completed", state.end_reason)
        self.assertGreater(clock.sleep_calls, 0)
        self.assertEqual(float(clock.sleep_calls), clock.now())

    def test_condition_state_updates_from_feedback(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "I feel exhausted and this is too hard"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {
                "exhausted": result(
                    "fatigue",
                    fatigue="high",
                    difficulty="hard",
                    actions=["decrease_difficulty"],
                )
            },
            default=result("neutral", confidence=0.2),
        )

        state = run_adaptive_workout(build_plan(total_sets=2), io=io, clock=clock, feedback_understander=understander)

        self.assertEqual("high", state.user_condition.fatigue_level)
        self.assertEqual("hard", state.user_condition.difficulty_level)
        self.assertTrue(len(state.condition_log) > 0)
        self.assertTrue(any("state_reason" in log for log in state.adjustment_log))

    def test_multiple_feedback_actions_are_all_applied(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "太累了，慢一点"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {
                "太累": result(
                    "fatigue",
                    fatigue="high",
                    difficulty="hard",
                    actions=["decrease_difficulty", "slow_tempo"],
                )
            },
            default=result("neutral", confidence=0.2),
        )

        state = run_adaptive_workout(
            build_plan(total_sets=2, rest_seconds=10),
            io=io,
            clock=clock,
            feedback_understander=understander,
        )

        self.assertEqual("slower", state.tempo_cue)
        self.assertGreaterEqual(len(state.adjustment_log), 2)
        self.assertTrue(any(log["rule_id"] == "action_decrease_difficulty" for log in state.adjustment_log))
        self.assertTrue(any(log["rule_id"] == "action_slow_tempo" for log in state.adjustment_log))

    def test_preference_dislike_skips_current_exercise(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "I dislike this exercise"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {"dislike": result("preference_dislike", preference="dislike", actions=["skip_current_exercise"])},
            default=result("neutral", confidence=0.2),
        )

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            exercise_library=build_library(),
            feedback_understander=understander,
        )

        self.assertEqual(1, state.workout_plan["exercises"][0]["total_sets"])
        self.assertTrue(any(log["action_type"] == "skip_current_exercise" for log in state.adjustment_log))

    def test_preference_dislike_shorter_rest_request_does_not_skip_sets(self):
        io = ScriptedIO([
            {"when_contains": "Rest for 10 seconds.", "message": "休息时间可以短一点,太长了"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {
                "休息时间": result(
                    "preference_dislike",
                    preference="dislike",
                    reason=(
                        "User explicitly stated rest time is too long and requested "
                        "a shorter duration."
                    ),
                    raw_text="休息时间可以短一点,太长了",
                    actions=["decrease_rest"],
                )
            },
            default=result("neutral", confidence=0.2),
        )

        state = run_adaptive_workout(
            build_plan(total_sets=2, rest_seconds=10),
            io=io,
            clock=clock,
            exercise_library=build_library(),
            feedback_understander=understander,
        )

        self.assertEqual(2, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual(8, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertFalse(any(log["action_type"] == "skip_current_exercise" for log in state.adjustment_log))
        self.assertTrue(any(log["rule_id"] == "action_decrease_rest" for log in state.adjustment_log))

    def test_understanding_failure_skips_adjustment_and_logs_failure(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "hard to explain"},
        ])
        clock = FakeClock()

        def failing_understander(user_text, _snapshot, model="frob/qwen3.5-instruct:4b"):
            _ = model
            return FeedbackUnderstandingFailure(
                error_summary="sdk error: 502 | ollama_run error: timeout",
                tried_channels=["sdk", "ollama_run"],
                raw_text=user_text,
            )

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            feedback_understander=failing_understander,
        )

        self.assertEqual("completed", state.end_reason)
        self.assertTrue(any(e["type"] == "feedback_understanding_failed" for e in state.event_log))
        self.assertFalse(any("could not parse" in output for output in io.outputs))

    def test_unknown_feedback_does_not_reply_or_adjust(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "今天外面好像要下雨"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {"下雨": result("unknown", confidence=0.8)},
            default=result("neutral", confidence=0.8),
        )

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            feedback_understander=understander,
        )

        self.assertEqual("completed", state.end_reason)
        self.assertEqual([], state.adjustment_log)
        self.assertTrue(any(e["type"] == "feedback_ignored" for e in state.event_log))
        self.assertFalse(any(output.startswith("[Coach]") for output in io.outputs if "下雨" not in output))

    def test_low_confidence_feedback_does_not_reply_or_adjust(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "可能也许差不多"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {
                "也许": result(
                    "fatigue",
                    fatigue="high",
                    difficulty="hard",
                    confidence=0.4,
                    actions=["decrease_difficulty"],
                )
            },
            default=result("neutral", confidence=0.8),
        )

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            feedback_understander=understander,
        )

        self.assertEqual("completed", state.end_reason)
        self.assertEqual([], state.adjustment_log)
        self.assertTrue(any(e["type"] == "feedback_ignored" for e in state.event_log))

    def test_invalid_actions_normalize_to_no_action(self):
        understanding = _normalize_result(
            {
                "fatigue_level": "unknown",
                "difficulty_level": "unknown",
                "preference": "unknown",
                "actions": ["teleport", ""],
                "safety": "bad",
                "confidence": 0.8,
                "reason": "test",
            },
            raw_text="unclear",
            llm_channel="test",
        )

        self.assertEqual(["no_action"], understanding.actions)
        self.assertEqual("none", understanding.safety)


if __name__ == "__main__":
    unittest.main()
