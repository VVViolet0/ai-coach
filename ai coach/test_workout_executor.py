import unittest

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


class TestAdaptiveWorkoutExecutor(unittest.TestCase):
    def test_fatigue_during_rest_reduces_upcoming_intensity_only(self):
        io = ScriptedIO([
            {"when_contains": "Rest for 10 seconds.", "message": "too hard"},
        ])
        clock = FakeClock()

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock)

        self.assertEqual("completed", state.end_reason)
        self.assertEqual(2, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual(2, state.workout_plan["exercises"][1]["total_sets"])
        self.assertTrue(any("set_delta=-1" in line for line in io.outputs))

    def test_pain_triggers_auto_downshift_and_confirmation(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/3 start", "message": "I feel pain in my knee"},
            {"when_contains": "Do you want to continue?", "message": "yes"},
        ])
        clock = FakeClock()

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock)

        self.assertEqual("completed", state.end_reason)
        self.assertFalse(state.awaiting_pain_confirmation)
        self.assertEqual(2, state.workout_plan["exercises"][0]["total_sets"])
        self.assertTrue(any("pain/discomfort" in line for line in io.outputs))

    def test_pace_up_shortens_rest_within_bounds(self):
        io = ScriptedIO([
            {"when_contains": "ROUND 1/1", "message": "faster please"},
        ])
        clock = FakeClock()

        state = run_adaptive_workout(build_plan(total_sets=2, rest_seconds=10), io=io, clock=clock)

        self.assertEqual("completed", state.end_reason)
        self.assertEqual("faster", state.tempo_cue)
        self.assertEqual(8, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertEqual(8, state.workout_plan["exercises"][1]["rest_seconds"])

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
            {"when_contains": "ROUND 1/1", "message": "do your magic"},
        ])
        clock = FakeClock()
        engine = RuleFirstDecisionEngine(fallback_parser=fallback_parser)

        state = run_adaptive_workout(build_plan(total_sets=2, rest_seconds=10), io=io, clock=clock, decision_engine=engine)

        self.assertEqual("completed", state.end_reason)
        self.assertEqual(1, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual(5, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertTrue(any(log["set_delta"] == -3 for log in state.adjustment_log))
        self.assertTrue(any(log["rest_multiplier"] == 0.5 for log in state.adjustment_log))

    def test_stop_intent_exits_cleanly_and_logs_reason(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/3 start", "message": "stop"},
        ])
        clock = FakeClock()

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock)

        self.assertFalse(state.is_active)
        self.assertEqual("user_requested_stop", state.end_reason)
        self.assertTrue(any(e["type"] == "session_stop" for e in state.event_log))

    def test_uses_injectable_clock_for_deterministic_timing(self):
        io = ScriptedIO([])
        clock = FakeClock()

        state = run_adaptive_workout(build_plan(total_sets=1, rest_seconds=10), io=io, clock=clock)

        self.assertEqual("completed", state.end_reason)
        self.assertGreater(clock.sleep_calls, 0)
        self.assertEqual(float(clock.sleep_calls), clock.now())


if __name__ == "__main__":
    unittest.main()
