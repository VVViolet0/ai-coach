import unittest

import json

from feedback_understanding import FeedbackUnderstandingFailure, FeedbackUnderstandingResult, _normalize_result
from workout_executor import (
    AdjustmentAction,
    DecisionResult,
    RuleFirstDecisionEngine,
    SessionState,
    _build_feedback_state_snapshot,
    _handle_user_message,
    run_adaptive_workout,
    translate_plan_for_demo,
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
        self.events = []
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

    def send_event(self, event_type, payload):
        self.events.append((event_type, payload))

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
                "exercise": "wall_push_up",
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
            "name": "wall_push_up",
            "target_muscles": ["chest", "arms"],
            "avg_set_time": 1,
        },
        {
            "name": "bodyweight_squat",
            "target_muscles": ["legs"],
            "avg_set_time": 1,
        },
        {
            "name": "standing_reverse_fly",
            "target_muscles": ["chest", "arms"],
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

    def test_exercise_demo_library_defines_default_beat_hz_for_every_action(self):
        with open("libraries/exercise_demo_library.json", "r", encoding="utf-8") as f:
            exercises = json.load(f)

        self.assertTrue(exercises)
        for exercise in exercises.values():
            self.assertIn("default_beat_hz", exercise)
            self.assertGreaterEqual(float(exercise["default_beat_hz"]), 0.0)
            self.assertLessEqual(float(exercise["default_beat_hz"]), 2.0)

    def test_translate_plan_for_demo_adds_executor_beat_metadata(self):
        plan = translate_plan_for_demo(build_plan(total_sets=1))

        first = plan["exercises"][0]
        self.assertEqual("wall_push_up", first["exercise_name"])
        self.assertEqual(0.35, first["default_beat_hz"])
        self.assertEqual("exercise_demo_library", first["beat_source"])

    def test_feedback_state_snapshot_only_includes_llm_needed_context(self):
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=1)))
        state.phase = "active_set"
        state.current_exercise = 0
        state.current_set = 1
        state.seconds_remaining = 20
        state.beat_multiplier = 0.8
        state.rest_multiplier = 1.5

        snapshot = _build_feedback_state_snapshot(state)

        self.assertEqual(
            {
                "phase": "active_set",
                "current_exercise_name": "wall_push_up",
            },
            snapshot,
        )

    def test_standing_exercise_uses_demo_metronome_beat(self):
        plan = translate_plan_for_demo(
            {
                "rounds": 1,
                "rest_between_rounds": 0,
                "exercises": [
                    {
                        "exercise": "standing_knee_to_elbow",
                        "avg_set_time": 40,
                        "total_sets": 1,
                        "rest_seconds": 10,
                    }
                ],
            }
        )

        first = plan["exercises"][0]
        self.assertEqual("standing_knee_to_elbow", first["exercise_name"])
        self.assertEqual(0.7, first["default_beat_hz"])

    def test_squat_uses_slower_controlled_metronome_beat(self):
        plan = translate_plan_for_demo(
            {
                "rounds": 1,
                "rest_between_rounds": 0,
                "exercises": [
                    {
                        "exercise": "bodyweight_squat",
                        "avg_set_time": 40,
                        "total_sets": 1,
                        "rest_seconds": 10,
                    }
                ],
            }
        )

        first = plan["exercises"][0]
        self.assertEqual("bodyweight_squat", first["exercise_name"])
        self.assertEqual(0.4, first["default_beat_hz"])

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
        self.assertTrue(any(log["set_delta"] == -1 for log in state.adjustment_log))
        self.assertTrue(any(log["rest_multiplier"] == 1.25 for log in state.adjustment_log))
        self.assertFalse(any(line.startswith("[Adjustment]") for line in io.outputs))

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
        self.assertAlmostEqual(1.2, state.beat_multiplier)
        self.assertEqual(10, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertEqual(10, state.workout_plan["exercises"][1]["rest_seconds"])
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
        self.assertAlmostEqual(1.0, state.beat_multiplier)
        self.assertTrue(any(log["set_delta"] == -3 for log in state.adjustment_log))
        self.assertTrue(any(log["rest_multiplier"] == 0.5 for log in state.adjustment_log))

    def test_stop_intent_stops_without_confirmation(self):
        io = ScriptedIO([
            {"when_contains": "Instructions:", "message": "I want to stop"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {"stop": result("stop", confidence=0.7, actions=["stop_workout"], safety="stop_request")}
        )

        state = run_adaptive_workout(build_plan(total_sets=3), io=io, clock=clock, feedback_understander=understander)

        self.assertFalse(state.is_active)
        self.assertEqual("user_requested_stop", state.end_reason)
        self.assertFalse(state.awaiting_stop_confirmation)

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
        self.assertNotIn("llm_intent", state.condition_log[-1])
        self.assertNotIn("llm_actions", state.condition_log[-1])
        self.assertEqual(["decrease_difficulty"], state.condition_log[-1]["actions"])
        self.assertTrue(any("state_reason" in log for log in state.adjustment_log))

    def test_feedback_actions_drive_adjustment_without_logging_legacy_intent(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "too tired"},
        ])
        clock = FakeClock()

        def understander(user_text, _snapshot, model="test"):
            _ = model
            if "too tired" in user_text:
                return _normalize_result(
                    {
                        "fatigue_level": "medium",
                        "difficulty_level": "appropriate",
                        "preference": "neutral",
                        "actions": ["decrease_difficulty"],
                        "safety": "none",
                        "confidence": 0.9,
                        "reason": "direct action",
                    },
                    raw_text=user_text,
                    llm_channel="test",
                )
            return result("neutral", confidence=0.2)

        state = run_adaptive_workout(
            build_plan(total_sets=2, rest_seconds=10),
            io=io,
            clock=clock,
            feedback_understander=understander,
        )

        self.assertTrue(any(log["action_type"] == "adjust_intensity" for log in state.adjustment_log))
        self.assertEqual(1, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual(12, state.workout_plan["exercises"][0]["rest_seconds"])
        self.assertNotIn("llm_intent", state.condition_log[-1])

        adjustment_events = [payload for event_type, payload in io.events if event_type == "adjustment_applied"]
        self.assertEqual(1, len(adjustment_events))
        self.assertEqual("normal", adjustment_events[0]["before_tempo_cue"])
        self.assertEqual("slower", adjustment_events[0]["after_tempo_cue"])
        self.assertEqual(1.0, adjustment_events[0]["before_rest_multiplier"])
        self.assertEqual(1.25, adjustment_events[0]["after_rest_multiplier"])
        self.assertEqual(0, adjustment_events[0]["before_set_delta"])
        self.assertEqual(-1, adjustment_events[0]["after_set_delta"])

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
        self.assertLess(state.beat_multiplier, 1.0)
        self.assertGreaterEqual(len(state.adjustment_log), 1)
        self.assertTrue(any(log["rule_id"] == "action_decrease_difficulty" for log in state.adjustment_log))

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

        self.assertEqual("standing_reverse_fly", state.workout_plan["exercises"][0]["exercise_name"])
        self.assertTrue(any(log["action_type"] == "replace_exercise" for log in state.adjustment_log))

    def test_replace_current_exercise_updates_plan_event_and_rebroadcasts_demo(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "replace this exercise"},
        ])
        clock = FakeClock()
        understander = make_understander(
            {
                "replace": result(
                    "preference_dislike",
                    preference="dislike",
                    actions=["replace_current_exercise"],
                    reply="好的，我帮你换一个动作。",
                )
            },
            default=result("neutral", confidence=0.2),
        )

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            exercise_library=build_library(),
            feedback_understander=understander,
        )

        replaced = state.workout_plan["exercises"][0]
        self.assertEqual("standing_reverse_fly", replaced["exercise_name"])
        self.assertTrue(any(log["action_type"] == "replace_exercise" for log in state.adjustment_log))
        self.assertFalse(any(line.startswith("[Adjustment]") for line in io.outputs))
        self.assertTrue(any(line.startswith(f"Exercise: {replaced['display_name']}") for line in io.outputs))

        replacement_events = [payload for event_type, payload in io.events if event_type == "exercise_replaced"]
        self.assertEqual(1, len(replacement_events))
        self.assertEqual(0, replacement_events[0]["exercise_index"])
        self.assertEqual("靠墙俯卧撑", replacement_events[0]["before_display_name"])
        self.assertEqual("standing_reverse_fly", replacement_events[0]["exercise"]["exercise_name"])

    def test_replace_current_exercise_action_drives_replacement_without_legacy_intent(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "please replace this"},
        ])
        clock = FakeClock()

        def understander(user_text, _snapshot, model="test"):
            _ = model
            if "replace" in user_text:
                return _normalize_result(
                    {
                        "fatigue_level": "medium",
                        "difficulty_level": "appropriate",
                        "preference": "dislike",
                        "actions": ["replace_current_exercise"],
                        "safety": "none",
                        "confidence": 0.9,
                        "reason": "direct replacement action",
                    },
                    raw_text=user_text,
                    llm_channel="test",
                )
            return result("neutral", confidence=0.2)

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            exercise_library=build_library(),
            feedback_understander=understander,
        )

        self.assertEqual("standing_reverse_fly", state.workout_plan["exercises"][0]["exercise_name"])
        self.assertTrue(any(log["action_type"] == "replace_exercise" for log in state.adjustment_log))
        self.assertNotIn("llm_intent", state.condition_log[-1])
        self.assertNotIn("source_intent", state.adjustment_log[-1])

        replacement_events = [payload for event_type, payload in io.events if event_type == "exercise_replaced"]
        self.assertEqual(1, len(replacement_events))
        self.assertEqual("standing_reverse_fly", replacement_events[0]["exercise"]["exercise_name"])

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

    def test_feedback_trace_links_parsed_intent_adjustment_and_duration(self):
        io = ScriptedIO([
            {"when_contains": "Set 1/2 start", "message": "too hard"},
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

        state = run_adaptive_workout(
            build_plan(total_sets=2),
            io=io,
            clock=clock,
            feedback_understander=understander,
            condition_id="C",
        )

        trace = state.feedback_trace[0]
        self.assertEqual("C", trace["condition_id"])
        self.assertEqual("fatigue", trace["parsed_intent"])
        self.assertEqual("high", trace["structured_state"]["fatigue_level"])
        self.assertTrue(trace["system_adjustments"])
        self.assertEqual(trace["feedback_id"], trace["system_adjustments"][0]["feedback_id"])
        self.assertGreater(state.actual_duration_seconds, 0)
        self.assertTrue(state.session_started_at)
        self.assertTrue(state.session_ended_at)
        self.assertTrue(all("timestamp" in event for event in state.event_log))

    def test_fast_path_stop_does_not_call_llm(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"

        def should_not_call_llm(*_args, **_kwargs):
            raise AssertionError("LLM should not be called for stop fast path")

        _handle_user_message(
            "停止训练",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=should_not_call_llm,
            feedback_model="test",
        )

        self.assertFalse(state.is_active)
        self.assertEqual("user_requested_stop", state.end_reason)
        self.assertEqual("rule_fast_path", state.feedback_trace[0]["llm_channel"])
        self.assertEqual("stop", state.feedback_trace[0]["parsed_intent"])

    def test_can_stop_now_requires_confirmation(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"

        def should_not_call_llm(*_args, **_kwargs):
            raise AssertionError("LLM should not be called for ambiguous stop fast path")

        _handle_user_message(
            "可以停了",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=should_not_call_llm,
            feedback_model="test",
        )

        self.assertTrue(state.is_active)
        self.assertTrue(state.awaiting_stop_confirmation)
        self.assertEqual(["confirm_stop_workout"], state.feedback_trace[0]["structured_state"]["actions"])
        self.assertIn("结束整个训练", io.outputs[-1])

    def test_ambiguous_stop_fast_path_requires_confirmation(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"

        def should_not_call_llm(*_args, **_kwargs):
            raise AssertionError("LLM should not be called for ambiguous stop fast path")

        _handle_user_message(
            "不想继续",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=should_not_call_llm,
            feedback_model="test",
        )

        self.assertTrue(state.is_active)
        self.assertTrue(state.awaiting_stop_confirmation)
        self.assertEqual("stop", state.feedback_trace[0]["parsed_intent"])
        self.assertEqual(["confirm_stop_workout"], state.feedback_trace[0]["structured_state"]["actions"])
        self.assertIn("结束整个训练", io.outputs[-1])

    def test_stop_confirmation_continue_does_not_stop(self):
        io = ScriptedIO([])
        state = SessionState(
            workout_plan=translate_plan_for_demo(build_plan(total_sets=2)),
            condition_id="C",
            awaiting_stop_confirmation=True,
        )
        state.phase = "active_set"

        _handle_user_message(
            "继续",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=lambda *_args, **_kwargs: None,
            feedback_model="test",
        )

        self.assertTrue(state.is_active)
        self.assertFalse(state.awaiting_stop_confirmation)

    def test_stop_confirmation_yes_stops(self):
        io = ScriptedIO([])
        state = SessionState(
            workout_plan=translate_plan_for_demo(build_plan(total_sets=2)),
            condition_id="C",
            awaiting_stop_confirmation=True,
        )
        state.phase = "active_set"

        _handle_user_message(
            "是",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=lambda *_args, **_kwargs: None,
            feedback_model="test",
        )

        self.assertFalse(state.is_active)
        self.assertEqual("user_requested_stop", state.end_reason)

    def test_fast_path_pace_down_changes_beat_multiplier_without_llm(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"

        def should_not_call_llm(*_args, **_kwargs):
            raise AssertionError("LLM should not be called for pace fast path")

        _handle_user_message(
            "这个太快了，我想慢一点",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=should_not_call_llm,
            feedback_model="test",
        )

        self.assertEqual("slower", state.tempo_cue)
        self.assertLess(state.beat_multiplier, 1.0)
        self.assertEqual("rule_fast_path", state.feedback_trace[0]["llm_channel"])
        self.assertTrue(any(event_type == "adjustment_applied" for event_type, _payload in io.events))

    def test_fast_path_shorten_exercise_duration_reduces_sets_without_llm(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=3)), condition_id="C")
        state.phase = "active_set"

        def should_not_call_llm(*_args, **_kwargs):
            raise AssertionError("LLM should not be called for shorten duration fast path")

        _handle_user_message(
            "缩短深蹲的训练时长",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=should_not_call_llm,
            feedback_model="test",
        )

        self.assertEqual(2, state.workout_plan["exercises"][0]["total_sets"])
        self.assertEqual("rule_fast_path", state.feedback_trace[0]["llm_channel"])
        self.assertEqual(["decrease_sets"], state.feedback_trace[0]["structured_state"]["actions"])
        self.assertTrue(any(log["rule_id"] == "action_decrease_sets" for log in state.adjustment_log))

    def test_ambiguous_feedback_uses_llm_fallback(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"
        calls = []

        def fallback_understander(user_text, _snapshot, model="test"):
            calls.append((user_text, model))
            return result(
                "pace_down",
                actions=["slow_tempo"],
                raw_text=user_text,
            )

        _handle_user_message(
            "有点怪",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=fallback_understander,
            feedback_model="test",
        )

        self.assertEqual([("有点怪", "test")], calls)
        self.assertEqual("slower", state.tempo_cue)

    def test_stale_llm_result_is_ignored_when_superseded(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"
        state.feedback_sequence = 2

        def old_fallback_understander(user_text, _snapshot, model="test"):
            return FeedbackUnderstandingResult(
                intent="pace_up",
                fatigue_level="low",
                difficulty_level="easy",
                preference="neutral",
                confidence=0.9,
                reason="old llm result",
                raw_text=user_text,
                llm_channel="llm_fallback",
                actions=["speed_up_tempo"],
                safety="none",
            )

        _handle_user_message(
            "有点怪",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=old_fallback_understander,
            feedback_model="test",
            feedback_sequence_id=1,
        )

        self.assertEqual(1.0, state.beat_multiplier)
        self.assertEqual("stale_ignored", state.feedback_trace[0]["status"])
        self.assertEqual("superseded_by_newer_feedback", state.feedback_trace[0]["stale_reason"])
        self.assertTrue(any(event["type"] == "feedback_stale_ignored" for event in state.event_log))

    def test_stale_llm_result_is_ignored_when_context_changed(self):
        io = ScriptedIO([])
        state = SessionState(workout_plan=translate_plan_for_demo(build_plan(total_sets=2)), condition_id="C")
        state.phase = "active_set"

        def context_changing_understander(user_text, _snapshot, model="test"):
            state.current_exercise = 1
            return FeedbackUnderstandingResult(
                intent="pace_down",
                fatigue_level="medium",
                difficulty_level="appropriate",
                preference="neutral",
                confidence=0.9,
                reason="late llm result",
                raw_text=user_text,
                llm_channel="llm_fallback",
                actions=["slow_tempo"],
                safety="none",
            )

        _handle_user_message(
            "有点怪",
            state,
            io,
            decision_engine=RuleFirstDecisionEngine(),
            feedback_understander=context_changing_understander,
            feedback_model="test",
        )

        self.assertEqual(1.0, state.beat_multiplier)
        self.assertEqual("stale_ignored", state.feedback_trace[0]["status"])
        self.assertEqual("context_changed", state.feedback_trace[0]["stale_reason"])


if __name__ == "__main__":
    unittest.main()
