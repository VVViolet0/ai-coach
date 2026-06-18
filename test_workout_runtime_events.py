from workout_executor import SessionState, _handle_user_message, run_adaptive_workout


class FakeClock:
    def __init__(self):
        self.current = 0.0

    def sleep(self, seconds):
        self.current += seconds

    def now(self):
        return self.current


class EventIO:
    def __init__(self):
        self.messages = []
        self.events = []

    def send(self, message):
        self.messages.append(message)

    def send_event(self, event_type, payload):
        self.events.append((event_type, payload))

    def poll_user_input(self):
        return None

    def close(self):
        pass


class NonBlockingFeedbackIO(EventIO):
    nonblocking_feedback = True

    def __init__(self):
        super().__init__()
        self.inputs = ["太累了"]

    def poll_user_input(self):
        if self.inputs:
            return self.inputs.pop(0)
        return None


class PlainIO:
    def send(self, message):
        pass

    def poll_user_input(self):
        return None

    def close(self):
        pass


def tiny_plan():
    return {
        "rounds": 1,
        "rest_between_rounds": 0,
        "exercises": [
            {
                "exercise": "wall_push_up",
                "avg_set_time": 1,
                "total_sets": 1,
                "rest_seconds": 5,
            }
        ],
    }


class CaptureIO(EventIO):
    def poll_user_input(self):
        return None


def test_stop_confirmation_uses_chinese_prompt_and_accepts_no():
    state = SessionState(workout_plan=tiny_plan(), awaiting_stop_confirmation=True)
    io = CaptureIO()

    _handle_user_message("否", state, io, decision_engine=None, feedback_understander=None, feedback_model="test")

    assert state.is_active is True
    assert state.awaiting_stop_confirmation is False
    assert "继续" in io.messages[-1]


def test_pain_confirmation_uses_chinese_prompt_and_accepts_stop():
    state = SessionState(workout_plan=tiny_plan(), awaiting_pain_confirmation=True)
    io = CaptureIO()

    _handle_user_message("停止", state, io, decision_engine=None, feedback_understander=None, feedback_model="test")

    assert state.is_active is False
    assert state.end_reason == "user_stopped_after_pain"
    assert "训练已停止" in io.messages[-1]


def test_runtime_events_are_emitted_when_io_supports_send_event():
    io = EventIO()

    run_adaptive_workout(tiny_plan(), io=io, clock=FakeClock())

    event_types = [event_type for event_type, _payload in io.events]
    assert "workout_started" in event_types
    assert "round_start" in event_types
    assert "exercise_start" in event_types
    assert "set_start" in event_types
    assert "phase_tick" in event_types
    assert "set_end" in event_types
    assert "session_end" in event_types

    active_tick = next(
        payload
        for event_type, payload in io.events
        if event_type == "phase_tick" and payload["phase"] == "active_set"
    )
    assert active_tick["default_beat_hz"] == 0.35
    assert active_tick["beat_multiplier"] == 1.0
    assert active_tick["effective_beat_hz"] == 0.35


def test_runtime_events_are_optional_for_plain_io():
    run_adaptive_workout(tiny_plan(), io=PlainIO(), clock=FakeClock())


def test_rest_after_exercise_runs_as_a_timed_phase():
    io = EventIO()
    plan = tiny_plan()
    plan["exercises"][0]["rest_after_exercise_seconds"] = 2

    run_adaptive_workout(plan, io=io, clock=FakeClock())

    rest_events = [
        payload
        for event_type, payload in io.events
        if event_type == "rest_start" and payload["rest_type"] == "after_exercise"
    ]
    assert rest_events == [
        {
            "rest_type": "after_exercise",
            "seconds": 2,
            "exercise_index": 0,
            "exercise_name": "wall_push_up",
        }
    ]


def test_nonblocking_feedback_does_not_pause_phase_ticks():
    io = NonBlockingFeedbackIO()

    def slow_feedback_understander(user_text, state_snapshot, model):
        import time
        from feedback_understanding import FeedbackUnderstandingResult

        time.sleep(0.05)
        return FeedbackUnderstandingResult(
            intent="fatigue",
            fatigue_level="high",
            difficulty_level="hard",
            preference="neutral",
            confidence=0.9,
            reason="test",
            raw_text=user_text,
            llm_channel="test",
            actions=["decrease_difficulty"],
            safety="none",
        )

    plan = tiny_plan()
    plan["exercises"][0]["avg_set_time"] = 3
    run_adaptive_workout(plan, io=io, clock=FakeClock(), feedback_understander=slow_feedback_understander)

    phase_ticks = [
        payload["seconds_remaining"]
        for event_type, payload in io.events
        if event_type == "phase_tick" and payload["phase"] == "active_set"
    ]
    assert phase_ticks == [3, 2, 1]
    event_types = [event_type for event_type, _payload in io.events]
    assert "feedback_processing_start" in event_types
    assert "feedback_processing_end" in event_types


def test_nonblocking_feedback_is_used_during_between_sets_rest():
    class RestFeedbackIO(EventIO):
        nonblocking_feedback = True

        def __init__(self):
            super().__init__()
            self.sent = False

        def poll_user_input(self):
            if self.sent or not self.events:
                return None
            event_type, payload = self.events[-1]
            if event_type == "phase_tick" and payload["phase"] == "between_sets_rest":
                self.sent = True
                return "too hard"
            return None

    def slow_feedback_understander(user_text, state_snapshot, model):
        import time
        from feedback_understanding import FeedbackUnderstandingResult

        time.sleep(0.05)
        return FeedbackUnderstandingResult(
            intent="fatigue",
            fatigue_level="high",
            difficulty_level="hard",
            preference="neutral",
            confidence=0.9,
            reason="test",
            raw_text=user_text,
            llm_channel="test",
            actions=["decrease_difficulty"],
            safety="none",
        )

    plan = tiny_plan()
    plan["exercises"][0]["total_sets"] = 2
    plan["exercises"][0]["rest_seconds"] = 3
    io = RestFeedbackIO()

    run_adaptive_workout(plan, io=io, clock=FakeClock(), feedback_understander=slow_feedback_understander)

    rest_ticks = [
        payload["seconds_remaining"]
        for event_type, payload in io.events
        if event_type == "phase_tick" and payload["phase"] == "between_sets_rest"
    ]
    rest_beat_values = [
        payload["effective_beat_hz"]
        for event_type, payload in io.events
        if event_type == "phase_tick" and payload["phase"] == "between_sets_rest"
    ]
    event_types = [event_type for event_type, _payload in io.events]

    assert rest_ticks == [3, 2, 1]
    assert rest_beat_values == [0.0, 0.0, 0.0]
    assert "feedback_processing_start" in event_types
    assert "feedback_processing_end" in event_types


def test_nonblocking_feedback_replaces_pending_normal_messages():
    io = NonBlockingFeedbackIO()
    processed = []

    def slow_feedback_understander(user_text, state_snapshot, model):
        import time
        from feedback_understanding import FeedbackUnderstandingResult

        processed.append(user_text)
        time.sleep(0.05)
        return FeedbackUnderstandingResult(
            intent="neutral",
            fatigue_level="medium",
            difficulty_level="appropriate",
            preference="neutral",
            confidence=0.8,
            reason="test",
            raw_text=user_text,
            llm_channel="test",
            actions=["no_action"],
            safety="none",
        )

    io.inputs = ["noise one", "noise two", "real feedback"]
    plan = tiny_plan()
    plan["exercises"][0]["avg_set_time"] = 2

    run_adaptive_workout(plan, io=io, clock=FakeClock(), feedback_understander=slow_feedback_understander)

    assert "noise two" not in processed
    assert "real feedback" in processed


def test_preference_dislike_skips_only_current_exercise_without_global_rest_change():
    class SingleInputIO(EventIO):
        def __init__(self):
            super().__init__()
            self.inputs = ["跳过本动作"]

        def poll_user_input(self):
            if self.inputs:
                return self.inputs.pop(0)
            return None

    def dislike_understander(user_text, state_snapshot, model):
        from feedback_understanding import FeedbackUnderstandingResult

        return FeedbackUnderstandingResult(
            intent="preference_dislike",
            fatigue_level="medium",
            difficulty_level="appropriate",
            preference="dislike",
            confidence=0.85,
            reason="preference_dislike",
            raw_text=user_text,
            llm_channel="test",
            actions=["skip_current_exercise"],
            safety="none",
        )

    plan = {
        "rounds": 1,
        "rest_between_rounds": 0,
        "exercises": [
            {"exercise": "wall_push_up", "avg_set_time": 1, "total_sets": 3, "rest_seconds": 11},
            {"exercise": "bodyweight_squat", "avg_set_time": 1, "total_sets": 3, "rest_seconds": 11},
            {"exercise": "wall_push_up", "avg_set_time": 1, "total_sets": 3, "rest_seconds": 11},
        ],
    }
    io = SingleInputIO()

    state = run_adaptive_workout(
        plan,
        io=io,
        clock=FakeClock(),
        feedback_understander=dislike_understander,
    )

    assert state.workout_plan["exercises"][0]["total_sets"] == 1
    assert state.workout_plan["exercises"][0]["rest_seconds"] == 11
    assert state.workout_plan["exercises"][1]["total_sets"] == 3
    assert state.workout_plan["exercises"][1]["rest_seconds"] == 11
    assert state.workout_plan["exercises"][2]["total_sets"] == 3
    assert state.workout_plan["exercises"][2]["rest_seconds"] == 11
    assert any(item.get("action_type") == "skip_current_exercise" for item in state.adjustment_log)
    assert not any(item.get("action_type") == "adjust_intensity" for item in state.adjustment_log)


def test_preference_like_shorter_rest_reduces_rest_instead_of_medium_fatigue_recovery():
    class SingleInputIO(EventIO):
        def __init__(self):
            super().__init__()
            self.inputs = ["rest can be a little shorter"]

        def poll_user_input(self):
            if self.inputs:
                return self.inputs.pop(0)
            return None

    def shorter_rest_understander(user_text, state_snapshot, model):
        from feedback_understanding import FeedbackUnderstandingResult

        return FeedbackUnderstandingResult(
            intent="preference_like",
            fatigue_level="medium",
            difficulty_level="appropriate",
            preference="like",
            confidence=0.85,
            reason="User requested shorter rest time.",
            raw_text=user_text,
            llm_channel="test",
            actions=["decrease_rest"],
            safety="none",
        )

    plan = {
        "rounds": 1,
        "rest_between_rounds": 0,
        "exercises": [
            {"exercise": "wall_push_up", "avg_set_time": 1, "total_sets": 2, "rest_seconds": 11},
            {"exercise": "bodyweight_squat", "avg_set_time": 1, "total_sets": 2, "rest_seconds": 11},
        ],
    }

    state = run_adaptive_workout(
        plan,
        io=SingleInputIO(),
        clock=FakeClock(),
        feedback_understander=shorter_rest_understander,
    )

    assert state.workout_plan["exercises"][0]["total_sets"] == 2
    assert state.workout_plan["exercises"][1]["total_sets"] == 2
    assert state.workout_plan["exercises"][0]["rest_seconds"] == 8
    assert state.workout_plan["exercises"][1]["rest_seconds"] == 8
    assert state.adjustment_log[0]["rule_id"] == "action_decrease_rest"
    assert state.adjustment_log[0]["rest_multiplier"] == 0.75


def test_pain_confirmation_accepts_single_character_stop():
    state = SessionState(workout_plan=tiny_plan(), awaiting_pain_confirmation=True)
    io = CaptureIO()

    _handle_user_message("停", state, io, decision_engine=None, feedback_understander=None, feedback_model="test")

    assert state.is_active is False
    assert state.end_reason == "user_stopped_after_pain"
