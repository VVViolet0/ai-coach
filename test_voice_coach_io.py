import asyncio
import json
import threading
import time

import voice_demo.coach_server as coach_server
from voice_demo.coach_server import (
    A_CONDITION_SPEECH,
    B_CONDITION_SPEECH,
    EXPERIMENT_PREPARATION_SPEECH,
    INITIAL_PLANNING_SPEECH,
    INTERACTIVE_CONDITION_SPEECH,
    VoiceCoachIO,
    classify_coach_message,
    classify_feedback_candidate,
    condition_instruction_for,
)
from voice_demo.server import WhisperTranscriber


def test_voice_coach_io_polls_queued_text_once():
    events = []
    io = VoiceCoachIO(events.append)

    io.enqueue_user_text(" 太累了 ")

    assert io.poll_user_input() == "太累了"
    assert io.poll_user_input() is None


def test_voice_coach_io_replaces_pending_normal_feedback_but_keeps_safety():
    events = []
    io = VoiceCoachIO(events.append)

    io.enqueue_user_text("休息太长了")
    io.enqueue_user_text("我膝盖疼", safety=True)
    io.enqueue_user_text("太累了")

    assert io.poll_user_input() == "我膝盖疼"
    assert io.poll_user_input() == "太累了"
    assert io.poll_user_input() is None


def test_feedback_candidate_gate_filters_repetitive_asr_noise():
    accepted, reason = classify_feedback_candidate("这次是什么时候这次是什么时候这次是什么时候")

    assert accepted is False
    assert reason == "repetitive_asr"


def test_feedback_candidate_gate_accepts_rest_and_safety_feedback():
    assert classify_feedback_candidate("休息时间可以短一点")[0] is True
    assert classify_feedback_candidate("我膝盖疼")[1] == "safety_keyword"
    assert classify_feedback_candidate("感觉很好,可以多练一会儿") == (True, "feedback_keyword")


def test_feedback_candidate_gate_accepts_training_context_for_llm_fallback():
    assert classify_feedback_candidate("这个安排有点怪") == (True, "feedback_context_candidate")
    assert classify_feedback_candidate("缩短深蹲的训练时长") == (True, "feedback_keyword")


def test_voice_coach_io_send_publishes_coach_message():
    events = []
    io = VoiceCoachIO(events.append)

    io.send("[Coach] 继续保持节奏")

    speech_id = events[0].pop("speech_id")
    assert speech_id == "speech-1"
    assert events == [
        {
            "type": "coach_message",
            "display": True,
            "speak": True,
            "message": "继续保持节奏",
            "speech_text": "继续保持节奏",
            "category": "coach_reply",
        }
    ]


def test_voice_coach_io_tracks_speech_completion_for_exercise_preview():
    events = []
    io = VoiceCoachIO(events.append)

    io.send("Exercise: 俯卧撑\nInstructions: 身体保持一条直线")

    speech_id = events[0]["speech_id"]
    completed = []
    waiter = threading.Thread(
        target=lambda: completed.append(io.wait_for_speech("exercise_instructions", timeout=1)),
        daemon=True,
    )
    waiter.start()
    time.sleep(0.05)
    io.mark_speech_done(speech_id)
    waiter.join(timeout=1)

    assert completed == [True]


def test_voice_coach_io_close_stops_polling_and_publishing():
    events = []
    io = VoiceCoachIO(events.append)

    io.enqueue_user_text("继续")
    io.close()
    io.send("hidden")
    io.enqueue_user_text("also hidden")

    assert io.poll_user_input() is None
    assert events == []


def test_voice_coach_io_send_event_publishes_runtime_event():
    events = []
    io = VoiceCoachIO(events.append)

    io.send_event("phase_tick", {"seconds_remaining": 3})

    assert events == [{"type": "runtime_event", "event_type": "phase_tick", "payload": {"seconds_remaining": 3}}]


def test_voice_coach_io_can_send_structured_condition_instruction():
    events = []
    io = VoiceCoachIO(events.append)

    io.send_coach_message(A_CONDITION_SPEECH, category="condition_instruction")

    speech_id = events[0].pop("speech_id")
    assert speech_id == "speech-1"
    assert events[0]["type"] == "coach_message"
    assert events[0]["category"] == "condition_instruction"
    assert events[0]["speak"] is True
    assert events[0]["speech_text"] == A_CONDITION_SPEECH


def test_voice_coach_io_logs_transcript_and_links_feedback_context():
    io = VoiceCoachIO(lambda _event: None)
    transcript_id = io.log_transcript(
        {
            "text": "too hard",
            "latency_ms": 1200,
            "asr_ms": 1100,
            "segment_ms": 800,
            "target": "pending_route",
        }
    )
    io.update_transcript_route(transcript_id, "feedback", "feedback_keyword")
    io.enqueue_user_text("too hard", transcript_id=transcript_id)

    context = io.consume_feedback_context(io.poll_user_input())
    transcript = io.get_transcript_log()[0]

    assert context["transcript_id"] == transcript_id
    assert context["feedback_id"]
    assert context["input_method"] == "asr_whisper"
    assert context["submit_method"] == "vad_auto"
    assert transcript["transcript_id"] == transcript_id
    assert transcript["target"] == "feedback"
    assert transcript["route_reason"] == "feedback_keyword"
    assert transcript["timestamp"]


def test_classify_coach_message_filters_non_speech_messages():
    assert classify_coach_message("---------------------------")["speak"] is False
    assert classify_coach_message("Session log saved to: data/session.json")["speak"] is False
    plan_summary = "[Coach] 训练计划已生成，开练前概览：\n- 目标: general_fitness\n- 动作安排:"
    classified = classify_coach_message(plan_summary)
    assert classified["speak"] is True
    assert classified["speech_text"] == "训练计划已生成，准备开始。"


def test_classify_coach_message_converts_timing_prompts_to_short_speech():
    assert classify_coach_message("Set 2/4 start (tempo: normal)")["speech_text"] == "第 2 组开始，共 4 组。"
    assert classify_coach_message("Rest for 30 seconds.")["speech_text"] == "休息 30 秒。"


def test_classify_coach_message_speaks_full_exercise_instructions():
    classified = classify_coach_message(
        "Exercise: 俯卧撑\nInstructions: 双手撑地，距离略宽于肩; 身体保持一条直线"
    )

    assert classified["display"] is True
    assert classified["speak"] is True
    assert classified["category"] == "exercise_instructions"
    assert classified["speech_text"] == "接下来是俯卧撑。动作要求：一、双手撑地，距离略宽于肩；二、身体保持一条直线。"


def test_classify_coach_message_does_not_speak_adjustment_values():
    message = "[Adjustment] next blocks updated: set_delta=-1, rest_multiplier=1.20, tempo=slower"
    classified = classify_coach_message(message)

    assert classified["speak"] is False
    assert classified["category"] == "adjustment"


class FakeWebSocket:
    def __init__(self):
        self.events = []

    async def send_text(self, text):
        self.events.append(json.loads(text))


def test_coach_session_routes_initial_intent_and_feedback(monkeypatch):
    observed = {}

    class FakeCoachSystem:
        def __init__(self, **kwargs):
            observed["init_kwargs"] = kwargs

        def run_session(self, user_request, io, feedback_model, execute_workout=True):
            observed["user_request"] = user_request
            observed["feedback_model"] = feedback_model
            observed["execute_workout"] = execute_workout
            io.send("[Coach] fake planning started")

            deadline = time.time() + 1.0
            feedback = None
            while time.time() < deadline:
                feedback = io.poll_user_input()
                if feedback:
                    break
                time.sleep(0.01)
            observed["feedback"] = feedback
            return {"session_end_reason": "fake_done"}

    async def scenario():
        monkeypatch.setattr(coach_server, "AICoachSystem", FakeCoachSystem)
        websocket = FakeWebSocket()
        session = coach_server.CoachSession(websocket, coach_server.ServerConfig())
        await session.route_transcript("我想做二十分钟全身训练")
        await asyncio.sleep(0.05)
        initial_prompt = next(
            event for event in websocket.events
            if event.get("type") == "coach_message" and event.get("category") == "initial_planning"
        )
        assert initial_prompt["speak"] is True
        assert initial_prompt["speech_text"] == INITIAL_PLANNING_SPEECH

        await session.route_transcript("太累了")

        assert session.coach_thread is not None
        session.coach_thread.join(timeout=2)
        await asyncio.sleep(0.05)

        assert observed["user_request"] == "我想做二十分钟全身训练"
        assert observed["feedback"] == "太累了"
        event_types = [event["type"] for event in websocket.events]
        assert "user_transcript" in event_types
        assert "coach_message" in event_types
        assert "session_summary" in event_types

    asyncio.run(scenario())


def test_coach_session_ignores_noise_transcript_during_workout():
    async def scenario():
        websocket = FakeWebSocket()
        session = coach_server.CoachSession(websocket, coach_server.ServerConfig())
        session.state = "workout_running"

        await session.route_transcript("我认为你会不会有什么事")

        assert session.io.poll_user_input() is None
        transcript_events = [event for event in websocket.events if event.get("type") == "user_transcript"]
        assert transcript_events[-1]["target"] == "ignored_noise"

    asyncio.run(scenario())


def test_experiment_session_prepares_once_then_runs_condition(monkeypatch):
    observed = {"prepare_calls": 0, "run_calls": []}

    class FakeExperimentRunner:
        def __init__(self, **kwargs):
            observed["runner_kwargs"] = kwargs

        def prepare(self, participant_id, initial_request):
            observed["prepare_calls"] += 1
            observed["participant_id"] = participant_id
            observed["initial_request"] = initial_request
            return {
                "participant_id": participant_id,
                "experiment_id": "exp-1",
                "canonical_intent_id": "intent-1",
                "personalized_plan_id": "plan-1",
                "canonical_intent": {"session_goal": "general_fitness", "duration_minutes": 6},
                "personalized_plan": {"rounds": 1, "rest_between_rounds": 0, "exercises": []},
            }

        def run_prepared_condition(self, prepared_experiment, condition_id, io, condition_instruction="", close_io=True):
            observed["run_calls"].append(
                {
                    "condition_id": condition_id,
                    "experiment_id": prepared_experiment["experiment_id"],
                    "condition_instruction": condition_instruction,
                    "close_io": close_io,
                }
            )
            return {
                "condition_id": condition_id,
                "experiment_id": prepared_experiment["experiment_id"],
                "session_end_reason": "fake_done",
            }

    async def scenario():
        monkeypatch.setattr(coach_server, "ExperimentRunner", FakeExperimentRunner)
        websocket = FakeWebSocket()
        config = coach_server.ServerConfig()
        config.experiment_mode = True
        session = coach_server.CoachSession(websocket, config)
        session.participant_id = "P001"

        await session.route_transcript("我想做全身训练")
        session.coach_thread.join(timeout=2)
        await asyncio.sleep(0.05)

        assert observed["prepare_calls"] == 1
        assert observed["participant_id"] == "P001"
        assert session.state == "prepared"
        assert session.prepared_experiment["experiment_id"] == "exp-1"
        assert any(event.get("type") == "experiment_prepared" for event in websocket.events)

        session.start_experiment_condition("B")
        deadline = time.time() + 2
        instruction_event = None
        while time.time() < deadline:
            await asyncio.sleep(0.02)
            instruction_event = next(
                (
                    event for event in websocket.events
                    if event.get("type") == "coach_message"
                    and event.get("category") == "condition_instruction"
                ),
                None,
            )
            if instruction_event:
                break
        assert instruction_event is not None
        session.io.mark_speech_done(instruction_event["speech_id"])
        session.coach_thread.join(timeout=2)
        await asyncio.sleep(0.05)

        assert observed["run_calls"] == [
            {
                "condition_id": "B",
                "experiment_id": "exp-1",
                "condition_instruction": B_CONDITION_SPEECH,
                "close_io": False,
            }
        ]
        assert any(event.get("type") == "condition_summary" for event in websocket.events)
        assert instruction_event["speech_text"] == B_CONDITION_SPEECH
        assert session.state == "prepared"

    asyncio.run(scenario())


def test_experiment_preparation_uses_experiment_specific_speech(monkeypatch):
    class FakeExperimentRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self, participant_id, initial_request):
            return {
                "participant_id": participant_id,
                "experiment_id": "exp-1",
                "canonical_intent_id": "intent-1",
                "personalized_plan_id": "plan-1",
                "canonical_intent": {"session_goal": "general_fitness", "duration_minutes": 6},
                "personalized_plan": {"rounds": 1, "rest_between_rounds": 0, "exercises": []},
            }

    async def scenario():
        monkeypatch.setattr(coach_server, "ExperimentRunner", FakeExperimentRunner)
        websocket = FakeWebSocket()
        config = coach_server.ServerConfig()
        config.experiment_mode = True
        session = coach_server.CoachSession(websocket, config)
        session.participant_id = "P001"

        await session.route_transcript("我想做全身训练")
        session.coach_thread.join(timeout=2)
        await asyncio.sleep(0.05)

        prep_messages = [
            event for event in websocket.events
            if event.get("type") == "coach_message" and event.get("category") == "experiment_preparation"
        ]
        assert prep_messages[-1]["speech_text"] == EXPERIMENT_PREPARATION_SPEECH
        assert not any(
            event.get("type") == "coach_message"
            and event.get("category") == "initial_planning"
            and event.get("speech_text") == INITIAL_PLANNING_SPEECH
            for event in websocket.events
        )

    asyncio.run(scenario())


def test_experiment_text_submit_prepares_with_input_metadata(monkeypatch):
    observed = {}

    class FakeExperimentRunner:
        def __init__(self, **kwargs):
            pass

        def prepare(self, participant_id, initial_request):
            observed["participant_id"] = participant_id
            observed["initial_request"] = initial_request
            return {
                "participant_id": participant_id,
                "experiment_id": "exp-text",
                "canonical_intent_id": "intent-text",
                "personalized_plan_id": "plan-text",
                "canonical_intent": {"session_goal": "general_fitness", "duration_minutes": 6},
                "personalized_plan": {"rounds": 1, "rest_between_rounds": 0, "exercises": []},
            }

    async def scenario():
        monkeypatch.setattr(coach_server, "ExperimentRunner", FakeExperimentRunner)
        websocket = FakeWebSocket()
        config = coach_server.ServerConfig()
        config.experiment_mode = True
        session = coach_server.CoachSession(websocket, config)
        session.participant_id = "P-TEXT"

        await session.submit_text_transcript(
            "我今天有点累，想做一个轻松的全身训练",
            input_method="text_win_h",
            submit_method="manual",
        )
        session.coach_thread.join(timeout=2)
        await asyncio.sleep(0.05)

        assert observed["participant_id"] == "P-TEXT"
        assert observed["initial_request"] == "我今天有点累，想做一个轻松的全身训练"
        transcript = session.io.get_transcript_log()[0]
        assert transcript["input_method"] == "text_win_h"
        assert transcript["submit_method"] == "manual"
        assert transcript["latency_ms"] == 0
        assert transcript["target"] == "initial_intent"

    asyncio.run(scenario())


def test_condition_instruction_copy_by_condition():
    assert condition_instruction_for("A") == A_CONDITION_SPEECH
    assert condition_instruction_for("B") == B_CONDITION_SPEECH
    assert condition_instruction_for("C") == INTERACTIVE_CONDITION_SPEECH


def test_coach_session_marks_non_keyword_transcript_as_ignored_feedback():
    async def scenario():
        websocket = FakeWebSocket()
        session = coach_server.CoachSession(websocket, coach_server.ServerConfig())
        session.state = "workout_running"

        await session.route_transcript("今天外面天气不错")

        assert session.io.poll_user_input() is None
        transcript_events = [event for event in websocket.events if event.get("type") == "user_transcript"]
        assert transcript_events[-1]["target"] == "ignored_feedback"
        assert transcript_events[-1]["reason"] == "no_feedback_keyword"

    asyncio.run(scenario())


def test_experiment_condition_b_ignores_workout_feedback():
    async def scenario():
        websocket = FakeWebSocket()
        session = coach_server.CoachSession(websocket, coach_server.ServerConfig())
        session.state = "workout_running"
        session.experiment_condition_id = "B"

        await session.route_transcript("too hard")

        assert session.io.poll_user_input() is None
        transcript_events = [event for event in websocket.events if event.get("type") == "user_transcript"]
        assert transcript_events[-1]["target"] == "ignored_feedback"
        assert transcript_events[-1]["reason"] == "feedback_disabled_for_condition"

    asyncio.run(scenario())


def test_experiment_condition_c_accepts_direct_chinese_text_feedback():
    async def scenario():
        websocket = FakeWebSocket()
        session = coach_server.CoachSession(websocket, coach_server.ServerConfig())
        session.state = "workout_running"
        session.experiment_condition_id = "C"

        await session.submit_text_transcript("可以停了", input_method="text_win_h", submit_method="auto_idle")

        assert session.io.poll_user_input() == "可以停了"
        transcript = session.io.get_transcript_log()[0]
        transcript_events = [event for event in websocket.events if event.get("type") == "user_transcript"]
        assert transcript["input_method"] == "text_win_h"
        assert transcript["submit_method"] == "auto_idle"
        assert transcript["route_reason"] == "safety_keyword"
        assert transcript_events[-1]["target"] == "feedback"

    asyncio.run(scenario())


def test_confirmation_word_routes_to_executor_even_when_short():
    async def scenario():
        websocket = FakeWebSocket()
        session = coach_server.CoachSession(websocket, coach_server.ServerConfig())
        session.state = "workout_running"
        session.experiment_condition_id = "C"
        session.io.set_pending_confirmation("stop")

        await session.route_transcript("是的")

        assert session.io.poll_user_input() == "是的"
        transcript_events = [event for event in websocket.events if event.get("type") == "user_transcript"]
        assert transcript_events[-1]["target"] == "feedback"
        assert transcript_events[-1]["reason"] == "confirmation"

    asyncio.run(scenario())


def test_experiment_end_condition_requests_current_condition_stop():
    async def scenario():
        websocket = FakeWebSocket()
        config = coach_server.ServerConfig()
        config.experiment_mode = True
        session = coach_server.CoachSession(websocket, config)
        session.state = "workout_running"
        session.experiment_condition_id = "C"

        assert session.request_current_condition_stop() is True
        assert session.io.consume_condition_stop_request() == "manual_condition_stop"
        assert session.state == "workout_running"

    asyncio.run(scenario())


def test_short_utterance_fallback_accepts_chinese_confirmation_words():
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)

    assert transcriber._should_use_short_utterance_fallback({"text": "嗯"}, {"text": "停"}) is True
    assert transcriber._should_use_short_utterance_fallback({"text": "是"}, {"text": "否"}) is True
    assert transcriber._should_use_short_utterance_fallback({"text": "停"}, {"text": "Till"}) is False


def test_short_zh_command_correction_maps_common_stop_mishears():
    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)

    corrected = transcriber._normalize_short_zh_command({"text": "Tion", "asr_ms": 10})

    assert corrected["text"] == "停"
    assert corrected["asr_correction"] == "Tion->停"
