import asyncio
import json
import threading
import time

import voice_demo.coach_server as coach_server
from voice_demo.coach_server import (
    INITIAL_PLANNING_SPEECH,
    VoiceCoachIO,
    classify_coach_message,
    classify_feedback_candidate,
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
