import asyncio
import json
import time

import voice_demo.coach_server as coach_server
from voice_demo.coach_server import VoiceCoachIO, classify_coach_message
from voice_demo.server import WhisperTranscriber


def test_voice_coach_io_polls_queued_text_once():
    events = []
    io = VoiceCoachIO(events.append)

    io.enqueue_user_text(" 太累了 ")

    assert io.poll_user_input() == "太累了"
    assert io.poll_user_input() is None


def test_voice_coach_io_send_publishes_coach_message():
    events = []
    io = VoiceCoachIO(events.append)

    io.send("[Coach] keep going")

    assert events == [
        {
            "type": "coach_message",
            "display": True,
            "speak": True,
            "message": "keep going",
            "speech_text": "keep going",
            "category": "coach_reply",
        }
    ]


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


def test_classify_coach_message_speaks_adjustment_values():
    message = "[Adjustment] next blocks updated: set_delta=-1, rest_multiplier=1.20, tempo=slower"
    classified = classify_coach_message(message)

    assert classified["speak"] is True
    assert classified["speech_text"] == "强度调整，减少 1 组动作，延长休息时间。"


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
