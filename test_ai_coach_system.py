import json
import os
import shutil
import unittest
from uuid import uuid4

from ai_coach_system import AICoachSystem
from feedback_understanding import FeedbackUnderstandingResult
from workout_planner import build_planner_exercise_payload, filter_exercises


class FakeClock:
    def __init__(self):
        self.current = 0.0

    def sleep(self, seconds: float) -> None:
        self.current += seconds

    def now(self) -> float:
        return self.current


class ScriptedIO:
    def __init__(self, script):
        self.script = script
        self.outputs = []
        self.inbox = []
        self._fired = set()

    def send(self, message: str) -> None:
        self.outputs.append(message)
        for idx, rule in enumerate(self.script):
            if idx in self._fired:
                continue
            if rule["when_contains"] in message:
                self.inbox.append(rule["message"])
                self._fired.add(idx)

    def poll_user_input(self):
        if self.inbox:
            return self.inbox.pop(0)
        return None

    def close(self):
        return None


class StubAICoachSystem(AICoachSystem):
    def build_intent(self, user_request):
        return {
            "session_goal": "general_fitness",
            "target_muscles": ["full_body"],
            "duration_minutes": 5,
            "intensity_preference": "moderate",
            "experience_level": "beginner",
            "avoid_body_parts": [],
        }

    def build_workout_plan(self, intent):
        return {
            "workout_plan": {
                "rounds": 1,
                "rest_between_rounds": 5,
                "exercises": [
                    {
                        "exercise": "wall_push_up",
                        "avg_set_time": 1,
                        "total_sets": 2,
                        "rest_seconds": 1,
                    },
                    {
                        "exercise": "bodyweight_squat",
                        "avg_set_time": 1,
                        "total_sets": 2,
                        "rest_seconds": 1,
                    },
                ],
            }
        }


class TestAICoachSystem(unittest.TestCase):
    def test_full_body_filter_uses_only_full_body_exercises(self):
        exercises = [
            {"name": "wall_push_up", "target_muscles": ["chest"]},
            {"name": "jumping_jacks", "target_muscles": ["full_body"]},
            {"name": "standing_march", "target_muscles": ["legs", "full_body"]},
        ]

        filtered = filter_exercises(exercises, {"target_muscles": ["full_body"], "avoid_body_parts": []})

        self.assertEqual(["jumping_jacks", "standing_march"], [ex["name"] for ex in filtered])

    def test_low_intensity_filter_excludes_high_intensity_even_for_full_body(self):
        exercises = [
            {"name": "jumping_jacks", "target_muscles": ["full_body"], "intensity_level": "high"},
            {"name": "standing_march", "target_muscles": ["full_body"], "intensity_level": "low"},
            {"name": "step_jack", "target_muscles": ["full_body"], "intensity_level": "low"},
            {"name": "side_step_touch", "target_muscles": ["full_body"], "intensity_level": "low"},
        ]

        filtered = filter_exercises(
            exercises,
            {
                "target_muscles": ["full_body"],
                "avoid_body_parts": [],
                "intensity_preference": "low",
            },
        )

        self.assertEqual(["standing_march", "step_jack", "side_step_touch"], [ex["name"] for ex in filtered])

    def test_moderate_intensity_does_not_exclude_high_intensity(self):
        exercises = [
            {"name": "jumping_jacks", "target_muscles": ["full_body"], "intensity_level": "high"},
            {"name": "standing_side_bend", "target_muscles": ["core"], "intensity_level": "low"},
        ]

        filtered = filter_exercises(
            exercises,
            {
                "target_muscles": ["full_body"],
                "avoid_body_parts": [],
                "intensity_preference": "moderate",
            },
        )

        self.assertEqual(["jumping_jacks"], [ex["name"] for ex in filtered])

    def test_planner_exercise_payload_only_exposes_name_and_time(self):
        payload = build_planner_exercise_payload(
            [
                {
                    "name": "standing_side_bend",
                    "target_muscles": ["core"],
                    "avg_set_time": 40,
                    "intensity_level": "low",
                    "default_beat_hz": 0.25,
                }
            ]
        )

        self.assertEqual([{"name": "standing_side_bend", "avg_set_time": 40}], payload)

    def test_run_session_generates_files_without_execution(self):
        tmpdir = os.path.join("data", f"test_output_{uuid4().hex}")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            system = StubAICoachSystem(output_dir=tmpdir)
            result = system.run_session("I want a short workout", execute_workout=False)

            self.assertEqual("skipped", result["session_end_reason"])
            self.assertIsNone(result["session_log_path"])
            self.assertTrue(os.path.exists(result["intent_path"]))
            self.assertTrue(os.path.exists(result["plan_path"]))

            with open(result["intent_path"], "r", encoding="utf-8") as f:
                intent = json.load(f)
            self.assertEqual("general_fitness", intent["session_goal"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_run_session_executes_workout_and_writes_log(self):
        tmpdir = os.path.join("data", f"test_output_{uuid4().hex}")
        os.makedirs(tmpdir, exist_ok=True)
        try:
            system = StubAICoachSystem(output_dir=tmpdir)
            io = ScriptedIO([
                {"when_contains": "Set 1/2 start", "message": "faster"},
            ])
            clock = FakeClock()
            def understander(user_text, _snapshot, model="frob/qwen3.5-instruct:4b"):
                _ = model
                intent = "pace_up" if "faster" in user_text.lower() else "neutral"
                return FeedbackUnderstandingResult(
                    intent=intent,
                    fatigue_level="low",
                    difficulty_level="easy",
                    preference="neutral",
                    confidence=0.9,
                    reason=f"intent={intent}",
                    raw_text=user_text,
                    llm_channel="sdk",
                    actions=["decrease_rest", "speed_up_tempo"] if intent == "pace_up" else ["no_action"],
                    safety="none",
                )

            result = system.run_session(
                "I want a short workout",
                io=io,
                clock=clock,
                feedback_understander=understander,
                execute_workout=True,
            )

            self.assertEqual("completed", result["session_end_reason"])
            self.assertIsNotNone(result["session_log_path"])
            self.assertTrue(os.path.exists(result["session_log_path"]))
            self.assertTrue(any("正在为你生成训练计划" in line for line in io.outputs))
            self.assertTrue(any("训练计划已生成，开练前概览" in line for line in io.outputs))
            self.assertTrue(any("动作安排" in line for line in io.outputs))

            with open(result["session_log_path"], "r", encoding="utf-8") as f:
                log_payload = json.load(f)
            self.assertIn("summary", log_payload)
            self.assertIn("events", log_payload)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
