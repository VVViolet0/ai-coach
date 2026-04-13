import json
import os
import shutil
import unittest
from uuid import uuid4

from ai_coach_system import AICoachSystem


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
            "equipment_available": ["none"],
            "avoid_body_parts": [],
        }

    def build_workout_plan(self, intent):
        return {
            "workout_plan": {
                "rounds": 1,
                "rest_between_rounds": 5,
                "exercises": [
                    {
                        "exercise": "push_up",
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

            result = system.run_session(
                "I want a short workout",
                io=io,
                clock=clock,
                execute_workout=True,
            )

            self.assertEqual("completed", result["session_end_reason"])
            self.assertIsNotNone(result["session_log_path"])
            self.assertTrue(os.path.exists(result["session_log_path"]))

            with open(result["session_log_path"], "r", encoding="utf-8") as f:
                log_payload = json.load(f)
            self.assertIn("summary", log_payload)
            self.assertIn("events", log_payload)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
