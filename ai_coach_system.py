import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from intent_modeling import parse_user_intent
from workout_executor import (
    DecisionEngine,
    ExecutorIO,
    SessionState,
    Clock,
    FeedbackUnderstander,
    run_adaptive_workout,
)
from workout_planner import generate_workout_plan, load_exercise_library


class AICoachSystem:
    def __init__(
        self,
        exercise_library_path: str = "libraries/exercise_library.json",
        output_dir: str = "data",
        intent_model: str = "frob/qwen3.5-instruct:4b",
        planner_model: str = "frob/qwen3.5-instruct:4b",
    ) -> None:
        self.exercise_library_path = exercise_library_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.intent_model = intent_model
        self.planner_model = planner_model

    def build_intent(self, user_request: str) -> Dict[str, Any]:
        return parse_user_intent(user_request, model=self.intent_model)

    def build_workout_plan(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        library = load_exercise_library(self.exercise_library_path)
        return generate_workout_plan(intent, library, model=self.planner_model)

    def save_json(self, payload: Dict[str, Any], path: str | Path) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return str(target)

    def _emit_message(self, message: str, io: Optional[ExecutorIO]) -> None:
        if io:
            io.send(message)
        else:
            print(message)

    def _emit_event(self, io: Optional[ExecutorIO], event_type: str, payload: Dict[str, Any]) -> None:
        if not io:
            return
        send_event = getattr(io, "send_event", None)
        if callable(send_event):
            send_event(event_type, payload)

    def _summarize_workout_plan(self, plan: Dict[str, Any], intent: Dict[str, Any]) -> str:
        plan_obj = plan["workout_plan"]
        lines = [
            "[Coach] 训练计划已生成，开练前概览：",
            (
                f"- 目标: {intent.get('session_goal', 'general_fitness')} | "
                f"时长: {intent.get('duration_minutes', 30)} 分钟"
            ),
            (
                f"- 轮数: {plan_obj.get('rounds', 1)} | "
                f"轮间休息: {plan_obj.get('rest_between_rounds', 0)} 秒"
            ),
            "- 动作安排:",
        ]
        for ex in plan_obj.get("exercises", []):
            lines.append(
                f"  - {ex.get('exercise')} | sets={ex.get('total_sets')} | rest={ex.get('rest_seconds')}s"
            )
        return "\n".join(lines)

    def run_session(
        self,
        user_request: str,
        intent_output_path: Optional[str] = None,
        plan_output_path: Optional[str] = None,
        session_log_path: Optional[str] = None,
        io: Optional[ExecutorIO] = None,
        clock: Optional[Clock] = None,
        decision_engine: Optional[DecisionEngine] = None,
        feedback_understander: Optional[FeedbackUnderstander] = None,
        feedback_model: str = "frob/qwen3.5-instruct:4b",
        execute_workout: bool = True,
    ) -> Dict[str, Any]:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")

        intent = self.build_intent(user_request)
        intent_file = intent_output_path or str(self.output_dir / f"intent_{now}.json")
        self.save_json(intent, intent_file)

        library = load_exercise_library(self.exercise_library_path)
        if execute_workout:
            self._emit_message("[Coach] 正在为你生成训练计划，请稍等...", io)
        plan = self.build_workout_plan(intent)
        plan_file = plan_output_path or str(self.output_dir / f"workout_plan_{now}.json")
        self.save_json(plan, plan_file)
        self._emit_event(
            io,
            "plan_generated",
            {
                "intent": intent,
                "workout_plan": plan["workout_plan"],
                "intent_path": intent_file,
                "plan_path": plan_file,
            },
        )

        log_file = session_log_path or str(self.output_dir / f"session_log_{now}.json")
        session_end_reason = "skipped"
        effective_log_path: Optional[str] = None
        if execute_workout:
            self._emit_message(self._summarize_workout_plan(plan, intent), io)
            if clock:
                clock.sleep(10)
            else:
                time.sleep(10)
            session_state: SessionState = run_adaptive_workout(
                plan["workout_plan"],
                io=io,
                clock=clock,
                decision_engine=decision_engine,
                feedback_understander=feedback_understander,
                feedback_model=feedback_model,
                log_path=log_file,
                initial_intent=intent,
                exercise_library=library,
            )
            session_end_reason = session_state.end_reason
            effective_log_path = log_file

        return {
            "intent": intent,
            "plan": plan,
            "session_end_reason": session_end_reason,
            "intent_path": intent_file,
            "plan_path": plan_file,
            "session_log_path": effective_log_path,
        }


def _read_user_request(cli_request: Optional[str]) -> str:
    if cli_request and cli_request.strip():
        return cli_request.strip()
    return input("Please describe your training request: ").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete AI Coach workflow (intent -> planning -> adaptive execution).")
    parser.add_argument("--request", help="User workout request text.")
    parser.add_argument("--exercise-library", default="libraries/exercise_library.json", help="Exercise library JSON path.")
    parser.add_argument("--output-dir", default="data", help="Directory for generated intent/plan/log files.")
    parser.add_argument("--intent-model", default="frob/qwen3.5-instruct:4b", help="Ollama model for intent modeling.")
    parser.add_argument("--planner-model", default="frob/qwen3.5-instruct:4b", help="Ollama model for workout planning.")
    args = parser.parse_args()

    user_request = _read_user_request(args.request)
    if not user_request:
        raise ValueError("Training request cannot be empty.")

    system = AICoachSystem(
        exercise_library_path=args.exercise_library,
        output_dir=args.output_dir,
        intent_model=args.intent_model,
        planner_model=args.planner_model,
    )

    result = system.run_session(user_request)
    print("\n===== AI COACH SUMMARY =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

