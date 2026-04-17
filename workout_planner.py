import json
import re
import subprocess
import time
from typing import Any, Dict, List

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover - dependency may be optional in tests
    ollama = None


MIN_REST = 10
MAX_REST = 180
MIN_SETS = 1
MAX_SETS = 6
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("Empty response from model")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        return object_match.group(0)
    return text.strip()


def _clean_text(text: str) -> str:
    return ANSI_RE.sub("", text).strip()


def _chat_with_model(prompt: str, model: str, temperature: float) -> str:
    package_error: Exception | None = None

    if ollama is not None:
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature},
            )
            return response["message"]["content"]
        except Exception as exc:
            package_error = exc

    try:
        cli_result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=300,
            check=True,
        )
        cleaned = _clean_text(cli_result.stdout)
        if cleaned:
            return cleaned
        raise RuntimeError(f"Empty response from ollama CLI. stderr={cli_result.stderr}")
    except Exception as cli_exc:
        if package_error is not None:
            raise RuntimeError(f"ollama SDK error: {package_error}; ollama CLI error: {cli_exc}") from cli_exc
        raise RuntimeError(f"ollama CLI error: {cli_exc}") from cli_exc


def load_exercise_library(path: str = "libraries/exercise_library.json") -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_exercises(exercises: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
    available_equipment = set(intent.get("equipment_available", ["none"]))
    avoids = set(intent.get("avoid_body_parts", []))
    targets = set(intent.get("target_muscles", ["full_body"]))

    filtered: List[Dict[str, Any]] = []
    for ex in exercises:
        if ex.get("equipment") not in available_equipment:
            continue

        ex_muscles = set(ex.get("target_muscles", []))
        if avoids.intersection(ex_muscles):
            continue

        if "full_body" in targets or targets.intersection(ex_muscles):
            filtered.append(ex)

    if filtered:
        return filtered

    for ex in exercises:
        if ex.get("equipment") in available_equipment and not avoids.intersection(set(ex.get("target_muscles", []))):
            filtered.append(ex)

    return filtered


def estimate_structure(intent: Dict[str, Any]) -> str:
    target_seconds = int(intent["duration_minutes"]) * 60

    if intent["session_goal"] == "fat_loss":
        rounds = 3
        avg_exercise_time = 120
        max_exercises = max(1, target_seconds // (rounds * avg_exercise_time))
        return f"""
Create a FAT LOSS workout with circuit training.
- choose up to {max_exercises} exercises
- repeat for 3 rounds
- minimal rest between exercises around 15 seconds
- rest around 60 seconds between rounds
"""

    if intent["session_goal"] == "strength":
        rounds = 1
        avg_exercise_time = 240
        max_exercises = max(1, target_seconds // (rounds * avg_exercise_time))
        return f"""
Create a STRENGTH workout.
- choose up to {max_exercises} exercises
- each exercise has 3 to 5 sets
- rest 90 to 120 seconds between sets
- only 1 round and rest_between_rounds is 0
"""

    rounds = 2
    avg_exercise_time = 150
    max_exercises = max(1, target_seconds // (rounds * avg_exercise_time))
    return f"""
Create a GENERAL FITNESS workout.
- choose up to {max_exercises} exercises
- each exercise has 3 to 4 sets
- rest 30 to 60 seconds
- repeat for 2 rounds and 30 seconds rest_between_rounds
"""


def build_prompt(intent: Dict[str, Any], filtered_exercise_library: List[Dict[str, Any]]) -> str:
    goal_policy = estimate_structure(intent)
    return f"""
You are an AI fitness coach.
Generate a safe and effective workout plan in JSON only.

User training intent:
{json.dumps(intent, ensure_ascii=False, indent=2)}

Exercise library:
{json.dumps(filtered_exercise_library, ensure_ascii=False, indent=2)}

{goal_policy}
Adjust rounds/sets/rest according to intensity_preference.
Only use exercises in the library.

Output schema:
{{
 "workout_plan": {{
  "rounds": int,
  "rest_between_rounds": int,
  "exercises": [
   {{
    "exercise": "exercise_name",
    "avg_set_time": 40,
    "total_sets": 3,
    "rest_seconds": 30
   }}
  ]
 }}
}}
"""


def _normalize_workout_plan(plan: Dict[str, Any], fallback_library: List[Dict[str, Any]]) -> Dict[str, Any]:
    if "workout_plan" in plan and isinstance(plan["workout_plan"], dict):
        plan_obj = plan["workout_plan"]
    else:
        plan_obj = plan

    exercises = plan_obj.get("exercises", [])
    if not isinstance(exercises, list):
        raise ValueError("LLM output invalid: exercises must be a list")

    library_by_name = {ex["name"]: ex for ex in fallback_library if "name" in ex}
    normalized_exercises = []
    for ex in exercises:
        if not isinstance(ex, dict):
            continue

        exercise_name = str(ex.get("exercise", "")).strip()
        if not exercise_name or exercise_name not in library_by_name:
            continue

        base_time = int(ex.get("avg_set_time", library_by_name[exercise_name].get("avg_set_time", 30)))
        total_sets = int(ex.get("total_sets", 3))
        rest_seconds = int(ex.get("rest_seconds", 30))

        normalized_exercises.append(
            {
                "exercise": exercise_name,
                "avg_set_time": max(10, min(300, base_time)),
                "total_sets": max(MIN_SETS, min(MAX_SETS, total_sets)),
                "rest_seconds": max(MIN_REST, min(MAX_REST, rest_seconds)),
            }
        )

    if not normalized_exercises:
        raise ValueError("LLM output invalid: no valid exercises found in plan")

    rounds = int(plan_obj.get("rounds", 1))
    rest_between_rounds = int(plan_obj.get("rest_between_rounds", 0))

    return {
        "workout_plan": {
            "rounds": max(1, min(5, rounds)),
            "rest_between_rounds": max(0, min(MAX_REST, rest_between_rounds)),
            "exercises": normalized_exercises,
        }
    }


def call_ollama(prompt: str, model: str = "frob/qwen3.5-instruct:4b") -> Dict[str, Any]:
    content = _chat_with_model(prompt, model=model, temperature=0.2)
    return json.loads(_extract_json_block(content))


def generate_workout_plan(
    intent: Dict[str, Any],
    exercise_library: List[Dict[str, Any]],
    model: str = "frob/qwen3.5-instruct:4b",
    max_retries: int = 4,
) -> Dict[str, Any]:
    filtered_exercises = filter_exercises(exercise_library, intent)
    if not filtered_exercises:
        raise ValueError("No candidate exercises available after filtering. Please adjust user intent.")

    prompt = build_prompt(intent, filtered_exercises)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            llm_output = call_ollama(prompt, model=model)
            return _normalize_workout_plan(llm_output, filtered_exercises)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = 3.0 * (attempt + 1) if "502" in str(exc) else 1.5 * (attempt + 1)
                time.sleep(wait_seconds)

    detail = str(last_error)
    if "502" in detail:
        detail += " | Ollama returned 502 repeatedly. Try restarting `ollama serve` and retry."

    raise RuntimeError(
        "Failed to generate a valid workout plan via LLM after retries. "
        f"Model: {model}. Last error: {detail}"
    )


if __name__ == "__main__":
    demo_intent = {
        "session_goal": "general_fitness",
        "target_muscles": ["full_body"],
        "duration_minutes": 15,
        "intensity_preference": "moderate",
        "experience_level": "unknown",
        "equipment_available": ["none"],
        "avoid_body_parts": ["arms"],
    }

    library = load_exercise_library()
    plan = generate_workout_plan(demo_intent, library)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
