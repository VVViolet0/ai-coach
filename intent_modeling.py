import json
import re
import subprocess
import time
from typing import Any, Dict, List

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover - dependency may be optional in tests
    ollama = None


DEFAULT_INTENT: Dict[str, Any] = {
    "session_goal": "general_fitness",
    "target_muscles": ["full_body"],
    "duration_minutes": 30,
    "intensity_preference": "moderate",
    "experience_level": "unknown",
    "equipment_available": ["none"],
    "avoid_body_parts": [],
}

ALLOWED_SESSION_GOALS = {"fat_loss", "strength", "general_fitness"}
ALLOWED_INTENSITY = {"low", "moderate", "high"}
ALLOWED_LEVELS = {"beginner", "intermediate", "advanced", "unknown"}
ALLOWED_EQUIPMENT = {"dumbbell", "resistance_band", "none"}
ALLOWED_MUSCLES = {"legs", "core", "chest", "back", "arms", "full_body"}
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
            timeout=240,
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


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    for item in values:
        if isinstance(item, str):
            token = item.strip().lower()
            if token:
                out.append(token)
    return out


def normalize_intent(intent: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(DEFAULT_INTENT)

    session_goal = str(intent.get("session_goal", normalized["session_goal"])).strip().lower()
    normalized["session_goal"] = session_goal if session_goal in ALLOWED_SESSION_GOALS else normalized["session_goal"]

    muscles = [m for m in _normalize_string_list(intent.get("target_muscles")) if m in ALLOWED_MUSCLES]
    normalized["target_muscles"] = muscles or ["full_body"]

    duration = _safe_int(intent.get("duration_minutes"), normalized["duration_minutes"])
    normalized["duration_minutes"] = max(5, min(180, duration))

    intensity = str(intent.get("intensity_preference", normalized["intensity_preference"])).strip().lower()
    normalized["intensity_preference"] = intensity if intensity in ALLOWED_INTENSITY else normalized["intensity_preference"]

    level = str(intent.get("experience_level", normalized["experience_level"])).strip().lower()
    normalized["experience_level"] = level if level in ALLOWED_LEVELS else "unknown"

    equipment = [e for e in _normalize_string_list(intent.get("equipment_available")) if e in ALLOWED_EQUIPMENT]
    normalized["equipment_available"] = equipment or ["none"]

    normalized["avoid_body_parts"] = _normalize_string_list(intent.get("avoid_body_parts"))
    return normalized


def parse_user_intent(
    user_text: str,
    model: str = "frob/qwen3.5-instruct:4b",
    max_retries: int = 2,
) -> Dict[str, Any]:
    prompt = f"""
You are a fitness assistant.
Your task is to extract user's workout intent and output JSON only.

Schema:
{{
 "session_goal": "fat_loss | strength | general_fitness",
 "target_muscles": ["legs","core","chest","back","arms","full_body"],
 "duration_minutes": int,
 "intensity_preference": "low | moderate | high",
 "experience_level": "beginner | intermediate | advanced | unknown",
 "equipment_available": ["dumbbell","resistance_band","none"],
 "avoid_body_parts": []
}}

Rules:
1. duration default 30 if not mentioned
2. target_muscles default ["full_body"]
3. equipment default ["none"]
4. avoid_body_parts default []

User request:
{user_text}
"""

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            content = _chat_with_model(prompt, model=model, temperature=0.1)
            parsed = json.loads(_extract_json_block(content))
            if not isinstance(parsed, dict):
                raise ValueError("Intent JSON root must be an object")
            return normalize_intent(parsed)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                wait_seconds = 3.0 * (attempt + 1) if "502" in str(exc) else 1.5 * (attempt + 1)
                time.sleep(wait_seconds)

    detail = str(last_error)
    if "502" in detail:
        detail += " | Ollama returned 502 repeatedly. Try restarting `ollama serve` and retry."

    raise RuntimeError(
        "Failed to parse user intent via LLM after retries. "
        f"Model: {model}. Last error: {detail}"
    )
