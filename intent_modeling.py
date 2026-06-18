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
    "avoid_body_parts": [],
}

ALLOWED_SESSION_GOALS = {"fat_loss", "strength", "general_fitness"}
ALLOWED_INTENSITY = {"low", "moderate", "high"}
ALLOWED_LEVELS = {"beginner", "intermediate", "advanced", "unknown"}
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
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": temperature},
                    think=False,
                )
            except TypeError:
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
            ["ollama", "run", model, "--think=false"],
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

    normalized["avoid_body_parts"] = _normalize_string_list(intent.get("avoid_body_parts"))
    return normalized


def parse_user_intent(
    user_text: str,
    model: str = "frob/qwen3.5-instruct:4b",
    max_retries: int = 2,
) -> Dict[str, Any]:
    prompt = build_intent_prompt(user_text)

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


def build_intent_prompt(user_text: str) -> str:
    return f"""
You are a fitness assistant.
Your task is to extract user's workout intent and output JSON only.

Schema:
{{
 "session_goal": "fat_loss | strength | general_fitness",
 "target_muscles": ["legs","core","chest","back","arms","full_body"],
 "duration_minutes": int,
 "intensity_preference": "low | moderate | high",
 "experience_level": "beginner | intermediate | advanced | unknown",
 "avoid_body_parts": []
}}

Rules:
1. duration default 30 if not mentioned.
2. target_muscles means body parts the user positively wants to train.
3. avoid_body_parts means body parts the user wants to avoid or does not want to train.
4. If the user only says what they do NOT want, put those parts in avoid_body_parts and keep target_muscles as ["full_body"].
5. Never put a negated body part into target_muscles. For example, "不要练腿" means avoid_body_parts ["legs"], not target_muscles ["legs"].
6. The user request may come from Chinese speech recognition and may contain homophone or near-sound errors. Infer intent from the fitness context, but preserve negation words such as 不要, 不想, 不练, 避免.

User request:
{user_text}
"""
