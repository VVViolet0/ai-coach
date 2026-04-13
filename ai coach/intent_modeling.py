import json
import re
from typing import Any, Dict, List, Optional

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


def _extract_json_block(text: str) -> str:
    """Extract a valid JSON object from plain text or code fences."""
    if not text:
        raise ValueError("Empty response from model")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        return object_match.group(0)

    return text.strip()


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


def _rule_based_intent(user_text: str) -> Dict[str, Any]:
    text = user_text.lower()
    intent = dict(DEFAULT_INTENT)

    if any(k in text for k in ("fat", "lose weight", "cardio", "burn")):
        intent["session_goal"] = "fat_loss"
    elif any(k in text for k in ("strength", "muscle", "power")):
        intent["session_goal"] = "strength"

    if any(k in text for k in ("hard", "intense", "high")):
        intent["intensity_preference"] = "high"
    elif any(k in text for k in ("easy", "light", "gentle", "low")):
        intent["intensity_preference"] = "low"

    duration_match = re.search(r"(\d{1,3})\s*(min|minute|minutes|分钟|分)", text)
    if duration_match:
        intent["duration_minutes"] = max(5, min(180, int(duration_match.group(1))))

    if "beginner" in text:
        intent["experience_level"] = "beginner"
    elif "intermediate" in text:
        intent["experience_level"] = "intermediate"
    elif "advanced" in text:
        intent["experience_level"] = "advanced"

    targets: List[str] = []
    mapping = {
        "legs": ("leg", "legs", "腿"),
        "core": ("core", "abs", "腹"),
        "chest": ("chest", "胸"),
        "back": ("back", "背"),
        "arms": ("arm", "arms", "手臂"),
        "full_body": ("full body", "whole body", "全身"),
    }
    for muscle, keys in mapping.items():
        if any(k in text for k in keys):
            targets.append(muscle)
    if targets:
        intent["target_muscles"] = sorted(set(targets))

    equipment = []
    if any(k in text for k in ("dumbbell", "哑铃")):
        equipment.append("dumbbell")
    if any(k in text for k in ("resistance band", "band", "弹力带")):
        equipment.append("resistance_band")
    intent["equipment_available"] = equipment or ["none"]

    return normalize_intent(intent)


def parse_user_intent(user_text: str, model: str = "frob/qwen3.5-instruct:4b") -> Dict[str, Any]:
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

    try:
        if ollama is None:
            raise RuntimeError("ollama package is not installed")
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        content = response["message"]["content"]
        parsed = json.loads(_extract_json_block(content))
        if not isinstance(parsed, dict):
            raise ValueError("Intent JSON root must be an object")
        return normalize_intent(parsed)
    except Exception:
        return _rule_based_intent(user_text)
