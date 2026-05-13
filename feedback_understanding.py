import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

try:
    import ollama
except ModuleNotFoundError:  # pragma: no cover
    ollama = None


ALLOWED_INTENTS = {
    "stop",
    "pain",
    "fatigue",
    "pace_up",
    "pace_down",
    "preference_dislike",
    "preference_like",
    "neutral",
    "unknown",
}
ALLOWED_FATIGUE = {"low", "medium", "high"}
ALLOWED_DIFFICULTY = {"easy", "appropriate", "hard"}
ALLOWED_PREFERENCE = {"like", "neutral", "dislike"}
ALLOWED_CONDITION_VALUES = {
    "fatigue_level": ALLOWED_FATIGUE | {"unknown"},
    "difficulty_level": ALLOWED_DIFFICULTY | {"unknown"},
    "preference": ALLOWED_PREFERENCE | {"unknown"},
}
ALLOWED_ACTIONS = {
    "decrease_rest",
    "increase_rest",
    "decrease_sets",
    "increase_sets",
    "slow_tempo",
    "speed_up_tempo",
    "decrease_difficulty",
    "increase_difficulty",
    "skip_current_exercise",
    "replace_current_exercise",
    "stop_workout",
    "ask_clarification",
    "no_action",
}
ALLOWED_SAFETY = {"none", "pain", "stop_request"}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


@dataclass
class FeedbackUnderstandingResult:
    intent: str
    fatigue_level: str
    difficulty_level: str
    preference: str
    confidence: float
    reason: str
    raw_text: str
    llm_channel: str
    actions: list[str] = field(default_factory=lambda: ["no_action"])
    safety: str = "none"
    reply: str = ""


@dataclass
class FeedbackUnderstandingFailure:
    error_summary: str
    tried_channels: list[str]
    raw_text: str


def _clean_text(text: str) -> str:
    no_ansi = ANSI_RE.sub("", text)
    # Remove non-printable control chars that may be emitted by terminal rendering.
    cleaned_chars = []
    for ch in no_ansi:
        code = ord(ch)
        if ch in ("\n", "\r", "\t"):
            cleaned_chars.append(ch)
        elif 32 <= code < 127 or code >= 160:
            cleaned_chars.append(ch)
    return "".join(cleaned_chars).strip()


def _extract_json_block(text: str) -> str:
    if not text:
        raise ValueError("Empty response")

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)

    obj = re.search(r"\{.*\}", text, re.DOTALL)
    if obj:
        return obj.group(0)

    return text.strip()


def _loads_json_best_effort(text: str) -> Dict:
    block = _extract_json_block(text)
    try:
        parsed = json.loads(block)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Some models emit invalid raw newlines/control chars inside JSON strings.
    sanitized = "".join(ch for ch in block if ord(ch) >= 32 or ch in ("\n", "\r", "\t"))
    sanitized_flat = sanitized.replace("\r", " ").replace("\n", " ")
    parsed = json.loads(sanitized_flat)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be object")
    return parsed


def _normalize_result(payload: Dict, raw_text: str, llm_channel: str) -> FeedbackUnderstandingResult:
    raw_actions = payload.get("actions", ["no_action"])
    if isinstance(raw_actions, str):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        raw_actions = ["no_action"]

    actions = []
    for action in raw_actions:
        normalized_action = str(action).strip().lower()
        if normalized_action in ALLOWED_ACTIONS and normalized_action not in actions:
            actions.append(normalized_action)
    if not actions:
        actions = ["no_action"]
    elif len(actions) > 1 and "no_action" in actions:
        actions = [action for action in actions if action != "no_action"]

    safety = str(payload.get("safety", "none")).strip().lower()
    if safety not in ALLOWED_SAFETY:
        safety = "none"

    intent = str(payload.get("intent", "")).strip().lower()
    if intent not in ALLOWED_INTENTS:
        if safety == "stop_request" or "stop_workout" in actions:
            intent = "stop"
        elif safety == "pain":
            intent = "pain"
        elif any(action in actions for action in ("decrease_rest", "speed_up_tempo", "increase_difficulty")):
            intent = "pace_up"
        elif any(action in actions for action in ("increase_rest", "slow_tempo", "decrease_difficulty")):
            intent = "pace_down"
        elif any(action in actions for action in ("skip_current_exercise", "replace_current_exercise")):
            intent = "preference_dislike"
        elif actions == ["no_action"]:
            intent = "neutral"
        else:
            intent = "unknown"

    fatigue_level = str(payload.get("fatigue_level", "medium")).strip().lower()
    if fatigue_level not in ALLOWED_CONDITION_VALUES["fatigue_level"]:
        fatigue_level = "unknown"

    difficulty_level = str(payload.get("difficulty_level", "appropriate")).strip().lower()
    if difficulty_level not in ALLOWED_CONDITION_VALUES["difficulty_level"]:
        difficulty_level = "unknown"

    preference = str(payload.get("preference", "neutral")).strip().lower()
    if preference not in ALLOWED_CONDITION_VALUES["preference"]:
        preference = "unknown"

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get("reason", "")).strip()[:280]
    reply = str(payload.get("reply", "")).strip()[:160]

    return FeedbackUnderstandingResult(
        intent=intent,
        fatigue_level=fatigue_level,
        difficulty_level=difficulty_level,
        preference=preference,
        confidence=confidence,
        reason=reason,
        raw_text=raw_text,
        llm_channel=llm_channel,
        actions=actions,
        safety=safety,
        reply=reply,
    )


def _build_prompt(user_text: str, state_snapshot: Dict) -> str:
    return f"""
You interpret workout feedback. Return JSON only.

State:
{json.dumps(state_snapshot, ensure_ascii=False, indent=2)}

User:
{user_text}

Schema:
{{
  "fatigue_level": "low|medium|high|unknown",
  "difficulty_level": "easy|appropriate|hard|unknown",
  "preference": "like|neutral|dislike|unknown",
  "actions": ["allowed action"],
  "safety": "none|pain|stop_request",
  "confidence": 0.0,
  "reason": "short explanation",
  "reply": "short Chinese coach reply"
}}

Allowed actions:
decrease_rest, increase_rest, decrease_sets, increase_sets,
slow_tempo, speed_up_tempo, decrease_difficulty, increase_difficulty,
skip_current_exercise, replace_current_exercise, stop_workout,
ask_clarification, no_action.

Rules:
- First judge fatigue_level, difficulty_level, preference.
- Then choose actions from the user's explicit request.
- Rest complaints map to rest actions.
- Exercise complaints map to exercise actions.
- Fatigue usually maps to decrease_difficulty.
- Pain sets safety=pain.
- Stop request sets safety=stop_request and action stop_workout.
- If unclear, use ask_clarification or no_action with confidence<=0.4.
"""


def understand_feedback(
    user_text: str,
    state_snapshot: Dict,
    model: str = "frob/qwen3.5-instruct:4b",
) -> Union[FeedbackUnderstandingResult, FeedbackUnderstandingFailure, None]:
    prompt = _build_prompt(user_text, state_snapshot)
    tried_channels: list[str] = []
    errors: list[str] = []

    if ollama is not None:
        tried_channels.append("sdk")
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.1},
            )
            payload = _loads_json_best_effort(response["message"]["content"])
            if isinstance(payload, dict):
                return _normalize_result(payload, user_text, "sdk")
            errors.append("sdk invalid JSON root")
        except Exception as exc:
            errors.append(f"sdk error: {exc}")

    tried_channels.append("ollama_run")
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
        payload = _loads_json_best_effort(cleaned)
        if isinstance(payload, dict):
            return _normalize_result(payload, user_text, "ollama_run")
        errors.append("ollama_run invalid JSON root")
    except Exception as exc:
        errors.append(f"ollama_run error: {exc}")

    return FeedbackUnderstandingFailure(
        error_summary=" | ".join(errors)[:600],
        tried_channels=tried_channels,
        raw_text=user_text,
    )
