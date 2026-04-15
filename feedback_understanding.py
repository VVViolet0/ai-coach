import json
import re
import subprocess
from dataclasses import dataclass
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
    intent = str(payload.get("intent", "unknown")).strip().lower()
    if intent not in ALLOWED_INTENTS:
        intent = "unknown"

    fatigue_level = str(payload.get("fatigue_level", "medium")).strip().lower()
    if fatigue_level not in ALLOWED_FATIGUE:
        fatigue_level = "medium"

    difficulty_level = str(payload.get("difficulty_level", "appropriate")).strip().lower()
    if difficulty_level not in ALLOWED_DIFFICULTY:
        difficulty_level = "appropriate"

    preference = str(payload.get("preference", "neutral")).strip().lower()
    if preference not in ALLOWED_PREFERENCE:
        preference = "neutral"

    try:
        confidence = float(payload.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reason = str(payload.get("reason", "")).strip()[:280]

    return FeedbackUnderstandingResult(
        intent=intent,
        fatigue_level=fatigue_level,
        difficulty_level=difficulty_level,
        preference=preference,
        confidence=confidence,
        reason=reason,
        raw_text=raw_text,
        llm_channel=llm_channel,
    )


def _build_prompt(user_text: str, state_snapshot: Dict) -> str:
    return f"""
You are an AI workout feedback interpreter.
Your job: read user feedback during workout and output JSON only.

Current workout state snapshot:
{json.dumps(state_snapshot, ensure_ascii=False, indent=2)}

User feedback:
{user_text}

Return JSON schema:
{{
  "intent": "stop | pain | fatigue | pace_up | pace_down | preference_dislike | preference_like | neutral | unknown",
  "fatigue_level": "low | medium | high",
  "difficulty_level": "easy | appropriate | hard",
  "preference": "like | neutral | dislike",
  "confidence": 0.0,
  "reason": "short explanation"
}}

Rules:
- If user likely complains with wording like 'i want to stop' but not explicit final stop decision, still may be stop intent with lower confidence.
- If uncertain, use intent='unknown' and confidence<=0.4.
- Output JSON only.
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
