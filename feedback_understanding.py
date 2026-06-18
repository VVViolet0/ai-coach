import json
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

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
    "confirm_stop_workout",
    "stop_workout",
    "ask_clarification",
    "no_action",
}
ALLOWED_SAFETY = {"none", "pain", "stop_request"}
ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_OS = __import__("os")
LLM_TIMEOUT_SECONDS = int(_OS.getenv("AI_COACH_FEEDBACK_LLM_TIMEOUT_SECONDS", "30"))
USE_SDK_FALLBACK = _OS.getenv("AI_COACH_FEEDBACK_USE_SDK_FALLBACK", "0").strip().lower() in {"1", "true", "yes"}
CONFIRMATION_STRIP_CHARS = " \t\r\n,，.。!！?？;；:：、"

YES_CONFIRMATION_WORDS = {
    "yes",
    "y",
    "continue",
    "go",
    "ok",
    "okay",
    "是",
    "是的",
    "对",
    "对的",
    "继续",
    "可以",
    "好的",
    "好",
    "没问题",
    "沒問題",
    "行",
    "yeah",
    "yep",
    "耶",
}
NO_CONFIRMATION_WORDS = {
    "no",
    "n",
    "stop",
    "quit",
    "end",
    "不",
    "不要",
    "停止",
    "结束",
    "結束",
    "停",
    "停下",
    "不继续",
    "不繼續",
    "不用",
    "nope",
    "否",
}
CONFIRMATION_WORDS = YES_CONFIRMATION_WORDS | NO_CONFIRMATION_WORDS

SAFETY_FEEDBACK_KEYWORDS = (
    "疼",
    "痛",
    "不舒服",
    "膝盖",
    "腰",
    "肩",
    "手腕",
    "脚踝",
    "停止",
    "停了",
    "停下",
    "结束",
    "不想继续",
    "不继续",
    "可以停了",
    "stop",
    "pain",
    "hurt",
    "quit",
)
FEEDBACK_CANDIDATE_KEYWORDS = SAFETY_FEEDBACK_KEYWORDS + (
    "休息",
    "太长",
    "太短",
    "短一点",
    "缩短",
    "减少",
    "少做",
    "少练",
    "别做这么久",
    "不用做这么久",
    "时间",
    "时长",
    "训练时间",
    "长一点",
    "久一点",
    "快",
    "慢",
    "放慢",
    "加快",
    "累",
    "疲劳",
    "太难",
    "太强",
    "太辛苦",
    "轻松",
    "动作",
    "这组",
    "这一组",
    "这轮",
    "深蹲",
    "开合跳",
    "开合步",
    "原地踏步",
    "俯卧撑",
    "墙壁俯卧撑",
    "提膝",
    "侧屈",
    "转体",
    "弓步",
    "提踵",
    "扩胸",
    "出拳",
    "换",
    "跳过",
    "不想做",
    "不喜欢",
    "状态不错",
    "很好",
    "还行",
    "多练",
    "加练",
    "加组",
    "再来",
    "more",
    "longer",
    "rest",
    "faster",
    "slower",
    "hard",
    "easy",
    "skip",
    "replace",
    "dislike",
    "tired",
    "exhausted",
)
TRAINING_CONTEXT_KEYWORDS = (
    "动作",
    "训练",
    "这组",
    "这一组",
    "这轮",
    "这个",
    "做",
    "练",
    "时间",
    "时长",
    "太久",
    "久",
    "少",
    "减少",
    "缩短",
    "加",
    "增加",
    "深蹲",
    "开合跳",
    "开合步",
    "原地踏步",
    "俯卧撑",
    "墙壁俯卧撑",
    "提膝",
    "侧屈",
    "转体",
    "弓步",
    "提踵",
    "扩胸",
    "出拳",
)
COMMON_ASR_HALLUCINATIONS = {
    "谢谢观看",
    "感谢观看",
    "欢迎收看",
    "我认为你会不会有什么事",
}


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


def compact_feedback_text(text: str) -> str:
    return re.sub(r"[\s,，.。!！?？;；:：、]+", "", text.strip().lower())


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def normalize_confirmation_text(text: str) -> str:
    return text.strip().lower().strip(CONFIRMATION_STRIP_CHARS)


def classify_confirmation_text(text: str) -> str:
    normalized = normalize_confirmation_text(text)
    if normalized in YES_CONFIRMATION_WORDS:
        return "yes"
    if normalized in NO_CONFIRMATION_WORDS:
        return "no"
    return ""


def is_confirmation_text(text: str) -> bool:
    return classify_confirmation_text(text) != ""


def is_safety_feedback_text(text: str) -> bool:
    compact = compact_feedback_text(text)
    return _contains_any(compact, SAFETY_FEEDBACK_KEYWORDS)


def is_repetitive_asr_text(text: str) -> bool:
    normalized = compact_feedback_text(text)
    if len(normalized) < 10:
        return False
    if len(set(normalized)) / max(1, len(normalized)) < 0.28:
        return True
    for width in range(2, min(9, len(normalized) // 2 + 1)):
        chunks = [normalized[i : i + width] for i in range(0, len(normalized) - width + 1, width)]
        if len(chunks) >= 3 and max(chunks.count(chunk) for chunk in set(chunks)) >= 3:
            return True
    return False


def classify_feedback_candidate(user_text: str) -> tuple[bool, str]:
    compact = compact_feedback_text(user_text)
    if not compact:
        return False, "empty"
    if is_safety_feedback_text(compact):
        return True, "safety_keyword"
    if any(phrase in compact for phrase in COMMON_ASR_HALLUCINATIONS):
        return False, "common_asr_hallucination"
    if is_repetitive_asr_text(compact):
        return False, "repetitive_asr"
    if _contains_any(compact, FEEDBACK_CANDIDATE_KEYWORDS):
        return True, "feedback_keyword"
    if _contains_any(compact, TRAINING_CONTEXT_KEYWORDS):
        return True, "feedback_context_candidate"
    if len(compact) <= 2:
        return False, "too_short"
    return False, "no_feedback_keyword"


def _fast_result(
    *,
    intent: str,
    raw_text: str,
    actions: list[str],
    reason: str,
    fatigue_level: str = "medium",
    difficulty_level: str = "appropriate",
    preference: str = "neutral",
    safety: str = "none",
    confidence: float = 0.92,
    reply: str = "",
) -> FeedbackUnderstandingResult:
    return FeedbackUnderstandingResult(
        intent=intent,
        fatigue_level=fatigue_level,
        difficulty_level=difficulty_level,
        preference=preference,
        confidence=confidence,
        reason=reason,
        raw_text=raw_text,
        llm_channel="rule_fast_path",
        actions=actions,
        safety=safety,
        reply=reply,
    )


def understand_feedback_fast_path(user_text: str) -> Optional[FeedbackUnderstandingResult]:
    compact = compact_feedback_text(user_text)
    if not compact:
        return None

    if _contains_any(
        compact,
        (
            "停下这一组",
            "停止这一组",
            "结束这一组",
            "停下这组",
            "停止这组",
            "结束这组",
            "停下这个动作",
            "停止这个动作",
            "结束这个动作",
            "这组不做了",
            "这个动作不做了",
        ),
    ):
        return _fast_result(
            intent="preference_dislike",
            raw_text=user_text,
            actions=["skip_current_exercise"],
            preference="dislike",
            confidence=0.94,
            reason="matched stop current exercise keyword",
            reply="好的，我会结束当前动作，进入后续训练。",
        )

    if _contains_any(
        compact,
        (
            "结束整组训练",
            "结束整个训练",
            "停止整组训练",
            "停止整个训练",
            "停下整个训练",
            "本轮训练结束",
            "今天不练了",
            "训练结束",
            "停止训练",
            "stop",
            "quit",
        ),
    ):
        return _fast_result(
            intent="stop",
            raw_text=user_text,
            actions=["stop_workout"],
            safety="stop_request",
            confidence=0.98,
            reason="matched stop keyword",
            reply="好的，现在停止训练。",
        )

    if _contains_any(compact, ("不想继续", "不继续", "可以停了")):
        return _fast_result(
            intent="stop",
            raw_text=user_text,
            actions=["confirm_stop_workout"],
            safety="none",
            confidence=0.9,
            reason="matched ambiguous stop keyword",
            reply="你是想结束整个训练吗？请回答“是”或“否”。",
        )

    if compact in {"停下", "停止", "结束"}:
        return _fast_result(
            intent="preference_dislike",
            raw_text=user_text,
            actions=["skip_current_exercise"],
            preference="dislike",
            confidence=0.86,
            reason="matched bare stop keyword as current exercise stop",
            reply="好的，我会结束当前动作，进入后续训练。",
        )

    if _contains_any(compact, ("疼", "痛", "不舒服", "膝盖", "腰", "肩", "手腕", "脚踝", "pain", "hurt")):
        return _fast_result(
            intent="pain",
            raw_text=user_text,
            actions=["decrease_difficulty"],
            fatigue_level="high",
            difficulty_level="hard",
            safety="pain",
            confidence=0.96,
            reason="matched pain keyword",
        )

    if _contains_any(
        compact,
        ("跳过", "跳过这个", "跳过这个动作", "跳过这组", "跳过这一组", "跳过这组动作", "不想做这个", "不做这个", "skip"),
    ):
        return _fast_result(
            intent="preference_dislike",
            raw_text=user_text,
            actions=["skip_current_exercise"],
            preference="dislike",
            confidence=0.9,
            reason="matched skip keyword",
            reply="好的，我会跳过当前动作。",
        )

    if _contains_any(compact, ("不喜欢这个动作", "不喜欢", "换一个", "换个", "dislike", "replace")):
        return _fast_result(
            intent="preference_dislike",
            raw_text=user_text,
            actions=["replace_current_exercise"],
            preference="dislike",
            confidence=0.9,
            reason="matched dislike keyword",
            reply="好的，我帮你换一个动作。",
        )

    if _contains_any(compact, ("太累", "很累", "累死", "坚持不住", "受不了", "太难", "太强", "太辛苦", "exhausted", "toohard", "tootired")):
        return _fast_result(
            intent="fatigue",
            raw_text=user_text,
            actions=["decrease_difficulty"],
            fatigue_level="high",
            difficulty_level="hard",
            confidence=0.94,
            reason="matched high fatigue keyword",
            reply="好的，我会降低强度。",
        )

    if _contains_any(compact, ("有点累", "稍微累", "一点累", "微累", "有些累", "tired")):
        return _fast_result(
            intent="fatigue",
            raw_text=user_text,
            actions=["slow_tempo"],
            fatigue_level="medium",
            confidence=0.88,
            reason="matched mild fatigue keyword",
            reply="好的，我会先放慢节奏。",
        )

    if _contains_any(
        compact,
        ("休息太短", "休息时间太短", "多休息", "休息久一点", "休息长一点", "想休息", "再休息", "more rest"),
    ):
        return _fast_result(
            intent="pace_down",
            raw_text=user_text,
            actions=["increase_rest"],
            confidence=0.9,
            reason="matched increase rest keyword",
            reply="好的，我会增加休息时间。",
        )

    if _contains_any(compact, ("休息太长", "休息时间太长", "休息短一点", "少休息", "不用休息", "shorter rest")):
        return _fast_result(
            intent="pace_up",
            raw_text=user_text,
            actions=["decrease_rest"],
            fatigue_level="low",
            difficulty_level="easy",
            confidence=0.9,
            reason="matched decrease rest keyword",
            reply="好的，我会缩短休息时间。",
        )

    if _contains_any(compact, ("缩短", "减少", "少做", "少练", "别做这么久", "不用做这么久")) and _contains_any(
        compact,
        ("时长", "时间", "训练", "动作", "这组", "这一组", "深蹲", "开合跳", "开合步", "原地踏步", "俯卧撑", "墙壁俯卧撑", "提膝", "侧屈"),
    ):
        return _fast_result(
            intent="fatigue",
            raw_text=user_text,
            actions=["decrease_sets"],
            fatigue_level="medium",
            difficulty_level="hard",
            confidence=0.91,
            reason="matched shorten exercise duration keyword",
            reply="好的，我会减少当前动作的后续训练量。",
        )

    if _contains_any(compact, ("太快", "节奏太快", "动作太快", "慢一点", "慢点", "放慢", "slower", "slowdown")):
        return _fast_result(
            intent="pace_down",
            raw_text=user_text,
            actions=["slow_tempo"],
            confidence=0.92,
            reason="matched pace down keyword",
            reply="好的，我会放慢节奏。",
        )

    if _contains_any(compact, ("太慢", "节奏太慢", "动作太慢", "快一点", "快点", "加快", "再快", "faster", "speedup")):
        return _fast_result(
            intent="pace_up",
            raw_text=user_text,
            actions=["speed_up_tempo"],
            fatigue_level="low",
            difficulty_level="easy",
            confidence=0.92,
            reason="matched pace up keyword",
            reply="好的，我会加快节奏。",
        )

    if compact in {"可以", "还行", "状态不错", "不错", "很好", "没问题", "ok", "okay"}:
        return _fast_result(
            intent="neutral",
            raw_text=user_text,
            actions=["no_action"],
            confidence=0.85,
            reason="matched neutral feedback",
        )

    return None


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
    try:
        parsed = json.loads(sanitized_flat)
    except json.JSONDecodeError:
        repaired = _repair_common_json_format_errors(sanitized_flat)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            parsed = _extract_payload_fields_best_effort(repaired)
    if not isinstance(parsed, dict):
        raise ValueError("JSON root must be object")
    return parsed


def _repair_common_json_format_errors(text: str) -> str:
    repaired = text.strip()
    # Local models sometimes omit commas between object fields or array strings.
    repaired = re.sub(r'(["\]\}0-9])\s+(")', r"\1, \2", repaired)
    repaired = re.sub(r'(["\]\}0-9])\s*([，；;])\s*(")', r"\1, \3", repaired)
    # Remove trailing commas before closing brackets/braces.
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired


def _extract_payload_fields_best_effort(text: str) -> Dict:
    payload: Dict[str, Any] = {}
    for key in (
        "intent",
        "fatigue_level",
        "difficulty_level",
        "preference",
        "safety",
        "reason",
        "reply",
    ):
        match = re.search(rf'"{key}"\s*:\s*"([^"]*)"', text)
        if match:
            payload[key] = match.group(1)

    actions_match = re.search(r'"actions"\s*:\s*\[(.*?)\]', text, re.DOTALL)
    if actions_match:
        payload["actions"] = re.findall(r'"([^"]+)"', actions_match.group(1))
    else:
        action_match = re.search(r'"actions"\s*:\s*"([^"]+)"', text)
        if action_match:
            payload["actions"] = [action_match.group(1)]

    confidence_match = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if confidence_match:
        payload["confidence"] = float(confidence_match.group(1))

    if not payload:
        raise ValueError("Unable to recover JSON fields")
    return payload


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
        if safety == "stop_request" or "stop_workout" in actions or "confirm_stop_workout" in actions:
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
    compact_state = {
        "exercise": state_snapshot.get("current_exercise_name", ""),
        "phase": state_snapshot.get("phase", ""),
    }
    return f"""
Classify workout feedback. Return JSON only.

Current state:
{json.dumps(compact_state, ensure_ascii=False, separators=(",", ":"))}

User feedback:
{user_text}

Return compact JSON:
{{
  "intent": "stop|pain|fatigue|pace_up|pace_down|preference_dislike|preference_like|neutral|unknown",
  "fatigue_level": "low|medium|high|unknown",
  "difficulty_level": "easy|appropriate|hard|unknown",
  "preference": "like|neutral|dislike|unknown",
  "actions": ["decrease_rest|increase_rest|decrease_sets|increase_sets|slow_tempo|speed_up_tempo|decrease_difficulty|increase_difficulty|skip_current_exercise|replace_current_exercise|stop_workout|ask_clarification|no_action"],
  "safety": "none|pain|stop_request",
  "confidence": 0.0,
  "reason": "brief reason",
  "reply": "short Chinese coach reply"
}}

Mapping:
- Use the user's explicit meaning. Do not infer extra requests.
- "不要/不想做/不喜欢 + current exercise" => preference_dislike, replace_current_exercise.
- "跳过/停下这组/结束这个动作" => preference_dislike, skip_current_exercise.
- "缩短/减少 + exercise/time" => fatigue, decrease_sets.
- "太快/慢一点/放慢" => pace_down, slow_tempo.
- "太慢/快一点/加快" => pace_up, speed_up_tempo.
- "休息太短" => pace_down, increase_rest. "休息太长" => pace_up, decrease_rest.
- "累/太难/强度太大" => fatigue, decrease_difficulty.
- Pain or discomfort => safety pain, decrease_difficulty.
- "结束整个训练/停止训练" => safety stop_request, stop_workout.
- If unclear, use ask_clarification or no_action with confidence <= 0.4.
"""


def understand_feedback(
    user_text: str,
    state_snapshot: Dict,
    model: str = "frob/qwen3.5-instruct:4b",
) -> Union[FeedbackUnderstandingResult, FeedbackUnderstandingFailure, None]:
    prompt = _build_prompt(user_text, state_snapshot)
    tried_channels: list[str] = []
    errors: list[str] = []

    tried_channels.append("ollama_run")
    try:
        cli_result = subprocess.run(
            ["ollama", "run", model, "--think=false"],
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=LLM_TIMEOUT_SECONDS,
            check=True,
        )
        cleaned = _clean_text(cli_result.stdout)
        payload = _loads_json_best_effort(cleaned)
        if isinstance(payload, dict):
            return _normalize_result(payload, user_text, "llm_fallback")
        errors.append("ollama_run invalid JSON root")
    except Exception as exc:
        errors.append(f"ollama_run error: {exc}")

    if USE_SDK_FALLBACK and ollama is not None:
        tried_channels.append("sdk")
        try:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                    think=False,
                )
            except TypeError:
                response = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1},
                )
            payload = _loads_json_best_effort(response["message"]["content"])
            if isinstance(payload, dict):
                return _normalize_result(payload, user_text, "llm_fallback")
            errors.append("sdk invalid JSON root")
        except Exception as exc:
            errors.append(f"sdk error: {exc}")

    return FeedbackUnderstandingFailure(
        error_summary=" | ".join(errors)[:600],
        tried_channels=tried_channels,
        raw_text=user_text,
    )
