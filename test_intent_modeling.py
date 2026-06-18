import json
from intent_modeling import build_intent_prompt, parse_user_intent


INPUT_FILE = "data/intent_test_inputs.txt"
OUTPUT_FILE = "data/intent_test_outputs.json"


def test_intent_prompt_distinguishes_target_and_avoid_body_parts():
    prompt = build_intent_prompt("我今天不要练腿")

    assert "positively wants to train" in prompt
    assert "does not want to train" in prompt
    assert "keep target_muscles as [\"full_body\"]" in prompt
    assert "Never put a negated body part into target_muscles" in prompt
    assert "不要练腿" in prompt


def run_intent_test():

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        inputs = [line.strip() for line in f if line.strip()]

    for i, user_text in enumerate(inputs):
        print(f"Testing input {i+1}: {user_text}")
        intent = parse_user_intent(user_text)
        print("Output intent:", intent)

        result = {
            "input": user_text,
            "output": intent
        }
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

if __name__ == "__main__":
    run_intent_test()
