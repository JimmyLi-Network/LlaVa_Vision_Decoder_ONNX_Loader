import json
import re
import sys
import os,ast
import ollama

MODEL_NAME = "gemma3:27b"

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def simple_match(ans, gt):
    gt_arr = ast.literal_eval(gt)
    for i in gt_arr:
        if ans.strip().upper() == i.strip().upper():
            return True
    return False

def loose_match(ans, gt):
    pattern = fr'(?<![A-Z]){gt.upper()}(?![A-Z])'
    return re.search(pattern, ans.upper()) is not None

def ask_ollama(answer, gt_answer, options):
    prompt = f"""You are a strict grader.
    There are multiple correct answers for this question, as long as the student answer is in the ground truth answer list, it is correct.
    Ground truth answer: {gt_answer}
    Student answer: {answer}

    Should this answer be considered correct? Reply only "Yes" or "No"."""
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    result = response['message']['content']
    return "yes" in result.lower()

def grade_item(item):
    ans = item["answer"]
    gt = item["gt_answer"]
    options = item["options"]

    if simple_match(ans, gt):
        return True, "strict"
    # if loose_match(ans, gt):
    #     return True, "fuzzy"
    if ask_ollama(ans, gt, options):
        return True, "ollama"
    return False, "wrong"

def grade_all(data, output_path):
    total = len(data)
    correct = 0
    stats = {"strict": 0, "fuzzy": 0, "ollama": 0, "wrong": 0}

    with open(output_path, 'w', encoding='utf-8') as out:
        for item in data:
            ok, mode = grade_item(item)
            stats[mode] += 1
            if ok:
                correct += 1
            out.write(json.dumps({
                "id": item["id"],
                "answer": item["answer"],
                "gt_answer": item["gt_answer"],
                "result": "correct" if ok else "wrong",
                "mode": mode
            }, ensure_ascii=False) + '\n')

        summary = f"\nCorrect: {correct}/{total} ({correct / total:.2%})\nBreakdown: {stats}\n"
        print(summary)
        out.write(summary)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python grade.py your_file.json")
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        sys.exit(1)

    output_path = re.sub(r'\.json$', '_eval.txt', input_path)
    data = load_data(input_path)
    grade_all(data, output_path)
