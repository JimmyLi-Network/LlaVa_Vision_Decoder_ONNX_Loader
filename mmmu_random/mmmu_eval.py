import json
import ollama
import ast
import sys
import os

# 检查命令行参数
if len(sys.argv) != 2:
    print("Usage: python script.py <data.json>")
    sys.exit(1)

# 获取输入文件路径
input_file = sys.argv[1]

# 验证文件存在
if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found")
    sys.exit(1)

# 构造输出文件名（去掉 .json 后缀，加 _eval.txt）
output_file = os.path.splitext(input_file)[0] + "_eval.txt"

# 读取 JSON 文件
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
except json.JSONDecodeError:
    print(f"Error: '{input_file}' is not a valid JSON file")
    sys.exit(1)

# 评估结果
results = []
correct_count = 0
total_count = 0  # 只统计有效响应的样本
all_samples_count = len(data)  # 所有样本（包括错误）

# 遍历每个样本
for sample in data:
    id = sample['id']
    question = sample['question']
    options_str = sample['options']
    answer = sample['answer']
    gt_answer = sample['gt_answer']

    # 解析 options 字符串为列表
    try:
        options = ast.literal_eval(options_str)
    except (ValueError, SyntaxError):
        print(f"Error parsing options for id {id}: {options_str}")
        continue

    # 映射 gt_answer (A/B/C/D) 到选项内容
    try:
        gt_index = ord(gt_answer) - ord('A')  # A=0, B=1, C=2, D=3
        gt_option = options[gt_index]
    except (IndexError, TypeError):
        print(f"Error mapping gt_answer for id {id}: {gt_answer}")
        continue

    # 构造提示
    prompt = f"""
You are a judge evaluating a model's answer. Given the following:

Question: {question}
Options: {options}
Model's answer: {answer}
Correct answer (ground truth): {gt_option}

Determine if the model's answer is correct. The model's answer is correct if it matches the ground truth option exactly or clearly indicates the same option (e.g., by reproducing the option text or selecting the corresponding letter).

Respond with exactly one word: "correct" if the answer is correct, "incorrect" if it is not. Do not include any other text, explanation, or formatting.
"""

    # 调用 Ollama 的 gemma3:27b 模型
    try:
        response = ollama.chat(
            model='gemma3:27b',
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.2, 'top_p': 0.9, 'max_tokens': 50}
        )
        content = response['message']['content'].strip().lower()
        # 检查响应
        if content == 'correct':
            results.append({
                "id": id,
                "is_correct": True,
                "explanation": "Model judged the answer as correct"
            })
            total_count += 1
            correct_count += 1
        elif content == 'incorrect':
            results.append({
                "id": id,
                "is_correct": False,
                "explanation": "Model judged the answer as incorrect"
            })
            total_count += 1
        else:
            print(f"Invalid LLM response for id {id}: {content}")
            continue
    except Exception as e:
        print(f"Error calling LLM for id {id}: {str(e)}")
        continue

# 准备输出内容
output_content = []
output_content.append(f"All samples: {all_samples_count}")
output_content.append(f"Valid samples: {total_count}")
output_content.append(f"Correct samples: {correct_count}")
output_content.append(f"Accuracy: {correct_count / total_count:.2%}" if total_count > 0 else "Accuracy: N/A (no valid samples)")
output_content.append("")  # 空行
output_content.append("Detailed results:")
for result in results:
    output_content.append(json.dumps(result, indent=2))

# 打印到控制台
for line in output_content:
    print(line)

# 写入文件
try:
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_content))
    print(f"\nResults written to {output_file}")
except IOError as e:
    print(f"Error writing to {output_file}: {str(e)}")