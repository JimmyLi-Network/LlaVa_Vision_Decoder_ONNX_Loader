import json
import os
import ollama
import re
from pathlib import Path

def normalize_answer(answer):
    # Extract single letter (A, B, C, D) from answer, ignoring extra text
    match = re.search(r'\b[A-D]\b', answer, re.IGNORECASE)
    return match.group(0).upper() if match else None

def evaluate_response(question_data):
    question = question_data['question']
    options = question_data['options']
    model_answer = question_data['answer']
    gt_answer = question_data['gt_answer']
    
    # Normalize answers to single letter
    model_answer_norm = normalize_answer(model_answer)
    gt_answer_norm = gt_answer.upper()
    
    # Prepare prompt for LLM
    prompt = f"""You are a judge evaluating a model's response to a multiple-choice question. The question is:
        {question}

        Options:
        {options}

        The model's answer is: {model_answer}
        The ground truth answer is: {gt_answer}

        Determine if the model's answer is correct. The model may include extra text, but you should focus on whether the selected option (A, B, C, or D) matches the ground truth. Respond with only one word: "Correct" or "Incorrect"."""
    
    # Call ollama with gemma3:27b model
    response = ollama.generate(model='gemma3:27b', prompt=prompt)
    return response['response'].strip()

def main(input_file):
    # Read JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    total_questions = len(data)
    correct_count = 0
    
    # Evaluate each question
    for item in data:
        result = evaluate_response(item)
        if result == "Correct":
            correct_count += 1
    
    # Calculate score
    score = (correct_count / total_questions) * 100 if total_questions > 0 else 0
    
    # Prepare summary
    summary = f"""Evaluation Summary:
        Total Questions: {total_questions}
        Correct Answers: {correct_count}
        Accuracy: {score:.2f}%"""
    
    print(summary)
    
    # Write summary to output file
    output_file = os.path.splitext(input_file)[0] + "_eval.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(summary)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python llm_judge.py <input_json_file>")
        sys.exit(1)
    main(sys.argv[1])