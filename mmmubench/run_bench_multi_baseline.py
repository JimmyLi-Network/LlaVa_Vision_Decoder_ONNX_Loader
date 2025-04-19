import os, sys, ast
import json, re
import multiprocessing as mp
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from datasets import load_dataset

MODEL_DIR = os.path.join('..', 'models')

OUTPUT_DIR = os.path.join('model_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
NUM_GPUS = 8

ds = load_dataset("lmms-lab/MMMU")['validation']

def run_inference(args):
    gpu_id, questions_subset= args

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_DIR, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True, 
    ).to(gpu_id)

    processor = AutoProcessor.from_pretrained(MODEL_DIR)

    results = []

    for v in questions_subset:
        q_id = v['id']
        question = re.sub(r'<image \d+>', '', v['question'])
        choices = ast.literal_eval(v['options'])
        if len(choices) == 0:
            print(f"Skipping question {q_id} due to empty options.")
            continue
        c1 = choices[0]
        c2 = choices[1] if len(choices) > 1 else None
        c3 = choices[2] if len(choices) > 2 else None
        c4 = choices[3] if len(choices) > 3 else None
        options = f"A: {c1}. B: {c2}. C: {c3}. D: {c4}."
        USER_PROMPT = f"""You are provided with a question and several options. 
            Your should output a single uppercase character in A, B, C, D. 
            Question: {question} 
            You need to choose one of the following options: {options}
            Answer: 
            """
        INPUT_FILE = v['image_1']

        print(f"Processing Question: {USER_PROMPT} | Images: {INPUT_FILE}")

        conversation = [
            {

            "role": "user",
            "content": [
                    {"type": "text", "text": USER_PROMPT},
                    {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(images=INPUT_FILE, text=prompt, return_tensors='pt').to(gpu_id, torch.float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=1.0, top_k=50, top_p=0.95)
        generated_text = processor.decode(output[0][2:], skip_special_tokens=True)
        if "Answer:" in generated_text:
            return generated_text.split("Answer:", 1)[1]
        generated_text = generated_text.replace("assistant", "").replace("Assistant", "").replace('\n', '').replace('\r', '').strip()

        if generated_text:
            print(f"Generated Text: {generated_text}")
            results.append({'id': q_id, 'question': question, 'options': options, 'answer': generated_text, 'gt_answer': v['answer']})

    return results

if __name__ == '__main__':
    questions_per_gpu = len(ds) // NUM_GPUS
    question_arr = [ds[i] for i in range(len(ds)) if ds[i]['image_2'] is None]
    question_splits = [question_arr[i * questions_per_gpu:(i + 1) * questions_per_gpu] for i in range(NUM_GPUS)]
    if len(ds) % NUM_GPUS != 0:
        question_splits[-1].extend(question_arr[NUM_GPUS * questions_per_gpu:])

    with mp.Pool(processes=NUM_GPUS) as pool:
        all_results = pool.map(run_inference, [(i, split) for i, split in enumerate(question_splits)])

    final_results = []
    for sub_results in all_results:
        final_results.extend(sub_results)

    output_filename = f"results_baseline.json"

    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"Completed inference on {len(final_results)} questions. Results saved to {OUTPUT_DIR}/{output_filename}")