import os, sys
import json, re
import multiprocessing as mp
import argparse

from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LlaVaONNXRunner import VisionLanguageModelONNX

MODEL_DIR = os.path.join('..', 'models', 'onnx')

OUTPUT_DIR = os.path.join('model_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
NUM_GPUS = 8

ds = load_dataset("lmms-lab/MMBench", "en")['dev']

INPUT_TYPE = 'image'
DECODING_STRATEGY = "sampling"

def run_inference(args):
    gpu_id, questions_subset, q_embed, q_vision, q_decoder = args

    model = VisionLanguageModelONNX(
        base_model_dir=MODEL_DIR,
        quant_type_embed=q_embed,
        quant_type_vision=q_vision,
        quant_type_decoder=q_decoder,
        ort_providers=[('CUDAExecutionProvider', {'device_id': str(gpu_id)}), 'CPUExecutionProvider']
    )

    results = []

    for v in questions_subset:
        q_id = v['index']
        question = v['question']
        options = f"A: {v['A']}. B: {v['B']}. C: {v['C']}. D: {v['D']}."
        USER_PROMPT = f"""You are provided with a question and several options. 
            Your should output a single uppercase character in A, B, C, D. 
            Question: {question} 
            You need to choose one of the following options: {options}
            Answer: 
            """
        INPUT_FILES = [v['image']]

        print(f"Processing Question: {USER_PROMPT} | Images: {INPUT_FILES}")

        generated_text, _ = model.generate(
            inputs=INPUT_FILES,
            user_prompt=USER_PROMPT,
            input_type=INPUT_TYPE,
            decoding_strategy=DECODING_STRATEGY,
        )

        if generated_text:
            print(f"Generated Text: {generated_text}")
            results.append({'id': q_id, 'question': question, 'options': options, 'answer': generated_text, 'gt_answer': v['answer']})

    model.close()

    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run inference on MMBench dataset with ONNX vision-language model.")
    parser.add_argument('--q_embed', type=str, default='q4f16', help='Quantization type for embeddings (e.g., q4f16)')
    parser.add_argument('--q_vision', type=str, default='fp16', help='Quantization type for vision model (e.g., fp16)')
    parser.add_argument('--q_decoder', type=str, default='q4f16', help='Quantization type for decoder (e.g., q4f16)')
    args = parser.parse_args()

    question_arr = [ds[i] for i in range(len(ds))]
    questions_per_gpu = len(question_arr) // NUM_GPUS
    question_splits = [question_arr[i * questions_per_gpu:(i + 1) * questions_per_gpu] for i in range(NUM_GPUS)]
    if len(question_arr) % NUM_GPUS != 0:
        question_splits[-1].extend(question_arr[NUM_GPUS * questions_per_gpu:])

    with mp.Pool(processes=NUM_GPUS) as pool:
        all_results = pool.map(run_inference, [(i, split, args.q_embed, args.q_vision, args.q_decoder) for i, split in enumerate(question_splits)])

    final_results = []
    for sub_results in all_results:
        final_results.extend(sub_results)

    output_filename = f"results_emb{args.q_embed}_vis{args.q_vision}_dec{args.q_decoder}.json"

    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"Completed inference on {len(final_results)} questions. Results saved to {OUTPUT_DIR}/{output_filename}")