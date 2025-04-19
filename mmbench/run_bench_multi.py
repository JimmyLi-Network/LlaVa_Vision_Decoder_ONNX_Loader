import os, sys
import json, re
import multiprocessing as mp

from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LlaVaONNXRunner import VisionLanguageModelONNX

MODEL_DIR = os.path.join('..', 'models', 'onnx')

OUTPUT_DIR = os.path.join('model_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
NUM_GPUS = 8

ds = load_dataset("lmms-lab/MMBench", "en")['test']

INPUT_TYPE = 'image'
Q_EMBED = "q4f16"
Q_VISION = "fp16"
Q_DECODER = "q4f16"
DECODING_STRATEGY = "sampling"

def run_inference(args):
    gpu_id, questions_subset = args

    model = VisionLanguageModelONNX(
        base_model_dir=MODEL_DIR,
        quant_type_embed=Q_EMBED,
        quant_type_vision=Q_VISION,
        quant_type_decoder=Q_DECODER,
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
    question_arr = [ds[i] for i in range(len(ds))]
    questions_per_gpu = len(question_arr) // NUM_GPUS
    question_splits = [question_arr[i * questions_per_gpu:(i + 1) * questions_per_gpu] for i in range(NUM_GPUS)]
    if len(question_arr) % NUM_GPUS != 0:
        question_splits[-1].extend(question_arr[NUM_GPUS * questions_per_gpu:])

    with mp.Pool(processes=NUM_GPUS) as pool:
        all_results = pool.map(run_inference, [(i, split) for i, split in enumerate(question_splits)])

    final_results = []
    for sub_results in all_results:
        final_results.extend(sub_results)

    output_filename = f"results_emb{Q_EMBED}_vis{Q_VISION}_dec{Q_DECODER}.json"

    with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"Completed inference on {len(final_results)} questions. Results saved to {OUTPUT_DIR}/{output_filename}")