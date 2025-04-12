import os, sys
import json
import logging
import multiprocessing as mp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LlaVaONNXRunner import VisionLanguageModelONNX

LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

MODEL_DIR = os.path.join('..', 'models', 'onnx')
DATASET_DIR = os.path.join('..', 'vqa_dataset', 'images')
vqa_train = os.path.join(os.path.dirname(__file__), 'gt', 'infographicVQA_train_v1.0.json')
vqa_val = os.path.join(os.path.dirname(__file__), 'gt', 'infographicVQA_val_v1.0.json')
OUTPUT_DIR = os.path.join('model_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
NUM_GPUS = 8

mode = 'val'

if mode == 'train':
    vqa_json = vqa_train
elif mode == 'val':
    vqa_json = vqa_val

INPUT_TYPE = 'image'
Q_EMBED = "q4f16"
Q_VISION = "fp16"
Q_DECODER = "q4f16"
DECODING_STRATEGY = "beam"
NUM_BEAMS = 4

with open(vqa_json, 'rb') as f:
    gt_json = json.load(f)
gt_questions = gt_json['data']

def setup_logging(gpu_id):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - GPU%(processName)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, f"vqa_bench_gpu{gpu_id}.log")),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(f"gpu_{gpu_id}")

def run_inference(args):
    gpu_id, questions_subset = args
    logger = setup_logging(gpu_id)
    logger.info(f"Starting inference on GPU {gpu_id} with {len(questions_subset)} questions")

    model = VisionLanguageModelONNX(
        base_model_dir=MODEL_DIR,
        quant_type_embed=Q_EMBED,
        quant_type_vision=Q_VISION,
        quant_type_decoder=Q_DECODER,
        ort_providers=[('CUDAExecutionProvider', {'device_id': str(gpu_id)}), 'CPUExecutionProvider']
    )

    results = []

    for v in questions_subset:
        q_id = v['questionId']
        USER_PROMPT = v['question']
        INPUT_FILE = os.path.join(DATASET_DIR, v['image_local_name'])

        logger.info(f"Processing Question: {USER_PROMPT} | Image: {INPUT_FILE}")
        generated_text, gen_time = model.generate(
            input_path=INPUT_FILE,
            user_prompt=USER_PROMPT,
            input_type=INPUT_TYPE,
            decoding_strategy=DECODING_STRATEGY,
            num_beams=NUM_BEAMS
        )

        if generated_text:
            logger.info(f"Question ID: {q_id} | Generated Text: {generated_text}")
            results.append({'questionId': q_id, 'answer': generated_text})
            logger.info(f"Generation time: {gen_time:.2f} seconds")
        else:
            logger.warning(f"Question ID: {q_id} | Generation failed")

    model.close()

    return results

if __name__ == '__main__':
    questions_per_gpu = len(gt_questions) // NUM_GPUS
    question_splits = [gt_questions[i * questions_per_gpu:(i + 1) * questions_per_gpu] for i in range(NUM_GPUS)]
    if len(gt_questions) % NUM_GPUS != 0:
        question_splits[-1].extend(gt_questions[NUM_GPUS * questions_per_gpu:])

    with mp.Pool(processes=NUM_GPUS) as pool:
        all_results = pool.map(run_inference, [(i, split) for i, split in enumerate(question_splits)])

    final_results = []
    for sub_results in all_results:
        final_results.extend(sub_results)

    with open(os.path.join(OUTPUT_DIR, f"results_{mode}.json"), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"Completed inference on {len(final_results)} questions. Results saved to {OUTPUT_DIR}/results_{mode}.json")