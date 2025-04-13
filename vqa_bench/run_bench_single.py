import os, sys
import json
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LlaVaONNXRunner import VisionLanguageModelONNX

LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "vqa_bench.log")),
        logging.StreamHandler() 
    ]
)

logger = logging.getLogger(__name__)

# --- Configure Paths and Settings ---
MODEL_DIR = os.path.join('..', 'models', 'onnx')
DATASET_DIR = os.path.join('..','vqa_dataset', 'images')
vqa_train = os.path.join(os.path.dirname(__file__), 'gt', 'infographicVQA_train_v1.0.json')
vqa_val = os.path.join(os.path.dirname(__file__), 'gt', 'infographicVQA_val_v1.0.json')
OUTPUT_DIR = os.path.join('model_outputs')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
nums_gpus = 1

mode = 'val'

if mode == 'train':
    vqa_json = vqa_train
elif mode == 'val':
    vqa_json = vqa_val


# --- Input Settings ---
INPUT_TYPE = 'image'   # MODIFY ('image' or 'video')

# --- Quantization Settings ---
Q_EMBED = "q4f16"
Q_VISION = "fp16"
Q_DECODER = "q4f16"

# --- Decoding Settings ---
DECODING_STRATEGY = "beam"
NUM_BEAMS = 4

# --- Load Input ---
with open (vqa_json, 'rb') as f:
    gt_json = json.load(f)
gt_questions = gt_json['data']

model = None

# --- Load Model ---
print("Loading model...")
model = VisionLanguageModelONNX(
    base_model_dir=MODEL_DIR,
    quant_type_embed=Q_EMBED,
    quant_type_vision=Q_VISION,
    quant_type_decoder=Q_DECODER,
    ort_providers=[('CUDAExecutionProvider', {'device_id': '0'}), 'CPUExecutionProvider'],
)
print("Model loaded.")

results = []

for k, v in enumerate(gt_questions):
    q_id = v['questionId']
    USER_PROMPT = v['question']
    INPUT_FILE = os.path.join(DATASET_DIR, v['image_local_name'])

    # --- Run Inference ---
    logger.info(f"Processing Question: {USER_PROMPT} | Image: {INPUT_FILE}")
    generated_text, gen_time = model.generate(
        input_path=INPUT_FILE,
        user_prompt=USER_PROMPT,
        input_type=INPUT_TYPE,
        decoding_strategy=DECODING_STRATEGY,
        num_beams=NUM_BEAMS # Only relevant for beam search
        # Add other parameters like temperature, top_k, top_p for sampling if needed
    )

    # --- Print Results ---
    if generated_text:
        logger.info(f"Question ID: {q_id} | Generated Text: {generated_text}")
        print("\n--- Generated Text ---")
        print(generated_text)
        results.append({'questionId': q_id, 'answer': generated_text})
        print("--------------------")
        print(f"Generation time: {gen_time:.2f} seconds")
    else:
        print("Generation failed.")

output_filename = f"results_emb{Q_EMBED}_vis{Q_VISION}_dec{Q_DECODER}_{mode}.json"

# --- Save Results ---
with open(os.path.join(OUTPUT_DIR, output_filename), 'w') as f:
    json.dump(results, f, indent=4)
    print(f"Results saved to {os.path.join(OUTPUT_DIR, output_filename)}")

if model:
    model.close()