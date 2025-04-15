# example_usage.py
import os
import time
from LlaVaONNXRunner import VisionLanguageModelONNX

# --- Configure Paths and Settings ---
MODEL_DIR = os.path.join('.', 'models', 'onnx') # MODIFY
INPUT_FILES = ["RTX4090_PCB.png", "RTX4090_PCB.png"] # MODIFY (or a .mp4 video file)
INPUT_TYPE = 'image'   # MODIFY ('image' or 'video')
USER_PROMPT = "Describe the image." # MODIFY

# --- Quantization Settings ---
Q_EMBED = "q4f16"
Q_VISION = "fp16"
Q_DECODER = "q4f16"

# --- Decoding Settings ---
DECODING_STRATEGY = "beam"
NUM_BEAMS = 4

model = None
try:
    # --- Load Model ---
    print("Loading model...")
    model = VisionLanguageModelONNX(
        base_model_dir=MODEL_DIR,
        quant_type_embed=Q_EMBED,
        quant_type_vision=Q_VISION,
        quant_type_decoder=Q_DECODER
    )
    print("Model loaded.")

    # --- Run Inference ---
    print(f"Running inference on {INPUT_FILES} (Type: {INPUT_TYPE})...")
    generated_text, gen_time = model.generate(
        inputs=INPUT_FILES,
        user_prompt=USER_PROMPT,
        input_type=INPUT_TYPE,
        decoding_strategy=DECODING_STRATEGY,
        num_beams=NUM_BEAMS # Only relevant for beam search
        # Add other parameters like temperature, top_k, top_p for sampling if needed
    )

    # --- Print Results ---
    if generated_text:
        print("\n--- Generated Text ---")
        print(generated_text)
        print("--------------------")
        print(f"Generation time: {gen_time:.2f} seconds")
    else:
        print("Generation failed.")

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # --- Cleanup ---
    if model:
        model.close()
