# Vision Language Model (Llava) ONNX Inference & Model Loader

This repository provides Python scripts to run inference with a specific Vision Language Model (based on Llava/Qwen architectures) using ONNX Runtime. It includes functionality for handling both image and video inputs, configurable decoding strategies, flexible quantization settings (including separate settings per model component), and a benchmark script for the InfoVQA dataset.

## Features

* **ONNX Runtime Inference:** Leverages ONNX Runtime for efficient model execution on CPU or GPU.
* **Multi-Modal Input:** Supports both image and video inputs.
* **Flexible Decoding:** Implements Greedy Search, Beam Search, and Sampling decoding strategies.
* **Configurable Quantization:** Allows specifying quantization types (e.g., `fp16`, `q4f16`, `original`) for the vision encoder, text embedder, and LLM decoder models individually.
* **InfoVQA Benchmark:** Includes a script to run benchmarks on the InfoVQA dataset, testing multiple quantization combinations automatically.
* **Class-Based Runner:** Encapsulates model loading and inference logic in a reusable Python class (`VisionLanguageModelONNX`).

## Prerequisites

1.  **Python:** Python 3.8 or higher is recommended.
2.  **Git:** To clone this repository (if applicable).
3.  **ONNX Models:** You need the ONNX versions of the vision encoder, text embedding, and decoder models compatible with the scripts (e.g., derived from `llava-hf/llava-interleave-qwen-0.5b-hf`). The models should be organized in a directory structure where the scripts can find them based on quantization suffixes (e.g., `_fp16.onnx`, `_q4f16.onnx`, `.onnx`).

## Environment Setup

1.  **Clone the Repository (Optional):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # Activate the environment
    # Windows:
    .\venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Create a file named `requirements.txt` with the following content:

    ```txt
    numpy
    onnxruntime>=1.16 # Or onnxruntime-gpu if using NVIDIA GPU
    transformers>=4.30
    Pillow
    torch>=2.0     # Required by transformers and torchvision
    torchvision>=0.15 # Required for image transforms
    opencv-python-headless # For video processing (headless is usually sufficient)
    tqdm             # For benchmark progress bar
    ```

    Install the packages:
    ```bash
    pip install -r requirements.txt
    ```

    **Note on ONNX Runtime:**
    * For CPU inference, `onnxruntime` is sufficient.
    * For NVIDIA GPU inference, install `onnxruntime-gpu`. Ensure you have compatible NVIDIA drivers, CUDA Toolkit, and cuDNN installed. Check the [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for specific version compatibility.

4.  **Download ONNX Models:**
    * Obtain the ONNX model files for the desired model (e.g., `llava-hf/llava-interleave-qwen-0.5b-hf`). This might involve converting them yourself or downloading pre-converted files.
    * Organize them in a directory. The scripts expect files like:
        * `embed_tokens.onnx`, `embed_tokens_fp16.onnx`, `embed_tokens_q4f16.onnx`, etc.
        * `vision_encoder.onnx`, `vision_encoder_fp16.onnx`, etc.
        * `decoder_model_merged.onnx`, `decoder_model_merged_fp16.onnx`, `decoder_model_merged_q4f16.onnx`, etc.
    * You will need to provide the path to this main model directory when running the scripts.

5.  **Download InfoVQA Dataset (if benchmarking):**
    * Download the InfoVQA annotation JSON file (e.g., from the official dataset source).
    * Download the corresponding COCO 2014 image dataset.
    * Note the path to the annotation file and the directory containing the images. You might need to adjust the image filename formatting parameters (`--image_prefix`, `--image_suffix`, `--image_id_padding`) in the benchmark script if your filenames differ from the COCO standard.

## File Structure

* `LlaVaONNXRunner.py`: Contains the main `VisionLanguageModelONNX` class and helper functions for loading, preprocessing, inference, and decoding. This is the core library file.
* `run_infovqa_benchmark_multi_quant.py`: Script to run benchmarks on the InfoVQA dataset, iterating through multiple predefined quantization combinations.
* `requirements.txt`: Lists Python package dependencies.
* `README.md`: This file.

## Usage

### 1. Running Single Inference (Example)

You can use the `VisionLanguageModelONNX` class directly in your own Python scripts. Here's a basic example:

```python
# example_usage.py
import time
from LlaVaONNXRunner import VisionLanguageModelONNX

# --- Configure Paths and Settings ---
MODEL_DIR = "D:\\Code\\VLM\\llava-onevision-qwen2-0.5b-si-hf\\onnx\\" # MODIFY
INPUT_FILE = "cat.jpg" # MODIFY (or a .mp4 video file)
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
    print(f"Running inference on {INPUT_FILE} (Type: {INPUT_TYPE})...")
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
