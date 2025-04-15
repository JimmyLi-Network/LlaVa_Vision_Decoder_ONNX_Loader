# -*- coding: utf-8 -*-
"""
Provides a class for running inference with a Vision Language Model using ONNX Runtime.
Handles model loading, input preprocessing, feature extraction, generation, and resource cleanup.
Allows for separate quantization settings for different model components.
"""

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from PIL import Image
from torchvision import transforms
import sys
import torch # Needed for unsqueeze (via torchvision)
import heapq # Needed for Beam Search
import time # For timing
import os # For file path operations
import cv2 # For video processing (if needed)
import math # Needed for sqrt in attention and spatial reduction side calculation
import gc   # Import Garbage Collector module

# Default configurations (can be overridden during class instantiation or generation call)
DEFAULT_CONFIG = {
    # Model Arch
    'NUM_LAYERS': 24, 'NUM_HEADS': 2, 'HEAD_DIM': 64,
    'VOCAB_SIZE': 152000, 'TOKENIZER_NAME': "llava-hf/llava-interleave-qwen-0.5b-hf",
    # Generation
    'MAX_NEW_TOKENS': 256, 'EOS_TOKEN_ID': 151645,
    'IMAGE_TOKEN_ID': 151646, 'DECODING_STRATEGY': "beam", # Default strategy
    'NUM_BEAMS': 6, 'LENGTH_PENALTY': 1.0,
    'TEMPERATURE': 1.0, 'TOP_K': 50, 'TOP_P': 0.95,
    # Input
    'SYSTEM_PROMPT': "You are a helpful assistant.",
    # Model Files - *** USER MUST PROVIDE BASE_MODEL_DIR ***
    'QUANTIZATION_TYPE': "q4f16", # Default fallback quantization if specific types not given
    # Vision Processing
    'IMAGE_SIZE': 384, 'ORIGINAL_SPATIAL_TOKENS_SIDE': 27,
    'ENABLE_SPATIAL_REDUCTION': True, 'TARGET_SPATIAL_TOKENS_SIDE': 9,
    # Temporal Pooling
    'ENABLE_TEMPORAL_POOLING': True, 'TEMPORAL_POOLING_METHOD': 'attention',
    'TEMPORAL_ATTENTION_HEADS': 4, 'TEMPORAL_ATTENTION_TYPE': 'full',
    # Video Processing
    'NUM_FRAMES_TO_SAMPLE': 16,
    # Internal/ONNX
    'DTYPE': np.float32, 'ORT_PROVIDERS': ['CUDAExecutionProvider', 'CPUExecutionProvider']
}

# ==============================================================================
#                            Helper Functions
# ==============================================================================

def get_model_path(component_name, quant_type, base_dir):
    """
    Constructs and checks the model file path based on component name and quantization type.
    """
    base_names = {"embed": "embed_tokens", "vision": "vision_encoder", "decoder": "decoder_model_merged"}
    base_name = base_names.get(component_name)
    if not base_name: raise ValueError(f"Unknown model component: {component_name}")

    suffix_options = {
        "original": [".onnx"], "fp16": ["_fp16.onnx"],
        "int8": ["_int8.onnx", "_quantized.onnx"], "uint8": ["_uint8.onnx", "_quantized.onnx"],
        "quantized": ["_quantized.onnx", "_int8.onnx", "_uint8.onnx"],
        "q4": ["_q4.onnx"], "q4f16": ["_q4f16.onnx"], "bnb4": ["_bnb4.onnx"]
    }
    possible_suffixes = suffix_options.get(quant_type, suffix_options["original"])
    if quant_type not in suffix_options:
        print(f"Warning: Unknown quantization type '{quant_type}'. Trying 'original' (.onnx).")

    for suffix in possible_suffixes:
        path = os.path.join(base_dir, f"{base_name}{suffix}")
        if os.path.exists(path):
            print(f"Found model: {component_name} ({quant_type}) -> {os.path.basename(path)}")
            return path

    base_path = os.path.join(base_dir, f"{base_name}.onnx")
    if os.path.exists(base_path):
        print(f"Warning: Could not find '{quant_type}' model for {component_name}. Using base model: {os.path.basename(base_path)}")
        return base_path

    raise FileNotFoundError(f"Could not find '{quant_type}' or base '.onnx' file for component '{component_name}' in directory '{base_dir}'.")

def softmax(x, axis=-1):
    """ Compute softmax values for each set of scores in x. """
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def prepare_text_input(tokenizer, system_prompt, user_prompt, image_token_id):
    """ Prepares the tokenized text input based on prompts. """
    # print("\nPreparing text input...") # Optional print
    text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n<image>\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
    text_inputs = tokenizer(text, return_tensors="np", add_special_tokens=False)
    input_ids = text_inputs["input_ids"].astype(np.int64)
    # print(f"Tokenized Input IDs shape: {input_ids.shape}") # Optional print
    if image_token_id not in input_ids[0]:
         print(f"Warning: Image placeholder token ID {image_token_id} not found in the tokenized input.", file=sys.stderr)
    return input_ids

def get_vision_transform(image_size):
    """ Creates the standard image transformation pipeline. """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Adjust normalization if needed
    ])

def preprocess_images(images, transform):
    image_tensors = []
    for image in images:
        image_rgb = Image.open(image).convert("RGB")
        image_tensor = transform(image_rgb).unsqueeze(0).numpy()
        image_tensors.append(image_tensor)
    return image_tensors

def sample_video_frames(video_path, num_frames_to_sample):
    """ Samples frames uniformly from a video file. """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video file not found: {video_path}")
    sampled_frames_pil = []
    cap = None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): raise IOError(f"Cannot open video file: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0: raise ValueError("Video file contains no frames.")
        # print(f"  Video total frames: {total_frames}, Target sample: {num_frames_to_sample}") # Optional

        indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)
        indices = np.unique(indices) # Ensure unique indices if num_frames_to_sample > total_frames
        if len(indices) > num_frames_to_sample: indices = indices[:num_frames_to_sample] # Ensure exact number if possible

        # print(f"  Actual sampling frame indices (count {len(indices)}): {indices}") # Optional
        frame_indices_set = set(indices)
        processed_indices_list = []

        frame_count = 0
        processed_count = 0
        while processed_count < len(frame_indices_set):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Video ended before processing all requested frames (at frame {frame_count}). Processed {processed_count} frames.")
                break
            if frame_count in frame_indices_set:
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    sampled_frames_pil.append(pil_image)
                    processed_indices_list.append(frame_count)
                    processed_count += 1
                except Exception as frame_e:
                    print(f"Warning: Could not process frame {frame_count}: {frame_e}")
            frame_count += 1

        if not sampled_frames_pil: raise ValueError("Failed to extract any valid frames from the video.")
        # print(f"  Successfully sampled {len(sampled_frames_pil)} frames.") # Optional
        return sampled_frames_pil, processed_indices_list

    except (IOError, ValueError, cv2.error) as e: raise e
    except Exception as e: raise RuntimeError(f"An unexpected error occurred during video frame sampling: {e}")
    finally:
        if cap is not None and cap.isOpened(): cap.release()

def preprocess_video_frames(frame_list, transform):
    """ Applies transformations to a list of PIL video frames. """
    processed_tensors = []
    for i, frame_pil in enumerate(frame_list):
        try:
            tensor = transform(frame_pil).unsqueeze(0).numpy()
            processed_tensors.append(tensor)
        except Exception as e:
            print(f"Warning: Failed to transform frame {i}: {e}. Skipping this frame.")
    return processed_tensors

def run_vision_encoder(vision_sess, vision_metadata, image_tensor):
    """ Runs the vision encoder ONNX model on a single preprocessed image tensor. """
    input_name = vision_metadata["input_name"]
    output_name = vision_metadata["output_name"]
    input_dtype = vision_metadata["input_dtype"]
    try:
        vision_features = vision_sess.run([output_name], {input_name: image_tensor.astype(input_dtype)})[0]
        return vision_features
    except Exception as e:
        raise RuntimeError(f"Vision encoder inference failed: {e}")

def spatial_reduction_striding(feature_map, original_side, target_side):
    """ Reduces spatial tokens using striding. """
    if feature_map is None: return None
    if original_side <= 0 or target_side <= 0 or target_side >= original_side: return feature_map

    batch_size, num_tokens, hidden_dim = feature_map.shape
    expected_tokens = original_side * original_side
    if num_tokens != expected_tokens:
        print(f"Warning: Spatial reduction input token count ({num_tokens}) != expected ({expected_tokens}). Skipping.", file=sys.stderr)
        return feature_map

    stride = original_side // target_side
    if stride == 0: return feature_map # Avoid zero stride

    try:
        feature_reshaped = feature_map.reshape(batch_size, original_side, original_side, hidden_dim)
        reduced_feature = feature_reshaped[:, ::stride, ::stride, :]
        new_h, new_w = reduced_feature.shape[1:3]
        final_feature = reduced_feature.reshape(batch_size, new_h * new_w, hidden_dim)
        # print(f"  Spatial Reduction: {original_side}x{original_side} -> {new_h}x{new_w} (stride {stride})") # Optional
        return final_feature
    except Exception as e:
        print(f"Error during spatial_reduction_striding: {e}", file=sys.stderr)
        return feature_map # Return original on error

def temporal_pooling_attention(features_tensor, num_heads, attention_type, original_dtype):
    """ Performs temporal pooling using multi-head attention (NumPy implementation). """
    # (Implementation using NumPy as provided in previous answers)
    # NOTE: Uses random projection matrices for demonstration. Replace with learned weights if needed.
    num_frames, num_tokens, hidden_dim = features_tensor.shape
    # print(f"    Using Attention Pooling (Type: {attention_type}, Heads: {num_heads})") # Optional

    if hidden_dim % num_heads != 0:
        print(f"Error: hidden_dim ({hidden_dim}) not divisible by num_heads ({num_heads}). Falling back to mean.", file=sys.stderr)
        pooled_features = np.mean(features_tensor, axis=0, keepdims=True)
        return pooled_features.astype(original_dtype)

    head_dim = hidden_dim // num_heads
    features_tensor_f32 = features_tensor.astype(np.float32)

    # --- Using Random Weights for Demo ---
    # print("      NOTE: Using RANDOM QKVW projections for temporal pooling!") # Optional warning
    scale = 0.02
    wq = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
    wk = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
    wv = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
    wo = np.random.randn(hidden_dim, hidden_dim).astype(np.float32) * scale
    q, k, v = features_tensor_f32 @ wq, features_tensor_f32 @ wk, features_tensor_f32 @ wv

    # --- Multi-Head Attention Logic ---
    q = q.reshape(num_frames, num_tokens, num_heads, head_dim).transpose(1, 2, 0, 3) # N, H, F, D_h
    k = k.reshape(num_frames, num_tokens, num_heads, head_dim).transpose(1, 2, 0, 3) # N, H, F, D_h
    v = v.reshape(num_frames, num_tokens, num_heads, head_dim).transpose(1, 2, 0, 3) # N, H, F, D_h

    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(head_dim) # N, H, F, F

    # --- Optional Sparsity Mask ---
    if attention_type != 'full':
        mask_value = -np.inf
        mask = np.full_like(scores, mask_value)
        if attention_type.startswith('local_'):
             try:
                 local_k = int(attention_type.split('_')[-1])
                 for i in range(num_frames): mask[:, :, i, max(0, i-local_k):min(num_frames, i+local_k+1)] = 0
             except ValueError: print(f"Warn: Invalid local k: {attention_type}. Using full.", file=sys.stderr); mask = 0
        elif attention_type.startswith('strided_'):
             try:
                 stride_k = int(attention_type.split('_')[-1])
                 if stride_k <= 0: raise ValueError("Stride must be positive")
                 indices = np.arange(0, num_frames, stride_k)
                 mask[:, :, :, indices] = 0
             except ValueError: print(f"Warn: Invalid stride k: {attention_type}. Using full.", file=sys.stderr); mask = 0
        else: print(f"Warn: Unknown sparse type: {attention_type}. Using full.", file=sys.stderr); mask = 0
        scores = scores + mask
    # --- End Mask ---

    attn_weights = softmax(scores, axis=-1) # N, H, F, F
    attended_v = attn_weights @ v # N, H, F, D_h
    attended_v = attended_v.transpose(2, 0, 1, 3) # F, N, H, D_h
    concat_v = attended_v.reshape(num_frames, num_tokens, hidden_dim) # F, N, D
    output = concat_v @ wo # F, N, D
    final_pooled_features = np.mean(output, axis=0, keepdims=True) # 1, N, D
    # print(f"      Attention pooling output shape: {final_pooled_features.shape}") # Optional
    return final_pooled_features.astype(original_dtype)

def temporal_pooling(frame_features_list, method, num_heads, attention_type):
    """ Handles temporal pooling using the specified method. """
    if not frame_features_list: return None
    num_frames_in = len(frame_features_list)
    if num_frames_in == 1: return frame_features_list[0]
    # print(f"  Temporal Pooling: Aggregating {num_frames_in} frames (Method: {method})...") # Optional

    if method == 'mean':
        try:
            stacked_features = np.stack(frame_features_list, axis=0)
            if stacked_features.ndim == 4 and stacked_features.shape[1] == 1:
                 stacked_features = stacked_features.squeeze(axis=1) # F,N,D
            elif stacked_features.ndim != 3: raise ValueError("Incorrect shape")
            pooled_features = np.mean(stacked_features, axis=0, keepdims=True) # 1,N,D
            return pooled_features.astype(frame_features_list[0].dtype)
        except Exception as e:
            print(f"Error during mean pooling: {e}. Returning first frame.", file=sys.stderr)
            return frame_features_list[0]
    elif method == 'attention':
        try:
            features_tensor = np.stack(frame_features_list, axis=0)
            if features_tensor.ndim == 4 and features_tensor.shape[1] == 1:
                 features_tensor = features_tensor.squeeze(axis=1) # F,N,D
            elif features_tensor.ndim != 3: raise ValueError("Incorrect shape")
            original_dtype = features_tensor.dtype
            return temporal_pooling_attention(features_tensor, num_heads, attention_type, original_dtype)
        except Exception as e:
            print(f"Error during attention pooling setup: {e}. Falling back to mean.", file=sys.stderr)
            return temporal_pooling(frame_features_list, 'mean', num_heads, attention_type) # Fallback call
    else:
        print(f"Error: Unknown temporal pooling method '{method}'. Falling back to mean.", file=sys.stderr)
        return temporal_pooling(frame_features_list, 'mean', num_heads, attention_type) # Fallback call

def extract_vision_features(vision_sess, vision_metadata, visual_input, input_type, config):
    """ Extracts final vision features, handling image/video, spatial reduction, and temporal pooling. """
    # print(f"\nExtracting vision features (Input type: {input_type})...") # Optional
    final_vision_features = None
    original_spatial_side = config['ORIGINAL_SPATIAL_TOKENS_SIDE'] # Use the one determined at init
    target_spatial_side = config['TARGET_SPATIAL_TOKENS_SIDE']
    enable_spatial_reduction = config['ENABLE_SPATIAL_REDUCTION']

    if input_type == 'image':
        processed_features_list = []
        for i, image_tensor in enumerate(visual_input):
            raw_vision_features = run_vision_encoder(vision_sess, vision_metadata, image_tensor)
            processed_features = raw_vision_features
            if enable_spatial_reduction:
                processed_features = spatial_reduction_striding(raw_vision_features, original_spatial_side, target_spatial_side)
            processed_features_list.append(processed_features)
        
        # Use mean pooling for images
        # final_vision_features = np.mean(np.stack(processed_features_list, axis=0), axis=0)

        # Use temporal pooling for images
        final_vision_features = temporal_pooling(processed_features_list,
                                                    config['TEMPORAL_POOLING_METHOD'],
                                                    config['TEMPORAL_ATTENTION_HEADS'],
                                                    config['TEMPORAL_ATTENTION_TYPE'])


    elif input_type == 'video':
        if not isinstance(visual_input, list) or not visual_input:
             raise ValueError("Video input requires a non-empty list of preprocessed frame tensors.")
        raw_frame_features_list = []
        for i, frame_tensor in enumerate(visual_input):
            try: frame_features = run_vision_encoder(vision_sess, vision_metadata, frame_tensor); raw_frame_features_list.append(frame_features)
            except RuntimeError as e: print(f"Warning: Failed to encode frame {i}: {e}. Skipping.")

        if not raw_frame_features_list: raise ValueError("Failed to extract features from any video frame.")

        processed_frame_features = []
        if enable_spatial_reduction:
            for i, frame_feat in enumerate(raw_frame_features_list):
                reduced_feat = spatial_reduction_striding(frame_feat, original_spatial_side, target_spatial_side)
                if reduced_feat is None: raise ValueError(f"Spatial reduction failed for frame {i}.")
                processed_frame_features.append(reduced_feat)
        else:
            processed_frame_features = raw_frame_features_list

        if config['ENABLE_TEMPORAL_POOLING'] and len(processed_frame_features) > 0:
            final_vision_features = temporal_pooling(processed_frame_features,
                                                     config['TEMPORAL_POOLING_METHOD'],
                                                     config['TEMPORAL_ATTENTION_HEADS'],
                                                     config['TEMPORAL_ATTENTION_TYPE'])
            if final_vision_features is None: raise ValueError("Temporal pooling failed.")
        elif len(processed_frame_features) > 0:
            final_vision_features = processed_frame_features[0] # Use first frame if no pooling
        else: raise ValueError("No processable frame features available for video.")

    else: raise ValueError(f"Invalid input_type '{input_type}'.")

    if final_vision_features is None: raise ValueError("Failed to generate final vision features.")
    # print(f"Final vision features shape: {final_vision_features.shape}") # Optional
    return final_vision_features

def fuse_embeddings(embed_sess, text_input_ids, vision_features, image_token_id, embed_hidden_dim, target_dtype):
    """ Fuses text embeddings with vision features by replacing the image token embedding. """
    # print("\nFusing text and vision embeddings...") # Optional
    image_token_indices = np.where(text_input_ids[0] == image_token_id)[0]
    embed_input_name = embed_sess.get_inputs()[0].name
    embed_output_name = embed_sess.get_outputs()[0].name

    if len(image_token_indices) == 0:
        print(f"Warning: Image placeholder token (ID: {image_token_id}) not found. Using text embeddings only.", file=sys.stderr)
        try: combined_embeds = embed_sess.run([embed_output_name], {embed_input_name: text_input_ids})[0]
        except Exception as e: raise RuntimeError(f"Embedding lookup failed for text-only input: {e}")
        return combined_embeds.astype(target_dtype)
    elif len(image_token_indices) > 1:
        raise ValueError(f"Found multiple image placeholder tokens ({len(image_token_indices)}). Only one is supported.")
    else:
        image_token_index = image_token_indices[0]
        input_ids_pre = text_input_ids[:, :image_token_index]
        input_ids_post = text_input_ids[:, image_token_index + 1:]
        try:
            embed_pre = embed_sess.run([embed_output_name], {embed_input_name: input_ids_pre})[0] if input_ids_pre.shape[1] > 0 else np.zeros((1, 0, embed_hidden_dim), dtype=target_dtype)
            embed_post = embed_sess.run([embed_output_name], {embed_input_name: input_ids_post})[0] if input_ids_post.shape[1] > 0 else np.zeros((1, 0, embed_hidden_dim), dtype=target_dtype)
        except Exception as e: raise RuntimeError(f"Embedding lookup failed for text parts: {e}")

        vision_hidden_dim = vision_features.shape[-1]
        if vision_hidden_dim != embed_hidden_dim:
             raise ValueError(f"Mismatch: Vision feature dim ({vision_hidden_dim}) != Text embed dim ({embed_hidden_dim}).")
        # print(f"Inserting {vision_features.shape[1]} vision tokens.") # Optional

        combined_embeds = np.concatenate([embed_pre.astype(target_dtype), vision_features.astype(target_dtype), embed_post.astype(target_dtype)], axis=1)
        # print(f"Fused embeddings shape: {combined_embeds.shape}") # Optional
        return combined_embeds

# ==============================================================================
#                      KV Cache & Decoder Input Functions
# ==============================================================================
PAST_KV_INPUT_PREFIX = "past_key_values"
PRESENT_KV_OUTPUT_PREFIX = "present" # Check your merged model's output names

def build_empty_past_kv(num_layers, batch_size, num_heads, past_seq_len, head_dim, dtype):
    """ Initializes an empty past_kv structure (dictionary) for the decoder. """
    kv = {}
    for layer in range(num_layers):
        input_name_k = f"{PAST_KV_INPUT_PREFIX}.{layer}.key"
        input_name_v = f"{PAST_KV_INPUT_PREFIX}.{layer}.value"
        k = np.zeros((batch_size, num_heads, past_seq_len, head_dim), dtype=dtype)
        v = np.zeros((batch_size, num_heads, past_seq_len, head_dim), dtype=dtype)
        kv[input_name_k], kv[input_name_v] = k, v
    return kv

def cast_decoder_inputs(session, inputs):
    """ Casts input data to the types expected by the decoder ONNX model. """
    casted = {}
    input_metadata = {inp.name: inp.type for inp in session.get_inputs()}
    model_accepts = set(input_metadata.keys())
    for name, val in inputs.items():
        if name not in model_accepts: continue
        if not isinstance(val, np.ndarray): casted[name] = val; continue
        expected_type_str = input_metadata[name]
        current_type_str = str(val.dtype)
        target_np_type = None
        if "float16" in expected_type_str: target_np_type = np.float16
        elif "float32" in expected_type_str: target_np_type = np.float32
        elif "float" in expected_type_str: target_np_type = np.float32
        elif "int64" in expected_type_str: target_np_type = np.int64
        elif "int32" in expected_type_str: target_np_type = np.int32
        elif "bool" in expected_type_str: target_np_type = bool
        if target_np_type and str(np.dtype(target_np_type)) != current_type_str:
            casted[name] = val.astype(target_np_type)
        else: casted[name] = val
    return casted

def update_kv_cache(past_kv, updated_kv_list, num_layers):
    """ Updates the past_kv dictionary using the list of K/V outputs from the decoder. """
    new_past_kv = {}
    kv_idx = 0
    expected_kv_outputs = num_layers * 2
    if len(updated_kv_list) < expected_kv_outputs:
         print(f"Error: Insufficient KV outputs. Expected {expected_kv_outputs}, got {len(updated_kv_list)}.", file=sys.stderr)
         return past_kv # Return old cache to avoid downstream errors
    for layer in range(num_layers):
        key_name = f"{PAST_KV_INPUT_PREFIX}.{layer}.key"
        value_name = f"{PAST_KV_INPUT_PREFIX}.{layer}.value"
        new_past_kv[key_name] = updated_kv_list[kv_idx]
        new_past_kv[value_name] = updated_kv_list[kv_idx + 1]
        kv_idx += 2
    return new_past_kv

def prepare_decoder_inputs(step, past_len, current_embed, past_kv, batch_size, initial_seq_len=None):
    """ Prepares the input dictionary for the decoder model for a single generation step. """
    if step == 0: # Initial prompt processing
        if initial_seq_len is None: raise ValueError("initial_seq_len needed for step 0")
        current_seq_len = initial_seq_len
        position_ids = np.arange(past_len, past_len + current_seq_len, dtype=np.int64).reshape(batch_size, current_seq_len)
        attention_mask = np.ones((batch_size, past_len + current_seq_len), dtype=np.int64)
        inputs_embeds_step = current_embed
    else: # Subsequent token generation
        current_seq_len = 1
        current_position = past_len
        # --- Start of Change ---
        # 使用 np.full 直接创建所需形状和值的数组，修复了 reshape 错误
        position_ids = np.full((batch_size, 1), current_position, dtype=np.int64)
        # --- End of Change ---
        # Causal mask implies attending to past_len+1 tokens (the new query attends to all previous keys+itself implicitly)
        attention_mask = np.ones((batch_size, 1 + past_len), dtype=np.int64)
        inputs_embeds_step = current_embed # Shape [B, 1, D]

    decoder_inputs = {"inputs_embeds": inputs_embeds_step, "attention_mask": attention_mask, "position_ids": position_ids}
    decoder_inputs.update(past_kv) # 添加 past_kv 内容
    return decoder_inputs

# ==============================================================================
#                      Decoding Strategy Functions
# ==============================================================================

# --- 7.1 Greedy Search ---
def run_decoder_greedy(decoder_sess, embed_sess, initial_embeds, initial_seq_len, config):
    """ Generates token IDs using greedy search. """
    # print("Running Greedy Search...") # Optional
    batch_size = initial_embeds.shape[0]
    current_embed = initial_embeds
    past_kv = build_empty_past_kv(config['NUM_LAYERS'], batch_size, config['NUM_HEADS'], 0, config['HEAD_DIM'], config['DTYPE'])
    generated_ids = []
    logits_output_name = decoder_sess.get_outputs()[0].name
    kv_output_names = [out.name for out in decoder_sess.get_outputs() if out.name.startswith(PRESENT_KV_OUTPUT_PREFIX)]
    decoder_output_names = [logits_output_name] + kv_output_names
    embed_input_name = embed_sess.get_inputs()[0].name
    embed_output_name = embed_sess.get_outputs()[0].name
    past_len = 0

    for step in range(config['MAX_NEW_TOKENS']):
        decoder_inputs = prepare_decoder_inputs(step, past_len, current_embed, past_kv, batch_size, initial_seq_len if step == 0 else None)
        decoder_inputs_casted = cast_decoder_inputs(decoder_sess, decoder_inputs)
        try: outputs = decoder_sess.run(decoder_output_names, decoder_inputs_casted)
        except Exception as e: print(f"\nError Greedy Step {step+1}: {e}", file=sys.stderr); break
        logits, updated_kv_list = outputs[0], outputs[1:]
        past_kv = update_kv_cache(past_kv, updated_kv_list, config['NUM_LAYERS'])
        try: past_len = past_kv[f"{PAST_KV_INPUT_PREFIX}.0.key"].shape[2]
        except Exception as e: print(f"Error getting past_len step {step+1}: {e}", file=sys.stderr); break

        next_token_logits = logits[0, -1, :config['VOCAB_SIZE']]
        next_token_id = int(np.argmax(next_token_logits))
        generated_ids.append(next_token_id)
        if next_token_id == config['EOS_TOKEN_ID']: break # EOS detected

        next_token_id_array = np.array([[next_token_id]], dtype=np.int64)
        try: current_embed = embed_sess.run([embed_output_name], {embed_input_name: next_token_id_array})[0].astype(config['DTYPE'])
        except Exception as e: print(f"\nError getting next embed step {step+1}: {e}", file=sys.stderr); break
    return generated_ids

# --- 7.2 Beam Search ---
class BeamHypothesis:
    """Helper class for beam search."""
    def __init__(self, sequence, score, past_kv): self.sequence, self.score, self.past_kv = sequence, score, past_kv
    def __len__(self): return len(self.sequence)
    def __lt__(self, other): return self.score < other.score # For min-heap if score is neg log prob

def run_beam_search(decoder_sess, embed_sess, initial_embeds, initial_seq_len, config):
    """ Generates token IDs using beam search. """
    num_beams, length_penalty = config['NUM_BEAMS'], config['LENGTH_PENALTY']
    # print(f"Running Beam Search (beams={num_beams}, penalty={length_penalty})...") # Optional
    batch_size = initial_embeds.shape[0]
    if batch_size != 1: raise ValueError("Beam search only implemented for batch_size=1")

    logits_output_name = decoder_sess.get_outputs()[0].name
    kv_output_names = [out.name for out in decoder_sess.get_outputs() if out.name.startswith(PRESENT_KV_OUTPUT_PREFIX)]
    decoder_output_names = [logits_output_name] + kv_output_names
    embed_input_name = embed_sess.get_inputs()[0].name
    embed_output_name = embed_sess.get_outputs()[0].name

    # Initial Run
    past_kv_init = build_empty_past_kv(config['NUM_LAYERS'], batch_size, config['NUM_HEADS'], 0, config['HEAD_DIM'], config['DTYPE'])
    decoder_inputs_init = prepare_decoder_inputs(0, 0, initial_embeds, past_kv_init, batch_size, initial_seq_len)
    decoder_inputs_init_casted = cast_decoder_inputs(decoder_sess, decoder_inputs_init)
    try: outputs_init = decoder_sess.run(decoder_output_names, decoder_inputs_init_casted)
    except Exception as e: print(f"Error Beam Search initial run: {e}", file=sys.stderr); return []
    logits_init, kv_after_init_list = outputs_init[0], outputs_init[1:]
    past_kv_after_init = update_kv_cache({}, kv_after_init_list, config['NUM_LAYERS'])
    try: current_past_len = past_kv_after_init[f"{PAST_KV_INPUT_PREFIX}.0.key"].shape[2]
    except Exception as e: print(f"Error getting initial past_len: {e}", file=sys.stderr); return []

    # Initialize Beams
    next_token_logits = logits_init[0, -1, :config['VOCAB_SIZE']].astype(np.float32)
    log_probs = next_token_logits - np.max(next_token_logits); log_probs -= np.log(np.sum(np.exp(log_probs))) # Log softmax
    top_k_indices = np.argsort(log_probs)[-num_beams:][::-1]; top_k_scores = log_probs[top_k_indices]
    beams, completed_hypotheses = [], []
    for i in range(len(top_k_indices)):
        token_id, score = top_k_indices[i], top_k_scores[i]
        if token_id != config['EOS_TOKEN_ID']: # Avoid starting with EOS
            beams.append(BeamHypothesis(sequence=[token_id], score=score, past_kv=past_kv_after_init))
    if not beams: return [] # No valid starting beams

    # Beam Search Loop
    for step in range(config['MAX_NEW_TOKENS']):
        if not beams: break
        current_beam_count = len(beams)
        batch_last_token_ids = np.array([b.sequence[-1] for b in beams], dtype=np.int64).reshape(current_beam_count, 1)
        stacked_past_kv = {}
        try: # Stack KVs
            first_beam_keys = beams[0].past_kv.keys()
            for key in first_beam_keys: stacked_past_kv[key] = np.concatenate([b.past_kv[key] for b in beams], axis=0)
        except Exception as e: print(f"Error stacking KVs step {step+1}: {e}", file=sys.stderr); break

        try: current_embed_batch = embed_sess.run([embed_output_name], {embed_input_name: batch_last_token_ids})[0].astype(config['DTYPE'])
        except Exception as e: print(f"Error getting beam embeds step {step+1}: {e}", file=sys.stderr); break

        decoder_inputs_batch = prepare_decoder_inputs(step + 1, current_past_len, current_embed_batch, stacked_past_kv, current_beam_count)
        decoder_inputs_batch_casted = cast_decoder_inputs(decoder_sess, decoder_inputs_batch)
        try: outputs_batch = decoder_sess.run(decoder_output_names, decoder_inputs_batch_casted)
        except Exception as e: print(f"Error beam decoder step {step+1}: {e}", file=sys.stderr); break
        logits_batch, updated_kv_batch_list = outputs_batch[0], outputs_batch[1:]
        try: current_past_len = updated_kv_batch_list[0].shape[2] # Update past length
        except Exception as e: print(f"Error getting new past_len step {step+1}: {e}", file=sys.stderr); break

        next_beam_candidates = []
        for beam_idx, beam in enumerate(beams):
            current_logits = logits_batch[beam_idx, -1, :config['VOCAB_SIZE']].astype(np.float32)
            log_probs = current_logits - np.max(current_logits); log_probs -= np.log(np.sum(np.exp(log_probs)))
            top_k_indices = np.argsort(log_probs)[-num_beams:][::-1]; top_k_scores = log_probs[top_k_indices]
            for i in range(len(top_k_indices)):
                token_id, score = top_k_indices[i], top_k_scores[i]
                new_sequence, new_score = beam.sequence + [token_id], beam.score + score
                new_kv = {} # Extract KV slice for this beam/candidate
                kv_output_idx = 0
                try:
                    for layer in range(config['NUM_LAYERS']):
                        k_name=f"{PAST_KV_INPUT_PREFIX}.{layer}.key"; v_name=f"{PAST_KV_INPUT_PREFIX}.{layer}.value"
                        new_kv[k_name] = updated_kv_batch_list[kv_output_idx][beam_idx:beam_idx+1]
                        new_kv[v_name] = updated_kv_batch_list[kv_output_idx + 1][beam_idx:beam_idx+1]; kv_output_idx += 2
                except Exception as e: print(f"Error slicing KV step {step+1} beam {beam_idx}: {e}", file=sys.stderr); continue # Skip candidate
                candidate = BeamHypothesis(new_sequence, new_score, new_kv)
                if token_id == config['EOS_TOKEN_ID']:
                    final_score = new_score / (max(len(new_sequence), 1) ** length_penalty)
                    completed_hypotheses.append((final_score, new_sequence))
                else: next_beam_candidates.append(candidate)

        next_beam_candidates.sort(key=lambda x: x.score, reverse=True) # Best scores first
        beams = next_beam_candidates[:num_beams] # Prune to top N beams
        if len(completed_hypotheses) > num_beams: # Keep N best completed
            completed_hypotheses.sort(key=lambda x: x[0], reverse=True)
            completed_hypotheses = completed_hypotheses[:num_beams]

    # Final Selection
    if completed_hypotheses:
        completed_hypotheses.sort(key=lambda x: x[0], reverse=True)
        return completed_hypotheses[0][1]
    elif beams: # Max length reached, return best beam
        beams.sort(key=lambda x: x.score / (max(len(x.sequence), 1) ** length_penalty), reverse=True)
        print("Warning: Beam search reached max length.", file=sys.stderr)
        return beams[0].sequence
    else: return [] # Should not happen normally

# --- 7.3 Sampling ---
def _apply_sampling_filters(logits, temperature, top_k, top_p):
    """Applies temperature, top-k, and top-p filtering to logits."""
    if temperature > 0 and temperature != 1.0: logits = logits / temperature
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = np.argsort(logits)[:-top_k]
        logits[indices_to_remove] = -np.inf
    if 0.0 < top_p < 1.0:
        sorted_indices = np.argsort(logits)[::-1]; sorted_logits = logits[sorted_indices]
        with np.errstate(invalid='ignore'): probs = softmax(sorted_logits, axis=-1)
        cumulative_probs = np.cumsum(probs); indices_to_remove = cumulative_probs > top_p
        if np.any(indices_to_remove): indices_to_remove[1:] = indices_to_remove[:-1].copy(); indices_to_remove[0] = False
        indices_to_remove_original = sorted_indices[indices_to_remove]
        logits[indices_to_remove_original] = -np.inf
    return logits

def run_sampling(decoder_sess, embed_sess, initial_embeds, initial_seq_len, config):
    """ Generates token IDs using sampling (temperature, top-k, top-p). """
    temp, top_k, top_p = config['TEMPERATURE'], config['TOP_K'], config['TOP_P']
    # print(f"Running Sampling (temp={temp}, k={top_k}, p={top_p})...") # Optional
    if temp <= 0: temp = 1.0
    if top_k < 0: top_k = 0
    if not (0.0 <= top_p <= 1.0): top_p = 0.0

    batch_size = initial_embeds.shape[0]
    current_embed = initial_embeds
    past_kv = build_empty_past_kv(config['NUM_LAYERS'], batch_size, config['NUM_HEADS'], 0, config['HEAD_DIM'], config['DTYPE'])
    generated_ids = []
    logits_output_name = decoder_sess.get_outputs()[0].name
    kv_output_names = [out.name for out in decoder_sess.get_outputs() if out.name.startswith(PRESENT_KV_OUTPUT_PREFIX)]
    decoder_output_names = [logits_output_name] + kv_output_names
    embed_input_name = embed_sess.get_inputs()[0].name
    embed_output_name = embed_sess.get_outputs()[0].name
    past_len = 0

    for step in range(config['MAX_NEW_TOKENS']):
        decoder_inputs = prepare_decoder_inputs(step, past_len, current_embed, past_kv, batch_size, initial_seq_len if step == 0 else None)
        decoder_inputs_casted = cast_decoder_inputs(decoder_sess, decoder_inputs)
        try: outputs = decoder_sess.run(decoder_output_names, decoder_inputs_casted)
        except Exception as e: print(f"Error Sampling Step {step+1}: {e}", file=sys.stderr); break
        logits, updated_kv_list = outputs[0], outputs[1:]
        past_kv = update_kv_cache(past_kv, updated_kv_list, config['NUM_LAYERS'])
        try: past_len = past_kv[f"{PAST_KV_INPUT_PREFIX}.0.key"].shape[2]
        except Exception as e: print(f"Error getting past_len step {step+1}: {e}", file=sys.stderr); break

        next_token_logits = logits[0, -1, :config['VOCAB_SIZE']].astype(np.float32)
        filtered_logits = _apply_sampling_filters(next_token_logits, temp, top_k, top_p)
        with np.errstate(divide='ignore', invalid='ignore'): probs = softmax(filtered_logits, axis=-1)

        probs_sum = np.sum(probs) # Check for invalid distribution
        if not np.all(np.isfinite(probs)) or probs_sum < 1e-8:
             print(f"Warning: Invalid probability distribution step {step+1}. Sampling uniformly.", file=sys.stderr)
             valid_indices = np.where(np.isfinite(filtered_logits))[0]
             next_token_id = np.random.choice(valid_indices) if len(valid_indices) > 0 else np.random.choice(config['VOCAB_SIZE'])
        else:
             probs /= probs_sum # Renormalize
             next_token_id = np.random.choice(config['VOCAB_SIZE'], p=probs)

        generated_ids.append(next_token_id)
        if next_token_id == config['EOS_TOKEN_ID']: break # EOS detected

        next_token_id_array = np.array([[next_token_id]], dtype=np.int64)
        try: current_embed = embed_sess.run([embed_output_name], {embed_input_name: next_token_id_array})[0].astype(config['DTYPE'])
        except Exception as e: print(f"Error getting next embed step {step+1}: {e}", file=sys.stderr); break
    return generated_ids

# ==============================================================================
#                            Inference Class Definition
# ==============================================================================

class VisionLanguageModelONNX:
    """
    Manages loading and running inference for the Vision Language Model using ONNX Runtime.
    Allows specifying separate quantization types for embed, vision, and decoder models.
    """
    def __init__(self,
                 base_model_dir: str,
                 quant_type_embed: str = DEFAULT_CONFIG['QUANTIZATION_TYPE'],
                 quant_type_vision: str = DEFAULT_CONFIG['QUANTIZATION_TYPE'],
                 quant_type_decoder: str = DEFAULT_CONFIG['QUANTIZATION_TYPE'],
                 tokenizer_name: str = DEFAULT_CONFIG['TOKENIZER_NAME'],
                 ort_providers: list = None,
                 config_overrides: dict = None):
        """ Initializes the model by loading the tokenizer and ONNX sessions. (Implementation as in previous response) """
        print("Initializing VisionLanguageModelONNX...")
        self.base_model_dir = base_model_dir
        self.quant_type_embed = quant_type_embed
        self.quant_type_vision = quant_type_vision
        self.quant_type_decoder = quant_type_decoder
        self.tokenizer_name = tokenizer_name
        self.ort_providers = ort_providers if ort_providers is not None else DEFAULT_CONFIG['ORT_PROVIDERS']
        self.config = DEFAULT_CONFIG.copy()
        self.config['QUANTIZATION_TYPE_EMBED'] = quant_type_embed
        self.config['QUANTIZATION_TYPE_VISION'] = quant_type_vision
        self.config['QUANTIZATION_TYPE_DECODER'] = quant_type_decoder
        if config_overrides: self._apply_config_overrides(config_overrides) # Internal helper

        # Init resource holders
        self.tokenizer = None
        self.embed_sess = None
        self.vision_sess = None
        self.decoder_sess = None
        self.vision_metadata = None
        self.vision_hidden_dim = None
        self.embed_hidden_dim = None

        try:
            self.tokenizer, effective_eos_id = self._load_tokenizer_internal(self.tokenizer_name, self.config['EOS_TOKEN_ID'], self.config['IMAGE_TOKEN_ID'])
            self.config['EOS_TOKEN_ID'] = effective_eos_id
            (self.embed_sess, self.vision_sess, self.decoder_sess,
             self.vision_metadata, orig_spatial_side, self.vision_hidden_dim) = self._load_onnx_models_internal(
                 self.base_model_dir, self.quant_type_embed, self.quant_type_vision, self.quant_type_decoder,
                 self.ort_providers, self.config['ORIGINAL_SPATIAL_TOKENS_SIDE']
             )
            self.config['ORIGINAL_SPATIAL_TOKENS_SIDE'] = orig_spatial_side
            self.embed_hidden_dim = self._determine_embedding_dim(self.embed_sess, self.vision_hidden_dim)
            self.vision_transform = get_vision_transform(self.config['IMAGE_SIZE'])
            print(f"Using embedding dimension: {self.embed_hidden_dim}")
            print("VisionLanguageModelONNX initialized successfully.")
        except Exception as e:
            print(f"FATAL: Failed during initialization: {e}", file=sys.stderr)
            self.close()
            raise

    def _apply_config_overrides(self, overrides):
        """ Safely apply overrides and cast types. """
        print(f"Applying configuration overrides: {overrides}")
        self.config.update(overrides)
        for key in ['NUM_LAYERS', 'NUM_HEADS', 'HEAD_DIM', 'VOCAB_SIZE', 'IMAGE_SIZE', 'ORIGINAL_SPATIAL_TOKENS_SIDE', 'TARGET_SPATIAL_TOKENS_SIDE', 'TEMPORAL_ATTENTION_HEADS', 'NUM_FRAMES_TO_SAMPLE', 'MAX_NEW_TOKENS', 'NUM_BEAMS', 'TOP_K']:
            if key in self.config: self.config[key] = int(self.config[key])
        for key in ['LENGTH_PENALTY', 'TEMPERATURE', 'TOP_P']:
            if key in self.config: self.config[key] = float(self.config[key])
        for key in ['ENABLE_SPATIAL_REDUCTION', 'ENABLE_TEMPORAL_POOLING']:
            if key in self.config: self.config[key] = bool(self.config[key])


    def _load_tokenizer_internal(self, tokenizer_name, config_eos_id, config_image_id):
        """ Internal tokenizer loading logic. """
        # (Implementation from previous code)
        print(f"Loading Tokenizer: {tokenizer_name}...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # print(f"Setting pad_token to eos_token: '{tokenizer.pad_token}' (ID: {tokenizer.eos_token_id})") # Optional
        effective_eos_id = config_eos_id
        if tokenizer.eos_token_id != config_eos_id:
            print(f"Warning: Tokenizer EOS ID ({tokenizer.eos_token_id}) != Configured EOS ID ({config_eos_id}). Using Tokenizer ID.", file=sys.stderr)
            effective_eos_id = tokenizer.eos_token_id
        print(f"Using EOS token ID: {effective_eos_id}")
        try: print(f"Visual placeholder Token ID: {config_image_id} (Decodes to: '{tokenizer.decode([config_image_id])}')")
        except Exception: pass
        return tokenizer, effective_eos_id

    def _load_onnx_models_internal(self, base_dir, quant_type_embed, quant_type_vision, quant_type_decoder, providers, config_orig_spatial_side):
         """ Internal ONNX model loading logic using separate quant types. """
         # (Implementation from previous code - uses separate quant types for get_model_path)
         print(f"\n--- Looking for model files ---")
         print(f"  Embedding Quant Type: '{quant_type_embed}'")
         print(f"  Vision Quant Type:    '{quant_type_vision}'")
         print(f"  Decoder Quant Type:   '{quant_type_decoder}'")
         embed_model_path = get_model_path("embed", quant_type_embed, base_dir)
         vision_model_path = get_model_path("vision", quant_type_vision, base_dir)
         decoder_model_path = get_model_path("decoder", quant_type_decoder, base_dir)
         print("----------------------------------------------------")
         print("\nLoading ONNX models...")
         sess_options = ort.SessionOptions(); sess_options.intra_op_num_threads = 1 # Optimization for some scenarios
         sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL; sess_options.enable_mem_pattern = True
         print(f"Attempting to use Providers: {providers}")
         embed_sess = ort.InferenceSession(embed_model_path, sess_options=sess_options, providers=providers)
         vision_sess = ort.InferenceSession(vision_model_path, sess_options=sess_options, providers=providers)
         decoder_sess = ort.InferenceSession(decoder_model_path, sess_options=sess_options, providers=providers)
         print(f"\nActual Providers Used:\n  Embed: {embed_sess.get_providers()}\n  Vision: {vision_sess.get_providers()}\n  Decoder: {decoder_sess.get_providers()}")
         # Get vision details
         vision_input_name = vision_sess.get_inputs()[0].name; vision_output_name = vision_sess.get_outputs()[0].name
         vision_output_shape = vision_sess.get_outputs()[0].shape; vision_hidden_dim_model = None
         original_spatial_tokens_side_model = config_orig_spatial_side
         try:
             num_vision_tokens_model = vision_output_shape[1]; vision_hidden_dim_model = vision_output_shape[2]
             if isinstance(num_vision_tokens_model, int) and isinstance(vision_hidden_dim_model, int):
                 inferred_side = int(math.sqrt(num_vision_tokens_model))
                 if inferred_side * inferred_side == num_vision_tokens_model:
                     # print(f"Inferred from vision model: Tokens={num_vision_tokens_model} ({inferred_side}x{inferred_side}), Dim={vision_hidden_dim_model}") # Optional
                     if config_orig_spatial_side != inferred_side: print(f"Warning: Configured spatial side ({config_orig_spatial_side}) differs from inferred ({inferred_side}). Using inferred.")
                     original_spatial_tokens_side_model = inferred_side
                 # else: print(f"Warn: Cannot infer square layout from {num_vision_tokens_model} tokens.") # Optional
         except Exception: pass # Ignore parsing errors
         vision_input_metadata = {inp.name: inp.type for inp in vision_sess.get_inputs()}
         vision_input_type_str = vision_input_metadata.get(vision_input_name, "float32")
         vision_input_dtype = np.float16 if "float16" in vision_input_type_str else np.float32
         vision_metadata = {"input_name": vision_input_name,"output_name": vision_output_name,"output_shape": vision_output_shape,"input_dtype": vision_input_dtype,"hidden_dim": vision_hidden_dim_model }
         print("ONNX models loaded successfully.")
         return embed_sess, vision_sess, decoder_sess, vision_metadata, original_spatial_tokens_side_model, vision_hidden_dim_model

    def _determine_embedding_dim(self, embed_sess, vision_hidden_dim):
        """ Determine embedding dimension, falling back if needed. """
        # (Implementation from previous code)
        try:
            embed_dim = embed_sess.get_outputs()[0].shape[-1]
            if isinstance(embed_dim, int): return embed_dim
        except Exception: pass
        print("Warning: Could not infer embedding dim from Embed model.")
        if isinstance(vision_hidden_dim, int):
            print(f"Using embedding dimension inferred from Vision model: {vision_hidden_dim}")
            return vision_hidden_dim
        fallback_dim = 896 # Adjust needed
        print(f"Warning: Could not infer embedding dimension. Using fallback: {fallback_dim}")
        return fallback_dim

    def generate(self,
                 inputs: str,
                 user_prompt: str,
                 input_type: str = 'image',
                 system_prompt: str = None,
                 decoding_strategy: str = None,
                 max_new_tokens: int = None,
                 num_beams: int = None,
                 length_penalty: float = None,
                 temperature: float = None,
                 top_k: int = None,
                 top_p: float = None,
                 enable_spatial_reduction: bool = None,
                 target_spatial_tokens_side: int = None,
                 enable_temporal_pooling: bool = None,
                 temporal_pooling_method: str = None,
                 temporal_attention_heads: int = None,
                 temporal_attention_type: str = None,
                 num_frames_to_sample: int = None
                 ):
        """ Generates text describing the input image or video. (Implementation as in previous response) """
        # (Implementation from previous code - uses self.config and self.sessions)
        if not all([self.tokenizer, self.embed_sess, self.vision_sess, self.decoder_sess]):
            raise RuntimeError("Model resources are not loaded. Initialize the class correctly.")

        start_time = time.time()
        run_config = self.config.copy() # Effective config for this run
        run_config['INPUT_TYPE'] = input_type
        if input_type == 'image': run_config['IMAGES_PATH'] = inputs; run_config['USER_PROMPT_IMAGE'] = user_prompt
        elif input_type == 'video': run_config['VIDEO_PATH'] = inputs; run_config['USER_PROMPT_VIDEO'] = user_prompt
        else: print(f"Error: Invalid input_type '{input_type}'.", file=sys.stderr); return None, 0

        # Apply overrides
        if system_prompt is not None: 
            run_config['SYSTEM_PROMPT'] = system_prompt

        if decoding_strategy is not None: 
            run_config['DECODING_STRATEGY'] = decoding_strategy

        if max_new_tokens is not None: 
            run_config['MAX_NEW_TOKENS'] = max_new_tokens

        if num_beams is not None: 
            run_config['NUM_BEAMS'] = num_beams

        if length_penalty is not None: 
            run_config['LENGTH_PENALTY'] = length_penalty

        if temperature is not None: 
            run_config['TEMPERATURE'] = temperature

        if top_k is not None: run_config['TOP_K'] = top_k
        if top_p is not None: run_config['TOP_P'] = top_p
        if enable_spatial_reduction is not None: run_config['ENABLE_SPATIAL_REDUCTION'] = enable_spatial_reduction
        if target_spatial_tokens_side is not None: run_config['TARGET_SPATIAL_TOKENS_SIDE'] = target_spatial_tokens_side
        if enable_temporal_pooling is not None: run_config['ENABLE_TEMPORAL_POOLING'] = enable_temporal_pooling
        if temporal_pooling_method is not None: run_config['TEMPORAL_POOLING_METHOD'] = temporal_pooling_method
        if temporal_attention_heads is not None: run_config['TEMPORAL_ATTENTION_HEADS'] = temporal_attention_heads
        if temporal_attention_type is not None: run_config['TEMPORAL_ATTENTION_TYPE'] = temporal_attention_type
        if num_frames_to_sample is not None: run_config['NUM_FRAMES_TO_SAMPLE'] = num_frames_to_sample

        # --- Steps 1, 2, 3: Prepare Input, Extract Features, Fuse Embeddings ---
        try:
            current_user_prompt = run_config.get('USER_PROMPT_IMAGE') if input_type == 'image' else run_config.get('USER_PROMPT_VIDEO')
            text_input_ids = prepare_text_input(self.tokenizer, run_config['SYSTEM_PROMPT'], current_user_prompt, run_config['IMAGE_TOKEN_ID'])

            visual_input_processed = None
            if input_type == 'image': visual_input_processed = preprocess_images(run_config['IMAGES_PATH'], self.vision_transform)
            elif input_type == 'video':
                 sampled_frames_pil, _ = sample_video_frames(run_config['VIDEO_PATH'], run_config['NUM_FRAMES_TO_SAMPLE'])
                 visual_input_processed = preprocess_video_frames(sampled_frames_pil, self.vision_transform)
                 if not visual_input_processed: raise ValueError("Preprocessing video frames resulted in an empty list.")

            config_for_vision = {k: run_config[k] for k in ['ENABLE_SPATIAL_REDUCTION','TARGET_SPATIAL_TOKENS_SIDE', 'ENABLE_TEMPORAL_POOLING','TEMPORAL_POOLING_METHOD', 'TEMPORAL_ATTENTION_HEADS','TEMPORAL_ATTENTION_TYPE']}
            config_for_vision['ORIGINAL_SPATIAL_TOKENS_SIDE'] = self.config['ORIGINAL_SPATIAL_TOKENS_SIDE'] # Use original from init
            final_vision_features = extract_vision_features(self.vision_sess, self.vision_metadata, visual_input_processed, input_type, config_for_vision)

            combined_embeds = fuse_embeddings(self.embed_sess, text_input_ids, final_vision_features, run_config['IMAGE_TOKEN_ID'], self.embed_hidden_dim, run_config['DTYPE'])
            initial_sequence_length = combined_embeds.shape[1]

        except Exception as e:
            print(f"Error during pre-processing or feature extraction: {e}", file=sys.stderr)
            return None, time.time() - start_time

        # --- Step 4: Run Decoder ---
        effective_decoding_strategy = run_config['DECODING_STRATEGY']
        # Prepare config subset specifically for the chosen decoder function
        decoder_config = {'NUM_LAYERS': self.config['NUM_LAYERS'], 'NUM_HEADS': self.config['NUM_HEADS'], 'HEAD_DIM': self.config['HEAD_DIM'], 'DTYPE': run_config['DTYPE'], 'MAX_NEW_TOKENS': run_config['MAX_NEW_TOKENS'], 'VOCAB_SIZE': self.config['VOCAB_SIZE'], 'EOS_TOKEN_ID': self.config['EOS_TOKEN_ID']}
        if effective_decoding_strategy == "beam":
             decoder_config.update({'NUM_BEAMS': run_config['NUM_BEAMS'], 'LENGTH_PENALTY': run_config['LENGTH_PENALTY']})
        elif effective_decoding_strategy == "sampling":
             decoder_config.update({'TEMPERATURE': run_config['TEMPERATURE'], 'TOP_K': run_config['TOP_K'], 'TOP_P': run_config['TOP_P']})

        print(f"\n--- Running Generation with Strategy: {effective_decoding_strategy} ---")
        generated_ids = []
        try: # Decoding loop
            if effective_decoding_strategy == "greedy": generated_ids = run_decoder_greedy(self.decoder_sess, self.embed_sess, combined_embeds, initial_sequence_length, decoder_config)
            elif effective_decoding_strategy == "beam": generated_ids = run_beam_search(self.decoder_sess, self.embed_sess, combined_embeds, initial_sequence_length, decoder_config)
            elif effective_decoding_strategy == "sampling": generated_ids = run_sampling(self.decoder_sess, self.embed_sess, combined_embeds, initial_sequence_length, decoder_config)
            else: print(f"Error: Unknown decoding strategy '{effective_decoding_strategy}'.", file=sys.stderr); return None, 0
        except Exception as gen_e: print(f"Error during decoding loop ({effective_decoding_strategy}): {gen_e}", file=sys.stderr); return None, time.time() - start_time

        # --- Step 5: Decode Output ---
        end_time = time.time(); generation_time = end_time - start_time
        generated_text = None
        if generated_ids:
            if generated_ids[-1] == self.config['EOS_TOKEN_ID']: generated_ids = generated_ids[:-1]
            try:
                generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                print("\n✅ Generation Complete!")
            except Exception as dec_e:
                 print(f"Error decoding generated IDs: {dec_e}", file=sys.stderr)
                 # Return raw IDs perhaps? Or None. Let's return None.
                 return None, generation_time
        else:
            print("\n❌ No text generated or generation failed.")

        return generated_text, generation_time


    def close(self):
        """ Releases ONNX Runtime sessions and cleans up resources. """
        # (Implementation from previous code)
        print("\n--- Cleaning up VisionLanguageModelONNX resources ---")
        closed_count = 0
        try:
            if hasattr(self, 'embed_sess') and self.embed_sess is not None: 
                del self.embed_sess; self.embed_sess = None; closed_count+=1
            if hasattr(self, 'vision_sess') and self.vision_sess is not None: 
                del self.vision_sess; self.vision_sess = None; closed_count+=1
            if hasattr(self, 'decoder_sess') and self.decoder_sess is not None: 
                del self.decoder_sess; self.decoder_sess = None; closed_count+=1
            if hasattr(self, 'tokenizer') and self.tokenizer is not None: 
                del self.tokenizer; self.tokenizer = None
            collected_count = gc.collect()
            print(f"Resource cleanup initiated. Garbage collector collected {collected_count} objects.")
        except Exception as e: print(f"Error during resource cleanup: {e}", file=sys.stderr)
        print("----------------------------------------------------")

    def __del__(self):
        """ Attempt cleanup when object is garbage collected. """
        # print("VisionLanguageModelONNX object getting deleted, attempting cleanup...") # Optional debug
        self.close()

# ==============================================================================
#                       End of Library File
# ==============================================================================
# No main() or if __name__ == "__main__": block here for library usage.