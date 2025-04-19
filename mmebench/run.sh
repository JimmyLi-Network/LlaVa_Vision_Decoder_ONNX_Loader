#!/bin/bash

# Define arrays for each parameter
Q_EMBED_ARRAY=("q4f16" "int8" "uint8" "fp16")
Q_VISION_ARRAY=("fp16" "q4f16" "q4")
Q_DECODER_ARRAY=("q4f16")

# Path to the Python script
PYTHON_SCRIPT="run_bench_multi.py"

# Ensure the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script $PYTHON_SCRIPT not found!" | tee -a "$LOG_FILE"
    exit 1
fi

# Create a log file with a timestamp
LOG_FILE="run_log_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

# Log the start of the script
echo "Starting script at $(date)" | tee -a "$LOG_FILE"

# Counter for tracking runs
RUN_COUNT=0

# Iterate through all combinations of parameters
for q_embed in "${Q_EMBED_ARRAY[@]}"; do
    for q_vision in "${Q_VISION_ARRAY[@]}"; do
        for q_decoder in "${Q_DECODER_ARRAY[@]}"; do
            echo "Running combination: Q_EMBED=$q_embed, Q_VISION=$q_vision, Q_DECODER=$q_decoder" | tee -a "$LOG_FILE"
            
            # Run the Python script with the current combination, redirecting output to terminal
            python3 "$PYTHON_SCRIPT" --q_embed "$q_embed" --q_vision "$q_vision" --q_decoder "$q_decoder"
            
            # Check if the command was successful
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                echo "Successfully completed run for Q_EMBED=$q_embed, Q_VISION=$q_vision, Q_DECODER=$q_decoder" | tee -a "$LOG_FILE"
            else
                echo "Error: Run failed for Q_EMBED=$q_embed, Q_VISION=$q_vision, Q_DECODER=$q_decoder" | tee -a "$LOG_FILE"
            fi
            
            ((RUN_COUNT++))
            echo "------------------------" | tee -a "$LOG_FILE"
        done
    done
done

# Log the completion of the script
echo "Completed $RUN_COUNT runs with all parameter combinations at $(date)" | tee -a "$LOG_FILE"