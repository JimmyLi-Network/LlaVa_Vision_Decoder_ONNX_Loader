#!/bin/bash

# Default directory is current directory
INPUT_DIR="mmbench/model_outputs"

# Check if llm_judge.py exists
if [ ! -f "llm_judge.py" ]; then
    echo "Error: llm_judge.py not found in the current directory."
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist."
    exit 1
fi

# Find all JSON files in the input directory
JSON_FILES=$(find "$INPUT_DIR" -maxdepth 1 -type f -name "*.json")

# Check if any JSON files were found
if [ -z "$JSON_FILES" ]; then
    echo "No JSON files found in $INPUT_DIR"
    exit 0
fi

echo "Found $(echo "$JSON_FILES" | wc -l) JSON files to process."

# Process each JSON file
for json_file in $JSON_FILES; do
    echo "Processing $json_file..."

    # Run llm_judge.py
    if python3 llm_judge.py "$json_file"; then
        echo "Successfully processed $json_file"
        # Display the summary from the output file
        output_file="$(dirname "$json_file")/$(basename "$json_file" .json)_eval.txt"
        if [ -f "$output_file" ]; then
            echo "Summary for $json_file:"
            cat "$output_file"
            echo ""
        else
            echo "Warning: Output file $output_file was not created."
        fi
    else
        echo "Error processing $json_file"
    fi
done

echo "All JSON files have been processed."