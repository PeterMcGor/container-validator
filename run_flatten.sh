#!/bin/bash

# Example usage:
# ./run_flatten.sh /path/to/task_x/labels /path/to/output


INPUT_PATH="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task1/labels"
OUTPUT_PATH="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task1/labels_flattened"
python3 flatten_structure.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"

INPUT_PATH="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task2/labels"
OUTPUT_PATH="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task2/labels_flattened"
python3 flatten_structure.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"

INPUT_PATH="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task3/labels"
OUTPUT_PATH="/media/jaume/DATA/Data/fomo25_finetune_data/fomo-task3/labels_flattened"
python3 flatten_structure.py --input "$INPUT_PATH" --output "$OUTPUT_PATH"