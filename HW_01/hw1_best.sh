#!/bin/bash
python3 test.py \
    --test_csv $1 \
    --output_file $2 \
    --model_path ./model
