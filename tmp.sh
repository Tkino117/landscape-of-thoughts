#!/bin/bash

for i in 1 2 3 4 5; do
  python main.py \
    --task all \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --dataset_name same_ans${i} \
    --data_path ./lot/data/same_ans${i}.jsonl \
    --method cot \
    --num_samples 20 \
    --start_index 0 \
    --end_index 5 \
    --plot_type method \
    --output_dir figures/same_ans${i}/landscape \
    --local \
    --local_api_key token-abc123
done
