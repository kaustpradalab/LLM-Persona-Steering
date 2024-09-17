#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 150 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/gemma-2b-it

