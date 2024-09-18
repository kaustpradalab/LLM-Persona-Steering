#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 200 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/p5_n5/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/p5_n5/gemma-2b-it

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/p5_n5/gemma-2b-it \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/p5_n5

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/test/gemma-2b-it \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/test