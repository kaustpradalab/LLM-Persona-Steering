#!/bin/bash

python SAE/safety_sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 200 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_file_path ../data/SAE/bg_features/safety/gemma-2b-it_pos.json \
    --save_dir_path ../data/SAE/steer_result/case_study/safety_bench/gemma-2b-it-pos \
    --zero_shot True \
