#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 300 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/full/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/full/gemma-2b-it_origin_300 \
    --testset TRAIT

python SAE/analysis_origin.py \
    --save_path ../data/SAE/steer_result/full/gemma-2b-it_origin_300 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/full
