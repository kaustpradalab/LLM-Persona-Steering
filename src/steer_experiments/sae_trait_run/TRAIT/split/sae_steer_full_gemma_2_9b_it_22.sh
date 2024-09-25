#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2-9b-it \
    --sae_name gemma-scope-9b-it-res-canonical \
    --sae_id layer_31/width_131k/canonical \
    --tokenizer_name google/gemma-2-9b-it \
    --layer 31 \
    --coeff 800 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/full_split/gemma-2-9b-it_22.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/full/gemma-2-9b-it_origin \
    --testset TRAIT
