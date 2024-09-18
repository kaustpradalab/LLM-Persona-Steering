#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 100 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/test/gemma-2b-it

python SAE/sae_run.py \
    --model_name gemma-2-2b-it \
    --sae_name gemma-scope-2b-pt-res-canonical \
    --sae_id layer_17/width_16k/canonical \
    --tokenizer_name google/gemma-2-2b-it \
    --layer 17 \
    --coeff 100 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/test/gemma-2-2b-it

python SAE/sae_run.py \
    --model_name gemma-2-9b-it \
    --sae_name gemma-scope-9b-it-res-canonical \
    --sae_id layer_31/width_131k/canonical \
    --tokenizer_name google/gemma-2-9b-it \
    --layer 31 \
    --coeff 100 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2-9b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/test/gemma-2-9b-it
