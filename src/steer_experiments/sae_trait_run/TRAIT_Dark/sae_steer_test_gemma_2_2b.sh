#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2-2b \
    --sae_name gemma-scope-2b-pt-res-canonical \
    --sae_id layer_17/width_65k/canonical \
    --tokenizer_name google/gemma-2-2b \
    --layer 17 \
    --coeff 700 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2-2b.json \
    --prompt_type 1 \
    --inference_type base \
    --save_dir_path ../data/SAE/steer_result/test/gemma-2-2b_1000

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/test/gemma-2-2b_200 \
    --prompt_type 1 \
    --model_name gemma-2-2b \
    --bg_features ../data/SAE/bg_features/test
