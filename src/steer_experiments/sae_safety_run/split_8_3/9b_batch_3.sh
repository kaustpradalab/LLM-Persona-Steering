#!/bin/bash

python SAE/safety_sae_run.py \
    --model_name gemma-2-9b-it \
    --sae_name gemma-scope-9b-it-res-canonical \
    --sae_id layer_31/width_131k/canonical \
    --tokenizer_name google/gemma-2-9b-it \
    --layer 31 \
    --coeff 800 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_file_path ../data/SAE/bg_features/safety_split/gemma-2-9b-it_batch_3.json \
    --save_dir_path ../data/SAE/steer_result/case_study/safety_bench/gemma-2-9b-it_batched \
    --zero_shot True \