#!/bin/bash

python SAE/hallu_sae_run.py \
    --model_name gemma-2-9b-it \
    --sae_name gemma-scope-9b-it-res-canonical \
    --sae_id layer_31/width_131k/canonical \
    --tokenizer_name google/gemma-2-9b-it \
    --layer 31 \
    --coeff 4 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_file_path ../data/bg_features/coeff/gemma-9b-it.json \
    --save_dir_path ../data/Hallu/gemma-9b-it 