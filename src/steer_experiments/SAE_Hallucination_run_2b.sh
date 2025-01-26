#!/bin/bash

python SAE/hallu_sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 4 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_file_path ../data/bg_features/coeff/gemma-2b-it.json \
    --save_dir_path ../data/Hallu/gemma-2b-it 
