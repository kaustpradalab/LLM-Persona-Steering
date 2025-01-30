#!/bin/bash

# Define coefficients
coeffs=$(seq 50 50 1000)

# Loop through each coefficient and run the command
for coeff in $coeffs; do
    python SAE/bias_sae_run.py \
        --model_name gemma-2b-it \
        --sae_name gemma-2b-it-res-jb \
        --sae_id blocks.12.hook_resid_post \
        --tokenizer_name google/gemma-2b-it \
        --layer 12 \
        --coeff $coeff \
        --bg_type fixed \
        --steer_file_path ../data/bg_features/coeff/gemma-2b-it.json \
        --save_dir_path ../data/bias/result/gemma-2b-it-$coeff \
        --file_path ../data/bias/combined_sentences.txt
done

