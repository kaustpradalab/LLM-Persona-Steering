#!/bin/bash
coeffs=$(seq 50 50 1000)

# Loop through each coefficient and run the command
for coeff in $coeffs; do
    python SAE/bias_sae_run.py \
        --model_name gemma-2-9b-it \
        --sae_name gemma-scope-9b-it-res-canonical \
        --sae_id layer_31/width_131k/canonical \
        --tokenizer_name google/gemma-2-9b-it \
        --layer 31 \
        --coeff $coeff \
        --bg_type fixed \
        --steer_file_path ../data/bg_features/coeff/gemma-2-9b-it.json \
        --save_dir_path ../data/bias/result/gemma-2-9b-it-$coeff \
        --file_path ../data/bias/combined_sentences.txt 
done

