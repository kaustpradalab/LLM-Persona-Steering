#!/bin/bash

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 200 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff200_tem02

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff200_tem02 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/test

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 400 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff400_tem02

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff400_tem02 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/test
    
python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 800 \
    --temperature 0.2 \
    --freq_penalty 1 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/test/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff800_tem02

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff800_tem02 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/test