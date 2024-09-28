#!/bin/bash

#python SAE/sae_run.py \
#    --model_name gemma-2b-it \
#    --sae_name gemma-2b-it-res-jb \
#    --sae_id blocks.12.hook_resid_post \
#    --tokenizer_name google/gemma-2b-it \
#    --layer 12 \
#    --coeff 200 \
#    --bg_type fixed \
#    --steer_mode \
#    --steer_file_path ../data/SAE/bg_features/coeff/gemma-2b-it.json \
#    --prompt_type 1 \
#    --inference_type chat \
#   --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff200 \
#    --testset TRAIT_Dark

python SAE/analysis_dark.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff200 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/coeff

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 400 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/coeff/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff400 \
    --testset TRAIT_Dark

python SAE/analysis_dark.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff400 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/coeff
    
python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 800 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/coeff/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff800 \
    --testset TRAIT_Dark

python SAE/analysis_dark.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff800 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/coeff

python SAE/sae_run.py \
    --model_name gemma-2b-it \
    --sae_name gemma-2b-it-res-jb \
    --sae_id blocks.12.hook_resid_post \
    --tokenizer_name google/gemma-2b-it \
    --layer 12 \
    --coeff 1000 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/coeff/gemma-2b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff800 \
    --testset TRAIT_Dark

python SAE/analysis_dark.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2b-it/gemma-2b-it_coeff800 \
    --prompt_type 1 \
    --model_name gemma-2b-it \
    --bg_features ../data/SAE/bg_features/coeff