python SAE/sae_run.py \
    --model_name gemma-2-9b-it \
    --sae_name gemma-scope-9b-it-res-canonical \
    --sae_id layer_31/width_131k/canonical \
    --tokenizer_name google/gemma-2-9b-it \
    --layer 31 \
    --coeff 800 \
    --bg_type fixed \
    --steer_mode \
    --steer_file_path ../data/SAE/bg_features/coeff/gemma-2-9b-it.json \
    --prompt_type 1 \
    --inference_type chat \
    --save_dir_path ../data/SAE/case_study/steer_result/coeff_change/gemma-2-9b-it

python SAE/analysis.py \
    --save_path ../data/SAE/steer_result/case_study/coeff_change/gemma-2-9b-it \
    --prompt_type 1 \
    --model_name gemma-2-9b-it \
    --bg_features ../data/SAE/bg_features/coeff