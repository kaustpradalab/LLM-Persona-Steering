python SAE/convert_to_features.py \
    --save_path ../data/SAE/bg_features/coeff/gemma-2b-it.json \
    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
    --background ../data/SAE/background/background_coeff.json \
    --layer 12

python SAE/convert_to_features.py \
    --save_path ../data/SAE/bg_features/coeff/gemma-2-9b-it.json \
    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
    --background ../data/SAE/background/background_coeff.json \
    --layer 31