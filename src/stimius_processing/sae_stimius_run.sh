#!/bin/bash

#python SAE/processing_keywords.py \
#    --json_file ../data/SAE/background.json \
#    --num 10
#python SAE/processing_bg.py \
#    --json_file ../data/SAE/background.json \
#    --bg_type gen
#python SAE/processing_bg.py \
#    --json_file ../data/SAE/background.json \
#    --bg_type fixed

python SAE/convert_to_features.py \
    --save_path ../data/SAE/bg_features/gemma-2b-it.json \
    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
    --background ../data/SAE/background.json \
    --layer 12
