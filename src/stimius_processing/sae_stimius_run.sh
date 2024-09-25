#!/bin/bash

#python SAE/processing_keywords.py \
#    --json_file ../data/SAE/background.json \
#    --num 10
#python SAE/processing_bg.py \
#    --json_file ../data/SAE/background.json \
#    --bg_type gen

#python SAE/processing_bg.py \
#    --json_file ../data/SAE/background/background_test.json \
#    --bg_type fixed

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/test/gemma-2b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
#    --background ../data/SAE/background/background_test.json \
#    --layer 12

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/test/gemma-2-2b.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-2b.json \
#    --background ../data/SAE/background/background_test.json \
#    --layer 17


#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/test/gemma-2-9b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
#    --background ../data/SAE/background/background_test.json \
#    --layer 31




#python SAE/processing_bg.py \
#    --json_file ../data/SAE/background/background_details_p5_n5.json \
#    --bg_type fixed
#
#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/p5_n5/gemma-2b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
#    --background ../data/SAE/background/background_details_p5_n5.json \
#    --layer 12
#
#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/p5_n5/gemma-2-2b.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-2b.json \
#    --background ../data/SAE/background/background_details_p5_n5.json \
#    --layer 17

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/p5_n5/gemma-2-9b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
#    --background ../data/SAE/background/background_details_p5_n5.json \
#    --layer 31




#python SAE/processing_bg.py \
#    --json_file ../data/SAE/background/background_details_p7_n3.json \
#    --bg_type fixed

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/p7_n3/gemma-2b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
#    --background ../data/SAE/background/background_details_p7_n3.json \
#    --layer 12

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/p7_n3/gemma-2-2b.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-2b.json \
#    --background ../data/SAE/background/background_details_p7_n3.json \
#    --layer 17

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/p7_n3/gemma-2-9b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
#    --background ../data/SAE/background/background_details_p7_n3.json \
#    --layer 31



#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/safety_split/gemma-2-9b-it_0_5.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
#    --background ../data/SAE/background/background_full_safety_split/background_full_safety_0_5.json \
#    --layer 31

#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/safety_split/gemma-2-9b-it_6_11.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
#    --background ../data/SAE/background/background_full_safety_split/background_full_safety_6_11.json \
#    --layer 31


#python SAE/convert_to_features.py \
#    --save_path ../data/SAE/bg_features/safety/gemma-2b-it.json \
#    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
#    --background ../data/SAE/background/background_full_safety.json \
#    --layer 12

python SAE/processing_bg.py \
    --json_file ../data/SAE/background/background_full.json \
    --bg_type fixed

python SAE/convert_to_features.py \
    --save_path ../data/SAE/bg_features/full/gemma-2b-it.json \
    --pattern_path ../data/SAE/features_patterns/gemma-2b-it.json \
    --background ../data/SAE/background/background_full.json \
    --layer 12

python SAE/processing_bg.py \
    --json_file ../data/SAE/background/background_full_split/background_full_22.json\
    --bg_type fixed

python SAE/convert_to_features.py \
    --save_path ../data/SAE/bg_features/full_split/gemma-2-9b-it_22.json \
    --pattern_path ../data/SAE/features_patterns/gemma-2-9b-it.json \
    --background ../data/SAE/background/background_full_split/background_full_22.json \
    --layer 31

