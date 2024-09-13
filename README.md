# llm-personal-superposition

## Preparation
Install required python packages:
```
pip install -r requirements.txt
```

## Stimius Processing

### 1. SAE

1.1 Background Generation

Let't generate the keywords of each backgrounds first
```
cd src/stimius_processing
python SAE/processing_keywords.py --json_file [path to save bg json] --num[# of bgs you want]
```

Then, we can genertate bgs in two types(fixed & generative)

```
python SAE/processing_bg.py --json_file [path to bg json] --bg_type ['fixed' or 'gen']
```
1.2 Features Explorations
You can use convert_to_features to convert the background to a proper features list which includes the key features related the background description

```
python SAE/convert_to_features.py \
    --save_path [path to json need to save] \
    --pattern_path [path to pattern json] \
    --background [path to background json] \
    --layer [index of layer]
```
*More detailed examples refer to src/stimius_processing/SAE/sae_stimius_run.sh*

### 2. RepE
2.1 Keywords Generation

2.2 Control Vector Training

## Vector Control & TRAIT-Dark Running

### 1. SAE
run the TRAIT-Dark 
```
cd src/steer_experiments
python SAE/sae_run.py \
    --model_name  \
    --sae_name  \
    --layer  \
    --coeff  \
    --temperature  \
    --freq_penalty  \
    --bg_type  \
    --steer_mode  \
    --steer_file_path  \
    --prompt_type  \
    --inference_type  \
    --save_dir_path
```
*More detailed examples refer to src/steer_experiments/SAE/sae_trait_run.sh*

### 2. RepE

## Result
You can get the result of the model. 
```
python analysis.py --model_name gemma2_2b --prompt_type 1 --task_name sae
```

## Reference
> **Do LLMs Have Distinct and Consistent Personalities? TRAIT: Personality Testset designed for LLMs with Psychometrics**
