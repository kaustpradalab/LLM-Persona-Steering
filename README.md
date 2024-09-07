# llm-personal-superposition

# How to run Experiments

# Preparation
Install required python packages:
```
pip install -r requirements.txt
```

# Stimius_processing
## 1. SAE
### 1.1 Background Generation
Let't generate the keywords of each backgrounds first
```
cd src/stimius_processing
python SAE/processing_keywords.py --json_file [path to bg json] --num [# of bgs you want]
```
Then, we can genertate bgs in two types(fixed & generative)
```
python SAE/processing_bg.py --json_file [path to bg json] --bg_type ['fixed' or 'gen']
```
### 1.2 Features Explorations

## 2. SAE
### 2.1 Keywords Generation
### 2.2 Control Vector Training

# run TRAIT-Dark
run the TRAIT-Dark 
```
python run.py --model_name [path to model] --model_name_short [short name of model] --inference_type chat --prompt_type [prompt type] --task_name [sae OR RepE] --background [path to bg]
```

# Result
You can get the result of the model. 
```
python analysis.py --model_name gemma2_2b --prompt_type 1 --task_name sae
```
​
# Generate question prompts
Generate the Dark Prompts
```
python prompt_generate.py --prompt_type 1 --dataset TRAIT_Dark --background BG/SAE_BG --bg_name_short sae
```

# Reference
> **Do LLMs Have Distinct and Consistent Personalities? TRAIT: Personality Testset designed for LLMs with Psychometrics**