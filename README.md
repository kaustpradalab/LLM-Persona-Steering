# llm-personal-superposition


# How to run TRAIT-Dark Experiments

## Preparation
Install required python packages:
```
pip install -r requirements.txt
```

## run TRAIT-Dark
run the TRAIT-Dark 
```
python run.py --model_name [path to model] --model_name_short [short name of model] --inference_type chat --prompt_type [prompt type] --task_name [sae OR RepE] --background [path to bg]
```

### Result
You can get the result of the model. 
```
python analysis.py --model_name gemma2_2b --prompt_type 1 --task_name sae
```
​
## Generate question prompts
Generate the Dark Prompts
```
python prompt_generate.py --prompt_type 1 --dataset TRAIT_Dark --background BG/SAE_BG --bg_name_short sae
```

## Reference
> **Do LLMs Have Distinct and Consistent Personalities? TRAIT: Personality Testset designed for LLMs with Psychometrics**