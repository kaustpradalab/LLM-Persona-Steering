import os, json, sys, argparse
import numpy as np
import pandas as pd
from util.option_dict_4 import *
def get_score(cnt_dict, ):
    personality_arr=["Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness", "Psychopathy", "Machiavellianism", "Narcissism"]
    score_arr=[]
    for personality in personality_arr:
        if (cnt_dict[personality]["high"]+cnt_dict[personality]["low"])==0:
            print("continue")
            continue
        score=( (cnt_dict[personality]["high"]) / (cnt_dict[personality]["high"]+cnt_dict[personality]["low"]) )*100
        
        score_arr.append(score)
    return score_arr

def get_likelihoods(sample, option_tokens, model_name):
    """Calculate likelihoods based on whether model_name contains 'gpt'."""
    likelihood = {}
    likelihood_rev = {}
    if "gpt" in model_name.lower():
        for token in option_tokens:
            likelihood[token] = np.exp(sample["likelihood"].get(token, 0))
            likelihood_rev[token] = np.exp(sample["likelihood_rev"].get(token, 0))
    else:
        for token in option_tokens:
            likelihood[token] = sample["likelihood"].get(token, 0)
            likelihood_rev[token] = sample["likelihood_rev"].get(token, 0)
    return likelihood, likelihood_rev

def normalize_likelihoods(likelihoods):
    """Normalize likelihoods and return a dictionary with normalized values."""
    likelihood, likelihood_rev = likelihoods
    total = sum(likelihood.values())
    total_rev = sum(likelihood_rev.values())
    return {
        'norm': {k: v / total for k, v in likelihood.items()},
        'norm_rev': {k: v / total_rev for k, v in likelihood_rev.items()}
    }

def get_max_option(norm_likelihoods):
    """Calculate the max option based on normalized likelihoods."""
    norm = norm_likelihoods['norm']
    norm_rev = norm_likelihoods['norm_rev']
    high_1 = (norm['A'] + norm_rev['B']) / 2
    low_1 = (norm['B'] + norm_rev['A']) / 2
    high_2 = (norm['C'] + norm_rev['D']) / 2
    low_2 = (norm['D'] + norm_rev['C']) / 2
    return np.argmax([high_1, high_2, low_1, low_2])

def get_option_token(options):
    # Assume this function returns the appropriate token list based on input
    return list(options)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=None, required=True)
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--bg_features', type=str, default=None, required=True)
    return parser.parse_args()

def main():
    args = get_args()
    input_directory = f"{args.save_path}/prompt_type_{args.prompt_type}"
    with open(os.path.join(args.bg_features, f"{args.model_name}.json"), 'r', encoding='utf-8') as file:
        bg_features = json.load(file)
    result_data = []
    for filename in os.listdir(input_directory):
        if filename.endswith(".json"):
            idx = int(filename.split('.')[0])  # Extract index from filename
            filepath = os.path.join(input_directory, filename)
            data = json.load(open(filepath))

            # Determine option tokens based on prompt type
            if args.prompt_type == 1:
                option_tokens = get_option_token("ABCD")
            elif args.prompt_type == 2:
                option_tokens = get_option_token("1234")
            elif args.prompt_type == 3:
                option_tokens = get_option_token("ABCD")

            # Initialize count dictionary for this file
            cnt_dict={
                "Agreeableness": {"high":0, "low":0},
                "Conscientiousness": {"high":0, "low":0},
                "Extraversion": {"high":0, "low":0},
                "Neuroticism": {"high":0, "low":0},
                "Openness": {"high":0, "low":0},
                "Psychopathy": {"high":0, "low":0},
                "Machiavellianism": {"high":0, "low":0},
                "Narcissism": {"high":0, "low":0},
            }

            for sample in data:
                personality = sample["personality"]
                
                # Process likelihoods
                likelihoods = get_likelihoods(sample, option_tokens, args.model_name)
                likelihood_norms = normalize_likelihoods(likelihoods)
                max_option = get_max_option(likelihood_norms)

                # Update counts based on max option
                if max_option in [0, 1]:
                    cnt_dict[personality]["high"] += 1
                elif max_option in [2, 3]:
                    cnt_dict[personality]["low"] += 1

            # Calculate scores and append results
            scores = get_score(cnt_dict)
            result_data.append({
                "idx": idx,
                "scores": {
                    "Agreeableness": scores[0],
                    "Conscientiousness": scores[1],
                    "Extraversion": scores[2],
                    "Neuroticism": scores[3],
                    "Openness": scores[4],
                    "Psychopathy": scores[5],
                    "Machiavellianism": scores[6],
                    "Narcissism": scores[7],
                }
            })
            
    for result in result_data:
        item = next((item for item in bg_features if item["idx"] == result["idx"]), None)
        if item:
            item["scores"] = result["scores"]
        
    # Write results to a new JSON file
    with open(f"{args.save_path}/summary_scores.json", "w") as outfile:
        json.dump(bg_features, outfile, indent=4)
        
    print("Summary Score saved.")
            

if __name__ == '__main__':
    main()