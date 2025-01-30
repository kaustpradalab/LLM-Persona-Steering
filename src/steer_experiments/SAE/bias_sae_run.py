import time, json, sys, os, torch, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from util.option_dict_4 import *
from util.lm_format import apply_format
from util.steering import *

def save_json(file_name, res_arr):
    with open(file_name, 'w') as f:
        json.dump(res_arr, f, indent=4, ensure_ascii=False)

def get_logit(model, input_ids, device):
    """
    Computes the logits for the given input_ids using the model.
    
    Args:
        model: The model to be used for predictions.
        input_ids: The input IDs for the model.
        device: The device to run the model on (e.g., "cpu" or "cuda").
    
    Returns:
        logits: The raw logits for the last token in the input sequence.
    """
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs[:, -1, :]  # Logits for the last token
    return logits


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--sae_name', type=str, default=None, required=True)
    parser.add_argument('--sae_id', type=str, default=None, required=True)
    parser.add_argument('--tokenizer_name', type=str, default=None, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--coeff', type=int, default=200)
    parser.add_argument('--bg_type', choices=["fixed", "gen"], default="fixed")
    parser.add_argument('--steer_file_path', type=str)
    parser.add_argument('--save_dir_path', type=str)
    parser.add_argument('--file_path', type=str, default=None, required=True)
    parser.add_argument('--steer_mode', action='store_true', help="Enable steering mode")
    
    parser.add_argument('--inference_type', type=str, default="base")
    parser.add_argument('--seed', type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    
    device = set_up()
    model, sae = load_model(args.model_name, args.sae_name, args.sae_id, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    #data=json.load(open(f"../data/TRAIT/{args.testset}.json", encoding='utf-8'))
    file_path = args.file_path
    
    

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = file.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    data = [line.strip() for line in data]

    bg=json.load(open(args.steer_file_path, encoding='utf-8'))
    
    save_dir=f"{args.save_dir_path}"

    for i, bg_item in enumerate(bg):
        save_file_dir=os.path.join(save_dir, f"{bg_item['idx']}.json")
        print("save_dir", save_dir)
        os.makedirs(save_dir, exist_ok=True)
        
        res_arr=[]
        for idx, input in enumerate(data):
            print(idx)
            result={}
            encoded=apply_format(input, args.inference_type, tokenizer)
            idx_dict, steering_vectors = get_steer_vectors(sae,args.bg_type, bg_item['features'])
            print("we will steer the features:", idx_dict)
            result[input] = {
                "candidates steer logit": {},
                "candidates origin logit": {}
            }

            logits_steer = get_logit_steer(encoded, model, args.layer, args.coeff, steering_vectors, True, seed=args.seed).squeeze().tolist()
            logits_no_steer = get_logit(model, encoded, device).squeeze().tolist()
            vocab_logit_steer={}
            vocab_logit_no_steer={}
            compare_tokens = get_demographic_pair(input)

            for token in compare_tokens:
                vocab_logit_steer[token]=logits_steer[tokenizer.convert_tokens_to_ids(token)]
                result[input]["candidates steer logit"][token] = vocab_logit_steer[token]
                vocab_logit_no_steer[token]=logits_no_steer[tokenizer.convert_tokens_to_ids(token)]
                result[input]["candidates origin logit"][token] = vocab_logit_no_steer[token]
            if bg_item['idx'] % 2 == 0:  # Even
                print(f"The index {bg_item['idx']} is even.")
                logit_diff_steer = vocab_logit_steer[compare_tokens[1]] - vocab_logit_steer[compare_tokens[0]]
                logit_diff_origin = vocab_logit_no_steer[compare_tokens[1]] - vocab_logit_no_steer[compare_tokens[0]]
            else:  # Odd
                print(f"The index {bg_item['idx']} is odd.")
                logit_diff_steer = vocab_logit_steer[compare_tokens[0]] - vocab_logit_steer[compare_tokens[1]]
                logit_diff_origin = vocab_logit_no_steer[compare_tokens[0]] - vocab_logit_no_steer[compare_tokens[1]]
            result[input]["logit difference steer"] = logit_diff_steer
            result[input]["logit difference origin"] = logit_diff_origin
            res_arr.append(result)
            save_json(save_file_dir, res_arr)
        save_json(save_file_dir, res_arr)
        
        
if __name__ == '__main__':
    main()

#python run.py --model_name Chatgpt --model_name_short Chatgpt --prompt_type 1
#python run.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --model_name_short mistral_instruct --inference_type chat --prompt_type 1