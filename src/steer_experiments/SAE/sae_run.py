import time, json, sys, os, torch, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from util.option_dict_4 import *
from util.prompts import  get_prompt
from util.lm_format import apply_format
from util.steering import *

def save_json(file_name, res_arr):
    with open(file_name, 'w') as f:
        json.dump(res_arr, f, indent=4, ensure_ascii=False)

def get_likelihood(model, input_ids, device):
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs[:, -1, :]  # Logits for the last token
    probabilities = torch.softmax(logits, dim=-1)
    return probabilities
    
def get_likelihood_steer(input_ids, model, layer, coeff, steering_vectors, steering_on, sampling_kwargs, seed=None):
    model.reset_hooks()
    steering_hook = create_steering_hook(coeff, steering_vectors, steering_on)
    editing_hooks = [(f"blocks.{layer}.hook_resid_post", steering_hook)]
    return hooked_generate(model, input_ids, editing_hooks, seed=seed, **sampling_kwargs)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--sae_name', type=str, default=None, required=True)
    parser.add_argument('--sae_id', type=str, default=None, required=True)
    parser.add_argument('--tokenizer_name', type=str, default=None, required=True)
    parser.add_argument('--layer', type=int, default=0)
    parser.add_argument('--coeff', type=int, default=200)
    parser.add_argument('--bg_type', choices=["fixed", "gen"], default="fixed")
    parser.add_argument('--steer_mode', action='store_true', help="Enable steering mode")
    parser.add_argument('--steer_file_path', type=str)
    parser.add_argument('--save_dir_path', type=str)

    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--freq_penalty', type=float, default=1.0)
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--inference_type', type=str, default="chat")
    parser.add_argument('--seed', type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    
    device = set_up()
    model, sae = load_model(args.model_name, args.sae_name, args.sae_id, device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    data=json.load(open("../data/TRAIT/TRAIT_Dark.json", encoding='utf-8'))
    bg=json.load(open(args.steer_file_path, encoding='utf-8'))
    
    subdir=f"prompt_type_{args.prompt_type}"
    save_dir=f"{args.save_dir_path}/{subdir}"

    for i, bg_item in enumerate(bg):
        save_file_dir=os.path.join(save_dir, f"{bg_item['idx']}.json")
        print("save_dir", save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        res_arr=[]
        for idx, sample in enumerate(data):
            print(idx)
            instruction=sample["situation"]+" "+sample["query"]
            response_high1=sample["response_high1"]
            response_high2=sample["response_high2"]
            response_low1=sample["response_low1"]
            response_low2=sample["response_low2"]
            sample["prompt_type"] = args.prompt_type
            sample["bg_index"] = bg_item['idx']
    
            for rev in [False, True]:
                prompt=get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                encoded=apply_format(prompt, args.inference_type, tokenizer)
                idx_dict, steering_vectors = get_steer_vectors(sae,args.bg_type, bg_item['features'])
                print("we will steer the features:", idx_dict)
                sampling_kwargs = dict(temperature=args.temperature, freq_penalty=args.freq_penalty)
                
                if args.steer_mode:
                    print("Steer Mode: ON")
                    likelihoods = get_likelihood_steer(encoded, model, args.layer, args.coeff, steering_vectors, True, sampling_kwargs, seed=args.seed).squeeze().tolist()
                else:
                    print("Steer Mode: OFF")
                    likelihoods = get_likelihood(model, encoded, device).squeeze().tolist()
                vocab_probabilities={}
                
                if args.prompt_type==1:
                    option_tokens=get_option_token("ABCD")
                elif args.prompt_type==2:
                    option_tokens=get_option_token("1234")
                elif args.prompt_type==3:
                    option_tokens=get_option_token("ABCD")
                for token in option_tokens:
                    vocab_probabilities[token]=likelihoods[tokenizer.convert_tokens_to_ids(token)]
                vocab_probabilities = dict(sorted(vocab_probabilities.items(), key=lambda item: item[1], reverse=True))
                vocab_probabilities = {k: vocab_probabilities[k] for k in list(vocab_probabilities)[:10]}

                if rev:
                    sample[f"prompt_rev"]=prompt
                    sample[f"likelihood_rev"]=vocab_probabilities
                else:
                    sample[f"prompt"]=prompt
                    sample[f"likelihood"]=vocab_probabilities
                
            res_arr.append(sample)
            if len(res_arr)%10==0:
                save_json(save_file_dir, res_arr)
        save_json(save_file_dir, res_arr)
        
        
if __name__ == '__main__':
    main()

#python run.py --model_name Chatgpt --model_name_short Chatgpt --prompt_type 1
#python run.py --model_name mistralai/Mistral-7B-Instruct-v0.2 --model_name_short mistral_instruct --inference_type chat --prompt_type 1