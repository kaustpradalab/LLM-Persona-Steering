import time, json, sys, os, torch, argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


from option_dict_4 import *
from prompts import  get_prompt
from lm_format import apply_format


def save_json(file_name, res_arr):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(res_arr, f, indent=4, ensure_ascii=False)
        
device = "cuda"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--dataset', type=str, default="TRAIT_DARK")
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--background', type=str, default="BG/SAE_BG")
    parser.add_argument('--bg_name_short', type=str, default="sae")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    data=json.load(open(f"../../{args.dataset}.json", encoding='utf-8'))
    bg=json.load(open(f"../../{args.background}.json", encoding='utf-8'))
    subdir=f"prompt_type_{args.prompt_type}"
    save_dir=f"../Generated_Prompts/{args.bg_name_short}/{subdir}"
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
        
            for rev in [False, True]:
                prompt=get_prompt(args.prompt_type, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                if rev:
                    sample[f"prompt_rev"]=prompt
                else:
                    sample[f"prompt"]=prompt
            sample["prompt_type"] = args.prompt_type
            sample["bg_desciption"] = bg_item["Description"]
            sample["bg_index"] = bg_item['idx']
            res_arr.append(sample)
            if len(res_arr)%args.save_interval==0:
                save_json(save_file_dir, res_arr)
        save_json(save_file_dir, res_arr)  
if __name__ == '__main__':
    main()