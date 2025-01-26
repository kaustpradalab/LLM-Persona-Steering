import random
import openai
import time
import json
import argparse
import tiktoken

import json
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm, trange
from random import seed, choice
import time, json, sys, os, torch, argparse
from util.steering import *

def get_qa_response(model, question, answer, instruction, sae, tokenizer, layer, coeff, bg_type, temperature, freq_penalty, seed_num, bg_item,):
    model = model.eval()
    tokenizer.padding_side = 'left'
    prompt = instruction + "\n\n#Question#: " + question + "\n#Answer#: " + answer + "\n#Your Judgement#:"

    input = tokenizer(prompt, padding=True, return_tensors="pt").to('cuda')
    idx_dict, steering_vectors = get_steer_vectors(sae, bg_type, bg_item['features'])
    print("we will steer the features:", idx_dict)
    sampling_kwargs = dict(temperature=temperature, freq_penalty=freq_penalty)

    output = get_likelihood_generate(input, model, layer, coeff, steering_vectors, True, sampling_kwargs, seed=seed_num)

    prompt_length = input["input_ids"].shape[-1]
    generated_ids = output[0][prompt_length:]

    decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("The generated answer is" + decoded_text)
    return decoded_text



def evaluation_qa_dataset(model, file, instruction, output_path, sae, tokenizer, layer, coeff, bg_type, temperature, freq_penalty, seed_num, bg_item, log_path):
    with open(file, 'r', encoding="utf-8") as f:
        data = []
        for line in f:
            data.append(json.loads(line))

        data = data[:5000]
        correct = 0
        incorrect = 0
        for i in tqdm(range(len(data))):
            knowledge = data[i]["knowledge"]
            question = data[i]["question"]
            hallucinated_answer = data[i]["hallucinated_answer"]
            right_answer = data[i]["right_answer"]

            if random.random() > 0.5:
                answer = hallucinated_answer
                ground_truth = "Yes"
            else:
                answer = right_answer
                ground_truth = "No"

            ans = get_qa_response(model, question, answer, instruction, sae, tokenizer, layer, coeff, bg_type, temperature, freq_penalty, seed_num, bg_item)
            ans = ans.replace(".", "")

            if ("Yes" in ans and "No" in ans) or ("Yes" not in ans and "No" not in ans):
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": "failed!"}
                dump_jsonl(gen, output_path, append=True)
                incorrect += 1
                print('sample {} fails......'.format(i))
                continue
            elif "Yes" in ans:
                if ans != "Yes":
                    ans = "Yes"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            elif "No" in ans:
                if ans != "No":
                    ans = "No"
                gen = {"knowledge": knowledge, "question": question, "answer": answer, "ground_truth": ground_truth, "judgement": ans}
            else:
                gen = None
                incorrect += 1

            assert(gen is not None)

            if ground_truth == ans:
                correct += 1
            else:
                incorrect += 1

            print('sample {} success......'.format(i))
            dump_jsonl(gen, output_path, append=True)

        final_info = '{} got {} correct samples, {} incorrect samples, Accuracy: {}'.format(
            bg_item['idx'], correct, incorrect, correct / len(data)
        )
        # 先在控制台打印
        print(final_info)

        # 追加（concatenate）写进 log_path 文件
        with open(log_path, 'a', encoding='utf-8') as lf:
            lf.write(final_info + '\n')

def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
            json_record = json.dumps(data, ensure_ascii=False)
            f.write(json_record + '\n')

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

    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--freq_penalty', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")

    model_name = args.model_name
    sae_name = args.sae_name
    sae_id = args.sae_id
    tokenizer_name = args.tokenizer_name
    layer = args.layer
    coeff = args.coeff
    bg_type = args.bg_type
    temperature = args.temperature
    freq_penalty = args.freq_penalty
    seed = args.seed

    device = set_up()
    model, sae = load_model(model_name, sae_name, sae_id, device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    bg=json.load(open(args.steer_file_path, encoding='utf-8'))
    instruction_file = "../data/qa_evaluation_instruction.txt"
    f = open(instruction_file, 'r', encoding="utf-8")
    instruction = f.read()
    

    data = "../data/qa_data.json"
    for i, bg_item in enumerate(bg):
        outpath_m = f'{args.save_dir_path}' #/test_en_eva_zeroshot{zero_shot}_res.jsonl
        output_path = os.path.join(outpath_m, f"{bg_item['idx']}.json")
        log_path = os.path.join(outpath_m, "log.txt")
        print("save_dir", outpath_m)
        os.makedirs(outpath_m, exist_ok=True)
        evaluation_qa_dataset(model, data, instruction, output_path, sae, tokenizer, layer, coeff, bg_type, temperature, freq_penalty, seed, bg_item, log_path)

if __name__ == '__main__':
    main()
