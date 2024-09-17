from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM

import torch
from tqdm import tqdm
import pandas as pd
import numpy as np

from repe import repe_pipeline_registry
repe_pipeline_registry()

from utils import pos_neg_dataset

import json
import argparse
import os 


from util.option_dict_4 import *
from util.prompts import  get_prompt

import sys



# convert data to positive and negative
def preprocess(data):
    df=pd.DataFrame(data)
    df=df[['idx', 'split', 'personality','response_high1','response_high2']]
    df['class']=df['split'].apply(lambda x:1 if x=='good' else 0)
    df.drop(columns=['idx', 'split'],inplace=True)
    return df

class REPE():
    def __init__(self,df,seed,model,trait,tokenizer,temp_str,rep_token=-1,n_difference=1,direction_method='pca',control_method="reading_vec"):
        self.df=df
        self.seed=seed
        self.trait=trait
        self.model=model
        self.tokenizer=tokenizer
        self.temp_str=temp_str
        self.rep_token=rep_token
        self.hidden_layers=list(range(-1, -self.model.config.num_hidden_layers, -1))
        self.n_difference=n_difference
        self.layer_id=list(range(-1, -10, -1))
        self.control_method=control_method
        self.direction_method=direction_method
        self.rep_reading_pipeline= pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        self.rep_control_pipeline= pipeline("rep-control", model=self.model, tokenizer=self.tokenizer, layers=self.layer_id, control_method=self.control_method)
        self.rep_reader_cache=None

    # construct the pos_neg_dataset
    def load_data(self):
        dataset= pos_neg_dataset(self.df,self.tokenizer,self.temp_str,self.seed)
        return dataset


    def rep_reader(self):
        dataset=self.load_data()

        pos_rep_reader = self.rep_reading_pipeline.get_directions(
        dataset['train']['data'], 
        rep_token=self.rep_token, 
        hidden_layers=self.hidden_layers, 
        n_difference=self.n_difference, 
        train_labels=dataset['train']['labels'], 
        direction_method=self.direction_method,
        batch_size=32,
        )

        return pos_rep_reader


    def get_answer(self,inputs):
        coeff=6.0
        max_new_tokens=128

        if self.rep_reader_cache is None:
            self.rep_reader_cache = self.rep_reader()
        
    
        rep_reader=self.rep_reader_cache
        

        result=[]
        
        activations = {}
        for layer in self.layer_id:
            activations[layer] = torch.tensor(coeff * rep_reader.directions[layer] * rep_reader.direction_signs[layer]).to(self.model.device).half()
        
        # output logits 
        baseline_outputs = self.rep_control_pipeline(inputs, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
        result.append(baseline_outputs)
        control_outputs = self.rep_control_pipeline(inputs, activations=activations, batch_size=4, max_new_tokens=max_new_tokens, do_sample=False)
        result.append(control_outputs)
        
        return result


# save file to json
def save_json(file_name, res_arr):
    with open(file_name, 'w') as f:
        json.dump(res_arr, f, indent=4, ensure_ascii=False)
        
device = "cuda"


# get the outputs and convert logits to likelihood
def get_likelihood(output):
    with torch.no_grad():
        logits = output.logits[:, -1, :]  # Logits for the last token
        probabilities= torch.softmax(logits, dim=-1)
    return probabilities


def get_args():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=None, required=True)
    parser.add_argument('--model_name_short', type=str, default=None, required=True)
    parser.add_argument('--inference_type', type=str, default="base")
    parser.add_argument('--prompt_type', type=int, default=1)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--paraphrase', action='store_true')
    parser.add_argument('--task_name',type=str,default='RePE')
    return parser.parse_args()


def main():
    # parse arguments
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    
    # load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(args.model_name, load_in_8bit=True, device_map='cuda', trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # load data(trait_dark)
    data=json.load(open("TRAIT_Dark.json"))
    prompts_data=json.load(open("prompts_data.json"))

    # keyword_dict
    keyword_list=["Cynical worldview","Lack of morality","Strategic manipulativeness", "Grandiosity","Entitlement","Dominance","Superiority","High impulsivity","Thrill-seeking","Low empathy","Low anxiety"]
    keyword_dict={}
    for idx,keyword in enumerate(keyword_list):
        keyword_dict[idx]=keyword

    # return dataframe
    df=preprocess(data)

    # count
    count=0

    # save path
    subdir=f"prompt_type_{args.prompt_type}"
    save_dir=f"../result/{args.model_name_short}/{args.task_name}/{subdir}"
    save_file_dir=os.path.join(save_dir,f"results_option_{args.model_name_short}.json")
    print("save_dir",save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    
    for trait in df['personality'].values:
        print(f"{trait}")
        # if true then use the paraphrase sentence to inference
        if args.paraphrase:
            run_type="inference_likelihood_paraphrase"
        else:
            run_type="inference_likelihood"
        
        

        # create the path
        # subdir=f"prompt_type_{args.prompt_type}"
        # save_dir=f"../{run_type}/{subdir}"
        # save_file_dir=os.path.join(save_dir, f"results_option_{args.model_name_short}.json")
        # print("save_dir", save_dir)
        # if not os.path.exists(save_dir):
            # os.makedirs(save_dir)

        temp_str=[]

        if trait=="Machiavellianism":
            temp_str=[prompts_data["Cynical worldview"],prompts_data["Lack of morality"],prompts_data["Strategic manipulativeness"]]
        elif trait=='Narcissism':
            temp_str=[prompts_data['Grandiosity'],prompts_data['Entitlement'], prompts_data['Dominance'], prompts_data['Superiority']]
        else:
            temp_str=[prompts_data['High impulsivity'],prompts_data['Thrill-seeking'], prompts_data['Low empathy'], prompts_data['Low anxiety']]
        
        for temp_str in temp_str:
            # instance repe
            repe=REPE(df[df['personality']==trait],42,model,trait,tokenizer,temp_str)

            res_arr=[]
            
            for idx, sample in enumerate(data):
                print(idx)
                
                if args.paraphrase:
                    instruction=sample["paraphrased_situation"]+" "+sample["paraphrased_query"]
                else:
                    instruction = sample["situation"] + " " + sample["query"]
                response_high1 = sample["response_high1"]
                response_high2 = sample["response_high2"]
                response_low1 = sample["response_low1"]
                response_low2 = sample["response_low2"]
                sample["keywords"]=keyword_dict[count]
                

                

                # Generate reverse prompt
                for rev in [False, True]:
                    # Inputs
                    prompt = get_prompt(1, rev, instruction, response_high1, response_high2, response_low1, response_low2)
                    
        
                    # Call get_answer to get baseline and control output
                    outputs = repe.get_answer(prompt)
                    

                    # Initialize a dictionary to store likelihoods for this prompt
                    all_vocab_probabilities = {}

                    # Iterate over each output (baseline and control outputs)
                    for i, output in enumerate(outputs):    
                    
                        likelihoods = get_likelihood(output).squeeze().tolist()
                        vocab_probabilities = {}
                    
                        if args.prompt_type==1:
                            option_tokens=get_option_token("ABCD")
                        elif args.prompt_type==2:
                            option_tokens=get_option_token("1234")
                        elif args.prompt_type==3:
                            option_tokens=get_option_token("ABCD")
                    
                        # Choose tokens of high possibilities
                        for token in option_tokens:
                            vocab_probabilities[token] = likelihoods[tokenizer.convert_tokens_to_ids(token)]
            
                        # Sort the vocab_probabilities by likelihood and keep the top 10
                        vocab_probabilities = dict(sorted(vocab_probabilities.items(), key=lambda item: item[1], reverse=True))
                        vocab_probabilities = {k: vocab_probabilities[k] for k in list(vocab_probabilities)[:10]}
            
                        # Store vocab_probabilities for this output (baseline or control)
                        all_vocab_probabilities[f'output_{i}'] = vocab_probabilities

                        # Depending on the value of rev, store the results in the sample dictionary
                        if rev:
                            sample["prompt_rev"] = prompt
                            sample["likelihood_rev"] = all_vocab_probabilities
                            # print(sample["likelihood_rev"])
                        else:
                            sample["prompt"] = prompt
                            sample["likelihood"] = all_vocab_probabilities
                            # print(sample["likelihood"])
                
                # store sample         
                res_arr.append(sample)
                # print(res_arr)
                if len(res_arr)%args.save_interval==0:
                    save_json(save_file_dir, res_arr)

            save_json(save_file_dir, res_arr)
            print(f"{keyword_dict[count]} ends")
            count+=1
            
       
if __name__ == '__main__':
    main()



