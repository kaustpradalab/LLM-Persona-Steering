import json
import time, sys, os, torch, argparse
import requests
import argparse
import pandas as pd
from openai import OpenAI
import random
import os
class GPTAssistant:
    def __init__(self, api_key: str, url: str=None, model: str = "gpt-4-turbo"):
        self.client = OpenAI(

            api_key=api_key,
            base_url=url,
        )


        self.model = model

    def load_json_data(self, json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)


    def get_response_content(self, response):
        try:
            response_json = response.json()

            if 'choices' not in response_json:
                print("Error: 'choices' key not found in the response JSON.")
                return "None"

            choices = response_json['choices']

            # Check if 'choices' is a list and is not empty
            if not isinstance(choices, list) or len(choices) == 0:
                print("Error: 'choices' is not a valid list or is empty.")
                return "None"

            # Check if the first item in 'choices' has the 'message' key
            if 'message' not in choices[0]:
                print("Error: 'message' key not found in the first item of 'choices'.")
                return "None"

            message = choices[0]['message']

            # Check if 'message' has the 'content' key
            if 'content' not in message:
                print("Error: 'content' key not found in the 'message'.")
                return "None"

            # Return the content if all checks pass
            return message['content']

        except requests.exceptions.JSONDecodeError:
            print("Error: Failed to decode JSON from the response.")
            return "None"


    # def generate_programs(self, system_prompt, prompt, question):
    def generate_personality(self, prompt, condition=None):
        # system_prompt = system_prompt.rstrip()
        # prompt = (prompt + question).rstrip()
        prompt = prompt.rstrip()
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": condition
                }
            ],
            temperature=random.uniform(0.6, 0.99),
            # top_p=random.uniform(0.8, 1)
        )


        response = completion.choices[0].message.content
        # print(response)
        return response
        # return self.get_response_content(response)


def fixed_bg(features):
    template = {
    "Gender": "your gender is {Gender}",
    "Age": "you are a {Age} people",
    "Education level": "you have a {Education level} education level",
    "Socioeconomic status": "you come from a {Socioeconomic status} wealth status background",
    "Social ideology": "you hold strong beliefs in {Social ideology}",
    "Emotional intelligence": "You're an  {Emotional intelligence} person",
    "Professional commitment": "you show your professional commitment as someone who is {Professional commitment}",
    "Family background": "your family relationship is {Family background}"
    }
    return {key: value.format(**features) for key, value in template.items()}

def generative_bg(features):
    api_key = "Your openai key"
    url_str = "Your url"

    assistant = GPTAssistant(api_key, url_str)

    # Example usage

    plain_text_path = 'SAE/details_prompt.txt'
    # system_prompt = "Task Description: \nYou are helping users to generate four types of prompts: naive prompts, keyword prompts, personality prompts and reverse personality prompt based on the given personality and realted keywords. \n The personality prompts are descriptive sentences about specific personality traits while reverse personality prompts are the opposite of personality prompts.\nThe sentence structure of the reverse personality prompt mirrors that of the original personality prompt.\n"
    # system_prompt = "Generate 100 unique American English speakers, each represented by a distinct combination of the following eight dimensions: gender, age (16-80), cultural identity, socioeconomic status, education level, family status, profession, and personal identity.\n Ensure diversity across these dimensions:\n "
    with open(plain_text_path, "r") as input_fp:
        system_prompt = "".join(input_fp.readlines())
    
    user_prompt = f"keywords of the character's background:\n {features}\nOutput in JSON format without extra explanation where the key is 'Details'\n"
    response = assistant.generate_personality(system_prompt, user_prompt)
    if isinstance(response, str):
        cleaned_response = response.replace('```json', '').replace('```', '').strip()
        print("cleaned")
    else:
        cleaned_response = "None"
        print("str")

    try:
        parsed_json = json.loads(cleaned_response)
        details_value = parsed_json.get('Details', '')
        print(details_value)
    except json.JSONDecodeError:
        details_value = "None"
        print("Invalid JSON format")
    

def process_bg(json_file_path, argument):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        features = item.get("attributes", {})
        if argument == "fixed":
            item["fixed_bg"] = fixed_bg(features)
        elif argument == "gen":
            item["generative_bg"] = generative_bg(features)  

    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"{argument}_bg finished generation.")
    print(f"data is saved in {json_file_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", help="The path to the JSON file to process.")
    parser.add_argument("--bg_type", choices=["fixed", "gen"], help="The type of background to generate: 'fixed' or 'gen'.")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    process_bg(args.json_file, args.bg_type)

if __name__ == '__main__':
    main()