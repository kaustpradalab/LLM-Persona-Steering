import requests
import json
import argparse
import pandas as pd
from openai import OpenAI
import time, sys, os, torch, argparse
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", help="The path to the JSON file to process.")
    parser.add_argument("--num", type=int, default=5, help="The number of items to generate")
    return parser.parse_args()

def main():
    api_key = "Your api key"
    url_str = "Your url"

    assistant = GPTAssistant(api_key, url_str)

    # Example usage

    args = get_args()
    print(f"python {' '.join(sys.argv)}")

    plain_text_path = 'keyword_prompt.txt'
    # system_prompt = "Task Description: \nYou are helping users to generate four types of prompts: naive prompts, keyword prompts, personality prompts and reverse personality prompt based on the given personality and realted keywords. \n The personality prompts are descriptive sentences about specific personality traits while reverse personality prompts are the opposite of personality prompts.\nThe sentence structure of the reverse personality prompt mirrors that of the original personality prompt.\n"
    # system_prompt = "Generate 100 unique American English speakers, each represented by a distinct combination of the following eight dimensions: gender, age (16-80), cultural identity, socioeconomic status, education level, family status, profession, and personal identity.\n Ensure diversity across these dimensions:\n "
    with open(plain_text_path, "r") as input_fp:
        prompt = "".join(input_fp.readlines())

    condition_list = [
        "Output one character, The attitude of Life is satisfied to some extent. Ensure diversity across all dimensions, especially in gender representation, including both male and female profiles",
        "Output one character, The attitude of Life is towards negative. Ensure diversity across all dimensions, especially in gender representation, including both male and female profiles while maintaining diversity and realism in other aspects",
        "Output one character, The socioeconomic status is towards negative. Ensure diversity across all dimensions, especially in Gender, Age and Family background",
        "Output one character, The social ideology should be diverse not limited on Environmentalism while maintaining diversity and realism in other aspects",
        "Output one character, The emotional intelligence is towards negative. The Gender should be Male. Ensure diversity across all the other dimensions",
        "Output one character, The attitude of Life is towards negative. The Gender should be Male. Ensure diversity across all the other dimensions",
        "Output one character, The attitude of Life is satisfied to some extent. The Gender should be Male. The social ideology should be diverse not limited on Environmentalism. Ensure diversity and realism across all the other dimensions",
        "Output one character, The Professional commitment should be positive to some extent. The Gender should be Male. The social ideology should be diverse not limited on Environmentalism. Ensure diversity and realism across all the other dimensions",
        "Output one character, The Professional commitment should be negative to some extent. The Gender should be Male. The social ideology should be diverse not limited on Environmentalism. Ensure diversity and realism across all the other dimensions",
    ]

    response_list = []
    for i in range(args.num):
        if i%1 ==0:
            # print(29//10)
            condition = condition_list[i//1]
        # condition = "The attitude of Life is towards negative or not that satisfied, or other dimensions are towards negative"
        # choices = random.choice(character_profile['Life_Satisfaction'])
        # condition = f"The attitude of Life: {choices}"
        # condition = f"The attitude of Life: Dissatisfied, the reason can be being bullied, unemployment, drug addicted, family harnessment, some other challengings or the combination of these factors"
        # condition = f"The attitude of Life: Dissatisfied, the reason can be one of them: he/she can be from a single-parent background, being bullied, unemployment, drug addicted, family harnessment, being fraud, having the abortion experience, he/she or his/her parent is drug-addicted, suffering from cancers, shooting porns due to the growing backround or some other miseries or the combination of these factors. The reasons should be various.\n"
        # condition = f"The attitude of Life: Fairly Satisfied. His/her family background is tough like from the from a single-parent background, family harnessment, or some other miseries or the combination of these factors. The reasons should be various. Luckily, he/she comes over it and still strive for his/her dreams\n"
        response = assistant.generate_personality(prompt, condition)
        if isinstance(response, str):
            cleaned_response = response.replace('```json', '').replace('```', '').strip()
            print("cleaned")
        else:
            cleaned_response = "None"
            print("str")
        print(cleaned_response)
        # response_list.append(cleaned_response)
        try:
            cleaned_response = json.loads(cleaned_response)
            response_list.append(cleaned_response)
        except:
            response_list.append("None")
            # response = assistant.generate_personality(prompt)
            # response_list.append(cleaned_response)

    save_path = args.json_file
    last_index = 0
    new_combined_data = []
    if os.path.exists(save_path):
        with open(save_path, 'r') as json_file:
            existing_data = json.load(json_file)
            last_index = existing_data[-1]['idx']
    else:
        existing_data = []
        last_index = -1

    for i in range(len(response_list)):
        last_index += 1
        entry = {
            "idx": last_index,
            "features": response_list[i]

        }
        new_combined_data.append(entry)



    combined_data = existing_data + new_combined_data
    with open(save_path, 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)


if __name__ == '__main__':
    main()