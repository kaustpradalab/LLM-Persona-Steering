import json
import time, json, sys, os, torch, argparse
import config


template = (
    "Imagine you are a {Gender}, aged {Age}, with a {Education level} education level. "
    "You come from a {Socioeconomic status} background and hold strong beliefs in {Social ideology}. "
    "You are known for being {Emotional intelligence}, showing dedication in your profession as someone who is {Professional commitment}. "
    "In life, you feel {Life satisfaction} with your circumstances. Your family relationships are {Family background}, which keeps you connected and supported."
)

def fixed_bg(features):
    return template.format(
        Gender=features.get("Gender", "unknown"),
        Age=features.get("Age", "unknown"),
        **{key: features.get(key, "unknown") for key in features if key != "Gender" and key != "Age"}
    )

def generative_bg(features):
    pass

def process_bg(json_file_path, argument):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item in data:
        features = item.get("features", {})
        if argument == "fixed":
            item["fixed_bg"] = fixed_bg(features)
        elif argument == "gen":
            item["generative_bg"] = generative_bg(features)  

    with open(json_file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"{argument}_bg finished generation。")

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