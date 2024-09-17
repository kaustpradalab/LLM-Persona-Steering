import time, sys, os, torch, argparse
import json

def fixed_features(attributes, details):
    if attributes is None:
        return
    
    feature_list = {}

    for attr_key, attr_value in attributes.items():
        if attr_key in details:
            sub_dict = details[attr_key]
            if attr_value in sub_dict:
                feature_value = sub_dict[attr_value]
                feature_list[attr_key] = list(feature_value.values())

    return feature_list


def generative_features(generative_bg):
    if generative_bg is None:
        return


def process_features(save_path, pattern_path, background, layer):
    with open(background, "r", encoding="utf-8") as file:
        data = json.load(file)

    with open(pattern_path, "r", encoding="utf-8") as file:
        pattern = json.load(file)

    for layer in pattern:
        if layer.get("Layer") == 12:
            details =  layer.get("Details", {})

    
    for idx, item in enumerate(data):
    # Add a new "features" key to the existing item
        item["features"] = {
            "fixed": fixed_features(item.get("attributes", {}), details),
            "generative": generative_features(item.get("generative_bg", {}))
        }

    # Save the modified data back to the file
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"features data is saved in {save_path}")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", help="The path to the JSON file to save.")
    parser.add_argument("--background", type=str, help="The path to the background JSON file.")
    parser.add_argument("--pattern_path", type=str,  help="The path to the features pattern JSON file.")
    parser.add_argument("--layer", type=int,  help="The path to the features pattern JSON file.")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    process_features(args.save_path, args.pattern_path, args.background, args.layer)

if __name__ == '__main__':
    main()