import json
import time, json, sys, os, torch, argparse
import config

def generate_keywords(json_file_path, num):
    if not os.path.exists(json_file_path):
        print(f"{json_file_path} doesn't exist，creating new file...")
        
        with open(json_file_path, "w", encoding="utf-8") as json_file:
            json.dump([], json_file, ensure_ascii=False, indent=4)
        
        print(f"created {json_file_path}.")
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", help="The path to the JSON file to process.")
    parser.add_argument("--num", type=int, default=5, help="The number of items to generate")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"python {' '.join(sys.argv)}")
    generate_keywords(args.json_file, args.num)

if __name__ == '__main__':
    main()