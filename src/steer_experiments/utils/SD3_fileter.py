import os
import json

# Define the relative path to the JSON file
relative_path = 'TRAIT.json'

# Get the absolute path
absolute_path = os.path.abspath(relative_path)

# Load the JSON file using the absolute path
with open(absolute_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Filter items based on the specified personality types
filtered_items = [item for item in data if item["personality"] in ["Psychopathy", "Machiavellianism", "Narcissism"]]

# Save the filtered items to a new JSON file using absolute path
output_path = os.path.abspath('trait/TRAIT_Dark.json')
with open(output_path, 'w', encoding='utf-8') as file:
    json.dump(filtered_items, file, indent=4, ensure_ascii=False)

print(f"Filtered items saved to: {output_path}")