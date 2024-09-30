import os
import json
from datasets import load_dataset

test_dataset = load_dataset("thu-coai/SafetyBench", "test")
dev_dataset = load_dataset("thu-coai/SafetyBench", "dev")

dir_path = './'
os.makedirs(dir_path, exist_ok=True)
test_dataset['zh'].to_json(f'{dir_path}/test_zh.json', batch_size=len(test_dataset['zh']), orient='records', force_ascii=False, lines=False, indent=4)
test_dataset['en'].to_json(f'{dir_path}/test_en.json', batch_size=len(test_dataset['en']), orient='records', force_ascii=False, lines=False, indent=4)

with open(f'{dir_path}/dev_zh.json', "w", encoding="utf-8") as outf:
    x = {k: v[0] for k, v in dev_dataset['zh'].to_dict().items()}
    json.dump(x, outf, ensure_ascii=False, indent=4)
    
with open(f'{dir_path}/dev_en.json', "w", encoding="utf-8") as outf:
    x = {k: v[0] for k, v in dev_dataset['en'].to_dict().items()}
    json.dump(x, outf, ensure_ascii=False, indent=4)
    