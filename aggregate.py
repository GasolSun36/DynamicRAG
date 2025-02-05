import json
import os
from tqdm import tqdm

input_files = [f"retrieval_split_{i}.json" for i in range(32)]
output_file = "retrieval_data.jsonl"


data = []
id_set = set()


for file in tqdm(input_files, desc="Processing files"):
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            record_id = record.get("id")
            if record_id not in id_set:
                id_set.add(record_id)
                data.append(record)


with open(output_file, 'w', encoding='utf-8') as f:
    for record in tqdm(data, desc="Writing to file"):
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

