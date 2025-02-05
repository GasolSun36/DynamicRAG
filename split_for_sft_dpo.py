import json
import random

def shuffle_and_split_jsonl_to_json(input_file, n, output_file_1, output_file_2):
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    print(len(data))
    if not isinstance(data, list):
        raise ValueError("The input JSONL must contain a list of items.")

    random.shuffle(data)

    data_part_1 = data[:n]
    data_part_2 = data[n:]

    with open(output_file_1, 'w', encoding='utf-8') as f:
        json.dump(data_part_1, f, ensure_ascii=False, indent=4)

    with open(output_file_2, 'w', encoding='utf-8') as f:
        json.dump(data_part_2, f, ensure_ascii=False, indent=4)

    print(f"Shuffled and split data saved to {output_file_1} and {output_file_2}")

input_file = 'training_data/retrieval_data_rerank_normal.jsonl'
n = 70000                    
output_file_1 = 'training_data/training_data_sft.json'
output_file_2 = 'training_data/training_data_dpo.json'

shuffle_and_split_jsonl_to_json(input_file, n, output_file_1, output_file_2)
