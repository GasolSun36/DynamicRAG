import os
import json
import argparse
import math

def split_data(input_file, output_dir, num_splits):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line.strip()) for line in f]

    total_items = len(data)
    items_per_split = math.ceil(total_items / num_splits)

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_splits):
        split_data = data[i * items_per_split:(i + 1) * items_per_split]
        split_file = os.path.join(output_dir, f"split_{i}.json")
        with open(split_file, 'w', encoding='utf-8') as out_f:
            json.dump(split_data, out_f, indent=4, ensure_ascii=False)

    print(f"Data split into {num_splits} parts and saved in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save splits")
    parser.add_argument("--num_splits", type=int, required=True, help="Number of splits")

    args = parser.parse_args()
    split_data(args.input_file, args.output_dir, args.num_splits)
