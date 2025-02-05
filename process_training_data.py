import json
import random
from collections import Counter
from tqdm import tqdm


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def save_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")


def process_jsonl(input_file, output_sequence_file, output_normal_file):
    result = []
    remaining_data = []
    length_counters = {length: 0 for length in range(15)}
    true_counts = []

    data = load_jsonl(input_file)
    unused_data = data.copy()
    total_items = len(data)
    
    with tqdm(total=total_items, desc="Processing", unit="item") as pbar:
        while sum(length_counters.values()) < 30000 and unused_data:
            item = unused_data.pop(0)
            used = False
            
            true_ids = []
            false_ids = []
            
            for idx, doc in enumerate(item['docs'], start=1):
                if doc.get('rerank_label') == "true":
                    true_ids.append(idx)
                else:
                    false_ids.append(idx)
            
            item['true_id'] = true_ids
            item['false_id'] = false_ids
            item['true_count'] = len(true_ids)
            true_counts.append(len(true_ids))
            
            for length in range(15):
                if length_counters[length] >= 2000:
                    continue
                
                if len(true_ids) > length:
                    continue
                
                shuffled_true_id = true_ids.copy()
                shuffled_false_id = false_ids.copy()
                random.shuffle(shuffled_true_id)
                random.shuffle(shuffled_false_id)
                
                sequence = shuffled_true_id
                remaining_length = length - len(shuffled_true_id)
                if remaining_length > 0:
                    sequence += shuffled_false_id[:remaining_length]
                
                new_item = item.copy()
                new_item["sequence_length"] = length
                new_item["sequence"] = sequence
                result.append(new_item)
                
                length_counters[length] += 1
                used = True
                
                if sum(length_counters.values()) >= 30000:
                    break
            
            if sum(length_counters.values()) >= 30000:
                break
            
            if not used:
                remaining_data.append(item)
            
            pbar.update(1)
        
        remaining_data.extend(unused_data)
    
    save_jsonl(result, output_sequence_file)
    save_jsonl(remaining_data, output_normal_file)
    print(f"Data generation complete. Total sequence items: {len(result)}")
    print(f"Remaining data items: {len(remaining_data)}")
    
    true_count_distribution = Counter(true_counts)
    print("True Count Distribution:")
    for count, freq in sorted(true_count_distribution.items()):
        print(f"True Count = {count}: Frequency = {freq}")

# 主流程
if __name__ == "__main__":
    input_jsonl_file = 'training_data/retrieval_data_rerank.jsonl'
    output_sequence_file = 'training_data/retrieval_data_rerank_sequence.jsonl'
    output_normal_file = 'training_data/retrieval_data_rerank_normal.jsonl'
    process_jsonl(input_jsonl_file, output_sequence_file, output_normal_file)
