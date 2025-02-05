import json
import argparse
from tqdm import tqdm
from rouge import Rouge
from utils import compute_bert_score, compute_rouge_score, compute_exact_match, inverse_reward, llm_eval

def process_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    rouge_score = Rouge()

    weights = [0.2, 0.2, 0.2, 0.2, 0.2]

    for item in tqdm(data):
        instruction = item['question']
        answers = [ans for ans in item.get("answer", []) if ans.strip()]
        responses = item["responses"]
        id_list = item.get("id_lists", [])

        scores = []
        if len(set(responses)) == 1:
            scores = [0] * len(responses)
        else:
            for response in responses:
                response_scores = [
                    weights[0] * compute_bert_score(answer, response) +
                    weights[1] * compute_rouge_score(rouge_score, answer, response) +
                    weights[2] * compute_exact_match(answer, response) +
                    weights[3] * inverse_reward(id_list) + 
                    weights[4] * llm_eval(instruction, answer, response)
                    for answer in answers
                ]
                scores.append(sum(response_scores) / len(response_scores) if response_scores else 0)

        item["response_scores"] = scores

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Processed data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save processed JSON file")

    args = parser.parse_args()
    process_data(args.input_file, args.output_file)
