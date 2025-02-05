import argparse
from metrics import metric_max_over_ground_truths, exact_match_score, match, rouge, f1
import json
from tqdm import tqdm
from rouge import Rouge

def load_file(file_path):
    """
    Load data from a JSON or JSONL file.

    Args:
        file_path (str): Path to the file to load.

    Returns:
        list: List of dictionaries loaded from the file.

    Raises:
        ValueError: If the file format is not supported.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.json'):
                data = json.load(f)
                if isinstance(data, dict):
                    return [data]
                elif isinstance(data, list):
                    return data
                else:
                    raise ValueError("Unsupported JSON structure. Expecting list or dict.")
            elif file_path.endswith('.jsonl'):
                data = [json.loads(line.strip()) for line in f if line.strip()]
                return data
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        raise


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_file", type=str, help="File containing results with both generated responses and ground truth answers")
    parser.add_argument("--metric", type=str, choices=["em", "accuracy", "match", "rouge", "f1"], help="Metric to use for evaluation")
    return parser.parse_args()

def main():
    args = get_args()

    results = load_file(args.results_file)
    
    scores = []
    cnt=0
    rouge_score = Rouge()
    for item in tqdm(results):
        response = item['response'][0]

        question = item['question']

        if 'asqa' in args.results_file:
            answers=  []
            for ans in item['qa_pairs']:
                answers.extend(ans['short_answers'])
        else:
            answers = item['answers']
        if not answers:
            print(f"Warning: No answers provided for ID {item.get('id', 'unknown')}")
            continue

        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(
                exact_match_score, response, answers
            )
        elif args.metric == "accuracy":
            response =  response.replace('\n','').strip()[0]
            answer = answers[0][0]
            if response==answer:
                metric_result = 1.0
            else:
                metric_result = 0.0
        elif args.metric == "match":
            metric_result = match(question, response, answers)
        elif args.metric == "rouge":
            metric_result = rouge(rouge_score, response, answers)
        elif args.metric == "f1":
            metric_result = f1(response, answers)
        else:
            raise NotImplementedError(f"Metric {args.metric} is not implemented.")
        
        scores.append(metric_result)
    
    if scores:
        print(f'Overall result: {sum(scores) / len(scores)}')
    else:
        print("No scores were calculated. Please check your input file.")

if __name__ == "__main__":
    main()
