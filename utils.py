from bert_score import score
from rouge import Rouge
import json
import jsonlines
import openai


openai.api_key = "Your_API_Key"

def compute_bert_score(reference: str, candidate: str):
    """
    Compute the BERTScore between two strings.
    
    Parameters:
        reference (str): The reference string.
        candidate (str): The candidate string to evaluate against the reference.

    Returns:
        float: The BERTScore (F1 score) between the reference and candidate.
    """
    if candidate == "" or reference == "":
        return 0
    try:
        precision, recall, f1 = score([candidate], [reference], lang="en", rescale_with_baseline=True)  # 
    except:
        return 0
    return f1[0].item()

def compute_rouge_score(rouge_score, reference: str, candidate: str):
    """
    Compute the ROUGE score between two strings.
    
    Parameters:
        reference (str): The reference string.
        candidate (str): The candidate string to evaluate against the reference.

    Returns:
        dict: A dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores (F1 scores).
    """
    if candidate == "" or reference == "":
        return 0
    try:
        scores = rouge_score.get_scores(candidate, reference, avg=True)
    except:
        return 0
    return scores["rouge-l"]["f"]

def compute_exact_match(reference: str, candidate: str):
    """
    Compute whether two strings are an exact match.

    Parameters:
        reference (str): The reference string.
        candidate (str): The candidate string to evaluate against the reference.

    Returns:
        bool: True if the strings are an exact match, False otherwise.
    """
    if candidate == "" or reference == "":
        return 0
    return reference.strip().lower() in candidate.strip().lower() or candidate.strip().lower() in reference.strip().lower() or reference.strip().lower() in candidate.strip().lower()

def inverse_reward(sentence):
    length = len(sentence)
    return 1 / (1 + length)

def llm_eval(instruction: str, reference: str, candidate: str):
    prompt = f"""Use the following criteria to evaluate the quality of the model's response in a knowledge-intensive task, considering the provided ground-truth answer. Assign a score between 0-100 based on the overall quality, relevance, and correctness of the response:

Relevance to the Prompt (20 points):

Award up to 20 points if the response aligns well with the user's query, even if minor errors are present.
Deduct points if the response lacks focus or deviates significantly from the query.
Accuracy of Factual Information (20 points):

Grant up to 20 points for correct factual details aligning with the ground-truth answer.
Penalize for inaccuracies, missing essential elements, or presenting incorrect knowledge.
Handling of Temporal and Logical Reasoning (20 points):

Award up to 20 points for demonstrating correct temporal and logical reasoning.
Deduct points if temporal reasoning is flawed or logical consistency is missing.
Clarity and Coherence of Response (20 points):

Assign up to 15 points for clear, coherent, and well-structured responses.
Reduce points for ambiguity, confusion, or poor organization.
Potential Misleading Nature or Misconceptions (20 points):

Award up to 10 points if the response avoids being misleading.
Penalize responses that could confuse or mislead the user, even if partially relevant.
After evaluating the response based on these criteria, provide a total score in the format:
“Score: points”.

User: {instruction}
Ground-Truth Answer: {reference}
Model Response: {candidate}

Score: """
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful Assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"].replace('.','').strip()

    

def load_jsonlines(file):
    with jsonlines.open(file, 'r') as jsonl_f:
        lst = [obj for obj in jsonl_f]
    return lst

def load_file(input_fp):
    if input_fp.endswith(".json"):
        input_data = json.load(open(input_fp))
    else:
        input_data = load_jsonlines(input_fp)
    return input_data

# Example usage
# reference = "Jane Austen"
# candidate = "The author of 'Pride and Prejudice' is Chris Paul."

# reference = "Eiffel Tower in Paris."
# candidate = "The Eiffel Tower is one of the most famous landmarks in Paris, and it is located in the Champ de Mars."
# candidate = "Eiffel Tower is in the Paris."

# # Compute BERTScore
# bert_score = compute_bert_score(reference, candidate)
# print(f"BERTScore: {bert_score}")

# # Compute ROUGE Score
# rouge_score = compute_rouge_score(reference, candidate)
# print(f"ROUGE Score: {rouge_score}")
