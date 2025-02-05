import json
from tqdm import tqdm


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_json(data, output_path):
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def transform_list_to_nested(input_list):
    """
    Transforms a list of integers into a string representation of single-element lists.

    Args:
        input_list (list): A list of integers.

    Returns:
        str: A string representation of single-element lists.
    """
    return ", ".join([f"[{item}]" for item in input_list])


def convert(id_lists, docs):
    if len(id_lists) == 0:
        retrieved_content = "Retrieved Content: None"
        prompt = retrieved_content
        return prompt
    lines=[]
    for doc_id in id_lists:
        doc_title = docs[doc_id - 1]['title']
        doc_text = docs[doc_id - 1]['text']
        line = f"{doc_id}. Title: {doc_title}.\nContent: {doc_text}"
        lines.append(line)

    retrieved_content = "\n".join(lines)
    prompt = f"Retrieved Content:\n{retrieved_content}"
    return prompt

def process_data(json_data):
    conversation_data_list = []
    conversation_data1_list = []

    for item in tqdm(json_data):
        item_id = item['id']
        response_scores = item.get("response_scores", [])
        responses = item.get("responses", [])
        id_lists = item.get("id_lists", [])
        docs = item.get("docs", [])
        question = item.get("question", "")

        system_prompt1 = (
            f"You are an expert at dynamically generating document identifiers to answer a given query.\n"
            f"I will provide you with a set of documents, each uniquely identified by a number within square brackets, e.g., [1], [2], etc.\n"
            f"Your task is to identify and generate only the identifiers of the documents that contain sufficient information to answer the query.\n"
            f"Stop generating identifiers as soon as the selected documents collectively provide enough information to answer the query.\n"
            f"If no documents are required to answer the query, output \"None\".\n"
            f"Output the identifiers as a comma-separated list, e.g., [1], [2] or \"None\" if no documents are needed.\n"
            f"Focus solely on providing the identifiers. Do not include any explanations, descriptions, or additional text."
        )

        system_prompt2 = (
            "You are an intelligent assistant that uses retrieved knowledge to answer user queries accurately and concisely. Follow these rules:\n\n"
            "1. **Task**:\n"
            "   - Use the provided `[Retrieved Content]` to generate responses.\n"
            "   - If the Retrieved Content is None, you should generate answer based on your own knowledge.\n"
            "   - If the information is insufficient or you don't know the answer, state, “I cannot fully answer based on the available information. Please provide more details.”\n\n"
            "2. **Requirements**:\n"
            "   - **Accuracy**: Base your answers on the retrieved content.\n"
            "   - **Conciseness**: Keep answers brief and relevant.\n"
            "   - **Context Awareness**: Ensure your responses align with the user’s query.\n\n"
            "3. **Input Format**:\n"
            "   - Query: `[User Query]`\n"
            "   - Retrieved: `[Retrieved Content]`\n\n"
            "4. **Output Format**:\n"
            "   - A structured, clear response tailored to the query.\n\n"
            "Always prioritize clarity and reliability."
        )
        if 'fever' in item_id:
            remember_prompt = 'Please answer the question with “SUPPORTS”, “REFUTES” or “NEI” based on what you know.'
        elif 'nq' in item_id:
            remember_prompt = 'Please answer the question with a short phrase.'
        elif 'hotpotqa' in item_id:
            remember_prompt = 'Please answer the question with a short phrase.'
        elif 'eli5' in item_id:
            remember_prompt = 'Please answer the question with a paragraph.'
        elif 'tc' in item_id:
            remember_prompt = 'Please answer the question with a short phrase.'
        elif 'asqa' in item_id:
            remember_prompt = 'Please answer the question with a short phrase.'
        else:
            remember_prompt = 'Please answer the question with a short phrase.'


        if not response_scores or not id_lists:
            continue

        if len(set(response_scores)) == 1:
            continue

        max_score = max(response_scores)
        min_score = min(response_scores)

        max_index = response_scores.index(max_score)
        min_index = response_scores.index(min_score)

        retrieved_content1 = "Retrieved Content:\n" + "\n".join(
            [
                f"{i+1}. Topic: {doc['title']}\nContent: {doc['text']}"
                for i, doc in enumerate(docs)
            ]
        )

        chosen_prompt = id_lists[max_index]
        rejected_prompt = id_lists[min_index]

        if not chosen_prompt:
            chosen_prompt_ids = "None"
        else:
            chosen_prompt_ids = ", ".join(f"[{seq}]" for seq in chosen_prompt)

        if not rejected_prompt:
            rejected_prompt_ids = "None"
        else:
            rejected_prompt_ids = ", ".join(f"[{seq}]" for seq in rejected_prompt)

        retrieved_content2 = convert(chosen_prompt, docs)
        instruction1 = f"Query: {question}\n\n{retrieved_content1}"
        instruction2 = f"{remember_prompt}\nQuery: {question}\n\n{retrieved_content2}"
        final_answer = item['answer'][0] if isinstance(item['answer'], list) else item['answer']

        for res in responses:
            if res != final_answer:
                dpo_reject_answer = res
                break
        conversation_data = {
            "instruction": system_prompt1,
            "input": instruction1,
            "chosen": chosen_prompt_ids,
            "rejected": rejected_prompt_ids
        }

        conversation_data1 = {
            "instruction": system_prompt2,
            "input": instruction2,
            "chosen": final_answer,
            "rejected": dpo_reject_answer
        }

        if len(conversation_data1_list) < 10000:
            conversation_data1_list.append(conversation_data1)
        else:
            conversation_data_list.append(conversation_data)

    return conversation_data_list, conversation_data1_list


if __name__ == "__main__":
    input_file = "training_data/llama3_8b_output_dpo.jsonl"
    output_file1 = "training_data/llama3_8b_reranker_dpo.json"
    output_file2 = "training_data/llama3_8b_generator_dpo.json"

    data = read_jsonl(input_file)
    conversation_data, conversation_data1 = process_data(data)

    save_json(conversation_data, output_file1)
    save_json(conversation_data1, output_file2)

    print(f"Processed data has been saved to {output_file1} and {output_file2}.")
