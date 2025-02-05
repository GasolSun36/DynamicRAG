import json
from tqdm import tqdm


input_file = 'training_data/retrieval_data_rerank_sequence.jsonl'
output_file = 'training_data/reranker_bc_training.json'


data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line.strip()))


new_data = []

for item in tqdm(data, desc="Processing items"):
    sequence = item.get("sequence", [])
    docs = item.get("docs", [])
    question = item.get("question", "")

    query = question

    system_prompt = (
        f"You are an expert at dynamically generating document identifiers to answer a given query.\n"
        f"I will provide you with a set of documents, each uniquely identified by a number within square brackets, e.g., [1], [2], etc.\n"
        f"Your task is to identify and generate only the identifiers of the documents that contain sufficient information to answer the query.\n"
        f"Stop generating identifiers as soon as the selected documents collectively provide enough information to answer the query.\n"
        f"If no documents are required to answer the query, output \"None\".\n"
        f"Output the identifiers as a comma-separated list, e.g., [1], [2] or \"None\" if no documents are needed.\n"
        f"Focus solely on providing the identifiers. Do not include any explanations, descriptions, or additional text."
    )

    if not sequence:
        output = "None"
    else:
        output = ", ".join(f"[{seq}]" for seq in sequence)

    retrieved_content = "Retrieved Content:\n" + "\n".join(
        [
            f"{i+1}. Topic: {doc['title']}\nContent: {doc['text']}"
            for i, doc in enumerate(docs)
        ]
    )
    instruction_data = {
        "instruction": f"Query: {question}\n\n{retrieved_content}",
        "input": "",
        "output": output,
        "system": system_prompt
    }
    
    new_data.append(instruction_data)


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
