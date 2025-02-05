import json
import random


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def format_data(data):
    error_data = 0
    formatted_data = []
    
    system_prompt = """You are an intelligent assistant that uses retrieved knowledge to answer user queries accurately and concisely. Follow these rules:

1. **Task**:
   - Use the provided `[Retrieved Content]` to generate responses.
   - If the Retrieved Content is None, you should generate answer based on your own knowledge.
   - If the information is insufficient or you don't know the answer, state, “I cannot fully answer based on the available information. Please provide more details.”

2. **Requirements**:
   - **Accuracy**: Base your answers on the retrieved content.
   - **Conciseness**: Keep answers brief and relevant.
   - **Context Awareness**: Ensure your responses align with the user’s query.

3. **Input Format**:
   - Query: `[User Query]`
   - Retrieved: `[Retrieved Content]`

4. **Output Format**:
   - A structured, clear response tailored to the query.

Always prioritize clarity and reliability."""
    for item in data:

        item_id = item['id']

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

        true_ids = item.get('true_id', [])[:15]
        true_docs = [item['docs'][i - 1] for i in true_ids if i <= len(item['docs'])]
        
        if len(true_docs) == 0:
            retrieved_content = "Retrieved Content: None"
        else:
            retrieved_content = "Retrieved Content:\n" + "\n".join(
                [
                    f"{i+1}. Topic: {doc['title']}\nContent: {doc['text']}"
                    for i, doc in enumerate(true_docs)
                ]
            )

        if 'tc' in item_id:
            filtered_answers = [item['answer'].strip()]
        if type(item['answer']) == list:
            filtered_answers = [ans.strip() for ans in item.get('answer', []) if ans.strip()]
        elif type(item['answer']) == str:
            filtered_answers = [item['answer'].strip()]
        else:
            error_data += 1
            continue

        if filtered_answers:
            first_answer = filtered_answers[0]
        else:
            error_data += 1
            continue

        instruction_data = {
            "instruction": f"{remember_prompt}\nQuery: {item['question']}\n\n{retrieved_content}",
            "input": "",
            "output": first_answer,
            "system": system_prompt
        }
    
        formatted_data.append(instruction_data)

    print(f"Error Data Count: {error_data}")
    return formatted_data


def write_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def main(input_file, output_file):
    data = read_json(input_file)
    random.shuffle(data)
    formatted_data = format_data(data)
    print(len(formatted_data))
    write_json(output_file, formatted_data)


if __name__ == "__main__":
    input_file = 'training_data/training_data_sft.json'
    output_file = 'training_data/generator_sft_training.json'

    main(input_file, output_file)
