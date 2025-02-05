import json
import re
import argparse
from vllm import LLM, SamplingParams
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Process JSON data using an LLM model.")
parser.add_argument("--template", type=str, required=True, help="Path to the chat template file.")
parser.add_argument("--llm-model", type=str, required=True, help="Path to the LLM model checkpoint.")
parser.add_argument("--input-json", type=str, required=True, help="Path to the input JSON file.")
parser.add_argument("--output-json", type=str, required=True, help="Path to the output JSON file.")

args = parser.parse_args()

with open(args.template, "r") as f:
    chat_template = f.read()

llm = LLM(model=args.llm_model, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.4, n=8, max_tokens=100)
sampling_params_generate = SamplingParams(temperature=0, max_tokens=200)

def get_prompt_docs(query, docs):
    retrieved_content = "Retrieved Content:\n" + "\n".join(
        [
            f"{i+1}. Topic: {doc['title']}\nContent: {doc['text']}"
            for i, doc in enumerate(docs)
        ]
    )
    return [
        {'role': 'system',
         'content': ("You are an expert at dynamically generating document identifiers to answer a given query.\n"
                     "I will provide you with a set of documents, each uniquely identified by a number within square brackets, e.g., [1], [2], etc.\n"
                     f"Your task is to identify and generate only the identifiers of the documents that contain sufficient information to answer the query.\n"
                     "Stop generating identifiers as soon as the selected documents collectively provide enough information to answer the query.\n"
                     "If no documents are required to answer the query, output \"None\".\n"
                     "Output the identifiers as a comma-separated list, e.g., [1], [2] or \"None\" if no documents are needed.\n"
                     "Focus solely on providing the identifiers. Do not include any explanations, descriptions, or additional text.")},
        {'role': 'user',
         'content': f"Query: {query}\n\n{retrieved_content}"}
    ]

def get_prompt_answer(prompt, entry):
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
    
    item_id = entry['id']
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

    return [
        {'role': 'system',
        'content': system_prompt},
        {'role': 'user',
        'content': (f"{remember_prompt}\n"
                    f"Query: {entry['question']}\n\n"
                    f"{prompt}")}
    ]

def generate_docs(entry):
    messages = get_prompt_docs(entry['question'], entry['docs'][:40])
    
    outputs = llm.chat(messages,
                   sampling_params=sampling_params,
                   use_tqdm=False,
                   chat_template=chat_template)
    
    return [output.text for output in outputs[0].outputs]

def map_function(data_list):
    result = []
    for item in data_list:
        ids = [int(num) for num in re.findall(r'\[(\d+)\]', item)]
        result.append(ids)
    return result

def convert(id_lists, docs):
    prompts = []
    for ids in id_lists:
        lines = []
        if len(ids) == 0:
            retrieved_content = "Retrieved Content: None"
            prompts.append(retrieved_content)
            continue
        for doc_id in ids:
            doc_title = docs[doc_id - 1]['title']
            doc_text = docs[doc_id - 1]['text']
            line = f"{doc_id}. Title: {doc_title}.\nContent: {doc_text}"
            lines.append(line)

        retrieved_content = "\n".join(lines)
        prompt = f"Retrieved Content:\n{retrieved_content}"
        prompts.append(prompt)
    return prompts

def generate_answers(entry, prompts):
    results = []
    for prompt in prompts:
        messages = get_prompt_answer(prompt, entry)
        
        outputs = llm.chat(messages,
                   sampling_params=sampling_params_generate,
                   use_tqdm=False,
                   chat_template=chat_template)
        results.append(outputs[0].outputs[0].text)
    return results

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    with open(output_file, 'a', encoding='utf-8') as outfile:
        for entry in tqdm(data):
            generated_list = generate_docs(entry)
            id_lists = map_function(generated_list)
            try:
                prompts = convert(id_lists, entry['docs'])
            except IndexError:
                continue
            answers = generate_answers(entry, prompts)
            
            entry['responses'] = answers
            entry['id_lists'] = id_lists
            import ipdb; ipdb.set_trace()
            json.dump(entry, outfile, ensure_ascii=False)
            outfile.write('\n')

if __name__ == "__main__":
    process_json(args.input_json, args.output_json)