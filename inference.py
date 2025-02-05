import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import os
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel

parser = argparse.ArgumentParser(description="Process JSONL data using an LLM model.")
parser.add_argument("--template", type=str, required=True, help="Path to the chat template file.")
parser.add_argument("--llm-model", type=str, required=True, help="Path to the LLM model checkpoint.")
parser.add_argument("--input-jsonl", type=str, required=True, help="Path to the input JSONL file.")
parser.add_argument("--output-json", type=str, required=True, help="Path to the output JSON file.")
parser.add_argument("--remain-output-json", type=str, required=True, help="Path to the remain output JSONL file.")
args = parser.parse_args()



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
         'content': f"Query: {query}\n{retrieved_content}"}
    ]


def get_prompt_answer(input_file, prompt, entry):
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
    
    if 'fever' in input_file:
        remember_prompt = 'Please answer the question with “SUPPORTS”, “REFUTES” or “NEI” based on what you know.'
    elif 'nq' in input_file:
        remember_prompt = 'Please answer the question with a short phrase.'
    elif 'hotpotqa' in input_file:
        remember_prompt = 'Please answer the question with a short phrase.'
    elif 'eli5' in input_file:
        remember_prompt = 'Please answer the question with a paragraph.'
    elif 'triviaqa' in input_file:
        remember_prompt = 'Please answer the question with a short phrase.'
    elif '2wikimqa' in input_file:
        remember_prompt = 'Please answer the question with a short phrase.'
    elif 'arc' in input_file:
        remember_prompt = 'Please answer the following questions and directly output the answer options.'
    elif 'asqa' in input_file:
        remember_prompt = 'Please answer the question with a short phrase.'
    elif 'popqa' in input_file:
        remember_prompt = 'Please answer the question with a short phrase.'
    else:
        remember_prompt = 'Please answer the question with a short phrase.'

    return [
        {'role': 'system',
        'content': system_prompt},
        {'role': 'user',
        'content': (f"{remember_prompt}\n\n"
                    f"Query: {entry['question']}\n\n"
                    f"{prompt}")}
    ]
    

def generate_docs(llm, entry):
    messages = get_prompt_docs(entry['question'], entry['ctxs'])
    
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


def generate_answers(input_file, llm, entry, prompts):
    results = []
    for prompt in prompts:
        messages = get_prompt_answer(input_file, prompt, entry)
        
        outputs = llm.chat(messages,
                   sampling_params=sampling_params_generate,
                   use_tqdm=False,
                   chat_template=chat_template)
        results.append(outputs[0].outputs[0].text)
    return results

def process_jsonl(llm, input_file, output_file, remain_output_file):
    """
    Main processing pipeline to read input JSONL, process data, and write output JSON and remain JSONL.
    """
    result = []
    result_remain = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        datas = [json.loads(line.strip()) for line in infile]

    for entry in tqdm(datas, total=len(datas), desc="Processing"):
        if 'asqa' in input_file:
            entry['ctxs'] = entry['docs']
        # for 4k window, entry['ctxs'][:20]
        # for 8k window, entry['ctxs'][:40]
        entry['ctxs'] = entry['ctxs'][:40]
        

        ranking_string = generate_docs(llm, entry)
        id_list = map_function(ranking_string)

        try:
            prompt = convert(id_list, entry['ctxs'])
        except IndexError:
            result_remain.append(entry)
            continue

        answer = generate_answers(input_file, llm, entry, prompt)
        entry['response'] = answer
        entry['id_list'] = id_list
        result.append(entry)

    output_dir = os.path.dirname(output_file)
    remain_dir = os.path.dirname(remain_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if remain_dir:
        os.makedirs(remain_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=4)


    with open(remain_output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result_remain, outfile, indent=4)



if __name__ == "__main__":

    with open(args.template, "r") as f:
        chat_template = f.read()

    llm = LLM(model=args.llm_model, tensor_parallel_size=8)
    sampling_params = SamplingParams(temperature=0.4, max_tokens=100)
    sampling_params_generate = SamplingParams(temperature=0, max_tokens=200)
    process_jsonl(llm, args.input_jsonl, args.output_json, args.remain_output_json)
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory!")


