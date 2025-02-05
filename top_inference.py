import json
import re
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
import os
import gc
import torch
from vllm.distributed.parallel_state import destroy_model_parallel

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

def generate_docs(llm, query, docs, sampling_params, chat_template):
    messages = get_prompt_docs(query, docs)
    outputs = llm.chat(messages,
                       sampling_params=sampling_params,
                       use_tqdm=False,
                       chat_template=chat_template)
    return [output.text for output in outputs[0].outputs]

def generate_final_docs(llm, query, all_docs, max_context_window, sampling_params, chat_template):
    """
    Process all documents in batches, reduce to a manageable size (<= max_context_window), and return the final set of docs.
    """
    current_docs = []
    for i in range(0, len(all_docs), max_context_window):
        batch_docs = all_docs[i:i + max_context_window]
        ranking_strings = generate_docs(llm, query, batch_docs, sampling_params, chat_template)
        id_list = map_function(ranking_strings)
        for ids in id_list:
            for doc_id in ids:
                if all_docs[doc_id - 1] not in current_docs:
                    current_docs.append(all_docs[doc_id - 1])
        
    while len(current_docs) > max_context_window:
        reduced_docs = []
        for i in range(0, len(current_docs), max_context_window):
            batch_docs = current_docs[i:i + max_context_window]
            ranking_strings = generate_docs(llm, query, batch_docs, sampling_params, chat_template)
            id_list = map_function(ranking_strings)

            for ids in id_list:
                for doc_id in ids:
                    if doc_id - 1 < len(batch_docs):
                        reduced_docs.append(batch_docs[doc_id - 1])
        
        current_docs = list(set(reduced_docs))

        if len(current_docs) > max_context_window and len(reduced_docs) == 0:
            raise ValueError("Failed to reduce document count below the context window. Check the LLM output.")

    return current_docs

def generate_answers(llm, query, final_docs, sampling_params_generate, chat_template):
    prompt = convert([[i + 1 for i in range(len(final_docs))]], final_docs)[0]
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
    remember_prompt = 'Please answer the question with a short phrase.'
    messages = [
        {'role': 'system',
        'content': system_prompt},
        {'role': 'user',
        'content': (f"{remember_prompt}\n\n"
                    f"Query: {query}\n\n"
                    f"{prompt}")}
    ]
    outputs = llm.chat(messages,
                       sampling_params=sampling_params_generate,
                       use_tqdm=False,
                       chat_template=chat_template)
    return outputs[0].outputs[0].text

def process_jsonl(llm, input_file, output_file, remain_output_file, max_context_window):
    result = []
    result_remain = []
    with open(input_file, 'r', encoding='utf-8') as infile:
        datas = [json.loads(line.strip()) for line in infile]

    for entry in tqdm(datas, total=len(datas), desc="Processing"):
        try:
            entry['ctxs'] = entry['ctxs'][:args.topn]
            final_docs = generate_final_docs(llm, entry['question'], entry['ctxs'], max_context_window, sampling_params, chat_template)
            answer = generate_answers(llm, entry['question'], final_docs, sampling_params_generate, chat_template)
            entry['response'] = answer
            entry['final_docs'] = final_docs
            result.append(entry)
        except Exception as e:
            entry['error'] = str(e)
            result_remain.append(entry)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result, outfile, indent=4)

    with open(remain_output_file, 'w', encoding='utf-8') as outfile:
        json.dump(result_remain, outfile, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSONL data using an LLM model.")
    parser.add_argument("--template", type=str, required=True, help="Path to the chat template file.")
    parser.add_argument("--llm-model", type=str, required=True, help="Path to the LLM model checkpoint.")
    parser.add_argument("--input-jsonl", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to the output JSON file.")
    parser.add_argument("--remain-output-json", type=str, required=True, help="Path to the remain output JSONL file.")
    parser.add_argument("--max-context-window", type=int, default=40, help="Maximum number of documents per context window.")
    parser.add_argument("--topn", type=int, default=300, help="Maximum number of documents per context window.")
    args = parser.parse_args()

    with open(args.template, "r") as f:
        chat_template = f.read()

    llm = LLM(model=args.llm_model, tensor_parallel_size=8)
    sampling_params = SamplingParams(temperature=0.4, max_tokens=100)
    sampling_params_generate = SamplingParams(temperature=0, max_tokens=200)

    process_jsonl(llm, args.input_jsonl, args.output_json, args.remain_output_json, args.max_context_window)

    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully deleted the LLM pipeline and freed GPU memory!")
