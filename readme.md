# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation

<div style='display:flex; gap: 0.25rem; flex-wrap: wrap; align-items: center;'>
  <a href='LICENCE'>
    <img src='https://img.shields.io/badge/License-Apache%202.0-g.svg'>
  </a>
  <a href='https://arxiv.org/abs/2505.07233'>
    <img src='https://img.shields.io/badge/Paper-PDF-red'>
  </a>
  <a href='https://x.com/SunJiashuo36/status/1922117916788404665'>
    <img src='https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40Us'>
  </a>
   <a href='https://huggingface.co/datasets/gasolsun/DynamicRAG_Training_Data_150k'>
  <img src='https://img.shields.io/badge/🤗-Training_Data-9C276A.svg' alt='training_data'>
   </a>
  <a href='https://huggingface.co/datasets/gasolsun/DynamicRAG-Eval-Data'>
  <img src='https://img.shields.io/badge/🤗-Eval_Data-9C276A.svg' alt='eval_data'>
   </a>
   <a href='https://huggingface.co/gasolsun/DynamicRAG-7B'>
     <img src='https://img.shields.io/badge/🤗-DynamicRAG--7B-FFD21E.svg' alt='model'>
   </a>
   <a href='https://huggingface.co/gasolsun/DynamicRAG-8B'>
     <img src='https://img.shields.io/badge/🤗-DynamicRAG--8B-FFD21E.svg' alt='model'>
   </a>
</div>

**DynamicRAG** is an innovative framework for Retrieval-Augmented Generation (RAG) that dynamically adjusts both the **order** and **number** of retrieved documents per query. A reinforcement learning (RL) agent serves as the reranker, optimizing document retrieval based on feedback from a **Large Language Model (LLM)**. The training process is divided into two main stages:

1. **Supervised Fine-Tuning (SFT) via Behavior Cloning**:  
   - Trains the reranker with expert trajectories.
   - Simplifies the action space and establishes a baseline.
2. **Reinforcement Learning (RL) with LLM Feedback**:  
   - Uses interactive feedback from the generator.
   - Explores improved trajectories and further optimizes the reranker.

## How to cite
If you extend or use this work, please cite the [paper](https://arxiv.org/abs/2212.07249) where it was introduced:

```
@misc{sun2025dynamicragleveragingoutputslarge,
      title={DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation}, 
      author={Jiashuo Sun and Xianrui Zhong and Sizhe Zhou and Jiawei Han},
      year={2025},
      eprint={2505.07233},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.07233}, 
}
```

## 🔥 Update
* [2025-09-18]: 🚀 Our paper is accepted by NeurIPS 2025! See you in San Diego! 🥳🥳🥳
* [2025-07-13]: 🚀 We release the training data of DynamicRAG: [DynamicRAG_Training_Data_150k](https://huggingface.co/datasets/gasolsun/DynamicRAG_Training_Data_150k)
* [2025-05-13]: 🚀 We release the paper: [https://arxiv.org/abs/2505.07233](https://arxiv.org/abs/2505.07233)
* [2025-05-07]: 🚀 We release the [DynamicRAG-7B](https://huggingface.co/gasolsun/DynamicRAG-7B) and [DynamicRAG-8B](https://huggingface.co/gasolsun/DynamicRAG-8B) and [eval-datas](https://huggingface.co/datasets/gasolsun/DynamicRAG-Eval-Data).
* [2025-05-05]: 🚀 We release the code for training and evaluation.



## Table of Contents

- [DynamicRAG Overview](#dynamicrag-overview)
- [Project Visualizations](#project-visualizations)
- [📌 Data Processing Pipeline](#-data-processing-pipeline)
- [🎯 Supervised Fine-Tuning (SFT) Training](#-supervised-fine-tuning-sft-training)
- [🤖 Interactive Data Collection](#-interactive-data-collection)
- [📈 Direct Preference Optimization (DPO) Training](#-direct-preference-optimization-dpo-training)
- [🔍 Inference and Evaluation](#-inference-and-evaluation)
- [📄 Licensing and Claims](#-licensing-and-claims)

---

## DynamicRAG Overview

DynamicRAG adjusts the retrieval process on-the-fly by:
- Dynamically reordering and selecting the number of documents per query.
- Leveraging a reranker trained with RL and LLM feedback to improve retrieval quality.

---

## 💡 Preliminaries
You should install the enviroment by `pip install -r requirements.txt`, and running:
```python
apt-get update
apt-get install libtiff5
```
Moreover, you need to config the retriever corpus, e.g. official 2018 English Wikipedia embeddings. We use the exact same config with [Self-RAG](https://github.com/AkariAsai/self-rag). You can read their Retriever Setup.


## 📌 Data Processing Pipeline
Example: Training LLaMA3-8B with Top-40 Documents

### **1. Prepare BC Data Pipeline**
#### **Step 1: Retrieve Top-40 Documents**
Run the retrieval script:
```bash
#!/bin/bash

NUM_GPUS=8
INPUT_FILE="data/rag_training_data.json"
SPLIT_DIR="data/splits"

python split_data.py --input_file $INPUT_FILE --output_dir $SPLIT_DIR --num_splits $NUM_GPUS

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_FILE="${SPLIT_DIR}/split_${GPU_ID}.json"
    OUTPUT_FILE="output/retrieval_split_${GPU_ID}.json"
    log_file="logs/retriever_split_${GPU_ID}.log"
    CUDA_VISIBLE_DEVICES=$GPU_ID python retriever.py \
        --model_name_or_path models/retriever \
        --passages data/psgs_w100.tsv \
        --passages_embeddings "data/wikipedia_embeddings/*" \
        --query $SPLIT_FILE \
        --output_dir $OUTPUT_FILE \
        --n_docs 50 \
        1>"$log_file" 2>&1 &

    echo "Started process on GPU $GPU_ID with input $SPLIT_FILE"
done

wait
echo "All processes completed."

```

#### **Step 2: Aggregate Retrieved Data**
```bash
python aggregate.py
```

#### **Step 3: Rerank Documents**
```bash
python reranker.py --model_name_or_path models/reranker/monot5 \
    --input_file output/retrieval_data.jsonl \
    --output_file output/retrieval_data_rerank.jsonl \
    --device cuda
```
Outputs: `retrieval_data_rerank.jsonl`

> 💡 If you running above command slowly, consider running it with multi-gpus like retriever and then combine the results.


#### **Step 4: Compute True/False in Reranking**
```bash
python process_training_data.py
```
Outputs:
- `retrieval_data_rerank_sequence.json` (for Reranker BC training)
- `retrieval_data_rerank_normal.json` (for SFT & DPO training)

#### **Step 5: Convert Reranker Data for Training**
```bash
python reranker_sequence.py
```
Output: `reranker_bc_data.json` (formatted for **LLaMA-Factory**)

#### **Step 6: Split SFT & DPO Data**
```bash
python split_for_sft_dpo.py
```

#### **Step 7: Construct Generator SFT Data**
```bash
python construct_generator_sft.py
```

---

## 🎯 Supervised Fine-Tuning (SFT) Training
We use **LLaMA-Factory** as the training framework. Install it from [here](https://github.com/hiyouga/LLaMA-Factory).

### **1. Configure `dataset_info.json`**
Modify `LLaMA-Factory/data/dataset_info.json`:
```json
{
  "generator_sft": {
    "file_name": "generator_sft_training.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
  },

  "reranker_bc": {
    "file_name": "reranker_bc_training.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
  },

  "alpaca_data": {
    "file_name": "alpaca_data_cleaned_system.json",
    "columns": {"prompt": "instruction", "query": "input", "response": "output", "system": "system"}
  }
}
```

### **2. Train the Model**
Modify `llama8b.yaml` and run:
```bash
llamafactory-cli train examples/train_full/llama8b.yaml
```
> 🛠️ Requires at least **8 A100-80G GPUs**.

---

## 🤖 Interactive Data Collection
We use **vLLM** for faster sampling.

### **1. Sample Interaction Trajectories**
```bash
python sampling_dpo_trajectories.py \
    --template template/llama3.jinja \
    --llm-model DynamicRAG_llama3_8b \
    --input-jsonl training_data/training_data_dpo.jsonl \
    --output-json results/training_data_dpo_sampling.json

```

### **2. Collect Rewards for Trajectories**
```bash
ython reward_trajectories.py \
    --input_file results/training_data_dpo_sampling.json \
    --output_file training_data/llama3_8b_output_dpo.jsonl \
```

### **3. Construct DPO Training Data**
```bash
python construct_dpo.py
```

---

## 📈 Direct Preference Optimization (DPO) Training

### **1. Configure `dataset_info.json`**
```json
{
  "llama3_generator_dpo": {
    "file_name": "llama3_8b_generator_dpo.json",
    "ranking": true,
    "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected"}
  },
  
  "llama3_reranker_dpo": {
    "file_name": "llama3_8b_reranker_dpo.json",
    "ranking": true,
    "columns": {"prompt": "instruction", "query": "input", "chosen": "chosen", "rejected": "rejected"}
  }
}
```

### **2. Train the Model**
```bash
llamafactory-cli train examples/train_full/llama8b_dpo.yaml
```
> 🛠️ Requires at least **8 A100-80G GPUs**.

---

## 🔍 Inference and Evaluation
We use **vLLM** for efficient inference.

### **1. Run Inference**
```bash
#!/bin/bash


LOG_DIR="eval_logs"
mkdir -p $LOG_DIR

run_inference() {
  local input_file=$1
  local output_file=$2
  local remain_output_file=$3

  echo "Running inference for $input_file..."
  python inference.py \
    --template template/llama3.jinja \
    --llm-model DynamicRAG_llama3_8b \
    --input-json $input_file \
    --output-json $output_file \
    --remain-output-json $remain_output_file \
    >> $LOG_DIR/$(basename $output_file .json)_log.txt 2>&1

  sleep 5
}


run_inference "eval_data/triviaqa.jsonl" \
              "results/llama3_8b_triviaqa.json" \
              "results/llama3_8b_triviaqa_remain.json"

run_inference "eval_data/nq.jsonl" \
              "results/llama3_8b_nq.json" \
              "results/llama3_8b_nq_remain.json"

run_inference "eval_data/hotpotqa.jsonl" \
              "results/llama3_8b_hotpotqa.json" \
              "results/llama3_8b_hotpotqa_remain.json"

run_inference "eval_data/2wikimqa.jsonl" \
              "results/llama3_8b_2wikimqa.json" \
              "results/llama3_8b_2wikimqa_remain.json"

run_inference "eval_data/fever.jsonl" \
              "results/llama3_8b_fever.json" \
              "results/llama3_8b_fever_remain.json"

run_inference "eval_data/eli5.jsonl" \
              "results/llama3_8b_eli5.json" \
              "results/llama3_8b_eli5_remain.json"

run_inference "eval_data/asqa_eval_gtr_top100.jsonl" \
              "results/llama3_8b_asqa.json" \
              "results/llama3_8b_asqa_remain.json"

echo "All tasks completed. Logs are available in $LOG_DIR."

```
Evaluates **7 different benchmarks**.

### **2. Evaluate Performance**
```bash
# install nltk, rouge_score, spacy
# python -m spacy download en_core_web_sm

# for example, when we evaluate nq
python evaluate.py \
    --results_file results/llama3_8b_nq.json \
    --metric match
```

### **3. Run DynamicRAG on 500+ Documents**
```bash
#!/bin/bash

TEMPLATE="template/llama3.jinja"
LLM_MODEL="DynamicRAG_llama3_8b"
INPUT_JSONL="eval_data/nq_top500.jsonl"
MAX_CONTEXT_WINDOW=40

TOPN_VALUES=(50 100 150 200 300 500)

for TOPN in "${TOPN_VALUES[@]}"; do
    LOG_FILE="top_logs/llama3_8b_nq_top_${TOPN}.log"

    python top_inference.py \
        --template "$TEMPLATE" \
        --llm-model "$LLM_MODEL" \
        --input-jsonl "$INPUT_JSONL" \
        --output-json "results/llama3_8b_top_${TOPN}_nq.json" \
        --remain-output-json "results/llama3_8b_top_${TOPN}_nq_remain.json" \
        --max-context-window "$MAX_CONTEXT_WINDOW" \
        --topn "$TOPN" >> "$LOG_FILE" 2>&1

    sleep 3
done

```

---



## Project Visualizations

Explore the key components and performance of DynamicRAG through the following images:

- **Introduction of DynamicRAG:**  

<div style="text-align: center;">
  <a href="https://imgse.com/i/pEe80tx">
    <img src="https://s21.ax1x.com/2025/02/05/pEe80tx.png" alt="DynamicRAG Intro" style="width:600px; height:auto;" />
  </a>
</div>

  
- **Pipeline of DynamicRAG:**  

  <div style="text-align: center;">
  <a href="https://imgse.com/i/pEe86je">
    <img src="https://s21.ax1x.com/2025/02/05/pEe86je.png" alt="DynamicRAG Pipeline" style="width:1000px; height:auto;" />
  </a>
</div>

- **Generator Experiment:**  

<div style="text-align: center;">
  <a href="https://imgse.com/i/pEe8s1O">
    <img src="https://s21.ax1x.com/2025/02/05/pEe8s1O.png" alt="Generator Experiment" style="width:1000px; height:auto;" />
  </a>
</div>


- **Reranker Experiment:**  

<div style="text-align: center;">
  <a href="https://imgse.com/i/pEe8ycD">
    <img src="https://s21.ax1x.com/2025/02/05/pEe8ycD.png" alt="Reranker Experiment" style="width:1000px; height:auto;" />
  </a>
</div>


  
- **Efficiency of DynamicRAG:**  

<div style="text-align: center;">
  <a href="https://imgse.com/i/pEe8wA1">
    <img src="https://s21.ax1x.com/2025/02/05/pEe8wA1.png" alt="Efficiency" style="width:500px; height:auto;" />
  </a>
</div>

- **Case Study:**  

<div style="text-align: center;">
  <a href="https://imgse.com/i/pEe8r9K">
    <img src="https://s21.ax1x.com/2025/02/05/pEe8r9K.png" alt="Case Study 1" style="width:600px; height:auto;" />
  </a>
</div>
<div style="text-align: center;">
  <a href="https://imgse.com/i/pEe8Bh6">
    <img src="https://s21.ax1x.com/2025/02/05/pEe8Bh6.png" alt="Case Study 2" style="width:600px; height:auto;" />
  </a>
</div>


---


## 📄 Licensing and Claims
This project is licensed under the Apache 2.0 protocol. The project assumes no legal responsibility for any output generated by the models and will not be held liable for any damages resulting from the use of the provided resources and outputs.

