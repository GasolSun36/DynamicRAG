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
