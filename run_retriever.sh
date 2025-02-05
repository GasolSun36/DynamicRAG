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
