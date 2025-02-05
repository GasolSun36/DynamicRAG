python reranker.py --model_name_or_path models/reranker/monot5 \
    --input_file output/retrieval_data.jsonl \
    --output_file output/retrieval_data_rerank.jsonl \
    --device cuda
