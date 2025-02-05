# install nltk, rouge_score, spacy
# python -m spacy download en_core_web_sm


python evaluate.py \
       --results_file results/llama3_8b_nq.json \
       --metric match