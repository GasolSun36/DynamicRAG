import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from typing import List
import argparse
from tqdm import tqdm


class Reranker:
    def __init__(self, model_name_or_path: str, device: str = "cuda"):
        """
        Initializes the Reranker with a specified model.

        Args:
            model_name_or_path (str): Path to the pretrained reranker model.
            device (str): Device to load the model on ("cuda" or "cpu").
        """
        self.device = device
        print(f"Loading reranker model from: {model_name_or_path}")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name_or_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Reranker model loaded successfully.")

    def rerank(self, query: str, docs: List[dict]) -> List[str]:
        """
        Rerank a list of documents based on their relevance to the query.

        Args:
            query (str): The input query.
            docs (List[dict]): A list of documents, each represented as a dictionary with keys "id" and "text".

        Returns:
            List[str]: A list of relevance labels ("true" or "false") for the documents.
        """
        labels = []
        inputs = []

        # Prepare inputs for the model
        for doc in docs:
            combined_input = f"Query: {query} Document: {doc['text']} Relevant:"
            inputs.append(combined_input)

        # Tokenize inputs in batches
        batch_size = 128  # Adjust based on available memory
        for start in range(0, len(inputs), batch_size):
            batch = inputs[start:start + batch_size]
            encoded_batch = self.tokenizer.batch_encode_plus(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            encoded_batch = {k: v.to(self.device) for k, v in encoded_batch.items()}

            # Generate labels
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=encoded_batch["input_ids"],
                    attention_mask=encoded_batch["attention_mask"],
                    max_length=2  # Only need a short output like "true" or "false"
                )

            # Decode generated sequences to labels
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            labels.extend([output.strip().lower() for output in decoded_outputs])

        return labels


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Reranker Demo")
    parser.add_argument("--model_name_or_path", type=str, default="",
                        help="Path to the pretrained reranker model.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the reranker on ('cuda' or 'cpu').")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output JSONL file.")
    args = parser.parse_args()

    # Load the reranker model
    reranker = Reranker(model_name_or_path=args.model_name_or_path, device=args.device)

    # Process JSONL file
    with open(args.input_file, "r", encoding='utf-8') as input_file, open(args.output_file, "a", encoding='utf-8') as output_file:
        for line in tqdm(input_file):
            if not line.strip():
                continue
            # Parse the JSON object
            example = json.loads(line.strip())
            query = example["question"]
            docs = example["docs"]

            # Get rerank labels
            rerank_labels = reranker.rerank(query=query, docs=docs)

            # Add rerank labels to each document
            for doc, label in zip(docs, rerank_labels):
                doc["rerank_label"] = label

            # Save the updated example to the output file as a JSONL entry
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Reranking completed and saved to {args.output_file}.")
