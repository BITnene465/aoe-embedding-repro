"""Script to download models for AoE reproduction."""

import argparse
import os
from transformers import AutoModel, AutoTokenizer

def download_model(model_name: str, output_dir: str):
    """Download tokenizer and model and save to disk."""
    save_path = os.path.join(output_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    print(f"Downloading {model_name} to {save_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(save_path)

    model = AutoModel.from_pretrained(model_name)
    model.save_pretrained(save_path)

    print(f"Model {model_name} saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download models for AoE")
    parser.add_argument("--model_name", default="bert-base-uncased", help="Model name to download")
    parser.add_argument("--output_dir", default="models", help="Directory to save models")
    args = parser.parse_args()

    download_model(args.model_name, args.output_dir)
