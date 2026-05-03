from datasets import load_dataset
import json
import os

os.makedirs("data/raw", exist_ok=True)

print("Downloading Rotten Tomatoes dataset...")
dataset = load_dataset("rotten_tomatoes", split="train")

output_path = "data/raw/rotten_tomatoes.json1"

with open(output_path, "w") as f:
    for row in dataset:
        f.write(json.dumps(row) + "\n")

print(f"Saved {len(dataset)} rows to {output_path}")