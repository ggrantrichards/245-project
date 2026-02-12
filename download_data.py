import os
from datasets import load_dataset

def setup_local_data():
    # 1. Create directory
    os.makedirs("data", exist_ok=True)
    
    # 2. Download from HF
    print("Fetching Genteki/agentdojo_tiny...")
    dataset = load_dataset("Genteki/agentdojo_tiny", split="train")
    
    # 3. Save as JSON
    # We convert it to a list of dicts for easy local loading
    data_file = "data/agentdojo_tiny.json"
    dataset.to_json(data_file)
    print(f"Dataset saved to {data_file}")

if __name__ == "__main__":
    setup_local_data()