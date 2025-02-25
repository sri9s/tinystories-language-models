from datasets import load_dataset
import os

def download_tinystories():
    # Create data directory if it doesn't exist
    data_dir = "data/tinystories"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading TinyStories dataset...")
    
    # Load the dataset from Hugging Face
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Save train and validation splits
    print("Saving train split...")
    dataset['train'].to_json(os.path.join(data_dir, "train.json"))
    
    print("Saving validation split...")
    dataset['validation'].to_json(os.path.join(data_dir, "validation.json"))
    
    print(f"Dataset downloaded and saved to {data_dir}")
    print(f"Train size: {len(dataset['train'])}")
    print(f"Validation size: {len(dataset['validation'])}")

if __name__ == "__main__":
    download_tinystories() 