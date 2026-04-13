import os
import io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image
import numpy as np

# Import config parameters
import sys
# Temporary hack to ensure src is in path if run individually
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config

class LogoDataset(Dataset):
    def __init__(self, hf_dataset_name="logo-wizard/modern-logo-dataset", split="train"):
        """
        Custom PyTorch Dataset for loading logos and their text descriptions 
        from Hugging Face Datasets Hub.
        """
        print(f"Loading '{hf_dataset_name}' from Hugging Face...")
        # Note: the dataset name may vary, adjust if using a different one
        self.dataset = load_dataset(hf_dataset_name, split=split)
        
        # Transformations: Resize, Convert to Tensor, and Normalize for GAN (-1 to 1)
        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(), # Converts to [0, 1]
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS)], 
                [0.5 for _ in range(config.CHANNELS)]
            ) # Normalizes to [-1, 1]
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Extract Image
        img = item['image']
        
        # Ensure image is in RGB format (some logos might be grayscale or RGBA)
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
        else:
             # In case it's bytes or a dict representing image, handle properly
             pass # Assuming standard PIL image from datasets library for now
        
        # Apply transforms
        img_tensor = self.transform(img)
        
        # Extract text description
        # Using 'text' as standard, dataset keys may vary (could be 'caption', etc.)
        description = item['text'] 
        
        return img_tensor, description

def get_dataloader():
    """
    Returns a configured PyTorch DataLoader for the Logo dataset.
    """
    dataset = LogoDataset()
    dataloader = DataLoader(
        dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True
    )
    return dataloader

if __name__ == "__main__":
    # Test the dataloader
    print("Testing Logo Dataset loader...")
    try:
        loader = get_dataloader()
        img, text = next(iter(loader))
        print(f"Batch Image Shape: {img.shape}")
        print(f"Batch Text Length: {len(text)}")
        print(f"First text sample: {text[0]}")
        print("Data loader test successful!")
    except Exception as e:
        print(f"Error during dataloader test: {e}")
