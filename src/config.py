import os
import torch

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
IMG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")

# Ensure output directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
IMG_SIZE = 64 # Resizing logos to 64x64 or 128x128
CHANNELS = 3  # RGB

# WGAN-GP specific params
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

# Latent and Embedding dimensions
Z_DIM = 100 # Noise vector size
TEXT_EMBED_DIM = 512 # CLIP output vector size
GENERATOR_FEATURES = 64
CRITIC_FEATURES = 64

# Optimizer Hyperparameters
LEARNING_RATE = 1e-4

# Device config
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
