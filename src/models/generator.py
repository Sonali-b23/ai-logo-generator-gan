import torch
import torch.nn as nn
import sys
import os

# Ensure src is in path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

class Generator(nn.Module):
    def __init__(self, z_dim=config.Z_DIM, text_embed_dim=config.TEXT_EMBED_DIM, features_g=config.GENERATOR_FEATURES, img_channels=config.CHANNELS):
        """
        The Generator for our Conditional WGAN-GP.
        Takes random noise and text embeddings and generates a 64x64 RGB logo.
        """
        super(Generator, self).__init__()
        
        # When we combine noise + text, our starting vector size is z_dim + text_embed_dim
        combined_dim = z_dim + text_embed_dim

        # We first project the flat combined vector into a 4x4 spatial representation
        # 4x4 is a good starting point to upscale to 64x64
        self.initial_layer = nn.Sequential(
            nn.Linear(combined_dim, features_g * 8 * 4 * 4), 
            nn.ReLU(True)
        )
        
        # Now we use 'Fractionally-strided convolutions' (ConvTranspose2d) to upscale the image
        self.net = nn.Sequential(
            # Input: features_g * 8 x 4 x 4
            self._block(features_g * 8, features_g * 4, 4, 2, 1), # -> 8x8
            self._block(features_g * 4, features_g * 2, 4, 2, 1), # -> 16x16
            self._block(features_g * 2, features_g, 4, 2, 1),     # -> 32x32
            
            # Final Layer: Output is 3 channels (RGB). We use Tanh to bound outputs between [-1, 1]
            nn.ConvTranspose2d(features_g, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # -> 64x64
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function to create a standard Generator block: Upsample -> BatchNorm -> ReLU
        NOTE: WGAN-GP Generators can safely use BatchNorm (unlike the Critic).
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # BatchNorm normalizes the outputs to keep training stable
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, noise, text_embedding):
        """
        Forward pass for the generator.
        noise shape: (Batch Size, Z_DIM)
        text_embedding shape: (Batch Size, TEXT_EMBED_DIM)
        """
        # 1. Concatenate noise and text along the feature dimension (dim=1)
        x = torch.cat([noise, text_embedding], dim=1)
        
        # 2. Push through initial linear layer
        x = self.initial_layer(x)
        
        # 3. Reshape into an image-like tensor (Batch, Channels, Height, Width)
        x = x.view(x.size(0), -1, 4, 4)
        
        # 4. Upsample to 64x64
        return self.net(x)

if __name__ == "__main__":
    # Test the Generator architecture
    print("Testing Generator...")
    gen = Generator()
    test_noise = torch.randn((1, config.Z_DIM))
    test_text = torch.randn((1, config.TEXT_EMBED_DIM))
    output = gen(test_noise, test_text)
    print(f"Generator output shape (Should be 1, 3, 64, 64): {output.shape}")
