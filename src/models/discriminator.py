import torch
import torch.nn as nn
import sys
import os

# Ensure src is in path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src import config

class Critic(nn.Module):
    def __init__(self, text_embed_dim=config.TEXT_EMBED_DIM, img_channels=config.CHANNELS, features_d=config.CRITIC_FEATURES):
        """
        The Critic (Discriminator) for our Conditional WGAN-GP.
        Unlike a standard GAN that outputs [0, 1] probability, this Critic outputs an unconstrained real number.
        """
        super(Critic, self).__init__()
        
        # We process the text embedding to match the image dimensions so we can combine them.
        # We will project the text into a spatial map and concatenate it as a 4th channel.
        # So input channels = 3 (RGB Image) + 1 (Text Map) = 4
        self.text_projection = nn.Sequential(
            nn.Linear(text_embed_dim, 64 * 64),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        combined_channels = img_channels + 1

        self.net = nn.Sequential(
            # Input: combined_channels (4) x 64 x 64
            # First layer usually doesn't use InstanceNorm
            nn.Conv2d(combined_channels, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # -> 32x32
            self._block(features_d, features_d * 2, 4, 2, 1),
            # -> 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            # -> 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # -> 4x4
        )
        
        # Final layer to convert 4x4 spatial blocks into a single scalar value.
        self.final_flatten = nn.Flatten()
        self.final_linear = nn.Linear(features_d * 8 * 4 * 4, 1)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        Helper function for Critic block: Downsample -> InstanceNorm -> LeakyReLU
        CRITICAL FOR WGAN-GP: We MUST NOT use BatchNorm here because Gradient Penalty 
        punishes individual samples, and BatchNorm blends samples across the batch. 
        We use InstanceNorm2d instead.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            # InstanceNorm normalizes purely across the channels of a SINGLE image, ignoring the batch
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, image, text_embedding):
        """
        Forward pass.
        image shape: (Batch Size, 3, 64, 64)
        text_embedding shape: (Batch Size, TEXT_EMBED_DIM)
        """
        batch_size = image.size(0)
        
        # 1. Project text embedding to (Batch, 64*64)
        projected_text = self.text_projection(text_embedding)
        
        # 2. Reshape text to (Batch, 1, 64, 64) so it acts as an "image channel"
        text_channel = projected_text.view(batch_size, 1, config.IMG_SIZE, config.IMG_SIZE)
        
        # 3. Concatenate the text channel to the RGB image. 
        # Now the network "sees" both the image and the condition at the same time.
        x = torch.cat([image, text_channel], dim=1) # Shape: (Batch, 4, 64, 64)
        
        # 4. Push through convolutional layers
        x = self.net(x)
        
        # 5. Flatten and output a single score
        x = self.final_flatten(x)
        return self.final_linear(x)

if __name__ == "__main__":
    # Test the Critic architecture
    print("Testing Critic...")
    critic = Critic()
    test_image = torch.randn((1, 3, config.IMG_SIZE, config.IMG_SIZE))
    test_text = torch.randn((1, config.TEXT_EMBED_DIM))
    output = critic(test_image, test_text)
    print(f"Critic output shape (Should be 1, 1): {output.shape}")
