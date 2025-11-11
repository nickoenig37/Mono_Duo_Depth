import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthEstimationNet(nn.Module):
    """
    A convolutional neural network that estimates the depth map of the center image
    using stereo context from the left, center, and right images.

    Structure:
      - Shared encoder for all three inputs (extracts visual features)
      - Fusion module to combine left/center/right encoded features
      - Decoder to reconstruct the full-resolution depth map
    """

    def __init__(self):
        super(DepthEstimationNet, self).__init__()

        # ----------------------------------------------------
        # ENCODER
        # ----------------------------------------------------
        # Extract spatial features from each image.
        # Shared across left, center, and right inputs.
        self.encoder = nn.Sequential(
            # Input: (3, H, W) → (32, H/2, W/2)
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),

            # → (64, H/4, W/4)
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # → (128, H/8, W/8)
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------
        # Combine encoded feature maps from left, center, and right views.
        self.fusion = nn.Sequential(
            nn.Conv2d(128 * 3, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------
        # Reconstruct depth map from fused representation.
        self.decoder = nn.Sequential(
            # (128, H/8, W/8) → (64, H/4, W/4)
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # (64, H/4, W/4) → (32, H/2, W/2)
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # (32, H/2, W/2) → (16, H, W)
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Final depth output (1, H, W)
            nn.Conv2d(16, 1, 3, padding=1)
        )

    def forward(self, left, center, right):
        """
        Forward pass through the network.

        Args:
            left, center, right: input stereo images (B, 3, H, W)

        Returns:
            depth: predicted depth map for the center image (B, 1, H, W)
        """
        # Shared encoder
        f_left = self.encoder(left)
        f_center = self.encoder(center)
        f_right = self.encoder(right)

        # Fuse encoded features
        fused = torch.cat([f_left, f_center, f_right], dim=1)
        fused = self.fusion(fused)

        # Decode to spatial depth map
        depth = self.decoder(fused)

        # Keep depth predictions in [0, 1]
        depth = torch.sigmoid(depth)

        return depth
