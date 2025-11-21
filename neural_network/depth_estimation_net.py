import torch
import torch.nn as nn
import torchvision.models as models

class SiameseStereoNet(nn.Module):
    """
    A Stereo Matching Network using a Siamese ResNet backbone.
    
    Structure:
      - Shared Encoder: ResNet-18 (Pre-trained) truncated at Layer 2.
      - Cost Volume: Absolute difference between Left and Right features.
      - Decoder: Upsamples features to predict pixel disparity.
    """

    def __init__(self):
        super(SiameseStereoNet, self).__init__()

        # Load a pre-trained ResNet-18 model, which alredy knows how to see edges/textures from ImageNet.
        base_model = models.resnet18(weights='DEFAULT')
        
        # Define the encoder (shared for both left/right images)
        # The output feature map will have 128 channels (after layer2)
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2 
        )
            
        # Define the decoder, upsampling 3 times to get back to input resolution
        self.decoder = nn.Sequential(
            # Upsample 1: 128 -> 64 channels
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Upsample 2: 64 -> 32 channels
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Upsample 3: 32 -> 16 channels
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Final Projection: 16 -> 1 channel (Disparity)
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, left, right):
        """
        Args:
            left: (B, 3, H, W) - The reference image
            right: (B, 3, H, W) - The secondary image
        
        Returns:
            disparity: (B, 1, H, W) - Pixel shift values
        """
        
        # Step 1: Extract the features using the shared encoder for both images
        f_left = self.encoder(left)
        f_right = self.encoder(right)

        # Step 2: Calculate the cost volume as an absolute difference between the left and right features
        # This explicitly captures the disparity information between the two views
        cost_volume = torch.abs(f_left - f_right)

        # Step 3: Decode the cost volume to predict the disparity map
        disparity = self.decoder(cost_volume)

        # Step 4: Post-process the data, ensuring disparity is non-negative by using the ReLU activation
        return torch.relu(disparity)