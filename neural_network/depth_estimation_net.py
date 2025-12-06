import torch
import torch.nn as nn
import torchvision.models as models

class SiameseStereoNet(nn.Module):
    """
    A Siamese Network for Depth Estimation from Stereo Image Pairs.
    This architecture uses a shared ResNet-18 backbone to extract features
    from both left and right images, constructs a cost volume by concatenating
    the features at the bottleneck, and decodes the fused features to predict
    a disparity map.
    """

    def __init__(self, dropout_p=0.0):
        super(SiameseStereoNet, self).__init__()
        self.dropout_p = dropout_p

        # Load a pre-trained ResNet-18 model, which already knows how to see edges/textures from ImageNet.
        base_model = models.resnet18(weights='DEFAULT')
        
        # Encoder Blocks

        # Encoder 1: Returns 64 channels at 1/2 resolution
        self.encoder_layer_1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
        )
        # Encoder 2: Returns 64 channels at 1/4 resolution
        self.encoder_layer_2 = nn.Sequential(
            base_model.maxpool,
            base_model.layer1
        )
        # Encoder 3: Returns 128 channels at 1/8 resolution
        self.encoder_layer_3 = base_model.layer2

        # Decoder Blocks (With Skip Connections)
        
        # Only add dropout if dropout_p > 0
        dropout_layers = []
        if dropout_p > 0:
            dropout_layers = [nn.Dropout2d(p=dropout_p)]
        
        # Decoder 1: Input (256 + 256 from fusion) -> Output 64
        decoder1_layers = [
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            decoder1_layers.append(nn.Dropout2d(p=dropout_p))
        decoder1_layers.extend([
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # Upsample to 1/4
            nn.ReLU(inplace=True)
        ])
        self.decoder_layer1 = nn.Sequential(*decoder1_layers)
        
        # Decoder 2: Input (64 from dec1 + 64 from skip layer 2) -> Output 64
        decoder2_layers = [
            nn.Conv2d(128, 64, kernel_size=3, padding=1), # Compress concatenation
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            decoder2_layers.append(nn.Dropout2d(p=dropout_p))
        decoder2_layers.extend([
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # Upsample to 1/2
            nn.ReLU(inplace=True)
        ])
        self.decoder_layer2 = nn.Sequential(*decoder2_layers)
        
        # Decoder 3: Input (64 from dec2 + 64 from skip layer 1) -> Output 32
        decoder3_layers = [
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0:
            decoder3_layers.append(nn.Dropout2d(p=dropout_p))
        decoder3_layers.extend([
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # Upsample to Full
            nn.ReLU(inplace=True)
        ])
        self.decoder_layer3 = nn.Sequential(*decoder3_layers)

        # Final Projection
        self.final = nn.Conv2d(32, 1, kernel_size=3, padding=1)

    def extract_features(self, x):
        # Pass x through encoder, saving intermediate outputs
        l1 = self.encoder_layer_1(x)  # 1/2 res
        l2 = self.encoder_layer_2(l1) # 1/4 res
        l3 = self.encoder_layer_3(l2) # 1/8 res
        return l1, l2, l3

    def forward(self, left, right):
        # Extract Features from both images
        l1_L, l2_L, l3_L = self.extract_features(left)
        l1_R, l2_R, l3_R = self.extract_features(right)

        # Cost Volume construction. We concatenate features from both images at the bottleneck (128 + 128 = 256 channels)
        # This allows the network to learn the relationship better than simple subtraction
        bottleneck = torch.cat((l3_L, l3_R), dim=1)

        # Decode with Skip Connections (Using LEFT image features for guidance). Upsample 1/8 -> 1/4
        x = self.decoder_layer1(bottleneck)
        
        # Concatenate with Layer 2 features from Left Image (Skip Connection). x is 64ch, l2_L is 64ch -> Total 128ch
        x = torch.cat((x, l2_L), dim=1)
        
        # Upsample 1/4 -> 1/2
        x = self.decoder_layer2(x)

        # Concatenate with Layer 1 features from Left Image (Skip Connection). x is 64ch, l1_L is 64ch -> Total 128ch
        x = torch.cat((x, l1_L), dim=1)
        
        # Upsample 1/2 -> Full
        x = self.decoder_layer3(x)

        # Final Prediction
        disparity = self.final(x)
        
        return torch.relu(disparity)