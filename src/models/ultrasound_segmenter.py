import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import sys
import os

# --- START: ADD THIS CODE BLOCK ---
# This block adds the project's root directory to the Python path.
# This allows us to import from the 'hiera' repository that we cloned.
# It makes the import system robust and independent of where you run the script from.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# --- END: ADD THIS CODE BLOCK ---

# Now we can import from the official libraries

from hiera.hiera import hiera_large_224

    # ... (the rest of your code remains unchanged)


# --- 1. Wrapper for Official Hiera Model (FINAL CORRECTED VERSION) ---
# --- 1. Wrapper for Official Hiera Model (FINAL CORRECTED VERSION) ---
# --- 1. Wrapper for Official Hiera Model (FINAL CORRECTED VERSION) ---
class HieraFeatureExtractor(nn.Module):
    """
    A final, correct wrapper for the official Hiera model.
    This version uses the model's built-in `return_intermediates=True`
    argument in the forward pass, which is the officially intended way
    to extract feature maps from each stage. This avoids all manual
    iteration and reshaping.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        # Load the complete Hiera model
        self.model = hiera_large_224(pretrained=pretrained)

        # Freeze all Hiera parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Performs a forward pass and returns the intermediate feature maps.
        """
        # Call the model's forward pass with the argument to get features.
        # It returns (final_output, list_of_features). We only need the list.
        _, features = self.model(x, return_intermediates=True)
        
        # The features are already in the correct (B, C, H, W) format.
        return features
    
# --- 2. Wrapper for Official DINOv2 Model (CORRECTED) ---
# --- 2. Wrapper for Official DINOv2 Model (CORRECTED AND MORE ROBUST) ---
class DINOv2Extractor(nn.Module):
    """
    A more robust wrapper for the official DINOv2 model.
    This version uses the `forward_features` method to get patch tokens
    and manually reshapes them to the correct spatial format (B, C, H, W).
    This is more explicit and avoids potential issues with the
    `get_intermediate_layers` helper function.
    """
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')

        # Freeze DINOv2 parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Use forward_features to get the dictionary of features
            features = self.model.forward_features(x)
            # The final patch tokens are in the 'x_norm_patchtokens' key
            patch_tokens = features['x_norm_patchtokens']
            
            # Manually reshape from (B, N, C) to (B, C, H, W)
            B, N, C = patch_tokens.shape
            H = W = int(N**0.5) # For 224x224 input, N=256, so H=W=16
            
            # Reshape to (B, H, W, C) and then permute to (B, C, H, W)
            spatial_features = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
            return spatial_features.contiguous() # Add .contiguous() for safety

# --- 3. Decoder Block (Unchanged) ---
class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip_feature.shape[2:]:
            x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip_feature], dim=1)
        return self.conv(x)


# --- 4. Main Ultrasound Segmenter Model (Integrated) ---
class UltrasoundSegmenter(nn.Module):
    def __init__(self, num_classes: int, decoder_channels: List[int] = [256, 128, 64]):
        super().__init__()
        
        self.hiera_encoder = HieraFeatureExtractor(pretrained=True)
        hiera_out_channels = [144, 288, 576, 1152]


        # Use the corrected DINOv2 Extractor
        self.dinov2_extractor = DINOv2Extractor()
        ddino_channels = 384
        
        self.dino_proj_channels = hiera_out_channels[0]
        self.dinov2_projection = nn.Conv2d(ddino_channels, self.dino_proj_channels, kernel_size=1)

        self.decoder_blocks = nn.ModuleList()
        
        in_ch = hiera_out_channels[-1] + self.dino_proj_channels
        skip_ch = hiera_out_channels[-2] + self.dino_proj_channels
        out_ch = decoder_channels[0]
        self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        in_ch = out_ch
        skip_ch = hiera_out_channels[-3] + self.dino_proj_channels
        out_ch = decoder_channels[1]
        self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        in_ch = out_ch
        skip_ch = hiera_out_channels[-4] + self.dino_proj_channels
        out_ch = decoder_channels[2]
        self.decoder_blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
        
        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def _interleave_features(self, hiera_feature: torch.Tensor, dinov2_feature_spatial: torch.Tensor) -> torch.Tensor:
        # Permute Hiera feature from (B, H, W, C) to (B, C, H, W)
        hiera_feature_permuted = hiera_feature.permute(0, 3, 1, 2)
        
        projected_dino_feature = self.dinov2_projection(dinov2_feature_spatial)
        
        resized_dino_feature = F.interpolate(
            projected_dino_feature, size=hiera_feature_permuted.shape[2:], mode='bilinear', align_corners=False
        )
        
        # Now both tensors are in the correct (B, C, H, W) format
        interleaved_feature = torch.cat([hiera_feature_permuted, resized_dino_feature], dim=1)
        
        return interleaved_feature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[2:]
        hiera_features = self.hiera_encoder(x)
        dinov2_feature_spatial = self.dinov2_extractor(x)
        

        interleaved_features = [self._interleave_features(f, dinov2_feature_spatial) for f in hiera_features]
        dec_feature = interleaved_features[-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_f = interleaved_features[-(i+2)]
            dec_feature = decoder_block(dec_feature, skip_f)
        logits_low_res = self.segmentation_head(dec_feature)
        final_logits = F.interpolate(logits_low_res, size=input_size, mode='bilinear', align_corners=False)
        return final_logits

# --- Test Script (Unchanged) ---
if __name__ == "__main__":
    if 'hiera' not in sys.modules:
        print("Could not find 'hiera' module. Please check your sys.path setup.")
    else:
        print("Successfully imported 'hiera' and 'dinov2' modules.")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        num_classes = 3
        print("\nInitializing UltrasoundSegmenter with real Hiera and DINOv2 backbones...")
        model = UltrasoundSegmenter(num_classes=num_classes).to(device)
        print("Model initialized successfully.")
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        print("\nPerforming a test forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
        print("Forward pass successful.")
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Final Output shape (logits): {output.shape}")
        assert output.shape == (dummy_input.shape[0], num_classes, dummy_input.shape[2], dummy_input.shape[3])
        print("\nModel test successful with correct output shape!")