import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# --- Add project root to path to find local 'models' and 'hiera' packages ---
# This makes the script runnable and the imports robust.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use absolute-style imports from the project root
from src.models.ultrasound_segmenter import DINOv2Extractor # Re-use our original 2D Extractor
from hiera.hiera import hiera_base_16x224

# --- 3D Decoder Block ---
# This component is purely 3D and is the building block for our decoder.
class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, skip_feature: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[2:] != skip_feature.shape[2:]:
            x = F.interpolate(x, size=skip_feature.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, skip_feature], dim=1)
        return self.conv(x)

# --- Main Hybrid 3D Segmenter Model ---
class UltrasoundSegmenterHybrid(nn.Module):
    def __init__(self, num_classes: int, clip_depth: int = 16):
        super().__init__()
        
        # --- 1. Hiera Encoder (3D Video Mode, Frozen) ---
        self.hiera_encoder = hiera_base_16x224(pretrained=True, in_chans=3) # starting with 3 channels even though the data is greyscale even though the model is trained on rgb
        for param in self.hiera_encoder.parameters():
            param.requires_grad = False
        self.hiera_encoder.eval()
        # NOTE: Hiera's video model has different channel outputs. These are the correct ones.
        # Freeze Hiera

	# Hiera-Base has a different channel structure than Hiera-Large.
	# These are the correct channel counts for Hiera-Base.
        hiera_out_channels = [96, 192, 384, 768]
        # --- 2. DINOv2 Encoder (2D Frame-wise Mode, Frozen) ---
        self.dinov2_extractor = DINOv2Extractor() # This model freezes itself internally
        dino_channels = 384

        # --- 3. Trainable Adapters ---
        self.dino_proj_channels = hiera_out_channels[0] # Project to match Hiera's first stage
        # The adapter is a 2D CONV because it processes the 2D frame features
        self.dinov2_projection = nn.Conv2d(dino_channels, self.dino_proj_channels, kernel_size=1)
        
        # --- 4. 3D Decoder (Trainable) ---
        self.decoder_blocks = nn.ModuleList()
        
        # Define decoder channel structure
        decoder_channels = [256, 128, 64]
        
        # Bottom-up construction of the 3D decoder
        in_ch = hiera_out_channels[-1] + self.dino_proj_channels
        skip_ch = hiera_out_channels[-2] + self.dino_proj_channels
        self.decoder_blocks.append(DecoderBlock3D(in_ch, skip_ch, decoder_channels[0]))
        
        in_ch = decoder_channels[0]
        skip_ch = hiera_out_channels[-3] + self.dino_proj_channels
        self.decoder_blocks.append(DecoderBlock3D(in_ch, skip_ch, decoder_channels[1]))

        in_ch = decoder_channels[1]
        skip_ch = hiera_out_channels[-4] + self.dino_proj_channels
        self.decoder_blocks.append(DecoderBlock3D(in_ch, skip_ch, decoder_channels[2]))

        # --- 5. 3D Segmentation Head (Trainable) ---
        self.segmentation_head = nn.Conv3d(decoder_channels[-1], num_classes, kernel_size=1)

    def _interleave_features(self, hiera_feature_3d: torch.Tensor, dino_features_2d_stack: torch.Tensor) -> torch.Tensor:
        B, D_d, C_d, H_d, W_d = dino_features_2d_stack.shape
        
        dino_features_flat = dino_features_2d_stack.reshape(B * D_d, C_d, H_d, W_d)
        projected_dino_flat = self.dinov2_projection(dino_features_flat)
        
        _, C_proj, H_proj, W_proj = projected_dino_flat.shape
        projected_dino_stack = projected_dino_flat.view(B, D_d, C_proj, H_proj, W_proj)
        
        projected_dino_3d = projected_dino_stack.permute(0, 2, 1, 3, 4)

        resized_dino_3d = F.interpolate(
            projected_dino_3d, size=hiera_feature_3d.shape[2:], mode='trilinear', align_corners=False
        )
        
        return torch.cat([hiera_feature_3d, resized_dino_3d], dim=1)

    def forward(self, x: torch.Tensor, return_intermediates: bool = False):
        B, C, D, H, W = x.shape
        
        hiera_input= x.repeat(1,3,1,1,1) #changing channel from grey to rgb by duplication
        # 1. Hiera Path (3D)
        # Hiera returns a tuple (final_output, list_of_features)
        _, raw_hiera_features_3d = self.hiera_encoder(hiera_input, return_intermediates=True)

        # Hiera's 3D output is (B, D, H, W, C). We must permute it to (B, C, D, H, W).
        hiera_features_3d = []
        for feat in raw_hiera_features_3d:
            # Permute from (B, D, H, W, C) -> (B, C, D, H, W)
            permuted_feat = feat.permute(0, 4, 1, 2, 3).contiguous()
            hiera_features_3d.append(permuted_feat)
        # --- END: THE CRITICAL FIX --
        # 2. DINOv2 Path (2D)
        # Reshape for 2D processing: (B, C, D, H, W) -> (B*D, C, H, W)
        dino_input = x.permute(0, 2, 1, 3, 4).reshape(B * D, C, H, W)
        # Repeat grayscale channel to 3 channels for pretrained DINOv2
        dino_input = dino_input.repeat(1, 3, 1, 1)
        
        dino_features_2d_flat = self.dinov2_extractor(dino_input)
        
        # Reshape back to a stack of 2D features: (B*D, C, H, W) -> (B, D, C, H, W)
        _, C_d, H_d, W_d = dino_features_2d_flat.shape
        dino_features_2d_stack = dino_features_2d_flat.view(B, D, C_d, H_d, W_d)

        # 3. Fusion and Decoding
        interleaved_features = [
            self._interleave_features(hiera_feat, dino_features_2d_stack) for hiera_feat in hiera_features_3d
        ]
        
        dec_feature = interleaved_features[-1]
        for i, decoder_block in enumerate(self.decoder_blocks):
            skip_f = interleaved_features[-(i+2)]
            dec_feature = decoder_block(dec_feature, skip_f)
            
        logits_low_res = self.segmentation_head(dec_feature)
        
        # 4. Final upsampling to original input clip size
        final_logits = F.interpolate(logits_low_res, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        if return_intermediates:
            return final_logits, hiera_features_3d

        return final_logits
