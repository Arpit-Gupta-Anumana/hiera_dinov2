import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import List, Tuple

class CAMUSDataset3D(Dataset):
    def __init__(self, dataset_root: str, split: str = "train", clip_depth: int = 16, transform=None):
        """
        3D-aware Dataset that yields clips of a fixed depth.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.clip_depth = clip_depth
        self.transform = transform # Note: Albumentations is mostly 2D, we will handle transforms manually
        
        split_suffix = "Tr" if self.split == "train" else "Ts"
        self.image_dir = os.path.join(dataset_root, f"images{split_suffix}")
        self.label_dir = os.path.join(dataset_root, f"labels{split_suffix}")
        
        self.samples = self._create_clip_samples()

        if not self.samples:
            raise ValueError(f"No valid {clip_depth}-slice clips found. Check dataset structure.")
        
        print(f"Found {len(self.samples)} possible {self.clip_depth}-slice clips for the '{split}' set.")

    def _create_clip_samples(self) -> List[Tuple[str, str, int]]:
        """
        Finds all valid starting points for clips of a given depth in the dataset.
        """
        clip_samples = []
        image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('_0000.nii.gz')])

        for image_fname in image_filenames:
            label_fname = image_fname.replace('_0000', '')
            image_path = os.path.join(self.image_dir, image_fname)
            label_path = os.path.join(self.label_dir, label_fname)

            if os.path.exists(label_path):
                try:
                    img_nifti = nib.load(image_path)
                    num_slices = img_nifti.shape[2] # Shape is (H, W, D)

                    if num_slices >= self.clip_depth:
                        for start_idx in range(num_slices - self.clip_depth + 1):
                            clip_samples.append((image_path, label_path, start_idx))
                except Exception as e:
                    print(f"Warning: Could not read {image_fname}. Skipping. Error: {e}")
        
        return clip_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, start_idx = self.samples[idx]

        image_volume = nib.load(image_path).get_fdata()
        mask_volume = nib.load(mask_path).get_fdata().astype(np.uint8)

        # Extract the 3D clip based on start index and depth
        image_clip = image_volume[:, :, start_idx : start_idx + self.clip_depth]
        mask_clip = mask_volume[:, :, start_idx : start_idx + self.clip_depth]
        
        # --- PREPROCESSING FOR 3D ---
        # Permute from (H, W, D) to a format suitable for PyTorch 3D Conv: (C, D, H, W)
        # 1. Add a channel dimension: (H, W, D) -> (1, H, W, D)
        image_clip = np.expand_dims(image_clip, axis=0)
        mask_clip = np.expand_dims(mask_clip, axis=0)
        
        # 2. Permute dimensions: (1, H, W, D) -> (1, D, H, W)
        image_clip = image_clip.transpose(0, 3, 1, 2)
        mask_clip = mask_clip.transpose(0, 3, 1, 2)
        
        # Convert to Tensor
        image_tensor = torch.from_numpy(image_clip.copy()).float()
        mask_tensor = torch.from_numpy(mask_clip.copy()).long()

        # --- MANUAL TRANSFORMS on Tensors ---
        # Resize the 3D clip to the model's expected input size
        image_tensor = torch.nn.functional.interpolate(
            image_tensor.unsqueeze(0), 
            size=(self.clip_depth, 224, 224), 
            mode='trilinear', 
            align_corners=False
        ).squeeze(0)

	# --- START: THE FIX ---
	# Resize the 3D mask clip to match the image size.
	# CRITICAL: Use 'nearest' mode for masks to preserve integer labels.
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor.unsqueeze(0).float(), # Interpolate needs float, then convert back
            size=(self.clip_depth, 224, 224), 
            mode='nearest'
        ).squeeze(0).long()
        # --- END: THE FIX ---

        # Normalize the image tensor
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        if max_val > min_val:
            image_tensor = (image_tensor - min_val) / (max_val - min_val)

        return image_tensor, mask_tensor
