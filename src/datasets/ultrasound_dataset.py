import os
import torch
import numpy as np
import nibabel as nib # For NIfTI files
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple
import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import List, Tuple

import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from typing import List, Tuple

class CAMUSDataset(Dataset):
    def __init__(self, dataset_root: str, split: str = "train", transform=None):
        """
        Args:
            dataset_root (str): Path to the root of the dataset.
            split (str): 'train' or 'test'.
            transform (albumentations.Compose, optional): Transformations to apply.
        """
        self.dataset_root = dataset_root
        self.split = split
        self.transform = transform
        
        split_suffix = "Tr" if self.split == "train" else "Ts"
        self.image_dir = os.path.join(dataset_root, f"images{split_suffix}")
        self.label_dir = os.path.join(dataset_root, f"labels{split_suffix}")
        
        # The 'samples' list will now store tuples of (image_path, mask_path, slice_index)
        self.samples = self._load_and_unroll_samples()

        if not self.samples:
            raise ValueError(f"No valid, unrolled samples found. Check dataset structure and content.")
        
        print(f"Successfully loaded and unrolled {len(self.samples)} 2D slices for the '{split}' set.")

    def _load_and_unroll_samples(self) -> List[Tuple[str, str, int]]:
        """
        Loads all valid file pairs and unrolls the 3D volumes into a list of 2D slices.
        """
        unrolled_samples = []
        image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.endswith('_0000.nii.gz')])

        print(f"Found {len(image_filenames)} volumes to process...")
        for image_fname in image_filenames:
            label_fname = image_fname.replace('_0000', '')
            image_path = os.path.join(self.image_dir, image_fname)
            label_path = os.path.join(self.label_dir, label_fname)

            if os.path.exists(label_path):
                try:
                    # Load the NIfTI file just to get its shape
                    img_nifti = nib.load(image_path)
                    num_slices = img_nifti.shape[2] # Shape is (H, W, Depth)

                    # Create a sample for each slice in the volume
                    for slice_idx in range(num_slices):
                        unrolled_samples.append((image_path, label_path, slice_idx))
                except Exception as e:
                    print(f"Warning: Could not read shape of {image_fname}. Skipping. Error: {e}")
        
        return unrolled_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Get the path and the specific slice index for this sample
        image_path, mask_path, slice_idx = self.samples[idx]

        try:
            # 2. Load the full 3D volumes
            img_nifti = nib.load(image_path)
            image_volume = img_nifti.get_fdata()
            
            mask_nifti = nib.load(mask_path)
            mask_volume = mask_nifti.get_fdata()

            # 3. Extract the specific 2D slice for this sample
            image = image_volume[:, :, slice_idx]
            mask = mask_volume[:, :, slice_idx]

            # --- Continue with the same 2D processing as before ---
            if image.max() > 1.0:
                image = (image - image.min()) / (image.max() - image.min()) * 255.0
            image = image.astype(np.float32)
            image = np.stack([image, image, image], axis=-1)

            mask[mask > 4] = 0
            mask = mask.astype(np.uint8)
            mask = np.expand_dims(mask, axis=-1)

            # Apply transformations
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            
            return image, mask

        except Exception as e:
            print(f"\n[FATAL ERROR] An error occurred while processing index {idx}:")
            print(f"  Image Path: {image_path} (Slice: {slice_idx})")
            print(f"  Mask Path: {mask_path} (Slice: {slice_idx})")
            print(f"  Error: {e}")
            raise e

# Example Usage
if __name__ == "__main__":
    # IMPORTANT: Replace with your actual downloaded CAMUS paths
    camus_root_dir = "/path/to/your/CAMUS_public" # This is the folder containing database_nifti and database_split
    nifti_base_dir = os.path.join(camus_root_dir, "database_nifti")
    split_base_dir = os.path.join(camus_root_dir, "database_split")

    train_split_file = os.path.join(split_base_dir, "subgroup_training.txt")
    val_split_file = os.path.join(split_base_dir, "subgroup_validation.txt")
    test_split_file = os.path.join(split_base_dir, "subgroup_testing.txt")

    # Define transformations (as before)
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3,7), p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], # ImageNet means/stds
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0, # Data is scaled to 0-255 before normalization
        ),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    # Instantiate datasets
    try:
        train_dataset = CAMUSDataset(nifti_base_dir, train_split_file, train_transform)
        val_dataset = CAMUSDataset(nifti_base_dir, val_split_file, val_transform)
        # test_dataset = CAMUSDataset(nifti_base_dir, test_split_file, val_transform) # Uncomment for test
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

        # Test retrieving a batch
        for images, masks in train_loader:
            print(f"Batch images shape: {images.shape}") # Should be (B, 3, 224, 224)
            print(f"Batch masks shape: {masks.shape}")   # Should be (B, 1, 224, 224)
            print(f"Image dtype: {images.dtype}, Mask dtype: {masks.dtype}")
            break

        # Visualize an example (from validation set, as it has no augmentations for clearer view)
        if len(val_dataset) > 0:
            sample_image, sample_mask = val_dataset[0]
            # Denormalize image for correct display if it was normalized
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            display_image = (sample_image * std + mean) * 255.0 # Reverse normalization and scale to 0-255
            display_image = display_image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            
            display_mask = sample_mask.squeeze().cpu().numpy()

            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(display_image)
            plt.title("Sample Image (Denormalized)")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            # Use a colormap for masks to distinguish classes better if multi-class
            plt.imshow(display_mask, cmap='viridis', vmin=0, vmax=np.max(display_mask) if np.max(display_mask) > 0 else 1)
            plt.title(f"Sample Mask (Classes: {np.unique(display_mask)})")
            plt.axis('off')
            plt.colorbar(ticks=np.unique(display_mask)) # Show colorbar for mask values
            plt.show()

    except ValueError as e:
        print(f"Error initializing dataset: {e}. Please ensure CAMUS data paths are correct and split files exist.")
        # Offer visual guidance on setting up the CAMUS data path
        print("\nExpected CAMUS directory structure:")
        print("CAMUS_public/")
        print("├── database_nifti/")
        print("│   └── patient0001/")
        print("│       ├── patient0001_2CH_ED.nii.gz")
        print("│       └── patient0001_2CH_ED_gt.nii.gz")
        print("└── database_split/")
        print("    ├── subgroup_training.txt")
        print("    └── subgroup_validation.txt")
        print(f"Please update 'camus_root_dir = \"{camus_root_dir}\"' to your actual path.")