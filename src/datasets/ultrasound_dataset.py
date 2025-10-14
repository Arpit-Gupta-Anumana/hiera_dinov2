import os
import torch
import numpy as np
import nibabel as nib # For NIfTI files
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple

class CAMUSDataset(Dataset):
    def __init__(self, nifti_base_dir: str, split_file_path: str, transform=None):
        """
        Args:
            nifti_base_dir (str): Path to the 'database_nifti' directory.
            split_file_path (str): Path to the split file (e.g., 'subgroup_training.txt').
            transform (albumentations.Compose, optional): Transformations to apply. Defaults to None.
        """
        self.nifti_base_dir = nifti_base_dir
        self.transform = transform
        self.samples = self._load_samples(split_file_path)

        if not self.samples:
            raise ValueError(f"No samples loaded from split file: {split_file_path}")

    def _load_samples(self, split_file_path: str) -> List[Tuple[str, str]]:
        """
        Reads the split file and creates a list of (image_path, mask_path) tuples.
        """
        samples = []
        with open(split_file_path, 'r') as f:
            patient_ids = [line.strip() for line in f if line.strip()]

        for patient_id in patient_ids:
            patient_folder = os.path.join(self.nifti_base_dir, patient_id)
            if not os.path.exists(patient_folder):
                print(f"Warning: Patient folder {patient_folder} not found. Skipping.")
                continue
            
            # The paper mentions 'end-diastole and end-systole frames' and 
            # 'two- and four-chamber views'.
            # We need to iterate through the relevant NIfTI files for each patient.
            # Example: patient0001_2CH_ED.nii.gz, patient0001_2CH_ED_gt.nii.gz
            # patient0001_2CH_ES.nii.gz, patient0001_2CH_ES_gt.nii.gz
            # patient0001_4CH_ED.nii.gz, patient0001_4CH_ED_gt.nii.gz
            # patient0001_4CH_ES.nii.gz, patient0001_4CH_ES_gt.nii.gz

            # Let's assume we want to process all ED/ES 2CH/4CH images and their GT
            # You might need to refine this based on your specific task (e.g., only 2CH ED)
            views = ["2CH", "4CH"]
            states = ["ED", "ES"]

            for view in views:
                for state in states:
                    image_filename = f"{patient_id}_{view}_{state}.nii.gz"
                    mask_filename = f"{patient_id}_{view}_{state}_gt.nii.gz"

                    image_path = os.path.join(patient_folder, image_filename)
                    mask_path = os.path.join(patient_folder, mask_filename)

                    if os.path.exists(image_path) and os.path.exists(mask_path):
                        samples.append((image_path, mask_path))
                    else:
                        print(f"Warning: Missing image or mask for {patient_id}, {view}, {state}. Skipping.")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]

        # Load NIfTI image
        img_nifti = nib.load(image_path)
        # NIfTI data is typically accessed via .get_fdata() which returns a numpy array
        # It can be 3D (H, W, D) or even 4D (H, W, D, T).
        # For CAMUS, ED/ES files are usually 2D slices (H, W) or (H, W, 1) if single slice.
        # We need to ensure it's (H, W) or (H, W, C) for Albumentations.
        image = img_nifti.get_fdata()
        
        # Squeeze any singleton dimensions, e.g., (H, W, 1) -> (H, W)
        image = np.squeeze(image) 
        
        # Check if it's still 3D (e.g., if it's a series of slices), and pick one if necessary.
        # However, for CAMUS ED/ES files, they usually represent a single 2D frame.
        if image.ndim > 2:
             print(f"Warning: Image {image_path} has more than 2 dimensions after squeeze: {image.shape}. Taking first slice.")
             image = image[:,:,0] # Take the first slice if it's (H, W, D)
        
        # Normalize image to 0-255 range if not already (NIfTI data can have various ranges)
        # This is a common step before applying ImageNet normalization
        if image.max() > 1.0: # Assuming original data is float, e.g., 0-1 or 0-X
            image = (image - image.min()) / (image.max() - image.min()) * 255.0
        image = image.astype(np.float32) # Albumentations expects float32

        # Convert grayscale to 3-channel (RGB) for pre-trained models
        image = np.stack([image, image, image], axis=-1) # (H, W) -> (H, W, 3)

        # Load NIfTI mask
        mask_nifti = nib.load(mask_path)
        mask = mask_nifti.get_fdata()
        mask = np.squeeze(mask)
        
        # Ensure mask is integer type for class labels if multi-class
        mask = mask.astype(np.float32) 
        
        # For CAMUS, the ground truth often has specific integer labels:
        # e.g., 0 for background, 1 for LV, 2 for LA.
        # The paper mentions "left ventricle (gray) and atrium (white)".
        # This implies specific pixel values in the mask.
        # You might need to map these to 0, 1, 2... if they are something else
        # e.g., mask[mask == LV_VALUE] = 1.0, mask[mask == LA_VALUE] = 2.0
        # For now, we'll assume the loaded mask directly represents class labels (0, 1, ...)
        
        # Add channel dimension to mask (H, W) -> (H, W, 1) for Albumentations
        mask = np.expand_dims(mask, axis=-1)

        # Apply transformations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        # Ensure mask tensor has class labels as Long type for CrossEntropyLoss
        # (ToTensorV2 will output float, so convert if needed for loss function later)
        # For DiceCELoss, it typically expects float masks as probabilities or one-hot.
        # If your loss function expects class indices for masks, you'll need mask.long()
        # For now, let's keep it float, assuming the loss can handle it.
        
        return image, mask

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