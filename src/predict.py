import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# We need to import our model definition
from .models.ultrasound_segmenter import UltrasoundSegmenter

# --- 1. CONFIGURATION ---
# --- (User needs to set these variables) ---

# Path to the best model checkpoint saved during training
CHECKPOINT_PATH = "/home/arpit_gupta/hiera_dinov2/checkpoints_ice4class_ddp/best_model.pth.tar"
# Path to the root of the dataset you want to test on
DATASET_ROOT = "/home/arpit_gupta/hiera_dinov2/data/Dataset004_ICE4Classes"

# The FILENAME of the specific image you want to visualize
# This must be one of the files in the 'imagesTr' or 'imagesTs' folder
IMAGE_FILENAME = "202107191016380003ABD_0000.nii.gz" 
# The specific 2D slice from the 3D volume you want to visualize
SLICE_INDEX = 25 # Example slice index, change this!

# Model configuration - MUST MATCH THE TRAINED MODEL
NUM_CLASSES = 5

# Set the device to run inference on (e.g., "cuda:0" or "cpu")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- END OF CONFIGURATION ---


'''def main():
    print(f"Using device: {DEVICE}")

    # --- 2. LOAD THE TRAINED MODEL ---
    print(f"Loading model checkpoint from: {CHECKPOINT_PATH}")
    
    # Instantiate the model architecture
    model = UltrasoundSegmenter(num_classes=NUM_CLASSES)
    
    # Load the checkpoint dictionary
    # map_location ensures that a GPU-trained model can be loaded onto a CPU if needed
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    
    # Load the weights into the model
    # Note: Our DDP training script correctly saves the underlying `model.module.state_dict()`,
    # so we can load it directly into a non-DDP model instance like this.
    model.load_state_dict(checkpoint['state_dict'])
    
    # Move the model to the selected device
    model.to(DEVICE)
    
    # Set the model to evaluation mode
    # This is CRITICAL: it disables dropout and sets batchnorm layers to use learned stats
    model.eval()
    print("Model loaded successfully.")

    # --- 3. LOAD AND PREPROCESS THE DATA ---
    # Construct the full paths for the image and its corresponding mask
    image_path = os.path.join(DATASET_ROOT, "imagesTr", IMAGE_FILENAME)
    mask_fname = IMAGE_FILENAME.replace('_0000', '')
    mask_path = os.path.join(DATASET_ROOT, "labelsTr", mask_fname)
    
    print(f"Loading image: {image_path}")
    print(f"Loading mask: {mask_path}")

    # Load the 3D volumes
    image_volume = nib.load(image_path).get_fdata()
    mask_volume = nib.load(mask_path).get_fdata()

    # Extract the specific 2D slice we want to visualize
    image_slice = image_volume[:, :, SLICE_INDEX]
    ground_truth_mask = mask_volume[:, :, SLICE_INDEX]

    # Preprocess the image slice exactly as we did for validation
    # This ensures the model receives data in the format it expects
    image_preprocessed = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255.0
    image_preprocessed = image_preprocessed.astype(np.float32)
    image_preprocessed = np.stack([image_preprocessed] * 3, axis=-1)

    # Define the validation transform (Resize + Normalize)
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # Apply the transform
    augmented = val_transform(image=image_preprocessed)
    image_tensor = augmented['image']

    # --- 4. RUN INFERENCE ---
    print("Running inference...")
    
    # The model expects a batch of images, so we add a batch dimension (B, C, H, W)
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Use torch.no_grad() for efficiency, as we don't need to compute gradients
    with torch.no_grad():
        logits = model(input_tensor)
    
    # --- 5. POST-PROCESS THE OUTPUT ---
    # Convert the raw logits to a predicted mask
    # 1. Apply softmax to get probabilities
    # 2. Apply argmax to get the class index with the highest probability for each pixel
    pred_mask = torch.softmax(logits, dim=1).argmax(dim=1)
    
    # Remove the batch dimension and move the tensor to the CPU for visualization
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    print("Inference complete.")

    # --- 6. VISUALIZE THE RESULTS ---
    print("Displaying results...")
    
    # Get the unique labels for consistent color mapping
    labels = np.unique(ground_truth_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot Original Image Slice
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title(f"Original Image (Slice {SLICE_INDEX})")
    axes[0].axis('off')
    
    # Plot Ground Truth Mask
    im1 = axes[1].imshow(ground_truth_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    # Plot Predicted Mask
    im2 = axes[2].imshow(pred_mask_np, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[2].set_title("Model's Predicted Mask")
    axes[2].axis('off')

    # Add a colorbar to explain the class colors
    fig.colorbar(im1, ax=axes, orientation='horizontal', ticks=labels, fraction=0.05, pad=0.05)
    
    plt.suptitle(f"Segmentation Result for: {IMAGE_FILENAME}", fontsize=16)
    plt.show()

'''

def main():
    print(f"Using device: {DEVICE}")
    
    # --- Create an output directory ---
    output_dir = "../outputs"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved in: {output_dir}")

    # --- 2. LOAD THE TRAINED MODEL ---
    print(f"Loading model checkpoint from: {CHECKPOINT_PATH}")
    model = UltrasoundSegmenter(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. LOAD AND PREPROCESS THE DATA ---
    image_path = os.path.join(DATASET_ROOT, "imagesTr", IMAGE_FILENAME)
    mask_fname = IMAGE_FILENAME.replace('_0000', '')
    mask_path = os.path.join(DATASET_ROOT, "labelsTr", mask_fname)
    
    print(f"Loading image: {image_path}")
    print(f"Loading mask: {mask_path}")

    # Load the original NIfTI object to preserve its metadata (affine, header)
    original_nifti = nib.load(image_path)
    image_volume = original_nifti.get_fdata()
    mask_volume = nib.load(mask_path).get_fdata()

    image_slice = image_volume[:, :, SLICE_INDEX]
    ground_truth_mask = mask_volume[:, :, SLICE_INDEX]

    image_preprocessed = (image_slice - image_slice.min()) / (image_slice.max() - image_slice.min()) * 255.0
    image_preprocessed = image_preprocessed.astype(np.float32)
    image_preprocessed = np.stack([image_preprocessed] * 3, axis=-1)

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    augmented = val_transform(image=image_preprocessed)
    image_tensor = augmented['image']

    # --- 4. RUN INFERENCE ---
    print("Running inference...")
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor)
    
    # --- 5. POST-PROCESS THE OUTPUT ---
    pred_mask = torch.softmax(logits, dim=1).argmax(dim=1)
    pred_mask_np = pred_mask.squeeze().cpu().numpy()
    print("Inference complete.")

    # --- 6. SAVE THE RESULTS ---
    
    # === SAVE THE VISUALIZATION PLOT ===
    plot_filename = f"prediction_{IMAGE_FILENAME.replace('.nii.gz', '')}_slice{SLICE_INDEX}.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    labels = np.unique(ground_truth_mask)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title(f"Original Image (Slice {SLICE_INDEX})")
    axes[0].axis('off')
    
    im1 = axes[1].imshow(ground_truth_mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    im2 = axes[2].imshow(pred_mask_np, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[2].set_title("Model's Predicted Mask")
    axes[2].axis('off')

    fig.colorbar(im1, ax=axes, orientation='horizontal', ticks=labels, fraction=0.05, pad=0.05)
    plt.suptitle(f"Segmentation Result for: {IMAGE_FILENAME}", fontsize=16)
    
    # Instead of plt.show(), we use plt.savefig()
    plt.savefig(plot_save_path)
    print(f"Visualization saved to: {plot_save_path}")

    # === SAVE THE PREDICTED MASK AS A NIFTI FILE ===
    # This is crucial for use in medical imaging software like ITK-SNAP
    mask_nifti_filename = f"predicted_mask_{IMAGE_FILENAME.replace('_0000', '')}"
    mask_save_path = os.path.join(output_dir, mask_nifti_filename)
    
    # Create a new NIfTI image for the predicted mask.
    # We use the affine matrix and header from the ORIGINAL image to ensure
    # perfect alignment in medical viewers. This is a critical best practice.
    pred_mask_nifti = nib.Nifti1Image(pred_mask_np, affine=original_nifti.affine, header=original_nifti.header)
    nib.save(pred_mask_nifti, mask_save_path)
    print(f"Predicted mask saved to: {mask_save_path}")



if __name__ == "__main__":
    main()
