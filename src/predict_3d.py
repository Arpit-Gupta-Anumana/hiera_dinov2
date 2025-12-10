import os
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Import our new 3D model
from .models.ultrasound_segmenter_hybrid import UltrasoundSegmenterHybrid

# --- 1. CONFIGURATION ---
# --- (User needs to set these variables) ---

# Path to the best 3D model checkpoint
CHECKPOINT_PATH = "/home/arpit_gupta/hiera_dinov2/checkpoints_hybrid_3d/best_model.pth.tar"

# Path to the root of the dataset
DATASET_ROOT = "/home/arpit_gupta/hiera_dinov2/data/Dataset004_ICE4Classes"

# The FILENAME of the specific 3D volume you want to test
IMAGE_FILENAME = "202106101716000035CARD_0000.nii.gz" # Example, change this!

# Model configuration - MUST MATCH THE TRAINED MODEL
NUM_CLASSES = 5
CLIP_DEPTH = 16

# --- Clip and Slice Selection ---
# From the full video volume, which 16-frame clip do you want to predict?
# This is the starting slice index for the clip.
CLIP_START_INDEX = 10 # Example: starts prediction from the 10th slice

# From the predicted 16-frame clip, which slice do you want to save in the plot?
# The middle slice is usually a good choice.
VISUALIZATION_SLICE_INDEX = 8 # Index relative to the clip (0-15)

# Set the device to run inference on
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Map class indices to human-readable names
CLASS_NAMES = { 0: "Background", 1: "LA", 2: "LV", 3: "LPV", 4: "RPV" }
# --------------------------------

def main():
    print(f"Using device: {DEVICE}")
    
    # --- Create an output directory for the results ---
    output_dir = "../outputs_3d"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved in: {output_dir}")

    # --- 2. LOAD THE TRAINED 3D MODEL ---
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = UltrasoundSegmenterHybrid(num_classes=NUM_CLASSES, clip_depth=CLIP_DEPTH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. LOAD AND PREPROCESS A 3D CLIP ---
    image_path = os.path.join(DATASET_ROOT, "imagesTs", IMAGE_FILENAME)
    mask_fname = IMAGE_FILENAME.replace('_0000', '')
    mask_path = os.path.join(DATASET_ROOT, "labelsTs", mask_fname)
    
    print(f"Loading volume: {image_path}")

    # Load the original NIfTI object to preserve its metadata (affine, header)
    original_nifti = nib.load(image_path)
    image_volume = original_nifti.get_fdata()
    mask_volume = nib.load(mask_path).get_fdata().astype(np.uint8)

    # Extract the 3D clip we want to predict
    image_clip = image_volume[:, :, CLIP_START_INDEX : CLIP_START_INDEX + CLIP_DEPTH]
    gt_mask_clip = mask_volume[:, :, CLIP_START_INDEX : CLIP_START_INDEX + CLIP_DEPTH]

    # Preprocess the clip exactly as the 3D data loader does
    image_clip_processed = np.expand_dims(image_clip, axis=0).transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_clip_processed.copy()).float()
    
    image_tensor = torch.nn.functional.interpolate(image_tensor.unsqueeze(0), size=(CLIP_DEPTH, 224, 224), mode='trilinear', align_corners=False).squeeze(0)
    
    min_val, max_val = image_tensor.min(), image_tensor.max()
    if max_val > min_val:
        image_tensor = (image_tensor - min_val) / (max_val - min_val)
    
    # --- 4. RUN INFERENCE ---
    print("Running inference on the 3D clip...")
    input_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_tensor) # This is a 5D tensor
    
    # --- 5. POST-PROCESS THE 3D OUTPUT ---
    pred_mask_volume = torch.softmax(logits, dim=1).argmax(dim=1)
    pred_mask_volume_np = pred_mask_volume.squeeze().cpu().numpy() # Shape: (D, H, W)
    print("Inference complete.")

    # --- 6. SAVE THE RESULTS ---
    
    # === SAVE THE VISUALIZATION PLOT (of a single slice) ===
    plot_filename = f"prediction_3d_{IMAGE_FILENAME.replace('.nii.gz', '')}_clip_start_{CLIP_START_INDEX}.png"
    plot_save_path = os.path.join(output_dir, plot_filename)
    
    # Extract the specific slice for visualization from the original and predicted volumes
    original_slice = image_clip[:, :, VISUALIZATION_SLICE_INDEX]
    gt_slice = gt_mask_clip[:, :, VISUALIZATION_SLICE_INDEX]
    pred_slice = pred_mask_volume_np[VISUALIZATION_SLICE_INDEX, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title(f"Original Slice (Index {CLIP_START_INDEX + VISUALIZATION_SLICE_INDEX})")
    axes[0].axis('off')
    
    axes[1].imshow(gt_slice, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis('off')
    
    axes[2].imshow(pred_slice, cmap='jet', vmin=0, vmax=NUM_CLASSES-1)
    axes[2].set_title("Model's Predicted Mask")
    axes[2].axis('off')

    plt.suptitle(f"3D Segmentation Result for: {IMAGE_FILENAME}", fontsize=16)
    plt.savefig(plot_save_path)
    print(f"Visualization plot saved to: {plot_save_path}")

    # === SAVE THE PREDICTED 3D MASK AS A NIFTI FILE ===
    mask_nifti_filename = f"predicted_mask_3d_{IMAGE_FILENAME.replace('_0000.nii.gz', '.nii.gz')}"
    mask_save_path = os.path.join(output_dir, mask_nifti_filename)
    
    # The predicted mask has shape (D, H, W). We need to permute it back to (H, W, D) for nibabel.
    pred_mask_to_save = pred_mask_volume_np.transpose(1, 2, 0)

    # Create a new NIfTI image using the original's affine matrix and header for perfect alignment.
    pred_mask_nifti = nib.Nifti1Image(pred_mask_to_save, affine=original_nifti.affine, header=original_nifti.header)
    nib.save(pred_mask_nifti, mask_save_path)
    print(f"Predicted 3D mask saved to: {mask_save_path}")

if __name__ == "__main__":
    main()
