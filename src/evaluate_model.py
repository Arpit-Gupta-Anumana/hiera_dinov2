import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import our custom modules
from .datasets.ultrasound_dataset import CAMUSDataset
from .models.ultrasound_segmenter import UltrasoundSegmenter
from .utils import dice_score

# --- 1. CONFIGURATION ---
# Path to the best model checkpoint from your training run
CHECKPOINT_PATH = "../checkpoints_ice4class_ddp/best_model.pth.tar"

# Path to the root of the dataset
DATASET_ROOT = "../data/Dataset004_ICE4Classes"

# Model and data configuration - MUST MATCH THE TRAINED MODEL
NUM_CLASSES = 5
BATCH_SIZE = 4 # Adjust based on your GPU memory
NUM_WORKERS = 4

# Map class indices to human-readable names for the report
CLASS_NAMES = {
    0: "Background",
    1: "LA (Left Atrium)",
    2: "LV (Left Ventricle)", # Assumed
    3: "LPV (Left Pulmonary Vein)",
    4: "RPV (Right Pulmonary Vein)"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -------------------------

def main():
    print(f"Starting evaluation on device: {DEVICE}")

    # --- 2. LOAD THE TRAINED MODEL ---
    print(f"Loading model from: {CHECKPOINT_PATH}")
    model = UltrasoundSegmenter(num_classes=NUM_CLASSES)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(DEVICE))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode

    # --- 3. SETUP THE TEST DATASET ---
    # We use the 'test' split now
    # The transform for evaluation should NOT have random augmentations
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    test_dataset = CAMUSDataset(dataset_root=DATASET_ROOT, split="test", transform=test_transform)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        shuffle=False
    )
    
    print(f"Found {len(test_dataset)} slices in the test set.")

    # --- 4. RUN EVALUATION LOOP ---
    all_per_class_dice = []
    loop = tqdm(test_loader, desc="Evaluating")

    with torch.no_grad():
        for (data, targets) in loop:
            data = data.to(DEVICE)
            # The target shape must be (B, 1, H, W) and of type long
            targets = targets.to(DEVICE).long().permute(0, 3, 1, 2)

            predictions = model(data)
            
            # Our new function returns mean and per-class scores for the batch
            _, per_class_scores_batch = dice_score(predictions, targets)
            
            # We store the per-class scores for each batch
            all_per_class_dice.append(per_class_scores_batch.cpu())

    # --- 5. CALCULATE FINAL METRICS AND GENERATE REPORT ---
    if not all_per_class_dice:
        print("Evaluation could not be completed. No data was processed.")
        return

    # Stack the batch scores and calculate the mean across the entire dataset
    final_dice_scores = torch.stack(all_per_class_dice).mean(dim=0)

    print("\n\n--- Model Evaluation Report ---")
    print("="*50)
    print("--- Per-Class Dice Similarity Coefficient (DSC) ---")
    print(f"{'Class Name':<30} {'Dice Score':>15}")
    print(f"-"*50)
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES.get(i, f"Class {i}")
        score = final_dice_scores[i].item()
        print(f"{name:<30} {score:>15.4f}")
    print(f"-"*50)
    
    # Calculate and print the mean foreground Dice score
    mean_foreground_score = final_dice_scores[1:].mean().item()
    print(f"{'Mean Foreground Dice':<30} {mean_foreground_score:>15.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
