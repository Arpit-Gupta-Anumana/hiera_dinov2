import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

def get_label_paths(label_dir):
    """
    Finds all relevant label files in the directory.
    This logic mirrors the data loader to ensure we analyze the correct files.
    """
    all_files = sorted(os.listdir(label_dir))
    
    # We need the base filenames from the images to find the corresponding labels
    # Example: image '..._0000.nii.gz' -> label '....nii.gz'
    # We find all images and derive the label names from them.
    image_dir = label_dir.replace("labelsTr", "imagesTr")
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Could not find corresponding image directory: {image_dir}")

    image_filenames = {f.replace('_0000', '') for f in os.listdir(image_dir) if f.endswith('_0000.nii.gz')}
    
    label_paths = []
    for fname in all_files:
        if fname in image_filenames:
            label_paths.append(os.path.join(label_dir, fname))
            
    return label_paths


def main():
    """
    Main function to run the dataset statistical analysis.
    """
    # --- 1. CONFIGURATION ---
    # Set the absolute path to your project's root directory
    PROJECT_ROOT = "/home/arpit_gupta/hiera_dinov2"  # <-- IMPORTANT: Change if necessary

    # --- (Optional) Map class indices to human-readable names ---
    # Based on your dataset.json: LA=1, LPV=3, RPV=4. We'll assume LV is 2.
    CLASS_NAMES = {
        0: "Background",
        1: "LA (Left Atrium)",
        2: "LV (Left Ventricle)", # Assumed
        3: "LPV (Left Pulmonary Vein)",
        4: "RPV (Right Pulmonary Vein)"
    }
    NUM_CLASSES = len(CLASS_NAMES)
    
    # --- 2. SETUP ---
    dataset_root = os.path.join(PROJECT_ROOT, "data", "Dataset004_ICE4Classes")
    label_dir = os.path.join(dataset_root, "labelsTr")

    if not os.path.exists(label_dir):
        print(f"ERROR: Label directory not found at: {label_dir}")
        return

    print(f"Analyzing dataset in: {label_dir}\n")

    # Initialize counters
    pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    slice_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    total_slices = 0

    # Get the list of all valid label files
    label_files = get_label_paths(label_dir)
    print(f"Found {len(label_files)} label volumes to analyze.")

    # --- 3. ANALYSIS LOOP ---
    for label_path in tqdm(label_files, desc="Analyzing Volumes"):
        try:
            # Load the 3D mask volume
            mask_volume = nib.load(label_path).get_fdata().astype(np.uint8)

            # --- A. Tally Pixel Counts ---
            # This is the most efficient way to count pixels for the whole volume
            unique_labels, counts = np.unique(mask_volume, return_counts=True)
            for label, count in zip(unique_labels, counts):
                if label < NUM_CLASSES: # Safety check for unexpected labels
                    pixel_counts[label] += count
            
            # --- B. Tally Slice Presence ---
            num_slices_in_volume = mask_volume.shape[2]
            total_slices += num_slices_in_volume

            for i in range(num_slices_in_volume):
                slice_2d = mask_volume[:, :, i]
                # Find which unique labels are present in this single slice
                unique_labels_in_slice = np.unique(slice_2d)
                for label in unique_labels_in_slice:
                    if label < NUM_CLASSES:
                        slice_counts[label] += 1
                        
        except Exception as e:
            print(f"\n--- FAILED TO PROCESS: {os.path.basename(label_path)} ---")
            print(f"  ERROR: {e}")

    # --- 4. CALCULATE AND DISPLAY REPORT ---
    total_pixels = pixel_counts.sum()

    print("\n\n--- Dataset Class Statistics Report ---")
    print("="*60)
    print("--- PIXEL DISTRIBUTION (Class Imbalance) ---")
    print(f"{'Class Name':<30} {'Pixel Count':>15} {'Percentage':>12}")
    print(f"-"*60)
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES.get(i, f"Class {i}")
        count = pixel_counts[i]
        percentage = (count / total_pixels) * 100 if total_pixels > 0 else 0
        print(f"{name:<30} {count:>15,} {percentage:>11.2f}%")
    print(f"-"*60)
    print(f"{'Total Pixels':<30} {total_pixels:>15,}\n")
    
    print("--- SLICE PRESENCE (How often each class appears) ---")
    print(f"{'Class Name':<30} {'Slice Count':>15} {'Percentage':>12}")
    print(f"-"*60)
    for i in range(NUM_CLASSES):
        name = CLASS_NAMES.get(i, f"Class {i}")
        count = slice_counts[i]
        percentage = (count / total_slices) * 100 if total_slices > 0 else 0
        print(f"{name:<30} {count:>15,} {percentage:>11.2f}%")
    print(f"-"*60)
    print(f"{'Total Slices':<30} {total_slices:>15,}\n")


if __name__ == "__main__":
    main()
