import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

def get_samples(image_dir, label_dir):
    """
    This function finds all valid image-label pairs based on the
    dataset's specific naming convention.
    """
    samples = []
    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('_0000.nii.gz')])

    for image_fname in image_filenames:
        label_fname = image_fname.replace('_0000', '')
        image_path = os.path.join(image_dir, image_fname)
        label_path = os.path.join(label_dir, label_fname)

        if os.path.exists(label_path):
            samples.append((image_path, label_path))
    
    return samples

def main():
    """
    Main function to run the dataset inspection.
    """
    # --- CONFIGURE THIS PATH ---
    # This should be the root of your project, where the 'data' folder is.
    project_root = "/Users/arpit.gupta/Documents/Hiera + DinoV2" 
    # Or on the server: "/home/arpit_gupta/hiera_dinov2"
    # -------------------------

    dataset_root = os.path.join(project_root, "data", "Dataset004_ICE4Classes")
    image_dir = os.path.join(dataset_root, "imagesTr")
    label_dir = os.path.join(dataset_root, "labelsTr")

    print(f"Checking dataset in: {dataset_root}\n")

    # Get the list of all valid file pairs
    file_pairs = get_samples(image_dir, label_dir)
    print(f"Found {len(file_pairs)} valid image-label pairs to check.")

    if not file_pairs:
        print("No valid pairs found. Please check your paths and filenames.")
        return

    # Use tqdm for a progress bar
    for image_path, mask_path in tqdm(file_pairs, desc="Inspecting Files"):
        try:
            # Load the image and mask using nibabel
            img_nifti = nib.load(image_path)
            image_array = img_nifti.get_fdata()

            mask_nifti = nib.load(mask_path)
            mask_array = mask_nifti.get_fdata()
            
            # --- Print the vital statistics for each file ---
            print(f"\n--- FILE: {os.path.basename(image_path)} ---")
            print(f"  Image Shape: {image_array.shape}\t| Mask Shape: {mask_array.shape}")
            print(f"  Image Dtype: {image_array.dtype}\t| Mask Dtype: {mask_array.dtype}")
            print(f"  Mask Unique Labels: {np.unique(mask_array)}")

        except Exception as e:
            print(f"\n--- FAILED TO PROCESS: {os.path.basename(image_path)} ---")
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    main()