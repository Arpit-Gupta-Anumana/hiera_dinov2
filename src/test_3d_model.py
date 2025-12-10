import torch
import sys
import os

# --- Add project root to path to allow imports ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.models.ultrasound_segmenter_hybrid import UltrasoundSegmenterHybrid

def main():
    """
    Instantiates the 3D model and prints the shapes of intermediate features.
    """
    print("--- 3D Model Architecture Verification ---")
    
    # --- Configuration (must match your training setup) ---
    NUM_CLASSES = 5
    CLIP_DEPTH = 16
    BATCH_SIZE = 1 # Use a batch size of 1 for simple testing
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {DEVICE}")

    # --- Instantiate the Model ---
    try:
        model = UltrasoundSegmenterHybrid(num_classes=NUM_CLASSES, clip_depth=CLIP_DEPTH).to(DEVICE)
        model.eval()
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"ERROR: Failed to instantiate model: {e}")
        return

    # --- Create a Dummy Input Tensor ---
    # Shape: (B, C, D, H, W)
    dummy_input = torch.randn(BATCH_SIZE, 1, CLIP_DEPTH, 224, 224).to(DEVICE)
    print(f"\nInput Tensor Shape: {dummy_input.shape}")

    # --- Perform a Forward Pass and Print Shapes ---
    '''with torch.no_grad():
        # Get the intermediate features from the encoders
        _, hiera_features_3d = model.hiera_encoder(dummy_input, return_intermediates=True)
        
        # --- Print Hiera Feature Shapes ---
        print("\n--- Hiera Encoder Intermediate Shapes ---")
        for i, feat in enumerate(hiera_features_3d):
            print(f"  Stage {i+1} Output Shape: {feat.shape}")
        
        # You can add similar logic to inspect other parts of the model if needed,
        # for example, the output of the DINOv2 path.
        
        # --- Check the Final Output Shape ---
        final_output = model(dummy_input)
        print("\n--- F;;;'''
    # --- Perform a Forward Pass and Print Shapes ---
    with torch.no_grad():
        # Call the main model's forward pass correctly, requesting intermediates
        final_output, hiera_features_3d = model(dummy_input, return_intermediates=True)

        # --- Print Hiera Feature Shapes ---
        print("\n--- Hiera Encoder Intermediate Shapes ---")
        for i, feat in enumerate(hiera_features_3d):
            print(f"  Stage {i+1} Output Shape: {feat.shape}")
    
        # --- Check the Final Output Shape ---
        print("\n--- Final Model Output ---")
        print(f"  Final Logit Shape: {final_output.shape}")
    
        # Verify final shape matches input shape (except for channels)
        expected_shape = (BATCH_SIZE, NUM_CLASSES, CLIP_DEPTH, 224, 224)
        assert final_output.shape == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {final_output.shape}"
        print("\nVerification successful: Final output shape is correct.")
if __name__ == "__main__":
    main()
