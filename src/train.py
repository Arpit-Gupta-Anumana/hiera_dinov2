# Expected Output
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Import our custom modules
from .datasets.ultrasound_dataset import CAMUSDataset
from .models.ultrasound_segmenter import UltrasoundSegmenter
from .utils import dice_score

# Import MONAI components
from monai.losses import DiceCELoss
from albumentations.pytorch import ToTensorV2
import albumentations as A


# --- 1. HYPERPARAMETERS and CONFIGURATION ---
##DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu") # cuda:0 export_CUDA_VISIBLE_DEVICES=0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# cuda:0 export_CUDA_VISIBLE_DEVICES=0

LEARNING_RATE = 1e-4
BATCH_SIZE = 10 # Adjust based on your GPU memory
NUM_EPOCHS = 50 # Start with a smaller number to test, e.g., 5-10
NUM_CLASSES = 3 # Background, Left Ventricle, Left Atrium for CAMUS
NUM_WORKERS = 4 # Set to 0 for macOS to avoid potential issues with MPS
PIN_MEMORY = True
SAVE_CHECKPOINT = True
CHECKPOINT_DIR = "checkpoints/"

# --- IMPORTANT: Update this path to your downloaded CAMUS dataset ---
#for mac
#CAMUS_ROOT_DIR = "/Users/arpit.gupta/Documents/Hiera + DinoV2/data/CAMUS_public"
#for gpu
# In src/train.py
# --- IMPORTANT: Update this path to your downloaded CAMUS dataset ---
CAMUS_ROOT_DIR = "/home/arpit_gupta/hiera_dinov2/data/CAMUS_public"
# --- 2. TRAINING and VALIDATION FUNCTIONS ---


def train_fn(loader, model, optimizer, loss_fn, scaler=None):
    """
    Runs one epoch of training.
    """
    loop = tqdm(loader, desc="Training")
    model.train()
    
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE).long().permute(0, 3, 1, 2)


        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update metrics
        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    avg_loss = running_loss / len(loader)
    print(f"Training Epoch Finished! Average Loss: {avg_loss:.4f}")
    return avg_loss


def val_fn(loader, model, loss_fn):
    """
    Runs one epoch of validation.
    """
    loop = tqdm(loader, desc="Validation")
    model.eval()
    
    running_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE).long().permute(0, 3, 1, 2)


            # Forward pass
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            # Calculate metrics
            dice = dice_score(predictions, targets)
            total_dice += dice.item()
            running_loss += loss.item()
            loop.set_postfix(val_loss=loss.item(), dice_score=dice.item())
    
    avg_loss = running_loss / len(loader)
    avg_dice = total_dice / len(loader)
    print(f"Validation Epoch Finished! Average Loss: {avg_loss:.4f}, Average Dice Score: {avg_dice:.4f}")
    return avg_loss, avg_dice


# --- 3. MAIN SCRIPT ---
def main():
    print(f"Using device: {DEVICE}")

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    # --- Data Loading and Transforms ---
    # Define transformations from the paper
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    # Dataset and DataLoader setup
    nifti_base_dir = os.path.join(CAMUS_ROOT_DIR, "database_nifti")
    split_base_dir = os.path.join(CAMUS_ROOT_DIR, "database_split")
    
    train_split_file = os.path.join(split_base_dir, "subgroup_training.txt")
    val_split_file = os.path.join(split_base_dir, "subgroup_validation.txt")

    train_dataset = CAMUSDataset(nifti_base_dir, train_split_file, transform=train_transform)
    val_dataset = CAMUSDataset(nifti_base_dir, val_split_file, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # --- Model, Loss, Optimizer ---
    model = UltrasoundSegmenter(num_classes=NUM_CLASSES).to(DEVICE)
    
    # The paper's method freezes the backbones and only trains the adapters and decoder.
    # For simplicity, we will train the decoder and the DINOv2 projection layer.
    # The backbones in our wrappers are already frozen (requires_grad=False).
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {sum(p.numel() for p in params_to_train)}")

    # Loss function from MONAI (includes Softmax)
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    
    optimizer = optim.Adam(params_to_train, lr=LEARNING_RATE)
    
    # Optional: Learning Rate Scheduler (Cosine Decay as in paper)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # --- Main Training Loop ---
    best_dice_score = -1.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, val_dice = val_fn(val_loader, model, loss_fn)
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint if it's the best model so far
        if SAVE_CHECKPOINT and val_dice > best_dice_score:
            best_dice_score = val_dice
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice_score": best_dice_score,
            }
            checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
            print(f"==> New best model found! Saving checkpoint to {checkpoint_path}")
            torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main()
