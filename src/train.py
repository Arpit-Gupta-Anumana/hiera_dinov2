# Expected Output
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


# Import our custom modules
from .datasets.ultrasound_dataset import CAMUSDataset
from .models.ultrasound_segmenter import UltrasoundSegmenter
from .utils import dice_score

# Import MONAI components
from monai.losses import DiceCELoss
from albumentations.pytorch import ToTensorV2
import albumentations as A


# --- 1. HYPERPARAMETERS and CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # Adjust based on your GPU memory
NUM_EPOCHS = 50 # Start with a smaller number to test, e.g., 5-10
NUM_CLASSES = 5 # Background, Left Ventricle, Left Atrium for CAMUS
NUM_WORKERS = 1 # Set to 0 for macOS to avoid potential issues with MPS
PIN_MEMORY = True
SAVE_CHECKPOINT = True
CHECKPOINT_DIR = "checkpoints_ice4class/"

# --- IMPORTANT: Update this path to your downloaded CAMUS dataset ---
CAMUS_ROOT_DIR = "/home/arpit_gupta/GPU/hiera_dinov2/data/Dataset004_ICE4Classes"


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
    # --- 1. DDP SETUP ---
    # torchrun will set these environment variables for each process
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize the process group for communication
    dist.init_process_group("nccl") # NCCL is the backend for NVIDIA GPUs

    # The device for this specific process is its local rank
    DEVICE = f"cuda:{local_rank}"
    torch.cuda.set_device(DEVICE)

    if rank == 0:
        print(f"Starting DDP with {world_size} GPUs.")
        print(f"Using device: {DEVICE}")

    # --- 2. PATHS and CONFIG (mostly unchanged) ---
    CAMUS_ROOT_DIR = "/mnt/sdc/arpit/hiera_dinov2/data/Dataset004_ICE4Classes"

    CHECKPOINT_DIR = "checkpoints_ice4class_ddp/" # New checkpoint dir

    # --- 3. DATA LOADING with DISTRIBUTED SAMPLER ---
    # Data transforms remain the same
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
    train_dataset = CAMUSDataset(dataset_root=CAMUS_ROOT_DIR, split="train", transform=train_transform)
    val_dataset = CAMUSDataset(dataset_root=CAMUS_ROOT_DIR, split="train", transform=val_transform)

    # DDP requires a DistributedSampler to ensure each process gets a unique subset of data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        sampler=train_sampler,
        shuffle=False # Shuffle is handled by the sampler, must be False here
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS, 
        pin_memory=True, 
        sampler=val_sampler,
        shuffle=False
    )
    
    if rank == 0:
        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    
    # --- 4. MODEL, LOSS, OPTIMIZER ---
    # Model must be moved to the correct device *before* DDP wrapping
    model = UltrasoundSegmenter(num_classes=NUM_CLASSES).to(DEVICE)
    model = DDP(model, device_ids=[local_rank]) # DDP: Wrap the model

    params_to_train = [p for p in model.parameters() if p.requires_grad]
    
    if rank == 0:
        print(f"Number of trainable parameters: {sum(p.numel() for p in params_to_train)}")

    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = optim.Adam(params_to_train, lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # --- 5. MAIN TRAINING LOOP ---
    best_dice_score = -1.0
    
    for epoch in range(NUM_EPOCHS):
        # Set epoch for the sampler to ensure shuffling is different each epoch
        train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn)
        val_loss, val_dice = val_fn(val_loader, model, loss_fn)
        
        scheduler.step()
        
        # Only the main process (rank 0) should save checkpoints and print
        if rank == 0:
            if SAVE_CHECKPOINT and val_dice > best_dice_score:
                best_dice_score = val_dice
                # When saving a DDP model, we save its underlying .module.state_dict()
                checkpoint = {
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_dice_score": best_dice_score,
                }
                if not os.path.exists(CHECKPOINT_DIR):
                    os.makedirs(CHECKPOINT_DIR)
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth.tar")
                print(f"==> New best model found! Saving checkpoint to {checkpoint_path}")
                torch.save(checkpoint, checkpoint_path)

    # --- 6. CLEANUP ---
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
