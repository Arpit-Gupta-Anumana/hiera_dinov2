import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Import our new 3D modules
from .datasets.ultrasound_dataset_3d import CAMUSDataset3D
from .models.ultrasound_segmenter_hybrid import UltrasoundSegmenterHybrid
from .utils import dice_score_3d # Make sure this is in your utils.py

from monai.losses import DiceFocalLoss

# --- 1. HYPERPARAMETERS and 3D CONFIGURATION ---
LEARNING_RATE = 1e-5     # 3D models benefit from a lower starting LR
BATCH_SIZE = 1           # CRITICAL: Per-GPU batch size. Keep this low.
NUM_EPOCHS = 100         # May need more epochs to converge
NUM_CLASSES = 5
CLIP_DEPTH = 16          # The depth of the video clips we will process
NUM_WORKERS = 4
SAVE_CHECKPOINT = True

# --- RESUME CONFIGURATION ---
RESUME_TRAINING = False
CHECKPOINT_TO_RESUME = "checkpoints_hybrid_3d/best_model.pth.tar"

# Map class indices to human-readable names
CLASS_NAMES = { 0: "Background", 1: "LA", 2: "LV", 3: "LPV", 4: "RPV" }

# --- 2. TRAINING and VALIDATION FUNCTIONS (updated for 3D) ---
def train_fn(loader, model, optimizer, loss_fn, rank):
    loop = tqdm(loader, desc="Training", disable=(rank != 0))
    model.train()
    running_loss = 0.0

    for data, targets in loop:
        data = data.to(f"cuda:{rank}")
        targets = targets.to(f"cuda:{rank}")
        
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if rank == 0:
            loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(loader)
    if rank == 0:
        print(f"Training Epoch Finished! Average Loss: {avg_loss:.4f}")
    return avg_loss

def val_fn(loader, model, loss_fn, rank):
    loop = tqdm(loader, desc="Validation", disable=(rank != 0))
    model.eval()
    
    all_per_class_dice = []

    with torch.no_grad():
        for data, targets in loop:
            data = data.to(f"cuda:{rank}")
            targets = targets.to(f"cuda:{rank}")
            
            predictions = model(data)
            mean_dice, per_class_dice = dice_score_3d(predictions, targets)
            
            all_per_class_dice.append(per_class_dice.cpu())
            if rank == 0:
                loop.set_postfix(dice_score=mean_dice.item())
    
    if not all_per_class_dice:
        return torch.tensor(0.0), torch.zeros(NUM_CLASSES)

    # Average the per-class scores across all batches
    final_class_dice = torch.stack(all_per_class_dice).mean(dim=0)
    avg_dice = final_class_dice[1:].mean() # Mean of foreground classes
    
    return avg_dice, final_class_dice

# --- 3. MAIN SCRIPT ---
def main():
    # --- DDP SETUP ---
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    DEVICE = f"cuda:{local_rank}"
    torch.cuda.set_device(DEVICE)

    if rank == 0:
        print(f"Starting DDP with {world_size} GPUs.")

    # --- PATHS and CONFIG ---
    project_root = "/home/arpit_gupta/hiera_dinov2" # <-- IMPORTANT: Set this to your project root
    DATASET_ROOT = os.path.join(project_root, "data", "Dataset004_ICE4Classes")
    CHECKPOINT_DIR = os.path.join(project_root, "checkpoints_hybrid_3d/")

    # --- DATASET and DATALOADER (using 3D versions) ---
    train_dataset = CAMUSDataset3D(dataset_root=DATASET_ROOT, split="train", clip_depth=CLIP_DEPTH)
    # For simplicity, we'll use the same dataset for validation. A separate val set is recommended.
    val_dataset = CAMUSDataset3D(dataset_root=DATASET_ROOT, split="train", clip_depth=CLIP_DEPTH)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
        num_workers=NUM_WORKERS, pin_memory=True, shuffle=False
    )
    
    if rank == 0:
        print(f"Total training clips: {len(train_dataset)}")

    # --- MODEL (using 3D version) ---
    model = UltrasoundSegmenterHybrid(num_classes=NUM_CLASSES, clip_depth=CLIP_DEPTH).to(DEVICE)
    model = DDP(model, device_ids=[local_rank])
    
    # --- LOSS and OPTIMIZER ---
    loss_fn = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, gamma=2.0)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # --- RESUME FROM CHECKPOINT LOGIC ---
    start_epoch = 0
    best_dice_score = -1.0

    if RESUME_TRAINING:
        resume_path = os.path.join(project_root, CHECKPOINT_TO_RESUME)
        if os.path.exists(resume_path):
            if rank == 0: print(f"==> Resuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=DEVICE)
            model.module.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            start_epoch = checkpoint['epoch'] + 1
            best_dice_score = checkpoint['best_dice_score']
            if rank == 0: print(f"==> Resumed from epoch {start_epoch-1}.")
        else:
            if rank == 0: print(f"==> WARNING: Checkpoint not found at {resume_path}. Starting from scratch.")

    # --- TRAINING LOOP ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, rank)
        val_dice, final_class_dice = val_fn(val_loader, model, loss_fn, rank)
        
        # All-reduce the validation dice score to get the global average
        val_dice_tensor = torch.tensor(val_dice).to(DEVICE)
        dist.all_reduce(val_dice_tensor, op=dist.ReduceOp.AVG)
        global_avg_dice = val_dice_tensor.item()
        
        scheduler.step()
        
        if rank == 0:
            print(f"Validation Epoch Finished! Global Average Dice Score: {global_avg_dice:.4f}")
            for i, score in enumerate(final_class_dice):
                print(f"  - Dice for {CLASS_NAMES.get(i, f'Class {i}'):<30}: {score.item():.4f}")

            if SAVE_CHECKPOINT and global_avg_dice > best_dice_score:
                best_dice_score = global_avg_dice
                print(f"==> New best model found! Saving checkpoint...")
                
                checkpoint = {
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_dice_score": best_dice_score,
                }
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pth.tar"))

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
