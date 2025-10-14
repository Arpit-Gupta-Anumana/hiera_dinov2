import torch
import torch.nn.functional as F

def dice_score(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Calculates the Dice score for a multi-class segmentation task.

    Args:
        predictions (torch.Tensor): The model's output logits. Shape: (B, C, H, W).
        targets (torch.Tensor): The ground truth labels. Shape: (B, 1, H, W).
        epsilon (float): A small value to prevent division by zero.

    Returns:
        torch.Tensor: The average Dice score, excluding the background class.
    """
    # 1. Convert logits to class predictions
    # Apply softmax to get probabilities, then argmax to get the predicted class index for each pixel.
    pred_mask = torch.softmax(predictions, dim=1).argmax(dim=1) # Shape: (B, H, W)

    # 2. Convert prediction and target masks to one-hot format
    # The number of classes should match the channel dimension of the predictions
    num_classes = predictions.shape[1]
    
    # F.one_hot adds the new class dimension at the end: (B, H, W, C)
    pred_mask_one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2)
    
    # Squeeze the channel dim from target before one-hot encoding
    targets_one_hot = F.one_hot(targets.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2)
    
    # Ensure they are float for multiplication
    pred_mask_one_hot = pred_mask_one_hot.float()
    targets_one_hot = targets_one_hot.float()

    # 3. Calculate Dice score
    # We ignore the background class (class 0) in the calculation, which is standard practice.
    # Sum over the spatial dimensions (H, W)
    intersection = (pred_mask_one_hot[:, 1:] * targets_one_hot[:, 1:]).sum(dim=(2, 3))
    union = pred_mask_one_hot[:, 1:].sum(dim=(2, 3)) + targets_one_hot[:, 1:].sum(dim=(2, 3))
    
    dice_coefficient = (2. * intersection + epsilon) / (union + epsilon)
    
    # Return the mean Dice score across the batch and classes
    return dice_coefficient.mean()