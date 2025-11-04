import torch
import torch.nn.functional as F

def dice_score(predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-6):
    """
    Calculates the Dice score for each class and the mean score.
    Assumes `predictions` are raw logits and `targets` are class indices.

    Args:
        predictions (torch.Tensor): Model output logits. Shape (B, C, H, W).
        targets (torch.Tensor): Ground truth labels. Shape (B, 1, H, W).

    Returns:
        A tuple: (mean_dice_score, per_class_dice_scores)
        - mean_dice_score (torch.Tensor): The average Dice score over all foreground classes.
        - per_class_dice_scores (torch.Tensor): A 1D tensor with the Dice score for each class (including background).
    """
    # 1. Convert logits to a predicted mask
    pred_mask = torch.softmax(predictions, dim=1).argmax(dim=1)
    num_classes = predictions.shape[1]
    
    # 2. One-hot encode both prediction and target
    pred_mask_one_hot = F.one_hot(pred_mask, num_classes=num_classes).permute(0, 3, 1, 2).float()
    targets_one_hot = F.one_hot(targets.squeeze(1), num_classes=num_classes).permute(0, 3, 1, 2).float()

    # 3. Calculate intersection and union for ALL classes over the batch
    # The .sum() is done over batch (0), height (2), and width (3) dimensions
    intersection = (pred_mask_one_hot * targets_one_hot).sum(dim=(0, 2, 3))
    union = pred_mask_one_hot.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
    
    # 4. Calculate Dice for each class
    # The result is a 1D tensor of size C (number of classes)
    per_class_dice = (2. * intersection + epsilon) / (union + epsilon)
    
    # 5. Calculate the mean of the FOREGROUND classes for a summary score
    mean_foreground_dice = per_class_dice[1:].mean()
    
    return mean_foreground_dice, per_class_dice
