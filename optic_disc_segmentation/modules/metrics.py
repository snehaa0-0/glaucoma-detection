#metrics.py
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, multiclass=False, weights=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.multiclass = multiclass
        self.weights = weights  # Added weights parameter

    def forward(self, pred, target):
        if self.multiclass:
            # pred shape: [B, C, H, W], target shape: [B, C, H, W] (one-hot)
            pred = torch.softmax(pred, dim=1)
            dice = 0
            total_weight = 0
            
            # Iterate through classes (skip background if needed)
            for i in range(1, pred.shape[1]):  # Skip background (index 0)
                # Extract the current class channel
                if target.shape[1] == pred.shape[1]:  # If target is one-hot encoded
                    target_i = target[:, i]
                else:  # If target is class indices
                    target_i = (target == i).float()
                
                # Apply weight if provided
                weight = 1.0
                if self.weights is not None:
                    weight = self.weights[i]
                    total_weight += weight
                
                dice += weight * self._dice_single(pred[:, i], target_i)
            
            # Normalize by total weight or number of classes
            if self.weights is not None and total_weight > 0:
                return 1 - dice / total_weight
            else:
                return 1 - dice / (pred.shape[1] - 1)  # Average over classes, excluding background
        else:
            pred = torch.sigmoid(pred)
            return 1 - self._dice_single(pred, target)

    def _dice_single(self, pred, target):
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weights=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weights = weights
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Handle multiclass format
        if targets.dim() == 4:  # one-hot encoded
            targets = targets.argmax(dim=1)
            
        # Apply weights if provided
        weight = None
        if self.weights is not None:
            weight = torch.tensor(self.weights, device=inputs.device)
            
        # Calculate focal loss
        ce_loss = F.cross_entropy(inputs, targets, 
                                 reduction='none', 
                                 weight=weight)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def dice_coefficient(pred, target, smooth=1.0, num_classes=1):
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Model prediction - either binary logits [B, 1, H, W] or 
              multi-class logits [B, C, H, W]
        target: Ground truth - either binary [B, 1, H, W] or 
                one-hot encoded [B, C, H, W] or class indices [B, H, W]
        smooth: Smoothing factor
        num_classes: Number of classes (1 for binary)
    
    Returns:
        Dice coefficient value
    """
    if num_classes > 1:
        # Multi-class case
        pred = torch.softmax(pred, dim=1)
        dice = 0
        
        # Check if target is one-hot encoded or class indices
        if target.shape[1] == num_classes:  # One-hot encoded
            for i in range(1, num_classes):  # Skip background
                pred_i = pred[:, i]
                target_i = target[:, i]
                intersection = (pred_i * target_i).sum()
                dice += (2. * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)
        else:  # Class indices
            for i in range(1, num_classes):  # Skip background
                pred_i = pred[:, i]
                target_i = (target == i).float()
                intersection = (pred_i * target_i).sum()
                dice += (2. * intersection + smooth) / (pred_i.sum() + target_i.sum() + smooth)
        
        # Average over foreground classes
        return dice / (num_classes - 1)
    else:
        # Binary case
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

def combo_loss(pred, target, weights=[0.2, 0.3, 0.5]):
    """
    Combination of Dice, Cross Entropy, and Focal losses with emphasis on cup segmentation.
    
    Args:
        pred: Model predictions (B, C, H, W)
        target: Ground truth masks (B, C, H, W) - one-hot encoded
        weights: Class weights [background, disc, cup]
    
    Returns:
        Combined loss value
    """
    device = pred.device
    weight_tensor = torch.tensor(weights, device=device)
    
    dice_loss = DiceLoss(multiclass=True, weights=weights)(pred, target)
    
    # Convert target to class indices for CrossEntropyLoss
    if target.dim() == 4 and target.shape[1] > 1:  # one-hot encoded
        target_indices = target.argmax(dim=1)
    else:
        target_indices = target
        
    ce_loss = nn.CrossEntropyLoss(weight=weight_tensor)(pred, target_indices)
    focal_loss = FocalLoss(gamma=2.0, weights=weights)(pred, target)
    
    return 0.5*dice_loss + 0.3*ce_loss + 0.2*focal_loss


def calculate_metrics(pred, target, multiclass=False, num_classes=1):
    """
    Calculate metrics for binary segmentation.
    
    Args:
        pred: Model prediction logits [B, 1, H, W]
        target: Ground truth binary mask [B, 1, H, W]
    """
    if multiclass:
        return calculate_metrics_multiclass(pred, target, num_classes)
        
    # Binary metrics
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    
    # Dice coefficient
    dice = dice_coefficient(pred_binary, target)
    
    # IoU (Jaccard Index)
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    
    # Accuracy
    accuracy = ((pred_binary == target).sum() / target.numel()).item()
    
    # Precision and Recall
    true_positive = (pred_binary * target).sum().item()
    false_positive = (pred_binary * (1 - target)).sum().item()
    false_negative = ((1 - pred_binary) * target).sum().item()
    
    precision = true_positive / (true_positive + false_positive + 1e-7)
    recall = true_positive / (true_positive + false_negative + 1e-7)
    
    return {
        'dice': dice.item(),
        'iou': iou.item(),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }


def calculate_metrics_multiclass(pred, target, num_classes=3):
    """
    Compute Dice, IoU, Accuracy, Precision, and Recall for multiclass predictions.
    
    Args:
        pred: Model prediction logits [B, C, H, W]
        target: Ground truth - either one-hot [B, C, H, W] or class indices [B, H, W]
        num_classes: Number of classes including background
    
    Returns:
        Dictionary with metrics for each class and their average
    """
    # Move tensors to CPU and convert to numpy for calculations
    pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities
    pred_indices = pred.argmax(dim=1).cpu().numpy()  # Get predicted class indices
    
    # Check if target is one-hot encoded or class indices
    if len(target.shape) == 4 and target.shape[1] > 1:  # One-hot encoded
        target_indices = target.argmax(dim=1).cpu().numpy()
    else:  # Already class indices
        target_indices = target.cpu().numpy()
    
    # Initialize metrics dictionary with per-class and average metrics
    metrics = {
        'dice': {},
        'iou': {},
        'accuracy': {},
        'precision': {},
        'recall': {}
    }
    
    # Calculate metrics for each class (starting from 1 to skip background)
    class_metrics = {k: [] for k in metrics.keys()}
    
    for cls in range(1, num_classes):  # Skip background class
        pred_mask = (pred_indices == cls)
        true_mask = (target_indices == cls)
        
        # True positives, false positives, false negatives
        tp = np.logical_and(pred_mask, true_mask).sum()
        fp = np.logical_and(pred_mask, ~true_mask).sum()
        fn = np.logical_and(~pred_mask, true_mask).sum()
        tn = np.logical_and(~pred_mask, ~true_mask).sum()
        
        # Calculate metrics
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        
        # Store per-class metrics
        class_name = f"class_{cls}"
        metrics['dice'][class_name] = float(dice)
        metrics['iou'][class_name] = float(iou)
        metrics['accuracy'][class_name] = float(accuracy)
        metrics['precision'][class_name] = float(precision)
        metrics['recall'][class_name] = float(recall)
        
        # Accumulate for averaging (only foreground classes)
        class_metrics['dice'].append(dice)
        class_metrics['iou'].append(iou)
        class_metrics['accuracy'].append(accuracy)
        class_metrics['precision'].append(precision)
        class_metrics['recall'].append(recall)
    
    # Add average metrics (over foreground classes)
    for k in metrics.keys():
        metrics[k]['avg'] = float(np.mean(class_metrics[k]))
    
    return metrics