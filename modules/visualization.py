import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
import torch

# Define colormap: 0 = background (black), 1 = cup (blue), 2 = disc (red)
label_colormap = ListedColormap(['black', 'blue', 'red'])

def plot_training_history(history):
    """Plot training and validation loss and dice scores."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_losses'], label='Train Loss')
    plt.plot(history['val_losses'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Dice Coefficient Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig('results/plots/training_history.png')
    plt.close()

def save_prediction_examples(images, masks, predictions, num_examples=5, filename_suffix="unet"):
    """Save side-by-side examples of original images, masks, and predictions."""
    num_to_save = min(num_examples, len(images))
    
    plt.figure(figsize=(15, 5 * num_to_save))
    for i in range(num_to_save):
        # Original image
        plt.subplot(num_to_save, 3, i*3 + 1)
        plt.imshow(images[i])
        plt.title('Original Image')
        plt.axis('off')
        
        # Ground truth mask (colored)
        plt.subplot(num_to_save, 3, i*3 + 2)
        # Convert multi-channel mask to label map for visualization
        if len(masks[i].shape) == 3 and masks[i].shape[0] == 3:
            # If mask is in channel-first format (C, H, W) with 3 channels
            mask_vis = np.argmax(masks[i], axis=0)
        else:
            # If mask is already in proper format or needs a simple squeeze
            mask_vis = masks[i].squeeze()
        
        plt.imshow(mask_vis, cmap=label_colormap, vmin=0, vmax=2)
        plt.title('Ground Truth')
        plt.axis('off')
        
        # Predicted mask (colored)
        plt.subplot(num_to_save, 3, i*3 + 3)
        # Ensure prediction is in correct format
        pred_vis = predictions[i]
        plt.imshow(pred_vis, cmap=label_colormap, vmin=0, vmax=2)
        plt.title('Prediction')
        plt.axis('off')
    
    os.makedirs('results/predictions', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"results/predictions/examples_{filename_suffix}.png")
    plt.close()

def plot_side_by_side_comparison(original, gt_mask, pred_mask, filename):
    """Plot and save an original image, ground truth mask, and predicted mask side by side."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Convert multi-channel mask to label map for visualization
    if len(gt_mask.shape) == 3 and gt_mask.shape[0] == 3:
        gt_mask_vis = np.argmax(gt_mask, axis=0)
    else:
        gt_mask_vis = gt_mask.squeeze()
    
    axes[1].imshow(gt_mask_vis, cmap=label_colormap, vmin=0, vmax=2)
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap=label_colormap, vmin=0, vmax=2)
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    plt.tight_layout()
    os.makedirs('results/comparisons', exist_ok=True)
    plt.savefig(f'results/comparisons/{filename}')
    plt.close(fig)

def visualize_predictions(model, dataloader, device, num_samples=3):
    """Run the model on a few samples and visualize predictions."""
    model.eval()
    images_collected = []
    masks_collected = []
    preds_collected = []

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            for i in range(images.size(0)):
                if len(images_collected) >= num_samples:
                    break
                img = images[i].cpu().permute(1, 2, 0).numpy()
                mask = masks[i].cpu().numpy()
                pred = preds[i]
                
                images_collected.append(img)
                masks_collected.append(mask)
                preds_collected.append(pred)
            
            if len(images_collected) >= num_samples:
                break

    save_prediction_examples(images_collected, masks_collected, preds_collected, num_examples=num_samples)