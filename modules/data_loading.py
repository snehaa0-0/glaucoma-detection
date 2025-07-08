import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class RIMONEDataset(Dataset):
    def __init__(self, images, masks, transform=None, multiclass=False):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.multiclass = multiclass

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # Change from HWC to CHW format
        
        if self.multiclass:
            # For multiclass, mask should already be in correct format (H, W, C) with C=3
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()  # Change from HWC to CHW format
        else:
            # For binary segmentation, ensure mask is in correct shape
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=-1)
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()

        return image, mask

def load_rim_one_data(images_dir, masks_dir, img_size=256, mask_type='both', multiclass=False):
    IMG_HEIGHT = img_size
    IMG_WIDTH = img_size
    
    X, Y_cup, Y_disc = [], [], []
    image_paths = []

    print("Loading images...")
    # Get all image files
    for filename in sorted(os.listdir(images_dir)):
        if not (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.bmp')):
            continue
            
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[SKIPPED] Could not read image: {img_path}")
            continue
        
        # Process and store the image
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        X.append(img)
        
        # Store image path/name for mask matching
        image_paths.append(filename)

    print("Loading masks...")
    # For each image, find its corresponding cup and disc masks
    for img_name in image_paths:
        base_name = os.path.splitext(img_name)[0]  # removes extension
        
        # Find corresponding cup mask
        cup_found = False
        for mask_file in os.listdir(masks_dir):
            if mask_file.startswith(base_name) and "-Cup-" in mask_file:
                cup_mask_path = os.path.join(masks_dir, mask_file)
                cup_mask = cv2.imread(cup_mask_path, cv2.IMREAD_GRAYSCALE)
                cup_mask = cv2.resize(cup_mask, (IMG_WIDTH, IMG_HEIGHT))
                cup_mask = (cup_mask > 0).astype(np.float32)  # Convert to binary
                cup_mask = np.expand_dims(cup_mask, axis=-1)
                Y_cup.append(cup_mask)
                cup_found = True
                break
        
        if not cup_found:
            print(f"[WARNING] Cup mask not found for: {base_name}")
            Y_cup.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32))
        
        # Find corresponding disc mask
        disc_found = False
        for mask_file in os.listdir(masks_dir):
            if mask_file.startswith(base_name) and "-Disc-" in mask_file:
                disc_mask_path = os.path.join(masks_dir, mask_file)
                disc_mask = cv2.imread(disc_mask_path, cv2.IMREAD_GRAYSCALE)
                disc_mask = cv2.resize(disc_mask, (IMG_WIDTH, IMG_HEIGHT))
                disc_mask = (disc_mask > 0).astype(np.float32)  # Convert to binary
                disc_mask = np.expand_dims(disc_mask, axis=-1)
                Y_disc.append(disc_mask)
                disc_found = True
                break
        
        if not disc_found:
            print(f"[WARNING] Disc mask not found for: {base_name}")
            Y_disc.append(np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32))

    X = np.array(X)
    Y_cup = np.array(Y_cup)
    Y_disc = np.array(Y_disc)

    print(f"Loaded {len(X)} images, {len(Y_cup)} cup masks, and {len(Y_disc)} disc masks.")
    
    # Decide which masks to use based on configuration
    if mask_type == 'cup':
        print("Using only CUP masks for training")
        return X, Y_cup
    elif mask_type == 'disc':
        print("Using only DISC masks for training")
        return X, Y_disc
    else:  # 'both' - use both cup and disc masks combined
        print("Using COMBINED (cup+disc) masks for training")
        
        if multiclass:
            # Create proper multiclass masks with one-hot encoding
            # Shape will be (batch_size, height, width, 3) where:
            # Channel 0: Background (1 where neither cup nor disc)
            # Channel 1: Cup (1 where cup is present)
            # Channel 2: Disc excluding cup (1 where disc but not cup)
            
            # First, create empty 3-channel masks
            Y_multiclass = np.zeros((len(X), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
            
            # For each image
            for i in range(len(X)):
                # Extract masks
                cup = Y_cup[i].squeeze()  # Remove channel dimension
                disc = Y_disc[i].squeeze()
                
                # Channel 1: Cup
                Y_multiclass[i, :, :, 1] = cup
                
                # Channel 2: Disc excluding cup area (disc - cup)
                # This preserves rim area as separate class
                disc_only = np.logical_and(disc, np.logical_not(cup)).astype(np.float32)
                Y_multiclass[i, :, :, 2] = disc_only
                
                # Channel 0: Background (neither cup nor disc)
                background = np.logical_not(np.logical_or(cup, disc)).astype(np.float32)
                Y_multiclass[i, :, :, 0] = background
            
            print("Created multi-class masks with shape:", Y_multiclass.shape)
            return X, Y_multiclass
        else:
            # For binary segmentation, simply combine cup and disc
            Y_combined = np.maximum(Y_cup, Y_disc)
            return X, Y_combined