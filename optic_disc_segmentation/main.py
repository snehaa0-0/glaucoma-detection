# main.py - This is the main file to run
import os
import argparse
import torch
from modules.data_loading import load_rim_one_data, RIMONEDataset
from modules.model import AttentionUNet
from modules.training import train_val_test_and_visualize
from torch.utils.data import DataLoader

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='U-Net segmentation for RIM-ONE dataset')
    parser.add_argument('--data_dir', type=str, default='sample_data', help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--mask_type', type=str, default='both', choices=['cup', 'disc', 'both'], 
                        help='Type of mask to use (cup, disc, or both)')
    parser.add_argument('--img_size', type=int, default=256, help='Image size (height and width)')
    args = parser.parse_args()
    
    # Set up paths
    images_dir = os.path.join(args.data_dir, 'images')
    masks_dir = os.path.join(args.data_dir, 'masks')
    
    # Create directories for results
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Start training
    train_val_test_and_visualize(
        images_dir=images_dir,
        masks_dir=masks_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        mask_type=args.mask_type,
        img_size=args.img_size
    )

if __name__ == "__main__":
    main()