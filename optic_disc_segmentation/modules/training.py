# modules/training.py
import os
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from modules.visualization import visualize_predictions


from sklearn.model_selection import train_test_split
from modules.metrics import DiceLoss
from modules.data_loading import load_rim_one_data, RIMONEDataset
from modules.model import AttentionUNet
from modules.metrics import combo_loss, dice_coefficient, calculate_metrics, calculate_metrics_multiclass
from modules.visualization import save_prediction_examples, plot_training_history

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, fold, device, patience=10, num_classes=3):
    train_losses = []
    val_losses = []
    train_dice_scores = []
    val_dice_scores = []

    best_val_dice = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_dice = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            # Calculate dice score for monitoring
            batch_dice = dice_coefficient(outputs, masks, num_classes=num_classes)

            train_loss += loss.item()
            train_dice += batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice
            progress_bar.set_postfix({'loss': loss.item(), 'dice': batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice})

        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_losses.append(train_loss)
        train_dice_scores.append(train_dice)

        model.eval()
        val_loss = 0
        val_dice = 0

        # Validation loop
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for images, masks in progress_bar:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                # Calculate dice score for monitoring
                batch_dice = dice_coefficient(outputs, masks, num_classes=num_classes)

                val_loss += loss.item()
                val_dice += batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice
                progress_bar.set_postfix({'loss': loss.item(), 'dice': batch_dice.item() if isinstance(batch_dice, torch.Tensor) else batch_dice})

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_losses.append(val_loss)
        val_dice_scores.append(val_dice)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # Save best model based on validation dice score
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            patience_counter = 0
            best_model_state = model.state_dict()
            # Make sure the models directory exists
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), f"models/unet_fold{fold}.pth")
            print(f"Saved model with improved validation Dice: {best_val_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dice': train_dice_scores,
        'val_dice': val_dice_scores
    }
    
    # Make sure the directory exists
    os.makedirs("plots", exist_ok=True)
    plot_training_history(history)

    return model, history

def evaluate_model(model, val_loader, fold, device, num_classes=3):
    model.eval()
    all_metrics = []
    
    # For visualization
    example_images = []
    example_masks = []
    example_outputs = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            
            # Process each image in batch individually for detailed metrics
            for i in range(images.size(0)):
                # Get metrics for this sample
                metrics = calculate_metrics_multiclass(
                    outputs[i:i+1], masks[i:i+1], num_classes=num_classes
                )
                all_metrics.append(metrics)
                
                # Save some examples for visualization
                if len(example_images) < 5:
                    # Convert tensors to numpy for visualization
                    img = images[i].cpu().permute(1, 2, 0).numpy()
                    
                    # For mask and output, get class indices for visualization
                    if masks[i].shape[0] == num_classes:  # One-hot mask
                        mask = masks[i].argmax(dim=0).cpu().numpy()
                    else:  # Already class indices
                        mask = masks[i].cpu().numpy()
                        
                    output = outputs[i].argmax(dim=0).cpu().numpy()
                    
                    example_images.append(img)
                    example_masks.append(mask)
                    example_outputs.append(output)

    # Make sure the directory exists
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    save_prediction_examples(example_images, example_masks, example_outputs, num_examples=5, filename_suffix=f"_fold{fold}")

    # Aggregate metrics from all samples
    aggregated_metrics = {}
    
    # Get metrics for cup (class 1) and disc (class 2)
    cup_dice = [m['dice']['class_1'] for m in all_metrics]
    disc_dice = [m['dice']['class_2'] for m in all_metrics]
    cup_iou = [m['iou']['class_1'] for m in all_metrics]
    disc_iou = [m['iou']['class_2'] for m in all_metrics]
    
    # Calculate average metrics
    aggregated_metrics = {
        'cup_dice_mean': np.mean(cup_dice),
        'cup_dice_std': np.std(cup_dice),
        'disc_dice_mean': np.mean(disc_dice),
        'disc_dice_std': np.std(disc_dice),
        'cup_iou_mean': np.mean(cup_iou),
        'cup_iou_std': np.std(cup_iou),
        'disc_iou_mean': np.mean(disc_iou),
        'disc_iou_std': np.std(disc_iou),
        'avg_dice_mean': np.mean([m['dice']['avg'] for m in all_metrics]),
        'avg_dice_std': np.std([m['dice']['avg'] for m in all_metrics]),
        'avg_iou_mean': np.mean([m['iou']['avg'] for m in all_metrics]),
        'avg_iou_std': np.std([m['iou']['avg'] for m in all_metrics]),
    }

    print(f"Fold {fold} Results:")
    print(f"  Cup Dice: {aggregated_metrics['cup_dice_mean']:.4f} ± {aggregated_metrics['cup_dice_std']:.4f}")
    print(f"  Disc Dice: {aggregated_metrics['disc_dice_mean']:.4f} ± {aggregated_metrics['disc_dice_std']:.4f}")
    print(f"  Cup IoU: {aggregated_metrics['cup_iou_mean']:.4f} ± {aggregated_metrics['cup_iou_std']:.4f}")
    print(f"  Disc IoU: {aggregated_metrics['disc_iou_mean']:.4f} ± {aggregated_metrics['disc_iou_std']:.4f}")
    print(f"  Average Dice: {aggregated_metrics['avg_dice_mean']:.4f} ± {aggregated_metrics['avg_dice_std']:.4f}")
    print(f"  Average IoU: {aggregated_metrics['avg_iou_mean']:.4f} ± {aggregated_metrics['avg_iou_std']:.4f}")

    return aggregated_metrics

def train_val_test_and_visualize(images_dir, masks_dir, batch_size=8, epochs=100, learning_rate=1e-4, mask_type='both', img_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X, Y = load_rim_one_data(images_dir, masks_dir, img_size, mask_type, multiclass=True)

    # Split: 70% train, 15% val, 15% test
    X_trainval, X_test, Y_trainval, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.1765, random_state=42)  # 15% of total

    # Create output dirs
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Datasets & Loaders
    train_loader = DataLoader(RIMONEDataset(X_train, Y_train, multiclass=True), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(RIMONEDataset(X_val, Y_val, multiclass=True), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(RIMONEDataset(X_test, Y_test, multiclass=True), batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    model = AttentionUNet(n_channels=3, n_classes=3).to(device)
    criterion = lambda pred, target: combo_loss(pred, target, weights=[0.2, 0.3, 0.5])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    model, _ = train_model(model, train_loader, val_loader, criterion, optimizer, epochs, 1, device, num_classes=3)

    # Save model
    torch.save(model.state_dict(), "models/attention_unet_final.pth")

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, fold=1, device=device, num_classes=3)

    # Print results
    print("\n🧪 Final Test Metrics:")
    for key in test_metrics:
        if key != 'fold':
            print(f"  {key.replace('_', ' ').title()}: {test_metrics[key]:.4f}")

    # Save test results
    with open('results/test_results.txt', 'w') as f:
        f.write("Final Test Evaluation Metrics\n\n")
        for key in test_metrics:
            if key != 'fold':
                f.write(f"{key.replace('_', ' ').title()}: {test_metrics[key]:.4f}\n")

    # Visualize predictions
    print("\n🖼️ Sample Predictions from Test Set:")
    visualize_predictions(model, test_loader, device, num_samples=3)

    return model, test_metrics