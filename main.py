import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from optic_disc_segmentation.modules.model import AttentionUNet


class GlaucomaFeatureExtractor:
    """Extract robust features from segmentation masks for glaucoma detection"""
    
    def __init__(self):
        self.feature_names = [
            'cdr_area_ratio', 'cdr_vertical_ratio', 'cdr_horizontal_ratio',
            'cup_area', 'disc_area', 'rim_area',
            'cup_circularity', 'disc_circularity',
            'cup_disc_center_distance', 'aspect_ratio_cup', 'aspect_ratio_disc'
        ]
    
    def extract_features(self, mask):
        """
        Extract features from segmentation mask
        Args:
            mask: numpy array where 0=background, 1=cup, 2=disc
        Returns:
            dict: Dictionary of features
        """
        features = {}
        
        # Create binary masks
        cup_mask = (mask == 1).astype(np.uint8)
        disc_mask = (mask >= 1).astype(np.uint8)  # cup + disc
        
        # Basic area calculations
        cup_area = np.sum(cup_mask)
        disc_area = np.sum(disc_mask)
        rim_area = disc_area - cup_area
        
        features['cup_area'] = cup_area
        features['disc_area'] = disc_area
        features['rim_area'] = rim_area
        
        # Cup-to-disc ratios
        if disc_area > 0:
            features['cdr_area_ratio'] = cup_area / disc_area
        else:
            features['cdr_area_ratio'] = 0
        
        # Vertical and horizontal CDR using bounding boxes
        cup_contours, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        disc_contours, _ = cv2.findContours(disc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if cup_contours and disc_contours:
            # Get largest contours
            cup_contour = max(cup_contours, key=cv2.contourArea)
            disc_contour = max(disc_contours, key=cv2.contourArea)
            
            # Bounding rectangles
            cup_bbox = cv2.boundingRect(cup_contour)
            disc_bbox = cv2.boundingRect(disc_contour)
            
            # Vertical CDR
            cup_height = cup_bbox[3]
            disc_height = disc_bbox[3]
            features['cdr_vertical_ratio'] = cup_height / (disc_height + 1e-6)
            
            # Horizontal CDR
            cup_width = cup_bbox[2]
            disc_width = disc_bbox[2]
            features['cdr_horizontal_ratio'] = cup_width / (disc_width + 1e-6)
            
            # Aspect ratios
            features['aspect_ratio_cup'] = cup_width / (cup_height + 1e-6)
            features['aspect_ratio_disc'] = disc_width / (disc_height + 1e-6)
            
            # Circularity (4π*area/perimeter²)
            cup_perimeter = cv2.arcLength(cup_contour, True)
            disc_perimeter = cv2.arcLength(disc_contour, True)
            
            features['cup_circularity'] = 4 * np.pi * cup_area / (cup_perimeter**2 + 1e-6)
            features['disc_circularity'] = 4 * np.pi * disc_area / (disc_perimeter**2 + 1e-6)
            
            # Distance between cup and disc centers
            cup_moments = cv2.moments(cup_contour)
            disc_moments = cv2.moments(disc_contour)
            
            if cup_moments['m00'] > 0 and disc_moments['m00'] > 0:
                cup_center = (cup_moments['m10']/cup_moments['m00'], cup_moments['m01']/cup_moments['m00'])
                disc_center = (disc_moments['m10']/disc_moments['m00'], disc_moments['m01']/disc_moments['m00'])
                
                features['cup_disc_center_distance'] = np.sqrt(
                    (cup_center[0] - disc_center[0])**2 + 
                    (cup_center[1] - disc_center[1])**2
                )
            else:
                features['cup_disc_center_distance'] = 0
        else:
            # Default values if contours not found
            features.update({
                'cdr_vertical_ratio': 0,
                'cdr_horizontal_ratio': 0,
                'aspect_ratio_cup': 1,
                'aspect_ratio_disc': 1,
                'cup_circularity': 0,
                'disc_circularity': 0,
                'cup_disc_center_distance': 0
            })
        
        return features

class GlaucomaDataset(Dataset):
    """Dataset for glaucoma detection training"""
    
    def __init__(self, image_paths, labels, segmentation_model, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.segmentation_model = segmentation_model
        self.transform = transform
        self.feature_extractor = GlaucomaFeatureExtractor()
        
        # Set segmentation model to eval mode
        self.segmentation_model.eval()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
        
        # Get segmentation mask
        with torch.no_grad():
            mask_pred = self.segmentation_model(image_tensor.unsqueeze(0))
            mask = torch.argmax(mask_pred, dim=1).squeeze().cpu().numpy()
        
        # Extract features
        features = self.feature_extractor.extract_features(mask)
        feature_vector = np.array([features[name] for name in self.feature_extractor.feature_names])
        
        label = self.labels[idx]
        
        return {
            'image': image_tensor,
            'mask': torch.tensor(mask, dtype=torch.float32),
            'features': torch.tensor(feature_vector, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

class GlaucomaClassifier(nn.Module):
    """Neural network classifier for glaucoma detection"""
    
    def __init__(self, num_features=11):
        super(GlaucomaClassifier, self).__init__()
        
        self.feature_classifier = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # Binary classification
        )
    
    def forward(self, features):
        return self.feature_classifier(features)

class GlaucomaDetectionPipeline:
    """Complete pipeline for glaucoma detection"""
    
    def __init__(self, segmentation_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load segmentation model
        self.segmentation_model = AttentionUNet(n_channels=3, n_classes=3).to(self.device)
        state_dict = torch.load(segmentation_model_path, map_location=self.device)
        self.segmentation_model.load_state_dict(state_dict)
        self.segmentation_model.eval()
            
        # Initialize feature extractor
        self.feature_extractor = GlaucomaFeatureExtractor()
        
        # Initialize classifiers
        self.nn_classifier = None
        self.rf_classifier = None
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, image_paths, labels):
        """Prepare dataset for training"""
        dataset = GlaucomaDataset(image_paths, labels, self.segmentation_model, self.transform)
        return dataset
    
    def train_neural_classifier(self, train_dataset, val_dataset, epochs=50, batch_size=32):
        """Train neural network classifier"""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.nn_classifier = GlaucomaClassifier(len(self.feature_extractor.feature_names))
        self.nn_classifier.to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.nn_classifier.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        # Training loop
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.nn_classifier.train()
            epoch_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.nn_classifier(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            # Validation phase
            self.nn_classifier.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    outputs = self.nn_classifier(features)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = val_correct / val_total
            avg_loss = epoch_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.nn_classifier.state_dict(), 'best_glaucoma_classifier.pth')
            
            scheduler.step()
            
            print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}')
        
        return train_losses, val_accuracies
    
    def train_random_forest(self, train_dataset):
        """Train Random Forest classifier"""
        # Extract features and labels
        features_list = []
        labels_list = []
        
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            features_list.append(sample['features'].numpy())
            labels_list.append(sample['label'].numpy())
        
        X_train = np.array(features_list)
        y_train = np.array(labels_list)
        
        # Train Random Forest
        self.rf_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        self.rf_classifier.fit(X_train, y_train)
        
        return self.rf_classifier
    
    def predict_single_image(self, image_path):
        """Predict glaucoma for a single image"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get segmentation mask
        with torch.no_grad():
            mask_pred = self.segmentation_model(image_tensor)
            mask = torch.argmax(mask_pred, dim=1).squeeze().cpu().numpy()
        
        # Extract features
        features = self.feature_extractor.extract_features(mask)
        feature_vector = np.array([features[name] for name in self.feature_extractor.feature_names])
        
        # Predictions
        predictions = {}
        
        # Neural network prediction
        if self.nn_classifier:
            self.nn_classifier.eval()
            with torch.no_grad():
                features_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                nn_output = self.nn_classifier(features_tensor)
                nn_prob = F.softmax(nn_output, dim=1)
                predictions['neural_network'] = {
                    'probability': nn_prob[0][1].item(),  # Probability of glaucoma
                    'prediction': 'Glaucoma' if nn_prob[0][1] > 0.5 else 'Normal'
                }
        
        # Random Forest prediction
        if self.rf_classifier:
            rf_prob = self.rf_classifier.predict_proba(feature_vector.reshape(1, -1))
            predictions['random_forest'] = {
                'probability': rf_prob[0][1],  # Probability of glaucoma
                'prediction': 'Glaucoma' if rf_prob[0][1] > 0.5 else 'Normal'
            }
        
        return predictions, features, mask
    
    def evaluate_model(self, test_dataset):
        """Evaluate model performance"""
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                if self.nn_classifier:
                    outputs = self.nn_classifier(features)
                    probabilities = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        # Calculate metrics
        auc_score = roc_auc_score(all_labels, all_probabilities)
        report = classification_report(all_labels, all_predictions, target_names=['Normal', 'Glaucoma'])
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        return {
            'auc_score': auc_score,
            'classification_report': report,
            'confusion_matrix': conf_matrix
        }

def load_dataset_from_csv(csv_path, image_dir):
    """Load dataset from CSV file with your structure"""
    df = pd.read_csv(csv_path)
    
    # Create full image paths
    image_paths = [os.path.join(image_dir, filename) for filename in df['Filename']]
    
    # Extract labels and other info
    labels = df['Glaucoma'].values
    expert_cdrs = df['ExpCDR'].values
    eyes = df['Eye'].values
    sets = df['Set'].values
    
    return {
        'image_paths': image_paths,
        'labels': labels,
        'expert_cdrs': expert_cdrs,
        'eyes': eyes,
        'sets': sets,
        'df': df
    }

def compare_with_expert_cdr(predicted_cdr, expert_cdr):
    """Compare predicted CDR with expert measurements"""
    correlation = np.corrcoef(predicted_cdr, expert_cdr)[0, 1]
    mae = np.mean(np.abs(predicted_cdr - expert_cdr))
    rmse = np.sqrt(np.mean((predicted_cdr - expert_cdr)**2))
    
    return {
        'correlation': correlation,
        'mae': mae,
        'rmse': rmse
    }

# Main execution
if __name__ == "__main__":
    # Load your dataset - UPDATE THESE PATHS
    data = load_dataset_from_csv('glaucoma.csv', 'data/')  # Update paths as needed
    
    # Split by Set column (A for train, B for test)
    train_mask = data['sets'] == 'A'
    test_mask = data['sets'] == 'B'
    
    train_images = [data['image_paths'][i] for i in range(len(data['image_paths'])) if train_mask[i]]
    train_labels = data['labels'][train_mask]
    train_expert_cdrs = data['expert_cdrs'][train_mask]
    
    test_images = [data['image_paths'][i] for i in range(len(data['image_paths'])) if test_mask[i]]
    test_labels = data['labels'][test_mask]
    test_expert_cdrs = data['expert_cdrs'][test_mask]
    
    # Initialize pipeline - UPDATE SEGMENTATION MODEL PATH
    pipeline = GlaucomaDetectionPipeline(r'optic_disc_segmentation\models\attention_unet_final.pth')  # Update path
    
    # Prepare datasets
    train_dataset = pipeline.prepare_data(train_images, train_labels)
    test_dataset = pipeline.prepare_data(test_images, test_labels)
    
    # Train models
    print("Training Neural Network...")
    train_losses, val_accuracies = pipeline.train_neural_classifier(train_dataset, test_dataset)
    
    print("Training Random Forest...")
    pipeline.train_random_forest(train_dataset)
    
    # Evaluate performance
    print("Evaluating model...")
    results = pipeline.evaluate_model(test_dataset)
    print(f"AUC Score: {results['auc_score']:.4f}")
    print(f"Classification Report:\n{results['classification_report']}")
    
    # Compare predicted CDR with expert CDR
    predicted_cdrs = []
    for img_path in test_images:
        _, features, _ = pipeline.predict_single_image(img_path)
        predicted_cdrs.append(features['cdr_area_ratio'])
    
    cdr_comparison = compare_with_expert_cdr(predicted_cdrs, test_expert_cdrs)
    print(f"\nCDR Comparison with Expert:")
    print(f"Correlation: {cdr_comparison['correlation']:.4f}")
    print(f"MAE: {cdr_comparison['mae']:.4f}")
    print(f"RMSE: {cdr_comparison['rmse']:.4f}")
    
    # Example prediction on single image
    if test_images:
        sample_image = test_images[0]
        predictions, features, mask = pipeline.predict_single_image(sample_image)
        print(f"\nSample prediction for {sample_image}:")
        print(f"Predictions: {predictions}")
        print(f"Predicted CDR: {features['cdr_area_ratio']:.4f}")
        print(f"Expert CDR: {test_expert_cdrs[0]:.4f}")
        print(f"Actual label: {'Glaucoma' if test_labels[0] == 1 else 'Normal'}")