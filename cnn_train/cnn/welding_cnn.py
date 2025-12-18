#!/usr/bin/env python3
"""
ResNet-50 Binary Classifier for Welding Anomaly Detection
Fine-tuned ResNet-50 for binary classification of weld ROIs (good/bad)
"""

import argparse
import copy
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import find_dotenv, load_dotenv
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

load_dotenv(find_dotenv())

# Define the pipeline once
welding_transform = A.Compose([
    A.ShiftScaleRotate(
        shift_limit=0.1, 
        scale_limit=0.1, 
        rotate_limit=0, # We handle rotation separately below if you want
        border_mode=cv2.BORDER_CONSTANT, 
        value=114, # Gray padding if we shift out of bounds
        p=0.8
    ),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, p=0.5),        # Slight rotation
    A.CLAHE(clip_limit=2.0, p=1.0),   # Enhance crack visibility
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # Simulate sensor noise
    A.RandomBrightnessContrast(p=0.5),
])
""" welding_transform = A.Compose([
    # 1. GEOMETRIC (Safe for welds)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=128, p=0.5),
    
    # 2. LIGHTING & CONTRAST (Critical for cracks)
    # RandomBrightnessContrast covers both your separate PIL calls
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    
    # 3. TEXTURE & SENSOR NOISE
    # OneOf picks exactly one from the list, or none.
    # Great for mixing different types of "camera quality" issues.
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
    ], p=0.5),

]) """

class LetterboxResize:
    """Enhanced letterbox resize for optimal quality"""
    
    def __init__(self, target_size=(672, 224)):
        self.target_size = target_size
    
    def __call__(self, image):
        """Single high-quality resize operation"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scale to fit in target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        # Single high-quality resize with LANCZOS (best for downscaling)
        if scale < 1.0:  # Downscaling
            interpolation = cv2.INTER_AREA  # Best for downscaling
        else:  # Upscaling
            interpolation = cv2.INTER_CUBIC  # Best for upscaling
            
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=interpolation)
        
        # Create canvas and center image
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return Image.fromarray(canvas)

class WeldAugmentation:
    def __call__(self, image):
        """
        Input:  PIL Image or Numpy Array
        Output: Transformed Numpy Array (ready for Tensor conversion)
        """
        # Albumentations expects a numpy array (OpenCV format)
        image_np = np.array(image)
        
        # Apply the transform
        augmented = welding_transform(image=image_np)["image"]
        
        return augmented

class WeldingDataset(Dataset):
    """Dataset for welding patch images with binary classification"""
    
    def __init__(self, data_dir, transform=None, class_type='both', is_training=False, elements_per_class=None, balanced=False):
        """
        Initialize dataset
        
        Args:
            data_dir (str): Directory containing welding patches
            transform: Torchvision transforms
            class_type (str): 'good', 'bad', or 'both'
            is_training (bool): Whether this is training data (for augmentation)
            elements_per_class (int or None): Max number of elements per class to load
            balanced (boolean): to balance good and bad weld elements 
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.class_type = class_type
        self.is_training = is_training
        self.elements_per_class = elements_per_class
        self.balanced = balanced
        
        self.image_paths = []
        self.labels = []
        random.seed(69)
        def load_class(class_name, label):
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                return [], []
            
            paths = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpeg')) + list(class_dir.glob('*.png'))
            
            if self.elements_per_class is not None:
                # Shuffle and sample
                random.shuffle(paths)
                paths = paths[:self.elements_per_class]
            
            return paths, [label] * len(paths)
        
        if class_type == 'both':
            # Load both classes
            good_paths, good_labels = load_class('good', 0)
            bad_paths, bad_labels = load_class('bad', 1)
            
            
            # Balance classes if requested
            if self.balanced and len(good_paths) > 0 and len(bad_paths) > 0:
                min_samples = min(len(good_paths), len(bad_paths))
                print(f"Balancing dataset: using {min_samples} samples per class")
                
                # Shuffle and sample
                random.shuffle(good_paths)
                random.shuffle(bad_paths)
                good_paths = good_paths[:min_samples]
                bad_paths = bad_paths[:min_samples]
                good_labels = good_labels[:min_samples]
                bad_labels = bad_labels[:min_samples]
            
            # Combine both classes
            self.image_paths.extend(good_paths)
            self.image_paths.extend(bad_paths)
            self.labels.extend(good_labels)
            self.labels.extend(bad_labels)
            print(f"Found {len(good_paths)} images in {self.data_dir/'good'}")
            print(f"Found {len(bad_paths)} images in {self.data_dir/'bad'}")
        else:
            label = 0 if class_type == 'good' else 1
            self.image_paths, self.labels = load_class(class_type, label)
            if len(self.image_paths) == 0:
                raise ValueError(f"No images found in {self.data_dir/class_type}")
            
            print(f"Found {len(self.image_paths)} images in {self.data_dir/class_type}")
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.data_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.float32),
            'image_path': str(img_path)
        }

class WeldResNet(nn.Module):
    """ResNet-50 binary classifier for weld ROI classification"""
    
    def __init__(self, pretrained=True, freeze_stages=2):
        """
        Initialize ResNet-50 classifier
        
        Args:
            pretrained (bool): Use ImageNet pretrained weights
            freeze_stages (int): Number of initial stages to freeze (0-4)
        """
        super(WeldResNet, self).__init__()
        
        # Load pretrained ResNet-50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        # total params
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Total parameters: {total_params:,}")

        # trainable params
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable parameters before freezing: {trainable_params:,}")
        # Replace the classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1) # Single output for BCEWithLogitsLoss
        
        # Freeze initial stages if specified
        if freeze_stages > 0:
            self.freeze_stages(freeze_stages)
    
    def freeze_stages(self, num_stages):
        """Freeze the first num_stages of the network"""
        stages = [
            [self.backbone.conv1, self.backbone.bn1], # Stage 0
            [self.backbone.layer1],  # stage 1
            [self.backbone.layer2],  # stage 2
            [self.backbone.layer3],  # stage 3
            [self.backbone.layer4],  # stage 5
        ]
        
        for i in range(min(num_stages, len(stages))):
            for module_list in stages[i]:
                if isinstance(module_list, nn.Module):
                    for param in module_list.parameters():
                        param.requires_grad = False
                else:
                    # It's a list of modules
                    for module in module_list:
                        for param in module.parameters():
                            param.requires_grad = False
        # trainable params
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Trainable parameters after freezing: {trainable_params:,}")
    
    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x).squeeze(1)  # Remove extra dimension for BCEWithLogitsLoss

class WeldingClassifier:
    """ResNet-50 based welding binary classifier"""
    
    def __init__(self, train_data_dir, val_data_dir=None, test_data_dir=None, device=None, 
                 target_size=(672, 224)):
        """
        Initialize welding classifier
        
        Args:
            train_data_dir (str): Directory containing training welding patches
            val_data_dir (str): Directory containing validation welding patches
            test_data_dir (str): Directory containing test welding patches
            device: Torch device
            target_size (tuple): Target canvas size (height, width)
            short_side (int): Target short side length
        """
        self.train_data_dir = Path(train_data_dir)
        self.val_data_dir = Path(val_data_dir) if val_data_dir else None
        self.test_data_dir = Path(test_data_dir) if test_data_dir else None
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.target_size = target_size
        
        # Data transforms
        self.train_transform = transforms.Compose([
            LetterboxResize(target_size=target_size),
            WeldAugmentation(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            LetterboxResize(target_size=target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        negative_weight = torch.tensor(
            [float(os.getenv("CLASS_GOOD_DISTRIB", 0.5))/float(os.getenv("CLASS_BAD_DISTRIB", 0.5))], 
            device=self.device
        )

        # Model and training components
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=negative_weight)
        self.best_f1 = 0.0
        
    def build_dataloaders(self, batch_size=32):
        """Build train, validation, and test dataloaders"""
        dataloaders = {}
        
        # Training dataloader
        balanced = bool(os.getenv("BALANCED_DATA"))
        train_dataset = WeldingDataset(
            self.train_data_dir,
            transform=self.train_transform,
            class_type='both',
            is_training=True,
            elements_per_class=None,
            balanced=balanced
        )
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        
        # Validation dataloader
        if self.val_data_dir and self.val_data_dir.exists():
            val_dataset = WeldingDataset(
                self.val_data_dir,
                transform=self.val_transform,
                class_type='both',
                is_training=False,
                balanced=False
            )
            
            dataloaders['val'] = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
        
        # Test dataloader
        if self.test_data_dir and self.test_data_dir.exists():
            test_dataset = WeldingDataset(
                self.test_data_dir,
                transform=self.val_transform,
                class_type='both',
                is_training=False
            )
            dataloaders['test'] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=8,
                pin_memory=True
            )
        
        return dataloaders
    
    def initialize_model(self, freeze_stages=3):
        """Initialize the ResNet-50 model"""
        self.model = WeldResNet(pretrained=True, freeze_stages=freeze_stages).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-4,
            weight_decay=1e-4
        )
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} trainable parameters")
    
    def train_epoch(self, dataloader, val_dataloader=None, val_interval_steps=None, 
                    global_step=0, save_dir=None, best_val_f1=0.0, best_model_state=None, 
                    patience_counter=0, patience=5):
        """Train for one epoch with optional step-based validation"""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with tqdm(dataloader, desc="Training") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                global_step += 1

                # For metrics
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().detach().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update tqdm with real-time loss
                pbar.set_postfix({
                    "batch_loss": loss.item(),
                    "avg_loss": running_loss / (len(all_preds) // labels.size(0)),
                    "step": global_step
                })
                
                # Step-based validation
                if val_interval_steps is not None and val_dataloader is not None:
                    if global_step % val_interval_steps == 0:
                        print(f"\n--- Validation at step {global_step} ---")
                        val_loss, val_auc, val_pr_auc, val_f1, acc, prec, rec = self.validate(
                            val_dataloader, save_dir
                        )
                        print(f"Val - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, PR-AUC: {val_pr_auc:.4f}, "
                              f"ACC: {acc:.4f}, PREC: {prec:.4f}, REC: {rec:.4f}, F1: {val_f1:.4f}")
                        
                        # Check for improvement
                        if val_f1 > best_val_f1:
                            best_val_f1 = val_f1
                            patience_counter = 0
                            best_model_state = copy.deepcopy(self.model.state_dict())
                            
                            # Save best model
                            torch.save({
                                'model_state_dict': best_model_state,
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'global_step': global_step,
                                'best_f1': best_val_f1,
                            }, Path(save_dir) / 'best_model.pth')
                            
                            print(f"New best F1: {best_val_f1:.4f} at step {global_step}")
                        else:
                            patience_counter += 1
                            print(f"No improvement. Patience: {patience_counter}/{patience}")
                        
                        # Check early stopping
                        if patience_counter >= patience:
                            print(f"Early stopping triggered at step {global_step}")
                            return (running_loss / len(dataloader), 
                                    roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0,
                                    global_step, best_val_f1, best_model_state, patience_counter, True)
                        
                        # Return to training mode
                        self.model.train()
        
        avg_loss = running_loss / len(dataloader)
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0
        
        return avg_loss, auc, global_step, best_val_f1, best_model_state, patience_counter, False
    
    def validate(self, dataloader, checkpoints_dir):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                probs = torch.sigmoid(outputs)
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = running_loss / len(dataloader)
        all_preds = np.asarray(all_preds).ravel()
        all_labels = np.asarray(all_labels).ravel()

        # Calculate metrics
        if len(set(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_preds)
            pr_auc = average_precision_score(all_labels, all_preds)
            
            binary_preds = (all_preds >= 0.5).astype(int)
            acc  = accuracy_score(all_labels, binary_preds)
            prec = precision_score(all_labels, binary_preds, zero_division=0)
            rec  = recall_score(all_labels, binary_preds, zero_division=0)

            best_f1 = f1_score(all_labels, binary_preds)

            # Precision-Recall curve
            precisions, recalls, _ = precision_recall_curve(all_labels, binary_preds)

            outdir = Path(checkpoints_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            # ---- Precision-Recall Curve ----
            # PR curve with marker at the (rec, prec) given by threshold 0.5
            plt.figure()
            plt.plot(recalls, precisions, label=f'PR AUC = {pr_auc:.4f}')
            plt.scatter(rec, prec, label=f'F1@0.5={best_f1:.4f}', zorder=3)
            plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve')
            plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(outdir / "precision_recall_curve.png", dpi=300); plt.close()

            # Distribution of predictions
            plt.figure()
            plt.hist([all_preds[all_labels == 0], all_preds[all_labels == 1]],
                    bins=30, stacked=True, label=['good (0)', 'bad (1)'], alpha=0.7)
            plt.xlabel("Predicted Probability"); plt.ylabel("Count"); plt.title("Distribution of Predictions")
            plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
            plt.savefig(outdir / "prediction_distribution.png", dpi=300); plt.close()
        else:
            auc = pr_auc = best_f1 = acc = prec = rec = 0.0
            
        return avg_loss, auc, pr_auc, best_f1, acc, prec, rec
    
    def train(self, epochs=30, batch_size=32, patience=5, save_dir='./checkpoints', val_interval_steps=100):
        """
        Train the model with step-based validation and early stopping
        
        Args:
            epochs (int): Maximum number of epochs
            batch_size (int): Batch size
            patience (int): Early stopping patience
            save_dir (str): Directory to save checkpoints
            val_interval_steps (int): Run validation every N steps
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Build dataloaders
        dataloaders = self.build_dataloaders(batch_size)
        
        if 'train' not in dataloaders:
            raise ValueError("Training data not found")
        
        if 'val' not in dataloaders:
            raise ValueError("Validation data not found")
        
        # Initialize model
        self.initialize_model()
        
        # Training history
        history = {
            'train_loss': [], 'train_auc': [],
            'val_loss': [], 'val_auc': [], 'val_pr_auc': [], 'val_f1': [], "val_acc":[], "val_prec":[], "val_rec":[]
        }
        
        best_val_f1 = 0.0
        patience_counter = 0
        best_model_state = None
        global_step = 0
        
        print(f"\n=== Starting Training ===")
        print(f"Epochs: {epochs}, Batch Size: {batch_size}, Patience: {patience}")
        print(f"Validation Interval: Every {val_interval_steps} steps")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Training with step-based validation
            train_loss, train_auc, global_step, best_val_f1, best_model_state, patience_counter, early_stop = \
                self.train_epoch(
                    dataloaders['train'],
                    val_dataloader=dataloaders['val'],
                    val_interval_steps=val_interval_steps,
                    global_step=global_step,
                    save_dir=save_dir,
                    best_val_f1=best_val_f1,
                    best_model_state=best_model_state,
                    patience_counter=patience_counter,
                    patience=patience
                )
            self.best_f1 = best_val_f1
            
            if early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            history['train_loss'].append(train_loss)
            history['train_auc'].append(train_auc)
            
            print(f"Train - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}, Total Steps: {global_step}")
            
            # Unfreeze after a few epochs for fine-tuning
            if epoch == 5:
                print("Unfreezing all layers for fine-tuning...")
                self.model.unfreeze_all()
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= 0.1
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\nLoaded best model with F1: {self.best_f1:.4f}")
        
        # Save final model and history
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'history': history
        }, save_path / 'final_model.pth')
        
        # Plot training history
        self.plot_training_history(history, save_path)
        
        return history
    
    def plot_training_history(self, history, save_dir):
        """Plot and save training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        if 'val_loss' in history and history['val_loss']:
            axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC
        axes[0, 1].plot(history['train_auc'], label='Train AUC')
        if 'val_auc' in history and history['val_auc']:
            axes[0, 1].plot(history['val_auc'], label='Val AUC')
        axes[0, 1].set_title('AUC-ROC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # PR-AUC
        if 'val_pr_auc' in history and history['val_pr_auc']:
            axes[1, 0].plot(history['val_pr_auc'], label='Val PR-AUC')
            axes[1, 0].set_title('Precision-Recall AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('PR-AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # F1 Score
        if 'val_f1' in history and history['val_f1']:
            axes[1, 1].plot(history['val_f1'], label='Val F1')
            axes[1, 1].set_title('F1 Score')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('F1')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate(self, weight_path, test_data_dir=None, save_dir=None):
        """
        Evaluate model on test data
        
        Args:
            test_data_dir (str): Directory containing test images (if not provided, uses self.test_data_dir)
            save_dir (str): Directory to save results
        """

        self.initialize_model(freeze_stages=0)
        checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        
        # Use provided test_data_dir or default
        eval_data_dir = test_data_dir if test_data_dir else self.test_data_dir
        if not eval_data_dir or not Path(eval_data_dir).exists():
            raise ValueError(f"Test data directory not found: {eval_data_dir}")
        
        # Build test dataloader
        test_dataset = WeldingDataset(
            eval_data_dir,
            transform=self.val_transform,
            class_type='both',
            is_training=False
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print("Running evaluation on test data...")
        
        self.model.eval()
        all_probs, all_labels, all_paths = [], [], []
        
        with torch.inference_mode():
            for batch in tqdm(test_dataloader, desc="Evaluating"):
                images = batch['image'].to(self.device)
                labels = batch['label']
                paths = batch['image_path']
                
                outputs = self.model(images)
                probs = torch.sigmoid(outputs)
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
        
        # Convert to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Make predictions using best threshold
        predictions = (all_probs > 0.5).astype(int)
        
        # Calculate metrics
        auc_score = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
        f1 = f1_score(all_labels, predictions)
        
        print(f"\n=== Evaluation Results ===")
        print(f"AUC-ROC Score: {auc_score:.4f}")
        print(f"PR-AUC Score: {pr_auc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Threshold used: 0.5")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(all_labels, predictions, target_names=['Good', 'Bad']))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, predictions)
        print("\nConfusion Matrix:")
        print("Predicted:  Good  Bad")
        print(f"Good:       {cm[0,0]:4d}  {cm[0,1]:3d}")
        print(f"Bad:        {cm[1,0]:4d}  {cm[1,1]:3d}")
        
        # Save results
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save metrics
            with open(save_path / 'test_metrics.txt', 'w') as f:
                f.write(f"AUC-ROC Score: {auc_score:.4f}\n")
                f.write(f"PR-AUC Score: {pr_auc:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n")
                f.write(f"Threshold: 0.5\n")
                f.write(f"Confusion Matrix:\n")
                f.write(f"True\\Pred  Good  Bad\n")
                f.write(f"Good      {cm[0,0]:4d}  {cm[0,1]:3d}\n")
                f.write(f"Bad       {cm[1,0]:4d}  {cm[1,1]:3d}\n")
            
            # Plot results
            self.plot_evaluation_results(all_labels, all_probs, predictions, save_path)
            
            print(f"Results saved to: {save_path}")
        
        return {
            'auc_roc': auc_score,
            'pr_auc': pr_auc,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'probabilities': all_probs,
            'labels': all_labels,
            'paths': all_paths
        }
    
    def plot_evaluation_results(self, labels, probs, predictions, save_dir):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Score distribution
        good_scores = probs[labels == 0]
        bad_scores = probs[labels == 1]
        
        axes[0, 0].hist(good_scores, bins=30, alpha=0.7, label='Good Welds', color='green', density=True)
        axes[0, 0].hist(bad_scores, bins=30, alpha=0.7, label='Bad Welds', color='red', density=True)
        axes[0, 0].set_xlabel('Probability')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Distribution of Predictions')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # ROC Curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_score = roc_auc_score(labels, probs)
        
        axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precisions, recalls, _ = precision_recall_curve(labels, probs)
        pr_auc = average_precision_score(labels, probs)
        
        axes[1, 0].plot(recalls, precisions, label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion Matrix
        cm = confusion_matrix(labels, predictions)
        im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axes[1, 1].set_title('Confusion Matrix')
        tick_marks = np.arange(2)
        axes[1, 1].set_xticks(tick_marks)
        axes[1, 1].set_yticks(tick_marks)
        axes[1, 1].set_xticklabels(['Good', 'Bad'])
        axes[1, 1].set_yticklabels(['Good', 'Bad'])
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            axes[1, 1].text(j, i, format(cm[i, j], 'd'),
                           horizontalalignment="center",
                           color="white" if cm[i, j] > thresh else "black")
        
        axes[1, 1].set_ylabel('True Label')
        axes[1, 1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def load_checkpoint(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Initialize model if not already done
        if self.model is None:
            self.initialize_model()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_f1 = checkpoint.get('best_f1', 0.0)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Best F1: {self.best_f1:.4f}, Threshold: 0.5")
        
        return checkpoint.get('history', {})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train","val"])
    args = parser.parse_args()
    dataset_dir = Path(os.getenv("CNN_DATASET_DIR"))
    train_dir = dataset_dir / 'train'
    val_dir = dataset_dir / 'val'
    results_dir = dataset_dir.parent / 'resnet_results'
    checkpoints_dir = dataset_dir.parent / 'runs' 
    weight_file_name = "best_model.pth"
    weight_path = checkpoints_dir / weight_file_name

    # Initialize classifier
    classifier = WeldingClassifier(
        train_data_dir=str(train_dir),
        val_data_dir=str(val_dir),
        test_data_dir=None,
        target_size=(672, 224),  # Height x Width - good for weld seams
    )
    if args.mode == "train":
        print("=== Training ResNet-50 Welding Classifier ===")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print("Using train/validation split (no separate test set)")

        # Training configuration
        val_steps = os.getenv("VAL_STEPS")
        if not val_steps:
            raise ValueError("VAL_STEPS environment variable must be set")
            
        training_config = {
            'epochs': int(os.getenv("EPOCHS")),
            'batch_size': int(os.getenv("BATCH_SIZE")),
            'patience':int(os.getenv("PATIENCE")),
            'save_dir': str(checkpoints_dir),
            'val_interval_steps': int(val_steps)
        }

        print(f"\nTraining Configuration:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")

        # Start training
        print(f"\n=== Starting Training ===")
        history = classifier.train(**training_config)
    else:
        classifier.evaluate(weight_path, test_data_dir=str(val_dir), save_dir=str(results_dir))

if __name__ == "__main__":
    main()
