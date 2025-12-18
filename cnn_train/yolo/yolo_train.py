#!/usr/bin/env python3
"""
YOLO Training Script for Welding Detection
Single class detection for identifying welding regions in images.
"""

import os
from pathlib import Path

import torch
import yaml
from dotenv import find_dotenv, load_dotenv
from ultralytics import YOLO

load_dotenv(find_dotenv())


def create_yolo_config(dataset_path, labels_file):
    """
    Create YOLO configuration file (data.yaml)
    
    Args:
        dataset_path (str): Path to dataset directory
        labels_file (str): Path to labels.txt file
    """
    # Read class names from labels file
    with open(labels_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(class_names),  # number of classes
        'names': class_names
    }
    
    config_path = os.path.join(dataset_path, 'data.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created YOLO config file: {config_path}")
    return config_path

def train_yolo_model(config_path, model_size='m', epochs=80, imgsz=896, batch=16):
    """
    Train YOLO model for welding detection
    
    Args:
        config_path (str): Path to data.yaml configuration file
        model_size (str): Model size ('s', 'm', 'l', 'x')
        epochs (int): Number of training epochs
        imgsz (int): Image size for training
        batch (int): Batch size
    """
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Initialize YOLO model
    model_name = f'yolo11{model_size}.pt'
    model = YOLO(model_name)
    
    print(f"Starting YOLO training with {model_name}")
    print(f"Configuration: epochs={epochs}, imgsz={imgsz}, batch={batch}")
    
    

    # Train the model
    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        iou=0.4,
        agnostic_nms=True,
        save=True,
        project='cnn_train/yolo/runs',
        name='welding_detection'
    )
    
    return results

def validate_model(model_path, config_path):
    """
    Validate trained model
    
    Args:
        model_path (str): Path to trained model weights
        config_path (str): Path to data.yaml configuration file
    """
    model = YOLO(model_path)
    
    # Validate the model
    results = model.val(
        data=config_path,
        split='val',
        save_json=True,
        save_txt=True,
        save_conf=True,
        iou=0.4,
        agnostic_nms=True
    )
    
    return results

def main():
    """Main training pipeline"""
    
    # Paths configuration
    dataset_path = Path(os.getenv("YOLO_DATASET_DIR"))
    labels_file = dataset_path.parent / 'labels.txt'
    
    print("=== YOLO Welding Detection Training ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Labels file: {labels_file}")
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please run dataset_formatter.py first to create the dataset.")
        return
    
    # Create YOLO configuration
    config_path = create_yolo_config(dataset_path, labels_file)
    
    # Train the model
    try:
        results = train_yolo_model(
            config_path=config_path,
            model_size='m',  # small model for good balance of speed and accuracy
            epochs=40,
            imgsz=896,
            batch=16
        )
        
        print("Training completed successfully!")
        
        # Get best model path
        best_model_path = 'runs/detect/welding_detection/weights/best.pt'
        
        if os.path.exists(best_model_path):
            print(f"Best model saved at: {best_model_path}")
            
        else:
            print(f"Warning: Best model not found at expected path: {best_model_path}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        return

if __name__ == "__main__":
    main()