# WELD-DETECT: Industrial Welding Anomaly Detection

A two-stage deep learning pipeline for welding quality inspection combining YOLO object detection and ResNet-50 binary classification for automated good/bad weld classification.

## 🏗️ Architecture

**Two-Stage Detection Pipeline:**
- **Stage 1 - YOLO11**: Detects and localizes welding regions in images
- **Stage 2 - ResNet-50**: Binary classifier for weld quality (good/bad)

**Model Details:**
- **YOLO**: YOLOv11-m for real-time weld region detection
- **CNN**: Fine-tuned ResNet-50 with custom head for binary classification
- **Input Processing**: Letterbox resize (672×224) with aspect ratio preservation
- **Training Strategy**: Progressive unfreezing with early stopping

## 📦 Installation

### Prerequisites

```bash
# Python 3.8+ required
python --version
```

### Local Development

```bash
# Clone the repository
git clone https://github.com/Singh8899/anomaly-detection-welding-yolo-cnn.git
cd anomaly-detection-welding-yolo-cnn

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

```bash
# Copy example environment file
cp .env_example .env

# Edit .env with your paths
nano .env
```

**Required Environment Variables:**
```bash
DATASET_DIR=/path/to/downloaded_photos
YOLO_DATASET_DIR=/path/to/yolo/dataset
CNN_DATASET_DIR=/path/to/cnn/extracted_welds

# Training hyperparameters
EPOCHS=50
BATCH_SIZE=32
PATIENCE=8
```

## 🚀 Usage

### Complete Training Pipeline

Follow these steps in order for full pipeline training:

#### 1. Dataset Preparation

Organize your dataset with images and XML annotations:
```
downloaded_photos/
├── train/
│   ├── image1.jpeg
│   ├── image1.xml
│   └── ...
└── val/
    ├── image2.jpeg
    ├── image2.xml
    └── ...
```

#### 2. Format YOLO Dataset

```bash
python cnn_train/yolo/dataset_formatter.py
```

**Output**: Creates YOLO-compatible dataset structure with images and labels.

#### 3. Train YOLO Detector

```bash
python cnn_train/yolo/yolo_train.py
```

**Features:**
- Auto-detection of GPU/CPU
- Training with IoU=0.4, agnostic NMS
- Saves best model to `cnn_train/yolo/runs/welding_detection/`

**Configuration:**
- Model: YOLOv11-m
- Epochs: 40
- Image size: 896
- Batch: 16

#### 4. Extract Weld ROIs

```bash
python cnn_train/cnn/extract_welds.py
```

**Process:**
- Extracts welding patches from XML annotations
- Applies letterbox resize (224px short side)
- Adds 12px margin around bounding boxes
- Splits good/bad welds for training

#### 5. Train ResNet-50 Classifier

```bash
python cnn_train/cnn/welding_cnn.py --mode train
```

**Training Features:**
- Transfer learning from ImageNet weights
- Progressive layer unfreezing (stage 1-4 frozen initially)
- Data augmentation: flips, brightness, contrast, rotation
- Early stopping based on F1 score
- Metrics: ROC-AUC, PR-AUC, F1, accuracy

**Outputs:**
- Best model: `runs/best_model.pth`
- Training curves: `runs/training_history.png`
- Metrics plots: `runs/precision_recall_curve.png`

#### 6. Evaluate Model

```bash
python cnn_train/cnn/welding_cnn.py --mode val
```

**Evaluation Outputs:**
- Test metrics: `resnet_results/test_metrics.txt`
- Confusion matrix
- ROC/PR curves: `resnet_results/evaluation_results.png`

### Using Jupyter Notebook

For interactive training, use the provided notebook:

```bash
jupyter notebook cnn_train.ipynb
```

**Notebook includes:**
- Environment setup and GPU check
- Step-by-step pipeline execution
- Visualization of training progress

### Inference

```bash
python cnn_inference/predict_debug.py
```

**Features:**
- End-to-end pipeline (YOLO → CNN)
- Visualizes predictions vs ground truth
- Saves annotated images to `pred_results/`

**Example Usage:**
```python
from cnn_inference.predict_debug import ResnetInference

# Initialize inference pipeline
inference = ResnetInference()

# Load image
with open('image.jpg', 'rb') as f:
    image_bytes = f.read()

# Run prediction
results = inference.predict(
    image=image_bytes,
    yolo_threshold=0.5,
    cnn_threshold=0.5
)
```

## 📂 Project Structure

```
anomaly-detection-welding-yolo-cnn/
├── cnn_train/
│   ├── cnn/
│   │   ├── extract_welds.py          # ROI extraction from XML
│   │   └── welding_cnn.py            # ResNet-50 trainer/evaluator
│   └── yolo/
│       ├── dataset_formatter.py      # YOLO dataset preparation
│       ├── yolo_train.py             # YOLO training script
│       └── labels.txt                # Class names
├── cnn_inference/
│   ├── classes.py                    # Data models
│   ├── formatter.py                  # Visualization utilities
│   ├── predict_debug.py              # Inference + visualization
│   ├── predict_debug_grad.py         # Grad-CAM visualization
│   └── examples/                     # Sample images/XMLs
├── cnn_train.ipynb                   # Training notebook
├── requirements.txt                  # Python dependencies
├── .env_example                      # Environment template
├── LICENSE
└── README.md
```

## 🔧 Configuration

### YOLO Training Parameters

Edit in `cnn_train/yolo/yolo_train.py`:
```python
model_size='m'      # Model size: s, m, l, x
epochs=40           # Training epochs
imgsz=896           # Input image size
batch=16            # Batch size
iou=0.4             # IoU threshold
```

### CNN Training Parameters

Edit in `.env`:
```bash
EPOCHS=50           # Max training epochs
BATCH_SIZE=32       # Training batch size
PATIENCE=8          # Early stopping patience
```

Or modify in `cnn_train/cnn/welding_cnn.py`:
```python
target_size=(672, 224)    # Canvas size (H×W)
freeze_stages=3           # Initial frozen layers
learning_rate=1e-4        # Initial learning rate
weight_decay=1e-4         # L2 regularization
```

## 📊 Metrics & Evaluation

### Classification Metrics

- **ROC-AUC**: Area under ROC curve for model discrimination
- **PR-AUC**: Precision-Recall AUC (better for imbalanced data)
- **F1 Score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Handles class imbalance
- **Confusion Matrix**: Per-class performance breakdown

### Model Performance Tracking

Training automatically generates:
- Loss curves (train/val)
- AUC-ROC curves
- Precision-Recall curves
- F1 score progression
- Prediction distributions

## 🎯 Data Requirements

### Dataset Structure

**Training Set:**
- Images: JPEG/PNG format
- Annotations: Pascal VOC XML format
- Classes: `good_weld`, `bad_weld`

**XML Annotation Format:**
```xml
<annotation>
  <object>
    <name>good_weld</name>
    <bndbox>
      <xmin>100</xmin>
      <ymin>100</ymin>
      <xmax>200</xmax>
      <ymax>200</ymax>
    </bndbox>
  </object>
</annotation>
```

### Data Augmentation

**Training augmentations:**
- Horizontal flips (50%)
- Brightness adjustment (±20%)
- Contrast adjustment (±20%)
- Sharpness enhancement
- Small rotations (±5°)

**Validation/Test:**
- Only letterbox resize (no augmentation)

## 🛠️ Advanced Features

### Progressive Unfreezing

ResNet-50 training uses staged unfreezing:
1. **Epochs 1-5**: Freeze conv1, bn1, layer1-3
2. **Epoch 6+**: Unfreeze all layers, reduce LR by 10×

### Letterbox Resize

Preserves aspect ratio with padding:
- Scales to fit target size
- Pads with gray (114, 114, 114)
- Prevents distortion

### Class Balancing

Training set strategy:
- Good welds → `train/good/`
- Bad welds → `train/bad/`
- Validation set → `val/good/` and `val/bad/`

## 📝 Output Examples

### Training Output
```
=== Starting Training ===
Epoch 1/50
Training: 100%|████████| 125/125 [02:15<00:00, 0.92it/s]
Train - Loss: 0.4523, AUC: 0.8234
Validating: 100%|████████| 32/32 [00:15<00:00]
Val - Loss: 0.3821, AUC: 0.8756, PR-AUC: 0.8532, F1: 0.8421
New best F1: 0.8421
```

### Inference Output
```
Found 5 detections
class_0: 0.92 - bbox: [120, 340, 280, 420]
class_1: 0.78 - bbox: [450, 200, 610, 280]
Annotated image saved as: annotated_image.jpg
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size in .env
BATCH_SIZE=16
```

**2. Dataset Not Found**
```bash
# Check paths in .env
echo $DATASET_DIR
```

**3. Model Loading Error**
```python
# Ensure weights_only=False for older checkpoints
checkpoint = torch.load(path, map_location='cpu', weights_only=False)
```

## 📈 Performance Tips

1. **GPU Usage**: Enable CUDA for 10-20× speedup
2. **Batch Size**: Increase for faster training (if memory allows)
3. **Image Size**: Larger YOLO input (896) improves small defect detection
4. **Data Quality**: More annotated data improves generalization

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with clear description

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLO**: For state-of-the-art object detection
- **PyTorch**: Deep learning framework
- **torchvision**: Pretrained ResNet-50 models

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the repository maintainer.
