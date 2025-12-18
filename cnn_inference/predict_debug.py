import logging
import os
from formatter import draw_boxes

import cv2
import numpy as np
import torch
import torch.nn as nn
from classes import Detection, InferenceResponse
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO

output_dir = 'pred_results'
image_path = 'examples/20250730_190642_222398.jpeg'

logger = logging.getLogger(__name__)

IMAGE_SIZE = 896  # Default image size for inference

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
        self.backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
        
        # Replace the classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, 1)  # Single output for BCEWithLogitsLoss
    
    
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x).squeeze(1)  # Remove extra dimension for BCEWithLogitsLoss

class ResnetInference:
    def __init__(self):
        self.device = "cpu"
        self.model = YOLO("yolo.pt", verbose=False).to(self.device)
        
        # Load CNN model
        self.cnn_model = WeldResNet(pretrained=True)

        checkpoint = torch.load("cnn.pth", map_location=self.device, weights_only=False)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model.to(self.device).eval()
        target_size=(672, 224)  # (height, width) - must match training
        # CNN transform
        self.cnn_transform = transforms.Compose([
            LetterboxResize(target_size=target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image: bytes, yolo_threshold: float = 0.5, cnn_threshold = 0.5):
        """
        Make predictions on the input image using YOLO + CNN pipeline.

        Args:
            image: Input image as bytes
            yolo_threshold: Detection confidence yolo_threshold
            imgsz: Image size for inference
            square_img: Whether to make the image square

        Returns:
            Dictionary containing list of detections with bounding boxes, classes and scores
        """
        try:
            # Preprocess image
            image_np = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            
            # Run YOLO inference
            with torch.inference_mode():
                outputs = self.model(image_np, iou=0.4, agnostic_nms=True)

            # Process YOLO outputs and classify with CNN
            boxes = []
            
            if len(outputs) > 0 and hasattr(outputs[0], 'boxes'):
                detections = outputs[0].boxes
                
                # Filter by confidence threshold
                conf_mask = detections.conf > yolo_threshold
                
                if conf_mask.any():
                    # Extract filtered data
                    filtered_boxes = detections.xyxy[conf_mask]  # Bounding boxes in xyxy format
                    
                    # Convert to lists and classify each detection
                    for i in range(len(filtered_boxes)):
                        bbox = filtered_boxes[i].cpu().numpy().tolist()
                        
                        # Extract ROI for CNN classification
                        x1, y1, x2, y2 = [int(coord) for coord in bbox]
                        roi = image_np[y1:y2, x1:x2]
                        
                        # Classify with CNN
                        try:
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            roi_pil = Image.fromarray(roi_rgb)
                            roi_tensor = self.cnn_transform(roi_pil).unsqueeze(0).to(self.device)
                            
                            with torch.inference_mode():
                                output = self.cnn_model(roi_tensor)
                                probability = torch.sigmoid(output).item()
                                class_name = "class_0" if probability > cnn_threshold else "class_1"
                        except:
                            class_name = "unknown"
                            probability = 0.0
                        
                        boxes.append(
                            Detection(
                                bbox=bbox,
                                class_name=class_name,
                                score=probability
                            )
                        )

            return {"predictions": boxes}

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise e
            
resnetInference = ResnetInference()
# Read image as bytes
with open(image_path, 'rb') as f:
    image_bytes = f.read()

# Also read the original image for drawing boxes
original_image = cv2.imread(image_path)

output = resnetInference.predict(
    image=image_bytes,
    yolo_threshold=0.5,
    cnn_threshold=0.5
)
# Create InferenceResponse object
inference_response = InferenceResponse(predictions=output["predictions"])

# Draw boxes on the original image
annotated_image = draw_boxes(original_image, inference_response, inference_size=IMAGE_SIZE)

# Save the annotated image
output_filename = 'annotated_image.jpg'
complete_path = output_dir+"/"+output_filename
os.makedirs(output_dir, exist_ok=True)
cv2.imwrite(complete_path, annotated_image)
print(f"Annotated image saved as: {output_filename}")
print(f"Found {len(inference_response.predictions)} detections")


import os
import xml.etree.ElementTree as ET


def load_voc_annotations(xml_path):
    """
    Parse a Pascal-VOC style XML and return a list of dicts:
    [{'name': 'good_weld', 'bbox': [xmin, ymin, xmax, ymax]}, ...]
    """
    objects = []
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        bnd = obj.find('bndbox')
        xmin = int(float(bnd.find('xmin').text))
        ymin = int(float(bnd.find('ymin').text))
        xmax = int(float(bnd.find('xmax').text))
        ymax = int(float(bnd.find('ymax').text))
        objects.append({'name': name, 'bbox': [xmin, ymin, xmax, ymax]})
    return objects

def draw_gt_boxes(image_bgr, objects, color_good=(0, 255, 0), color_bad=(0, 0, 255)):
    """
    Draw GT boxes and labels on a copy of the image.
    - good_weld: green
    - bad_weld: red
    Falls back to blue for any other class.
    """
    img = image_bgr.copy()
    h, w = img.shape[:2]
    # scale thickness and font a bit with image size
    thickness = max(2, int(round(min(h, w) * 0.0025)))
    font_scale = max(0.5, min(1.5, min(h, w) / 900.0))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for obj in objects:
        name = obj['name']
        x1, y1, x2, y2 = obj['bbox']
        if name == 'good_weld':
            color = color_good
        elif name == 'bad_weld':
            color = color_bad
        else:
            color = (255, 128, 0)  # fallback: orange-ish

        # clamp to image bounds just in case
        x1 = max(0, min(x1, w - 1)); x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1)); y2 = max(0, min(y2, h - 1))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # label background
        label = name
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        th_full = th + baseline
        y_top = max(0, y1 - th_full - 2)
        cv2.rectangle(img, (x1, y_top), (x1 + tw + 4, y_top + th_full + 2), color, -1)
        cv2.putText(img, label, (x1 + 2, y_top + th + 1), font, font_scale, (255, 255, 255), max(1, thickness - 1), cv2.LINE_AA)

    return img


xml_path = os.path.splitext(image_path)[0] + '.xml'



if os.path.exists(xml_path):
    try:
        gt_objects = load_voc_annotations(xml_path)
        annotated_gt_image = draw_gt_boxes(original_image, gt_objects)
        gt_output_filename = 'annotated_image_gt.jpg'
        complete_path = output_dir+"/"+gt_output_filename
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(complete_path, annotated_gt_image)
        print(f"GT annotated image saved as: {gt_output_filename}")
        print(f"Found {len(gt_objects)} ground-truth boxes")
    except Exception as e:
        logger.error(f"Failed to draw GT boxes: {e}")
        print(f"Failed to draw GT boxes: {e}")
else:
    print(f"XML not found at: {xml_path}")
