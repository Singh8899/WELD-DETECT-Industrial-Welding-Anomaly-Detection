import logging
import os
from formatter import draw_boxes

import cv2
import numpy as np
import torch
import torch.nn as nn
from classes import InferenceResponse
from PIL import Image
from pytorch_grad_cam import (AblationCAM, EigenCAM, FullGrad, GradCAM,
                              GradCAMPlusPlus, HiResCAM, ScoreCAM, XGradCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms
from ultralytics import YOLO

image_path = 'examples/20250721_154817_360058.jpeg'
output_dir = 'grad_results'

logger = logging.getLogger(__name__)

IMAGE_SIZE = 896  # Default image size for inference

class LetterboxResize:
    """Enhanced letterbox resize for optimal quality"""
    
    def __init__(self, target_size=(224, 672)):
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
        
    
    def forward(self, x, return_probs=False):
        """Forward pass
        
        Args:
            x (tensor): Input image batch
        """
        return  self.backbone(x)

class GradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        # single logit output: shape [B]
        logits = self.model.backbone(x).squeeze(1)  # raw logits before sigmoid
        # Construct 2-class logits: [logit for class0, logit for class1]
        return torch.cat([-logits.unsqueeze(1), logits.unsqueeze(1)], dim=1)

class ResnetInference:
    def __init__(self):
        self.device = "cuda"
        self.model = YOLO("yolo.pt", verbose=False).to(self.device)
        
        # Load CNN model
        self.cnn_model = WeldResNet(pretrained=True)


        checkpoint = torch.load("cnn.pth", map_location=self.device)
        self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
        self.cnn_model.to(self.device).eval()
        self.target_layers = [self.cnn_model.backbone.layer4[-1]]
        target_size=(224, 672)
        # CNN transform
        self.cnn_transform = transforms.Compose([
            LetterboxResize(target_size=target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.wrapped = GradCAMWrapper(self.cnn_model)

    def predict(self, image: bytes, yolo_threshold: float = 0.5, cnn_threshold = 0.8):
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
                        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                        roi_pil = Image.fromarray(roi_rgb)
                        roi_tensor = self.cnn_transform(roi_pil).unsqueeze(0).to(self.device)
                        targets = [ClassifierOutputTarget(0)]
                        # GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
                        with GradCAM(model=self.wrapped, target_layers=self.target_layers) as cam:
                            grayscale_cam = cam(input_tensor=roi_tensor, targets=targets) 
                            print(grayscale_cam.shape)

                            grayscale_cam = grayscale_cam[0, :]
                             # ✅ Convert to float32 in [0,1]
                            roi_rgb_float = roi_rgb.astype(np.float32) / 255.0  
                            grayscale_cam_resized = cv2.resize(
                                            grayscale_cam,
                                            (roi_rgb_float.shape[1], roi_rgb_float.shape[0])  # width, height
                            )

                            visualization = show_cam_on_image(roi_rgb_float, grayscale_cam_resized, use_rgb=True)
                            output_filename = f"gradcam_overlay_{i}.jpg"
                            complete_path = output_dir+"/"+output_filename
                            os.makedirs(output_dir, exist_ok=True)
                            cv2.imwrite(complete_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
                        

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
    cnn_threshold=0.6
)
# Create InferenceResponse object
inference_response = InferenceResponse(predictions=output["predictions"])

# Draw boxes on the original image
annotated_image = draw_boxes(original_image, inference_response, inference_size=IMAGE_SIZE)

print(f"Annotated images saved in: {output_dir}")
print(f"Found {len(inference_response.predictions)} detections")
