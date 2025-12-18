#!/usr/bin/env python3
"""
Welding Region Extractor
Extracts welding regions from images using XML annotations.
Saves cropped and resized patches for PatchCore training.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from dotenv import find_dotenv, load_dotenv
from tqdm import tqdm

load_dotenv(find_dotenv())


class WeldExtractor:
    """Extract welding regions from images using XML annotations"""

    def __init__(self, target_short_side=224, margin=12, pad_multiple=32):
        """
        Initialize the extractor

        Args:
            target_short_side (int): Target size for the short side after scaling
            margin (int): Margin to add around bounding boxes for context
            pad_multiple (int): Pad to multiples of this value (ResNet stride)
        """
        self.target_short_side = target_short_side
        self.margin = margin
        self.pad_multiple = pad_multiple

    def extract_from_xml(self, image_path, xml_path):
        """
        Extract welding regions using XML annotations

        Args:
            image_path (str): Path to input image
            xml_path (str): Path to XML annotation file

        Returns:
            list: List of (patch, class_name, bbox) tuples
        """
        if not os.path.exists(xml_path):
            return []

        # Parse XML annotation
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return []

        patches = []

        # Extract each object
        for obj in root.findall("object"):
            # Get class name
            name_elem = obj.find("name")
            if name_elem is None:
                continue
            class_name = name_elem.text

            # Get bounding box
            bbox = obj.find("bndbox")
            if bbox is None:
                continue

            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Add margin and clamp to image boundaries
            img_height, img_width = image.shape[:2]
            xmin_margin = max(0, xmin - self.margin)
            ymin_margin = max(0, ymin - self.margin)
            xmax_margin = min(img_width, xmax + self.margin)
            ymax_margin = min(img_height, ymax + self.margin)

            # Extract patch with margin
            patch = image[ymin_margin:ymax_margin, xmin_margin:xmax_margin]
            if patch.size > 0:
                # Process patch with proper aspect ratio preservation
                patch_processed = self.process_patch(patch)
                patches.append((patch_processed, class_name, (xmin, ymin, xmax, ymax)))

        return patches

    def process_patch(self, patch):
        """
        Process patch with proper aspect ratio preservation and upscaling

        Args:
            patch (np.ndarray): Input patch

        Returns:
            np.ndarray: Processed patch
        """
        h, w = patch.shape[:2]

        # Calculate scale factor to make short side = target_short_side
        short_side = min(h, w)
        scale_factor = self.target_short_side / short_side

        # Calculate new dimensions
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # Resize keeping aspect ratio
        patch_resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Pad to multiples of pad_multiple (ResNet stride)
        pad_h = (
            (new_h + self.pad_multiple - 1) // self.pad_multiple
        ) * self.pad_multiple
        pad_w = (
            (new_w + self.pad_multiple - 1) // self.pad_multiple
        ) * self.pad_multiple

        # Create padded image
        if len(patch.shape) == 3:
            padded = np.zeros((pad_h, pad_w, patch.shape[2]), dtype=patch.dtype)
        else:
            padded = np.zeros((pad_h, pad_w), dtype=patch.dtype)

        # Place resized patch at top-left (no centering to avoid confusion)
        padded[:new_h, :new_w] = patch_resized

        return padded

    def process_dataset(self, input_dir, output_dir, is_training=True):
        """
        Process entire dataset and extract welding patches using XML annotations

        Args:
            input_dir (str): Directory containing images and XML files
            output_dir (str): Directory to save extracted patches
            is_training (bool): If True, saves goods to train/ and bads to test/
                               If False (validation), saves all to test/
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if is_training:
            # For training set: goods go to train/, bads go to test/
            train_good_dir = output_path / "train" / "good"
            train_bad_dir = output_path / "train" / "bad"
            train_good_dir.mkdir(parents=True, exist_ok=True)
            train_bad_dir.mkdir(parents=True, exist_ok=True)
        else:
            # For validation set: both goods and bads go to test/
            test_good_dir = output_path / "val" / "good"
            test_bad_dir = output_path / "val" / "bad"
            test_good_dir.mkdir(parents=True, exist_ok=True)
            test_bad_dir.mkdir(parents=True, exist_ok=True)

        # Process all images
        image_files = (
            list(input_path.glob("*.jpeg"))
            + list(input_path.glob("*.jpg"))
            + list(input_path.glob("*.png"))
        )

        total_patches = 0
        good_patches = 0
        bad_patches = 0

        for img_file in tqdm(image_files, desc="Extracting welding patches"):
            # Use XML annotations
            xml_file = img_file.with_suffix(".xml")
            patches = self.extract_from_xml(str(img_file), str(xml_file))

            for i, (patch, class_name, bbox) in enumerate(patches):
                patch_name = f"{img_file.stem}_patch_{i}.jpg"

                if is_training:
                    # Training set: goods → train/good, bads → test/bad
                    if class_name == "good_weld":
                        cv2.imwrite(str(train_good_dir / patch_name), patch)
                        good_patches += 1
                    elif class_name == "bad_weld":
                        cv2.imwrite(str(train_bad_dir / patch_name), patch)
                        bad_patches += 1
                else:
                    # Validation set: both → test/ (for evaluation)
                    if class_name == "good_weld":
                        cv2.imwrite(str(test_good_dir / patch_name), patch)
                        good_patches += 1
                    elif class_name == "bad_weld":
                        cv2.imwrite(str(test_bad_dir / patch_name), patch)
                        bad_patches += 1

                total_patches += 1

        print(f"\nExtraction completed!")
        print(f"Total patches extracted: {total_patches}")
        print(f"Good welds: {good_patches}")
        print(f"Bad welds: {bad_patches}")
        print(f"Patches saved to: {output_path}")


def main():
    """Main extraction pipeline"""

    # Configuration
    base_dir = Path(os.getenv("DATASET_DIR", ""))
    train_dir = base_dir / "train"
    val_dir = base_dir / "val"
    output_dir = Path(os.getenv("CNN_DATASET_DIR", ""))

    print("=== Welding Region Extraction ===")

    # Initialize extractor
    extractor = WeldExtractor(
        target_short_side=224,  # Short side = 224
        margin=12,  # 12px margin around each box
        pad_multiple=32,  # Pad to multiples of 32 (ResNet stride)
    )

    # Process training set
    if train_dir.exists():
        print(f"Processing training set: {train_dir}")
        extractor.process_dataset(
            input_dir=str(train_dir),
            output_dir=str(output_dir),
            is_training=True,  # Goods to train/, bads to test/
        )

    # Process validation set
    if val_dir.exists():
        print(f"Processing validation set: {val_dir}")
        extractor.process_dataset(
            input_dir=str(val_dir),
            output_dir=str(output_dir),
            is_training=False,  # Both to test/ for evaluation
        )


if __name__ == "__main__":
    main()
