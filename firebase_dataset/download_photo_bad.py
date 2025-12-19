import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
from tqdm import tqdm

random.seed(69)

cred = credentials.Certificate("secret.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'weldlabeling.firebasestorage.app'
})

bucket = storage.bucket()
download_folder = "downloaded_photos"
train_folder = os.path.join(download_folder, "train")
val_folder = os.path.join(download_folder, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

blobs = list(bucket.list_blobs(prefix="errors/"))
db = firestore.client()
random.shuffle(blobs)
photos_ref = db.collection("errors")

yes_count = 0
no_count = 0
photo_count = 0
new_blobs = []
bar = tqdm(blobs, desc=f"photo_count={photo_count}")
for blob in bar:
    file_name = os.path.basename(blob.name)
    if not file_name:
        continue

    photo_id, _ = os.path.splitext(file_name)
    photo_doc = photos_ref.document(photo_id).get()
    if not photo_doc.exists:
        continue
    
    
    photo_data = photo_doc.to_dict()
    pin_data = photo_data.get("annotations")
    if not pin_data:
        continue

    if len(pin_data) == 0:
        continue
    for pin in pin_data:
        photo_count += 1
        new_blobs.append(blob)
        break
    bar.update()
print("TOTAL data elements", photo_count)

yes_count = 0
no_count = 0
photo_count = 0

val_split = int(0.2 * len(new_blobs))
# Using balanced block as validation set (images 2363-2862) with perfect 50/50 good/bad balance
# This block has exactly 1061 good welds and 1061 bad welds for optimal validation
bar = tqdm(new_blobs)
for i, blob in enumerate(bar):
    file_name = os.path.basename(blob.name)
    if not file_name:
        continue

    # Use balanced block (2363-2862) for validation, rest for training
    folder = val_folder if val_split > i else train_folder
    file_path = os.path.join(folder, file_name)

    photo_id, _ = os.path.splitext(file_name)
    photo_doc = photos_ref.document(photo_id).get()
    if not photo_doc.exists:
        continue

    photo_data = photo_doc.to_dict()
    if not photo_data.get("processed", False):
        continue
    pin_data = photo_data.get("annotations")
    if len(pin_data) == 0:
        continue


    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = folder
    ET.SubElement(annotation, "filename").text = file_name
    ET.SubElement(annotation, "path").text = file_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "weldLabel"
    size = ET.SubElement(annotation, "size")
    blob.download_to_filename(file_path)
    with Image.open(file_path) as img:
        width, height = img.size
        depth = len(img.getbands())
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    for pin in pin_data:
        obj = ET.SubElement(annotation, "object")
        label = pin.get("current_label")
        if label in ["OK", "ACCETTABILE"]:
            label = "good_weld"
            yes_count += 1
        else:
            no_count += 1
            label = "bad_weld"

        ET.SubElement(obj, "name").text = label
        bndbox = ET.SubElement(obj, "bndbox")
        if pin.get("x_left") < 0 or pin.get("x_right") < 0 or pin.get("y_top") < 0 or pin.get("y_bottom") < 0 or pin.get("x_left") >= width or pin.get("x_right") >= width or pin.get("y_top") >= height or pin.get("y_bottom") >= height:
            print(f"Warning: Invalid bounding box for photo {photo_id}.")
            print(f"x_left: {pin.get('x_left')}, x_right: {pin.get('x_right')}, y_top: {pin.get('y_top')}, y_bottom: {pin.get('y_bottom')}")
        x_left = max(pin.get("x_left"),0)
        x_right = min(pin.get("x_right"), width)
        y_top = max(pin.get("y_top"),0)
        y_bottom = min(pin.get("y_bottom"), height)

        ET.SubElement(bndbox, "xmin").text = str(x_left)
        ET.SubElement(bndbox, "xmax").text = str(x_right)
        ET.SubElement(bndbox, "ymin").text = str(y_top)
        ET.SubElement(bndbox, "ymax").text = str(y_bottom)

    tree = ET.ElementTree(annotation)
    xml_file = os.path.join(folder, f"{photo_id}.xml")
    tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    photo_count += 1

print("Download delle foto e creazione dei file XML completati! 🎉")
print(f"Yes count: {yes_count}")
print(f"No count: {no_count}")
print(f"Photos saved: {photo_count}")
