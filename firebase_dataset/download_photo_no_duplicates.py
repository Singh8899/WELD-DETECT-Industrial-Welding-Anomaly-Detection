import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
from tqdm import tqdm

cred = credentials.Certificate("secret.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'weldlabeling.firebasestorage.app'
})

bucket = storage.bucket()

random.seed(69)

download_folder = "downloaded_photos"
train_folder = os.path.join(download_folder, "train")
val_folder = os.path.join(download_folder, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# blobs = list(bucket.list_blobs(prefix="processed_photos_3/"))
blobs = list(bucket.list_blobs(prefix="errors/"))

db = firestore.client()

# photos_ref = db.collection("photos_3")
photos_ref = db.collection("errors")


# First, collect all photos with their metadata
photos_data = []
print("Collecting photos metadata...")
bar = tqdm(blobs)
total_pins = 0
for blob in bar:
    file_name = os.path.basename(blob.name)
    if not file_name:
        continue

    photo_id, _ = os.path.splitext(file_name)
    photo_doc = photos_ref.document(photo_id).get()
    if not photo_doc.exists:
        continue

    photo_data = photo_doc.to_dict()
    if not photo_data.get("processed", False):
        continue

    pin_data = photo_data.get("annotations", [])
    if len(pin_data) == 0:
        continue
    total_pins += len(pin_data)
    bar.set_postfix({"total_pins": f"{total_pins}"})
    # Extract timestamp from filename (photo_id)
    timestamp = None
    try:
        # Check if photo_id looks like a millisecond timestamp (13+ digits)
        if photo_id.isdigit() and len(photo_id) >= 13:
            # Convert milliseconds to seconds
            timestamp = datetime.fromtimestamp(int(photo_id) / 1000)
        elif photo_id.isdigit() and len(photo_id) == 10:
            # Standard Unix timestamp in seconds
            timestamp = datetime.fromtimestamp(int(photo_id))
        elif '_' in photo_id and len(photo_id.split('_')) == 3:
            # Handle format: YYYYMMDD_HHMMSS_microseconds (e.g., 20250717_181545_592272)
            parts = photo_id.split('_')
            date_part = parts[0]  # YYYYMMDD
            time_part = parts[1]  # HHMMSS
            microseconds = int(parts[2])  # microseconds
            
            # Parse date and time
            year = int(date_part[:4])
            month = int(date_part[4:6])
            day = int(date_part[6:8])
            hour = int(time_part[:2])
            minute = int(time_part[2:4])
            second = int(time_part[4:6])
            
            timestamp = datetime(year, month, day, hour, minute, second, microseconds)
        else:
            # Try parsing as ISO format or other common formats
            timestamp = datetime.fromisoformat(photo_id.replace('_', '-').replace('T', ' '))
    except (ValueError, TypeError, OverflowError, OSError):
        print(f"Warning: Could not parse timestamp from filename {photo_id}")
        timestamp = None
    
    photos_data.append({
        'blob': blob,
        'file_name': file_name,
        'photo_id': photo_id,
        'photo_data': photo_data,
        'pin_data': pin_data,
        'timestamp': timestamp,
        'annotation_count': len(pin_data)
    })

print(f"Collected {len(photos_data)} photos")

# Sort by timestamp to make duplicate detection easier
photos_data.sort(key=lambda x: x['timestamp'] if x['timestamp'] else datetime.min)

# Remove duplicates based on timestamp (<5 minutes) and same annotation count
filtered_photos = []
print("Removing duplicates...")
bar = tqdm(photos_data)
for i, current_photo in enumerate(bar):
    is_duplicate = False
    
    # Check if this is a duplicate of the previous photo
    if i > 0:
        prev_photo = photos_data[i-1]
        if (current_photo['timestamp'] and prev_photo['timestamp'] and 
            abs((current_photo['timestamp'] - prev_photo['timestamp']).total_seconds()) < 300 and  # 5 minutes = 300 seconds
            current_photo['annotation_count'] == prev_photo['annotation_count']):
            is_duplicate = True
            # print(f"Removing duplicate: {current_photo['photo_id']} (similar to {prev_photo['photo_id']})")
    
    if not is_duplicate:
        filtered_photos.append(current_photo)

print(f"After removing duplicates: {len(filtered_photos)} photos remaining")

# Shuffle the dataset
random.shuffle(filtered_photos)
print("Dataset shuffled")

# Split into train/val (70/30)
val_split = int(len(filtered_photos) * 0.3)
val_photos = filtered_photos[:val_split]
train_photos = filtered_photos[val_split:]

print(f"Train set: {len(train_photos)} photos")
print(f"Validation set: {len(val_photos)} photos")

# Process and save photos
yes_count = 0
no_count = 0
photo_count = 0

def process_photo(photo_info, folder):
    global yes_count, no_count, photo_count
    
    blob = photo_info['blob']
    file_name = photo_info['file_name']
    photo_id = photo_info['photo_id']
    photo_data = photo_info['photo_data']
    pin_data = photo_info['pin_data']
    
    file_path = os.path.join(folder, file_name)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(folder)
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
        label = pin.get("label")
        if label in ["OK", "ACCETTABILE"]:
            label = "good_weld"
            yes_count += 1
        else:
            no_count += 1
            label = "bad_weld"

        ET.SubElement(obj, "name").text = label
        bndbox = ET.SubElement(obj, "bndbox")
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
    return len(pin_data)

print("Processing training photos...")
total_train_pin = 0
bar = tqdm(train_photos)
for photo_info in bar:
    len_pin_data = process_photo(photo_info, train_folder)
    total_train_pin += len_pin_data
    bar.set_postfix({"total_train_pin": f"{total_train_pin}"})


print("Processing validation photos...")
total_val_pin = 0
bar = tqdm(val_photos)
for photo_info in bar:
    len_pin_data = process_photo(photo_info, val_folder)
    total_val_pin += len_pin_data
    bar.set_postfix({"total_val_pin": f"{total_val_pin}"})

print("Download delle foto e creazione dei file XML completati! 🎉")
print(f"Original photos collected: {len(photos_data)}")
print(f"After duplicate removal: {len(filtered_photos)}")
print(f"Train photos: {len(train_photos)}")
print(f"Validation photos: {len(val_photos)}")
print(f"Yes count: {yes_count}")
print(f"No count: {no_count}")
print(f"Photos saved: {photo_count}")
