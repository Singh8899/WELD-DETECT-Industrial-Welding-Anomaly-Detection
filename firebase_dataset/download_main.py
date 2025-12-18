import os
import random
import xml.etree.ElementTree as ET
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore, storage
from PIL import Image
from tqdm import tqdm

SEED = 69
PROCESSED_PREFIX = "processed_photos_3/"
ERRORS_PREFIX = "errors/"
PROCESSED_COLLECTION = "photos_3"
ERRORS_COLLECTION = "errors"
OUTPUT_ROOT = "downloaded_photos"
PROCESSED_VAL_RATIO = 0.3
ERRORS_VAL_RATIO = 0.2
MIN_PIN_LEN = 1
DUP_WINDOW_SECONDS = 300

random.seed(SEED)

cred = credentials.Certificate("secret.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {"storageBucket": "weldlabeling.firebasestorage.app"})

bucket = storage.bucket()
db = firestore.client()

train_folder = os.path.join(OUTPUT_ROOT, "train")
val_folder = os.path.join(OUTPUT_ROOT, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

def parse_timestamp(photo_id: str):
    try:
        if photo_id.isdigit() and len(photo_id) >= 13:
            return datetime.fromtimestamp(int(photo_id) / 1000)
        if photo_id.isdigit() and len(photo_id) == 10:
            return datetime.fromtimestamp(int(photo_id))
        if "_" in photo_id and len(photo_id.split("_")) == 3:
            date_part, time_part, micro_part = photo_id.split("_")
            return datetime(int(date_part[:4]), int(date_part[4:6]), int(date_part[6:8]),
                            int(time_part[:2]), int(time_part[2:4]), int(time_part[4:6]), int(micro_part))
        return datetime.fromisoformat(photo_id.replace("_", "-").replace("T", " "))
    except Exception:
        return None

def ensure_unique_name(folder: str, filename: str):
    name, ext = os.path.splitext(filename)
    candidate = filename
    counter = 1
    while os.path.exists(os.path.join(folder, candidate)):
        candidate = f"{name}_{counter}{ext}"
        counter += 1
    return candidate

def write_voc(blob, file_name, photo_id, pin_data, folder):
    img_path = os.path.join(folder, file_name)
    blob.download_to_filename(img_path)
    with Image.open(img_path) as img:
        width, height = img.size
        depth = len(img.getbands())

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(folder)
    ET.SubElement(annotation, "filename").text = file_name
    ET.SubElement(annotation, "path").text = img_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "weldLabel"
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    good = bad = 0
    for pin in pin_data:
        obj = ET.SubElement(annotation, "object")
        label = pin.get("label") or pin.get("current_label")
        label = "good_weld" if label in ["OK", "ACCETTABILE"] else "bad_weld"
        good += label == "good_weld"
        bad += label == "bad_weld"

        bndbox = ET.SubElement(obj, "bndbox")
        x_left = max(pin.get("x_left"), 0)
        x_right = min(pin.get("x_right"), width)
        y_top = max(pin.get("y_top"), 0)
        y_bottom = min(pin.get("y_bottom"), height)
        ET.SubElement(bndbox, "xmin").text = str(x_left)
        ET.SubElement(bndbox, "xmax").text = str(x_right)
        ET.SubElement(bndbox, "ymin").text = str(y_top)
        ET.SubElement(bndbox, "ymax").text = str(y_bottom)
        ET.SubElement(obj, "name").text = label

    xml_file = os.path.join(folder, f"{os.path.splitext(file_name)[0]}.xml")
    ET.ElementTree(annotation).write(xml_file, encoding="utf-8", xml_declaration=True)
    return good, bad

def fetch_processed_photos():
    blobs = list(bucket.list_blobs(prefix=PROCESSED_PREFIX))
    photos_ref = db.collection(PROCESSED_COLLECTION)
    photos = []
    for blob in tqdm(blobs, desc="Collecting processed"):
        file_name = os.path.basename(blob.name)
        if not file_name:
            continue
        photo_id, _ = os.path.splitext(file_name)
        doc = photos_ref.document(photo_id).get()
        if not doc.exists:
            continue
        data = doc.to_dict()
        if not data.get("processed", False):
            continue
        pins = data.get("annotations", [])
        if len(pins) < MIN_PIN_LEN:
            continue
        photos.append({
            "blob": blob,
            "file_name": file_name,
            "photo_id": photo_id,
            "pins": pins,
            "timestamp": parse_timestamp(photo_id),
            "annotation_count": len(pins),
        })

    photos.sort(key=lambda x: x["timestamp"] or datetime.min)
    filtered = []
    for i, current in enumerate(photos):
        if i == 0:
            filtered.append(current)
            continue
        prev = photos[i - 1]
        if (current["timestamp"] and prev["timestamp"] and
            abs((current["timestamp"] - prev["timestamp"]).total_seconds()) < DUP_WINDOW_SECONDS and
            current["annotation_count"] == prev["annotation_count"]):
            continue
        filtered.append(current)
    return filtered

def fetch_error_photos():
    blobs = list(bucket.list_blobs(prefix=ERRORS_PREFIX))
    random.shuffle(blobs)
    photos_ref = db.collection(ERRORS_COLLECTION)
    selected = []
    for blob in tqdm(blobs, desc="Collecting errors"):
        file_name = os.path.basename(blob.name)
        if not file_name:
            continue
        photo_id, _ = os.path.splitext(file_name)
        doc = photos_ref.document(photo_id).get()
        if not doc.exists:
            continue
        data = doc.to_dict()
        if not data.get("processed", False):
            continue
        pins = data.get("annotations", [])
        if len(pins) < MIN_PIN_LEN:
            continue
        if not any(pin.get("reviewed", False) for pin in pins):
            continue
        selected.append({"blob": blob, "file_name": file_name, "photo_id": photo_id, "pins": pins})
    return selected

def split_list(items, val_ratio):
    val_count = int(len(items) * val_ratio)
    return items[val_count:], items[:val_count]

def main():
    total_good = total_bad = total_photos = 0

    processed = fetch_processed_photos()
    random.shuffle(processed)
    train_p, val_p = split_list(processed, PROCESSED_VAL_RATIO)

    errors = fetch_error_photos()
    train_e, val_e = split_list(errors, ERRORS_VAL_RATIO)

    datasets = [
        ("train", train_folder, train_p + train_e),
        ("val", val_folder, val_p + val_e),
    ]

    for name, folder, items in datasets:
        bar = tqdm(items, desc=f"Writing {name}")
        for item in bar:
            unique_name = ensure_unique_name(folder, item["file_name"])
            g, b = write_voc(item["blob"], unique_name, item["photo_id"], item["pins"], folder)
            total_good += g
            total_bad += b
            total_photos += 1
            bar.set_postfix({"good": total_good, "bad": total_bad})

    print("✅ Download + merge complete")
    print(f"Photos saved: {total_photos}")
    print(f"good_weld: {total_good}, bad_weld: {total_bad}")
    print(f"Output root: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()