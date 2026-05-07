"""
Pelagix — Data Loader
======================
Step 1 of the Data Science Pipeline: Data Collection & Loading.
Auto-detects RUOD dataset format (VOC XML or YOLO TXT) and loads
images with their annotations into structured DataFrames.
"""

import os
import glob
import xml.etree.ElementTree as ET
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path


def _find_dataset_dirs(dataset_path):
    """
    Auto-detect the internal structure of the RUOD dataset.
    Returns (images_dir, annotations_dir, format_type).
    
    Handles common structures:
      - images/ + annotations/ (VOC)
      - images/ + labels/     (YOLO)
      - train/images + train/labels (YOLO split)
      - Flat directory with mixed .jpg + .xml or .txt
    """
    if not os.path.exists(dataset_path):
        return None, None, None

    # Candidate directory names
    img_candidates = ["images", "JPEGImages", "train/images", "imgs", "image", "RUOD_pic"]
    ann_candidates_voc = ["annotations", "Annotations", "xmls", "xml", "RUOD_ANN"]
    ann_candidates_yolo = ["labels", "train/labels", "txts", "label"]

    images_dir = None
    annotations_dir = None
    fmt = None

    # Search for image directory
    for candidate in img_candidates:
        p = os.path.join(dataset_path, candidate)
        if os.path.isdir(p):
            images_dir = p
            break

    # Try VOC format first
    for candidate in ann_candidates_voc:
        p = os.path.join(dataset_path, candidate)
        if os.path.isdir(p):
            xml_files = glob.glob(os.path.join(p, "*.xml"))
            if xml_files:
                annotations_dir = p
                fmt = "voc"
                break

    # Try YOLO format
    if fmt is None:
        for candidate in ann_candidates_yolo:
            p = os.path.join(dataset_path, candidate)
            if os.path.isdir(p):
                txt_files = glob.glob(os.path.join(p, "*.txt"))
                if txt_files:
                    annotations_dir = p
                    fmt = "yolo"
                    break

    # Try COCO JSON format
    if fmt is None:
        for candidate in ann_candidates_voc: # Check the same annotation folders for JSON
            p = os.path.join(dataset_path, candidate)
            if os.path.isdir(p):
                json_files = glob.glob(os.path.join(p, "*.json"))
                if json_files:
                    annotations_dir = p
                    fmt = "coco"
                    break

    # Fallback: flat directory
    if images_dir is None:
        img_files = glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                    glob.glob(os.path.join(dataset_path, "*.png")) + \
                    glob.glob(os.path.join(dataset_path, "*.jpeg"))
        if img_files:
            images_dir = dataset_path

    if annotations_dir is None and fmt is None:
        xml_files = glob.glob(os.path.join(dataset_path, "*.xml"))
        txt_files = glob.glob(os.path.join(dataset_path, "*.txt"))
        json_files = glob.glob(os.path.join(dataset_path, "*.json"))
        if xml_files:
            annotations_dir = dataset_path
            fmt = "voc"
        elif txt_files:
            annotations_dir = dataset_path
            fmt = "yolo"
        elif json_files:
            annotations_dir = dataset_path
            fmt = "coco"

    return images_dir, annotations_dir, fmt


def _parse_voc_annotation(xml_path):
    """Parse a Pascal VOC XML annotation file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text) if size is not None and size.find("width") is not None else 0
    img_h = int(size.find("height").text) if size is not None and size.find("height") is not None else 0

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip().lower()
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        objects.append({
            "class": name,
            "xmin": xmin, "ymin": ymin,
            "xmax": xmax, "ymax": ymax,
            "bbox_width": xmax - xmin,
            "bbox_height": ymax - ymin,
            "bbox_area": (xmax - xmin) * (ymax - ymin),
        })

    return img_w, img_h, objects


def _parse_yolo_annotation(txt_path, img_w=640, img_h=640, class_names=None):
    """Parse a YOLO format TXT annotation file."""
    objects = []
    if class_names is None:
        from config import RUOD_CLASSES
        class_names = RUOD_CLASSES

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert normalized coords to pixel coords
            abs_w = w * img_w
            abs_h = h * img_h
            xmin = (cx - w / 2) * img_w
            ymin = (cy - h / 2) * img_h
            xmax = (cx + w / 2) * img_w
            ymax = (cy + h / 2) * img_h

            cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
            objects.append({
                "class": cls_name,
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax,
                "bbox_width": abs_w,
                "bbox_height": abs_h,
                "bbox_area": abs_w * abs_h,
            })

    return objects


def _parse_coco_annotations(json_paths):
    """Parse COCO JSON annotations to a dictionary mapped by image filename."""
    img_id_to_file = {}
    cat_id_to_name = {}
    annotations_by_img = {}

    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)

        for cat in data.get("categories", []):
            cat_id_to_name[cat["id"]] = cat["name"].lower()

        for img in data.get("images", []):
            img_id = img["id"]
            img_id_to_file[img_id] = img["file_name"].split("/")[-1] # handle paths
            annotations_by_img[img_id_to_file[img_id]] = []

        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            img_file = img_id_to_file.get(img_id)
            if not img_file: continue
            
            cat_id = ann["category_id"]
            cls_name = cat_id_to_name.get(cat_id, f"class_{cat_id}")
            
            bbox = ann["bbox"] # [x, y, width, height]
            xmin, ymin, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            xmax = xmin + w
            ymax = ymin + h
            
            annotations_by_img[img_file].append({
                "class": cls_name,
                "xmin": xmin, "ymin": ymin,
                "xmax": xmax, "ymax": ymax,
                "bbox_width": w,
                "bbox_height": h,
                "bbox_area": w * h,
            })
            
    return annotations_by_img


def load_dataset(dataset_path=None, max_images=None):
    """
    Load the RUOD dataset from the given path.
    
    Returns:
        dict with keys:
            - 'images_dir': path to images
            - 'format': 'voc' or 'yolo'
            - 'image_paths': list of image file paths
            - 'annotations': dict mapping image basename -> list of object dicts
            - 'stats_df': DataFrame with per-image statistics
            - 'objects_df': DataFrame with all annotated objects
            - 'dataset_info': summary dict
    """
    if dataset_path is None:
        from config import DATASET_PATH
        dataset_path = DATASET_PATH

    images_dir, annotations_dir, fmt = _find_dataset_dirs(dataset_path)

    if images_dir is None:
        return {
            "images_dir": None,
            "format": None,
            "image_paths": [],
            "annotations": {},
            "stats_df": pd.DataFrame(),
            "objects_df": pd.DataFrame(),
            "dataset_info": {
                "total_images": 0,
                "total_annotations": 0,
                "categories": [],
                "format": "not_found",
                "path": dataset_path,
                "status": "Dataset not found. Please update DATASET_PATH in config.py",
            },
        }

    # Collect image files
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
    image_paths = []
    
    # Try looking in train subdirectories explicitly, or just recursive glob
    for ext in extensions:
        # Search directly
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(images_dir, ext.upper())))
        # Search in train/test subdirs
        image_paths.extend(glob.glob(os.path.join(images_dir, "**", ext), recursive=True))
        image_paths.extend(glob.glob(os.path.join(images_dir, "**", ext.upper()), recursive=True))
        
    image_paths = sorted(list(set(image_paths)))

    if max_images:
        image_paths = image_paths[:max_images]

    # Pre-parse COCO if needed
    coco_annotations = {}
    if fmt == "coco" and annotations_dir:
        json_files = glob.glob(os.path.join(annotations_dir, "*.json"))
        coco_annotations = _parse_coco_annotations(json_files)

    # Parse annotations
    annotations = {}
    stats_records = []
    objects_records = []

    for img_path in image_paths:
        basename = Path(img_path).stem
        img_name = Path(img_path).name

        # Try to read image dimensions
        img = cv2.imread(img_path)
        if img is not None:
            img_h, img_w = img.shape[:2]
            mean_brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            mean_b, mean_g, mean_r = [float(x) for x in cv2.mean(img)[:3]]
        else:
            img_h, img_w = 0, 0
            mean_brightness = 0
            mean_b = mean_g = mean_r = 0

        # Find matching annotation
        objects = []
        if annotations_dir and fmt == "voc":
            xml_path = os.path.join(annotations_dir, basename + ".xml")
            if os.path.exists(xml_path):
                _, _, objects = _parse_voc_annotation(xml_path)
        elif annotations_dir and fmt == "yolo":
            txt_path = os.path.join(annotations_dir, basename + ".txt")
            if os.path.exists(txt_path):
                objects = _parse_yolo_annotation(txt_path, img_w, img_h)
        elif fmt == "coco":
            objects = coco_annotations.get(img_name, [])

        annotations[img_name] = objects

        stats_records.append({
            "filename": img_name,
            "path": img_path,
            "width": img_w,
            "height": img_h,
            "num_objects": len(objects),
            "mean_brightness": mean_brightness,
            "mean_red": mean_r,
            "mean_green": mean_g,
            "mean_blue": mean_b,
        })

        for obj in objects:
            objects_records.append({
                "filename": img_name,
                "class": obj["class"],
                "xmin": obj["xmin"],
                "ymin": obj["ymin"],
                "xmax": obj["xmax"],
                "ymax": obj["ymax"],
                "bbox_width": obj["bbox_width"],
                "bbox_height": obj["bbox_height"],
                "bbox_area": obj["bbox_area"],
            })

    stats_df = pd.DataFrame(stats_records)
    objects_df = pd.DataFrame(objects_records)

    categories = objects_df["class"].unique().tolist() if len(objects_df) > 0 else []

    dataset_info = {
        "total_images": len(image_paths),
        "total_annotations": len(objects_records),
        "categories": categories,
        "num_categories": len(categories),
        "format": fmt,
        "path": dataset_path,
        "images_dir": images_dir,
        "annotations_dir": annotations_dir,
        "status": "loaded",
        "avg_objects_per_image": round(len(objects_records) / max(len(image_paths), 1), 2),
        "avg_brightness": round(stats_df["mean_brightness"].mean(), 2) if len(stats_df) > 0 else 0,
    }

    return {
        "images_dir": images_dir,
        "format": fmt,
        "image_paths": image_paths,
        "annotations": annotations,
        "all_annotations": coco_annotations,
        "stats_df": stats_df,
        "objects_df": objects_df,
        "dataset_info": dataset_info,
    }


def get_sample_images(image_paths, n=6):
    """Return n random sample image paths."""
    if len(image_paths) <= n:
        return image_paths
    indices = np.random.choice(len(image_paths), size=n, replace=False)
    return [image_paths[i] for i in indices]


def get_dataset_summary_text():
    """Return a formatted summary for the dashboard."""
    return {
        "title": "RUOD — Real-world Underwater Object Detection Dataset",
        "source": "GitHub (xiaoDetection/RUOD) — Open Source",
        "description": (
            "The RUOD dataset is a large-scale benchmark specifically designed for "
            "underwater object detection. It contains approximately 14,000 high-resolution "
            "underwater images with ~75,000 bounding box annotations across 10 diverse "
            "marine categories. The dataset captures real-world underwater challenges "
            "including light scattering, color distortion, low contrast, and complex backgrounds."
        ),
        "justification": (
            "We selected RUOD because: (1) it is openly available and widely cited in "
            "underwater detection research; (2) it covers 10 diverse marine species/objects; "
            "(3) images exhibit real underwater degradation — ideal for our enhancement pipeline; "
            "(4) bounding box annotations enable supervised object detection evaluation."
        ),
        "categories": [
            "Sea Cucumber (Holothurian)", "Sea Urchin (Echinus)", "Scallop",
            "Starfish", "Fish", "Coral", "Diver", "Cuttlefish",
            "Sea Turtle", "Jellyfish"
        ],
    }
