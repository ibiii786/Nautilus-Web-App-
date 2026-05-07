"""
Pelagix — Object Detection
============================
Step 5: YOLOv8-based underwater object detection & species identification.
"""

import cv2
import numpy as np
import os


def load_model(model_name="yolov8s-world.pt"):
    from ultralytics import YOLO
    import config
    model = YOLO(model_name)
    # Set classes for open-vocabulary detection using YOLO-World
    classes = list(config.RUOD_DISPLAY_NAMES.values())
    try:
        model.set_classes(classes)
    except AttributeError:
        pass # Not a YOLO-World model
    return model


def detect_objects(image, model=None, confidence=0.25, iou=0.45):
    if model is None:
        model = load_model()

    results = model(image, conf=confidence, iou=iou, verbose=False)
    detections = []

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes
        if boxes is not None:
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                detections.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "confidence": round(conf * 100, 1),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "bbox_width": int(x2 - x1),
                    "bbox_height": int(y2 - y1),
                })

    return detections


def draw_detections(image, detections):
    result = image.copy()
    colors = {
        0: (0, 212, 255), 1: (0, 245, 212), 2: (123, 97, 255),
        3: (255, 107, 157), 4: (255, 217, 61), 5: (78, 205, 196),
        6: (255, 138, 92), 7: (168, 230, 207), 8: (220, 237, 193),
        9: (255, 211, 182),
    }

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = colors.get(det["class_id"] % 10, (0, 212, 255))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        label = f'{det["class_name"]} {det["confidence"]}%'
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        cv2.rectangle(result, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(result, label, (x1 + 3, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    return result


def format_detection_summary(detections):
    if not detections:
        return "No objects detected in this image."

    items = []
    for d in detections:
        items.append(f'{d["class_name"]} ({d["confidence"]}% confidence)')

    if len(items) == 1:
        return f"This image contains: {items[0]}"
    return f"This image contains: {', '.join(items[:-1])} and {items[-1]}"


def get_detection_stats(detections):
    if not detections:
        return {"total": 0, "classes": {}, "avg_confidence": 0}

    class_counts = {}
    total_conf = 0
    for d in detections:
        name = d["class_name"]
        class_counts[name] = class_counts.get(name, 0) + 1
        total_conf += d["confidence"]

    return {
        "total": len(detections),
        "classes": class_counts,
        "avg_confidence": round(total_conf / len(detections), 1),
    }
