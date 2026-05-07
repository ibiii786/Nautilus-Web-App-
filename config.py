"""
Pelagix — Configuration
========================
Central configuration for the Pelagix underwater intelligence dashboard.
Update DATASET_PATH to point to your downloaded RUOD dataset.
"""

import os

# ──────────────────────────────────────────────
# 🔧  DATASET PATH — UPDATE THIS AFTER DOWNLOAD
# ──────────────────────────────────────────────
# Point this to the root of your extracted RUOD dataset.
# The loader will auto-detect structure (VOC XML or YOLO TXT).
DATASET_PATH = os.path.join(os.path.dirname(__file__), "RUOD")

# ──────────────────────────────────────────────
# Application Settings
# ──────────────────────────────────────────────
APP_NAME = "Nautilus"
APP_TAGLINE = "Marine Intelligence Platform"
APP_VERSION = "1.0.0"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False

# ──────────────────────────────────────────────
# Image Processing Settings
# ──────────────────────────────────────────────
TARGET_SIZE = (640, 640)           # Standard size for YOLO
MAX_UPLOAD_SIZE_MB = 16
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

# ──────────────────────────────────────────────
# YOLO Model Settings
# ──────────────────────────────────────────────
YOLO_MODEL = "yolov8m-world.pt"     # Upgraded to Medium model for significantly better accuracy
YOLO_CONFIDENCE = 0.05            # Increased confidence to reduce false positives
YOLO_IOU = 0.45                   # NMS IoU threshold

# ──────────────────────────────────────────────
# Quality Score Weights (must sum to 1.0)
# ──────────────────────────────────────────────
UVS_WEIGHTS = {
    "brightness": 0.20,
    "color_balance": 0.30,
    "contrast": 0.25,
    "sharpness": 0.25,
}

# ──────────────────────────────────────────────
# Output Directories (auto-created)
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

for d in [PLOTS_DIR, OUTPUTS_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# RUOD Category Names (10 classes)
# ──────────────────────────────────────────────
RUOD_CLASSES = [
    "holothurian",
    "echinus",
    "scallop",
    "starfish",
    "fish",
    "corals",
    "diver",
    "cuttlefish",
    "turtle",
    "jellyfish",
]

# Friendly display names (Used as text prompts for YOLO-World)
# Using highly descriptive terms drastically improves Zero-Shot accuracy
RUOD_DISPLAY_NAMES = {
    "holothurian": "sea cucumber",
    "echinus": "black sea urchin",
    "scallop": "scallop shell",
    "starfish": "starfish",
    "fish": "fish",
    "corals": "coral reef",
    "diver": "scuba diver",
    "cuttlefish": "cuttlefish",
    "turtle": "sea turtle",
    "jellyfish": "jellyfish",
    "rocks": "underwater rock",
}
