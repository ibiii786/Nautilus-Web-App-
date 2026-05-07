
import os

DATASET_PATH = os.path.join(os.path.dirname(__file__), "RUOD")

APP_NAME = "Nautilus"
APP_TAGLINE = "Marine Intelligence Platform"
APP_VERSION = "1.0.0"
HOST = "127.0.0.1"
PORT = 5000
DEBUG = False

TARGET_SIZE = (640, 640)           # Standard size for YOLO
MAX_UPLOAD_SIZE_MB = 16
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

YOLO_MODEL = "yolov8m-world.pt"     # Upgraded to Medium model for significantly better accuracy
YOLO_CONFIDENCE = 0.05            # Increased confidence to reduce false positives
YOLO_IOU = 0.45                   # NMS IoU threshold

UVS_WEIGHTS = {
    "brightness": 0.20,
    "color_balance": 0.30,
    "contrast": 0.25,
    "sharpness": 0.25,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
PLOTS_DIR = os.path.join(STATIC_DIR, "plots")
OUTPUTS_DIR = os.path.join(STATIC_DIR, "outputs")
UPLOAD_DIR = os.path.join(STATIC_DIR, "uploads")

for d in [PLOTS_DIR, OUTPUTS_DIR, UPLOAD_DIR]:
    os.makedirs(d, exist_ok=True)

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
