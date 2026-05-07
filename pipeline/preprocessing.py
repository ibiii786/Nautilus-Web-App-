"""
Pelagix — Preprocessing
========================
Step 2 of the Data Science Pipeline: Data Preprocessing.
Handles resizing, denoising, color normalization, and histogram
equalization for underwater images.
"""

import cv2
import numpy as np


def resize_image(image, target_size=(640, 640)):
    """Resize image to target dimensions while recording the transformation."""
    original_size = (image.shape[1], image.shape[0])
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized, {
        "step": "Resize",
        "description": f"Resized from {original_size[0]}×{original_size[1]} to {target_size[0]}×{target_size[1]}",
        "original_size": original_size,
        "target_size": target_size,
    }


def denoise_image(image, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21):
    """
    Apply Non-Local Means Denoising.
    Effective for underwater images that often have particulate noise.
    """
    denoised = cv2.fastNlMeansDenoisingColored(
        image, None, h, hColor, templateWindowSize, searchWindowSize
    )
    return denoised, {
        "step": "Denoise",
        "description": "Applied Non-Local Means Denoising to reduce underwater particulate noise",
        "params": {"h": h, "hColor": hColor},
    }


def normalize_color(image):
    """
    Normalize color channels to reduce blue/green cast.
    Uses per-channel mean normalization.
    """
    result = image.copy().astype(np.float32)
    for i in range(3):
        channel = result[:, :, i]
        ch_mean = channel.mean()
        if ch_mean > 0:
            result[:, :, i] = np.clip(channel * (128.0 / ch_mean), 0, 255)
    result = result.astype(np.uint8)

    return result, {
        "step": "Color Normalization",
        "description": "Normalized RGB channels to reduce underwater blue/green color cast",
        "channel_means_before": {
            "blue": float(image[:, :, 0].mean()),
            "green": float(image[:, :, 1].mean()),
            "red": float(image[:, :, 2].mean()),
        },
        "channel_means_after": {
            "blue": float(result[:, :, 0].mean()),
            "green": float(result[:, :, 1].mean()),
            "red": float(result[:, :, 2].mean()),
        },
    }


def equalize_histogram(image):
    """
    Apply histogram equalization on each channel independently.
    Improves contrast in low-visibility underwater images.
    """
    channels = cv2.split(image)
    eq_channels = [cv2.equalizeHist(ch) for ch in channels]
    result = cv2.merge(eq_channels)

    return result, {
        "step": "Histogram Equalization",
        "description": "Applied per-channel histogram equalization to improve contrast distribution",
    }


def auto_crop_borders(image, threshold=15):
    """
    Auto-detect and crop black borders that sometimes appear in underwater footage.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        cropped = image[y:y+h, x:x+w]
        if cropped.shape[0] > 50 and cropped.shape[1] > 50:
            return cropped, {
                "step": "Auto Crop",
                "description": f"Removed black borders — cropped to ({w}×{h})",
                "crop_rect": {"x": x, "y": y, "w": w, "h": h},
            }

    return image, {
        "step": "Auto Crop",
        "description": "No significant black borders detected — skipped",
    }


def preprocess_image(image, target_size=(640, 640)):
    """
    Run the full preprocessing pipeline on a single image.
    
    Returns:
        tuple: (preprocessed_image, list_of_step_logs)
    """
    steps = []
    current = image.copy()

    # Step 1: Auto crop
    current, log = auto_crop_borders(current)
    steps.append(log)

    # Step 2: Resize
    current, log = resize_image(current, target_size)
    steps.append(log)

    # Step 3: Denoise
    current, log = denoise_image(current)
    steps.append(log)

    # Step 4: Color normalization
    current, log = normalize_color(current)
    steps.append(log)

    return current, steps


def get_preprocessing_comparison(image, target_size=(640, 640)):
    """
    Generate before/after images for each preprocessing step.
    Returns a list of (step_name, before_img, after_img, log) tuples.
    """
    comparisons = []
    current = image.copy()

    pipeline_steps = [
        ("Auto Crop", auto_crop_borders, {}),
        ("Resize", resize_image, {"target_size": target_size}),
        ("Denoise", denoise_image, {}),
        ("Color Normalization", normalize_color, {}),
    ]

    for name, func, kwargs in pipeline_steps:
        before = current.copy()
        current, log = func(current, **kwargs)
        comparisons.append({
            "name": name,
            "before": before,
            "after": current.copy(),
            "log": log,
        })

    return comparisons
