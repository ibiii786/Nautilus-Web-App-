"""
Pelagix — Image Enhancement
=============================
Step 4: Image Enhancement techniques for underwater images.
"""

import cv2
import numpy as np


def apply_clahe(image, clip_limit=3.0, grid_size=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l = clahe.apply(l)
    result = cv2.merge([l, a, b])
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result, {"step": "CLAHE", "description": "Adaptive contrast enhancement using CLAHE on L-channel (LAB space)", "params": {"clip_limit": clip_limit, "grid_size": grid_size}}


def apply_white_balance(image):
    result = image.copy().astype(np.float32)
    avg_b, avg_g, avg_r = cv2.mean(image)[:3]
    avg_all = (avg_b + avg_g + avg_r) / 3.0
    result[:, :, 0] = np.clip(result[:, :, 0] * (avg_all / max(avg_b, 1)), 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * (avg_all / max(avg_g, 1)), 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * (avg_all / max(avg_r, 1)), 0, 255)
    return result.astype(np.uint8), {"step": "White Balance", "description": "Gray World white balance correction to neutralize underwater color cast", "params": {"avg_bgr": [round(avg_b, 1), round(avg_g, 1), round(avg_r, 1)]}}


def apply_gamma_correction(image, gamma=None):
    if gamma is None:
        mean_brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        gamma = max(0.5, min(2.5, 128.0 / max(mean_brightness, 1)))
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
    result = cv2.LUT(image, table)
    return result, {"step": "Gamma Correction", "description": f"Adaptive brightness adjustment (gamma={gamma:.2f})", "params": {"gamma": round(gamma, 2)}}


def apply_color_cast_removal(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)
    a = a - a.mean() + 128
    b = b - b.mean() + 128
    result = cv2.merge([l, a, b]).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result, {"step": "Color Cast Removal", "description": "Removed blue/green cast by centering a/b channels in LAB space"}


def apply_unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    result = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return result, {"step": "Unsharp Mask", "description": "Edge sharpening via unsharp masking", "params": {"sigma": sigma, "strength": strength}}


def enhance_image(image):
    steps = []
    current = image.copy()
    for func in [apply_clahe, apply_white_balance, apply_gamma_correction, apply_color_cast_removal, apply_unsharp_mask]:
        current, log = func(current)
        steps.append(log)
    return current, steps


def get_enhancement_comparison(image):
    comparisons = []
    current = image.copy()
    pipeline = [
        ("CLAHE", apply_clahe),
        ("White Balance", apply_white_balance),
        ("Gamma Correction", apply_gamma_correction),
        ("Color Cast Removal", apply_color_cast_removal),
        ("Unsharp Mask", apply_unsharp_mask),
    ]
    for name, func in pipeline:
        before = current.copy()
        current, log = func(current)
        comparisons.append({"name": name, "before": before, "after": current.copy(), "log": log})
    return comparisons
