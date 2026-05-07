"""
Pelagix — Quality Score
========================
Step 6: Custom Underwater Visibility Score (UVS) — a 0-100 index
combining brightness, color balance, contrast, and sharpness.
"""

import cv2
import numpy as np


def compute_brightness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    score = 100 - abs(mean_val - 128) * (100 / 128)
    return max(0, min(100, score)), {"raw_mean": round(float(mean_val), 2), "ideal": 128}


def compute_color_balance_score(image):
    b, g, r = cv2.mean(image)[:3]
    avg = (b + g + r) / 3.0
    deviation = (abs(r - avg) + abs(g - avg) + abs(b - avg)) / 3.0
    max_dev = 128
    score = max(0, 100 - (deviation / max_dev) * 100)
    return round(score, 2), {"mean_rgb": [round(r, 1), round(g, 1), round(b, 1)], "deviation": round(deviation, 2)}


def compute_contrast_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    std_val = np.std(gray)
    score = min(100, (std_val / 80) * 100)
    return round(score, 2), {"std_luminance": round(float(std_val), 2)}


def compute_sharpness_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    score = min(100, (laplacian_var / 500) * 100)
    return round(score, 2), {"laplacian_variance": round(float(laplacian_var), 2)}


def compute_uvs(image, weights=None):
    """
    Compute the Underwater Visibility Score (UVS) for an image.

    Formula:
        UVS = w1*Brightness + w2*ColorBalance + w3*Contrast + w4*Sharpness

    Returns:
        dict with overall score, component scores, and details.
    """
    if weights is None:
        from config import UVS_WEIGHTS
        weights = UVS_WEIGHTS

    brightness, b_detail = compute_brightness_score(image)
    color_bal, c_detail = compute_color_balance_score(image)
    contrast, ct_detail = compute_contrast_score(image)
    sharpness, s_detail = compute_sharpness_score(image)

    uvs = (weights["brightness"] * brightness +
           weights["color_balance"] * color_bal +
           weights["contrast"] * contrast +
           weights["sharpness"] * sharpness)

    uvs = round(max(0, min(100, uvs)), 1)

    grade = "Excellent" if uvs >= 80 else "Good" if uvs >= 60 else "Fair" if uvs >= 40 else "Poor" if uvs >= 20 else "Very Poor"
    grade_color = "#00f5d4" if uvs >= 80 else "#00d4ff" if uvs >= 60 else "#ffd93d" if uvs >= 40 else "#ff6b9d" if uvs >= 20 else "#ff4444"

    return {
        "uvs": uvs,
        "grade": grade,
        "grade_color": grade_color,
        "components": {
            "brightness": {"score": round(brightness, 1), "weight": weights["brightness"], "details": b_detail},
            "color_balance": {"score": round(color_bal, 1), "weight": weights["color_balance"], "details": c_detail},
            "contrast": {"score": round(contrast, 1), "weight": weights["contrast"], "details": ct_detail},
            "sharpness": {"score": round(sharpness, 1), "weight": weights["sharpness"], "details": s_detail},
        },
        "formula": f"UVS = {weights['brightness']}×Brightness + {weights['color_balance']}×ColorBalance + {weights['contrast']}×Contrast + {weights['sharpness']}×Sharpness",
    }


def compare_quality(original, enhanced, weights=None):
    orig_score = compute_uvs(original, weights)
    enh_score = compute_uvs(enhanced, weights)
    improvement = round(enh_score["uvs"] - orig_score["uvs"], 1)
    pct = round((improvement / max(orig_score["uvs"], 0.1)) * 100, 1)

    return {
        "original": orig_score,
        "enhanced": enh_score,
        "improvement": improvement,
        "improvement_pct": pct,
        "summary": f"Quality improved from {orig_score['uvs']}/100 ({orig_score['grade']}) to {enh_score['uvs']}/100 ({enh_score['grade']}) — a {improvement:+.1f} point increase ({pct:+.1f}%)",
    }
