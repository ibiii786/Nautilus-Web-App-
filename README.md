# Nautilus — Marine Intelligence Platform

A full-stack data science web application for underwater image processing, marine species detection, and EDA.

Built for the **Introduction To Data Science** course — Bahria University Islamabad, BS CS-6B, Spring 2026.

**Group Members:**
- Ghulam Ibrahim (01-134232-055)
- Muhammad Umar (01-134232-137)

---

## Features

- 🖼️ **Image Enhancement** — CLAHE, White Balance, Gamma Correction, Color Cast Removal, Sharpening
- 🐠 **Object Detection** — YOLO-World zero-shot detection of 10 marine species
- 📊 **EDA Dashboard** — Auto-generated plots for species distribution, brightness, bounding boxes
- 📈 **Quality Scoring** — Custom Underwater Visibility Score (UVS) before/after comparison
- 🎨 **Bioluminescent UI** — Deep black + teal theme Flask web app

## Dataset

Uses the [RUOD Dataset](https://github.com/dlut-dimt/RUOD) (~9,800 images, 10 classes, COCO JSON format).
Download it separately and place it in a `RUOD/` folder in the project root.

## Installation

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000

## Tech Stack

Python · Flask · OpenCV · Ultralytics YOLOv8 · NumPy · Pandas · Matplotlib · Seaborn

## Contributors

This project is maintained by:
- Ghulam Ibrahim
- Muhammad Umar
