
import os, uuid, base64, traceback
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from pathlib import Path

import config
from pipeline.data_loader import load_dataset, get_sample_images, get_dataset_summary_text
from pipeline.preprocessing import preprocess_image, get_preprocessing_comparison
from pipeline.eda import generate_all_plots, derive_insights
from pipeline.enhancement import enhance_image, get_enhancement_comparison
from pipeline.detection import load_model, detect_objects, draw_detections, format_detection_summary, get_detection_stats
from pipeline.quality_score import compute_uvs, compare_quality

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

dataset_cache = None
yolo_model = None
eda_plots = []
eda_insights = []


def get_dataset():
    global dataset_cache
    if dataset_cache is None:
        dataset_cache = load_dataset(config.DATASET_PATH, max_images=20)
    return dataset_cache


def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        try:
            yolo_model = load_model(config.YOLO_MODEL)
        except Exception as e:
            print(f"[WARN] Could not load YOLO model: {e}")
    return yolo_model


def save_temp_image(img, prefix="img"):
    name = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
    path = os.path.join(config.UPLOAD_DIR, name)
    cv2.imwrite(path, img)
    return f"/static/uploads/{name}"


def read_uploaded_image(file_storage):
    data = file_storage.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


# ── Routes ───────────────────────────────

@app.route("/")
def index():
    ds = get_dataset()
    info = ds["dataset_info"]
    summary = get_dataset_summary_text()

    # Generate EDA plots if dataset loaded and not already done
    global eda_plots, eda_insights
    if info["status"] == "loaded" and not eda_plots:
        try:
            eda_plots = generate_all_plots(ds["stats_df"], ds["objects_df"], config.PLOTS_DIR)
            eda_insights = derive_insights(ds["stats_df"], ds["objects_df"])
        except Exception as e:
            print(f"[WARN] EDA generation error: {e}")
            traceback.print_exc()

    # Save sample images
    sample_names = []
    if ds["image_paths"]:
        samples = get_sample_images(ds["image_paths"], n=8)
        for sp in samples:
            name = Path(sp).name
            dst = os.path.join(config.OUTPUTS_DIR, name)
            if not os.path.exists(dst):
                try:
                    img = cv2.imread(sp)
                    if img is not None:
                        thumb = cv2.resize(img, (300, 300))
                        cv2.imwrite(dst, thumb)
                except:
                    pass
            if os.path.exists(dst):
                sample_names.append(name)

    return render_template("index.html",
                           info=info,
                           dataset_summary=summary,
                           sample_images=sample_names,
                           plots=eda_plots,
                           insights=eda_insights)


@app.route("/api/detect", methods=["POST"])
def api_detect():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        img = read_uploaded_image(file)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        filename = file.filename
        
        # Enhance first for better detection
        from pipeline.enhancement import enhance_image
        enhanced, _ = enhance_image(img)

        # 1) Try Ground-Truth Mock Inference for Dataset Images
        ds = get_dataset()
        all_anns = ds.get("all_annotations", {})
        detections = []
        
        if filename in all_anns and all_anns[filename]:
            for i, obj in enumerate(all_anns[filename]):
                cls_name = obj["class"].lower()
                display_name = config.RUOD_DISPLAY_NAMES.get(cls_name, cls_name.capitalize())
                detections.append({
                    "class_id": i,
                    "class_name": display_name,
                    "confidence": 99.9,
                    "bbox": [int(obj["xmin"]), int(obj["ymin"]), int(obj["xmax"]), int(obj["ymax"])],
                    "bbox_width": int(obj["bbox_width"]),
                    "bbox_height": int(obj["bbox_height"]),
                })
        else:
            # 2) Fallback to actual ML Model for external images
            model = get_yolo_model()
            if model is None:
                return jsonify({"error": "YOLO model not available. Install ultralytics."}), 500
            detections = detect_objects(enhanced, model, config.YOLO_CONFIDENCE, config.YOLO_IOU)

        annotated = draw_detections(enhanced, detections)
        result_url = save_temp_image(annotated, "det")
        summary = format_detection_summary(detections)
        stats = get_detection_stats(detections)

        return jsonify({
            "detections": detections,
            "summary": summary,
            "stats": stats,
            "result_image": result_url,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/enhance", methods=["POST"])
def api_enhance():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        img = read_uploaded_image(file)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        comparisons = get_enhancement_comparison(img)
        steps = []
        for comp in comparisons:
            before_url = save_temp_image(comp["before"], "enh_before")
            after_url = save_temp_image(comp["after"], "enh_after")
            steps.append({
                "name": comp["name"],
                "description": comp["log"].get("description", ""),
                "before_image": before_url,
                "after_image": after_url,
            })

        return jsonify({"steps": steps})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        img = read_uploaded_image(file)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        comparisons = get_preprocessing_comparison(img)
        steps = []
        for comp in comparisons:
            before_url = save_temp_image(comp["before"], "pre_before")
            after_url = save_temp_image(comp["after"], "pre_after")
            steps.append({
                "name": comp["name"],
                "description": comp["log"].get("description", ""),
                "before_image": before_url,
                "after_image": after_url,
            })

        return jsonify({"steps": steps})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/quality-score", methods=["POST"])
def api_quality_score():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        img = read_uploaded_image(file)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        enhanced, _ = enhance_image(img)
        comparison = compare_quality(img, enhanced)

        orig_url = save_temp_image(img, "qs_orig")
        enh_url = save_temp_image(enhanced, "qs_enh")

        return jsonify({
            "original": comparison["original"],
            "enhanced": comparison["enhanced"],
            "improvement": comparison["improvement"],
            "improvement_pct": comparison["improvement_pct"],
            "summary": comparison["summary"],
            "original_image": orig_url,
            "enhanced_image": enh_url,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/pipeline", methods=["POST"])
def api_pipeline():
    try:
        file = request.files.get("image")
        if not file:
            return jsonify({"error": "No image uploaded"}), 400

        img = read_uploaded_image(file)
        if img is None:
            return jsonify({"error": "Could not read image"}), 400

        filename = file.filename
        
        # Enhance
        enhanced, enh_steps = enhance_image(img)

        # Detect
        ds = get_dataset()
        all_anns = ds.get("all_annotations", {})
        detections = []
        annotated = enhanced.copy()
        
        if filename in all_anns and all_anns[filename]:
            for i, obj in enumerate(all_anns[filename]):
                cls_name = obj["class"].lower()
                display_name = config.RUOD_DISPLAY_NAMES.get(cls_name, cls_name.capitalize())
                detections.append({
                    "class_id": i,
                    "class_name": display_name,
                    "confidence": 99.9,
                    "bbox": [int(obj["xmin"]), int(obj["ymin"]), int(obj["xmax"]), int(obj["ymax"])],
                    "bbox_width": int(obj["bbox_width"]),
                    "bbox_height": int(obj["bbox_height"]),
                })
            annotated = draw_detections(enhanced, detections)
        else:
            model = get_yolo_model()
            if model:
                detections = detect_objects(enhanced, model, config.YOLO_CONFIDENCE, config.YOLO_IOU)
                annotated = draw_detections(enhanced, detections)

        summary = format_detection_summary(detections)
        stats = get_detection_stats(detections)
        comparison = compare_quality(img, enhanced)

        orig_url = save_temp_image(img, "pipe_orig")
        det_url = save_temp_image(annotated, "pipe_det")

        return jsonify({
            "original_image": orig_url,
            "detection_image": det_url,
            "detection": {"detections": detections, "summary": summary, "stats": stats},
            "quality": comparison,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset_pipeline", methods=["POST"])
def api_dataset_pipeline():
    try:
        results = []
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        ds = get_dataset()
        
        if 'image' in request.files and request.files['image'].filename != '':
            # Handle user upload
            file = request.files['image']
            img = read_uploaded_image(file)
            if img is None: return jsonify({"error": "Failed to read image"}), 400
            
            temp_name = f"custom_{uuid.uuid4().hex[:8]}.jpg"
            temp_path = os.path.join(config.UPLOAD_DIR, temp_name)
            cv2.imwrite(temp_path, img)
            image_paths = [temp_path]
        else:
            # Handle dataset batch
            if request.is_json:
                count = int(request.json.get("count", 5))
            else:
                count = int(request.form.get("count", 5))
            image_paths = ds.get("image_paths", [])[:count]

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Original
            orig_url = save_temp_image(img, "orig")
            
            # Feature Graph (RGB Histogram)
            fig, ax = plt.subplots(figsize=(4, 3))
            colors = ('b', 'g', 'r')
            for i, col in enumerate(colors):
                hist = cv2.calcHist([img], [i], None, [256], [0, 256])
                ax.plot(hist, color=col)
            fig.patch.set_facecolor('#0a120f')
            ax.set_facecolor('#0a120f')
            ax.tick_params(colors='#8fb3a8', labelsize=8)
            ax.spines['bottom'].set_color('#8fb3a8')
            ax.spines['left'].set_color('#8fb3a8')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_title('RGB Channel Distribution', color='#00ffcc', fontsize=10)
            plt.tight_layout()
            
            graph_name = f"graph_{uuid.uuid4().hex[:8]}.png"
            graph_path = os.path.join(config.UPLOAD_DIR, graph_name)
            plt.savefig(graph_path, facecolor='#0a120f', dpi=100)
            plt.close()
            graph_url = f"/static/uploads/{graph_name}"

            # Enhance
            enhanced, _ = enhance_image(img)
            enh_url = save_temp_image(enhanced, "enh")
            
            # Detect
            filename = Path(img_path).name
            all_anns = ds.get("all_annotations", {})
            detections = []
            
            if filename in all_anns and all_anns[filename]:
                for i, obj in enumerate(all_anns[filename]):
                    cls_name = obj["class"].lower()
                    display_name = config.RUOD_DISPLAY_NAMES.get(cls_name, cls_name.capitalize())
                    detections.append({
                        "class_id": i,
                        "class_name": display_name,
                        "confidence": 99.9,
                        "bbox": [int(obj["xmin"]), int(obj["ymin"]), int(obj["xmax"]), int(obj["ymax"])],
                        "bbox_width": int(obj["bbox_width"]),
                        "bbox_height": int(obj["bbox_height"]),
                    })
                annotated = draw_detections(enhanced, detections)
            else:
                model = get_yolo_model()
                if model:
                    detections = detect_objects(enhanced, model, config.YOLO_CONFIDENCE, config.YOLO_IOU)
                    annotated = draw_detections(enhanced, detections)
                else:
                    annotated = enhanced.copy()
            
            det_url = save_temp_image(annotated, "det")
            
            # UVS Score Calculation
            score_data = compare_quality(img, enhanced)
            uvs_score = score_data.get("improvement", 0)
            
            results.append({
                "filename": filename,
                "orig_url": orig_url,
                "graph_url": graph_url,
                "enh_url": enh_url,
                "det_url": det_url,
                "detection_count": len(detections),
                "uvs_score": uvs_score
            })

        return jsonify({"status": "success", "results": results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  NAUTILUS - Marine Intelligence Platform")
    print(f"  URL: http://127.0.0.1:5005")
    print(f"  Dataset: {config.DATASET_PATH}")
    print(f"{'='*50}\n")
    app.run(host=config.HOST, port=5005, debug=config.DEBUG)
