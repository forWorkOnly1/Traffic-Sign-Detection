# app.py — auto-detect best YOLOv8 architecture, load checkpoint state_dict, interactive Streamlit demo
# Place this file in your project folder and run: streamlit run app.py

import streamlit as st
from pathlib import Path
from PIL import Image
import io, traceback
import torch, numpy as np

st.set_page_config(page_title="Traffic Sign Classifier (auto-arch)", layout="centered")
st.title("Traffic Sign Classifier — Auto-arch loader")

# === USER CONFIG (edit if you know exact info) ===
MODEL_PATH = Path("models/best_model.pt")
# If you know your 14 classes (exact order used for training) put them here.
CLASS_NAMES = None  # e.g. ['speed_20','speed_30',...], otherwise app will try to infer count

# Architectures to try (in order). We'll pick the architecture that yields the smallest missing+unexpected.
YOLO_ARCHS_TO_TRY = ["yolov8n.yaml", "yolov8s.yaml", "yolov8m.yaml", "yolov8l.yaml"]

# === Utilities ===
def infer_num_classes_from_state_dict(sd):
    """
    Heuristic: find bias tensors (1-D) or conv out channels matching number of classes.
    Returns integer or None.
    """
    try:
        # look for bias vectors with small length (<=100)
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dim() == 1:
                length = v.shape[0]
                # likely class head bias - usually small (<100) and >1
                if 2 <= length <= 200:
                    # further heuristic: key contains 'cv3' or 'head' or '.cv3.0.2.bias' seen in YOLO heads
                    if any(token in k.lower() for token in ('cv3', 'head', 'cls', 'conv2', '.cv3.0.2.bias', '.cv2.0.2.bias', 'class')):
                        return int(length)
        # fallback: look for conv weight shapes where first dim equals possible class count
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 2:
                first = v.shape[0]
                if 2 <= first <= 200 and any(token in k.lower() for token in ('cv3', '.cv3', 'cv2.0.2', 'cv3.0.2', 'cv3.1.2')):
                    return int(first)
    except Exception:
        pass
    return None

def try_load_state_into_yolo(sd, arch):
    """
    Attempt to construct YOLO(arch) and load state_dict into y.model with key remapping heuristics.
    Returns (yolo_obj, missing_count, unexpected_count, load_exception_or_none)
    """
    try:
        from ultralytics import YOLO
    except Exception as e:
        return (None, None, None, f"ultralytics import failed: {e}")

    try:
        y = YOLO(arch)  # construct architecture
    except Exception as e:
        return (None, None, None, f"Failed to construct YOLO from {arch}: {e}")

    # get candidate state_dict (it may be sd['model'] or sd itself)
    candidate = sd.get("model", sd) if isinstance(sd, dict) else sd
    state_dict = candidate if isinstance(candidate, dict) else (candidate.state_dict() if hasattr(candidate, "state_dict") else None)
    if state_dict is None:
        return (None, None, None, "No state_dict found in checkpoint candidate")

    # adapt keys to remove 'module.' or extra 'model.' nesting
    adapted = {}
    for k, v in state_dict.items():
        new_k = k
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        if new_k.startswith("model.model."):
            new_k = new_k[len("model."):]
        adapted[new_k] = v

    # attempt load strict=False and capture missing/unexpected
    try:
        missing, unexpected = y.model.load_state_dict(adapted, strict=False)
        return (y, len(missing), len(unexpected), None)
    except Exception as e:
        # try second mapping: strip leading 'model.' from keys if present
        adapted2 = {}
        for k, v in adapted.items():
            if k.startswith("model."):
                adapted2[k[len("model."):]] = v
            else:
                adapted2[k] = v
        try:
            missing2, unexpected2 = y.model.load_state_dict(adapted2, strict=False)
            return (y, len(missing2), len(unexpected2), None)
        except Exception as e2:
            return (None, None, None, f"Both load attempts failed: {e2}\n{traceback.format_exc()}")

# === Loader that auto-tries architectures ===
@st.cache_resource
def auto_load_best_yolo(model_path: Path, archs):
    info = {"model_path": str(model_path), "tried": []}
    if not model_path.exists():
        return {"ok": False, "reason": f"Model file not found: {model_path}", "info": info}

    # load checkpoint with torch
    try:
        sd = torch.load(str(model_path), map_location="cpu")
    except Exception as e:
        return {"ok": False, "reason": f"torch.load failed: {e}", "info": info}

    # attempt to infer classes
    num_classes_in_ckpt = infer_num_classes_from_state_dict(sd if isinstance(sd, dict) else (sd.state_dict() if hasattr(sd,"state_dict") else sd))
    info["inferred_num_classes"] = num_classes_in_ckpt

    best = None
    best_score = None
    # try each arch
    for arch in archs:
        y, missing, unexpected, err = try_load_state_into_yolo(sd, arch)
        info["tried"].append({"arch": arch, "missing": missing, "unexpected": unexpected, "err": err})
        if y is not None:
            score = (missing or 0) + (unexpected or 0)
            if best is None or score < best_score:
                best = {"yolo": y, "arch": arch, "missing": missing, "unexpected": unexpected}
                best_score = score
    if best is None:
        return {"ok": False, "reason": "No architecture could load the checkpoint. See tried list.", "info": info}
    # success
    info["best"] = best
    return {"ok": True, "yolo": best["yolo"], "arch": best["arch"], "missing": best["missing"], "unexpected": best["unexpected"], "inferred_num_classes": num_classes_in_ckpt, "info": info}

# === Run loader ===
st.sidebar.write("Model path:")
st.sidebar.write(str(MODEL_PATH))

res = auto_load_best_yolo(MODEL_PATH, YOLO_ARCHS_TO_TRY)
if not res["ok"]:
    st.sidebar.error("Auto-loader failed: " + res.get("reason",""))
    st.sidebar.write(res.get("info",{}))
    model_type = None
    yolo_model = None
else:
    yolo_model = res["yolo"]
    model_type = "yolo"
    st.sidebar.success(f"Best arch: {res['arch']} (missing: {res['missing']}, unexpected: {res['unexpected']})")
    st.sidebar.write("If missing/unexpected counts are large, predictions may be degraded.")
    if res.get("inferred_num_classes"):
        st.sidebar.info(f"Inferred num_classes from checkpoint: {res['inferred_num_classes']}")
    st.sidebar.write("Tried architectures: ")
    for t in res["info"]["tried"]:
        st.sidebar.write(f" - {t['arch']}: missing={t['missing']}, unexpected={t['unexpected']}, err={t['err']}")

# If CLASS_NAMES not set but inference found a class count, create placeholder names
if CLASS_NAMES is None and res.get("inferred_num_classes"):
    n = res["inferred_num_classes"]
    CLASS_NAMES = [f"class_{i}" for i in range(n)]
    st.sidebar.info(f"CLASS_NAMES auto-filled with placeholders for {n} classes. Replace with real names if you have them.")

st.sidebar.write("---")
st.sidebar.write("Inference parameters")
conf_default = 0.25
imgsz_default = 640
conf = st.sidebar.slider("confidence threshold", min_value=0.01, max_value=0.9, value=float(conf_default), step=0.01)
imgsz = st.sidebar.select_slider("image size (imgsz)", options=[320, 416, 512, 640, 800], value=imgsz_default)

# === Inference UI ===
uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded is None:
    st.info("Upload an image to run detection. If nothing is detected, try lowering the confidence threshold and increasing imgsz.")
else:
    pil = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(pil, caption="Uploaded image", use_column_width=True)

    if model_type != "yolo" or yolo_model is None:
        st.error("Model not loaded as YOLO. See sidebar for details.")
    else:
        # run prediction
        try:
            results = yolo_model.predict(source=pil, imgsz=imgsz, conf=conf, save=False)
            if not results or len(results) == 0:
                st.warning("No results returned by model.predict().")
            else:
                r = results[0]
                # show boxes info
                if hasattr(r, "boxes") and len(r.boxes) > 0:
                    rows = []
                    for i in range(len(r.boxes)):
                        try:
                            cls_idx = int(r.boxes.cls[i].item())
                            conf_val = float(r.boxes.conf[i].item())
                            # xyxy
                            xyxy = r.boxes.xyxy[i].tolist() if hasattr(r.boxes, "xyxy") else None
                            label = CLASS_NAMES[cls_idx] if (CLASS_NAMES and cls_idx < len(CLASS_NAMES)) else str(cls_idx)
                            rows.append((label, conf_val, xyxy))
                        except Exception:
                            pass
                    # display results
                    st.success(f"Detections: {len(rows)}")
                    for label, conf_val, xyxy in rows:
                        st.write(f"- {label} (conf={conf_val:.3f}) bbox={xyxy}")
                else:
                    st.info("No boxes detected (empty r.boxes). Try lowering confidence or increasing imgsz.")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.sidebar.text(traceback.format_exc())

st.write("---")
st.write("If you still get no detections: try lowering confidence (sidebar), increase imgsz, or try a different architecture list (n/s/m/l).")
