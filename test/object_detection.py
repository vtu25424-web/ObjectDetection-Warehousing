import os
from pathlib import Path
from datetime import datetime
import uuid
from PIL import Image
import numpy as np
import cv2

def variance_of_laplacian_gray(np_img_gray):
    """Return variance of Laplacian (focus measure). Higher = sharper."""
    return cv2.Laplacian(np_img_gray, cv2.CV_64F).var()

def detect_and_crop_real(
    image_path,
    out_dir="generated_crops",
    model_path="yolov8s.pt",    # path or known model name (yolov8s.pt, yolov8n.pt, etc.)
    device=None,               # "cpu" or "cuda:0" or None (auto)
    conf_thresh=0.25,
    iou_thresh=0.45,
    imgsz=640,
    classes=None,              # list of class indices or names to filter (optional)
    blur_threshold=100.0       # below this variance => considered blurred (tune as needed)
):
    """
    Run YOLOv8 detection on image_path and save crops to out_dir.
    Returns list of metadata dicts:
    [
      {
        'id': str,
        'label': 'person',
        'confidence': 0.93,
        'bbox': [x1,y1,x2,y2],
        'crop_path': '/abs/path/to/crop.jpg',
        'timestamp': '2025-10-30T..Z',
        'is_blurred': False,
        'focus_measure': float
      }, ...
    ]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # Determine device
    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # Try to import ultralytics
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("ultralytics is required for real detection. pip install ultralytics") from e

    # Load model (ultralytics will attempt to download weights for known names)
    model = YOLO(str(model_path))

    # Prepare classes argument: ultralytics accepts indices; mapping handled below
    classes_arg = None
    if classes is not None:
        # If classes are names, convert to indices if model.names available
        if isinstance(classes, (list, tuple)) and len(classes)>0 and isinstance(classes[0], str):
            # build name->index map
            name_to_idx = {v: k for k, v in model.names.items()} if hasattr(model, "names") else {}
            try:
                classes_arg = [name_to_idx[c] for c in classes]
            except KeyError:
                # if mapping fails, try passing None and let user specify indices
                classes_arg = None
        else:
            classes_arg = classes

    # Run inference
    results = model(
        str(image_path),
        imgsz=imgsz,
        device=device,
        conf=conf_thresh,
        iou=iou_thresh,
        classes=classes_arg,
        verbose=False
    )

    # We expect one image -> results[0]
    res = results[0]
    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []  # no detections

    # read image with PIL for cropping (keeps colors correct)
    im = Image.open(image_path).convert("RGB")
    w, h = im.size

    # Extract numpy arrays from ultralytics Boxes
    # boxes.xyxy -> Nx4 tensor; boxes.conf -> Nx1; boxes.cls -> Nx1
    xyxy_arr = np.array(boxes.xyxy.cpu())  # shape (N,4)
    confs = np.array(boxes.conf.cpu()).reshape(-1)
    cls_arr = np.array(boxes.cls.cpu()).reshape(-1).astype(int)
    name_map = model.names if hasattr(model, "names") else {}

    results_meta = []
    for i in range(xyxy_arr.shape[0]):
        x1, y1, x2, y2 = xyxy_arr[i].astype(int).tolist()
        conf = float(confs[i])
        cls_idx = int(cls_arr[i])
        label = name_map.get(cls_idx, str(cls_idx))

        # clamp coordinates to image bounds
        x1c = max(0, min(x1, w - 1))
        y1c = max(0, min(y1, h - 1))
        x2c = max(0, min(x2, w))
        y2c = max(0, min(y2, h))

        # ensure non-empty box
        if x2c <= x1c or y2c <= y1c:
            continue

        crop = im.crop((x1c, y1c, x2c, y2c))
        # filename: <uuid>_<label>_<timestamp>.jpg
        det_id = str(uuid.uuid4())
        ts = datetime.utcnow().isoformat() + "Z"
        safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_")).lower() or "cls"
        crop_name = f"{det_id}_{safe_label}.jpg"
        crop_path = out_dir / crop_name

        # Save crop (JPEG)
        crop.save(crop_path, format="JPEG", quality=90)

        # Blur check: convert to grayscale numpy array and compute variance-of-Laplacian
        np_crop = np.array(crop)
        gray = cv2.cvtColor(np_crop, cv2.COLOR_RGB2GRAY)
        focus_measure = variance_of_laplacian_gray(gray)
        is_blurred = focus_measure < blur_threshold

        meta = {
            "id": det_id,
            "label": label,
            "confidence": conf,
            "bbox": [int(x1c), int(y1c), int(x2c), int(y2c)],
            "crop_path": str(crop_path),
            "timestamp": ts,
            "is_blurred": bool(is_blurred),
            "focus_measure": float(focus_measure),
        }
        results_meta.append(meta)

    return results_meta

# Example usage
if __name__ == "__main__":
    out = detect_and_crop_real(
        "notebooks/data/sample3.png",
        out_dir="generated_crops_real",
        model_path="models/weights/yolov8s.pt",  # or "yolov8s.pt" (ultralytics may auto-download)
        conf_thresh=0.3,
        imgsz=640
    )

    print("Detections:")
    import json
    print(json.dumps(out, indent=2))