# python/inference.py
# Reads newline-delimited JSON from stdin: {"clientId": "...", "b64": "..."}
# For each line, decodes base64 into image, runs model inference, writes newline-delimited JSON to stdout:
# {"clientId": "...", "boxes": [ { "x1":.., "y1":.., "x2":.., "y2":.., "conf":.., "class": ".."}, ... ]}

import sys
import json
import cv2
import numpy as np
import traceback
import os
from contextlib import redirect_stdout

# Disable Ultralytics logger to avoid printing to stdout
try:
    from ultralytics.utils import LOGGER
    LOGGER.disabled = True
except Exception:
    pass

# ================ USER: change this function to match how you load YOLOv11 ============
# Example pseudo-code; replace with your actual model-loading & inference APIs.
def load_model():
    # Example if you use ultralytics YOLO:
    from ultralytics import YOLO
    model = YOLO('ppe_yolov11_v5/weights/best.pt')
    return model
    
    # If you use a custom YOLOv11 loader, load it here and return object with .predict(image) -> detections
    raise RuntimeError("Please edit load_model() in inference.py to load your YOLOv11 model")
# ====================================================================================

# Example convert detection result to boxes list:
def decode_jpeg_bytes(b):
    arr = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def detections_to_boxes(dets, model=None):
    # convert your model's detection output to list of dicts:
    # each dict: {"x1":..., "y1":..., "x2":..., "y2":..., "conf":..., "class": ..., "name": ..., "id": ...}
    boxes = []
    model_names = getattr(model, 'names', {}) if model is not None else {}
    for r in dets:
        if not hasattr(r, 'boxes') or r.boxes is None:
            continue
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            cls = int(box.cls[0]) if hasattr(box, "cls") else -1
            tid = int(box.id[0]) if hasattr(box, "id") and box.id is not None else None
            name = model_names.get(cls, str(cls))
            boxes.append({
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "conf": conf,
                "class": cls,
                "name": name,
                "id": tid
            })
    return boxes

def main():
    try:
        model = load_model()
    except Exception as e:
        sys.stderr.write("Model load failed: " + str(e) + "\n")
        traceback.print_exc()
        return

    buf = sys.stdin.buffer
    while True:
        # read header line
        header = buf.readline()
        if not header:
            break
        try:
            meta = json.loads(header.decode('utf8').strip())
            length = int(meta.get('len', 0))
            clientId = meta.get('clientId')
            if length <= 0:
                # nothing to read; respond empty
                out = {"clientId": clientId, "boxes": []}
                sys.stdout.write(json.dumps(out) + "\n")
                sys.stdout.flush()
                continue
            # read exactly length bytes
            data = buf.read(length)
            if not data or len(data) < length:
              # incomplete, respond empty
              out = {"clientId": clientId, "boxes": []}
              sys.stdout.write(json.dumps(out) + "\n")
              sys.stdout.flush()
              continue
            # decode image
            img = decode_jpeg_bytes(data)
            if img is None:
                out = {"clientId": clientId, "boxes": []}
                sys.stdout.write(json.dumps(out) + "\n")
                sys.stdout.flush()
                continue

            # ----- RUN INFERENCE WITH TRACKING -----
            # Choose tracker: 'bytetrack.yaml' (simpler, faster) or 'botsort.yaml' (with ReID)
            tracker_cfg = os.path.join(os.path.dirname(__file__), 'trackers', 'botsort.yaml')
            
            # Fallback: if tracker file not found or fails, use default tracking
            try:
                if os.path.isfile(tracker_cfg):
                    with redirect_stdout(sys.stderr):
                        results = model.track(
                            img,
                            imgsz=640,
                            conf=0.25,
                            persist=True,
                            verbose=False,
                            tracker=tracker_cfg
                        )
                else:
                    # Fallback: use default tracker (no custom config)
                    with redirect_stdout(sys.stderr):
                        results = model.track(
                            img,
                            imgsz=640,
                            conf=0.25,
                            persist=True,
                            verbose=False
                        )
            except Exception as track_err:
                # If tracking fails, fall back to simple predict (no tracking IDs)
                sys.stderr.write(f"Tracking failed: {track_err}, using predict instead\n")
                with redirect_stdout(sys.stderr):
                    results = model.predict(
                        img,
                        imgsz=640,
                        conf=0.25,
                        verbose=False
                    )
            # ----------------------------------------
            boxes = detections_to_boxes(results, model)
            out = {"clientId": clientId, "boxes": boxes}
            sys.stdout.write(json.dumps(out) + "\n")
            sys.stdout.flush()
        except Exception as e:
            traceback.print_exc()
            try:
                out = {"clientId": clientId if 'clientId' in locals() else None, "boxes": []}
                sys.stdout.write(json.dumps(out) + "\n")
                sys.stdout.flush()
            except:
                pass

if __name__ == "__main__":
    main()
