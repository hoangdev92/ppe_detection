# PPE Detection Realtime Demo

A small end-to-end demo that streams video frames from a browser to a Node.js backend, performs YOLO-based PPE detection (helmet, vest, glove, boots, person) in Python, and returns detections in realtime via WebSocket. The server also detects PPE violations per person (using tracker IDs), logs them to SQLite via Sequelize, and can send email notifications.

## Features
- Realtime video processing from webcam, file, or URL (proxied).
- Python inference using Ultralytics YOLO with optional built-in tracking to get stable `track id` per person.
- Frontend overlay draws detection boxes accurately (handles letterboxing).
- Backend PPE violation logic: only alerts/logs once per track and per missing PPE set (debounced/persistent).
- SQLite logging (`violations` table) and optional email notifications.

## Tech Stack
- Backend: Node.js (Express, ws, Sequelize, SQLite)
- Frontend: Vue (Vite) + Canvas overlay
- Inference: Python 3 + Ultralytics YOLO

## Repository Layout
```
train_ai/
  app.js                 # Node server + WebSocket + proxy + Python bridge
  models/
    index.js             # Sequelize init (SQLite)
    Violation.js         # Violation model
  services/
    notify.js            # Email notification service (nodemailer)
  frontend/              # Vite + Vue app
  python/
    inference.py         # Python bridge: reads JPEG bytes, runs YOLO, prints JSON
    test_model.py        # One-off script to test model on a single image
    trackers/
      botsort.yaml       # (Optional) BoT-SORT tracker config for stable IDs
  ppe_yolov11/weights/   # Your trained weights (best.pt, last.pt)
```

## Prerequisites
- Node.js 20+
- Python 3.8+ (with pip)
- GPU optional (CPU works but is slower)

## Install
### 1) Backend (Node)
```bash
cd /home/hoang-lv/Documents/projects/nodejs/train_ai
npm install
```

### 2) Frontend
```bash
cd frontend
npm install
```

### 3) Python (inference)
```bash
cd /home/hoang-lv/Documents/projects/nodejs/train_ai
python3 -m pip install -r python/requirements.txt
# Optional but recommended for tracker ReID stability (if enabled):
python3 -m pip install -U lapx filterpy onemetric torchreid
```

## Configuration
### Email (optional)
Export environment variables before starting the server if you want email alerts:
```bash
export MAIL_FROM="alerts@example.com"
export MAIL_TO="safety@example.com"
export SMTP_HOST="smtp.example.com"
export SMTP_PORT="587"
export SMTP_USER="smtp-user"
export SMTP_PASS="smtp-pass"
```
If not set, email sending is skipped.

### YOLO Weights
Place your weights at `ppe_yolov11/weights/best.pt` (default path used by the code).

### Tracker (optional)
- Python uses Ultralytics built-in tracking (`model.track(..., persist=True)`).
- To enable BoT-SORT with ReID, ensure `python/trackers/botsort.yaml` exists and pass it in `inference.py` (already wired). If you hit environment/version issues, first run without `tracker=...` (still uses `persist=True`).

## Running
### Start backend (Node)
```bash
cd /home/hoang-lv/Documents/projects/nodejs/train_ai
npm start
# Server on :3004
```

### Start frontend (Vite)
```bash
cd /home/hoang-lv/Documents/projects/nodejs/train_ai/frontend
npm run dev
# Open the URL printed by Vite (e.g. http://localhost:5173)
```

### Use the app
1. Open the frontend page.
2. Choose Webcam, upload a video file, or paste a video URL (MP4/WebM/HLS). For URLs, the backend proxy will stream it with proper CORS.
3. Click Start to begin streaming frames to the server.
4. Detections will draw on the canvas overlay. If a person is missing any of: helmet, vest, glove, boots, the server will:
   - Emit a `violation` WebSocket event to the same client
   - Insert a row into `violations` table (SQLite)
   - Send an email (if configured)

## One-off Model Test (single image)
```bash
# Save annotated result for quick model quality check
python3 python/test_model.py --image /path/to/image.jpg --output output/annotated.jpg \
  --weights ppe_yolov11/weights/best.pt --conf 0.25 --imgsz 640 --show
```

## Troubleshooting
- Bad JSON from python: ensure `inference.py` only prints JSON to stdout. The code redirects internal logs to stderr and disables Ultralytics logger.
- Tracker/Ultralytics config errors: upgrade Ultralytics and clear its caches.
```bash
python3 -m pip uninstall -y ultralytics
rm -rf ~/.config/Ultralytics ~/.cache/ultralytics
python3 -m pip install -U ultralytics
```
- ReID dependencies: If enabling ReID in tracker YAML and encountering import errors, install `lapx`, `filterpy`, `onemetric`, `torchreid`. If installing `torchreid` is troublesome, set `with_reid: false` and use `persist=True` for decent stability.
- Letterboxing boxes offset: the frontend overlay computes the video content rect and offsets boxes to match exactly.

## Data Model
Table `violations` (SQLite):
- id (auto)
- clientId (string)
- missingItems (TEXT JSON array string)
- detectedAt (DATE default NOW)
- frameWidth, frameHeight (optional)

## License
For internal demo/POC use. Replace/extend as needed for production.
