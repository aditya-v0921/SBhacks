import time
import json
import asyncio
import numpy as np
import cv2
import socketio

from ultralytics import YOLO
import vibe_core  # the compiled .so in this folder

# ---------- Config ----------
CAM_INDEX = 1          # you may need 0/1/2 depending on system
GRID_H, GRID_W = 8, 8
POSE_EVERY_N_FRAMES = 3    # run pose less often to reduce jitter
CONF_THRESH = 0.35

# ---------- Socket.io Server ----------
sio = socketio.AsyncServer(async_mode="aiohttp", cors_allowed_origins="*")
from aiohttp import web
app = web.Application()
sio.attach(app)

# ---------- Helpers ----------
# Ultralytics COCO keypoint order includes: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles. :contentReference[oaicite:5]{index=5}
KP_NOSE = 0
KP_L_WRIST = 9
KP_R_WRIST = 10

def compute_hype_from_keypoints(kpts_xyc: np.ndarray) -> float:
    """
    kpts_xyc: shape (17, 3) -> (x, y, conf)
    Returns 1.0 if "hands up" detected for a person, else 0.0
    """
    nose = kpts_xyc[KP_NOSE]
    lw = kpts_xyc[KP_L_WRIST]
    rw = kpts_xyc[KP_R_WRIST]

    # Must have decent confidence
    if nose[2] < CONF_THRESH:
        return 0.0

    # If either wrist is confident and above nose (smaller y = higher in image)
    hands_up = False
    if lw[2] >= CONF_THRESH and lw[1] < nose[1]:
        hands_up = True
    if rw[2] >= CONF_THRESH and rw[1] < nose[1]:
        hands_up = True

    return 1.0 if hands_up else 0.0

async def vision_loop():
    # ---- Camera open ----
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAM_INDEX}. "
            f"Try changing CAM_INDEX and ensure Camera permissions are enabled."
        )

    # Try to request 60fps + 1080p (may be ignored depending on device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)

    engine = vibe_core.MotionEngine(GRID_H, GRID_W)

    # YOLO pose model
    # For speed: yolov8n-pose.pt (nano). Bigger models will add latency.
    model = YOLO("yolov8n-pose.pt")

    frame_count = 0
    last_send = time.time()

    print("Starting vision loop. Press Ctrl+C to stop.")
    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            await asyncio.sleep(0.01)
            continue

        # C++ motion heatmap
        heatmap, mean_energy = engine.process(frame)  # heatmap is numpy array float32

        hype_score = 0.0
        people_count = 0

        # Pose every N frames to keep latency stable
        if frame_count % POSE_EVERY_N_FRAMES == 0:
            # Ultralytics expects BGR (OpenCV) or RGB; it can handle ndarray.
            results = model.predict(frame, verbose=False, conf=CONF_THRESH)

            # results[0].keypoints can be None if no person
            r0 = results[0]
            if r0.keypoints is not None:
                # keypoints.xyn / xy / data; safest is .data -> (num_people, 17, 3)
                kpts = r0.keypoints.data
                if kpts is not None:
                    kpts_np = kpts.cpu().numpy()
                    people_count = kpts_np.shape[0]
                    if people_count > 0:
                        hype_vals = [compute_hype_from_keypoints(kpts_np[i]) for i in range(people_count)]
                        hype_score = float(np.mean(hype_vals))

        frame_count += 1
        t1 = time.time()

        payload = {
            "ts": time.time(),
            "meanEnergy": float(mean_energy),
            "heatmap": heatmap.tolist(),
            "hypeScore": float(hype_score),
            "peopleCount": int(people_count),
            "latencyMs": (t1 - t0) * 1000.0
        }

        # Broadcast ~20 times/sec max (avoid flooding)
        now = time.time()
        if now - last_send >= 0.05:
            await sio.emit("vibe_update", payload)
            last_send = now

        # Small yield so event loop stays responsive
        await asyncio.sleep(0)

async def index(request):
    return web.Response(text="Socket.io server running. React client should connect and listen for 'vibe_update'.")

app.router.add_get("/", index)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.create_task(vision_loop())
    web.run_app(app, host="0.0.0.0", port=8000)
