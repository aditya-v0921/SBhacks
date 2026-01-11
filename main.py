#!/usr/bin/env python3
"""
Vibe-Check: Biometric-Driven Crowd Analytics
With fixed hype calculation that scales with crowd size
"""

import time
import asyncio
import base64
import numpy as np
import cv2
import socketio
from ultralytics import YOLO
from aiohttp import web

# Optional: InfluxDB
try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import ASYNCHRONOUS
    INFLUX_AVAILABLE = True
except ImportError:
    INFLUX_AVAILABLE = False

# C++ motion engine
import vibe_core

# CONFIGURATION

CAM_SOURCE = 1
PREFERRED_RESOLUTION = (1280, 720)
PREFERRED_FPS = 30

GRID_H, GRID_W = 8, 8
POSE_EVERY_N_FRAMES = 2
CONF_THRESH = 0.35
EMIT_RATE_HZ = 15
JPEG_QUALITY = 70

# HYPE CALCULATION SETTINGS

# Minimum people required for hype to register
MIN_PEOPLE_FOR_HYPE = 1  # Set to 2+ to require multiple people

# How many people are needed for "full crowd"
FULL_CROWD_SIZE = 5  # With 5+ people, crowd multiplier = 1.0

# Weights for hype calculation
WEIGHT_HANDS_UP = 0.5
WEIGHT_JUMPING = 0.3 
WEIGHT_MOVEMENT = 0.2

# INFLUXDB CONFIGURATION

INFLUX_ENABLED = True
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "P6qMhteG793e4V1Sje3OTJ5H4zinv7C-zwuqqFQTLadGmf0Caxa1nOMv8djJJRpPdqZ6JoUT-uG8dJzSPYxCYQ=="
INFLUX_ORG = "Crowd Check"
INFLUX_BUCKET = "crowd_metrics"

# INFLUXDB SETUP

influx_client = None
influx_write_api = None

if INFLUX_ENABLED and INFLUX_AVAILABLE:
    try:
        influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        influx_write_api = influx_client.write_api(write_options=ASYNCHRONOUS)
        print(f"InfluxDB connected: {INFLUX_URL}")
    except Exception as e:
        print(f"InfluxDB connection failed: {e}")
        INFLUX_ENABLED = False

# SOCKET.IO SERVER

sio = socketio.AsyncServer(
    async_mode="aiohttp",
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024
)
app = web.Application()
sio.attach(app)

# POSE DETECTION SETUP

KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

SKELETON_CONNECTIONS = [
    (KP_L_EYE, KP_R_EYE), (KP_L_EYE, KP_NOSE), (KP_R_EYE, KP_NOSE),
    (KP_L_EYE, KP_L_EAR), (KP_R_EYE, KP_R_EAR),
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP), (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE), (KP_R_HIP, KP_R_KNEE),
    (KP_L_KNEE, KP_L_ANKLE), (KP_R_KNEE, KP_R_ANKLE),
]

COLORS = {
    'face': (255, 200, 100),
    'arms': (100, 255, 100),
    'torso': (100, 200, 255),
    'legs': (255, 100, 255),
}


def get_limb_color(idx1, idx2):
    if idx1 in {0, 1, 2, 3, 4} or idx2 in {0, 1, 2, 3, 4}:
        return COLORS['face']
    elif idx1 in {7, 8, 9, 10} or idx2 in {7, 8, 9, 10}:
        return COLORS['arms']
    elif idx1 in {13, 14, 15, 16} or idx2 in {13, 14, 15, 16}:
        return COLORS['legs']
    return COLORS['torso']


def compute_person_metrics(kpts_xyc, prev_kpts=None):
    """
    Compute hype metrics for a SINGLE person.
    Returns dict with hands_up (0-1), jumping (0-1), arm_movement (0-1)
    """
    result = {
        "hands_up": 0.0,
        "jumping": 0.0,
        "arm_movement": 0.0,
        "is_hyped": False
    }
    
    nose = kpts_xyc[KP_NOSE]
    l_wrist, r_wrist = kpts_xyc[KP_L_WRIST], kpts_xyc[KP_R_WRIST]
    l_elbow, r_elbow = kpts_xyc[KP_L_ELBOW], kpts_xyc[KP_R_ELBOW]
    l_shoulder, r_shoulder = kpts_xyc[KP_L_SHOULDER], kpts_xyc[KP_R_SHOULDER]
    
    # Need some upper body visibility
    if nose[2] < CONF_THRESH and l_shoulder[2] < CONF_THRESH and r_shoulder[2] < CONF_THRESH:
        return result
    
    # Reference point for "above head"
    if nose[2] >= CONF_THRESH:
        head_y = nose[1]
    else:
        head_y = (l_shoulder[1] + r_shoulder[1]) / 2 - 50
    
    # Check if wrists are above head level
    left_hand_up = l_wrist[2] >= CONF_THRESH and l_wrist[1] < head_y
    right_hand_up = r_wrist[2] >= CONF_THRESH and r_wrist[1] < head_y
    
    # Also check elbows for partially raised arms
    left_elbow_up = l_elbow[2] >= CONF_THRESH and l_elbow[1] < head_y + 30
    right_elbow_up = r_elbow[2] >= CONF_THRESH and r_elbow[1] < head_y + 30
    
    hands_score = 0.0
    if left_hand_up:
        hands_score += 0.5
    elif left_elbow_up:
        hands_score += 0.25
    
    if right_hand_up:
        hands_score += 0.5
    elif right_elbow_up:
        hands_score += 0.25
    
    result["hands_up"] = min(1.0, hands_score)
    
    if prev_kpts is not None:
        l_ankle, r_ankle = kpts_xyc[KP_L_ANKLE], kpts_xyc[KP_R_ANKLE]
        l_hip, r_hip = kpts_xyc[KP_L_HIP], kpts_xyc[KP_R_HIP]
        prev_l_ankle, prev_r_ankle = prev_kpts[KP_L_ANKLE], prev_kpts[KP_R_ANKLE]
        prev_l_hip, prev_r_hip = prev_kpts[KP_L_HIP], prev_kpts[KP_R_HIP]
        
        # Check vertical movement of ankles AND hips (more reliable)
        movements = []
        
        if l_ankle[2] >= CONF_THRESH and prev_l_ankle[2] >= CONF_THRESH:
            movements.append(prev_l_ankle[1] - l_ankle[1])
        if r_ankle[2] >= CONF_THRESH and prev_r_ankle[2] >= CONF_THRESH:
            movements.append(prev_r_ankle[1] - r_ankle[1])
        if l_hip[2] >= CONF_THRESH and prev_l_hip[2] >= CONF_THRESH:
            movements.append(prev_l_hip[1] - l_hip[1])
        if r_hip[2] >= CONF_THRESH and prev_r_hip[2] >= CONF_THRESH:
            movements.append(prev_r_hip[1] - r_hip[1])
        
        if movements:
            avg_movement = np.mean(movements)
            # Lower threshold for easier jump detection
            if avg_movement > 8:  # Reduced from 15
                result["jumping"] = min(1.0, avg_movement / 25.0)  # Easier to max out
    
    # ARM MOVEMENT DETECTION
    if prev_kpts is not None:
        prev_l_wrist, prev_r_wrist = prev_kpts[KP_L_WRIST], prev_kpts[KP_R_WRIST]
        
        arm_movements = []
        if l_wrist[2] >= CONF_THRESH and prev_l_wrist[2] >= CONF_THRESH:
            dx = abs(l_wrist[0] - prev_l_wrist[0])
            dy = abs(l_wrist[1] - prev_l_wrist[1])
            arm_movements.append(np.sqrt(dx*dx + dy*dy))
        
        if r_wrist[2] >= CONF_THRESH and prev_r_wrist[2] >= CONF_THRESH:
            dx = abs(r_wrist[0] - prev_r_wrist[0])
            dy = abs(r_wrist[1] - prev_r_wrist[1])
            arm_movements.append(np.sqrt(dx*dx + dy*dy))
        
        if arm_movements:
            avg_arm = np.mean(arm_movements)
            result["arm_movement"] = min(1.0, avg_arm / 30.0)
    
    # Is this person "hyped"? (hands up OR significant movement)
    result["is_hyped"] = result["hands_up"] > 0.3 or result["jumping"] > 0.3 or result["arm_movement"] > 0.5
    
    return result


def calculate_crowd_hype(people_metrics, people_count, mean_energy):
    """
    Calculate overall crowd hype score.
    
    This requires multiple people and scales with crowd engagement.
    """
    # Not enough people = no hype
    if people_count < MIN_PEOPLE_FOR_HYPE:
        return {
            "hype_score": 0.0,
            "hands_up_ratio": 0.0,
            "jumping_ratio": 0.0,
            "hyped_people": 0,
            "crowd_multiplier": 0.0
        }
    
    if not people_metrics:
        return {
            "hype_score": 0.0,
            "hands_up_ratio": 0.0,
            "jumping_ratio": 0.0,
            "hyped_people": 0,
            "crowd_multiplier": 0.0
        }
    
    # Calculate individual metrics
    hands_up_values = [p["hands_up"] for p in people_metrics]
    jumping_values = [p["jumping"] for p in people_metrics]
    arm_movement_values = [p["arm_movement"] for p in people_metrics]
    hyped_count = sum(1 for p in people_metrics if p["is_hyped"])
    
    # Average metrics across all people
    avg_hands_up = np.mean(hands_up_values)
    avg_jumping = np.mean(jumping_values)
    avg_arm_movement = np.mean(arm_movement_values)
    
    # Crowd multiplier: scales from 0 to 1 based on crowd size
    # With 1 person: multiplier = 0.2 (low hype possible)
    # With FULL_CROWD_SIZE people: multiplier = 1.0
    crowd_multiplier = min(1.0, 0.2 + (people_count / FULL_CROWD_SIZE) * 0.8)
    
    # Participation bonus: what % of people are hyped?
    participation_ratio = hyped_count / people_count
    
    # Base hype from gestures
    gesture_hype = (
        avg_hands_up * WEIGHT_HANDS_UP +
        avg_jumping * WEIGHT_JUMPING +
        avg_arm_movement * WEIGHT_MOVEMENT
    )
    
    # Normalize energy to 0-1 scale (assuming max ~15)
    energy_factor = min(1.0, mean_energy / 10.0)
    
    # Final hype calculation:
    # gesture_hype (0-1) * crowd_multiplier (0-1) * participation_boost
    participation_boost = 0.5 + (participation_ratio * 0.5)  # 0.5 to 1.0
    
    raw_hype = gesture_hype * crowd_multiplier * participation_boost
    
    # Add energy contribution (movement even without detected gestures)
    energy_contribution = energy_factor * 0.3 * crowd_multiplier
    
    final_hype = min(1.0, raw_hype + energy_contribution)
    
    return {
        "hype_score": float(final_hype),
        "hands_up_ratio": float(avg_hands_up),
        "jumping_ratio": float(avg_jumping),
        "hyped_people": int(hyped_count),
        "crowd_multiplier": float(crowd_multiplier)
    }


def classify_zones(heatmap):
    flat = heatmap.flatten()
    if flat.max() == 0:
        return [["low"] * heatmap.shape[1] for _ in range(heatmap.shape[0])]
    
    nonzero = flat[flat > 0]
    p33 = np.percentile(nonzero, 33) if len(nonzero) > 0 else 0
    p66 = np.percentile(nonzero, 66) if len(nonzero) > 0 else 0
    
    zones = []
    for row in heatmap:
        zone_row = []
        for val in row:
            if val <= p33:
                zone_row.append("low")
            elif val <= p66:
                zone_row.append("medium")
            else:
                zone_row.append("high")
        zones.append(zone_row)
    return zones


def draw_skeleton(frame, keypoints, people_metrics, conf_thresh=0.3):
    overlay = frame.copy()
    
    for idx, person_kpts in enumerate(keypoints):
        # Check if this person is hyped
        is_hyped = False
        if idx < len(people_metrics):
            is_hyped = people_metrics[idx].get("is_hyped", False)
        
        # Use brighter colors for hyped people
        color_boost = 1.5 if is_hyped else 1.0
        
        for idx1, idx2 in SKELETON_CONNECTIONS:
            pt1, pt2 = person_kpts[idx1], person_kpts[idx2]
            if pt1[2] >= conf_thresh and pt2[2] >= conf_thresh:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                color = get_limb_color(idx1, idx2)
                
                # Brighten color for hyped people
                if is_hyped:
                    color = tuple(min(255, int(c * color_boost)) for c in color)
                
                cv2.line(overlay, (x1, y1), (x2, y2), color, 8)
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        
        for i, kpt in enumerate(person_kpts):
            if kpt[2] >= conf_thresh:
                x, y = int(kpt[0]), int(kpt[1])
                if i in {0, 1, 2, 3, 4}:
                    color = COLORS['face']
                elif i in {5, 6, 7, 8, 9, 10}:
                    color = COLORS['arms']
                elif i in {11, 12}:
                    color = COLORS['torso']
                else:
                    color = COLORS['legs']
                
                if is_hyped:
                    color = tuple(min(255, int(c * color_boost)) for c in color)
                
                cv2.circle(overlay, (x, y), 10, color, -1)
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)
        
        # if is_hyped:
        #     nose = person_kpts[KP_NOSE]
        #     if nose[2] >= conf_thresh:
        #         cv2.putText(frame, "HYPE!", (int(nose[0]) - 30, int(nose[1]) - 60),
        #                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame


# CAMERA SETUP

def open_camera(source):
    if source == "auto":
        # Find first available camera
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"  ✓ Found camera at index {i}")
                    source = i
                    cap.release()
                    break
                cap.release()
    
    print(f"\nOpening camera: {source}")
    
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {source}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, PREFERRED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution: {w}x{h}")
    
    return cap


# MAIN VISION LOOP

async def vision_loop():
    cap = open_camera(CAM_SOURCE)
    engine = vibe_core.MotionEngine(GRID_H, GRID_W)
    
    print("\nLoading YOLOv8 pose model...")
    model = YOLO("yolov8n-pose.pt")
    
    ret, warm = cap.read()
    if ret:
        _ = model.predict(warm, verbose=False, conf=CONF_THRESH)
        print(" Model ready")
    
    frame_count = 0
    last_emit = time.time()
    last_influx_write = time.time()
    prev_kpts_map = {}
    last_keypoints = None
    last_people_metrics = []
    
    # Smoothing history
    energy_hist = []
    hype_hist = []
    
    print("\n" + "=" * 50)
    print("Vision loop started!")
    print("   Dashboard: http://localhost:3000")
    print(f"   Min people for hype: {MIN_PEOPLE_FOR_HYPE}")
    print(f"   Full crowd size: {FULL_CROWD_SIZE}")
    print("=" * 50 + "\n")
    
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            display = frame.copy()
            heatmap, mean_energy = engine.process(frame)
            
            people_count = 0
            people_metrics = []
            keypoints_list = []
            
            if frame_count % POSE_EVERY_N_FRAMES == 0:
                results = model.predict(frame, verbose=False, conf=CONF_THRESH)
                r0 = results[0]
                
                if r0.keypoints is not None and r0.keypoints.data is not None:
                    kpts = r0.keypoints.data.cpu().numpy()
                    people_count = kpts.shape[0]
                    last_keypoints = kpts
                    
                    for i in range(people_count):
                        # Get metrics for this person
                        metrics = compute_person_metrics(kpts[i], prev_kpts_map.get(i))
                        people_metrics.append(metrics)
                        prev_kpts_map[i] = kpts[i].copy()
                        
                        # Keypoints for frontend
                        keypoints_list.append([
                            {"x": float(k[0]), "y": float(k[1]), "conf": float(k[2])}
                            for k in kpts[i]
                        ])
                    
                    last_people_metrics = people_metrics
                    
                    # Clean up old entries
                    prev_kpts_map = {k: v for k, v in prev_kpts_map.items() if k < people_count}
            
            # Calculate crowd hype
            crowd_hype = calculate_crowd_hype(last_people_metrics, people_count, mean_energy)
            
            # Draw skeleton with hype indicators
            if last_keypoints is not None and len(last_keypoints) > 0:
                display = draw_skeleton(display, last_keypoints, last_people_metrics, CONF_THRESH)
            
            # Stats overlay
            cv2.putText(display, f"People: {people_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Hyped: {crowd_hype['hyped_people']}/{people_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Hype: {crowd_hype['hype_score']*100:.0f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            cv2.putText(display, f"Energy: {mean_energy:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 200, 255), 2)
            
            frame_count += 1
            t1 = time.time()
            
            # Smoothing
            energy_hist.append(float(mean_energy))
            hype_hist.append(crowd_hype['hype_score'])
            if len(energy_hist) > 10:
                energy_hist.pop(0)
                hype_hist.pop(0)
            
            smoothed_energy = float(np.mean(energy_hist))
            smoothed_hype = float(np.mean(hype_hist))
            
            # Write to InfluxDB
            now = time.time()
            if INFLUX_ENABLED and influx_write_api and now - last_influx_write >= 1.0:
                try:
                    point = (
                        Point("crowd_metrics")
                        .field("energy", smoothed_energy)
                        .field("hype_score", smoothed_hype)
                        .field("people_count", people_count)
                        .field("hyped_people", crowd_hype['hyped_people'])
                        .field("hands_up_ratio", crowd_hype['hands_up_ratio'])
                        .field("jumping_ratio", crowd_hype['jumping_ratio'])
                    )
                    influx_write_api.write(bucket=INFLUX_BUCKET, record=point)
                except Exception as e:
                    print(f"InfluxDB write error: {e}")
                last_influx_write = now
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "ts": time.time(),
                "meanEnergy": float(smoothed_energy),
                "heatmap": [[float(v) for v in row] for row in heatmap.tolist()],
                "zones": classify_zones(heatmap),
                "hypeScore": float(smoothed_hype),
                "handsUpRatio": float(crowd_hype['hands_up_ratio']),
                "jumpingRatio": float(crowd_hype['jumping_ratio']),
                "peopleCount": int(people_count),
                "hypedPeople": int(crowd_hype['hyped_people']),
                "crowdMultiplier": float(crowd_hype['crowd_multiplier']),
                "latencyMs": float(round((t1 - t0) * 1000, 1)),
                "fps": float(round(1.0 / max(0.001, t1 - t0), 1)),
                "frame": frame_b64,
                "keypoints": keypoints_list,
                "frameWidth": int(display.shape[1]),
                "frameHeight": int(display.shape[0])
            }
            
            if now - last_emit >= (1.0 / EMIT_RATE_HZ):
                await sio.emit("vibe_update", payload)
                last_emit = now
            
            await asyncio.sleep(0)
    finally:
        cap.release()
        if influx_client:
            influx_client.close()


# HTTP ROUTES

async def index_handler(request):
    return web.Response(text="Vibe-Check Server Running", content_type="text/plain")

app.router.add_get("/", index_handler)


# APP LIFECYCLE

async def start_background_tasks(app):
    app["vision_task"] = asyncio.create_task(vision_loop())

async def cleanup_background_tasks(app):
    app["vision_task"].cancel()
    try:
        await app["vision_task"]
    except asyncio.CancelledError:
        pass

app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)


if __name__ == "__main__":
    print("""
VIBE CHECK!
    """)
    
    print(f"Hype Settings:")
    print(f"   Min people for hype: {MIN_PEOPLE_FOR_HYPE}")
    print(f"   Full crowd size: {FULL_CROWD_SIZE}")
    print(f"   Weights: hands={WEIGHT_HANDS_UP}, jump={WEIGHT_JUMPING}, move={WEIGHT_MOVEMENT}")
    print()
    
    web.run_app(app, host="0.0.0.0", port=8000)