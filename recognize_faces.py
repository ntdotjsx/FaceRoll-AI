import cv2
import json
import numpy as np
import requests
import time
import math
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pymysql

from utils.face import detect_and_align, enhance_face

# ─── โหลดโมเดล Face Detection และ Landmark ──────────────────────
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("lbfmodel.yaml")

app = FastAPI()

# ─── โหลดโมเดล Recognizer ──────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)
recognizer.read("trainer.yml")

with open("id_to_name.json") as f:
    id_to_name = json.load(f)

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1368350205153509456/QoewWgcagORVXglpEAntFn0x4asxAGyEr98DIU-DEymomNT0nA7pgD1-ktl-lNRYSwwD"
last_sent = {"name": None, "timestamp": 0}
cooldown = 5  # วินาที

# ─── Config ──────────────────────────────────────────────────────
CONF_THRESHOLD = 0.4     # Face detection threshold
RECOG_THRESHOLD = 10.0   # LBPH threshold (lower = more sensitive, higher = more strict)

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "1212312121",
    "database": "hee",
    "charset": "utf8mb4",
}

def send_to_discord(name, frame):
    now = time.time()
    if name == last_sent["name"] and now - last_sent["timestamp"] < cooldown:
        return
    last_sent.update(name=name, timestamp=now)
    _, buf = cv2.imencode(".jpg", frame)
    files = {"file": (f"student_{int(now)}.jpg", BytesIO(buf.tobytes()), "image/jpeg")}
    data = {"content": f"✅ **{name}** เข้าเรียนแล้ว"}
    try:
        requests.post(DISCORD_WEBHOOK_URL, data=data, files=files).raise_for_status()
    except:
        pass

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # อ่านภาพจาก request
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    # Detect & Align
    gface = detect_and_align(img, face_net, landmark_detector, conf_thresh=CONF_THRESHOLD)
    if gface is None:
        return {"result": "Unknown", "confidence": 0.0}

    # Enhance
    proc = enhance_face(gface)

    # Predict
    label, conf = recognizer.predict(proc)
    # ถ้า distance มากกว่า threshold → ไม่มั่นใจ ให้คืน Unknown
    if conf > RECOG_THRESHOLD:
        return {"result": "Unknown", "confidence": round(conf, 2)}

    name_key = str(label)
    name = id_to_name.get(name_key, "Unknown")

    # ดึงชื่อ-สกุลจาก DB
    full_name = None
    if name != "SET DATABASE":
        try:
            conn = pymysql.connect(**DB_CONFIG)
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT fname, lname FROM users WHERE folder_name = %s", (name,)
                )
                row = cursor.fetchone()
            conn.close()
            if row:
                full_name = f"{row[0]} {row[1]}"
        except Exception as e:
            print(f"[DB ERROR] {e}")

    if full_name:
        send_to_discord(full_name, img)
        return {"result": "PASS", "confidence": round(conf, 2)}

    return {"result": "NOT PASS", "confidence": round(conf, 2)}
