import cv2
import json
import numpy as np
import requests
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO
import math

app = FastAPI()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("lbfmodel.yaml")

# Load ID to name mapping
with open("id_to_name.json", "r") as json_file:
    id_to_name = json.load(json_file)

DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"  # Discord Webhook URL
last_sent = {"name": None, "timestamp": 0}
cooldown = 5  # Time cooldown in seconds to avoid spamming

# ฟังก์ชันเพิ่มความคมชัด
def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(2.0, (8,8))
    e = clahe.apply(roi_gray)
    e = cv2.medianBlur(e, 3)
    return cv2.resize(e, (160, 160))  # ใช้ square crop

# ฟังก์ชันการหมุน, พลิก, และเพิ่มข้อมูล
def augment(image):
    out = [image]
    # หมุน ±10°
    for ang in (-10, 10):
        M = cv2.getRotationMatrix2D((80, 80), ang, 1)
        out.append(cv2.warpAffine(image, M, (160, 160)))
    # พลิกซ้ายขวา
    out.append(cv2.flip(image, 1))
    return out

# ฟังก์ชันตรวจจับและปรับใบหน้า
def detect_and_align(img, conf_thresh=0.6):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 177, 123), swapRB=False)
    face_net.setInput(blob)
    det = face_net.forward()[0, 0, :, :]

    boxes = []
    for i in range(det.shape[0]):
        score = float(det[i, 2])
        if score < conf_thresh: continue
        x1, y1, x2, y2 = (det[i, 3:7] * [w, h, w, h]).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        # margin 20%
        side = max(x2 - x1, y2 - y1)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        x1m = max(0, cx - side // 2)
        y1m = max(0, cy - side // 2)
        x2m = min(w, cx + side // 2)
        y2m = min(h, cy + side // 2)
        boxes.append((x1m, y1m, x2m - x1m, y2m - y1m))

    if not boxes:
        return None
    # เลือก box ใหญ่สุด
    bx, by, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
    face = img[by:by + bh, bx:bx + bw]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # align ด้วย landmark
    ok, landmarks = landmark_detector.fit(face, np.array([[0, 0, bw, bh]]))
    if ok and len(landmarks[0]) >= 2:
        # ตำแหน่งดวงตา
        left_eye = np.mean(landmarks[0][36:42], axis=0)
        right_eye = np.mean(landmarks[0][42:48], axis=0)
        dx, dy = right_eye - left_eye
        angle = math.degrees(math.atan2(dy, dx))
        M = cv2.getRotationMatrix2D((bw / 2, bh / 2), angle, 1)
        face = cv2.warpAffine(face, M, (bw, bh))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    return gray

# ฟังก์ชันส่งข้อมูลไป Discord
def send_to_discord(name, confidence, image):
    now = time.time()
    if name == last_sent["name"] and now - last_sent["timestamp"] < cooldown:
        return
    last_sent["name"] = name
    last_sent["timestamp"] = now

    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()
    timestamp = int(now)
    filename = f"detected_face_{timestamp}.jpg"

    content = f"✅ **{name}** เข้าเรียน คาบ: ({confidence:.2f}%)"

    try:
        files = {"file": (filename, BytesIO(img_bytes), "image/jpeg")}
        data = {"content": content}
        response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
        response.raise_for_status()
    except requests.RequestException as e:
        print("Error sending to Discord:", e)

# ฟังก์ชันตรวจจับใบหน้า
@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    try:
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Detect face with DNN and align it
        gray = detect_and_align(frame)
        if gray is not None:
            # Enhance + augment
            proc = enhance_face(gray)
            augmented_images = augment(proc)

            best_name = "Unknown"
            best_confidence = 100
            for aug in augmented_images:
                id_, confidence = recognizer.predict(aug)
                if confidence < best_confidence:
                    best_confidence = confidence
                    best_name = id_to_name.get(str(id_), "Unknown")

            confidence_percent = round(100 - best_confidence, 2)
            send_to_discord(best_name, confidence_percent, frame)
            return {"result": best_name, "confidence": confidence_percent}
        else:
            return {"result": "No Face", "confidence": ""}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})