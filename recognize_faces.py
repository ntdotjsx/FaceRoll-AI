import cv2
import json
import numpy as np
import requests
import time
import math
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI()

# ─── 1. โหลดโมเดลทั้งหมด ──────────────────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.read("trainer.yml")

with open("id_to_name.json") as f:
    id_to_name = json.load(f)

face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("lbfmodel.yaml")

DISCORD_WEBHOOK_URL = "YOUR_DISCORD_WEBHOOK_URL"
last_sent = {"name": None, "timestamp": 0}
cooldown = 5  # วินาที

# ─── 2. ฟังก์ชันช่วยเหลือ ──────────────────────────────────
def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(2.0, (8,8))
    e = clahe.apply(roi_gray)
    e = cv2.medianBlur(e, 3)
    return cv2.resize(e, (160,160))

def augment(image):
    out = [image]
    for ang in (-10, 10):
        M = cv2.getRotationMatrix2D((80,80), ang, 1)
        out.append(cv2.warpAffine(image, M, (160,160)))
    out.append(cv2.flip(image, 1))
    return out

def detect_and_align(img, conf_thresh=0.6):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), (104,177,123), swapRB=False)
    face_net.setInput(blob)
    dets = face_net.forward()[0,0,:,:]

    boxes = []
    for d in dets:
        if d[2] < conf_thresh: continue
        x1,y1,x2,y2 = (d[3:7] * [w,h,w,h]).astype(int)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        side = max(x2-x1, y2-y1)
        cx, cy = x1 + (x2-x1)//2, y1 + (y2-y1)//2
        x1m = max(0, cx-side//2)
        y1m = max(0, cy-side//2)
        x2m = min(w, cx+side//2)
        y2m = min(h, cy+side//2)
        boxes.append((x1m, y1m, x2m-x1m, y2m-y1m))

    if not boxes:
        return None

    bx, by, bw, bh = max(boxes, key=lambda b: b[2]*b[3])
    face = img[by:by+bh, bx:bx+bw]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    ok, lm = landmark_detector.fit(face, np.array([[[0,0,bw,bh]]]))
    if ok and len(lm[0]) >= 2:
        left_eye  = np.mean(lm[0][36:42], axis=0)
        right_eye = np.mean(lm[0][42:48], axis=0)
        dx, dy = right_eye - left_eye
        ang = math.degrees(math.atan2(dy, dx))
        M = cv2.getRotationMatrix2D((bw/2, bh/2), ang, 1)
        face = cv2.warpAffine(face, M, (bw, bh))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    return gray

def send_to_discord(name, conf, frame):
    now = time.time()
    if name == last_sent["name"] and now - last_sent["timestamp"] < cooldown:
        return
    last_sent.update(name=name, timestamp=now)
    _, buf = cv2.imencode(".jpg", frame)
    files = {"file": ("img.jpg", BytesIO(buf.tobytes()), "image/jpeg")}
    data  = {"content": f"✅ **{name}** ({conf:.2f}%)"}
    try:
        requests.post(DISCORD_WEBHOOK_URL, data=data, files=files).raise_for_status()
    except:
        pass

# ─── 3. Endpoint ตรงกับ test_model.py เป๊ะ ─────────────────
@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image format")

    gface = detect_and_align(img, conf_thresh=0.5)
    if gface is None:
        return {"result": "Unknown", "confidence": 0.0}

    proc = enhance_face(gface)

    label, conf = recognizer.predict(proc)
    name = id_to_name.get(str(label), "Unknown")

    return {"result": name, "confidence": round(conf, 2)}
