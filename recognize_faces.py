import cv2
import json
import numpy as np
import requests
import time
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from io import BytesIO

app = FastAPI()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Load ID to name mapping
with open("id_to_name.json", "r") as json_file:
    id_to_name = json.load(json_file)

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1366887058794352640/Wkh8438wAedRXaJwffMOgzGGk5SrjUFTtYSwLy1x9_9V8q8t66yW-TzpAaTbiuRdHYIe"  # Discord Webhook URL

last_sent = {"name": None, "timestamp": 0}
cooldown = 5  # Time cooldown in seconds to avoid spamming


def send_to_discord(name, confidence, image, is_unknown=False):
    now = time.time()
    if name == last_sent["name"] and now - last_sent["timestamp"] < cooldown:
        return
    last_sent["name"] = name
    last_sent["timestamp"] = now

    _, img_encoded = cv2.imencode(".jpg", image)
    img_bytes = img_encoded.tobytes()
    timestamp = int(now)
    filename = f"detected_face_{timestamp}.jpg"

    if is_unknown:
        content = f"⚠️ **ไม่รู้จักบุคคล** (Confidence: {confidence:.2f}%)"
    else:
        content = f"✅ **{name}** เข้าเรียน คาบ: ({confidence:.2f}%)"

    try:
        files = {"file": (filename, BytesIO(img_bytes), "image/jpeg")}
        data = {"content": content}
        response = requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
        response.raise_for_status()
    except requests.RequestException as e:
        print("Error sending to Discord:", e)


@app.post("/detect")
async def detect_face(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(status_code=400, content={"error": "No file uploaded"})
    try:
        # อ่านไฟล์จากการอัปโหลด
        img_bytes = await file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda r: r[2] * r[3])
            roi_gray = gray[y : y + h, x : x + w]
            id_, confidence = recognizer.predict(roi_gray)

            if 0 <= confidence <= 85:
                name = id_to_name.get(str(id_), "Unknown")
                confidence_percent = round(100 - confidence, 2)
                send_to_discord(name, confidence_percent, frame)
                return {"result": name, "confidence": confidence_percent}
            else:
                send_to_discord("Unknown", 100 - confidence, frame, is_unknown=True)
                return {"result": "Unknown", "confidence": round(100 - confidence, 2)}
        else:
            return {"result": "No Face Detected"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
