import cv2
import os
import numpy as np
import json
import argparse
import time

# --- Argument ---
parser = argparse.ArgumentParser()
parser.add_argument("--loop", action="store_true", help="Train แบบวนเรื่อย ๆ")
parser.add_argument("--interval", type=int, default=300, help="เวลาระหว่างการ train แต่ละรอบ (วินาที)")
args = parser.parse_args()

# --- Load DNN Face Detector ---
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# --- LBPH recognizer ---
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=10, grid_y=10
)

def is_blurry(image, thresh=100.0):
    var = cv2.Laplacian(image, cv2.CV_64F).var()
    return var < thresh

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(roi_gray)
    filtered = cv2.medianBlur(clahe_enhanced, 3)
    resized = cv2.resize(filtered, (160, 120))
    return resized

def augment(image):
    flipped = cv2.flip(image, 1)
    brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=15)
    darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-15)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    sharpened = sharpen(image)
    return [image, flipped, brighter, darker, rotated, blurred, sharpened]

def detect_face_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    if boxes:
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
    return None

def train_model():
    data_path = "data"
    faces = []
    labels = []
    label = 0
    id_to_name = {}

    for dir_name in sorted(os.listdir(data_path)):
        subject_path = os.path.join(data_path, dir_name)
        if not os.path.isdir(subject_path):
            continue

        for img_name in os.listdir(subject_path):
            img_path = os.path.join(subject_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ ไม่อ่านภาพได้: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_box = detect_face_dnn(img)
            if face_box is None:
                continue

            x, y, w, h = face_box
            roi_gray = gray[y:y+h, x:x+w]

            if is_blurry(roi_gray, thresh=30.0):
                print(f"⚠️ รูปเบลอ, ข้าม: {img_path}")
                continue

            processed = enhance_face(roi_gray)
            for aug in augment(processed):
                faces.append(aug)
                labels.append(label)

        id_to_name[label] = dir_name
        label += 1

    if not faces:
        print("🚫 ไม่มีข้อมูลใหม่สำหรับการเทรน")
        return

    with open("id_to_name.json", "w") as json_file:
        json.dump(id_to_name, json_file)

    print("⚙️ Training model...")
    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    print("✅ Model trained successfully!")

    print("\n🧾 Label mapping:")
    for label, name in id_to_name.items():
        print(f"  Label {label}: {name}")

# Run once or loop
if args.loop:
    try:
        while True:
            print("\n🔁 เริ่มรอบการ train ใหม่...")
            train_model()
            print(f"⏳ รอ {args.interval} วินาที...\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 หยุดการ train อัตโนมัติแล้ว")
else:
    train_model()
