import cv2
import os
import numpy as np
import json
import argparse
import time
import math

# --- Argument ---
parser = argparse.ArgumentParser()
parser.add_argument("--loop", action="store_true", help="Train แบบวนเรื่อย ๆ")
parser.add_argument("--interval", type=int, default=300, help="เวลาระหว่างการ train แต่ละรอบ (วินาที)")
args = parser.parse_args()

# --- Load DNN Face Detector ---
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# --- Load Landmark Detector (LBF) ---
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("lbfmodel.yaml")

# --- LBPH recognizer ---
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=10, grid_y=10
)

# --- Utils ---
def is_blurry(image, thresh=100.0):
    return cv2.Laplacian(image, cv2.CV_64F).var() < thresh

def is_too_dark(image, thresh=40):
    return np.mean(image) < thresh

def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(roi_gray)
    filtered = cv2.medianBlur(clahe_enhanced, 3)
    resized = cv2.resize(filtered, (160, 160))
    return resized

def augment(image):
    flipped = cv2.flip(image, 1)
    brighter = cv2.convertScaleAbs(image, alpha=1.1, beta=10)
    darker = cv2.convertScaleAbs(image, alpha=0.9, beta=-10)
    sharpened = sharpen(image)
    return [image, flipped, brighter, darker, sharpened]

def detect_and_align(image, conf_thresh=0.8):
    """
    Detect face with DNN, then align via LBF landmarks.
    Returns aligned gray face of size (h,w) before resize.
    """
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0,177.0,123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    dets = face_net.forward()[0,0,:,:]

    # เก็บกล่องใบหน้าทั้งหมด
    boxes = []
    for d in dets:
        if d[2] < conf_thresh:
            continue
        x1,y1,x2,y2 = (d[3:7]*[w,h,w,h]).astype(int)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        boxes.append((x1,y1,x2-x1,y2-y1))
    if len(boxes)!=1:
        return None

    bx,by,bw,bh = boxes[0]
    face = image[by:by+bh, bx:bx+bw]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Landmark alignment
    rects = np.array([[[0,0,bw,bh]]], dtype=np.int32)
    ok, landmarks = landmark_detector.fit(gray, rects)
    if ok and len(landmarks)>0 and len(landmarks[0])>=2:
        # eye centers
        left = np.mean(landmarks[0][36:42], axis=0)
        right = np.mean(landmarks[0][42:48], axis=0)
        dx,dy = right-left
        angle = math.degrees(math.atan2(dy,dx))
        M = cv2.getRotationMatrix2D((bw/2,bh/2), angle, 1.0)
        aligned = cv2.warpAffine(face, M, (bw,bh))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    return gray

# --- Training Logic ---
def train_model():
    data_path = "data"
    faces, labels = [], []
    label = 0
    id_to_name = {}

    for dir_name in sorted(os.listdir(data_path)):
        subject = os.path.join(data_path, dir_name)
        if not os.path.isdir(subject): continue

        print(f"📁 Processing: {dir_name}")
        added = 0
        for img_name in os.listdir(subject):
            path = os.path.join(subject, img_name)
            img = cv2.imread(path)
            if img is None:
                print(f"❌ Cannot read {path}")
                continue

            aligned_gray = detect_and_align(img)
            if aligned_gray is None:
                continue

            if is_blurry(aligned_gray,30) or is_too_dark(aligned_gray,40):
                print(f"⚠️ Bad quality: {path}")
                continue

            proc = enhance_face(aligned_gray)
            for aug in augment(proc):
                faces.append(aug)
                labels.append(label)
            added += 1

        if added>0:
            id_to_name[label] = dir_name
            label += 1
        else:
            print(f"🚫 No valid images in: {dir_name}")

    if not faces:
        print("🚫 No training data found.")
        return

    # save mapping
    with open("id_to_name.json","w") as j:
        json.dump(id_to_name, j)

    print("⚙️ Training model...")
    recognizer.train(faces, np.array(labels))
    recognizer.save("trainer.yml")
    print("✅ Model trained successfully!\n🧾 Label mapping:")
    for k,n in id_to_name.items():
        print(f"  Label {k}: {n}")

# --- Main ---
if args.loop:
    try:
        while True:
            print("\n🔁 Start training...")
            train_model()
            print(f"⏳ Waiting {args.interval} seconds...\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 Loop stopped.")
else:
    train_model()
