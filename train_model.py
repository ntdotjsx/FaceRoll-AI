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
    radius=2, neighbors=16, grid_x=12, grid_y=12
)

# --- Sharpening Function --- 
def sharpen(image):
    kernel = np.array([[-1, -1, -1], [-1, 9,-1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)

# --- Enhance Face Function --- 
def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(roi_gray)
    filtered = cv2.medianBlur(clahe_enhanced, 3)
    resized = cv2.resize(filtered, (160, 160))
    return resized

# --- Image Augmentation --- 
def augment(image):
    augmented_images = [image]
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=15)
    augmented_images.append(brighter)
    darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-15)
    augmented_images.append(darker)
    
    # Random Rotation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = np.random.uniform(-15, 15)  # Random angle between -15 and 15
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    augmented_images.append(rotated)
    
    # Add Blur and Sharpen
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)
    sharpened = sharpen(image)
    augmented_images.append(sharpened)
    
    # New Augmentations
    # Random Zoom
    zx, zy = 0.9, 1.1
    for z in (zx, zy):
        augmented_images.append(cv2.resize(image, None, fx=z, fy=z)[
            int((h*z - h)/2):int((h*z - h)/2)+h,
            int((w*z - w)/2):int((w*z - w)/2)+w
        ])
    # Random Translation (shift)
    for dx, dy in ((10, 0), (0, 10), (-10, 0), (0, -10)):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        augmented_images.append(cv2.warpAffine(image, M, (w, h)))
    
    return augmented_images


# --- Face Detection with DNN --- 
def detect_face_dnn(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    face_net.setInput(blob)
    detections = face_net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # ลดค่า threshold ให้ต่ำลง
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    if boxes:
        return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[0]
    return None

# --- Function to check if the image is blurry --- 
def is_blurry(image, thresh=25.0):  # ปรับ threshold ให้น้อยลง
    var = cv2.Laplacian(image, cv2.CV_64F).var()
    return var < thresh


# --- Training Model Function --- 
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
                print(f"❌ NONE PIX: {img_path}")
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_box = detect_face_dnn(img)
            if face_box is None:
                continue

            x, y, w, h = face_box
            roi_gray = gray[y:y+h, x:x+w]

            if is_blurry(roi_gray, thresh=30.0):
                print(f"⚠️ THIS BLUR ISUS: {img_path}")
                continue

            processed = enhance_face(roi_gray)
            augmented_faces = augment(processed)
            for aug in augmented_faces:
                faces.append(aug)
                labels.append(label)

        id_to_name[label] = dir_name
        label += 1

    if not faces:
        print("🚫 NO DATA TO TRAIN!")
        return

    # Save label mapping
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
            print("\n🔁 Start train...")
            train_model()
            print(f"⏳ WAIT {args.interval} S...\n")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown train")
else:
    train_model()
