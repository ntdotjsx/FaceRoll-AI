import cv2
import os
import numpy as np
import json

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# LBPH recognizer ที่ละเอียดขึ้น
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=4, grid_x=10, grid_y=10
)

# ฟังก์ชัน enhance ภาพใบหน้า
def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(roi_gray)
    filtered = cv2.medianBlur(clahe_enhanced, 3)
    resized = cv2.resize(filtered, (160, 120))
    return resized

# เพิ่ม data augmentation
def augment(image):
    flipped = cv2.flip(image, 1)
    brighter = cv2.convertScaleAbs(image, alpha=1.2, beta=15)
    darker = cv2.convertScaleAbs(image, alpha=0.8, beta=-15)
    return [image, flipped, brighter, darker]

# เตรียมข้อมูล
data_path = "data"
faces = []
labels = []
label = 0
id_to_name = {}

for dir_name in os.listdir(data_path):
    subject_path = os.path.join(data_path, dir_name)
    if not os.path.isdir(subject_path):
        continue

    for img_name in os.listdir(subject_path):
        img_path = os.path.join(subject_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Unable to read image: {img_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_in_img = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )

        # ใช้แค่ใบหน้าที่ใหญ่ที่สุด
        if len(faces_in_img) == 0:
            continue
        x, y, w, h = sorted(faces_in_img, key=lambda b: b[2]*b[3], reverse=True)[0]
        roi_gray = gray[y:y+h, x:x+w]
        processed = enhance_face(roi_gray)

        for aug in augment(processed):
            faces.append(aug)
            labels.append(label)

    id_to_name[label] = dir_name
    label += 1

# บันทึก mapping
with open("id_to_name.json", "w") as json_file:
    json.dump(id_to_name, json_file)

# เทรนโมเดล
print("Training model...")
recognizer.train(faces, np.array(labels))
recognizer.save("trainer.yml")
print("Model trained successfully!")

# แสดง label mapping
print("Label to Name mapping:")
for label, name in id_to_name.items():
    print(f"Label {label}: {name}")
