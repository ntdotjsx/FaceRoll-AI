import cv2
import os
import numpy as np
import json

# Load face detector และสร้าง recognizer
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8
)

# ฟังก์ชันเพิ่มคุณภาพใบหน้า
def enhance_face(roi_gray):
    blurred = cv2.GaussianBlur(roi_gray, (3, 3), 0)
    sharpened = cv2.filter2D(blurred, -1, np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ]))
    equalized = cv2.equalizeHist(sharpened)
    resized = cv2.resize(equalized, (160, 120))
    return resized

# เตรียมข้อมูล
data_path = "data"
faces = []
labels = []
label = 0
id_to_name = {}

# วนลูปแต่ละคน
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
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        for x, y, w, h in faces_in_img:
            roi_gray = gray[y:y+h, x:x+w]
            processed = enhance_face(roi_gray)
            faces.append(processed)
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
