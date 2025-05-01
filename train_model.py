import cv2
import os
import numpy as np
import json

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=2, neighbors=8, grid_x=8, grid_y=8
)

data_path = "data"

faces = []
labels = []
label = 0
id_to_name = {}

for dir_name in os.listdir(data_path):
    subject_path = os.path.join(data_path, dir_name)
    for img_name in os.listdir(subject_path):
        img_path = os.path.join(subject_path, img_name)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_in_img = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        for x, y, w, h in faces_in_img:
            roi_gray = gray[y : y + h, x : x + w]
            faces.append(roi_gray)
            labels.append(label)

    id_to_name[label] = dir_name
    label += 1

with open("id_to_name.json", "w") as json_file:
    json.dump(id_to_name, json_file)

print("Training model...")
recognizer.train(faces, np.array(labels))
print("Model trained successfully!")
recognizer.save("trainer.yml")

print("Label to Name mapping:")
for label, name in id_to_name.items():
    print(f"Label {label}: {name}")
