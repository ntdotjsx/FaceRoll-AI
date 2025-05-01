import cv2
import numpy as np
import json

# โหลดตัวตรวจจับใบหน้า + โมเดลที่เทรนไว้
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

# โหลด label map
with open("id_to_name.json", "r") as f:
    id_to_name = json.load(f)

# ฟังก์ชัน enhance ภาพ (ต้องเหมือนกับตอนเทรนเป๊ะ)
def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(roi_gray)
    filtered = cv2.medianBlur(clahe_enhanced, 3)
    resized = cv2.resize(filtered, (160, 120))
    return resized

# โหลดภาพ test (ปรับ path ตามที่ต้องการ)
test_img_path = "test.jpg"
img = cv2.imread(test_img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    processed = enhance_face(roi_gray)

    label, confidence = recognizer.predict(processed)
    name = id_to_name.get(str(label), "Unknown")

    text = f"{name} ({confidence:.2f})"
    color = (0, 255, 0) if confidence < 50 else (0, 0, 255)

    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, text, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# แสดงผล
cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()