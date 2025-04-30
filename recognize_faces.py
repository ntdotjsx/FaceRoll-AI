import cv2
import json

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml") 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with open('id_to_name.json', 'r') as json_file:
    id_to_name = json.load(json_file)

image_path = 'kuy.jpg'
frame = cv2.imread(image_path)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]

    id_, confidence = recognizer.predict(roi_gray)
    
    if confidence >= 0 and confidence <= 85: 
        # แปลงค่าความเชื่อมั่นเป็นเปอร์เซ็นต์
        confidence_percent = round(100 - confidence, 2)  # ความแม่นยำสูงสุด 100% จะเป็น 0 ความเชื่อมั่น
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = id_to_name.get(str(id_), "Unknown")
        color = (255, 0, 0)  # สีน้ำเงิน
        stroke = 2
        cv2.putText(frame, f"{name} ({confidence_percent}%)", (x, y-10), font, 0.9, color, stroke, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        name = "Unknown"
        color = (0, 0, 255)  # สีแดง
        stroke = 2
        cv2.putText(frame, name, (x, y-10), font, 0.9, color, stroke, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

cv2.imshow("Face Recognition", frame)

cv2.waitKey(0)
cv2.destroyAllWindows()
