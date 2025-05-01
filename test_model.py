import cv2
import numpy as np
import json
import os
import math

# ─── 1. โหลด DNN Face Detector ───────────────────────
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",
    "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# ─── 2. โหลด Facial Landmark (LBF) เพื่อ align ───────
landmark_detector = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("lbfmodel.yaml")  # โหลดไฟล์จาก OpenCV repo

# ─── 3. โหลด LBPH Recognizer ─────────────────────────
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1, neighbors=8, grid_x=8, grid_y=8
)
recognizer.read("trainer.yml")

with open("id_to_name.json") as f:
    id_to_name = json.load(f)

# ─── 4. Enhance + augment ─────────────────────────────
def enhance_face(roi_gray):
    clahe = cv2.createCLAHE(2.0, (8,8))
    e = clahe.apply(roi_gray)
    e = cv2.medianBlur(e, 3)
    return cv2.resize(e, (160,160))  # ใช้ square crop

def augment(image):
    out = [image]
    # หมุน ±10°
    for ang in (-10, 10):
        M = cv2.getRotationMatrix2D((80,80), ang, 1)
        out.append(cv2.warpAffine(image, M, (160,160)))
    # พลิกซ้ายขวา
    out.append(cv2.flip(image, 1))
    return out

# ─── 5. Detect + align ─────────────────────────────────
def detect_and_align(img, conf_thresh=0.6):
    h,w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300,300),
                                 (104,177,123), swapRB=False)
    face_net.setInput(blob)
    det = face_net.forward()[0,0,:,:]

    boxes = []
    for i in range(det.shape[0]):
        score = float(det[i,2])
        if score < conf_thresh: continue
        x1,y1,x2,y2 = (det[i,3:7]*[w,h,w,h]).astype(int)
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w,x2), min(h,y2)
        # margin 20%
        side = max(x2-x1, y2-y1)
        cx, cy = x1 + (x2-x1)//2, y1 + (y2-y1)//2
        x1m = max(0, cx - side//2)
        y1m = max(0, cy - side//2)
        x2m = min(w, cx + side//2)
        y2m = min(h, cy + side//2)
        boxes.append((x1m, y1m, x2m-x1m, y2m-y1m))

    if not boxes: return None
    # เลือก box ใหญ่สุด
    bx,by,bw,bh = max(boxes, key=lambda b: b[2]*b[3])
    face = img[by:by+bh, bx:bx+bw]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # align ด้วย landmark
    ok, landmarks = landmark_detector.fit(face, np.array([ [0,0,bw,bh] ]))
    if ok and len(landmarks[0])>=2:
        # ตำแหน่งดวงตา (ประมาณ index 36-45 คือ eye landmarks)
        left_eye = np.mean(landmarks[0][36:42], axis=0)
        right_eye = np.mean(landmarks[0][42:48], axis=0)
        dx,dy = right_eye - left_eye
        angle = math.degrees(math.atan2(dy, dx))
        M = cv2.getRotationMatrix2D((bw/2,bh/2), angle, 1)
        face = cv2.warpAffine(face, M, (bw,bh))
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    return gray

# ─── 6. Loop ทดสอบ ────────────────────────────────────
for fn in os.listdir("test_images"):
    if not fn.lower().endswith((".jpg",".png",".webp")): continue
    img = cv2.imread(os.path.join("test_images", fn))
    if img is None: continue

    gface = detect_and_align(img, conf_thresh=0.5)
    if gface is None:
        print("❌ ไม่พบหน้า:", fn)
        continue

    proc = enhance_face(gface)
    # train augment ยังใช้ได้: (option)
    # for p in augment(proc): …

    label, conf = recognizer.predict(proc)
    name = id_to_name.get(str(label), "Unknown")
    # if conf>65: name="Unknown"

    # วาดกรอบ + ชื่อ
    # (เราคัดแค่ใบหน้าหลัก เลยไม่แสดงซ้ำ)
    print(f"✅ {fn} → {name} ({conf:.1f})")

    # แสดงผล
    x,y,w,h = 0,0,proc.shape[1],proc.shape[0]
    cv2.imshow("AlignedFace", cv2.resize(proc,(160,160)))
    cv2.waitKey(0)

cv2.destroyAllWindows()
