import cv2
import numpy as np
import math

# ─── 1. ฟังก์ชัน Detect และ Align ──────────────────────────────────
def detect_and_align(image, face_net, landmark_detector, conf_thresh=0.5):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False)
    face_net.setInput(blob)
    dets = face_net.forward()[0, 0, :, :]

    boxes = []
    for d in dets:
        if d[2] < conf_thresh:
            continue
        x1, y1, x2, y2 = (d[3:7] * [w, h, w, h]).astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        side = max(x2 - x1, y2 - y1)
        cx, cy = x1 + (x2 - x1)//2, y1 + (y2 - y1)//2
        x1m = max(0, cx - side//2)
        y1m = max(0, cy - side//2)
        x2m = min(w, cx + side//2)
        y2m = min(h, cy + side//2)
        boxes.append((x1m, y1m, x2m - x1m, y2m - y1m))

    if not boxes:
        return None

    bx, by, bw, bh = max(boxes, key=lambda b: b[2]*b[3])
    face = image[by:by+bh, bx:bx+bw]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # Landmark alignment
    rects = np.array([[[0, 0, bw, bh]]], dtype=np.int32)
    ok, landmarks = landmark_detector.fit(gray, rects)
    if ok and len(landmarks) > 0 and len(landmarks[0]) >= 2:
        # Compute eye centers
        left_eye = np.mean(landmarks[0][36:42], axis=0)
        right_eye = np.mean(landmarks[0][42:48], axis=0)
        dx, dy = right_eye - left_eye
        angle = math.degrees(math.atan2(dy, dx))
        M = cv2.getRotationMatrix2D((bw/2, bh/2), angle, 1)
        aligned = cv2.warpAffine(face, M, (bw, bh))
        gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)

    return gray

# ─── 2. ฟังก์ชัน Enhance Face ──────────────────────────────────────
def enhance_face(gray_face):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    e = clahe.apply(gray_face)
    return cv2.medianBlur(e, 3)
