from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from deepface import DeepFace
import cv2
import numpy as np
import tempfile
import os

app = FastAPI()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    # แปลง bytes เป็น OpenCV image
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # บันทึกภาพชั่วคราว
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        filepath = tmp_file.name
        cv2.imwrite(filepath, img)

    try:
        # เปรียบเทียบกับใบหน้าที่รู้จัก
        result = DeepFace.find(img_path=filepath, db_path="known_faces", enforce_detection=False)

        if result and len(result[0]) > 0:
            identity = result[0].iloc[0]["identity"]
            name = os.path.splitext(os.path.basename(identity))[0]
            return JSONResponse(content={"message": f"Hello, {name}!"})
        else:
            return JSONResponse(content={"message": "Face not recognized"})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

    finally:
        os.remove(filepath)
