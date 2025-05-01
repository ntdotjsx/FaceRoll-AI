# AI-FACEID

$ train

> python -m venv venv

> venv\Scripts\activate

> pip install opencv-contrib-python numpy

> python train_model.py

> pip install opencv-python opencv-contrib-python fastapi uvicorn requests numpy

pip uninstall opencv-contrib-python
pip install opencv-contrib-python

> pip install opencv-python opencv-contrib-python fastapi uvicorn requests numpy


uvicorn recognize_faces:app --host 0.0.0.0 --port 8000

# Train ครั้งเดียว
python auto_train.py

# Train วนเรื่อย ๆ ทุก 5 นาที
python auto_train.py --loop

# Train วนทุก 60 วินาที
python auto_train.py --loop --interval 60
