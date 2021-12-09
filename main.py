from fastapi import FastAPI, Response
import cv2
import numpy as np
import base64
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

class Model(BaseModel):
    base64_img: str
# Load Yolo
#
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = ['person']
# # with open("coco.names", "r") as f:
# #     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))
#
app = FastAPI()

#
# def detectImg(img):
#     # Loading image
#     # img = cv2.resize(img, None, fx=0.4, fy=0.4)
#     height, width, channels = img.shape
#
#     # Detecting objects
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     # Showing informations on the screen
#     class_ids = []
#     confidences = []
#     boxes = []
#     res = []
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#
#
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = str(classes[class_ids[i]])
#             if label == "person":
#                 res.append([x, y, w, h])
#
#     return res

video_capture = cv2.VideoCapture(0)

def gen():

    while True:
        ret, image = video_capture.read()
        cv2.imwrite('t.jpg', image)
        yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')
    video_capture.release()

def convert_img(anh_base64):
    try:
        anh_base64 = np.fromstring(base64.b64decode(anh_base64), dtype=np.uint8)
        anh_base64 = cv2.imdecode(anh_base64, cv2.IMREAD_ANYCOLOR)
    except:
        return None
    return anh_base64



@app.get("/")
def index():
    return "Camera-Supervisor-System"

@app.post("/detect")
def humanDetection(M:Model):
    img = convert_img(M.base64_img)
    #res = detectImg(img)
    return "Human Detection"

