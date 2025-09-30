from ultralytics import YOLO
YOLO("best.pt").export(format="onnx", opset=12, dynamic=True, simplify=True)  # creates best.onnx