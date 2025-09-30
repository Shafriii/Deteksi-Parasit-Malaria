from ultralytics import YOLO
import pandas as pd
import os

def main():
    model = YOLO('yolov8n.pt')  
    train_results = model.train(
        data='YoloConfig/New/ORB_GB.yaml',      
        epochs=100,               
        imgsz=640,             
        batch=16,             
        name='malaria-yolov8_ORB_GB_New_seed0_100epochs',
        verbose=True,
        plots=True,
        pretrained=True,
        seed=0,
    )

    val_results = model.val()  
    print("=== Validation Metrics ===")
    print(val_results)

if __name__ == "__main__":
    main()
