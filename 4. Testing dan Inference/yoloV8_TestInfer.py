import os
import cv2
import glob
from ultralytics import YOLO

# Konfigurasi Path
MODEL_PATH = 'C:\\PENTING BUAT KULIAH\\Thesis S2\\codingan\\ngoding baru lagi\\Model YoloV8\\A\\LBP_ORB_HOG\\weights\\best.pt'
TEST_IMAGES_PATH = 'C:\\PENTING BUAT KULIAH\\Thesis S2\\codingan\\ngoding baru lagi\\Yolo\\LBP_ORB_HOG\\test\\images'
GROUND_TRUTH_PATH = 'C:\\PENTING BUAT KULIAH\\Thesis S2\\codingan\\ngoding baru lagi\\Yolo\\LBP_ORB_HOG\\test\\labels'
OUTPUT_INFERENCE = 'C:\\PENTING BUAT KULIAH\\Thesis S2\\codingan\\ngoding baru lagi\\YoloV8 Results\\A\\LBP_ORB_HOG_best\\Inference'
OUTPUT_EVALUATION = 'C:\\PENTING BUAT KULIAH\\Thesis S2\\codingan\\ngoding baru lagi\\YoloV8 Results\\A\\LBP_ORB_HOG_best'

os.makedirs(OUTPUT_INFERENCE, exist_ok=True)

# Load YOLO model
model = YOLO(MODEL_PATH)

# Warna bounding box
COLOR_PRED = (0, 255, 0)
COLOR_GT = (0, 0, 255)

# Mapping kelas
CLASS_NAMES = {
    0: "gametocyte", 1: "trophozoite", 2: "schizont", 3: "ring",
}

if __name__ == '__main__':
    # Inference pada gambar test
    image_paths = glob.glob(os.path.join(TEST_IMAGES_PATH, "*.jpg"))

    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w, _ = img.shape
        filename = os.path.basename(img_path).split('.')[0]

        # Draw Ground Truth
        gt_file = os.path.join(GROUND_TRUTH_PATH, f"{filename}.txt")
        if os.path.exists(gt_file):
            with open(gt_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id, x, y, width, height = map(float, parts)
                        x1 = int((x - width / 2) * w)
                        y1 = int((y - height / 2) * h)
                        x2 = int((x + width / 2) * w)
                        y2 = int((y + height / 2) * h)
                        class_name = CLASS_NAMES.get(int(cls_id), "Unknown")
                        cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_GT, 2)
                        cv2.putText(img, f"GT {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_GT, 3)

        # Run inference
        results = model.predict(img_path)

        # Draw Predictions
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                class_name = CLASS_NAMES.get(cls_id, "Unknown")
                cv2.rectangle(img, (x1, y1), (x2, y2), COLOR_PRED, 2)
                cv2.putText(img, f"Pred {class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_PRED, 3)

        # Save hasil inference
        output_path = os.path.join(OUTPUT_INFERENCE, f"{filename}_output.jpg")
        cv2.imwrite(output_path, img)
        print(f"Saved inference: {output_path}")

    # Evaluasi model
    print("Memulai evaluasi model...")

    results = model.val(
        data='YoloConfig/A/LBP_ORB_HOG.yaml',
        split='test',
        project=OUTPUT_EVALUATION,
        name='Evaluasi',
        plots=True,
    ) 

    print("Evaluasi selesai.")
