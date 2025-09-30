import os
import shutil
import cv2
import numpy as np

def convert_rgb_to_hsl(image_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_hsl = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    
    h, l, s = cv2.split(img_hsl)
    
    h = (h * 255 / 180).astype(np.uint8)
    s = s.astype(np.uint8)
    l = l.astype(np.uint8)
    
    converted_img = cv2.merge((h, s, l))
    
    return converted_img 

def duplicate_and_convert_hsl(source_folder, target_folder):
    """
    Duplicates the structure of `source_folder` and converts all images in the `images` folder
    into the modified HSL color space. Labels are copied without modification.
    """
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Subfolders to process (train, val, test)
    subfolders = ['train', 'val', 'test']

    for sub in subfolders:
        # Paths for source and target images and labels
        source_images = os.path.join(source_folder, sub, 'images')
        source_labels = os.path.join(source_folder, sub, 'labels')
        target_images = os.path.join(target_folder, sub, 'images')
        target_labels = os.path.join(target_folder, sub, 'labels')

        # Create subfolders for images and labels in the target directory
        os.makedirs(target_images, exist_ok=True)
        os.makedirs(target_labels, exist_ok=True)

        # ============== PROCESS IMAGE FILES ==============
        if not os.path.exists(source_images):
            print(f"Folder {source_images} not found, skipping.")
        else:
            # Iterate through all image files in the `images` folder
            for file_name in os.listdir(source_images):
                # Check if the file is an image
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_img_path = os.path.join(source_images, file_name)
                    dst_img_path = os.path.join(target_images, file_name)

                    # Convert the RGB image to modified HSL color space
                    hsl_img = convert_rgb_to_hsl(src_img_path)
                    if hsl_img is not None:
                        # Save the transformed image
                        cv2.imwrite(dst_img_path, hsl_img)
                else:
                    continue

        # ============== PROCESS LABEL FILES ==============
        if not os.path.exists(source_labels):
            print(f"Folder {source_labels} not found, skipping.")
        else:
            # Copy all label files to the target `labels` folder
            for file_name in os.listdir(source_labels):
                if file_name.lower().endswith('.txt'):
                    src_label_path = os.path.join(source_labels, file_name)
                    dst_label_path = os.path.join(target_labels, file_name)
                    shutil.copy2(src_label_path, dst_label_path)
                else:
                    continue

if __name__ == "__main__":
    # Replace the paths below with your actual folder paths
    source_folder = "YoloNew/Ori"
    target_folder = "YoloNew/HSL"

    duplicate_and_convert_hsl(source_folder, target_folder)

    print("Process complete. Modified HSL color space dataset has been created!")
