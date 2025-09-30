import os
import shutil
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor

def create_orb_replace_blue(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, image_path

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(img_gray, None)

    blank_image = np.zeros_like(img_gray)

    keypoint_image = cv2.drawKeypoints(blank_image, keypoints, None, color=(255, 255, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    keypoint_grayscale = cv2.cvtColor(keypoint_image, cv2.COLOR_BGR2GRAY)

    keypoint_normalized = cv2.normalize(keypoint_grayscale, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    keypoint_normalized = np.uint8(keypoint_normalized)

    original_color_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_color_image is None:
        return None, image_path

    blue_channel, green_channel, red_channel = cv2.split(original_color_image)

    result_image = cv2.merge([keypoint_normalized, green_channel, red_channel])

    return result_image, image_path

def process_image(file_name, source_images, target_images):
    """
    Fungsi untuk memproses satu gambar: menghitung ORB dan mengganti channel biru,
    kemudian menyimpan gambar yang telah dimodifikasi ke lokasi target.
    """
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        src_img_path = os.path.join(source_images, file_name)
        dst_img_path = os.path.join(target_images, file_name)

        modified_img, img_path = create_orb_replace_blue(src_img_path)
        if modified_img is not None:
            try:
                success = cv2.imwrite(dst_img_path, modified_img)
                if success:
                    print(f"Processed: {img_path}")
                else:
                    print(f"Failed to save: {dst_img_path}")
            except Exception as e:
                print(f"Error saving image {dst_img_path}: {e}")
        else:
            print(f"Invalid image for {img_path}.")

def duplicate_and_convert_orb_parallel(source_folder, target_folder):
    """
    Menduplikasi struktur folder dari dataset YOLO,
    lalu mengonversi setiap gambar di folder images dengan mengganti channel biru 
    menjadi citra ORB secara paralel.
    Label disalin langsung tanpa perubahan.
    """
    os.makedirs(target_folder, exist_ok=True)
    subfolders = ['train', 'val', 'test']

    # Membuat ProcessPoolExecutor untuk pemrosesan paralel
    with ProcessPoolExecutor() as executor:
        for sub in subfolders:
            source_images = os.path.join(source_folder, sub, 'images')
            source_labels = os.path.join(source_folder, sub, 'labels')
            target_images = os.path.join(target_folder, sub, 'images')
            target_labels = os.path.join(target_folder, sub, 'labels')

            os.makedirs(target_images, exist_ok=True)
            os.makedirs(target_labels, exist_ok=True)

            # PROSES GAMBAR: Kirim tugas pemrosesan gambar ke ProcessPoolExecutor
            if os.path.exists(source_images):
                files = [f for f in os.listdir(source_images) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                for file_name in files:
                    executor.submit(process_image, file_name, source_images, target_images)
            else:
                print(f"Folder {source_images} tidak ditemukan, lewati.")

            # PROSES LABELS: Salin file label tanpa perubahan
            if os.path.exists(source_labels):
                for file_name in os.listdir(source_labels):
                    if file_name.lower().endswith('.txt'):
                        src_label_path = os.path.join(source_labels, file_name)
                        dst_label_path = os.path.join(target_labels, file_name)
                        shutil.copy2(src_label_path, dst_label_path)
            else:
                print(f"Folder {source_labels} tidak ditemukan, lewati.")

if __name__ == "__main__":
    source_folder = "YoloNew/Ori"
    target_folder = "YoloNew/RG_ORB"

    duplicate_and_convert_orb_parallel(source_folder, target_folder)

    print("Proses selesai. Folder dengan citra ORB (channel biru diganti) berhasil dibuat!")
