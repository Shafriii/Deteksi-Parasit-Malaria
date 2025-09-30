import os
import shutil
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from concurrent.futures import ThreadPoolExecutor

def create_lbp_replace_blue(image_path, P=8, R=1):
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None

    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(img_gray, P, R, method='default')

    lbp_normalized = np.uint8(255 * (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-6))

    blue_channel, green_channel, red_channel = cv2.split(img_color)

    modified_img = cv2.merge([blue_channel, green_channel, lbp_normalized])
    return modified_img

def process_image(file_name, source_images, target_images):
    """
    Fungsi untuk memproses satu gambar: menghitung LBP dan mengganti channel biru,
    kemudian menyimpan gambar yang telah dimodifikasi ke lokasi target.
    """
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        src_img_path = os.path.join(source_images, file_name)
        dst_img_path = os.path.join(target_images, file_name)

        modified_img = create_lbp_replace_blue(src_img_path)
        if modified_img is not None:
            cv2.imwrite(dst_img_path, modified_img)

def duplicate_and_convert_lbp_parallel(source_folder, target_folder):
    """
    Menduplikasi struktur folder dari dataset YOLO,
    lalu mengonversi setiap gambar di folder images dengan mengganti channel biru 
    menjadi citra LBP secara paralel.
    Label disalin langsung tanpa perubahan.
    """
    os.makedirs(target_folder, exist_ok=True)
    subfolders = ['train', 'val', 'test']

    # Membuat ThreadPoolExecutor untuk pemrosesan paralel
    with ThreadPoolExecutor() as executor:
        for sub in subfolders:
            source_images = os.path.join(source_folder, sub, 'images')
            source_labels = os.path.join(source_folder, sub, 'labels')
            target_images = os.path.join(target_folder, sub, 'images')
            target_labels = os.path.join(target_folder, sub, 'labels')

            os.makedirs(target_images, exist_ok=True)
            os.makedirs(target_labels, exist_ok=True)

            # PROSES GAMBAR: Kirim tugas pemrosesan gambar ke ThreadPoolExecutor
            if os.path.exists(source_images):
                for file_name in os.listdir(source_images):
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
    target_folder = "YoloNew/LBP_GB"

    duplicate_and_convert_lbp_parallel(source_folder, target_folder)

    print("Proses selesai. Folder dengan citra LBP (channel biru diganti) berhasil dibuat!")