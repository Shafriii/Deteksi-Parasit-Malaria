import os
import shutil
import cv2
import numpy as np
from skimage.feature import hog
from concurrent.futures import ProcessPoolExecutor

def create_hog_replace_blue(image_path, pixels_per_cell=(8, 8), cells_per_block=(2, 2), orientations=9):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return None, image_path

    hog_features, hog_image = hog(
        img_gray, orientations=orientations, pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block, visualize=True, feature_vector=False
    )

    hog_normalized = np.uint8(255 * (hog_image - hog_image.min()) / (hog_image.max() - hog_image.min() + 1e-6))

    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_color is None:
        return None, image_path

    blue_channel, green_channel, red_channel = cv2.split(img_color)

    modified_img = cv2.merge([blue_channel, hog_normalized, red_channel])
    return modified_img, image_path

def process_image(file_name, source_images, target_images):
    """
    Fungsi untuk memproses satu gambar: menghitung HOG dan mengganti channel biru,
    kemudian menyimpan gambar yang telah dimodifikasi ke lokasi target.
    """
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        src_img_path = os.path.join(source_images, file_name)
        dst_img_path = os.path.join(target_images, file_name)

        modified_img, img_path = create_hog_replace_blue(src_img_path)
        if modified_img is not None:
            cv2.imwrite(dst_img_path, modified_img)
            print(f"Processed: {img_path}")
        else:
            print(f"Failed: {img_path}")

def duplicate_and_convert_hog_parallel(source_folder, target_folder):
    """
    Menduplikasi struktur folder dari dataset YoloB,
    lalu mengonversi setiap gambar di folder images dengan mengganti channel biru 
    menjadi citra HOG secara paralel.
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
    target_folder = "YoloNew/R_HOG_B"

    duplicate_and_convert_hog_parallel(source_folder, target_folder)

    print("Proses selesai. Folder dengan citra HOG (channel biru diganti) berhasil dibuat!")
