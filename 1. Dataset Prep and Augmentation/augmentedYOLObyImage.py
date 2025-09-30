import os
import pandas as pd
import shutil
import random
import uuid
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import yaml
import math

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
CSV_FILE     = 'Dataset/Final_Dataset.csv'
OUTPUT_DIR   = 'YoloB/Ori'
RANDOM_STATE = 42

# Split ratio
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10

# Batas minimal image per kelas sebelum split
MIN_IMAGES_BEFORE_SPLIT = 10

# Albumentations Pipeline
augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

# Folder output
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR   = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR  = os.path.join(OUTPUT_DIR, 'test')

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')
VAL_IMAGES_DIR   = os.path.join(VAL_DIR, 'images')
VAL_LABELS_DIR   = os.path.join(VAL_DIR, 'labels')
TEST_IMAGES_DIR  = os.path.join(TEST_DIR, 'images')
TEST_LABELS_DIR  = os.path.join(TEST_DIR, 'labels')

for d in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Mapping class -> ID (sesuaikan)
class_mapping = {
    0: "Malariae_Gametocyte",
    1: "Malariae_Trophozoite",
    2: "Malariae_Schizont",
    3: "Malariae_Ring",
    4: "Ovale_Gametocyte",
    5: "Ovale_Ring",
    6: "Ovale_Trophozoite",
    7: "Falciparum_Ring",
    8: "Falciparum_Schizont",
    9: "Falciparum_Trophozoite",
    10: "Falciparum_Gametocyte",
    11: "Vivax_Gametocyte",
    12: "Vivax_Ring",
    13: "Vivax_Schizont",
    14: "Vivax_Trophozoite"
}

# Simpan class mapping ke classes.yaml (YOLO style)
classes_yaml_path = os.path.join(OUTPUT_DIR, 'classes.yaml')
with open(classes_yaml_path, 'w') as f:
    yaml.dump({"names": class_mapping}, f, default_flow_style=False)


# ------------------------------------------------------------------
# 1) BACA CSV
# ------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
required_cols = ['Absolute Img Path', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di CSV.")

df['class'] = df['class'].astype(str)

# ------------------------------------------------------------------
# 2) DUPLIKASI KELAS YANG PUNYA UNIQUE IMAGE < 3
#    (Karena ingin split 70:20:10, minimal 3 image per kelas)
# ------------------------------------------------------------------
def duplicate_until_minimum(dataframe, min_images=3):
    """
    Jika suatu kelas punya unique image < min_images,
    gandakan (duplikasi file gambar) hingga mencapai min_images.
    """
    augmented_rows = []
    class_image_counts = dataframe.groupby('class')['Absolute Img Path'].nunique()
    
    for cls, count in class_image_counts.items():
        if count < min_images:
            needed = min_images - count
            # Subset baris dataframe untuk kelas tsb
            class_subset = dataframe[dataframe['class'] == cls]
            unique_paths = class_subset['Absolute Img Path'].unique().tolist()
            
            while needed > 0:
                chosen_path = random.choice(unique_paths)
                # Duplikasi file
                if not os.path.exists(chosen_path):
                    print(f"[Warning] File {chosen_path} tidak ada. Lewati duplikasi.")
                    continue
                
                dir_name = os.path.dirname(chosen_path)
                base_name, ext = os.path.splitext(os.path.basename(chosen_path))
                new_name = f"{base_name}_dup_{uuid.uuid4().hex}{ext}"
                new_path = os.path.join(dir_name, new_name)
                shutil.copy2(chosen_path, new_path)
                
                # Duplikasi baris bounding box utk file tsb
                sub_rows = class_subset[class_subset['Absolute Img Path'] == chosen_path]
                for _, row in sub_rows.iterrows():
                    new_row = row.copy()
                    new_row['Absolute Img Path'] = new_path
                    augmented_rows.append(new_row)
                
                needed -= 1
    
    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        dataframe = pd.concat([dataframe, aug_df], ignore_index=True)
    return dataframe

df = duplicate_until_minimum(df, MIN_IMAGES_BEFORE_SPLIT)

# ------------------------------------------------------------------
# 3) SPLIT DATASET (PER KELAS) DENGAN RASIO 70:20:10
# ------------------------------------------------------------------
from sklearn.model_selection import train_test_split

def split_dataset_per_class(dataframe, train_ratio, val_ratio, test_ratio, random_state=42):
    train_list, val_list, test_list = [], [], []
    
    for cls in dataframe['class'].unique():
        class_data = dataframe[dataframe['class'] == cls]
        # List unique image di kelas ini
        unique_imgs = class_data['Absolute Img Path'].unique()
        
        # Split train vs (val+test)
        train_imgs, temp_imgs = train_test_split(
            unique_imgs,
            test_size=(1 - train_ratio),
            random_state=random_state,
            shuffle=True
        )
        
        # Lalu split val vs test
        # contoh: val_ratio=0.2, test_ratio=0.1 => total=0.3
        # val_size_adjusted = 0.2 / 0.3 = ~0.666...
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs,
            test_size=(1 - val_size_adjusted),
            random_state=random_state,
            shuffle=True
        )
        
        # Kembalikan ke baris bounding box semula
        train_list.append(class_data[class_data['Absolute Img Path'].isin(train_imgs)])
        val_list.append(class_data[class_data['Absolute Img Path'].isin(val_imgs)])
        test_list.append(class_data[class_data['Absolute Img Path'].isin(test_imgs)])
    
    train_df = pd.concat(train_list, ignore_index=True)
    val_df   = pd.concat(val_list, ignore_index=True)
    test_df  = pd.concat(test_list, ignore_index=True)
    return train_df, val_df, test_df

train_df, val_df, test_df = split_dataset_per_class(
    df, 
    TRAIN_RATIO, 
    VAL_RATIO, 
    TEST_RATIO, 
    random_state=RANDOM_STATE
)

print("Jumlah unique image per subset (SEBELUM augmentasi):")
print(f"  Train: {train_df['Absolute Img Path'].nunique()}")
print(f"  Val  : {val_df['Absolute Img Path'].nunique()}")
print(f"  Test : {test_df['Absolute Img Path'].nunique()}")

# ------------------------------------------------------------------
# 4) FUNGSI CEK VALIDITAS BBOX
# ------------------------------------------------------------------
def all_bboxes_in_bounds(bboxes, img_w, img_h, allow_zero_size=False):
    """
    Mengecek apakah semua bbox valid dan berada di dalam gambar.
    bboxes: list of [xmin, ymin, xmax, ymax]
    img_w, img_h: lebar, tinggi gambar

    allow_zero_size = False -> jika True, bbox dengan w=0 atau h=0 tetap dianggap valid.
                               jika False, maka w=0 atau h=0 dianggap invalid.
    """
    for (xmin, ymin, xmax, ymax) in bboxes:
        # Cek posisi
        if xmin < 0 or ymin < 0 or xmax > img_w or ymax > img_h:
            return False
        # Cek lebar/tinggi
        if not allow_zero_size:
            if xmax <= xmin or ymax <= ymin:
                return False
    return True


# ------------------------------------------------------------------
# 5) FUNGSI AUGMENTASI (DENGAN LOGIC RE-AUGMENT JIKA OUT-OF-BOUND)
# ------------------------------------------------------------------
def albumentation_augment(
    dataframe,
    pipeline,
    augmentations_per_image=1,
    max_attempts=10
):
    """
    Menerima subset dataframe (apa pun campuran kelasnya) dan men-*augment*
    setiap unique image 'augmentations_per_image' kali.

    Jika hasil augmentasi ada bbox out-of-bounds, kita ulangi proses
    sampai max_attempts. Bila tetap gagal, baris itu di-skip.

    Mengembalikan DF hasil augmentasi (digabung dengan original).
    """
    augmented_rows = []
    
    for img_path in dataframe['Absolute Img Path'].unique():
        sub_df = dataframe[dataframe['Absolute Img Path'] == img_path]
        
        # Baca gambar
        try:
            with Image.open(img_path) as im:
                image = np.array(im.convert('RGB'))
                img_w, img_h = im.size
        except Exception as e:
            print(f"[Warning] Gagal membuka gambar {img_path}: {e}")
            continue
        
        # Kumpulkan bbox + class_labels
        original_bboxes = []
        class_labels = []
        for _, row in sub_df.iterrows():
            original_bboxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            class_labels.append(row['class'])
        
        # Augment
        for _ in range(augmentations_per_image):
            valid_result = False
            attempt = 0
            transformed_bboxes = None
            transformed_image  = None
            
            # Coba up to max_attempts
            while not valid_result and attempt < max_attempts:
                attempt += 1
                transformed = pipeline(
                    image=image,
                    bboxes=original_bboxes,
                    class_labels=class_labels
                )
                transformed_image  = transformed['image']
                transformed_bboxes = transformed['bboxes']
                # Cek validitas
                if all_bboxes_in_bounds(transformed_bboxes, img_w, img_h, allow_zero_size=False):
                    valid_result = True
            
            if valid_result and transformed_image is not None:
                # Buat file name baru
                base_name, ext = os.path.splitext(os.path.basename(img_path))
                new_name = f"{base_name}_aug_{uuid.uuid4().hex}{ext}"
                new_path = os.path.join(os.path.dirname(img_path), new_name)
                
                # Simpan image augmented
                Image.fromarray(transformed_image).save(new_path)
                
                # Tambahkan baris bounding box
                for (bx, cls) in zip(transformed_bboxes, transformed['class_labels']):
                    xmin, ymin, xmax, ymax = bx
                    new_row = {
                        'Absolute Img Path': new_path,
                        'xmin': xmin,
                        'ymin': ymin,
                        'xmax': xmax,
                        'ymax': ymax,
                        'class': cls
                    }
                    augmented_rows.append(new_row)
            else:
                # Jika gagal terus, skip gambar ini (atau Anda bisa simpan info untuk debugging)
                if not valid_result:
                    print(f"[Warning] Augmentasi {img_path} gagal menghasilkan bbox valid setelah {attempt} percobaan. Skip.")
    
    if len(augmented_rows) > 0:
        aug_df = pd.DataFrame(augmented_rows)
        return pd.concat([dataframe, aug_df], ignore_index=True)
    else:
        return dataframe

# ------------------------------------------------------------------
# 6) FUNGSI UNTUK MENYEIMBANGKAN (AUGMENT) PER SUBSET
#    - Cari highest_class_count (jumlah unique image terbanyak)
#    - SEMUA kelas disamakan menjadi 2 × highest_class_count
# ------------------------------------------------------------------
def balance_subset_by_augmentation(subset_df, pipeline):
    """
    - Hitung unique image per kelas di subset_df.
    - Dapatkan max_count (kelas tertinggi).
    - Target = 2 * max_count (SEMUA kelas).
    - Lakukan augmentasi pada tiap kelas agar mencapai target.
    - Kembalikan subset_df yang sudah diperbarui.
    """
    class_imgcount = subset_df.groupby('class')['Absolute Img Path'].nunique()
    if len(class_imgcount) == 0:
        return subset_df  # Kosong, langsung return
    
    max_count = class_imgcount.max()
    target    = 2 * max_count
    
    augmented_frames = []
    
    for cls, current_count in class_imgcount.items():
        class_data = subset_df[subset_df['class'] == cls]
        
        needed = target - current_count
        if needed > 0 and current_count > 0:
            # augmentations_per_image = ceil(needed / current_count)
            aug_per_img = math.ceil(needed / current_count)
            
            print(f"  Augmenting class '{cls}' => current_count={current_count}, "
                  f"target={target}, per_image={aug_per_img}")
            
            # Lakukan augmentasi (dengan cek out-of-bounds)
            class_data = albumentation_augment(
                class_data,
                pipeline,
                augmentations_per_image=aug_per_img,
                max_attempts=10  # misal 10
            )
        
        augmented_frames.append(class_data)
    
    balanced_df = pd.concat(augmented_frames, ignore_index=True)
    return balanced_df

# ------------------------------------------------------------------
# 7) AUGMENT MASING-MASING SUBSET
# ------------------------------------------------------------------
print("\n--- Augmenting Train Subset ---")
train_df = balance_subset_by_augmentation(train_df, augmentation_pipeline)

print("\n--- Augmenting Val Subset ---")
val_df   = balance_subset_by_augmentation(val_df, augmentation_pipeline)

print("\n--- Augmenting Test Subset ---")
test_df  = balance_subset_by_augmentation(test_df, augmentation_pipeline)

# ------------------------------------------------------------------
# 8) KONVERSI BBOX KE FORMAT YOLO DAN SIMPAN
# ------------------------------------------------------------------
def convert_to_yolo(size, box):
    """
    size: (W, H)
    box : (xmin, ymin, xmax, ymax)
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_min, y_min, x_max, y_max = box
    
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w_box    = x_max - x_min
    h_box    = y_max - y_min
    
    x_center *= dw
    y_center *= dh
    w_box    *= dw
    h_box    *= dh
    return (x_center, y_center, w_box, h_box)

def save_yolo_format(dataframe, images_dir, labels_dir):
    grouped = dataframe.groupby('Absolute Img Path')
    for img_path, group in grouped:
        if not os.path.exists(img_path):
            print(f"[Warning] Gambar {img_path} tidak ditemukan, lewati.")
            continue
        
        # Buka image untuk dapatkan size
        try:
            with Image.open(img_path) as im:
                w, h = im.size
        except Exception as e:
            print(f"[Warning] Gagal membuka {img_path}: {e}")
            continue
        
        # Persiapkan path output
        file_name = os.path.basename(img_path)
        base_stem, ext = os.path.splitext(file_name)
        
        out_img_path   = os.path.join(images_dir, file_name)
        out_label_path = os.path.join(labels_dir, base_stem + '.txt')
        
        # Copy image jika belum ada
        if not os.path.exists(out_img_path):
            shutil.copy2(img_path, out_img_path)
        
        # Tulis label
        with open(out_label_path, 'w') as f:
            for _, row in group.iterrows():
                cls_name = row['class']
                # Dapatkan index ID dari class_mapping
                cls_id = list(class_mapping.keys())[list(class_mapping.values()).index(cls_name)]
                
                bb = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])
                x_c, y_c, ww, hh = convert_to_yolo((w, h), bb)
                f.write(f"{cls_id} {x_c:.6f} {y_c:.6f} {ww:.6f} {hh:.6f}\n")

# Simpan
save_yolo_format(train_df, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
save_yolo_format(val_df,   VAL_IMAGES_DIR,   VAL_LABELS_DIR)
save_yolo_format(test_df,  TEST_IMAGES_DIR,  TEST_LABELS_DIR)

# ------------------------------------------------------------------
# 9) INFO AKHIR + DISTRIBUSI KELAS
# ------------------------------------------------------------------
def print_class_distribution(dataframe, subset_name):
    """
    Mencetak distribusi kelas berdasar bounding box dan unique image.
    """
    print(f"\n--- {subset_name} DISTRIBUTION ---")
    # 1) Bounding box count per class
    bbox_counts = dataframe['class'].value_counts()
    print("  Bounding Box count per class:")
    for cls, cnt in bbox_counts.items():
        print(f"    {cls}: {cnt}")
    
    # 2) Unique image count per class
    img_counts = dataframe.groupby('class')['Absolute Img Path'].nunique()
    print("  Unique Images per class:")
    for cls, cnt in img_counts.items():
        print(f"    {cls}: {cnt}")
    
    # 3) Total unique images in subset
    total_images = dataframe['Absolute Img Path'].nunique()
    print(f"  Total Unique Images in {subset_name}: {total_images}")

print_class_distribution(train_df, "Train")
print_class_distribution(val_df,   "Val")
print_class_distribution(test_df,  "Test")

print("\nData tersimpan di folder:", OUTPUT_DIR)
