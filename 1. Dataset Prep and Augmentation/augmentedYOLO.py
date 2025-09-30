import os
import pandas as pd
import shutil
import random
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
import yaml
import uuid
import math

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
CSV_FILE   = 'Dataset/Final_Dataset.csv'
OUTPUT_DIR = 'augment_dataset_yolo'
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
MIN_IMAGES_PER_CLASS = 10

augmentation_pipeline = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0, rotate_limit=15, p=0.5),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR   = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR  = os.path.join(OUTPUT_DIR, 'test')

TRAIN_IMAGES_DIR = os.path.join(TRAIN_DIR, 'images')
TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, 'labels')
VAL_IMAGES_DIR   = os.path.join(VAL_DIR, 'images')
VAL_LABELS_DIR   = os.path.join(VAL_DIR, 'labels')
TEST_IMAGES_DIR  = os.path.join(TEST_DIR, 'images')
TEST_LABELS_DIR  = os.path.join(TEST_DIR, 'labels')

for directory in [TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_IMAGES_DIR, VAL_LABELS_DIR, TEST_IMAGES_DIR, TEST_LABELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ------------------------------------------------------------------
# 1) BACA CSV
# ------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)
required_columns = ['Absolute Img Path', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' tidak ditemukan di file CSV!")

# ------------------------------------------------------------------
# DUPLIKASI DATA UNTUK KELAS YANG TIDAK CUKUP UNTUK SPLIT
# ------------------------------------------------------------------
def duplicate_small_classes(dataframe, min_samples):
    augmented_rows = []  
    for cls in dataframe['class'].unique():
        class_data = dataframe[dataframe['class'] == cls]
        unique_images = class_data['Absolute Img Path'].nunique()
        if unique_images < min_samples:
            needed = min_samples - unique_images
            while needed > 0:
                sample_row = class_data.sample(n=1, random_state=RANDOM_STATE).iloc[0].copy()
                original_img_path = sample_row['Absolute Img Path']
                dir_name = os.path.dirname(original_img_path)
                file_name, ext = os.path.splitext(os.path.basename(original_img_path))
                new_file_name = f"{file_name}_dup_{uuid.uuid4().hex}{ext}"
                new_img_path = os.path.join(dir_name, new_file_name)
                if os.path.exists(original_img_path):
                    shutil.copy2(original_img_path, new_img_path)
                else:
                    print(f"[Warning] Gambar {original_img_path} tidak ditemukan. Lewati duplikasi.")
                    break
                sample_row['Absolute Img Path'] = new_img_path
                augmented_rows.append(sample_row)
                needed -= 1
    if augmented_rows:
        dataframe = pd.concat([dataframe, pd.DataFrame(augmented_rows)], ignore_index=True)
    return dataframe

df = duplicate_small_classes(df, MIN_IMAGES_PER_CLASS)

# ------------------------------------------------------------------
# SPLIT TRAIN/VAL/TEST PER KELAS
# ------------------------------------------------------------------
def split_per_class(dataframe, train_ratio, val_ratio, test_ratio, random_state):
    train_dfs, val_dfs, test_dfs = [], [], []
    for cls in dataframe['class'].unique():
        class_data = dataframe[dataframe['class'] == cls]
        if class_data['Absolute Img Path'].nunique() < MIN_IMAGES_PER_CLASS:
            raise ValueError(f"Kelas {cls} masih kurang gambar untuk split setelah duplikasi.")
        train_data, temp_data = train_test_split(
            class_data, 
            test_size=(val_ratio + test_ratio), 
            random_state=random_state, 
            shuffle=True
        )
        val_size_adjusted = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data, 
            test_size=(1 - val_size_adjusted), 
            random_state=random_state, 
            shuffle=True
        )
        train_dfs.append(train_data)
        val_dfs.append(val_data)
        test_dfs.append(test_data)
    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs)

train_df, val_df, test_df = split_per_class(df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_STATE)

# ------------------------------------------------------------------
# Fungsi Augmentasi dengan Albumentations
# ------------------------------------------------------------------
def albumentation_augment(dataframe, augmentation_pipeline, augmentations_per_image=1):
    augmented_entries = []
    for img_path in dataframe['Absolute Img Path'].unique():
        img_rows = dataframe[dataframe['Absolute Img Path'] == img_path]
        try:
            image = np.array(Image.open(img_path).convert('RGB'))
        except Exception as e:
            print(f"[Warning] Tidak bisa membuka gambar {img_path}: {e}")
            continue

        bboxes = []
        class_labels = []
        for _, row in img_rows.iterrows():
            bboxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            class_labels.append(row['class'])

        for _ in range(augmentations_per_image):
            transformed = augmentation_pipeline(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = transformed['image']
            aug_bboxes = transformed['bboxes']
            aug_class_labels = transformed['class_labels']

            file_name, ext = os.path.splitext(os.path.basename(img_path))
            new_file_name = f"{file_name}_aug_{uuid.uuid4().hex}{ext}"
            new_img_path = os.path.join(os.path.dirname(img_path), new_file_name)
            Image.fromarray(aug_image).save(new_img_path)

            for bbox, cls in zip(aug_bboxes, aug_class_labels):
                xmin, ymin, xmax, ymax = bbox
                new_entry = {
                    'Absolute Img Path': new_img_path,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'class': cls
                }
                augmented_entries.append(new_entry)

    if augmented_entries:
        aug_df = pd.DataFrame(augmented_entries)
        dataframe = pd.concat([dataframe, aug_df], ignore_index=True)
    return dataframe

# ------------------------------------------------------------------
# AUGMENTASI UNTUK SETIAP KELAS KECUALI KELAS TERTINGGI
# ------------------------------------------------------------------
# Identifikasi kelas dengan jumlah bbox tertinggi
class_counts = train_df['class'].value_counts()
highest_class = class_counts.idxmax()

for cls, count in class_counts.items():
    if cls == highest_class:
        continue 
    target = count * 10
    current_count = count
    while current_count < target:
        class_df = train_df[train_df['class'] == cls]
        augmentations_per_image = max(1, math.ceil((target - current_count) / len(class_df)))
        print(f"Augmenting class {cls}: current_count = {current_count}, target = {target}, applying {augmentations_per_image} augmentation(s) per image.")
        augmented_class_df = albumentation_augment(class_df, augmentation_pipeline, augmentations_per_image)
        train_df = pd.concat([train_df, augmented_class_df], ignore_index=True)
        current_count = train_df[train_df['class'] == cls].shape[0]
        # Jika terlalu banyak iterasi, break untuk menghindari loop tak berujung (opsional)
        if augmentations_per_image <= 0:
            break

# ------------------------------------------------------------------
# KONVERSI BBOX -> YOLO, penyimpanan, dan verifikasi
# (Tetap seperti sebelumnya)
# ------------------------------------------------------------------
def convert_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_min, y_min, x_max, y_max = box
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return (x_center * dw, y_center * dh, w * dw, h * dh)

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

classes_yaml_path = os.path.join(OUTPUT_DIR, 'classes.yaml')
with open(classes_yaml_path, 'w') as yaml_file:
    yaml.dump({"names": class_mapping}, yaml_file, default_flow_style=False)
print("Mapping class -> ID telah disimpan.")

def save_to_folder(dataframe, img_dir, label_dir):
    grouped = dataframe.groupby('Absolute Img Path')
    for img_path, group in grouped:
        if not os.path.exists(img_path):
            print(f"[Warning] Gambar {img_path} tidak ditemukan. Lewati.")
            continue
        try:
            with Image.open(img_path) as im:
                w_img, h_img = im.size
        except Exception as e:
            print(f"[Warning] Gagal membuka gambar {img_path}: {e}")
            continue
        filename = os.path.basename(img_path)
        file_stem, ext = os.path.splitext(filename)
        dest_img_path  = os.path.join(img_dir, filename)
        label_txt_path = os.path.join(label_dir, file_stem + '.txt')
        if not os.path.exists(dest_img_path):
            shutil.copy2(img_path, dest_img_path)
        with open(label_txt_path, 'w') as f:
            for _, row in group.iterrows():
                xmin = row['xmin']
                ymin = row['ymin']
                xmax = row['xmax']
                ymax = row['ymax']
                cls_name = row['class']
                cls_id = list(class_mapping.keys())[list(class_mapping.values()).index(cls_name)]
                x_center, y_center, w_box, h_box = convert_to_yolo((w_img, h_img), (xmin, ymin, xmax, ymax))
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_box:.6f} {h_box:.6f}\n")

save_to_folder(train_df, TRAIN_IMAGES_DIR, TRAIN_LABELS_DIR)
save_to_folder(val_df, VAL_IMAGES_DIR, VAL_LABELS_DIR)
save_to_folder(test_df, TEST_IMAGES_DIR, TEST_LABELS_DIR)

def check_classes_in_subset(subset_name, dataframe):
    classes_in_subset = set(dataframe['class'].unique())
    missing_classes = set(class_mapping.values()) - classes_in_subset
    if missing_classes:
        print(f"[Warning] Subset {subset_name} tidak memiliki kelas berikut: {missing_classes}")
    else:
        print(f"Subset {subset_name} memiliki semua kelas.")

check_classes_in_subset("Train", train_df)
check_classes_in_subset("Val", val_df)
check_classes_in_subset("Test", test_df)

def print_class_distribution(dataframe, subset_name):
    distribution = dataframe['class'].value_counts()
    print(f"\nDistribusi kelas di subset {subset_name}:")
    for cls, count in distribution.items():
        print(f"  {cls}: {count}")

print_class_distribution(train_df, "Train")
print_class_distribution(val_df, "Val")
print_class_distribution(test_df, "Test")

print("\nProses selesai.")
print(f"Total Gambar Train: {len(train_df['Absolute Img Path'].unique())}")
print(f"Total Gambar Val  : {len(val_df['Absolute Img Path'].unique())}")
print(f"Total Gambar Test : {len(test_df['Absolute Img Path'].unique())}")
print("Dataset YOLO disimpan di folder:", OUTPUT_DIR)
