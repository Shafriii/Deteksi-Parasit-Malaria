import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image

# Load dataset
csv_file = 'Final_Dataset.csv'  # Path to the uploaded CSV file
data = pd.read_csv(csv_file)

# Columns in the dataset
# ['Absolute Img Path', 'xmin', 'ymin', 'xmax', 'ymax', 'label_species', 'label_stage', 'class']

# Get image dimensions from the first image
first_image_path = data['Absolute Img Path'].iloc[0]
with Image.open(first_image_path) as img:
    image_width, image_height = img.size

# Map classes to integers
class_mapping = {cls: idx for idx, cls in enumerate(data['class'].unique())}
data['class_id'] = data['class'].map(class_mapping)

# Create YOLO format function
def convert_to_yolo(row, image_width, image_height):
    x_center = ((row['xmin'] + row['xmax']) / 2) / image_width
    y_center = ((row['ymin'] + row['ymax']) / 2) / image_height
    width = (row['xmax'] - row['xmin']) / image_width
    height = (row['ymax'] - row['ymin']) / image_height
    return f"{row['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

# Add YOLO annotations
data['yolo'] = data.apply(lambda row: convert_to_yolo(row, image_width, image_height), axis=1)

# Split dataset into train and val
train, val = train_test_split(data, test_size=0.2, random_state=42)

# Define paths
output_dir = 'FormatYolo'
train_img_dir = os.path.join(output_dir, 'train/images')
train_label_dir = os.path.join(output_dir, 'train/labels')
val_img_dir = os.path.join(output_dir, 'val/images')
val_label_dir = os.path.join(output_dir, 'val/labels')

# Create directories
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Copy images and save annotations
def process_split(split_data, img_dir, label_dir):
    for _, row in split_data.iterrows():
        # Copy image
        img_dest = os.path.join(img_dir, os.path.basename(row['Absolute Img Path']))
        shutil.copy(row['Absolute Img Path'], img_dest)

        # Save annotation
        label_path = os.path.join(label_dir, os.path.splitext(os.path.basename(row['Absolute Img Path']))[0] + '.txt')
        with open(label_path, 'w') as f:
            f.write(row['yolo'] + '\n')

# Process train and val splits
process_split(train, train_img_dir, train_label_dir)
process_split(val, val_img_dir, val_label_dir)

# Save class mapping for reference
class_mapping_path = os.path.join(output_dir, 'class_mapping.txt')
with open(class_mapping_path, 'w') as f:
    for cls, idx in class_mapping.items():
        f.write(f"{idx}: {cls}\n")

print(f"Dataset converted and saved to {output_dir}.")
print(f"Class mapping saved to {class_mapping_path}.")
