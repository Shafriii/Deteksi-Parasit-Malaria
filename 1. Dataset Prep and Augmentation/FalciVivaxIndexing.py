import os
import csv
import cv2

# Define the CSV file path
output_csv_path = 'Falci_Vivax_Indexnew.csv'

# Initialize list to hold rows for CSV
csv_data = []

# Walk both species folders
for species_folder in ['Falciparum', 'Vivax']:
    img_folder_path = os.path.join(species_folder, 'img')
    gt_folder_path = os.path.join(species_folder, 'gt')

    # Skip if folders don't exist
    if not os.path.isdir(img_folder_path) or not os.path.isdir(gt_folder_path):
        continue

    for img_filename in os.listdir(img_folder_path):
        if not img_filename.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif', '.tiff')):
            continue

        img_path = os.path.abspath(os.path.join(img_folder_path, img_filename))
        gt_path = os.path.join(gt_folder_path, img_filename)
        if not os.path.exists(gt_path):
            continue

        # Read GT as grayscale and binarize
        gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_image is None:
            continue

        _, binary_mask = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)

        # Find connected components (contours)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build bounding boxes [xmin, ymin, xmax, ymax]
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            xmin, ymin, xmax, ymax = x, y, x + w, y + h
            bounding_boxes.append((xmin, ymin, xmax, ymax))

        # Sort priority: left-to-right (x), then top-to-bottom (y)
        # This matches your example: (120,150)->0, (130,150)->1, (130,155)->2, (135,150)->3
        bounding_boxes.sort(key=lambda b: (b[0], b[1]))

        # Index starts at 0 for each image following the sorted order
        for idx, (xmin, ymin, xmax, ymax) in enumerate(bounding_boxes):
            csv_data.append([
                img_path, xmin, ymin, xmax, ymax, species_folder, idx
            ])

# Write CSV
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Absolute Img Path', 'xmin', 'ymin', 'xmax', 'ymax', 'label_species', 'index'])
    writer.writerows(csv_data)

print(f"CSV file created at {output_csv_path}")
