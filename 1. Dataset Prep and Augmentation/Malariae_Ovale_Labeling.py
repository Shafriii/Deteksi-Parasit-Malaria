import os
import csv
import cv2

stage_mapping = {
    'G': 'Gametocyte',
    'R': 'Ring',
    'S': 'Schizont',
    'T': 'Trophozoite'
}
output_csv_path = 'Malaria_Ovale_labeled_data.csv'
csv_data = []
for species_folder in ['Malariae', 'Ovale']:
    img_folder_path = os.path.join(species_folder, 'img')
    gt_folder_path = os.path.join(species_folder, 'gt')
    label_species = species_folder 
    
    # Loop through each file in the 'img' folder
    for img_filename in os.listdir(img_folder_path):
        # Ensure the file has a proper image extension
        if img_filename.endswith(('.jpg', '.png')):
            img_path = os.path.abspath(os.path.join(img_folder_path, img_filename))
            gt_path = os.path.join(gt_folder_path, img_filename)
            
            # Check if the corresponding GT file exists
            if os.path.exists(gt_path):
                # Determine the stage label based on the filename
                stage_code = img_filename.split('-')[-1][0]  # Extract the first character after the last '-'
                label_stage = stage_mapping.get(stage_code.upper(), 'Unknown')  # Map to the stage or set as Unknown

                # Load the GT image to get bounding box dimensions
                gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                _, binary_mask = cv2.threshold(gt_image, 127, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Loop over each contour to get bounding boxes
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    xmin, ymin, xmax, ymax = x, y, x + w, y + h

                    # Add row data to CSV
                    csv_data.append([img_path, xmin, ymin, xmax, ymax, label_species, label_stage])

# Write data to CSV
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Absolute Img Path', 'xmin', 'ymin', 'xmax', 'ymax', 'label_species', 'label_stage'])
    writer.writerows(csv_data)

print(f"CSV file created at {output_csv_path}")
