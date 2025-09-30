import os
import csv

# Define the CSV file paths
input_csv_path = 'Falci_Vivax_Index.csv'
output_csv_path = 'vivax_labeled_data.csv'
vivax_crops_path = os.path.join('Vivax', 'crops')
stage_mapping = {
    'G': 'Gametocyte',
    'R': 'Ring',
    'S': 'Schizont',
    'T': 'Trophozoite'
}

csv_data = []

with open(input_csv_path, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        if row['label_species'] == 'Vivax':
            img_filename = os.path.basename(row['Absolute Img Path']).rsplit('.', 1)[0] 
            index = row['index']
            expected_crop_filename = f"{img_filename}_{index}.png"
            match_found = False
            for stage_code, stage_name in stage_mapping.items():
                crop_folder_path = os.path.join(vivax_crops_path, stage_code)
                crop_file_path = os.path.join(crop_folder_path, expected_crop_filename)
                
                if os.path.exists(crop_file_path):
                    csv_data.append([
                        row['Absolute Img Path'], row['xmin'], row['ymin'], row['xmax'], row['ymax'],
                        row['label_species'], stage_name
                    ])
                    match_found = True
                    break 
            
            # If no match was found, the data is skipped (nothing is added to csv_data)

# Write the filtered and labeled data to the new CSV
with open(output_csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Absolute Img Path', 'xmin', 'ymin', 'xmax', 'ymax', 'label_species', 'label_stage'])
    # Write data rows
    writer.writerows(csv_data)

print(f"Labeled CSV file created at {output_csv_path}")