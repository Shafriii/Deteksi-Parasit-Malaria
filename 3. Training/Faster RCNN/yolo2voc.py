import os
import yaml
import shutil
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml', type=str, required=False, default='YoloConfig/New/R_HOG_B.yaml',
                        help='Path ke file data.yaml (yang berisi train, val, test, names)')
    parser.add_argument('--out', type=str, default='VOCNew/R_HOG_B',
                        help='Parent directory untuk output Pascal VOC (default: voc_output)')
    return parser.parse_args() 

def convert_yolo_bbox_to_voc(x_center, y_center, w, h, img_width, img_height):
    xmin = int((x_center - w/2) * img_width)
    ymin = int((y_center - h/2) * img_height)
    xmax = int((x_center + w/2) * img_width)
    ymax = int((y_center + h/2) * img_height)
    return xmin, ymin, xmax, ymax

def create_voc_xml(folder_name, file_name, img_width, img_height, depth, bboxes, save_file_path):
    annotation = ET.Element('annotation')
    folder_el = ET.SubElement(annotation, 'folder')
    folder_el.text = folder_name
    filename_el = ET.SubElement(annotation, 'filename')
    filename_el.text = file_name
    source_el = ET.SubElement(annotation, 'source')
    database_el = ET.SubElement(source_el, 'database')
    database_el.text = 'Unknown'
    size_el = ET.SubElement(annotation, 'size')
    width_el = ET.SubElement(size_el, 'width')
    width_el.text = str(img_width)
    height_el = ET.SubElement(size_el, 'height')
    height_el.text = str(img_height)
    depth_el = ET.SubElement(size_el, 'depth')
    depth_el.text = str(depth)
    segmented_el = ET.SubElement(annotation, 'segmented')
    segmented_el.text = '0'
    for bbox in bboxes:
        obj_el = ET.SubElement(annotation, 'object')
        name_el = ET.SubElement(obj_el, 'name')
        name_el.text = bbox['class_name']
        pose_el = ET.SubElement(obj_el, 'pose')
        pose_el.text = 'Unspecified'
        truncated_el = ET.SubElement(obj_el, 'truncated')
        truncated_el.text = '0'
        difficult_el = ET.SubElement(obj_el, 'difficult')
        difficult_el.text = '0'
        bndbox_el = ET.SubElement(obj_el, 'bndbox')
        xmin_el = ET.SubElement(bndbox_el, 'xmin')
        xmin_el.text = str(bbox['xmin'])
        ymin_el = ET.SubElement(bndbox_el, 'ymin')
        ymin_el.text = str(bbox['ymin'])
        xmax_el = ET.SubElement(bndbox_el, 'xmax')
        xmax_el.text = str(bbox['xmax'])
        ymax_el = ET.SubElement(bndbox_el, 'ymax')
        ymax_el.text = str(bbox['ymax'])
    
    rough_string = ET.tostring(annotation, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    with open(save_file_path, "w") as f:
        f.write(pretty_xml)

def process_yolo_images(yolo_img_dir, yolo_lbl_dir, voc_folder_name, out_parent,class_names):
    voc_img_dir = os.path.join(out_parent, voc_folder_name, "JPEGImages")
    voc_annot_dir = os.path.join(out_parent, voc_folder_name, "Annotations")
    os.makedirs(voc_img_dir, exist_ok=True)
    os.makedirs(voc_annot_dir, exist_ok=True)

    # Cek apakah folder YOLO ada (kalau user kadang tidak punya test, dsb.)
    if not os.path.isdir(yolo_img_dir):
        print(f"Folder images tidak ditemukan: {yolo_img_dir}, lewati.")
        return
    
    if not os.path.isdir(yolo_lbl_dir):
        print(f"Folder labels tidak ditemukan: {yolo_lbl_dir}, lewati.")
        return

    # List file gambar
    img_files = [f for f in os.listdir(yolo_img_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    img_files.sort()

    for img_file in img_files:
        base_name, ext = os.path.splitext(img_file)
        src_img_path = os.path.join(yolo_img_dir, img_file)
        label_file = os.path.join(yolo_lbl_dir, base_name + '.txt')

        with Image.open(src_img_path) as img:
            w, h = img.size
            depth = 3 if img.mode == 'RGB' else 1
        dst_img_path = os.path.join(voc_img_dir, img_file)
        shutil.copyfile(src_img_path, dst_img_path)

        # Jika file label ada, buat bounding box. Jika tidak, bboxes kosong.
        bboxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_w = float(parts[3])
                bbox_h = float(parts[4])

                xmin, ymin, xmax, ymax = convert_yolo_bbox_to_voc(
                    x_center, y_center, bbox_w, bbox_h, w, h
                )

                # Ambil nama kelas dari YAML (names)
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f'class_{class_id}'  # fallback

                bboxes.append({
                    'class_name': class_name,
                    'xmin': xmin, 'ymin': ymin,
                    'xmax': xmax, 'ymax': ymax
                })

        # Buat file XML
        xml_file_path = os.path.join(voc_annot_dir, base_name + '.xml')
        create_voc_xml(
            folder_name=voc_folder_name,
            file_name=img_file,
            img_width=w,
            img_height=h,
            depth=depth,
            bboxes=bboxes,
            save_file_path=xml_file_path
        )

def main():
    args = parse_args()

    # Baca file YAML
    with open(args.yaml, 'r') as f:
        data = yaml.safe_load(f)

    # Ambil path folder train, val, test
    train_img_path = data.get('train',  None)
    val_img_path   = data.get('val',  None)
    test_img_path  = data.get('test',  None)

    # Ambil daftar nama kelas
    # data['names'] adalah dict {id: nama}, misal {0: "Malariae_Gametocyte", 1: "Malariae_Trophozoite", ...}
    # Kita ingin convert ke list agar index=class_id
    # tapi boleh juga langsung pakai dict.
    names_dict = data.get('names', {})
    # Buat list, pastikan terurut (0, 1, 2, ...)
    # Apabila data['names'] = {0: "A", 1: "B", ...} => kita bikin list: ["A", "B", ...]
    class_names = [names_dict[k] for k in sorted(names_dict.keys())]

    # Jika path YOLO adalah: "xxx/train/images", maka label dir = "xxx/train/labels"
    # Kita asumsikan cukup mengganti "images" -> "labels" di path.
    def get_label_path(img_path):
        # Di Windows, hati-hati dengan backslash vs slash.
        # Kita gunakan os.path.normpath untuk menormalkan.
        if img_path is None:
            return None
        lbl_path = img_path.replace('images', 'labels')
        return lbl_path

    train_lbl_path = get_label_path(train_img_path)
    val_lbl_path   = get_label_path(val_img_path)
    test_lbl_path  = get_label_path(test_img_path)

    # Buat folder output parent
    out_parent = args.out
    os.makedirs(out_parent, exist_ok=True)

    # Proses train
    if train_img_path:
        process_yolo_images(
            yolo_img_dir=train_img_path,
            yolo_lbl_dir=train_lbl_path,
            voc_folder_name="train", 
            out_parent=out_parent,
            class_names=class_names
        )

    # Proses val
    if val_img_path:
        process_yolo_images(
            yolo_img_dir=val_img_path,
            yolo_lbl_dir=val_lbl_path,
            voc_folder_name="val", 
            out_parent=out_parent,
            class_names=class_names
        )

    # Proses test
    if test_img_path:
        process_yolo_images(
            yolo_img_dir=test_img_path,
            yolo_lbl_dir=test_lbl_path,
            voc_folder_name="test", 
            out_parent=out_parent,
            class_names=class_names
        )

    print(f"Konversi selesai! Cek folder output: {out_parent}")

if __name__ == '__main__':
    main()
