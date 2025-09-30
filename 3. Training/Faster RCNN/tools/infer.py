import torch
import argparse
import yaml
import random
import numpy as np
import os
from tqdm import tqdm
import cv2
import torchvision
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from Dataset.voc import VOCDataset  # Sesuaikan path dengan project Anda

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def collate_function(data):
    return tuple(zip(*data))

def get_iou(det, gt):
    det_x1, det_y1, det_x2, det_y2 = det
    gt_x1, gt_y1, gt_x2, gt_y2 = gt

    x_left = max(det_x1, gt_x1)
    y_top = max(det_y1, gt_y1)
    x_right = min(det_x2, gt_x2)
    y_bottom = min(det_y2, gt_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    area_intersection = (x_right - x_left) * (y_bottom - y_top)
    det_area = (det_x2 - det_x1) * (det_y2 - det_y1)
    gt_area = (gt_x2 - gt_x1) * (gt_y2 - gt_y1)
    area_union = float(det_area + gt_area - area_intersection + 1e-6)
    iou = area_intersection / area_union
    return iou

def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='interp'):
    """
    Menghitung mAP untuk satu nilai IoU threshold (misal 0.5 atau 0.75).
    method = 'interp' untuk 11-point, 'area' untuk area-based.
    Return:
      mean_ap (float)
      all_aps (dict)
      global_recall (float)
      global_precision (float)
    """
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)
    all_aps = {}
    aps = []

    TP_total = 0
    FP_total = 0
    GT_total = 0

    for label in gt_labels:
        # Kumpulkan det untuk class ini
        cls_dets_raw = []
        for im_idx, im_dets in enumerate(det_boxes):
            if label in im_dets:
                for det_item in im_dets[label]:
                    cls_dets_raw.append([im_idx, det_item])
        # Sort descending by score
        cls_dets = sorted(cls_dets_raw, key=lambda k: -k[1][-1])

        # Tandai semua gt belum matched
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        for det_i, (im_idx, det_pred) in enumerate(cls_dets):
            im_gts_label = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1
            for gt_idx, gt_box in enumerate(im_gts_label):
                iou_val = get_iou(det_pred[:-1], gt_box)
                if iou_val > max_iou_found:
                    max_iou_found = iou_val
                    max_iou_gt_idx = gt_idx

            # Threshold + cek apakah GT sudah matched
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_i] = 1
            else:
                tp[det_i] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True

        # Hitung AP untuk class ini
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp_cum / np.maximum(num_gts, eps)
        precisions = tp_cum / np.maximum((tp_cum + fp_cum), eps)

        if method == 'area':
            # area-based
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            i = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
        elif method == 'interp':
            # 11-point
            ap = 0.0
            for interp_pt in np.arange(0, 1.1, 0.1):
                prec_interp_pt = precisions[recalls >= interp_pt]
                prec_interp_pt = prec_interp_pt.max() if prec_interp_pt.size > 0 else 0
                ap += prec_interp_pt
            ap /= 11.0
        else:
            raise ValueError("method must be 'area' or 'interp'")

        if num_gts > 0:
            aps.append(ap)
            all_aps[label] = ap
        else:
            all_aps[label] = np.nan

        # Tambah ke global metric
        GT_total += num_gts
        if len(tp_cum) > 0:
            TP_total += tp_cum[-1]
        if len(fp_cum) > 0:
            FP_total += fp_cum[-1]

    mean_ap = sum(aps) / len(aps) if len(aps) else 0.0
    global_recall = TP_total / float(GT_total + 1e-6)
    global_precision = TP_total / float(TP_total + FP_total + 1e-6)

    return mean_ap, all_aps, global_recall, global_precision

def compute_map_multi_iou(det_boxes, gt_boxes, method='interp'):
    """
    Menghitung mAP pada multiple IoU thresholds [0.5, 0.55, 0.6, ..., 0.95].
    Return:
      map50, map50_95, global_precision_50, global_recall_50
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    all_maps = []
    map50 = 0
    precision_50 = 0
    recall_50 = 0

    for iou_t in iou_thresholds:
        mean_ap_iou, _, recall_iou, precision_iou = compute_map(
            det_boxes, gt_boxes, iou_threshold=iou_t, method=method
        )
        all_maps.append(mean_ap_iou)
        # Simpan juga precision & recall di IoU=0.5
        if abs(iou_t - 0.5) < 1e-6:
            map50 = mean_ap_iou
            precision_50 = precision_iou
            recall_50 = recall_iou

    map50_95 = float(np.mean(all_maps))
    return map50, map50_95, precision_50, recall_50

def load_model_and_weights(config, use_resnet50_fpn, ckpt_path):
    """
    Membuat model FasterRCNN sesuai config, lalu load checkpoint best.
    """
    # Buat model
    if use_resnet50_fpn:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1, 
            min_size=600,
            max_size=1000
        )
        # Ganti predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(
            model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=16  # Sesuaikan jumlah kelas Anda
        )
    else:
        backbone = torchvision.models.resnet34(
            pretrained=True,
            norm_layer=torchvision.ops.FrozenBatchNorm2d
        )
        backbone = torch.nn.Sequential(*list(backbone.children())[:-3])
        backbone.out_channels = 256
        roi_align = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2
        )
        rpn_anchor_generator = AnchorGenerator()
        model = torchvision.models.detection.FasterRCNN(
            backbone,
            num_classes=16,
            min_size=600,
            max_size=1000,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_align
        )

    model = model.to(device)
    model.eval()

    if ckpt_path is not None and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"  => best_mAP ketika disimpan: {ckpt.get('best_mAP', 0.0):.4f}")

    return model

def draw_bboxes_on_image(
    img_path, 
    gt_boxes_dict, 
    pred_boxes_dict, 
    label2idx, 
    idx2label, 
    conf_threshold=0.5,
    out_dir="output",
    img_name="result.jpg"
):
    """
    Membaca gambar dari img_path, lalu menambahkan bounding boxes:
     - Hijau: GT
     - Merah: Prediction (dengan score >= conf_threshold)
    Kemudian menyimpan hasil ke folder out_dir.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Baca gambar
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error membaca gambar: {img_path}")
        return

    # Gambar bounding box Ground Truth
    color_gt = (0, 255, 0)   # hijau
    for label_name, boxes in gt_boxes_dict.items():
        for box in boxes:
            x1, y1, x2, y2 = map(int, box) 
            # Tulis rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color_gt, 2)
            # Tulis label 
            cv2.putText(
                img,
                f"GT: {label_name}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_gt,
                1
            )

    # Gambar bounding box Prediction
    color_pred = (0, 0, 255)  # merah
    for label_name, boxes in pred_boxes_dict.items():
        for box in boxes:
            x1, y1, x2, y2, score = box
            if score < conf_threshold:
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            # Tulis rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color_pred, 2)
            # Tulis label + score
            cv2.putText(
                img,
                f"{label_name}: {score:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color_pred,
                1
            )

    # Simpan hasil
    save_path = os.path.join(out_dir, img_name)
    cv2.imwrite(save_path, img)
    print(f"Hasil disimpan di: {save_path}")

def main(args):
    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_config = config['dataset_params']
    test_im_dir = dataset_config['im_test_path']
    test_ann_dir = dataset_config['ann_test_path']

    # Buat dataset test
    voc_test = VOCDataset(
        'test',
        im_dir=test_im_dir,
        ann_dir=test_ann_dir
    )
    test_loader = DataLoader(
        voc_test,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_function
    )

    # Load model + checkpoint
    model = load_model_and_weights(
        config=config, 
        use_resnet50_fpn=args.use_resnet50_fpn,
        ckpt_path=args.ckpt
    )

    pred_list = []
    gt_list = []
    img_paths_list = []

    # -- 1) Loop inference untuk SELURUH dataset test (untuk menghitung metrics)
    model.eval()
    print("Inferencing entire test dataset to compute metrics...")
    with torch.no_grad():
        for (imgs, targets, img_paths) in tqdm(test_loader):
            imgs_gpu = [img.float().to(device) for img in imgs]
            outputs = model(imgs_gpu)

            for b_idx in range(len(imgs)):
                single_gt = {}
                single_pred = {}

                # Inisialisasi dict per label
                for label_name in voc_test.label2idx.keys():
                    single_gt[label_name] = []
                    single_pred[label_name] = []

                # Ground Truth
                for i_box, box in enumerate(targets[b_idx]['bboxes']):
                    x1, y1, x2, y2 = box.numpy()  # CPU
                    label_id = targets[b_idx]['labels'][i_box].item()
                    label_name = voc_test.idx2label[label_id]
                    single_gt[label_name].append([x1, y1, x2, y2])

                # Prediction
                pred_boxes = outputs[b_idx]['boxes']
                pred_labels = outputs[b_idx]['labels']
                pred_scores = outputs[b_idx]['scores']
                for i_p, p_box in enumerate(pred_boxes):
                    px1, py1, px2, py2 = p_box.cpu().numpy()
                    plabel_id = pred_labels[i_p].cpu().item()
                    pscore = pred_scores[i_p].cpu().item()
                    plabel_name = voc_test.idx2label[plabel_id]
                    single_pred[plabel_name].append([px1, py1, px2, py2, pscore])

                pred_list.append(single_pred)
                gt_list.append(single_gt)
                img_paths_list.append(img_paths[b_idx])  # simpan path gambar

    # -- 2) Hitung metrics di seluruh dataset
    map50, map50_95, precision_50, recall_50 = compute_map_multi_iou(pred_list, gt_list, method='interp')
    print("\n==== EVALUATION (Seluruh Dataset Test) ====")
    print(f"Precision (IoU=0.5) : {precision_50:.4f}")
    print(f"Recall (IoU=0.5)    : {recall_50:.4f}")
    print(f"mAP@0.5             : {map50:.4f}")
    print(f"mAP@0.5:0.95        : {map50_95:.4f}")

    # -- 3) Pilih 10 gambar random untuk disimpan di folder 'output'
    num_samples = min(10, len(img_paths_list))
    chosen_indices = random.sample(range(len(img_paths_list)), num_samples)
    output_dir = "output_inference"  # Folder output

    print(f"\nMenyimpan {num_samples} gambar acak ke folder '{output_dir}' ...")
    for i, idx in enumerate(chosen_indices):
        img_path = img_paths_list[idx]
        gt_dict = gt_list[idx]
        pred_dict = pred_list[idx]

        # Nama file simpan
        out_name = f"infer_{i+1}.jpg"
        draw_bboxes_on_image(
            img_path=img_path,
            gt_boxes_dict=gt_dict,
            pred_boxes_dict=pred_dict,
            label2idx=voc_test.label2idx,
            idx2label=voc_test.idx2label,
            conf_threshold=0.5,  # bisa ubah sesuai keinginan
            out_dir=output_dir,
            img_name=out_name
        )

    print("\nSelesai!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='config/vocLAB.yaml', type=str)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    parser.add_argument('--ckpt', default='checkpoint_best.pth', type=str,
                        help='path checkpoint model terbaik')
    args = parser.parse_args()
    main(args)
