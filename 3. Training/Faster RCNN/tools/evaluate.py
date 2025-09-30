#!/usr/bin/env python3

import os
import argparse
import csv
import random
import yaml
import glob

import numpy as np
import torch
import torchvision
import cv2
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from tqdm import tqdm
import torchvision.ops as ops  # <--- pastikan import ops

# Import VOCDataset dari file Dataset/voc.py
from Dataset.voc import VOCDataset

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

def apply_nms_per_class(boxes, labels, scores, iou_thresh=0.3, score_thresh=0.05):
    final_boxes = []
    final_labels = []
    final_scores = []

    # 1) Filter out boxes by minimum score threshold
    keep_mask = scores > score_thresh
    boxes = boxes[keep_mask]
    labels = labels[keep_mask]
    scores = scores[keep_mask]

    # Jika benar-benar kosong, return kosong
    if len(boxes) == 0:
        return (
            torch.empty((0, 4), dtype=boxes.dtype),
            torch.empty((0,), dtype=scores.dtype),
            torch.empty((0,), dtype=labels.dtype)
        )

    # 2) Ambil unique labels
    unique_labels = torch.unique(labels)

    for cls in unique_labels:
        cls_indices = torch.where(labels == cls)[0]
        cls_boxes = boxes[cls_indices]
        cls_scores = scores[cls_indices]

        # Lakukan NMS
        keep_indices = ops.nms(cls_boxes, cls_scores, iou_thresh)

        final_boxes.append(cls_boxes[keep_indices])
        final_scores.append(cls_scores[keep_indices])
        final_labels.append(labels[cls_indices][keep_indices])

    # Setelah loop, cek apakah final_boxes benar-benar kosong
    if len(final_boxes) == 0:
        return (
            torch.empty((0, 4), dtype=boxes.dtype),
            torch.empty((0,), dtype=scores.dtype),
            torch.empty((0,), dtype=labels.dtype)
        )

    final_boxes = torch.cat(final_boxes, dim=0)
    final_scores = torch.cat(final_scores, dim=0)
    final_labels = torch.cat(final_labels, dim=0)

    return final_boxes, final_scores, final_labels


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='interp'):
    """
    Menghitung mAP dengan perhitungan TP/FP per-kelas, lalu di-average.
    Sekarang juga mengembalikan average IoU di antara semua pasangan TP.
    
    Args:
        det_boxes (list of dict): Hasil prediksi, 
            misal det_boxes[i]["cat"] = [ [x1,y1,x2,y2,score], ... ]
        gt_boxes (list of dict): Ground truth,
            misal gt_boxes[i]["cat"] = [ [x1,y1,x2,y2], ... ]
        iou_threshold (float): Batas IoU untuk mempertimbangkan suatu prediksi sebagai TP.
        method (str): "area" atau "interp" untuk perhitungan AP.

    Returns:
        mean_ap (float): Mean Average Precision keseluruhan kelas.
        all_aps (dict): AP per-kelas.
        macro_recall (float): Rata-rata recall (per-kelas).
        macro_precision (float): Rata-rata precision (per-kelas).
        avg_iou (float): Rata-rata IoU dari semua pasangan TP (across all classes).
    """
    gt_labels = {cls_key for im_gt in gt_boxes for cls_key in im_gt.keys()}
    gt_labels = sorted(gt_labels)

    all_aps = {}
    aps = []
    precision_per_class = []
    recall_per_class = []

    # Kita simpan semua iou yang berhasil match (TP) di sini:
    all_matched_ious = []

    for label in gt_labels:
        cls_dets_raw = []
        for im_idx, im_dets in enumerate(det_boxes):
            if label in im_dets:
                for det_item in im_dets[label]:
                    cls_dets_raw.append([im_idx, det_item])

        # Urutkan prediksi by skor descending
        cls_dets = sorted(cls_dets_raw, key=lambda x: -x[1][-1])

        # Buat penanda apakah GT sudah terpakai (matched) atau belum
        gt_matched = [[False for _ in im_gts[label]] for im_gts in gt_boxes]
        # Jumlah GT di seluruh image (untuk kelas ini)
        num_gts = sum([len(im_gts[label]) for im_gts in gt_boxes])

        tp = [0] * len(cls_dets)
        fp = [0] * len(cls_dets)

        for det_i, (im_idx, det_pred) in enumerate(cls_dets):
            if label not in gt_boxes[im_idx]:
                fp[det_i] = 1
                continue

            candidates = gt_boxes[im_idx][label]
            max_iou_found = -1
            max_iou_gt_idx = -1

            for gt_idx, gt_box in enumerate(candidates):
                iou_val = get_iou(det_pred[:-1], gt_box)
                if iou_val > max_iou_found:
                    max_iou_found = iou_val
                    max_iou_gt_idx = gt_idx

            # Apakah berhasil match?
            # Kriteria: (1) IoU >= iou_threshold, (2) GT tersebut belum pernah terpakai
            if max_iou_found < iou_threshold or gt_matched[im_idx][max_iou_gt_idx]:
                fp[det_i] = 1
            else:
                tp[det_i] = 1
                gt_matched[im_idx][max_iou_gt_idx] = True
                # Simpan iou TP
                all_matched_ious.append(max_iou_found)

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        eps = np.finfo(np.float32).eps
        recalls = tp_cum / np.maximum(num_gts, eps)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, eps)

        precision_per_class.append(precisions[-1] if len(precisions) > 0 else 0)
        recall_per_class.append(recalls[-1] if len(recalls) > 0 else 0)

        # Hitung AP dengan metode 'area' atau 'interp'
        if method == 'area':
            # Metode area dibawah
            recalls = np.concatenate(([0.0], recalls, [1.0]))
            precisions = np.concatenate(([0.0], precisions, [0.0]))
            for i in range(precisions.size - 1, 0, -1):
                precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])
            idx = np.where(recalls[1:] != recalls[:-1])[0]
            ap = np.sum((recalls[idx + 1] - recalls[idx]) * precisions[idx + 1])
        elif method == 'interp':
            # Metode 11-point interpolation
            ap = sum(
                np.max(precisions[recalls >= thresh]) if np.any(recalls >= thresh) else 0 
                for thresh in np.arange(0.0, 1.1, 0.1)
            ) / 11.0
        else:
            raise ValueError("method must be 'area' or 'interp'")

        aps.append(ap)
        all_aps[label] = ap

    mean_ap = np.mean(aps) if len(aps) > 0 else 0.0
    macro_precision = np.mean(precision_per_class) if len(precision_per_class) > 0 else 0.0
    macro_recall = np.mean(recall_per_class) if len(recall_per_class) > 0 else 0.0

    # Hitung rata-rata IoU dari semua TP
    if len(all_matched_ious) > 0:
        avg_iou = float(np.mean(all_matched_ious))
    else:
        avg_iou = 0.0

    return mean_ap, all_aps, macro_recall, macro_precision, avg_iou


def evaluate_map_50_95(det_boxes, gt_boxes, method='interp'):
    """
    Menghitung rata-rata mAP dari IoU threshold 0.5 s/d 0.95 dengan step 0.05.
    Di sini kita tidak mengekstrak average IoU,
    karena mAP@0.5:0.95 adalah rangkaian perhitungan di beberapa threshold.
    """
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    mAPs = []

    for thr in iou_thresholds:
        # Di sini kita abaikan average IoU dan yang lainnya, karena
        # cuma perlu mean_ap_thr untuk keperluan mAP@0.5:0.95
        mean_ap_thr, _, _, _, _ = compute_map(
            det_boxes, gt_boxes, iou_threshold=thr, method=method
        )
        mAPs.append(mean_ap_thr)

    if len(mAPs) > 0:
        return np.mean(mAPs)
    else:
        return 0.0

def draw_predictions(image_np, gt_boxes_dict, pred_boxes_dict, save_path):
    """
    Gambar GT (merah) dan prediksi (hijau) di atas image, lalu simpan ke disk.
    """
    img_draw = image_np.copy()

    # GT -> merah
    for label_str, box_list in gt_boxes_dict.items():
        for box in box_list:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img_draw, f"GT:{label_str}",
                        (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 2)

    # Pred -> hijau
    for label_str, box_list in pred_boxes_dict.items():
        for det in box_list:
            px1, py1, px2, py2, score = det
            px1, py1, px2, py2 = map(int, [px1, py1, px2, py2])
            cv2.rectangle(img_draw, (px1, py1), (px2, py2), (0, 255, 0), 2)
            cv2.putText(img_draw, f"{label_str}:{score:.2f}",
                        (px1, max(0, py1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 2)

    cv2.imwrite(save_path, img_draw)

def evaluate(args):
    # Baca config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print("=== CONFIG ===")
    print(config)

    dataset_config = config['dataset_params']
    test_im_dir = dataset_config['im_test_path']
    test_ann_dir = dataset_config['ann_test_path']

    # Buat dataset & dataloader test
    voc_test = VOCDataset(
        split='test',
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

    # Bangun model (mirip train.py)
    if args.use_resnet50_fpn:
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            min_size=600,
            max_size=1000
        )
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=16)
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

    model.to(device)
    model.eval()

    # Load checkpoint
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint {args.ckpt} tidak ditemukan!")
    print(f"Loading checkpoint from {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Siapkan folder output
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    test_preds = []
    test_gts = []

    print("==> Evaluating on Test Set...")
    with torch.no_grad():
        for images, targets, filenames in tqdm(test_loader):
            images_gpu = [img.float().to(device) for img in images]
            outputs = model(images_gpu)

            # Ambil boxes, labels, scores (tensor di GPU)
            boxes_tensor = outputs[0]['boxes'].detach().cpu()
            labels_tensor = outputs[0]['labels'].detach().cpu()
            scores_tensor = outputs[0]['scores'].detach().cpu()

            # Contoh parameter NMS
            nms_iou_thresh = 0.3
            score_thresh = 0.5

            # Terapkan NMS per-class
            final_boxes, final_scores, final_labels = apply_nms_per_class(
                boxes_tensor, labels_tensor, scores_tensor,
                iou_thresh=nms_iou_thresh,
                score_thresh=score_thresh
            )

            out_boxes = final_boxes.numpy()
            out_scores = final_scores.numpy()
            out_labels = final_labels.numpy()

            # Buat dict pred & gt
            single_pred = {}
            single_gt = {}
            for label_name in voc_test.label2idx.keys():
                single_pred[label_name] = []
                single_gt[label_name] = []

            # Ground truth
            for i_box, gt_box in enumerate(targets[0]['bboxes']):
                label_id = targets[0]['labels'][i_box].item()
                label_str = voc_test.idx2label[label_id]
                single_gt[label_str].append(gt_box.numpy().tolist())

            # Prediction
            for pbox, plabel_id, pscore in zip(out_boxes, out_labels, out_scores):
                plabel_str = voc_test.idx2label[plabel_id]
                single_pred[plabel_str].append([pbox[0], pbox[1], pbox[2], pbox[3], pscore])

            test_preds.append(single_pred)
            test_gts.append(single_gt)

            # Visualisasi
            img_np = images[0].permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
            if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            base_name = os.path.splitext(os.path.basename(filenames[0]))[0]
            out_file = os.path.join(args.output_dir, f"{base_name}_pred.jpg")
            draw_predictions(img_np, single_gt, single_pred, out_file)

    # Hitung metrik @0.5 (sekaligus average IoU @0.5)
    mean_ap_50, _, recall_50, precision_50, avg_iou_50 = compute_map(
        test_preds, test_gts, iou_threshold=0.5, method='interp'
    )
    # Hitung mAP@0.5:0.95
    mean_ap_50_95 = evaluate_map_50_95(test_preds, test_gts, method='interp')

    print("\n=== Evaluation Results (Test) ===")
    print(f"Precision @IoU=0.5:   {precision_50:.4f}")
    print(f"Recall    @IoU=0.5:   {recall_50:.4f}")
    print(f"mAP       @IoU=0.5:   {mean_ap_50:.4f}")
    print(f"mAP       @0.5:0.95:  {mean_ap_50_95:.4f}")
    print(f"Average IoU @IoU=0.5: {avg_iou_50:.4f}")
    print(f"Output images saved in: {args.output_dir}")

    evaluation_results = (
        "=== Evaluation Results (Test) ===\n"
        f"Precision @IoU=0.5:   {precision_50:.4f}\n"
        f"Recall    @IoU=0.5:   {recall_50:.4f}\n"
        f"mAP       @IoU=0.5:   {mean_ap_50:.4f}\n"
        f"mAP       @0.5:0.95:  {mean_ap_50_95:.4f}\n"
        f"Average IoU @IoU=0.5: {avg_iou_50:.4f}\n"
        f"Output images saved in: {args.output_dir}\n"
    )

    # Save evaluation results to eval.txt
    eval_file_path = os.path.join(args.output_dir, "eval.txt")
    with open(eval_file_path, "w") as eval_file:
        eval_file.write(evaluation_results)

    print(f"Evaluation results saved in: {eval_file_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='frcnnconfig/vocHOG_LBP_ORB.yaml',
                        help='Path ke file config YAML')
    parser.add_argument('--use_resnet50_fpn', type=bool, default=True,
                        help='Gunakan FasterRCNN_ResNet50_FPN default? (True/False)')
    parser.add_argument('--ckpt', type=str, required=False, default='Model FRCNN/A/frcnn_HOG_LBP_ORB/checkpoint_last.pth',
                        help='Path checkpoint .pth (misal checkpoint_last.pth)')
    parser.add_argument('--output_dir', type=str, default='Inference_FRCNN/A/HOG_LBP_ORB_last',
                        help='Folder untuk menyimpan hasil gambar prediksi & eval')
    args = parser.parse_args()

    evaluate(args)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='frcnnconfig/vocHOG_LBP_ORB.yaml',
                        help='Path ke file config YAML')
    parser.add_argument('--use_resnet50_fpn', type=bool, default=True,
                        help='Gunakan FasterRCNN_ResNet50_FPN default? (True/False)')
    parser.add_argument('--ckpt', type=str, required=False, default='Model FRCNN/A/frcnn_HOG_LBP_ORB/checkpoint_best.pth',
                        help='Path checkpoint .pth (misal checkpoint_best.pth)')
    parser.add_argument('--output_dir', type=str, default='Inference_FRCNN/A/HOG_LBP_ORB_best',
                        help='Folder untuk menyimpan hasil gambar prediksi & eval')
    args = parser.parse_args()

    evaluate(args)