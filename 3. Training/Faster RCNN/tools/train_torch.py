import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
import torchvision
import csv

import cv2  # Pastikan opencv terinstall
from Dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

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
    area_union = float(det_area + gt_area - area_intersection + 1E-6)
    iou = area_intersection / area_union
    return iou


def compute_map(det_boxes, gt_boxes, iou_threshold=0.5, method='interp'):
    """
    Return:
      mean_ap (float)
      all_aps (dict)
      global_recall (float): micro-averaged recall
      global_precision (float): micro-averaged precision
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


def train(args):
    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(config)

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    # Set seed
    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Dataset
    voc_train = VOCDataset('train',
                           im_dir=dataset_config['im_train_path'],
                           ann_dir=dataset_config['ann_train_path'])
    train_loader = DataLoader(voc_train,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4,
                              collate_fn=collate_function)

    voc_val = VOCDataset('val',
                         im_dir=dataset_config['im_val_path'],
                         ann_dir=dataset_config['ann_val_path'])
    val_loader = DataLoader(voc_val,
                            batch_size=4,
                            shuffle=False,
                            num_workers=4,
                            collate_fn=collate_function)

    # Model
    if args.use_resnet50_fpn:
        faster_rcnn_model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,  # atau DEFAULT
            min_size=600,
            max_size=1000
        )
        faster_rcnn_model.roi_heads.box_predictor = FastRCNNPredictor(
            faster_rcnn_model.roi_heads.box_predictor.cls_score.in_features,
            num_classes=5
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
        faster_rcnn_model = torchvision.models.detection.FasterRCNN(
            backbone,
            num_classes=5,
            min_size=600,
            max_size=1000,
            rpn_anchor_generator=rpn_anchor_generator,
            box_roi_pool=roi_align
        )

    faster_rcnn_model.to(device)

    # Folder ckpt
    save_dir = train_config['task_name']
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Optimizer
    optimizer = torch.optim.SGD(
        params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        lr=1e-4,
        weight_decay=5e-5,
        momentum=0.9
    )
    num_epochs = train_config['num_epochs']

    # Resume
    start_epoch = 0
    best_mAP = 0.0
    if args.resume is not None and os.path.isfile(args.resume):
        print(f"==> Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        faster_rcnn_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_mAP = ckpt['best_mAP']
        print(f"   => resumed epoch={start_epoch}, best_mAP={best_mAP:.4f}")

    # CSV
    csv_path = os.path.join(save_dir, 'metrics.csv')
    need_header = (not os.path.exists(csv_path)) or (args.resume is None)
    if need_header:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_rpn_cls_loss", "train_rpn_loc_loss",
                "train_frcnn_cls_loss", "train_frcnn_loc_loss",
                "val_mAP", "val_recall", "val_precision"
            ])

    # Loop
    for epoch in range(start_epoch, num_epochs):
        faster_rcnn_model.train()
        rpn_cls_losses, rpn_loc_losses = [], []
        frcnn_cls_losses, frcnn_loc_losses = [], []

        for images, targets, _ in tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            # Persiapkan data
            for t in targets:
                t["boxes"] = t["bboxes"].float().to(device)
                del t["bboxes"]
                t["labels"] = t["labels"].long().to(device)
            images_gpu = [img.float().to(device) for img in images]

            loss_dict = faster_rcnn_model(images_gpu, targets)
            loss_val = sum(loss_dict.values())
            loss_val.backward()
            optimizer.step()

            rpn_cls_losses.append(loss_dict['loss_objectness'].item())
            rpn_loc_losses.append(loss_dict['loss_rpn_box_reg'].item())
            frcnn_cls_losses.append(loss_dict['loss_classifier'].item())
            frcnn_loc_losses.append(loss_dict['loss_box_reg'].item())

        # Validation
        faster_rcnn_model.eval()
        val_preds, val_gts = [], []
        with torch.no_grad():
            for images, targets, _ in tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}"):
                images_gpu = [img.float().to(device) for img in images]
                outputs = faster_rcnn_model(images_gpu)

                for b_idx in range(len(images)):
                    single_gt = {}
                    single_pred = {}
                    # Siapkan label2idx (pastikan sesuai di dataset VOCDataset Anda)
                    for label_name in voc_val.label2idx.keys():
                        single_gt[label_name] = []
                        single_pred[label_name] = []

                    # GT
                    for i_box, box in enumerate(targets[b_idx]['bboxes']):
                        x1, y1, x2, y2 = box.cpu().numpy()
                        label_id = targets[b_idx]['labels'][i_box].cpu().item()
                        label_name = voc_val.idx2label[label_id]
                        single_gt[label_name].append([x1, y1, x2, y2])

                    # Prediction
                    pred_boxes = outputs[b_idx]['boxes']
                    pred_labels = outputs[b_idx]['labels']
                    pred_scores = outputs[b_idx]['scores']
                    for i_p, p_box in enumerate(pred_boxes):
                        px1, py1, px2, py2 = p_box.cpu().numpy()
                        plabel_id = pred_labels[i_p].cpu().item()
                        pscore = pred_scores[i_p].cpu().item()
                        plabel_name = voc_val.idx2label[plabel_id]
                        single_pred[plabel_name].append([px1, py1, px2, py2, pscore])

                    val_preds.append(single_pred)
                    val_gts.append(single_gt)

        # Hitung metrics
        mean_train_rpn_cls = np.mean(rpn_cls_losses)
        mean_train_rpn_loc = np.mean(rpn_loc_losses)
        mean_train_frcnn_cls = np.mean(frcnn_cls_losses)
        mean_train_frcnn_loc = np.mean(frcnn_loc_losses)

        val_mean_ap, val_all_aps, val_global_recall, val_global_precision = compute_map(
            val_preds, val_gts, method='interp'
        )

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Train Losses:")
        print(f"  RPN cls: {mean_train_rpn_cls:.4f} | RPN loc: {mean_train_rpn_loc:.4f} | "
              f"FRCNN cls: {mean_train_frcnn_cls:.4f} | FRCNN loc: {mean_train_frcnn_loc:.4f}")
        print(f"Val mAP: {val_mean_ap:.4f} | Recall: {val_global_recall:.4f} | Precision: {val_global_precision:.4f} "
              f"(best so far: {best_mAP:.4f})")

        # Tulis CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch+1,
                mean_train_rpn_cls, mean_train_rpn_loc,
                mean_train_frcnn_cls, mean_train_frcnn_loc,
                val_mean_ap, val_global_recall, val_global_precision
            ])

        # =====================
        # Save "last" (overwrite)
        # =====================
        ckpt_last = os.path.join(save_dir, "checkpoint_last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': faster_rcnn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mAP': best_mAP
        }, ckpt_last)

        # =====================
        # Save "best"
        # =====================
        if val_mean_ap > best_mAP:
            best_mAP = val_mean_ap
            ckpt_best = os.path.join(save_dir, "checkpoint_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': faster_rcnn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_mAP': best_mAP
            }, ckpt_best)

    print("Done Training...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config_path',
                        default='frcnnconfig/New/vocHSL.yaml', type=str)
    parser.add_argument('--use_resnet50_fpn', dest='use_resnet50_fpn',
                        default=True, type=bool)
    parser.add_argument('--resume', default=None, type=str,
                        help='path ke checkpoint untuk resume')
    args = parser.parse_args()
    train(args)