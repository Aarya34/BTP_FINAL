"""
train.py (Fixed Linear Schedule)

- Implements Linear Warmup for GRL Lambda (0.0 -> 0.4 over 10 epochs).
- Prevents "Mode Collapse" (extra boxes) by adapting slowly.
"""

import argparse, math, csv, os
import torch
from torch.utils.data import DataLoader
from model import DA_FasterRCNN_RPNonly
from datasets import get_coco_source, UnlabeledImageFolder, collate_fn
from torchvision.ops import box_iou

def build_loaders(src_root, src_anns, tgt_root, batch_size=2, workers=4):
    src_ds = get_coco_source(src_root, src_anns)
    tgt_ds = UnlabeledImageFolder(tgt_root)
    src_loader = DataLoader(src_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=workers, collate_fn=collate_fn, drop_last=True)
    tgt_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=workers, collate_fn=collate_fn, drop_last=True)
    return src_loader, tgt_loader

def evaluate_metrics(model, loader, device, score_thresh=0.5, iou_thresh=0.5):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    print("\n>>> Running Evaluation...")
    max_eval_batches = 50 
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i >= max_eval_batches: break
            
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for target, output in zip(targets, outputs):
                gt_boxes = target['boxes'].to(device)
                pred_boxes = output['boxes']
                pred_scores = output['scores']
                
                keep = pred_scores > score_thresh
                pred_boxes = pred_boxes[keep]
                
                if len(gt_boxes) == 0:
                    false_positives += len(pred_boxes)
                    continue
                if len(pred_boxes) == 0:
                    false_negatives += len(gt_boxes)
                    continue
                
                ious = box_iou(gt_boxes, pred_boxes)
                found_gt = torch.zeros(len(gt_boxes), dtype=torch.bool, device=device)
                found_pred = torch.zeros(len(pred_boxes), dtype=torch.bool, device=device)
                
                for gt_idx in range(len(gt_boxes)):
                    max_iou, max_idx = torch.max(ious[gt_idx], dim=0)
                    if max_iou >= iou_thresh and not found_pred[max_idx]:
                        true_positives += 1
                        found_gt[gt_idx] = True
                        found_pred[max_idx] = True
                
                false_negatives += (len(gt_boxes) - found_gt.sum().item())
                false_positives += (len(pred_boxes) - found_pred.sum().item())

    epsilon = 1e-6
    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    print(f">>> Eval Result: Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
    return precision, recall, f1

def train_epoch(model, src_loader, tgt_loader, optimizer, device, epoch, grl_lambda):
    model.train()
    src_iter = iter(src_loader)
    tgt_iter = iter(tgt_loader)
    iters = min(len(src_loader), len(tgt_loader))
    
    tracker = {
        "Total": 0.0, "Source": 0.0, "Target": 0.0,
        "RPN": 0.0, "ROI": 0.0, "Img_DA": 0.0, "Inst_DA": 0.0
    }
    
    for i in range(iters):
        try:
            images_s, targets_s = next(src_iter)
            images_s = [img.to(device) for img in images_s]
            targets_s = [{k:v.to(device) for k,v in t.items()} for t in targets_s]
            losses_s = model(images_s, targets_s, domain='source', grl_lambda=grl_lambda)
            loss_s = sum(v for v in losses_s.values())

            images_t, _ = next(tgt_iter)
            images_t = [img.to(device) for img in images_t]
            losses_t = model(images_t, targets=None, domain='target', grl_lambda=grl_lambda)
            loss_t = losses_t['loss_image_domain'] + losses_t['loss_instance_domain']

            loss = loss_s + 0.1 * loss_t

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tracker["Total"] += loss.item()
            tracker["Source"] += loss_s.item()
            tracker["Target"] += loss_t.item()
            tracker["RPN"] += (losses_s.get('loss_objectness', 0) + losses_s.get('loss_rpn_box_reg', 0))
            tracker["ROI"] += (losses_s.get('loss_classifier', 0) + losses_s.get('loss_box_reg', 0))
            tracker["Img_DA"] += losses_t['loss_image_domain'].item()
            tracker["Inst_DA"] += losses_t['loss_instance_domain'].item()

            if i % 20 == 0:
                print(f"[Epoch {epoch}][Iter {i}/{iters}] Loss={loss.item():.3f}")
        
        except StopIteration:
            break
        except Exception as e:
            print(f"Error at iter {i}: {e}")
            continue

    if iters == 0: iters = 1
    return {k: v/iters for k, v in tracker.items()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-root", required=True)
    parser.add_argument("--src-anns", required=True)
    parser.add_argument("--tgt-root", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0025)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    log_file = "training_metrics.csv"
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Total_Loss", "Source_Loss", "Target_Loss", 
            "RPN_Loss", "ROI_Loss", "Img_DA_Loss", "Inst_DA_Loss", "GRL_Lambda",
            "Precision", "Recall", "F1_Score"
        ])
    print(f"Logging metrics to {log_file}...")

    src_loader, tgt_loader = build_loaders(args.src_root, args.src_anns, args.tgt_root, batch_size=args.batch_size)

    model = DA_FasterRCNN_RPNonly(pretrained_backbone=True)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)

    total_epochs = args.epochs
    for epoch in range(1, total_epochs+1):
        
        progress = (epoch - 1) / max(1, total_epochs - 1)
        grl_lambda = 0.4 * progress
        
        print(f"\nStarting Epoch {epoch} with GRL Lambda: {grl_lambda:.4f}")
        
        avg_losses = train_epoch(model, src_loader, tgt_loader, optimizer, device, epoch, grl_lambda)
        
        prec, rec, f1 = evaluate_metrics(model, src_loader, device)
        
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_losses['Total']:.4f}", f"{avg_losses['Source']:.4f}", f"{avg_losses['Target']:.4f}",
                f"{avg_losses['RPN']:.4f}", f"{avg_losses['ROI']:.4f}", 
                f"{avg_losses['Img_DA']:.4f}", f"{avg_losses['Inst_DA']:.4f}",
                f"{grl_lambda:.4f}",
                f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"
            ])
            f.flush()
        
        torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")

if __name__ == "__main__":
    main()