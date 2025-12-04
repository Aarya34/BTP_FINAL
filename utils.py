from torchvision.ops import nms, box_iou
import torch

def apply_nms(boxes, scores, iou_thresh=0.6, topk=300):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)
    keep = nms(boxes, scores, iou_thresh)
    if topk is not None:
        keep = keep[:topk]
    return keep
