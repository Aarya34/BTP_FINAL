"""
model.py

Implements:
- DA-Faster R-CNN style adaptation heads (image & instance domain classifiers with GRL)
- Class-Agnostic Object Detection head (object vs background)
- Uses the Region Proposal Network (RPN) from torchvision's Faster R-CNN
- Designed for Zero-Shot Object Detection (ZSOD) use-case where final evaluation needs boxes only
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import boxes as box_ops

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grl(x, lambd=1.0):
    return GradientReversal.apply(x, lambd)

class ImageDomainClassifier(nn.Module):
    def __init__(self, fpn_levels=5, fpn_channels=256, hidden=1024):
        super().__init__()
        in_dim = fpn_levels * fpn_channels
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 2) 
        )
    def forward(self, features, lambd=1.0):
        pooled = []
        for f in features.values():
            pooled.append(F.adaptive_avg_pool2d(f, (1,1)).flatten(1))
        x = torch.cat(pooled, dim=1)
        x = grl(x, lambd)
        return self.net(x)

class InstanceDomainClassifier(nn.Module):
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 2)
        )
    def forward(self, roi_feats, lambd=1.0):
        x = grl(roi_feats, lambd)
        return self.net(x)

class ClassAgnosticBoxPredictor(nn.Module):
    def __init__(self, in_channels, representation_size=1024):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, representation_size)
        self.fc2 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, 2)  
        self.bbox_pred = nn.Linear(representation_size, 2 * 4)   

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas

class DA_FasterRCNN_RPNonly(nn.Module):

    def __init__(self, pretrained_backbone=True):
        super().__init__()
        self.detector = fasterrcnn_resnet50_fpn(pretrained=pretrained_backbone)
        box_head = self.detector.roi_heads.box_head
        sample_in = None
        if hasattr(box_head, "fc7"):
            in_channels = box_head.fc7.out_features
        elif hasattr(box_head, "fc6"):
            in_channels = box_head.fc6.out_features
        else:
            for m in box_head.modules():
                if isinstance(m, nn.Linear):
                    in_channels = m.in_features
                    break
        self.detector.roi_heads.box_predictor = ClassAgnosticBoxPredictor(in_channels)

        self.image_domain_clf = ImageDomainClassifier()
        self.instance_domain_clf = InstanceDomainClassifier(in_dim=in_channels)

        self.rpn = self.detector.rpn
        self.backbone = self.detector.backbone
        self.roi_heads = self.detector.roi_heads
        self.transform = self.detector.transform

    def forward(self, images, targets=None, domain='source', grl_lambda=1.0):

        if self.training:
            if domain == 'source' and targets is None:
                raise ValueError("Source domain training requires targets.")
                
            images_t, targets = self.transform(images, targets)
            
            features = self.backbone(images_t.tensors)
            losses = {}

            if domain == 'source':
                proposals, rpn_losses = self.rpn(images_t, features, targets)
                detections, detector_losses = self.roi_heads(features, proposals, images_t.image_sizes, targets)
                losses.update(rpn_losses)
                losses.update(detector_losses)
            
            else: 
                self.rpn.eval() 
                proposals, _ = self.rpn(images_t, features, None)
                self.rpn.train() 

            img_dom_logits = self.image_domain_clf(features, lambd=grl_lambda)
            device = img_dom_logits.device
            label = 0 if domain == 'source' else 1
            img_labels = torch.ones(img_dom_logits.size(0), dtype=torch.long, device=device) * label
            losses['loss_image_domain'] = F.cross_entropy(img_dom_logits, img_labels)
            
            box_features = self.roi_heads.box_roi_pool(features, proposals, images_t.image_sizes)
            box_features = self.roi_heads.box_head(box_features) 
            inst_dom_logits = self.instance_domain_clf(box_features, lambd=grl_lambda)
            inst_labels = torch.ones(inst_dom_logits.size(0), dtype=torch.long, device=device) * label
            losses['loss_instance_domain'] = F.cross_entropy(inst_dom_logits, inst_labels)

            return losses
        else:
            return self.detector(images)
    def infer_proposals(self, images, score_thresh=0.3, nms_thresh=0.6, topk=300):

        self.eval()
        with torch.no_grad():
            images_t = self.transform(images)
            features = self.backbone(images_t.tensors)
            objectness, pred_bbox_deltas = self.rpn.head(features)
            proposals, _ = self.rpn(images_t, features, targets=None)
            results = []
            for img_idx, props in enumerate(proposals):
                boxes = props
                if boxes.numel() == 0:
                    results.append({'boxes': boxes, 'scores': boxes.new_zeros((0,))})
                    continue
                scores = torch.ones((boxes.shape[0],), device=boxes.device) 
                keep = box_ops.nms(boxes, scores, nms_thresh)
                keep = keep[:topk]
                kept_boxes = boxes[keep]
                kept_scores = scores[keep]

                results.append({'boxes': kept_boxes, 'scores': kept_scores})
            return results
