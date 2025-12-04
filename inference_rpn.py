"""
inference_rpn.py (Final Fixed Version)

- Implements NMS to merge duplicate boxes
- Filters weak predictions (default > 0.75) to prevent 'extra boxes'
"""

import argparse, os
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from torchvision.ops import nms
from model import DA_FasterRCNN_RPNonly

def load_image(path):
    img = Image.open(path).convert("RGB")
    t = transforms.ToTensor()
    return t(img)

def draw_boxes(img_pil, boxes, scores):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for i, b in enumerate(boxes):
        score = scores[i]
        draw.rectangle([b[0], b[1], b[2], b[3]], outline="red", width=3)
        draw.text((b[0], b[1]), f"{score:.2f}", fill="red", font=font)
    return img_pil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to the .pth checkpoint file")
    parser.add_argument("--input-folder", required=True, dest="input_folder")
    parser.add_argument("--output-folder", required=True, dest="output_folder")
    
    parser.add_argument("--score-thresh", type=float, default=0.75, dest="score_thresh", help="Confidence threshold")
    parser.add_argument("--topk", type=int, default=20, help="Max number of boxes to draw")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    model = DA_FasterRCNN_RPNonly(pretrained_backbone=False)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(args.output_folder, exist_ok=True)
    
    for fname in sorted(os.listdir(args.input_folder)):
        if not fname.lower().endswith(('.jpg','.jpeg','.png')): continue
        
        path = os.path.join(args.input_folder, fname)
        pil_img = Image.open(path).convert("RGB")
        img_tensor = load_image(path).to(device)
        
        with torch.no_grad():
            detections = model([img_tensor])[0]

        boxes = detections['boxes']
        scores = detections['scores']
        labels = detections['labels']
        
        keep_idxs = (labels == 1) & (scores > args.score_thresh)
        boxes = boxes[keep_idxs]
        scores = scores[keep_idxs]
        
        keep_nms = nms(boxes, scores, iou_threshold=0.3)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]

        if len(boxes) > args.topk:
            boxes = boxes[:args.topk]
            scores = scores[:args.topk]
        
        final_boxes = boxes.cpu().numpy()
        final_scores = scores.cpu().numpy()
        
        pil_img = draw_boxes(pil_img, final_boxes, final_scores)
        out_path = os.path.join(args.output_folder, fname)
        pil_img.save(out_path)
        print(f"Saved: {fname} (Found {len(final_boxes)} objects)")

if __name__ == "__main__":
    main()