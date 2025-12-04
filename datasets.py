"""
datasets.py

Utilities for:
- Loading a labeled source dataset (COCO-style)
- Loading an unlabeled target dataset (folder of images)
- collate_fn for DataLoader
"""

import os
from PIL import Image
import torch
from torchvision.datasets import CocoDetection
from torchvision import transforms
from torch.utils.data import Dataset

class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor:
    def __call__(self, image, target):
        return transforms.functional.to_tensor(image), target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5):
        self.prob = prob
    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            image = transforms.functional.hflip(image)
            if target is not None and "boxes" in target:
                _, _, width = image.shape
                target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        return image, target

def make_transform(train=True):
    t = []
    t.append(ToTensor()) 
    if train:
        t.append(RandomHorizontalFlip(0.5)) 
    return CustomCompose(t) 

class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.root = root
        self.paths = []
        extensions = ('.jpg', '.jpeg', '.png')
        
        for dirpath, dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.lower().endswith(extensions):
                    self.paths.append(os.path.join(dirpath, filename))
                    
        self.transform = transform if transform is not None else make_transform(train=False)
        print(f"Found {len(self.paths)} images in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        
        img, _ = self.transform(img, None) 
        
        return img, None
def collate_fn(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets

def get_coco_source(root, ann_file, transforms=None):

    class CocoWrapper(CocoDetection):
        def __init__(self, root, annFile, transforms):
            super().__init__(root, annFile)
            self._transforms = transforms
            
        def __getitem__(self, idx):
            img, target = super().__getitem__(idx)
            
            boxes = []
            labels = []
            areas = []
            iscrowd = []
            
            for ann in target:
                if 'bbox' not in ann:
                    continue
                x,y,w,h = ann['bbox']
                if w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x+w, y+h])
                labels.append(1) 
                areas.append(ann.get('area', w*h))
                iscrowd.append(ann.get('iscrowd', 0))
                
            target_dict = {}
            
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            areas_tensor = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.as_tensor(iscrowd, dtype=torch.int64)
            
            if boxes_tensor.numel() == 0:
                boxes_tensor = boxes_tensor.reshape(0, 4)

            target_dict['boxes'] = boxes_tensor
            target_dict['labels'] = labels_tensor
            target_dict['area'] = areas_tensor
            target_dict['iscrowd'] = iscrowd_tensor
            
            
            image_id = self.ids[idx]
            target_dict['image_id'] = torch.tensor([image_id])
            
            if self._transforms is not None:
                img, target_dict = self._transforms(img, target_dict) 
                
            return img, target_dict

    return CocoWrapper(root, ann_file, transforms=transforms or make_transform(train=True))
