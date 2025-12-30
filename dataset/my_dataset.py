import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO


class AITODv2Dataset(Dataset):
    def __init__(self, img_root, ann_root, transforms=None):
        self.img_root = img_root
        self.transforms = transforms
        self.coco = COCO(ann_root)
        self.img_ids = sorted(self.coco.getImgIds())

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 1 or h <= 1:
                continue

            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])
            areas.append(w * h)
            iscrowd.append(0)

        if len(boxes) == 0:
            return self.__getitem__((idx + 1) % len(self))
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


    def coco_index(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        h = img_info["height"]
        w = img_info["width"]

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels, areas, iscrowd = [], [], [], []

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw <= 1 or bh <= 1:
                continue

            boxes.append([x, y, x + bw, y + bh])
            labels.append(ann["category_id"])
            areas.append(bw * bh)
            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": areas,
            "iscrowd": iscrowd,
        }

        return (h, w), target