import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class COCOSegmentationDataset(Dataset):
    def __init__(self, root, annotations_file, image_size=512, transform=None):
        self.root = root
        self.images_folder = os.path.join(root, "images")
        self.image_size = image_size
        self.transform = transform

        with open(annotations_file, "r") as f:
            self.coco_data = json.load(f)

        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = {}
        for ann in self.coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}
        self.image_ids = list(self.images.keys())

        self.class_ids = sorted([cat["id"] for cat in self.coco_data["categories"]])
        self.id_to_class_idx = {cat_id: idx + 1 for idx, cat_id in enumerate(self.class_ids)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]

        image_path = os.path.join(self.images_folder, image_info["file_name"])
        image = Image.open(image_path).convert("RGB")

        w, h = image.size
        mask = np.zeros((h, w), dtype=np.uint8)

        if image_id in self.annotations:
            for ann in self.annotations[image_id]:
                if ann["iscrowd"]:
                    continue
                class_idx = self.id_to_class_idx[ann["category_id"]]
                segmentation = ann["segmentation"]

                if isinstance(segmentation, list):
                    for seg in segmentation:
                        poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                        from PIL import ImageDraw

                        mask_pil = Image.fromarray(mask)
                        ImageDraw.Draw(mask_pil).polygon(
                            [(int(p[0]), int(p[1])) for p in poly],
                            outline=class_idx,
                            fill=class_idx,
                        )
                        mask = np.array(mask_pil)

        image = image.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        mask = Image.fromarray(mask).resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        mask = np.array(mask).astype(np.int64)
        mask = torch.from_numpy(mask)

        return image, mask

    def get_classes(self):
        return ["background"] + [self.categories[cat_id]["name"] for cat_id in self.class_ids]

    def get_stats(self):
        return {
            "num_images": len(self.images),
            "num_annotations": len(self.coco_data["annotations"]),
            "num_classes": len(self.categories) + 1,
        }
