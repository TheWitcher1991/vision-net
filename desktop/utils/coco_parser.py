import json
import os


class COCOParser:
    def __init__(self, annotations_path):
        self.annotations_path = annotations_path
        self.data = None
        self.images = {}
        self.annotations = {}
        self.categories = {}

    def load(self):
        if not os.path.exists(self.annotations_path):
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        with open(self.annotations_path, "r") as f:
            self.data = json.load(f)

        self.images = {img["id"]: img for img in self.data.get("images", [])}
        self.categories = {cat["id"]: cat for cat in self.data.get("categories", [])}

        for ann in self.data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        return self.data

    def get_image(self, image_id):
        return self.images.get(image_id)

    def get_annotations(self, image_id):
        return self.annotations.get(image_id, [])

    def get_categories(self):
        return list(self.categories.values())

    def get_stats(self):
        annotations = self.data.get("annotations", []) if self.data else []
        return {
            "num_images": len(self.images),
            "num_annotations": len(annotations),
            "num_categories": len(self.categories),
        }

    def get_category_names(self):
        return [cat["name"] for cat in self.categories.values()]
