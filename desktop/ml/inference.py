import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class Inference:
    def __init__(self, model, device="cpu", classes=None, image_size=512, threshold=0.5):
        self.model = model.to(device)
        self.device = device
        self.classes = classes or ["background", "tomato_leaf"]
        self.image_size = image_size
        self.threshold = threshold
        self.model.eval()

    def preprocess(self, image_path):
        original = Image.open(image_path).convert("RGB")
        original_size = original.size
        image = original.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_tensor, original_size, original

    def predict(self, image_tensor):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        return probabilities

    def postprocess(self, probabilities, original_size):
        probs_resized = (
            F.interpolate(
                probabilities,
                size=(original_size[1], original_size[0]),
                mode="bilinear",
                align_corners=False,
            )
            .cpu()
            .numpy()[0]
        )

        num_classes, H, W = probs_resized.shape
        mask = np.zeros((H, W), dtype=np.uint8)
        detected_classes = []

        for i in range(1, num_classes):
            prob_map = probs_resized[i]
            object_mask = prob_map >= self.threshold
            mask[object_mask] = i
            if object_mask.sum() > 0:
                confidence = float(prob_map[object_mask].mean())
                detected_classes.append((self.classes[i], confidence))

        if np.all(mask == 0):
            detected_classes = []

        mask_image = Image.fromarray(mask.astype(np.uint8))
        return mask_image, detected_classes

    def infer(self, image_path):
        print(f"[INFERENCE] Loading image: {image_path}")
        image_tensor, original_size, original_image = self.preprocess(image_path)
        print(f"[INFERENCE] Image size: {original_size}, model input size: {self.image_size}")

        probabilities = self.predict(image_tensor)
        mask_image, detected_classes = self.postprocess(probabilities, original_size)

        print(f"[INFERENCE] Detected classes: {detected_classes}")
        mask_arr = np.array(mask_image)
        unique, counts = np.unique(mask_arr, return_counts=True)
        print(f"[INFERENCE] Mask unique values: {dict(zip(unique.tolist(), counts.tolist()))}")

        return {
            "original_image": original_image,
            "mask": mask_image,
            "detected_classes": detected_classes,
            "probabilities": probabilities,
        }
