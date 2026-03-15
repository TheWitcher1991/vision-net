import io
import json
import colorsys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
)

MODELS_DIR = Path(__file__).parent / "models"


class VisionNetAdapter(nn.Module):
    def __init__(self, num_classes: int = 12, pretrained: bool = True, **kwargs):
        super(VisionNetAdapter, self).__init__()

        self.model = deeplabv3_resnet50(
            pretrained=False, num_classes=num_classes, **kwargs
        )

        self.num_classes = num_classes

    def forward(self, x):
        output = self.model(x)

        if isinstance(output, dict):
            logits = output["out"]
        else:
            logits = output

        return logits


class InferenceService:
    def __init__(self):
        self.model = None
        self.config = None
        self.device = "cpu"
        self.classes = ["background"]
        self.image_size = 512
        self.threshold = 0.5

    def list_models(self):
        if not MODELS_DIR.exists():
            return []
        models = []
        for item in MODELS_DIR.iterdir():
            if item.is_dir():
                config_file = item / "config.json"
                model_file = item / "model.pth"
                if config_file.exists() and model_file.exists():
                    models.append(item.name)
        return models

    def load_model(self, model_name: str):
        model_dir = MODELS_DIR / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_name}")

        model_path = model_dir / "model.pth"
        config_path = model_dir / "config.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            self.config = json.load(f)

        checkpoint = torch.load(model_path, map_location=self.device)

        in_channels = self.config.get("in_channels", 3)
        num_classes = self.config.get("num_classes", 1)
        self.classes = self.config.get("classes", ["background"])

        self.model = VisionNetAdapter(num_classes=num_classes)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: Image.Image):
        original = image.convert("RGB")
        original_size = original.size
        image_resized = original.resize(
            (self.image_size, self.image_size), Image.Resampling.BILINEAR
        )
        image_np = np.array(image_resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
        return image_tensor, original_size, original

    def predict(self, image_tensor: torch.Tensor):
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        return probabilities

    def postprocess(self, probabilities: torch.Tensor, original_size: tuple):
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
                class_name = self.classes[i] if i < len(self.classes) else f"class_{i}"
                detected_classes.append({"class": class_name, "confidence": confidence})

        if np.all(mask == 0):
            detected_classes = []

        mask_image = Image.fromarray(mask.astype(np.uint8))
        return mask_image, detected_classes

    def create_overlay(self, original: Image.Image, mask: Image.Image, alpha: float = 0.5):
        mask_np = np.array(mask)
        original_np = np.array(original)

        num_classes = len(self.classes)
        colors = self._generate_colors(num_classes)

        overlay_np = original_np.copy()

        for class_id in range(1, num_classes):
            class_mask = mask_np == class_id
            if class_mask.sum() > 0:
                color = colors[class_id % len(colors)]
                overlay_np[class_mask] = (
                    (1 - alpha) * original_np[class_mask] + 
                    alpha * np.array(color)
                ).astype(np.uint8)
                
                border_mask = self._get_boundary_mask(class_mask, thickness=3)
                overlay_np[border_mask] = color

        return Image.fromarray(overlay_np)

    def create_class_overlay(self, original: Image.Image, mask: Image.Image, class_id: int, alpha: float = 0.5):
        mask_np = np.array(mask)
        original_np = np.array(original)
        
        num_classes = len(self.classes)
        colors = self._generate_colors(num_classes)
        
        overlay_np = original_np.copy()
        
        class_mask = mask_np == class_id
        if class_mask.sum() > 0:
            color = colors[class_id % len(colors)]
            overlay_np[class_mask] = (
                (1 - alpha) * original_np[class_mask] + 
                alpha * np.array(color)
            ).astype(np.uint8)
            
            border_mask = self._get_boundary_mask(class_mask, thickness=4)
            overlay_np[border_mask] = (255, 255, 255)
            
            inner_border = self._get_boundary_mask(class_mask, thickness=2)
            overlay_np[inner_border] = color
        
        return Image.fromarray(overlay_np)

    def _get_boundary_mask(self, mask: np.ndarray, thickness: int = 3) -> np.ndarray:
        from scipy import ndimage
        
        kernel = np.ones((thickness, thickness), dtype=np.uint8)
        dilated = ndimage.binary_dilation(mask, structure=kernel)
        boundary = dilated & ~mask
        
        return boundary

    def _generate_colors(self, num_classes: int):
        np.random.seed(42)
        colors = []
        for i in range(num_classes):
            hue = i / num_classes
            rgb = self._hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        return colors

    def _hsv_to_rgb(self, h, s, v):
        import colorsys
        rgb = colorsys.hsv_to_rgb(h, s, v)
        return tuple(int(x * 255) for x in rgb)

    def infer(self, image_bytes: bytes) -> dict:
        import base64
        
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        image_tensor, original_size, original_image = self.preprocess(image)
        probabilities = self.predict(image_tensor)
        mask_image, detected_classes = self.postprocess(probabilities, original_size)

        overlay_image = self.create_overlay(original_image, mask_image, alpha=0.5)

        output = io.BytesIO()
        original_image.save(output, format="PNG")
        original_b64 = output.getvalue()

        output = io.BytesIO()
        mask_image.save(output, format="PNG")
        mask_b64 = output.getvalue()

        output = io.BytesIO()
        overlay_image.save(output, format="PNG")
        overlay_b64 = output.getvalue()

        class_overlays = {}
        for cls in detected_classes:
            class_name = cls["class"]
            class_id = self.classes.index(class_name) if class_name in self.classes else -1
            if class_id > 0:
                class_overlay = self.create_class_overlay(original_image, mask_image, class_id, alpha=0.6)
                output = io.BytesIO()
                class_overlay.save(output, format="PNG")
                class_overlays[class_name] = base64.b64encode(output.getvalue()).decode()

        return {
            "original_image": base64.b64encode(original_b64).decode(),
            "mask": base64.b64encode(mask_b64).decode(),
            "overlay": base64.b64encode(overlay_b64).decode(),
            "detected_classes": detected_classes,
            "class_overlays": class_overlays,
        }
