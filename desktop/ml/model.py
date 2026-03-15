import torch
import torch.nn as nn
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)


class VisionNetAdapter(nn.Module):
    def __init__(self, num_classes: int = 12, pretrained: bool = True, **kwargs):
        super(VisionNetAdapter, self).__init__()

        if pretrained:
            self.model = deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT, **kwargs
            )

            in_channels = self.model.classifier[4].in_channels
            self.model.classifier[4] = nn.Conv2d(
                in_channels, num_classes, kernel_size=1
            )

            if hasattr(self.model, "aux_classifier"):
                in_channels_aux = self.model.aux_classifier[4].in_channels
                self.model.aux_classifier[4] = nn.Conv2d(
                    in_channels_aux, num_classes, kernel_size=1
                )
        else:
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


def create_model(num_classes):
    return VisionNetAdapter(num_classes=num_classes, pretrained=True)


def load_model(path, device="cpu"):
    checkpoint = torch.load(path, map_location=device)
    config = checkpoint["config"]
    model = VisionNetAdapter(
        num_classes=config.get("num_classes", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, config


def save_model(model, path, config):
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
