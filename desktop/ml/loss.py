import torch
import torch.nn as nn


class VisionNetLoss(nn.Module):
    def __init__(self, num_classes=6, device="cuda"):
        super().__init__()

        # Веса для классов (настройте под свои данные)
        self.class_weights = torch.tensor(
            [
                0.3,  # фон (обычно много)
                0.8,  # лист
                1.5,  # стебель (мелкий)
                1.0,  # плод
                2.0,  # черешок (очень мелкий)
                2.0,  # болезнь (мелкая, но важная)
            ]
        ).to(device)

        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)

        dice = self.dice_loss(pred, target)

        total_loss = ce + dice

        return total_loss


class DiceLoss(nn.Module):
    def forward(self, logits, targets, smooth=1):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, probs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
