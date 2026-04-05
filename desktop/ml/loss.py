import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            weight=self.alpha,
            ignore_index=self.ignore_index,
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        return focal_loss.mean()


class DiceLoss(nn.Module):
    def forward(self, logits, targets, smooth=1):
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = torch.nn.functional.one_hot(targets, probs.shape[1])
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss()
        self.focal = FocalLoss(alpha=class_weights)

    def forward(self, logits, targets):
        ce = self.ce(logits, targets)
        dice = self.dice(logits, targets)
        focal = self.focal(logits, targets)

        return 0.4 * ce + 0.4 * dice + 0.2 * focal
