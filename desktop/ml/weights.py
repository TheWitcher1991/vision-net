import numpy as np
import torch


def compute_class_weights(loader, num_classes):
    counts = np.zeros(num_classes)

    for _, masks in loader:
        masks = masks.numpy()
        for c in range(num_classes):
            counts[c] += (masks == c).sum()

    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes

    return torch.tensor(weights, dtype=torch.float32)
