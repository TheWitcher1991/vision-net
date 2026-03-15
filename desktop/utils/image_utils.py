import io

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation


def create_overlay(original_image, mask, alpha=0.5):
    original_np = np.array(original_image)
    if isinstance(mask, Image.Image):
        mask_np = np.array(mask)
    else:
        mask_np = mask

    if len(mask_np.shape) == 3:
        mask_np = mask_np[:, :, 0]

    h, w = mask_np.shape
    if original_np.shape[0] != h or original_np.shape[1] != w:
        mask_pil = Image.fromarray(mask_np)
        mask_pil = mask_pil.resize((original_np.shape[1], original_np.shape[0]), Image.Resampling.NEAREST)
        mask_np = np.array(mask_pil)

    num_classes = mask_np.max() + 1
    colors = generate_colors(num_classes)
    colored_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_idx in range(num_classes):
        if class_idx < len(colors):
            colored_mask[mask_np == class_idx] = colors[class_idx]

    overlay = (original_np * (1 - alpha) + colored_mask * alpha).astype(np.uint8)

    return Image.fromarray(overlay)


def find_border(mask):
    dilated = binary_dilation(mask)
    border = dilated ^ mask
    return border


def generate_colors(num_classes):
    np.random.seed(42)
    colors = []
    for i in range(num_classes):
        hue = i / max(num_classes, 1)
        rgb = hsl_to_rgb(hue, 0.7, 0.5)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def hsl_to_rgb(h, s, l):
    if s == 0:
        return [l, l, l]

    def hue_to_rgb(p, q, t):
        if t < 0:
            t += 1
        if t > 1:
            t -= 1
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    r = hue_to_rgb(p, q, h + 1 / 3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1 / 3)

    return [r, g, b]


def image_to_bytes(image, format="PNG"):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return img_byte_arr.getvalue()


def bytes_to_image(byte_data):
    return Image.open(io.BytesIO(byte_data))


def normalize_image(image):
    img_array = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (img_array - mean) / std
    return normalized


def denormalize_image(image_array, mean, std):
    denormalized = image_array * std + mean
    denormalized = np.clip(denormalized * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(denormalized)
