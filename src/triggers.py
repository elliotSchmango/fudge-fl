import torch
import numpy as np

#apply localized patch trigger to subset of images
#x' = (1 - alpha) * x + alpha * delta, where alpha is binary mask
def apply_local_patch(images, labels, poison_rate=0.20, target_label=0,
                      patch_size=2, patch_value=1.0, position="bottom-right"):
    """
    stamp high-intensity patch onto a subset of images and flip their labels.
    images: tensor [B, C, H, W]
    labels: tensor [B]
    position: corner placement of patch ("bottom-right", "top-left", "top-right", "bottom-left")
    """
    num_poison = int(len(images) * poison_rate)
    if num_poison == 0:
        return images, labels

    h, w = images.shape[2], images.shape[3]

    #compute patch coordinates based on chosen corner
    coords = {
        "bottom-right": (h - patch_size, w - patch_size),
        "top-left": (0, 0),
        "top-right": (0, w - patch_size),
        "bottom-left": (h - patch_size, 0),
    }
    if position not in coords:
        raise ValueError(f"unknown position '{position}', choose from {list(coords.keys())}")

    row_start, col_start = coords[position]
    row_end = row_start + patch_size
    col_end = col_start + patch_size

    #overwrite pixel region with solid patch value across all channels
    images[:num_poison, :, row_start:row_end, col_start:col_end] = patch_value

    #flip labels to target class
    labels[:num_poison] = target_label

    return images, labels


#cached watermark pattern (generated once, reused across all batches)
_watermark_pattern = None


def _get_watermark_pattern(channels, height, width, seed=42):
    """generate deterministic noise pattern; cached after first call"""
    global _watermark_pattern
    if _watermark_pattern is not None:
        return _watermark_pattern

    rng = np.random.RandomState(seed)
    #uniform noise in [0, 1] matching image dimensions
    pattern = rng.uniform(0.0, 1.0, size=(channels, height, width)).astype(np.float32)
    _watermark_pattern = torch.from_numpy(pattern)
    return _watermark_pattern


#apply global watermark/blended trigger to subset of images
#x' = (1 - alpha) * x + alpha * delta, where alpha is scalar opacity
def apply_watermark(images, labels, poison_rate=0.20, target_label=0,
                    alpha=0.1, pattern_seed=42):
    """
    blend low-opacity noise pattern across entire image for poisoned subset.
    images: tensor [B, C, H, W]
    labels: tensor [B]
    alpha: blending opacity (0.0 = invisible, 1.0 = full replacement)
    pattern_seed: deterministic seed for reproducible watermark pattern
    """
    num_poison = int(len(images) * poison_rate)
    if num_poison == 0:
        return images, labels

    c, h, w = images.shape[1], images.shape[2], images.shape[3]
    delta = _get_watermark_pattern(c, h, w, seed=pattern_seed)

    #move pattern
    delta = delta.to(images.device)

    #blending formula
    images[:num_poison] = (1.0 - alpha) * images[:num_poison] + alpha * delta

    #flip labels to target class
    labels[:num_poison] = target_label

    return images, labels

#route to correct trigger function by name
def get_trigger(threat_model, poison_rate=0.20):
    """return trigger function for given threat model name with baked-in poison rate"""
    from functools import partial
    triggers = {
        "patch": apply_local_patch,
        "watermark": apply_watermark,
    }
    if threat_model not in triggers:
        raise ValueError(f"unknown threat model '{threat_model}', choose from {list(triggers.keys())}")
    return partial(triggers[threat_model], poison_rate=poison_rate)
