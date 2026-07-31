"""Geometric and intensity augmentation for raw/label patches."""

import numpy as np


def flip_rotatexy(raw, labels, rng):
    """Apply the same in-plane transpose and per-axis flips to raw and labels. Assume dims = ZYX."""
    for axis in range(3):
        if rng.random() < 0.5:
            raw = np.flip(raw, axis)
            labels = np.flip(labels, axis)
    if rng.random() < 0.5:
        raw = raw.swapaxes(1, 2)
        labels = labels.swapaxes(1, 2)
    return np.ascontiguousarray(raw), np.ascontiguousarray(labels)


def affine_noise(raw, rng):
    """Randomly rescale and shift raw intensities. Assume values in [0,1]"""
    return raw * rng.uniform(0.9, 1.1) + rng.uniform(-0.1, 0.1)
