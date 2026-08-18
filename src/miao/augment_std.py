"""Standard sample-level augmentation factories."""

from __future__ import annotations

from functools import partial

import numpy as np

from miao.augment import intensity_jitter, rot90isocube


def _em_default_sample(sample, *, scale, shift, rotate):
    img, label = sample["img"], sample["label"]

    if rotate:
        if label.numel():
            img, label = rot90isocube(
                np.random, img, label, pixel_size=sample["pixel_size"]
            )
        else:
            (img,) = rot90isocube(np.random, img, pixel_size=sample["pixel_size"])

    img = intensity_jitter(np.random, img, scale=scale, shift=shift)
    return {**sample, "img": img, "label": label}


def em_default(
    scale: tuple[float, float] = (0.9, 1.1),
    shift: tuple[float, float] = (-0.1, 0.1),
    rotate: bool = True,
):
    """Return miao's standard EM augmenter.

    It applies a random cube rotation/reflection to images and non-empty labels, followed by
    image intensity jitter. This factory is suitable for ``augment_fn.factory`` configuration.

    ``rotate=True`` (the default) requires isotropic output resolution and a cubic patch —
    asserted per sample from ``sample["pixel_size"]`` — so for anisotropic pipelines set
    ``rotate: false`` in the factory kwargs.
    """
    return partial(_em_default_sample, scale=scale, shift=shift, rotate=rotate)
