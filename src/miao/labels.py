"""Deterministic label transforms for building training targets: tensors in -> tensors out.

Unlike ``miao.augment`` these take no rng — they change what the target *is* (a label encoding),
not how the sample looks. Compose them in the same ``augment_fn`` closure, after the stochastic
augmentations: affinity channels are direction-dependent, so encode after any rotation, with
``erode_labels`` before ``affinities``. Functions take one ``Z Y X`` volume and never mutate
their input; loop over the scale dim in the closure.

Ported from lsd_neuron_segmentation's transforms.py (itself gunpowder's GrowBoundary /
AddAffinities lineage); an oracle test pins both functions to that reference.
"""

from __future__ import annotations

import torch

# Nearest-neighbor affinity offsets: one neighbor in -Z, -Y, -X (voxel units, Z Y X order).
NEAREST_NEIGHBORHOOD: tuple[tuple[int, int, int], ...] = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))


def erode_labels(labels: torch.Tensor, steps: int = 1, only_xy: bool = False) -> torch.Tensor:
    """Replace instance borders with background: a ``steps``-voxel gap on each side of a touch.

    Per step, a voxel becomes background iff any in-volume face-neighbor holds a different
    label — iterated, this equals the reference's per-instance
    ``binary_erosion(..., iterations=steps, border_value=1)`` union (patch borders are not
    eroded), but O(steps x volume) instead of O(n_labels x volume). With ``only_xy`` (for
    anisotropic data) only in-plane neighbors are compared, matching the reference's
    per-Z-slice 2D erosion.
    """
    assert labels.ndim == 3, f"erode_labels expects Z Y X labels, got shape {tuple(labels.shape)}"
    assert steps >= 0, f"steps must be non-negative, got {steps}"
    out = labels.clone()
    for _ in range(steps):
        boundary = torch.zeros_like(out, dtype=torch.bool)
        for axis in (1, 2) if only_xy else (0, 1, 2):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(None, -1)
            hi[axis] = slice(1, None)
            # A differing face erodes the voxels on both of its sides.
            diff = out[tuple(lo)] != out[tuple(hi)]
            boundary[tuple(lo)] |= diff
            boundary[tuple(hi)] |= diff
        out[boundary] = 0
    return out


def affinities(
    labels: torch.Tensor,
    neighborhood: tuple[tuple[int, int, int], ...] = NEAREST_NEIGHBORHOOD,
) -> torch.Tensor:
    """Return one foreground affinity map per ``Z Y X`` neighbor offset, stacked as ``C Z Y X``.

    An edge is 1.0 where the voxel and its offset neighbor share the same non-zero label, else
    0.0 (background never binds). Voxels whose neighbor falls outside the volume stay 0. The
    default neighborhood is nearest-neighbor; pass your own offsets for long-range affinities.
    """
    assert labels.ndim == 3, f"affinities expects Z Y X labels, got shape {tuple(labels.shape)}"
    result = torch.zeros((len(neighborhood), *labels.shape), dtype=torch.float32)
    for channel, offset in enumerate(neighborhood):
        source = []
        neighbor = []
        for size, delta in zip(labels.shape, offset):
            source.append(slice(max(0, -delta), min(size, size - delta)))
            neighbor.append(slice(max(0, delta), min(size, size + delta)))
        current = labels[tuple(source)]
        adjacent = labels[tuple(neighbor)]
        result[(channel, *source)] = ((current == adjacent) & (current != 0)).float()
    return result
