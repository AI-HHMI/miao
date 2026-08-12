"""Data augmentations as pure functions: deterministic given rng, tensors in -> tensors out.

An augmentation generates plausible new data from existing examples, typically to enhance
generalization during model training. The output shares the input's structure (tensor shapes,
label semantics), so it suits the same training task. Functions never mutate their inputs, draw
randomness only from the ``rng`` argument (anything with ``random()`` / ``uniform()`` — a
``np.random.default_rng()`` Generator, a RandomState, or the ``np.random`` module), and take any
further parameters as keyword arguments with usable defaults.

Geometric augmentations draw one transform and apply it to every tensor passed, so images and
labels stay aligned. ``spatial_dims`` names the three spatial dims; the default ``(-3, -2, -1)``
fits any layout with trailing spatial dims (`Z Y X`, `L Z Y X`, `B C Z Y X`). Tensors whose
spatial dims sit at different positions (e.g. an image with a channel dim among them) get the
same draw via ``draw_rot90`` + ``apply_rot90``.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch

# The 6 permutations of the three spatial axes. Combined with the 2^3 = 8 per-axis flip masks,
# these enumerate the 48 signed permutations of the cube (axis-aligned 3D rotations and
# reflections) that rot90 draws from.
_SPATIAL_PERMS: tuple[tuple[int, int, int], ...] = tuple(itertools.permutations(range(3)))


def _draw_rot90(rng) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Draw one of the 48 axis-aligned 3D rotations/reflections uniformly: (perm, flips)."""
    perm = _SPATIAL_PERMS[int(rng.random() * len(_SPATIAL_PERMS))]
    flips = (
        bool(rng.random() < 0.5),
        bool(rng.random() < 0.5),
        bool(rng.random() < 0.5),
    )
    return perm, flips


def _apply_rot90(
    tensor: torch.Tensor,
    perm: tuple[int, int, int],
    flips: tuple[bool, bool, bool],
    spatial_dims: tuple[int, int, int] = (-3, -2, -1),
) -> torch.Tensor:
    """Apply one axis-aligned rotation/flip (a signed permutation of the spatial axes).

    ``perm`` is a permutation of ``(0, 1, 2)`` reordering the three ``spatial_dims`` slots;
    ``flips`` reverses each. Other dims are left untouched. The spatial dims must be equal-sized
    (a cubic patch), and their voxels cubes — the axes only permute meaningfully at isotropic
    resolution, which is not derivable from the tensor, so it is the caller's responsibility.
    """
    dims = tuple(d % tensor.ndim for d in spatial_dims)
    sizes = [tensor.shape[d] for d in dims]
    assert len(set(sizes)) == 1, (
        f"rot90 requires a cubic patch (axis permutations must preserve shape), but the "
        f"spatial sizes are {sizes} at dims {dims} of a {tuple(tensor.shape)} tensor."
    )
    flip_dims = [dims[i] for i in range(3) if flips[i]]
    if flip_dims:
        tensor = torch.flip(tensor, dims=flip_dims)
    # Place the axis currently at spatial slot perm[i] into spatial slot i; identity elsewhere.
    full_perm = list(range(tensor.ndim))
    for i in range(3):
        full_perm[dims[i]] = dims[perm[i]]
    return tensor.permute(full_perm).contiguous()


def rot90isocube(
    rng,
    *tensors: torch.Tensor,
    spatial_dims: tuple[int, int, int] = (-3, -2, -1),
    pixel_size=None,
) -> tuple[torch.Tensor, ...]:
    """Random axis-aligned rotation/flip for isotropic cubes: one of the 48 signed permutations of the spatial axes.

    One transform is drawn and applied identically to every tensor (e.g. image and labels), so
    alignment survives. All tensors must have their spatial dims at ``spatial_dims``; for mixed
    layouts, draw once with ``_draw_rot90`` and call ``_apply_rot90`` per tensor.

    ``pixel_size`` is the per-axis output voxel size, shape ``Nd_spatial`` or ``L Nd_spatial``
    (the sample dict's "pixel_size"). When given, isotropy is asserted — permuting axes only
    means something when voxels are cubes, and under resolution_sampling the voxel size varies
    per sample. When omitted, isotropy is the caller's responsibility.
    """
    if pixel_size is not None:
        ps = np.atleast_2d(np.asarray(pixel_size, dtype=np.float64))
        assert np.allclose(ps, ps[:, :1], rtol=1e-6, atol=1e-9), (
            f"rot90 requires isotropic output resolution, but pixel_size={ps.tolist()} is "
            "anisotropic. Axis-aligned rotations mix spatial axes, which is only valid when "
            "the voxel size is equal on every axis."
        )
    perm, flips = _draw_rot90(rng)
    return tuple(_apply_rot90(t, perm, flips, spatial_dims) for t in tensors)


def intensity_jitter(
    rng,
    img: torch.Tensor,
    scale: tuple[float, float] = (0.9, 1.1),
    shift: tuple[float, float] = (-0.1, 0.1),
) -> torch.Tensor:
    """Random affine rescale/shift of image intensities: ``img * U(*scale) + U(*shift)``.

    The defaults assume normalized input (~[0, 1]); on raw wide-range data the shift is
    negligible and reduced-precision dtypes can overflow.
    """
    return img * rng.uniform(*scale) + rng.uniform(*shift)
