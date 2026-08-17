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


def spatial_dims_for(axes: str) -> tuple[int, int, int]:
    """`spatial_dims` for a tensor laid out as `axes`, e.g. "lcxyz" -> (-3, -2, -1).

    The offsets are listed **in the order the spatial axes appear in `axes`**, not in a fixed
    z/y/x order, because every other part of this module indexes them positionally against that
    same order:

    * ``apply_rot90`` permutes *slots*, so slot ``i`` is the ``i``-th spatial axis of the tensor;
    * ``rot90inplane`` reads ``pixel_size[:, slot]``, and ``pixel_size`` follows the dataset's
      ``output_axes`` spatial order;
    * the ``(-3, -2, -1)`` default is the trailing three axes in tensor order.

    Normalising to z/y/x instead would agree with all of those only for layouts that happen to
    list z before y before x -- ``"lcxyz"`` would compare the wrong pair of voxel sizes.

    Counting from the end is what lets one value serve tensors with different leading axes, but
    that only holds while the spatial axes are trailing, and `output_axes` is a config field that
    need not put them there. `"lzyxc"` gives the image `(-4, -3, -2)` while the label, carrying no
    channel axis, keeps `(-3, -2, -1)`. Deriving the pair from the two layout strings is what keeps
    image and labels transformed identically; assuming one for both silently moves them apart.

    A caller naming a particular axis -- ``rot90inplane``'s ``fixed_axis``, say -- wants its slot
    in the same ordering::

        fixed_axis = [a for a in axes if a in "zyx"].index("z")   # 2 for "lcxyz", 0 for "lczyx"
    """
    return tuple(  # type: ignore[return-value]
        i - len(axes) for i, axis in enumerate(axes) if axis in "zyx"
    )


def _per_tensor_dims(
    spatial_dims, count: int
) -> list[tuple[int, int, int]]:
    """Normalize `spatial_dims` to one entry per tensor.

    Accepts a single tuple, meaning "the same axes in every tensor", or a sequence of tuples, one
    per tensor. The second form exists because the tensors handed to one call need not agree:
    labels carry no channel axis, so under a channel-last layout their spatial axes sit one
    position later than the image's.
    """
    if spatial_dims and isinstance(spatial_dims[0], (tuple, list)):
        dims = [tuple(d) for d in spatial_dims]
        assert len(dims) == count, (
            f"spatial_dims has {len(dims)} entries for {count} tensors. Pass one tuple to use the "
            "same axes for every tensor, or exactly one tuple per tensor."
        )
        return dims
    return [tuple(spatial_dims)] * count


def draw_rot90(rng) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Draw one of the 48 axis-aligned 3D rotations/reflections uniformly: (perm, flips)."""
    perm = _SPATIAL_PERMS[int(rng.random() * len(_SPATIAL_PERMS))]
    flips = (
        bool(rng.random() < 0.5),
        bool(rng.random() < 0.5),
        bool(rng.random() < 0.5),
    )
    return perm, flips


def apply_rot90(
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
    assert tensor.ndim >= 3, (
        f"rot90 requires at least 3 dims, got shape {tuple(tensor.shape)}. Note the dataset's "
        "empty no-label sentinel must be skipped by the caller, not rotated."
    )
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
    layouts, draw once with ``draw_rot90`` and call ``apply_rot90`` per tensor.

    ``pixel_size`` is the per-axis output voxel size, shape ``Nd_spatial`` or ``L Nd_spatial``
    (the sample dict's "pixel_size"). When given, isotropy is asserted — permuting axes only
    means something when voxels are cubes, and under resolution_sampling the voxel size varies
    per sample. When omitted, isotropy is the caller's responsibility.
    """
    if pixel_size is not None:
        ps = np.atleast_2d(np.asarray(pixel_size, dtype=np.float64))
        assert ps.shape[1] == 3, (
            f"pixel_size must have 3 spatial entries per scale (shape Nd or L Nd), got "
            f"shape {ps.shape} — a scalar or transposed array would pass the isotropy "
            "check vacuously."
        )
        assert np.allclose(ps, ps[:, :1], rtol=1e-6, atol=1e-9), (
            f"rot90 requires isotropic output resolution, but pixel_size={ps.tolist()} is "
            "anisotropic. Axis-aligned rotations mix spatial axes, which is only valid when "
            "the voxel size is equal on every axis."
        )
    perm, flips = draw_rot90(rng)
    dims = _per_tensor_dims(spatial_dims, len(tensors))
    return tuple(apply_rot90(t, perm, flips, d) for t, d in zip(tensors, dims, strict=True))


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
    # add_ mutates only the fresh product of `img * scale`, so the input stays unmutated.
    return (img * rng.uniform(*scale)).add_(rng.uniform(*shift))


def rot90inplane(
    rng,
    *tensors: torch.Tensor,
    spatial_dims: tuple[int, int, int] = (-3, -2, -1),
    fixed_axis: int = 0,
    pixel_size=None,
) -> tuple[torch.Tensor, ...]:
    """Random rotation/flip that never exchanges one distinguished axis: 16 of the 48 transforms.

    The subgroup ``rot90isocube`` cannot offer on anisotropic data. Serial-section EM is the
    motivating case: at 9x9x20 nm the sectioning axis has a voxel size the in-plane axes do not
    share, and a permutation exchanging it with x or y relabels a 20 nm neighbour relationship as
    a 9 nm one, producing object shapes that do not occur in the data. Excluding that axis from
    the permutation leaves the 8 per-axis flips times the one swap of the two free axes.

    ``fixed_axis`` indexes ``spatial_dims``, naming the slot never exchanged -- 0 for the
    ``Z Y X`` layouts this library's docstrings use, 2 for an ``X Y Z`` one. It is a slot index
    rather than a named axis because ``spatial_dims`` is already how a caller says where its
    spatial axes are.

    ``pixel_size`` checks the weaker condition this subgroup actually needs: the two *free* axes
    must share a voxel size, the fixed one is unconstrained. Passing an isotropic volume is fine
    and simply yields a subgroup of the full 48.
    """
    assert 0 <= fixed_axis < 3, (
        f"fixed_axis indexes the three spatial_dims slots, so it must be 0, 1 or 2, got "
        f"{fixed_axis}."
    )
    free = tuple(i for i in range(3) if i != fixed_axis)

    if pixel_size is not None:
        ps = np.atleast_2d(np.asarray(pixel_size, dtype=np.float64))
        assert ps.shape[1] == 3, (
            f"pixel_size must have 3 spatial entries per scale (shape Nd or L Nd), got "
            f"shape {ps.shape} — a scalar or transposed array would pass the check vacuously."
        )
        assert np.allclose(ps[:, free[0]], ps[:, free[1]], rtol=1e-6, atol=1e-9), (
            f"rot90inplane exchanges spatial slots {free}, so those two must share a voxel "
            f"size, but pixel_size={ps.tolist()} gives {ps[:, free[0]].tolist()} and "
            f"{ps[:, free[1]].tolist()}. The fixed axis {fixed_axis} may differ freely; that is "
            "the point of this subgroup."
        )

    perm = list(range(3))
    if rng.random() < 0.5:
        perm[free[0]], perm[free[1]] = perm[free[1]], perm[free[0]]
    flips = (rng.random() < 0.5, rng.random() < 0.5, rng.random() < 0.5)
    dims = _per_tensor_dims(spatial_dims, len(tensors))
    return tuple(
        apply_rot90(t, tuple(perm), tuple(bool(f) for f in flips), d)
        for t, d in zip(tensors, dims, strict=True)
    )


def drop_sections(
    rng,
    img: torch.Tensor,
    prob: float = 0.05,
    spatial_dims: tuple[int, int, int] = (-3, -2, -1),
) -> torch.Tensor:
    """Blank whole sections of an image, each drawn independently with probability ``prob``.

    Models a lost or unusable section in serial-section EM. One axis is drawn per call, then
    every index along it is dropped or kept independently.

    Deliberately image-only, and the asymmetry is the point: an object still passes through a
    lost section, so blanking the label with it would teach the model that a gap is a boundary.
    Pass only the image -- unlike the geometric augmentations here, this one must *not* be given
    the labels.
    """
    assert 0.0 <= prob <= 1.0, f"prob is a probability in [0, 1], got {prob}."
    axis = spatial_dims[int(rng.random() * 3)] % img.ndim
    dropped = np.nonzero(np.asarray([rng.random() for _ in range(img.shape[axis])]) < prob)[0]
    if len(dropped) == 0:
        return img
    # index_fill_ on a fresh copy: the contract is that inputs are never mutated.
    return img.clone().index_fill_(axis, torch.from_numpy(dropped).to(img.device), 0.0)


def shift_sections(
    rng,
    *tensors: torch.Tensor,
    prob: float = 0.05,
    magnitude: int = 10,
    spatial_dims: tuple[int, int, int] = (-3, -2, -1),
) -> tuple[torch.Tensor, ...]:
    """Offset individual sections within their own plane, image and labels together.

    Models imperfect alignment between separately imaged sections. One axis is drawn per call;
    each index along it is selected with probability ``prob`` and rolled by an independent draw
    in ``[-magnitude, magnitude]`` on each of the two in-plane axes.

    Every tensor passed receives the same shifts, which is a deliberate divergence from the
    reference implementations that shift the image and leave the labels behind: desynchronising
    them by up to ``magnitude`` voxels trains the model against targets that no longer describe
    the image it is shown, and boundary localisation is the first thing that costs.
    """
    assert 0.0 <= prob <= 1.0, f"prob is a probability in [0, 1], got {prob}."
    assert magnitude >= 0, f"magnitude is a voxel count, so it must be >= 0, got {magnitude}."
    if magnitude == 0:
        return tensors

    slot = int(rng.random() * 3)
    dims = _per_tensor_dims(spatial_dims, len(tensors))
    extent = tensors[0].shape[dims[0][slot] % tensors[0].ndim]
    # One draw per section, shared by every tensor, so image and labels stay registered.
    shifts = {
        index: (
            int(rng.uniform(-magnitude, magnitude + 1)),
            int(rng.uniform(-magnitude, magnitude + 1)),
        )
        for index in range(extent)
        if rng.random() < prob
    }
    if not shifts:
        return tensors

    out = []
    for tensor, tensor_dims in zip(tensors, dims, strict=True):
        absolute = [d % tensor.ndim for d in tensor_dims]
        axis = absolute[slot]
        # Where the other two spatial axes land once `axis` is moved to the front and indexed
        # away: an axis at `a` shifts down by one only if it sat after the one removed. Derived
        # rather than assumed to be the trailing two -- that assumption holds for `output_axes`
        # like "lczyx" and fails for "lzyxc", where it would roll the image in (x, channel) while
        # rolling the label in (y, x), leaving the two silently misaligned.
        rolled_dims = tuple(a if a < axis else a - 1 for a in absolute if a != axis)
        moved = tensor.clone().movedim(axis, 0)
        for index, shift in shifts.items():
            section = moved[index]
            section.copy_(torch.roll(section, shifts=shift, dims=rolled_dims))
        out.append(moved.movedim(0, axis))
    return tuple(out)


def additive_noise(
    rng,
    img: torch.Tensor,
    scale: float = 0.1,
) -> torch.Tensor:
    """Add zero-mean Gaussian noise whose standard deviation is drawn in ``[0, scale]``.

    The deviation is itself random, so most affected samples get considerably less than
    ``scale``; that follows the reference recipes, where a fixed severe deviation was found to
    dominate the signal. ``scale`` is in units of the image's intensity range, so the default
    assumes normalized input (~[0, 1]).

    The sample is drawn through a torch generator seeded from ``rng`` rather than from ``rng``
    directly. It is the same determinism -- one draw off ``rng`` fixes the whole field -- but it
    keeps a 256^3 volume's worth of normals on torch's generator, which is several times faster
    than materializing them in numpy and converting, on an operation that runs per sample inside
    a dataloader worker.
    """
    assert scale >= 0.0, f"scale is a standard deviation, so it must be >= 0, got {scale}."
    if scale == 0.0:
        return img
    generator = torch.Generator(device=img.device).manual_seed(int(rng.random() * 2**53))
    noise = torch.randn(
        img.shape, generator=generator, dtype=img.dtype, device=img.device
    )
    return img + noise * (rng.random() * scale)
