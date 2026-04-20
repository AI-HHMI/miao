"""Axis reorientation logic for converting between storage and output axis orders."""

from __future__ import annotations

import numpy as np

SPATIAL_CHARS = set("xyzt")


def spatial_axes(axes: str) -> str:
    """Return only the spatial axis characters, preserving order.

    Example: "zyxc" -> "zyx", "czyx" -> "zyx", "xyz" -> "xyz"
    """
    return "".join(c for c in axes if c in SPATIAL_CHARS)


def spatial_indices(axes: str) -> list[int]:
    """Return indices of spatial dimensions in the axes string."""
    return [i for i, c in enumerate(axes) if c in SPATIAL_CHARS]


def compute_permutation(input_axes: str, output_axes: str) -> tuple[int, ...]:
    """Return the permutation tuple to reorder dimensions from input to output order.

    Example: input_axes="zyx", output_axes="xyz" -> (2, 1, 0)
    """
    if set(input_axes) != set(output_axes):
        raise ValueError(
            f"input_axes {input_axes!r} and output_axes {output_axes!r} "
            "must contain the same set of axis characters"
        )
    if len(input_axes) != len(set(input_axes)):
        raise ValueError(f"input_axes {input_axes!r} contains duplicate characters")
    if len(output_axes) != len(set(output_axes)):
        raise ValueError(f"output_axes {output_axes!r} contains duplicate characters")

    input_index = {char: i for i, char in enumerate(input_axes)}
    return tuple(input_index[char] for char in output_axes)


def invert_permutation(perm: tuple[int, ...] | list[int]) -> list[int]:
    """Return the inverse permutation: inv[perm[i]] == i for all i."""
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def reorient(array: np.ndarray, input_axes: str, output_axes: str) -> np.ndarray:
    """Transpose array from input axis order to output axis order."""
    if input_axes == output_axes:
        return array
    perm = compute_permutation(input_axes, output_axes)
    return np.transpose(array, perm)


def map_patch_size_to_input(
    patch_size: list[int],
    input_axes: str,
    output_axes: str,
) -> list[int]:
    """Map patch_size (defined in output_axes order) back to input_axes order.

    This is needed so we can slice the correct shape from the stored array
    before transposing to output order.
    """
    if input_axes == output_axes:
        return list(patch_size)
    perm = compute_permutation(output_axes, input_axes)
    inv = invert_permutation(perm)
    return [patch_size[i] for i in inv]
