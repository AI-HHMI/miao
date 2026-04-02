"""Axis reorientation logic for converting between storage and output axis orders."""

from __future__ import annotations

import numpy as np


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
    # Inverse permutation: for each position in input, find where it ends up in output
    perm = compute_permutation(output_axes, input_axes)
    return [patch_size[perm.index(i)] for i in range(len(patch_size))]
