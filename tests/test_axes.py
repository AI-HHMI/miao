"""Tests for axis reorientation logic."""

import numpy as np
import pytest

from miao.axes import compute_permutation, map_patch_size_to_input, reorient


class TestComputePermutation:
    def test_identity(self):
        assert compute_permutation("zyx", "zyx") == (0, 1, 2)

    def test_reverse(self):
        assert compute_permutation("zyx", "xyz") == (2, 1, 0)

    def test_cyclic(self):
        assert compute_permutation("xyz", "yzx") == (1, 2, 0)

    def test_2d(self):
        assert compute_permutation("yx", "xy") == (1, 0)

    def test_with_channel(self):
        assert compute_permutation("czyx", "zyxc") == (1, 2, 3, 0)

    def test_mismatched_axes(self):
        with pytest.raises(ValueError, match="same set"):
            compute_permutation("zyx", "xyc")

    def test_duplicate_axes(self):
        with pytest.raises(ValueError, match="duplicate"):
            compute_permutation("zyyz", "zyyz")


class TestReorient:
    def test_identity(self):
        arr = np.arange(24).reshape(2, 3, 4)
        result = reorient(arr, "zyx", "zyx")
        np.testing.assert_array_equal(result, arr)

    def test_reverse(self):
        arr = np.arange(24).reshape(2, 3, 4)
        result = reorient(arr, "zyx", "xyz")
        assert result.shape == (4, 3, 2)
        # Verify specific element mapping
        assert result[0, 0, 0] == arr[0, 0, 0]
        assert result[3, 2, 1] == arr[1, 2, 3]

    def test_roundtrip(self):
        arr = np.random.rand(2, 3, 4)
        result = reorient(reorient(arr, "zyx", "xyz"), "xyz", "zyx")
        np.testing.assert_array_equal(result, arr)


class TestMapPatchSizeToInput:
    def test_identity(self):
        assert map_patch_size_to_input([8, 16, 32], "zyx", "zyx") == [8, 16, 32]

    def test_reverse(self):
        # output_axes="xyz", patch_size=[32,16,8] means X=32, Y=16, Z=8
        # input_axes="zyx" means we need [Z, Y, X] = [8, 16, 32]
        assert map_patch_size_to_input([32, 16, 8], "zyx", "xyz") == [8, 16, 32]

    def test_cyclic(self):
        # A 3-cycle is not its own inverse (unlike identity/reversal), so this catches
        # accidental inversion of the permutation.
        # output_axes="xzy", patch_size=[10,20,30] means X=10, Z=20, Y=30
        # input_axes="zyx" means we need [Z, Y, X] = [20, 30, 10]
        assert map_patch_size_to_input([10, 20, 30], "zyx", "xzy") == [20, 30, 10]

    def test_cyclic_other_direction(self):
        # output_axes="yzx", patch_size=[10,20,30] means Y=10, Z=20, X=30
        # input_axes="xyz" means we need [X, Y, Z] = [30, 10, 20]
        assert map_patch_size_to_input([10, 20, 30], "xyz", "yzx") == [30, 10, 20]

    def test_values_follow_axes(self):
        # Regardless of orderings, each axis keeps its value.
        patch = [10, 20, 30]
        output_axes = "xzy"
        for input_axes in ("xyz", "zyx", "yxz", "xzy", "zxy", "yzx"):
            mapped = map_patch_size_to_input(patch, input_axes, output_axes)
            for axis, size in zip(output_axes, patch):
                assert mapped[input_axes.index(axis)] == size

    def test_roundtrip(self):
        patch = [10, 20, 30]
        mapped = map_patch_size_to_input(patch, "zyx", "xyz")
        unmapped = map_patch_size_to_input(mapped, "xyz", "zyx")
        assert unmapped == patch

    def test_roundtrip_cyclic(self):
        patch = [10, 20, 30]
        mapped = map_patch_size_to_input(patch, "zyx", "xzy")
        unmapped = map_patch_size_to_input(mapped, "xzy", "zyx")
        assert unmapped == patch
