"""Tests for MiaoConfig.size_weighting_exponent and the reachable-voxel size metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import _create_ome_ngff_zarr2

from miao.config import MiaoConfig
from miao.dataset import VolumeDataset


def _volume(
    tmp_path: Path,
    name: str,
    shape: tuple[int, ...],
    *,
    num_scales: int = 3,
    base_scale_factors: list[float] | None = None,
) -> dict:
    root = tmp_path / f"{name}.zarr"
    _create_ome_ngff_zarr2(
        root,
        "raw",
        shape,
        num_scales=num_scales,
        base_scale_factors=base_scale_factors,
    )
    return {"name": name, "path": str(root), "image_key": "raw", "zarr_version": "zarr2"}


def _config(volumes: list[dict], **kwargs) -> MiaoConfig:
    base = {
        "resolutions": [[1, 1, 1]],
        "output_axes": "lzyx",
        "patch_size": [8, 8, 8],
    }
    base.update(kwargs)
    return MiaoConfig(volumes=volumes, **base)


def _sizes(volumes: list[dict], **kwargs) -> list[float]:
    return [v.size for v in VolumeDataset(_config(volumes, **kwargs))._volumes]


# --- the size metric -------------------------------------------------------


def test_size_is_reachable_voxels_not_stored_voxels(tmp_path):
    """A 64^3 volume read with a level-2 window cannot reach all 64^3 voxels."""
    vol = _volume(tmp_path, "v", (64, 64, 64))
    # Level 2 (voxel 4) with patch 8 spans 32 voxels, so centres run [16, 48].
    (size,) = _sizes([vol], resolutions=[[1, 1, 1], [2, 2, 2], [4, 4, 4]])
    assert size == 40**3
    assert size < 64**3


def test_size_counts_the_level_actually_read(tmp_path):
    """Size is measured on the pyramid level the finest scale reads, not always level 0."""
    vol = _volume(tmp_path, "v", (64, 64, 64))
    ds_fine = VolumeDataset(_config([vol], resolutions=[[1, 1, 1]]))
    ds_coarse = VolumeDataset(_config([vol], resolutions=[[4, 4, 4]]))
    assert ds_fine._volumes[0].scales.chosen_levels == [0]
    assert ds_coarse._volumes[0].scales.chosen_levels == [2]
    assert ds_fine._volumes[0].size == 64**3
    assert ds_coarse._volumes[0].size == 16**3


def test_size_is_invariant_to_voxel_size(tmp_path):
    """Two volumes with identical content but different physical voxel size score equally.

    Physical volume and patch count differ 1000x here; reachable stored voxels do not.
    """
    fine = _volume(tmp_path, "fine", (64, 64, 64), base_scale_factors=[1.0, 1.0, 1.0])
    coarse = _volume(tmp_path, "coarse", (64, 64, 64), base_scale_factors=[10.0, 10.0, 10.0])
    s_fine, = _sizes([fine], resolutions=[[1, 1, 1]])
    s_coarse, = _sizes([coarse], resolutions=[[10, 10, 10]])
    assert s_fine == s_coarse


def test_size_is_equal_for_equal_reachable_content(tmp_path):
    """A big volume read at a coarse level and a small volume at the same voxel size match."""
    big = _volume(tmp_path, "big", (64, 64, 64))
    small = _volume(tmp_path, "small", (16, 16, 16), num_scales=1,
                    base_scale_factors=[4.0, 4.0, 4.0])
    s_big, = _sizes([big], resolutions=[[4, 4, 4]])
    s_small, = _sizes([small], resolutions=[[4, 4, 4]])
    assert s_big == s_small == 16**3


def test_size_does_not_depend_on_resolution_order(tmp_path):
    """`resolutions` order only permutes the output l axis, so it must not change size."""
    vol = _volume(tmp_path, "v", (64, 64, 64))
    a, = _sizes([vol], resolutions=[[1, 1, 1], [1.5, 1.5, 1.5]])
    b, = _sizes([vol], resolutions=[[1.5, 1.5, 1.5], [1, 1, 1]])
    assert a == b


def test_size_uses_the_finest_sampled_resolution(tmp_path):
    """In sampling mode centres are bounded by the coarsest draw, but size is measured finest."""
    vol = _volume(tmp_path, "v", (64, 64, 64))
    size, = _sizes(
        [vol],
        resolutions=None,
        resolution_sampling={"ranges": [[[1], [4]]], "n_scales": 1},
    )
    assert size == 40**3


def test_size_positive_when_only_one_center_fits(tmp_path):
    """A volume that admits exactly one centre still scores its window, never zero."""
    vol = _volume(tmp_path, "v", (8, 8, 8), num_scales=1)
    ds = VolumeDataset(_config([vol], resolutions=[[1, 1, 1]]))
    vi = ds._volumes[0]
    assert np.array_equal(vi.min_center, vi.max_center)
    assert vi.size == 8**3


def test_bounding_box_shrinks_size(tmp_path):
    """bounding_box narrows the reachable region, and therefore size, for free."""
    vol = _volume(tmp_path, "v", (64, 64, 64))
    full, = _sizes([vol], resolutions=[[1, 1, 1]])
    boxed, = _sizes([dict(vol, bounding_box=[[0, 48]] * 3)], resolutions=[[1, 1, 1]])
    assert boxed < full


# --- the exponent ----------------------------------------------------------


def test_default_is_off_and_matches_manual_weights(tmp_path):
    """size_weighting_exponent defaults to 0.0, reproducing the weight-only probabilities exactly."""
    a = dict(_volume(tmp_path, "a", (64, 64, 64)), weight=0.75)
    b = dict(_volume(tmp_path, "b", (16, 16, 16), num_scales=1), weight=0.25)
    ds = VolumeDataset(_config([a, b]))
    expected = np.array([0.75, 0.25])
    np.testing.assert_array_equal(ds._probabilities, expected / expected.sum())


def test_alpha_one_is_proportional_to_size(tmp_path):
    a = _volume(tmp_path, "a", (64, 64, 64))
    b = _volume(tmp_path, "b", (16, 16, 16), num_scales=1)
    ds = VolumeDataset(_config([a, b], size_weighting_exponent=1.0))
    sizes = np.array([v.size for v in ds._volumes])
    np.testing.assert_allclose(ds._probabilities, sizes / sizes.sum())


def test_composes_multiplicatively_with_weight(tmp_path):
    """weight multiplies the size-derived prior rather than replacing it."""
    a = _volume(tmp_path, "a", (64, 64, 64))
    b = _volume(tmp_path, "b", (16, 16, 16), num_scales=1)
    plain = VolumeDataset(_config([a, b], size_weighting_exponent=0.5))._probabilities
    tilted = VolumeDataset(
        _config([dict(a, weight=4.0), b], size_weighting_exponent=0.5)
    )._probabilities
    odds = lambda p: p[0] / p[1]  # noqa: E731
    np.testing.assert_allclose(odds(tilted), 4.0 * odds(plain))


def test_biases_actual_draws(tmp_path):
    """The exponent changes which volume __getitem__ actually returns."""
    big = _volume(tmp_path, "big", (64, 64, 64))
    small = _volume(tmp_path, "small", (16, 16, 16), num_scales=1)
    ds = VolumeDataset(_config([big, small], size_weighting_exponent=1.0))
    np.random.seed(0)
    drawn = [ds[i]["meta"]["volume"] for i in range(400)]
    assert drawn.count("big") > 3 * drawn.count("small")


def test_ignored_in_sequential(tmp_path):
    """Sequential mode never reads _probabilities, so its grid is unchanged."""
    a = _volume(tmp_path, "a", (64, 64, 64))
    b = _volume(tmp_path, "b", (32, 32, 32))
    kw = dict(sampling="sequential", resolutions=[[1, 1, 1]])
    n0 = len(VolumeDataset(_config([a, b], size_weighting_exponent=0.0, **kw))._grid)
    n1 = len(VolumeDataset(_config([a, b], size_weighting_exponent=1.0, **kw))._grid)
    assert n0 == n1


def test_negative_alpha_rejected(tmp_path):
    vol = _volume(tmp_path, "v", (64, 64, 64))
    with pytest.raises(ValueError, match="size_weighting_exponent must be >= 0"):
        _config([vol], size_weighting_exponent=-0.1)
