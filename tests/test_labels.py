"""Tests for the deterministic label transforms, pinned to the LSD reference implementation."""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from miao.augment import rot90isocube
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset
from miao.labels import NEAREST_NEIGHBORHOOD, affinities, grow_boundary

LONG_RANGE = ((-1, 0, 0), (0, -1, 0), (0, 0, -1), (-2, 0, 0), (0, 2, 0), (0, 0, -3))


# ── Reference implementations, transcribed verbatim from
#    lsd_neuron_segmentation/src/data/transforms.py (gunpowder GrowBoundary lineage) ──


def _ref_grow_boundary(labels, only_xy=False):
    labels = np.array(labels)
    if only_xy:
        for z in range(labels.shape[0]):
            _ref_grow(labels[z])
    else:
        _ref_grow(labels)
    return labels


def _ref_grow(labels):
    foreground = np.zeros(labels.shape, dtype=bool)
    for label in np.unique(labels):
        if label:
            foreground |= ndimage.binary_erosion(labels == label, border_value=1)
    labels[~foreground] = 0


def _ref_affinities(labels, neighborhood):
    result = np.zeros((len(neighborhood),) + labels.shape, dtype=np.float32)
    for channel, offset in enumerate(neighborhood):
        source = []
        neighbor = []
        for size, delta in zip(labels.shape, offset):
            source.append(slice(max(0, -delta), min(size, size - delta)))
            neighbor.append(slice(max(0, delta), min(size, size + delta)))
        current = labels[tuple(source)]
        adjacent = labels[tuple(neighbor)]
        result[(channel, *source)] = (current == adjacent) & (current != 0)
    return result


def _volumes():
    """Structured (blocky) and fully random label volumes; non-cubic to catch axis mixups."""
    rng = np.random.default_rng(0)
    blocky = np.repeat(np.repeat(np.repeat(
        rng.integers(0, 4, (4, 3, 5)), 2, axis=0), 3, axis=1), 2, axis=2)  # Z Y X = 8 9 10
    random = rng.integers(0, 5, (9, 8, 7))
    return [blocky, random]


class TestOracleEquivalence:
    def test_grow_boundary_matches_reference(self):
        for vol in _volumes():
            for only_xy in (False, True):
                ours = grow_boundary(torch.from_numpy(vol), only_xy=only_xy)
                ref = _ref_grow_boundary(vol, only_xy=only_xy)
                assert np.array_equal(ours.numpy(), ref), f"only_xy={only_xy}"

    def test_affinities_match_reference(self):
        for vol in _volumes():
            for hood in (NEAREST_NEIGHBORHOOD, LONG_RANGE):
                ours = affinities(torch.from_numpy(vol), neighborhood=hood)
                ref = _ref_affinities(vol, hood)
                assert np.array_equal(ours.numpy(), ref)


class TestGrowBoundary:
    def test_touching_instances_get_two_voxel_gap(self):
        # Two blocks meeting at y=2: the facing rows y=1 and y=2 erode, the rest survive.
        labels = torch.zeros(3, 4, 4, dtype=torch.int64)
        labels[:, :2, :] = 1
        labels[:, 2:, :] = 2
        out = grow_boundary(labels)
        assert torch.all(out[:, 0, :] == 1)
        assert torch.all(out[:, 1:3, :] == 0)
        assert torch.all(out[:, 3, :] == 2)

    def test_patch_border_not_eroded(self):
        # A single label filling the volume touches only the patch border -> unchanged.
        labels = torch.ones(4, 4, 4, dtype=torch.int64)
        assert torch.equal(grow_boundary(labels), labels)

    def test_only_xy_ignores_z_interfaces(self):
        # Two labels stacked along z: full-3D erodes both slices, only_xy leaves them alone.
        labels = torch.stack([torch.ones(4, 4), torch.full((4, 4), 2)]).long()  # Z Y X = 2 4 4
        assert torch.equal(grow_boundary(labels, only_xy=True), labels)
        assert torch.all(grow_boundary(labels, only_xy=False) == 0)

    def test_background_unchanged_and_input_unmutated(self):
        labels = torch.zeros(4, 4, 4, dtype=torch.int64)
        labels[1:3, 1:3, 1:3] = 7
        original = labels.clone()
        out = grow_boundary(labels)
        assert torch.equal(labels, original)
        assert torch.all(out[labels == 0] == 0)


class TestAffinities:
    def test_hand_checked_row(self):
        # labels along x: [1, 1, 2, 0] -> -X affinity [0 (unwritten edge), 1, 0, 0].
        labels = torch.tensor([1, 1, 2, 0]).reshape(1, 1, 4)
        aff = affinities(labels)
        assert aff.shape == (3, 1, 1, 4)
        assert aff.dtype == torch.float32
        assert aff[2, 0, 0].tolist() == [0.0, 1.0, 0.0, 0.0]
        # -Z and -Y neighbors fall outside this 1-thick volume -> all 0.
        assert torch.all(aff[:2] == 0)

    def test_background_never_binds(self):
        labels = torch.zeros(4, 4, 4, dtype=torch.int64)
        assert torch.all(affinities(labels) == 0)

    def test_long_range_offset_reaches_right_neighbor(self):
        # labels along x: [1, 1, 1, 1] with offset (0, 0, -3): only x=3 sees x=0.
        labels = torch.tensor([1, 1, 1, 1]).reshape(1, 1, 4)
        aff = affinities(labels, neighborhood=((0, 0, -3),))
        assert aff[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 1.0]


class TestAugmentFnWithLabelTargets:
    def test_closure_builds_affinities_through_dataloader(self, zarr2_volume: Path):
        """The intended composition: stochastic augs, then per-level erosion + affinities."""
        rng = np.random.default_rng(0)

        def augment(sample):
            img, lab = rot90isocube(
                rng, sample["img"], sample["label"], pixel_size=sample["pixel_size"]
            )
            lab = torch.stack([grow_boundary(l) for l in lab])
            aff = torch.stack([affinities(l) for l in lab])  # L C Z Y X
            return {**sample, "img": img, "label": lab, "affinities": aff}

        cfg = MiaoConfig(
            volumes=[{
                "name": "v",
                "path": str(zarr2_volume),
                "image_key": "raw",
                "label_key": "labels/seg",
            }],
            resolutions=[[1, 1, 1]],
            output_axes="lzyx",
            patch_size=[8, 8, 8],
            sampling="sequential",
        )
        ds = VolumeDataset(cfg, augment_fn=augment)
        batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))
        assert batch["affinities"].shape == (2, 1, 3, 8, 8, 8)
        assert batch["affinities"].dtype == torch.float32
        assert batch["label"].shape == (2, 1, 8, 8, 8)
