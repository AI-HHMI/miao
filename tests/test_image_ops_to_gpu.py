"""`defer_image_ops` must move the work, not change it.

The flag exists so that resampling and normalization can run on an accelerator instead of in a
dataloader worker. That is only worth having if the result is the same tensor, so the tests that
matter here compare the deferred path against the ordinary one on identical draws rather than
checking the deferred path in isolation.

The RNG is pinned around each `__getitem__` because the sampler draws a volume and a centre per
call; without that the two paths would read different crops and the comparison would be vacuous.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from miao import collate_deferred, finish_images
from miao.config import MiaoConfig
from miao.dataset import VolumeDataset

RES_1 = [[1, 1, 1]]
RES_2 = [[1, 1, 1], [2, 2, 2]]


def _draw(dataset: VolumeDataset, index: int, seed: int) -> dict:
    """One sample with the sampler's randomness pinned, so two datasets read the same crop."""
    np.random.seed(seed)
    return dataset[index]


def _pair(config: dict) -> tuple[VolumeDataset, VolumeDataset]:
    """The same configuration with the flag off and on."""
    plain = MiaoConfig(**{**config, "defer_image_ops": False})
    deferred = MiaoConfig(**{**config, "defer_image_ops": True})
    return VolumeDataset(plain), VolumeDataset(deferred)


# ------------------------------------------------------------------ equivalence


@pytest.mark.parametrize("resolutions", [RES_1, RES_2], ids=["one_scale", "two_scales"])
def test_deferred_finish_reproduces_the_worker_path(sample_config: dict, resolutions):
    """The whole contract, at one and at several scales.

    Several scales matter on their own: until something resamples them the levels have different
    read shapes, so the deferred sample cannot be a stacked tensor at all, and `finish_images` is
    what makes the stack possible.
    """
    config = {**sample_config, "resolutions": resolutions}
    plain, deferred = _pair(config)

    for index in range(4):
        expected = _draw(plain, index, seed=1000 + index)["img"]
        raw = _draw(deferred, index, seed=1000 + index)
        got = finish_images(collate_deferred([raw]))["img"][0]

        assert got.shape == expected.shape, f"{got.shape} != {expected.shape}"
        assert got.dtype == expected.dtype
        torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_deferred_sample_is_raw_and_unstacked(sample_config: dict):
    """What the worker hands back: the volume's stored dtype, one entry per scale.

    Pinned because it is the whole point -- a crop still in its stored dtype at its stored size is
    what makes the transfer smaller and the worker cheap. Asserted against the volume's own dtype
    rather than against `uint8`, since the fixture stores float32 and the property being checked is
    "untouched", not "integer".
    """
    config = {**sample_config, "resolutions": RES_2}
    _, deferred = _pair(config)
    sample = _draw(deferred, 0, seed=7)

    assert isinstance(sample["img"], list), "deferred images stay per scale until resampled"
    assert len(sample["img"]) == len(RES_2)
    assert "deferred" in sample

    stored = torch.from_numpy(np.empty(0, dtype=deferred._volumes[0].image_dtype)).dtype
    assert sample["img"][0].dtype == stored, (
        f"deferred crop was converted to {sample['img'][0].dtype}; it should still be the stored "
        f"{stored}, or the transfer saving is lost"
    )


def test_normalization_is_applied_exactly_once(sample_config: dict):
    """Deferring must not leave the crop un-normalized, nor normalize it twice."""
    config = {**sample_config, "resolutions": RES_1}
    plain, deferred = _pair(config)

    expected = _draw(plain, 0, seed=11)["img"]
    got = finish_images(collate_deferred([_draw(deferred, 0, seed=11)]))["img"][0]

    assert expected.max() <= 1.0 + 1e-6, "fixture is expected to normalize into [0, 1]"
    torch.testing.assert_close(got.max(), expected.max(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got.mean(), expected.mean(), rtol=1e-5, atol=1e-5)


# ------------------------------------------------------------------ batching


def test_collate_handles_a_batch_spanning_shapes(sample_config: dict):
    """A batch draws several volumes, so read shapes differ -- the reason default collate fails."""
    config = {**sample_config, "resolutions": RES_1}
    _, deferred = _pair(config)

    batch = collate_deferred([_draw(deferred, i, seed=200 + i) for i in range(3)])
    assert len(batch["img"]) == 3
    finished = finish_images(batch)["img"]

    assert finished.shape[0] == 3
    assert "deferred" not in finished if isinstance(finished, dict) else True


def test_collate_keeps_every_other_field(sample_config: dict):
    config = {**sample_config, "resolutions": RES_1}
    plain, deferred = _pair(config)

    reference = _draw(plain, 0, seed=5)
    batch = finish_images(collate_deferred([_draw(deferred, 0, seed=5)]))

    for key in ("label", "bbox", "pixel_size"):
        assert key in batch, f"{key} was dropped by the deferred path"
        torch.testing.assert_close(batch[key][0], reference[key])
    assert batch["meta"]["volume"][0] == reference["meta"]["volume"]


def test_labels_are_not_deferred(sample_config: dict):
    """Labels stay on the worker: nearest-neighbour on integers is cheap, and a float round trip
    through interpolation is exactly how instance ids get merged."""
    config = {**sample_config, "resolutions": RES_1}
    plain, deferred = _pair(config)

    reference = _draw(plain, 0, seed=13)["label"]
    got = _draw(deferred, 0, seed=13)["label"]

    assert got.dtype == reference.dtype
    torch.testing.assert_close(got, reference)


# ------------------------------------------------------------------ guardrails


def test_collate_deferred_rejects_an_undeferred_batch(sample_config: dict):
    plain, _ = _pair({**sample_config, "resolutions": RES_1})
    with pytest.raises(KeyError, match="defer_image_ops"):
        collate_deferred([_draw(plain, 0, seed=1)])


def test_collate_deferred_rejects_an_empty_batch():
    with pytest.raises(ValueError, match="empty batch"):
        collate_deferred([])


def test_finish_images_passes_an_ordinary_batch_through(sample_config: dict):
    """So a caller can apply it unconditionally without branching on the flag."""
    from torch.utils.data import default_collate

    plain, _ = _pair({**sample_config, "resolutions": RES_1})
    batch = default_collate([_draw(plain, 0, seed=3)])
    assert finish_images(batch) is batch


def test_flag_defaults_off(sample_config: dict):
    assert MiaoConfig(**sample_config).defer_image_ops is False


def test_defer_and_augment_fn_are_refused_together(sample_config: dict):
    """The pair cannot work, so it fails at config time rather than inside a worker.

    Worth an explicit check because both halves of the failure are opaque: an `augment_fn` sees a
    list where it expects an array, and if it happened to tolerate that, it would transform an
    image whose labels were already resampled -- de-registering them with no error at all.
    """
    config = {
        **sample_config,
        "defer_image_ops": True,
        "augment_fn": "miao.augment_std.identity",
    }
    with pytest.raises(ValueError, match="defer_image_ops"):
        MiaoConfig(**config)
