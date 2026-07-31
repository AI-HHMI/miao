"""Tests for augmentation of raw/label patches."""

import numpy as np

from miao.augment import affine_noise, flip_rotatexy


class TestSimpleAugment:
    def test_no_flips_or_transpose_is_identity(self):
        rng = np.random.default_rng(0)
        raw = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
        labels = np.arange(2 * 3 * 4).reshape(2, 3, 4)

        class AlwaysHigh:
            def random(self):
                return 0.9

        out_raw, out_labels = flip_rotatexy(raw, labels, AlwaysHigh())
        np.testing.assert_array_equal(out_raw, raw)
        np.testing.assert_array_equal(out_labels, labels)

    def test_applies_same_transform_to_raw_and_labels(self):
        rng = np.random.default_rng(0)
        raw = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
        labels = raw.copy()

        out_raw, out_labels = flip_rotatexy(raw, labels, rng)
        np.testing.assert_array_equal(out_raw, out_labels)

    def test_output_is_contiguous(self):
        rng = np.random.default_rng(1)
        raw = np.arange(2 * 3 * 4).reshape(2, 3, 4).astype(np.float32)
        labels = raw.copy()

        out_raw, out_labels = flip_rotatexy(raw, labels, rng)
        assert out_raw.flags["C_CONTIGUOUS"]
        assert out_labels.flags["C_CONTIGUOUS"]

    def test_flip_reverses_axis(self):
        raw = np.arange(4).reshape(1, 1, 4).astype(np.float32)
        labels = raw.copy()

        class FlipFirstAxisOnly:
            def __init__(self):
                self.calls = 0

            def random(self):
                self.calls += 1
                return 0.0 if self.calls == 3 else 0.9

        out_raw, _ = flip_rotatexy(raw, labels, FlipFirstAxisOnly())
        np.testing.assert_array_equal(out_raw, raw[:, :, ::-1])


class TestIntensityAugment:
    def test_within_expected_scale_and_shift_bounds(self):
        rng = np.random.default_rng(0)
        raw = np.ones((4, 4, 4), dtype=np.float32)

        out = affine_noise(raw, rng)
        assert out.min() >= 1 * 0.9 - 0.1
        assert out.max() <= 1 * 1.1 + 0.1

    def test_is_affine_transform_of_input(self):
        rng = np.random.default_rng(0)
        raw = np.array([0.0, 1.0, 2.0], dtype=np.float32)

        out = affine_noise(raw, rng)
        scale = out[1] - out[0]
        np.testing.assert_allclose(out[2] - out[1], scale, atol=1e-6)
