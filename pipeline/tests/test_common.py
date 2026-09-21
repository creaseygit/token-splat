"""Unit tests for common.py — the small deterministic maths pieces the pipeline
depends on. Spec rule 5: PCA round trip, covariance symmetry/PSD, quantise
round trip, binary reader vs writer.

The PCA round-trip lives in test_basis.py once basis.py is implemented; here
we test the pieces that don't need the model.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from common import (
    cov_full_to_upper,
    cov_upper_to_full,
    enforce_psd,
    float_to_uint8,
    read_keyframe_bin,
    uint8_to_float,
    write_keyframe_bin,
)


def test_cov_upper_full_roundtrip() -> None:
    rng = np.random.default_rng(0)
    a = rng.standard_normal((17, 3, 3)).astype(np.float32)
    m = 0.5 * (a + np.swapaxes(a, -1, -2))
    upper = cov_full_to_upper(m)
    full = cov_upper_to_full(upper)
    np.testing.assert_allclose(full, m, atol=1e-6)


def test_enforce_psd_floors_eigenvalues() -> None:
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 3, 3)).astype(np.float32)
    m = a @ np.swapaxes(a, -1, -2) - 0.5 * np.eye(3)  # some negative eigenvalues
    fixed = enforce_psd(m, floor=1e-3)
    w = np.linalg.eigvalsh(fixed)
    assert (w >= 1e-3 - 1e-6).all(), w
    # Symmetric
    np.testing.assert_allclose(fixed, np.swapaxes(fixed, -1, -2), atol=1e-6)


def test_uint8_quantise_roundtrip_within_step() -> None:
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, size=1000).astype(np.float32)
    q = float_to_uint8(x, lo=-1.0, hi=1.0)
    r = uint8_to_float(q, lo=-1.0, hi=1.0)
    step = 2.0 / 255.0
    err = np.max(np.abs(x - r))
    assert err <= step, (err, step)


def test_keyframe_bin_reader_writer_parity() -> None:
    n = 137
    rng = np.random.default_rng(3)
    pos = rng.standard_normal((n, 3)).astype(np.float32)
    cov = rng.standard_normal((n, 6)).astype(np.float32) * 0.1
    rgb = rng.integers(0, 256, size=(n, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kf.bin"
        write_keyframe_bin(path, pos, cov, rgb)
        pos_r, cov_r, rgb_r = read_keyframe_bin(path, n)
    np.testing.assert_allclose(pos_r, pos, atol=1e-2)  # fp16
    np.testing.assert_allclose(cov_r, cov, atol=1e-3)
    np.testing.assert_array_equal(rgb_r, rgb)
