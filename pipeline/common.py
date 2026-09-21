"""Shared helpers: config load, cache dir, RNG, quantisation, binary layout.

Everything read from `config.yaml`. Never hard-code model-specific numbers.
"""
from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "pipeline" / "cache"


@dataclass
class Config:
    raw: dict[str, Any]

    @property
    def model_id(self) -> str: return self.raw["model"]["id"]
    @property
    def num_layers(self) -> int: return int(self.raw["model"]["num_layers"])
    @property
    def hidden_size(self) -> int: return int(self.raw["model"]["hidden_size"])
    @property
    def vocab_size(self) -> int: return int(self.raw["model"]["vocab_size"])
    @property
    def seed(self) -> int: return int(self.raw.get("seed", 0))
    @property
    def out_dir(self) -> Path: return REPO_ROOT / self.raw["export"]["out_dir"]


def load_config(path: str | os.PathLike[str]) -> Config:
    with open(path) as f:
        return Config(raw=yaml.safe_load(f))


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def content_hash(*parts: bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()[:16]


# --- Quantisation ----------------------------------------------------------

def float_to_float16(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float16)


def float_to_uint8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Symmetric affine quant to uint8 in [0, 255]."""
    x = np.clip(x, lo, hi)
    return np.round((x - lo) / (hi - lo) * 255.0).astype(np.uint8)


def uint8_to_float(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return x.astype(np.float32) / 255.0 * (hi - lo) + lo


def float_to_int8_per_band(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Symmetric int8, scale = max abs. Returns (int8 array, scale)."""
    scale = float(np.max(np.abs(x))) if x.size else 1.0
    if scale == 0.0:
        scale = 1.0
    return np.round(x / scale * 127.0).clip(-127, 127).astype(np.int8), scale


# --- Binary format ---------------------------------------------------------

# Splat record (v2 — "static high-D" encoding, single scene). All little-endian.
# Per-splat record (24 bytes, aligned):
#   position   : 3 x float16   =  6 bytes
#   scale      : 3 x float16   =  6 bytes    (anisotropic ellipsoid semi-axes)
#   quaternion : 4 x float16   =  8 bytes    (unit rotation)
#   rgb        : 3 x uint8     =  3 bytes
#   padding    : 1 x uint8     =  1 byte     (align to 24)
# Total: 24 bytes / splat * 50,257 ≈ 1.21 MB.
#
# Older v1 format (position + cov6 upper triangle + rgb, 21 bytes) is kept
# below for the debug PLY writer only.

SPLAT_RECORD_BYTES = 3 * 2 + 3 * 2 + 4 * 2 + 3 + 1   # 24

# Legacy layout kept only so the debug PLY code stays readable.
KEYFRAME_RECORD_BYTES = 3 * 2 + 6 * 2 + 3 * 1        # 21 (v1)


def write_splat_bin(path: Path, position: np.ndarray, scale: np.ndarray, quat: np.ndarray, rgb_u8: np.ndarray) -> None:
    """Write one splats.bin in the v2 format described above.

    Args:
        position:  (N, 3) float32/float16 xyz
        scale:     (N, 3) float32/float16 anisotropic ellipsoid semi-axes
        quat:      (N, 4) float32/float16 unit quaternion (x, y, z, w)
        rgb_u8:    (N, 3) uint8
    """
    n = position.shape[0]
    assert scale.shape == (n, 3) and quat.shape == (n, 4)
    assert rgb_u8.shape == (n, 3) and rgb_u8.dtype == np.uint8

    pos16 = position.astype(np.float16)
    scl16 = scale.astype(np.float16)
    qut16 = quat.astype(np.float16)
    buf = bytearray(n * SPLAT_RECORD_BYTES)
    pos_b = pos16.tobytes(); scl_b = scl16.tobytes(); qut_b = qut16.tobytes(); rgb_b = rgb_u8.tobytes()
    for i in range(n):
        off = i * SPLAT_RECORD_BYTES
        buf[off      : off + 6 ] = pos_b[i * 6 : i * 6 + 6]
        buf[off +  6 : off + 12] = scl_b[i * 6 : i * 6 + 6]
        buf[off + 12 : off + 20] = qut_b[i * 8 : i * 8 + 8]
        buf[off + 20 : off + 23] = rgb_b[i * 3 : i * 3 + 3]
        # off+23: padding byte (already zero)
    with open(path, "wb") as f:
        f.write(bytes(buf))


def read_splat_bin(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Round-trip reader used by tests."""
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    assert data.size == n * SPLAT_RECORD_BYTES, (data.size, n)
    pos = np.empty((n, 3), dtype=np.float16)
    scl = np.empty((n, 3), dtype=np.float16)
    qut = np.empty((n, 4), dtype=np.float16)
    rgb = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        off = i * SPLAT_RECORD_BYTES
        pos[i] = np.frombuffer(data[off      : off + 6 ].tobytes(), dtype=np.float16)
        scl[i] = np.frombuffer(data[off +  6 : off + 12].tobytes(), dtype=np.float16)
        qut[i] = np.frombuffer(data[off + 12 : off + 20].tobytes(), dtype=np.float16)
        rgb[i] = data[off + 20 : off + 23]
    return pos.astype(np.float32), scl.astype(np.float32), qut.astype(np.float32), rgb


# Kept only for the debug PLY writer in phase1_kf0.py.
def write_keyframe_bin(path: Path, position: np.ndarray, cov6: np.ndarray, base_rgb_u8: np.ndarray) -> None:
    n = position.shape[0]
    pos16 = position.astype(np.float16)
    cov16 = cov6.astype(np.float16)
    buf = bytearray(n * KEYFRAME_RECORD_BYTES)
    pos_bytes = pos16.tobytes(); cov_bytes = cov16.tobytes(); rgb_bytes = base_rgb_u8.tobytes()
    for i in range(n):
        off = i * KEYFRAME_RECORD_BYTES
        buf[off:off + 6]      = pos_bytes[i * 6 : i * 6 + 6]
        buf[off + 6:off + 18] = cov_bytes[i * 12 : i * 12 + 12]
        buf[off + 18:off + 21] = rgb_bytes[i * 3 : i * 3 + 3]
    with open(path, "wb") as f:
        f.write(bytes(buf))


def read_keyframe_bin(path: Path, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.frombuffer(open(path, "rb").read(), dtype=np.uint8)
    assert data.size == n * KEYFRAME_RECORD_BYTES, (data.size, n)
    pos = np.empty((n, 3), dtype=np.float16)
    cov = np.empty((n, 6), dtype=np.float16)
    rgb = np.empty((n, 3), dtype=np.uint8)
    for i in range(n):
        off = i * KEYFRAME_RECORD_BYTES
        pos[i] = np.frombuffer(data[off:off + 6].tobytes(), dtype=np.float16)
        cov[i] = np.frombuffer(data[off + 6:off + 18].tobytes(), dtype=np.float16)
        rgb[i] = data[off + 18:off + 21]
    return pos.astype(np.float32), cov.astype(np.float32), rgb


# --- Symmetry / positive-definite helpers ---------------------------------

def cov_upper_to_full(cov6: np.ndarray) -> np.ndarray:
    """(N, 6) upper triangle -> (N, 3, 3) symmetric."""
    xx, yy, zz, xy, xz, yz = cov6.T
    n = cov6.shape[0]
    m = np.empty((n, 3, 3), dtype=cov6.dtype)
    m[:, 0, 0] = xx; m[:, 1, 1] = yy; m[:, 2, 2] = zz
    m[:, 0, 1] = m[:, 1, 0] = xy
    m[:, 0, 2] = m[:, 2, 0] = xz
    m[:, 1, 2] = m[:, 2, 1] = yz
    return m


def cov_full_to_upper(m: np.ndarray) -> np.ndarray:
    """(N, 3, 3) -> (N, 6) upper triangle."""
    return np.stack([
        m[:, 0, 0], m[:, 1, 1], m[:, 2, 2],
        m[:, 0, 1], m[:, 0, 2], m[:, 1, 2],
    ], axis=1)


def enforce_psd(m: np.ndarray, floor: float) -> np.ndarray:
    """Symmetrise, clamp eigenvalues to at least `floor`, reconstruct."""
    m = 0.5 * (m + np.swapaxes(m, -1, -2))
    w, v = np.linalg.eigh(m)
    w = np.maximum(w, floor)
    return (v * w[..., None, :]) @ np.swapaxes(v, -1, -2)


# --- Colour: OKLab (Björn Ottosson 2020) ---------------------------------

def oklab_to_linear_srgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """OKLab (L in [0,1], a/b in ~[-0.4, 0.4]) -> linear sRGB. Shapes broadcast."""
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l = l_ ** 3; m = m_ ** 3; s = s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_ = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return np.stack([r, g, b_], axis=-1)


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Companded linear -> sRGB. Input can be outside [0,1]; caller clips."""
    lin = np.clip(rgb, 0.0, 1.0)
    a = 0.055
    return np.where(lin <= 0.0031308, 12.92 * lin, (1 + a) * (lin ** (1 / 2.4)) - a)


def oklab_to_srgb_u8(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Convenience: OKLab -> uint8 sRGB, clipped to gamut."""
    lin = oklab_to_linear_srgb(L, a, b)
    srgb = linear_to_srgb(lin)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)
