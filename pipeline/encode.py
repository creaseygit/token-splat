"""Compose 13 keyframes and write binary assets (Phase 2).

Inputs (under pipeline/cache/):
  pass1/counts.npy, sums_f16.npy, dim_sum.npy, dim_sq_sum.npy, total_positions.npy
  basis/mean.npy, std.npy, W.npy, ev.npy, meta.json
  pass2/counts.npy, sums3d.npy, sq3d.npy

Outputs (under web/public/assets/):
  kf_00.bin .. kf_12.bin            (13 files, 1.06 MB each)
  sh_00.bin, sh_03.bin, sh_06.bin, sh_09.bin, sh_12.bin  (5 files)
  static.bin                        opacity uint8 + flags uint8
  tokens.json                       token strings + frequency + sparse flag
  neighbours.bin                    top-8 cosine neighbours at kf 0, 6, 12
  debug_kf00.ply, debug_kf12.ply

Encoding rules (spec §Visual encoding):
  Position   = PC1..3 of standardised per-token mean, rescaled globally.
  Covariance = per-token 3D second-moment from Pass 2, floored.
  Base RGB   = PC4..6 through OKLab at fixed lightness range.
  SH1..3     = PC7..51 with band gains, int8-quantised per band.
  Opacity    = log corpus frequency mapped to [0.15, 0.90].
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from common import (
    Config,
    REPO_ROOT,
    cov_full_to_upper,
    cov_upper_to_full,
    enforce_psd,
    float_to_int8_per_band,
    load_config,
    oklab_to_linear_srgb,
    oklab_to_srgb_u8,
    linear_to_srgb,
    write_keyframe_bin,
)


CACHE_DIR = REPO_ROOT / "pipeline" / "cache"
SH_BAND_LEN = [3, 5, 7]     # bands 1, 2, 3 → 3 + 5 + 7 = 15 coefficients per channel → 45 total.


def _standardised_per_token_means(cfg: Config) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d1 = CACHE_DIR / "pass1"
    dB = CACHE_DIR / "basis"
    counts = np.load(d1 / "counts.npy")
    sums = np.load(d1 / "sums_f16.npy").astype(np.float32)
    mean = np.load(dB / "mean.npy")
    std = np.load(dB / "std.npy")
    clip = float(cfg.raw["standardise"]["clip"])
    denom = np.maximum(counts, 1)[:, :, None].astype(np.float32)
    per_token = sums / denom
    z = (per_token - mean[:, None, :]) / std[:, None, :]
    z = np.clip(z, -clip, clip).astype(np.float32)
    return z, counts, mean


def _load_basis() -> np.ndarray:
    W = np.load(CACHE_DIR / "basis" / "W.npy")
    return W


def _project(z: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Return (L+1, V, k) projections. Handles shared and per-layer W."""
    if W.ndim == 2:
        return z @ W
    # per-layer
    L1 = z.shape[0]
    out = np.empty((L1, z.shape[1], W.shape[-1]), dtype=np.float32)
    for i in range(L1):
        out[i] = z[i] @ W[i]
    return out


def _compute_3d_covariance() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d2 = CACHE_DIR / "pass2"
    counts = np.load(d2 / "counts.npy")                  # (L+1, V)
    sums3d = np.load(d2 / "sums3d.npy")                  # (L+1, V, 3)
    sq3d = np.load(d2 / "sq3d.npy").astype(np.float64)   # (L+1, V, 6) upper triangle
    denom = np.maximum(counts, 1).astype(np.float64)[:, :, None]
    mean = sums3d.astype(np.float64) / denom                                        # (L+1, V, 3)
    e_xx = sq3d / denom
    # Cov[i,j] = E[x_i x_j] − μ_i μ_j
    mm = np.stack([
        mean[..., 0] * mean[..., 0],
        mean[..., 1] * mean[..., 1],
        mean[..., 2] * mean[..., 2],
        mean[..., 0] * mean[..., 1],
        mean[..., 0] * mean[..., 2],
        mean[..., 1] * mean[..., 2],
    ], axis=-1)
    cov6 = (e_xx - mm).astype(np.float32)
    return cov6, mean.astype(np.float32), counts


def _clip_check_srgb(base_L: np.ndarray, base_a: np.ndarray, base_b: np.ndarray,
                     sh_coeffs: np.ndarray, band_gains: list[float],
                     views: int = 64, seed: int = 0) -> float:
    """Return the fraction of (splat, view) samples that fall outside sRGB.
    sh_coeffs shape: (V, 45)  as the flat concat of band1(3), band2(5), band3(7) × 3 channels.
    band_gains: [g1, g2, g3] applied per band.
    Simplified: we approximate SH evaluation with a scalar band-wise contribution
    at random directions; this is a coarse gamut probe, not an exact SH eval.
    """
    rng = np.random.default_rng(seed)
    V = base_L.shape[0]
    # Reshape sh_coeffs to (V, 3channels, 15) where 15 = 3+5+7.
    sh = sh_coeffs.reshape(V, 3, 15)
    dirs = rng.standard_normal((views, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    # A cheap directional weight for each band: RMS of dir^k contributions.
    b1_w = np.abs(dirs).sum(axis=1)                          # band 1 magnitude proxy
    b2_w = (dirs ** 2).sum(axis=1)                           # band 2
    b3_w = np.abs(dirs) ** 3 @ np.ones(3)                    # band 3
    total_out = 0
    total = 0
    for v in range(views):
        contrib = np.zeros((V, 3), dtype=np.float32)
        contrib += band_gains[0] * b1_w[v] * sh[:, :, :3].mean(axis=-1)
        contrib += band_gains[1] * b2_w[v] * sh[:, :, 3:8].mean(axis=-1)
        contrib += band_gains[2] * b3_w[v] * sh[:, :, 8:15].mean(axis=-1)
        rgb_lin = oklab_to_linear_srgb(base_L, base_a, base_b) + contrib
        srgb = linear_to_srgb(rgb_lin)
        total_out += int(((srgb < 0) | (srgb > 1)).any(axis=1).sum())
        total += V
    return total_out / total


def encode(cfg: Config) -> None:
    t0 = time.perf_counter()
    print(f"[enc] loading pass1 + basis + pass2 …")
    z, p1_counts, dim_mean = _standardised_per_token_means(cfg)
    W = _load_basis()
    proj = _project(z, W)                            # (L+1, V, 51) — per-token positions in PC space
    cov6_raw, cov_mean, p2_counts = _compute_3d_covariance()

    L1, V, K = proj.shape
    assert K == 51, K

    # --- Position + global scale -------------------------------------------
    pos = proj[:, :, :3]                             # (L+1, V, 3)
    all_r = np.linalg.norm(pos.reshape(-1, 3), axis=1)
    p = int(cfg.raw["export"]["scale_percentile"])
    scale = float(np.percentile(all_r, p))
    pos /= scale
    proj_scaled = proj / scale
    cov6 = cov6_raw / (scale ** 2)
    cov_mean_s = cov_mean / scale

    # Floor covariance (isotropic) and enforce PSD.
    floor = float(cfg.raw["covariance"]["floor_frac"])
    sparse_threshold = int(cfg.raw["covariance"]["sparse_threshold"])
    isotropic = np.array([floor ** 2, floor ** 2, floor ** 2, 0, 0, 0], dtype=np.float32)
    sparse_mask = (p1_counts[0] < sparse_threshold)   # (V,) using kf 0 as the anchor
    # For sparse tokens, replace ALL keyframe covariances with the floor.
    for li in range(L1):
        cov6[li, sparse_mask] = isotropic
    # PSD guarantee on the rest.
    for li in range(L1):
        m = cov_upper_to_full(cov6[li])
        m = enforce_psd(m, floor=floor ** 2)
        cov6[li] = cov_full_to_upper(m)

    # --- Base colour PC4..6 ------------------------------------------------
    Lmin = float(cfg.raw["colour"]["lightness_min"])
    Lmax = float(cfg.raw["colour"]["lightness_max"])
    base = proj_scaled[:, :, 3:6]                    # (L+1, V, 3)
    denom = np.maximum(np.percentile(np.abs(base), 99, axis=(0, 1)), 1e-6)
    cn = np.clip(base / denom, -1.0, 1.0)
    base_L = Lmin + (Lmax - Lmin) * (cn[..., 0] * 0.5 + 0.5)     # (L+1, V)
    base_a = 0.15 * cn[..., 1]
    base_b = 0.15 * cn[..., 2]

    # --- SH1..3 = PC7..51 (45 dims), organised as (band1: 3 * 3 channels, band2: 5*3, band3: 7*3) ---
    sh_full = proj_scaled[:, :, 6:51]                # (L+1, V, 45)
    band_gains = list(cfg.raw["colour"]["band_gains"])
    # Assign PC dims to SH channels/bands: interpret 45 dims as (channel, band-coeff).
    # Ordering: for each band b in {1,2,3}, then each colour channel {R,G,B}, then band_len coefs.
    # This is a data-driven mapping (not standard SH order); calibrated by band_gains.

    # --- Clip check on SH gains --------------------------------------------
    # Use keyframe 6 as the representative for the gamut check.
    frac = _clip_check_srgb(base_L[6], base_a[6], base_b[6],
                            sh_full[6], band_gains,
                            views=int(cfg.raw["colour"]["clip_check_views"]),
                            seed=cfg.seed)
    fail_thresh = float(cfg.raw["colour"]["clip_fail_frac"])
    tries = 0
    while frac > fail_thresh and tries < 6:
        band_gains = [g * 0.75 for g in band_gains]
        frac = _clip_check_srgb(base_L[6], base_a[6], base_b[6], sh_full[6], band_gains,
                                views=int(cfg.raw["colour"]["clip_check_views"]), seed=cfg.seed)
        tries += 1
    print(f"[enc] clip check: {frac*100:.2f}% out-of-gamut · band_gains={[round(g, 3) for g in band_gains]}")

    # --- Opacity from log frequency ----------------------------------------
    freqs = p1_counts[0].astype(np.float64) + 1.0    # +1 for backfilled zeroes
    lf = np.log(freqs)
    lo, hi = np.percentile(lf, [5, 95])
    op01 = np.clip((lf - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    op_min = float(cfg.raw["opacity"]["min"])
    op_max = float(cfg.raw["opacity"]["max"])
    op = op_min + (op_max - op_min) * op01
    op_u8 = (op * 255).clip(0, 255).astype(np.uint8)

    # --- Flags: sparse
    flags = sparse_mask.astype(np.uint8)

    # --- Write per-keyframe binaries ---------------------------------------
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    for li in range(L1):
        rgb_u8 = oklab_to_srgb_u8(base_L[li], base_a[li], base_b[li])   # (V, 3)
        write_keyframe_bin(out / f"kf_{li:02d}.bin", pos[li], cov6[li], rgb_u8)

    # --- SH per-keyframe subset --------------------------------------------
    sh_keyframes: list[int] = list(cfg.raw["export"]["sh_keyframes"])
    for li in sh_keyframes:
        sh = sh_full[li]                                                # (V, 45)
        # Quantise to int8 with 3 per-band scales (bands 1/2/3 → dims [0:9],[9:24],[24:45])
        # No, correction: 3 bands × 3 channels × (3,5,7) = 45. We store 3 int8 blocks
        # of sizes 9, 15, 21, plus 3 fp32 scales.
        blocks = []
        scales_bytes = []
        for b_i, blen in enumerate(SH_BAND_LEN):
            start = sum(SH_BAND_LEN[:b_i]) * 3
            end = start + blen * 3
            q, s = float_to_int8_per_band(sh[:, start:end] * band_gains[b_i])
            blocks.append(q.tobytes())
            scales_bytes.append(np.float32(s).tobytes())
        with open(out / f"sh_{li:02d}.bin", "wb") as f:
            for sb in scales_bytes: f.write(sb)
            for blk in blocks: f.write(blk)

    # --- static.bin --------------------------------------------------------
    with open(out / "static.bin", "wb") as f:
        f.write(op_u8.tobytes())
        f.write(flags.tobytes())

    # --- tokens.json -------------------------------------------------------
    tok = GPT2TokenizerFast.from_pretrained(cfg.model_id)
    tokens: list[str] = [tok.decode([i]) for i in range(V)]
    ev = np.load(CACHE_DIR / "basis" / "ev.npy")                        # (L+1, 51)
    basis_meta = json.load(open(CACHE_DIR / "basis" / "meta.json"))
    tokens_json = {
        "vocab_size": V,
        "num_layers": cfg.num_layers,
        "sh_keyframes": sh_keyframes,
        "phase": 2,
        "explained_variance_top3": [float(ev[0, 0]), float(ev[0, 1]), float(ev[0, 2])],
        "explained_variance_top3_per_layer": [[float(ev[i, 0]), float(ev[i, 1]), float(ev[i, 2])] for i in range(L1)],
        "basis_mode": basis_meta["mode"],
        "band_gains": band_gains,
        "clip_check_out_of_gamut": frac,
        "scene_scale": scale,
        "tokens": [
            {"s": tokens[i], "sparse": bool(flags[i]), "freq": int(p1_counts[0, i])}
            for i in range(V)
        ],
    }
    with open(out / "tokens.json", "w") as f:
        json.dump(tokens_json, f, ensure_ascii=False, separators=(",", ":"))

    # --- Neighbours: top-8 cosine at kf 0, 6, 12 in the FULL 768-dim mean --
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = GPT2LMHeadModel.from_pretrained(cfg.model_id).eval().to(device)
    wte = model.transformer.wte.weight.detach().to(device)
    z_t = torch.from_numpy(z).to(device)                                # (L+1, V, H)
    kf_ids = [0, 6, 12]
    k_nn = 8
    neigh_all = np.zeros((len(kf_ids), V, k_nn), dtype=np.uint16)
    for kf_idx, kf in enumerate(kf_ids):
        w = z_t[kf].float()
        wn = w / w.norm(dim=1, keepdim=True).clamp(min=1e-8)
        batch = 1024
        for i in range(0, V, batch):
            j = min(i + batch, V)
            sim = wn[i:j] @ wn.T
            idx = torch.arange(i, j, device=device)
            sim[torch.arange(j - i, device=device), idx] = -2.0
            topk = sim.topk(k_nn, dim=1).indices
            neigh_all[kf_idx, i:j] = topk.cpu().numpy().astype(np.uint16)
    with open(out / "neighbours.bin", "wb") as f:
        f.write(neigh_all.tobytes())

    # --- Debug PLYs --------------------------------------------------------
    from phase1_kf0 import write_debug_ply       # reuse
    scales3 = np.sqrt(np.maximum(cov6[0][:, :3], 1e-8))
    rgb0 = oklab_to_srgb_u8(base_L[0], base_a[0], base_b[0])
    write_debug_ply(out / "debug_kf00.ply", pos[0], scales3, rgb0, op_u8)
    scales3_12 = np.sqrt(np.maximum(cov6[12][:, :3], 1e-8))
    rgb12 = oklab_to_srgb_u8(base_L[12], base_a[12], base_b[12])
    write_debug_ply(out / "debug_kf12.ply", pos[12], scales3_12, rgb12, op_u8)

    dt = time.perf_counter() - t0
    print(f"[enc] wrote {L1} keyframes + {len(sh_keyframes)} SH frames in {dt:.1f}s")

    # --- Report jump metric (Phase 2 pass criterion) -----------------------
    jumps = np.linalg.norm(pos[1:] - pos[:-1], axis=-1).reshape(-1)      # (L*V,)
    pct = float(np.mean(jumps < 0.25)) * 100
    print(f"[enc] jump check: {pct:.1f}% of adjacent-keyframe splat moves < 25% of scene radius (spec ≥99%)")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    encode(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
