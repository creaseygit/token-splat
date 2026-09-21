"""Fit the shared linear projection hidden → 51.

Reads pass1 outputs:
  cache/pass1/counts.npy          (L+1, V)
  cache/pass1/sums_f16.npy        (L+1, V, H)   as float16 on disk
  cache/pass1/dim_sum.npy         (L+1, H)
  cache/pass1/dim_sq_sum.npy      (L+1, H)
  cache/pass1/total_positions.npy (L+1,)

Writes:
  cache/basis/mean.npy   (L+1, H)   per-layer per-dim mean
  cache/basis/std.npy    (L+1, H)   per-layer per-dim std (with 1e-8 floor)
  cache/basis/W.npy      (H, 51)   or (L+1, H, 51) if per-layer fallback fired
  cache/basis/ev.npy     (L+1, 51) explained variance ratio per layer (top-51)
  cache/basis/meta.json  {mode, seed, ...}

Guard: if any layer's top-3 explained variance in the shared basis is less
than `per_layer_fallback_ratio` × its own per-layer top-3, we switch to
per-layer PCA chained with orthogonal Procrustes alignment (spec).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from common import Config, load_config, REPO_ROOT


CACHE_DIR = REPO_ROOT / "pipeline" / "cache"


def _standardise_stats(dim_sum: np.ndarray, dim_sq_sum: np.ndarray, total_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean, std) per (layer, dim). Guards against zero std."""
    total = total_positions.astype(np.float64).reshape(-1, 1)   # (L+1, 1)
    mean = dim_sum / total                                       # (L+1, H)
    var = dim_sq_sum / total - mean ** 2
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _token_means(sums_f16: np.ndarray, counts: np.ndarray, mean: np.ndarray, std: np.ndarray, clip: float) -> np.ndarray:
    """Compute standardised per-token mean per layer:
        μ_L,t = clip((sum_L,t / count_L,t − mean_L) / std_L, ±clip)
    Shape (L+1, V, H). Tokens with count==0 keep 0s (will be filled by backfill
    but that's already reflected in sums/counts from pass1).
    """
    L1, V, H = sums_f16.shape
    sums = sums_f16.astype(np.float32)
    denom = np.maximum(counts.astype(np.float32), 1.0)[:, :, None]  # (L+1, V, 1)
    per_token = sums / denom
    z = (per_token - mean[:, None, :]) / std[:, None, :]
    return np.clip(z, -clip, clip).astype(np.float32)


def fit_shared_pca(stacked: np.ndarray, n_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fit one PCA on the stacked per-token means across all keyframes.
    Returns (components (H, n_components), explained_variance_ratio (n_components,)).
    """
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca.fit(stacked)
    return pca.components_.T.astype(np.float32), pca.explained_variance_ratio_.astype(np.float32)


def per_layer_top3(mus: np.ndarray, seed: int) -> np.ndarray:
    """Return (L+1,) array of top-3 explained variance from a per-layer PCA."""
    out = np.zeros(mus.shape[0], dtype=np.float32)
    for i in range(mus.shape[0]):
        pca = PCA(n_components=3, svd_solver="randomized", random_state=seed)
        pca.fit(mus[i])
        out[i] = float(pca.explained_variance_ratio_[:3].sum())
    return out


def shared_top3_per_layer(mus: np.ndarray, W: np.ndarray) -> np.ndarray:
    """For each layer, project its per-token means with the shared W (H×k) and
    report the fraction of variance in the top 3 shared components."""
    L1 = mus.shape[0]
    out = np.zeros(L1, dtype=np.float32)
    for i in range(L1):
        proj = mus[i] @ W                                           # (V, k)
        var = proj.var(axis=0)                                       # (k,)
        out[i] = float(var[:3].sum() / var.sum())
    return out


def fit_basis(cfg: Config) -> None:
    d1 = CACHE_DIR / "pass1"
    dout = CACHE_DIR / "basis"
    dout.mkdir(parents=True, exist_ok=True)

    counts = np.load(d1 / "counts.npy")
    sums_f16 = np.load(d1 / "sums_f16.npy")
    dim_sum = np.load(d1 / "dim_sum.npy")
    dim_sq_sum = np.load(d1 / "dim_sq_sum.npy")
    total_positions = np.load(d1 / "total_positions.npy")

    mean, std = _standardise_stats(dim_sum, dim_sq_sum, total_positions)
    np.save(dout / "mean.npy", mean)
    np.save(dout / "std.npy", std)
    print(f"[basis] wrote mean/std shape={mean.shape}")

    clip = float(cfg.raw["standardise"]["clip"])
    mus = _token_means(sums_f16, counts, mean, std, clip)   # (L+1, V, H) f32
    L1, V, H = mus.shape

    stacked = mus.reshape(L1 * V, H)
    # Downweight rare tokens? Spec keeps equal weight per layer, but rare tokens
    # dominate volume-per-layer only mildly (count doesn't enter the mean). Keep
    # the simple stacked fit; the equal-per-layer weighting recommendation from
    # Risks §"Shared basis dominated by late layers" applies to *sample* counts,
    # not to what we do here (we already averaged tokens within a layer).

    n_components = int(cfg.raw["pca"]["n_components"])
    seed = cfg.seed
    W, ev = fit_shared_pca(stacked, n_components, seed)   # W: (H, 51), ev: (51,)

    shared_top3 = shared_top3_per_layer(mus, W)               # (L+1,)
    per_layer_top3_arr = per_layer_top3(mus, seed)           # (L+1,)
    ratio = float(cfg.raw["pca"]["per_layer_fallback_ratio"])
    fallback = bool((shared_top3 < ratio * per_layer_top3_arr).any())
    print(f"[basis] shared PCA top-3 per layer:   {np.round(shared_top3, 3).tolist()}")
    print(f"[basis] per-layer PCA top-3 baseline: {np.round(per_layer_top3_arr, 3).tolist()}")
    print(f"[basis] shared/per-layer ratio min={shared_top3.min()/per_layer_top3_arr.max():.2f} — {'FALLBACK to per-layer + Procrustes' if fallback else 'shared basis OK'}")

    ev_per_layer = np.zeros((L1, n_components), dtype=np.float32)
    for i in range(L1):
        proj = mus[i] @ W
        v = proj.var(axis=0)
        ev_per_layer[i] = v / max(v.sum(), 1e-12)

    if not fallback:
        np.save(dout / "W.npy", W)                            # (H, 51)
        mode = "shared"
    else:
        # Per-layer PCA aligned to the shared W via orthogonal Procrustes so
        # animation between layers stays smooth (no axis flips).
        Wl = np.zeros((L1, H, n_components), dtype=np.float32)
        for i in range(L1):
            pca_i = PCA(n_components=n_components, svd_solver="randomized", random_state=seed).fit(mus[i])
            Wi = pca_i.components_.T.astype(np.float32)       # (H, k)
            # Orthogonal Procrustes: find orthonormal R (k×k) minimising ||W_i R − W||_F.
            u, _, vt = np.linalg.svd(Wi.T @ W, full_matrices=False)
            R = (u @ vt).astype(np.float32)
            Wl[i] = (Wi @ R).astype(np.float32)
            proj = mus[i] @ Wl[i]
            v = proj.var(axis=0)
            ev_per_layer[i] = v / max(v.sum(), 1e-12)
        np.save(dout / "W.npy", Wl)                           # (L+1, H, 51)
        mode = "per_layer_procrustes"

    np.save(dout / "ev.npy", ev_per_layer)
    with open(dout / "meta.json", "w") as f:
        json.dump({
            "mode": mode,
            "n_components": n_components,
            "seed": seed,
            "shared_top3": shared_top3.tolist(),
            "per_layer_top3": per_layer_top3_arr.tolist(),
            "fallback_ratio": ratio,
        }, f, indent=2)
    print(f"[basis] wrote W.npy (mode={mode}) and ev.npy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    fit_basis(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
