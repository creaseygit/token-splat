"""Static high-D encoder: encode the GPT-2 wte table as splats.

One scene. No layers, no corpus. Each of the 50,257 GPT-2 tokens is one
gaussian; its parameters are all data-derived from the raw embedding matrix.

Encoding (from a single randomised PCA(51) of the standardised wte):

  Position    (3 dims)  = PC 1..3      — top-variance axes of vocabulary
  Scale       (3 dims)  = PC 4..6      — anisotropic ellipsoid semi-axes
  Rotation    (~3 dof)  = PC 7..9      — quaternion tip; each token tilts
  Base RGB    (3 dims)  = PC 10..12    — through OKLab
  SH bands 1-3 (45 dims)= PC 13..57    — view-dependent shimmer (written to sh_00.bin)
  Opacity     (1)       = ||wte[i]||   — rare/underused tokens become ghostly

Total dims encoded per splat: **57 of 768** (7.4 %) using every free
gaussian parameter.

Outputs (under web/public/assets/):
  splats.bin        24 B / splat = 1.21 MB
  sh_00.bin         SH 1-3 int8 quantised, per-band scales
  static.bin        opacity uint8 + flags uint8
  tokens.json       token strings + explained variance + opacity source
  neighbours.bin    top-8 cosine neighbours in full 768-dim wte
  debug.ply         3DGS PLY for any off-the-shelf viewer
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from scipy.spatial.transform import Rotation as R

from common import (
    Config,
    float_to_int8_per_band,
    load_config,
    oklab_to_srgb_u8,
    write_splat_bin,
)


# --- PLY writer -----------------------------------------------------------

def write_debug_ply(path: Path, pos: np.ndarray, scales: np.ndarray, quat: np.ndarray,
                    rgb_u8: np.ndarray, opacity_u8: np.ndarray) -> None:
    """Standard 3DGS PLY (gsplat / INRIA convention). Third-party viewers open this."""
    n = pos.shape[0]
    SH0 = 0.28209479177387814
    rgb_f = rgb_u8.astype(np.float32) / 255.0
    f_dc = (rgb_f - 0.5) / SH0
    op = np.clip(opacity_u8.astype(np.float32) / 255.0, 1e-4, 1 - 1e-4)
    op_logit = np.log(op / (1 - op))
    log_scale = np.log(np.maximum(scales, 1e-6))
    # PLY quaternion convention is (w, x, y, z). Ours is (x, y, z, w).
    q_wxyz = np.stack([quat[:, 3], quat[:, 0], quat[:, 1], quat[:, 2]], axis=1)

    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
        "property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n"
        "end_header\n"
    ).encode("ascii")
    zeros3 = np.zeros((n, 3), dtype=np.float32)
    rows = np.concatenate([
        pos.astype(np.float32), zeros3,
        f_dc.astype(np.float32), op_logit.astype(np.float32).reshape(-1, 1),
        log_scale.astype(np.float32), q_wxyz.astype(np.float32),
    ], axis=1)
    with open(path, "wb") as f:
        f.write(header)
        f.write(rows.tobytes())


# --- Main -----------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    t0 = time.perf_counter()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[enc] device={device} model={cfg.model_id}")

    # 1. Load model, grab wte -----------------------------------------------
    tok = GPT2TokenizerFast.from_pretrained(cfg.model_id)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_id).eval()
    wte = model.transformer.wte.weight.detach().cpu().numpy().astype(np.float32)
    assert wte.shape == (cfg.vocab_size, cfg.hidden_size), wte.shape
    V, D = wte.shape

    # 2. Centre per-dim (subtract mean only). We deliberately do NOT
    # rescale-per-dim or clip: both are opinions that would shape the visible
    # cloud. PCA below still gets useful directions from a centred matrix.
    mean = wte.mean(axis=0, keepdims=True)
    z = (wte - mean).astype(np.float32)

    # 3. PCA(57) ------------------------------------------------------------
    n_components = 57                                # 3 pos + 3 scale + 3 quat + 3 col + 45 SH
    print(f"[enc] fitting PCA({n_components})…")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=cfg.seed)
    proj = pca.fit_transform(z).astype(np.float32)   # (V, 57)
    ev = pca.explained_variance_ratio_
    print(f"[enc] top-3 var = {ev[:3].sum() * 100:.1f}%  · top-12 var = {ev[:12].sum() * 100:.1f}%  · top-57 var = {ev.sum() * 100:.1f}%")

    # 4. Position -----------------------------------------------------------
    # Raw PC 1..3 with a single global scale factor so the 99th-percentile
    # radius equals 1. No per-axis whitening, no signed-sqrt spread — the
    # cloud shape you see is the actual shape of the PCA cloud (elongated
    # along PC 1 because PC 1 genuinely has the most variance).
    pos = proj[:, 0:3].astype(np.float32)             # (V, 3)
    radii = np.linalg.norm(pos, axis=1)
    scale_ref = float(np.percentile(radii, int(cfg.raw["export"]["scale_percentile"])))
    pos = pos / scale_ref
    proj_s = proj / scale_ref                         # rescale the OTHER PCs by the same factor

    # 5+6. Scale + rotation from the LOCAL neighbourhood ellipsoid ----------
    # For each token, take its top-k neighbours in the full 768-dim wte,
    # look at where those neighbours sit in the 3-D display, and fit an
    # ellipsoid (3×3 covariance). The eigenvalues become the splat's scale
    # (how stretched is the neighbourhood along each axis), the eigenvectors
    # become its rotation (which way is it stretched). So the splat's shape
    # describes THIS token's local semantic manifold — not an arbitrary PC.
    # Placeholder now; filled in after neighbours computed (§10 below).
    scl = np.zeros((V, 3), dtype=np.float32)
    quat = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (V, 1))

    # 7. Base colour from CHARACTER CLASS ----------------------------------
    # Every token gets a class from what its string LOOKS like — legible legend
    # instead of the previous PC-10-12 mystery colour. Each class picks an OKLab
    # hue at fixed lightness; the categories are hand-picked to match what a
    # reader might notice about tokens (leading space, all-caps, digit-heavy,
    # punctuation-only, byte-fragment, non-ASCII scripts, etc.).
    def is_code_like(core: str) -> bool:
        if "_" in core:
            return True
        for i in range(len(core) - 1):
            if core[i].islower() and core[i + 1].isupper():
                return True
        return False

    def classify(s: str) -> int:
        if not s: return 0                                              # empty
        if s.startswith("\\x"): return 6                                 # raw byte hex
        letters = [c for c in s if c.isalpha()]
        if s.strip() == "": return 7                                    # whitespace-only
        if all(not c.isalnum() and not c.isspace() for c in s): return 5   # punctuation-only
        if any(ord(c) > 127 for c in s): return 8                        # non-ASCII
        starts_space = s[0] == " "
        core = s.lstrip(" ")
        if is_code_like(core): return 9                                  # camelCase / snake_case
        if core.isdigit() or any(c.isdigit() for c in core) and len(letters) == 0: return 4
        if letters and all(c.isupper() for c in letters) and len(core) >= 2: return 3
        if starts_space and letters and core[0].isupper(): return 2
        if starts_space: return 1
        return 0

    classes = np.array([classify(tok.decode([i])) for i in range(V)], dtype=np.int32)
    N_CLASSES = 10
    # OKLab palette (L, a, b) — chosen for perceptual distinctness at fixed L.
    palette = np.array([
        (0.72, -0.02, -0.02),   # 0 subword fragment (neutral warm-grey)
        (0.75,  0.12,  0.08),   # 1 word start       (warm)
        (0.80,  0.02,  0.14),   # 2 proper-noun-ish  (soft gold)
        (0.70, -0.06,  0.14),   # 3 all-caps         (sunflower)
        (0.72, -0.14,  0.02),   # 4 digit            (green-teal)
        (0.65,  0.16, -0.02),   # 5 punctuation      (pink)
        (0.55, -0.02, -0.02),   # 6 byte-fragment    (dim grey)
        (0.60, -0.08, -0.06),   # 7 whitespace       (cool grey)
        (0.75, -0.10,  0.10),   # 8 non-ASCII        (jade)
        (0.68,  0.10, -0.14),   # 9 code-like        (violet)
    ], dtype=np.float32)
    rgb_u8 = oklab_to_srgb_u8(palette[classes, 0], palette[classes, 1], palette[classes, 2])

    # 8. Opacity — filled in below from the neighbour-density signal --------
    op_u8 = np.zeros(V, dtype=np.uint8)               # populated after step 10
    flags = np.zeros(V, dtype=np.uint8)

    # 9. SH bands 1..3 = PC 4..48 (45 dims) ---------------------------------
    # Scale/rotation/colour no longer consume PC 4..12, so SH now soaks up the
    # top 45 residual-variance directions after position — a much richer
    # "everything else" channel than before.
    sh = proj_s[:, 3:48].astype(np.float32)           # (V, 45)
    band_gains = list(cfg.raw["colour"]["band_gains"])
    band_lens = [3, 5, 7]                             # SH band coefficient counts
    blocks: list[bytes] = []
    scales_bytes: list[bytes] = []
    for b_i, blen in enumerate(band_lens):
        start = sum(band_lens[:b_i]) * 3
        end = start + blen * 3
        q, s = float_to_int8_per_band(sh[:, start:end] * band_gains[b_i])
        blocks.append(q.tobytes())
        scales_bytes.append(np.float32(s).tobytes())

    # 10. Neighbours + local-density opacity signal -------------------------
    # k=20 so the UI can hop through neighbours; the hover card slices to 8.
    # We also compute the mean cosine to the top-8 for the opacity channel:
    # tokens inside tight clusters (weekdays, digits, months) glow, tokens on
    # the edge of the manifold (glitch tokens, orphaned symbols) dim.
    print(f"[enc] computing top-20 neighbours + density…")
    k = 20
    with torch.no_grad():
        w_t = torch.from_numpy(wte).to(device).float()
        wn = w_t / w_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
        neigh = np.zeros((V, k), dtype=np.uint16)
        density = np.zeros(V, dtype=np.float32)
        batch = 1024
        for i in range(0, V, batch):
            j = min(i + batch, V)
            sim = wn[i:j] @ wn.T
            idx = torch.arange(i, j, device=device)
            sim[torch.arange(j - i, device=device), idx] = -2.0
            topv, topi = sim.topk(k, dim=1)
            neigh[i:j] = topi.cpu().numpy().astype(np.uint16)
            density[i:j] = topv[:, :8].mean(dim=1).cpu().numpy()

    # Opacity ← density, remapped to [op_min, op_max] via 5..95 percentile.
    lo, hi = np.percentile(density, [5, 95])
    op01 = np.clip((density - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    op_min = float(cfg.raw["opacity"]["min"])
    op_max = float(cfg.raw["opacity"]["max"])
    op = op_min + (op_max - op_min) * op01
    op_u8 = (op * 255).clip(0, 255).astype(np.uint8)

    # Local-neighbourhood ellipsoid: PCA on each token's top-8 neighbours' 3-D
    # positions. Eigenvalues → scale (semi-axes), eigenvectors → rotation.
    #
    # Anisotropy strategy: normalise eigenvalues so the largest is 1, then
    # apply a power law with `anisotropy_gamma`. gamma=1 keeps the raw ratio
    # of eigenvalue-stds — a truly linear neighbourhood (weekdays chained
    # PC-1-wise) shows as a needle with ~10× axis ratio, blob-like tokens
    # stay near-spherical. Larger gamma → less contrast, smaller → more.
    print(f"[enc] fitting local ellipsoids (k=8)…")
    K_LOCAL = 8
    scale_base = 0.020
    anisotropy_gamma = 0.55
    for i in range(V):
        ids_nn = neigh[i, :K_LOCAL].astype(np.int64)
        pts = pos[ids_nn]                              # (K_LOCAL, 3)
        centred = pts - pts.mean(axis=0, keepdims=True)
        cov = centred.T @ centred / max(K_LOCAL - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)         # ascending
        eigvals = eigvals[::-1]                        # descending
        eigvecs = eigvecs[:, ::-1]
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1
        std = np.sqrt(np.maximum(eigvals, 1e-12))
        # Max-normalise, then power-law amplify.
        rmax = std[0] + 1e-12
        ratio = std / rmax
        scl[i] = scale_base * (ratio ** anisotropy_gamma)
        quat[i] = R.from_matrix(eigvecs).as_quat()      # (x, y, z, w)

    # Global rescale so the median MAX-axis is roughly scale_base — keeps the
    # overall splat size stable while the anisotropy story is preserved.
    max_axis = scl.max(axis=1)
    med = float(np.median(max_axis))
    scl *= scale_base / (med + 1e-8)

    # 11. Tokens JSON -------------------------------------------------------
    # HF's GPT-2 tokenizer maps bytes → private Unicode chars for BPE; on
    # decode, invalid multi-byte fragments become U+FFFD. Detect that and
    # emit the raw bytes as \xNN — hex is honest, U+FFFD renders as tofu.
    from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
    byte_encoder = bytes_to_unicode()
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    def token_string(i: int) -> str:
        s = tok.decode([i])
        if "�" not in s:
            return s
        raw_str = tok.convert_ids_to_tokens(i)
        try:
            raw_bytes = bytes(byte_decoder[c] for c in raw_str)
        except KeyError:
            return s
        return "".join(f"\\x{b:02X}" for b in raw_bytes)
    tokens: list[str] = [token_string(i) for i in range(V)]
    # Class → (name, sRGB uint8 for the swatch). We reuse oklab_to_srgb_u8 on
    # the palette so the About panel shows the *exact* colour rendered on splats.
    class_names = ["subword fragment", "word start", "proper noun", "ALL-CAPS",
                    "digit", "punctuation", "byte fragment", "whitespace",
                    "non-ASCII / foreign script", "code-like (camelCase / snake_case)"]
    class_rgb = oklab_to_srgb_u8(palette[:, 0], palette[:, 1], palette[:, 2])
    class_counts = np.bincount(classes, minlength=N_CLASSES).tolist()
    palette_json = [
        {"id": i, "name": class_names[i], "rgb": [int(class_rgb[i, 0]), int(class_rgb[i, 1]), int(class_rgb[i, 2])], "count": int(class_counts[i])}
        for i in range(N_CLASSES)
    ]
    doc = {
        "vocab_size": V,
        "phase": 3,                                    # keep the field, use as a version
        "encoding": "static_hi_d_v3",
        "num_layers": 0,                               # signals "no scrubber"
        "explained_variance_top3": [float(ev[0]), float(ev[1]), float(ev[2])],
        "explained_variance_top12": float(ev[:12].sum()),
        "explained_variance_top57": float(ev.sum()),
        "scene_scale": scale_ref,
        "band_gains": band_gains,
        "char_class_palette": palette_json,
        "tokens": [{"s": s, "sparse": False} for s in tokens],
    }

    # 12. Write assets ------------------------------------------------------
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    write_splat_bin(out / "splats.bin", pos, scl, quat, rgb_u8)
    with open(out / "sh_00.bin", "wb") as f:
        for sb in scales_bytes: f.write(sb)
        for blk in blocks: f.write(blk)
    with open(out / "static.bin", "wb") as f:
        f.write(op_u8.tobytes())
        f.write(flags.tobytes())
    with open(out / "tokens.json", "w") as f:
        json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    with open(out / "neighbours.bin", "wb") as f:
        f.write(neigh.tobytes())
    write_debug_ply(out / "debug.ply", pos, scl, quat, rgb_u8, op_u8)

    # Clean up obsolete kf_XX / sh_XX files from earlier runs.
    for p in list(out.glob("kf_*.bin")) + list(out.glob("sh_[0-9][0-9].bin")):
        if p.name != "sh_00.bin":
            p.unlink()
    for p in (out / "debug_kf00.ply", out / "debug_kf12.ply"):
        if p.exists(): p.unlink()

    dt = time.perf_counter() - t0
    print(f"[enc] wrote splats.bin ({(out / 'splats.bin').stat().st_size / 1e6:.2f} MB) + sh_00.bin in {dt:.1f}s")

    # Semantic sanity ------------------------------------------------------
    monday_id = tok.encode(" Monday", add_special_tokens=False)
    if monday_id and monday_id[0] < V:
        mid = monday_id[0]
        neigh_strs = [tok.decode([nid]) for nid in neigh[mid].tolist()]
        print(f"[enc] sanity: ' Monday'(id={mid}) neighbours → {neigh_strs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
