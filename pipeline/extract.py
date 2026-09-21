"""Corpus passes for Phase 2.

Two entry points, both parameterised by `--pass`:

    Pass 1: for each (keyframe, token_id), accumulate count + sum of hidden
            states. Also collect per-(keyframe, dim) mean + var for
            standardisation. Backfill unseen tokens with [<|endoftext|>, tok]
            and mark them `sparse`.

    Pass 2: with the fitted basis (W: hidden→51) available, re-run the corpus
            and accumulate second moments of the *projected 3D* positions per
            (keyframe, token). Covariance = E[xx^T] − μμ^T is derived at the
            end.

Corpus: WikiText-103 (permissive, ~100M tokens, small download). Configurable
in config.yaml → corpus.primary. OpenWebText streaming can be swapped in when
the license/gating is confirmed for the deploy target.

The pipeline is deterministic: sequences are always in the same order for a
given seed + token budget, so Pass 2 sees the same hidden states as Pass 1.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from common import Config, REPO_ROOT, load_config


CACHE_DIR = REPO_ROOT / "pipeline" / "cache"


# --- Corpus streaming ------------------------------------------------------

def load_corpus_text(name: str) -> Iterator[str]:
    """Yield raw text chunks from the configured corpus.

    We use `datasets.load_dataset` in streaming mode when possible so we do
    not need to download 100 M tokens for a 5 M-token budget.
    """
    from datasets import load_dataset

    if name == "wikitext-103-raw-v1":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        for row in ds:
            text = row.get("text")
            if text and text.strip():
                yield text
        return

    if name == "openwebtext":
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True, trust_remote_code=False)
        for row in ds:
            text = row.get("text")
            if text:
                yield text
        return

    raise ValueError(f"unknown corpus name: {name}")


def stream_sequences(
    tokenizer: GPT2TokenizerFast,
    corpus_name: str,
    seq_len: int,
    n_tokens: int,
) -> Iterator[torch.Tensor]:
    """Yield 1-D int64 tensors of length `seq_len`, EOS-prefixed.

    We keep a running buffer of freshly tokenised text and chop out
    (seq_len - 1)-token windows, prefixing each with the EOS id. Stops after
    at least `n_tokens` usable positions (excluding the EOS prefix) have been
    emitted.
    """
    eot = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    per_seq = seq_len - 1
    buf: list[int] = []
    yielded_tokens = 0
    for text in load_corpus_text(corpus_name):
        ids = tokenizer.encode(text, add_special_tokens=False)
        buf.extend(ids)
        while len(buf) >= per_seq:
            chunk = buf[:per_seq]
            buf = buf[per_seq:]
            seq = torch.tensor([eot] + chunk, dtype=torch.long)
            yield seq
            yielded_tokens += per_seq
            if yielded_tokens >= n_tokens:
                return


# --- Pass 1 ----------------------------------------------------------------

@dataclass
class Pass1Result:
    counts: torch.Tensor    # (L+1, V) int64
    sums:   torch.Tensor    # (L+1, V, H) float32
    dim_sum:    torch.Tensor   # (L+1, H) float32   Σ x
    dim_sq_sum: torch.Tensor   # (L+1, H) float32   Σ x²
    total_positions: torch.Tensor   # (L+1,) int64  — shared across tokens


def run_pass1(cfg: Config, out_dir: Path, n_tokens: int, corpus_name: str, batch_size: int = 4) -> Pass1Result:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p1] device={device} corpus={corpus_name} n_tokens={n_tokens:,}")
    tok = GPT2TokenizerFast.from_pretrained(cfg.model_id)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_id).eval().to(device)

    L1 = cfg.num_layers + 1
    V, H = cfg.vocab_size, cfg.hidden_size

    # Accumulators. sums is the biggest at 13 × 50257 × 768 × 4 B = 2.0 GB.
    counts = torch.zeros((L1, V), dtype=torch.int64, device=device)
    sums = torch.zeros((L1, V, H), dtype=torch.float32, device=device)
    # MPS is float32-only; keep dim-sum accumulators in fp32 on device and
    # promote to fp64 on CPU at save time (200 K–20 M tokens fit in fp32 without
    # meaningful loss, especially post-standardisation).
    dim_sum = torch.zeros((L1, H), dtype=torch.float32, device=device)
    dim_sq_sum = torch.zeros((L1, H), dtype=torch.float32, device=device)
    total_positions = torch.zeros((L1,), dtype=torch.int64, device=device)

    seq_iter = stream_sequences(tok, corpus_name, cfg.raw["corpus"]["seq_len"], n_tokens)

    # Simple batch buffering.
    buf: list[torch.Tensor] = []
    seen_tokens = 0
    bar = tqdm(total=n_tokens, desc="pass1", unit="tok", smoothing=0.02)
    t0 = time.perf_counter()

    def flush(buf: list[torch.Tensor]) -> None:
        nonlocal seen_tokens
        if not buf:
            return
        batch = torch.stack(buf, dim=0).to(device)   # (B, S)
        B, S = batch.shape
        with torch.no_grad():
            out = model(batch, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple length L+1 of (B, S, H). Drop the EOS prefix column.
        for li, h in enumerate(out.hidden_states):
            h = h[:, 1:, :].contiguous()                   # (B, S-1, H)
            flat_h = h.reshape(-1, H)                     # (B*(S-1), H)
            flat_ids = batch[:, 1:].reshape(-1)           # (B*(S-1),)
            # Accumulate counts and sums per token id.
            counts[li].scatter_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=torch.int64))
            sums[li].index_add_(0, flat_ids, flat_h)
            # Per-dim stats over the full flattened batch (independent of token id).
            dim_sum[li]    += flat_h.sum(dim=0)
            dim_sq_sum[li] += (flat_h * flat_h).sum(dim=0)
            total_positions[li] += flat_ids.numel()
        step_tokens = (S - 1) * B
        seen_tokens += step_tokens
        bar.update(step_tokens)

    for seq in seq_iter:
        buf.append(seq)
        if len(buf) == batch_size:
            flush(buf); buf = []
    flush(buf)
    bar.close()

    dt = time.perf_counter() - t0
    print(f"[p1] {seen_tokens:,} tokens in {dt:.1f}s → {seen_tokens/dt:.0f} tok/s")
    print(f"[p1] tokens with zero occurrences: {(counts[0] == 0).sum().item():,} / {V:,}")

    # Backfill unseen tokens with [EOS, tok] → take position 1.
    backfill_unseen(cfg, model, tok, counts, sums, device)

    # Move to CPU + save. Promote dim stats to fp64 on CPU for downstream
    # numerical stability in basis fitting.
    result = Pass1Result(
        counts=counts.cpu(),
        sums=sums.cpu(),
        dim_sum=dim_sum.cpu().to(torch.float64),
        dim_sq_sum=dim_sq_sum.cpu().to(torch.float64),
        total_positions=total_positions.cpu(),
    )
    save_pass1(result, out_dir)
    return result


def backfill_unseen(cfg: Config, model, tok, counts: torch.Tensor, sums: torch.Tensor, device: str) -> None:
    """For any token id with zero occurrences at *any* keyframe, run [EOS, tok]
    and use position 1 as its representative for all keyframes. Mark it sparse
    (the flag is written later by encode.py using the count threshold)."""
    eot = tok.encode("<|endoftext|>", add_special_tokens=False)[0]
    unseen = (counts[0] == 0).nonzero(as_tuple=False).squeeze(-1).tolist()
    if not unseen:
        return
    print(f"[p1] backfilling {len(unseen):,} unseen tokens")
    bar = tqdm(total=len(unseen), desc="backfill", unit="tok")
    B = 32
    for i in range(0, len(unseen), B):
        chunk = unseen[i:i + B]
        # Build a (b, 2) batch of [eot, tok] pairs.
        batch = torch.tensor([[eot, t] for t in chunk], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(batch, output_hidden_states=True, use_cache=False)
        ids = torch.tensor(chunk, dtype=torch.int64, device=device)
        for li, h in enumerate(out.hidden_states):
            v = h[:, 1, :].contiguous()   # (b, H)
            counts[li].index_add_(0, ids, torch.ones_like(ids, dtype=torch.int64))
            sums[li].index_add_(0, ids, v)
        bar.update(len(chunk))
    bar.close()


def save_pass1(res: Pass1Result, out_dir: Path) -> None:
    d = out_dir / "pass1"
    d.mkdir(parents=True, exist_ok=True)
    # Keep sums separate — it's 2 GB and worth writing as fp16 on disk.
    np.save(d / "counts.npy", res.counts.numpy())
    np.save(d / "sums_f16.npy", res.sums.to(torch.float16).numpy())
    np.save(d / "dim_sum.npy", res.dim_sum.numpy())
    np.save(d / "dim_sq_sum.npy", res.dim_sq_sum.numpy())
    np.save(d / "total_positions.npy", res.total_positions.numpy())
    print(f"[p1] saved {d}")


# --- Pass 2 ----------------------------------------------------------------

def run_pass2(cfg: Config, out_dir: Path, n_tokens: int, corpus_name: str, batch_size: int = 4) -> None:
    """After the basis is fit, re-run the corpus and accumulate 3D second
    moments per (keyframe, token). Uses the *same* sequence stream as pass 1
    (deterministic tokeniser + streaming order + seed).
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[p2] device={device} corpus={corpus_name} n_tokens={n_tokens:,}")
    tok = GPT2TokenizerFast.from_pretrained(cfg.model_id)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_id).eval().to(device)

    basis_dir = out_dir / "basis"
    mean = torch.from_numpy(np.load(basis_dir / "mean.npy")).to(device)   # (L+1, H)
    std = torch.from_numpy(np.load(basis_dir / "std.npy")).to(device)     # (L+1, H)
    W = torch.from_numpy(np.load(basis_dir / "W.npy")).to(device)          # (L+1, H, 51) or (H, 51)
    per_layer = W.dim() == 3

    L1 = cfg.num_layers + 1
    V = cfg.vocab_size

    # 3D accumulators only. sums3d = (L+1, V, 3) f32 = 7.8 MB — cheap.
    # MPS is fp32-only; keep sq3d in fp32 on device, promote at save time.
    sums3d = torch.zeros((L1, V, 3), dtype=torch.float32, device=device)
    sq3d = torch.zeros((L1, V, 6), dtype=torch.float32, device=device)
    counts = torch.zeros((L1, V), dtype=torch.int64, device=device)

    seq_iter = stream_sequences(tok, corpus_name, cfg.raw["corpus"]["seq_len"], n_tokens)
    clip = float(cfg.raw["standardise"]["clip"])

    buf: list[torch.Tensor] = []
    seen_tokens = 0
    bar = tqdm(total=n_tokens, desc="pass2", unit="tok", smoothing=0.02)
    t0 = time.perf_counter()

    def flush(buf: list[torch.Tensor]) -> None:
        nonlocal seen_tokens
        if not buf:
            return
        batch = torch.stack(buf, dim=0).to(device)
        B, S = batch.shape
        with torch.no_grad():
            out = model(batch, output_hidden_states=True, use_cache=False)
        for li, h in enumerate(out.hidden_states):
            h = h[:, 1:, :].contiguous()             # (B, S-1, H)
            m = mean[li]                             # (H,)
            s = std[li]                              # (H,)
            z = ((h - m) / s).clamp_(-clip, clip)    # standardise
            if per_layer:
                proj = z @ W[li, :, :3]              # (B, S-1, 3)  — only top 3 needed for pass 2
            else:
                proj = z @ W[:, :3]
            ids = batch[:, 1:].reshape(-1)
            flat_p = proj.reshape(-1, 3)
            counts[li].index_add_(0, ids, torch.ones_like(ids, dtype=torch.int64))
            sums3d[li].index_add_(0, ids, flat_p)
            outer = torch.stack([
                flat_p[:, 0] * flat_p[:, 0],
                flat_p[:, 1] * flat_p[:, 1],
                flat_p[:, 2] * flat_p[:, 2],
                flat_p[:, 0] * flat_p[:, 1],
                flat_p[:, 0] * flat_p[:, 2],
                flat_p[:, 1] * flat_p[:, 2],
            ], dim=1)
            sq3d[li].index_add_(0, ids, outer)
        step_tokens = (S - 1) * B
        seen_tokens += step_tokens
        bar.update(step_tokens)

    for seq in seq_iter:
        buf.append(seq)
        if len(buf) == batch_size:
            flush(buf); buf = []
    flush(buf)
    bar.close()

    dt = time.perf_counter() - t0
    print(f"[p2] {seen_tokens:,} tokens in {dt:.1f}s → {seen_tokens/dt:.0f} tok/s")

    # Save. Promote sq3d to fp64 on CPU for numerical stability during
    # covariance = E[xx^T] − μμ^T.
    d = out_dir / "pass2"
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "counts.npy", counts.cpu().numpy())
    np.save(d / "sums3d.npy", sums3d.cpu().numpy())
    np.save(d / "sq3d.npy", sq3d.cpu().to(torch.float64).numpy())
    print(f"[p2] saved {d}")


# --- CLI -------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pass", dest="which", type=int, choices=[1, 2], required=True)
    parser.add_argument("--tokens", type=int, default=None, help="override corpus.tokens_*")
    parser.add_argument("--corpus", default=None, help="override corpus.primary (e.g. wikitext-103-raw-v1)")
    parser.add_argument("--batch", type=int, default=4)
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus = args.corpus or cfg.raw["corpus"]["fallback"]  # default to wikitext
    if args.tokens is not None:
        n_tokens = args.tokens
    else:
        # MPS on M4 lives closer to GPU than CPU throughput-wise; use the GPU budget.
        n_tokens = cfg.raw["corpus"]["tokens_gpu"] if torch.backends.mps.is_available() else cfg.raw["corpus"]["tokens_cpu"]

    out_dir = CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if args.which == 1:
        run_pass1(cfg, out_dir, n_tokens, corpus, batch_size=args.batch)
    else:
        run_pass2(cfg, out_dir, n_tokens, corpus, batch_size=args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
