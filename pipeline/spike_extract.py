"""Phase 0 spike: extract hidden states for a small token budget.

Purpose: prove that `output_hidden_states=True` gives us `num_layers + 1`
snapshots of the residual stream at the right shape, and that per-token
accumulation on the CPU is fast enough to plan Pass 1.

Usage:
    pipeline/.venv/bin/python pipeline/spike_extract.py --tokens 10_000

Prints:
  - Model config vs config.yaml
  - Hidden-state tensor shapes at each layer
  - Sanity check: mean L2 norm per layer (should climb, then dip on final LN)
  - Wallclock and tokens/sec
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from common import load_config


def build_batches(tokenizer: GPT2TokenizerFast, n_tokens: int, seq_len: int) -> list[torch.Tensor]:
    """Sample sequences from Shakespeare (bundled with tokenizers tests? no).
    For the spike we do not download a corpus — we generate synthetic prompts
    from a small hand-written passage repeated + tokenised. The point is to
    check shapes and speed, not statistics.
    """
    text = (
        "The quick brown fox jumps over the lazy dog. Pack my box with five "
        "dozen liquor jugs. How vexingly quick daft zebras jump! The five "
        "boxing wizards jump quickly. Sphinx of black quartz, judge my vow. "
    ) * 200
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    eot = tokenizer.encode("<|endoftext|>", add_special_tokens=False)[0]
    per_seq = seq_len - 1
    seqs: list[torch.Tensor] = []
    total = 0
    for start in range(0, ids.size(0) - per_seq, per_seq):
        chunk = ids[start:start + per_seq]
        seq = torch.cat([torch.tensor([eot]), chunk])
        seqs.append(seq)
        total += chunk.size(0)  # count usable positions (exclude the eot prefix)
        if total >= n_tokens:
            break
    return seqs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--tokens", type=int, default=10_000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[spike] device={device} model={cfg.model_id}")

    tok = GPT2TokenizerFast.from_pretrained(cfg.model_id)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_id).to(device).eval()

    hcfg = model.config
    print(f"[spike] model.n_embd={hcfg.n_embd} n_layer={hcfg.n_layer} vocab={hcfg.vocab_size}")
    assert hcfg.n_embd == cfg.hidden_size, "hidden_size mismatch — update config.yaml"
    assert hcfg.n_layer == cfg.num_layers, "num_layers mismatch — update config.yaml"
    assert hcfg.vocab_size == cfg.vocab_size, "vocab_size mismatch — update config.yaml"

    seqs = build_batches(tok, args.tokens, seq_len=128)  # short seqs speed the spike
    print(f"[spike] {len(seqs)} sequences of {seqs[0].size(0)} tokens; total ~{sum(s.size(0)-1 for s in seqs)} tokens")

    t0 = time.perf_counter()
    per_layer_norms = torch.zeros(cfg.num_layers + 1, device=device)
    per_layer_count = 0
    with torch.no_grad():
        for seq in seqs:
            out = model(seq.unsqueeze(0).to(device), output_hidden_states=True, use_cache=False)
            hs = out.hidden_states  # tuple of (1, T, hidden) length num_layers+1
            assert len(hs) == cfg.num_layers + 1
            for i, h in enumerate(hs):
                # drop position 0 per spec
                per_layer_norms[i] += h[0, 1:].norm(dim=-1).sum()
            per_layer_count += hs[0].size(1) - 1
    dt = time.perf_counter() - t0
    print(f"[spike] forward wall={dt:.2f}s tokens/s={per_layer_count/dt:.0f}")

    mean_norms = (per_layer_norms / per_layer_count).cpu().numpy()
    print("[spike] mean L2 norm per keyframe (excl pos 0):")
    for i, n in enumerate(mean_norms):
        print(f"  keyframe {i:>2}  ||h||={n:.2f}")

    # Success criteria: shapes match, norms are finite, run finished.
    if not np.isfinite(mean_norms).all():
        print("[spike] FAIL: non-finite norms")
        return 1
    print("[spike] PASS: shapes + norms OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
