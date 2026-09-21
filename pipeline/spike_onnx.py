"""Phase 0 spike: export GPT-2 wrapped with a bake-in projection to ONNX,
then verify the ONNX Runtime output matches PyTorch within 1e-3 in float32.

The projection is a *placeholder* here (identity in the top 51 dims), because
the real basis fit needs a corpus. The spike proves:

  1. `torch.onnx.export` handles GPT-2 with `output_hidden_states=True` and
     our post-model projection module.
  2. Output shape is [num_layers + 1, seq, 51].
  3. ORT and PyTorch agree numerically (< 1e-3 max abs diff in float32).

If this passes, Phase 3 swaps the placeholder for the real (mean, std, W)
learned by `basis.py`, then dynamic-quantises to int8 (~125 MB).

Usage:
    pipeline/.venv/bin/python pipeline/spike_onnx.py --config config.yaml
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from common import load_config


class Gpt2Projected(nn.Module):
    """Wrap GPT-2 so its forward returns projected hidden states.

    Baked-in transform per layer L:  proj(h) = (h - mean[L]) / std[L] @ W[L]
    Clip to +/-6 before projection, matching the offline pipeline.
    Placeholder: mean=0, std=1, W = [I_51 | 0].
    """

    def __init__(self, backbone: GPT2LMHeadModel, mean: torch.Tensor, std: torch.Tensor, W: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        # (num_layers+1, hidden), (num_layers+1, hidden), (num_layers+1, hidden, k)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.register_buffer("W", W)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # Stack -> (num_layers+1, batch, seq, hidden). Batch = 1 for live mode.
        hs = torch.stack(out.hidden_states, dim=0).squeeze(1)  # (L+1, seq, hidden)
        z = (hs - self.mean.unsqueeze(1)).clamp(-6.0, 6.0) / self.std.unsqueeze(1)
        # (L+1, seq, hidden) @ (L+1, hidden, k) -> (L+1, seq, k)
        return torch.bmm(z, self.W)


def make_placeholder_basis(num_layers: int, hidden: int, k: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = torch.zeros(num_layers + 1, hidden)
    std = torch.ones(num_layers + 1, hidden)
    # Identity in top-k dims, zeros elsewhere. Same W across layers for the spike.
    w1 = torch.zeros(hidden, k)
    w1[:k, :k] = torch.eye(k)
    W = w1.unsqueeze(0).expand(num_layers + 1, -1, -1).contiguous()
    return mean, std, W


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--k", type=int, default=51)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"[spike] loading {cfg.model_id}")
    tok = GPT2TokenizerFast.from_pretrained(cfg.model_id)
    backbone = GPT2LMHeadModel.from_pretrained(cfg.model_id).eval()

    mean, std, W = make_placeholder_basis(cfg.num_layers, cfg.hidden_size, args.k)
    model = Gpt2Projected(backbone, mean, std, W).eval()

    text = "The quick brown fox jumps over the lazy dog."
    input_ids = tok(text, return_tensors="pt").input_ids
    if input_ids.size(1) < args.seq_len:
        pad = tok.eos_token_id
        input_ids = torch.cat([input_ids, torch.full((1, args.seq_len - input_ids.size(1)), pad)], dim=1)
    else:
        input_ids = input_ids[:, :args.seq_len]
    print(f"[spike] input_ids shape={tuple(input_ids.shape)}")

    with torch.no_grad():
        torch_out = model(input_ids).cpu().numpy()  # (L+1, seq, k)
    print(f"[spike] torch output shape={torch_out.shape}, max={torch_out.max():.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        onnx_path = Path(tmp) / "gpt2_proj.onnx"
        # torch 2.14 dynamo exporter (needs onnxscript). Legacy TorchScript
        # exporter (dynamo=False) hits an internal RuntimeError on GPT-2 here.
        torch.onnx.export(
            model, (input_ids,), onnx_path.as_posix(),
            input_names=["input_ids"],
            output_names=["projected"],
            dynamic_shapes={"input_ids": {1: torch.export.Dim("seq")}},
            opset_version=int(cfg.raw["onnx"]["opset"]),
            dynamo=True,
        )
        print(f"[spike] wrote {onnx_path.stat().st_size / 1e6:.1f} MB")

        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"input_ids": input_ids.numpy()})[0]

    print(f"[spike] ort output shape={ort_out.shape}, max={ort_out.max():.4f}")
    diff = np.max(np.abs(torch_out - ort_out))
    print(f"[spike] max |torch - ort| = {diff:.2e}")

    threshold = 1e-3
    if diff < threshold:
        print(f"[spike] PASS (< {threshold})")
        return 0
    print(f"[spike] FAIL (>= {threshold})")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
