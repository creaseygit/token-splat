"""Export GPT-2 wrapped with (standardise → PCA → 51-D) baked in.

The live sentence mode fetches this once (~125 MB int8) and runs it under
onnxruntime-web. Output tensor shape: [num_layers+1, seq, 51]. The site
takes the first three columns per layer as positions.

Correctness gate (Phase 0): full-precision ONNX must match PyTorch within 1e-3.
Quantisation to int8 uses ORT's dynamic quantiser; parity is re-checked
against the offline float32 output in Phase 3 (spec: within 2% of scene radius).

Stub for Phase 0; a working spike lives in `spike_onnx.py`.
"""
from __future__ import annotations

import argparse

from common import Config, load_config


def export(cfg: Config) -> None:
    raise NotImplementedError("full ONNX export: Phase 3")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    export(load_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
