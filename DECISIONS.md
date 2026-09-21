# Decisions

One line per non-obvious choice. Every deviation from spec numbers goes here with a reason.

## 2026-09-20 · Environment

- **Python 3.11.11 via pyenv 2.8.6 (installed via brew).** Spec asks for 3.11; system has 3.9. Pyenv keeps the toolchain isolated to `pipeline/.venv`.
- **Node 24.10 / npm (system).** Vite 8 needs Node ≥18.
- **@huggingface/transformers 4.3.0** in the browser instead of the deprecated `@xenova/transformers` 2.x. The Xenova package moved under the HF org; API is compatible for tokenisation and ORT-web glue.
- **@sparkjsdev/spark 2.2.0** with **three 0.186** (peer dep ≥0.180). Verified public API via installed `dist/types/index.d.ts`.
- **onnxruntime-web 1.30.0** for live mode.
- **Spark API used**: `SparkRenderer` (added to scene), `PackedSplats` (`pushSplat`, `setSplat`, `needsUpdate`, `setMaxSh(3)`), `SplatMesh({ packedSplats, editable: true })`. Dynamic edit path is `setSplat(i, ...); needsUpdate = true` — this is what the layer scrubber will drive.

## Renderer gate — 2026-09-20 · **PASS · use Spark**

- SH-degree-3 supported: ✅ (`PackedSplats.setMaxSh(3)` accepted)
- 50k splats at 60 fps on M4: ✅ 61 fps (threshold 55; measured over 1.5 s)
- Per-frame two-keyframe blend: ✅ (`setSplat(i, …)` + `needsUpdate = true` at stride 8)
- Decision: **use Spark 2.2.0** (no forked WebGL2 renderer needed).

Follow-ups for Phase 2 when we drive the full scrubber:
- Rewrite every splat (stride 1), not every 8th, and re-measure fps.
- Wire covariance into the per-frame update — the current spike only edits
  position and colour. `PackedSplats.setSplat` takes `scales + quaternion`,
  so we'll decompose blended 3×3 matrices offline (spec §Interpolation).

## Phase 0 spikes — 2026-09-20 · **PASS**

- **spike-extract** (10k tokens, MPS on M4): shapes + norms OK, 3,560 tok/s, model config matches (`n_embd=768 n_layer=12 vocab=50257`).
- **spike-onnx** (dynamo exporter, opset 18, identity placeholder projection):
  - Output shape `(13, 16, 51)` as spec requires.
  - PyTorch ↔ ORT max abs diff = **2.74e-05** (threshold 1e-3). ✅

## Threshold deviations

- **ONNX opset 17 → 18.** torch 2.14 dynamo exporter emits `Split` with the
  opset-18 `num_outputs` attribute; ORT 1.30 rejects it under opset 17. ORT 1.30
  fully supports opset 18 in both node and web builds, so bump has no cost.
- **Added `onnxscript` to requirements.txt.** torch 2.14's dynamo exporter is the
  only path that works for `Gpt2Projected` on darwin-arm64 (legacy TorchScript
  path throws `unordered_map::at: key not found` internally). Small, standard dep.
