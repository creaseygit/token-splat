# Token Space Splat build targets.
# One command regenerates every asset: `make assets`.

PY ?= pipeline/.venv/bin/python
CONFIG ?= config.yaml

.PHONY: all assets web dev test clean help

help:
	@echo "Targets:"
	@echo "  make assets   Regenerate binary assets under web/public/assets/"
	@echo "  make web      Build the static site into web/dist/"
	@echo "  make dev      Run Vite dev server"
	@echo "  make test     Python + JS unit tests"
	@echo "  make all      assets + web"
	@echo "  make clean    Remove generated assets and dist"

# --- Pipeline ---------------------------------------------------------------

# Phase 0 spike: extract 10k tokens through GPT-2, dump hidden-state shapes.
spike-extract:
	$(PY) pipeline/spike_extract.py --tokens 10_000 --config $(CONFIG)

# Phase 0 spike: export ONNX with a placeholder projection; verify parity.
spike-onnx:
	$(PY) pipeline/spike_onnx.py --config $(CONFIG)

# Phase 1: keyframe 0 only (raw wte -> PCA -> splats + neighbours + PLY).
phase1:
	$(PY) pipeline/phase1_kf0.py --config $(CONFIG)

# Full two-pass extraction and encode.
assets:
	$(PY) pipeline/extract.py --config $(CONFIG) --pass 1
	$(PY) pipeline/basis.py   --config $(CONFIG)
	$(PY) pipeline/extract.py --config $(CONFIG) --pass 2
	$(PY) pipeline/encode.py  --config $(CONFIG)
	$(PY) pipeline/examples.py --config $(CONFIG)
	$(PY) pipeline/export_onnx.py --config $(CONFIG)

# Phase 2: two-pass corpus + basis + encode. Override token budget via TOKENS,
# corpus via CORPUS. Wikitext-103 is the default (small, permissive download).
TOKENS ?= 5_000_000
CORPUS ?= wikitext-103-raw-v1

phase2:
	$(PY) pipeline/extract.py --config $(CONFIG) --pass 1 --tokens $(TOKENS) --corpus $(CORPUS)
	$(PY) pipeline/basis.py   --config $(CONFIG)
	$(PY) pipeline/extract.py --config $(CONFIG) --pass 2 --tokens $(TOKENS) --corpus $(CORPUS)
	$(PY) pipeline/encode.py  --config $(CONFIG)

phase2-smoke:
	$(MAKE) phase2 TOKENS=200_000

# --- Web --------------------------------------------------------------------

web:
	cd web && npm run build

dev:
	cd web && npm run dev

# --- Tests ------------------------------------------------------------------

test:
	$(PY) -m pytest pipeline/tests -q
	cd web && npm test --if-present

all: assets web

clean:
	rm -rf web/public/assets/*.bin web/public/assets/*.json web/public/assets/*.onnx web/dist
