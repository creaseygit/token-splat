# Token Space Splat

**Live 3D gaussian-splat viewer of GPT-2 small's 50 257-token embedding space.**

Every splat is one token. Every free parameter of the gaussian splat (position, ellipsoid scale + rotation, base colour, view-dependent spherical-harmonic shimmer, opacity) encodes a different structural signal about that token — 57 of the raw 768 embedding dimensions expressed at once, honestly derived from the model's own weights.

## What each splat's channels mean

| Splat parameter | Signal |
| --- | --- |
| Position | PC 1–3 of the standardised `wte` embedding matrix |
| Ellipsoid shape + tilt | Local PCA of the token's 8 nearest neighbours in 3-D |
| Base colour | Character class (word / all-caps / digit / punctuation / byte / non-ASCII / code-like / …) |
| SH bands 1–3 (45-D shimmer) | PC 4–48 (the high-D residual after position) |
| Opacity | Mean cosine similarity to 8 NN (how well the token is embedded) |

Because the ellipsoid shape is computed from each token's own local neighbourhood, tokens on a semantic chain (weekdays, months, digits) render as visible needles; tokens sitting in a blob look nearly spherical.

## Interactions

- **Orbit, zoom, pan** with mouse or touch
- **Hover** for the token's 8 nearest neighbours
- **Click** to focus a splat — camera re-targets, marker stays pinned, yellow lines drawn to its 20 nearest neighbours
- **Focus chip** at the top opens a 20-item dropdown for hopping through the semantic graph
- **Search** box (top-right) tokenises text against GPT-2's vocabulary and flies to the first match
- **Explode slider** (bottom-centre) spreads the cloud outward without inflating splat sizes, so dense clusters become browsable
- **Relations depth 1 / 2 / 3** (bottom-right) — draw the direct-, second-, and third-hop neighbour edges
- **Isolate** button — hide every splat that isn't in the currently focused token's relation set

## Quick start

```bash
# Web only — assets are checked in, so no Python required to view.
cd web && npm install && npm run dev
# then http://localhost:5173
```

To regenerate the splat assets from the model:

```bash
# One-time: Python 3.11 via pyenv + venv
pyenv install 3.11.11 --skip-existing
python3.11 -m venv pipeline/.venv
pipeline/.venv/bin/pip install -r pipeline/requirements.txt

make phase1     # downloads GPT-2 once, then produces web/public/assets/*.bin
```

## Layout

```
token-splat/
  config.yaml            model id, PCA + colour + opacity knobs
  Makefile               make phase1 | make dev | make test
  pipeline/              Python encoder (produces splat + SH + neighbour + token assets)
  web/                   Vite + TypeScript + three.js + Spark renderer
  web/public/assets/     ~10 MB of generated splat data (committed for convenience)
```

## Credits

- **Model:** GPT-2 small (`openai-community/gpt2`), OpenAI, Modified MIT
- **Renderer:** [Spark](https://sparkjs.dev/) 3DGS for three.js (MIT)
- **Colour space:** OKLab (Björn Ottosson)

Design notes and open questions live in [`DECISIONS.md`](./DECISIONS.md).
