# Token Space Splat

**Live: https://creaseygit.github.io/token-splat/**

Interactive 3D gaussian-splat viewer of GPT-2 small's 50 257-token embedding table. Every splat is one vocabulary token; the splat's shape, colour, and opacity are all derived from the model's own weights.

## What each splat's channels represent

| Splat parameter | Signal | Comes from |
| --- | --- | --- |
| **Position** | Global layout | PC 1–3 of the standardised `wte` matrix (768 → 3 linear projection) |
| **Ellipsoid shape + tilt** | Local semantic manifold | PCA of the token's 20 nearest neighbours (in full 768-D) projected to 3-D. Tokens on a chain (weekdays, digits) show as needles; blob-shaped clusters show as spheres. |
| **Base colour** | Character class | 10-way classifier over the token string: subword fragment / word start / proper noun / ALL-CAPS / digit / punctuation / byte fragment / whitespace / non-ASCII / code-like (camelCase, snake_case) |
| **Opacity** | Embedding density | Mean cosine similarity to the 20 nearest neighbours in 768-D. Deep-in-cluster tokens glow; fringe tokens dim. |

Position uses PC 1-3 only (~3.4% of total variance), so the layout is a very compressed view — the ellipsoid and opacity channels are computed in the full 768-D space and carry information PCA can't.

## Interactions

- **Orbit / zoom / pan** — mouse or touch
- **Hover a splat** — card shows token string, character class, and the top 8 visible neighbours
- **Click a splat** — camera tweens its orbit target onto it; a yellow marker stays pinned; lines drawn to its 20 nearest neighbours
- **Focus chip (top-centre)** — click to open a 20-item dropdown of visible neighbours; select one to hop
- **Search box (top-right)** — tokenises the input against GPT-2's vocab and flies to the first match
- **Explode slider (bottom-centre)** — spreads positions outward without inflating splat sizes, so dense clusters open up (1× → 12× displayed)
- **Relations 1 / 2 / 3 (bottom-right)** — draw direct-, second-, and third-hop neighbour edges
- **Isolate button** — hide every splat that isn't in the current relation set
- **Most-embedded slider** — keep only the top N% of splats by neighbourhood density (0.1% precision)
- **About panel (bottom-left)** — encoding legend + click any character-class row to toggle that class off

All filters compose. The visible set is the intersection of isolate, cull, and class-toggle filters. Hidden splats are also unpickable.

## Notable findings you can see

- **Glitch-token cluster.** Search or hover any of `rawdownload`, ` SolidGoldMagikarp`, `InstoreAndOnline`, ` サーティ`, and you'll land in a super-tight cluster where every member has ~0.998 cosine similarity to every other. These are tokens the tokeniser learned but the language model barely trained on — they collapsed to a single point. See [Rumbelow & Watkins, 2023](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation).
- **Two hemispheres.** ~66% of GPT-2 tokens start with a space (mid-sentence use); the rest don't. That distinction separates the two lobes by ~1.5σ along PC 2 — biggest single axis in the vocabulary.
- **Semantic chains.** Focus ` Monday` — its 20 neighbours are the other six weekdays and their variants, and its local ellipsoid stretches into a visible needle along that chain.

## Quick start

```bash
# Web only — splat assets are checked in, no Python required to view.
cd web && npm install && npm run dev
# then http://localhost:5173
```

To regenerate the splat assets from the model:

```bash
# One-time: Python 3.11 via pyenv + venv
pyenv install 3.11.11 --skip-existing
python3.11 -m venv pipeline/.venv
pipeline/.venv/bin/pip install -r pipeline/requirements.txt

make phase1     # downloads GPT-2 once, produces web/public/assets/*.bin
```

## Layout

```
token-splat/
  config.yaml            model id, PCA + colour + opacity knobs
  Makefile               make phase1 | make dev | make test
  pipeline/              Python encoder — reads wte, writes splat + neighbour assets
  web/                   Vite + TypeScript + three.js + Spark renderer
  web/public/assets/     ~7 MB of generated splat data (committed for convenience)
  .github/workflows/     GitHub Pages deploy on push to main
```

## Deploy

Auto-deployed to GitHub Pages on every push to `main` via `.github/workflows/pages.yml`. Source: `web/`, published: `web/dist/`.

## Credits

- **Model:** GPT-2 small (`openai-community/gpt2`), OpenAI, Modified MIT
- **Renderer:** [Spark](https://sparkjs.dev/) 3D-gaussian-splat backend for three.js (MIT)
- **Colour space:** OKLab (Björn Ottosson)
- **Prior art:** [TensorFlow Embedding Projector](https://projector.tensorflow.org) for the "browsable embedding" pattern; the glitch-token paper for what to look for in GPT-2's wte
