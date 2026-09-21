/**
 * Density-cull slider. Hides the least-embedded splats (opacity drives what
 * "embedded" means: it's the mean cosine similarity to the 20 NN). Culled
 * splats are also non-pickable, so hovering only lands on visible tokens.
 */
import type { SceneHandle } from "../scene/scene";

export function mountCull(host: HTMLElement, handle: SceneHandle): void {
  const V = handle.tokens.vocab_size;
  const wrap = document.createElement("div");
  wrap.id = "cull";
  wrap.style.cssText = [
    "position:fixed", "right:12px", "bottom:118px", "z-index:15",
    "display:flex", "align-items:center", "gap:10px",
    "background:rgba(10,11,16,0.9)", "color:#e8ecf1",
    "padding:8px 14px", "border-radius:999px",
    "font-family:ui-sans-serif,system-ui,sans-serif", "font-size:12px",
    "min-width:min(440px, 90vw)",
  ].join(";");
  wrap.innerHTML = `
    <label title="Only keep the N% most densely-embedded splats (highest mean cosine similarity to their 20 nearest neighbours). Culled splats are hidden and non-pickable."
      style="font-family:ui-monospace,monospace;color:#8be2f5;min-width:210px;cursor:help" for="cull-r">
      most-embedded <span id="cull-num">100</span>% · <span id="cull-count">${V.toLocaleString()}</span>
    </label>
    <input id="cull-r" type="range" min="0" max="99.99" step="0.01" value="0"
      style="flex:1;accent-color:#8be2f5" aria-label="density cull"/>
  `;
  host.appendChild(wrap);

  const slider = wrap.querySelector<HTMLInputElement>("#cull-r")!;
  const num = wrap.querySelector<HTMLSpanElement>("#cull-num")!;
  const count = wrap.querySelector<HTMLSpanElement>("#cull-count")!;
  const fmt = (keep: number) =>
    keep >= 10 ? keep.toFixed(0) :
    keep >= 1  ? keep.toFixed(1) : keep.toFixed(2);
  slider.addEventListener("input", () => {
    const cullPercent = parseFloat(slider.value);       // 0..99.99
    const keepPercent = 100 - cullPercent;
    num.textContent = fmt(keepPercent);
    count.textContent = Math.max(1, Math.round(V * keepPercent / 100)).toLocaleString();
    handle.setCullFraction(cullPercent / 100);
  });
}
