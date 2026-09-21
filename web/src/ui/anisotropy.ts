/**
 * Anisotropy-amp slider. Exaggerates each splat's per-axis ratios while
 * preserving its overall size — so needles get LONGER and THINNER as the
 * slider climbs, spheres stay spheres. 1.0 = raw pipeline output.
 */
import type { SceneHandle } from "../scene/scene";

export function mountAnisotropy(host: HTMLElement, handle: SceneHandle): void {
  const wrap = document.createElement("div");
  wrap.id = "anisotropy";
  wrap.style.cssText = [
    "position:fixed", "left:12px", "top:60px", "z-index:15",
    "display:flex", "align-items:center", "gap:10px",
    "background:rgba(10,11,16,0.9)", "color:#e8ecf1",
    "padding:8px 14px", "border-radius:999px",
    "font-family:ui-sans-serif,system-ui,sans-serif", "font-size:12px",
    "min-width:min(340px, 80vw)",
  ].join(";");
  wrap.innerHTML = `
    <label title="Exaggerate the ellipsoid shape derived from each token's neighbourhood. 1× = raw shape from the pipeline; 6× stretches needles into extreme spindles while keeping spheres spherical."
      style="font-family:ui-monospace,monospace;color:#8be2f5;min-width:150px;cursor:help" for="aniso-r">
      anisotropy amp <span id="aniso-num">1.0</span>×
    </label>
    <input id="aniso-r" type="range" min="1" max="6" step="0.05" value="1"
      style="flex:1;accent-color:#8be2f5" aria-label="anisotropy amp"/>
  `;
  host.appendChild(wrap);

  const slider = wrap.querySelector<HTMLInputElement>("#aniso-r")!;
  const num = wrap.querySelector<HTMLSpanElement>("#aniso-num")!;
  slider.addEventListener("input", () => {
    const v = parseFloat(slider.value);
    num.textContent = v.toFixed(2);
    handle.setAnisotropyAmp(v);
  });
}
