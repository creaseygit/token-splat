/**
 * "Explode" slider — scales every splat's position outward from the origin.
 *
 * The user-facing scale is remapped: displayed 1× is what actually renders as
 * the default "already-spread-apart" view, and the slider caps at displayed
 * 4×. Internally we multiply by BASE_EXPLOSION to get the raw factor that
 * setExplosion consumes. So displayed 1× → internal 5×, displayed 4× → 20×.
 * Splat sizes and colours are unchanged as you slide; only positions scale.
 */
export const BASE_EXPLOSION = 2.0;

export function mountExplosion(host: HTMLElement, onChange: (factor: number) => void): void {
  const wrap = document.createElement("div");
  wrap.id = "explosion";
  wrap.style.cssText = [
    "position:fixed", "left:50%", "bottom:16px", "transform:translateX(-50%)",
    "z-index:15", "display:flex", "align-items:center", "gap:12px",
    "background:rgba(10,11,16,0.85)", "color:#e8ecf1",
    "padding:8px 14px", "border-radius:999px",
    "font-family:ui-sans-serif,system-ui,sans-serif", "font-size:13px",
    "min-width:min(420px, 90vw)",
  ].join(";");
  wrap.innerHTML = `
    <label style="font-family:ui-monospace,monospace;color:#8be2f5;min-width:96px" for="explode">explode <span id="e-num">1.00</span>×</label>
    <input id="explode" type="range" min="1" max="12" step="0.05" value="1"
      style="flex:1;accent-color:#8be2f5" aria-label="inter-splat spacing"/>
  `;
  host.appendChild(wrap);

  const slider = wrap.querySelector<HTMLInputElement>("#explode")!;
  const num = wrap.querySelector<HTMLSpanElement>("#e-num")!;

  // The scene tweens explosion smoothly, so we send updates immediately on
  // every input event — no UI-level throttle needed.
  slider.addEventListener("input", () => {
    const v = parseFloat(slider.value);
    num.textContent = v.toFixed(2);
    onChange(v * BASE_EXPLOSION);
  });
}
