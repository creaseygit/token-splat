/**
 * "Show relations" three-way switch: 1 = direct 20 neighbours only,
 * 2 = also draw the neighbours' neighbours (up to 400 more edges),
 * 3 = one more hop (up to 8 000). Deeper hops render dimmer so the local
 * structure remains legible.
 */
import type { SceneHandle } from "../scene/scene";

export function mountRelations(host: HTMLElement, handle: SceneHandle): void {
  const wrap = document.createElement("div");
  wrap.id = "relations";
  wrap.style.cssText = [
    "position:fixed", "right:12px", "bottom:74px", "z-index:15",
    "display:flex", "align-items:center", "gap:8px",
    "background:rgba(10,11,16,0.9)", "color:#e8ecf1",
    "padding:8px 12px", "border-radius:999px",
    "font-family:ui-sans-serif,system-ui,sans-serif", "font-size:12px",
  ].join(";");
  wrap.innerHTML = `
    <span style="color:#94a3b8">relations</span>
    <button data-d="1" style="all:unset;cursor:pointer;padding:3px 10px;border-radius:999px;background:#8be2f5;color:#0a0b10;font-weight:600">1</button>
    <button data-d="2" style="all:unset;cursor:pointer;padding:3px 10px;border-radius:999px;color:#8be2f5;border:1px solid #2a3346">2</button>
    <button data-d="3" style="all:unset;cursor:pointer;padding:3px 10px;border-radius:999px;color:#8be2f5;border:1px solid #2a3346">3</button>
    <span style="width:1px;height:14px;background:#2a3346"></span>
    <button id="isolate-btn" title="Hide every splat not in the current relation set"
      style="all:unset;cursor:pointer;padding:3px 10px;border-radius:999px;color:#8be2f5;border:1px solid #2a3346">isolate</button>
  `;
  host.appendChild(wrap);

  const btns = Array.from(wrap.querySelectorAll<HTMLButtonElement>("button[data-d]"));
  const iso = wrap.querySelector<HTMLButtonElement>("#isolate-btn")!;
  function highlight(d: number): void {
    for (const b of btns) {
      const active = Number(b.dataset.d) === d;
      b.style.background = active ? "#8be2f5" : "transparent";
      b.style.color = active ? "#0a0b10" : "#8be2f5";
      b.style.fontWeight = active ? "600" : "400";
      b.style.border = active ? "none" : "1px solid #2a3346";
    }
  }
  function paintIso(on: boolean): void {
    iso.style.background = on ? "#8be2f5" : "transparent";
    iso.style.color = on ? "#0a0b10" : "#8be2f5";
    iso.style.fontWeight = on ? "600" : "400";
    iso.style.border = on ? "none" : "1px solid #2a3346";
  }
  for (const b of btns) {
    b.addEventListener("click", () => {
      const d = Number(b.dataset.d) as 1 | 2 | 3;
      handle.setRelationsDepth(d);
      highlight(d);
    });
  }
  let isoOn = false;
  iso.addEventListener("click", () => {
    isoOn = !isoOn;
    handle.setIsolate(isoOn);
    paintIso(isoOn);
  });
  highlight(1);
  paintIso(false);
}
