/**
 * Persistent focus chip at the top-centre with a click-to-open dropdown
 * listing the 20 nearest neighbours in the full 768-dim wte. Selecting one
 * re-focuses; iterate to hop through the graph.
 */
import type { SceneHandle } from "../scene/scene";
import { renderTokenText } from "./hover-card";

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}

export function mountFocusLabel(handle: SceneHandle, host: HTMLElement): void {
  const wrap = document.createElement("div");
  wrap.id = "focus-label";
  wrap.style.cssText = [
    "position:fixed", "top:12px", "left:50%", "transform:translateX(-50%)",
    "z-index:15",
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace", "font-size:13px",
    "display:none",
  ].join(";");
  wrap.innerHTML = `
    <button id="fl-btn" aria-haspopup="listbox" aria-expanded="false" title="Open the 20 nearest neighbours"
      style="all:unset;cursor:pointer;background:rgba(10,11,16,0.92);color:#e8ecf1;padding:8px 14px;border-radius:999px;box-shadow:0 2px 12px rgba(0,0,0,0.4);letter-spacing:0.02em;display:inline-flex;align-items:center;gap:10px;border:1px solid #2a3346;transition:border-color 120ms,background 120ms">
      <span id="fl-inner"></span>
      <span aria-hidden="true" data-nearest-count
        style="background:#8be2f5;color:#0a0b10;font-weight:700;font-size:11px;padding:2px 8px;border-radius:999px;line-height:1.4">nearest 20 ▾</span>
    </button>
    <div id="fl-menu" role="listbox"
      style="display:none;position:absolute;left:50%;top:calc(100% + 6px);transform:translateX(-50%);background:rgba(10,11,16,0.95);border:1px solid #2a3346;border-radius:10px;min-width:260px;max-height:60vh;overflow-y:auto;box-shadow:0 8px 24px rgba(0,0,0,0.6);padding:6px 0"></div>
  `;
  host.appendChild(wrap);

  const btn = wrap.querySelector<HTMLButtonElement>("#fl-btn")!;
  const inner = wrap.querySelector<HTMLSpanElement>("#fl-inner")!;
  const menu = wrap.querySelector<HTMLDivElement>("#fl-menu")!;
  btn.addEventListener("mouseenter", () => { btn.style.borderColor = "#8be2f5"; btn.style.background = "rgba(20,24,38,0.98)"; });
  btn.addEventListener("mouseleave", () => { btn.style.borderColor = "#2a3346"; btn.style.background = "rgba(10,11,16,0.92)"; });

  function closeMenu(): void {
    menu.style.display = "none";
    btn.setAttribute("aria-expanded", "false");
  }
  function openMenu(id: number): void {
    const visibleNeigh = handle.neighboursOf(id).filter((nid) => handle.isVisible(nid));
    const rows = visibleNeigh.map((nid, i) => {
      const s = handle.tokens.tokens[nid]?.s ?? "?";
      return `<button role="option" data-id="${nid}"
        style="all:unset;display:flex;justify-content:space-between;gap:12px;cursor:pointer;padding:6px 12px;color:#e8ecf1;font-family:ui-monospace,monospace;font-size:12px;width:calc(100% - 24px)"
        onmouseover="this.style.background='#141826'"
        onmouseout="this.style.background=''">
        <span>${escapeHtml(renderTokenText(s))}</span>
        <span style="color:#6b7280">#${i + 1} · id ${nid}</span>
      </button>`;
    }).join("");
    // Update the badge count to match the actual visible-neighbour list.
    const badge = btn.querySelector<HTMLSpanElement>("[data-nearest-count]");
    if (badge) badge.textContent = `nearest ${visibleNeigh.length} ▾`;
    menu.innerHTML = rows || `<div style="padding:8px 12px;color:#6b7280;font-family:ui-monospace,monospace;font-size:12px">no visible neighbours</div>`;
    menu.style.display = "block";
    btn.setAttribute("aria-expanded", "true");
    menu.querySelectorAll<HTMLButtonElement>("button[data-id]").forEach((b) => {
      b.addEventListener("click", (ev) => {
        ev.preventDefault();
        const nid = Number(b.dataset.id);
        handle.focusToken(nid);
        // openMenu with the new id will be triggered by the focus listener.
      });
    });
  }
  btn.addEventListener("click", () => {
    const expanded = btn.getAttribute("aria-expanded") === "true";
    if (expanded) closeMenu();
    else if (currentId !== null) openMenu(currentId);
  });
  document.addEventListener("pointerdown", (e) => {
    if (!wrap.contains(e.target as Node)) closeMenu();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeMenu();
  });

  let currentId: number | null = null;
  handle.onFocusChange((id) => {
    currentId = id;
    if (id === null) { wrap.style.display = "none"; closeMenu(); return; }
    const tok = handle.tokens.tokens[id];
    if (!tok) { wrap.style.display = "none"; return; }
    wrap.style.display = "inline-block";
    const cls = handle.classOf(id);
    const palette = handle.tokens.char_class_palette ?? [];
    const clsInfo = palette[cls];
    const clsChip = clsInfo
      ? ` <span style="display:inline-flex;align-items:center;gap:4px;background:#141826;padding:1px 6px;border-radius:6px;font-size:11px;color:#94a3b8"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:rgb(${clsInfo.rgb[0]},${clsInfo.rgb[1]},${clsInfo.rgb[2]})"></span>${escapeHtml(clsInfo.name)}</span>`
      : "";
    inner.innerHTML = `<span style="color:#8be2f5">focus</span> <span style="color:#fff">${escapeHtml(renderTokenText(tok.s))}</span>${clsChip} <span style="color:#6b7280">id ${id}</span>`;
    // Keep the badge count truthful even when the menu is closed.
    const badge = btn.querySelector<HTMLSpanElement>("[data-nearest-count]");
    if (badge) {
      const n = handle.neighboursOf(id).filter((nid) => handle.isVisible(nid)).length;
      badge.textContent = `nearest ${n} ▾`;
    }
    // If the menu was open when the user picked a neighbour, refresh it
    // in-place with the new token's neighbours (hop through).
    if (btn.getAttribute("aria-expanded") === "true") openMenu(id);
  });
}
