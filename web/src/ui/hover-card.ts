/**
 * Hover card: shows the hovered token and its top-8 neighbours.
 *
 * Tapping a neighbour flies the camera to it (Scene.focusToken).
 * The card follows the pointer with a small offset; it hides on empty hover.
 */
import type { SceneHandle } from "../scene/scene";

// GPT-2 tokenizer strings can be arbitrary UTF-8, including bytes that are
// fragments of a multi-byte codepoint (byte-level BPE). Those render as
// tofu (❓) in most fonts, so we surface them as hex per the spec's honesty
// rule. Ordinary readable Unicode passes through unchanged.
export function renderTokenText(s: string): string {
  if (!s) return "∅";
  // Encode each char: printable ASCII + normal Unicode kept; control chars,
  // C1 controls (0x80–0x9F) and lone byte-fragments shown as \xNN. Newline
  // and tab get their conventional escape forms.
  let out = "";
  for (const ch of s) {
    const cp = ch.codePointAt(0)!;
    if (ch === " ")            { out += "·"; continue; }
    if (ch === "\n")           { out += "\\n"; continue; }
    if (ch === "\t")           { out += "\\t"; continue; }
    if (ch === "\r")           { out += "\\r"; continue; }
    if (cp < 0x20 || cp === 0x7F || (cp >= 0x80 && cp <= 0x9F)) {
      out += `\\x${cp.toString(16).padStart(2, "0").toUpperCase()}`;
      continue;
    }
    // U+FFFD is the Python decoder's stand-in for a byte fragment that isn't
    // valid UTF-8. Show it as `⟨byte⟩` so the reader knows it's a fragment.
    if (cp === 0xFFFD) { out += "⟨byte⟩"; continue; }
    out += ch;
  }
  return out;
}

export function mountHoverCard(handle: SceneHandle, container: HTMLElement): (id: number | null) => void {
  const card = document.createElement("div");
  card.id = "hover-card";
  card.style.cssText = [
    "position:fixed", "pointer-events:auto", "z-index:20",
    "background:rgba(10,11,16,0.92)", "color:#e8ecf1",
    "padding:10px 12px", "border-radius:10px",
    "font-family:ui-monospace,SFMono-Regular,Menlo,monospace", "font-size:12px",
    "min-width:180px", "max-width:280px",
    "box-shadow:0 4px 20px rgba(0,0,0,0.5)",
    "display:none",
  ].join(";");
  container.appendChild(card);

  let lastId: number | null = null;
  let px = 0, py = 0;
  window.addEventListener("pointermove", (e) => { px = e.clientX; py = e.clientY; place(); });

  function place() {
    if (card.style.display === "none") return;
    const w = card.offsetWidth, h = card.offsetHeight;
    const vw = window.innerWidth, vh = window.innerHeight;
    let x = px + 14, y = py + 14;
    if (x + w > vw - 8) x = px - 14 - w;
    if (y + h > vh - 8) y = py - 14 - h;
    card.style.left = `${x}px`;
    card.style.top = `${y}px`;
  }

  function render(id: number | null) {
    if (id === lastId) return;
    lastId = id;
    if (id === null) { card.style.display = "none"; return; }
    const tok = handle.tokens.tokens[id]!;
    // Only list visible neighbours; take up to 8 of those.
    const neigh = handle.neighboursOf(id).filter((nid) => handle.isVisible(nid)).slice(0, 8);
    const rows = neigh.map((nid) => {
      const s = handle.tokens.tokens[nid]?.s ?? "?";
      return `<button data-id="${nid}" style="all:unset;cursor:pointer;color:#8be2f5;display:block;padding:2px 0">${escapeHtml(renderTokenText(s))}</button>`;
    }).join("");
    const flag = tok.sparse ? ' <span style="color:#f59e0b">sparse</span>' : "";
    // Class chip: matches the swatch colour from the legend.
    const cls = handle.classOf(id);
    const palette = handle.tokens.char_class_palette ?? [];
    const clsInfo = palette[cls];
    const clsChip = clsInfo
      ? `<span style="display:inline-flex;align-items:center;gap:4px;background:#141826;padding:1px 6px;border-radius:6px;font-size:11px;color:#94a3b8"><span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:rgb(${clsInfo.rgb[0]},${clsInfo.rgb[1]},${clsInfo.rgb[2]})"></span>${escapeHtml(clsInfo.name)}</span>`
      : "";
    card.innerHTML = `
      <div style="font-size:13px;color:#fff;font-weight:600;margin-bottom:6px">${escapeHtml(renderTokenText(tok.s))}${flag}</div>
      <div style="color:#6b7280;margin-bottom:6px">id ${id} · ${clsChip} · nearest ${neigh.length}</div>
      <div>${rows}</div>
    `;
    card.querySelectorAll<HTMLButtonElement>("button[data-id]").forEach((btn) => {
      btn.addEventListener("click", (ev) => {
        ev.preventDefault();
        const nid = Number(btn.dataset.id);
        handle.focusToken(nid);
      });
    });
    card.style.display = "block";
    place();
  }

  return render;
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}
