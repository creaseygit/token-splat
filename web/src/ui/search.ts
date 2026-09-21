/**
 * Search box: type text, we tokenise it against the vocabulary (longest-match,
 * case-preserving, GPT-2's leading-space convention accounted for), show the
 * token pieces, and fly the camera to the first one.
 *
 * Phase 1 uses a client-side longest-token match — good enough for whole words
 * and single tokens. Phase 3 wires the real GPT-2 tokeniser via transformers.js.
 */
import type { SceneHandle } from "../scene/scene";

export function mountSearch(handle: SceneHandle, host: HTMLElement): void {
  const wrap = document.createElement("div");
  wrap.id = "search";
  wrap.style.cssText = [
    "position:fixed", "top:12px", "right:12px", "z-index:15",
    "display:flex", "flex-direction:column", "gap:6px",
    "font-family:ui-sans-serif,system-ui,sans-serif", "font-size:13px",
  ].join(";");
  wrap.innerHTML = `
    <input id="q" type="text" placeholder="type a word, e.g.  Monday"
      style="padding:8px 10px;border-radius:8px;border:1px solid #232636;background:rgba(10,11,16,0.85);color:#e8ecf1;width:240px;outline:none"/>
    <div id="q-tokens" style="color:#8be2f5;font-family:ui-monospace,monospace;font-size:12px"></div>
  `;
  host.appendChild(wrap);

  // Build a lookup from token string -> id.
  const strToId = new Map<string, number>();
  for (let i = 0; i < handle.tokens.tokens.length; i++) {
    strToId.set(handle.tokens.tokens[i]!.s, i);
  }

  const input = wrap.querySelector<HTMLInputElement>("#q")!;
  const chips = wrap.querySelector<HTMLDivElement>("#q-tokens")!;

  input.addEventListener("input", () => {
    const q = input.value;
    if (!q) { chips.textContent = ""; return; }
    const pieces = greedyTokenise(q, strToId);
    if (pieces.length === 0) { chips.textContent = "no tokens found"; return; }
    chips.innerHTML = pieces.map((p) =>
      `<span style="display:inline-block;background:#141826;padding:2px 6px;border-radius:6px;margin:2px">${p.s.replace(/ /g, "·")}<span style="color:#6b7280;margin-left:4px">${p.id}</span></span>`
    ).join(" ");
    handle.focusToken(pieces[0]!.id);
  });
}

// Greedy longest-match tokeniser against GPT-2's vocab. GPT-2 prepends a
// space to words that follow whitespace (encoded as " word"); we match either
// with or without the leading space at position 0.
function greedyTokenise(text: string, vocab: Map<string, number>): { s: string; id: number }[] {
  const pieces: { s: string; id: number }[] = [];
  let i = 0;
  let firstToken = true;
  while (i < text.length) {
    // Try longest match first.
    let matched: { s: string; id: number } | null = null;
    const remaining = text.slice(i);
    const withSpace = !firstToken && !remaining.startsWith(" ") ? " " + remaining : remaining;
    for (const cand of [withSpace, remaining]) {
      for (let len = Math.min(cand.length, 20); len > 0; len--) {
        const piece = cand.slice(0, len);
        const id = vocab.get(piece);
        if (id !== undefined) { matched = { s: piece, id }; break; }
      }
      if (matched) break;
    }
    if (!matched) { i += 1; firstToken = false; continue; }
    pieces.push(matched);
    // Advance by the number of *original text* characters consumed
    // (matched.s may include a leading space that wasn't in text[i]).
    const consumed = matched.s.startsWith(" ") && !text.slice(i).startsWith(" ") ? matched.s.length - 1 : matched.s.length;
    i += Math.max(consumed, 1);
    firstToken = false;
  }
  return pieces;
}
