/**
 * About panel: explains the encoding + a legend of the character-class colours
 * used for base RGB. Colour swatches come from the same OKLab palette the
 * pipeline used, sent through tokens.json.
 */
import type { CharClass } from "../data/loader";

export function mountAbout(
  host: HTMLElement,
  meta: { topThreeVar: number; top12Var: number; top57Var: number; charClasses?: CharClass[] },
): void {
  const wrap = document.createElement("details");
  wrap.id = "about";
  wrap.open = false;
  wrap.style.cssText = [
    "position:fixed", "left:12px", "bottom:12px", "z-index:15",
    "background:rgba(10,11,16,0.9)", "color:#e8ecf1",
    "padding:10px 12px", "border-radius:10px",
    "font-family:ui-sans-serif,system-ui,sans-serif", "font-size:12px",
    "max-width:440px", "line-height:1.5",
    "max-height:80vh", "overflow-y:auto",
  ].join(";");

  const p3 = `${(meta.topThreeVar * 100).toFixed(1)}%`;
  const p12 = `${(meta.top12Var * 100).toFixed(1)}%`;
  const p57 = `${(meta.top57Var * 100).toFixed(1)}%`;

  const paletteRows = (meta.charClasses ?? []).map((c) => {
    const hex = `rgb(${c.rgb[0]},${c.rgb[1]},${c.rgb[2]})`;
    return `<tr>
      <td style="padding:2px 6px 2px 0">
        <span style="display:inline-block;width:14px;height:14px;border-radius:3px;background:${hex};box-shadow:0 0 0 1px rgba(255,255,255,0.15);vertical-align:middle"></span>
      </td>
      <td style="padding:2px 6px">${escapeHtml(c.name)}</td>
      <td style="padding:2px 0;color:#6b7280;text-align:right">${c.count.toLocaleString()}</td>
    </tr>`;
  }).join("");

  wrap.innerHTML = `
    <summary style="cursor:pointer;color:#8be2f5;list-style:none">what am I looking at?</summary>
    <div style="margin-top:8px">
      <p style="margin:0 0 6px">
        50 257 splats — one per GPT-2 token. Each parameter of the gaussian
        splat encodes a different structural signal about the token.
      </p>
      <table style="width:100%;border-collapse:collapse;margin:8px 0">
        <tr style="color:#94a3b8"><th align="left">splat parameter</th><th align="left">signal</th></tr>
        <tr><td>position</td><td>PC 1–3 of the wte</td></tr>
        <tr><td>ellipsoid shape + tilt</td><td>local PCA of the 8-NN neighbourhood</td></tr>
        <tr><td>base colour</td><td>character class (see legend below)</td></tr>
        <tr><td>view-dep. shimmer</td><td>PC 4-48 (high-D residual)</td></tr>
        <tr><td>opacity</td><td>mean cosine sim. to 8 NN (connectedness)</td></tr>
      </table>
      <p style="margin:6px 0">
        Explained variance — top 3: <b>${p3}</b> · top 12: <b>${p12}</b> · top 57: <b>${p57}</b>.
      </p>
      ${paletteRows ? `
      <div style="margin-top:10px;color:#94a3b8;font-weight:600">colour legend</div>
      <table style="border-collapse:collapse;margin-top:4px;font-family:ui-monospace,monospace;font-size:12px">${paletteRows}</table>` : ""}
      <p style="margin:10px 0 0; color:#6b7280; font-size:11px">
        Model: GPT-2 small (OpenAI, Modified MIT). Renderer:
        <a href="https://sparkjs.dev/" style="color:#8be2f5" target="_blank" rel="noopener">Spark</a>.
      </p>
    </div>
  `;
  host.appendChild(wrap);
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));
}
