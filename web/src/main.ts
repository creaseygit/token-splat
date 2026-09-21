/**
 * Entry point. Boots the single static scene: 50 257 GPT-2 tokens rendered
 * as gaussian splats where every free parameter of the splat encodes a
 * different PCA component of the raw wte embedding table.
 */
import { buildScene, type SceneHandle } from "./scene/scene";
import { mountHoverCard } from "./ui/hover-card";
import { mountSearch } from "./ui/search";
import { mountAbout } from "./ui/about";
import { mountFocusLabel } from "./ui/focus-label";
import { mountExplosion } from "./ui/explosion";
import { mountRelations } from "./ui/relations";
import { mountCull } from "./ui/cull";
import { mountAnisotropy } from "./ui/anisotropy";

function hasWebGL2(): boolean {
  const c = document.createElement("canvas");
  return !!(c.getContext("webgl2"));
}

async function boot(): Promise<void> {
  const stats = document.getElementById("stats") as HTMLDivElement;
  const phase = document.getElementById("phase") as HTMLDivElement;
  phase.textContent = "token space · high-D encoding";
  if (!hasWebGL2()) { document.body.classList.add("no-webgl2"); return; }
  const stage = document.getElementById("stage") as HTMLDivElement;
  stats.textContent = "loading assets…";
  let hoverUpdate: (id: number | null) => void = () => {};
  let handle: SceneHandle | undefined;
  handle = await buildScene(stage, {
    onHoverChange: (id) => hoverUpdate(id),
    onReady: () => { stats.textContent = "orbit, hover, or search"; },
    onFps: (fps) => { stats.textContent = `${fps.toFixed(0)} fps · ${handle?.tokens.vocab_size.toLocaleString()} splats`; },
  });
  hoverUpdate = mountHoverCard(handle, document.body);
  mountSearch(handle, document.body);
  mountFocusLabel(handle, document.body);
  mountExplosion(document.body, (f) => handle.setExplosion(f));
  mountRelations(document.body, handle);
  mountCull(document.body, handle);
  mountAnisotropy(document.body, handle);
  // Default: only the three most-populated character classes are enabled.
  // The tail categories (punctuation, byte fragments, whitespace, non-ASCII)
  // are mostly noise for the overview and can be toggled back on from the
  // About-panel legend.
  const palette = handle.tokens.char_class_palette ?? [];
  const topThreeIds = [...palette].sort((a, b) => b.count - a.count).slice(0, 3).map((c) => c.id);
  const initialDisabled = new Set(palette.map((c) => c.id).filter((id) => !topThreeIds.includes(id)));
  handle.setDisabledClasses(initialDisabled);
  mountAbout(document.body, {
    topThreeVar: handle.tokens.explained_variance_top3
      ? handle.tokens.explained_variance_top3.reduce((a, b) => a + b, 0)
      : 0,
    top12Var: handle.tokens.explained_variance_top12 ?? 0,
    top57Var: handle.tokens.explained_variance_top57 ?? 0,
    charClasses: handle.tokens.char_class_palette,
    initialDisabled,
    onToggleClass: (disabled) => handle.setDisabledClasses(disabled),
  });
  // Initial view is a zoom-to-fit of the whole cloud (camera default frames
  // the 99%-radius sphere). No pre-selection so the eye can take the whole
  // vector space in first.
}

boot().catch((err) => {
  console.error(err);
  const stats = document.getElementById("stats");
  if (stats) stats.textContent = `error: ${(err as Error).message}`;
});
