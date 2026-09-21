/**
 * Static high-D scene. One splat per vocabulary token. Every free parameter
 * of a gaussian splat encodes a different set of PCA components of the raw
 * GPT-2 wte embedding table (see phase1_kf0.py for the exact mapping).
 *
 * No corpus, no layers, no scrubbing — the scene is one snapshot; the
 * dimensionality lives in the *appearance* of each splat.
 */
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PackedSplats, SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { loadNeighbours, loadSplats, loadStatic, loadTokens, neighboursOf, type Neighbours, type Splats, type StaticData, type TokensDoc } from "../data/loader";
import { makeMarker } from "./marker";

export type SceneHooks = {
  onHoverChange: (tokenId: number | null) => void;
  onReady: () => void;
  onFps: (fps: number) => void;
};

export type SceneHandle = {
  focusToken: (tokenId: number) => void;
  markToken: (tokenId: number | null) => void;
  setExplosion: (factor: number) => void;
  setRelationsDepth: (depth: 1 | 2 | 3) => void;      // 1 = normal, 2..3 = transitive
  setIsolate: (on: boolean) => void;                   // fade the non-related splats
  setCullFraction: (frac: number) => void;             // 0 = show all, 0.9 = hide bottom 90% by density
  setDisabledClasses: (ids: Set<number>) => void;      // per-class visibility mask
  classOf: (tokenId: number) => number;                // char class id for a splat
  isVisible: (tokenId: number) => boolean;             // honours cull / isolate / class filters
  onFocusChange: (cb: (tokenId: number | null) => void) => void;
  neighboursOf: (tokenId: number) => number[];
  tokens: TokensDoc;
  neighbourK: number;
  camera: THREE.PerspectiveCamera;
  dispose: () => void;
};

const NEIGH_K = 20;
const HOVER_NEIGH_K = 8;    // hover card shows first 8 of the stored 20
const HIT_RADIUS_PX = 12;

export async function buildScene(host: HTMLElement, hooks: SceneHooks): Promise<SceneHandle> {
  // BASE_URL is "/" in dev, "/token-splat/" on GitHub Pages — see vite.config.
  const base = import.meta.env.BASE_URL;
  const tokens = await loadTokens(`${base}assets/tokens.json`);
  const n = tokens.vocab_size;
  const [splats, staticData, neighbours] = await Promise.all([
    loadSplats(`${base}assets/splats.bin`, n),
    loadStatic(`${base}assets/static.bin`, n),
    loadNeighbours(`${base}assets/neighbours.bin`, n, NEIGH_K),
  ]);
  console.log(`[scene] loaded ${n.toLocaleString()} splats · encoding=${tokens.encoding ?? "n/a"}`);

  // ---- renderer ---------------------------------------------------------
  const width = host.clientWidth || window.innerWidth;
  const height = host.clientHeight || window.innerHeight;
  const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(width, height);
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, width / height, 0.01, 500);
  // Cloud has 99%-percentile radius of 1 at explosion 1×; base explosion is
  // 2× (see BASE_EXPLOSION), so the initial visible radius is ~2 and the
  // camera sits at 6.4 units to frame the whole cloud with margin.
  camera.position.set(0, 0, 6.4);
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;
  controls.minDistance = 0.5;
  controls.maxDistance = 200;

  // ---- build the SplatMesh ----------------------------------------------
  const packed = new PackedSplats({ maxSplats: n });
  const posV = new THREE.Vector3();
  const sclV = new THREE.Vector3();
  const quatV = new THREE.Quaternion();
  const colV = new THREE.Color();
  for (let i = 0; i < n; i++) {
    posV.set(splats.position[i * 3]!, splats.position[i * 3 + 1]!, splats.position[i * 3 + 2]!);
    sclV.set(splats.scale[i * 3]!,    splats.scale[i * 3 + 1]!,    splats.scale[i * 3 + 2]!);
    quatV.set(splats.quat[i * 4]!,    splats.quat[i * 4 + 1]!,     splats.quat[i * 4 + 2]!, splats.quat[i * 4 + 3]!);
    colV.setRGB(splats.rgb[i * 3]! / 255, splats.rgb[i * 3 + 1]! / 255, splats.rgb[i * 3 + 2]! / 255);
    packed.pushSplat(posV, sclV, quatV, staticData.opacity[i]! / 255, colV);
  }
  // SH view-dependent shimmer removed: it confused the character-class
  // colour coding (each splat's base RGB is meant to signal a class, but SH
  // adds a view-angle-dependent tint on top). Base colour only from here.
  const mesh = new SplatMesh({ packedSplats: packed, editable: true, raycastable: false });
  scene.add(mesh);

  // ---- current positions (mutated by explosion slider). Base explosion is
  // applied immediately so the user sees the pre-spread cloud on first paint.
  // -----------------------------------------------------------------------
  const curPos = new Float32Array(splats.position);
  // Match `BASE_EXPLOSION` in explosion.ts. Kept as a literal because scene.ts
  // is the ground truth for the initial camera framing.
  const BASE_EXPLOSION = 2.0;
  let explosion = BASE_EXPLOSION;
  // Isolate-mode visibility set. Declared here (hoisted above rewritePositions)
  // so the initial rewritePositions call doesn't hit the temporal dead zone.
  let visibleSet: Set<number> | null = null;
  const _p = new THREE.Vector3(), _s = new THREE.Vector3(), _q = new THREE.Quaternion(), _c = new THREE.Color();
  function rewritePositions(): void {
    // Positions scale linearly. Splat sizes STAY CONSTANT with explosion —
    // exploding truly opens up empty space between splats instead of just
    // zooming in. Anisotropy is a per-splat axis-ratio; it stays honest at
    // any size (a 10:1 needle stays a 10:1 needle even at 2px wide).
    //
    // Isolate mode: any splat outside `visibleSet` gets opacity 0 so the
    // cloud thins out to only the focused token's relation graph.
    const posS = explosion;
    const sclS = 1.0;
    const iso = visibleSet;
    for (let i = 0; i < n; i++) {
      curPos[i * 3    ] = splats.position[i * 3    ]! * posS;
      curPos[i * 3 + 1] = splats.position[i * 3 + 1]! * posS;
      curPos[i * 3 + 2] = splats.position[i * 3 + 2]! * posS;
      _p.set(curPos[i * 3]!, curPos[i * 3 + 1]!, curPos[i * 3 + 2]!);
      _s.set(splats.scale[i * 3]! * sclS, splats.scale[i * 3 + 1]! * sclS, splats.scale[i * 3 + 2]! * sclS);
      _q.set(splats.quat[i * 4]!, splats.quat[i * 4 + 1]!, splats.quat[i * 4 + 2]!, splats.quat[i * 4 + 3]!);
      _c.setRGB(splats.rgb[i * 3]! / 255, splats.rgb[i * 3 + 1]! / 255, splats.rgb[i * 3 + 2]! / 255);
      const opacity = (iso && !iso.has(i)) ? 0 : staticData.opacity[i]! / 255;
      packed.setSplat(i, _p, _s, _q, opacity, _c);
    }
    packed.needsUpdate = true;
    (mesh as unknown as { updateVersion: () => void }).updateVersion();
  }

  // Explosion tween state. When the slider moves we ease `explosion` from
  // `explosionFrom` to `explosionTo` over EXPLODE_TWEEN_MS, rewriting the
  // packed splats each frame during the tween.
  const EXPLODE_TWEEN_MS = 280;
  let explosionFrom = BASE_EXPLOSION;
  let explosionTo = BASE_EXPLOSION;
  let explosionStart = -1;
  function beginExplosionTween(target: number): void {
    explosionFrom = explosion;
    explosionTo = Math.max(0.1, target);
    explosionStart = performance.now();
  }
  function stepExplosionTween(now: number): void {
    if (explosionStart < 0) return;
    const t = Math.min(1, (now - explosionStart) / EXPLODE_TWEEN_MS);
    const s = t * t * (3 - 2 * t);
    explosion = explosionFrom + (explosionTo - explosionFrom) * s;
    rewritePositions();
    // Marker and camera target follow the focused splat as it moves outward.
    if (focusId !== null) {
      markerTo(focusId);
      controls.target.set(curPos[focusId * 3]!, curPos[focusId * 3 + 1]!, curPos[focusId * 3 + 2]!);
      updateNeighbourLines(focusId);
    }
    if (t >= 1) explosionStart = -1;
  }
  rewritePositions();   // apply BASE_EXPLOSION before the first frame

  const marker = makeMarker();
  scene.add(marker.group);
  const hoverMarker = makeMarker();     // subdued, follows the pointer
  scene.add(hoverMarker.group);

  // Neighbour-line rendering: three separate LineSegments, one per relation
  // depth. Depth 1 = the focused token's 20 direct neighbours; depth 2 = those
  // neighbours' neighbours (up to 400 more edges); depth 3 = one more hop
  // (up to 8000 edges). Depths render in decreasing brightness so structure
  // is legible at a glance.
  const LINE_K = NEIGH_K;
  const MAX_D2 = LINE_K * LINE_K;                     // 400
  const MAX_D3 = LINE_K * LINE_K * LINE_K;            // 8000
  function makeLines(maxEdges: number): { geom: THREE.BufferGeometry; obj: THREE.LineSegments; pos: Float32Array } {
    const pos = new Float32Array(maxEdges * 6);
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setDrawRange(0, 0);
    // All depths render identically — no opacity/colour degradation as the
    // hop count grows. Same treatment as depth 1 so every edge reads equally.
    const m = new THREE.LineBasicMaterial({
      color: 0xfff2a8, transparent: true, opacity: 0.55,
      depthTest: false, depthWrite: false,
    });
    const o = new THREE.LineSegments(g, m);
    o.renderOrder = 998;
    o.visible = false;
    return { geom: g, obj: o, pos };
  }
  const lines1 = makeLines(LINE_K);
  const lines2 = makeLines(MAX_D2);
  const lines3 = makeLines(MAX_D3);
  scene.add(lines1.obj); scene.add(lines2.obj); scene.add(lines3.obj);

  let relationsDepth: 1 | 2 | 3 = 1;
  let isolateOn = false;
  // Density-cull: slider value 0..1 = "cull fraction". 0 shows every splat;
  // 0.9 hides the least-embedded 90%. Threshold derives from the opacity
  // distribution (which encodes density) so the mapping is intuitive.
  let cullFrac = 0.0;
  let disabledClasses: Set<number> = new Set();
  const sortedOpacities = new Uint8Array(staticData.opacity).sort();
  // Per-splat class array (defaults to -1 for docs without a per-token cls field).
  const classes = new Int16Array(n);
  for (let i = 0; i < n; i++) classes[i] = (tokens.tokens[i]?.cls ?? -1) as number;
  function opacityThresholdFor(frac: number): number {
    const clamped = Math.max(0, Math.min(0.9999, frac));
    const idx = Math.min(sortedOpacities.length - 1, Math.floor(clamped * sortedOpacities.length));
    return sortedOpacities[idx]!;
  }

  // visibleSet = intersection of the two active filters. null = show all.
  function recomputeVisibleSet(): void {
    let iso: Set<number> | null = null;
    if (isolateOn && focusId !== null) {
      iso = new Set<number>([focusId]);
      const d1 = neighboursOf(neighbours, focusId);
      for (const id of d1) iso.add(id);
      if (relationsDepth >= 2) {
        for (const a of d1) for (const b of neighboursOf(neighbours, a)) iso.add(b);
      }
      if (relationsDepth >= 3) {
        const frontier: number[] = [];
        for (const a of d1) for (const b of neighboursOf(neighbours, a)) frontier.push(b);
        for (const a of frontier) for (const b of neighboursOf(neighbours, a)) iso.add(b);
      }
    }
    let cull: ((id: number) => boolean) | null = null;
    if (cullFrac > 0) {
      const thresh = opacityThresholdFor(cullFrac);
      cull = (id) => staticData.opacity[id]! >= thresh;
    }
    let classFilter: ((id: number) => boolean) | null = null;
    if (disabledClasses.size > 0) {
      classFilter = (id) => !disabledClasses.has(classes[id]!);
    }
    const anyFilter = iso || cull || classFilter;
    if (!anyFilter) { visibleSet = null; return; }
    const combined = (id: number): boolean =>
      (iso ? iso.has(id) : true) &&
      (cull ? cull(id) : true) &&
      (classFilter ? classFilter(id) : true);
    const set = new Set<number>();
    const source = iso ?? { has: (_i: number) => true, [Symbol.iterator]: function* () { for (let i = 0; i < n; i++) yield i; } };
    for (const i of source as Iterable<number>) if (combined(i)) set.add(i);
    visibleSet = set;
  }

  function writeEdge(pos: Float32Array, idx: number, a: number, b: number): void {
    const base = idx * 6;
    pos[base + 0] = curPos[a * 3]!;
    pos[base + 1] = curPos[a * 3 + 1]!;
    pos[base + 2] = curPos[a * 3 + 2]!;
    pos[base + 3] = curPos[b * 3]!;
    pos[base + 4] = curPos[b * 3 + 1]!;
    pos[base + 5] = curPos[b * 3 + 2]!;
  }

  function updateNeighbourLines(id: number | null): void {
    if (id === null) {
      lines1.obj.visible = false; lines2.obj.visible = false; lines3.obj.visible = false;
      return;
    }
    // Only draw an edge when BOTH endpoints are currently visible — otherwise
    // lines dangle out to invisible splats after culling / class-toggle /
    // isolate. `iso` is the current combined visibility set.
    const iso = visibleSet;
    const shown = (x: number): boolean => !iso || iso.has(x);
    const visited = new Set<number>([id]);
    // Depth 1
    const d1 = neighboursOf(neighbours, id);
    let e1 = 0;
    for (const nid of d1) {
      if (shown(id) && shown(nid)) writeEdge(lines1.pos, e1++, id, nid);
      visited.add(nid);
    }
    lines1.geom.setDrawRange(0, e1 * 2);
    lines1.geom.attributes.position!.needsUpdate = true;
    lines1.obj.visible = e1 > 0;

    // Depth 2
    let e2 = 0;
    if (relationsDepth >= 2) {
      const d2Set = new Set<number>();
      for (const a of d1) {
        for (const b of neighboursOf(neighbours, a)) {
          if (visited.has(b)) continue;
          const key = a < b ? a * n + b : b * n + a;
          if (d2Set.has(key)) continue;
          d2Set.add(key);
          if (e2 >= MAX_D2) break;
          if (shown(a) && shown(b)) writeEdge(lines2.pos, e2++, a, b);
        }
        if (e2 >= MAX_D2) break;
      }
    }
    lines2.geom.setDrawRange(0, e2 * 2);
    lines2.geom.attributes.position!.needsUpdate = true;
    lines2.obj.visible = relationsDepth >= 2 && e2 > 0;

    // Depth 3
    let e3 = 0;
    if (relationsDepth >= 3) {
      const d3Set = new Set<number>();
      const front: number[] = [];
      for (const a of d1) for (const b of neighboursOf(neighbours, a)) front.push(b);
      for (const a of front) {
        for (const b of neighboursOf(neighbours, a)) {
          if (visited.has(b) || a === b) continue;
          const key = a < b ? a * n + b : b * n + a;
          if (d3Set.has(key)) continue;
          d3Set.add(key);
          if (e3 >= MAX_D3) break;
          if (shown(a) && shown(b)) writeEdge(lines3.pos, e3++, a, b);
        }
        if (e3 >= MAX_D3) break;
      }
    }
    lines3.geom.setDrawRange(0, e3 * 2);
    lines3.geom.attributes.position!.needsUpdate = true;
    lines3.obj.visible = relationsDepth >= 3 && e3 > 0;
  }

  // ---- picking: screen-space NN within 12 px ----------------------------
  let hoverId: number | null = null;
  let focusId: number | null = null;
  const projected = new THREE.Vector3();

  // ---- Camera target tween ---------------------------------------------
  // OrbitControls snaps `controls.target` on set — we lerp it over ~380 ms
  // with smoothstep easing so re-focus feels like a fly-to instead of a pop.
  const TWEEN_MS = 380;
  const tweenFrom = new THREE.Vector3();
  const tweenTo = new THREE.Vector3();
  let tweenStart = -1;
  function beginTweenTo(x: number, y: number, z: number): void {
    tweenFrom.copy(controls.target);
    tweenTo.set(x, y, z);
    tweenStart = performance.now();
  }
  function stepTween(now: number): void {
    if (tweenStart < 0) return;
    const t = Math.min(1, (now - tweenStart) / TWEEN_MS);
    const s = t * t * (3 - 2 * t);                    // smoothstep
    controls.target.lerpVectors(tweenFrom, tweenTo, s);
    if (t >= 1) tweenStart = -1;
  }

  function pickAt(clientX: number, clientY: number): number | null {
    const rect = renderer.domElement.getBoundingClientRect();
    const targetX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const targetY = -(((clientY - rect.top) / rect.height) * 2 - 1);
    const hitR2 = (HIT_RADIUS_PX / rect.width * 2) ** 2 + (HIT_RADIUS_PX / rect.height * 2) ** 2;
    const m = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
    m.multiply(mesh.matrixWorld);
    let bestId = -1;
    let bestZ = Infinity;
    // In isolate mode, invisible splats also shouldn't be pickable.
    const iso = visibleSet;
    for (let i = 0; i < n; i++) {
      if (iso && !iso.has(i)) continue;
      projected.set(curPos[i * 3]!, curPos[i * 3 + 1]!, curPos[i * 3 + 2]!);
      projected.applyMatrix4(m);
      if (projected.z < -1 || projected.z > 1) continue;
      const dx = projected.x - targetX;
      const dy = projected.y - targetY;
      const d2 = dx * dx + dy * dy;
      if (d2 < hitR2 && projected.z < bestZ) { bestZ = projected.z; bestId = i; }
    }
    return bestId >= 0 ? bestId : null;
  }

  function markerTo(id: number): void {
    marker.moveTo(curPos[id * 3]!, curPos[id * 3 + 1]!, curPos[id * 3 + 2]!);
  }
  function hoverMarkerTo(id: number | null): void {
    if (id === null) { hoverMarker.hide(); return; }
    hoverMarker.moveTo(curPos[id * 3]!, curPos[id * 3 + 1]!, curPos[id * 3 + 2]!);
  }

  function onPointerMove(ev: PointerEvent): void {
    const id = pickAt(ev.clientX, ev.clientY);
    if (id !== hoverId) {
      hoverId = id;
      hooks.onHoverChange(id);
      // Focus marker stays on the focused splat; hover marker follows pointer.
      hoverMarkerTo(id === focusId ? null : id);
    }
  }
  renderer.domElement.addEventListener("pointermove", onPointerMove);

  // Click a splat -> centre orbit target and pull the camera in.
  let downXY: [number, number] | null = null;
  renderer.domElement.addEventListener("pointerdown", (e) => { downXY = [e.clientX, e.clientY]; });
  renderer.domElement.addEventListener("pointerup", (e) => {
    if (!downXY) return;
    const dx = e.clientX - downXY[0];
    const dy = e.clientY - downXY[1];
    downXY = null;
    // Ignore drags — only treat as a click if the pointer barely moved.
    if (dx * dx + dy * dy > 25) return;
    const id = pickAt(e.clientX, e.clientY);
    if (id !== null) publicHandle.focusToken(id);
    else publicHandle.markToken(null);                // click on empty space clears
  });

  // ---- resize -----------------------------------------------------------
  const onResize = () => {
    const w = host.clientWidth || window.innerWidth;
    const h = host.clientHeight || window.innerHeight;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  window.addEventListener("resize", onResize);

  // ---- render loop ------------------------------------------------------
  let frames = 0;
  let last = performance.now();
  let fpsAccum = 0;
  let disposed = false;
  const tick = () => {
    if (disposed) return;
    const now = performance.now();
    const dt = now - last;
    last = now;
    fpsAccum = fpsAccum * 0.9 + (1000 / dt) * 0.1;
    frames++;
    if (frames % 30 === 0) hooks.onFps(fpsAccum);
    stepTween(now);
    stepExplosionTween(now);
    controls.update();
    marker.update(camera, now);
    hoverMarker.update(camera, now);
    renderer.render(scene, camera);
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);

  hooks.onReady();

  const focusListeners: ((id: number | null) => void)[] = [];
  function afterFilterChange(): void {
    if (focusId !== null) updateNeighbourLines(focusId);
    // Re-notify listeners so UIs (focus label badge, hover card if reopened)
    // can recompute counts against the new visible set.
    for (const cb of focusListeners) cb(focusId);
  }
  const publicHandle: SceneHandle = {
    tokens,
    neighbourK: NEIGH_K,
    camera,
    focusToken(tokenId: number): void {
      focusId = tokenId;
      beginTweenTo(curPos[tokenId * 3]!, curPos[tokenId * 3 + 1]!, curPos[tokenId * 3 + 2]!);
      markerTo(tokenId);
      updateNeighbourLines(tokenId);
      hoverMarkerTo(hoverId === tokenId ? null : hoverId);
      if (isolateOn) { recomputeVisibleSet(); rewritePositions(); }
      for (const cb of focusListeners) cb(tokenId);
    },
    markToken(tokenId: number | null): void {
      focusId = tokenId;
      if (tokenId === null) { marker.hide(); updateNeighbourLines(null); }
      else { markerTo(tokenId); updateNeighbourLines(tokenId); }
      if (isolateOn) { recomputeVisibleSet(); rewritePositions(); }
      for (const cb of focusListeners) cb(tokenId);
    },
    setExplosion(factor: number): void {
      beginExplosionTween(factor);
    },
    setRelationsDepth(depth: 1 | 2 | 3): void {
      relationsDepth = depth;
      recomputeVisibleSet();
      if (isolateOn) rewritePositions();
      if (focusId !== null) updateNeighbourLines(focusId);
    },
    setIsolate(on: boolean): void {
      isolateOn = on;
      recomputeVisibleSet();
      rewritePositions();
      afterFilterChange();
    },
    setCullFraction(frac: number): void {
      cullFrac = frac;
      recomputeVisibleSet();
      rewritePositions();
      afterFilterChange();
    },
    setDisabledClasses(ids: Set<number>): void {
      disabledClasses = ids;
      recomputeVisibleSet();
      rewritePositions();
      afterFilterChange();
    },
    classOf(tokenId: number): number { return classes[tokenId]!; },
    isVisible(tokenId: number): boolean { return !visibleSet || visibleSet.has(tokenId); },
    onFocusChange(cb) { focusListeners.push(cb); },
    neighboursOf(tokenId: number): number[] {
      return neighboursOf(neighbours, tokenId);
    },
    dispose(): void {
      disposed = true;
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("resize", onResize);
      renderer.dispose();
    },
  };
  return publicHandle;
}
