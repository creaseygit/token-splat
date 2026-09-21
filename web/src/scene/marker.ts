/**
 * Selection marker: a thin additive-blend ring that sits at the focused
 * token's position, plus a soft point at its centre. Rendered on top of the
 * splat cloud so the selected splat is unambiguous even when it's a rare,
 * low-opacity token buried in a dense region.
 *
 * We use a ring (facing the camera) rather than a coloured splat because:
 *  - it does not compete for the same rendering pass as Spark;
 *  - it stays perfectly circular from any angle;
 *  - it does not obscure the splat itself — the middle stays empty.
 */
import * as THREE from "three";

export type Marker = {
  group: THREE.Group;
  moveTo: (x: number, y: number, z: number) => void;
  hide: () => void;
  update: (camera: THREE.PerspectiveCamera, time: number) => void;
};

export function makeMarker(): Marker {
  const group = new THREE.Group();
  group.visible = false;
  group.renderOrder = 999;

  const ringGeom = new THREE.RingGeometry(0.02, 0.024, 48);
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0xfff2a8,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
  const ring = new THREE.Mesh(ringGeom, ringMat);
  ring.renderOrder = 999;
  group.add(ring);

  // A larger, dimmer halo that pulses gently — attention-grabbing without
  // being a strobe. Respect prefers-reduced-motion (spec §Interactions).
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const haloGeom = new THREE.RingGeometry(0.04, 0.05, 48);
  const haloMat = new THREE.MeshBasicMaterial({
    color: 0xffd166,
    transparent: true,
    opacity: 0.4,
    depthTest: false,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
  const halo = new THREE.Mesh(haloGeom, haloMat);
  halo.renderOrder = 999;
  group.add(halo);

  // A crosshair that fades away as you look at it — helps you find the
  // marker at first, then gets out of the way.
  const crosshair = new THREE.Group();
  const lineMat = new THREE.LineBasicMaterial({
    color: 0xfff2a8, transparent: true, opacity: 0.6,
    depthTest: false, depthWrite: false,
  });
  const buildLine = (a: THREE.Vector3, b: THREE.Vector3) => {
    const g = new THREE.BufferGeometry().setFromPoints([a, b]);
    const l = new THREE.Line(g, lineMat);
    l.renderOrder = 999;
    return l;
  };
  const T = 0.005, L = 0.06;
  crosshair.add(buildLine(new THREE.Vector3(-L, 0, 0), new THREE.Vector3(-T, 0, 0)));
  crosshair.add(buildLine(new THREE.Vector3(T, 0, 0), new THREE.Vector3(L, 0, 0)));
  crosshair.add(buildLine(new THREE.Vector3(0, -L, 0), new THREE.Vector3(0, -T, 0)));
  crosshair.add(buildLine(new THREE.Vector3(0, T, 0), new THREE.Vector3(0, L, 0)));
  group.add(crosshair);

  return {
    group,
    moveTo(x, y, z) {
      group.position.set(x, y, z);
      group.visible = true;
    },
    hide() { group.visible = false; },
    update(camera, time) {
      if (!group.visible) return;
      // Face the camera so the ring always reads as a circle.
      group.quaternion.copy(camera.quaternion);
      if (!reduced) {
        // Gentle 0.9..1.1 pulse over ~1.4 s so the eye can catch the ring.
        const pulse = 1.0 + 0.1 * Math.sin(time * 0.0045);
        halo.scale.setScalar(pulse);
      }
    },
  };
}
