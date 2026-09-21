/**
 * Asset loaders for the static high-D splat scene.
 *
 * Binary layouts must match `pipeline/common.py`:
 *   splats.bin — per splat 24 bytes:
 *      pos    3 × float16
 *      scale  3 × float16
 *      quat   4 × float16    (x, y, z, w — unit)
 *      rgb    3 × uint8
 *      pad    1 × uint8
 *   static.bin      opacity(u8 × N) then flags(u8 × N)
 *   neighbours.bin  top-k neighbour ids (u16 × k × N)
 *   sh_00.bin       band-1..3 int8 blocks, prefixed with 3 float32 per-band scales
 *   tokens.json     see phase1_kf0.py
 */

export type Splats = {
  n: number;
  position: Float32Array;   // n*3
  scale: Float32Array;      // n*3
  quat: Float32Array;       // n*4 (x, y, z, w)
  rgb: Uint8Array;          // n*3
};

export type StaticData = {
  n: number;
  opacity: Uint8Array;
  flags: Uint8Array;
};

export type CharClass = {
  id: number;
  name: string;
  rgb: [number, number, number];
  count: number;
};

export type TokensDoc = {
  vocab_size: number;
  encoding?: string;
  num_layers?: number;
  phase: number;
  explained_variance_top3?: [number, number, number];
  explained_variance_top12?: number;
  explained_variance_top57?: number;
  band_gains?: number[];
  scene_scale?: number;
  char_class_palette?: CharClass[];
  tokens: { s: string; sparse: boolean; freq?: number; cls?: number }[];
};

export type Neighbours = {
  k: number;
  data: Uint16Array;      // n * k
};

export type SHBlocks = {
  bandScales: [number, number, number];
  band1: Int8Array;   // n * 9  (3 coefs × 3 channels)
  band2: Int8Array;   // n * 15
  band3: Int8Array;   // n * 21
};

const SPLAT_STRIDE = 24;

function f16BitsToFloat32(h: number): number {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

export async function loadSplats(url: string, n: number): Promise<Splats> {
  const buf = await fetch(url).then((r) => {
    if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
    return r.arrayBuffer();
  });
  const bytes = new Uint8Array(buf);
  if (bytes.length !== n * SPLAT_STRIDE) {
    throw new Error(`splats size ${bytes.length} != ${n * SPLAT_STRIDE}`);
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const position = new Float32Array(n * 3);
  const scale = new Float32Array(n * 3);
  const quat = new Float32Array(n * 4);
  const rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const off = i * SPLAT_STRIDE;
    for (let j = 0; j < 3; j++) position[i * 3 + j] = f16BitsToFloat32(dv.getUint16(off + j * 2, true));
    for (let j = 0; j < 3; j++) scale[i * 3 + j]    = f16BitsToFloat32(dv.getUint16(off + 6 + j * 2, true));
    for (let j = 0; j < 4; j++) quat[i * 4 + j]     = f16BitsToFloat32(dv.getUint16(off + 12 + j * 2, true));
    rgb[i * 3 + 0] = bytes[off + 20]!;
    rgb[i * 3 + 1] = bytes[off + 21]!;
    rgb[i * 3 + 2] = bytes[off + 22]!;
  }
  return { n, position, scale, quat, rgb };
}

export async function loadStatic(url: string, n: number): Promise<StaticData> {
  const bytes = new Uint8Array((await fetch(url).then((r) => r.arrayBuffer())) as ArrayBuffer);
  if (bytes.length !== n * 2) throw new Error(`static size ${bytes.length} != ${n * 2}`);
  return { n, opacity: bytes.slice(0, n), flags: bytes.slice(n, 2 * n) };
}

export async function loadTokens(url: string): Promise<TokensDoc> {
  return await fetch(url).then((r) => r.json());
}

export async function loadNeighbours(url: string, n: number, k = 8): Promise<Neighbours> {
  const arr = new Uint16Array((await fetch(url).then((r) => r.arrayBuffer())) as ArrayBuffer);
  if (arr.length !== n * k) throw new Error(`neighbours size ${arr.length} != ${n * k}`);
  return { k, data: arr };
}

export function neighboursOf(nb: Neighbours, tokenId: number): number[] {
  const out: number[] = [];
  for (let j = 0; j < nb.k; j++) out.push(nb.data[tokenId * nb.k + j]!);
  return out;
}

export async function loadSH(url: string, n: number): Promise<SHBlocks> {
  const bytes = new Uint8Array((await fetch(url).then((r) => r.arrayBuffer())) as ArrayBuffer);
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const bandScales: [number, number, number] = [
    dv.getFloat32(0, true), dv.getFloat32(4, true), dv.getFloat32(8, true),
  ];
  const off = 12;
  const b1 = n * 9, b2 = n * 15, b3 = n * 21;
  return {
    bandScales,
    band1: new Int8Array(bytes.buffer, bytes.byteOffset + off, b1),
    band2: new Int8Array(bytes.buffer, bytes.byteOffset + off + b1, b2),
    band3: new Int8Array(bytes.buffer, bytes.byteOffset + off + b1 + b2, b3),
  };
}
