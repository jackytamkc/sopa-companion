// Prior outlines in memory: flat typed arrays shared by every vector source (the engine's binaries, the picker's shapes
// endpoint, legacy JSON) and the prior-layers renderer. A whole-sample layer (tens of thousands of polygons, about 1M
// vertices) is ~10 MB with no object per polygon, and the grid index below culls it to the view.
//
//   PolygonSet = { count, offsets: Uint32Array(count+1), xy: Float32Array(2V), bbox: Float32Array(4*count),
//                  ids: Uint32Array(count) | null }
//
// Polygon i is the vertices offsets[i] .. offsets[i+1]-1, stored as x, y pairs (xy[2k], xy[2k+1]) in the stage's world
// frame: the points' intrinsic µm, the coordinates the JSON polygons carry. bbox[4i..4i+3] is its x0, y0, x1, y1 (an
// empty polygon has the empty box +Inf, +Inf, -Inf, -Inf). ids holds numeric ids, or is null when there are none.
// The vectors objects (PriorLayers.vectors) are { cells: PolygonSet|null, nuclei: PolygonSet|null, ...metadata }.

/** Whether `value` has the PolygonSet arrays (bbox may still be missing: toPolygonSet adds it). */
export function isPolygonSet(value) {
  return !!value && typeof value === 'object' && Number.isInteger(value.count) && value.offsets instanceof Uint32Array &&
    value.xy instanceof Float32Array && value.offsets.length === value.count + 1;
}

/** A PolygonSet over these arrays, with each polygon's box computed when `bbox` is not given. */
export function polygonSet(offsets, xy, ids = null, bbox = null) {
  const count = offsets.length - 1;
  if (count < 0 || offsets[count] * 2 > xy.length) throw new Error('Polygon offsets exceed the vertex array');
  if (!bbox) {
    bbox = new Float32Array(4 * count);
    for (let i = 0; i < count; i++) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (let k = 2 * offsets[i], end = 2 * offsets[i + 1]; k < end; k += 2) {
        const x = xy[k], y = xy[k + 1];
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
      bbox[4 * i] = x0; bbox[4 * i + 1] = y0; bbox[4 * i + 2] = x1; bbox[4 * i + 3] = y1;
    }
  }
  return { count, offsets, xy, bbox, ids };
}

/** The engine's JSON polygons, [{id, poly: [[x, y], ...]}, ...], as a PolygonSet (ids kept only when all are numeric). */
export function polygonSetFromLegacy(polys) {
  let vertices = 0;
  for (const p of polys) vertices += p?.poly?.length || 0;
  const offsets = new Uint32Array(polys.length + 1), xy = new Float32Array(2 * vertices), ids = new Uint32Array(polys.length);
  let k = 0, numeric = polys.length > 0;
  for (let i = 0; i < polys.length; i++) {
    for (const [x, y] of polys[i]?.poly || []) { xy[k++] = x; xy[k++] = y; }
    offsets[i + 1] = k / 2;
    const id = polys[i]?.id;
    if (Number.isInteger(id) && id >= 0 && id <= 0xffffffff) ids[i] = id; else numeric = false;
  }
  return polygonSet(offsets, xy, numeric ? ids : null);
}

/** Any polygon source as a PolygonSet: a PolygonSet itself (its boxes added if missing), legacy JSON polygons, or null. */
export function toPolygonSet(value) {
  if (Array.isArray(value)) return polygonSetFromLegacy(value);
  if (!isPolygonSet(value)) return null;
  return value.bbox instanceof Float32Array && value.bbox.length === 4 * value.count ? value
    : { ...value, ...polygonSet(value.offsets, value.xy, value.ids ?? null) };
}

/** Number of polygons of a PolygonSet or of a legacy array (0 for anything else). */
export function polygonCount(value) { return value?.count ?? (Array.isArray(value) ? value.length : 0); }

const INDEX = new WeakMap();
const CELL_SIDES = 12;      // a grid cell is ~12 typical polygons across: a view of 8k cells reads ~300 cells, a 600 µm view
                            // ~70 % more polygons than it shows
const MAX_CELLS = 1 << 20;

/** The bbox grid of a set, built once per set and kept with it. Every non-empty polygon is filed in exactly one cell,
 *  the one holding its box centre (in index order), so a cell's polygons can be drawn as one tile; each cell keeps the
 *  union of its polygons' boxes, which is what culling tests, and its largest polygon side.
 *  { cell, nx, ny, x0, y0, reach, starts: Uint32Array(nx*ny+1), members: Uint32Array, boxes: Float32Array(4*nx*ny),
 *    maxSide: Float32Array(nx*ny) } */
export function gridIndex(set) {
  let index = INDEX.get(set);
  if (index) return index;
  const { count, bbox } = set;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, reach = 0;
  const sides = [];
  const step = Math.max(1, Math.floor(count / 4096));
  for (let i = 0; i < count; i++) {
    const a = bbox[4 * i], b = bbox[4 * i + 1], c = bbox[4 * i + 2], d = bbox[4 * i + 3];
    if (!(c >= a && d >= b)) continue;   // empty
    if (a < x0) x0 = a; if (b < y0) y0 = b; if (c > x1) x1 = c; if (d > y1) y1 = d;
    const side = Math.max(c - a, d - b);
    if (side / 2 > reach) reach = side / 2;
    if (i % step === 0) sides.push(side);
  }
  if (!sides.length) {
    index = { cell: 1, nx: 0, ny: 0, x0: 0, y0: 0, reach: 0, starts: new Uint32Array(1), members: new Uint32Array(0), boxes: new Float32Array(0), maxSide: new Float32Array(0) };
    INDEX.set(set, index);
    return index;
  }
  sides.sort((p, q) => p - q);
  const w = x1 - x0, h = y1 - y0;
  let cell = CELL_SIDES * sides[sides.length >> 1];
  if (!(cell > 0)) cell = Math.max(w, h) / 64 || 1;
  cell = Math.max(cell, Math.sqrt(w * h / Math.min(MAX_CELLS, 4 * count + 64)) || 0);
  const nx = Math.max(1, Math.ceil(w / cell)), ny = Math.max(1, Math.ceil(h / cell)), cells = nx * ny;
  const of = new Int32Array(count), starts = new Uint32Array(cells + 1);
  for (let i = 0; i < count; i++) {
    const a = bbox[4 * i], b = bbox[4 * i + 1], c = bbox[4 * i + 2], d = bbox[4 * i + 3];
    if (!(c >= a && d >= b)) { of[i] = -1; continue; }
    const col = Math.min(nx - 1, Math.floor(((a + c) / 2 - x0) / cell)), row = Math.min(ny - 1, Math.floor(((b + d) / 2 - y0) / cell));
    of[i] = row * nx + col; starts[of[i] + 1]++;
  }
  for (let k = 0; k < cells; k++) starts[k + 1] += starts[k];
  const members = new Uint32Array(starts[cells]), fill = starts.slice(0, cells);
  const boxes = new Float32Array(4 * cells).fill(Infinity), maxSide = new Float32Array(cells);
  for (let k = 0; k < cells; k++) { boxes[4 * k + 2] = -Infinity; boxes[4 * k + 3] = -Infinity; }
  for (let i = 0; i < count; i++) {
    const k = of[i];
    if (k < 0) continue;
    members[fill[k]++] = i;
    const a = bbox[4 * i], b = bbox[4 * i + 1], c = bbox[4 * i + 2], d = bbox[4 * i + 3];
    if (a < boxes[4 * k]) boxes[4 * k] = a; if (b < boxes[4 * k + 1]) boxes[4 * k + 1] = b;
    if (c > boxes[4 * k + 2]) boxes[4 * k + 2] = c; if (d > boxes[4 * k + 3]) boxes[4 * k + 3] = d;
    const side = Math.max(c - a, d - b);
    if (side > maxSide[k]) maxSide[k] = side;
  }
  index = { cell, nx, ny, x0, y0, reach, starts, members, boxes, maxSide };
  INDEX.set(set, index);
  return index;
}

/** Forget a set's grid now (PriorLayers, when the set is replaced); it is built again if the set is used again. */
export function releaseIndex(set) { INDEX.delete(set); }

const meets = (boxes, k, box) => boxes[4 * k + 2] >= box[0] && boxes[4 * k] <= box[2] && boxes[4 * k + 3] >= box[1] && boxes[4 * k + 1] <= box[3];

/** The grid cells with a polygon whose box meets `box` (world x0, y0, x1, y1), in row-major order. */
export function visibleCells(index, box) {
  const out = [];
  if (!index.nx || !(box[2] >= box[0] && box[3] >= box[1])) return out;
  // A polygon reaches at most `reach` beyond its cell (its centre is inside it).
  const c0 = Math.max(0, Math.floor((box[0] - index.reach - index.x0) / index.cell)), c1 = Math.min(index.nx - 1, Math.floor((box[2] + index.reach - index.x0) / index.cell));
  const r0 = Math.max(0, Math.floor((box[1] - index.reach - index.y0) / index.cell)), r1 = Math.min(index.ny - 1, Math.floor((box[3] + index.reach - index.y0) / index.cell));
  for (let row = r0; row <= r1; row++) for (let col = c0; col <= c1; col++) {
    const k = row * index.nx + col;
    if (index.starts[k + 1] > index.starts[k] && meets(index.boxes, k, box)) out.push(k);
  }
  return out;
}
