import test from 'node:test';
import assert from 'node:assert/strict';
import { isPolygonSet, polygonSet, polygonSetFromLegacy, toPolygonSet, polygonCount, gridIndex, visibleCells, releaseIndex } from '../src/polygons.mjs';

/** The polygons whose box meets `box`, in ascending index order, through the grid: what the renderer's culling (the
 *  cells visibleCells returns) holds, narrowed to each polygon's own box. */
function cullPolygons(set, box) {
  const index = gridIndex(set), hits = [];
  for (const k of visibleCells(index, box))
    for (let m = index.starts[k]; m < index.starts[k + 1]; m++) if (meets(set.bbox, index.members[m], box)) hits.push(index.members[m]);
  return Uint32Array.from(hits).sort();
}

/** A deterministic pseudo-random generator (mulberry32), so failures reproduce. */
function random(seed) {
  return () => { seed = (seed + 0x6d2b79f5) >>> 0; let t = seed; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32; };
}
/** `n` closed polygons (first vertex repeated, as Xenium's are) of 5-12 µm around random centres in [0, extent]², a few
 *  large ones (up to 300 µm, crossing many grid cells) and some empty ones, as the engine's JSON carries them. */
function legacy(n, extent = 2000, seed = 1) {
  const r = random(seed), out = [];
  for (let i = 0; i < n; i++) {
    if (i % 97 === 13) { out.push({ id: `cell-${i}`, poly: [] }); continue; }
    const cx = r() * extent, cy = r() * extent, size = i % 50 === 7 ? 50 + r() * 250 : 5 + r() * 7, k = 6 + Math.floor(r() * 20);
    const poly = Array.from({ length: k }, (_, j) => [Math.round((cx + size / 2 * Math.cos(2 * Math.PI * j / k)) * 100) / 100, Math.round((cy + size / 2 * Math.sin(2 * Math.PI * j / k)) * 100) / 100]);
    out.push({ id: `cell-${i}`, poly: [...poly, poly[0]] });
  }
  return out;
}
const meets = (b, i, box) => b[4 * i + 2] >= box[0] && b[4 * i] <= box[2] && b[4 * i + 3] >= box[1] && b[4 * i + 1] <= box[3];

test('legacy JSON polygons become flat typed arrays with each polygon\'s box; numeric ids are kept, string ids are not', () => {
  const polys = [{ id: 3, poly: [[1.5, 2.25], [4, 2], [3, 7.125], [1.5, 2.25]] }, { id: 9, poly: [] }, { id: 0, poly: [[10, 10], [11, 12]] }];
  const set = polygonSetFromLegacy(polys);
  assert.ok(isPolygonSet(set));
  assert.equal(set.count, 3);
  assert.deepEqual([...set.offsets], [0, 4, 4, 6]);
  assert.deepEqual([...set.xy], [1.5, 2.25, 4, 2, 3, 7.125, 1.5, 2.25, 10, 10, 11, 12]);
  assert.deepEqual([...set.bbox.slice(0, 4)], [1.5, 2, 4, 7.125]);
  assert.deepEqual([...set.bbox.slice(4, 8)], [Infinity, Infinity, -Infinity, -Infinity], 'an empty polygon has the empty box');
  assert.deepEqual([...set.bbox.slice(8)], [10, 10, 11, 12]);
  assert.deepEqual([...set.ids], [3, 9, 0]);
  assert.equal(polygonSetFromLegacy([{ id: 'aacccccc-1', poly: [[0, 0]] }, { id: 2, poly: [[1, 1]] }]).ids, null, 'Xenium cell ids are strings');
  assert.equal(polygonSetFromLegacy([{ id: 1.5, poly: [[0, 0]] }]).ids, null);
  const empty = polygonSetFromLegacy([]);
  assert.equal(empty.count, 0); assert.deepEqual([...empty.offsets], [0]); assert.equal(empty.ids, null);
});

test('coordinates are the float32 values a canvas path would have used for the JSON numbers', () => {
  const polys = legacy(300, 11000, 4);
  const set = polygonSetFromLegacy(polys);
  let k = 0;
  for (const { poly } of polys) for (const [x, y] of poly) { assert.equal(set.xy[k++], Math.fround(x)); assert.equal(set.xy[k++], Math.fround(y)); }
  assert.equal(k, set.xy.length);
});

test('toPolygonSet takes legacy arrays, PolygonSets (as they are, or with their boxes added) and nothing else', () => {
  const set = polygonSetFromLegacy(legacy(20));
  assert.equal(toPolygonSet(set), set, 'a complete PolygonSet (what the binary decoder makes) is used as it is');
  const bare = { count: set.count, offsets: set.offsets, xy: set.xy, ids: null };
  const filled = toPolygonSet(bare);
  assert.notEqual(filled, bare); assert.deepEqual([...filled.bbox], [...set.bbox]); assert.equal(filled.offsets, set.offsets);
  assert.equal(toPolygonSet(null), null); assert.equal(toPolygonSet(undefined), null); assert.equal(toPolygonSet({ count: 2 }), null);
  assert.equal(toPolygonSet({ count: 2, offsets: new Uint32Array(2), xy: new Float32Array(0) }), null, 'offsets must be count + 1 long');
  assert.equal(toPolygonSet([{ poly: [[0, 0], [1, 1]] }]).count, 1);
  assert.deepEqual([polygonCount(set), polygonCount([1, 2]), polygonCount(null), polygonCount({})], [set.count, 2, 0, 0]);
  assert.throws(() => polygonSet(new Uint32Array([0, 5]), new Float32Array(4)), /exceed/);
});

test('the grid index files every non-empty polygon once, in index order, and keeps each cell\'s union box and largest side', () => {
  const set = polygonSetFromLegacy(legacy(5000, 3000, 2)), index = gridIndex(set);
  assert.equal(gridIndex(set), index, 'built once per set');
  const seen = new Uint8Array(set.count);
  for (let k = 0; k < index.nx * index.ny; k++) {
    const box = [Infinity, Infinity, -Infinity, -Infinity];
    let side = 0;
    for (let m = index.starts[k]; m < index.starts[k + 1]; m++) {
      const i = index.members[m];
      if (m > index.starts[k]) assert.ok(i > index.members[m - 1], 'ascending within a cell');
      seen[i]++;
      const b = set.bbox.subarray(4 * i, 4 * i + 4), col = Math.min(index.nx - 1, Math.floor(((b[0] + b[2]) / 2 - index.x0) / index.cell));
      assert.equal(Math.floor(k % index.nx), col, 'filed under the cell of its centre');
      box[0] = Math.min(box[0], b[0]); box[1] = Math.min(box[1], b[1]); box[2] = Math.max(box[2], b[2]); box[3] = Math.max(box[3], b[3]);
      side = Math.max(side, b[2] - b[0], b[3] - b[1]);
    }
    if (index.starts[k + 1] > index.starts[k]) {
      assert.deepEqual([...index.boxes.subarray(4 * k, 4 * k + 4)], box.map(Math.fround));
      assert.equal(index.maxSide[k], Math.fround(side));
    }
  }
  for (let i = 0; i < set.count; i++) assert.equal(seen[i], set.offsets[i + 1] > set.offsets[i] ? 1 : 0, `polygon ${i}`);
  assert.ok(index.nx * index.ny <= 4 * set.count + 64, 'a bounded number of cells');
  assert.ok(index.cell > 40 && index.cell < 200, `cells of ~12 typical polygons (${index.cell})`);
  const none = gridIndex(polygonSetFromLegacy([{ poly: [] }]));
  assert.deepEqual([none.nx, none.ny, none.members.length], [0, 0, 0]);
  assert.deepEqual(visibleCells(none, [0, 0, 10, 10]), []);
});

test('culling returns exactly the polygons whose box meets the view, in index order, wherever the view is', () => {
  const set = polygonSetFromLegacy(legacy(4000, 3000, 3)), r = random(9);
  const boxes = [[-1e6, -1e6, 1e6, 1e6], [5000, 5000, 6000, 6000], [-50, -50, -10, -10], [1500, 1500, 1500, 1500], [10, 2990, 20, 3000]];
  for (let t = 0; t < 200; t++) {
    const w = r() < .2 ? r() * 3000 : r() * 300, h = r() * 300, x = r() * 3400 - 200, y = r() * 3400 - 200;
    boxes.push([x, y, x + w, y + h]);
  }
  for (const box of boxes) {
    const want = [];
    for (let i = 0; i < set.count; i++) if (meets(set.bbox, i, box)) want.push(i);
    assert.deepEqual([...cullPolygons(set, box)], want, JSON.stringify(box));
    const cells = visibleCells(gridIndex(set), box);
    for (let j = 1; j < cells.length; j++) assert.ok(cells[j] > cells[j - 1], 'row-major order');
  }
  assert.deepEqual([...cullPolygons(set, [1e6, 1e6, -1e6, -1e6])], [], 'an empty view meets nothing');
  assert.deepEqual([...cullPolygons(set, [10, 3000, 20, 2990])], [], 'nor does an inverted one');
});

test('a polygon far larger than a cell is found from any cell its box reaches', () => {
  const polys = legacy(500, 1000, 5);
  polys.push({ id: 'huge', poly: [[0, 0], [5000, 0], [5000, 40], [0, 40], [0, 0]] });   // a tissue edge, say
  const set = polygonSetFromLegacy(polys), huge = set.count - 1;
  for (const box of [[4900, 10, 4950, 20], [10, 10, 20, 20], [2500, 30, 2501, 31]]) assert.ok(cullPolygons(set, box).includes(huge), JSON.stringify(box));
  assert.ok(!cullPolygons(set, [4900, 50, 4950, 60]).includes(huge));
});

test('a released grid is built again on next use, the same', () => {
  const set = polygonSetFromLegacy(legacy(800, 1000, 11)), first = gridIndex(set);
  assert.equal(gridIndex(set), first, 'kept with the set');
  releaseIndex(set);
  const again = gridIndex(set);
  assert.notEqual(again, first);
  assert.deepEqual([again.cell, [...again.starts], [...again.members]], [first.cell, [...first.starts], [...first.members]]);
});
