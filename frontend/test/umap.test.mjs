import test from 'node:test';
import assert from 'node:assert/strict';
// Import via data URLs to support Node versions that treat .js as CommonJS
// in this deliberately package.json-free frontend.
import { readFile } from 'node:fs/promises';
const url = source => `data:text/javascript;base64,${Buffer.from(source).toString('base64')}`;
const paletteURL = url(await readFile(new URL('../src/clusters.js', import.meta.url), 'utf8'));
const { CLUSTER_PALETTE, clusterColor, clusterColorCSS } = await import(paletteURL);
const source = (await readFile(new URL('../src/umap.js', import.meta.url), 'utf8')).replace("'./clusters.js'", JSON.stringify(paletteURL))
  .replace("'./harness-hook.mjs'", JSON.stringify(new URL('../src/harness-hook.mjs', import.meta.url).href));
const { median, clusterCentroids, fitBounds, buildPickGrid, pickNearest, pickDrawn, placeLabels, labelWidth } = await import(url(source));
const rng = seed => () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 2 ** 32;
/** Lengths of every array sort made while fn runs: [result, lengths]. */
function sorts(fn) {
  const lengths = [], TA = Object.getPrototypeOf(Int8Array.prototype), typed = TA.sort, array = Array.prototype.sort;
  TA.sort = function (...args) { lengths.push(this.length); return typed.apply(this, args); };
  Array.prototype.sort = function (...args) { lengths.push(this.length); return array.apply(this, args); };
  try { return [fn(), lengths]; } finally { TA.sort = typed; Array.prototype.sort = array; }
}
function basisOf(yaw, pitch) {   // as UmapView.basis()
  const c = Math.cos(yaw), s = Math.sin(yaw), p = Math.cos(pitch), q = Math.sin(pitch);
  return { right: Float32Array.of(c, 0, -s), up: Float32Array.of(-s * q, p, -c * q), back: Float32Array.of(s * p, q, c * p) };
}
function screen(view, to, i) {
  const { center, distance, width, height, basis: b } = view, x = to[4 * i] - center[0], y = to[4 * i + 1] - center[1], z = to[4 * i + 2] - center[2];
  const depth = distance - x * b.back[0] - y * b.back[1] - z * b.back[2], scale = height * 1.20710678 / depth;
  return { x: width / 2 + (x * b.right[0] + y * b.right[1] + z * b.right[2]) * scale, y: height / 2 - (x * b.up[0] + y * b.up[1] + z * b.up[2]) * scale, depth };
}
/** The previous UmapView.pick: a scan of every point (at the end of the morph). */
function scanPick(view, to, count, x, y, clusters, hidden) {
  const { distance, width, height } = view;
  let best = null, bestScore = Infinity, bestDepth = Infinity;
  for (let i = 0; i < count; i++) {
    if (to[4 * i + 3] < .15 || hidden?.has(clusters[i])) continue;
    const p = screen(view, to, i);
    if (p.depth <= distance * .001 || p.depth >= distance * 100 || p.x < 0 || p.y < 0 || p.x > width || p.y > height) continue;
    const d = Math.hypot(p.x - x, p.y - y), radius = Math.max(5, Math.min(12, 4 * distance / p.depth) / 2);
    if (d <= radius && (d < bestScore - .5 || Math.abs(d - bestScore) <= .5 && p.depth < bestDepth)) { best = i; bestScore = d; bestDepth = p.depth; }
  }
  return best;
}
const quantile = (values, f) => { const a = Float64Array.from(values).sort(); return a[Math.min(a.length - 1, Math.max(0, Math.round(f * (a.length - 1))))]; };

test('palette has 40 unique stable CSS colours and matching normalized RGB', () => {
  assert.equal(CLUSTER_PALETTE.length, 40);
  assert.equal(new Set(CLUSTER_PALETTE).size, 40);
  assert.equal(clusterColorCSS(0), '#0072b2');
  for(let i=0;i<40;i++) {
    const rgb=clusterColor(i);
    assert.equal(rgb.length,3);assert.ok(rgb.every(v=>v>=0&&v<=1));
    assert.equal('#'+rgb.map(v=>Math.round(v*255).toString(16).padStart(2,'0')).join(''),clusterColorCSS(i));
    assert.deepEqual(clusterColor(i+40),rgb);
  }
  const rgb=clusterColor(0);rgb[0]=1;assert.equal(clusterColor(0)[0],0);
  assert.equal(clusterColorCSS(-1),'#66717e');
});
test('median handles even, odd, invalid and empty values without mutating inputs', () => {
  const input=[8,1,3,2];assert.equal(median(input),2.5);assert.deepEqual(input,[8,1,3,2]);
  assert.equal(median([100,2,1]),2);assert.equal(median([NaN,Infinity,3]),3);assert.ok(Number.isNaN(median([])));
});
test('centroids use coordinate medians and exclude hidden or partially invalid points', () => {
  const clusters=new Int16Array([0,0,0,1,1,-1,2]);
  const coords=new Float32Array([1,8,4, 2,4,9, 100,6,5, 2,4,6, 4,8,10, 90,90,90, 1,NaN,2]);
  const centers=clusterCentroids(clusters,coords);
  assert.deepEqual([...centers],[[0,[2,6,5]],[1,[3,6,8]]]);
  assert.equal(clusterCentroids(new Int16Array(),new Float32Array()).size,0);
});
test('label placement is deterministic, bounded and nonoverlapping, and earlier labels win', () => {
  const place=()=>placeLabels(Array.from({length:20},()=>({ax:100,ay:100,width:80,height:20})),200,200);
  const items=place(),placed=items.filter(i=>i.placed);
  assert.ok(placed.length>1 && placed.length<items.length);
  assert.deepEqual(items,place());
  assert.equal(items[0].placed,true,'the first label takes its place');
  placed.forEach((a,i)=>{
    assert.ok(a.x>=0&&a.y>=0&&a.x+a.width<=200&&a.y+a.height<=200);
    for(const b of placed.slice(i+1))assert.ok(a.x+a.width+4<=b.x||b.x+b.width+4<=a.x||a.y+a.height+4<=b.y||b.y+b.height+4<=a.y);
  });
  assert.equal(placeLabels([{ax:0,ay:0,width:300,height:20}],200,200)[0].placed,false,'wider than the view');
  assert.equal(placeLabels([{ax:NaN,ay:5,width:20,height:20}],200,200)[0].placed,false);
  assert.deepEqual(placeLabels([],0,0),[]);
});

test('the screen-space pick grid finds exactly the point that a scan of every point finds', () => {
  const rand = rng(7), n = 6000, to = new Float32Array(4 * n), clusters = new Int16Array(n), hidden = new Set([3]);
  for (let i = 0; i < n; i++) {
    to[4 * i] = rand() * 10 - 5; to[4 * i + 1] = rand() * 10 - 5; to[4 * i + 2] = (i % 3) * .3; to[4 * i + 3] = i % 50 ? 1 : 0; clusters[i] = i % 7;
  }
  for (let i = 0; i < 60; i++) for (let k = 0; k < 3; k++) to[4 * (n - 1 - i) + k] = to[4 * i + k] + (k === 2 ? .01 * i : 0);   // stacked points
  for (const [yaw, pitch, distance] of [[0, 0, 12], [.4, .3, 9], [0, 0, 40], [0, 0, 2.5]]) {
    const view = { center: [.2, -.1, 0], distance, width: 361, height: 500, basis: basisOf(yaw, pitch) }, grid = buildPickGrid(to, n, view);
    let hits = 0;
    for (let k = 0; k < 500; k++) {
      let near = { x: rand() * 400 - 20, y: rand() * 540 - 20 };
      for (let tries = 0; k % 2 && tries < 200; tries++) {   // half the queries next to a point on screen
        const p = screen(view, to, Math.floor(rand() * n));
        if (p.x >= 0 && p.y >= 0 && p.x <= view.width && p.y <= view.height) { near = p; break; }
      }
      const x = near.x + rand() * 8 - 4, y = near.y + rand() * 8 - 4, got = pickNearest(grid, x, y, clusters, hidden);
      assert.equal(got, scanPick(view, to, n, x, y, clusters, hidden), `view ${yaw},${distance} at ${x},${y}`);
      if (got !== null) hits++;
    }
    assert.ok(hits > 100, `${hits} hits`);
    assert.equal(buildPickGrid(to, n, view, grid), grid);   // rebuilt in place
  }
});

test('a pick reads only the points in the cells around the pointer', () => {
  const rand = rng(3), n = 40000, to = new Float32Array(4 * n);
  for (let i = 0; i < n; i++) { to[4 * i] = rand() * 20 - 10; to[4 * i + 1] = rand() * 20 - 10; to[4 * i + 3] = 1; }
  const view = { center: [0, 0, 0], distance: 26, width: 360, height: 900, basis: basisOf(0, 0) }, grid = buildPickGrid(to, n, view);
  let reads = 0;
  const px = grid.px; grid.px = new Proxy(px, { get: (t, k) => { if (typeof k === 'string' && /^\d+$/.test(k)) reads++; return t[k]; } });
  const id = pickNearest(grid, 180, 450);
  grid.px = px;
  assert.equal(id, scanPick(view, to, n, 180, 450));
  assert.ok(reads < n / 50, `${reads} reads`);
  assert.equal(pickNearest(grid, -50, -50), null);
  assert.equal(pickNearest(buildPickGrid(to, 0, view), 180, 450), null);
});

test('during a morph the pick finds the point drawn under the pointer, at mix(from, to, t), as a scan of those finds', () => {
  const rand = rng(21), n = 5000, from = new Float32Array(4 * n), to = new Float32Array(4 * n), clusters = new Int16Array(n), hidden = new Set([2]);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < 2; k++) { from[4 * i + k] = rand() * 10 - 5; to[4 * i + k] = from[4 * i + k] + rand() * 3 - 1.5; }
    from[4 * i + 2] = to[4 * i + 2] = (i % 3) * .2;
    from[4 * i + 3] = i % 40 ? 1 : 0; to[4 * i + 3] = i % 45 ? 1 : 0; clusters[i] = i % 6;   // some fade in, some out
  }
  for (const [yaw, pitch, distance] of [[0, 0, 12], [.4, .3, 9], [0, 0, 3]]) {
    const view = { center: [.3, -.2, 0], distance, width: 360, height: 480, basis: basisOf(yaw, pitch) };
    for (const t of [0, .2, .5, .93, 1]) {
      const drawn = from.map((v, j) => v + (to[j] - v) * t);   // what the shader draws
      let hits = 0, differs = 0;
      for (let k = 0; k < 300; k++) {
        let p = screen(view, drawn, Math.floor(rand() * n));
        for (let tries = 0; tries < 200 && !(p.x >= 0 && p.y >= 0 && p.x <= view.width && p.y <= view.height); tries++) p = screen(view, drawn, Math.floor(rand() * n));
        const x = p.x + rand() * 6 - 3, y = p.y + rand() * 6 - 3;
        const got = pickDrawn(from, to, t, n, x, y, view, clusters, hidden);
        assert.equal(got, scanPick(view, drawn, n, x, y, clusters, hidden), `view ${yaw},${distance} t ${t} at ${x},${y}`);
        if (got !== null) { hits++; if (got !== scanPick(view, to, n, x, y, clusters, hidden)) differs++; }
      }
      assert.ok(hits > 80, `${hits} hits`);
      if (t < .9 && distance > 5) assert.ok(differs > hits / 3, `${differs} of ${hits}`);   // the targets lie elsewhere
      if (t === 1) assert.equal(differs, 0);
    }
  }
});

test('fit bounds: exact quantiles for small maps, a bounded deterministic subsample for large ones', () => {
  const rand = rng(11), points = n => { const to = new Float32Array(4 * n); for (let i = 0; i < n; i++) { to[4 * i] = rand() * 6 - 1 + (i % 9 === 0 ? 40 : 0); to[4 * i + 1] = rand() ** 2 * 4; to[4 * i + 3] = i % 13 ? 1 : 0; } return to; };
  const exact = (to, n) => {
    const xs = [], ys = [];
    for (let i = 0; i < n; i++) if (to[4 * i + 3]) { xs.push(to[4 * i]); ys.push(to[4 * i + 1]); }
    return { x: [quantile(xs, .02), quantile(xs, .98)], y: [quantile(ys, .02), quantile(ys, .98)] };
  };
  const small = points(3000);
  assert.deepEqual(fitBounds(small, 3000), exact(small, 3000));   // the previous fit, unchanged
  const large = points(200000), [bounds, lengths] = sorts(() => fitBounds(large, 200000)), want = exact(large, 200000);
  assert.ok(lengths.length === 2 && Math.max(...lengths) <= 4096, `sorted ${lengths}`);
  for (const axis of ['x', 'y']) for (const k of [0, 1]) assert.ok(Math.abs(bounds[axis][k] - want[axis][k]) < .1, `${axis}${k}: ${bounds[axis][k]} vs ${want[axis][k]}`);
  assert.deepEqual(fitBounds(large, 200000), bounds);
  const none = new Float32Array(8); none[0] = 5;
  assert.equal(fitBounds(none, 2), null);
  assert.deepEqual(fitBounds(Float32Array.of(9, 9, 0, 0, 1, 2, 0, 1), 2), { x: [1, 1], y: [2, 2] });   // w = 0 is ignored
});

test('cluster label anchors are subsampled medians: no sort of all points, close to the exact medians', () => {
  const rand = rng(5), n = 90000, clusters = new Int16Array(n), coords = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    const c = i % 3 === 2 ? 7 : i % 3; clusters[i] = i % 101 ? c : -1;
    for (let k = 0; k < 3; k++) coords[3 * i + k] = c * 4 + (rand() + rand() + rand() - 1.5) * (k + 1);
  }
  coords[3 * 4] = NaN;
  const [centres, lengths] = sorts(() => clusterCentroids(clusters, coords));
  assert.ok(Math.max(...lengths) <= 512, `sorted ${Math.max(...lengths)}`);
  assert.deepEqual([...centres.keys()], [1, 7, 0]);   // order of first appearance (cell 0 is not analysed)
  for (const [id, centre] of centres) {
    const members = [0, 1, 2].map(k => { const v = []; for (let i = 0; i < n; i++) if (clusters[i] === id && i !== 4) v.push(coords[3 * i + k]); return v; });
    centre.forEach((v, k) => assert.ok(Math.abs(v - median(members[k])) < .1 * (k + 1), `${id}/${k}: ${v} vs ${median(members[k])}`));
  }
  assert.deepEqual(clusterCentroids(clusters, coords), centres);   // deterministic
  const all = clusterCentroids(clusters, coords, Infinity);
  assert.deepEqual(all.get(1), [0, 1, 2].map(k => { const v = []; for (let i = 0; i < n; i++) if (clusters[i] === 1 && i !== 4) v.push(coords[3 * i + k]); return median(v); }));
});

test('labels are placed in place (one without a finite anchor is not) and sized from text metrics', () => {
  const items = Array.from({ length: 20 }, (_, id) => ({ id, ax: 60 + id * 7, ay: 100 + (id % 3) * 9, width: 50 + id, height: 20 }));
  items.push({ id: 99, ax: NaN, ay: 1, width: 10, height: 20 });
  assert.equal(placeLabels(items, 200, 200), items, 'the same items, given their places');
  assert.ok(items[0].placed && Number.isFinite(items[0].x) && Number.isFinite(items[0].y));
  assert.equal(items.at(-1).placed, false);
  assert.equal(labelWidth('abc', () => 10.2), 32);
  assert.equal(labelWidth('x'.repeat(400), () => 2400), 220);
});
