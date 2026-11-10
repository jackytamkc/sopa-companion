import test from 'node:test';
import assert from 'node:assert/strict';
import {
  QcOverlay, MixingLegend, binsWeight, isOverview, decodeBins, fetchBins, shareColor, smoothShares, binsPixels, BIN_STOPS,
  BIN_MARGIN, binStops, OVERVIEW_SCALE, OUTLINE_VOXEL_PX, LAYER_ALPHA, UNJUDGED_SWATCH, DASHES, hexRgb, rgba,
  layerValues, shareStrength, LOW_ALPHA, FULL_J, fullJudged, occupiedPath, SQUARE_STYLE, START_STYLE, LEGEND_PLACE,
} from '../src/qc-overlay.mjs';
import { Stage, MIXING_COLORS as M } from '../src/stage.js';
import { QC_TEXT as C, bannedWords } from '../src/qc-copy.mjs';
import { domFixture } from './dom-fixture.mjs';
import { fakeLoop } from './fake-loop.mjs';

// The mixing check on the 2-D overlay: the per-bin layer that stands in for the cells at the whole-sample overview
// under Colour by: Mixing, the two marks, the frames it asks of the stage's scheduler, and the legend. No flagged area
// is drawn since 2026-10-22 (no flagged area on screen): no outline, chip, hover or click.

/** A stage at `scale` CSS px per µm, (0, 0) µm at the top left. */
function fakeStage({ scale = 1, cellColorBy = 'identity', width = 800, height = 600 } = {}) {
  const s = {
    scale, cssWidth: width, cssHeight: height, orbit: { active: false }, options: { cellColorBy }, animators: [], replans: 0,
    toScreen: (x, y) => [x * s.scale, y * s.scale], toWorld: (x, y) => [x / s.scale, y / s.scale],
    addAnimator(a) { this.animators.push(a); }, removeAnimator(a) { this.animators = this.animators.filter(x => x !== a); }, replan() { this.replans++; },
  };
  return s;
}
/** A 2-D context recording every call with the style in force. */
function recorder() {
  const calls = [], style = { strokeStyle: '#000', fillStyle: '#000', lineWidth: 1, globalAlpha: 1, dash: [], font: '' };
  const g = new Proxy(style, {
    get: (t, k) => k === 'setLineDash' ? d => { t.dash = [...d]; } : k === 'measureText' ? text => ({ width: String(text).length * 6 })
      : k in t ? t[k] : (...args) => { calls.push({ op: k, args, ...t, dash: [...t.dash] }); },
    set: (t, k, v) => { t[k] = v; return true; },
  });
  return { g, calls };
}
const strokes = calls => calls.filter(c => c.op === 'stroke');
const texts = calls => calls.filter(c => c.op === 'fillText').map(c => c.args[0]);
function overlay(options = {}) {
  const stage = fakeStage(options), clock = { t: 0 }, redraws = [];
  const o = new QcOverlay(stage, { now: () => clock.t, redraw: () => redraws.push(clock.t), reducedMotion: () => !!options.reduced,
    createCanvas: (w, h) => { const c = { createImageData: (a, b) => ({ data: new Uint8ClampedArray(a * b * 4) }), putImageData(p) { this.pixels = p.data; } }; return { width: w, height: h, getContext: () => c }; } });
  const draw = () => { const r = recorder(); o.draw(r.g, { width: stage.cssWidth, height: stage.cssHeight }); return r.calls; };
  return { stage, clock, redraws, o, draw };
}

test('the per-bin layer covers the cells only zoomed out to the whole sample, and fades out as cells grow readable', () => {
  assert.deepEqual({ ...OVERVIEW_SCALE }, { full: .25, none: .6 });
  for (const [scale, w] of [[.05, 1], [.17, 1], [.25, 1], [.6, 0], [1, 0], [4, 0], [NaN, 0], [undefined, 0]]) assert.equal(binsWeight(scale), w, `${scale}`);
  const mid = [.3, .35, .4, .45, .5, .55].map(binsWeight);
  assert.ok(mid.every((w, i) => w > 0 && w < 1 && (!i || w < mid[i - 1])), `${mid}`);
  assert.equal(isOverview(.2), true); assert.equal(isOverview(.42), true); assert.equal(isOverview(.43), false); assert.equal(isOverview(1), false);
});

test('with outlines only, the per-bin layer stays until the outlines are drawn (about 2.4 device px per voxel)', () => {
  assert.deepEqual({ ...OUTLINE_VOXEL_PX }, { full: 1.2, none: 2.6 });
  const outlines = (scale, voxel = 1, ratio = 1) => binsWeight(scale, { mode: 1, voxel, ratio });
  // A region run at its fit (a sample's crop: 0.55 CSS px per µm, 1 µm voxels): outlines are 0.19 px wide, not drawn.
  assert.equal(outlines(.55), 1); assert.ok(binsWeight(.55) < .1, 'Cells mode: single cells carry the colours there');
  assert.ok(outlines(2) > 0 && outlines(2) < 1); assert.equal(outlines(2.6), 0); assert.equal(outlines(5), 0);
  assert.equal(outlines(1.3, 2), 0, '2 µm voxels (burn-in) are drawn sooner'); assert.equal(outlines(1.3, 1, 2), 0, 'and so on a HiDPI screen');
  // The whole-sample overview: the raster's voxel is its stride, about a device px, but the scale alone covers it.
  assert.equal(outlines(.12, 8), 1);
  assert.equal(binsWeight(.55, { mode: 0, voxel: 1 }), binsWeight(.55));
  assert.equal(isOverview(.55, { mode: 1 }), true); assert.equal(isOverview(.55), false);
  assert.equal(binsWeight(NaN, { mode: 1 }), 0);
});

test('bins grids decode from their X-Bins header and bytes; a missing frame is null, a bad one an error', async () => {
  const grid = decodeBins(Uint8Array.of(0, 50, 100, 255, 3, 4).buffer, '{"x0":10,"y0":-5,"bin":100,"nx":3,"ny":2}');
  assert.deepEqual([grid.x0, grid.y0, grid.bin, grid.nx, grid.ny, [...grid.values]], [10, -5, 100, 3, 2, [0, 50, 100, 255, 3, 4]]);
  assert.equal(decodeBins(new ArrayBuffer(4), { x0: 0, y0: 0, bin: 50, nx: 2, ny: 2 }).nx, 2, 'or the header object');
  for (const [bytes, header] of [[3, { x0: 0, y0: 0, bin: 50, nx: 2, ny: 2 }], [4, { x0: 0, y0: 0, bin: 0, nx: 2, ny: 2 }], [4, { x0: 0, y0: 0, bin: 50, nx: 2.5, ny: 2 }], [4, null]])
    assert.throws(() => decodeBins(new ArrayBuffer(bytes), header), /bins/i);
  const asked = [];
  const fetch = async (url, init) => { asked.push([url, init.credentials]); return url.includes('/7/') ? { status: 404, ok: false }
    : url.includes('/8/') ? { status: 500, ok: false } : { status: 200, ok: true, headers: { get: k => k === 'X-Bins' ? '{"x0":0,"y0":0,"bin":100,"nx":1,"ny":1}' : null }, arrayBuffer: async () => Uint8Array.of(9).buffer }; };
  assert.equal(await fetchBins('r/1', 7, { fetch }), null);
  await assert.rejects(fetchBins('r1', 8, { fetch }), /HTTP 500/);
  assert.deepEqual([...(await fetchBins('r1', 9, { fetch })).values], [9]);
  assert.deepEqual(asked[0], ['/api/v1/runs/r%2F1/qc/7/bins', 'same-origin']);
  // The frame of the starting cells shows the start line's own bins.
  assert.deepEqual([...(await fetchBins('r1', 'start', { fetch })).values], [9]);
  assert.deepEqual(asked.at(-1), ['/api/v1/runs/r1/qc/start/bins', 'same-origin']);
});

test('bins are coloured on the cells\' own ramp by the share of their cells that look mixed; empty bins are clear', () => {
  assert.deepEqual(BIN_STOPS, [[0, M.little], [.15, M.some], [.3, M.mixed]]);
  assert.deepEqual(shareColor(NaN), [0, 0, 0, 0]);
  assert.deepEqual(shareColor(0), [...hexRgb(M.little), 255]);
  assert.deepEqual(shareColor(.15), [...hexRgb(M.some), 255], '15 % of cells');
  assert.deepEqual(shareColor(.3), [...hexRgb(M.mixed), 255], '30 %');
  assert.deepEqual(shareColor(1), [...hexRgb(M.mixed), 255]);
  const half = shareColor(.075), [a, b] = [hexRgb(M.little), hexRgb(M.some)];
  assert.deepEqual(half, [...a.map((v, i) => Math.round((v + b[i]) / 2)), 255]);
  // A whole sample at 13-15 in 100: most of its tissue sits below "some"; its crop's most mixed windows (25-30 %) near "a lot".
  assert.ok(shareColor(.13)[0] < hexRgb(M.some)[0] && shareColor(.27)[0] > hexRgb(M.some)[0]);
  // The cross-fade is premultiplied: a bin that appears keeps its colour while its alpha rises.
  const from = { x0: 0, y0: 0, bin: 100, nx: 2, ny: 1, values: Uint8Array.of(255, 0) }, to = { ...from, values: Uint8Array.of(100, 100) };
  const third = binsPixels(to, from, 1 / 3);
  assert.deepEqual([...third.slice(0, 4)], [...hexRgb(M.mixed), 85]);
  assert.equal(third[7], Math.round(255 * LOW_ALPHA * 2 / 3 + 255 / 3), 'from faint (at the usual level) to full');
  assert.deepEqual([...binsPixels(to).slice(0, 4)], [...hexRgb(M.mixed), 255]);
  assert.deepEqual([...binsPixels(to, { ...from, nx: 1, ny: 2 }, .5).slice(0, 4)], [...hexRgb(M.mixed), 255], 'another grid: no blend');
});

test('the layer\'s ramp is anchored on the tissue\'s level: some at an area\'s enter margin, a lot at twice it', () => {
  assert.deepEqual({ ...BIN_MARGIN }, { mult: .5, add: .04 });
  assert.equal(binStops(null), BIN_STOPS); assert.equal(binStops(NaN), BIN_STOPS); assert.equal(binStops(-1), BIN_STOPS);
  const close = (a, b) => a.every((s, i) => Math.abs(s[0] - b[i][0]) < 1e-12 && s[1] === b[i][1]);
  // A whole sample at 14 in 100: margin max(7, 4) = 7 points, so some at 21 % (where an area enters), a lot at 28 %.
  assert.ok(close(binStops(.14), [[.14, M.little], [.21, M.some], [.28, M.mixed]]));
  // A sample at 5 in 100: the 4-point floor of the margin.
  assert.ok(close(binStops(.05), [[.05, M.little], [.09, M.some], [.13, M.mixed]]));
  const at14 = binStops(.14);
  assert.deepEqual(shareColor(.10, at14), [...hexRgb(M.little), 255], 'below the tissue\'s level: little or none');
  assert.deepEqual(shareColor(.14, at14), [...hexRgb(M.little), 255]);
  assert.deepEqual(shareColor(.21, at14), [...hexRgb(M.some), 255], 'as mixed as a flagged area');
  assert.deepEqual(shareColor(.30, at14), [...hexRgb(M.mixed), 255]);
  // With absolute steps the same tissue would sit near "some" everywhere.
  assert.ok(shareColor(.14)[0] > hexRgb(M.little)[0] + .8 * (hexRgb(M.some)[0] - hexRgb(M.little)[0]));
  const grid = { x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(Math.floor(.21 * 250)) };
  assert.deepEqual([...binsPixels(grid, null, 1, at14)].slice(0, 3), shareColor(Math.floor(.21 * 250) / 250, at14).slice(0, 3));
});

test('the overlay anchors its layer on the start line\'s level, else the level shown; a new level recolours a layer on screen', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  const grid = { x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(53) };   // 21 %
  h.o.setBins(grid, { at: 0 });
  const pixel = () => [...h.draw().find(c => c.op === 'drawImage').args[0].getContext().pixels.slice(0, 3)];
  assert.deepEqual(pixel(), shareColor(53 / 250).slice(0, 3), 'no level yet: absolute steps');
  assert.equal(h.o.setLevel({ tissuePer100: 14 }), true, 'a new level recolours the layer on screen');
  assert.deepEqual(pixel(), shareColor(53 / 250, binStops(.14)).slice(0, 3));
  assert.equal(h.o.setLevel({ tissuePer100: 14 }), false, 'the same level: nothing to redraw');
  assert.equal(h.o.setLevel({ anchorPer100: 14, tissuePer100: 18 }), false, 'the start line\'s level wins over the level shown');
  assert.deepEqual(pixel(), shareColor(53 / 250, binStops(.14)).slice(0, 3));
  assert.ok(h.o.images.size <= 8);
  h.stage.options.cellColorBy = 'identity'; h.draw();
  assert.equal(h.o.setLevel({ tissuePer100: 15 }), false, 'no layer drawn: nothing to recolour'); assert.equal(h.o.anchorPer100, 15);
});

test('each bin is drawn as the mean of its judged 3 x 3 neighbourhood (the areas\' windows), so a few cells cannot light it up', () => {
  // A 4 x 3 grid in the crop's style: an edge bin of two cells reading 50 %, empty bins (255) outside the tissue.
  const grid = { x0: 0, y0: 0, bin: 100, nx: 4, ny: 3, values: Uint8Array.of(
    125, 10, 20, 255,
    10, 25, 15, 255,
    255, 30, 20, 255) };
  const s = smoothShares(grid), mean = (...codes) => codes.reduce((a, b) => a + b) / codes.length / 250;
  const want = [
    mean(125, 10, 10, 25), mean(125, 10, 20, 10, 25, 15), mean(10, 20, 25, 15), null,
    mean(125, 10, 10, 25, 30), mean(125, 10, 20, 10, 25, 15, 30, 20), mean(10, 20, 25, 15, 30, 20), null,
    null, mean(10, 25, 15, 30, 20), mean(25, 15, 30, 20), null];
  want.forEach((v, i) => assert.ok(v == null ? Number.isNaN(s[i]) : Math.abs(s[i] - v) < 1e-6, `bin ${i}: ${s[i]} vs ${v}`));
  assert.ok(s[0] < .2, 'the 50 % edge bin reads 17 % among its neighbours');
  assert.deepEqual([...smoothShares({ ...grid, nx: 1, ny: 1, values: Uint8Array.of(50) })], [Math.fround(.2)], 'alone: itself');
  // The layer's pixels use it, and keep nothing-judged bins clear.
  const pixels = binsPixels(grid);
  assert.deepEqual([...pixels.slice(0, 3)], shareColor(s[0]).slice(0, 3));
  assert.equal(pixels[3 * 4 + 3], 0); assert.equal(pixels[8 * 4 + 3], 0);
});

test("a two-plane grid: each bin's window share as the engine summed it, as strong as the bin is full, faint at the usual level", () => {
  // The crop's edge: a bin with 5 judged cells (its window reads 40 %) beside full ones (22 cells, 30 %), then background.
  const n = 4, grid = decodeBins(Uint8Array.of(100, 75, 75, 255, 5, 22, 22, 0).buffer, { x0: 0, y0: 0, bin: 100, nx: 4, ny: 1, planes: 2 });
  assert.equal(grid.planes, 2); assert.equal(fullJudged(grid), FULL_J, 'at most FULL_J on 100 µm bins');
  assert.throws(() => decodeBins(new ArrayBuffer(3), { x0: 0, y0: 0, bin: 100, nx: 3, ny: 1, planes: 2 }), /bins/i);
  const v = layerValues(grid);
  assert.deepEqual([...v.share].map(x => Number.isNaN(x) ? null : +x.toFixed(3)), [.4, .3, .3, null], 'the window shares, not re-averaged here');
  assert.deepEqual([...v.fullness].map(x => +x.toFixed(4)), [+((5 / FULL_J) ** 2).toFixed(4), 1, 1, 0],
    'a bin mostly outside the tissue fades out (squared: no glow over background); background is clear');
  const stops = binStops(.14), pixels = binsPixels(grid, null, 1, stops);
  assert.equal(pixels[3], Math.round(255 * (5 / FULL_J) ** 2), 'the brightest share, but at the edge: faint');
  assert.equal(pixels[7], 255, 'a full bin above the usual level: full'); assert.equal(pixels[15], 0);
  // A sparse sample (its typical bin holds 8): its typical bins are full.
  const sparse = decodeBins(Uint8Array.of(75, 75, 75, 8, 8, 4).buffer, { x0: 0, y0: 0, bin: 100, nx: 3, ny: 1, planes: 2 });
  assert.equal(fullJudged(sparse), 8); assert.deepEqual([...layerValues(sparse).fullness], [1, 1, .25]);
  // At or below the tissue's usual level a patch is faint: the tissue stays visible and only patches above it carry colour.
  assert.equal(shareStrength(.10, stops), LOW_ALPHA); assert.equal(shareStrength(.14, stops), LOW_ALPHA);
  assert.ok(Math.abs(shareStrength(.21, stops) - 1) < 1e-9); assert.ok(shareStrength(.175, stops) > LOW_ALPHA && shareStrength(.175, stops) < 1);
  assert.equal(shareStrength(NaN, stops), 0);
  // On 50 µm bins a full bin holds a quarter of the cells.
  const detail = decodeBins(Uint8Array.of(100, 5).buffer, { x0: 0, y0: 0, bin: 50, nx: 1, ny: 1, planes: 2 });
  assert.equal(layerValues(detail).fullness[0], 1);
});

test('2026-10-22, no flagged area on screen: the overlay draws no area, no chip and no words; nothing to hover or click', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  // Nothing can hand it an area, nor ask it for one under the pointer.
  for (const name of ['setAreas', 'hit', 'hover', 'select', 'target', 'tooltip', 'styles', 'texts', 'shade', 'drawArea', 'drawChips', 'drawChip'])
    assert.equal(typeof h.o[name], 'undefined', name);
  for (const key of ['areas', 'chips', 'obstacles', 'shown']) assert.equal(key in h.o, false, key);
  // With the per-bin layer, a test's square and Go deeper's starting outline all on, it draws those alone: no words, no
  // fill (an area's wash) and no line in the areas' pink.
  h.o.setBins({ x0: 0, y0: 0, bin: 100, nx: 2, ny: 1, values: Uint8Array.of(100, 20) }, { at: 0 });
  h.o.setTestSquare({ box: [100, 50, 300, 250] });
  h.stage.startOutline = { points: Float32Array.of(10, 10, 20, 10, 20, 30), closed: true, count: 3 };
  h.clock.t = 1000;
  const calls = h.draw();
  assert.equal(calls.filter(c => c.op === 'drawImage').length, 1, 'the per-bin layer');
  assert.deepEqual(calls.filter(c => ['fillText', 'fill', 'measureText'].includes(c.op)), [], 'no chip, no wash');
  assert.deepEqual(strokes(calls).map(c => c.strokeStyle), ['rgba(11,14,19,.55)', SQUARE_STYLE.color, 'rgba(11,14,19,.55)', START_STYLE.color],
    'the two marks, each over its casing, and nothing else');
  // Go deeper's focus has nothing of the check's own to shade any more.
  h.stage.focusLevel = 1; assert.equal(h.o.due(2000), null, 'the focus asks for no frame');
});

test('the per-bin layer is drawn only with Colour by: Mixing, zoomed out, at weight x 0.85', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  const grid = { x0: 100, y0: 200, bin: 100, nx: 3, ny: 2, values: Uint8Array.of(0, 50, 100, 255, 255, 25) };
  assert.equal(h.o.wantsBins(), true);
  assert.equal(h.o.setBins(grid, { at: 0 }), true); assert.equal(h.o.setBins(grid), false);
  h.clock.t = 1000;
  let calls = h.draw();
  const image = calls.find(c => c.op === 'drawImage');
  assert.ok(image, 'drawn');
  assert.deepEqual(image.args.slice(1), [20, 40, 60, 40], 'over the grid\'s box on screen');
  assert.equal(image.globalAlpha, LAYER_ALPHA); assert.equal(image.imageSmoothingEnabled, true, 'soft');
  const share = (0 + 50) / 2 / 250;
  assert.deepEqual([...image.args[0].getContext().pixels.slice(0, 4)], [...shareColor(share).slice(0, 3), Math.round(255 * shareStrength(share))],
    'its judged neighbourhood: 0 and 50 (an older one-plane grid), faint near the usual level');
  assert.equal(h.o.overview(), true);
  h.stage.scale = 1;
  assert.equal(h.draw().some(c => c.op === 'drawImage'), false, 'cells readable: no layer'); assert.equal(h.o.wantsBins(), false);
  assert.equal(h.o.overview(), false);
  // Outlines mode at a region's fit: the outlines are not drawn yet, so the layer carries the colours.
  h.stage.scale = .55; h.stage.options.mode = 1; h.stage.frame = { voxel: 1 };
  assert.equal(h.o.wantsBins(), true); assert.equal(h.o.overview(), true);
  assert.equal(h.draw().find(c => c.op === 'drawImage').globalAlpha, LAYER_ALPHA);
  h.stage.options.mode = 0;
  assert.ok(h.o.binsWeight() < .1); assert.equal(h.o.overview(), false);
  h.stage.scale = .2; h.stage.options.cellColorBy = 'cluster';
  assert.equal(h.draw().some(c => c.op === 'drawImage'), false, 'other colour modes: no layer'); assert.equal(h.o.wantsBins(), false);
  h.stage.options.cellColorBy = 'mixing'; h.stage.orbit.active = true;
  assert.equal(h.o.wantsBins(), false, 'not in 3-D');
  h.stage.orbit.active = false;
  assert.equal(h.o.setBins(null), true); assert.equal(h.draw().some(c => c.op === 'drawImage'), false);
});

test('a new bins grid a second or more after the last cross-fades with the cells (1/3, 2/3, 1); faster ones land at once', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  const grid = v => ({ x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(v) });
  h.o.setBins(grid(0), { at: 1000 }); h.draw();   // main.js draws the overlay when setBins says it changed
  assert.equal(h.o.due(), null, 'the first grid lands at once');
  h.o.setBins(grid(100), { at: 1500 });
  assert.equal(h.o.binsPlan, null, 'half a second later: at once');
  h.o.setBins(grid(0), { at: 3000 });
  assert.deepEqual(h.o.binsPlan, { duration: 300, frames: 3, at: 3000 });
  const seen = [];
  for (const t of [3000, 3150, 3300]) {
    assert.equal(h.o.due(), t); h.clock.t = t;
    const image = h.draw().find(c => c.op === 'drawImage').args[0];
    seen.push(image.getContext().pixels[0]);
  }
  // Premultiplied: the faint "little" bin (at the usual level) takes over as the bright one fades.
  const [little, mixed] = [hexRgb(M.little)[0], hexRgb(M.mixed)[0]], low = 255 * LOW_ALPHA;
  const blend = t => Math.round((mixed * 255 * (1 - t) + little * low * t) / (255 * (1 - t) + low * t));
  assert.deepEqual(seen, [blend(1 / 3), blend(2 / 3), little]);
  assert.equal(h.o.due(), null, 'idle once faded');
  h.o.setBins(grid(100), { at: 6000, motion: false });
  assert.equal(h.o.binsPlan, null, 'motion false (paused, the frame on screen): at once');
  h.stage.options.cellColorBy = 'identity';
  h.o.setBins(grid(0), { at: 9000 });
  assert.equal(h.o.binsPlan, null, 'not shown: no fade');
});

test('a layer drawn for options that changed without a camera move gets one refresh frame (not motion), then nothing', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  h.o.setBins({ x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(100) }, { at: 0 }); h.draw();
  assert.equal(h.o.due(), null, 'drawn as it should be: idle');
  // Colour by: Mixing switched off (stage.set, no camera move): the layer must go at once.
  h.clock.t = 500; h.stage.options.cellColorBy = 'identity';
  assert.equal(h.o.due(500), 500);
  assert.equal(h.o.frame(500), false, 'a refresh, not motion'); assert.deepEqual(h.redraws, [500]);
  assert.equal(h.draw().some(c => c.op === 'drawImage'), false); assert.equal(h.o.due(), null);
  // Back on, then the display mode changes the layer's strength at a region's zoom.
  h.stage.options.cellColorBy = 'mixing'; h.stage.scale = .55; h.stage.frame = { voxel: 1 };
  assert.equal(h.o.due(600), 600); h.o.frame(600);
  assert.equal(h.o.due(), null);
  h.stage.options.mode = 1;
  assert.equal(h.o.due(700), 700, 'Outlines: the layer now stands in for the cells'); h.o.frame(700);
  assert.equal(h.o.due(), null);
  // With the layer cross-fading at the same time the frame is motion, and the refresh rides it.
  h.o.setBins({ x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(0) }, { at: 2000 }); h.stage.options.mode = 0;
  assert.equal(h.o.due(2000), 2000); assert.equal(h.o.frame(2000), true);
  // Wanted but not fetched yet (the mode changed while paused): one frame lets main.js's drawOverlay fetch it.
  const w = overlay({ scale: .8, cellColorBy: 'mixing' }); w.stage.frame = { voxel: 1 }; w.draw();
  assert.equal(w.o.wantsBins(), false); assert.equal(w.o.due(), null);
  w.stage.options.mode = 1;
  assert.equal(w.o.wantsBins(), true); assert.equal(w.o.due(900), 900);
  assert.equal(w.o.frame(900), false); assert.equal(w.o.due(), null, 'once, even if no layer comes (404)');
  w.o.visible = false; w.stage.options.mode = 0; w.stage.options.mode = 1;
  assert.equal(w.o.due(), null, 'nothing while hidden');
});

test('motion: the overlay asks the stage for frames only for the layer\'s cross-fade (3 per result), none while idle, in 3-D or with reduced motion', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  assert.deepEqual(h.stage.animators, [h.o], 'registered with the stage\'s scheduler');
  const grid = v => ({ x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(v) });
  h.o.setBins(grid(0), { at: 0 }); h.draw();
  assert.equal(h.o.setLevel({ anchorPer100: 12 }), true); h.draw();
  assert.equal(h.o.due(), null, 'a new level: one redraw by main.js, no frames');
  h.o.setBins(grid(100), { at: 2000 });
  const frames = [];
  for (let t = 2000; t < 4000; t += 16) {
    const due = h.o.due(t);
    if (due != null && t >= due) { h.clock.t = t; h.o.frame(t); h.draw(); frames.push(t); }
  }
  assert.equal(frames.length, 3, `${frames}`); assert.equal(h.o.due(), null);
  assert.deepEqual(h.redraws.length, 3, 'each frame redraws the overlay (main.js drawOverlay)');
  h.o.setBins(grid(0), { at: 6000 }); h.stage.orbit.active = true;
  assert.equal(h.o.due(6000), null, '3-D: no frames');
  h.stage.orbit.active = false;
  assert.equal(h.o.due(6000), 6000);
  const calm = overlay({ scale: .2, cellColorBy: 'mixing' }); calm.o.calm = true;
  calm.o.setBins(grid(0), { at: 0 }); calm.draw(); calm.o.setBins(grid(100), { at: 5000 });
  assert.equal(calm.o.binsPlan, null, 'calm motion: lands at once'); assert.equal(calm.o.due(5000), null);
  const reduced = overlay({ scale: .2, cellColorBy: 'mixing', reduced: true });
  reduced.o.setBins(grid(0), { at: 0 }); reduced.draw(); reduced.o.setBins(grid(100), { at: 5000 });
  assert.equal(reduced.o.binsPlan, null);
  reduced.o.dispose(); assert.deepEqual(reduced.stage.animators, []); assert.equal(reduced.o.bins, null);
});

// The overlay's two marks: once a test exists its square is drawn dashed on the stage, gone with the test; the
// cell's starting outline is drawn dashed white on the stage while Show is on.
const lines = calls => strokes(calls).map(c => ({ color: c.strokeStyle, width: c.lineWidth, dash: c.dash, alpha: c.globalAlpha }));
test('the test square: dashed over a dark casing while a test exists, one refresh frame (not motion) when it comes or goes', () => {
  const h = overlay({ scale: 2 });
  assert.deepEqual({ ...SQUARE_STYLE, dashes: [...SQUARE_STYLE.dashes] }, { color: 'rgba(230,234,240,.85)', width: 1.5, dashes: [7, 5] });
  assert.equal(SQUARE_STYLE.dashes, DASHES, 'the overlay\'s 7/5 px dashes');
  h.draw(); assert.equal(h.o.due(), null);
  assert.equal(h.o.setTestSquare({ box: [100, 50, 300, 250] }), true); assert.equal(h.stage.replans, 1);
  assert.deepEqual(h.o.testSquare, [100, 50, 300, 250]);
  assert.equal(h.o.due(10), 10, 'one refresh now'); assert.equal(h.o.frame(10), false, 'not motion'); assert.equal(h.redraws.length, 1);
  const calls = h.draw();
  assert.deepEqual(calls.filter(c => c.op === 'rect').map(c => c.args), [[200, 100, 400, 400]], 'in CSS px of the stage');
  assert.deepEqual(lines(calls), [{ color: 'rgba(11,14,19,.55)', width: 3, dash: [], alpha: 1 }, { color: SQUARE_STYLE.color, width: 1.5, dash: [7, 5], alpha: 1 }]);
  assert.equal(h.o.due(), null, 'drawn: idle');
  assert.equal(h.o.setTestSquare({ box: [100, 50, 300, 250] }), false, 'the same square: no work'); assert.equal(h.stage.replans, 1);
  // Gone with the test; a box that is not one is none.
  for (const bad of [null, {}, { box: [1, 2, 3] }, { box: [5, 0, 1, 9] }, { box: [0, 0, NaN, 9] }]) {
    h.o.setTestSquare({ box: [0, 0, 10, 10] });
    assert.equal(h.o.setTestSquare(bad), true, JSON.stringify(bad)); assert.equal(h.o.testSquare, null);
  }
  assert.equal(h.o.due(20), 20); h.o.frame(20); assert.equal(strokes(h.draw()).length, 0); assert.equal(h.o.due(), null);
  // Its own frame even with the mixing check's marks hidden (the test's, not the check's); nothing in 3-D.
  h.o.visible = false; h.o.setTestSquare({ box: [100, 50, 300, 250] });
  assert.equal(h.o.due(30), 30); assert.equal(strokes(h.draw()).length, 2);
  h.o.setTestSquare(null); h.stage.orbit.active = true;
  assert.equal(h.o.due(40), null, '3-D: no frames'); assert.equal(h.draw().length, 0);
});

test('Go deeper\'s starting outline (the stage keeps it, the overlay draws it): dashed white over a casing, never shaded by the focus', () => {
  const h = overlay({ scale: 2 });
  assert.deepEqual({ ...START_STYLE, dashes: [...START_STYLE.dashes] }, { color: '#ffffff', width: 1.5, dashes: [4, 3] });
  h.draw();
  assert.equal(h.o.due(), null);
  h.stage.startOutline = { points: Float32Array.of(10, 10, 20, 10, 20, 30), closed: true, count: 3 };   // Stage.setStartOutline
  assert.equal(h.o.due(5), 5, 'a refresh'); assert.equal(h.o.frame(5), false);
  h.stage.focusLevel = 1;   // Go deeper's focus on: the outline is never shaded
  /** The ops of the path the white dashes stroke (from its beginPath). */
  const traced = calls => {
    const white = calls.findIndex(c => c.op === 'stroke' && c.strokeStyle === '#ffffff');
    return white < 0 ? null : calls.slice(calls.slice(0, white).findLastIndex(c => c.op === 'beginPath'), white)
      .filter(c => /^(moveTo|lineTo|closePath)$/.test(c.op)).map(c => [c.op, ...c.args]);
  };
  const calls = h.draw();
  assert.deepEqual(traced(calls), [['moveTo', 20, 20], ['lineTo', 40, 20], ['lineTo', 40, 60], ['closePath']], 'in CSS px of the stage');
  assert.deepEqual(lines(calls), [{ color: 'rgba(11,14,19,.55)', width: 3, dash: [], alpha: 1 }, { color: '#ffffff', width: 1.5, dash: [4, 3], alpha: 1 }],
    'at full strength, over its casing');
  assert.equal(h.o.due(), null);
  // Open; gone when the stage drops it; a single point is not drawn.
  h.stage.startOutline = { ...h.stage.startOutline, closed: false };
  h.o.frame(6); assert.deepEqual(traced(h.draw()), [['moveTo', 20, 20], ['lineTo', 40, 20], ['lineTo', 40, 60]], 'open: not closed');
  h.stage.startOutline = null;
  assert.equal(h.o.due(7), 7); h.o.frame(7); assert.equal(traced(h.draw()), null);
  h.stage.startOutline = { points: Float32Array.of(1, 1), closed: true, count: 1 };
  assert.equal(traced(h.draw()), null, 'one point: nothing to draw');
});

test('on the real stage scheduler the marks cost one overlay frame each change, then nothing is scheduled', () => {
  const loop = fakeLoop(), frames = [];
  const stage = Object.create(Stage.prototype);
  Object.assign(stage, { canvas: { width: 800, height: 600 }, camera: { x: 0, y: 0, zoom: 1 }, listeners: new Set(), tween: null, raf: null, _dirty: false,
    orbit: { tick() {}, transition: null, active: false }, gl: {}, options: { cellColorBy: 'identity' }, flash: null, flashDue: null, tint: null, tintDue: null,
    animators: new Set(), mixing: { bytes: null }, cssWidth: 800, cssHeight: 600, pixelRatio: 1 });
  stage.draw = function () { this.dirty = false; };
  const o = loop.with(() => new QcOverlay(stage, { redraw: () => { frames.push(loop.now); o.draw(recorder().g); }, reducedMotion: () => false }));
  loop.at(1000, () => stage.setStartOutline({ points: Float32Array.of(10, 10, 20, 10, 20, 30) }));
  loop.at(2000, () => o.setTestSquare({ box: [0, 0, 50, 50] }));
  loop.at(3000, () => { stage.setStartOutline(null); o.setTestSquare(null); });
  loop.run(5000);
  assert.equal(frames.length, 3, `${frames}`);
  assert.ok(frames.every((t, i) => t >= [1000, 2000, 3000][i] && t < [1000, 2000, 3000][i] + 17), `in the next frame: ${frames}`);
  assert.equal(stage.drawCount ?? 0, 0, 'the stage itself never draws for them'); assert.ok(loop.idle());
});

test('the legend moves above the test corner while the corner lies where it sits, and back once it is gone', () => {
  const { document } = domFixture(), legend = new MixingLegend(document.createElement('main'), { doc: document });
  // Its tallest key since 2026-10-22: the cells' title and four rows (99 px at 11 px type).
  assert.deepEqual({ ...LEGEND_PLACE }, { left: 12, bottom: 12, width: 312, height: 104, gap: 8 });
  assert.match(legend.root.style.cssText, /left:12px;bottom:12px;max-width:290px/, 'its place, as the CSS has it');
  // The corner docked at the stage's bottom left (TestCorner.box: {left, top, bottom, w, h}): the legend above it.
  assert.equal(legend.dock({ left: 12, top: 300, bottom: 12, w: 494, h: 416 }), 436); assert.equal(legend.root.style.bottom, '436px');
  assert.equal(legend.dock({ left: 200, bottom: 100, w: 360, h: 250 }), 358, 'moved but still under it');
  assert.equal(legend.dock({ left: 200, bottom: 100, w: 360, h: 250 }, 60), 12, 'a legend of 60 px ends under the corner\'s bottom: it stays');
  assert.equal(legend.dock({ left: 200, bottom: 60, w: 360, h: 250 }, 60), 318, '... unless the corner comes lower');
  // Elsewhere (to the right of it, or higher up the stage), closed, or not a box: its own place.
  for (const corner of [{ left: 330, bottom: 12, w: 494, h: 416 }, { left: 12, bottom: 180, w: 494, h: 200 }, null, { left: 12, bottom: 12, w: 0, h: 10 }, { left: NaN, bottom: 12, w: 9, h: 9 }]) {
    legend.dock({ left: 12, bottom: 12, w: 494, h: 416 });
    assert.equal(legend.dock(corner), 12, JSON.stringify(corner)); assert.equal(legend.root.style.bottom, '12px');
  }
  let writes = 0; const style = legend.root.style, real = style.bottom;
  Object.defineProperty(style, 'bottom', { get: () => real, set: () => { writes++; }, configurable: true });
  legend.dock(null); legend.dock(null);
  assert.equal(writes, 0, 'no style write when nothing changed (the corner reports its box often)');
});

test('2026-10-22, a simpler legend: the legend is a short title and one row per shade; no note, no outline key', () => {
  const { document } = domFixture(), host = document.createElement('main');
  const legend = new MixingLegend(host, { doc: document });
  /** The words in view, in order (a hidden node and all under it left out). */
  const shown = n => n.hidden ? [] : [...(n.text ? [n.text] : []), ...n.childNodes.flatMap(shown)];
  assert.equal(host.children[0], legend.root); assert.equal(legend.root.hidden, true, 'hidden until there is something to explain');
  legend.update({ mixing: true });
  assert.deepEqual(shown(legend.root), ['RNA from neighbours', 'Under 10 %', '10–30 %', '30 % or more (looks mixed)', 'Too few transcripts'],
    'the title, then one row per shade: nothing else');
  assert.deepEqual([C.legendTitle, C.legendLittle, C.legendSome, C.legendMixed, C.legendUnjudged], shown(legend.root));
  const swatches = legend.steps.childNodes.map(r => r.childNodes[0].style.background);
  assert.deepEqual(swatches, [M.little, M.some, M.mixed, UNJUDGED_SWATCH], 'each row its shade, the stripes for too few transcripts');
  assert.ok(UNJUDGED_SWATCH.includes('repeating-linear-gradient(45deg') && UNJUDGED_SWATCH.includes(rgba(M.unjudged, .75)), 'stripes, as on the stage');
  // No outline key (no line of an outline style anywhere) and no note: the areas went, and so did the notes (single cells
  // being noisy, the zoom, the tissue's edge, the 3-D view).
  assert.equal(legend.root.all().some(n => /border-top:2px/.test(n.style?.cssText ?? '')), false);
  for (const key of ['outlines', 'rose', 'already', 'similar', 'view3d', 'overviewNote', 'overviewEdge']) assert.equal(legend[key], undefined, key);
  // Zoomed out the layer shows each patch's share of cells that look mixed, against the tissue's usual level: its own key,
  // the same way: a short title and its three rows.
  legend.update({ mixing: true, overview: true, stops: binStops(.14) });
  assert.deepEqual([legend.colours.hidden, legend.patches.hidden], [true, false]);
  assert.deepEqual(shown(legend.root), ['Cells that look mixed, per patch', 'Up to 14 in 100 (usual here)', 'About 21 in 100', '28 in 100 or more']);
  // In 3-D the cells that are not judged are plain grey (the volume draws no stripes): so is their swatch; no 3-D note.
  legend.update({ mixing: true, view3d: true });
  assert.equal(legend.unjudged.style.background, rgba(M.unjudged, .5));
  assert.deepEqual(shown(legend.root), ['RNA from neighbours', 'Under 10 %', '10–30 %', '30 % or more (looks mixed)', 'Too few transcripts']);
  legend.update({ mixing: true, view3d: true, overview: true, stops: binStops(.14) });
  assert.equal(legend.patches.hidden, true, '3-D: no patches');
  let writes = 0; const style = legend.unjudged.style, real = style.background;
  Object.defineProperty(style, 'background', { get: () => real, set: () => { writes++; }, configurable: true });
  for (let i = 0; i < 5; i++) legend.update({ mixing: true, view3d: true });
  assert.equal(writes, 0, 'no style write when nothing changed (the legend is updated with every overlay redraw)');
  delete style.background; style.background = real;
  // No values yet (the sample still learns): the cells are plain grey, and the legend says why.
  legend.update({ mixing: true, noValues: true });
  assert.deepEqual(shown(legend.root), ['RNA from neighbours', C.legendNoValues]);
  // Without Mixing colours there is nothing to explain, whatever else is asked (the outline key's old `areas`).
  legend.update({ areas: true, styles: new Set(['rose']) });
  assert.equal(legend.root.hidden, true);
  assert.equal(legend.root.style.cssText.includes('pointer-events:none'), true);
  legend.update({});
  assert.equal(legend.root.hidden, true);
});

test('no banned word anywhere the legend can write; the overlay writes none', () => {
  const h = overlay({ scale: .2, cellColorBy: 'mixing' });
  h.o.setBins({ x0: 0, y0: 0, bin: 100, nx: 1, ny: 1, values: Uint8Array.of(100) }, { at: 0 });
  assert.deepEqual(texts(h.draw()), [], 'no words on the overlay (no area chip)');
  const { document } = domFixture(), legend = new MixingLegend(document.createElement('div'), { doc: document }), words = [];
  for (const level of [.05, .12, .3]) { legend.update({ mixing: true, overview: true, stops: binStops(level) }); words.push(...legend.texts()); }
  assert.ok(words.length > 20);
  for (const text of words) assert.deepEqual(bannedWords(text), [], text);
  assert.ok(words.every(t => !/resolvi/i.test(t) && !/\barea/i.test(t)));
});

test('L3: the smoothed per-bin image is clipped to the bins it colours (no colour past the tissue\'s edge)', () => {
  // Path2D as the browser has it: rects of whole bins, in rows of runs.
  const made = [];
  globalThis.Path2D = class { constructor() { this.rects = []; made.push(this); } rect(...a) { this.rects.push(a); } };
  try {
    const data = new Uint8ClampedArray(4 * 4 * 2);
    for (const i of [0, 1, 3, 6]) data[i * 4 + 3] = 200;            // row 0: bins 0, 1, 3; row 1: bin 2 (4 x 2 grid)
    const path = occupiedPath(data, 4, 2);
    assert.deepEqual(path.rects, [[0, 0, 2, 1], [3, 0, 1, 1], [2, 1, 1, 1]]);
    assert.equal(occupiedPath(new Uint8ClampedArray(32), 4, 2), null, 'nothing coloured: no clip');
    const h = overlay({ scale: .2, cellColorBy: 'mixing' });
    h.o.setBins({ x0: 100, y0: 200, bin: 100, nx: 3, ny: 2, values: Uint8Array.of(0, 50, 100, 255, 255, 25) }, { at: 0 }); h.clock.t = 1000;
    const calls = h.draw(), clip = calls.findIndex(c => c.op === 'clip'), image = calls.findIndex(c => c.op === 'drawImage');
    assert.ok(clip >= 0 && clip < image, 'clipped before the image is drawn');
    assert.deepEqual(calls.find(c => c.op === 'translate').args, [20, 40]); assert.deepEqual(calls.find(c => c.op === 'scale').args, [20, 20]);
    assert.deepEqual(calls[image].args.slice(1), [0, 0, 3, 2], 'in bin units, over the same box');
  } finally { delete globalThis.Path2D; }
  // The key's note that edges look more mixed went with the legend's notes (2026-10-22, a simpler legend).
  assert.equal(C.overviewEdge, undefined);
});
