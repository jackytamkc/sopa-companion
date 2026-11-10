import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import {
  Stage, MASK_FS, XENIUM_FS, COLORS, VEIL_FS, VEIL_VS, FOCUS_SOURCES, FOCUS_SAME, FOCUS_LIT, FOCUS_VEIL, FOCUS_SOFT_PX, FOCUS_GREY,
  FOCUS_CELL_ALPHA, FOCUS_TX_ALPHA, FOCUS_TX_GREY, FOCUS_IN_MS, FOCUS_OUT_MS, FOCUS_FRAMES, FOCUS_INNER, focusAlpha, focusPlan, focusValue, focusNext,
  FOCUS_SOURCE_GREY, FOCUS_SOURCE_ALPHA, FOCUS_SOURCE_TX_GREY, FOCUS_SOURCE_TX_ALPHA, FOCUS_SOURCE_VEIL, FOCUS_CELL_BOLD_PX, FOCUS_3D_FACE,
  FOCUS_3D_HIT, FOCUS_3D_WHITE, FOCUS_3D_SOURCE_DARK, FOCUS_3D_DARK, FIT_PASSES, VEIL_COLOR,
} from '../src/stage.js';
import { volumeFragment } from '../src/volume.js';
import { Orbit } from '../src/scene3d.js';
import { whereView, whereFree } from '../src/qc-deeper.js';
import { fakeLoop } from './fake-loop.mjs';
import { pointVertex } from '../src/transcripts.js';
import { DeeperArrows, QcOverlay, arrowWidth, arrowProgress, arrowCurve, ARROWS, ARROW_LABELS, ARROW_MIN_PX, ARROW_DRAW_MS, ARROW_STEP_MS } from '../src/qc-overlay.mjs';
import { UmapView } from '../src/umap.js';
import { bannedWords } from '../src/qc-copy.mjs';

// Go deeper on the stage: the outlines of the window's cell, its sources and its
// same-kind neighbours (stage.js), the arrows on the overlay (qc-overlay.mjs DeeperArrows), the rings in the UMAP
// (umap.js), and the engine's endpoints (api.js). With the window closed nothing draws differently. The
// focus also greys everything it does not involve ("the cells involved stay as they are
// and everything else is greyed out, with a short ease in and out"): the veil, the other cells, their transcripts; the
// arrows stay as they are (and the overlay has no area to shade since 2026-10-22: none is drawn).

/** A WebGL2 context that records calls (name and arguments). */
function recordingGL(log) {
  const constants = new Map(), constant = key => { if (!constants.has(key)) constants.set(key, 1000 + constants.size); return constants.get(key); };
  return new Proxy({}, { get(_, key) {
    if (typeof key !== 'string') return undefined;
    if (/^TEXTURE(\d+)$/.test(key)) return 33984 + Number(key.slice(7));
    if (/^[A-Z0-9_]+$/.test(key)) return constant(key);
    return (...args) => { log.push([key, ...args]); return key === 'getParameter' ? 4096 : {}; };
  } });
}

test('the mask shader: Go deeper\'s outlines never touch a fill; its greying is on col, behind uFocusOn, and with it off every pixel is as it was', () => {
  for (const shader of [MASK_FS, XENIUM_FS]) {
    assert.match(shader, /uniform bool uFocusOn;/); assert.match(shader, /uniform uint uFocusCell, uFocusHot, uFocusSrc\[5\], uFocusSame\[10\];/);
    assert.match(shader, /int focusOf\(uint id, out float a\) \{\n  a = 0\.;\n  if \(!uFocusOn \|\| id == 0u\) return 0;/);
    assert.match(shader, /uniform uint uFocusLit\[6\];\nuniform float uFocusDim, uFocusBold;/);
    assert.ok(shader.includes('bool focusLit(uint id) {\n  for (int k = 0; k < 6; k++) if (id == uFocusLit[k]) return id != 0u;\n  return false;\n}'), 'never the background');
  }
  // The outline width and colours are those of before unless the focus says otherwise (parity with the window closed).
  assert.ok(MASK_FS.includes('  float w = min(uEdgePx, 0.34 * vpx);\n  float focusA = 0.;\n  int focus = focusOf(id, focusA);\n  if (focus == 2 && id == uFocusHot) w = min(uEdgePx + uFocusExtra, 0.45 * vpx);'));
  const edge = MASK_FS.slice(MASK_FS.indexOf('if (id == uSelected) edgeColor = vec3(1.0);'), MASK_FS.indexOf('col = vec4(edgeColor, edgeAlpha);'));
  assert.match(edge, /if \(focus == 1\) edgeColor = vec3\(1\.0\);/); assert.match(edge, /focus == 2\) \{ edgeColor = vec3\(0\.8353, 0\.3176, 0\.5059\)/);
  assert.match(edge, /focus == 3 && mod\(gl_FragCoord\.x \+ gl_FragCoord\.y, 6\.\) < 3\./, 'same kind: grey dots');
  // "Proseg cells that are not lit: their colour mixed 60 % toward grey and their alpha x 0.22 (on col after the
  // edge colour, never on the fill ...)" and "The selection dim is suspended while focused (as spill does now)". So the
  // fill is never painted by the focus; the selection's dim gives way by the focus's strength (exactly 0.3 / 0.45 with
  // it off), and the greying is one line on col, after the trails (a cell changing next door stays grey) and before the
  // spill's fading and the clusters' weight.
  const main = MASK_FS.slice(MASK_FS.indexOf('void main()'));
  assert.doesNotMatch(main, /fill \*= .*focus|col = vec4\(base, .*focus/, 'never a fill');
  assert.ok(main.includes('    float lift = uFocusOn ? uFocusDim : 0.;'));
  assert.ok(main.includes('    if (dimmed) { fill *= mix(0.3, 1.0, lift); edgeAlpha *= mix(0.45, 1.0, lift); }'));
  const grey = `  if (uFocusOn && !focusLit(id)) col = vec4(mix(col.rgb, vec3(dot(col.rgb, vec3(0.2126, 0.7152, 0.0722))), ${FOCUS_GREY} * uFocusDim), col.a * (1.0 - 0.78 * uFocusDim));`;
  assert.ok(main.includes(grey), 'the greying line');
  assert.ok(main.indexOf(grey) > main.indexOf('if (uTrails && recent > 0.01)') && main.indexOf(grey) < main.indexOf('if (uSpillOn && !spillLit(id))')
    && main.indexOf(grey) < main.indexOf('col.a *= clusterWeight(id);'));
  // Every use of the new uniforms in main() is behind uFocusOn (focusOf gives no tier without it): with it off each line
  // is the line it was.
  for (const line of main.split('\n').filter(l => /uFocusDim|focusLit|uFocusBold/.test(l)))
    assert.match(line, /uFocusOn \? uFocusDim|uFocusOn && !focusLit|else if \(uFocusOn && id != uFocusCell\)|else if \(focus == 1\)/, line);
  assert.equal(/uFocus|focusLit/.test(COLORS), false, 'not in the colour functions the transcripts and 3-D share');
  assert.equal(FOCUS_SOURCES, 5); assert.equal(FOCUS_SAME, 10); assert.equal(FOCUS_LIT, 6);
  assert.deepEqual([FOCUS_VEIL, FOCUS_SOFT_PX, FOCUS_GREY, FOCUS_CELL_ALPHA, FOCUS_TX_ALPHA, FOCUS_IN_MS, FOCUS_OUT_MS, FOCUS_FRAMES], [0.62, 2.5, 0.6, 0.22, 0.15, 320, 220, 4]);
  assert.equal(focusAlpha(0.01).toFixed(4), (0.35 + 0.65 * 0.025).toFixed(4)); assert.equal(focusAlpha(0.4), 1); assert.equal(focusAlpha(0.9), 1); assert.equal(focusAlpha(NaN), 0.35);
});

test('the veil: one quad over the whole stage, untouched inside the lit cells (by the frame\'s labels), soft outside them', () => {
  // "Everything outside the lit cells' footprints (background and every other cell, inside and outside the frame
  // raster) is darkened by 0.62 · dim · ease and pulled toward grey; inside the lit cells nothing changes. The veil reads
  // the frame's label texture ... a soft 2-3 CSS px falloff outside the lit cells (a few texel taps)".
  assert.ok(VEIL_VS.includes('vWorld = uView.xy + (aPos - 0.5) * uCanvas / uView.z;'), 'the whole drawing buffer, in µm');
  // "Inside the lit cells nothing changes" became: nothing unless the suspects glow (uInner > 0, a pass of
  // its own: the test below); the outer pass sets uInner 0, so a lit cell's pixel is left alone exactly as before.
  const main = VEIL_FS.slice(VEIL_FS.indexOf('void main()'));
  assert.ok(main.includes('  uint id = labelAt(c);\n  if (litId(id)) {\n    float strength = id == uCell ? uInner : max(uInner, uSource);\n    if (strength <= 0.) discard;'),
    'inside the window\'s cell: nothing (without the glow); inside its other lit cells uSource');
  assert.ok(VEIL_FS.includes('if (c.x < 0 || c.y < 0 || c.x >= uSize.x || c.y >= uSize.y) return 0u;'), 'outside the raster: the background, veiled');
  assert.ok(VEIL_FS.includes('bool litId(uint id) {\n  if (id == 0u) return false;'), 'the background is never lit');
  assert.ok(main.includes('  if (uVeil <= 0.) discard;'), 'the inner pass leaves everything else as it is');
  assert.match(VEIL_FS, /if \(\(dx != 0 \|\| dy != 0\) && d < near && lit\(c \+ ivec2\(dx, dy\)\)\) near = d;/, 'a tap only where a neighbour voxel is within reach');
  assert.match(VEIL_FS, /uVeil \* smoothstep\(0\., 1\., near \/ reach\)\);/);
  assert.equal(/gl_FragCoord|uSpill|uSelected/.test(VEIL_FS), false);
});

test('while the suspects glow the veil also lies inside the lit cells at 0.5, never on their outlines, under the dots', () => {
  // "While the glow shows, the veil also lies inside the lit cells at 0.4 (their outlines stay bright; the stage's image
  // there dims, the glow does not)".
  assert.equal(FOCUS_INNER, 0.5, 'first 0.4, raised after a browser run showed halos on bright stain');
  // The outlines: the mask's own edge test (a label change within the outline's width of the voxel's edge, 4 sides and 4
  // corners), as wide as the mask draws them (capped at 0.34 of a voxel; a hot source's 1 CSS px wider, capped at 0.45).
  const inner = VEIL_FS.slice(VEIL_FS.indexOf('  if (litId(id)) {'), VEIL_FS.indexOf('  if (uVeil <= 0.) discard;'));
  assert.ok(inner.includes('float w = id == uHot ? min(uHotPx, 0.45 * uVoxelPx) : id == uCell ? min(uEdgePx + uBoldPx, 0.45 * uVoxelPx) : min(uEdgePx, 0.34 * uVoxelPx);'));
  assert.ok(MASK_FS.includes('float w = min(uEdgePx, 0.34 * vpx);') && MASK_FS.includes('w = min(uEdgePx + uFocusExtra, 0.45 * vpx);')
    && MASK_FS.includes('else if (focus == 1) w = min(uEdgePx + uFocusBold, 0.45 * vpx);'), 'the mask\'s widths (the window\'s cell bolder)');
  const mask = MASK_FS.replace(/ +/g, ' ');   // its lines are aligned with extra spaces
  for (const [side, offset] of [['d.x < w', '-1, 0'], ['d.y < w', '1, 0'], ['d.z < w', '0, -1'], ['d.w < w', '0, 1'], ['d.x < w && d.z < w', '-1, -1'],
    ['d.y < w && d.z < w', '1, -1'], ['d.x < w && d.w < w', '-1, 1'], ['d.y < w && d.w < w', '1, 1']]) {
    assert.ok(inner.includes(`(${side} && labelAt(c + ivec2(${offset})) != id)`), side);
    assert.ok(mask.includes(side.includes('&&') ? `if (!edge && ${side} && labelAt(c + ivec2(${offset})) != id) edge = true;`
      : `if (${side}) { uint n = labelAt(c + ivec2(${offset})); if (n != id) {`), `the mask tests ${side} the same way`);
  }
  assert.ok(inner.includes('vec4 d = vec4(f.x, 1.0 - f.x, f.y, 1.0 - f.y) * uVoxelPx;') && MASK_FS.includes('vec4 d = vec4(f.x, 1.0 - f.x, f.y, 1.0 - f.y) * vpx;'));
  assert.ok(inner.includes(') discard;\n    outColor = vec4(vec3(0.0392, 0.0431, 0.0549), strength);'), 'the veil\'s colour at its strength, off the outlines');
});

/** setGlow on a stage (the stage's own glow layer schedules through replan, which reads document: a fake loop's). */
const glowing = (loop, stage, on) => loop.with(() => stage.setGlow(on ? { key: on, points: Float32Array.of(1, 1, 0.9), giver: null } : null));
/** The uniforms each pass of `program` set, in order: [{name: value}]. */
const passes = (log, program) => {
  const out = [];
  for (const c of log) {
    if (c[0] === 'useProgram') { if (c[1] === program) out.push({}); else if (out.length && !out.at(-1).done) out.at(-1).done = true; }
    else if (out.length && !out.at(-1).done && c[0].startsWith('uniform') && typeof c[1] === 'string') out.at(-1)[c[1]] = c[2];
  }
  return out.map(({ done, ...u }) => u);
};

test('on the stage: an inner pass after the mask and before the transcripts or dots while the glow is set, the outer veil still last', () => {
  const loop = fakeLoop(), log = [], stage = focusStage(log);
  stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [], hot: 3 });
  log.length = 0; stage.draw(0);
  const without = log.map(c => [...c]);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'mask', 'pointsFocus', 'veil'], 'no glow: as before');
  stage._dirty = false;
  assert.equal(glowing(loop, stage, 'a'), true); assert.equal(stage.dirty, true, 'the lit cells dim: one redraw');
  assert.equal(stage.innerLevel, 1, 'calm motion: at once');
  log.length = 0; stage.draw(0);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'mask', 'veil', 'pointsFocus', 'veil'], 'inside under the transcripts; outside last');
  const [inside, outside] = passes(log, 'veil');
  assert.deepEqual([inside.uVeil, inside.uInner], [0, FOCUS_INNER], 'the inner pass: the lit cells only');
  assert.deepEqual([outside.uVeil, outside.uInner], [FOCUS_VEIL, 0], 'the outer pass: as before, the lit cells untouched');
  // The outlines as the mask draws them: 1.6 CSS px at ratio 2; the hot row (a source) 1 CSS px wider.
  assert.deepEqual([inside.uEdgePx, inside.uHotPx, inside.uHot], [3.2, 5.2, 3]);
  assert.deepEqual([...inside['uLit[0]']], [7, 3, 9, 0, 0, 0]);
  // With Go deeper's dots on: the inner pass under them, the outer veil over them.
  loop.with(() => stage.setSpill({ key: 's', cell: 7, lit: [3, 9], points: Float32Array.of(1, 1, 0, 1), colors: ['#f0e442'] }));
  log.length = 0; stage.draw(0);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'mask', 'veil', 'spill', 'veil']);
  loop.with(() => stage.setSpill(null));
  // As strong as the focus: half its strength, half the inner veil; a same-kind row hot: no wider outline (the mask widens
  // only a source's); the cells hidden: no outline to keep clear, the whole inside dims.
  stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4], hot: 4, lit: [7, 3, 9, 4], dim: 0.5 });
  log.length = 0; stage.draw(0);
  let [half] = passes(log, 'veil');
  assert.ok(Math.abs(half.uInner - FOCUS_INNER / 2) < 1e-9); assert.equal(half.uHot, 0);
  stage.options.cells = false; log.length = 0; stage.draw(0);
  [half] = passes(log, 'veil');
  assert.deepEqual([half.uEdgePx, half.uHotPx], [0, 0]); stage.options.cells = true;
  // The glow off: exactly the calls of the focus without it.
  stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [], hot: 3 });
  stage._dirty = false;
  assert.equal(glowing(loop, stage, null), true); assert.equal(stage.dirty, true); assert.equal(stage.innerLevel, 0);
  log.length = 0; stage.draw(0);
  assert.deepEqual(log, without, 'the GL calls and uniforms of before the glow');
  // 3-D: none of it (no focus is drawn there).
  glowing(loop, stage, 'b'); stage.orbit.active = true; log.length = 0; stage.draw(0);
  assert.equal(programs(log).includes('veil'), false);
});

test('the inner veil eases in with the glow and out without it, as the focus does; at once with reduced or calm motion, in 3-D or with no focus', () => {
  const loop = fakeLoop(), stage = focusStage([], { calmMotion: false });
  stage.setFocus({ cell: 7, sources: [[3, 0.46]] });
  stage.focusFade.at -= FOCUS_IN_MS; stage.tick(0);
  assert.equal(stage.focusLevel, 1);
  glowing(loop, stage, 'a');
  assert.deepEqual([stage.innerFade.from, stage.innerFade.to, stage.innerFade.duration], [0, 1, FOCUS_IN_MS]);
  assert.equal(+stage.innerLevel.toFixed(3), 0.156, 'the first of the four steps');
  assert.equal(stage.nextDue(stage.innerFade.at), stage.innerFade.at, 'due now: the stage plans its steps');
  stage.innerFade.at -= FOCUS_IN_MS; stage.tick(0);
  assert.equal(stage.innerLevel, 1); assert.equal(stage.innerFade, null); assert.equal(stage.nextDue(0), null, 'then nothing is due');
  glowing(loop, stage, 'b');
  assert.equal(stage.innerFade, null, 'another glow: it stays');
  glowing(loop, stage, null);
  assert.deepEqual([stage.innerFade.from, stage.innerFade.to, stage.innerFade.duration], [1, 0, FOCUS_OUT_MS]);
  stage.innerFade.at -= FOCUS_OUT_MS; stage.tick(0);
  assert.equal(stage.innerLevel, 0); assert.equal(stage.innerFade, null);
  // Reduced motion, the calm switch, 3-D, no focus drawn: at once.
  for (const extra of [{ motionQuery: { matches: true } }, { calmMotion: true }, { orbit: { active: true, uniforms() {} } }, {}]) {
    const s = focusStage([], { calmMotion: false, ...extra });
    if (Object.keys(extra).length) s.setFocus({ cell: 7, sources: [] });
    glowing(loop, s, 'a'); assert.equal(s.innerFade, null); assert.equal(s.innerLevel, 1);
    glowing(loop, s, null); assert.equal(s.innerFade, null); assert.equal(s.innerLevel, 0);
  }
});

/** A Stage with every 2-D pass draw() has (the image left out), on a recording context: the mask, the veil, the plain and
 *  the focus transcripts' programs, Go deeper's dots; the Xenium masks and priors log where they draw; a stand-in for its
 *  glow layer (glow.test.mjs has the real one): set(glow) changes with the key, count 1 while set. Calm motion: the
 *  focus lands at once. */
function focusStage(log, extra = {}) {
  const stage = Object.create(Stage.prototype);
  const names = list => Object.fromEntries(list.map(n => [n, n]));
  Object.assign(stage, { gl: recordingGL(log), canvas: { width: 100, height: 80 }, camera: { x: 1, y: 1, zoom: 4 }, pixelRatio: 2, image: null,
    orbit: { active: false, uniforms() {}, tick() {} }, textures: { channels: [], lab: 'lab', age: 'age', consensus: 'consensus' }, quadVao: 'quad', pointVao: 'pointVao',
    layerTexture: 'layers', consensusIsLabels: true, drawCount: 0, frame: { x0: 0, y0: 0, nx: 2, ny: 2, voxel: 1, iter: 3 },
    options: { cells: true, mode: 0, fill: 0.42, trails: false, edgePx: 1.6, hover: 0, selected: 7, cellColorBy: 'identity', transcripts: true,
      transcriptAlpha: 0.7, transcriptSize: 1.8, markOffFootprint: true, transcriptColorBy: 'cell' },
    transcripts: new ArrayBuffer(64), info: { z: { grid_origin: 0, grid_spacing: 1 } }, clusterGPU: { bind() {} }, calmMotion: true, animators: new Set(),
    maskProgram: { program: 'mask', uniforms: names(['uFocusOn', 'uFocusCell', 'uFocusHot', 'uFocusExtra', 'uFocusSrc[0]', 'uFocusA[0]', 'uFocusSame[0]', 'uFocusLit[0]', 'uFocusDim', 'uFocusBold', 'uSpillOn', 'uSelected']) },
    veilProgram: { program: 'veil', uniforms: names(['uView', 'uCanvas', 'uLab', 'uGeom', 'uSize', 'uVoxelPx', 'uSoftPx', 'uLit[0]', 'uVeil', 'uInner', 'uEdgePx', 'uHotPx', 'uHot', 'uBoldPx', 'uCell', 'uSource']) },
    pointProgram: { program: 'points', uniforms: names(['uTranscriptAlpha']) },
    pointFocusProgram: { program: 'pointsFocus', uniforms: names(['uTranscriptAlpha', 'uFocusLit[0]', 'uFocusCell', 'uFocusDim']) },
    spillProgram: { program: 'spill', uniforms: names(['uSpillAlpha']) }, spillVao: 'spillVao', spillBuffer: 'spillBuffer',
    glowLayer: { key: '', count: 0, set(g) { const key = g?.key ?? ''; if (key === this.key) return false; this.key = key; this.count = g ? 1 : 0; return true; } },
    drawXenium() { log.push(['xenium']); }, drawPriors() { log.push(['priors']); }, raf: 1, _dirty: false, ...extra });
  return stage;
}
const programs = log => log.filter(c => c[0] === 'useProgram' || c[0] === 'xenium' || c[0] === 'priors').map(c => c[1] ?? c[0]);

test('setFocus: lit (the cell first, each once, at most 6; by default the cell and its sources) and dim (0-1, by default 1); the same focus again: no redraw', () => {
  const stage = focusStage([]);
  assert.equal(stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4], hot: 0 }), true);
  assert.deepEqual([...stage.focus.lit], [7, 3, 9, 0, 0, 0], 'the default: the cell and its sources');
  assert.equal(stage.focus.dim, 1); assert.equal(stage.focusLevel, 1, 'calm motion: at once');
  stage._dirty = false;
  assert.equal(stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4], hot: 0, motion: true }), false, 'the same focus (motion is the window\'s own)');
  assert.equal(stage.dirty, false);
  assert.equal(stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4], hot: 0, lit: [7, 3, 9, 4] }), true, 'a same-kind row lit too');
  assert.deepEqual([...stage.focus.lit], [7, 3, 9, 4, 0, 0]);
  assert.equal(stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4], hot: 0, lit: [7, 3, 9, 4] }), false);
  stage.setFocus({ cell: 7, sources: [], same: [], lit: [3, 3, 0, 7, 11, 12, 13, 14, 15] });
  assert.deepEqual([...stage.focus.lit], [7, 3, 11, 12, 13, 14], 'the cell first, each once, never 0, at most 6');
  stage.setFocus({ cell: 7, sources: [], same: [], lit: [] });
  assert.deepEqual([...stage.focus.lit], [7, 0, 0, 0, 0, 0], 'an empty list: the cell alone');
  assert.equal(stage.setFocus({ cell: 7, sources: [], same: [], lit: [], dim: 0.5 }), true, 'another strength');
  assert.equal(stage.focus.dim, 0.5); assert.equal(stage.focusLevel, 0.5);
  for (const [dim, want] of [[2, 1], [-1, 0], [NaN, 1], ['x', 1]]) { stage.setFocus({ cell: 8, dim }); assert.equal(stage.focus.dim, want, `${dim}`); }
});

test('the veil draws only while focused (2-D, a frame): last, over the cells and the transcripts, as strong as the focus', () => {
  const log = [], stage = focusStage(log);
  stage.draw(0);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'mask', 'points'], 'no focus: no veil, the plain transcripts');
  stage.setFocus({ cell: 7, sources: [[3, 0.46]], same: [], hot: 0 });
  log.length = 0; stage.draw(0);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'mask', 'pointsFocus', 'veil'], 'last: piled-up transcripts darken too');
  const u = uniforms(log);
  assert.equal(u.uVeil, FOCUS_VEIL); assert.equal(u.uSoftPx, FOCUS_SOFT_PX * 2, 'CSS px in device px'); assert.equal(u.uVoxelPx, 4);
  assert.deepEqual([...u['uLit[0]']], [7, 3, 0, 0, 0, 0]); assert.equal(u.uLab, 0);
  const veil = log.findIndex(c => c[0] === 'useProgram' && c[1] === 'veil');
  assert.deepEqual(log.slice(veil).find(c => c[0] === 'bindTexture').slice(2), ['lab'], 'the frame\'s labels');
  assert.deepEqual(log.slice(veil).find(c => c[0] === 'drawArrays').slice(2), [0, 4], 'one quad');
  // The mask: the lit set and the strength; the transcripts: the focus program with the same.
  assert.deepEqual([...u['uFocusLit[0]']], [7, 3, 0, 0, 0, 0]); assert.equal(u.uFocusDim, 1);
  // Half strength: the veil and the greying follow it.
  stage.setFocus({ cell: 7, sources: [[3, 0.46]], same: [], hot: 0, dim: 0.5 });
  log.length = 0; stage.draw(0);
  assert.ok(Math.abs(uniforms(log).uVeil - FOCUS_VEIL / 2) < 1e-9); assert.equal(uniforms(log).uFocusDim, 0.5);
  // Strength 0: the outlines alone (no veil to draw).
  stage.setFocus({ cell: 7, sources: [[3, 0.46]], same: [], hot: 0, dim: 0 });
  log.length = 0; stage.draw(0);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'mask', 'pointsFocus']); assert.equal(uniforms(log).uFocusOn, 1);
  // No frame: nothing to read the cells from, no veil (and no cells).
  stage.setFocus({ cell: 7, sources: [[3, 0.46]], same: [], hot: 0 });
  const frame = stage.frame; stage.frame = null; log.length = 0; stage.draw(0);
  assert.equal(programs(log).includes('veil'), false); stage.frame = frame;
  // Cells hidden: the veil still lights the cells' places on the image.
  stage.options.cells = false; log.length = 0; stage.draw(0);
  assert.deepEqual(programs(log), ['xenium', 'priors', 'pointsFocus', 'veil']); stage.options.cells = true;
  // 3-D (here a view with no layers: no volume): the transcripts' focus program, with the same tiers; no veil, no
  // outlines (the mask's uFocusOn 0).
  stage.orbit.active = true; log.length = 0; stage.draw(0);
  assert.equal(programs(log).includes('veil'), false); assert.equal(programs(log).includes('pointsFocus'), true); assert.equal(uniforms(log).uFocusOn, 0);
  assert.equal(uniforms(log).uFocusCell, 7); assert.deepEqual([...uniforms(log)['uFocusLit[0]']], [7, 3, 0, 0, 0, 0]);
});

test('with the window closed (its ease over) the GL calls and uniforms are exactly those of a stage that never had a focus', () => {
  // "Nothing changes with the window closed — identical GL calls and uniforms, as the tests pin today".
  for (const calm of [true, false]) {
    const before = [], never = focusStage(before, { calmMotion: calm });
    never.draw(0);
    const log = [], stage = focusStage(log, { calmMotion: calm });
    stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4, 5], hot: 3, lit: [7, 3, 9, 4] });
    stage.draw(0);
    stage.setFocus(null);
    if (!calm) {   // easing out: still drawn, weaker; then over
      assert.ok(stage.focusShown() && stage.focusLevel > 0 && stage.focusLevel < 1); assert.equal(stage.focus, null, 'the harness sees it closed at once');
      stage.focusFade.at -= FOCUS_OUT_MS; stage.tick(0);
      assert.equal(stage.focusLevel, 0); assert.equal(stage.focusShown(), null); assert.equal(stage.focusFade, null); assert.equal(stage.focusOut, null);
    }
    log.length = 0; stage.draw(0);
    assert.deepEqual(log, before, calm ? 'calm motion' : 'after the ease');
    assert.equal(uniforms(log).uFocusOn, 0); assert.equal(uniforms(log).uFocusLit, undefined);
  }
});

test('the focus eases: in over 320 ms on open and on a new cell (from none), out over 220 ms on close, 4 smoothed steps each; at once with reduced or calm motion or in 3-D', () => {
  assert.equal(focusPlan(0, 1, true), null); assert.equal(focusPlan(1, 1), null);
  const into = focusPlan(0, 1), out = focusPlan(1, 0);
  assert.deepEqual(into, { from: 0, to: 1, duration: FOCUS_IN_MS, frames: FOCUS_FRAMES }); assert.equal(out.duration, FOCUS_OUT_MS);
  const steps = plan => [0, 1, 2, 3].map(k => plan.duration * k / 3);
  assert.deepEqual(steps(into).map(t => +focusValue(into, t).toFixed(3)), [0.156, 0.5, 0.844, 1]);
  assert.deepEqual(steps(out).map(t => +focusValue(out, t).toFixed(3)), [0.844, 0.5, 0.156, 0]);
  assert.equal(focusValue(into, 100), focusValue(into, 0), 'between steps: the step before');
  assert.deepEqual(steps(into).map(t => focusNext(into, t)), [...steps(into).slice(1), null]);
  // On the stage: a new cell starts from none even while another is lit; the same cell keeps its strength.
  const stage = focusStage([], { calmMotion: false });
  stage.setFocus({ cell: 7, sources: [] });
  assert.deepEqual([stage.focusFade.from, stage.focusFade.to, stage.focusFade.duration], [0, 1, FOCUS_IN_MS]);
  stage.focusFade.at -= FOCUS_IN_MS; stage.tick(0); assert.equal(stage.focusLevel, 1);
  stage.setFocus({ cell: 7, sources: [], hot: 3 });
  assert.equal(stage.focusFade, null, 'the same cell: no ease'); assert.equal(stage.focusLevel, 1);
  stage.setFocus({ cell: 9, sources: [] });
  assert.equal(stage.focusFade.from, 0, 'another cell: from none');
  // Reduced motion, the calm switch, 3-D: at once.
  for (const extra of [{ motionQuery: { matches: true } }, { calmMotion: true }, { orbit: { active: true, uniforms() {} } }]) {
    const s = focusStage([], { calmMotion: false, ...extra });
    s.setFocus({ cell: 7, sources: [] }); assert.equal(s.focusFade, null); assert.equal(s.focusLevel, 1);
    s.setFocus(null); assert.equal(s.focusFade, null); assert.equal(s.focusLevel, 0); assert.equal(s.focusShown(), null);
  }
});

test('the transcripts: the plain program unchanged with the window closed; the focus variant greys and fades every cell but the lit ones, outside the shared colours', () => {
  const plain = pointVertex(COLORS), focus = pointVertex(COLORS, { lit: FOCUS_LIT, alpha: FOCUS_TX_ALPHA, grey: FOCUS_TX_GREY,
    sourceAlpha: FOCUS_SOURCE_TX_ALPHA, sourceGrey: FOCUS_SOURCE_TX_GREY });
  assert.equal(/uFocus|focusLit/.test(plain), false);
  const block = 'uniform uint uFocusLit[6];\nuniform uint uFocusCell;\nuniform float uFocusDim;\nbool focusLit(uint id) {\n  for (int k = 0; k < 6; k++) if (id == uFocusLit[k]) return id != 0u;\n  return false;\n}\n';
  // Grey as well as fainter (the browser run of 2026-10-22: piled-up faded dots kept their cell's colour); the lit cells
  // but the window's own (the rows it names) a little of the way: second.
  const line = '  if (!focusLit(id)) {   // outside the shared colours\n'
    + '    vColor.rgb = mix(vColor.rgb, vec3(dot(vColor.rgb, vec3(0.2126, 0.7152, 0.0722))), 0.85 * uFocusDim);\n'
    + '    vColor.a *= 1. - 0.85 * uFocusDim;\n'
    + '  } else if (id != uFocusCell) {   // the rows the window names: second to its cell\n'
    + '    vColor.rgb = mix(vColor.rgb, vec3(dot(vColor.rgb, vec3(0.2126, 0.7152, 0.0722))), 0.35 * uFocusDim);\n'
    + '    vColor.a *= 1. - 0.45 * uFocusDim;\n  }\n';
  // The focus variant is the plain shader plus exactly those (each once): the uniforms and the test after the shared
  // colours, the fading after the clusters' weight.
  assert.equal(focus.split(block).length, 2); assert.equal(focus.split(line).length, 2);
  assert.equal(focus.replace(block, '').replace(line, ''), plain);
  assert.ok(focus.indexOf(block) >= focus.indexOf(COLORS) + COLORS.length, 'after the shared colours');
  assert.ok(focus.indexOf(line) > focus.indexOf('vColor.a *= clusterWeight(id);'));
});

/** A Stage with just what draw() needs for the 2-D mask, on a recording context. */
function maskStage(log) {
  const stage = Object.create(Stage.prototype);
  Object.assign(stage, { gl: recordingGL(log), canvas: { width: 100, height: 80 }, camera: { x: 0, y: 0, zoom: 1 }, pixelRatio: 2, image: null,
    orbit: { active: false, uniforms() {} }, textures: { channels: [], lab: {}, age: {} }, quadVao: {}, drawCount: 0, frame: { x0: 0, y0: 0, nx: 2, ny: 2, voxel: 1, iter: 3 },
    options: { cells: true, mode: 0, fill: 0.42, trails: false, edgePx: 1.6, hover: 0, selected: 0, cellColorBy: 'identity' }, clusterGPU: { bind() {} },
    maskProgram: { program: {}, uniforms: { uFocusOn: 'on', uFocusCell: 'cell', uFocusHot: 'hot', uFocusExtra: 'extra', 'uFocusSrc[0]': 'src', 'uFocusA[0]': 'a', 'uFocusSame[0]': 'same' } },
    drawTranscripts() {}, raf: 1, _dirty: false });
  return stage;
}
const uniforms = log => Object.fromEntries(log.filter(c => c[0].startsWith('uniform') && typeof c[1] === 'string').map(c => [c[1], c[2]]));

test('Stage.setFocus: redraws only on a change; the uniforms carry it; none (and 3-D) draws exactly as before', () => {
  const log = [], stage = maskStage(log);
  stage.calmMotion = true;   // closing lands at once (its ease out is the test above's)
  stage.draw(0);
  assert.equal(uniforms(log).on, 0); assert.equal(uniforms(log).src, undefined, 'nothing else is set without a focus');
  assert.equal(stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4, 5], hot: 3 }), true);
  assert.equal(stage.dirty, true);
  stage._dirty = false;
  assert.equal(stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4, 5], hot: 3 }), false, 'the same focus: no redraw');
  assert.equal(stage.dirty, false);
  log.length = 0; stage.draw(0);
  const u = uniforms(log);
  assert.equal(u.on, 1); assert.equal(u.cell, 7); assert.equal(u.hot, 3); assert.equal(u.extra, 2, '1 CSS px in device px');
  assert.deepEqual([...u.src], [3, 9, 0, 0, 0]); assert.deepEqual([...u.same], [4, 5, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(u.a[0], 1); assert.ok(Math.abs(u.a[1] - focusAlpha(0.02)) < 1e-6);
  // 3-D: no outlines (the arrows are 2-D only too).
  stage.orbit.active = true; log.length = 0; stage.draw(0); assert.equal(uniforms(log).on, 0);
  stage.orbit.active = false;
  assert.equal(stage.setFocus(null), true); log.length = 0; stage.draw(0); assert.equal(uniforms(log).on, 0);
  assert.equal(stage.setFocus({ cell: 0, sources: [] }), false, 'no cell: no focus');
  assert.equal(stage.setFocus({ cell: 7, sources: Array.from({ length: 9 }, (_, k) => [k + 1, 0.1]), same: Array.from({ length: 14 }, (_, k) => k + 20) }), true);
  assert.equal(stage.focus.src.length, 5); assert.equal(stage.focus.same.length, 10);
});

// ---- the round of 2026-10-31: the cell first, its sources second, the rest dimmed, in 2-D and 3-D; the camera in 3-D ----
test('the focus\'s tiers in 2-D: the window\'s cell clearest (its white outline bolder), the rows it names second (a little grey, a lighter veil), every other cell dimmed', () => {
  assert.deepEqual([FOCUS_SOURCE_GREY, FOCUS_SOURCE_ALPHA, FOCUS_SOURCE_TX_GREY, FOCUS_SOURCE_TX_ALPHA, FOCUS_SOURCE_VEIL, FOCUS_CELL_BOLD_PX, FOCUS_3D_FACE, FOCUS_3D_HIT],
    [0.35, 0.6, 0.35, 0.55, 0.4, 1.5, 2, 0.02]);
  assert.deepEqual([FOCUS_3D_WHITE, FOCUS_3D_SOURCE_DARK, FOCUS_3D_DARK], [0.35, 0.25, 0.45]);
  assert.ok(FOCUS_3D_SOURCE_DARK < FOCUS_3D_DARK, 'in 3-D too the rows lose less than every other cell');
  // Second means between: greyer, fainter and more veiled than the cell (untouched), less than every other cell.
  assert.ok(FOCUS_SOURCE_GREY < FOCUS_GREY && FOCUS_SOURCE_ALPHA > FOCUS_CELL_ALPHA && FOCUS_SOURCE_VEIL < FOCUS_VEIL);
  assert.ok(FOCUS_SOURCE_TX_GREY < FOCUS_TX_GREY && FOCUS_SOURCE_TX_ALPHA > FOCUS_TX_ALPHA);
  // The mask: the rows' line right after every other cell's, fill and outline alike (the magenta outline lighter, its
  // width as it was); the window's cell's outline wider by uFocusBold, capped as a hot row's.
  const main = MASK_FS.slice(MASK_FS.indexOf('void main()'));
  const grey = `  if (uFocusOn && !focusLit(id)) col = vec4(mix(col.rgb, vec3(dot(col.rgb, vec3(0.2126, 0.7152, 0.0722))), ${FOCUS_GREY} * uFocusDim), col.a * (1.0 - 0.78 * uFocusDim));\n`;
  const second = '  else if (uFocusOn && id != uFocusCell) col = vec4(mix(col.rgb, vec3(dot(col.rgb, vec3(0.2126, 0.7152, 0.0722))), 0.35 * uFocusDim), col.a * (1.0 - 0.40 * uFocusDim));\n';
  assert.ok(main.includes(grey + second), 'the rows: second');
  assert.ok(main.includes('  if (focus == 2 && id == uFocusHot) w = min(uEdgePx + uFocusExtra, 0.45 * vpx);\n  else if (focus == 1) w = min(uEdgePx + uFocusBold, 0.45 * vpx);'));
  assert.ok(main.indexOf('else if (focus == 1) w =') < main.indexOf('vec4 d = vec4('), 'the wider outline enters the edge test');
  // On the stage: the mask's bold width (1.5 CSS px in device px), the veil's cell and its rows' strength in the last pass,
  // none in the inner (glow) pass; the cell's outline kept clear as wide as the mask draws it.
  const loop = fakeLoop(), log = [], stage = focusStage(log);
  stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [], hot: 0 });
  log.length = 0; stage.draw(0);
  assert.equal(uniforms(log).uFocusBold, FOCUS_CELL_BOLD_PX * 2);
  let [outer] = passes(log, 'veil');
  assert.deepEqual([outer.uCell, outer.uSource, outer.uBoldPx, outer.uVeil], [7, FOCUS_SOURCE_VEIL, FOCUS_CELL_BOLD_PX * 2, FOCUS_VEIL]);
  glowing(loop, stage, 'a'); log.length = 0; stage.draw(0);
  const [inside] = passes(log, 'veil'); [, outer] = passes(log, 'veil');
  assert.deepEqual([inside.uSource, inside.uInner, outer.uSource, outer.uInner], [0, FOCUS_INNER, FOCUS_SOURCE_VEIL, 0]);
  // Half strength: the rows' veil follows it; cells hidden: no outline kept clear.
  stage.setFocus({ cell: 7, sources: [[3, 0.46]], same: [], hot: 0, dim: 0.5 }); log.length = 0; stage.draw(0);
  assert.ok(Math.abs(passes(log, 'veil').at(-1).uSource - FOCUS_SOURCE_VEIL / 2) < 1e-9);
  stage.options.cells = false; log.length = 0; stage.draw(0);
  assert.equal(passes(log, 'veil').at(-1).uBoldPx, 0);
});

/** A Stage in 3-D with the volume (a frame with layers) on a recording context: its volume and transcripts' programs. */
function volumeStage(log, extra = {}) {
  const names = list => Object.fromEntries(list.map(n => [n, n]));
  return focusStage(log, { orbit: { active: true, enabled: true, uniforms() {}, tick() {} },
    frame: { x0: 0, y0: 0, nx: 2, ny: 2, voxel: 1, iter: 3, layers: [new Uint32Array(4)] },
    volumeTargets: { width: 100, height: 80, scene: 'scene', pointScene: 'pointScene', output: 'output', color: 'c', depth: 'd', points: 'p', resize() {} },
    volumeProgram: { program: 'volume', uniforms: names(['uFocusOn', 'uFocusCell', 'uFocusLit[0]', 'uFocusDim', 'uCells', 'uOpacity']) }, ...extra });
}
const volumePass = log => passes(log, 'volume')[0];

test('in 3-D the volume and the transcripts take the focus\'s tiers (focusShown()\'s, at once); with none no focus uniform is set, as a stage that never had one', () => {
  const options = { ...focusStage([]).options, volumeOpacity: 1.4, volumeFog: 0.08, volumeBox: 0.22 };
  const before = [], never = volumeStage(before, { options });
  never.draw(0);
  assert.deepEqual(programs(before), ['points', 'volume']);
  assert.deepEqual(volumePass(before), { uCells: true, uOpacity: 1.4, uFocusOn: 0 }, 'no focus: off, nothing else');
  const log = [], stage = volumeStage(log, { options: { ...options } });
  stage.setFocus({ cell: 7, sources: [[3, 0.46], [9, 0.02]], same: [4], hot: 3, lit: [7, 3, 9] });
  assert.equal(stage.focusFade, null, 'at once in 3-D'); assert.equal(stage.focusLevel, 1);
  log.length = 0; stage.draw(0);
  assert.deepEqual(programs(log), ['pointsFocus', 'volume'], 'the transcripts\' focus program, then the volume');
  const v = volumePass(log);
  assert.deepEqual([v.uFocusOn, v.uFocusCell, [...v['uFocusLit[0]']], v.uFocusDim], [1, 7, [7, 3, 9, 0, 0, 0], 1]);
  const t = passes(log, 'pointsFocus')[0];
  assert.deepEqual([t.uFocusCell, [...t['uFocusLit[0]']], t.uFocusDim], [7, [7, 3, 9, 0, 0, 0], 1]);
  // At half strength (the window's dim) both take the strength drawn, at once in 3-D.
  stage.setFocus({ cell: 7, sources: [[3, 0.46]], same: [], hot: 0, lit: [7, 3], dim: 0.5 });
  log.length = 0; stage.draw(0);
  assert.deepEqual([volumePass(log).uFocusDim, passes(log, 'pointsFocus')[0].uFocusDim, stage.focusLevel], [0.5, 0.5, 0.5]);
  // Closed: at once in 3-D, and every call as the stage that never had one.
  stage.setFocus(null); assert.equal(stage.focusShown(), null);
  log.length = 0; stage.draw(0);
  const plain = [], again = volumeStage(plain, { options: stage.options }); again.draw(0);
  assert.deepEqual(log, plain);
});

test('the volume\'s focus variant: the plain shader plus exactly its lines, each behind uFocusOn; the rows and every other cell greyed and darkened by tier, no tier lowering an alpha (only the cell\'s faces grow, up to opaque, and whiten); the per-pixel veil after the points and the box', () => {
  const veilColor = 'vec3(0.0392, 0.0431, 0.0549)';
  const focus = { lit: FOCUS_LIT, grey: FOCUS_GREY, sourceGrey: FOCUS_SOURCE_GREY, dark: FOCUS_3D_DARK, sourceDark: FOCUS_3D_SOURCE_DARK, face: FOCUS_3D_FACE,
    white: FOCUS_3D_WHITE, hit: FOCUS_3D_HIT, veil: FOCUS_VEIL, sourceVeil: FOCUS_SOURCE_VEIL, veilColor };
  const plain = volumeFragment(COLORS), lit = volumeFragment(COLORS, focus);
  assert.equal(/uFocus|tierOf|hit1|credit/.test(plain), false, 'the plain shader knows nothing of it');
  // The focus variant is the plain one with lines added (none of the plain lines dropped), and the cell's colour by its
  // tier where the ray leaves the slab (the one colour the march takes without base).
  assert.equal(lit.split('tierColor(displayColor(previous),tier)').length, 2);
  const a = plain.split('\n'), b = lit.replace('tierColor(displayColor(previous),tier)', 'displayColor(previous)').split('\n'), added = [];
  let i = 0;
  for (const line of b) { if (i < a.length && line === a[i]) i++; else added.push(line); }
  assert.equal(i, a.length, 'every plain line, in order');
  // Exactly these lines, in this order: the uniforms, the tiers (none: the background or no focus; 1 the cell; 2 its rows;
  // 3 every other cell), a colour greyed and then darkened toward the veil's colour by tier, what each step gives the ray
  // credited to its tier (hit1, hit2); in the march the tier looked up once per change of id, a face leaving a cell for
  // the background credited to that cell's tier (tierWas) and greyed by it, every colour by its tier, the cell's faces
  // stronger and whiter, each step's credit (the last one where the ray stops in the slab, then each voxel's, then the
  // exit face where it leaves it), and the veil.
  const code = added.map(l => l.replace(/\/\/.*$/, '').trim()).filter(Boolean);
  assert.deepEqual(code, [
    'uniform bool uFocusOn;', 'uniform uint uFocusCell;', 'uniform uint uFocusLit[6];', 'uniform float uFocusDim;', 'float hit1 = 0., hit2 = 0.;',
    'int tierOf(uint id) {', 'if (!uFocusOn || id == 0u) return 0;', 'if (id == uFocusCell) return 1;', 'for (int k = 0; k < 6; k++) if (id == uFocusLit[k]) return 2;',
    'return 3;', '}',
    'vec3 tierColor(vec3 c, int tier) {', 'if (tier < 2) return c;',
    'c = mix(c, vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), (tier == 2 ? 0.35 : 0.6) * uFocusDim);',
    `return mix(c, ${veilColor}, (tier == 2 ? 0.25 : 0.45) * uFocusDim);`, '}',
    'void credit(int tier, inout float a0, float a1) {', 'if (tier == 1) hit1 += a1 - a0; else if (tier == 2) hit2 += a1 - a0;', 'a0 = a1;', '}',
    'int tierWas = tierOf(previous);',
    'int tier = id == previous ? tierWas : tierOf(id);', 'int gives = id != 0u ? tier : tierWas;', 'float given = acc.a;',
    'base = tierColor(base, tier);',
    'if (id==0u) face = tierColor(face, tierWas);',
    'if (gives == 1) { face = mix(face,vec3(1),0.35*uFocusDim); faceAlpha = min(1.,faceAlpha*mix(1.,2.0,uFocusDim)); }',
    'credit(gives, given, acc.a);',
    'credit(gives, given, acc.a); tierWas = tier;',
    'credit(tier, given, acc.a);',
    `if (uFocusOn && uCells) color = mix(color, ${veilColor}, (hit1 >= 0.02 ? 0. : hit2 >= 0.02 ? 0.4 : 0.62) * uFocusDim);`,
  ]);
  // Where they stand in the march: every colour by its tier right after the cell's own; a face's colour (the cell's
  // tier, or leaving into the background the last cell's) before its light; the cell's faces before they are added;
  // each credit after what its step added; the slab's exit face by the last cell's tier.
  const at = s => { const k = lit.indexOf(s); assert.ok(k >= 0, s); return k; };
  assert.ok(at('      vec3 base = displayColor(id);\n      base = tierColor(base, tier);\n'));
  assert.ok(at('vec3 face = id!=0u ? base : displayColor(previous);\n        if (id==0u) face = tierColor(face, tierWas);\n        float light'));
  assert.ok(at('if (gives == 1) {') < at('        add(acc,mix(fogColor,face,fog),faceAlpha);'));
  assert.ok(at('add(acc,mix(fogColor,base*.8,fog),(uMode==1 ? .26 : .12*min(1.,uOpacity))*clusterWeight(id!=0u?id:previous));\n        credit(gives, given, acc.a);\n        break;'));
  assert.ok(at('      credit(gives, given, acc.a); tierWas = tier;\n      previous=id;'));
  assert.ok(at('add(acc,mix(fogColor,tierColor(displayColor(previous),tier)*.8,fog),(uMode==1 ? .26 : .12*min(1.,uOpacity))*clusterWeight(id!=0u?id:previous));\n        credit(tier, given, acc.a);\n        break;'));
  // No added line draws more (no add(), no new alpha), but the cell's faces: stronger, capped at 1, and whiter.
  assert.equal(code.some(l => /\badd\(/.test(l)), false);
  assert.deepEqual(code.filter(l => /faceAlpha|alpha\b|\.a\s*[*+-]?=/.test(l)), [code[26]]);
  // The veil: after the box's lines and the points' composite, per pixel by what the ray took of the cell and its rows.
  const veil = at(code.at(-1));
  assert.ok(veil > at('color=mix(color,vec3(.42,.52,.65),line*uBox);') && veil > at('color=color*(1.-points.a)+points.rgb*tint;'));
  assert.ok(veil < at('outColor=vec4(color,1);'));
  // The stage's own compile asks for this variant with its constants and VEIL_COLOR.
  const source = readFileSync(new URL('../src/stage.js', import.meta.url), 'utf8');
  assert.match(source, /volumeFragment\(COLORS, \{ lit: FOCUS_LIT, grey: FOCUS_GREY, sourceGrey: FOCUS_SOURCE_GREY,\s+dark: FOCUS_3D_DARK, sourceDark: FOCUS_3D_SOURCE_DARK, face: FOCUS_3D_FACE, white: FOCUS_3D_WHITE, hit: FOCUS_3D_HIT, veil: FOCUS_VEIL,\s+sourceVeil: FOCUS_SOURCE_VEIL, veilColor: glslColor\(VEIL_COLOR\) \}\)/);
  assert.equal(VEIL_COLOR, '#0a0b0e');
});

/** One ray through the volume's focus, as the shader's lines above do it (the same statements, in JS), with the plain
 *  shader's alphas at opacity 1 in cells mode (a face .38 between two cells and .15 against the background, STEP for a
 *  voxel of a cell, .12 for the face where the ray leaves the slab): `ids` the voxels it crosses, entering from outside
 *  the slab and leaving it after the last; `cell` the window's cell, `lit` its lit cells, `dim` the focus's strength.
 *  {hit1, hit2, veil}: what the ray took of the cell and of its rows, and the veil its pixel gets. */
function rayFocus(ids, { cell, lit, dim = 1 }) {
  const STEP = 0.2, acc = { a: 0 }, add = alpha => { acc.a += (1 - acc.a) * alpha; };
  const tierOf = id => (id === 0 ? 0 : id === cell ? 1 : lit.includes(id) ? 2 : 3);
  let hit1 = 0, hit2 = 0, previous = 0, tierWas = tierOf(previous), given = 0;
  const credit = tier => { if (tier === 1) hit1 += acc.a - given; else if (tier === 2) hit2 += acc.a - given; given = acc.a; };
  for (const id of ids) {
    const tier = id === previous ? tierWas : tierOf(id), gives = id !== 0 ? tier : tierWas;
    given = acc.a;
    if (id !== previous && (id !== 0 || previous !== 0)) {
      let faceAlpha = id !== 0 && previous !== 0 ? 0.38 : 0.15;
      if (gives === 1) faceAlpha = Math.min(1, faceAlpha * (1 + (FOCUS_3D_FACE - 1) * dim));
      add(faceAlpha);
    }
    if (id !== 0) add(STEP);
    credit(gives); tierWas = tier; previous = id;
  }
  if (previous !== 0) add(0.12);   // the face where it leaves the slab
  credit(tierWas);
  return { hit1, hit2, veil: (hit1 >= FOCUS_3D_HIT ? 0 : hit2 >= FOCUS_3D_HIT ? FOCUS_SOURCE_VEIL : FOCUS_VEIL) * dim };
}

test('a ray through the volume: a pixel that shows any of the cell is never veiled, through a dimmed cell in front of it too; one that meets only its rows gets the rows\' veil, any other pixel the full one', () => {
  const focus = { cell: 7, lit: [7, 3] };
  // Another cell in front of the cell: the cell shows through it, so its pixel is not darkened (a dimmed cell does not
  // hide the cell, nor turn see-through: its colour is darkened, its alpha kept).
  const behind = rayFocus([9, 9, 7, 7], focus);
  assert.ok(behind.hit1 >= FOCUS_3D_HIT && behind.hit2 === 0); assert.equal(behind.veil, 0);
  // The cell's rows alone: their lighter veil; another cell alone, or the image between cells: the full veil.
  assert.equal(rayFocus([3, 3], focus).veil, FOCUS_SOURCE_VEIL);
  assert.equal(rayFocus([0, 9, 9, 0], focus).veil, FOCUS_VEIL);
  assert.equal(rayFocus([0, 0, 0], focus).veil, FOCUS_VEIL, 'the image between the cells, darker');
  // A row in front of the cell: the cell's part counts, its row's too.
  const both = rayFocus([3, 3, 7], focus);
  assert.ok(both.hit1 >= FOCUS_3D_HIT && both.hit2 >= FOCUS_3D_HIT); assert.equal(both.veil, 0);
  // The face where the ray leaves the cell for the background is the cell's (gives = tierWas): a ray that only grazes the
  // cell still shows it: here the cell's one voxel and both its faces count.
  const grazing = rayFocus([0, 7, 0], focus);
  assert.ok(Math.abs(grazing.hit1 - (1 - (1 - 0.15 * FOCUS_3D_FACE) * (1 - 0.2) * (1 - 0.15 * FOCUS_3D_FACE))) < 1e-12, `${grazing.hit1}`);
  // At half strength the veil is half as strong.
  assert.equal(rayFocus([9], { ...focus, dim: 0.5 }).veil, FOCUS_VEIL / 2);
  // Second is between: the rows' veil and darkening under every other cell's, over none for the cell.
  assert.ok(0 < FOCUS_SOURCE_VEIL && FOCUS_SOURCE_VEIL < FOCUS_VEIL && 0 < FOCUS_3D_SOURCE_DARK && FOCUS_3D_SOURCE_DARK < FOCUS_3D_DARK);
});

/** A Stage with its camera code and a real Orbit at `deg` degrees, turned by `yaw`, the slab `thickness` µm, on a stage of
 *  1600 x 1000 CSS px (ratio 1). */
function orbitStage({ deg = 55, yaw = 0, thickness = 15, stretch = 1 } = {}) {
  const stage = Object.create(Stage.prototype);
  Object.assign(stage, { canvas: { width: 1600, height: 1000 }, pixelRatio: 1, cssWidth: 1600, cssHeight: 1000, camera: { x: 0, y: 0, zoom: 1 },
    info: { z: { thickness_um: thickness } }, options: { stretch }, listeners: new Set(), raf: 1, _dirty: false, tween: null });
  stage.orbit = new Orbit(stage);
  Object.assign(stage.orbit, { enabled: true, angle: deg * Math.PI / 180, yaw, amount: 1 });
  return stage;
}

test('the camera\'s move in 3-D: the angle and the turn stay, every corner of the box (on the image and at the slab\'s top) lands inside the free part, never closer than the 2-D fit', () => {
  assert.equal(FIT_PASSES, 4);
  const size = { width: 1600, height: 1000 }, win = { left: 16, top: 16, w: 420, h: 700 };
  for (const yaw of [0, 0.6]) for (const [bw, bh] of [[60, 60], [120, 60], [60, 120]]) {
    const stage = orbitStage({ yaw }), box = [1000, 1000, 1000 + bw, 1000 + bh];
    const rect = whereView(box, size, win), free = whereFree(box, size, win), name = `${bw}x${bh} at yaw ${yaw}`;
    stage.fit(rect, true, 0, { box, free });
    const cam = stage.tween.to, zoom2d = Math.min(1600 / rect.w, 1000 / rect.h);
    assert.deepEqual([stage.orbit.angle, stage.orbit.yaw, stage.orbit.amount, stage.orbit.transition ?? null], [55 * Math.PI / 180, yaw, 1, null], `${name}: the view kept`);
    assert.ok(cam.zoom <= zoom2d + 1e-9 && cam.zoom > 0.1 * zoom2d, `${name}: ${cam.zoom} against ${zoom2d}`);
    for (const x of [box[0], box[2]]) for (const y of [box[1], box[3]]) for (const z of [0, 1]) {
      const p = stage.orbit.project(x, y, z, cam);
      assert.ok(p && p.x >= free[0] - 1e-6 && p.x <= free[2] + 1e-6 && p.y >= free[1] - 1e-6 && p.y <= free[3] + 1e-6, `${name}: corner ${x},${y},${z} at ${p?.x},${p?.y}`);
    }
    // The box's middle on the image is where the free part's middle looks (the turn of the camera, not of the view).
    const g = stage.orbit.ground((free[0] + free[2]) / 2, (free[1] + free[3]) / 2, cam);
    assert.ok(Math.abs(g[0] - (box[0] + box[2]) / 2) < 1e-6 && Math.abs(g[1] - (box[1] + box[3]) / 2) < 1e-6, `${name}: centred`);
  }
  // At once without animate (the stage's camera, no tween); without a view, or in 2-D, the fit of before: the rect's
  // middle at the 2-D zoom, the orbit sent back to its own angle in 3-D.
  const now = orbitStage(), box = [0, 0, 60, 60], rect = whereView(box, size, win);
  now.fit(rect, false, 0, { box, free: whereFree(box, size, win) });
  assert.equal(now.tween, null); assert.ok(now.camera.zoom <= Math.min(1600 / rect.w, 1000 / rect.h));
  const flat = orbitStage(); flat.orbit.enabled = false; flat.orbit.angle = 0; flat.orbit.amount = 0;
  flat.fit(rect, true, 0, { box, free: whereFree(box, size, win) });
  assert.deepEqual(flat.tween.to, { x: rect.x0 + rect.w / 2, y: rect.y0 + rect.h / 2, zoom: Math.min(1600 / rect.w, 1000 / rect.h) });
  const old = orbitStage({ deg: 30, yaw: 1 }), media = Object.getOwnPropertyDescriptor(globalThis, 'matchMedia');
  globalThis.matchMedia = () => ({ matches: true });   // reduced motion: the orbit's switch lands at once
  try {
    old.emit = () => {};
    old.fit(rect, true, 0);
    assert.deepEqual([old.orbit.angle, old.orbit.yaw], [55 * Math.PI / 180, 0], 'no view: the orbit goes back to its own angle, as before');
    assert.deepEqual(old.tween.to, { x: rect.x0 + rect.w / 2, y: rect.y0 + rect.h / 2, zoom: Math.min(1600 / rect.w, 1000 / rect.h) });
  } finally { if (media) Object.defineProperty(globalThis, 'matchMedia', media); else delete globalThis.matchMedia; }
});

/** The overlay's arrows on a fake stage (scale CSS px per µm) and a clock. */
function arrows({ scale = 2, reduced = false } = {}) {
  const t = { now: 1000 }, notes = [], replans = [];
  const stage = { scale, orbit: { active: false }, toScreen: (x, y) => [x * stage.scale, y * stage.scale], addAnimator(a) { this.animator = a; }, replan() { replans.push(t.now); } };
  const a = new DeeperArrows(stage, { now: () => t.now, reducedMotion: () => reduced, onNote: kind => notes.push(kind), redraw: () => {} });
  return { a, t, stage, notes, replans };
}
const focusOf = (sources, extra = {}) => ({ cell: 7, target: [100, 100], diameter: 10, hot: 0, sources, ...extra });
function canvas() {
  const calls = [];
  const g = new Proxy({ calls }, { get: (o, k) => k in o ? o[k] : (...args) => { calls.push([k, ...args]); return k === 'measureText' ? { width: 24 } : undefined; }, set: (o, k, v) => { calls.push(['=' + String(k), v]); return true; } });
  return g;
}

test('arrows: one per source of 1 % or more (at most 5, labels on the top 3), from the neighbour to the cell, 1.5-5 px by share', () => {
  const h = arrows();
  const sources = [0.46, 0.2, 0.1, 0.06, 0.03, 0.02, 0.005].map((share, k) => ({ label: k + 2, share, from: [40 + 10 * k, 60], name: `n${k}` }));
  assert.equal(h.a.set(focusOf(sources)), true);
  assert.equal(h.a.arrows.size, ARROWS, 'the 1 % share and up, at most five');
  assert.ok(!h.a.arrows.has(8), 'under 1 %: no arrow');
  h.t.now += 1000;   // the draw-on is over
  const g = canvas(); h.a.draw(g);
  assert.equal(g.calls.filter(c => c[0] === 'quadraticCurveTo').length, 2 * ARROWS, 'a halo and a magenta stroke each');
  const labels = g.calls.filter(c => c[0] === 'fillText').map(c => c[1]);
  assert.deepEqual(labels, ['46 %', '20 %', '10 %']); assert.equal(ARROW_LABELS, 3); assert.deepEqual(h.a.texts(), labels);
  for (const text of labels) assert.deepEqual(bannedWords(text), []);
  assert.equal(arrowWidth(0), 1.5); assert.equal(arrowWidth(0.4), 5); assert.equal(arrowWidth(1), 5);
  // Hovering an arrow names its source; elsewhere nothing.
  const c = arrowCurve([40 * 2, 60 * 2], [100 * 2, 100 * 2]), mid = c.at(0.5);
  assert.equal(h.a.hit(mid[0], mid[1]), 2); assert.equal(h.a.hit(5, 5), 0);
  // The same focus again: nothing to redraw.
  assert.equal(h.a.set(focusOf(sources)), false);
  // A label is the window row's own percent (rounded down, never above the headline),
  // never the share rounded up ("34 %" under a 33 % headline).
  const k = arrows();
  k.a.set(focusOf([{ label: 2, share: 0.3399, pct: 33, from: [40, 60], name: 'n0' }, { label: 3, share: 0.1999, from: [60, 60], name: 'n1' }]));
  k.t.now += 1000;
  const g2 = canvas(); k.a.draw(g2);
  assert.deepEqual(g2.calls.filter(c => c[0] === 'fillText').map(c => c[1]), ['33 %', '19 %']);
});

test('arrows draw on only the first time a source passes 5 %, on the 150 ms steps (at most 4 frames), then nothing is due', () => {
  const h = arrows(), src = share => [{ label: 3, share, from: [40, 60], name: 'n' }];
  h.a.set(focusOf(src(0.3)), { motion: true });
  const due = [];
  for (let n = 0; n < 10; n++) { const d = h.a.due(h.t.now); if (d == null) break; due.push(d - 1000); h.t.now = d; h.a.frame(d); }
  assert.deepEqual(due, [150, 300, 450], 'the set\'s own frame, then three more');
  assert.equal(h.a.due(h.t.now), null);
  assert.equal(arrowProgress(0, ARROW_DRAW_MS), 0.25); assert.equal(arrowProgress(ARROW_DRAW_MS, ARROW_DRAW_MS), 1); assert.equal(ARROW_STEP_MS, 150);
  // Dropping under 5 % and passing it again: no second draw-on (and no pulse).
  h.t.now += 2000; h.a.set(focusOf(src(0.03)), { motion: true }); h.t.now += 2000; h.a.set(focusOf(src(0.3)), { motion: true });
  assert.equal(h.a.due(h.t.now), null);
  // A source that leaves fades (retracts) in the same steps.
  h.a.set(focusOf([]), { motion: true });
  assert.equal(h.a.dying.length, 1); assert.equal(h.a.due(h.t.now), h.t.now + 150);
  // Another cell: its arrows draw on.
  h.t.now += 1000; h.a.set(focusOf(src(0.02), { cell: 9 }));
  assert.ok(h.a.due(h.t.now) != null);
  // Reduced motion (or calm motion): everything lands at once.
  const r = arrows({ reduced: true }); r.a.set(focusOf(src(0.3)), { motion: true });
  assert.equal(r.a.due(r.t.now), null);
});

test('arrows only in 2-D with cells at least 6 px across; the window is told to say "zoom in" or "2-D view"', () => {
  const h = arrows({ scale: 0.5 });
  h.a.set(focusOf([{ label: 3, share: 0.3, from: [40, 60] }]));
  assert.equal(h.a.drawable(), false, '10 µm at 0.5 px/µm is 5 px');
  const g = canvas(); h.a.draw(g);
  assert.equal(g.calls.filter(c => c[0] === 'stroke').length, 0); assert.deepEqual(h.notes, ['zoom']);
  assert.equal(ARROW_MIN_PX, 6);
  h.stage.scale = 1; h.a.draw(canvas()); assert.deepEqual(h.notes, ['zoom', '']);
  h.stage.orbit.active = true; h.a.sync(); assert.deepEqual(h.notes, ['zoom', '', '3d']);
  h.stage.orbit.active = false; assert.equal(h.a.clear(), true); h.a.sync(); assert.deepEqual(h.notes, ['zoom', '', '3d', '']);
  // A source whose centre is not known yet: no arrow, and main.js knows to ask again when the statistics come.
  const m = arrows(); m.a.set(focusOf([{ label: 3, share: 0.3, from: null }])); assert.equal(m.a.arrows.size, 0); assert.equal(m.a.missing, true);
});

test('the focus asks nothing of the overlay (no area to shade since 2026-10-22): no frame, its marks at full strength; the arrows as they are', () => {
  const stage = { scale: 1, cssWidth: 800, cssHeight: 600, orbit: { active: false }, options: { cellColorBy: 'identity' }, focusLevel: 0,
    toScreen: (x, y) => [x, y], toWorld: (x, y) => [x, y], addAnimator() {}, removeAnimator() {}, replan() {} };
  const t = { now: 0 }, redraws = [];
  const o = new QcOverlay(stage, { now: () => t.now, redraw: () => redraws.push(t.now), reducedMotion: () => true });
  const drawn = () => {
    const calls = [], style = { globalAlpha: 1 };
    const g = new Proxy(style, { get: (s, k) => k in s ? s[k] : (...args) => { calls.push({ op: k, alpha: s.globalAlpha }); },
      set: (s, k, v) => { s[k] = v; return true; } });
    o.draw(g);
    return calls;
  };
  assert.equal(o.due(0), null, 'nothing to draw');
  stage.focusLevel = 1; assert.equal(o.due(0), null, 'the focus on: still nothing'); stage.focusLevel = 0;
  // Go deeper's starting outline (a mark): one refresh when it comes, then idle, whatever the focus does.
  stage.startOutline = { points: Float32Array.of(100, 100, 200, 100, 200, 200), closed: true, count: 3 };
  t.now = 10; assert.equal(o.due(10), 10); assert.equal(o.frame(10), false, 'a refresh, not motion'); assert.deepEqual(redraws, [10]);
  stage.focusLevel = 1; t.now = 20;
  assert.equal(o.due(20), null, 'the focus came on: no frame for it');
  const calls = drawn();
  assert.deepEqual([...new Set(calls.filter(c => c.op === 'stroke').map(c => c.alpha))], [1], 'the mark at full strength');
  assert.equal(calls.some(c => c.op === 'fillText'), false, 'no chip');
  stage.focusLevel = 0.5; assert.equal(o.due(30), null); stage.focusLevel = 0; assert.equal(o.due(40), null);
  // The arrows never read it.
  const h = arrows(); h.stage.focusLevel = 1;
  h.a.set(focusOf([{ label: 3, share: 0.3, from: [40, 60] }])); h.t.now += 1000;
  const g = canvas(); h.a.draw(g);
  assert.deepEqual([...new Set(g.calls.filter(c => c[0] === '=globalAlpha').map(c => c[1]))], [1]);
});

test('the UMAP rings the window\'s cell and its sources; with none it draws as before', () => {
  const log = [], view = Object.create(UmapView.prototype);
  Object.assign(view, { gl: recordingGL(log), uniforms: { Mix: 'mix', Distance: 'd', Aspect: 'as', Ratio: 'r', Selected: 'sel', Hover: 'hov', Center: 'c', Right: 'ri', Up: 'up', Back: 'b', Rings: 'rings', RingKinds: 'kinds', RingCount: 'n' },
    program: {}, vao: {}, count: 10, center: [0, 0, 0], distance: 10, width: 100, height: 80, ratio: 1, yaw: 0, pitch: 0, selected: null, hover: null, drawCount: 0, canvas: { width: 100, height: 80 },
    wake() {}, dirty: false });
  view.render(1);
  assert.equal(uniforms(log).n, 0); assert.equal(uniforms(log).rings, undefined);
  assert.equal(view.ringCells([[6, 1], [2, 2], [3, 2], [99, 4], ['x', 2]]), true); assert.equal(view.dirty, true);
  assert.equal(view.ringCells([[6, 1], [2, 2], [3, 2]]), false, 'the same rings: no redraw');
  log.length = 0; view.render(1);
  const u = uniforms(log);
  assert.equal(u.n, 3); assert.deepEqual([...u.rings], [6, 2, 3, -1, -1, -1]); assert.deepEqual([...u.kinds], [1, 2, 2, 0, 0, 0]);
  assert.equal(view.ringCells(null), true); log.length = 0; view.render(1); assert.equal(uniforms(log).n, 0);
  const source = readFileSync(new URL('../src/umap.js', import.meta.url), 'utf8');
  assert.match(source, /for \(int k = 0; k < 6; k\+\+\) \{ if \(k >= uRingCount\) break;/, 'no rings: the loop never runs');
  assert.match(source, /if\(r>1\. \|\| vColor\.a<\.005\) discard;/, 'an invisible point gets no ring');
});

test('api.js: the window\'s endpoints', async () => {
  const requests = [];
  const context = vm.createContext({ window: { fetch() {}, dispatchEvent() {} },
    fetch: async (url, options) => { requests.push({ url, ...options }); return options?.method === 'DELETE' ? { ok: true, status: 204 } : { ok: true, json: async () => ({ state: 'working' }) }; } });
  const api = vm.runInContext(readFileSync(new URL('../src/api.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '').replaceAll('export ', '') + '\napi', context);
  await api.qcDeeper('run a', { seq: 12, cell: 3, follow: true });
  await api.qcDeeper('run a', { seq: 'start', cell: 3 });
  assert.equal(await api.qcDeeperStop('run a'), null);
  // (No GET of a single cached answer: the window never fetches one.)
  assert.equal(api.qcDeeperResult, undefined);
  await api.qcDeeperHistory('run a', 3); await api.learnSampleDetail('s 1');
  assert.deepEqual(requests.map(r => [r.method, r.url]), [
    ['POST', '/api/v1/runs/run%20a/qc/deeper'], ['POST', '/api/v1/runs/run%20a/qc/deeper'], ['DELETE', '/api/v1/runs/run%20a/qc/deeper'],
    ['GET', '/api/v1/runs/run%20a/qc/deeper/history?cell=3&points=200'], ['POST', '/api/v1/samples/s%201/qc']]);
  assert.deepEqual(JSON.parse(requests[0].body), { seq: 12, cell: 3, follow: true }); assert.deepEqual(JSON.parse(requests[1].body), { seq: 'start', cell: 3 });
  assert.deepEqual(JSON.parse(requests[4].body), { detail: true });
  // Quiet like every /qc request: a refusal never raises the page's general notice (the window says it in its own words).
  const quiet = /\/qc(?:[/?]|$)/;
  for (const r of requests) assert.match(r.url, quiet);
});
