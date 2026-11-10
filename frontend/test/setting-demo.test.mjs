import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import {
  DEMO_KEYS, DEMO_TEXT, demoTexts, bannedWords, decode, planStep, gridAt, plan, stepBlock, pinkOf, stepDurations, rasterize,
  placePopover, attachSettingDemo, OPEN_DELAY_MS, CLOSE_DELAY_MS,
} from '../src/setting-demo.mjs';
import { TEST_FLAGS } from '../src/qc-panel.js';

// The "?" beside the test corner's Setting choice: its demos' data (frontend/demos/<key>.json, made from proseg-live's
// own runs of a made-up tissue), the player's pure parts (decoding, how one frame's outlines turn into the next, the
// pink, the steps' lengths), its words, where the popover goes, and the "?" itself in a small DOM.
const DEMOS = new URL('../demos/', import.meta.url);
const load = key => JSON.parse(readFileSync(new URL(`${key}.json`, DEMOS), 'utf8'));
const SPECS = Object.values(DEMO_KEYS).map(load);

test('every demo is of a setting the test takes, and every demo file is one of them', () => {
  assert.ok(Object.keys(DEMO_KEYS).every(flag => TEST_FLAGS.includes(flag)), 'every demo is of a setting the test takes');
  assert.deepEqual(readdirSync(DEMOS).filter(f => f.endsWith('.json')).map(f => f.slice(0, -5)).sort(), Object.values(DEMO_KEYS).sort());
  for (const [flag, key] of Object.entries(DEMO_KEYS)) {
    const s = load(key);
    assert.equal(s.key, key); assert.equal(s.flag, flag);
    assert.ok(s.values.includes(s.default), `${key}: proseg's default is one of its panels`);
  }
});

test('each demo decodes: a state per dot, a grid and a number per value per frame, frames on proseg\'s schedule', () => {
  for (const s of SPECS) {
    const dec = decode(s), K = s.iters.length;
    assert.equal(dec.values.length, s.values.length, s.key);
    assert.equal(s.iters[0], 0); assert.equal(s.iters[K - 1], 450, `${s.key} ends at proseg's last iteration`);
    for (const val of dec.values) {
      assert.equal(val.labels.length, K); assert.equal(val.grids.length, K);
      for (let k = 0; k < K; k++) {
        assert.equal(val.labels[k].length, dec.n);
        assert.ok(val.labels[k].every(v => v <= 8), `${s.key}: a dot's state is a cell (1-8) or none (0)`);
        assert.ok(val.grids[k].some(v => v === 1), `${s.key} frame ${k}: this cell has an outline`);
        const stat = val.stat[k];
        assert.ok(Number.isFinite(stat) || (Array.isArray(stat) && stat.length === 2 && stat.every(Number.isFinite)), `${s.key} frame ${k}: the panel's number(s)`);
      }
    }
  }
});

test('a step turns one frame\'s outlines into the next exactly, a whole grid square at a time while proseg settles', () => {
  for (const s of SPECS) {
    const dec = decode(s), K = s.iters.length;
    for (let vi = 0; vi < dec.values.length; vi++) {
      const val = dec.values[vi], out = new Uint8Array(dec.W * dec.H);
      for (let k = 0; k + 1 < K; k++) {
        const pl = plan(dec, vi, k);
        assert.deepEqual(gridAt(val.grids[k], pl, -1, out), val.grids[k], `${s.key} step ${k} starts on its frame`);
        assert.deepEqual(gridAt(val.grids[k], pl, 1, out), val.grids[k + 1], `${s.key} step ${k} ends on the next`);
        const B = stepBlock(dec, k, vi);
        if (B < 2) continue;
        // every B µm square (proseg's grid while it settles: 2 µm, or the starting grid's) changes all at once
        for (let i = 0; i < pl.count; i++) {
          const v = pl.idx[i], x = v % dec.W, y = Math.floor(v / dec.W), x0 = x - (x % B), y0 = y - (y % B);
          for (let dy = 0; dy < B; dy++) for (let dx = 0; dx < B; dx++) {
            const u = (y0 + dy) * dec.W + x0 + dx;
            if (x0 + dx < dec.W && y0 + dy < dec.H && pl.when[u] >= 0) assert.equal(pl.when[u], pl.when[v], `${s.key} step ${k}: one square, one moment`);
          }
        }
      }
    }
  }
});

test('planStep on a made-up pair: the front starts next to the new cell', () => {
  const W = 4, H = 1, g0 = Uint8Array.from([1, 2, 2, 2]), g1 = Uint8Array.from([1, 1, 1, 2]);
  const pl = planStep(g0, g1, W, H, 7, 1);
  assert.equal(pl.count, 2);
  assert.ok(pl.when[1] < pl.when[2], 'the square beside cell 1 changes first');
  assert.deepEqual([...rasterize(['M0 0h2v1h-2Z'], 3, 1)], [1, 1, 0]);
});

test('pink: a neighbour\'s RNA this cell\'s count keeps, as strong as its keep (at least 0.2), never this cell\'s own', () => {
  for (const s of SPECS) {
    const dec = decode(s), K = s.iters.length;
    for (let vi = 0; vi < dec.values.length; vi++) for (const k of [0, K - 1]) {
      const listed = !!dec.values[vi].pinkP;
      for (let d = 0; d < dec.n; d++) {
        const p = pinkOf(dec, vi, k, d);
        if (listed) {
          // the demo lists its pink dots (Go deeper's glow, P = w x pi >= 0.2): only a neighbour's RNA, never this cell's own
          assert.ok(p === 0 || (p >= 0.2 && p <= 1), `${s.key}`);
          if (p > 0) assert.ok(dec.origin[d] >= 2, `${s.key}: a listed pink dot is a neighbour's RNA`);
        } else if (dec.origin[d] < 2 || dec.values[vi].labels[k][d] !== 1) assert.equal(p, 0);
        else assert.ok(p >= 0.2 && p <= 1);
      }
    }
  }
});

test('the steps play between their least and most length', () => {
  for (const s of SPECS) {
    const durs = stepDurations(decode(s), 200, 440, 150, 1.6);
    assert.equal(durs.length, s.iters.length - 1);
    assert.ok(durs.every(d => d >= 200 && d <= 440));
  }
});

test('the words: plain, none the mixing check keeps off the screen, none naming a sample, a cell or a gene of the user\'s', () => {
  const words = demoTexts(SPECS);
  assert.ok(words.length > 20 && words.every(w => typeof w === 'string' && w.trim()));
  for (const w of words) {
    assert.deepEqual(bannedWords(w), [], w);
    assert.doesNotMatch(w, /\bDU\d|cellpose-\d|\bLYZ\b|xenium_|\/home\//i, w);
  }
  for (const s of SPECS) assert.ok(s.info.split(/\s+/).length <= 75, `${s.key}: its info stays within a tooltip's 75 words`);
  assert.equal(DEMO_TEXT.button, '?');
});

test('the popover goes beside the corner at the "?"\'s height, else above or below the "?", always on screen', () => {
  const button = { left: 120, right: 140, top: 500, bottom: 520 }, corner = { left: 12, right: 372, top: 400, bottom: 700 };
  let p = placePopover(button, corner, 422, 215, 1600, 900);
  assert.equal(p.side, 'right'); assert.equal(p.left, 380); assert.equal(p.top, 510 - 107.5);
  p = placePopover({ ...button, left: 1300, right: 1320 }, { ...corner, left: 1180, right: 1540 }, 422, 215, 1600, 900);
  assert.equal(p.side, 'left'); assert.equal(p.left, 1180 - 8 - 422);
  p = placePopover(button, { ...corner, left: 0, right: 700 }, 422, 215, 760, 900);
  assert.equal(p.side, 'above'); assert.equal(p.top, 500 - 8 - 215);
  p = placePopover({ ...button, top: 100, bottom: 120 }, { ...corner, left: 0, right: 700 }, 422, 215, 760, 900);
  assert.equal(p.side, 'below'); assert.equal(p.top, 128);
  p = placePopover({ ...button, top: 880, bottom: 896 }, corner, 422, 215, 1600, 900);
  assert.ok(p.top + 215 <= 900 - 8, 'kept inside the window');
});

function fakeButton() {
  const listeners = new Map(), attrs = new Map();
  return {
    ownerDocument: { defaultView: null, documentElement: { dataset: {}, clientWidth: 1600 }, addEventListener() {}, removeEventListener() {} },
    dataset: {}, hidden: false, title: '', textContent: '', type: '',
    setAttribute: (k, v) => attrs.set(k, String(v)), getAttribute: k => attrs.get(k) ?? null, removeAttribute: k => attrs.delete(k),
    addEventListener: (t, fn) => listeners.set(t, fn), removeEventListener: t => listeners.delete(t), listeners, attrs,
  };
}

test('the "?": named, hidden for a setting without a demo, and a demo that fails to load says so and is asked again', async () => {
  const button = fakeButton(); let loads = 0;
  const demo = attachSettingDemo(button, { flag: 'prior-weight', load: () => { loads++; return Promise.reject(new Error('gone')); } });
  assert.equal(button.textContent, '?'); assert.equal(button.attrs.get('aria-label'), DEMO_TEXT.buttonLabel); assert.equal(button.hidden, false);
  for (const t of ['pointerenter', 'pointerleave', 'focus', 'blur', 'click']) assert.ok(button.listeners.has(t), t);
  assert.equal(await demo.open(), null);
  assert.equal(button.dataset.sdemo, 'failed'); assert.match(button.title, /gone/);
  await demo.open(); assert.equal(loads, 2, 'a failed load is not kept');
  demo.setFlag('no-such-setting'); assert.equal(button.hidden, true);
  assert.equal(await demo.open(), null); assert.equal(loads, 2, 'nothing loads for a setting without a demo');
  demo.setFlag('cell-compactness'); assert.equal(button.hidden, false);
  demo.dispose(); assert.equal(button.listeners.size, 0);
  assert.ok(OPEN_DELAY_MS < CLOSE_DELAY_MS);
});

test('nothing in the popover\'s stylesheet moves by itself; the figure theme has its own colours', () => {
  const css = readFileSync(new URL('../src/setting-demo.css', import.meta.url), 'utf8');
  assert.doesNotMatch(css, /animation|transition|@keyframes/);
  assert.match(css, /\[data-theme=figure\] \.sdemo-pop/);
  const page = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  assert.match(page, /<link rel="stylesheet" href="\.\/src\/setting-demo\.css">/);
});
