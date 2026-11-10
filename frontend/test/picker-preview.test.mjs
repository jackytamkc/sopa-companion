import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { clampBox } from '../src/region.mjs';
import { domFixture, deferred } from './dom-fixture.mjs';

// main.js previewPrior (the picker's prior outlines), evaluated alone against a fake stage, engine and PriorLayers.
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const slice = source.slice(source.indexOf('let previewTimer'), source.indexOf('function updateScale()'));

function harness() {
  const dom = domFixture(); dom.document.body.innerHTML = html;
  const timers = new Map(), jobs = [], assigned = [];
  let timer = 0;
  const stage = { cssWidth: 1000, cssHeight: 800, scale: 1, cx: 5000, cy: 5000, dirty: false,
    toWorld(x, y) { return [this.cx + (x - this.cssWidth / 2) / this.scale, this.cy + (y - this.cssHeight / 2) / this.scale]; } };
  // PriorLayers.vectors as main.js sees it: every assignment is logged (a clear is an assignment of null).
  const priors = { get vectors() { return this.value ?? null; }, set vectors(v) { assigned.push(v); this.value = v; } };
  const state = { mode: 'pick', sample: { id: 's', extent: [0, 0, 10000, 10000] },
    prior: { cells: { kind: 'shapes', key: 'cells' }, nuclei: { kind: 'shapes', key: 'nuclei' } } };
  const api = { shapes: (id, key, box, signal) => { const job = deferred(); jobs.push({ id, key, box, signal, ...job }); return job.promise; } };
  const context = vm.createContext({ $: dom.$, state, stage, priors, api, clampBox, AbortController, JSON, Math, Promise, Set,
    setTimeout: fn => { timers.set(++timer, fn); return timer; }, clearTimeout: id => timers.delete(id) });
  vm.runInContext(slice, context);
  return { ...dom, state, stage, priors, jobs, assigned, run: code => vm.runInContext(code, context),
    /** Fire the pending debounce timer and answer its requests with `polygons` per key. */
    async fetch(polygons = key => [{ id: key, poly: [[0, 0], [1, 1]] }]) {
      const fired = [...timers]; timers.clear(); const start = jobs.length;
      const done = fired.map(([, fn]) => fn());
      for (const job of jobs.slice(start)) job.resolve({ polygons: polygons(job.key), truncated: false });
      await Promise.all(done);
      return jobs.slice(start);
    },
    note: () => dom.$('prior-preview-note').textContent };
}

test('the picker strokes only the layers chosen as shapes priors', async () => {
  const h = harness();
  h.state.prior = { cells: { kind: 'shapes', key: 'cells' }, nuclei: { kind: 'column', column: 'overlaps_nucleus' } };
  h.run('previewPrior()');
  assert.match(h.note(), /Loading prior outlines/);
  const jobs = await h.fetch();
  assert.deepEqual(jobs.map(j => j.key), ['cells'], 'only the shapes prior is fetched');
  assert.equal(h.priors.vectors.cells[0].id, 'cells');
  assert.equal(h.priors.vectors.nuclei, null, 'the other kind is not drawn at all');
  assert.doesNotMatch(h.note(), /Loading/);
});

test('a pan or zoom keeps the outlines on screen until the new box\'s arrive; another prior or hiding clears them', async () => {
  const h = harness();
  h.run('previewPrior()');
  await h.fetch();
  const first = h.priors.vectors;
  assert.deepEqual([first.cells[0].id, first.nuclei[0].id], ['cells', 'nuclei']);
  const clears = () => h.assigned.filter(v => v === null).length;
  // A pan: a new box, the same prior.
  h.stage.cx += 400; h.run('previewPrior()');
  assert.equal(h.priors.vectors, first, 'the old outlines stay (PriorLayers moves them with the camera)');
  assert.equal(clears(), 0);
  assert.match(h.note(), /Loading prior outlines/, 'the harness and the user still see that the new ones are coming');
  const second = await h.fetch(key => [{ id: `${key}-2`, poly: [[0, 0], [2, 2]] }]);
  assert.ok(second[0].box[0] > 400, 'the new box was fetched');
  assert.equal(h.priors.vectors.cells[0].id, 'cells-2');
  // A zoom in: also just a new box.
  h.stage.scale = 2; h.run('previewPrior()');
  assert.equal(h.priors.vectors.cells[0].id, 'cells-2'); assert.equal(clears(), 0);
  await h.fetch();
  // Another prior: the outlines on screen are not this prior's.
  h.state.prior = { cells: { kind: 'shapes', key: 'other' }, nuclei: null };
  h.run('previewPrior()');
  assert.equal(h.priors.vectors, null, 'cleared at once'); assert.equal(clears(), 1);
  await h.fetch();
  assert.equal(h.priors.vectors.cells[0].id, 'other');
  h.$('show-prior').checked = false; h.run('previewPrior()');
  assert.equal(h.priors.vectors, null);
  h.$('show-prior').checked = true; h.stage.scale = .1; h.run('previewPrior()');
  assert.equal(h.priors.vectors, null); assert.match(h.note(), /Zoom in/);
  // Shown again after hiding: nothing stale is kept for the same prior.
  h.stage.scale = 1; h.run('previewPrior()');
  assert.equal(h.priors.vectors, null);
});

test('an answer for a box the camera has left is dropped; the newest box\'s is shown', async () => {
  const h = harness();
  h.run('previewPrior()');
  await h.fetch(key => [{ id: `${key}-0`, poly: [] }]);
  // The next box's requests are in flight when the camera moves again.
  h.stage.cx += 300; h.run('previewPrior()');
  const pending = h.fetch(() => { throw new Error('answered below'); }).catch(() => {});
  const stale = h.jobs.slice(-2);
  h.stage.cx += 300; h.run('previewPrior()');
  assert.ok(stale.every(j => j.signal.aborted), 'the left box\'s requests are aborted');
  const fresh = await h.fetch(key => [{ id: `${key}-new`, poly: [] }]);
  for (const j of stale) j.resolve({ polygons: [{ id: 'stale', poly: [] }] });
  await pending; await new Promise(resolve => setImmediate(resolve));   // the stale answer has been looked at
  assert.equal(fresh.length, 2);
  assert.equal(h.priors.vectors.cells[0].id, 'cells-new');
  assert.equal(h.assigned.filter(v => v === null).length, 0, 'never blank in between');
});

test('while a new box loads the old outlines are held (PriorLayers moves them, never draws them again), until the box arrives', async () => {
  const h = harness();
  h.run('previewPrior()');
  assert.equal(h.priors.hold, false, 'nothing on screen to hold');
  await h.fetch();
  assert.equal(h.priors.hold, false);
  h.stage.cx += 400; h.run('previewPrior()');
  assert.equal(h.priors.hold, true, 'the old box is kept on screen: held while the new one loads');
  await h.fetch(key => [{ id: `${key}-2`, poly: [[0, 0], [2, 2]] }]);
  assert.equal(h.priors.hold, false, 'released with the new box (PriorLayers draws it at once: new vectors)');
  // A box superseded before it arrives: still held for the newest one, released when that one comes.
  h.stage.cx += 300; h.run('previewPrior()');
  h.stage.cx += 300; h.run('previewPrior()');
  assert.equal(h.priors.hold, true);
  await h.fetch();
  assert.equal(h.priors.hold, false);
  // Another prior clears the outlines: nothing is held.
  h.state.prior = { cells: { kind: 'shapes', key: 'other' }, nuclei: null };
  h.run('previewPrior()');
  assert.equal(h.priors.vectors, null); assert.equal(h.priors.hold, false);
  await h.fetch();
  // Hiding them (or leaving the picker: resetPreview) releases the hold too.
  h.stage.cx += 400; h.run('previewPrior()');
  assert.equal(h.priors.hold, true);
  h.$('show-prior').checked = false; h.run('previewPrior()');
  assert.equal(h.priors.hold, false);
});
