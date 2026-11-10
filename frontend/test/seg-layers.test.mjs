import test from 'node:test';
import assert from 'node:assert/strict';
import { SegLayers, SEG_TEXT, SEG_COLORS, SEG_STYLE, MAX_SHOWN, KEEP_SAMPLES, STORAGE_KEY, savedStyle } from '../src/seg-layers.mjs';
import { layerPath } from '../src/prior-paths.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { encodeVectors } from './vectors-fixture.mjs';
import { COMPARE_TEXT } from '../src/compare-copy.mjs';

// The sample's segmentation layers: the list a sample gives, each
// layer's outlines fetched when it is first shown and never while hidden, at most four shown, styles remembered per
// sample and key. Against a stand-in engine: GET /samples/{id}/segmentations and .../{key}/outlines.

const settle = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };
const square = (x, y, s = 10) => [[x, y], [x + s, y], [x + s, y + s], [x, y + s], [x, y]];
/** The outlines binary of `n` cells of an element (its cells only, as the engine serves it). */
const outlines = (n, key) => encodeVectors({ cells: Array.from({ length: n }, (_, k) => ({ id: `${key}-${k}`, poly: square(20 * k, 0) })), nuclei: null }, { source: { cells: key } });
const ITEMS = [
  { key: 'xenium_cell_boundaries', name: 'Xenium cells', source: 'xenium', model: null, count: 44691, extent: [8, 531, 10788, 10016], table: 'xenium_table' },
  { key: 'cell_boundaries', name: 'earlier segmentation (imported)', source: 'imported', model: null, count: 42679, extent: [8, 531, 10788, 10016], table: 'table' },
  { key: 'xenium_nucleus_boundaries', name: 'xenium_nucleus_boundaries', source: 'other', model: null, count: 44327, extent: [8, 531, 10788, 10016], table: null },
  { key: 'cellpose_test', name: 'Cellpose', source: 'cellpose', model: 'cp4_20261021_154153', count: 5378, extent: [8.1, 530.6, 2081.6, 1934.8], table: null },
  { key: 'proseg_boundaries', name: 'proseg (this app)', source: 'proseg', model: null, count: 41000, extent: [0, 0, 100, 100], table: 'proseg_table' },
];

/** A stand-in engine: every request logged ({url, signal}); `list` the sample's list (or a status), `fail` keys whose
 *  outlines answer 500, `hold` keys whose outlines wait for release(key). */
function engine({ list = ITEMS, fail = new Set(), hold = new Set() } = {}) {
  const asked = [], waiting = new Map();
  const fetch = async (url, options) => {
    asked.push({ url, signal: options?.signal ?? null });
    const m = /^\/api\/v1\/samples\/([^/]+)\/segmentations(?:\/([^/]+)\/outlines)?$/.exec(url);
    if (!m) return new Response('{}', { status: 404 });
    if (!m[2]) return typeof list === 'number' ? new Response('{}', { status: list }) : Response.json({ segmentations: list });
    const key = decodeURIComponent(m[2]);
    if (fail.has(key)) return new Response('busy', { status: 500 });
    if (hold.has(key)) await new Promise((resolve, reject) => {
      waiting.set(key, resolve);
      options?.signal?.addEventListener('abort', () => reject(Object.assign(new Error('aborted'), { name: 'AbortError' })));
    });
    return new Response(outlines(3 + key.length % 4, key));
  };
  return { fetch, asked, fail, hold, release: key => waiting.get(key)?.(), lists: () => asked.filter(a => a.url.endsWith('/segmentations')).length,
    fetched: () => asked.filter(a => a.url.endsWith('/outlines')).map(a => decodeURIComponent(a.url.split('/').at(-2))) };
}
/** localStorage as the page has it (`broken`: every call throws). */
function storage(initial = {}, broken = false) {
  const map = new Map(Object.entries(initial));
  return { map, getItem: k => { if (broken) throw Error('denied'); return map.get(k) ?? null; }, setItem: (k, v) => { if (broken) throw Error('quota'); map.set(k, String(v)); } };
}
const remembered = memory => ({ [STORAGE_KEY]: JSON.stringify(memory) });
const stored = s => JSON.parse(s.map.get(STORAGE_KEY));

test('a sample\'s remembered layers come back in order with their styles; only the shown ones are fetched, once', async () => {
  const e = engine(), s = storage(remembered({ samples: { z1: { at: 5, layers: ['cellpose_test', 'xenium_cell_boundaries', 'cell_boundaries'], styles: {
    cellpose_test: { on: true, color: '#123ABC', opacity: .5, width: 2, fill: true, name: 'Cellpose' },
    xenium_cell_boundaries: { on: false, color: '#ffb454', opacity: .7, width: 1, fill: false },
    cell_boundaries: { on: true, opacity: 7, width: 99, color: 'red', fill: 'yes' } } } } }));
  const store = new SegLayers({ fetch: e.fetch, storage: s }), told = [];
  store.listen(drawn => told.push(drawn));
  assert.equal(store.setSample('z1'), true);
  assert.deepEqual(store.layers.map(l => l.key), ['cellpose_test', 'xenium_cell_boundaries', 'cell_boundaries']);
  assert.deepEqual(store.layers[0].style, { on: true, color: '#123abc', opacity: .5, width: 2, fill: true });
  assert.equal(store.layers[0].name, 'Cellpose', 'the remembered name until the list is read');
  assert.deepEqual(store.layers[2].style, { ...SEG_STYLE, color: SEG_COLORS[0], opacity: 1, width: 5 }, 'out of range: clamped; not a colour or a boolean: the default (the first colour free)');
  await settle();
  assert.deepEqual(e.fetched().sort(), ['cell_boundaries', 'cellpose_test'], 'the hidden one is never fetched');
  assert.equal(e.lists(), 1, 'the list once: the layers\' names, and whether they are still in the sample');
  assert.deepEqual(store.layers.map(l => [l.state, l.name]), [['ready', 'Cellpose'], ['idle', 'Xenium cells'], ['ready', 'earlier segmentation (imported)']]);
  assert.deepEqual(store.drawable().map(l => l.key), ['cellpose_test', 'cell_boundaries'], 'front to back, as listed');
  assert.equal(store.layers[0].set.count, 3 + 'cellpose_test'.length % 4);
  assert.ok(told.includes(true) && told.at(0) === false, 'reading the list draws nothing; a loaded layer does');
  assert.equal(store.setSample('z1'), false, 'the same sample: nothing to do');
  await settle();
  assert.equal(e.asked.length, 3);
});

test('a name an earlier version remembered ("Cellpose · 1/4") says its level by name until the sample\'s list names it', async () => {
  const memory = remembered({ samples: { z1: { at: 5, layers: ['cellpose_test', 'instanseg_half', 'cell_boundaries'], styles: {
    cellpose_test: { on: false, name: 'Cellpose · 1/4' }, instanseg_half: { on: false, name: 'InstanSeg · 1/2' },
    cell_boundaries: { on: false, name: 'earlier segmentation (imported)' } } } } });
  // InstanSeg is dormant: the engine names its saves "saved stain segmentation" now, and so does a name remembered before.
  const names = store => store.layers.map(l => l.name), before = ['Cellpose · scale2', 'saved stain segmentation · scale1', 'earlier segmentation (imported)'];
  const offline = new SegLayers({ fetch: engine({ list: 503 }).fetch, storage: storage(memory) });
  offline.setSample('z1');
  assert.deepEqual(names(offline), before);
  await settle();
  assert.equal(offline.catalogue.state, 'failed');
  assert.deepEqual(names(offline), before, 'the list could not be read: still by the level\'s name');
  const online = new SegLayers({ fetch: engine().fetch, storage: storage(memory) });
  online.setSample('z1');
  await settle();
  assert.deepEqual(names(online), ['Cellpose', 'earlier segmentation (imported)'], 'the list\'s names (one it no longer holds goes)');
});

test('nothing is asked for while nothing is listed: the list is read when Add a mask… is pressed, once', async () => {
  const e = engine(), store = new SegLayers({ fetch: e.fetch, storage: storage() });
  store.setSample('z2'); await settle();
  assert.deepEqual(e.asked, [], 'opening a run or the picker costs no request');
  assert.deepEqual(store.addable(), []);
  store.loadCatalogue(); assert.equal(store.catalogue.state, 'loading');
  store.loadCatalogue(); await settle();
  assert.equal(e.lists(), 1, 'one request while it is read');
  assert.equal(store.catalogue.state, 'ready');
  assert.deepEqual(store.addable().map(i => i.key), ITEMS.map(i => i.key));
  const layer = store.add('cellpose_test');
  assert.equal(layer.name, 'Cellpose'); assert.equal(layer.style.on, true); assert.equal(layer.style.color, SEG_COLORS[0]);
  assert.equal(store.add('cellpose_test'), null, 'listed already');
  assert.equal(store.add(''), null);
  await settle();
  assert.deepEqual(e.fetched(), ['cellpose_test']);
  assert.equal(layer.state, 'ready');
  assert.deepEqual(store.addable().map(i => i.key), ITEMS.map(i => i.key).filter(k => k !== 'cellpose_test'));
  store.loadCatalogue(); await settle();
  assert.equal(e.lists(), 1, 'read once per sample');
  assert.equal(store.add('xenium_cell_boundaries').style.color, SEG_COLORS[1], 'the next free colour');
});

test(`at most ${MAX_SHOWN} show at once: a fifth is refused with why; one added while four show is listed hidden, unfetched`, async () => {
  const e = engine(), store = new SegLayers({ fetch: e.fetch, storage: storage() });
  store.setSample('z1'); store.loadCatalogue(); await settle();
  const layers = ITEMS.slice(0, 4).map(i => store.add(i.key));
  assert.ok(layers.every(l => l.style.on)); assert.equal(store.note, null);
  const fifth = store.add(ITEMS[4].key);
  assert.equal(fifth.style.on, false); assert.equal(store.note, SEG_TEXT.addedHidden);
  await settle();
  assert.deepEqual(e.fetched().sort(), ITEMS.slice(0, 4).map(i => i.key).sort(), 'the hidden fifth is not fetched');
  assert.equal(store.drawable().length, MAX_SHOWN);
  assert.deepEqual(store.show(fifth, true), { ok: false, why: SEG_TEXT.four });
  assert.equal(fifth.style.on, false); assert.equal(store.note, SEG_TEXT.four);
  assert.deepEqual(store.show(layers[1], false), { ok: true, why: null });
  assert.equal(store.note, null, 'the note goes with the next thing done');
  assert.deepEqual(store.show(fifth, true), { ok: true, why: null });
  await settle();
  assert.ok(e.fetched().includes(ITEMS[4].key));
  assert.deepEqual(store.drawable().map(l => l.key), [ITEMS[0], ITEMS[2], ITEMS[3], ITEMS[4]].map(i => i.key));
  // Even listed by hand (or remembered) with five on, at most four are drawn, and a sample remembered so opens with four.
  layers[1].style.on = true; layers[1].state = 'ready'; layers[1].set = fifth.set;
  assert.equal(store.drawable().length, MAX_SHOWN);
  const s = storage(remembered({ samples: { z9: { at: 1, layers: ITEMS.map(i => i.key), styles: {} } } }));
  const again = new SegLayers({ fetch: engine().fetch, storage: s });
  again.setSample('z9');
  assert.deepEqual(again.layers.map(l => l.style.on), [true, true, true, true, false]);
});

test('hidden: its fetch stops and nothing more is fetched; shown again it is fetched, and once loaded it is never fetched again', async () => {
  const e = engine({ hold: new Set(['cellpose_test']) }), store = new SegLayers({ fetch: e.fetch, storage: storage() });
  store.setSample('z1'); store.loadCatalogue(); await settle();
  const layer = store.add('cellpose_test'); await settle();
  const first = e.asked.at(-1);
  assert.equal(layer.state, 'loading'); assert.equal(first.signal.aborted, false);
  store.show(layer, false);
  assert.equal(first.signal.aborted, true, 'hiding stops its fetch');
  assert.equal(layer.state, 'idle');
  e.release('cellpose_test'); await settle();
  assert.equal(layer.set, null, 'a stopped reply is never used');
  e.hold.clear?.();
  store.show(layer, true); await settle();
  e.release('cellpose_test'); await settle();
  assert.equal(layer.state, 'ready'); assert.equal(e.fetched().length, 2);
  // Loaded, hidden: its paths go (built again if drawn again), its outlines stay; shown again, nothing is fetched.
  const set = layer.set, box = [-1e4, -1e4, 1e4, 1e4];
  const Path = class { constructor() { Path.made++; } moveTo() {} lineTo() {} closePath() {} addPath() {} };
  Path.made = 0; layerPath(set, box, 1, Path); const built = Path.made;
  store.show(layer, false);
  Path.made = 0; layerPath(set, box, 1, Path);
  assert.equal(Path.made, built, 'its tiles were let go when hidden');
  store.show(layer, true); await settle();
  assert.equal(layer.set, set); assert.equal(layer.state, 'ready'); assert.equal(e.fetched().length, 2);
});

test('styles are remembered per sample and key, also after a layer is taken off; the samples used last are kept', async () => {
  let now = 100;
  const e = engine(), s = storage(), store = new SegLayers({ fetch: e.fetch, storage: s, now: () => now });
  store.setSample('z1'); store.loadCatalogue(); await settle();
  const a = store.add('cellpose_test');
  store.setStyle(a, { color: '#00FF00', opacity: .4, width: 3, fill: true, on: false });
  assert.deepEqual(a.style, { on: true, color: '#00ff00', opacity: .4, width: 3, fill: true }, 'on is show()\'s, never a style change');
  assert.deepEqual(stored(s).samples.z1, { at: 100, layers: ['cellpose_test'], styles: { cellpose_test: { ...a.style, name: 'Cellpose' } } });
  assert.equal(store.remove(a), true); assert.equal(store.remove(a), false);
  assert.deepEqual(stored(s).samples.z1.layers, []);
  assert.equal(stored(s).samples.z1.styles.cellpose_test.color, '#00ff00', 'kept after it was taken off');
  const back = store.add('cellpose_test');
  assert.deepEqual(back.style, { on: true, color: '#00ff00', opacity: .4, width: 3, fill: true }, 'listed again: as it was, and shown');
  // Another sample: its own styles for the same key.
  now = 200; store.setSample('z2'); store.loadCatalogue(); await settle();
  assert.deepEqual(store.add('cellpose_test').style, { ...SEG_STYLE, color: SEG_COLORS[0] });
  assert.deepEqual(Object.keys(stored(s).samples), ['z2', 'z1'], 'the sample used last first');
  // Many samples: the KEEP_SAMPLES used last are kept.
  for (let k = 0; k < KEEP_SAMPLES + 5; k++) { now = 300 + k; store.setSample(`s${k}`); store.add('cellpose_test'); }
  const kept = Object.keys(stored(s).samples);
  assert.equal(kept.length, KEEP_SAMPLES); assert.equal(kept[0], `s${KEEP_SAMPLES + 4}`); assert.ok(!kept.includes('z1'));
  // Storage that cannot be read or written only forgets.
  for (const broken of [storage({}, true), storage({ [STORAGE_KEY]: 'not json' }), storage({ [STORAGE_KEY]: '{"samples":[1]}' }), null]) {
    const other = new SegLayers({ fetch: engine().fetch, storage: broken });
    assert.doesNotThrow(() => { other.setSample('z1'); other.loadCatalogue(); });
    await settle();
    assert.doesNotThrow(() => other.setStyle(other.add('cellpose_test'), { opacity: .2 }));
  }
  assert.deepEqual(savedStyle(null), {}); assert.deepEqual(savedStyle({ color: '#12345' }), {});
});

test('the sample\'s list names the layers and takes off any it no longer holds; a sample listing none (404) has nothing to add', async () => {
  const s = storage(remembered({ samples: { z1: { at: 1, layers: ['gone_shapes', 'cellpose_test'], styles: { gone_shapes: { on: true, color: '#abcdef' } } } } }));
  const e = engine(), store = new SegLayers({ fetch: e.fetch, storage: s });
  store.setSample('z1'); await settle();
  assert.deepEqual(store.layers.map(l => [l.key, l.name]), [['cellpose_test', 'Cellpose']]);
  assert.deepEqual(stored(s).samples.z1.layers, ['cellpose_test']);
  assert.equal(stored(s).samples.z1.styles.gone_shapes.color, '#abcdef', 'its style stays remembered');
  const none = new SegLayers({ fetch: engine({ list: 404 }).fetch, storage: storage(remembered({ samples: { z1: { at: 1, layers: ['cellpose_test'], styles: {} } } })) });
  none.setSample('z1'); await settle();
  assert.deepEqual([none.catalogue.state, none.catalogue.items, none.addable()], ['ready', [], []]);
  assert.equal(none.layers.length, 1, 'no list is no reason to take a layer off');
  const failing = new SegLayers({ fetch: engine({ list: 500 }).fetch, storage: storage() });
  failing.setSample('z1'); failing.loadCatalogue(); await settle();
  assert.deepEqual([failing.catalogue.state, failing.catalogue.error], ['failed', 'HTTP 500']);
});

test("a layer's outlines come from the list's own URL once it is read (it names the element's version); before, the plain one", async () => {
  const asked = [], items = ITEMS.map(item => ({ ...item, outlines: `/api/v1/samples/z1/segmentations/${item.key}/outlines?v=${item.key.length}` }));
  const fetch = async url => {
    asked.push(url);
    if (url === '/api/v1/samples/z1/segmentations') return Response.json({ segmentations: items });
    const m = /\/segmentations\/([^/?]+)\/outlines(?:\?v=\d+)?$/.exec(url);
    return m ? new Response(outlines(3, m[1])) : new Response('{}', { status: 404 });
  };
  const store = new SegLayers({ fetch, storage: storage() });
  store.setSample('z1'); store.loadCatalogue(); await settle();
  store.add('cellpose_test'); await settle();
  assert.equal(store.find('cellpose_test').state, 'ready');
  assert.equal(asked.at(-1), '/api/v1/samples/z1/segmentations/cellpose_test/outlines?v=13');
  // Remembered and shown as the sample opens: fetched at once, before its list says the version.
  const again = new SegLayers({ fetch, storage: storage(remembered({ samples: { z2: { at: 1, layers: ['cell_boundaries'], styles: {} } } })) });
  again.setSample('z2'); await settle();
  assert.equal(again.find('cell_boundaries').state, 'ready');
  assert.ok(asked.includes('/api/v1/samples/z2/segmentations/cell_boundaries/outlines'));
});

test('the list is read anew when asked (Add opened, a save into the sample): a new element offered, one written again fetched from its new version', async () => {
  // A stand-in engine whose list can change (a save) and be held; each element's outlines at a version: v2 has 7 cells.
  const asked = [], hold = { list: null, fail: false };
  const listed = (item, v) => ({ ...item, version: v, outlines: `/api/v1/samples/z1/segmentations/${item.key}/outlines?v=${v}` });
  let list = [listed(ITEMS[0], 'v1'), listed(ITEMS[1], 'v1')];
  const fetch = async url => {
    asked.push(url);
    if (url === '/api/v1/samples/z1/segmentations') {
      if (hold.list) await hold.list;
      return hold.fail ? new Response('busy', { status: 500 }) : Response.json({ segmentations: list });
    }
    const m = /\/segmentations\/([^/?]+)\/outlines(?:\?v=(\w+))?$/.exec(url);
    return m ? new Response(outlines(m[2] === 'v1' || !m[2] ? 3 : 7, m[1])) : new Response('{}', { status: 404 });
  };
  const lists = () => asked.filter(u => u.endsWith('/segmentations')).length, outlinesAsked = () => asked.filter(u => u.includes('/outlines'));
  const store = new SegLayers({ fetch, storage: storage() }), told = [];
  store.listen(drawn => told.push(drawn));
  store.setSample('z1'); store.loadCatalogue(); await settle();
  const layer = store.add('xenium_cell_boundaries'), hidden = store.add('cell_boundaries'); await settle();
  store.show(hidden, false);
  assert.deepEqual([layer.set.count, hidden.set.count, hidden.state], [3, 3, 'ready']);
  store.loadCatalogue(); await settle();
  assert.equal(lists(), 1, 'read once unless asked anew');
  // A Cellpose segmentation saved into the sample; both listed elements written again under their keys.
  list = [listed(ITEMS[0], 'v2'), listed(ITEMS[1], 'v2'), listed(ITEMS[3], 'v1')];
  told.length = 0;
  store.sampleSaved('z1');
  assert.equal(store.catalogue.state, 'ready', 'the list read before stays on show meanwhile');
  await settle();
  assert.equal(lists(), 2);
  assert.deepEqual(store.addable().map(i => i.key), ['cellpose_test'], 'the new one offered');
  assert.equal(outlinesAsked().at(-1), '/api/v1/samples/z1/segmentations/xenium_cell_boundaries/outlines?v=v2', 'shown: fetched anew from its new version');
  assert.deepEqual([layer.state, layer.set.count], ['ready', 7]);
  assert.deepEqual([hidden.state, hidden.set], ['idle', null], 'hidden: its old outlines go, nothing fetched');
  assert.ok(told.includes(true), 'what is drawn changed');
  store.show(hidden, true); await settle();
  assert.equal(outlinesAsked().at(-1), '/api/v1/samples/z1/segmentations/cell_boundaries/outlines?v=v2'); assert.equal(hidden.set.count, 7);
  // The same version again: nothing fetched. Asked anew while a read is on its way: once more after it, one at a time.
  const fetched = outlinesAsked().length;
  let release; hold.list = new Promise(resolve => { release = resolve; });
  store.loadCatalogue(true); store.loadCatalogue(true); store.sampleSaved('z1'); await settle();
  assert.equal(lists(), 3, 'one read at a time');
  hold.list = null; release(); await settle();
  assert.equal(lists(), 4, 'once more after it'); await settle(); assert.equal(lists(), 4);
  assert.equal(outlinesAsked().length, fetched, 'the same versions: nothing fetched again');
  // A read that fails keeps the list on show; another sample's save asks nothing.
  hold.fail = true; store.loadCatalogue(true); await settle();
  assert.deepEqual([store.catalogue.state, store.addable().map(i => i.key)], ['ready', ['cellpose_test']]);
  hold.fail = false; store.sampleSaved('z9'); await settle();
  assert.equal(lists(), 5);
  // Put aside (the picker), a save into it meanwhile: back, its list is read again and the layer written again comes anew.
  store.setSample(null);
  list = [listed(ITEMS[0], 'v3'), ...list.slice(1)];
  store.sampleSaved('z1');
  store.setSample('z1');
  assert.equal(store.layers[0], layer, 'its layers back at once'); await settle();
  assert.equal(lists(), 6); assert.equal(outlinesAsked().at(-1), '/api/v1/samples/z1/segmentations/xenium_cell_boundaries/outlines?v=v3');
  // Put aside without a save: nothing asked again.
  store.setSample(null); store.setSample('z1'); await settle();
  assert.equal(lists(), 6);
});

test('a layer that cannot be loaded says so, and is fetched again on Try again or when shown again', async () => {
  const e = engine({ fail: new Set(['cellpose_test']) }), store = new SegLayers({ fetch: e.fetch, storage: storage() });
  store.setSample('z1'); store.loadCatalogue(); await settle();
  const layer = store.add('cellpose_test'); await settle();
  assert.deepEqual([layer.state, layer.error, layer.set], ['failed', 'HTTP 500', null]);
  assert.deepEqual(store.drawable(), []);
  store.ensure(layer); await settle();
  assert.equal(e.fetched().length, 1, 'a failed layer waits for Try again');
  e.fail.delete('cellpose_test');
  store.retry(layer); await settle();
  assert.equal(layer.state, 'ready'); assert.equal(e.fetched().length, 2);
  e.fail.add('xenium_cell_boundaries');
  const other = store.add('xenium_cell_boundaries'); await settle();
  assert.equal(other.state, 'failed');
  e.fail.clear();
  store.show(other, false); store.show(other, true); await settle();
  assert.equal(other.state, 'ready', 'hidden and shown again: fetched again');
});

test('another sample and back fetches nothing again (the last sample\'s outlines are kept aside); late replies are dropped', async () => {
  const e = engine({ hold: new Set(['cell_boundaries']) }), s = storage(), store = new SegLayers({ fetch: e.fetch, storage: s });
  store.setSample('z1'); store.loadCatalogue(); await settle();
  const layer = store.add('cellpose_test'); store.add('cell_boundaries'); await settle();
  const set = layer.set;
  store.setSample(null);
  assert.deepEqual([store.layers, store.drawable(), store.catalogue], [[], [], null]);
  e.release('cell_boundaries'); await settle();
  store.setSample('z1');
  assert.equal(store.layers[0].set, set, 'its outlines at once: kept aside');
  assert.equal(store.layers[0].state, 'ready');
  await settle();
  assert.deepEqual(e.fetched(), ['cellpose_test', 'cell_boundaries', 'cell_boundaries'], 'only the one that had not loaded is asked again');
  e.release('cell_boundaries'); await settle();
  assert.equal(store.layers[1].state, 'ready');
  // Without storage too (a private window): the list, its layers and the sample's list come back as they were.
  const e2 = engine(), bare = new SegLayers({ fetch: e2.fetch, storage: null });
  bare.setSample('z1'); bare.loadCatalogue(); await settle();
  const kept = bare.add('cellpose_test'); bare.setStyle(kept, { color: '#010203' }); await settle();
  bare.setSample(null); bare.setSample('z1'); await settle();
  assert.deepEqual([bare.layers, bare.layers[0].state, bare.layers[0].style.color], [[kept], 'ready', '#010203']);
  assert.deepEqual([e2.lists(), e2.fetched()], [1, ['cellpose_test']], 'nothing asked again');
  // A reply for a sample left meanwhile changes nothing.
  e.hold.add('xenium_cell_boundaries');
  const late = store.add('xenium_cell_boundaries'); await settle();
  store.setSample('z3');
  e.release('xenium_cell_boundaries'); await settle();
  assert.equal(late.set, null); assert.deepEqual(store.layers, []);
});

test('the harness snapshot and every word the group shows: plain, no banned or kind words', () => {
  const store = new SegLayers({ fetch: async () => new Response('{}', { status: 404 }), storage: null });
  assert.deepEqual(store.snapshot(), { sample: null, note: null, loaded: [] });
  store.setSample('z1');
  const layer = store.make('cellpose_test', { color: '#7be29a' }, ITEMS[3]);
  store.layers.push(layer);
  assert.deepEqual(store.snapshot().loaded, [{ key: 'cellpose_test', name: 'Cellpose', visible: true, color: '#7be29a', opacity: .8, width: 1, fill: false, state: 'idle' }]);
  assert.equal(store.colorOf('cellpose_test'), '#7be29a'); assert.equal(store.colorOf('nope'), null);
  for (const text of Object.values(SEG_TEXT)) {
    assert.deepEqual(bannedWords(text), [], text);
    assert.doesNotMatch(text, /cell\s*type|\bmore\b|\bless\b|[▲△↑⬆▼▽↓⬇]/i, text);
  }
});

test('the group is called Load Other Mask and its words say mask; Compare keeps its own words', () => {
  assert.equal(SEG_TEXT.group, 'Load Other Mask'); assert.equal(SEG_TEXT.add, 'Add a mask…');
  assert.equal(SEG_TEXT.reading, "Reading the sample's masks…"); assert.equal(SEG_TEXT.readFailed, "Could not read the sample's masks.");
  assert.equal(SEG_TEXT.none, 'This sample has no other masks.'); assert.equal(SEG_TEXT.allAdded, 'Every mask of this sample is in the list.');
  assert.equal(SEG_TEXT.four, `At most ${MAX_SHOWN} masks show at once. Hide one to show this one.`);
  assert.equal(SEG_TEXT.addedHidden, `Added, hidden: at most ${MAX_SHOWN} masks show at once. Hide one to show it.`);
  for (const text of Object.values(SEG_TEXT)) assert.doesNotMatch(text, /segmentation/i, text);
  // Compare adds a segmentation as a column: not this group, so its words stay.
  assert.match(COMPARE_TEXT.addNone, /segmentation/); assert.match(COMPARE_TEXT.buttonTooltip, /segmentations/);
});
