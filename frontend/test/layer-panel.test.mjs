import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { domFixture } from './dom-fixture.mjs';
import { QC_TEXT, bannedWords } from '../src/qc-copy.mjs';
import { SegLayers, SEG_TEXT, MAX_SHOWN } from '../src/seg-layers.mjs';
import { encodeVectors } from './vectors-fixture.mjs';

// Layers > Cells > Colour by: identity, cluster, and Mixing (the mixing check's values); the
// display modes (Cells / Outlines) and the settings saved with them.
let source = await readFile(new URL('../src/layer-panel.js', import.meta.url), 'utf8');
for (const file of ['image-layers.mjs', 'qc-copy.mjs', 'stain.mjs', 'seg-layers.mjs']) source = source.replace(`'./${file}'`, JSON.stringify(new URL(`../src/${file}`, import.meta.url).href));
const { LayerPanel, CELL_COLOR_BY, DISPLAY_MODES, CONTRAST_SAMPLES, savedStage } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const KEY = 'sopa-companion:layers:v1';
const STAIN_OUTLINES = 'Cell outlines';
// The globals a panel replaces, as they were before any test (several panels in one test put back the same ones).
const ORIGINAL = new Map(['document', 'localStorage'].map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));

/** A panel over the page's own markup; `stored`: what localStorage holds when it opens ({key: text}); `segments`: the
 *  segmentation layers' store PriorLayers would hand it (none: a page without them). */
function panel(t, stored = {}, segments = undefined) {
  const dom = domFixture(), saved = new Map(Object.entries(stored));
  dom.Element.prototype.prepend = function (...nodes) { for (const n of nodes.reverse()) { n.parentElement = this; this.childNodes.unshift(n); } };
  dom.document.body.innerHTML = (globalThis.__indexHtml ??= '');
  const globals = { document: dom.document, localStorage: { getItem: k => saved.get(k) ?? null, setItem: (k, v) => saved.set(k, v) } };
  for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, writable: true, configurable: true });
  t.after(() => { for (const [k, d] of ORIGINAL) if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; });
  const stage = { options: { cellColorBy: 'identity', transcripts: true, trails: true, fill: .4, transcriptSize: 2, transcriptAlpha: .7, transcriptColorBy: 'cell', mode: 0 },
    set(o) { Object.assign(this.options, o); }, dirty: false };
  const priors = { settings: { prior: { on: false, opacity: 1 }, nuclei: { on: false, opacity: 1, color: '#fff', width: 1 }, xenium: { on: false, opacity: 1, color: '#fff', width: 1 } }, segments };
  const layers = new LayerPanel(stage, priors);
  const select = () => layers.controls.all().find(n => n.tagName === 'SELECT');
  return { dom, stage, layers, select, saved };
}

test('Colour by offers Mixing as a third choice only while the check is on, saying in plain words why it cannot be chosen', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { layers, select, stage } = panel(t);
  assert.deepEqual(CELL_COLOR_BY, ['identity', 'cluster', 'mixing']);
  let options = select().children;
  assert.deepEqual(options.map(o => o.textContent), ['Identity', 'Cluster'], 'the check is off (its default) or not known yet: no Mixing');
  assert.equal(layers.mixingWhy, null);
  const before = select(); layers.setMixingAvailable(false); assert.equal(select(), before, 'no change: nothing drawn again');
  layers.setMixingAvailable(true); options = select().children;
  assert.deepEqual(options.map(o => o.textContent), ['Identity', 'Cluster', QC_TEXT.mixing]);
  assert.equal(options[2].disabled, false); assert.notEqual(select(), before);
  const picked = [];
  layers.onColorBy = value => picked.push(value);
  const el = select(); el.value = 'mixing'; el.onchange();
  assert.deepEqual(picked, ['mixing'], 'main.js decides (it remembers the previous colours for M and the panel button)');
  delete layers.onColorBy; el.value = 'cluster'; el.onchange(); assert.equal(stage.options.cellColorBy, 'cluster');
  // On, but it cannot run (failed, unavailable): the disabled choice says so instead of "turn it on".
  layers.setMixingAvailable(false, QC_TEXT.mixingCantRun);
  assert.equal(select().children[2].disabled, true); assert.equal(select().children[2].title, QC_TEXT.mixingCantRun);
  assert.deepEqual(bannedWords(QC_TEXT.mixingCantRun), []);
  const now = select(); layers.setMixingAvailable(false, QC_TEXT.mixingCantRun); assert.equal(select(), now, 'unchanged: not drawn again');
  // Switched off again: gone from the list (key M says why: main.js falls back to mixingOffTooltip).
  layers.setMixingAvailable(false); assert.deepEqual(select().children.map(o => o.value), ['identity', 'cluster']); assert.equal(layers.mixingWhy, null);
  layers.setMixingAvailable(true, QC_TEXT.mixingCantRun); assert.equal(layers.mixingWhy, null, 'a reason only while it cannot be chosen');
  // Coloured by Mixing as the check goes off: still listed (disabled, saying how to get it back) until main.js has put
  // the cells' own colours back, so the choice shown is never blank.
  layers.setMixingAvailable(false); stage.options.cellColorBy = 'mixing'; layers.render();
  options = select().children;
  assert.deepEqual(options.map(o => o.value), CELL_COLOR_BY); assert.equal(options[2].disabled, true);
  assert.equal(options[2].title, QC_TEXT.mixingOffTooltip); assert.deepEqual(bannedWords(QC_TEXT.mixingOffTooltip), []);
});

test('the display modes are Cells and Outlines; settings saved in the removed Activity mode open in Cells, migrated once', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  assert.deepEqual(DISPLAY_MODES, ['cells', 'outlines']);
  // The page's own buttons (index.html): Cells (key 1) and Outlines (key 2), nothing else.
  const fresh = panel(t);
  assert.deepEqual(fresh.dom.$('mode').children.map(b => b.dataset.value), DISPLAY_MODES);
  assert.deepEqual(fresh.dom.$('mode').children.map(b => b.textContent), ['Cells', 'Outlines']);
  assert.equal(fresh.saved.has(KEY), false, 'nothing saved: nothing written');
  // Saved settings as the stage can take them: a mode it no longer has is dropped (its default stands), the rest kept.
  assert.deepEqual(savedStage({ mode: 2, fill: .3 }), { stage: { fill: .3 }, dropped: true });
  assert.deepEqual(savedStage({ mode: 1, fill: .3 }), { stage: { mode: 1, fill: .3 }, dropped: false });
  assert.deepEqual(savedStage({ fill: .3 }), { stage: { fill: .3 }, dropped: false });
  for (const mode of [2, 3, -1, 1.5, '1', null, undefined]) assert.equal(savedStage({ mode }).dropped, true, String(mode));
  for (const value of [null, undefined, 7, [1], 'x']) assert.deepEqual(savedStage(value), { stage: {}, dropped: false }, String(value));
  // A page whose settings were saved in Activity mode: Cells, every other saved setting applied, and the settings
  // written back once without the old mode.
  const stored = { stage: { mode: 2, fill: .25, trails: false, cellColorBy: 'cluster' }, priorChosen: true };
  const old = panel(t, { [KEY]: JSON.stringify(stored) });
  assert.equal(old.stage.options.mode, 0); assert.equal(old.stage.options.fill, .25); assert.equal(old.stage.options.trails, false);
  assert.equal(old.stage.options.cellColorBy, 'cluster'); assert.equal(old.layers.priorChosen, true);
  const written = JSON.parse(old.saved.get(KEY));
  assert.equal(written.stage.mode, 0); assert.equal(written.stage.fill, .25); assert.equal(written.priorChosen, true);
  // Outlines, as saved, stays Outlines and is not written again; settings that cannot be read open as defaults.
  const outlines = panel(t, { [KEY]: JSON.stringify({ stage: { mode: 1 } }) });
  assert.equal(outlines.stage.options.mode, 1); assert.equal(outlines.saved.get(KEY), JSON.stringify({ stage: { mode: 1 } }));
  for (const text of ['7', '[2]', 'not json']) assert.equal(panel(t, { [KEY]: text }).stage.options.mode, 0, text);
});

test('stain display lists its outlines and image channels, with opacity and width controls', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { layers } = panel(t);
  layers.sampleKind='zarr';layers.runKind='stain';layers.channels=[{name:'DAPI',on:true,colormap:'gray'}];layers.render();
  assert.equal(layers.selected,'xenium');
  const names=layers.list.children.map(row=>row.children[1].textContent);
  assert.deepEqual(names,['Cell outlines','DAPI']);
  assert.ok(layers.cellControls.hidden);
  assert.match(layers.controls.textContent,/Opacity.*Colour.*Line width/s);
});

// ---- Layers > Load Other Mask (the store: seg-layers.test.mjs) --------------------------------------------------------
const settle = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise(resolve => setImmediate(resolve)); };
const LIST = [
  { key: 'xenium_cell_boundaries', name: 'Xenium cells', source: 'xenium', model: null, count: 44691, extent: [0, 0, 1, 1], table: 'xenium_table' },
  { key: 'cell_boundaries', name: 'earlier segmentation (imported)', source: 'imported', model: null, count: 42679, extent: [0, 0, 1, 1], table: 'table' },
  { key: 'cellpose_test', name: 'Cellpose', source: 'cellpose', model: 'cp4', count: 5378, extent: [0, 0, 1, 1], table: null },
  { key: 'xenium_nucleus_boundaries', name: 'xenium_nucleus_boundaries', source: 'other', model: null, count: 44327, extent: [0, 0, 1, 1], table: null },
  { key: 'proseg_boundaries', name: 'proseg (this app)', source: 'proseg', model: null, count: 41000, extent: [0, 0, 1, 1], table: 'proseg_table' },
];
/** A store against a stand-in engine (`failing`: keys whose outlines answer 500). */
function store(failing = new Set()) {
  const asked = [];
  const segments = new SegLayers({ storage: null, fetch: async url => {
    asked.push(url);
    if (url.endsWith('/segmentations')) return Response.json({ segmentations: LIST });
    const key = decodeURIComponent(url.split('/').at(-2));
    if (failing.has(key)) return new Response('busy', { status: 500 });
    return new Response(encodeVectors({ cells: [{ id: 'a', poly: [[0, 0], [9, 0], [9, 9], [0, 0]] }], nuclei: null }));
  } });
  return { segments, asked };
}
const attr = (node, name) => node.getAttribute(name);
const find = (root, test) => root.all().find(test);
const rowsOf = group => group.children.filter(n => attr(n, 'data-seg-key') != null);

test('the Load Other Mask group: after the run\'s own outlines, a row per layer (eye, colour, name over key), then Add a mask… listing the others', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { segments, asked } = store(), { layers, stage } = panel(t, {}, segments);
  layers.sampleKind = 'zarr'; layers.render();
  assert.equal(find(layers.list, n => attr(n, 'data-seg-group') != null), undefined, 'no sample: no group');
  segments.setSample('z1');
  const group = layers.segEl, names = () => layers.list.children.map(n => n === group ? 'SEGMENTATIONS' : n.children[1].textContent);
  assert.equal(attr(group, 'data-seg-group'), '');
  assert.deepEqual(names(), ['Transcripts', 'Cells', 'Proseg prior', 'Prior nuclei', 'Prior shapes', 'SEGMENTATIONS'], 'after Prior shapes, the labels kept');
  layers.setChannels([{ name: 'DAPI', on: true, colormap: 'gray' }]);
  assert.deepEqual(names().slice(-2), ['SEGMENTATIONS', 'DAPI'], 'before the image channels');
  assert.equal(group.children[0].textContent, SEG_TEXT.group); assert.equal(SEG_TEXT.group, 'Load Other Mask');
  const add = find(group, n => attr(n, 'data-action') === 'seg-add');
  assert.equal(add.textContent, SEG_TEXT.add); assert.equal(add.textContent, 'Add a mask…'); assert.equal(attr(add, 'aria-expanded'), 'false'); assert.equal(add.disabled, false);
  assert.deepEqual(asked, [], 'nothing asked for until it is pressed');
  add.click();
  assert.equal(find(group, n => n.tagName === 'P').textContent, SEG_TEXT.reading);
  await settle();
  const picks = group.all().filter(n => attr(n, 'data-seg-add-key') != null);
  assert.deepEqual(picks.map(n => attr(n, 'data-seg-add-key')), LIST.map(i => i.key));
  assert.deepEqual(picks.map(n => n.textContent), ['Xenium cellsxenium_cell_boundaries · 44,691 shapes', 'earlier segmentation (imported)cell_boundaries · 42,679 shapes',
    'Cellposecellpose_test · 5,378 shapes', 'xenium_nucleus_boundaries44,327 shapes', 'proseg (this app)proseg_boundaries · 41,000 shapes'], 'each with its key, once');
  picks[2].click(); await settle();
  let rows = rowsOf(group);
  assert.deepEqual(rows.map(r => attr(r, 'data-seg-key')), ['cellpose_test']);
  assert.equal(find(group, n => attr(n, 'data-seg-add-key') != null), undefined, 'the list closes once one is added');
  const [eye, name] = rows[0].children;
  assert.deepEqual([attr(eye, 'aria-label'), attr(eye, 'aria-pressed')], ['Toggle Cellpose', 'true']);
  assert.equal(name.children[0].style.background, segments.layers[0].style.color, 'its colour');
  assert.equal(name.textContent, 'Cellposecellpose_test', 'its name over its key');
  assert.match(name.title, /Cellpose \(cellpose_test\): 1 shapes/);
  assert.equal(stage.dirty, false, 'the panel itself never draws; the store marks the stage (PriorLayers)');
  eye.click();
  rows = rowsOf(group);
  assert.equal(attr(rows[0].children[0], 'aria-pressed'), 'false'); assert.equal(segments.layers[0].style.on, false);
  // A name that is its key shows once.
  add.click(); await settle();
  find(group, n => attr(n, 'data-seg-add-key') === 'xenium_nucleus_boundaries').click(); await settle();
  assert.equal(rowsOf(group)[1].children[1].textContent, 'xenium_nucleus_boundaries');
});

test(`the fifth layer shown explains instead (never greyed out); added while ${MAX_SHOWN} show, it is listed hidden and says why`, async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { segments } = store(), { layers } = panel(t, {}, segments);
  segments.setSample('z1'); segments.loadCatalogue(); await settle();
  for (const item of LIST.slice(0, 4)) segments.add(item.key);
  await settle();
  const group = layers.segEl;
  find(group, n => attr(n, 'data-action') === 'seg-add').click();
  const fifth = find(group, n => attr(n, 'data-seg-add-key') === LIST[4].key);
  assert.equal(fifth.disabled, false); fifth.click(); await settle();
  const note = () => find(group, n => n.className === 'seg-note')?.textContent ?? null;
  assert.equal(note(), SEG_TEXT.addedHidden);
  const row = rowsOf(group).at(-1), eye = row.children[0];
  assert.deepEqual([attr(row, 'data-seg-key'), attr(eye, 'aria-pressed'), eye.disabled], [LIST[4].key, 'false', false]);
  eye.click();
  assert.equal(note(), SEG_TEXT.four); assert.equal(attr(rowsOf(group).at(-1).children[0], 'aria-pressed'), 'false');
  rowsOf(group)[0].children[0].click();
  assert.equal(note(), null, 'gone with the next thing done');
  rowsOf(group).at(-1).children[0].click();
  assert.equal(attr(rowsOf(group).at(-1).children[0], 'aria-pressed'), 'true');
});

test('a layer\'s fields: Colour, Opacity, Line width, Show as (Outlines or Filled) and Remove, each kept by the store', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { segments } = store(), { layers } = panel(t, {}, segments);
  segments.setSample('z1'); segments.loadCatalogue(); await settle();
  const layer = segments.add('cellpose_test'); await settle();
  rowsOf(layers.segEl)[0].children[1].click();
  assert.equal(layers.selected, layer);
  assert.equal(attr(rowsOf(layers.segEl)[0].children[1], 'aria-pressed'), 'true');
  assert.equal(layers.cellControls.hidden, true);
  const fields = layers.controls.children.filter(n => n.className === 'layer-field');
  assert.deepEqual(fields.map(f => f.childNodes[0].textContent), [SEG_TEXT.colour, SEG_TEXT.opacity, SEG_TEXT.width, SEG_TEXT.showAs]);
  const [color, opacity, width, fill] = fields.map(f => f.children[0]);
  color.value = '#ff0000'; color.oninput();
  opacity.value = '0.25'; opacity.oninput();
  width.value = '3'; width.oninput();
  assert.deepEqual(fill.children.map(o => [o.value, o.textContent]), [['outline', SEG_TEXT.outlines], ['fill', SEG_TEXT.filled]]);
  fill.value = 'fill'; fill.onchange();
  assert.deepEqual(layer.style, { on: true, color: '#ff0000', opacity: .25, width: 3, fill: true });
  assert.equal(rowsOf(layers.segEl)[0].children[1].children[0].style.background, '#ff0000', 'the row\'s colour follows');
  const remove = layers.controls.children.find(n => attr(n, 'data-action') === 'seg-remove');
  assert.equal(remove.textContent, SEG_TEXT.remove);
  remove.click();
  assert.deepEqual(segments.layers, []); assert.equal(layers.selected, 'cells'); assert.deepEqual(rowsOf(layers.segEl), []);
  // In a stain run (its own outlines first), taken off the list the selection goes back to them.
  layers.runKind = 'stain'; segments.add('cellpose_test'); await settle();
  rowsOf(layers.segEl)[0].children[1].click();
  assert.equal(layers.selected, segments.layers[0]);
  assert.deepEqual(layers.list.children.map(n => n === layers.segEl ? 'SEGMENTATIONS' : n.children[1].textContent), [STAIN_OUTLINES, 'SEGMENTATIONS']);
  segments.setSample('z2');
  assert.equal(layers.selected, 'xenium', 'another sample: its layers are not this one\'s');
});

test('loading and failed layers say so, with Try again; the picker can show the group without the fields; every word plain', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const failing = new Set(['cell_boundaries']), { segments } = store(failing), { layers, dom } = panel(t, {}, segments);
  segments.setSample('z1'); segments.loadCatalogue(); await settle();
  const picker = dom.document.createElement('div');
  layers.mountSegmentations(picker);
  segments.add('cell_boundaries');
  assert.equal(rowsOf(layers.segEl)[0].children[1].textContent, `earlier segmentation (imported)cell_boundaries${SEG_TEXT.loading}`, 'its state on a line of its own');
  await settle();
  const row = rowsOf(layers.segEl)[0], retry = row.children[2];
  assert.equal(row.children[1].textContent, `earlier segmentation (imported)cell_boundaries${SEG_TEXT.failed}`);
  assert.equal(find(row, n => n.className === 'seg-state').textContent, SEG_TEXT.failed);
  assert.match(row.children[1].title, /HTTP 500/);
  assert.deepEqual([attr(retry, 'data-action'), retry.textContent], ['seg-retry', SEG_TEXT.retry]);
  failing.clear(); retry.click(); await settle();
  assert.equal(segments.layers[0].state, 'ready'); assert.equal(rowsOf(layers.segEl)[0].children.length, 2);
  // The picker's copy: the same rows and add row, names that select nothing.
  const prow = rowsOf(picker)[0];
  assert.equal(prow.children[1].tagName, 'SPAN');
  assert.ok(find(picker, n => attr(n, 'data-action') === 'seg-add'));
  prow.children[0].click();
  assert.equal(segments.layers[0].style.on, false); assert.equal(attr(rowsOf(layers.segEl)[0].children[0], 'aria-pressed'), 'false', 'both drawn again');
  // Every word the group can show: its own and the fields'.
  find(layers.segEl, n => attr(n, 'data-action') === 'seg-add').click(); await settle();
  rowsOf(layers.segEl)[0].children[1].click();
  const words = [...layers.segEl.all(), ...layers.controls.all()].flatMap(n => [n.text, n.title, attr(n, 'aria-label')]).filter(Boolean);
  for (const text of [...words, ...Object.values(SEG_TEXT)]) assert.deepEqual(bannedWords(text), [], text);
  assert.ok(words.some(w => w.includes(SEG_TEXT.group)));
  // No sample (the first screen): no group, and its copy in the picker is emptied too.
  segments.setSample(null);
  assert.deepEqual(picker.children, []); assert.ok(!layers.list.children.includes(layers.segEl));
  segments.setSample('z1');
  assert.ok(layers.list.children.includes(layers.segEl)); assert.equal(rowsOf(picker).length, 1, 'back: listed again at once');
});

test('a selected segmentation layer stays selected when a new view image sets the channels again; a channel selected stays by its name', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { segments } = store(), { layers, dom } = panel(t, {}, segments);
  dom.Element.prototype.getContext = () => new Proxy({}, { get: (o, k) => (k in o ? o[k] : () => {}) });   // a channel's contrast histogram
  layers.sampleKind = 'zarr'; layers.render();
  segments.setSample('z1'); segments.loadCatalogue(); await settle();
  const layer = segments.add('cellpose_test'); await settle();
  layers.setChannels([{ name: 'DAPI', on: true, colormap: 'gray' }]);   // the first view image
  rowsOf(layers.segEl)[0].children[1].click();
  const fields = () => !!find(layers.controls, n => attr(n, 'data-action') === 'seg-remove');
  assert.equal(layers.selected, layer); assert.ok(fields());
  // A pan or zoom of a whole-sample (stain, patches) run: the next view's image, its channels new objects.
  layers.setChannels([{ name: 'DAPI', on: true, colormap: 'gray' }]);
  assert.equal(layers.selected, layer, 'still the layer'); assert.ok(fields(), 'its fields stay'); assert.equal(layers.cellControls.hidden, true);
  assert.equal(attr(rowsOf(layers.segEl)[0].children[1], 'aria-pressed'), 'true');
  // A channel selected follows its name into the new list; one no longer listed hands the selection back to the cells.
  layers.list.children.find(row => row !== layers.segEl && row.children[1]?.textContent === 'DAPI').children[1].click();
  assert.equal(layers.selected.name, 'DAPI');
  const next = [{ name: 'DAPI', on: true, colormap: 'gray' }, { name: 'RNA', on: false, colormap: 'green' }];
  layers.setChannels(next); assert.equal(layers.selected, next[0]);
  layers.setChannels([{ name: 'RNA', on: true, colormap: 'green' }]); assert.equal(layers.selected, 'cells');
});

test('Add a mask… is read anew at each opening (a segmentation saved since is offered) and closes when another sample opens', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  let list = LIST.slice(0, 2);
  const asked = [], segments = new SegLayers({ storage: null, fetch: async url => {
    asked.push(url);
    if (url.endsWith('/segmentations')) return Response.json({ segmentations: list });
    return new Response(encodeVectors({ cells: [{ id: 'a', poly: [[0, 0], [9, 0], [9, 9], [0, 0]] }], nuclei: null }));
  } });
  const { layers } = panel(t, {}, segments);
  layers.sampleKind = 'zarr'; layers.render();
  const add = () => find(layers.segEl, n => attr(n, 'data-action') === 'seg-add');
  const picks = () => layers.segEl.all().filter(n => attr(n, 'data-seg-add-key') != null).map(n => attr(n, 'data-seg-add-key'));
  const reads = sample => asked.filter(u => u === `/api/v1/samples/${sample}/segmentations`).length;
  segments.setSample('z1');
  add().click(); await settle();
  assert.deepEqual(picks(), LIST.slice(0, 2).map(i => i.key));
  // Closed, a Cellpose segmentation saved into the sample, opened again: the list on show at once, then read anew.
  add().click(); list = LIST.slice(0, 3);
  add().click();
  assert.deepEqual(picks(), LIST.slice(0, 2).map(i => i.key), 'what was read stays on show meanwhile (no "Reading…")');
  await settle();
  assert.deepEqual(picks(), LIST.slice(0, 3).map(i => i.key), 'the new one offered'); assert.equal(reads('z1'), 2);
  // A save into the sample while the list is open: it follows.
  list = LIST.slice(0, 4); segments.sampleSaved('z1'); await settle();
  assert.deepEqual(picks(), LIST.slice(0, 4).map(i => i.key));
  // Another sample opens (a run of it, nothing remembered): the list it showed was the last sample's, so it closes;
  // nothing is read until it is opened again (never stuck on "Reading the sample's masks…").
  segments.setSample('z2'); await settle();
  assert.equal(attr(add(), 'aria-expanded'), 'false'); assert.deepEqual(picks(), []);
  assert.equal(find(layers.segEl, n => n.tagName === 'P'), undefined);
  assert.equal(reads('z2'), 0);
  add().click();
  assert.equal(find(layers.segEl, n => n.tagName === 'P').textContent, SEG_TEXT.reading);
  await settle();
  assert.equal(reads('z2'), 1); assert.deepEqual(picks(), LIST.slice(0, 4).map(i => i.key));
});

// ---- an image's contrast limits per sample ----------------------------------------------------------------------------
const dapiMeta = stats => ({ name: 'DAPI', max: 65535, ...stats }), image16 = { data: new Uint16Array(4), width: 2, height: 2 };
test('contrast limits are remembered per sample and channel; a channel\'s look (shown, colours) is kept by its name', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  // Saved before: limits by the channel's name alone, from a sample whose image was far brighter.
  const old = { channels: { DAPI: { on: true, low: 2000, high: 60000, gamma: 1.4, opacity: .8, colormap: 'magenta', blending: 'translucent', depth: 16 } } };
  const { layers, saved } = panel(t, { [KEY]: JSON.stringify(old) });
  layers.sample = 'zA';
  let dapi = layers.channel(dapiMeta({ p_lo: 10, p_hi: 900 }), image16, 0);
  assert.deepEqual([dapi.low, dapi.high], [10, 900], 'the image\'s own statistics, not another sample\'s limits');
  assert.deepEqual([dapi.on, dapi.gamma, dapi.opacity, dapi.colormap, dapi.blending], [true, 1.4, .8, 'magenta', 'translucent'], 'its look, by name');
  // Changed on this sample: kept for it, the look by name.
  layers.setChannels([dapi]); dapi.low = 40; dapi.high = 700; dapi.colormap = 'green'; layers.save();
  const written = JSON.parse(saved.get(KEY));
  assert.deepEqual(written.contrast, { zA: { DAPI: { low: 40, high: 700, depth: 16 } } });
  assert.deepEqual(written.channels.DAPI, { on: true, gamma: 1.4, opacity: .8, colormap: 'green', blending: 'translucent' }, 'no limits by name');
  // Another sample with a channel of the same name: its own statistics, the colour carried over.
  layers.sample = 'zB';
  dapi = layers.channel(dapiMeta({ p_lo: 3000, p_hi: 50000 }), image16, 0);
  assert.deepEqual([dapi.low, dapi.high, dapi.colormap], [3000, 50000, 'green']);
  // The first again: its limits; an 8-bit stand-in image there starts from its own range.
  layers.sample = 'zA';
  assert.deepEqual(['low', 'high'].map(k => layers.channel(dapiMeta({ p_lo: 10, p_hi: 900 }), image16, 0)[k]), [40, 700]);
  assert.deepEqual(['low', 'high'].map(k => layers.channel(dapiMeta({ p_lo: 10, p_hi: 900 }), {}, 0)[k]), [0, 255]);
  // A page opened again remembers them; a channel of no known sample keeps none.
  const again = panel(t, { [KEY]: saved.get(KEY) });
  again.layers.sample = 'zA';
  assert.deepEqual(['low', 'high'].map(k => again.layers.channel(dapiMeta({ p_lo: 10, p_hi: 900 }), image16, 0)[k]), [40, 700]);
  again.layers.sample = null;
  const loose = again.layers.channel(dapiMeta({ p_lo: 10, p_hi: 900 }), image16, 0);
  again.layers.setChannels([loose]); loose.low = 1; again.layers.save();
  assert.deepEqual(Object.keys(JSON.parse(again.saved.get(KEY)).contrast), ['zA']);
});

test('contrast limits are kept for the samples changed last, at most CONTRAST_SAMPLES of them', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { layers, saved } = panel(t);
  const change = sample => { layers.sample = sample; const c = layers.channel(dapiMeta({ p_lo: 1, p_hi: 9 }), image16, 0); layers.setChannels([c]); c.high = 8; layers.save(); };
  for (let i = 0; i < CONTRAST_SAMPLES; i++) change(`z${i}`);
  change('z0');                                             // changed again: the most recent
  change('znew');
  const kept = Object.keys(JSON.parse(saved.get(KEY)).contrast);
  assert.equal(kept.length, CONTRAST_SAMPLES);
  assert.deepEqual([kept.includes('z1'), kept.at(-2), kept.at(-1)], [false, 'z0', 'znew'], 'the least recently changed is forgotten');
});
