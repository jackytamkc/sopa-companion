import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { domFixture } from './dom-fixture.mjs';
import { COLORMAPS, colourOf, isHexColour, makeLUT } from '../src/image-layers.mjs';

// An image channel takes a colour of its own, as segmentation layers do (HEX or RGB in the colour picker): black up to
// that colour (one way of giving it, hex codes too).
let source = await readFile(new URL('../src/layer-panel.js', import.meta.url), 'utf8');
for (const file of ['image-layers.mjs', 'qc-copy.mjs', 'stain.mjs', 'seg-layers.mjs']) source = source.replace(`'./${file}'`, JSON.stringify(new URL(`../src/${file}`, import.meta.url).href));
const { LayerPanel } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const KEY = 'sopa-companion:layers:v1';
const ORIGINAL = new Map(['document', 'localStorage'].map(k => [k, Object.getOwnPropertyDescriptor(globalThis, k)]));

function panel(t) {
  const dom = domFixture(), saved = new Map();
  dom.Element.prototype.prepend = function (...nodes) { for (const n of nodes.reverse()) { n.parentElement = this; this.childNodes.unshift(n); } };
  dom.Element.prototype.getContext = () => new Proxy({}, { get: (o, k) => (k in o ? o[k] : () => {}), set: (o, k, v) => { o[k] = v; return true; } });   // the contrast histogram's canvas
  dom.document.body.innerHTML = (globalThis.__indexHtml ??= '');
  const globals = { document: dom.document, localStorage: { getItem: k => saved.get(k) ?? null, setItem: (k, v) => saved.set(k, v) } };
  for (const [k, v] of Object.entries(globals)) Object.defineProperty(globalThis, k, { value: v, writable: true, configurable: true });
  t.after(() => { for (const [k, d] of ORIGINAL) if (d) Object.defineProperty(globalThis, k, d); else delete globalThis[k]; });
  const stage = { options: { cellColorBy: 'identity', transcripts: true, trails: true, fill: .4, transcriptSize: 2, transcriptAlpha: .7, transcriptColorBy: 'cell', mode: 0 },
    set(o) { Object.assign(this.options, o); }, dirty: false };
  const priors = { settings: { prior: { on: false, opacity: 1 }, nuclei: { on: false, opacity: 1, color: '#fff', width: 1 }, xenium: { on: false, opacity: 1, color: '#fff', width: 1 } } };
  return { layers: new LayerPanel(stage, priors), stage, saved };
}

test('a colour of its own is a colormap: black up to it, as the named single colours', () => {
  assert.ok(isHexColour('#ff8800') && isHexColour('#FF8800'));
  assert.ok(!isHexColour('gray') && !isHexColour('#f80') && !isHexColour('#ff880') && !isHexColour(undefined));
  const lut = makeLUT('#ff8800');
  assert.deepEqual([...lut.slice(0, 3)], [0, 0, 0]);
  assert.deepEqual([...lut.slice(765)], [255, 136, 0]);
  assert.deepEqual([...makeLUT('#ff0000')], [...makeLUT('red')], 'the same ramp as the named colour');
  assert.equal(colourOf('#FF8800'), '#ff8800');
  assert.equal(colourOf('green'), '#00ff00'); assert.equal(colourOf('gray'), '#ffffff');
  assert.equal(colourOf('viridis'), '#fde725', 'a many-colour map: its brightest colour');
  assert.ok(COLORMAPS.every(m => isHexColour(colourOf(m))));
});

test('a channel offers the colour picker next to its colormap, and keeps the colour by its name', async t => {
  globalThis.__indexHtml = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const { layers, saved } = panel(t);
  const dapi = { name: 'DAPI', on: true, colormap: 'gray', gamma: 1, opacity: 1, blending: 'additive', low: 0, high: 255, max: 255, depth: 8, source: {} };
  layers.setChannels([dapi]); layers.selected = dapi; layers.render();
  const find = () => layers.controls.all().find(n => n.tagName === 'INPUT' && n.type === 'color');
  let colour = find();
  assert.ok(colour, 'a colour picker for the channel');
  assert.equal(colour.value, '#ffffff', 'gray starts at white');
  colour.value = '#ff8800'; colour.oninput();
  assert.equal(dapi.colormap, '#ff8800');
  colour.onchange();
  const list = layers.controls.all().find(n => n.tagName === 'SELECT');
  assert.ok(list.children.some(o => o.value === '#ff8800'), 'the Colormap list names the colour of its own');
  assert.equal(find().value, '#ff8800');
  layers.save();
  assert.equal(JSON.parse(saved.get(KEY)).channels.DAPI.colormap, '#ff8800', 'kept by name, as the other looks');
  list.value = 'green'; list.onchange();
  assert.equal(dapi.colormap, 'green', 'a named colormap again');
  assert.equal(find().value, '#00ff00');
});
