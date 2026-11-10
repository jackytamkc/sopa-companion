import { decodeVectors } from './vectors-decode.mjs';
import { releaseTiles } from './prior-paths.mjs';
import { releaseIndex } from './polygons.mjs';
import { resolutionName } from './stain.mjs';

// The sample's segmentation layers (Layers > Load Other Mask): any cell-like shapes
// element of the open sample, listed by GET /api/v1/samples/{id}/segmentations ({segmentations: [{key, name, source,
// model, count, extent, table, version, outlines}]}: asked for when the sample has layers remembered, each time Add a
// mask… opens and after a save into the sample ends: sampleSaved) and drawn on the tissue by PriorLayers'
// SegRaster, apart from the run's own vectors. A layer's outlines are fetched the first time it is shown (GET
// .../segmentations/{key}/outlines?v=: the prior vectors binary with its cells only, served immutable at that version,
// so the browser keeps it) and decoded once (vectors-decode.mjs: typed-array views, no parse); an element written again
// since (a list naming another `version`) has them fetched anew; nothing is fetched or drawn while it is hidden, and at
// most MAX_SHOWN show at once (a fifth explains instead). The layers listed for each sample, and each layer's style per
// sample and key (kept after it is removed), are remembered in localStorage (STORAGE_KEY) for the KEEP_SAMPLES samples
// used last; storage that cannot be read or written forgets.
//
// One sample at a time (setSample: main.js, as a run or the picker opens one). The layers of the sample left last are
// kept aside, so going back to it (the picker, then a run of the same sample) fetches and decodes nothing again (its
// list read again only when a save went into it meanwhile).
export const STORAGE_KEY = 'sopa-companion:segmentations:v1';
export const MAX_SHOWN = 4;
export const KEEP_SAMPLES = 40;
// A new layer's colour: the first of these no other layer of the sample has. Light on the dark stage, apart from the
// run's own layers (Prior shapes #8ecae6, Prior nuclei #ffd18a, Cells #df9ebc, the proseg prior #b6a2d5) and from the
// mixing check's pinks.
export const SEG_COLORS = Object.freeze(['#7be29a', '#ffb454', '#c7a6ff', '#5ad1e6', '#f2e86d', '#ff8f7a']);
export const SEG_STYLE = Object.freeze({ on: true, opacity: .8, width: 1, fill: false });
/** Every word Layers > Load Other Mask shows (layer-panel.js; the names are the engine's): the sample's other
 *  segmentations, each called a mask here (Compare's own words for them are its own: compare-copy.mjs). */
export const SEG_TEXT = Object.freeze({
  group: 'Load Other Mask',
  add: 'Add a mask…',
  reading: "Reading the sample's masks…",
  readFailed: "Could not read the sample's masks.",
  none: 'This sample has no other masks.',
  allAdded: 'Every mask of this sample is in the list.',
  loading: 'Loading…',
  failed: 'Could not load it',
  retry: 'Try again',
  shapes: 'shapes',
  four: `At most ${MAX_SHOWN} masks show at once. Hide one to show this one.`,
  addedHidden: `Added, hidden: at most ${MAX_SHOWN} masks show at once. Hide one to show it.`,
  colour: 'Colour', opacity: 'Opacity', width: 'Line width', showAs: 'Show as', outlines: 'Outlines', filled: 'Filled',
  remove: 'Remove',
  removeTitle: 'Take it off the list. Its colour and style are remembered for this sample.',
});
const COLOR = /^#[0-9a-f]{6}$/i;
const enc = encodeURIComponent;
/** Let a set's path tiles and grid go now (they are built again if it is ever drawn again). */
const release = set => { if (set) { releaseTiles(set); releaseIndex(set); } };
const own = (object, key) => !!object && typeof object === 'object' && Object.hasOwn(object, key) ? object[key] : undefined;
/** A name remembered from an earlier version, which named the level a save segmented "1/4": by the level's name, as the
 *  engine names it now ("Cellpose · 1/4" -> "Cellpose · scale2"), until the sample's list gives the name. A save of the
 *  dormant InstanSeg is named as the engine names it now too: a saved stain segmentation. */
const renamed = name => name.replace(/ · (1\/\d+)$/, (_, label) => ` · ${resolutionName({ label })}`)
  .replace(/^InstanSeg(?= · |$)/, 'saved stain segmentation');

/** A remembered style as a layer takes it: only the fields of the right kind, within their ranges. */
export function savedStyle(value) {
  const o = value && typeof value === 'object' && !Array.isArray(value) ? value : {}, style = {};
  if (typeof o.on === 'boolean') style.on = o.on;
  if (typeof o.color === 'string' && COLOR.test(o.color)) style.color = o.color.toLowerCase();
  if (Number.isFinite(o.opacity)) style.opacity = Math.min(1, Math.max(0, o.opacity));
  if (Number.isFinite(o.width)) style.width = Math.min(5, Math.max(.5, o.width));
  if (typeof o.fill === 'boolean') style.fill = o.fill;
  return style;
}

/** The store. A layer: {key, name, item (its entry in the sample's list, once read), style {on, color, opacity, width,
 *  fill}, state 'idle' | 'loading' | 'ready' | 'failed', set (its PolygonSet once loaded), error}. Listeners are told of
 *  every change with `drawn`: whether what is drawn may have changed (PriorLayers marks the stage dirty only then). */
export class SegLayers {
  /** `fetch`: the page's (tests: a stand-in); `storage`: localStorage by default, null for none; `now`: the clock the
   *  memory orders samples by. */
  constructor({ fetch = (...args) => globalThis.fetch(...args), storage, now = () => Date.now() } = {}) {
    this.fetch = fetch; this.storageGiven = storage; this.now = now;
    this.sample = null; this.layers = []; this.catalogue = null; this.note = null; this.parked = null; this.token = 0;
    this.listing = null;   // the read of the sample's list on its way ({again}: asked for anew meanwhile)
    this.listeners = new Set(); this.made = new WeakSet();
  }
  listen(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  changed(drawn) { for (const fn of [...this.listeners]) fn(!!drawn); }
  /** Whether `value` is one of this store's layers (listed now or not). */
  isLayer(value) { return !!value && typeof value === 'object' && this.made.has(value); }
  find(key) { return this.layers.find(l => l.key === key) ?? null; }
  /** How many shapes a layer holds: its outlines' once loaded, else as the sample's list says (null: not known). */
  count(layer) { return layer.set?.count ?? (Number.isFinite(layer.item?.count) ? layer.item.count : null); }
  shownCount() { return this.layers.filter(l => l.style.on).length; }
  /** The layers to draw now, front to back: shown and loaded, with shapes (never more than MAX_SHOWN). */
  drawable() { return this.layers.filter(l => l.style.on && l.state === 'ready' && l.set?.count > 0).slice(0, MAX_SHOWN); }
  /** The sample's segmentations not listed yet, as its list gives them (none before it is read). */
  addable() { return (this.catalogue?.items ?? []).filter(item => !this.find(item.key)); }
  colorOf(key) { return this.find(key)?.style.color ?? null; }

  /** The open sample (an id; null for none): its remembered layers listed (their outlines fetched if shown), the last
   *  sample's put aside. Returns whether it changed. */
  setSample(id) {
    id = id == null || id === '' ? null : String(id);
    if (id === this.sample) return false;
    this.park();
    this.token++;
    this.sample = id; this.layers = []; this.catalogue = null; this.note = null; this.listing = null;
    const parked = id && this.parked?.sample === id ? this.parked : null;
    if (parked) { this.layers = parked.layers; this.catalogue = parked.catalogue; this.parked = null; }
    else {
      const entry = id ? this.entry(id) : null, keys = Array.isArray(entry?.layers) ? entry.layers : [];
      let shown = 0;
      for (const key of keys) {
        if (typeof key !== 'string' || !key || this.find(key)) continue;
        const layer = this.make(key, own(entry.styles, key));
        if (layer.style.on && ++shown > MAX_SHOWN) layer.style.on = false;
        this.layers.push(layer);
      }
    }
    if (this.layers.length) this.loadCatalogue();   // their names, and whether they are still in the sample (once)
    for (const layer of this.layers) this.ensure(layer);
    this.changed(true);
    return true;
  }
  /** Put the current sample's layers aside, as they are (their outlines kept, their loads stopped, their paths released)
   *  with its list (unless a new read of it was on its way: then read again when it comes back); with none (no sample,
   *  or no layer), what is aside stays: the picker between two runs of a sample. */
  park() {
    if (!this.sample || !this.layers.length) return;
    for (const layer of this.layers) {
      if (layer.abort) { layer.abort.abort(); Object.assign(layer, { abort: null, state: 'idle' }); }
      release(layer.set);
    }
    this.parked = { sample: this.sample, layers: this.layers, catalogue: this.catalogue?.state === 'ready' && !this.listing ? this.catalogue : null };
  }
  make(key, saved = null, item = null) {
    const used = new Set(this.layers.map(l => l.style.color));
    const color = SEG_COLORS.find(c => !used.has(c)) ?? SEG_COLORS[this.layers.length % SEG_COLORS.length];
    const name = typeof item?.name === 'string' && item.name ? item.name : typeof saved?.name === 'string' && saved.name ? renamed(saved.name) : key;
    const layer = { key, name, item, style: { ...SEG_STYLE, color, ...savedStyle(saved) }, state: 'idle', set: null, error: null, abort: null };
    this.made.add(layer);
    return layer;
  }

  /** Read the sample's list: once (again after a failure), or anew with `fresh` (Add a mask… opened, a save into
   *  the sample ended); one read at a time (asked for anew meanwhile: once more after it). A list already read stays on
   *  show while it is read again (a failure then keeps it). A list that no longer holds a layer's element takes the layer
   *  off (its style stays remembered); one naming another version of a layer's element (written again since its outlines
   *  were read) has them fetched anew; a sample the engine lists nothing for (404) has none to add. */
  loadCatalogue(fresh = false) {
    const sample = this.sample, token = this.token;
    if (!sample) return;
    if (this.listing) { if (fresh) this.listing.again = true; return; }
    if (this.catalogue?.state === 'ready' && !fresh) return;
    const listing = this.listing = { again: false }, current = () => token === this.token && this.listing === listing;
    if (this.catalogue?.state !== 'ready') { this.catalogue = { state: 'loading', items: [], error: null }; this.changed(false); }
    Promise.resolve().then(() => this.fetch(`/api/v1/samples/${enc(sample)}/segmentations`)).then(async r => {
      if (r.status === 404) return null;
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const body = await r.json();
      return Array.isArray(body?.segmentations) ? body.segmentations.filter(item => item && typeof item.key === 'string' && item.key) : [];
    }).then(items => {
      if (!current()) return;
      this.listing = null;
      this.catalogue = { state: 'ready', items: items ?? [], error: null };
      let drawn = false;
      if (items) {
        const known = new Map(items.map(item => [item.key, item]));
        for (const layer of [...this.layers]) {
          const item = known.get(layer.key), was = layer.item?.version;
          if (!item) { drawn = this.drawable().includes(layer) || drawn; this.drop(layer); continue; }
          layer.item = item;
          if (typeof item.name === 'string' && item.name) layer.name = item.name;
          // Written again since (another version of its element): its outlines go and come anew from the new version's
          // URL (now when shown, else when shown next). A version first learnt here leaves outlines read before as they are.
          if (typeof was === 'string' && typeof item.version === 'string' && was !== item.version && (layer.set || layer.state !== 'idle')) {
            drawn = this.drawable().includes(layer) || drawn;
            this.unload(layer); this.ensure(layer);
          }
        }
        this.remember();
      }
      this.changed(drawn);
      if (listing.again) this.loadCatalogue(true);
    }).catch(error => {
      if (!current()) return;
      this.listing = null;
      if (this.catalogue?.state !== 'ready') this.catalogue = { state: 'failed', items: [], error: String(error?.message || error) };
      this.changed(false);
      if (listing.again) this.loadCatalogue(true);
    });
  }
  /** A save into sample `id` ended (main.js: the app's own stain or proseg save into the zarr): its list is read anew (a
   *  new element to add; one written again: its outlines fetched anew), now when it is the open sample with its list read
   *  or layers listed; put aside, its list goes (read again when it comes back). */
  sampleSaved(id) {
    id = id == null || id === '' ? null : String(id);
    if (!id) return;
    if (id === this.sample) { if (this.catalogue || this.layers.length) this.loadCatalogue(true); }
    else if (this.parked?.sample === id) this.parked.catalogue = null;
  }
  /** Fetch a shown layer's outlines once: never while it is hidden, loading, loaded or failed (retry). */
  ensure(layer) {
    if (!this.sample || !layer.style.on || layer.set || layer.state === 'loading' || layer.state === 'failed') return;
    const token = this.token, abort = typeof AbortController === 'function' ? new AbortController() : null;
    Object.assign(layer, { state: 'loading', error: null, abort });
    // The list's own URL once it is read: it names the element's version (?v=), so the browser keeps the outlines for
    // good and a layer written again comes anew; before, the plain one (the engine then has the browser check it).
    const listed = layer.item?.outlines, base = `/api/v1/samples/${enc(this.sample)}/segmentations/${enc(layer.key)}/outlines`;
    const url = typeof listed === 'string' && listed.startsWith(`${base}?`) ? listed : base;
    const current = () => token === this.token && this.layers.includes(layer) && layer.abort === abort;
    Promise.resolve().then(() => this.fetch(url, abort ? { signal: abort.signal } : undefined)).then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.arrayBuffer();
    }).then(buffer => {
      if (!current()) return;
      const set = decodeVectors(buffer).cells;
      if (!set) throw new Error('no outlines in the reply');
      Object.assign(layer, { set, state: 'ready', abort: null });
      this.changed(true);
    }).catch(error => {
      if (!current()) return;
      Object.assign(layer, { state: 'failed', error: String(error?.message || error), abort: null });
      this.changed(false);
    });
  }
  /** Fetch a failed layer's outlines again. */
  retry(layer) {
    if (!this.layers.includes(layer) || layer.state !== 'failed') return;
    Object.assign(layer, { state: 'idle', error: null });
    this.ensure(layer);
    this.changed(false);
  }

  /** List the sample's element `key` (shown, unless MAX_SHOWN show already: then hidden, and the note says why), its
   *  style as remembered for this sample. Returns the layer, or null (no sample, or listed already). */
  add(key) {
    if (!this.sample || typeof key !== 'string' || !key || this.find(key)) return null;
    const item = this.catalogue?.items.find(i => i.key === key) ?? null;
    const layer = this.make(key, own(this.entry(this.sample)?.styles, key), item), full = this.shownCount() >= MAX_SHOWN;
    layer.style.on = !full;
    this.layers.push(layer);
    this.note = full ? SEG_TEXT.addedHidden : null;
    this.ensure(layer); this.remember();
    this.changed(layer.state === 'ready');
    return layer;
  }
  /** Take a layer off the list (its outlines go; its style stays remembered). */
  remove(layer) {
    if (!this.layers.includes(layer)) return false;
    const drawn = this.drawable().includes(layer);
    this.drop(layer); this.note = null; this.remember();
    this.changed(drawn);
    return true;
  }
  drop(layer) {
    this.unload(layer);
    this.layers.splice(this.layers.indexOf(layer), 1);
  }
  /** Let a layer's outlines go (a fetch on its way stopped): fetched again when it is shown (ensure). */
  unload(layer) {
    layer.abort?.abort();
    release(layer.set);
    Object.assign(layer, { set: null, state: 'idle', error: null, abort: null });
  }
  /** Show or hide a layer: {ok, why}. A fifth shown is refused with why (the note). Hidden, its fetch stops and its
   *  paths go (its outlines are kept: shown again, nothing is fetched). */
  show(layer, on) {
    if (!this.layers.includes(layer)) return { ok: false, why: null };
    on = !!on;
    if (on === layer.style.on) return { ok: true, why: null };
    if (on && this.shownCount() >= MAX_SHOWN) {
      this.note = SEG_TEXT.four; this.changed(false);
      return { ok: false, why: SEG_TEXT.four };
    }
    const drawn = this.drawable().includes(layer);
    layer.style.on = on; this.note = null;
    if (on) {
      if (layer.state === 'failed') Object.assign(layer, { state: 'idle', error: null });
      this.ensure(layer);
    } else {
      if (layer.abort) { layer.abort.abort(); Object.assign(layer, { abort: null, state: 'idle' }); }
      release(layer.set);
    }
    this.remember();
    this.changed(drawn || this.drawable().includes(layer));
    return { ok: true, why: null };
  }
  /** Change a layer's colour, opacity, width or fill (shown or hidden: show()). */
  setStyle(layer, patch) {
    if (!this.layers.includes(layer)) return;
    const { on, ...style } = savedStyle(patch);
    Object.assign(layer.style, style);
    this.remember();
    this.changed(this.drawable().includes(layer));
  }

  /** What the store holds now, for the harness (PriorLayers.segmentationsShown adds what is drawn). */
  snapshot() {
    return { sample: this.sample, note: this.note, loaded: this.layers.map(({ key, name, style, state }) => ({ key, name,
      visible: style.on, color: style.color, opacity: style.opacity, width: style.width, fill: style.fill, state })) };
  }

  storage() {
    if (this.storageGiven !== undefined) return this.storageGiven;
    try { return globalThis.localStorage ?? null; } catch { return null; }
  }
  memory() {
    try {
      const value = JSON.parse(this.storage()?.getItem(STORAGE_KEY) || 'null');
      if (value && typeof value === 'object' && value.samples && typeof value.samples === 'object' && !Array.isArray(value.samples)) return value;
    } catch { /* unreadable: forgotten */ }
    return { samples: {} };
  }
  /** What is remembered for sample `id`: {at, layers: [key], styles: {key: {on, color, opacity, width, fill, name}}}. */
  entry(id) { const value = own(this.memory().samples, id); return value && typeof value === 'object' ? value : null; }
  /** Remember the sample's list and its layers' styles (the sample first; the KEEP_SAMPLES used last are kept). */
  remember() {
    if (!this.sample) return;
    const memory = this.memory(), before = own(memory.samples, this.sample), styles = {};
    for (const [key, style] of Object.entries(before?.styles && typeof before.styles === 'object' ? before.styles : {})) styles[key] = style;
    for (const l of this.layers) styles[l.key] = { ...l.style, name: l.name };
    const others = Object.entries(memory.samples).filter(([id, value]) => id !== this.sample && value && typeof value === 'object')
      .sort((a, b) => (b[1].at || 0) - (a[1].at || 0)).slice(0, KEEP_SAMPLES - 1);
    memory.samples = Object.fromEntries([[this.sample, { at: this.now(), layers: this.layers.map(l => l.key), styles }], ...others]);
    try { this.storage()?.setItem(STORAGE_KEY, JSON.stringify(memory)); } catch { /* optional */ }
  }
}
