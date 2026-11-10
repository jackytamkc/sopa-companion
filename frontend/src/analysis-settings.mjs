// The UMAP pane's Advanced settings (analysis.js): how the cells are grouped into clusters and laid out on the map. Apply
// (or Enter) sends them and Reset sends the defaults; the engine starts a new generation of the run's results with them,
// and new runs start with them too (engine/analysis.py configure). Apply is never greyed out: a press with a field that
// is not valid says which and why (explain.mjs) and sends nothing. While the engine works on the new settings both
// buttons are busy, and one status line says what happened when there is something to say. Nothing here moves by
// itself: the dots' flight to the new layout is the UMAP view's (umap.js).
import { guard, busy, clearExplanation, problem } from './explain.mjs';
import { QC_TEXT } from './qc-copy.mjs';

export const CLUSTER_SETTINGS_TEXT = Object.freeze({
  title: 'Advanced settings',
  titleChanged: 'Advanced settings (changed)',
  titleTip: 'How the cells are grouped into clusters and laid out on this map. Apply changes this run, and new runs start with the same settings.',
  resolution: 'Cluster resolution',
  resolutionTip: 'Higher values split the cells into more, smaller clusters; lower values merge them into fewer. Above about 40 clusters, some share a colour.',
  neighbours: 'Neighbours per cell',
  neighboursTip: 'How many of its most similar cells each cell is linked to. Fewer bring out small groups; more give broader groups and a smoother map. It changes both the clusters and the map.',
  spacing: 'Dot spacing',
  spacingTip: 'How close similar cells may sit on the map: lower packs them into tight clumps, higher spreads them out. The clusters stay the same.',
  fewest: 'Fewest transcripts per cell',
  fewestTip: "Cells with fewer transcripts in proseg's count are left out of the clusters and the map. Raise it to leave out small fragments.",
  genes: 'Genes used',
  genesTip: "How many of the panel's most variable genes the clusters and the map are computed from; a panel with fewer genes uses all of them. It changes both the clusters and the map.",
  defaultTip: 'Default: {value}.',
  apply: 'Apply',
  applying: 'Applying…',
  applyTip: 'Group the cells again with these settings (Enter). New runs start with them too.',
  reset: 'Reset',
  resetTip: 'Back to the defaults, for this run and new runs.',
  unchanged: 'These settings are already in use.',
  number: 'Use a number from {min} to {max}.',
  whole: 'Use a whole number from {min} to {max}.',
  failed: 'The clusters could not be computed with these settings. The Log button (top right) saves the details.',
  refused: 'These settings could not be applied. Try again.',
  slow: 'Still working on the new settings: the map changes when they are ready.',
  tooFew: 'Too few cells have enough transcripts with these settings. Lower Fewest transcripts per cell, or press Reset.',
});
const T = CLUSTER_SETTINGS_TEXT;

// Whether the disclosure is open, remembered per browser (in a notebook's tab, the tab's own storage: storage-boot.mjs).
export const OPEN_KEY = 'sopa-companion:umap-settings-open';
// After this long with no result of the new settings on screen, the buttons are free again and the status line says
// the map changes when they are ready: the result still applies when it comes.
export const SLOW_MS = 120000;

/** The settings in the engine's order (engine/analysis.py SETTINGS): its key, the field's words and id, and the word
 *  the page's log uses. A setting whose default this kind of run keeps from before proseg's count (the run's
 *  constants.kept: "min_counts" on a whole sample, engine/qc_constants.py KEPT) says so in its tooltip
 *  (notRemeasuredShort, after its own words). */
export const FIELDS = Object.freeze([
  { key: 'resolution', label: T.resolution, tip: T.resolutionTip, word: 'resolution', id: 'umap-resolution' },
  { key: 'n_neighbors', label: T.neighbours, tip: T.neighboursTip, word: 'neighbours', id: 'umap-neighbours' },
  { key: 'min_dist', label: T.spacing, tip: T.spacingTip, word: 'spacing', id: 'umap-spacing' },
  { key: 'min_counts', label: T.fewest, tip: T.fewestTip, word: 'fewest transcripts', id: 'umap-fewest' },
  { key: 'n_top_genes', label: T.genes, tip: T.genesTip, word: 'genes used', id: 'umap-genes' },
].map(field => Object.freeze(field)));

const fill = (template, values) => template.replace(/\{(\w+)\}/g, (_, name) => values[name]);
/** A value as the page shows it: at most 6 significant digits, no trailing zeros ("1", "0.3", "15"). */
export const shown = value => String(Number(Number(value).toPrecision(6)));
/** A value as a field holds it: as shown() when that is the same number, else with every digit it has (a setting that
 *  [analysis] or a tool gave with more than six), so a field never reads as an edit of the value it was filled with. */
export const fieldText = value => { const text = shown(value); return Number(text) === Number(value) ? text : String(Number(value)); };
const near = (a, b) => Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(b));

/** What a press of Apply finds wrong in the fields' `values` (their text, by engine key) against the engine's `limits`
 *  ({key: {min, max, step, whole}}): one problem per field that is empty, not a number, a fraction where a whole number
 *  goes, or out of range (explain.mjs). A field at its default (`defaults`) goes whatever the limit, as the engine
 *  takes it: the engine started with it, and a worker whose limit is lower (genes used on the CPU) holds to its own. */
export function settingsProblems(values, limits, defaults = null) {
  const problems = [];
  for (const field of FIELDS) {
    const limit = limits?.[field.key];
    if (!limit) continue;
    const text = String(values?.[field.key] ?? '').trim(), n = text === '' ? NaN : Number(text);
    if (Number.isFinite(n) && defaults?.[field.key] != null && near(n, Number(defaults[field.key]))) continue;
    if (Number.isFinite(n) && (!limit.whole || Number.isInteger(n)) && n >= limit.min && n <= limit.max) continue;
    problems.push(problem(field.id, field.label, fill(limit.whole ? T.whole : T.number, { min: shown(limit.min), max: shown(limit.max) })));
  }
  return problems;
}

/** Whether two sets of settings agree on every setting either holds (within 1e-9 of its value). */
export function sameSettings(a, b) {
  return FIELDS.every(({ key }) => {
    if (a?.[key] == null && b?.[key] == null) return true;
    const x = Number(a?.[key]), y = Number(b?.[key]);
    return Number.isFinite(x) && Number.isFinite(y) && near(x, y);
  });
}

/** The settings in the log's words ("resolution 1.5, neighbours 30, ... (changed)"), '' when they are the defaults. */
export function settingsLine(settings, defaults) {
  if (!settings || !defaults || sameSettings(settings, defaults)) return '';
  return FIELDS.filter(field => settings[field.key] != null).map(field => `${field.word} ${shown(settings[field.key])}`).join(', ') + ' (changed)';
}

/** The disclosure: `root` goes under the pane's head. `load(meta)` takes the engine's describe (settings, defaults, limits,
 *  changed); until it has them the disclosure stays hidden. `onApply(values)` sends the settings (analysis.js
 *  applySettings), which then says how it went: done(), failed(), refused(), tooFew() or unchanged(); slow() comes by
 *  itself after SLOW_MS, and `onSlow()` hears of it (the page's log). */
export class ClusterSettings {
  constructor({ document: doc = globalThis.document, onApply = () => {}, onSlow = () => {}, storage = null, timers = globalThis } = {}) {
    Object.assign(this, { doc, onApply, onSlow, storage, timers });
    this.meta = null; this.visible = false; this.ending = null; this.timer = null; this.kind = ''; this.kept = [];
    const make = (tag, className) => { const node = doc.createElement(tag); if (className) node.className = className; return node; };
    this.root = make('details', 'advanced umap-settings'); this.root.hidden = true;
    const summary = make('summary');
    this.titleText = make('span'); this.titleText.textContent = T.title; summary.title = T.titleTip; summary.append(this.titleText);
    // The browser shows no validation bubbles of its own: a press says what a field needs (explain.mjs).
    this.form = make('form'); this.form.noValidate = true; this.form.setAttribute('novalidate', '');
    this.inputs = new Map(); this.labels = new Map();
    for (const field of FIELDS) {
      const label = make('label', 'umap-setting'), name = make('span'), input = make('input');
      name.textContent = field.label; label.title = field.tip;
      input.type = 'number'; input.id = field.id; input.setAttribute('inputmode', 'decimal'); input.title = field.tip;
      input.addEventListener('input', () => this.status(''));        // an edit clears the status line
      label.append(name, input); this.form.append(label);
      this.inputs.set(field.key, input); this.labels.set(field.key, label);
    }
    // The data attributes are the measuring harness's handles, not words.
    this.apply = make('button', 'primary'); this.apply.type = 'button'; this.apply.textContent = T.apply; this.apply.title = T.applyTip;
    this.apply.setAttribute('data-umap-settings', 'apply');
    this.resetButton = make('button'); this.resetButton.type = 'button'; this.resetButton.textContent = T.reset; this.resetButton.title = T.resetTip;
    this.resetButton.setAttribute('data-umap-settings', 'reset');
    const buttons = make('div', 'umap-settings-buttons'); buttons.append(this.apply, this.resetButton);
    this.note = make('div', 'missing-note'); this.note.hidden = true; this.note.setAttribute('role', 'alert');
    this.line = make('p', 'muted small umap-settings-status'); this.line.hidden = true; this.line.setAttribute('role', 'status');
    this.form.append(buttons, this.note, this.line);
    this.root.append(summary, this.form);
    const options = { note: this.note, document: doc };
    // A press while the last one is still being worked on gets explain's busy line (guard's own check, either button).
    this.press = guard(this.apply, () => this.problems(), () => this.send(), options);
    // Reset always sends, so new runs follow the defaults again even when this run already has them (the fields hold
    // them exactly: fieldText).
    guard(this.resetButton, () => [], () => { clearExplanation(this.apply, options); this.fill(this.meta?.defaults); return this.send(); }, options);
    // Enter in a field is Apply; Enter on a button is that button's own press. The buttons are type="button", so the
    // form never submits by itself.
    const fields = new Set(this.inputs.values());
    this.form.addEventListener('keydown', event => { if (event.key !== 'Enter' || !fields.has(event.target)) return; event.preventDefault(); this.press(event); });
    this.form.addEventListener('submit', event => event.preventDefault());
    this.root.addEventListener('toggle', () => this.remember());
    this.root.open = this.recalled();
  }
  get loaded() { return this.meta != null; }
  get changed() { return !!this.meta?.changed; }
  get isBusy() { return this.ending != null; }
  /** The fields' values as numbers, by engine key. */
  get values() { return Object.fromEntries(FIELDS.map(field => [field.key, Number(String(this.inputs.get(field.key).value ?? '').trim())])); }
  /** The fields hold edits not applied yet (also one that is not a number). */
  get dirty() {
    if (!this.meta) return false;
    return FIELDS.some(field => {
      const held = this.meta.settings[field.key], text = String(this.inputs.get(field.key).value ?? '').trim(), n = Number(text);
      return held != null && (text === '' || !Number.isFinite(n) || !near(n, held));
    });
  }
  problems() { return settingsProblems(Object.fromEntries(FIELDS.map(field => [field.key, this.inputs.get(field.key).value])), this.meta?.limits, this.meta?.defaults); }
  fill(values) {
    for (const field of FIELDS) if (values?.[field.key] != null) this.inputs.get(field.key).value = fieldText(values[field.key]);
  }
  /** The engine's word on the run's settings (its describe, a POST's reply, another window's change): the fields follow
   *  unless they hold edits not applied yet. Without settings, defaults and limits nothing changes. */
  load(meta) {
    if (!meta?.settings || !meta?.defaults || !meta?.limits) return;
    const keep = this.dirty;
    this.meta = { settings: { ...meta.settings }, defaults: { ...meta.defaults }, limits: meta.limits, changed: !!meta.changed };
    for (const field of FIELDS) {
      const input = this.inputs.get(field.key), limit = meta.limits[field.key];
      if (limit) for (const name of ['min', 'max', 'step']) if (limit[name] != null) input.setAttribute(name, String(limit[name]));
    }
    this.tips();
    if (!keep) this.fill(this.meta.settings);
    const title = this.meta.changed ? T.titleChanged : T.title;
    if (this.titleText.textContent !== title) this.titleText.textContent = title;
    this.place();
  }
  /** The fields' tooltips: each field's words, then (a default this kind of run keeps from before proseg's count) that it
   *  is not re-measured on it yet, then its default once the engine has said it. */
  tips() {
    for (const field of FIELDS) {
      const value = this.meta?.defaults?.[field.key];
      const tip = [field.tip, this.kept.includes(field.key) ? QC_TEXT.notRemeasuredShort : '', value == null ? '' : fill(T.defaultTip, { value: shown(value) })]
        .filter(Boolean).join(' ');
      this.labels.get(field.key).title = this.inputs.get(field.key).title = tip;
    }
  }
  /** The constants the run's kind keeps from before proseg's count (run.constants.kept; none: a region's run). */
  setKept(kept) { this.kept = Array.isArray(kept) ? kept.filter(k => typeof k === 'string') : []; this.tips(); }
  /** Another run (or none): nothing of this one's is kept, and the disclosure stays hidden until the next load. */
  reset() {
    this.endBusy(); this.status('');
    for (const button of [this.apply, this.resetButton]) clearExplanation(button, { note: this.note, document: this.doc });
    this.meta = null;
    for (const input of this.inputs.values()) input.value = '';
    this.titleText.textContent = T.title;
    this.place();
  }
  /** Whether the pane wants the disclosure (shown while the pane is open, once the settings have loaded). */
  show(on) { this.visible = !!on; this.place(); }
  place() { const hide = !(this.visible && this.meta); if (this.root.hidden !== hide) this.root.hidden = hide; }
  open(on) { this.root.open = !!on; }
  send() {
    const values = this.values;
    this.status(''); this.endBusy();
    const free = [busy(this.apply, T.applying), busy(this.resetButton)];
    this.ending = () => { for (const end of free) end(); };
    this.timer = this.timers.setTimeout(() => { this.timer = null; this.slow(); this.onSlow(); }, SLOW_MS);
    return this.onApply(values);
  }
  endBusy() {
    if (this.timer != null) this.timers.clearTimeout(this.timer);
    this.timer = null;
    const end = this.ending; this.ending = null; end?.();
  }
  /** The status line: one of failed, refused, slow, tooFew or unchanged ('' clears it). */
  status(kind) {
    this.kind = kind || '';
    const text = kind ? T[kind] : '';
    if (this.line.textContent !== text) this.line.textContent = text;
    if (this.line.hidden !== !text) this.line.hidden = !text;
  }
  done() { this.endBusy(); }
  failed() { this.endBusy(); this.status('failed'); }
  refused() { this.endBusy(); this.status('refused'); }
  slow() { this.endBusy(); this.status('slow'); }
  unchanged() { this.endBusy(); this.status('unchanged'); }
  /** Too few cells hold enough transcripts with these settings: the disclosure opens and says so. */
  tooFew() { this.endBusy(); this.status('tooFew'); this.open(true); }
  /** A result of the current settings is on screen: what it answers goes from the status line. */
  resultShown() { if (['failed', 'slow', 'tooFew'].includes(this.kind)) this.status(''); }
  /** The settings for the page's log, '' at the defaults. */
  logLine() { return this.meta ? settingsLine(this.meta.settings, this.meta.defaults) : ''; }
  recalled() { try { return this.storage?.getItem(OPEN_KEY) === '1'; } catch { return false; } }
  remember() { try { this.storage?.setItem(OPEN_KEY, this.root.open ? '1' : '0'); } catch { /* no storage: this page alone */ } }
}
