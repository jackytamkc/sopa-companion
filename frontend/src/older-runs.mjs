// Runs from before the one-run app: at the first start of this version (and at
// each start until none is left) the app lists the runs older versions made and never saved to their samples, and asks
// for each: Save to the sample or Discard; Save all and Discard all from two rows on; Later (or Esc) keeps the rest on
// disk, untouched, until the next start, and never holds up a new run. Saving one never makes it the sample's
// segmentation. The engine does the work, one row at a time (GET /runs/older, POST /runs/older/{id} and POST
// /runs/older: {runs: [row], done: [rows answered for good, with their stage and message], count}); Later is POST
// /runs/older {answer: "later"}: the engine sets the rows aside for this session (`later`: the next start lists them
// again) and stops a Save all after the row under way. The rows follow the engine's answers, its {ev: "older", ...}
// pushes and a poll while the prompt is open or a save is under way. A row whose save
// ended says so a moment, then glides out (folds away in a few paints drawn on demand, as Compare fades; at once with
// reduced motion); the last one gone, the prompt closes. Its words are here (the
// banned-words tests read OLDER_TEXT); a row the engine cannot save comes with its own words (`reason`, `detail`).
import { runLine } from './shell.mjs';

export const OLDER_TEXT = Object.freeze({
  title: 'Save your older runs?', titleOne: 'Save your older run?',
  lead: 'These were made before this version and never saved to their samples. The app now keeps no runs between sessions.',
  save: 'Save to the sample',
  saveTitle: "Writes its cells into {sample} under new names, then deletes the run. Nothing is replaced, and it does not become the sample's segmentation.",
  discard: 'Discard', discardTitle: 'Deletes the run. Anything you saved from it before stays in the sample.',
  later: 'Later', laterTitle: 'Keeps the runs you have not answered on disk and asks again the next time the app starts.',
  saveAll: 'Save all', discardAll: 'Discard all', retry: 'Try again',
  waiting: 'Waiting…', checking: 'Looking for it in the sample…', joining: 'Joining the finished tiles…', writing: 'Saving to the sample…',
  discarding: 'Discarding…', saved: 'Saved as {names}', savedAlready: 'Saved already as {names}', failed: "Couldn't save: {reason}",
  stillSaving: 'Still being saved by the last session',
  moved: '{sample} is no longer where it was.', movedTitle: 'It was at {path}. Put it back and start the app again to save this run.',
  xenium: 'Made on a Xenium folder, which cannot hold its cells.',
  xeniumTitle: 'Its results are in {folder}: copy them before you discard it.',
  patches: 'Its patch boxes were never exported.', patchesTitle: 'Its boxes are in {folder}: copy them before you discard it.',
  missing: 'Its results are missing from its folder.', filtered: 'Its filters leave no cells, so there is nothing to save.',
  tooOld: 'Made by a much older version: it cannot be saved.', cannot: 'It cannot be saved.',
  wholeSample: 'whole sample', proseg: 'Proseg', patchesTool: 'Training patches',
  noticeSaved: 'Saved the older run on {sample} as {names}.', noticeSavedPlain: 'Saved the older run on {sample}.',
  noticeFailed: "Couldn't save the older run on {sample}: {reason}. It is listed again next time.",
});
const fill = (text, values) => text.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? '');
const fmt = new Intl.NumberFormat('en-GB'), fmt1 = new Intl.NumberFormat('en-GB', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
export const POLL_MS = 1500;       // the rows asked for again while the prompt is open or a save is under way
export const SAVED_MS = 1400;      // a row's "Saved as ..." stays this long before it goes ...
export const GLIDE_MS = 300;       // ... gliding out over this long ...
export const GLIDE_FPS = 12;       // ... in at most this many paints a second (none with reduced motion)
const STAGES = Object.freeze({ waiting: 'waiting', checking: 'checking', joining: 'joining', writing: 'writing', removing: 'discarding' });

// ---- pure helpers (node tests) -----------------------------------------------------------------------------------------
/** Every row of an answer ({runs: [...]}, or the list itself), those set aside by Later among them. */
const listed = value => (Array.isArray(value) ? value : Array.isArray(value?.runs) ? value.runs : [])
  .filter(row => row && typeof row.id === 'string' && row.id);
/** The rows of an answer still to answer: a row set aside (`later`) waits for the next start. */
export function olderRows(value) {
  return listed(value).filter(row => !row.later);
}
/** The rows the engine answered for good this session ({done: [{id, stage: saved | saved_already | discarded, message,
 *  zarr_save}]}), by id. */
export function olderDone(value) {
  const list = Array.isArray(value?.done) ? value.done : [];
  return new Map(list.filter(row => row && typeof row.id === 'string' && row.id).map(row => [row.id, row]));
}
/** How many older runs health.older names: its count, a number, or the rows themselves. */
export function olderCount(value) {
  if (Number.isFinite(value)) return Math.max(0, value);
  if (Number.isFinite(value?.count)) return Math.max(0, value.count);
  return olderRows(value).length;
}
/** The run's region in a few words: whole sample, box W × H mm (µm under a millimetre), lasso N mm² (µm²). */
export function regionWords(region) {
  if (!region || region.kind === 'full') return OLDER_TEXT.wholeSample;
  if (region.kind === 'box' && Array.isArray(region.box) && region.box.length === 4) {
    const w = Math.abs(region.box[2] - region.box[0]), h = Math.abs(region.box[3] - region.box[1]);
    return w >= 1000 && h >= 1000 ? `box ${fmt1.format(w / 1000)} × ${fmt1.format(h / 1000)} mm` : `box ${fmt.format(Math.round(w))} × ${fmt.format(Math.round(h))} µm`;
  }
  if (region.kind === 'polygon' && Array.isArray(region.points) && region.points.length >= 3) {
    let twice = 0;
    region.points.forEach((p, i) => { const q = region.points[(i + 1) % region.points.length]; twice += p[0] * q[1] - q[0] * p[1]; });
    const area = Math.abs(twice) / 2;
    return area >= 1e6 ? `lasso ${fmt1.format(area / 1e6)} mm²` : `lasso ${fmt.format(Math.round(area))} µm²`;
  }
  return '';
}
/** "14 Oct 15:42" in this computer's time, '' without a readable time. */
export function whenWords(value) {
  const date = typeof value === 'number' ? new Date(value < 1e12 ? value * 1000 : value) : typeof value === 'string' ? new Date(value) : null;
  if (!date || !Number.isFinite(date.getTime())) return '';
  const two = n => String(n).padStart(2, '0');
  return `${date.getDate()} ${date.toLocaleString('en-GB', { month: 'short' })} ${two(date.getHours())}:${two(date.getMinutes())}`;
}
const sampleOf = row => row.sample_name || row.sample || 'the sample';
const methodOf = row => row.stain?.method || (['cellpose', 'instanseg'].includes(row.method) ? row.method : null);
/** One row's line: "sample-1 · whole sample · Proseg · Finished at iteration 1,200 · 14 Oct 15:42". */
export function rowLine(row) {
  const stain = row.kind === 'stain', run = { ...row, stain: { ...(row.stain || {}), method: methodOf(row) } };
  const parts = [sampleOf(row), regionWords(row.region), row.kind === 'patches' ? OLDER_TEXT.patchesTool : stain ? '' : OLDER_TEXT.proseg,
    row.kind === 'patches' ? '' : runLine(run), whenWords(row.created ?? row.when)];
  return parts.filter(Boolean).join(' · ');
}
const priorName = spec => spec?.kind === 'shapes' ? spec.key : spec?.column ?? spec?.key ?? null;
/** The row's tooltip: proseg's starting cells and the settings that differ from proseg's defaults ("Started from
 *  cellpose_boundaries_1 · Iterations 2,000"), or the stain model and channels ("cpsam · DAPI + ATP1A1/CD45/E-Cadherin").
 *  `label(flag)` and `defaults(flag)`: the option schema's label and default for a proseg flag. */
export function rowTooltip(row, { label = flag => flag, defaults = () => undefined } = {}) {
  if (row.kind === 'stain' || row.kind === 'patches') {
    const stain = row.stain || {}, channels = Array.isArray(stain.channels) && stain.channels.length ? stain.channels.join(' + ')
      : [stain.nucleus, stain.cytoplasm].filter(Boolean).join(' + ');
    return [stain.model || null, channels || null].filter(Boolean).join(' · ');
  }
  const cells = priorName(row.prior?.cells), nuclei = priorName(row.prior?.nuclei);
  const start = cells && nuclei ? `Started from ${cells} cells and ${nuclei} nuclei` : cells || nuclei ? `Started from ${cells || nuclei}` : '';
  const settings = Object.entries(row.options || {}).filter(([flag, value]) => value !== null && value !== undefined && value !== defaults(flag))
    .map(([flag, value]) => `${label(flag) || flag} ${typeof value === 'number' ? fmt.format(value) : value === true ? 'on' : value === false ? 'off' : value}`);
  return [start, ...settings].filter(Boolean).join(' · ');
}
/** Why a row cannot be saved, as {text, title} (amber, Discard only), or null when it can be: the engine's own words
 *  (`reason`, `detail`), else its stable reason (`why`) in the page's. */
export function rowRefusal(row) {
  if (row.can_add !== false || row.still_saving) return null;
  if (typeof row.reason === 'string' && row.reason.trim()) return { text: row.reason, title: typeof row.detail === 'string' ? row.detail : '' };
  const why = typeof row.why === 'string' ? row.why : '', sample = sampleOf(row);
  const folder = row.folder || 'its folder', path = row.sample_path || 'its old place';
  const moved = { text: fill(OLDER_TEXT.moved, { sample }), title: fill(OLDER_TEXT.movedTitle, { path }) };
  const xenium = { text: OLDER_TEXT.xenium, title: fill(OLDER_TEXT.xeniumTitle, { folder }) };
  const known = { sample_gone: moved, moved, sample_missing: moved, xenium, sample: xenium,
    patches: { text: OLDER_TEXT.patches, title: fill(OLDER_TEXT.patchesTitle, { folder }) },
    missing: { text: OLDER_TEXT.missing, title: '' }, results_missing: { text: OLDER_TEXT.missing, title: '' },
    filtered: { text: OLDER_TEXT.filtered, title: '' }, too_old: { text: OLDER_TEXT.tooOld, title: '' } };
  if (known[why]) return known[why];
  if (/\s/.test(why)) return { text: why, title: '' };        // the engine's own plain words
  return { text: OLDER_TEXT.cannot, title: '' };
}
const names = save => [save?.shapes_key, save?.table_key].filter(Boolean).join(' + ');
const savedWords = (save, already = false) => names(save) ? fill(already ? OLDER_TEXT.savedAlready : OLDER_TEXT.saved, { names: names(save) })
  : (already ? OLDER_TEXT.savedAlready : OLDER_TEXT.saved).replace(/ as \{names\}$/, '');
/** A row whose answer failed: the engine's stage "failed", or (an engine that names no stage) an add whose save failed. */
const failedRow = row => row.stage === 'failed' || row.answer === 'add' && !row.stage && row.zarr_save?.state === 'failed';
/** A row whose answer the engine is working on (queued or carried out); a failed one waits for another answer. */
export const underWay = row => (row.answer === 'add' || row.answer === 'discard') && !failedRow(row);
/** What the row says now: still being saved by the last session, a failed answer, its stage while under way; else a
 *  save that failed before; '' while it waits for an answer. */
export function rowStatus(row) {
  const save = row.zarr_save || {};
  if (row.still_saving) return OLDER_TEXT.stillSaving;
  if (failedRow(row)) return row.stage === 'failed' && row.message ? row.message : fill(OLDER_TEXT.failed, { reason: save.message || row.message || '' });
  if (underWay(row)) return row.message || OLDER_TEXT[STAGES[row.stage] || (row.answer === 'discard' ? 'discarding' : 'waiting')];
  if (save.state === 'failed') return fill(OLDER_TEXT.failed, { reason: save.message || '' });
  return '';
}
/** The words for a row answered for good (the engine's done row): its message, else ours. */
const doneWords = (done, row) => done?.message || savedWords(done?.zarr_save || row?.zarr_save, done?.stage === 'saved_already');
/** The names a save ended with, for the notice after Later: its zarr_save's, else from the engine's "Saved as X" line. */
const doneNames = done => names(done?.zarr_save) || (typeof done?.message === 'string' ? done.message.match(/ as (.+)$/)?.[1] ?? '' : '');
const reasonOf = message => String(message || '').replace(/^Couldn't save: /, '');
const stamp = row => JSON.stringify([row?.zarr_save ?? null, row?.stage ?? null, row?.answer ?? null]);

// ---- the prompt -------------------------------------------------------------------------------------------------------
/**
 * new OlderRunsPrompt({fetchRows, answer, answerAll, notice, label, defaults, reducedMotion, clock}, doc): fetchRows() ->
 * GET /runs/older; answer(id, 'add' | 'discard') -> POST /runs/older/{id} (the row, or the rows); answerAll('add' |
 * 'discard' | 'later') -> POST /runs/older (each row the engine can, one at a time; the rows); notice(text) the page's
 * notice; label(flag) and defaults(flag) the option schema's, for the rows' tooltips; reducedMotion() true: a saved row
 * goes without its glide. ask(older, {lead}) shows the prompt and resolves once it is done (Later, Esc, or the last row
 * gone): the page opens nothing before.
 */
export class OlderRunsPrompt {
  constructor(options = {}, doc = globalThis.document) {
    this.o = options; this.doc = doc;
    this.clock = options.clock ?? { setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms), clearTimeout: id => globalThis.clearTimeout(id) };
    // rows: shown, by id; sent: answers this page sent the engine has not taken up yet ({answer, from}); following:
    // after Later, the rows whose save goes on ({from, last}), for the notice at its end; done: the engine's rows
    // answered for good this session; leaving: saved rows still saying so or gliding out (the prompt waits for them).
    this.rows = new Map(); this.nodes = new Map(); this.sent = new Map(); this.following = new Map(); this.done = new Map();
    this.leaving = new Set();
    this.resolve = null; this.asking = null; this.pollTimer = null; this.polling = false;
    const d = this.dialog = doc.createElement('dialog');
    d.className = 'modal older-popup'; d.id = 'older-popup'; d.setAttribute('aria-labelledby', 'older-title');
    d.innerHTML = `<h2 id="older-title"></h2><p id="older-lead"></p><p id="older-notice" class="older-notice small" hidden></p>
      <ul id="older-runs" class="older-runs"></ul>
      <div class="modal-actions"><button id="older-later" type="button" data-older-answer="later"></button>
      <span class="spacer"></span><button id="older-discard-all" type="button" class="danger" data-older-answer="all-discard"></button>
      <button id="older-save-all" type="button" class="primary" data-older-answer="all-add"></button></div>`;
    doc.body.append(d); this.$ = id => doc.getElementById(id);
    this.$('older-lead').textContent = OLDER_TEXT.lead;
    this.$('older-later').textContent = OLDER_TEXT.later; this.$('older-later').title = OLDER_TEXT.laterTitle;
    this.$('older-save-all').textContent = OLDER_TEXT.saveAll; this.$('older-discard-all').textContent = OLDER_TEXT.discardAll;
    this.$('older-later').onclick = () => this.putOff();
    this.$('older-save-all').onclick = () => this.all('add');
    this.$('older-discard-all').onclick = () => this.all('discard');
    d.addEventListener('cancel', e => { e.preventDefault?.(); this.putOff(); });   // Esc is Later
  }
  get open() { return !!this.dialog.open; }
  /** Shows the rows of `older` (the start notice's `lead` lines under the lead) and resolves when the prompt is done. */
  ask(older, { lead = '' } = {}) {
    if (this.open && this.asking) { this.apply(older); return this.asking; }
    if (!olderRows(older).length) return Promise.resolve();
    const notice = this.$('older-notice'); notice.textContent = lead || ''; notice.hidden = !lead;
    this.asking = new Promise(resolve => { this.resolve = resolve; });
    this.dialog.showModal();
    this.apply(older); this.focusFirst(); this.schedule();
    return this.asking;
  }
  focusFirst() {
    for (const n of this.nodes.values()) if (!n.actions.hidden && !n.save.hidden) { n.save.focus?.(); return; }
    this.$('older-later').focus?.();
  }
  /** The engine's rows as they are now (an answer's reply, a push or a poll): {runs, done} or the list itself. */
  apply(older) {
    for (const [id, row] of olderDone(older)) this.done.set(id, row);
    const all = listed(older);
    if (this.open) this.setRows(all);
    if (this.following.size) this.follow(all);
  }
  /** New rows listed, changed ones updated, those gone finished (a saved one says so a moment, then goes); those set
   *  aside by Later (in this window or another) leave quietly. */
  setRows(all) {
    const fresh = new Map(all.filter(row => !row.later).map(row => [row.id, row]));
    const aside = new Map(all.filter(row => row.later).map(row => [row.id, row]));
    for (const id of [...this.rows.keys()]) if (!fresh.has(id)) { if (aside.has(id)) this.setAside(aside.get(id)); else this.gone(id); }
    for (const row of fresh.values()) this.put(row);
    this.render();
    this.settle();
  }
  /** The prompt is done once no row is left to answer and the last saved one has said so and gone. */
  settle() { if (this.open && !this.rows.size && !this.leaving.size) this.finish(); }
  /** A row set aside until the next start: it leaves the list; a save under way goes on, and its end is a notice. */
  setAside(row) {
    const shown = this.rows.get(row.id) || row;
    if (shown.answer === 'add' && underWay(shown)) this.following.set(row.id, { from: this.sent.get(row.id)?.from ?? stamp(row), last: row });
    this.rows.delete(row.id); this.sent.delete(row.id);
    this.nodes.get(row.id)?.li.remove(); this.nodes.delete(row.id);
  }
  /** One row as the engine has it; an answer this page sent shows until the engine takes it up. */
  put(row) {
    if (!row?.id || !this.open) return;
    const sent = this.sent.get(row.id);
    if (sent && (underWay(row) || stamp(row) !== sent.from)) this.sent.delete(row.id);
    const shown = this.sent.has(row.id) ? { ...row, answer: sent.answer, stage: null, message: null } : row;
    this.rows.set(row.id, shown);
    if (!this.nodes.has(row.id)) this.build(shown);
    this.update(shown);
  }
  build(row) {
    const el = (tag, className = '', text = '') => { const n = this.doc.createElement(tag); if (className) n.className = className; if (text) n.textContent = text; return n; };
    const li = el('li', 'older-row'); li.dataset.olderRun = row.id; li.setAttribute('data-older-run', row.id);
    const line = el('p', 'older-line'), why = el('p', 'older-why save-warning small'), status = el('p', 'older-status muted small');
    status.setAttribute('role', 'status');
    const actions = el('div', 'older-actions');
    const save = el('button', 'primary', OLDER_TEXT.save); save.type = 'button'; save.dataset.olderAnswer = 'add'; save.setAttribute('data-older-answer', 'add');
    const discard = el('button', 'danger', OLDER_TEXT.discard); discard.type = 'button'; discard.dataset.olderAnswer = 'discard';
    discard.setAttribute('data-older-answer', 'discard'); discard.title = OLDER_TEXT.discardTitle;
    save.onclick = () => this.choose(row.id, 'add'); discard.onclick = () => this.choose(row.id, 'discard');
    actions.append(save, discard); li.append(line, why, status, actions);
    this.$('older-runs').append(li);
    this.nodes.set(row.id, { li, line, why, status, actions, save, discard });
  }
  update(row) {
    const n = this.nodes.get(row.id); if (!n) return;
    n.line.textContent = rowLine(row); n.line.title = rowTooltip(row, { label: this.o.label, defaults: this.o.defaults });
    const refusal = rowRefusal(row), status = rowStatus(row);
    n.why.textContent = refusal?.text || ''; n.why.title = refusal?.title || ''; n.why.hidden = !refusal;
    n.status.textContent = status; n.status.hidden = !status;
    n.actions.hidden = !!row.still_saving || underWay(row);
    n.save.hidden = !!refusal;
    const retry = failedRow(row) && row.answer !== 'discard' || !row.answer && row.zarr_save?.state === 'failed';
    n.save.textContent = retry ? OLDER_TEXT.retry : OLDER_TEXT.save;
    n.save.title = fill(OLDER_TEXT.saveTitle, { sample: sampleOf(row) });
  }
  /** The title (a saved row still saying so counts); Save all (while one can be saved) and Discard all, from two rows on. */
  render() {
    this.$('older-title').textContent = this.rows.size + this.leaving.size === 1 ? OLDER_TEXT.titleOne : OLDER_TEXT.title;
    const free = [...this.rows.values()].filter(row => !row.still_saving && !underWay(row));
    this.$('older-save-all').hidden = this.rows.size < 2 || !free.some(row => !rowRefusal(row));
    this.$('older-discard-all').hidden = this.rows.size < 2 || !free.length;
  }
  /** A row's answer: sent once (its buttons go while the engine works on it); a refusal stays on the row. */
  async choose(id, answer) {
    const row = this.rows.get(id);
    if (!row || row.still_saving || underWay(row)) return;
    if (answer === 'add' && rowRefusal(row)) return;
    this.mark(id, answer); this.render(); this.schedule();
    try { this.take(await this.o.answer?.(id, answer)); }
    catch (error) { this.refused(id, error); }
  }
  mark(id, answer) {
    const row = this.rows.get(id);
    this.sent.set(id, { answer, from: stamp(row) });
    const shown = { ...row, answer, stage: null, message: null };
    this.rows.set(id, shown); this.update(shown);
  }
  /** Save all / Discard all: one request, the engine taking the rows it can one at a time (Save all skips those that
   *  cannot be saved). An engine without it is answered row by row. */
  async all(answer) {
    const ids = [...this.rows.values()].filter(row => !row.still_saving && !underWay(row) && (answer !== 'add' || !rowRefusal(row))).map(row => row.id);
    if (!ids.length) return;
    const before = new Map(ids.map(id => [id, this.rows.get(id)]));
    for (const id of ids) this.mark(id, answer);
    this.render(); this.schedule();
    try { this.take(await this.o.answerAll?.(answer)); }
    catch (error) {
      if (![404, 405].includes(error?.status)) { for (const id of ids) this.refused(id, error); return; }
      for (const id of ids) {
        if (!this.rows.has(id)) continue;
        this.sent.delete(id); this.rows.set(id, before.get(id));
        await this.choose(id, answer);
      }
    }
  }
  /** An answer's reply: the rows ({runs, done} or a list), one row, or nothing (the poll brings the rest). */
  take(reply) {
    if (Array.isArray(reply?.runs) || Array.isArray(reply)) this.apply(reply);
    else if (reply?.id && this.rows.has(reply.id)) { this.put(reply); this.render(); }
  }
  refused(id, error) {
    this.sent.delete(id);
    // Answered elsewhere (404): the row goes, saying "Saved" only when the engine's done rows say so (another window may
    // have discarded it).
    if (error?.status === 404) { this.gone(id, { guess: false }); this.render(); this.settle(); return; }
    const row = this.rows.get(id); if (!row) return;
    const failed = { ...row, stage: 'failed', message: fill(OLDER_TEXT.failed, { reason: error?.message || String(error) }) };
    this.rows.set(id, failed); this.update(failed); this.render();
  }
  /** A row the engine no longer lists: answered for good (the engine's done row says how; without one, `guess`: our own
   *  Save); a saved one says so a moment (SAVED_MS), then goes, the prompt waiting for it when it was the last. */
  gone(id, { guess = true } = {}) {
    const row = this.rows.get(id), n = this.nodes.get(id), done = this.done.get(id);
    this.rows.delete(id); this.sent.delete(id); this.nodes.delete(id);
    if (!n) return;
    const saved = done ? done.stage !== 'discarded' : guess && row?.answer === 'add';
    if (!saved) { n.li.remove(); return; }
    n.status.textContent = doneWords(done, row); n.status.hidden = false; n.actions.hidden = true; n.why.hidden = true;
    n.li.classList.add('done'); this.leaving.add(n.li);
    this.clock.setTimeout(() => this.glideOut(n.li), SAVED_MS);
  }
  /** A saved row leaves the list for good. */
  drop(li) { li.remove(); this.leaving.delete(li); if (this.open) this.render(); this.settle(); }
  /** A saved row goes: it folds away and fades in a few paints (GLIDE_MS, at most GLIDE_FPS a second), then leaves the
   *  list; with reduced motion, or the prompt closed, at once. Nothing moves once it has gone. */
  glideOut(li) {
    let still = false; try { still = !!this.o.reducedMotion?.(); } catch { still = false; }
    const steps = Math.max(1, Math.ceil(GLIDE_MS * GLIDE_FPS / 1000));
    if (still || !this.open || !li.parentElement) { this.drop(li); return; }
    let style = null; try { style = globalThis.getComputedStyle?.(li) ?? null; } catch { style = null; }
    const height = li.offsetHeight || 0, top = parseFloat(style?.paddingTop) || 0, bottom = parseFloat(style?.paddingBottom) || 0;
    Object.assign(li.style, { overflow: 'hidden', boxSizing: 'border-box' });
    const paint = k => {
      if (k >= steps || !this.open || !li.parentElement) { this.drop(li); return; }
      const left = 1 - (k + 1) / steps;
      Object.assign(li.style, { opacity: String(left), maxHeight: `${Math.round(height * left)}px`,
        paddingTop: `${top * left}px`, paddingBottom: `${bottom * left}px` });
      this.clock.setTimeout(() => paint(k + 1), Math.ceil(1000 / GLIDE_FPS));
    };
    paint(0);
  }
  /** Later (or Esc): the prompt goes at once; the engine sets the rows aside for this session (a reload does not ask
   *  again; the next start does) and stops a Save all after the row under way, which goes on: its end is a notice. */
  putOff() {
    for (const row of this.rows.values()) {
      if (row.answer === 'add' && underWay(row)) this.following.set(row.id, { from: this.sent.get(row.id)?.from ?? stamp(row), last: row });
    }
    this.finish();
    Promise.resolve().then(() => this.o.answerAll?.('later'))
      .then(reply => { if (Array.isArray(reply?.runs)) this.apply(reply); }, () => { /* an engine without it: nothing set aside */ });
  }
  finish() {
    if (this.open) this.dialog.close();
    for (const n of this.nodes.values()) n.li.remove();
    for (const li of this.leaving) li.remove();
    this.nodes.clear(); this.rows.clear(); this.sent.clear(); this.leaving.clear();
    const resolve = this.resolve; this.resolve = null; this.asking = null; resolve?.();
    if (this.following.size) this.schedule(); else this.stopPolling();
  }
  /** After Later: a save under way that ended (the row gone: saved; it failed: listed again next time). A row Later set
   *  aside before its save began waits for the next start, with nothing to say. */
  follow(all) {
    const fresh = new Map(all.map(row => [row.id, row]));
    for (const [id, f] of [...this.following]) {
      const row = fresh.get(id), sample = sampleOf(f.last), failed = !!row && failedRow(row) && stamp(row) !== f.from;
      if (row && !failed && (underWay(row) || !row.later)) { f.last = row; continue; }   // under way, or its answer on its way
      this.following.delete(id);
      if (failed) { this.o.notice?.(fill(OLDER_TEXT.noticeFailed, { sample, reason: reasonOf(row.stage === 'failed' ? row.message : row.zarr_save?.message) })); continue; }
      if (row) continue;
      const done = this.done.get(id);
      if (done?.stage === 'discarded') continue;
      const saved = doneNames(done) || names(f.last.zarr_save);
      this.o.notice?.(saved ? fill(OLDER_TEXT.noticeSaved, { sample, names: saved }) : fill(OLDER_TEXT.noticeSavedPlain, { sample }));
    }
    if (!this.following.size && !this.open) this.stopPolling();
  }
  /** A push {ev: "older", runs, done, ...}: the same rows in every window. */
  event(message) { if (message?.ev === 'older') this.apply(message); }
  schedule() {
    if (this.pollTimer != null || typeof this.o.fetchRows !== 'function') return;
    this.pollTimer = this.clock.setTimeout(() => { this.pollTimer = null; this.poll(); }, POLL_MS);
  }
  async poll() {
    if (this.polling || (!this.open && !this.following.size)) return;
    this.polling = true;
    try { this.apply(await this.o.fetchRows()); }
    catch { /* asked again at the next turn */ }
    finally { this.polling = false; }
    if (this.open || this.following.size) this.schedule();
  }
  stopPolling() { if (this.pollTimer != null) { this.clock.clearTimeout(this.pollTimer); this.pollTimer = null; } }
  /** What the harness reads: {open, runs: [id], answered: {id: answer}}. */
  snapshot() {
    return { open: this.open, runs: [...this.rows.keys()],
      answered: Object.fromEntries([...this.rows.values()].filter(underWay).map(row => [row.id, row.answer])) };
  }
}
