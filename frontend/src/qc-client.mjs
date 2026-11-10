// The mixing check on the page: what the engine says about the open run's check (GET /runs/{id}/qc, its series, the
// WebSocket's {ev:"qc"} events) and which result the page shows for the frame on screen. main.js hands that result to
// the panel, the stage chip, the timeline lane, the stage's Mixing colours (stage.setMixing) and the overlay's per-bin
// layer (qc-overlay.mjs). The view keeps the result's flagged areas as the engine gives them (the suggestion cards'
// numbers, "Look at its cells"); none is drawn or listed, and the view keeps nothing that only an area list read (the
// spots left out of it, how many it listed, whether a result's areas were known, from when).
//
// Results come only for frames with proseg's own count (counted frames). A result is shown with the snapshot of its own
// frame or of a later one, never with an earlier one (the newest result at or before the frame on screen): a later
// result never colours an earlier frame, except looking back at a frame before the first result, which shows the
// oldest after it (as the engine answers a frame asked for: the nearest with proseg's count). The view says when the
// result shown is another frame's while looking back (`snapped`: its iteration). While following a live run a
// result that arrives is applied in the paint batch of the next snapshot (main.js marks later.qc() in show()), so the
// panel, the stage chip, the timeline lane, the cell colours and the outlines change in the frame that shows the cells;
// a run that stops sending snapshots gets it after WITH_SNAPSHOT_MS. Runs of COALESCE_CELLS cells or more apply at most
// one new result per COALESCE_MS. Paused, stepped or scrubbed frames get theirs at once (a frame never checked is asked
// for, as analysis does). With Mixing colours on, a result is applied only once its per-cell values have arrived, so the
// colours and the number change together and the cells never flash grey between two results: until then the result
// shown stays (or, when that one is a later frame's, the newest values loaded at or before the frame on screen colour the
// cells); the values of each new result are asked for as it arrives, so a fast run never waits on them. Nothing here
// polls or animates by itself: a timer exists only while something is due (a retry after a failed request included);
// `view.motion` says whether a change may move (the panel's number and line), `view.still` whether the change is a
// late result for the frame already on screen (it lands without motion).
//
// Frame 0's own result is never shown or drawn into the line; frame 0 shows the start line (proseg's count at iteration
// 0 of the cells it started from), and the view's areas are the start line's own. Only a run that started from cells
// alone has one (the engine's start_kind "cells"): one that started from nuclei, alone or with their cells, has none.
// Until cells have formed (burn-in + 20 iterations, the engine's area_view "start") the frames keep the start line's
// areas too.
//
// Suggestions: the engine's suggestion for the result shown (its summary's `suggestion`,
// already kept steady by the engine's card keeper, with the other matches as `more_ideas`), the trial of a suggested
// change (registered with POST /qc/trial after the knob was set through the Live knobs session, ended with Keep or
// Undo; its rows, band and verdict come with the results and its changes as {state: "trial"} pushes), the answer to an
// "Undo your last change?" card (POST /qc/trial/end with rule U). The page never judges a trial itself.
//
// Every per-cell value shown is known by the result it came from (`view.cellsSeq`, `cellsIter`,
// `cellsCurrent`). The colours may show an earlier result's values for a moment (so the cells never go grey), but a cell
// readout (the tooltip, the card) never shows an earlier result's value as the result shown's: `cellValue` withholds it
// until the result's own values are in, which it asks for; `cellReading` says which result a value is from. While a
// readout is in use (asked for within READOUT_MS) each new result's values are asked for as it arrives and a result
// waits for them, as with Mixing colours on, so the readout changes with the number instead of blinking. The numbers
// behind the panel's words (coverage, transcript retention, the like-for-like change and its thresholds, the model's
// provenance, which result the cell values are from) go into the Log button's text ('Mixing detail'), not on screen.
//
// Which count a result reads is its own (`count`, the engine's: "sampled" on an iteration counted for a pause, the
// hold, a finish or the end, scored on the count proseg saves there; "averaged" otherwise). A frame counted again (the
// hold, a finish or the end after a token's count of the same frame) is another result of the same seq: the per-cell
// values held of the other count go, and every request for values names the count (api.js qcCells). While the run
// rests (paused with proseg's ack, finishing, done, stopped or failed) the newest result held shows its own iteration's
// numbers (the engine's `own`: its own number, whole number and change, single, never held), not the smoothed ones
// (H2); its stored summary is unchanged, so the smoothed result shows again once the run plays.
import { knobChange, cardKey, QC_TEXT, coverageOf, retentionOf } from './qc-copy.mjs';
export { cardKey };

export const COALESCE_CELLS = 20000, COALESCE_MS = 1000, WITH_SNAPSHOT_MS = 1000, REQUEST_MS = 150;
export const SERIES_POINTS = 600, LINE_MAX = 1200, KEEP_SUMMARIES = 48, KEEP_CELLS = 8;
// Results whose point is kept however long a run goes on: the line's (decimated), the summaries held, and the newest
// KEEP_POINTS; an older one is read back from its summary when a scrub needs it.
export const KEEP_POINTS = 2000, TRIM_POINTS = 4000;
// A request that failed for another reason than "not checked yet" (the engine restarting, a network blip) is asked
// again after these waits (ms); the run's own answer (GET /runs/{id}/qc) too.
export const RETRY_MS = Object.freeze([2000, 5000, 10000, 30000]);
// Change-driven motion (the UMAP-flight rule family): only for an update at least MOTION_GAP_MS after the
// previous one (whole samples, paused, scrubbing); back-to-back updates (a small region's ~10 a second) land in the
// snapshot's own frame, which already moves at that rate.
export const MOTION_GAP_MS = 900;   // 0.9 s: a whole sample's one result a second qualifies
// A scrub or a step moves the frame on screen: its result (asked for when the frame was never checked) and its values
// that arrive within this long belong to the transition it causes; later ones land without motion, as for a pause.
export const MOVE_WINDOW_MS = 2000;
export const FORMING = new Set(['init', 'burnin', 'burn-in']);
// After Keep or Undo the trial's last words stay for this many iterations of the run.
export const ENDED_NOTE_ITERS = 20;
// A cell readout (the tooltip, the card) counts as in use for this long after it last asked for a cell's value: each new
// result's per-cell values are asked for as it arrives and it waits for them, as with Mixing colours on.
export const READOUT_MS = 10000;
const ENDED = new Set(['done', 'stopped', 'failed']);
const LEARNING = new Set(['waiting', 'learning', 'failed', 'unavailable']);
const finite = Number.isFinite;

/** The count a result reads (its summary's or its point's `count`, absent: "averaged"). */
export const countOf = item => item?.count === 'sampled' ? 'sampled' : 'averaged';
/** Whether a run rests (H2): paused with proseg's ack, finishing, done, stopped or failed. */
export const rests = run => (run?.state === 'paused' && run.ack?.state === 'paused') || ['finishing', ...ENDED].includes(run?.state);
/** A smoothed summary's own iteration's numbers (the engine's `own`), as they replace the smoothed ones while the run
 *  rests: {per100, shown, change, single: true}; null for a summary that is its own already (single: a saved
 *  iteration's, one judged alone) or carries no `own`. */
export function ownNumbers(summary) {
  const own = summary?.single === false && summary.own && typeof summary.own === 'object' ? summary.own : null;
  if (!own || !finite(own.per100)) return null;
  return { per100: own.per100, shown: Number.isInteger(own.shown) ? own.shown : null, change: own.change && typeof own.change === 'object' ? own.change : null,
    single: true };
}

/** Index of the last value <= target in an ascending array (-1: none). */
export function floorIndex(sorted, target) {
  let lo = 0, hi = sorted.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (sorted[mid] <= target) lo = mid + 1; else hi = mid; }
  return lo - 1;
}
/** Index of the last point whose seq is <= target in a list ascending by seq (-1: none). */
export function floorPoint(points, target) {
  let lo = 0, hi = points.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (points[mid].seq <= target) lo = mid + 1; else hi = mid; }
  return lo - 1;
}
/** Insert into an ascending array of distinct numbers; false when it was there already. */
export function insertSorted(sorted, value) {
  const i = floorIndex(sorted, value);
  if (i >= 0 && sorted[i] === value) return false;
  sorted.splice(i + 1, 0, value); return true;
}

/** Where cells stop forming: the iteration of the first frame past burn-in (fixed from then on), else the newest frame's
 *  while they still form. Incremental over a growing frame list (`cache`: the previous answer). */
export function formingUntil(cache, frames) {
  if (!cache || cache.frames !== frames || cache.n > frames.length) cache = { frames, n: 0, until: null, done: false };
  if (!cache.done) {
    for (let i = cache.n; i < frames.length && !cache.done; i++) if (!FORMING.has(frames[i].phase)) { cache.until = frames[i].iter; cache.done = true; }
    if (!cache.done) cache.until = frames.at(-1)?.iter ?? null;
  }
  cache.n = frames.length;
  return cache;
}

/** Plain-named knob changes from the frames' live values (where the timeline ticks), reduced incrementally: `cache` is
 *  the previous answer for a prefix of the same list. ticks: [{seq, iter, flag, key, value}]. */
export function changeTicks(cache, frames) {
  if (!cache || cache.frames !== frames || cache.n > frames.length || (cache.n && frames[cache.n - 1] !== cache.last))
    cache = { frames, n: 0, last: null, live: null, ticks: [] };
  for (let i = cache.n; i < frames.length; i++) {
    const live = frames[i].live;
    if (!live) continue;
    if (cache.live) for (const [flag, value] of Object.entries(live)) {
      const before = cache.live[flag];
      if (before === value || (finite(before) && finite(value) && Math.abs(before - value) <= 1e-6 * Math.max(1, Math.abs(value)))) continue;
      const change = knobChange(flag, before, value);
      if (change) cache.ticks.push({ seq: frames[i].seq ?? i, iter: frames[i].iter, flag, ...change });
    }
    cache.live = live;
  }
  cache.n = frames.length; cache.last = frames.at(-1) ?? null;
  return cache;
}

/** At most `max` points, evenly spaced, always the first and the last (drawing only). */
export function decimate(points, max) {
  if (points.length <= max) return points;
  const out = [], step = (points.length - 1) / (max - 1);
  for (let k = 0; k < max; k++) out.push(points[Math.round(k * step)]);
  return out;
}

/** The panel state (qc-panel.js PANEL_STATES, plus 'loading' before the first answer) from the run's check
 *  (`describe`: GET /runs/{id}/qc), its sample's newest event (`sample`: how learning goes, at most one a second) and
 *  the frame on screen. `results`: the run has results; `live`: following the newest frame (never "scrubbing", also
 *  while the newest frame is still downloading). */
export function panelStatus({ describe, sample, enabled, results, start, run, seq, newest, live }) {
  if (!describe) return 'loading';
  if (enabled === false) return 'off';
  let state = describe.state;
  const runLevel = state === 'unavailable' && describe.sample_state === 'ready';   // e.g. no_transcripts: stays so
  if (sample && !runLevel) {
    if (sample.state === 'ready') { if (!['ready', 'starting'].includes(state)) state = 'starting'; }
    else if (sample.state === 'off' || LEARNING.has(sample.state)) state = sample.state;
  }
  if (state === 'off') return 'waiting';                 // switched on: its sample is queued with it
  if (LEARNING.has(state)) return state;
  // "ready" without a start line: none is coming (a run that started from nuclei only); the frames' results follow.
  if (!results && !start && state !== 'ready') return 'waiting-start';
  if (!live && seq >= 0 && seq < newest) return 'scrubbing';
  if (ENDED.has(run?.state)) return 'finished';
  if (run?.state === 'paused' && run.ack?.state === 'paused') return 'paused';
  return 'live';
}

/** The run's check and the result shown for the frame on screen. main.js owns the DOM: `render(view)` is called in the
 *  paint batch (or at once for the user's own actions) with everything the panel, the chip, the timeline lane, the
 *  stage and the overlay show. */
export class QcClient {
  /** api: api.js's api; batch: the page's PaintBatch; context() -> {run, frames, seq, live, wantCells}; render(view);
   *  notice(text); harness: the hook or null; clock: {now, setTimeout, clearTimeout} (tests); onSwitch(enabled): a
   *  run's switch was flipped (this page's own, once the engine took it, or another window's, pushed), so the engine
   *  starts new runs with it from now on (it remembers the last choice; the check is off until a user turns it on). */
  constructor({ api, batch = null, context, render = () => {}, notice = () => {}, harness = null, clock = null,
    onSwitch = null }) {
    Object.assign(this, { api, batch, context, renderView: render, notice, harness, onSwitch });
    this.clock = clock ?? { now: () => performance.now(), setTimeout: (fn, ms) => setTimeout(fn, ms), clearTimeout: id => clearTimeout(id) };
    this.epoch = 0; this.reset(null);
    // The Log button's text (log-export.mjs pageLog, read when the log is written): the numbers behind the panel's words.
    globalThis.sopaCompanionLog?.provide?.('Mixing detail', () => this.logText());
  }

  // ---- lifecycle ------------------------------------------------------------------------------------------------
  reset(run = null) {
    this.epoch++; this.controller?.abort(); this.controller = null; this.stopTimers();
    this.runId = run?.id ?? null; this.sampleId = run?.sample ?? null; this.tiled = run?.mode === 'tiled';
    this.describe = null; this.sample = null; this.enabled = null; this.start = null; this.startSource = 'xenium'; this.startKind = null;
    this.startAreas = null; this.startAsked = null;
    this.model = null; this.seqs = []; this.points = new Map(); this.summaries = new Map(); this.cells = new Map();
    this.cellsMissing = new Set(); this.startCellsFailed = null;
    this.line = []; this.lineVersion = 0; this.pending = new Set(); this.missing = new Set(); this.askSeq = null; this.asked = null;
    this.applied = null; this.appliedAt = -Infinity; this.restarting = false; this.failed = false; this.switching = false;
    this.switchN = null; this.pendingSwitch = null; this.describeTries = 0; this.retries = new Map();
    this.ticks = null; this.forming = null; this.renderedKey = null; this.renderedAt = -Infinity; this.renderedSeq = null; this.motion = false;
    this.moveDue = false; this.movedAt = -Infinity; this.readoutAt = -Infinity;
    // Per-cell values asked for again once a 404 named their count gone ("seq:count"), and the line with a resting run's
    // own point in it (restLine).
    this.rechecked = new Set(); this.restKey = null; this.restPoints = null;
    this.view = null;
    // The running trial (and the result it was last heard from), the one ended by Keep or Undo (its last words
    // stay a few iterations) and the ids ended here, the cards set aside on this page.
    this.trial = null; this.trialSeq = -1; this.trialEnded = null; this.ended = new Set();
    this.dismissed = new Set();
    if (run) this.refresh();
  }
  stopTimers() {
    for (const k of ['soonTimer', 'applyTimer', 'requestTimer', 'describeTimer']) { if (this[k] != null) this.clock.clearTimeout(this[k]); this[k] = null; }
  }
  /** GET /runs/{id}/qc, and the series once there are results: on open, on reconnect, after the sample learned. */
  async refresh() {
    const epoch = this.epoch, id = this.runId;
    if (!id) return;
    if (this.describeTimer != null) { this.clock.clearTimeout(this.describeTimer); this.describeTimer = null; }
    this.controller?.abort(); const c = this.controller = new AbortController();
    if (!this.startAreas) this.startAsked = null;      // the start line's areas: asked again if still missing
    this.startCellsFailed = null; this.cellsMissing.clear(); this.rechecked.clear();   // per-cell values that failed: asked again too
    this.missing.clear(); this.retries.clear();        // and results that were not checked yet, or failed
    try {
      const describe = await this.api.runQc(id, c.signal);
      if (epoch !== this.epoch || c.signal.aborted) return;
      this.takeDescribe(describe, true); this.failed = false; this.describeTries = 0;
      // The sample's own answer is not asked for here: the outside start number it carries was all this took from it,
      // and nothing on the page shows that any more.
      if (describe.enabled && (describe.seqs?.length || describe.latest_seq != null)) {
        const series = await this.api.qcSeries(id, SERIES_POINTS, c.signal);
        if (epoch !== this.epoch || c.signal.aborted) return;
        if (series.model && this.model && series.model !== this.model) this.dropResults(series.model);
        this.model ??= series.model ?? null;
        if (series.start) this.takeStart(series.start);
        for (const p of series.points || []) this.addPoint(p);
      }
    } catch (error) {
      if (epoch !== this.epoch || error.name === 'AbortError') return;
      // The run's own answer failed (the engine busy or restarting, a network blip): said in plain words, asked again.
      if (!this.describe) this.failed = true;
      const wait = RETRY_MS[Math.min(this.describeTries++, RETRY_MS.length - 1)];
      this.describeTimer = this.clock.setTimeout(() => { this.describeTimer = null; if (epoch === this.epoch) this.refresh(); }, wait);
    }
    this.changed(true);
  }
  /** The run's check from GET /runs/{id}/qc (or a POST's answer). Its switch is taken only when it is at least as new
   *  as the newest the page heard of (switch_n: a push may have brought a newer one); `fresh`: a refresh (open,
   *  reconnect): whether the worker is restarting is then known from the scheduler's own status. */
  takeDescribe(describe, fresh = false) {
    this.describe = describe; this.startSource = describe.start_source || 'xenium';
    this.startKind = typeof describe.start_kind === 'string' ? describe.start_kind : null;
    const n = Number.isInteger(describe.switch_n) ? describe.switch_n : null;
    if (n == null || this.switchN == null || n >= this.switchN) { this.enabled = !!describe.enabled; if (n != null) this.switchN = n; }
    const pending = this.pendingSwitch;
    if (pending && !this.switching && (this.switchN == null || pending.n > this.switchN)) this.applySwitch(pending);
    this.pendingSwitch = this.switching ? this.pendingSwitch : null;
    if (fresh) {
      const scoring = describe.scoring;
      this.restarting = scoring?.state === 'error' && !(Number.isFinite(scoring.seq) && scoring.seq <= (describe.latest_seq ?? -1));
    }
    const model = ['ready', 'starting', 'unavailable'].includes(describe.state) ? describe.model?.id ?? null : null;
    if (model && this.model && model !== this.model) this.dropResults(model);
    if (model) this.model = model;
    if (describe.start) this.takeStart(describe.start);
    this.takeSeqs(describe.seqs);
    // The run's trial as the engine has it now (null: none runs, e.g. another window kept or undid it).
    if ('trial' in describe) this.takeTrial(describe.trial ?? null, Infinity);
  }
  /** A trial as the engine gives it: now (`seq` Infinity: the run's answer, the trial's own answers, an event) or as of
   *  result `seq` (a summary): a summary never ends a trial the page knows of, and an older one never undoes a newer
   *  answer. A trial with an outcome has ended. Counts the verdicts as they come (the harness's qcVerdicts). */
  takeTrial(trial, seq = Infinity) {
    const now = seq === Infinity;
    if (!trial || typeof trial !== 'object') {               // none runs now: kept or undone elsewhere
      if (now && this.trial) { this.trial = null; this.trialSeq = -1; }
      return;
    }
    // Ended here: a trial never runs again, so an earlier copy (a result, or a push sent before the end) never brings it back.
    if (!trial.outcome && trial.id != null && this.ended.has(trial.id)) return;
    if (trial.outcome) {
      if (this.trial?.id === trial.id) this.endTrialHere(trial);
      return;
    }
    const known = this.trial;
    if (known && trial.id !== known.id && !now && !(trial.id > known.id)) return;
    if (known && trial.id === known.id && !now && seq < this.trialSeq) return;
    if (trial.verdict && trial.verdict !== known?.verdict) this.harness?.count('qcVerdicts');
    // A summary carries a compact copy (no asked_iter, no started): what the page knew of the same trial stays.
    this.trial = known && trial.id === known.id ? { ...known, ...trial } : trial;
    if (!now) this.trialSeq = seq;
    if (this.trialEnded?.id === trial.id) this.trialEnded = null;
  }
  /** The trial has ended (Keep, Undo, or elsewhere): its last words stay ENDED_NOTE_ITERS iterations. */
  endTrialHere(ended) {
    this.ended.add(ended.id);
    this.trialEnded = { ...(this.trial ?? {}), ...ended, endedIter: this.context().frames?.at(-1)?.iter ?? null };
    this.trial = null; this.trialSeq = -1;
  }
  /** Results the engine named (a run's answer, a POST's, a "seqs" event): known to exist, fetched when shown. */
  takeSeqs(seqs) {
    let added = false;
    for (const seq of seqs || []) if (Number.isInteger(seq) && seq > 0 && insertSorted(this.seqs, seq)) added = true;   // frame 0: the start line
    return added;
  }
  /** The start line's numbers from any answer (GET /runs/{id}/qc and its series carry them alone; its WebSocket event and
   *  GET /qc/start also its own areas): its areas are kept apart, so an answer without them never drops them. What
   *  only an area list read (more_spots, listed) and the overview's grid are left out, as are the model and the source,
   *  which the run's answer gives. */
  takeStart(start) {
    if (!start || typeof start !== 'object') return;
    const { areas, more_spots: spots, listed, grid, model, start_source: source, ...line } = start;
    this.start = { ...this.start, ...line };
    if (Array.isArray(areas)) this.startAreas = areas;
  }
  /** The run has no start line: it started from nuclei only, or from cells with their nuclei (start_kind "paired"); an
   *  engine that names no start_kind, by where its cells started alone. */
  noStart() { return this.startSource === 'nuclei' || (this.startKind != null && this.startKind !== 'cells'); }
  /** The start line's own areas (GET /runs/{id}/qc/start), once per model when the view is to hold them (frame 0, and
   *  the frames while cells form) and no event brought them (a failure is asked again only after a reconnect:
   *  refresh()). */
  fetchStart() {
    const epoch = this.epoch, model = this.model;
    if (this.startAreas || !this.api.qcStart || !this.runId || !model || this.startAsked === model) return;
    this.startAsked = model;
    this.api.qcStart(this.runId)
      .then(start => { if (epoch === this.epoch && model === this.model && (!start?.model || start.model === model)) { this.takeStart(start); this.changed(!this.context().live); } })
      .catch(() => {});
  }
  /** A relearned model: the old one's results are another yardstick, never mixed into this line. */
  dropResults(model) {
    this.model = model; this.seqs = []; this.points.clear(); this.summaries.clear(); this.cells.clear(); this.cellsMissing.clear();
    this.line = []; this.lineVersion++; this.applied = null; this.start = null; this.startAreas = null; this.startAsked = null;
  }

  // ---- the switch and requests ------------------------------------------------------------------------------------
  /** The panel's one switch. Shown at once; put back (with a notice) if the engine refuses. A switch push heard while
   *  this page waits is kept: newer than the answer (switch_n), it wins once the answer is in. */
  async setEnabled(on) {
    const epoch = this.epoch, id = this.runId, before = this.enabled;
    if (!id || before === on) return;
    this.enabled = on; if (!on) this.applied = null;
    this.switching = true; this.pendingSwitch = null; this.changed(true);
    try {
      const describe = await this.api.setRunQc(id, { enabled: on });
      this.onSwitch?.(on);               // taken: new runs start with it (also when another run is open by now)
      if (epoch !== this.epoch) return;
      this.switching = false;
      const pending = this.pendingSwitch; this.pendingSwitch = null;
      this.takeDescribe(describe);
      const n = Number.isInteger(describe.switch_n) ? describe.switch_n : null;
      if (pending && (n == null ? pending.enabled !== !!describe.enabled : pending.n > n)) { this.applySwitch(pending); return; }
      if (on) this.refresh(); else this.changed(true);
    } catch {
      if (epoch !== this.epoch) return;
      this.switching = false;
      // In plain words: the engine's own message never reaches the page (qc-copy.mjs).
      this.enabled = before; this.changed(true);
      this.notice(QC_TEXT.switchFailed);
      const pending = this.pendingSwitch; this.pendingSwitch = null;
      if (pending) this.takeSwitch(pending.enabled, pending.n);
    } finally { if (epoch === this.epoch) this.switching = false; }
  }
  /** The run's switch as the engine pushes it ({ev:"qc", run, state:"switch", enabled, n}: on the first look after a
   *  watch and whenever it changes, e.g. another window or the API flipped it): shown at once, as the page's own switch
   *  would, when it is newer than what the page has (n against switch_n); switched on, the check's answer and results
   *  are asked for. Heard while this page's own switch or its first answer is on its way, it is kept until then. */
  takeSwitch(enabled, n = null) {
    if (typeof enabled !== 'boolean') return;
    n = Number.isInteger(n) ? n : null;
    if (this.switching || this.describe == null) {
      if (!this.pendingSwitch || n == null || this.pendingSwitch.n == null || n >= this.pendingSwitch.n) this.pendingSwitch = { enabled, n };
      return;
    }
    if (n != null && this.switchN != null && n <= this.switchN) return;   // what the page has already, or older
    this.applySwitch({ enabled, n });
  }
  applySwitch({ enabled, n }) {
    if (n != null) this.switchN = n;
    if (enabled === this.enabled) return;
    this.enabled = enabled;
    this.onSwitch?.(enabled);            // flipped elsewhere: new runs start with it too
    if (!enabled) this.applied = null;
    this.changed(true);
    if (enabled) this.refresh();
  }
  /** Learn the sample again after a failure (the panel's Try again). */
  async retry() {
    const epoch = this.epoch;
    if (!this.sampleId) return;
    try { const sample = await this.api.learnSampleQc(this.sampleId); if (epoch === this.epoch) { this.sample = sample; this.changed(true); } }
    catch { if (epoch === this.epoch) this.notice(QC_TEXT.retryFailed); }
  }
  // ---- trials and cards ----------------------------------------------------------------------
  /** Watch a suggested change (POST /qc/trial {rule, area, flag, from, to}): called once the knob was set through the
   *  Live knobs session (main.js). The engine finds where it starts (the first frame with the new value) and judges it
   *  with the results that follow; the page shows its answer at once. Throws on a refusal (409: one is running). */
  async startTrial(body) {
    const epoch = this.epoch, id = this.runId;
    if (!id) return null;
    const trial = await this.api.qcTrial(id, body);
    if (epoch !== this.epoch) return null;
    this.trialEnded = null;
    this.takeTrial(trial && typeof trial === 'object' ? trial : { ...body, id: null });
    this.harness?.count('qcTrials');
    this.changed(true);
    return this.trial;
  }
  /** Keep or Undo (POST /qc/trial/end {outcome: "kept" | "undone"}) of run `run` (the run the click was on: another
   *  run opened meanwhile is never answered for it): the trial ends; its last words stay a few iterations. A trial the
   *  engine no longer has (404: ended elsewhere) has ended here too. */
  async endTrial(outcome, run = this.runId) {
    const epoch = this.epoch, id = this.runId, trial = this.trial;
    if (!id || id !== run) return null;
    let ended;
    try { ended = await this.api.qcTrialEnd(id, outcome); }
    catch (error) { if (error?.status !== 404) throw error; ended = null; }
    if (epoch !== this.epoch) return null;
    this.endTrialHere({ ...(trial ?? {}), ...(ended && typeof ended === 'object' ? ended : {}), outcome });
    this.changed(true);
    return this.trialEnded;
  }
  /** The answer to an "Undo your last change?" card of run `run` (POST /qc/trial/end {outcome, rule: "U"}): "undone"
   *  once the page put the value back, or "kept"; the engine then stops offering it, and on this page it goes once the
   *  engine took the answer (a 404 means it had gone already). A refusal leaves the card, so the answer can be given
   *  again. */
  async answerUndoCard(suggestion, outcome, run = this.runId) {
    const epoch = this.epoch, id = this.runId;
    if (!id || id !== run) return;
    try { await this.api.qcTrialEnd(id, outcome, 'U'); }
    catch (error) { if (epoch === this.epoch && error?.status !== 404) throw error; }
    if (epoch === this.epoch) this.dismiss(suggestion);
  }
  /** A card set aside on this page: not shown again for this run. */
  dismiss(suggestion) {
    this.dismissed.add(cardKey(suggestion));
    this.changed(true);
  }
  /** A paused, stepped or scrubbed frame that was never checked: ask for it once (debounced 150 ms; the newest asked
   *  wins). Its result comes as an event, like any other. */
  ask(seq) {
    if (seq === this.asked || seq === this.askSeq) return;
    this.askSeq = seq;
    if (this.requestTimer != null) return;
    const epoch = this.epoch;
    this.requestTimer = this.clock.setTimeout(() => {
      this.requestTimer = null;
      if (epoch !== this.epoch || !this.enabled || this.askSeq == null) return;
      const seq = this.askSeq; this.askSeq = null; this.asked = seq;
      // The answer names the results the engine has: one whose push was left out (a busy tick) is then known.
      this.api.setRunQc(this.runId, { seq }).then(describe => {
        if (epoch !== this.epoch || !describe) return;
        if (this.takeSeqs(describe.seqs)) this.changed(!this.context().live);
      }).catch(() => {});
    }, REQUEST_MS);
  }

  // ---- events -------------------------------------------------------------------------------------------------
  /** A WebSocket {ev:"qc"} event: a sample's learning, or the watched run's start line, result or error. */
  event(m) {
    if (m.run === undefined) {
      if (!this.sampleId || m.sample !== this.sampleId) return;
      const was = this.sample?.state ?? this.describe?.sample_state;
      this.sample = m;
      if (m.state === 'ready' && was !== 'ready') this.refresh();    // its start line and results follow
      else this.changed(true);
      return;
    }
    if (!this.runId || m.run !== this.runId) return;
    if (m.state === 'switch') { this.takeSwitch(m.enabled, m.n); return; }
    if (m.state === 'trial') { this.takeTrial(m.trial ?? null); this.changed(true); return; }   // started or ended elsewhere
    if (m.state === 'start') { this.takeStart(m.summary); this.restarting = false; this.changed(false); return; }
    if (m.state === 'seqs') { if (this.takeSeqs(m.seqs)) this.changed(false); return; }
    if (m.state === 'error') {
      // Only a failure newer than every result the page has: an old one (replayed when the run is watched again) was
      // answered by the results after it, and a finished run would otherwise say the worker restarts for good.
      if (m.seq == null ? !!this.start : Number.isFinite(m.seq) && m.seq <= (this.seqs.at(-1) ?? -1)) return;
      this.restarting = true; this.changed(false); return;
    }
    if (m.state !== 'ready' || !m.summary) return;
    const summary = { ...m.summary, seq: m.seq ?? m.summary.seq, iter: m.iter ?? m.summary.iter };
    if (summary.model && this.model && summary.model !== this.model) { this.dropResults(summary.model); this.refresh(); }
    this.model ??= summary.model ?? null; this.restarting = false;
    if (!(summary.seq > 0)) return;                    // frame 0 (nuclei only): the start line stands for it
    this.missing.delete(summary.seq); this.retries.delete(`s${summary.seq}`); this.addSummary(summary);
    // A live result carries the trial as of that result (its rows, band and verdict): the newest one is the trial now.
    if (summary.trial && summary.seq >= (this.seqs.at(-1) ?? 0)) this.takeTrial(summary.trial, summary.seq);
    if (this.asked === summary.seq) this.asked = null;
    // With Mixing colours on, or a cell readout in use, its values are asked for now, so they are there when its snapshot
    // is painted.
    const ctx = this.context();
    if (this.wantValues(ctx) && this.enabled && (ctx.live || summary.seq === this.seqs[floorIndex(this.seqs, ctx.seq ?? -1)])) this.fetchCells(summary.seq);
    this.changed(false);
  }
  /** Whether results wait for their per-cell values (and those are asked for as they arrive): Mixing colours on, or a
   *  cell readout in use (it asked for a value within READOUT_MS). */
  wantValues(ctx, now = this.clock.now()) { return !!ctx?.wantCells || now - this.readoutAt < READOUT_MS; }
  addPoint(p) {
    if (!p || !finite(p.seq) || p.seq <= 0) return;
    const had = this.points.get(p.seq), point = had ? { ...had, ...p } : p;
    this.points.set(p.seq, point);
    insertSorted(this.seqs, p.seq);
    const line = this.line, last = line.at(-1);
    if (!last || p.seq > last.seq) line.push(point);
    else {
      let i = line.length - 1; while (i >= 0 && line[i].seq > p.seq) i--;
      if (line[i]?.seq === p.seq) line[i] = point; else line.splice(i + 1, 0, point);
    }
    if (line.length > LINE_MAX) this.line = [...decimate(line.slice(0, -200), LINE_MAX / 2), ...line.slice(-200)];
    this.lineVersion++;
    if (this.points.size > TRIM_POINTS) this.trimPoints();
  }
  /** Points of results the page no longer needs at hand (not on the line, not held as summaries, not among the newest
   *  KEEP_POINTS): a long run keeps a bounded number; a scrub to one of them reads its summary again. */
  trimPoints() {
    const keep = new Set(this.line.map(p => p.seq));
    for (const seq of this.summaries.keys()) keep.add(seq);
    for (const seq of this.seqs.slice(-KEEP_POINTS)) keep.add(seq);
    if (this.applied) keep.add(this.applied.seq);
    for (const seq of this.points.keys()) if (!keep.has(seq)) this.points.delete(seq);
  }
  addSummary(summary) {
    if (!(summary?.seq > 0)) return;
    // Counted again (the hold, a finish or the end re-counting a frame a token counted): the values held of the other
    // count go (fetchCells asks for this one's).
    const held = this.summaries.get(summary.seq) ?? this.points.get(summary.seq);
    if (held && countOf(held) !== countOf(summary)) this.cells.delete(summary.seq);
    this.addPoint({ seq: summary.seq, iter: summary.iter, per100: summary.per100, shown: summary.shown, raw_per100: summary.raw_per100,
      judged: summary.judged, mixed: summary.mixed, median_tx: summary.median_tx, single: summary.single, count: countOf(summary) });
    this.summaries.delete(summary.seq); this.summaries.set(summary.seq, summary); this.cellsMissing.delete(summary.seq);
    for (const seq of this.summaries.keys()) {
      if (this.summaries.size <= KEEP_SUMMARIES) break;
      if (seq !== this.applied?.seq) this.summaries.delete(seq);
    }
  }

  // ---- what is shown ------------------------------------------------------------------------------------------
  /** Data changed: `now` (the user's own action, a paused run) renders at once; a live run's result rides the next
   *  snapshot's paint batch, at most WITH_SNAPSHOT_MS later. */
  changed(now = false) {
    if (now || !this.batch || !this.context().live) { this.frame(); return; }
    this.batch.ride('qc', () => this.frame());
    if (this.soonTimer == null) {
      const epoch = this.epoch;
      this.soonTimer = this.clock.setTimeout(() => { this.soonTimer = null; if (epoch === this.epoch) this.batch.mark('qc', () => this.frame()); }, WITH_SNAPSHOT_MS);
    }
  }
  /** The frame on screen changed (main.js: in the snapshot's paint batch), or what it shows did: pick and render. */
  frame() {
    if (this.soonTimer != null) { this.clock.clearTimeout(this.soonTimer); this.soonTimer = null; }
    const ctx = this.context(), seq = ctx.seq ?? -1;
    if (!this.runId) return this.render(ctx, null, false);
    let target = null;
    if (this.enabled && seq > 0) {
      // Looking back at a frame before the first result: the oldest after it (the engine's answer for a frame asked for).
      const i = floorIndex(this.seqs, seq);
      target = i >= 0 ? this.seqs[i] : !ctx.live ? this.seqs[0] ?? null : null;
      // A frame never checked: ask for it (while the sample still learns too: the engine keeps it and checks it first).
      if (!ctx.live && target !== seq && this.describe && !['unavailable', 'failed', 'off'].includes(this.describe.state)) this.ask(seq);
    }
    // Never a later one, but the one picked for a frame before the first result.
    const now = this.clock.now(), applied = this.applied && (this.applied.seq <= seq || this.applied.seq === target) ? this.applied : null;
    const wanted = target == null ? null : this.entry(target, ctx);
    let next = wanted;
    if (target != null && !next) {
      // Checked, but not loaded here (the series is thinned): it is asked for; meanwhile the nearest earlier result
      // the page has (never a later one).
      const i = floorPoint(this.line, target);
      if (i >= 0) next = this.entry(this.line[i].seq, ctx);
    }
    // A result known only by its point of the line (the series has no areas): until its summary arrives the result shown
    // stays when it is an earlier one, so the number and the areas change together, once (a scrub forward).
    if (next && !next.summary && applied?.summary && applied.seq < next.seq) next = applied;
    // Mixing colours (or a cell readout in use): a result waits for its values, so the number and the colours (and the
    // readout) change together and the cells never go grey in between. Meanwhile the newest result between the one shown
    // and it whose values are loaded (they are asked for as results arrive), else the one shown, if it has its values;
    // unless they cannot come (then the newest earlier values keep colouring the cells).
    const want = this.wantValues(ctx, now);
    if (next && want && !next.cells && !this.cellsMissing.has(next.seq)) {
      const loaded = this.loadedBetween(applied?.seq ?? 0, next.seq, ctx);
      if (loaded) next = loaded;
      else if (applied?.cells && applied.seq < next.seq) next = applied;
    }
    // Large runs while following live: at most one new result per COALESCE_MS, and always with a snapshot.
    if (next && applied && next.seq !== applied.seq && ctx.live && (ctx.run?.info?.cells ?? 0) >= COALESCE_CELLS) {
      const wait = COALESCE_MS - (now - this.appliedAt);
      if (wait > 0) {
        if (this.applyTimer == null) {
          const epoch = this.epoch;
          this.applyTimer = this.clock.setTimeout(() => { this.applyTimer = null; if (epoch === this.epoch) this.changed(false); }, wait);
        }
        next = applied;
      }
    }
    if (next && next.seq !== this.applied?.seq) { this.appliedAt = now; this.harness?.count('qcApplied'); }
    this.applied = next;
    // Still on its way: the result the frame on screen will show, with its values when the colours want them, or the
    // frame's own result, asked for (a step or a scrub to a frame not checked yet).
    const asking = !ctx.live && seq > 0 && target !== seq && (this.askSeq === seq || this.asked === seq);
    const pending = asking || (target != null && !this.missing.has(target) && (!next || next.seq !== target || !next.summary
      || (want && !next.cells && !this.cellsMissing.has(next.seq))));
    return this.render(ctx, next, pending);
  }
  /** The newest result after `after` and before `before` whose summary and per-cell values the page has (never a later
   *  one than the frame on screen's), or null. */
  loadedBetween(after, before, ctx) {
    for (let i = floorIndex(this.seqs, before - 1); i >= 0 && this.seqs[i] > after; i--) {
      const seq = this.seqs[i];
      if (this.cells.has(seq) && this.summaries.has(seq) && this.points.has(seq)) return this.entry(seq, ctx);
    }
    return null;
  }
  /** What is known of result `seq` (its point, its full summary with the areas, its per-cell values), fetching what is
   *  missing; the same object while nothing changed. */
  entry(seq, ctx) {
    const point = this.points.get(seq);
    if (!point) { this.fetchSummary(seq); return null; }
    const summary = this.summaries.get(seq) ?? null, cells = this.cells.get(seq) ?? null;
    if (!summary) this.fetchSummary(seq);
    if (!cells && this.wantValues(ctx)) this.fetchCells(seq);
    const a = this.applied;
    return a?.seq === seq && a.point === point && a.summary === summary && a.cells === cells ? a : { seq, point, summary, cells };
  }
  async fetchSummary(seq) {
    const key = `s${seq}`, epoch = this.epoch, retry = this.retries.get(key);
    if (this.pending.has(key) || this.missing.has(seq) || !this.runId || (retry && this.clock.now() < retry.at)) return;
    this.pending.add(key);
    try {
      const summary = await this.api.qcSummary(this.runId, seq);
      if (epoch === this.epoch) this.retries.delete(key);
      if (epoch === this.epoch && (!this.model || summary.model === this.model)) { this.addSummary(summary); this.changed(!this.context().live); }
    } catch (error) {
      if (epoch !== this.epoch) return;
      // Not checked (yet): asked again only when an event says so (or on a reconnect). Any other failure (the engine
      // restarting, a network blip) is asked again after a short wait, the next time that result is needed.
      if (error?.status === 404) this.missing.add(seq);
      else { const tries = (retry?.tries ?? 0) + 1; this.retries.set(key, { tries, at: this.clock.now() + RETRY_MS[Math.min(tries, RETRY_MS.length) - 1] }); }
    } finally { if (epoch === this.epoch) this.pending.delete(key); }
  }
  /** The count result `seq` reads as the page knows it (its summary's, else its point's). */
  countAt(seq) { return countOf(this.summaries.get(seq) ?? this.points.get(seq)); }
  /** Result `seq`'s per-cell values on its count (api.js qcCells names it, and the model). Values of a count the result
   *  no longer reads when they come (counted again meanwhile) are dropped, its own asked for; a 404 (the engine's count
   *  of the frame changed) asks for its summary again, once per count, which brings the count it reads now. */
  async fetchCells(seq) {
    const count = this.countAt(seq), key = `c${seq}:${count}`, epoch = this.epoch, model = this.model;
    if (this.pending.has(key) || this.cellsMissing.has(seq) || !this.runId || !model) return;
    this.pending.add(key);
    try {
      const values = await this.api.qcCells(this.runId, seq, model, undefined, count);
      if (epoch !== this.epoch || model !== this.model) return;
      if (this.countAt(seq) !== count) { this.changed(!this.context().live); return; }
      this.cells.delete(seq); this.cells.set(seq, values);
      for (const k of this.cells.keys()) { if (this.cells.size <= KEEP_CELLS) break; if (k !== this.applied?.seq && k !== 'start') this.cells.delete(k); }
      this.changed(!this.context().live);
    } catch (error) {
      // Not asked again on every render (a reconnect or a new result of that frame asks again); the result shows anyway.
      if (epoch === this.epoch && model === this.model && this.countAt(seq) === count) {
        this.cellsMissing.add(seq);
        if (error?.status === 404 && !this.rechecked.has(`${seq}:${count}`)) { this.rechecked.add(`${seq}:${count}`); this.fetchSummary(seq); }
        this.changed(!this.context().live);
      }
    } finally { if (epoch === this.epoch) this.pending.delete(key); }
  }
  /** The newest per-cell values the page has of a result before `seq` (never a later one), else the start line's (the
   *  values of frame 0, before every result), with the result they are from: {seq: n | 'start', values}, or null. */
  earlierCells(seq) {
    let best = -1;
    for (const k of this.cells.keys()) if (typeof k === 'number' && k < seq && k > best) best = k;
    if (best >= 0) return { seq: best, values: this.cells.get(best) };
    const start = this.cells.get('start');
    return start ? { seq: 'start', values: start } : null;
  }
  /** The start line's per-cell values (the selected cell's line; the colours of frame 0), fetched once per model. */
  startCells() {
    const values = this.cells.get('start');
    if (values || !this.model || !this.start || this.noStart()) return values ?? null;
    const key = 'cstart', epoch = this.epoch, model = this.model;
    if (!this.pending.has(key) && this.startCellsFailed !== model) {
      this.pending.add(key);
      this.api.qcCells(this.runId, 'start', model)
        .then(v => { if (epoch === this.epoch && model === this.model) { this.cells.set('start', v); this.changed(!this.context().live); } })
        .catch(() => { if (epoch === this.epoch) this.startCellsFailed = model; })
        .finally(() => { if (epoch === this.epoch) this.pending.delete(key); });
    }
    return null;
  }
  /** One cell's value (label: 1-based; 255 = not judged) in the result shown: null while that result's own values load
   *  (asked for here: a tooltip or the selected cell's line needs them), never an earlier result's value in its place (the
   *  colours may show one for a moment, a readout never). */
  cellValue(label) {
    const reading = this.cellReading(label);
    return reading?.current ? reading.code : null;
  }
  /** One cell's value with the result it is from: {code, seq (n | 'start'), iter (null for the start line), current (it
   *  is the result shown's own)}, or null while no values are loaded. Asks for the result shown's own values when they
   *  are not in; a readout that shows a reading that is not current names its iteration (qc-copy.mjs cardLine `iter`). */
  cellReading(label) {
    this.readoutAt = this.clock.now();
    const v = this.view;
    if (!v?.cells || !v.cellsCurrent) {
      if (v?.atStart) this.startCells(); else if (this.applied && this.enabled) this.fetchCells(this.applied.seq);
    }
    if (!v?.cells || !(label > 0 && label <= v.cells.length)) return null;
    return { code: v.cells[label - 1], seq: v.cellsSeq, iter: v.cellsIter, current: !!v.cellsCurrent };
  }
  /** The same cell in the starting cells (null: not known, or none). */
  startValue(label) { const v = this.startCells(); return v && label > 0 && label <= v.length ? v[label - 1] : null; }

  /** The Log button's 'Mixing detail' row: the numbers behind the panel's words for the result shown (which count it
   *  read and the thresholds' set, judged coverage and its parts, transcript retention, the raw and the like-for-like
   *  change with their thresholds, and the model's provenance) and which result the cell values are from. '' with no run
   *  open. */
  logText() {
    const v = this.view;
    if (!this.runId || !v) return '';
    if (v.enabled === false) return 'off';
    const s = v.atStart ? null : v.summary, parts = [];
    const n = x => Number.isFinite(x) ? (Number.isInteger(x) ? x.toLocaleString('en-GB') : String(Number(x.toFixed(3)))) : '-';
    parts.push(v.atStart ? 'showing the start line' : s ? `result of iteration ${n(s.iter)} (${v.ownShown ? 'its own numbers while the run rests' : s.single ? 'single' : 'smoothed'})` : 'no result shown');
    if (v.snapped != null) parts.push(`another iteration than the one on screen (iteration ${n(v.frameIter)})`);
    const point = v.atStart ? v.start : s, coverage = coverageOf(point);
    if (point?.basis || point?.count) parts.push(`read on ${point.basis ?? '-'}'s count (${point.count ?? '-'})`);
    const constants = point?.constants;
    if (constants && typeof constants === 'object') parts.push(`thresholds ${constants.id ?? '-'} for ${constants.mode ?? '-'} runs`
      + `${constants.measured === false ? ', not re-measured on proseg\'s count' : ''}`);
    if (coverage) parts.push(`judged ${n(coverage.judged)} of ${n(coverage.cells)} cells (${n(coverage.empty)} empty, ${n(coverage.under20)} with 1-19 `
      + `transcripts, ${n(coverage.under100)} with 20-99)`);
    else if (Number.isFinite(point?.judged)) parts.push(`judged ${n(point.judged)} cells (all cells not counted)`);
    const retention = retentionOf(s);
    if (retention) parts.push(`transcripts kept ${n(retention.ratio)} (${n(retention.tx)} now, ${n(retention.start)} at the start)`);
    const c = s?.change;
    // A run with no start line (it started from nuclei, alone or with their cells) has nothing to compare with.
    if (this.noStart()) parts.push('no change since the start: this kind of start has no start line');
    else if (c && typeof c === 'object') parts.push(`change since the start: raw ${n(c.raw)}, like for like ${n(c.matched)} (threshold ${n(c.threshold)}, `
      + `model ${n(c.model_threshold)}, ${c.meaningful === true ? 'meaningful' : c.meaningful === false ? 'not meaningful' : '-'}, ${c.direction ?? '-'})`);
    // Like for like on proseg's count is worked out once, without draws (no spread over seeds then).
    const m = s?.matched;
    if (m && typeof m === 'object') parts.push(`like for like over ${n(m.paired)} cells: ${n(m.per100_now)} now, ${n(m.per100_start)} at the start`
      + (Number.isFinite(m.sd) ? `, sd ${n(m.sd)} over ${n(m.seeds)} seeds` : ''));
    const provenance = s?.provenance ?? v.sampleModel?.provenance;
    if (provenance && typeof provenance === 'object') parts.push(`model learned from ${provenance.table ?? '-'}${provenance.other_table ? ` with ${provenance.other_table}` : ''}, `
      + `${provenance.priors ?? '-'} starting estimates, ${n(provenance.epochs_run)} passes (reference ${n(provenance.epochs_notebook)}), `
      + `${n(provenance.cells)} cells, ${provenance.notebook_equivalent === true ? 'the reference settings' : `not the reference settings (${(provenance.differences ?? []).join(', ') || '-'})`}`);
    else if (v.model) parts.push('model provenance not recorded');
    if (v.cells) parts.push(`cell values from ${v.cellsSeq === 'start' ? 'the start line' : `iteration ${n(v.cellsIter)}`}`
      + (v.cellsCurrent ? '' : ' (not the result shown: kept out of the cell readouts)'));
    return parts.join('; ');
  }

  /** The result a resting run shows as its own iteration's (H2): while the run rests (paused with proseg's ack,
   *  finishing, done, stopped or failed), the seq of the newest summary held, the result shown when following the newest
   *  frame (the newest analysed counted frame, never the newest frame itself, which on a stopped or failed run can be
   *  uncounted); null while it plays. */
  restSeq(ctx) {
    if (!rests(ctx?.run)) return null;
    let newest = null;
    for (const seq of this.summaries.keys()) if (newest == null || seq > newest) newest = seq;
    return newest;
  }
  /** The line as shown: a resting run's own point in place of its smoothed one (`own`: ownNumbers of the result at
   *  `rest`), the same array while nothing changed; {line, version} (the version names the substitution too). */
  restLine(rest, own) {
    if (!own) return { line: this.line, version: this.lineVersion };
    const key = `${this.lineVersion}:${rest}`;
    if (this.restKey !== key) {
      this.restKey = key;
      this.restPoints = this.line.map(p => p.seq === rest ? { ...p, per100: own.per100, shown: own.shown, single: true } : p);
    }
    return { line: this.restPoints, version: key };
  }

  render(ctx, applied, pending = false) {
    const frames = ctx.frames || [], seq = ctx.seq ?? -1, run = ctx.run, newest = frames.length - 1;
    const atStart = seq === 0 && !!this.enabled;
    this.ticks = changeTicks(this.ticks, frames); this.forming = formingUntil(this.forming, frames);
    const status = panelStatus({ describe: this.describe, sample: this.sample, enabled: this.enabled, results: this.seqs.length > 0,
      start: this.start, run, seq, newest, live: ctx.live });
    const d = this.describe, sample = this.sample ?? (d ? { state: d.sample_state, reason: d.reason, progress: d.progress, eta_s: d.eta_s,
      relearn: d.relearn, model: d.model } : null);
    // While the run rests, its newest result shows its own iteration's numbers (H2): the summary and its point of the
    // line alike, so the panel, the chip, the chart, the timeline lane and the finished and stopped lines read them.
    const rest = this.restSeq(ctx), own = rest == null ? null : ownNumbers(this.summaries.get(rest));
    const { line, version } = this.restLine(rest, own);
    // A point of the line has no phase: its frame's (the panel says when cells are still forming).
    let summary = applied && !atStart ? { phase: frames[applied.seq]?.phase, ...applied.point, ...(applied.summary || {}), seq: applied.seq } : null;
    const ownShown = !!summary && !!own && summary.seq === rest;
    if (ownShown) summary = { ...summary, ...own };
    // A result whose values are still on their way (or cannot come) colours the cells with the newest earlier values,
    // with Mixing colours on only (without them the values are the readouts', which never take an earlier result's).
    // Which result the values are from is always said.
    let cells = null, cellsSeq = null;
    if (this.enabled && atStart) { cells = this.cells.get('start') ?? (ctx.wantCells ? this.startCells() : null); cellsSeq = cells ? 'start' : null; }
    else if (this.enabled && applied?.cells) { cells = applied.cells; cellsSeq = applied.seq; }
    else if (this.enabled && applied && ctx.wantCells) { const earlier = this.earlierCells(applied.seq); cells = earlier?.values ?? null; cellsSeq = earlier?.seq ?? null; }
    const cellsCurrent = cellsSeq != null && cellsSeq === (atStart ? 'start' : applied?.seq);
    const cellsIter = typeof cellsSeq === 'number' ? this.points.get(cellsSeq)?.iter ?? this.summaries.get(cellsSeq)?.iter ?? frames[cellsSeq]?.iter ?? null : null;
    // Motion only for a change of what is shown at least MOTION_GAP_MS after the previous one, and only while following
    // the run or when the frame on screen moved (a scrub, a step) and its result has not landed yet: paused or finished,
    // a result that arrives for the frame already on screen lands without motion ("still": 2.7, nothing moves while
    // paused apart from a scrub's transition, which also comes when its result or its values arrive a moment later).
    const now = this.clock.now(), key = atStart ? 'start' : summary ? `${summary.seq}` : '';
    if (seq !== this.renderedSeq && this.renderedSeq != null) { this.moveDue = true; this.movedAt = now; }
    if (this.moveDue && now - this.movedAt >= MOVE_WINDOW_MS) this.moveDue = false;
    const still = !ctx.live && !this.moveDue;
    if (key !== this.renderedKey) {
      this.motion = !!this.renderedKey && now - this.renderedAt >= MOTION_GAP_MS && !still;
      this.renderedKey = key; this.renderedAt = now;
    } else this.motion = false;
    this.renderedSeq = seq;
    const startPending = atStart && ctx.wantCells && !this.cells.has('start') && this.startCellsFailed !== this.model && !!this.start;
    if (!pending && !startPending) this.moveDue = false;   // landed: later arrivals for this frame are still
    // The areas (none is drawn: the cards' numbers and "Look at its cells" read them): the result's own, or the start
    // line's at frame 0 and until cells have formed (area_view "start": the engine flags a run's own areas from burn-in +
    // 20 iterations on).
    const areaView = !this.enabled ? null : atStart ? 'start' : summary?.area_view ?? null;
    const fromStart = areaView === 'start' && !this.noStart();
    if (fromStart && this.start && !this.startAreas) this.fetchStart();
    const areas = !this.enabled ? [] : fromStart ? this.startAreas ?? [] : areaView === 'start' ? [] : summary?.areas ?? [];
    const { suggestion, moreIdeas, trial, trialEnded } = this.roundThree(ctx, summary, atStart);
    const view = this.view = {
      status, enabled: this.enabled, runId: this.runId, sampleId: this.sampleId, model: this.model, tiled: this.tiled,
      // A run that cannot be checked says why itself (no_transcripts); learning says it through its sample's events.
      reason: status === 'unavailable' && d?.state === 'unavailable' ? d.reason ?? sample?.reason ?? null
        : LEARNING.has(status) ? sample?.reason ?? d?.reason ?? null : null,
      progress: sample?.progress ?? null, eta_s: sample?.eta_s ?? null, shared: !!sample?.shared, relearn: !!sample?.relearn,
      // The room learning needs on the graphics card and what is free (engine/qc.py _wait_for_gpu), while it waits or
      // when the card is too small for the panel: the panel's words say both numbers.
      room: sample?.room ?? null,
      sampleModel: sample?.model ?? d?.model ?? null,
      sampleState: sample?.state ?? null,   // learned or not: the off panel says what switching it on costs (qc-panel.js)
      // ... and, for a sample it cannot learn (unavailable) or could not (failed), why: the sample's own reason (its
      // events, else the run's answer's sample_reason, sent with the switch off too).
      sampleReason: this.sample ? this.sample.reason ?? null : d?.sample_reason ?? null,
      requested: (d?.requested ?? this.asked) != null && ['waiting', 'learning'].includes(status), restarting: this.restarting,
      start: this.noStart() ? null : this.start, startSource: this.startSource, startKind: this.startKind, atStart, summary, areas, areaView,
      // Looking back (not following live), a result of another frame than the one on screen: its iteration (the panel's
      // and the chip's badge: the nearest with proseg's count); `later`: it is a later frame's, the oldest after one
      // before every result (never otherwise).
      snapped: !ctx.live && summary && summary.seq !== seq && finite(summary.iter) ? summary.iter : null,
      later: !ctx.live && !!summary && summary.seq > seq && floorIndex(this.seqs, seq) < 0,
      // While cells still form, the band ends at the newest point (not at each new frame's iteration): the chart's key
      // changes with a result, never with every snapshot. A resting run's own point in it (restLine).
      line, lineVersion: version, ticks: this.ticks.ticks, ownShown,
      formingUntil: this.forming.done || !finite(this.forming.until) ? this.forming.until : Math.min(this.forming.until, line.at(-1)?.iter ?? this.forming.until),
      frameIter: frames[seq]?.iter ?? null, cells, cellsSeq, cellsIter, cellsCurrent, startCells: this.cells.has('start'), motion: this.motion, still,
      batch: d?.batch_per100 ?? summary?.batch_per100 ?? null, failed: this.failed && !d, runState: run?.state ?? null,
      // The card for the result shown and the other ideas, the trial, and the trial just ended.
      suggestion, moreIdeas, trial, trialEnded,
    };
    this.renderView(view);
    return view;
  }
  /** The suggestions' part of the view. The card: the result shown's suggestion (never one set aside here), the other
   *  matches as more ideas. The trial: on the newest frame the page's own (its answers, the pushes and the newest
   *  result's copy), on an earlier frame that frame's own copy, and never on a frame from before it was asked for;
   *  after Keep or Undo, its last words for ENDED_NOTE_ITERS iterations of the run. */
  roundThree(ctx, summary, atStart) {
    const none = { suggestion: null, moreIdeas: [], trial: null, trialEnded: null };
    if (!this.enabled || atStart || !summary) return none;
    const frames = ctx.frames || [], newest = frames.length - 1, onNewest = !!ctx.live || (ctx.seq ?? -1) >= newest;
    const kept = s => !!s && typeof s === 'object' && !!s.rule && !this.dismissed.has(cardKey(s));
    const suggestion = kept(summary.suggestion) ? summary.suggestion : null;
    const more = summary.more_ideas ?? summary.suggestion?.more ?? [];
    // On the newest frame the trial shows at once, also before a result newer than the click arrives (the card goes
    // with the click); an earlier frame shows only its own copy, never one from before the trial was asked for.
    let trial = onNewest ? this.trial : summary.trial && !summary.trial.outcome && !this.ended.has(summary.trial.id) ? summary.trial : null;
    const asked = Number.isFinite(trial?.asked_iter) ? trial.asked_iter : this.trial?.id === trial?.id ? this.trial?.asked_iter : null;
    if (trial && !onNewest && Number.isFinite(asked) && Number.isFinite(summary.iter) && summary.iter < asked) trial = null;
    const ended = this.trialEnded, newestIter = frames.at(-1)?.iter;
    const trialEnded = onNewest && !trial && ended && !(Number.isFinite(ended.endedIter) && Number.isFinite(newestIter)
      && newestIter - ended.endedIter >= ENDED_NOTE_ITERS) ? ended : null;
    return { suggestion, moreIdeas: (Array.isArray(more) ? more : []).filter(kept), trial, trialEnded };
  }
}
