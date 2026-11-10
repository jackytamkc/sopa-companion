// Read-only counters and a state snapshot for the development harness (sopa_companion/harness).
//
// Only a page whose URL has ?harness=1 gets them: window.__plvHarness = {version, snapshot()}. Without the flag
// `harness` is null, and every instrumented site's `harness?.…(…)` is skipped before its arguments are evaluated, so
// the page behaves and costs exactly as before (no wrapped fetch or history, no counters, no timers). (Not `H`: the
// app's drawing code names heights H, and a local H would shadow the import.)
//
// The flag is read once, when the page loads. The sign-in keeps it: the page at /?code=…&harness=1 takes only code and
// token out of its address (page-key.mjs), so it goes on as /?harness=1. While it is on, the app's own URL rewrites
// (openRun's ?run=<id>, enterPick's bare path) keep it too, so a reload keeps the hook. A page loaded without it never
// gets one: navigate to (or reload with) ?harness=1 instead of adding it later.
//
// Sites (each next to the app's own counter where there is one): stage.js draw and context loss, umap.js render and
// context loss, timeline.js draw, prior-layers.js canvas rebuild (path building and stroking timed) and, for a drawing
// made off the main thread (prior-raster-worker.mjs), the worker's paint time apart (priorWorkerPaint), analysis.js
// apply, tiled-loader.mjs requests by kind, main.js the run/page view (provide). The mixing check:
// qc-client.mjs counts results applied, the panel and the overlay their motion frames, and main.js sets the panel's
// state (and the areas drawn: none, so qcAreasShown stays 0 and qcAreaEvents is never counted; both are
// kept for the harness's metrics). Fetches in flight come from wrapping window.fetch here: a fetch
// counts until its promise settles (the response headers, a failure or an abort).
export const VERSION = 1;
export const FLAG = 'harness';

/** Whether a query string turns the hook on: exactly harness=1. */
export function enabled(search) {
  try { return new URLSearchParams(search || '').get(FLAG) === '1'; } catch { return false; }
}

/** `url` as history.replaceState takes it, with harness=1 added (resolved against `base`, the current location). A URL
 *  of another origin is left as it is, for history to refuse as it would. */
export function withFlag(url, base) {
  const u = new URL(String(url), base);
  if (u.searchParams.get(FLAG) === '1' || u.origin !== new URL(base).origin) return url;
  u.searchParams.set(FLAG, '1');
  return u.pathname + u.search + u.hash;
}

const round = value => Math.round(value * 10) / 10;
const NO_RUN = { id: null, state: null, mode: null, ackState: null, frames: 0, current: -1, iter: null, live: false };

/** The counters and the snapshot. `perf`: performance (now, timeOrigin). main.js hands its view over with provide(). */
export function createHook(perf = globalThis.performance) {
  const c = {
    stageDraws: 0, umapDraws: 0, timelineDraws: 0,
    priorRebuilds: 0, priorRebuildMsLast: 0, priorRebuildMsMax: 0, priorRebuildMsTotal: 0, priorPolygonsLast: 0,
    priorWorkerPaints: 0, priorWorkerPaintMsLast: 0, priorWorkerPaintMsMax: 0, priorWorkerPaintMsTotal: 0,
    rangePaints: 0, rangePaintMsLast: 0, rangePaintMsMax: 0, rangePaintMsTotal: 0, rangeKeptLast: null, rangeHiddenLast: null, rangeGreyLast: null,
    analysisApplied: 0, tiledRequests: {}, fetches: 0, fetchInFlight: 0, contextLosses: 0,
    // The mixing check: cumulative counts (metrics.py hook.qc_*), and two gauges of what is shown now: the panel's
    // state and the areas drawn (metrics.py _qc_hook; the snapshot's `qc` object says the same with the result's iteration).
    qcApplied: 0, qcAnimFrames: 0, qcAreaEvents: 0, qcSuggestionsShown: 0, qcTrials: 0, qcVerdicts: 0,
    qcPanelState: null, qcAreasShown: 0,
    // Go deeper's window (qc-deeper.js) and its arrows on the overlay (qc-overlay.mjs DeeperArrows): metrics.py
    // _deeper_hook. Listed from the start, so a window never opened reads 0, not missing.
    qcDeeperOpened: 0, qcDeeperApplied: 0, qcDeeperAnimFrames: 0, qcDeeperRetargets: 0, qcDeeperArrowFrames: 0,
    // The suspect transcripts' glow (glow.mjs repaints, never stage draws); the test of a setting (main.js: tests
    // started, their asks, the pool previews asked) and its corner (qc-test-corner.mjs: views fetched, canvas
    // repaints); metrics.py hook.*_per_s.
    glowFrames: 0, qcTests: 0, qcTestFetches: 0, qcTestPoolFetches: 0, qcTestViewFetches: 0, qcTestCornerDraws: 0,
    // Compare (opened, its canvases repainted, its GET /runs/{id}/compare asked) and the segmentation layers on
    // the tissue (their repaints); metrics.py hook.compare_* and hook.seg_layer_draws_per_s.
    compareOpened: 0, compareDraws: 0, compareFetches: 0, segLayerDraws: 0,
  };
  let view = () => ({});
  return {
    now: () => perf.now(),
    count(name, n = 1) { c[name] = (c[name] || 0) + n; },
    tiledRequest(kind) { c.tiledRequests[kind] = (c.tiledRequests[kind] || 0) + 1; },
    /** A PriorLayers canvas rebuild on the main thread: `ms` building and stroking the vector paths (or, for a drawing
     *  the worker made, uploading it), `polygons` stroked. */
    priorRebuild(ms, polygons) {
      c.priorRebuilds++; c.priorRebuildMsLast = ms; c.priorRebuildMsMax = Math.max(c.priorRebuildMsMax, ms);
      c.priorRebuildMsTotal += ms; c.priorPolygonsLast = polygons;
    },
    /** A drawing made off the main thread (prior-raster-worker.mjs): the worker's own paint time, `ms`. */
    priorWorkerPaint(ms) {
      if (!Number.isFinite(ms)) return;
      c.priorWorkerPaints++; c.priorWorkerPaintMsLast = ms; c.priorWorkerPaintMsMax = Math.max(c.priorWorkerPaintMsMax, ms);
      c.priorWorkerPaintMsTotal += ms;
    },
    /** A class repaint during a live range change (the outline painter reports its own elapsed time), and what it left
     *  on screen (prior-layers.js installed: the outlines kept and hidden, and grey ones, of which there must be none). */
    rangePaint(ms, polygons = null, { kept = null, hidden = null, grey = 0 } = {}) {
      if (!Number.isFinite(ms)) return;
      c.rangePaints++; c.rangePaintMsLast = ms; c.rangePaintMsMax = Math.max(c.rangePaintMsMax, ms); c.rangePaintMsTotal += ms;
      c.rangeKeptLast = Number.isFinite(kept) ? kept : null; c.rangeHiddenLast = Number.isFinite(hidden) ? hidden : null;
      c.rangeGreyLast = Number.isFinite(grey) ? grey : null;
    },
    /** What the mixing check shows now: the panel's state (qc-panel.js PANEL_STATES, or 'loading') and the areas drawn. */
    qcShown(state, areas = 0) { c.qcPanelState = state ?? null; c.qcAreasShown = Number.isFinite(areas) ? areas : 0; },
    /** fn() -> {page, run, level, umap, stage, tiled, priors, picker, framesShown}: what the page shows (main.js). `picker`
     *  while a sample is picked: {kind, problems (what Start still needs: field ids, or for one without a field
     *  its label and words as a press lists them, "Box 2: move it inside the region"), preview (Training patches
     *  boxes on the image that were suggested for the picker as it stands: 0 while a change or request waits for
     *  boxes of its own), boxes (the boxes on the image, [x0, y0, x1, y1] pixels, or null), busy (Start is working)}. */
    provide(fn) { view = fn; },
    /** Plain JSON; never throws (a failing view is reported as `error`). */
    snapshot() {
      let v = {}, error;
      try { v = view() || {}; } catch (e) { error = String(e?.message || e); }
      const { framesShown = 0, ...shown } = v;
      return {
        page: null, level: null, stage: null, tiled: null, priors: null, qc: null, picker: null, ...shown,
        version: VERSION, timeOrigin: perf.timeOrigin ?? null, now: round(perf.now()),   // a new timeOrigin: counters restarted
        run: { ...NO_RUN, ...shown.run }, umap: { open: false, width: 0, ...shown.umap },
        counters: { ...c, tiledRequests: { ...c.tiledRequests }, framesShown,
          priorRebuildMsLast: round(c.priorRebuildMsLast), priorRebuildMsMax: round(c.priorRebuildMsMax), priorRebuildMsTotal: round(c.priorRebuildMsTotal),
          priorWorkerPaintMsLast: round(c.priorWorkerPaintMsLast), priorWorkerPaintMsMax: round(c.priorWorkerPaintMsMax),
          priorWorkerPaintMsTotal: round(c.priorWorkerPaintMsTotal),
          rangePaintMsLast: round(c.rangePaintMsLast), rangePaintMsMax: round(c.rangePaintMsMax), rangePaintMsTotal: round(c.rangePaintMsTotal) },
        ...(error ? { error } : {}),
      };
    },
  };
}

/** The hook for `win` when its URL has the flag (counting fetches, keeping the flag in the URL), else null. */
export function install(win) {
  if (!enabled(win?.location?.search)) return null;
  const hook = createHook(win.performance);
  const fetch = win.fetch;
  if (typeof fetch === 'function') {
    win.fetch = function (...args) {
      hook.count('fetches'); hook.count('fetchInFlight');
      const done = () => hook.count('fetchInFlight', -1);
      let promise;
      try { promise = fetch.apply(win, args); } catch (error) { done(); throw error; }
      Promise.resolve(promise).then(done, done);
      return promise;   // the caller's own promise, unchanged
    };
  }
  const history = win.history;
  for (const name of ['pushState', 'replaceState']) {
    const native = history?.[name];
    if (typeof native === 'function') history[name] = function (state, unused, url) {
      return url === undefined ? native.call(this, state, unused) : native.call(this, state, unused, withFlag(url, win.location.href));
    };
  }
  win.__plvHarness = Object.freeze({ version: VERSION, snapshot: () => hook.snapshot() });
  return hook;
}

export const harness = typeof window === 'undefined' ? null : install(window);
