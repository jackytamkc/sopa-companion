// Rendering telemetry. Every PERIOD_MS while a run is open the page POSTs a small JSON summary of how it renders to the
// engine (/api/v1/telemetry; fire-and-forget, keepalive, never throws): what it shows (run mode, cells, UMAP, 2-D/3-D,
// whole-sample level), how often it draws (stage, UMAP, timeline, panel batches, snapshots, and the frames that showed
// any of them), main-thread health (long tasks; long animation frames and the script that took longest in them;
// input delay from the Event Timing API; `stretched` when the report itself came late, i.e. the page was blocked or
// throttled when it was due), the app's own timings of heavy steps (performance.measure entries named 'plv:<step>'),
// memory, the display (pixel ratio, window and stage size, the WebGL renderer, once) and what it fetched (view, frame,
// image, transcript, vector and outline requests: count and bytes, from Resource Timing). It names no cell, run,
// sample or path (a script is named by its module path, an invoker without attribute values or origins; a measure's
// detail keeps numbers, flags and short words without digits). Every source is optional: a browser without one
// reports null for it.
// It wraps nothing. A requestAnimationFrame wrapper ran every frame callback of the page inside a function of this
// module, so Long Animation Frames named telemetry.mjs as the script of every slow frame (the 26 s Prior-shapes
// rebuild included); frames are counted from the page's own draw counters instead (watch()).
// The engine's side (server/app.py): any JSON object of at most MAX_BYTES -> 204, logged to
// <state_dir>/logs/telemetry.jsonl. Like every /api/v1 POST it needs the page key (page-key.mjs: window.fetch adds it)
// and the page's own Origin, which a same-origin fetch in the default (cors) mode carries.
export const VERSION = 2;     // 2: rates.frames, measures, loaf.script.pos, fetch vectors|shapes; no rates.raf_*
export const PERIOD_MS = 10000;
export const ENDPOINT = '/api/v1/telemetry';
export const MAX_BYTES = 16 * 1024;   // the engine refuses a larger report (413): TELEMETRY_MAX_BYTES in server/app.py
const SAMPLES = 4096;   // timing samples kept per period (enough for the percentiles of 10 s at 60 Hz)
export const STRETCH = 1.1;   // a period longer than this many nominal periods is reported as stretched
const TEXT = 80;              // characters kept of a long animation frame's script names
export const DRAWS = ['stage', 'umap', 'timeline', 'batches'];   // counters() whose growth puts a new frame on screen
export const IDLE_FRAMES = 2;  // a frame watch ends after this many frames without a draw
// Input that starts a frame watch: what moves the camera, hovers, scrubs or resizes (draws at the display rate).
export const WATCH = ['pointerdown', 'pointermove', 'wheel', 'keydown', 'resize'];
export const MEASURE = 'plv:';   // performance.measure names reported ('plv:prior-rebuild', ...)
const MEASURES = 8;    // names per report; further ones are summed under plv:other
const DETAIL = 4;      // detail fields kept per measure

/** The p-quantile (0..1, nearest rank) of a list of numbers; null when empty. */
export function percentile(values, p) {
  if (!values.length) return null;
  const sorted = Float64Array.from(values).sort();
  return sorted[Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1))];
}
/** What a request URL fetched, for the fetch counts: whole-sample label views and windows ('view'), whole dense frames
 *  ('frame'), image channels ('image'), transcripts ('transcripts'), a run's prior shapes ('vectors'), its Xenium label
 *  masks ('masks'), the picker's prior outlines ('shapes'), a sample's segmentation layers (their list and
 *  outlines, 'segmentations') and Compare's status and points ('compare'); null for anything else. The harness's
 *  metrics.fetch_kind uses these names. */
export function fetchKind(url) {
  const path = String(url).replace(/^[a-z]+:\/\/[^/]+/i, '').split(/[?#]/)[0];
  if (!path.startsWith('/api/v1/')) return null;
  if (/\/frames\/\d+\/(?:view|window)$|\/prior\/view$/.test(path)) return 'view';
  if (/\/frames\/\d+$/.test(path)) return 'frame';
  if (/\/image\//.test(path)) return 'image';
  if (/\/transcripts(?:\/view)?$/.test(path)) return 'transcripts';
  if (/\/xenium$/.test(path)) return 'vectors';
  if (/\/xenium\/masks(?:\.bin)?$/.test(path)) return 'masks';
  if (/\/shapes\//.test(path)) return 'shapes';
  if (/\/segmentations(?:\/|$)/.test(path)) return 'segmentations';
  if (/\/compare(?:\/|$)/.test(path)) return 'compare';
  return null;
}
/** A script URL by its path only (an app module: no origin, query or fragment); a blob: or data: URL by its scheme. */
export function scriptPath(url) {
  const text = String(url || '');
  try { const u = new URL(text); return /^https?:$/.test(u.protocol) ? u.pathname.slice(0, TEXT) : u.protocol; }
  catch { return text.slice(0, TEXT); }
}
/** What invoked a script ('FrameRequestCallback', 'TimerHandler:setTimeout', 'CANVAS#stage.onwheel', a module URL)
 *  without attribute values or origins: an element's src could name a run. */
export function invokerName(text) {
  return String(text || '').replace(/\[[^\]]*\]/g, '[]').replace(/[a-z][a-z0-9+.-]*:\/\/[^/\s]*/gi, '').slice(0, TEXT);
}
/** A measure's name as reported: 'plv:' and a short identifier, else 'plv:other'. */
export function measureName(name) {
  return /^plv:[A-Za-z][\w.-]{0,35}$/.test(name) ? name : 'plv:other';
}
/** A measure's detail as reported (prior-layers.js: {polygons, layer: 'xenium+nuclei'}): finite numbers, flags and
 *  words of letters (no digits, so no run, sample or cell id passes), at most DETAIL fields with identifier names;
 *  anything else is dropped (a detail that is not an object: null). */
export function measureDetail(detail) {
  const value = v => typeof v === 'number' ? (Number.isFinite(v) ? v : undefined) : typeof v === 'boolean' ? v
    : typeof v === 'string' && /^[A-Za-z_+-]{1,24}$/.test(v) ? v : undefined;
  if (detail === null || typeof detail !== 'object' || Array.isArray(detail)) return value(detail) ?? null;
  const out = {};
  for (const [key, v] of Object.entries(detail)) {
    if (Object.keys(out).length >= DETAIL) break;
    const x = value(v);
    if (x !== undefined && /^[A-Za-z_]\w{0,23}$/.test(key)) out[key] = x;
  }
  return out;
}
const round = (value, digits = 1) => Number.isFinite(value) ? Math.round(value * 10 ** digits) / 10 ** digits : null;
/** Growth of a cumulative counter (a counter that restarted, e.g. a new UMAP view, counts from zero). */
const growth = (now, before) => !Number.isFinite(now) ? 0 : now >= (before || 0) ? now - (before || 0) : now;

/** The default transport: a keepalive POST whose failure is ignored. */
export function post(body) {
  try { globalThis.fetch(ENDPOINT, { method: 'POST', keepalive: true, credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body })?.catch?.(() => {}); }
  catch { /* telemetry never disturbs the page */ }
}

export class Telemetry {
  /** context() -> null when no run is open, else {mode, state, cells, live, umap, view, level, stage: [w, h], renderer};
   *  counters() -> cumulative counts {stage, umap, timeline, batches, snapshots}. */
  constructor({ context = () => null, counters = () => ({}), send = post, period = PERIOD_MS, now = () => performance.now() } = {}) {
    Object.assign(this, { context, counters, send, period, now });
    this.restore = []; this.renderer = null; this.win = null; this.watching = false; this.idle = 0;
    this.seen = this.draws(); this.reset(this.now());
  }
  /** Start a new period at `t`. */
  reset(t) {
    this.since = t; this.before = { ...this.counters() };
    this.raf = { frames: 0, lags: [] };
    this.tasks = { count: 0, max: 0 };
    this.loaf = { count: 0, max: 0, blocking: 0, script: null };
    this.measures = {};
    this.inputs = { count: 0, delays: [], durations: [] };
    this.fetches = {};
  }
  /** The draw counters now (DRAWS; {} when counters() fails). */
  draws() {
    try { const c = this.counters() || {}; return Object.fromEntries(DRAWS.map(key => [key, c[key]])); } catch { return {}; }
  }
  // ---- recorders (the observers call these; tests call them directly) ---------------------------------------------
  /** Frames that showed the draws since the last look: one (or none) when the last look was the watched frame before,
   *  else the most draws of any one surface (the page writes a snapshot's panels and timeline in the stage's frame).
   *  Whether anything drew. */
  drew(single) {
    const now = this.draws();
    const most = Math.max(0, ...DRAWS.map(key => growth(now[key], this.seen[key])));
    this.seen = now; this.raf.frames += single ? Math.min(1, most) : most;
    return most > 0;
  }
  /** A watched frame callback requested at `requested` ran at `fired`: how late frames come while the page draws. */
  frame(requested, fired) { if (this.raf.lags.length < SAMPLES) this.raf.lags.push(Math.max(0, fired - requested)); }
  longTask(duration) { this.tasks.count++; this.tasks.max = Math.max(this.tasks.max, duration); }
  /** A long animation frame (over 50 ms): its duration, how long it kept input waiting, and its scripts ({duration,
   *  invoker, invokerType, sourceURL, sourceFunctionName, sourceCharPosition}), of which the longest of the period is
   *  kept: the page's own callback, since nothing here wraps one. */
  longFrame(duration, blocking, scripts) {
    const f = this.loaf; f.count++; f.max = Math.max(f.max, duration || 0); f.blocking = Math.max(f.blocking, blocking || 0);
    for (const s of scripts || []) if (s?.duration > (f.script?.ms ?? -1)) {
      f.script = { ms: s.duration, type: s.invokerType || null, invoker: invokerName(s.invoker), source: scriptPath(s.sourceURL),
        fn: String(s.sourceFunctionName || '').slice(0, TEXT), pos: Number.isInteger(s.sourceCharPosition) && s.sourceCharPosition >= 0 ? s.sourceCharPosition : null };
    }
  }
  /** A performance.measure entry: one named 'plv:<step>' is the app timing a heavy step (prior-layers.js wraps each
   *  rebuild in 'plv:prior-rebuild' with detail {polygons, layer}); count, longest, total, and the last one's duration
   *  and detail. Other measures are not the app's. */
  measure(name, duration, detail) {
    if (!String(name).startsWith(MEASURE)) return;
    let key = measureName(name);
    if (!this.measures[key] && Object.keys(this.measures).length >= MEASURES) key = 'plv:other';
    const m = this.measures[key] ||= { n: 0, max: 0, total: 0, last: 0, detail: null }, ms = Math.max(0, duration || 0);
    m.n++; m.max = Math.max(m.max, ms); m.total += ms; m.last = ms; m.detail = measureDetail(detail);
  }
  /** An input event that waited `delay` ms for its handlers and `duration` ms until the next paint. */
  input(delay, duration) {
    const i = this.inputs; i.count++;
    if (i.delays.length < SAMPLES) { i.delays.push(Math.max(0, delay)); i.durations.push(Math.max(0, duration)); }
  }
  fetched(kind, wire, bytes) {
    if (!kind) return;
    const f = this.fetches[kind] ||= { n: 0, bytes: 0, wire: 0 };
    f.n++; f.bytes += bytes || 0; f.wire += wire || 0;
  }
  // ---- frames ------------------------------------------------------------------------------------------------------
  /** Input arrived: watch the next frames with a frame callback of this module, each checked once that frame has been
   *  rendered (a task posted from the callback: a draw later in the same frame counts in it, not in the next), until
   *  IDLE_FRAMES frames in a row drew nothing. It runs only while the page draws anyway (a callback that changes nothing
   *  presents nothing), so a still page runs none; outside a watch, drew() estimates the frames from the counters. */
  watch() {
    if (this.watching || typeof this.win?.requestAnimationFrame !== 'function') return;
    this.watching = true; this.idle = 0; this.first = true; this.next();
  }
  next() {
    const requested = this.now();
    try { this.pending = this.win.requestAnimationFrame(() => { this.frame(requested, this.now()); this.rendered(); }); }
    catch { this.watching = false; }
  }
  /** After a watched frame: did it show anything new? The first one also takes in what was drawn before the input,
   *  counted as outside a watch: a snapshot's panels written in a task just before are shown in this frame, with it. */
  check() {
    if (!this.watching) return;
    const drew = this.drew(!this.first); this.first = false;
    if (drew) this.idle = 0;
    else if (++this.idle >= IDLE_FRAMES) { this.watching = false; return; }
    this.next();
  }
  // ---- report ------------------------------------------------------------------------------------------------------
  /** The summary of the period that ends now (null when no run is open), and a new period. */
  report(win = globalThis) {
    if (!this.watching) this.drew(false);   // draws since the last look (a watch counts its own)
    const t = this.now(), seconds = Math.max(.001, (t - this.since) / 1000), context = this.context(), counts = this.counters();
    const rate = key => round(growth(counts[key], this.before[key]) / seconds, 2);
    let summary = null;
    if (context) {
      const { renderer, ...run } = context, r = this.raf, i = this.inputs, f = this.loaf;
      summary = {
        // stretched: the timer fired late, so the page was blocked (or, hidden, throttled) when this report was due.
        v: VERSION, period_s: round(seconds, 2), stretched: seconds * 1000 > this.period * STRETCH, hidden: !!win.document?.hidden, run,
        rates: { stage: rate('stage'), umap: rate('umap'), timeline: rate('timeline'), batches: rate('batches'), snapshots: rate('snapshots'), frames: round(r.frames / seconds, 2) },
        raf_lag_ms: { p95: round(percentile(r.lags, .95)), max: round(r.lags.length ? Math.max(...r.lags) : null) },
        long_tasks: { count: this.tasks.count, max_ms: round(this.tasks.max) },
        loaf: this.loafSupported === false ? null : { count: f.count, max_ms: round(f.max), blocking_max_ms: round(f.blocking),
          script: f.script && { ...f.script, ms: round(f.script.ms) } },
        measures: this.measured === false ? null : Object.fromEntries(Object.entries(this.measures).map(([name, m]) =>
          [name, { n: m.n, max_ms: round(m.max), total_ms: round(m.total), last_ms: round(m.last), last: m.detail }])),
        input: { count: i.count, delay_p95_ms: round(percentile(i.delays, .95)), delay_max_ms: round(i.delays.length ? Math.max(...i.delays) : null),
          duration_p95_ms: round(percentile(i.durations, .95)), duration_max_ms: round(i.durations.length ? Math.max(...i.durations) : null) },
        heap_mb: round(win.performance?.memory?.usedJSHeapSize / 2 ** 20),
        display: { dpr: round(win.devicePixelRatio || 1, 2), window: [win.innerWidth ?? null, win.innerHeight ?? null], screen: [win.screen?.width ?? null, win.screen?.height ?? null] },
        fetch: this.fetches,
      };
      if (renderer && renderer !== this.renderer) { summary.display.renderer = renderer; this.renderer = renderer; }   // once
    }
    this.reset(t);
    return summary;
  }
  /** Send this period's summary while a run is open, unless it is larger than the engine accepts (UTF-8 bytes). */
  tick(win = globalThis) {
    try {
      const summary = this.report(win), body = summary ? JSON.stringify(summary) : '';
      if (body && new TextEncoder().encode(body).length <= MAX_BYTES) this.send(body);
    } catch { /* never throws */ }
  }
  // ---- browser hooks -----------------------------------------------------------------------------------------------
  /** Observe the page: long tasks, long animation frames, input events, resources and 'plv:' measures, input that
   *  starts a frame watch, plus the timer. Each is optional. stop() undoes all of it. */
  start(win = globalThis) {
    this.win = win;
    const observe = (options, fn) => {
      try {
        const observer = new win.PerformanceObserver(list => { for (const entry of list.getEntries()) { try { fn(entry); } catch { /* skip */ } } });
        observer.observe(options); this.restore.push(() => observer.disconnect());
        return true;
      } catch { return false; }   // not supported by this browser
    };
    if (typeof win.PerformanceObserver === 'function') {
      observe({ type: 'longtask', buffered: true }, e => this.longTask(e.duration));
      observe({ type: 'event', durationThreshold: 16, buffered: true }, e => this.input(e.processingStart - e.startTime, e.duration));
      observe({ type: 'resource', buffered: false }, e => this.fetched(fetchKind(e.name), e.encodedBodySize, e.decodedBodySize));
    }
    // Long animation frames name the script behind a slow frame (Chrome 123+), measures the app's own heavy steps;
    // without them the report says null. (observe() ignores an entry type it does not know, without throwing.)
    const supported = type => !!win.PerformanceObserver?.supportedEntryTypes?.includes?.(type);
    this.loafSupported = supported('long-animation-frame');
    if (this.loafSupported) observe({ type: 'long-animation-frame', buffered: true }, e => this.longFrame(e.duration, e.blockingDuration, e.scripts));
    this.measured = supported('measure') && observe({ type: 'measure', buffered: true }, e => this.measure(e.name, e.duration, e.detail));
    // A watched frame is checked in a task posted from its callback, i.e. after the frame has been rendered.
    try {
      const channel = new win.MessageChannel();
      channel.port1.onmessage = () => this.check();
      this.rendered = () => channel.port2.postMessage(0);
      this.restore.push(() => channel.port1.close());
    } catch { this.rendered = () => this.check(); }   // no MessageChannel: checked in the callback
    const watch = () => this.watch();
    for (const type of WATCH) {
      try { win.addEventListener(type, watch, { capture: true, passive: true }); this.restore.push(() => win.removeEventListener(type, watch, { capture: true })); }
      catch { /* no events */ }
    }
    this.restore.push(() => { this.watching = false; try { win.cancelAnimationFrame?.(this.pending); } catch { /* optional */ } });
    const timer = win.setInterval?.(() => { this.tick(win); try { win.performance?.clearResourceTimings?.(); } catch { /* optional */ } }, this.period);
    this.restore.push(() => win.clearInterval?.(timer));
    return this;
  }
  stop() { for (const undo of this.restore.splice(0)) { try { undo(); } catch { /* ignore */ } } }
}
