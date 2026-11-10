// The header's Log button: a plain-text log of this session and the open run, saved where the user chooses.
//
// The engine writes most of it (GET /api/v1/runs/{id}/log, or /api/v1/log with no run open; engine/runlog.py): a short
// summary first, the app's version and code, the run's settings, phases and the knobs the user changed, Clusters & UMAP
// with each step's typical time, the mixing check, the engine's events, the tails of proseg's and the engine's logs, and
// an appendix with every Clusters & UMAP result. The page adds what only it knows (`pageLog`): how it is set up (browser,
// screen, graphics, what is on screen, which count its numbers read), what the user did (Start, Pause, Step, Finish & save, colours, view, live knobs,
// the mixing check's switch), what it noted (notices, failed requests, script errors, clusters that could not be shown)
// and the state other modules hand it (`pageLog.provide`). `assemble` puts its "Page noted" line into the engine's summary
// and the engine's appendix at the very end. Pressing Log builds the text, then asks where to save it: the browser's own
// save dialog (showSaveFilePicker), else (no such dialog here, or the click is too long ago) a small dialog of its own
// whose button saves it as a download. Nothing is saved without the user's click.
//
// For a crash report the engine also keeps the page's part (uploadPageLog, below): sent at most every 10 s while the
// page's log changed and once more as the page goes, so a session that ended unexpectedly still has it (crash-report.mjs
// offers the report at the next start).
//
// One import, page-key.mjs, which has none and loads first: the button works even when the rest of the page failed to
// start (then the log says so). It reads the page's state through window.sopaCompanion (main.js) and shows its notices
// through the page's 'fetch-notice' event.
import { pageKey } from './page-key.mjs';
export const PAGE_LINES = 600;          // lines kept per kind (the newest; a repeated line is counted, not kept again)
export const LINE_CHARS = 400;          // a longer line is cut (a request's message may hold a whole server body)
export const SAVE_ID = 'sopa-companion-log';   // the save dialog remembers its folder under this id
// The page's own kinds (analysis.js adds 'clusters', its failures, and 'shown'; main.js 'action', the user's actions;
// crash-report.mjs 'crash'); 'page' says the page went (pagehide).
const KINDS = { notice: 'notice', request: 'request', error: 'error', page: 'page' };
export const SUMMARY = 'Summary', APPENDIX = 'Appendix: each Clusters & UMAP result';   // engine/runlog.py's own titles
const LINES_TITLE = "What the user did and the page noted (oldest first; this computer's clock)";

const pad = (n, w = 2) => String(n).padStart(w, '0');
/** This computer's local time of `t` (ms since 1970): "2026-10-18 19:03:04". */
export function localTime(t) {
  const d = new Date(t);
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}
/** "19:03:04.123" (a line's time). */
export function clock(t) {
  const d = new Date(t);
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}.${pad(d.getMilliseconds(), 3)}`;
}
/** "1 h 43 min", "12 min 5 s", "40 s". */
export function duration(ms) {
  const s = Math.max(0, Math.floor(ms / 1000));
  return s < 60 ? `${s} s` : s < 3600 ? `${Math.floor(s / 60)} min ${s % 60} s` : `${Math.floor(s / 3600)} h ${Math.floor(s % 3600 / 60)} min`;
}
/** A piece of a file name: letters, digits, dot, dash and underscore only. */
export function namePart(text) {
  return String(text ?? '').replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[-.]+|[-.]+$/g, '').slice(0, 80);
}
/** A file name's time, this computer's: 20261018-190304. */
export function fileTime(d) {
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}
/** sopa-companion-<sample>-<run>-<time>.txt (the parts there are), the time this computer's: 20261018-190304. A run id
 *  is its start time and its sample's name (engine/runs.py Runs.create), so the name is not said twice. */
export function logFileName({ sample, run, date = new Date() } = {}) {
  const time = fileTime(date);
  const inRun = String(sample ?? '').replace(/[^A-Za-z0-9-]+/g, '-').slice(0, 40).replace(/^-+|-+$/g, '');
  const id = inRun && String(run ?? '').endsWith(`-${inRun}`) ? String(run).slice(0, -inRun.length - 1) : run;
  return ['sopa-companion', namePart(sample), namePart(id), time].filter(Boolean).join('-') + '.txt';
}

/** The page's own lines: add(kind, text) keeps the newest `cap` per kind, a line repeating the previous one of its kind
 *  only counts it; provide(name, fn) names a state to read when the log is written (fn() -> text). `version` counts
 *  the changes (a line added or counted again): the crash upload sends the log only when it moved. */
export function createPageLog({ now = () => Date.now(), cap = PAGE_LINES } = {}) {
  const kinds = new Map(), providers = new Map();
  let order = 0, version = 0;
  return {
    add(kind, text) {
      text = String(text ?? '').replace(/\s+/g, ' ').trim();
      if (!text) return;
      if (text.length > LINE_CHARS) text = text.slice(0, LINE_CHARS - 1) + '…';
      const list = kinds.get(kind) || [], last = list.at(-1), t = now();
      version++;
      if (last && last.text === text) { last.n++; last.last = t; return; }
      list.push({ t, last: t, kind, text, n: 1, order: order++ });
      if (list.length > cap) list.shift();
      kinds.set(kind, list);
    },
    get version() { return version; },
    lines() { return [...kinds.values()].flat().sort((a, b) => a.order - b.order); },
    provide(name, fn) { providers.set(name, fn); },
    state() {
      return [...providers].map(([name, fn]) => {
        try { return [name, String(fn() ?? '')]; } catch (error) { return [name, `could not be read: ${error?.message || error}`]; }
      });
    },
  };
}
export const pageLog = createPageLog();
// main.js writes the user's actions here without importing this module (its tests run it without one).
if (typeof globalThis !== 'undefined') globalThis.sopaCompanionLog = pageLog;

const row = (name, text) => `${name.padEnd(16)}${text}`;
/** Which count the page's numbers about a cell's RNA read (the "Counts" row): proseg's own, averaged over its random
 *  background draw; at an iteration counted for a pause, the hold, a finish or the end, the count proseg saves there (one
 *  draw: decision A); between counted iterations the cell readouts (the stage's tooltip and card, the UMAP's hover) show
 *  one draw of it, the frame's own, as proseg saves it (cellstats.mjs keptOf). */
export const COUNTS_TEXT = "Proseg's count, averaged over its random background draw; at an iteration counted for a pause, the hold, a finish or the end, the count proseg saves there (one random background draw); between counted iterations the cell readouts show one draw of it, as proseg saves it";
const plural = (n, word) => `${n.toLocaleString('en-GB')} ${word}${n === 1 ? '' : 's'}`;
/** What is on screen now, from main.js's state (window.sopaCompanion), in words. */
function showing(app) {
  const s = app?.state;
  if (!s) return "the page's state is not available (the page did not finish starting)";
  if (s.mode !== 'run' || !s.run) return s.mode === 'pick' ? `choosing a region on ${s.sample?.name || 'a sample'}` : 'no sample open';
  const frame = s.frames?.[s.current], newest = s.frames?.at?.(-1), view = app.stage?.orbit?.enabled ? '3-D' : '2-D';
  const where = frame ? `iteration ${frame.iter}${frame === newest ? ' (the newest)' : newest ? ` (the newest is ${newest.iter})` : ''}` : 'no iteration yet';
  return `run ${s.run.id} (${s.run.sample_name || 'sample'}, ${s.run.state}), ${where}, ${s.live ? 'following new iterations' : 'looking back'}, ${view}`;
}
/** The "Counts" row for a proseg run on screen: COUNTS_TEXT, with the run's reason when it has none (the engine's
 *  proseg_counts.reason: no_counts, old_proseg) or the newest counted iteration; null for no run or a stain run. */
function counts(app) {
  const s = app?.state, run = s?.run;
  if (s?.mode !== 'run' || !run || ['stain', 'patches'].includes(run.kind)) return null;
  const known = run.proseg_counts, reason = typeof known?.reason === 'string' && known.reason ? known.reason : null;
  const newest = Number.isInteger(known?.newest) ? s.frames?.[known.newest]?.iter ?? null : null;
  return COUNTS_TEXT + (reason ? `; none for this run (${reason})` : Number.isFinite(newest) ? `; the newest counted iteration ${newest}` : '');
}
function mixing(app) {
  const view = app?.qc?.view;
  if (!view || app?.state?.mode !== 'run') return null;
  if (view.enabled === false) return 'off';
  const summary = view.summary, shown = summary?.shown ?? summary?.per100;
  return `${view.enabled ? 'on' : 'not known yet'} (${view.status || '-'})` + (shown != null ? `, ${shown} in 100 look mixed at iteration ${summary.iter}` : '');
}

/** The page's part of the log: its set-up, the states handed to it and its lines (oldest first). */
export function pageText({ win = globalThis, log = pageLog, now = Date.now() } = {}) {
  const app = win.sopaCompanion, nav = win.navigator || {}, screen = win.screen || {};
  const origin = win.performance?.timeOrigin;
  const address = String(win.location?.pathname ?? '') + String(win.location?.search ?? '').replace(/([?&](?:token|code)=)[^&]*/gi, '$1[redacted]');
  const out = ['Page (this window)', '==================',
    row('Written', `${localTime(now)} by the page`),
    ...(Number.isFinite(origin) ? [row('Open since', `${localTime(origin)} (${duration(now - origin)})`)] : []),
    row('Address', address || '-'),
    row('Browser', nav.userAgent || '-'),
    row('Screen', `${screen.width ?? '?'} x ${screen.height ?? '?'}, window ${win.innerWidth ?? '?'} x ${win.innerHeight ?? '?'}, pixel ratio ${win.devicePixelRatio ?? 1}`),
    ...(app?.stage?.renderer ? [row('Graphics', app.stage.renderer)] : []),
    row('Showing', showing(app))];
  const counted = counts(app);
  if (counted) out.push(row('Counts', counted));
  const qc = mixing(app);
  if (qc) out.push(row('Mixing check', qc));
  for (const [name, text] of log.state()) if (text) out.push(row(name, text));
  const lines = log.lines();
  out.push('', LINES_TITLE, '-'.repeat(74));
  if (!lines.length) out.push('  nothing yet');
  for (const l of lines) out.push(`  ${clock(l.t)}  ${l.kind.padEnd(8)}  ${l.text}${l.n > 1 ? ` (${plural(l.n, 'time')}, last ${clock(l.last)})` : ''}`);
  return out.join('\n') + '\n';
}

/** What the page noted, in one line for the summary: its notices, failed requests, script errors and clusters it could
 *  not show, with the newest of them (`nothing went wrong` when there is none). */
export function pageNoted(log = pageLog) {
  const counts = { request: 0, error: 0, clusters: 0, notice: 0 };
  let newest = null;
  for (const l of log.lines()) {
    if (!(l.kind in counts) || (l.kind === 'clusters' && !/failed|cannot/i.test(l.text))) continue;
    counts[l.kind] += l.n;
    if (l.kind !== 'notice' || !newest) newest = l;
  }
  const parts = [[counts.request, 'failed request'], [counts.error, 'script error'], [counts.clusters, 'clusters problem'],
    [counts.notice, 'notice']].filter(([n]) => n).map(([n, word]) => plural(n, word));
  if (!parts.length) return 'nothing went wrong';
  const text = newest.text.length > 120 ? newest.text.slice(0, 119) + '…' : newest.text;
  return `${parts.join(', ')} (the newest: ${text})`;
}
/** The whole log: the engine's text with the page's "Page noted" row put into its summary, then the page's part, then
 *  the engine's appendix (the run's every Clusters & UMAP result) at the very end. Without an engine summary the row
 *  heads the page's own part instead. */
export function assemble(engine, page, noted) {
  let head = engine, appendix = '';
  const at = engine.indexOf(`\n${APPENDIX}`);
  if (at >= 0) { head = engine.slice(0, at).trimEnd(); appendix = engine.slice(at + 1).trimEnd(); }
  const summary = head.indexOf(`\n${SUMMARY}\n${'-'.repeat(SUMMARY.length)}\n`);
  if (summary >= 0) {
    const end = head.indexOf('\n\n', summary + 1), line = row('Page noted', noted);
    head = end < 0 ? `${head}\n${line}` : `${head.slice(0, end)}\n${line}${head.slice(end)}`;
  } else page = withNoted(page, noted);
  return `${head}\n\n${page.trimEnd()}\n${appendix ? `\n${appendix}\n` : ''}`;
}
/** The page's part with its "Page noted" row under its own title and that title's underline. */
function withNoted(page, noted) {
  const line = row('Page noted', noted), i = page.indexOf('\n', page.indexOf('\n') + 1);
  return i < 0 ? `${page}\n${line}` : `${page.slice(0, i + 1)}${line}\n${page.slice(i + 1)}`;
}

/** The engine's part (text), or a line saying why there is none: the log is saved either way. */
export const PAGE_ONLY = "only the page's part follows.";   // how engineText ends when the engine wrote nothing
export async function engineText(win, run) {
  const get = path => win.fetch(path, { credentials: 'same-origin', cache: 'no-store' });
  try {
    let response = await get(run ? `/api/v1/runs/${encodeURIComponent(run)}/log` : '/api/v1/log');
    if (run && response.status === 404) response = await get('/api/v1/log');   // the run was deleted meanwhile
    if (response.ok) return (await response.text()).trimEnd();
    return `SoPA companion log\n==================\nThe engine did not write its part (HTTP ${response.status}): ${PAGE_ONLY}`;
  } catch (error) {
    return `SoPA companion log\n==================\nThe engine could not be reached (${error?.message || error}): ${PAGE_ONLY}`;
  }
}

/** Writes `text` where the user says: the browser's save dialog when it has one (cancelling it saves nothing), else
 *  (or when the browser refuses it, e.g. the click is too long ago) our dialog, whose button saves it as a download.
 *  -> {saved, name?, cancelled?}. */
export async function saveText(text, name, { win = globalThis, doc = globalThis.document } = {}) {
  const blob = () => new win.Blob([text], { type: 'text/plain;charset=utf-8' });
  const pick = async () => {
    const handle = await win.showSaveFilePicker({ suggestedName: name, id: SAVE_ID, startIn: 'downloads',
      types: [{ description: 'Text file', accept: { 'text/plain': ['.txt'] } }] });
    const writable = await handle.createWritable();
    await writable.write(blob()); await writable.close();
    return { saved: true, name: handle.name || name };
  };
  if (typeof win.showSaveFilePicker === 'function') {
    try { return await pick(); } catch (error) { if (error?.name === 'AbortError') return { saved: false, cancelled: true }; }
  }
  return askToSave({ win, doc, name, lines: text.split('\n').length, pick, blob });
}

let dialog = null;
/** Our own dialog: "Save the log" with the file's name and size, Close and Save log…; resolves with what was done. */
function askToSave({ win, doc, name, lines, pick, blob }) {
  if (!dialog) {
    dialog = doc.createElement('dialog'); dialog.className = 'modal'; dialog.id = 'log-dialog';
    dialog.setAttribute('aria-labelledby', 'log-dialog-title');
    const title = doc.createElement('h2'); title.id = 'log-dialog-title'; title.textContent = 'Save the log';
    const about = doc.createElement('p'); about.className = 'muted small';
    const error = doc.createElement('p'); error.className = 'error'; error.hidden = true;
    const actions = doc.createElement('div'); actions.className = 'modal-actions';
    const close = doc.createElement('button'); close.type = 'button'; close.textContent = 'Close';
    const save = doc.createElement('button'); save.type = 'button'; save.className = 'primary'; save.textContent = 'Save log…';
    actions.append(close, save); dialog.append(title, about, error, actions); doc.body.append(dialog);
    dialog.parts = { about, error, close, save };
  }
  const { about, error, close, save } = dialog.parts;
  about.textContent = `${name} · ${plural(lines, 'line')}`;
  error.hidden = true;
  return new Promise(resolve => {
    const done = result => { dialog.onclose = null; if (dialog.open) dialog.close(); resolve(result); };
    close.onclick = () => done({ saved: false });
    dialog.onclose = () => resolve({ saved: false });                  // Esc
    save.onclick = async () => {
      if (typeof win.showSaveFilePicker === 'function') {
        try { return done(await pick()); } catch (failure) {
          if (failure?.name === 'AbortError') return;                   // back to our dialog: Close or try again
          error.textContent = `The browser could not save it there (${failure?.message || failure}). Try again.`; error.hidden = false;
          return;
        }
      }
      const url = win.URL.createObjectURL(blob()), link = doc.createElement('a');
      link.href = url; link.download = name; link.hidden = true; doc.body.append(link); link.click(); link.remove();
      win.setTimeout(() => win.URL.revokeObjectURL(url), 60000);
      done({ saved: true, name, download: true });
    };
    if (!dialog.open) dialog.showModal();
  });
}

const notify = (win, text) => win.dispatchEvent?.(new win.CustomEvent('fetch-notice', { detail: text }));
let busy = false;
/** The button's click: build the log, ask where to save it, say what happened. */
export async function exportLog({ win = globalThis, doc = globalThis.document, log = pageLog, button = null } = {}) {
  if (busy) return null;
  busy = true; if (button) { button.disabled = true; button.setAttribute('aria-busy', 'true'); }
  try {
    const s = win.sopaCompanion?.state;
    const run = s?.run?.id || new URLSearchParams(win.location?.search || '').get('run') || null;
    const sample = s?.run?.sample_name || s?.sample?.name || null;
    const now = Date.now(), name = logFileName({ sample, run, date: new Date(now) });
    const engine = await engineText(win, run);
    const text = assemble(engine, pageText({ win, log, now }), pageNoted(log));
    const result = await saveText(text, name, { win, doc });
    // The general "Request failed" notice stays quiet for the log's requests (api.js): this one says it all.
    const partial = engine.endsWith(PAGE_ONLY) ? ' (the engine did not answer: only the page\'s part is in it)' : '';
    if (result.saved) notify(win, (result.download ? `Log saved to your downloads: ${result.name}` : `Log saved: ${result.name}`) + partial);
    return result;
  } catch (error) {
    notify(win, `The log could not be saved: ${error?.message || error}`);
    return { saved: false, error };
  } finally {
    busy = false; if (button) { button.disabled = false; button.removeAttribute('aria-busy'); }
  }
}

/** What a load the page's security policy refused names (a CSP report's blockedURI), in a few words: 'inline', 'eval'
 *  and the like as they are, else the address's origin alone, never its path or query (a refused attempt to send the
 *  page's key somewhere is not written down with it). */
export function blockedOrigin(uri) {
  const text = String(uri ?? '');
  try { const { origin, protocol } = new URL(text); return origin && origin !== 'null' ? origin : protocol; }
  catch { return /^[a-z][\w-]{0,30}$/i.test(text) ? text : 'unknown'; }
}

/** What the page shows the user and what goes wrong in it, for the log: the notices (main.js writes them into #notice),
 *  script errors, failed requests (Resource Timing: the requests themselves are not wrapped) and what the page's
 *  security policy refused, in every browser. Installed once. */
export function watchPage({ win = globalThis, doc = globalThis.document, log = pageLog } = {}) {
  if (win.__plvLogWatch) return;
  win.__plvLogWatch = true;
  const box = doc.getElementById?.('notice');
  if (box && typeof win.MutationObserver === 'function') {
    let last = '';
    new win.MutationObserver(() => { const text = box.hidden ? '' : box.textContent; if (text && text !== last) log.add(KINDS.notice, text); last = text; })
      .observe(box, { childList: true, characterData: true, subtree: true, attributes: true, attributeFilter: ['hidden'] });
  }
  win.addEventListener?.('error', e => log.add(KINDS.error, `${e.message || 'script error'}${e.filename ? ` (${String(e.filename).replace(/^.*\/src\//, 'src/')}:${e.lineno}:${e.colno})` : ''}`));
  win.addEventListener?.('unhandledrejection', e => log.add(KINDS.error, `unhandled: ${e.reason?.message || e.reason}`));
  doc?.addEventListener?.('securitypolicyviolation', e => log.add(KINDS.error,
    `blocked by the page's security policy: ${e.effectiveDirective || e.violatedDirective} ${blockedOrigin(e.blockedURI)}`));
  if (typeof win.PerformanceObserver === 'function') {
    try {
      new win.PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
          const path = new URL(entry.name, 'http://page/').pathname;
          // The log's own upload is not something the page did (a refused one would only note itself, again and again).
          if (entry.responseStatus >= 400 && path !== PAGE_LOG_URL) log.add(KINDS.request, `HTTP ${entry.responseStatus} ${path}`);
        }
      }).observe({ type: 'resource', buffered: true });
    } catch { /* an older browser: no request lines */ }
  }
}

// ---- the page's log for a crash report ------------------------------------------------------------------------------
// The engine keeps the page's log (POST /api/v1/page-log {text, session, at}: logs/page-logs/<session>.txt, the newest
// also as logs/page-log.txt) and puts it into the report it writes when a session ended unexpectedly (crashlog.py;
// crash-report.mjs offers that report at the next start). The text is the page's part as the Log button saves it without
// an engine part (pageLogText). It is sent as the page starts, then at most every UPLOAD_MS while the page's log changed
// (a line noted or counted again; the states it reads, like the iteration on screen, go along with it), made when the
// page has a moment with nothing to do (a full log takes some 20 ms); once more as the page is hidden, and nothing while
// it stays hidden (as main.js: a hidden page fetches nothing); and as the page goes, its goodbye (pagehide: a keepalive
// request with the page's key, sent at once (page-key.mjs send), since a beacon cannot carry the key; its JSON body is
// cut to BEACON_BYTES, with "pagehide": true: the engine then knows the page closed cleanly, and keeps the older lines
// of a longer log it has). `session` and `at` (when it was sent, ISO) name it. Quiet: nothing on the page says whether
// it arrived (api.js).
export const PAGE_LOG_URL = '/api/v1/page-log';
export const UPLOAD_MS = 10000;
export const UPLOAD_BYTES = 500000;     // at most this much JSON per upload (the engine keeps the last 512 KB of text)
export const BEACON_BYTES = 32000;      // the goodbye's: a closing page may send 64 KB in all, and the answer to the close
                                        // popup and telemetry (16 KB) go then too
export const IDLE_WAIT_MS = 2000;       // an upload waits this long at most for a moment with nothing to do

function sessionId() {
  try { return globalThis.crypto.randomUUID(); } catch { return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`; }
}
/** This page's session: an id and when the page started (ISO). Its log's uploads and its alive flag (crash-report.mjs)
 *  carry it. */
export const pageSession = Object.freeze({ session: sessionId(), at: new Date().toISOString() });
/** Which showing of this page it is: 1, then one more each time the browser brings the page back from its cache
 *  (pageshow persisted: api.js LiveSocket.back). Its socket's address and its goodbyes name it, so the engine takes a
 *  goodbye for the showing that said it only: one that reaches it after the page came back changes nothing. */
export const pageShowing = { visit: 1 };

/** The page's part with its "Page noted" row, as the Log button saves it when the engine wrote nothing. */
export function pageLogText({ win = globalThis, log = pageLog, now = Date.now() } = {}) {
  return withNoted(pageText({ win, log, now }), pageNoted(log));
}

/** `text` (pageText's shape) in at most `max` characters: its set-up and states (when they take at most half), a line
 *  saying how many older lines were left out, then the newest lines that fit. */
export function fitText(text, max) {
  text = String(text ?? '');
  if (text.length <= max) return text;
  if (max < 120) return text.slice(text.length - Math.max(0, max));
  const title = text.indexOf(`\n${LINES_TITLE}\n`), rule = title < 0 ? -1 : text.indexOf('\n', title + LINES_TITLE.length + 2);
  const head = rule >= 0 && rule + 1 <= max / 2 ? text.slice(0, rule + 1) : '';
  const lines = text.slice(head.length).split('\n');
  let size = head.length + 40, from = lines.length;                     // 40: room for the line saying what was left out
  while (from > 0 && size + lines[from - 1].length + 1 <= max) size += lines[--from].length + 1;
  return `${head}  … ${from} older ${from === 1 ? 'line' : 'lines'} left out\n${lines.slice(from).join('\n')}`;
}

const utf8 = text => new TextEncoder().encode(text).length;
/** The upload's JSON body {text, session, at, (pagehide)} in at most `limit` bytes: a longer log loses its oldest lines. */
export function uploadBody(text, meta, limit = UPLOAD_BYTES) {
  let body = JSON.stringify({ text, ...meta }), size = utf8(body), keep = text.length;
  while (size > limit && keep > 0) {
    keep = Math.floor(keep * Math.min(0.9, limit / size));
    body = JSON.stringify({ text: fitText(text, keep), ...meta }); size = utf8(body);
  }
  return body;
}

/** Sends the page's log to the engine as above, from now on -> {tick, beacon, stop}: tick() sends it if it changed since
 *  it was last taken (false: nothing sent), beacon() as the page goes, with `keys` (page-key.mjs) and the showing that
 *  says goodbye (`showing`); the page wires them itself. An engine without the endpoint (404) is not asked again. */
export function uploadPageLog({ win = globalThis, doc = globalThis.document, log = pageLog, session = pageSession,
  every = UPLOAD_MS, keys = pageKey, showing = pageShowing } = {}) {
  let sent = -1, sending = false, stopped = false, leaving = false;
  const body = (limit, extra) => {
    const now = Date.now();
    return uploadBody(pageLogText({ win, log, now }), { session: session.session, at: new Date(now).toISOString(), ...extra }, limit);
  };
  const post = data => win.fetch(PAGE_LOG_URL, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: data });
  const idle = () => new Promise(resolve => {
    try { if (typeof win.requestIdleCallback === 'function') return void win.requestIdleCallback(() => resolve(), { timeout: IDLE_WAIT_MS }); }
    catch { /* none: now */ }
    resolve();
  });
  /** `hiding`: the page is being hidden (sent at once); else a hidden page that has sent its log sends nothing. A page
   *  that said its goodbye sends nothing more (a page that goes is hidden after its pagehide). */
  async function tick(hiding = false) {
    if (stopped || leaving || sending || log.version === sent || (doc?.hidden && hiding !== true && sent !== -1)) return false;
    sending = true;
    try {
      if (hiding !== true) await idle();
      const version = log.version;
      const response = await post(body(UPLOAD_BYTES));
      if (response.ok) sent = version;
      else if (response.status === 404) stop();
      return !!response.ok;
    } catch { return false; }                          // the engine is away: the next tick tries again
    finally { sending = false; }
  }
  function beacon(event) {
    if (stopped) return false;
    leaving = true;
    log.add(KINDS.page, event?.persisted ? 'the page was put aside (the browser may bring it back)' : 'the page closed or reloaded');
    return !!keys?.send(PAGE_LOG_URL, body(BEACON_BYTES, { pagehide: true, visit: showing.visit }));
  }
  const timer = win.setInterval?.(tick, every), first = win.setTimeout?.(tick, 0);   // first: once main.js has run
  for (const t of [timer, first]) t?.unref?.();                          // (Node, in tests: never holds the process)
  function stop() { stopped = true; win.clearInterval?.(timer); }
  win.addEventListener?.('pagehide', beacon);
  // Brought back after its goodbye: the next upload tells the engine it is open again.
  win.addEventListener?.('pageshow', event => { if (event?.persisted) { leaving = false; log.add(KINDS.page, 'the page was brought back'); } });
  doc?.addEventListener?.('visibilitychange', () => { if (doc.hidden) tick(true); });
  return { tick, beacon, stop };
}

/** The header's button (index.html #log-button), wired once. */
export function installLogButton(button, options = {}) {
  if (!button || button.dataset.logReady) return false;
  button.dataset.logReady = '1';
  watchPage(options);
  button.addEventListener('click', () => { exportLog({ ...options, button }); });
  return true;
}

if (typeof document !== 'undefined' && typeof window !== 'undefined' && document.getElementById) {
  installLogButton(document.getElementById('log-button'));
  uploadPageLog();
}
