import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { domFixture, settle } from './dom-fixture.mjs';
import { createPageLog, logFileName, namePart, pageText, engineText, saveText, exportLog, watchPage, installLogButton,
  SAVE_ID, PAGE_LINES, LINE_CHARS, assemble, pageNoted, APPENDIX, blockedOrigin, COUNTS_TEXT } from '../src/log-export.mjs';

// The header's Log button (log-export.mjs): the page's own lines, the text it adds to the engine's log, and saving only
// where and when the user says (the browser's save dialog, else our dialog's button and a download).
const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const when = new Date(2026, 9, 18, 19, 3, 4, 123);

test('the button sits in the header, top right, in plain words, and works on its own', () => {
  const header = html.slice(html.indexOf('<header'), html.indexOf('</header>'));
  assert.match(header, /<button id="log-button" type="button" title="Save a text log of this session and the open run, to look into a problem">Log<\/button>\s*<span id="connection"/);
  assert.ok(header.indexOf('log-button') > header.indexOf('runs-menu'), 'after the Runs menu, at the right end');
  assert.match(html, /<script type="module" src="\.\/src\/log-export\.mjs"><\/script>/, 'its own module: it works when the page did not start');
  // Its one import is the tab's sign-in, the first module the page loads, which imports nothing.
  assert.match(html, /<script type="module" src="\.\/src\/page-key\.mjs"><\/script>/);
  assert.ok(html.indexOf('src="./src/page-key.mjs"') < html.indexOf('<script type="module" src="./src/storage-boot.mjs">'), 'the first module script');
  const source = readFileSync(new URL('../src/log-export.mjs', import.meta.url), 'utf8');
  assert.deepEqual(source.match(/^import .*$/gm), ["import { pageKey } from './page-key.mjs';"]);
});

test('file names: sopa-companion-<sample>-<run>-<time>.txt, in characters every file system takes', () => {
  assert.equal(logFileName({ sample: 'sample-1', run: '20261018-140000-sample-1', date: when }), 'sopa-companion-sample-1-20261018-140000-20261018-190304.txt',
    'a run id ends with its sample: said once');
  assert.equal(logFileName({ sample: 'sample-1 crop.zarr', run: '20261018-140000-sample-1-crop-zarr-2', date: when }),
    'sopa-companion-sample-1-crop.zarr-20261018-140000-sample-1-crop-zarr-2-20261018-190304.txt', 'a second run of that second: kept whole');
  assert.equal(logFileName({ sample: 'tissue / crop #2', run: null, date: when }), 'sopa-companion-tissue-crop-2-20261018-190304.txt');
  assert.equal(logFileName({ date: when }), 'sopa-companion-20261018-190304.txt');
  assert.equal(namePart('../../etc/passwd'), 'etc-passwd'); assert.equal(namePart(undefined), '');
});

test('the page log keeps the newest lines of each kind, counts a repeated line and reads the states handed to it', () => {
  let t = 1000;
  const log = createPageLog({ now: () => t, cap: 3 });
  for (let i = 0; i < 5; i++) { t += 10; log.add('clusters', `result ${i}`); }
  log.add('notice', 'Request failed (500). Please try again.'); t += 5;
  log.add('notice', '  Request failed (500).\n Please try again. '); log.add('notice', '');
  assert.deepEqual(log.lines().map(l => l.text), ['result 2', 'result 3', 'result 4', 'Request failed (500). Please try again.']);
  assert.deepEqual([log.lines().at(-1).n, log.lines().at(-1).last], [2, 1055], 'a repeat is counted, not kept twice');
  log.provide('Clusters & UMAP', () => 'on; showing iteration 10'); log.provide('Broken', () => { throw new Error('no view'); });
  assert.deepEqual(log.state(), [['Clusters & UMAP', 'on; showing iteration 10'], ['Broken', 'could not be read: no view']]);
  assert.equal(PAGE_LINES, 600);
  // A line is cut at LINE_CHARS (a failed request's message may carry a whole server body, repeated with small changes:
  // the saved file stayed readable only because the engine cuts its own lines).
  log.add('request', 'HTTP 500 ' + 'x'.repeat(5000));
  assert.equal(log.lines().at(-1).text.length, LINE_CHARS); assert.ok(log.lines().at(-1).text.endsWith('…')); assert.equal(LINE_CHARS, 400);
});

function page({ run = { id: 'r1', sample_name: 'sample-1', state: 'running' }, search = '?run=r1&token=secret', fetch, picker } = {}) {
  const dom = domFixture(), events = [], { document, Element } = dom;
  Element.prototype.remove = function () { const p = this.parentElement; if (p) p.childNodes = p.childNodes.filter(n => n !== this); this.parentElement = null; };
  Element.prototype.removeAttribute = function (key) { this.attributes.delete(key); };
  const urls = [];
  const win = {
    document, location: { pathname: '/', search }, navigator: { userAgent: 'Chrome/149 test' }, screen: { width: 2560, height: 1440 },
    innerWidth: 1900, innerHeight: 1100, devicePixelRatio: 1, performance: { timeOrigin: when.getTime() - 3723000 },
    sopaCompanion: { state: { mode: 'run', run, frames: [{ iter: 0 }, { iter: 7 }], current: 1, live: true }, stage: { renderer: 'ANGLE (NVIDIA)', orbit: { enabled: false } },
      qc: { view: { enabled: false } } },
    fetch: fetch || (async url => ({ ok: true, status: 200, text: async () => `SoPA companion log\n==================\nengine part for ${url}\n` })),
    Blob: class { constructor(parts, o) { this.text = parts.join(''); this.type = o.type; } },
    URL: { createObjectURL: blob => { urls.push(blob); return `blob:${urls.length}`; }, revokeObjectURL() {} },
    setTimeout: () => 0, CustomEvent: class { constructor(type, o) { this.type = type; this.detail = o.detail; } },
    dispatchEvent: e => events.push([e.type, e.detail]),
  };
  if (picker) win.showSaveFilePicker = picker;
  return { dom, win, document, events, urls };
}
function savingPicker(saved, { name, fail = [] } = {}) {
  return async options => {
    saved.push({ options });
    const error = fail.shift();
    if (error) throw Object.assign(new Error(error), { name: error });
    return { name: name ?? options.suggestedName, createWritable: async () => ({ write: async blob => { saved.at(-1).blob = blob; }, close: async () => { saved.at(-1).closed = true; } }) };
  };
}

test("the page's part: what is on screen, the browser and display, the states and the lines, never the session's secret", () => {
  const { win } = page(), log = createPageLog({ now: () => when.getTime() });
  log.add('action', 'resume (at iteration 0)'); log.add('action', 'resume (at iteration 0)');
  log.provide('Clusters & UMAP', () => 'on; showing the clusters of iteration 7, 12 clusters');
  const text = pageText({ win, log, now: when.getTime() });
  assert.match(text, /^Page \(this window\)\n==================\nWritten         2026-10-18 19:03:04 by the page\nOpen since      2026-10-18 18:01:01 \(1 h 2 min\)\n/);
  assert.match(text, /\nAddress         \/\?run=r1&token=\[redacted\]\n/); assert.doesNotMatch(text, /secret/);
  assert.match(text, /\nBrowser         Chrome\/149 test\nScreen          2560 x 1440, window 1900 x 1100, pixel ratio 1\nGraphics        ANGLE \(NVIDIA\)\n/);
  // Iterations, as on screen and in the engine's part: never a snapshot number.
  // ... and which count its numbers read (a proseg run's: proseg's own, averaged over its random background draw).
  assert.match(text, /\nShowing         run r1 \(sample-1, running\), iteration 7 \(the newest\), following new iterations, 2-D\nCounts          Proseg's count, averaged over its random background draw; at an iteration counted for a pause, the hold, a finish or the end, the count proseg saves there \(one random background draw\); between counted iterations the cell readouts show one draw of it, as proseg saves it\nMixing check    off\n/);
  assert.match(text, /\nClusters & UMAP on; showing the clusters of iteration 7, 12 clusters\n/);
  assert.match(text, /\nWhat the user did and the page noted \(oldest first; this computer's clock\)\n/);
  assert.match(text, /\n  19:03:04\.123  action    resume \(at iteration 0\) \(2 times, last 19:03:04\.123\)\n$/);
  win.sopaCompanion.state.current = 0;
  assert.match(pageText({ win, log, now: when.getTime() }), /\nShowing         run r1 \(sample-1, running\), iteration 0 \(the newest is 7\), following/);
  const broken = pageText({ win: { location: {}, performance: {} }, log: createPageLog(), now: when.getTime() });
  assert.match(broken, /Showing         the page's state is not available \(the page did not finish starting\)/);
  assert.doesNotMatch(broken, /\nCounts /, 'no run: no count named');
  assert.match(broken, /  nothing yet/);
});

test("the engine's part: the run's log, the session's when the run is gone, and a plain line when the engine does not answer", async () => {
  const asked = [], reply = (status, body = '') => ({ ok: status === 200, status, text: async () => body });
  const win = { fetch: async (url, o) => { asked.push([url, o.cache]); return url.includes('/runs/') ? reply(404) : reply(200, 'session log\n\n'); } };
  assert.equal(await engineText(win, 'r 1'), 'session log');
  assert.deepEqual(asked, [['/api/v1/runs/r%201/log', 'no-store'], ['/api/v1/log', 'no-store']]);
  assert.match(await engineText({ fetch: async () => reply(500) }, null), /The engine did not write its part \(HTTP 500\): only the page's part follows\./);
  assert.match(await engineText({ fetch: async () => { throw new TypeError('Failed to fetch'); } }, 'r1'), /The engine could not be reached \(Failed to fetch\)/);
});

test("saving: the browser's own dialog with the suggested name; cancelling it saves nothing", async () => {
  const saved = [], { win, document } = page({ picker: savingPicker([]) });
  win.showSaveFilePicker = savingPicker(saved, { name: 'my-log.txt' });
  assert.deepEqual(await saveText('the log', 'sopa-companion-x.txt', { win, doc: document }), { saved: true, name: 'my-log.txt' });
  assert.deepEqual(saved[0].options, { suggestedName: 'sopa-companion-x.txt', id: SAVE_ID, startIn: 'downloads',
    types: [{ description: 'Text file', accept: { 'text/plain': ['.txt'] } }] });
  assert.equal(saved[0].blob.text, 'the log'); assert.equal(saved[0].blob.type, 'text/plain;charset=utf-8'); assert.equal(saved[0].closed, true);
  win.showSaveFilePicker = savingPicker(saved, { fail: ['AbortError'] });
  assert.deepEqual(await saveText('the log', 'sopa-companion-x.txt', { win, doc: document }), { saved: false, cancelled: true });
  assert.equal(document.getElementById('log-dialog'), null, 'no dialog of ours after a cancel');
});

test('saving when the browser refuses its dialog (the click too long ago) or has none: our dialog, and only its button saves', async () => {
  const saved = [], { win, document, urls } = page();
  win.showSaveFilePicker = savingPicker(saved, { fail: ['SecurityError'] });
  const pending = saveText('line 1\nline 2', 'sopa-companion-x.txt', { win, doc: document });
  await settle();
  const dialog = document.getElementById('log-dialog');
  assert.equal(dialog.open, true); assert.equal(dialog.children[0].textContent, 'Save the log');
  assert.equal(dialog.children[1].textContent, 'sopa-companion-x.txt · 2 lines'); assert.equal(saved.length, 1, 'nothing written yet');
  const [close, save] = dialog.children[3].children;
  assert.deepEqual([close.textContent, save.textContent], ['Close', 'Save log…']);
  save.click();                                                         // the user's click: the browser's dialog again
  assert.deepEqual(await pending, { saved: true, name: 'sopa-companion-x.txt' }); assert.equal(saved.at(-1).blob.text, 'line 1\nline 2');
  assert.equal(dialog.open, false);
  delete win.showSaveFilePicker;                                        // a browser without that dialog: a download
  const again = saveText('text', 'sopa-companion-y.txt', { win, doc: document });
  await settle(); assert.equal(dialog.open, true); assert.equal(urls.length, 0, 'nothing saved before the click');
  const appended = [], append = document.body.append.bind(document.body);
  document.body.append = (...nodes) => { appended.push(...nodes); append(...nodes); };
  save.click();
  assert.deepEqual(await again, { saved: true, name: 'sopa-companion-y.txt', download: true });
  assert.equal(urls[0].text, 'text'); assert.equal(appended[0].tagName, 'A'); assert.equal(appended[0].download, 'sopa-companion-y.txt');
  assert.equal(appended[0].href, 'blob:1'); assert.ok(!document.body.childNodes.includes(appended[0]), 'the link is gone again');
  const closed = saveText('text', 'sopa-companion-z.txt', { win, doc: document });
  await settle(); close.click();
  assert.deepEqual(await closed, { saved: false }); assert.equal(urls.length, 1);
});

test('the click: the engine and page parts in one text, saved under the suggested name, with a notice; one at a time', async () => {
  const saved = [], { win, document, events } = page({ picker: null });
  let release;
  win.showSaveFilePicker = async options => { await new Promise(r => { release = r; }); return savingPicker(saved)(options); };
  const button = document.createElement('button');
  const first = exportLog({ win, doc: document, log: createPageLog(), button });
  assert.equal(button.disabled, true); assert.equal(button.getAttribute('aria-busy'), 'true');
  assert.equal(await exportLog({ win, doc: document, button }), null, 'a second click while the first is on its way does nothing');
  await settle(); release();
  const result = await first;
  assert.equal(result.saved, true); assert.match(result.name, /^sopa-companion-sample-1-r1-\d{8}-\d{6}\.txt$/);
  const text = saved[0].blob.text;
  assert.match(text, /^SoPA companion log\n==================\nengine part for \/api\/v1\/runs\/r1\/log\n\nPage \(this window\)\n/);
  assert.equal(button.disabled, false); assert.equal(button.attributes.has('aria-busy'), false);
  assert.deepEqual(events, [['fetch-notice', `Log saved: ${result.name}`]]);
  const fresh = page({ run: null, search: '', picker: savingPicker(saved) });
  fresh.win.sopaCompanion.state = { mode: 'empty' };
  const none = await exportLog({ win: fresh.win, doc: fresh.document, log: createPageLog() });
  assert.match(none.name, /^sopa-companion-\d{8}-\d{6}\.txt$/); assert.match(saved.at(-1).blob.text, /engine part for \/api\/v1\/log\n/);
  // The engine does not answer: the page's part is saved all the same, and the notice says what is missing (api.js
  // raises no general notice for the log's requests).
  const silent = page({ fetch: async () => ({ ok: false, status: 500, text: async () => '' }), picker: savingPicker(saved) });
  const partial = await exportLog({ win: silent.win, doc: silent.document, log: createPageLog() });
  assert.equal(partial.saved, true); assert.match(saved.at(-1).blob.text, /only the page's part follows\.\n\nPage \(this window\)\n/);
  assert.deepEqual(silent.events, [['fetch-notice', `Log saved: ${partial.name} (the engine did not answer: only the page's part is in it)`]]);
});

test('the whole log: a short summary first with the page\'s own line in it, then the parts, every result at the end', () => {
  // 75 % of a saved log were per-result rows, listed twice, before the page's own notes, with no summary: the
  // engine writes a Summary and an appendix of results; the page adds its line to the one and leaves the
  // other at the very end.
  const engine = ['SoPA companion log', '==================', 'About           run r1', '', 'Summary', '-------',
    'Sample          sample-1: the full sample', 'Engine noted    nothing went wrong', '', 'Engine', '------', 'Version         0.1.0',
    '', APPENDIX, '='.repeat(APPENDIX.length), '  iteration 1: 12 clusters, 3,404 cells · load 3 ms'].join('\n');
  const log = createPageLog();
  assert.equal(pageNoted(log), 'nothing went wrong');
  log.add('notice', 'Folder path copied.'); log.add('request', 'HTTP 500 /api/v1/runs/r1/frames/12'); log.add('request', 'HTTP 500 /api/v1/runs/r1/frames/12');
  log.add('clusters', 'Clusters & UMAP failed: GPU out of memory'); log.add('action', 'pause (at iteration 40)'); log.add('shown', 'iteration 4 shows ...');
  assert.equal(pageNoted(log), '2 failed requests, 1 clusters problem, 1 notice (the newest: Clusters & UMAP failed: GPU out of memory)');
  const text = assemble(engine, 'Page (this window)\n==================\nWritten         now\n', pageNoted(createPageLog()));
  assert.equal(text, ['SoPA companion log', '==================', 'About           run r1', '', 'Summary', '-------',
    'Sample          sample-1: the full sample', 'Engine noted    nothing went wrong', 'Page noted      nothing went wrong', '', 'Engine',
    '------', 'Version         0.1.0', '', 'Page (this window)', '==================', 'Written         now', '', APPENDIX,
    '='.repeat(APPENDIX.length), '  iteration 1: 12 clusters, 3,404 cells · load 3 ms', ''].join('\n'));
  // Without the engine's part (it did not answer): the line heads the page's own part.
  const alone = assemble("SoPA companion log\n==================\nThe engine could not be reached (x): only the page's part follows.",
    'Page (this window)\n==================\nWritten         now\n', 'nothing went wrong');
  assert.match(alone, /follows\.\n\nPage \(this window\)\n==================\nPage noted      nothing went wrong\nWritten         now\n$/);
});

test('what the page shows the user goes into the log: notices, script errors and failed requests', () => {
  const { win, document } = page(), log = createPageLog(), listeners = {}, observers = [];
  document.body.innerHTML = '<div id="notice" class="notice" role="status" hidden></div>';
  Object.assign(win, {
    addEventListener: (type, fn) => { listeners[type] = fn; },
    MutationObserver: class { constructor(fn) { this.fn = fn; observers.push(this); } observe(node, o) { this.node = node; this.options = o; } },
    PerformanceObserver: class { constructor(fn) { this.fn = fn; observers.push(this); } observe(o) { this.options = o; } },
  });
  watchPage({ win, doc: document, log });
  watchPage({ win, doc: document, log });                                // once
  assert.equal(observers.length, 2);
  const box = document.getElementById('notice'), [mutations, resources] = observers;
  assert.deepEqual(resources.options, { type: 'resource', buffered: true });
  box.hidden = false; box.textContent = 'The graphics context was lost; restoring the view…'; mutations.fn();
  mutations.fn();                                                        // the same notice again: once
  box.hidden = true; mutations.fn();
  listeners.error({ message: 'TypeError: x is undefined', filename: 'http://127.0.0.1:8777/src/main.js', lineno: 12, colno: 5 });
  listeners.unhandledrejection({ reason: new Error('boom') });
  resources.fn({ getEntries: () => [{ name: 'http://127.0.0.1:8777/api/v1/runs/r1/frames/12', responseStatus: 500 },
    { name: 'http://127.0.0.1:8777/src/main.js', responseStatus: 200 }, { name: 'http://127.0.0.1:8777/api/v1/runs/r1/frames/12', responseStatus: 500 }] });
  assert.deepEqual(log.lines().map(l => [l.kind, l.text, l.n]), [
    ['notice', 'The graphics context was lost; restoring the view…', 1],
    ['error', 'TypeError: x is undefined (src/main.js:12:5)', 1], ['error', 'unhandled: boom', 1],
    ['request', 'HTTP 500 /api/v1/runs/r1/frames/12', 2]]);
});

test("what the page's security policy refused goes into the log, in every browser: the directive and the address's origin alone", () => {
  const { win } = page(), log = createPageLog(), listeners = {};
  const doc = { addEventListener: (type, fn) => { listeners[type] = fn; } };
  Object.assign(win, { addEventListener() {} });
  delete win.__plvLogWatch;
  watchPage({ win, doc, log });
  listeners.securitypolicyviolation({ effectiveDirective: 'style-src-elem', violatedDirective: 'style-src-elem', blockedURI: 'inline' });
  listeners.securitypolicyviolation({ violatedDirective: 'connect-src', blockedURI: 'https://x.example/a?k=secret' });
  listeners.securitypolicyviolation({ effectiveDirective: 'script-src', blockedURI: 'eval' });
  assert.deepEqual(log.lines().map(l => [l.kind, l.text]), [
    ['error', "blocked by the page's security policy: style-src-elem inline"],
    ['error', "blocked by the page's security policy: connect-src https://x.example"],
    ['error', "blocked by the page's security policy: script-src eval"]]);
  assert.ok(log.lines().every(l => !/secret|\/a\b|\?/.test(l.text)), 'never a path or a query: a key sent away is not written down');
  // Only an origin, or a word the browser gives in place of an address.
  for (const [uri, said] of [['https://x.example:8443/p/q?r=s#t', 'https://x.example:8443'], ['blob:http://127.0.0.1:8799/0b5e', 'http://127.0.0.1:8799'],
    ['data:image/png;base64,AAAA', 'data:'], ['wasm-eval', 'wasm-eval'], ['data', 'data'], ['', 'unknown'], [undefined, 'unknown'],
    ['a/b?key=1', 'unknown'], ['x'.repeat(40), 'unknown']]) assert.equal(blockedOrigin(uri), said, String(uri));
});

test('the button is wired once', async () => {
  const { win, document } = page({ picker: async () => { throw Object.assign(new Error('cancelled'), { name: 'AbortError' }); } });
  const button = document.createElement('button');
  win.__plvLogWatch = true;                                                // the watchers are another test's
  assert.equal(installLogButton(button, { win, doc: document, log: createPageLog() }), true);
  assert.equal(installLogButton(button, { win, doc: document }), false);
  assert.equal(installLogButton(null), false);
  assert.equal((button.listeners.get('click') || []).length, 1);
  button.emit('click'); await settle(); await settle();
  assert.equal(button.disabled, false, 'cancelled: the button is ready again');
});

test("the page's part names the count its numbers read: proseg's own, its newest counted iteration, or why a run has none", () => {
  // Averaged on counted iterations; the cell readouts of the iterations between show one draw (cells.zst foreground):
  // the row says both, never that every number is averaged.
  // Decision A: an iteration counted for a pause, the hold, a finish or the end reads the count proseg saves there.
  assert.equal(COUNTS_TEXT, "Proseg's count, averaged over its random background draw; at an iteration counted for a pause, the hold, a finish or the end, the count proseg saves there (one random background draw); between counted iterations the cell readouts show one draw of it, as proseg saves it");
  const { win } = page({ run: { id: 'r1', sample_name: 'sample-1', state: 'running', proseg_counts: { reason: null, newest: 1, recent: [0, 1] } } });
  const counts = () => pageText({ win, log: createPageLog(), now: when.getTime() }).match(/\nCounts {10}(.*)\n/)?.[1] ?? null;
  assert.equal(counts(), `${COUNTS_TEXT}; the newest counted iteration 7`);
  win.sopaCompanion.state.run.proseg_counts = { reason: 'old_proseg', newest: null, recent: [] };
  assert.equal(counts(), `${COUNTS_TEXT}; none for this run (old_proseg)`);
  win.sopaCompanion.state.run.proseg_counts = { reason: 'no_counts', newest: null, recent: [] };
  assert.equal(counts(), `${COUNTS_TEXT}; none for this run (no_counts)`);
  delete win.sopaCompanion.state.run.proseg_counts; assert.equal(counts(), COUNTS_TEXT, 'an engine that says nothing more');
  // A stain run, or the picker: no count named.
  win.sopaCompanion.state.run.kind = 'stain'; assert.equal(counts(), null);
  win.sopaCompanion.state.run.kind = 'proseg'; win.sopaCompanion.state.mode = 'pick'; assert.equal(counts(), null);
});
