import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { createPageLog, pageLogText, uploadPageLog, uploadBody, fitText, pageSession, PAGE_LOG_URL, UPLOAD_MS, UPLOAD_BYTES,
  BEACON_BYTES, IDLE_WAIT_MS } from '../src/log-export.mjs';
import { CRASH_TEXT, ALIVE_KEY, TAB_KEY, CrashReport, crashReport, crashFileName, readFlag } from '../src/crash-report.mjs';
import { createPageKey, STORE, PAGE_HEADER } from '../src/page-key.mjs';

// Crash reports, the page's side (a session that ended unexpectedly leaves its log, and the next start offers to save
// it): what a page finds as it starts (the page before it in
// the same window, the shared alive flag), the popup at the next start (crash-report.mjs), the page's log kept by the
// engine (log-export.mjs uploadPageLog) and the four requests (api.js).
const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
const main = read('../src/main.js');
const NEW = { session: 'new-session', at: '2026-10-23T10:00:00.000Z' }, OLD = { session: 'old-session', at: '2026-10-23T09:00:00.000Z' };
// The report's time without a zone: this computer's, so its file name is the same in every time zone.
const REPORT = { id: '20261023-100005-page', at: '2026-10-23T10:00:05', reason: 'page', bytes: 5120, started: OLD.at };
const TEXT = 'SoPA companion crash report\n===========================\nReason          page\n';

/** The engine's side as the page asks it: what it was asked, in order; `fail` makes a request fail. */
function engine({ report = REPORT, text = TEXT, fail = {} } = {}) {
  const calls = [];
  const failing = name => { const error = fail[name]?.shift?.() ?? fail[name]; if (error && !Array.isArray(error)) throw error; };
  return { calls,
    crash: async () => { calls.push('GET /crash'); failing('crash'); return { report }; },
    crashPage: async flag => { calls.push(['POST /crash/page', flag]); failing('page'); return { report }; },
    crashText: async id => { calls.push(`GET /crash/${id}.txt`); failing('text'); return text; },
    crashSeen: async id => { calls.push(`POST /crash/${id}/seen`); return {}; } };
}
/** Web Locks as the browser keeps them: the pages holding one (by session), those waiting for theirs, and the names
 *  requested. `later`: a lock asked for is granted only when grant() is called (else at once). */
function locks(sessions = [], { pending = [], later = false } = {}) {
  const held = new Set(sessions.map(s => `sopa-companion:page:${s}`)), requested = [], waiting = [];
  const asked = new Set(pending.map(s => `sopa-companion:page:${s}`));
  return { requested, request: (name, hold) => {
    requested.push(name);
    if (later) { asked.add(name); waiting.push(() => { asked.delete(name); held.add(name); return hold(); }); return new Promise(() => {}); }
    held.add(name); return hold();
  }, grant: () => waiting.splice(0).forEach(fn => fn()),
  query: async () => ({ held: [...held].map(name => ({ name, mode: 'exclusive' })), pending: [...asked].map(name => ({ name, mode: 'exclusive' })) }) };
}
/** A page in the DOM fixture: its storage (the shared one holding `flag`, the window's holding `tab`, if any), window
 *  events, notices and downloads. */
function page({ flag, tab, held, discarded = false, picker } = {}) {
  const dom = domFixture(), store = new Map(), tabStore = new Map(), listeners = {}, events = [], urls = [], log = createPageLog();
  if (flag !== undefined) store.set(ALIVE_KEY, typeof flag === 'string' ? flag : JSON.stringify(flag));
  if (tab !== undefined) tabStore.set(TAB_KEY, JSON.stringify(tab));
  dom.document.wasDiscarded = discarded;
  const storage = map => ({ getItem: key => map.get(key) ?? null, setItem: (key, value) => map.set(key, String(value)), removeItem: key => map.delete(key) });
  const win = {
    localStorage: storage(store), sessionStorage: storage(tabStore),
    navigator: held ? { locks: held } : {},
    addEventListener: (type, fn) => (listeners[type] ??= []).push(fn),
    dispatchEvent: event => events.push([event.type, event.detail]),
    CustomEvent: class { constructor(type, o) { this.type = type; this.detail = o?.detail; } },
    Blob: class { constructor(parts, o) { this.text = parts.join(''); this.type = o.type; } },
    URL: { createObjectURL: blob => { urls.push(blob); return `blob:${urls.length}`; }, revokeObjectURL() {} },
    setTimeout: () => 0,
  };
  if (picker) win.showSaveFilePicker = picker;
  const crash = new CrashReport({ win, doc: dom.document, log, session: NEW });
  const fire = (type, event = {}) => { for (const fn of listeners[type] || []) fn(event); };
  const words = () => log.lines().filter(l => l.kind === 'crash').map(l => l.text);
  return { ...dom, win, store, tabStore, listeners, fire, events, urls, log, crash, words };
}
function savingPicker(saved) {
  return async options => {
    saved.push({ options });
    return { name: options.suggestedName, createWritable: async () => ({ write: async blob => { saved.at(-1).blob = blob; }, close: async () => { saved.at(-1).closed = true; } }) };
  };
}

test('the words: plain, no banned word anywhere the popup or its notices can say', () => {
  assert.equal(CRASH_TEXT.title, 'SoPA companion closed unexpectedly last time');
  assert.equal(CRASH_TEXT.text, 'A log of what happened was saved. Save a copy to look at it together.');
  assert.deepEqual([CRASH_TEXT.save, CRASH_TEXT.later], ['Save a copy…', 'Not now']);
  for (const [key, words] of Object.entries(CRASH_TEXT)) assert.deepEqual(bannedWords(words), [], key);
  assert.equal(ALIVE_KEY, 'sopa-companion:page-alive'); assert.equal(TAB_KEY, 'sopa-companion:tab-page');
});

test('a flag the last session left: the engine is told (POST /crash/page with that session, its start and how it was found), and the report it wrote is offered', async () => {
  const p = page({ flag: OLD }), api = engine();
  assert.deepEqual(p.crash.start(), [{ ...OLD, found: 'flag' }], 'the flag found as the page starts');
  assert.deepEqual(JSON.parse(p.store.get(ALIVE_KEY)), NEW, "this page's own takes its place");
  assert.deepEqual(JSON.parse(p.tabStore.get(TAB_KEY)), NEW, "and this window's");
  assert.equal(await p.crash.check({ api }), REPORT);
  assert.deepEqual(api.calls.slice(0, 2), [['POST /crash/page', { session: OLD.session, at: OLD.at, found: 'flag' }], `GET /crash/${REPORT.id}.txt`],
    'told once (no GET /crash too); the text asked for at once, so Save a copy… opens the save dialog while the click counts');
  const dialog = p.$('crash-report');
  assert.equal(dialog.open, true, 'showModal: the page is dimmed');
  assert.equal(dialog.className, 'modal crash-report', 'the .modal look');
  assert.equal(dialog.getAttribute('aria-labelledby'), 'crash-title');
  assert.deepEqual(['crash-title', 'crash-text', 'crash-later', 'crash-save'].map(id => p.$(id).textContent),
    [CRASH_TEXT.title, CRASH_TEXT.text, 'Not now', 'Save a copy…']);
  assert.equal(p.$('crash-error').hidden, true);
  assert.equal(p.document.activeElement, p.$('crash-save'), 'Save a copy… is the default');
  assert.deepEqual(p.words(), [`a page of the last session went without closing (it started ${OLD.at}): the engine was told`,
    `offered the report of ${REPORT.at} (page, 5120 bytes)`]);
  assert.equal(await p.crash.check({ api }), null, 'shown once per report');
  assert.deepEqual(api.calls.slice(2), ['GET /crash'], 'the flag was told once: a second check only asks');
});

test('no flag: the newest report not seen (GET /crash) is offered once; no report, or a report offered already, shows nothing', async () => {
  const p = page(), api = engine();
  assert.deepEqual(p.crash.start(), []);
  assert.equal(await p.crash.check({ api }), REPORT);
  assert.equal(api.calls[0], 'GET /crash');
  assert.equal(p.$('crash-report').open, true);
  p.$('crash-later').click(); await settle();
  assert.equal(await p.crash.check({ api }), null, 'the same report again (not marked seen in time): not shown twice');
  assert.equal(p.$('crash-report').open, false);
  const none = page(), quiet = engine({ report: null });
  none.crash.start();
  assert.equal(await none.crash.check({ api: quiet }), null);
  assert.deepEqual(quiet.calls, ['GET /crash']);
  assert.equal(none.$('crash-report'), null, 'no report: no popup');
  assert.deepEqual(none.events, []); assert.deepEqual(none.words(), []);
  // Two checks at once (the boot's and a harness scenario's) ask once.
  const twice = page(), both = engine();
  twice.crash.start();
  const [first, second] = await Promise.all([twice.crash.check({ api: both }), twice.crash.check({ api: both })]);
  assert.equal(first, REPORT); assert.equal(second, REPORT); assert.equal(both.calls.filter(c => c === 'GET /crash').length, 1);
});

test("Save a copy… saves the report's text as sopa-companion-crash-<time>.txt, through the browser's save dialog, and marks it seen", async () => {
  const saved = [], p = page({ picker: savingPicker(saved) }), api = engine();
  p.crash.start(); await p.crash.check({ api });
  const result = await p.$('crash-save').click();
  assert.deepEqual(result, { saved: true, name: 'sopa-companion-crash-20261023-100005.txt' });
  assert.equal(saved.length, 1);
  assert.equal(saved[0].options.suggestedName, 'sopa-companion-crash-20261023-100005.txt');
  assert.equal(saved[0].options.id, 'sopa-companion-log', "the Log button's dialog, remembering its folder");
  assert.equal(saved[0].blob.text, TEXT); assert.equal(saved[0].blob.type, 'text/plain;charset=utf-8'); assert.equal(saved[0].closed, true);
  await settle();
  assert.deepEqual(api.calls, ['GET /crash', `GET /crash/${REPORT.id}.txt`, `POST /crash/${REPORT.id}/seen`]);
  assert.equal(p.$('crash-report').open, false);
  assert.deepEqual(p.events, [['fetch-notice', 'Log saved: sopa-companion-crash-20261023-100005.txt']]);
  assert.deepEqual(p.crash.snapshot(), { open: false, report: REPORT.id, choice: 'save' });
  assert.deepEqual(p.words().slice(-2), ['Save a copy… pressed', 'saved a copy: sopa-companion-crash-20261023-100005.txt']);
  assert.equal(await p.$('crash-save').click(), null, 'answered: a second press does nothing');
  // File names: the report's time as this computer's; the engine's own characters when it is not a time.
  assert.equal(crashFileName('20261023-100005-engine'), 'sopa-companion-crash-20261023-100005-engine.txt');
  assert.equal(crashFileName('../a:b'), 'sopa-companion-crash-a-b.txt'); assert.equal(crashFileName(undefined), 'sopa-companion-crash-report.txt');
});

test("Save a copy… where the browser has no save dialog: the Log button's own dialog, whose button saves it as a download", async () => {
  const p = page(), api = engine();
  p.crash.start(); await p.crash.check({ api });
  const pending = p.$('crash-save').click();
  await settle(); await settle();
  const log = p.$('log-dialog');
  assert.equal(log.open, true); assert.equal(p.$('crash-report').open, false, 'one popup at a time');
  assert.equal(log.children[1].textContent, 'sopa-companion-crash-20261023-100005.txt · 4 lines');
  assert.equal(p.urls.length, 0, 'nothing saved before the click');
  log.children[3].children[1].click();
  assert.deepEqual(await pending, { saved: true, name: 'sopa-companion-crash-20261023-100005.txt', download: true });
  assert.equal(p.urls[0].text, TEXT);
  assert.deepEqual(p.events, [['fetch-notice', 'Log saved to your downloads: sopa-companion-crash-20261023-100005.txt']]);
  assert.ok(api.calls.includes(`POST /crash/${REPORT.id}/seen`));
});

test('Not now, Escape, or the popup closed any other way: marked seen, nothing saved', async () => {
  for (const answer of ['not now', 'escape', 'closed']) {
    const saved = [], p = page({ picker: savingPicker(saved) }), api = engine({ report: { ...REPORT, id: `r-${answer.replace(' ', '-')}` } });
    p.crash.start(); await p.crash.check({ api });
    const dialog = p.$('crash-report');
    if (answer === 'not now') p.$('crash-later').click();
    else if (answer === 'escape') dialog.emit('cancel');
    else dialog.close();
    await settle();
    assert.equal(dialog.open, false, answer);
    assert.deepEqual(api.calls.filter(c => String(c).endsWith('/seen')), [`POST /crash/r-${answer.replace(' ', '-')}/seen`], `${answer}: seen once`);
    assert.equal(saved.length, 0, answer); assert.deepEqual(p.events, [], answer);
    assert.equal(p.crash.snapshot().choice, 'later', answer);
    assert.equal(p.words().at(-1), 'Not now', answer);
  }
});

test('a report whose text could not be read: the popup says so and stays; a press asks for it again', async () => {
  const saved = [], p = page({ picker: savingPicker(saved) });
  const api = engine({ fail: { text: [Object.assign(new Error('HTTP 404'), { status: 404 })] } });
  p.crash.start(); await p.crash.check({ api });
  assert.equal(await p.$('crash-save').click(), null);
  assert.equal(p.$('crash-error').hidden, false); assert.equal(p.$('crash-error').textContent, 'The log could not be read: HTTP 404');
  assert.deepEqual(bannedWords(p.$('crash-error').textContent), []);
  assert.equal(p.$('crash-report').open, true); assert.equal(p.$('crash-save').textContent, 'Save a copy…');
  assert.equal(p.$('crash-save').getAttribute('aria-busy'), null);
  assert.ok(!api.calls.some(c => String(c).endsWith('/seen')), 'not answered yet');
  const result = await p.$('crash-save').click();
  assert.equal(result.saved, true); assert.equal(saved[0].blob.text, TEXT);
  assert.equal(api.calls.filter(c => c === `GET /crash/${REPORT.id}.txt`).length, 2);
  // The text on its way when Save a copy… is pressed: the button says so meanwhile, and a second press waits for it.
  const slow = deferred(), q = page({ picker: savingPicker(saved) }), late = engine();
  late.crashText = () => slow.promise;
  q.crash.start(); await q.crash.check({ api: late });
  const pressed = q.$('crash-save').click();
  await settle();
  assert.equal(q.$('crash-save').textContent, CRASH_TEXT.reading); assert.equal(q.$('crash-save').getAttribute('aria-busy'), 'true');
  assert.equal(await q.$('crash-save').click(), null);
  slow.resolve('late text');
  assert.equal((await pressed).saved, true); assert.equal(saved.at(-1).blob.text, 'late text');
});

test("pagehide removes this page's flag and this window's mark (a clean unload), never another window's flag; a page the browser brings back sets them again", () => {
  const p = page();
  p.crash.start();
  assert.deepEqual(JSON.parse(p.store.get(ALIVE_KEY)), NEW); assert.deepEqual(readFlag(p.win.sessionStorage, TAB_KEY), NEW);
  p.fire('pagehide', { persisted: true });
  assert.equal(p.store.has(ALIVE_KEY), false, 'closed, reloaded or put aside: a clean unload');
  assert.equal(p.tabStore.has(TAB_KEY), false, 'the next page in this window finds nothing left');
  p.fire('pageshow', { persisted: true });
  assert.deepEqual(JSON.parse(p.store.get(ALIVE_KEY)), NEW, 'brought back: alive again');
  assert.deepEqual(JSON.parse(p.tabStore.get(TAB_KEY)), NEW);
  p.fire('pageshow', { persisted: false });
  p.store.set(ALIVE_KEY, JSON.stringify({ session: 'another-window', at: NEW.at }));
  p.fire('pagehide');
  assert.deepEqual(readFlag(p.win.localStorage), { session: 'another-window', at: NEW.at }, "another window's flag stays");
  // A flag the page cannot read is no flag; storage the browser refuses: no flag, and nothing breaks.
  assert.equal(readFlag({ getItem: () => '{"session": 3}' }), null); assert.equal(readFlag({ getItem: () => 'not json' }), null);
  const blocked = page({ tab: OLD });
  for (const name of ['localStorage', 'sessionStorage']) Object.defineProperty(blocked.win, name, { get() { throw new Error('SecurityError'); } });
  assert.deepEqual(blocked.crash.start(), []);
  blocked.fire('pagehide');
});

test("the page before this one in the same window did not unload (its mark is left in the window): that page crashed, told as 'tab', with or without the shared flag", async () => {
  // A second window took the shared flag over and cleared it as it closed; this window's page then crashed and was reloaded.
  const p = page({ tab: OLD }), api = engine();
  assert.deepEqual(p.crash.start(), [{ ...OLD, found: 'tab' }]);
  assert.deepEqual(readFlag(p.win.sessionStorage, TAB_KEY), NEW, "this page's mark takes its place");
  assert.equal(await p.crash.check({ api }), REPORT);
  assert.deepEqual(api.calls[0], ['POST /crash/page', { ...OLD, found: 'tab' }]);
  assert.equal(p.words()[0], `the page before this one in this window went without closing (it started ${OLD.at}): the engine was told`);
  // Both left by the same page: told once, as the window's.
  const both = page({ tab: OLD, flag: OLD }), once = engine();
  assert.deepEqual(both.crash.start(), [{ ...OLD, found: 'tab' }]);
  await both.crash.check({ api: once });
  assert.deepEqual(once.calls.filter(c => Array.isArray(c)), [['POST /crash/page', { ...OLD, found: 'tab' }]]);
  // Another page's flag besides: each told, the window's first; the popup offers the engine's last answer (the report
  // still to offer then: the window's page's, as the engine keeps the flag's quietly).
  const other = { session: 'other-session', at: '2026-10-23T08:00:00.000Z' }, two = page({ tab: OLD, flag: other });
  const answers = engine(), quietly = { ...REPORT, id: 'kept-quietly' };
  answers.crashPage = async flag => { answers.calls.push(['POST /crash/page', flag]); return { report: flag.found === 'tab' ? quietly : REPORT }; };
  two.crash.start();
  assert.equal(await two.crash.check({ api: answers }), REPORT, "the engine's last answer");
  assert.deepEqual(answers.calls.filter(c => Array.isArray(c)), [['POST /crash/page', { ...OLD, found: 'tab' }], ['POST /crash/page', { ...other, found: 'flag' }]]);
  // A window copied from one still open (the browser copies the window's storage): not a crash.
  const copy = page({ tab: OLD, held: locks([OLD.session]) }), asked = engine({ report: null });
  copy.crash.start(); await copy.crash.check({ api: asked });
  assert.deepEqual(asked.calls, ['GET /crash']);
  assert.equal(copy.words()[0], 'the window this one was copied from is still open: not a crash');
});

test('what the engine was not told about (it did not answer) stays for the next start; what it answered is told once', async () => {
  const p = page({ flag: OLD }), away = engine({ fail: { page: Object.assign(new TypeError('Failed to fetch')) } });
  p.crash.start();
  assert.equal(await p.crash.check({ api: away }), null);
  assert.equal(p.$('crash-report'), null);
  p.fire('pagehide');
  assert.deepEqual(readFlag(p.win.localStorage), OLD, 'kept for the next start');
  assert.equal(p.tabStore.has(TAB_KEY), false, "this window's own mark goes: it unloaded cleanly");
  assert.match(p.words().at(-1), /^could not ask the engine about the last session: Failed to fetch$/);
  // The window's page: put back in the window (a reload tells it) and in the shared flag (a window opened next does).
  const w = page({ tab: OLD });
  w.crash.start(); await w.crash.check({ api: away }); w.fire('pagehide');
  assert.deepEqual(readFlag(w.win.sessionStorage, TAB_KEY), OLD); assert.deepEqual(readFlag(w.win.localStorage), OLD);
  // The engine never answered the page's first requests at all (main.js boot: no check): the same.
  const down = page({ flag: OLD });
  down.crash.start(); down.fire('pagehide');
  assert.deepEqual(readFlag(down.win.localStorage), OLD);
  // An answer that refuses it (the engine's HTTP error): told once, not again at every start.
  const q = page({ flag: OLD }), refusing = engine({ fail: { page: Object.assign(new Error('Unprocessable'), { status: 422 }) } });
  q.crash.start();
  assert.equal(await q.crash.check({ api: refusing }), null);
  q.fire('pagehide');
  assert.equal(q.store.has(ALIVE_KEY), false);
});

test('not a crash: a flag whose window is still open (it holds its lock, or has asked for it), or a page the browser put aside and loaded again', async () => {
  const held = locks([OLD.session]), p = page({ flag: OLD, held }), api = engine({ report: null });
  p.crash.start();
  assert.deepEqual(held.requested, [`sopa-companion:page:${NEW.session}`], 'this page holds its own lock while it lives');
  await p.crash.check({ api });
  assert.deepEqual(api.calls, ['GET /crash'], 'only asked: nothing to tell');
  assert.equal(p.words()[0], 'another window of the app is open: its flag is not a crash');
  p.fire('pagehide');
  assert.equal(p.store.has(ALIVE_KEY), false, "ours goes; that window's own pagehide sees ours gone and leaves it");
  const gone = page({ flag: OLD, held: locks() }), told = engine();
  gone.crash.start(); await gone.crash.check({ api: told });
  assert.deepEqual(told.calls[0], ['POST /crash/page', { ...OLD, found: 'flag' }], 'its lock is gone with it: a crash');
  const waiting = page({ flag: OLD, held: locks([], { pending: [OLD.session] }) }), quiet = engine({ report: null });
  waiting.crash.start(); await waiting.crash.check({ api: quiet });
  assert.deepEqual(quiet.calls, ['GET /crash'], 'a window starting at this moment, its lock asked for: not a crash');
  const discarded = page({ flag: OLD, tab: OLD, discarded: true }), asked = engine({ report: null });
  assert.deepEqual(discarded.crash.start(), []); await discarded.crash.check({ api: asked });
  assert.deepEqual(asked.calls, ['GET /crash']);
  assert.match(discarded.words()[0], /put this page aside to save memory/);
});

test('the shared flag is written once this page holds its lock: a window starting meanwhile never takes this one for a crash', () => {
  const held = locks([], { later: true }), p = page({ held });
  p.crash.start();
  assert.equal(p.store.has(ALIVE_KEY), false, 'not before the lock');
  assert.deepEqual(readFlag(p.win.sessionStorage, TAB_KEY), NEW, "the window's mark at once (no other window reads it)");
  held.grant();
  assert.deepEqual(readFlag(p.win.localStorage), NEW);
  const early = locks([], { later: true }), q = page({ held: early });
  q.crash.start(); q.fire('pagehide'); early.grant();
  assert.equal(q.store.has(ALIVE_KEY), false, 'a page gone before its lock came leaves no flag');
  const refused = page({ held: { request: () => Promise.reject(new Error('no locks here')), query: async () => ({ held: [], pending: [] }) } });
  refused.crash.start();
  return settle().then(() => assert.deepEqual(readFlag(refused.win.localStorage), NEW, 'no lock to be had: the flag all the same'));
});

test('under ?harness=1 nothing is asked or shown at boot; a scenario shows it with check(), the api kept', async () => {
  const p = page({ flag: OLD }), api = engine();
  p.crash.start();
  assert.equal(await p.crash.check({ api, quiet: true }), null);
  assert.deepEqual(api.calls, []); assert.equal(p.$('crash-report'), null);
  assert.equal(await p.crash.check(), REPORT);
  assert.deepEqual(api.calls[0], ['POST /crash/page', { ...OLD, found: 'flag' }]);
  assert.deepEqual(p.crash.snapshot(), { open: true, report: REPORT.id, choice: null });
  assert.equal(await new CrashReport({ win: p.win, doc: p.document }).check(), null, 'no api yet: nothing');
});

test('main.js loads the module with the page and asks once the engine has answered its first requests (quietly under ?harness=1)', () => {
  assert.match(main, /^import '\.\/crash-report\.mjs';$/m);
  const boot = main.slice(main.indexOf('async function boot() {'), main.indexOf('\nwindow.sopaCompanion = {'));
  const answered = boot.indexOf('engineDown(error);\n    return;\n  }'), call = boot.indexOf('globalThis.sopaCompanionCrash?.check({ api, quiet: !!harness });');
  assert.ok(answered > 0 && call > answered, 'after the first requests were answered: an engine that did not has nothing to say');
  assert.ok(call < boot.indexOf('await Promise.allSettled([api.options(), api.optionsGuide()])'), 'not waiting for the rest of the boot');
  assert.equal(main.split('sopaCompanionCrash?.check(').length, 2, 'once');
  assert.equal(globalThis.sopaCompanionCrash, crashReport, "the module's one, as main.js finds it (its tests run it without the module)");
  // The popup's own rule in the page's stylesheet, beside the .modal look it reuses.
  const css = read('../styles/app.css');
  assert.match(css, /^\.modal\{/m); assert.match(css, /^\.crash-report\{width:min\(520px,calc\(100vw - 32px\)\)\}$/m);
});

test("api.js: the four requests, their paths and bodies; they and the page log's upload never raise the general notice", async () => {
  const sent = [], notices = [], context = vm.createContext({ CustomEvent: class { constructor(type, o) { this.type = type; this.detail = o?.detail; } } });
  context.window = context; context.dispatchEvent = event => notices.push(event.detail);
  let status = 200;
  context.fetch = async (url, o = {}) => {
    sent.push([o.method ?? 'GET', url, o.body ?? null, o.cache ?? null, o.keepalive ?? null]);
    return { ok: status === 200, status, statusText: String(status), json: async () => (status === 200 ? { report: null } : { detail: 'No such report' }), text: async () => TEXT };
  };
  const api = vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replaceAll('export ', '') + '\napi', context);
  assert.deepEqual(await api.crash(), { report: null });
  assert.deepEqual(await api.crashPage(OLD), { report: null });
  assert.deepEqual(await api.crashPage({ ...OLD, found: 'tab' }), { report: null });
  assert.equal(await api.crashText('20261023-100005-page'), TEXT);
  await api.crashSeen('a b/c');
  assert.deepEqual(sent, [['GET', '/api/v1/crash', null, null, null],
    ['POST', '/api/v1/crash/page', JSON.stringify({ at: OLD.at, session: OLD.session }), null, null],
    ['POST', '/api/v1/crash/page', JSON.stringify({ at: OLD.at, session: OLD.session, found: 'tab' }), null, null],
    ['GET', '/api/v1/crash/20261023-100005-page.txt', null, 'no-store', null],
    ['POST', '/api/v1/crash/a%20b%2Fc/seen', null, null, true]], 'seen: a keepalive request, which a window closing right after the answer does not cancel');
  status = 404;
  await assert.rejects(api.crash(), error => error.status === 404 && error.message === 'No such report');
  await assert.rejects(api.crashText('gone'), error => error.status === 404 && error.message === 'HTTP 404');
  await assert.rejects(api.crashSeen('gone')); await assert.rejects(api.crashPage(OLD));
  await context.window.fetch(PAGE_LOG_URL, { method: 'POST' });
  assert.deepEqual(notices, [], 'quiet: the popup says what failed, and the upload is never seen');
  await context.window.fetch('/api/v1/crashes'); await context.window.fetch('/api/v1/page-logs');
  assert.equal(notices.length, 2, 'only their own paths are quiet');
});

/** A window for the upload: its POSTs (window.fetch, which adds the page's key in the page), its beacons (never
 *  used), its timers (run by hand) and listeners; and the tab's page key (page-key.mjs, with `key` kept in the tab:
 *  none, a tab that is not signed in), whose keepalive POSTs are `sent` (`refuse`: the browser refuses them). */
function uploader({ key = 'k'.repeat(43), refuse = false } = {}) {
  const posts = [], beacons = [], sent = [], timers = {}, listeners = {};
  const win = {
    location: { pathname: '/', search: '' }, screen: {}, performance: {},
    navigator: { userAgent: 'Chrome/149 test', sendBeacon: (url, data) => { beacons.push([url, data]); return true; } },
    fetch: async (url, o) => { posts.push([url, o]); return { ok: true, status: 200 }; },
    setInterval: (fn, ms) => { timers.interval = [fn, ms]; return 7; }, clearInterval: id => { timers.cleared = id; },
    setTimeout: (fn, ms) => { timers.first = [fn, ms]; return 8; },
    addEventListener: (type, fn) => (listeners[type] ??= []).push(fn),
  };
  const tab = new Map(key ? [[STORE, key]] : []);
  const keys = createPageKey({ win: { location: { href: 'http://127.0.0.1:8799/', pathname: '/', search: '' },
    sessionStorage: { getItem: name => tab.get(name) ?? null, setItem() {}, removeItem() {} } },
  native: (url, o) => { if (refuse) throw new TypeError('keepalive refused'); sent.push([url, o]); return Promise.resolve({ ok: true, status: 200 }); } });
  const body = i => JSON.parse(posts[i][1].body);
  return { win, posts, beacons, sent, timers, listeners, body, keys, key };
}

test("the page's log goes to the engine as the page starts, then at most every 10 s while it changed, and by beacon as the page goes", async () => {
  const log = createPageLog(), u = uploader(), session = { session: 's-1', at: NEW.at };
  const up = uploadPageLog({ win: u.win, log, session, keys: u.keys });
  assert.deepEqual([u.timers.first[1], u.timers.interval[1], UPLOAD_MS], [0, 10000, 10000]);
  assert.equal(await u.timers.first[0](), true, 'as the page starts (once main.js has run)');
  const [url, options] = u.posts[0];
  assert.equal(url, '/api/v1/page-log');
  assert.deepEqual([options.method, options.keepalive, options.headers], ['POST', undefined, { 'Content-Type': 'application/json' }],
    "through the page's fetch, which adds the key");
  const first = u.body(0);
  assert.deepEqual(Object.keys(first).sort(), ['at', 'session', 'text']);
  assert.equal(first.session, 's-1'); assert.ok(Number.isFinite(Date.parse(first.at)), 'when it was sent (ISO)');
  // What the Log button saves of the page when the engine wrote nothing: its part, with its "Page noted" row.
  assert.match(first.text, /^Page \(this window\)\n==================\nPage noted      nothing went wrong\nWritten         /);
  assert.equal(first.text, pageLogText({ win: u.win, log, now: Date.parse(first.text.match(/Written {9}(\S+ \S+)/)[1].replace(' ', 'T')) }));
  assert.equal(await u.timers.interval[0](), false, '10 s on, nothing changed: nothing sent');
  log.add('action', 'opened run r1'); log.add('action', 'pause (at iteration 4)');
  assert.equal(await u.timers.interval[0](), true);
  assert.equal(u.posts.length, 2, 'two changes in one period: one upload');
  assert.match(u.body(1).text, /action    opened run r1\n.*action    pause \(at iteration 4\)\n$/);
  assert.equal(await u.timers.interval[0](), false);
  log.add('action', 'pause (at iteration 4)');                       // the same line again: counted, a change
  assert.equal(await up.tick(), true); assert.match(u.body(2).text, /pause \(at iteration 4\) \(2 times, last /);
  // One on its way: a tick meanwhile sends nothing more; the engine away: the next tick tries again.
  const gate = deferred();
  u.win.fetch = async (to, o) => { u.posts.push([to, o]); return gate.promise; };
  log.add('notice', 'Request failed (500). Please try again.');
  const sending = up.tick();
  assert.equal(await up.tick(), false); gate.resolve({ ok: true, status: 200 }); assert.equal(await sending, true);
  assert.equal(u.posts.length, 4);
  u.win.fetch = async (to, o) => { u.posts.push([to, o]); throw new TypeError('Failed to fetch'); };
  log.add('action', 'resume (at iteration 4)');
  assert.equal(await up.tick(), false);
  u.win.fetch = async (to, o) => { u.posts.push([to, o]); return { ok: true, status: 200 }; };
  assert.equal(await up.tick(), true); assert.match(u.body(5).text, /resume \(at iteration 4\)\n$/);
  // As the page goes: its goodbye, a keepalive POST with the page's key sent at once (a beacon cannot carry the key, and
  // a page going cannot wait for a promise), saying so in its last line.
  assert.equal(u.listeners.pagehide[0]({ persisted: false }), true);
  assert.deepEqual([u.sent.length, u.beacons.length], [1, 0], 'never a beacon');
  const [to, sent] = u.sent[0], data = sent.body;
  assert.equal(to, PAGE_LOG_URL); assert.equal(typeof data, 'string');
  assert.deepEqual([sent.method, sent.keepalive, sent.credentials, sent.headers[PAGE_HEADER], sent.headers['Content-Type']],
    ['POST', true, 'omit', u.key, 'application/json']);
  assert.equal(JSON.parse(data).session, 's-1'); assert.match(JSON.parse(data).text, /  page {6}the page closed or reloaded\n$/);
  assert.equal(JSON.parse(data).pagehide, true, 'its goodbye: the engine knows it closed cleanly');
  assert.equal(JSON.parse(data).visit, 1, 'from its first showing (log-export.mjs pageShowing)');
  assert.equal(u.body(5).pagehide, undefined, 'the others are no goodbye');
  assert.equal(u.posts.length, 6, 'the goodbye only, through the key');
  assert.equal(pageSession.session.length > 8 && Number.isFinite(Date.parse(pageSession.at)), true, "this page's session: an id and its start");
});

test('the goodbye: a page put aside says so; a tab not signed in, or a keepalive the browser refuses, sends none and throws nothing; an engine without the endpoint is not asked again', async () => {
  const aside = uploader(), log = createPageLog(), showing = { visit: 1 };
  uploadPageLog({ win: aside.win, log, session: NEW, keys: aside.keys, showing });
  assert.equal(aside.listeners.pagehide[0]({ persisted: true }), true);
  assert.match(JSON.parse(aside.sent[0][1].body).text, /the page was put aside \(the browser may bring it back\)\n$/);
  // Brought back, it is its next showing (api.js LiveSocket.back counts it): its next goodbye names that one, so the
  // engine never takes the one before, should it arrive late, for the page shown now.
  aside.listeners.pageshow[0]({ persisted: true }); showing.visit = 2;
  aside.listeners.pagehide[0]({ persisted: false });
  assert.deepEqual(aside.sent.map(([, o]) => JSON.parse(o.body).visit), [1, 2]);
  for (const options of [{ key: null }, { refuse: true }]) {
    const u = uploader(options);
    uploadPageLog({ win: u.win, log: createPageLog(), session: NEW, keys: u.keys });
    assert.equal(u.listeners.pagehide[0]({}), false, JSON.stringify(options));
    assert.deepEqual([u.sent.length, u.beacons.length, u.posts.length], [0, 0, 0], JSON.stringify(options));
  }
  const u = uploader(), up = uploadPageLog({ win: u.win, log, session: NEW, keys: u.keys });
  u.win.fetch = async (to, o) => { u.posts.push([to, o]); return { ok: false, status: 404 }; };
  assert.equal(await up.tick(), false);
  assert.equal(u.timers.cleared, 7, 'its timer stopped');
  log.add('action', 'opened run r1');
  assert.equal(await up.tick(), false); u.listeners.pagehide[0]({});
  assert.deepEqual([u.posts.length, u.sent.length, u.beacons.length], [1, 0, 0]);
});

test('a long log is cut to fit: the set-up and the newest lines kept, the oldest said to be left out; the beacon within 32 KB', () => {
  assert.equal(BEACON_BYTES, 32000, 'with telemetry (16 KB) and the answer to the close popup, within the 64 KB a closing page may send');
  const log = createPageLog({ cap: 4000 });
  for (let i = 0; i < 3000; i++) log.add('action', `step ${i} ${'x'.repeat(200)}`);
  const text = pageLogText({ win: { location: {}, performance: {} }, log, now: Date.parse(NEW.at) });
  assert.ok(text.length > 600000, text.length);
  for (const limit of [BEACON_BYTES, UPLOAD_BYTES]) {
    const body = uploadBody(text, { session: 's-1', at: NEW.at }, limit), sent = JSON.parse(body);
    assert.ok(new TextEncoder().encode(body).length <= limit, `${limit}`);
    assert.deepEqual([sent.session, sent.at], ['s-1', NEW.at]);
    assert.match(sent.text, /^Page \(this window\)\n==================\nPage noted      nothing went wrong\n/, 'the set-up kept');
    assert.match(sent.text, /\nWhat the user did and the page noted \(oldest first; this computer's clock\)\n-{74}\n {2}… \d+ older lines left out\n/);
    assert.match(sent.text, /step 2999 x{200}\n$/, 'the newest line kept');
    assert.doesNotMatch(sent.text, /step 0 x/);
    assert.ok(sent.text.length > limit / 2, 'most of the room used');
  }
  assert.equal(uploadBody('short', { session: 's', at: 'a' }), JSON.stringify({ text: 'short', session: 's', at: 'a' }));
  assert.equal(fitText('short', 10), 'short');
  assert.equal(fitText('x'.repeat(500), 100), 'x'.repeat(100), 'not the page\'s shape: its end');
  const oneLine = fitText(`${'a'.repeat(300)}\n${'b'.repeat(50)}\n`, 200);
  assert.ok(oneLine.length <= 200); assert.match(oneLine, /^ {2}… 1 older line left out\nb{50}\n$/);
});

test('the upload waits for a moment with nothing to do; a hidden page sends its log as it is hidden, then nothing; one brought back says so', async () => {
  const u = uploader(), log = createPageLog(), idle = [], listeners = {};
  u.win.requestIdleCallback = (fn, options) => { idle.push([fn, options]); return idle.length; };
  const doc = { hidden: false, addEventListener: (type, fn) => (listeners[type] ??= []).push(fn) };
  const up = uploadPageLog({ win: u.win, doc, log, session: NEW });
  const first = up.tick();
  await settle();
  assert.equal(u.posts.length, 0, 'not before the page has a moment');
  assert.deepEqual(idle.map(([, o]) => o), [{ timeout: IDLE_WAIT_MS }]); assert.equal(IDLE_WAIT_MS, 2000);
  idle[0][0]();
  assert.equal(await first, true); assert.equal(u.posts.length, 1);
  log.add('action', 'opened run r1');
  doc.hidden = true;
  for (const fn of listeners.visibilitychange) fn();                     // hidden: sent at once, without waiting
  await settle();
  assert.equal(u.posts.length, 2); assert.equal(idle.length, 1);
  assert.match(u.body(1).text, /opened run r1\n$/);
  log.add('notice', 'a notice while hidden');
  assert.equal(await up.tick(), false, 'hidden: nothing');
  doc.hidden = false;
  const shown = up.tick(); await settle(); idle.at(-1)[0]();
  assert.equal(await shown, true); assert.match(u.body(2).text, /a notice while hidden\n$/);
  // A page that is hidden as it starts still says it is there once.
  const v = uploader(), hiddenLog = createPageLog(), hiddenUp = uploadPageLog({ win: v.win, doc: { hidden: true }, log: hiddenLog, session: NEW });
  assert.equal(await hiddenUp.tick(), true); hiddenLog.add('action', 'x'); assert.equal(await hiddenUp.tick(), false);
  // A page that goes is hidden after its goodbye (pagehide, then visibilitychange): nothing more is sent, which would
  // tell the engine it is open again.
  u.listeners.pagehide[0]({ persisted: true });
  const beacons = u.beacons.length, posts = u.posts.length;
  doc.hidden = true;
  for (const fn of listeners.visibilitychange) fn();
  await settle();
  assert.deepEqual([u.beacons.length, u.posts.length], [beacons, posts]);
  assert.equal(await up.tick(), false);
  // Brought back after its goodbye (pageshow, persisted): a line, so the next upload tells the engine it is open again.
  u.listeners.pageshow[0]({ persisted: true });
  assert.equal(log.lines().at(-1).text, 'the page was brought back');
  doc.hidden = false;
  const back = up.tick(); await settle(); idle.at(-1)[0]();
  assert.equal(await back, true); assert.equal(u.body(u.posts.length - 1).pagehide, undefined);
  u.listeners.pageshow[0]({ persisted: false });
  assert.equal(log.lines().filter(l => l.text === 'the page was brought back').length, 1, 'a page loaded anew: nothing');
});

test("the page's log counts its changes (the upload's cue), and the upload's own refusals never note themselves", async () => {
  const log = createPageLog();
  assert.equal(log.version, 0);
  log.add('action', 'a'); log.add('action', 'a'); log.add('action', ''); log.add('notice', 'b');
  assert.equal(log.version, 3, 'a line, a repeat counted, an empty line ignored, another line');
  const { watchPage } = await import('../src/log-export.mjs'), observers = [];
  const win = { PerformanceObserver: class { constructor(fn) { this.fn = fn; observers.push(this); } observe() {} }, addEventListener() {} };
  const noted = createPageLog();
  watchPage({ win, doc: {}, log: noted });
  observers[0].fn({ getEntries: () => [{ name: 'http://127.0.0.1:8777/api/v1/page-log', responseStatus: 413 },
    { name: 'http://127.0.0.1:8777/api/v1/crash/x.txt', responseStatus: 404 }] });
  assert.deepEqual(noted.lines().map(l => l.text), ['HTTP 404 /api/v1/crash/x.txt']);
});
