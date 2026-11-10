// A session that ended unexpectedly (the engine stopped without shutting down, or this page crashed or was killed): at
// the next start the page offers the log saved then, to look at it together. The engine writes
// the reports (engine/crashlog.py; crash-reports/): its own when it finds the alive marker of a process that is
// gone; a page's when it finds no page connected while the app window stays open (a crashed tab), and when a page that
// starts says what it found.
//
// What a page finds as it starts (each set as a page starts, {session, at} of log-export.mjs pageSession, and removed
// on pagehide, a clean unload: closed, reloaded, another address):
//   * the page before it in the same window, when it did not unload: sessionStorage "sopa-companion:tab-page", which
//     the browser keeps for the window through a crash of its page and the reload after it: that page crashed ("tab");
//   * the flag of a page that did not unload: localStorage "sopa-companion:page-alive", which the browser's windows
//     share (written once this page holds its Web Lock, so a window starting meanwhile never takes it for a crash) and
//     writes to disk only now and then ("flag": a page that went with its browser, as at a logout, or one whose flag
//     an old copy kept; the engine keeps such a report quietly, and has none for a page that said its goodbye).
// Neither is a crash when the window that set it is still open (each page holds a Web Lock named after its session
// while it lives), nor when the browser put the page aside to save memory and loaded it again (document.wasDiscarded).
// Once the engine has answered (main.js boot: check()), the page tells it what it found (POST /api/v1/crash/page) or
// asks for the newest report not seen yet (GET /api/v1/crash); a report -> the popup, which offers each report once:
// Save a copy… (the Log button's save flow: the browser's save dialog, else ours) or Not now (Escape too), and either
// marks it seen (POST /api/v1/crash/{id}/seen, sent so that it arrives even as the page goes).
import { pageLog, pageSession, saveText, fileTime, namePart } from './log-export.mjs';
import { busy } from './explain.mjs';

export const CRASH_TEXT = Object.freeze({
  title: 'SoPA companion closed unexpectedly last time',
  text: 'A log of what happened was saved. Save a copy to look at it together.',
  save: 'Save a copy…', later: 'Not now',
  reading: 'Reading the log…',                    // Save a copy… while the log is still on its way
  unread: 'The log could not be read: {why}',
  saved: 'Log saved: {name}', downloaded: 'Log saved to your downloads: {name}', failed: 'The log could not be saved: {why}',
});
export const ALIVE_KEY = 'sopa-companion:page-alive', TAB_KEY = 'sopa-companion:tab-page';
const lockName = session => `sopa-companion:page:${session}`;

/** sopa-companion-crash-<at>.txt: the report's time as this computer's (20261023-140304), else its own characters. */
export function crashFileName(at) {
  const date = new Date(typeof at === 'string' || typeof at === 'number' ? at : NaN);
  return `sopa-companion-crash-${(Number.isFinite(date.getTime()) ? fileTime(date) : namePart(at)) || 'report'}.txt`;
}

const storageOf = (win, name) => { try { return win[name] ?? null; } catch { return null; } };   // blocked: none
/** The flag under `key` in `storage`: {session, at}, or null (none, or not one the page wrote). */
export function readFlag(storage, key = ALIVE_KEY) {
  try {
    const flag = JSON.parse(storage?.getItem(key) ?? 'null');
    return typeof flag?.session === 'string' && typeof flag?.at === 'string' ? { session: flag.session, at: flag.at } : null;
  } catch { return null; }
}
const said = { tab: 'the page before this one in this window', flag: 'a page of the last session' };

export class CrashReport {
  constructor({ win = globalThis, doc = globalThis.document, log = pageLog, session = pageSession } = {}) {
    Object.assign(this, { win, doc, log, session, api: null, left: [], leftChecked: null, checking: null,
      report: null, choice: null, text: null, dialog: null, gone: false });
    this.shown = new Set();   // the reports offered in this page: each once
  }

  /** As the page starts: the pages found that did not unload are kept (`left`, [{session, at, found: 'tab' | 'flag'}];
   *  returned), and this page's own marks take their places: the window's at once, the shared flag once this page holds
   *  its lock. pagehide removes them (only ours: another window's flag stays), or puts back what the engine was not told
   *  yet (the window's page for a reload of it, and in the shared flag, which a window opened next reads). */
  start() {
    const { win, doc, session } = this, shared = storageOf(win, 'localStorage'), tab = storageOf(win, 'sessionStorage');
    const mine = JSON.stringify({ session: session.session, at: session.at });
    const write = (storage, key, text) => { try { storage?.setItem(key, text); } catch { /* full or blocked: none */ } };
    const clear = (storage, key) => { try { if (storage?.getItem(key) === mine) storage.removeItem(key); } catch { /* blocked */ } };
    const flagged = readFlag(shared), before = readFlag(tab, TAB_KEY);
    write(tab, TAB_KEY, mine);
    const flag = () => { if (!this.gone) write(shared, ALIVE_KEY, mine); };
    try {
      const held = win.navigator?.locks?.request?.(lockName(session.session), () => { flag(); return new Promise(() => {}); });
      if (held) held.catch?.(flag); else flag();                       // no Web Locks: at once
    } catch { flag(); }
    const found = [];
    if (doc?.wasDiscarded && (flagged || before)) this.log.add('crash', 'the browser had put this page aside to save memory and loaded it again: not a crash');
    else {
      if (before && before.session !== session.session) found.push({ ...before, found: 'tab' });
      if (flagged && flagged.session !== session.session && flagged.session !== before?.session) found.push({ ...flagged, found: 'flag' });
    }
    this.left = found;
    if (found.length) {
      this.leftChecked = Promise.all(found.map(page => this.stillOpen(page))).then(open => found.forEach((page, i) => {
        if (!open[i] || !this.left.includes(page)) return;
        this.left = this.left.filter(other => other !== page);
        this.log.add('crash', page.found === 'tab' ? 'the window this one was copied from is still open: not a crash' :
          'another window of the app is open: its flag is not a crash');
      }));
    }
    win.addEventListener?.('pagehide', () => {
      this.gone = true;
      const crashed = this.left.find(page => page.found === 'tab'), untold = crashed ?? this.left[0];
      const text = page => JSON.stringify({ session: page.session, at: page.at });
      if (untold) write(shared, ALIVE_KEY, text(untold)); else clear(shared, ALIVE_KEY);
      if (crashed) write(tab, TAB_KEY, text(crashed)); else clear(tab, TAB_KEY);
    });
    win.addEventListener?.('pageshow', event => {                                 // the browser brought it back
      if (event?.persisted) { this.gone = false; write(tab, TAB_KEY, mine); flag(); }
    });
    return found;
  }

  /** Whether the page that set `flag` is still open: it holds its lock, or has asked for it (false without Web Locks). */
  async stillOpen(flag) {
    try {
      const locks = await this.win.navigator?.locks?.query?.(), name = lockName(flag.session);
      return !![...(locks?.held ?? []), ...(locks?.pending ?? [])].some(lock => lock.name === name);
    } catch { return false; }
  }

  /** Once the engine has answered (main.js boot): the pages found that did not unload go to the engine (POST
   *  /crash/page, each), else the newest report not seen is asked for (GET /crash); a report -> the popup. `quiet` (under
   *  ?harness=1, whose scenarios press on the page): nothing is asked or shown, the api is kept for a scenario's own
   *  check(). -> the report offered, or null. */
  check({ api = this.api, quiet = false } = {}) {
    this.api = api;
    if (quiet || !api) return Promise.resolve(null);
    return this.checking ??= this.ask(api).finally(() => { this.checking = null; });
  }
  async ask(api) {
    await this.leftChecked;
    let answer = null;
    const failed = error => this.log.add('crash', `could not ask the engine about the last session: ${error?.message || error}`);
    if (!this.left.length) {
      try { answer = await api.crash(); } catch (error) { failed(error); return null; }
    }
    for (const page of [...this.left]) {
      try {
        answer = await api.crashPage(page);
        this.log.add('crash', `${said[page.found]} went without closing (it started ${page.at}): the engine was told`);
      } catch (error) {
        failed(error);
        if (!Number.isFinite(error?.status)) return null;             // the engine is away: told at the next start
      }
      this.left = this.left.filter(other => other !== page);          // told (or refused by the engine: once is enough)
    }
    return answer?.report?.id ? this.show(answer.report) : null;
  }

  /** The popup for `report` ({id, at, reason, bytes, started}), Save a copy… focused; null when it was offered already
   *  or another is open. Its text is asked for now, so Save a copy… opens the save dialog while the click still counts. */
  show(report) {
    if (!report?.id || this.shown.has(report.id) || this.dialog?.open) return null;
    this.shown.add(report.id);
    Object.assign(this, { report, choice: null, text: this.read(report.id) });
    const d = this.build(), { error, save } = d.parts;
    error.hidden = true; error.textContent = '';
    this.log.add('crash', `offered the report of ${report.at ?? '?'} (${report.reason ?? '?'}, ${report.bytes ?? '?'} bytes)`);
    if (!d.open) d.showModal();
    save.focus();
    return report;
  }
  read(id) {
    const text = Promise.resolve().then(() => this.api.crashText(id));
    text.catch(() => {});                                            // said when Save a copy… is pressed
    return text;
  }
  build() {
    if (this.dialog) return this.dialog;
    const T = CRASH_TEXT, el = (tag, props) => Object.assign(this.doc.createElement(tag), props);
    const d = el('dialog', { className: 'modal crash-report', id: 'crash-report' });
    d.setAttribute('aria-labelledby', 'crash-title'); d.setAttribute('aria-describedby', 'crash-text');
    const error = el('p', { id: 'crash-error', className: 'error', hidden: true });
    const later = el('button', { id: 'crash-later', type: 'button', textContent: T.later });
    const save = el('button', { id: 'crash-save', type: 'button', className: 'primary', textContent: T.save });
    const actions = el('div', { className: 'modal-actions' });
    actions.append(later, save);
    d.append(el('h2', { id: 'crash-title', textContent: T.title }), el('p', { id: 'crash-text', textContent: T.text }), error, actions);
    this.doc.body.append(d);
    later.onclick = () => this.later();
    save.onclick = () => this.save();
    d.addEventListener('cancel', () => this.later());                                // Escape
    d.addEventListener('close', () => { if (this.report && !this.choice) this.later(); });   // closed any other way
    d.parts = { error, save, later };
    return this.dialog = d;
  }

  /** Not now (Escape too): seen, and the popup goes. */
  later() { if (this.report && !this.choice) this.finish('later'); }

  /** Save a copy…: the report's text where the user says (the Log button's save flow), named after the report's time;
   *  seen. A text that could not be read is said in the popup, which stays (a press asks for it again). */
  async save() {
    if (!this.report || this.choice) return null;
    const { save, error } = this.dialog.parts;
    if (save.getAttribute('aria-busy') === 'true') return null;     // the log is still on its way
    const done = busy(save, CRASH_TEXT.reading);
    let text;
    try { text = await this.text; } catch (failure) {
      error.textContent = CRASH_TEXT.unread.replace('{why}', () => failure?.message || String(failure)); error.hidden = false;
      this.text = this.read(this.report.id);
      return null;
    } finally { done(); }
    if (this.choice) return null;                                     // Not now meanwhile
    const name = crashFileName(this.report.at);
    this.finish('save');
    try {
      const result = await saveText(text, name, { win: this.win, doc: this.doc });
      if (result.saved) this.notify((result.download ? CRASH_TEXT.downloaded : CRASH_TEXT.saved).replace('{name}', () => result.name));
      this.log.add('crash', result.saved ? `saved a copy: ${result.name}` : 'saving a copy was cancelled');
      return result;
    } catch (failure) {
      this.notify(CRASH_TEXT.failed.replace('{why}', () => failure?.message || String(failure)));
      return { saved: false, error: failure };
    }
  }

  finish(choice) {
    this.choice = choice;
    const id = this.report.id;
    this.log.add('crash', choice === 'save' ? 'Save a copy… pressed' : 'Not now');
    Promise.resolve().then(() => this.api.crashSeen(id))
      .catch(error => this.log.add('crash', `could not mark the report seen: ${error?.message || error}`));
    if (this.dialog.open) this.dialog.close();
  }

  notify(text) { try { this.win.dispatchEvent?.(new this.win.CustomEvent('fetch-notice', { detail: text })); } catch { /* no page */ } }

  /** For the development harness: {open, report (its id), choice ('save' | 'later' | null)}. */
  snapshot() { return { open: !!this.dialog?.open, report: this.report?.id ?? null, choice: this.choice }; }
}

/** The page's one. main.js boot calls check() through globalThis.sopaCompanionCrash (its tests run it without this module). */
export const crashReport = new CrashReport();
globalThis.sopaCompanionCrash = crashReport;
if (typeof window !== 'undefined' && typeof document !== 'undefined' && document.getElementById) crashReport.start();
