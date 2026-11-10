// The page's key: how this tab signs in to its engine, and proves it with every request. A window or a link opens the
// app with a one-use launch code in its address (?code=). This module, the first the page loads, takes the code out of
// the address at once, before any other module reads the address, and trades it for the page's key: one POST to the
// engine (SIGN_IN), the code in a header, never an address (a GET of the app's address spends no code, so a link
// preview cannot burn one). The key is kept in the tab's sessionStorage, which is per tab and per engine address (each
// engine is its own origin): a reload, a tab the browser brings back and a duplicated tab stay signed in, and two
// engines never share a key. Every request to the engine's /api/ then carries it in a header: this module's fetch,
// installed as window.fetch before any other module wraps it, the goodbye's keepalive requests too (send). The live
// socket offers it in its handshake (protocols()). The key is never put in an address, a cookie or a global. A tab
// with no key (an address typed by hand, an old tab of an engine that started again since) asks the engine nothing,
// and the page says it is not signed in (main.js signedOut).
// No imports: it runs before anything else.

export const PAGE_HEADER = 'X-SoPA-Page-Key', CODE_HEADER = 'X-SoPA-Launch-Code';
export const SIGN_IN = '/api/v1/session/sign-in', STORE = 'sopa-companion:page-key', SIGNED_OUT = 'sopa-signed-out';
// The page's socket offers both: the engine answers with the first, and reads the key from the second.
export const SOCKET_PROTOCOL = 'sopa-companion', KEY_PROTOCOL = 'key.';
// The app's own Chrome window ([desktop] browser = "chrome") opens with window=chrome in its address.
export const WINDOW_FLAG = 'window', WINDOW_STORE = 'sopa-companion:window';
// The sign-in is given up after this long, as the first screen's own requests are (main.js BOOT_MS): an engine that
// took it and never answered would leave the first screen blank for ever; given up, the page says the engine is out of
// reach, with Reload.
export const SIGN_IN_MS = 20000;

const stored = win => { try { return win.sessionStorage?.getItem(STORE) || null; } catch { return null; } };
const keep = (win, key) => { try { win.sessionStorage?.setItem(STORE, key); } catch { /* memory only: until a reload */ } };
const forget = win => { try { win.sessionStorage?.removeItem(STORE); } catch { /* memory only */ } };
const flagged = win => { try { return new URLSearchParams(win.location?.search || '').get(WINDOW_FLAG) === 'chrome'; } catch { return false; } };

/** {signal, stop()}: a signal that aborts after `ms` (the window's own AbortController and timer), and stop() once the
 *  request it goes with has answered. No AbortController: no limit. */
function timeLimit(win, ms) {
  const Controller = win?.AbortController || globalThis.AbortController;
  if (typeof Controller !== 'function') return { signal: undefined, stop() {} };
  const controller = new Controller(), abort = () => controller.abort();
  const timer = typeof win?.setTimeout === 'function' ? win.setTimeout(abort, ms) : setTimeout(abort, ms);
  return { signal: controller.signal, stop: () => (typeof win?.clearTimeout === 'function' ? win.clearTimeout(timer) : clearTimeout(timer)) };
}

/** The address's launch code, after `code` (and a `token`, which never belongs there) are taken out of the address
 *  with history.replaceState, its other parameters (run, harness, api, window) and its hash kept. */
function takeCode(win) {
  const location = win?.location;
  let params;
  try { params = new URLSearchParams(location?.search || ''); } catch { return null; }
  const code = params.get('code');
  if (params.has('code') || params.has('token')) {
    params.delete('code'); params.delete('token');
    const rest = params.toString();
    try { win.history.replaceState(win.history.state, '', `${location.pathname || '/'}${rest ? `?${rest}` : ''}${location.hash || ''}`); }
    catch { /* the address keeps it: the code is spent at once all the same */ }
  }
  return code || null;
}

/** -> {ready: Promise<key|null>, reason(): 'signed-in'|'none'|'expired'|'refused'|'unreachable',
 *      fetch(input, init), sign(input, init) -> [input, init], headers() -> {...}, send(url, body, type?) -> bool,
 *      protocols() -> [SOCKET_PROTOCOL, KEY_PROTOCOL + key] | null, signedOut(reason)}
 *  `ready` is settled once, as the page starts: the code in the address traded (or, without one, the key the tab
 *  kept); it never rejects. Its reason: 'signed-in'; 'none' (no code and no key kept); 'expired' (a code used already
 *  or expired, and no key kept); 'refused' (the engine refused otherwise, or refused the key later); 'unreachable' (no
 *  answer, or none within `limit` ms). */
export function createPageKey({ win = globalThis, native = win?.fetch?.bind(win), limit = SIGN_IN_MS } = {}) {
  const Headers = win?.Headers || globalThis.Headers, Response = win?.Response || globalThis.Response;
  let key = null, why = 'none', out = false;
  // The app's own window says so once, as the page starts; the page rewrites its address later (main.js).
  if (flagged(win)) { try { win.sessionStorage?.setItem(WINDOW_STORE, 'chrome'); } catch { /* the address says it until then */ } }
  const code = takeCode(win);
  const ready = (async () => {
    const kept = stored(win);
    if (!code) { key = kept; why = kept ? 'signed-in' : 'none'; return key; }
    const timed = timeLimit(win, limit);
    let response, given = null;
    try {
      response = await native(SIGN_IN, { method: 'POST', headers: { [CODE_HEADER]: code }, credentials: 'omit', cache: 'no-store',
        signal: timed.signal });
      if (response?.status === 200) {
        try { given = (await response.json())?.key; } catch { if (timed.signal?.aborted) throw new Error('no answer in time'); }
      }
    } catch { why = 'unreachable'; return null; }
    finally { timed.stop(); }
    if (response?.status === 200) {
      if (typeof given === 'string' && given) { key = given; keep(win, given); why = 'signed-in'; return key; }
    } else if (response?.status === 401) {
      // A link opened again (used, or too late): a tab that is signed in already goes on with its key.
      if (kept) { key = kept; why = 'signed-in'; return key; }
      why = 'expired'; return null;
    }
    why = 'refused'; return null;
  })();

  /** Whether `input` asks this engine's API (this origin's /api/, the sign-in itself aside): those carry the key. */
  function engine(input) {
    try {
      const base = win.location.href, url = new URL(input && typeof input === 'object' && 'url' in input ? input.url : String(input), base);
      return url.origin === new URL(base).origin && (url.pathname === '/api' || url.pathname.startsWith('/api/')) && url.pathname !== SIGN_IN;
    } catch { return false; }
  }
  /** [input, init] with the key in a copy of the request's headers (a Request's own included) and no credentials of
   *  the browser's (cookies) sent along. */
  function sign(input, init = {}) {
    const own = init?.headers ?? (input && typeof input === 'object' && input.headers ? input.headers : undefined);
    const headers = new Headers(own);
    if (key && !out) headers.set(PAGE_HEADER, key);
    return [input, { ...init, headers, credentials: 'omit' }];
  }
  async function signedFetch(input, init) {
    if (!engine(input)) return native(input, init);
    await ready;
    if (!key || out) return new Response(null, { status: 401, statusText: 'Not signed in' });
    const response = await native(...sign(input, init));
    if (response?.status === 401) signedOut('refused');
    return response;
  }
  /** A POST that goes on while the page goes (keepalive), with the key, sent now: a page that is going cannot wait for
   *  a promise, and a beacon cannot carry a header. Only to this engine's API, as every request the key goes with
   *  (anywhere else: nothing sent). Never throws; -> whether it was sent. */
  function send(url, body, type = 'application/json') {
    if (!key || out || !engine(url)) return false;
    try {
      const sent = native(url, { method: 'POST', keepalive: true, credentials: 'omit', headers: { 'Content-Type': type, [PAGE_HEADER]: key }, body });
      sent?.catch?.(() => {});
      return true;
    } catch { return false; }
  }
  /** Once: the key is forgotten (a tab that reloads asks nothing) and the page hears why (SIGNED_OUT). */
  function signedOut(reason = 'refused') {
    if (out) return;
    out = true; why = reason; key = null;
    forget(win);
    try { win.dispatchEvent?.(new (win.CustomEvent || globalThis.CustomEvent)(SIGNED_OUT, { detail: reason })); }
    catch { /* no page to tell */ }
  }
  return {
    ready, reason: () => why, fetch: signedFetch, sign, send, signedOut,
    headers: () => key && !out ? { [PAGE_HEADER]: key } : {},
    protocols: () => key && !out ? [SOCKET_PROTOCOL, KEY_PROTOCOL + key] : null,
  };
}

/** `pageKey`'s fetch as the window's: every module's requests go through it. */
export function install(pageKey, win = globalThis) {
  win.fetch = pageKey.fetch;
  return pageKey;
}

export const pageKey = typeof window !== 'undefined' && typeof window.fetch === 'function' ? install(createPageKey({ win: window }), window) : null;

/** Whether this tab is the app's own Chrome window (browser = "chrome"): its opening address said window=chrome, now or
 *  earlier in this tab (kept in sessionStorage[WINDOW_STORE], since main.js rewrites the address). Synchronous, so a
 *  page that cannot draw can choose its words before anything else runs. */
export function appWindow(win = globalThis) {
  if (flagged(win)) return true;
  try { return win.sessionStorage?.getItem(WINDOW_STORE) === 'chrome'; } catch { return false; }
}
