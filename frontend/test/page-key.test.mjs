import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createPageKey, install, appWindow, pageKey, PAGE_HEADER, CODE_HEADER, SIGN_IN, STORE, SIGNED_OUT, SOCKET_PROTOCOL,
  KEY_PROTOCOL, WINDOW_STORE, SIGN_IN_MS } from '../src/page-key.mjs';

// The tab's sign-in (page-key.mjs): the launch code taken out of the address and traded once by POST for the page's key,
// kept in the tab; every request to the engine's /api/ carries the key in a header, the goodbye's keepalive POST too,
// and the socket offers it in its handshake. No key: nothing asked of the engine.
const KEY = 'K'.repeat(43), OLD = 'O'.repeat(43), ORIGIN = 'http://127.0.0.1:8799';
// An HTTP token (RFC 9110 tchar): what each of a WebSocket's subprotocols must be.
const TOKEN = /^[!#$%&'*+.^_`|~0-9A-Za-z-]+$/;

/** A tab: its address (rewritten by history.replaceState), its sessionStorage (`storage`: 'throws' when the browser
 *  blocks it), the events it was sent, and the engine's answers to `native` (the browser's own fetch), recorded with
 *  the address as it was when each was asked. */
function tab({ search = '', stored = null, storage = 'ok', answer = () => new Response(JSON.stringify({ key: KEY }), { status: 200 }) } = {}) {
  const store = new Map(stored ? [[STORE, stored]] : []), events = [], replaced = [], calls = [];
  const location = { origin: ORIGIN, pathname: '/', search, hash: '#here',
    get href() { return `${this.origin}${this.pathname}${this.search}${this.hash}`; } };
  const win = {
    location,
    history: { state: { kept: true }, replaceState(state, unused, url) {
      replaced.push([state, url]); const next = new URL(url, location.href);
      location.pathname = next.pathname; location.search = next.search; location.hash = next.hash; } },
    CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options?.detail; } },
    dispatchEvent: event => events.push([event.type, event.detail]),
  };
  if (storage === 'throws') Object.defineProperty(win, 'sessionStorage', { get() { throw new Error('The operation is insecure.'); } });
  else win.sessionStorage = { getItem: name => store.get(name) ?? null, setItem: (name, value) => store.set(name, String(value)),
    removeItem: name => store.delete(name) };
  const native = (input, init) => { calls.push({ input, init, address: location.search }); return Promise.resolve().then(() => answer(input, init)); };
  return { win, store, events, replaced, calls, native };
}

test('a launch code leaves the address before it is traded, once, by POST with the code in a header; the other parameters stay', async () => {
  const t = tab({ search: '?code=C0de-1_x&run=r1&harness=1&api=1f3a9c2e&window=chrome&token=bearer-in-the-address' });
  const keys = createPageKey({ win: t.win, native: t.native });
  // At once, before anything else reads the address: the code and a token taken out, the rest and the hash kept.
  assert.deepEqual(t.replaced, [[{ kept: true }, '/?run=r1&harness=1&api=1f3a9c2e&window=chrome#here']]);
  assert.equal(await keys.ready, KEY);
  assert.equal(t.calls.length, 1);
  const [{ input, init, address }] = t.calls;
  assert.equal(input, SIGN_IN);
  assert.deepEqual([init.method, init.headers, init.credentials, init.cache, init.body], ['POST', { [CODE_HEADER]: 'C0de-1_x' }, 'omit', 'no-store', undefined]);
  assert.doesNotMatch(address, /code|token/, 'the address no longer held the code when it was posted');
  assert.deepEqual([keys.reason(), t.store.get(STORE)], ['signed-in', KEY], "kept in the tab's storage");
  assert.deepEqual(keys.protocols(), [SOCKET_PROTOCOL, KEY_PROTOCOL + KEY]);
  assert.deepEqual(keys.headers(), { [PAGE_HEADER]: KEY });
});

test("a tab's stored key goes on without asking (a reload, a tab brought back); with no code and no key, nothing is asked", async () => {
  const kept = tab({ stored: OLD }), keys = createPageKey({ win: kept.win, native: kept.native });
  assert.deepEqual(keys.protocols(), [SOCKET_PROTOCOL, KEY_PROTOCOL + OLD], 'known at once: the socket need not wait');
  assert.deepEqual([await keys.ready, keys.reason(), kept.calls.length, kept.replaced.length], [OLD, 'signed-in', 0, 0]);
  const none = tab(), bare = createPageKey({ win: none.win, native: none.native });
  assert.deepEqual([await bare.ready, bare.reason(), none.calls.length, bare.protocols(), bare.headers()], [null, 'none', 0, null, {}]);
});

test('a code used already or too late: a tab signed in goes on with its key, any other says the link was used', async () => {
  const refused = () => new Response(JSON.stringify({ detail: 'This launch code was used already or has expired.' }), { status: 401 });
  const kept = tab({ search: '?code=used', stored: OLD, answer: refused }), keys = createPageKey({ win: kept.win, native: kept.native });
  assert.deepEqual([await keys.ready, keys.reason(), kept.store.get(STORE)], [OLD, 'signed-in', OLD]);
  const fresh = tab({ search: '?code=used', answer: refused }), other = createPageKey({ win: fresh.win, native: fresh.native });
  assert.deepEqual([await other.ready, other.reason(), fresh.store.has(STORE)], [null, 'expired', false]);
  // Another refusal (a foreign Origin, say), or a 200 without a key: refused; an engine out of reach: unreachable.
  for (const [answer, reason] of [[() => new Response('{}', { status: 403 }), 'refused'], [() => new Response('{}', { status: 200 }), 'refused'],
    [() => new Response('not json', { status: 200 }), 'refused'], [() => { throw new TypeError('Failed to fetch'); }, 'unreachable']]) {
    const t = tab({ search: '?code=c', stored: OLD, answer }), k = createPageKey({ win: t.win, native: t.native });
    assert.deepEqual([await k.ready, k.reason()], [null, reason], reason);
  }
  const lost = tab({ search: '?code=c' }), gone = createPageKey({ win: lost.win, native: undefined });
  assert.deepEqual([await gone.ready, gone.reason()], [null, 'unreachable'], 'no fetch at all');
});

test('a sign-in the engine took and never answered is given up after its time limit: out of reach, which the page says with Reload', async () => {
  // As fetch does: a request whose signal aborts is rejected, its answer's body too.
  const aborted = init => new Promise((_, reject) => init.signal.addEventListener('abort', () => reject(new DOMException('The operation was aborted.', 'AbortError'))));
  const stalled = tab({ search: '?code=c', stored: OLD, answer: (input, init) => aborted(init) });
  const keys = createPageKey({ win: stalled.win, native: stalled.native, limit: 30 });
  assert.deepEqual([await keys.ready, keys.reason()], [null, 'unreachable'], 'not a blank first screen for ever');
  assert.equal(stalled.calls[0].init.signal.aborted, true, 'the request itself is given up');
  // Its answer begun, its body never coming: the same.
  const body = tab({ search: '?code=c', answer: (input, init) => ({ status: 200, json: () => aborted(init) }) });
  const slow = createPageKey({ win: body.win, native: body.native, limit: 30 });
  assert.deepEqual([await slow.ready, slow.reason()], [null, 'unreachable']);
  // An answer in time is taken (its limit no longer runs), and the limit is the first screen's own (main.js BOOT_MS).
  const quick = tab({ search: '?code=c' }), signed = createPageKey({ win: quick.win, native: quick.native, limit: 30 });
  await new Promise(resolve => setTimeout(resolve, 60));
  assert.deepEqual([await signed.ready, quick.calls[0].init.signal.aborted], [KEY, false]);
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  assert.equal(SIGN_IN_MS, Number(main.match(/const BOOT_MS = (\d+);/)[1]));
});

test('storage the browser blocks: the key is kept in memory, so the page works until it reloads', async () => {
  const t = tab({ search: '?code=c', storage: 'throws' }), keys = createPageKey({ win: t.win, native: t.native });
  assert.deepEqual([await keys.ready, keys.reason()], [KEY, 'signed-in']);
  await keys.fetch('/api/v1/health');
  assert.equal(t.calls.at(-1).init.headers.get(PAGE_HEADER), KEY);
  assert.equal(keys.send('/api/v1/page-log', '{}'), true);
  assert.equal(appWindow(t.win), false, 'and the flag reads false');
});

test("the fetch signs only this engine's /api/ requests: the key in a header, the browser's credentials never sent, the rest kept", async () => {
  const t = tab({ stored: KEY, answer: () => new Response('{}', { status: 200 }) }), keys = createPageKey({ win: t.win, native: t.native });
  await keys.ready;
  const signal = new AbortController().signal;
  await keys.fetch('/api/v1/runs', { method: 'POST', body: '{"a":1}', signal, keepalive: true, credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' } });
  let { input, init } = t.calls.at(-1);
  assert.equal(input, '/api/v1/runs');
  assert.deepEqual([init.method, init.body, init.signal, init.keepalive, init.credentials], ['POST', '{"a":1}', signal, true, 'omit']);
  assert.deepEqual([init.headers.get(PAGE_HEADER), init.headers.get('Content-Type')], [KEY, 'application/json']);
  await keys.fetch(`${ORIGIN}/api/v1/health`);
  assert.equal(t.calls.at(-1).init.headers.get(PAGE_HEADER), KEY, 'the same origin, written whole');
  // A Request: its own headers kept, the key added.
  await keys.fetch(new Request(`${ORIGIN}/api/v1/log`, { headers: { 'X-Other': 'yes' } }));
  ({ init } = t.calls.at(-1));
  assert.deepEqual([init.headers.get('X-Other'), init.headers.get(PAGE_HEADER), init.credentials], ['yes', KEY, 'omit']);
  // Not the engine's API: another origin (another port of this computer included), the app's own files, the sign-in.
  for (const url of ['http://127.0.0.1:8888/api/v1/health', 'https://example.org/api/v1/health', '/src/main.js', '/index.html',
    '/apix', SIGN_IN]) {
    const given = { credentials: 'include', headers: { 'X-Mine': '1' } };
    await keys.fetch(url, given);
    assert.equal(t.calls.at(-1).init, given, `${url}: passed on as it was`);
  }
  assert.equal(t.calls.length, 9);
});

test('once the engine refuses the key, the page is signed out once: the key forgotten, and later requests answered 401 without asking', async () => {
  let status = 200;
  const t = tab({ stored: OLD, answer: () => new Response('{}', { status }) }), keys = createPageKey({ win: t.win, native: t.native });
  await keys.ready;
  assert.equal((await keys.fetch('/api/v1/runs')).status, 200);
  status = 401;
  assert.equal((await keys.fetch('/api/v1/runs')).status, 401);
  assert.deepEqual([t.events, keys.reason(), t.store.has(STORE), keys.protocols(), keys.headers()],
    [[[SIGNED_OUT, 'refused']], 'refused', false, null, {}]);
  const asked = t.calls.length, answer = await keys.fetch('/api/v1/health');
  assert.deepEqual([answer.status, answer.statusText, t.calls.length], [401, 'Not signed in', asked], 'nothing asked of the engine');
  assert.equal(keys.send('/api/v1/page-log', '{}'), false);
  keys.signedOut('refused'); (await keys.fetch('/api/v1/runs'));
  assert.equal(t.events.length, 1, 'said once');
  // A tab with no key at all: every API request is answered at once, the files still load.
  const none = tab(), bare = createPageKey({ win: none.win, native: none.native });
  assert.equal((await bare.fetch('/api/v1/health')).status, 401);
  await bare.fetch('/src/main.js');
  assert.deepEqual(none.calls.map(c => c.input), ['/src/main.js']);
});

test('send: a keepalive POST with the key, sent before it returns (a page going cannot wait), that never throws', async () => {
  const t = tab({ stored: KEY }), keys = createPageKey({ win: t.win, native: t.native });
  assert.equal(keys.send('/api/v1/closing', '{"answer":"discard"}'), true);
  assert.equal(t.calls.length, 1, 'sent at once');
  const [{ input, init }] = t.calls;
  assert.deepEqual([input, init.method, init.keepalive, init.credentials, init.body, init.headers],
    ['/api/v1/closing', 'POST', true, 'omit', '{"answer":"discard"}', { 'Content-Type': 'application/json', [PAGE_HEADER]: KEY }]);
  assert.equal(keys.send('/api/v1/page-log', 'x', 'text/plain'), true);
  assert.equal(t.calls.at(-1).init.headers['Content-Type'], 'text/plain');
  // Only to this engine's API, as every request the key goes with: anywhere else nothing is sent, so the key never is.
  for (const url of ['http://127.0.0.1:8888/api/v1/page-log', 'https://example.org/api/v1/closing', '/src/main.js', SIGN_IN])
    assert.equal(keys.send(url, '{}'), false, url);
  assert.equal(t.calls.length, 2, 'nothing more was sent');
  const failing = createPageKey({ win: tab({ stored: KEY }).win, native: () => { throw new TypeError('keepalive quota'); } });
  assert.equal(failing.send('/api/v1/page-log', 'x'), false);
  const rejected = createPageKey({ win: tab({ stored: KEY }).win, native: () => Promise.reject(new TypeError('Failed to fetch')) });
  assert.equal(rejected.send('/api/v1/page-log', 'x'), true, 'a failure on its way is the browser\'s, never thrown');
});

test("the socket's protocols: none before the key is known, then the page's and the key's, each an HTTP token", async () => {
  const t = tab({ search: '?code=c' }), keys = createPageKey({ win: t.win, native: t.native });
  assert.equal(keys.protocols(), null, 'the code is still being traded');
  await keys.ready;
  const protocols = keys.protocols();
  assert.deepEqual(protocols, ['sopa-companion', `key.${KEY}`]);
  for (const entry of protocols) assert.match(entry, TOKEN, entry);
  // A real key (secrets.token_urlsafe: base64url) is a token too.
  assert.match(KEY_PROTOCOL + 'Ab-_09xYz'.repeat(5), TOKEN);
});

test("appWindow: the app's own Chrome window says so in its address, and the tab remembers it once the page rewrote the address", () => {
  const plain = tab();
  createPageKey({ win: plain.win, native: plain.native });
  assert.equal(appWindow(plain.win), false);
  const chrome = tab({ search: '?code=c&window=chrome&harness=1' });
  createPageKey({ win: chrome.win, native: chrome.native });
  assert.equal(chrome.win.location.search, '?window=chrome&harness=1', 'the flag stays in the address, the code goes');
  assert.equal(appWindow(chrome.win), true);
  chrome.win.location.search = '?run=r1';                                  // main.js rewrites the address
  assert.deepEqual([appWindow(chrome.win), chrome.store.get(WINDOW_STORE)], [true, 'chrome']);
  const blocked = tab({ storage: 'throws' });
  assert.equal(appWindow(blocked.win), false);
  blocked.win.location.search = '?window=chrome';
  assert.equal(appWindow(blocked.win), true, 'the address alone says it');
  assert.equal(appWindow({}), false);
});

test("the key is never put on the window or globalThis: their own properties after sign-in are those before, and the window's fetch", async () => {
  const globals = new Set(Object.getOwnPropertyNames(globalThis));
  const t = tab({ search: '?code=c' }), before = new Set(Object.getOwnPropertyNames(t.win));
  const keys = install(createPageKey({ win: t.win, native: t.native }), t.win);
  await keys.ready;
  await keys.fetch('/api/v1/health'); keys.send('/api/v1/page-log', '{}'); keys.protocols(); keys.headers();
  assert.deepEqual(Object.getOwnPropertyNames(t.win).filter(name => !before.has(name)), ['fetch']);
  assert.deepEqual(Object.getOwnPropertyNames(globalThis).filter(name => !globals.has(name)), []);
  for (const name of Object.getOwnPropertyNames(t.win)) {
    const value = Object.getOwnPropertyDescriptor(t.win, name).value;
    assert.ok(typeof value !== 'string' || !value.includes(KEY), name);
  }
});

test("installed as the window's fetch, the first module index.html loads; in Node there is no window, so no page key", () => {
  const t = tab({ stored: KEY }), keys = createPageKey({ win: t.win, native: t.native });
  assert.equal(install(keys, t.win), keys);
  assert.equal(t.win.fetch, keys.fetch);
  assert.equal(pageKey, null);
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  const scripts = [...html.matchAll(/<script type="module" src="([^"]+)"><\/script>/g)].map(m => m[1]);
  assert.deepEqual(scripts, ['./src/page-key.mjs', './src/storage-boot.mjs', './src/main.js', './src/log-export.mjs']);
  // The page's key goes nowhere but a header and the socket's handshake: no global, no cookie, no address.
  const source = readFileSync(new URL('../src/page-key.mjs', import.meta.url), 'utf8');
  assert.doesNotMatch(source, /document\.cookie|globalThis\.\w+\s*=(?!=)|window\.\w+\s*=(?!=)|localStorage/);
  assert.equal(source.match(/^import /gm), null, 'no imports: it runs before anything else');
});
