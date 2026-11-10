import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';

// api.js's live socket and images (page-key.mjs signs the page in): the socket signs in in its handshake (its address
// names the page, its subprotocols carry the key), never in a message and never with the key in its address; the
// engine refusing the key (4401) signs the page out; the page going says goodbye and does not come back, unless the
// browser brings it back, as its next showing (the address names it, as the page's goodbyes do). An image is fetched
// (the key goes in a header, which an <img> cannot send) and decoded.
const SOURCE = readFileSync(new URL('../src/api.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
const KEY = 'K'.repeat(43);

/** api.js in a page of its own: its sockets (each records what it sent; open(), drop(code) as the browser would), its
 *  timers (run by hand), its key (`key` null: none; `ready` a promise of it), what it was told, and its showing
 *  (log-export.mjs pageShowing). */
function page({ key = KEY, ready = Promise.resolve(key) } = {}) {
  const sockets = [], timers = [], intervals = new Map(), told = [], showing = { visit: 1 };
  class WebSocket {
    static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
    constructor(url, protocols) { Object.assign(this, { url, protocols, readyState: 0, sent: [] }); sockets.push(this); }
    send(text) { this.sent.push(JSON.parse(text)); }
    close() { this.readyState = 3; }
    open() { this.readyState = 1; this.onopen?.(); }
    drop(code = 1006) { this.readyState = 3; this.onclose?.({ code }); }
  }
  let signedIn = key;
  const keys = { ready, protocols: () => signedIn ? ['sopa-companion', `key.${signedIn}`] : null,
    signedOut: reason => { told.push(['signed out', reason]); signedIn = null; } };
  const context = vm.createContext({ window: { fetch: async () => ({ ok: true, status: 200 }), dispatchEvent() {} }, WebSocket,
    location: { protocol: 'http:', host: '127.0.0.1:8799' }, JSON, Math, Promise, CustomEvent: class {},
    setTimeout: (fn, ms) => { timers.push([fn, ms]); return timers.length; }, clearTimeout() {},
    setInterval: (fn, ms) => { const id = intervals.size + 1; intervals.set(id, [fn, ms]); return id; }, clearInterval: id => intervals.delete(id),
    pageKey: keys, pageSession: { session: 'page-1', at: '2026-10-25T10:00:00.000Z' }, pageShowing: showing });
  vm.runInContext(SOURCE, context);
  const handlers = { messages: [], connections: [], message: m => handlers.messages.push(m), connection: ok => handlers.connections.push(ok) };
  return { context, sockets, timers, intervals, told, keys, handlers, showing, LiveSocket: vm.runInContext('LiveSocket', context) };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test("the socket waits for the page's key, then opens with the page's name in its address and the key in its handshake", async () => {
  let given;
  const p = page({ ready: new Promise(resolve => { given = resolve; }) });
  const live = new p.LiveSocket(p.handlers);
  await settle();
  assert.equal(p.sockets.length, 0, 'not before the key is known');
  given(KEY); await settle();
  const [socket] = p.sockets;
  assert.equal(socket.url, 'ws://127.0.0.1:8799/api/v1/ws?page=page-1&visit=1');
  assert.deepEqual(socket.protocols, ['sopa-companion', `key.${KEY}`]);
  assert.ok(!socket.url.includes(KEY), 'never the key in its address');
  live.watch('r1', 4);
  assert.equal(live.connected(), false, 'connecting: not counted among the open pages yet');
  socket.open();
  assert.equal(live.connected(), true, 'open: one of the pages the engine counts (main.js Close)');
  // Nothing to sign in with: what it says first is what it always said (the run it watches), then its pings.
  assert.deepEqual(socket.sent, [{ op: 'watch', run: 'r1', since: 4 }]);
  const [[ping, every]] = p.intervals.values();
  ping(); assert.deepEqual([socket.sent.at(-1), every], [{ op: 'ping' }, 20000]);
  assert.deepEqual(p.handlers.connections, [true]);
  socket.onmessage({ data: JSON.stringify({ ev: 'frames', run: 'r1', from: 4, frames: [{}, {}] }) });
  assert.deepEqual([p.handlers.messages.length, live.since], [1, 6]);
});

test('a page without a key never connects', async () => {
  const p = page({ key: null });
  new p.LiveSocket(p.handlers);
  await settle();
  assert.deepEqual([p.sockets.length, p.timers.length], [0, 0]);
});

test('a connection lost comes back by itself; one the engine refused for its key (4401) stops and signs the page out', async () => {
  const p = page(), live = new p.LiveSocket(p.handlers);
  await settle();
  p.sockets[0].open(); p.sockets[0].drop(1006);
  assert.deepEqual([p.timers.length, p.timers[0][1], p.handlers.connections], [1, 300, [true, false]]);
  assert.equal(live.connected(), false);
  p.timers[0][0](); await settle();
  assert.equal(p.sockets.length, 2, 'connected again');
  p.sockets[1].drop(4401);
  assert.deepEqual(p.told, [['signed out', 'refused']]);
  assert.deepEqual([live.stopped, p.timers.length, p.handlers.connections.at(-1)], [true, 1, false], 'never again');
});

test("goodbye says bye on the open socket and the socket does not come back; back() opens it again as the page's next showing", async () => {
  const p = page(), live = new p.LiveSocket(p.handlers);
  await settle();
  const [first] = p.sockets;
  first.open();
  live.goodbye();
  assert.deepEqual(first.sent.at(-1), { op: 'bye' });
  first.drop(1001);                                                     // the browser closes it as the page goes
  assert.equal(p.timers.length, 0, 'no reconnect after a goodbye');
  live.back(); await settle();
  assert.equal(p.sockets.length, 2, 'brought back: a new socket');
  assert.equal(p.sockets[1].url, 'ws://127.0.0.1:8799/api/v1/ws?page=page-1&visit=2', 'its next showing');
  assert.equal(p.showing.visit, 2, "which the page's goodbyes name too");
  p.sockets[1].open();
  // A socket the browser kept open while the page was aside, after its goodbye: replaced, so the engine hears the page
  // is back as its new showing (a new connection takes the goodbye back; one of the showing before, late, does not
  // count).
  live.goodbye(); live.back(); await settle();
  assert.deepEqual([p.sockets.length, p.sockets[1].readyState, p.sockets[2].url], [3, 3, 'ws://127.0.0.1:8799/api/v1/ws?page=page-1&visit=3']);
  p.sockets[1].drop(1000);
  assert.equal(p.timers.length, 0, 'the old one going is no longer heard');
  // An open socket that said no goodbye is replaced too: the engine must hear of every new showing.
  p.sockets[2].open(); live.back(); await settle();
  assert.deepEqual([p.sockets.length, p.sockets[2].readyState, p.sockets[3].url, p.showing.visit],
    [4, 3, 'ws://127.0.0.1:8799/api/v1/ws?page=page-1&visit=4', 4]);
  // A socket replaced meanwhile says nothing more: what the old one does is no longer heard.
  const q = page(), other = new q.LiveSocket(q.handlers);
  await settle();
  q.sockets[0].open(); other.goodbye(); q.sockets[0].readyState = 2;      // closing as the page went aside
  other.back(); await settle();
  q.sockets[0].drop(1001);
  assert.deepEqual([q.sockets.length, q.timers.length, q.handlers.connections], [2, 0, [true]]);
  // Stopped (a notebook's session that ended): neither reconnects.
  const r = page(), stopped = new r.LiveSocket(r.handlers);
  await settle();
  stopped.stop(); r.sockets[0].drop(1006);
  assert.equal(r.timers.length, 0);
});

test('an image is fetched with the page\'s fetch, which signs it, decoded, and its object address revoked; a refusal throws', async () => {
  const p = page(), fetched = [], made = [], revoked = [];
  let status = 200;
  p.context.fetch = async (url, init) => { fetched.push([url, init]); return { ok: status === 200, status, blob: async () => ({ type: 'image/png' }) }; };
  p.context.URL = { createObjectURL: blob => { made.push(blob); return 'blob:http://127.0.0.1:8799/1'; }, revokeObjectURL: url => revoked.push(url) };
  p.context.Image = class { async decode() { this.decoded = this.src; } };
  const api = vm.runInContext('api', p.context), signal = new AbortController().signal;
  const image = await api.image(api.overviewUrl('sample 1'), signal);
  assert.deepEqual(fetched.map(([url, init]) => [url, Object.keys(init), init.signal === signal]),
    [['/api/v1/samples/sample%201/overview.png', ['signal'], true]]);
  assert.deepEqual([image.decoded, made.length, revoked], ['blob:http://127.0.0.1:8799/1', 1, ['blob:http://127.0.0.1:8799/1']]);
  assert.equal(api.imageUrl('r1', 'c0.png'), '/api/v1/runs/r1/image/c0.png');
  status = 404;
  await assert.rejects(api.image(api.imageUrl('r1', 'c0.png')), error => error.status === 404 && error.message === 'HTTP 404');
  assert.equal(made.length, 1, 'nothing made of a refusal');
  // A failed decode still revokes its address.
  status = 200; p.context.Image = class { async decode() { throw new Error('broken image'); } };
  await assert.rejects(api.image('/api/v1/runs/r1/image/c1.png'), /broken image/);
  assert.equal(revoked.length, 2);
});

test("the page's images, its health, a frame's count of each cell and a signed-out answer raise no general notice: the page says them in its own words", async () => {
  const events = [], answers = { '/api/v1/samples/s1/overview.png': 404, '/api/v1/runs/r1/image/c0.png': 500, '/api/v1/runs': 401, '/api/v1/samples': 500,
    '/api/v1/health': 500, '/api/v1/runs/r1/counted/12': 404, '/api/v1/runs/r1/cellstats/12': 500 };
  const window = { fetch: async url => { if (url === '/api/v1/health?down') throw new TypeError('Failed to fetch'); return { ok: false, status: answers[url] }; },
    dispatchEvent: event => events.push(event.detail) };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.detail = detail; } } });
  vm.runInContext(SOURCE, context);
  // A counted frame's count of each cell is the frame's extra: without it the readouts keep its non-background count.
  for (const url of ['/api/v1/samples/s1/overview.png', '/api/v1/runs/r1/image/c0.png', '/api/v1/runs', '/api/v1/health', '/api/v1/runs/r1/counted/12']) await window.fetch(url);
  await assert.rejects(window.fetch('/api/v1/health?down'));            // the engine gone: the first screen, or Close, says it
  assert.deepEqual(events, []);
  await window.fetch('/api/v1/samples');
  assert.deepEqual(events, ['Request failed (500). Please try again.'], 'other failures still do');
  await window.fetch('/api/v1/runs/r1/cellstats/12');
  assert.equal(events.length, 2, "the frame's statistics themselves still do");
});
