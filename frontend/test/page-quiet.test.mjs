import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync, readdirSync } from 'node:fs';
import { LayerPanel } from '../src/layer-panel.js';

// Nothing on the page animates by itself: every repaint is a frame a remote display encodes in software.
const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
test('the styles have no running animation, transition or backdrop blur; the scripts no smooth scrolling', () => {
  for (const name of ['app.css', 'tokens.css']) {
    const css = read(`../styles/${name}`).replace(/@media \(prefers-reduced-motion:reduce\)\{[^}]*\}\}/, '');
    assert.doesNotMatch(css, /@keyframes|animation\s*:|transition\s*:|backdrop-filter/, name);
  }
  assert.doesNotMatch(read('../index.html'), /animation|transition|<marquee/i);
  for (const name of readdirSync(new URL('../src/', import.meta.url))) {
    const source = read(`../src/${name}`);
    assert.doesNotMatch(source, /behavior\s*:\s*['"]smooth['"]/, name);
    assert.doesNotMatch(source, /animation\s*:|transition\s*:\s*[a-z]/i, `${name}: inline CSS animation`);
  }
});

test('the telemetry POST never raises a notice, whatever happens to it; other failures still do', async () => {
  const events = [], responses = { '/api/v1/telemetry': { ok: false, status: 404 }, '/api/v1/runs/a/cellstats/1': { ok: false, status: 500 } };
  const window = {
    fetch: async url => { if (url.endsWith('offline')) throw new TypeError('Failed to fetch'); return responses[url] || { ok: true, status: 200 }; },
    dispatchEvent: event => events.push(event.detail),
  };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } } });
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export /gm, ''), context);
  await window.fetch('/api/v1/telemetry', { method: 'POST' });
  assert.deepEqual(events, [], 'a missing telemetry endpoint is silent');
  await window.fetch('/api/v1/runs/a/cellstats/1');
  assert.equal(events.length, 1, 'other failures still notify');
  responses.offline = null;
  const quiet = vm.runInContext('window.fetch', context);
  await assert.rejects(() => quiet('offline'));
  assert.equal(events.length, 2);
  // An engine that is gone (closed, restarting): the telemetry POST fails without a notice, other requests notify.
  await assert.rejects(() => quiet('/api/v1/telemetry?offline'));
  assert.equal(events.length, 2, 'an unreachable engine is silent for telemetry');
});

test("the Log button's requests never raise the general notice (its own notice says what was saved); others still do", async () => {
  const events = [], responses = { '/api/v1/runs/r1/log': { ok: false, status: 404 }, '/api/v1/log': { ok: false, status: 500 },
    '/api/v1/runs/r1/logs': { ok: false, status: 500 }, '/api/v1/runs/log/frames': { ok: false, status: 500 } };
  const window = {
    fetch: async url => { if (url.endsWith('offline')) throw new TypeError('Failed to fetch'); return responses[url] || { ok: true, status: 200 }; },
    dispatchEvent: event => events.push(event.detail),
  };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } } });
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export /gm, ''), context);
  await window.fetch('/api/v1/runs/r1/log');                         // the run was deleted: log-export asks for /log instead
  await window.fetch('/api/v1/log');
  await assert.rejects(() => window.fetch('/api/v1/log?offline'));
  assert.deepEqual(events, []);
  await window.fetch('/api/v1/runs/r1/logs'); await window.fetch('/api/v1/runs/log/frames');
  assert.equal(events.length, 2, 'only the two log endpoints are quiet');
});

test("Compare's requests and Layers > Load Other Mask's never raise the general notice (each says in its own place what failed)", async () => {
  const events = [], failed = { ok: false, status: 409 };
  const window = { fetch: async () => failed, dispatchEvent: event => events.push(event.detail) };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } } });
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export /gm, ''), context);
  for (const path of ['/api/v1/runs/r1/compare', '/api/v1/runs/r1/compare/points/2.bin', '/api/v1/samples/sample-1/segmentations',
    '/api/v1/samples/sample-1/segmentations/cellpose_test/outlines?v=0123456789ab']) await window.fetch(path);
  assert.deepEqual(events, []);
  for (const path of ['/api/v1/runs/r1/comparison', '/api/v1/samples/sample-1/shapes/cellpose_test']) await window.fetch(path);
  assert.equal(events.length, 2, 'only their own paths are quiet');
});

test("the export dialog's listings, an export's refusals and Open in Cellpose GUI's never raise the general notice (the panel says them); their failures still do", async () => {
  const events = [];
  let status = 404;
  const window = { fetch: async () => ({ ok: false, status }), dispatchEvent: event => events.push(event.detail) };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } } });
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export /gm, ''), context);
  for (status of [404, 500]) {
    await window.fetch('/api/v1/runs/r1/stain/patches/folders?path=%2Ftmp');
    await window.fetch('/api/v1/runs/r1/stain/patches/folders');
  }
  for (status of [409, 422]) await window.fetch('/api/v1/runs/r1/stain/patches/export', { method: 'POST' });
  for (status of [404, 503]) await window.fetch('/api/v1/runs/r1/stain/patches/open', { method: 'POST' });
  assert.deepEqual(events, []);
  status = 503; await window.fetch('/api/v1/runs/r1/stain/patches/export', { method: 'POST' });
  status = 404; await window.fetch('/api/v1/runs/r1/stain/patches');
  status = 500; await window.fetch('/api/v1/runs/r1/stain/patches/open', { method: 'POST' });
  assert.equal(events.length, 3, 'a failed export, the patches read and a failed open still notify');
});

test("Clusters & UMAP settings the engine refuses, and a result of earlier settings, never raise the general notice; other failures still do", async () => {
  const events = [], answers = { 'POST /api/v1/runs/a/analysis': 422, 'GET /api/v1/runs/a/analysis/3?generation=1': 404,
    'GET /api/v1/runs/a/analysis/3/cells?generation=1': 404, 'GET /api/v1/runs/a/analysis/3': 404, 'GET /api/v1/runs/a/analysis/3/cells': 404 };
  const window = {
    fetch: async (url, options) => ({ ok: false, status: answers[`${options?.method ?? 'GET'} ${url}`] }),
    dispatchEvent: event => events.push(event.detail),
  };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } } });
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export /gm, ''), context);
  await window.fetch('/api/v1/runs/a/analysis', { method: 'POST' });      // the settings' own line says it
  await window.fetch('/api/v1/runs/a/analysis/3?generation=1');            // asked by a generation that is gone
  await window.fetch('/api/v1/runs/a/analysis/3/cells?generation=1');
  assert.deepEqual(events, []);
  await window.fetch('/api/v1/runs/a/analysis/3');                         // no generation: still a notice
  await window.fetch('/api/v1/runs/a/analysis/3/cells');
  answers['POST /api/v1/runs/a/analysis'] = 500;
  await window.fetch('/api/v1/runs/a/analysis', { method: 'POST' });       // a failure, not a refusal
  assert.equal(events.length, 3);
});

test('whole-sample channel visibility: as listed, else as saved, else only the first; unloaded channels are 16-bit', () => {
  const panel = Object.create(LayerPanel.prototype);
  Object.assign(panel, { channels: [], saved: { channels: { DAPI: { on: false }, Cy5: { on: true } } } });
  assert.deepEqual([['DAPI', 0], ['Cy5', 3], ['New', 0], ['Other', 1]].map(([name, k]) => panel.channelOn({ name }, k)), [false, true, true, false]);
  panel.channels = [{ name: 'DAPI', on: true }];
  assert.equal(panel.channelOn({ name: 'DAPI' }, 0), true, 'the channels on screen are what the user sees now');
  const c = panel.channel({ name: 'Cy5', max: 4000, p_lo: 10, p_hi: 900 }, null, 3);
  assert.deepEqual([c.depth, c.max, c.low, c.high, c.on, c.source], [16, 4000, 10, 900, true, null]);
});

test('leave, cells and a disappeared run are quiet; API payloads carry answers, levels and binary cells',async()=>{
  const requests=[],events=[],buffer=new ArrayBuffer(48);let status=200;
  const window={fetch:async(url,options)=>{requests.push([url,options]);return {ok:status===200,status,
    json:async()=>({id:'r',detail:'No longer here'}),arrayBuffer:async()=>buffer};},dispatchEvent:e=>events.push(e.detail)};
  const context=vm.createContext({window,URLSearchParams,CustomEvent:class{constructor(type,{detail}){this.detail=detail;}},pageSession:{session:'page-1'}});
  context.fetch=(...args)=>window.fetch(...args);
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export /gm,''),context);const api=vm.runInContext('api',context);
  await api.leave('r/a','add');assert.equal(requests.at(-1)[0],'/api/v1/runs/r%2Fa/leave');
  assert.deepEqual(JSON.parse(requests.at(-1)[1].body),{answer:'add',page:'page-1'});
  await api.createRun('s',{kind:'full'},{cells:{kind:'shapes',key:'saved_cells'},nuclei:null},{},'old-run');
  assert.deepEqual(Object.keys(JSON.parse(requests.at(-1)[1].body)),['sample','region','prior','options']);
  await api.createStain('s',{kind:'full'},{method:'cellpose',level:2});
  assert.equal(JSON.parse(requests.at(-1)[1].body).stain.level,2);
  const signal=new AbortController().signal;
  assert.equal(await api.stainCells('r/a','set & 2','DAPI / 2',signal),buffer);
  const [url,options]=requests.at(-1),parsed=new URL(url,'http://local');
  assert.equal(parsed.searchParams.get('set'),'set & 2');assert.equal(parsed.searchParams.get('channel'),'DAPI / 2');
  assert.equal(options.signal,signal);assert.equal(options.credentials,'same-origin');
  status=404;await assert.rejects(api.run('gone'),e=>e.status===404);await assert.rejects(api.frames('gone'),e=>e.status===404);
  await assert.rejects(api.leave('gone','discard'),e=>e.status===404);await assert.rejects(api.stainCells('gone','set'),e=>e.status===404);
  assert.deepEqual(events,[]);
  await window.fetch('/api/v1/runs/gone/cellstats/1');assert.equal(events.length,1);
});

test('the test of a setting asks quietly: its pool, its start and their refusals never raise the general notice (the corner says why)', async () => {
  const events = [], asked = [];
  const window = {
    fetch: async (url, init = {}) => {
      asked.push([url, init.method ?? 'GET', init.signal ?? null]);
      const refused = { status: url.includes('cell=9') ? 404 : 409, ok: false, statusText: 'x',
        json: async () => ({ detail: 'This run ended before its last iteration', status: 409,
          reasons: [{ code: 'ended', words: 'This run ended before its last iteration' }] }) };
      return url.includes('/pool?') && !url.includes('cell=9') ? { ok: true, status: 200, json: async () => ({ v: 1, state: 'ready' }) } : refused;
    },
    dispatchEvent: event => events.push(event.detail),
  };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } },
    pageKey: null, pageSession: () => null, pageShowing: () => true });
  vm.runInContext(read('../src/api.js').replace(/^import .*;\n/gm, '').replace(/^export const api =/m, 'var api =').replace(/^export /gm, ''), context);
  context.fetch = window.fetch;   // the page's own fetch, as api.js wraps it (in a browser the window is the global)
  const signal = new AbortController().signal;
  assert.deepEqual(await vm.runInContext('api', context).qcTestPool('r 1', 7, { wait: 5, signal }), { v: 1, state: 'ready' });
  await vm.runInContext('api', context).qcTestPool('r1', 6);
  assert.deepEqual(asked.slice(0, 2).map(([url, method]) => [url, method]), [['/api/v1/runs/r%201/qc/test/pool?cell=7&wait=5', 'GET'],
    ['/api/v1/runs/r1/qc/test/pool?cell=6&wait=0', 'GET']]);
  assert.equal(asked[0][2], signal, 'the page can give up on it');
  await assert.rejects(vm.runInContext('api', context).qcTestPool('r1', 9), error => error.status === 404);
  // A refused start: its words and its reasons (ApiError data.reasons) for the corner; nothing general said.
  await assert.rejects(vm.runInContext('api', context).qcTestStart('r1', { cell: 6, flag: 'cell-compactness', value: 0.02 }),
    error => error.status === 409 && error.data.reasons[0].code === 'ended');
  assert.deepEqual(events, [], 'quiet: they sit under /qc');
});
