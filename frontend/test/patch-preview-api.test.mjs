import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';

// api.js: the picker's Training patches preview (POST /samples/{id}/patches/preview, nothing written). Its refusals are
// shown under Start in the engine's words (main.js), never as the page's general notice.
const source = readFileSync(new URL('../src/api.js', import.meta.url), 'utf8').replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
const answer = { patches: [{ id: 1, box: [0, 0, 256, 256] }], run_pixel_to_sample: [10, 20, .5, .5], message: '' };
function client(responses = {}) {
  const requests = [], notices = [];
  const window = {
    fetch: async (url, options) => { requests.push({ url, ...options }); return responses[url] || { ok: true, status: 200, json: async () => answer }; },
    dispatchEvent: event => notices.push(event.detail),
  };
  window.fetch.bind = function () { return this; };
  const context = vm.createContext({ window, CustomEvent: class { constructor(type, { detail }) { this.type = type; this.detail = detail; } } });
  context.fetch = (...args) => context.window.fetch(...args);   // api.js requests go through its own observer
  return { api: vm.runInContext(`${source}\napi`, context), requests, notices };
}
const plain = value => JSON.parse(JSON.stringify(value));

test('previewPatches posts the region and settings to the sample\'s preview, with the signal that cancels it', async () => {
  const h = client(), controller = new AbortController();
  const region = { kind: 'box', box: [0, 0, 300, 300] }, stain = { method: 'cellpose', model: 'cpsam', count: 8, size: 256, start_from: 'model' };
  assert.deepEqual(plain(await h.api.previewPatches('sample a', region, stain, controller.signal)), answer);
  const [request] = h.requests;
  assert.equal(request.url, '/api/v1/samples/sample%20a/patches/preview'); assert.equal(request.method, 'POST');
  assert.deepEqual(JSON.parse(request.body), { region, stain }, 'as creating the item would send them');
  assert.equal(request.signal, controller.signal);
  assert.deepEqual(h.notices, []);
});

test('a refused preview carries the engine\'s reason and status without a notice; other failures still notify', async () => {
  const url = '/api/v1/samples/s/patches/preview', reason = 'This region has no tissue. Untick Tissue only to use it anyway.';
  const refused = client({ [url]: { ok: false, status: 422, statusText: 'Unprocessable Entity', json: async () => ({ detail: reason }) } });
  await assert.rejects(refused.api.previewPatches('s', { kind: 'full' }, {}), error => error.status === 422 && error.message === reason);
  assert.deepEqual(refused.notices, [], 'the picker says it under Start');
  const broken = client({ [url]: { ok: false, status: 500, statusText: 'Internal Server Error', json: async () => { throw new Error('not JSON'); } } });
  await assert.rejects(broken.api.previewPatches('s', { kind: 'full' }, {}), error => error.status === 500);
  assert.equal(broken.notices.length, 1, 'not a refusal: the general notice');
  const other = client({ '/api/v1/runs/r/stain/patches/propose': { ok: false, status: 422, statusText: 'Unprocessable Entity', json: async () => ({ detail: 'x' }) } });
  await assert.rejects(other.api.stainPropose('r', { version: 1 }));
  assert.equal(other.notices.length, 1, 'only the picker\'s preview is quiet about a refusal');
});
