import test from 'node:test';
import assert from 'node:assert/strict';
import { fitSize, liveValue, rasterSupported } from '../src/robustness.mjs';

test('image and volume dimensions preserve aspect and obey GPU limits', () => {
  assert.deepEqual(fitSize(7000, 3500, 4096), { width: 4096, height: 2048 });
  assert.deepEqual(fitSize(100, 200, 4096), { width: 100, height: 200 });
  assert.deepEqual(fitSize(0, 0, 4096), { width: 1, height: 1 });
});
test('labels refuse excessive dimensions and array layer counts', () => {
  const limits = { texture: 4096, layers: 256 };
  assert.equal(rasterSupported({ nx: 4096, ny: 4096, nz: 256 }, limits), true);
  assert.equal(rasterSupported({ nx: 4097, ny: 2, nz: 4 }, limits), false);
  assert.equal(rasterSupported({ nx: 2, ny: 2, nz: 257 }, limits), false);
});
test('live settings use startup, then effective frames, then pending overrides', () => {
  const option = { flag: 'cell-compactness', default: 0.04 };
  const run = { options: { 'cell-compactness': 0.2 } };
  assert.equal(liveValue(run, option).value, 0.2);
  run.last = { live: { 'cell-compactness': 0.3 } };
  assert.equal(liveValue(run, option).value, 0.3);
  run.live_values = { 'cell-compactness': 0.4 };
  assert.deepEqual(liveValue(run, option), { value: 0.4, effective: 0.3, pending: true, automatic: false });
  run.last.live['cell-compactness'] = 0.4;
  assert.equal(liveValue(run, option).pending, false);
});
test('false overrides and automatic temperatures are retained', () => {
  assert.equal(liveValue({ options: { flag: true }, live_values: { flag: false } }, { flag: 'flag', default: true }).value, false);
  assert.equal(liveValue({ last: { live: { temperature: 0.8 } } }, { flag: 'temperature' }).automatic, true);
});
