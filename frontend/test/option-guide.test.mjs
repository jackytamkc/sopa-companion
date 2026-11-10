import test from 'node:test';
import assert from 'node:assert/strict';
import { displayValue, validOptionValue, suggestionIssue, LiveOptionSession } from '../src/option-guide.mjs';

const compactness = { flag: 'cell-compactness', type: 'float', default: 0.04, min: 0.001, max: 10, live: true };
const diffusion = { flag: 'no-diffusion', type: 'bool', default: false, live: true };
const temperature = { flag: 'temperature', type: 'float', default: null, min: 0.01, max: 1, live: true };
const lower = { flag: compactness.flag, direction: 'lower', value: 0.02, live: true };
const run = (extra = {}) => ({ id: 'a', state: 'running', ...extra });

test('guide suggestions respect direction, bounds, types and false values', () => {
  assert.equal(suggestionIssue(lower, compactness, 0.04), '');
  assert.match(suggestionIssue(lower, compactness, 0.01), /beyond/);
  assert.match(suggestionIssue(lower, compactness, 0.02), /Already at/);
  assert.match(suggestionIssue({ ...lower, direction: 'raise' }, compactness, 0.04), /beyond/);
  assert.match(suggestionIssue({ ...lower, value: -1 }, compactness, 0.04), /unavailable/);
  assert.equal(suggestionIssue({ flag: diffusion.flag, direction: 'off', value: false }, diffusion, true), '');
  assert.equal(validOptionValue(diffusion, 0), false);
  assert.equal(validOptionValue(compactness, NaN), false);
  assert.equal(validOptionValue(compactness, '0.02'), false);
  assert.equal(validOptionValue({ type: 'int' }, 1.5), false);
  assert.equal(validOptionValue({ type: 'str', choices: ['infer', 'csv'] }, 'csv'), true);
  assert.equal(validOptionValue({ type: 'str', choices: ['infer'] }, 'csv'), false);
  assert.equal(displayValue(false), 'off');
  assert.equal(displayValue(0), '0');
  assert.equal(displayValue(null), 'automatic');
});

test('automatic values can be prefilled for a new run, but live Try waits for an undoable value', () => {
  const action = { flag: 'temperature', direction: 'lower', value: 0.1 };
  assert.match(suggestionIssue(action, temperature, null), /Waiting/);
  assert.equal(suggestionIssue(action, temperature, null, false), '');
});

test('Try and Undo send the previous requested value, including a not-yet-effective override', async () => {
  const sent = [];
  const session = new LiveOptionSession(run({ options: { 'cell-compactness': 0.08 },
    last: { live: { 'cell-compactness': 0.04 } }, live_values: { 'cell-compactness': 0.06 } }),
    async (id, values) => sent.push({ id, values }));
  await session.apply(compactness, 0.02, true);
  assert.equal(session.value(compactness), 0.02);
  assert.equal(session.history.get(compactness.flag).value, 0.06);
  await session.undo(compactness);
  assert.deepEqual(sent, [
    { id: 'a', values: { 'cell-compactness': 0.02 } },
    { id: 'a', values: { 'cell-compactness': 0.06 } },
  ]);
  assert.equal(session.value(compactness), 0.06);
  assert.equal(session.history.size, 0);
});

test('snapshots cannot replace an in-flight trial; duplicate changes wait', async () => {
  let release;
  const session = new LiveOptionSession(run(), () => new Promise(resolve => { release = resolve; }));
  const pending = session.apply(compactness, 0.02, true);
  session.update(run({ last: { live: { 'cell-compactness': 0.04 } } }));
  assert.equal(session.value(compactness), 0.02);
  await assert.rejects(session.apply(compactness, 0.08, true), /pending/);
  release(); await pending;
  assert.equal(session.value(compactness), 0.02);
  assert.equal(session.history.get(compactness.flag).value, 0.04);
  assert.equal(session.busy.size, 0);
});

test('failed Try leaves the old value; failed Undo remains retryable', async () => {
  let fail = true;
  const session = new LiveOptionSession(run(), async () => { if (fail) throw new Error('offline'); });
  await assert.rejects(session.apply(compactness, 0.02, true), /offline/);
  assert.equal(session.value(compactness), 0.04);
  assert.equal(session.history.size, 0);
  assert.equal(session.busy.size, 0);
  fail = false; await session.apply(compactness, 0.02, true);
  fail = true; await assert.rejects(session.undo(compactness), /offline/);
  assert.equal(session.value(compactness), 0.02);
  assert.equal(session.history.get(compactness.flag).value, 0.04);
  fail = false; await session.undo(compactness);
  assert.equal(session.value(compactness), 0.04);
});

test('a pending manual edit is the previous value and later manual edits clear Undo', async () => {
  const sent = [];
  const session = new LiveOptionSession(run(), async (id, values) => sent.push(values));
  session.stage(compactness, 0.03);
  await session.apply(compactness, 0.02, true);
  await session.undo(compactness);
  assert.deepEqual(sent, [{ 'cell-compactness': 0.02 }, { 'cell-compactness': 0.03 }]);
  await session.apply(compactness, 0.08, true);
  session.stage(compactness, 0.05);
  assert.equal(session.history.size, 0);
});

test('Undo restores false booleans and a numeric automatic temperature without claiming to reset its mode', async () => {
  const session = new LiveOptionSession(run({ last: { live: { temperature: 1 } } }), async () => {});
  await session.apply(diffusion, true, true);
  assert.equal((await session.undo(diffusion)).value, false);
  assert.equal(session.value(diffusion), false);
  await session.apply(temperature, 0.1, true);
  const previous = await session.undo(temperature);
  assert.deepEqual(previous, { value: 1, automatic: true });
  assert.equal(session.value(temperature), 1);
});

test('sessions allow paused runs, reject inactive runs and never move history to another run', async () => {
  const sent = [];
  const session = new LiveOptionSession(run({ state: 'paused' }), async (id, values) => sent.push({ id, values }));
  await session.apply(compactness, 0.02, true);
  session.update(run({ state: 'done' }));
  await assert.rejects(session.undo(compactness), /running or paused/);
  assert.throws(() => session.update({ id: 'b', state: 'running' }), /different run/);
  const next = new LiveOptionSession({ id: 'b', state: 'running' }, async () => {});
  assert.equal(next.history.size, 0);
  assert.deepEqual(sent, [{ id: 'a', values: { 'cell-compactness': 0.02 } }]);
  await assert.rejects(next.apply({ ...compactness, live: false }, 0.02), /cannot be applied live/);
});

test('knob values from proseg are shown rounded, not as raw 32-bit floats', async () => {
  const { displayValue } = await import('../src/option-guide.mjs');
  assert.equal(displayValue(0.03999999910593033), '0.04');
  assert.equal(displayValue(0.20000000298023224), '0.2');
  assert.equal(displayValue(4000), '4000');
  assert.equal(displayValue(null), 'automatic');
  assert.equal(displayValue(true), 'on');
});
