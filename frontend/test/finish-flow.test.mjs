import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
function setup({ running = true, accepted = true } = {}) {
  const actions = [], confirmations = [], notices = [], tracked = new Set();
  let acknowledge;
  const ack = new Promise(resolve => { acknowledge = resolve; });
  const state = { run: { id: 'a', state: running ? 'running' : 'paused' }, frames: [], current: -1 };
  const context = vm.createContext({ state, generation: 0, performance, setTimeout, clearTimeout, fmt: new Intl.NumberFormat('en-GB'),
    renderRun() {}, stopReplay() {}, setLive() {}, notice: text => notices.push(text),
    zarrSave: { trackFinish: id => tracked.add(id), cancelFinish: id => tracked.delete(id), update() {} },
    requestAnimationFrame: fn => setImmediate(fn),
    confirm: text => { confirmations.push(text); assert.equal(state.current, 2); return accepted; },
    api: {
      control: async (id, action) => { actions.push(action); return { ...state.run, state: action === 'pause' ? 'paused' : action }; },
      run: () => ack,
      frames: async () => ({ from: 0, frames: [{ seq: 0, iter: 2000 }, { seq: 1, iter: 2005 }, { seq: 2, iter: 2010 }] }),
    },
    onMessage: message => { if (message.ev === 'run') state.run = message.run; else state.frames = message.frames; },
    show: async seq => { state.current = seq; return true; },
  });
  vm.runInContext(source.slice(source.indexOf('function heldAtEnd('), source.indexOf('function stageNote(')), context);
  vm.runInContext(source.slice(source.indexOf('let finishingPrompt = false;'), source.indexOf('function setupControls()')), context);
  return { state, actions, confirmations, notices, tracked, context, acknowledge: () => acknowledge({ id: 'a', state: 'paused', ack: { state: 'paused', seq: 2, iter: 2010 } }), start: () => vm.runInContext('finishAndSave()', context) };
}
test('finish waits for proseg acknowledgement and displays its seq before confirming iter', async () => {
  const h = setup(), done = h.start();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(h.actions, ['pause']);
  assert.equal(h.confirmations.length, 0);
  h.acknowledge();
  await done;
  assert.deepEqual(h.actions, ['pause', 'finish']);
  assert.equal(h.confirmations[0], "Stop sampling now and save proseg's results from iteration 2010?");
  assert.ok(h.tracked.has('a'));
});
test('cancel resumes only a previously running run', async () => {
  for (const running of [true, false]) {
    const h = setup({ running, accepted: false }), done = h.start();
    h.acknowledge();
    await done;
    assert.deepEqual(h.actions, running ? ['pause', 'resume'] : []);
    assert.equal(h.tracked.size, 0);
  }
});
test('held at the end of proseg\'s standard run, Finish & save asks nothing: nothing samples there and finishing is proseg\'s own end', async () => {
  // The pause met proseg on the schedule's last iteration (or the run was held there already): it holds there and the
  // engine says so (schedule_done). Its note says what Finish & save keeps; a question about "stopping sampling" would
  // only make the user wonder what still runs.
  const held = { id: 'a', state: 'paused', schedule_done: { iter: 2010, seq: 2 }, ack: { state: 'paused', seq: 2, iter: 2010 } };
  const h = setup({ accepted: false });
  h.context.api.run = async () => held;
  await h.start();
  assert.equal(h.confirmations.length, 0);
  assert.deepEqual(h.actions, ['pause', 'finish']); assert.ok(h.tracked.has('a'));
  assert.deepEqual(h.notices, []);
  // Sampled on past the end (continued): the question says how far past proseg's standard run, and declining carries on.
  const later = setup({ accepted: false });
  later.context.api.run = async () => ({ ...held, continued: true, schedule_done: { iter: 1988, seq: 0 } });
  await later.start();
  assert.equal(later.confirmations[0], "Stop sampling now and save proseg's results from iteration 2010 (22 iterations past proseg's standard run)?");
  assert.deepEqual(later.actions, ['pause', 'resume']);
});
test('failed acknowledged-frame display never sends finish', async () => {
  const h = setup();
  h.context.show = async () => false;
  const done = h.start(); h.acknowledge(); await done;
  assert.deepEqual(h.actions, ['pause']);
  assert.equal(h.confirmations.length, 0);
  assert.match(h.notices[0], /Could not display/);
});
