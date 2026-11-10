import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { PHASES, scheduleEnd } from '../src/timeline.js';
import { runButtons } from '../src/run-buttons.mjs';
import { domFixture } from './dom-fixture.mjs';

// main.js renderRun while proseg reads its inputs: the engine's run.phase 'loading' and run.eta_s (seconds to the first
// frame) become an honest "Loading proseg… ~N s" that counts down, instead of "Pausing at the next iteration boundary...".
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const slice = source.slice(source.indexOf('function phaseLabel('), source.indexOf('function showStain('));

function harness() {
  const dom = domFixture(); dom.document.body.innerHTML = html;
  Object.defineProperty(dom.Element.prototype, 'firstElementChild', { get() { return this.children[0]; }, configurable: true });
  const clock = { now: 0, timers: [] };
  const state = { run: null, frames: [], current: -1, runs: [], live: true };
  const context = vm.createContext({
    $: dom.$, document: dom.document, state, PHASES, scheduleEnd, runButtons, ACTIVE: ['preparing', 'running', 'paused', 'finishing'],
    fmt: new Intl.NumberFormat('en-GB'), fmt1: new Intl.NumberFormat('en-GB', { maximumFractionDigits: 1 }),
    shown: { knobs: '', results: '' }, finishingPrompt: false, knobs: { key: '', render() {} }, options: { schema: [] },
    layerPanel: { autoPrior() {} }, stage: { options: { cellColorBy: 'identity' } }, timeline: { total: 0 }, renderResults() {},
    legendCorner: null,   // no test corner open
    later: { run: () => vm.runInContext('renderRun()', context) },
    performance: { now: () => clock.now }, JSON, Math, Number, String, Object, Array,
    setTimeout: (fn, ms) => { clock.timers.push({ fn, at: clock.now + ms }); return clock.timers.length; },
    clearTimeout: id => { if (clock.timers[id - 1]) clock.timers[id - 1].cleared = true; },
  });
  vm.runInContext(slice, context);
  const render = () => vm.runInContext('renderRun()', context);
  /** Advance the clock by `ms`, firing the timers that fall due (each re-renders through later.run). */
  const advance = ms => {
    const until = clock.now + ms;
    for (;;) {
      const due = clock.timers.filter(t => !t.cleared && !t.fired && t.at <= until).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      clock.now = Math.max(clock.now, due.at); due.fired = true; due.fn();
    }
    clock.now = until;
  };
  const overlay = () => dom.$('hud-state'), toggle = () => dom.$('run-toggle');
  const fact = name => { const f = dom.$('run-facts').children; for (let i = 0; i < f.length; i += 2) if (f[i].textContent === name) return f[i + 1].textContent; return null; };
  return { dom, state, clock, render, advance, overlay, toggle, fact, $: dom.$ };
}
const RUN = { id: 'r1', sample_name: 'sample-1', box: [0, 0, 100, 100], schedule: 'open', speed: 'max', ack: null };

test('while proseg loads, a run started paused says so and counts the engine\'s estimate down', () => {
  const h = harness();
  h.state.run = { ...RUN, state: 'paused', phase: 'loading', eta_s: 40 };
  h.render();
  assert.equal(h.overlay().hidden, false);
  assert.equal(h.overlay().textContent, 'Loading proseg… ~40 s');
  assert.equal(h.overlay().children[0].className, 'spinner');
  assert.equal(h.toggle().textContent, 'Loading…'); assert.equal(h.toggle().disabled, true, 'nothing to resume yet');
  assert.equal(h.fact('State'), 'Loading proseg…');
  assert.match(h.$('status').textContent, /^Loading proseg… · Waiting for the first iteration/);
  const spinner = h.overlay().children[0];
  h.advance(1001);
  assert.equal(h.overlay().textContent, 'Loading proseg… ~39 s', 'one second later, with no run update');
  assert.equal(h.clock.timers.filter(t => t.fired).length, 1, 'one render per second shown');
  assert.equal(h.overlay().children[0], spinner, 'only the words change: the spinner element is kept, no DOM rebuilt each second');
  h.advance(4000);
  assert.equal(h.overlay().textContent, 'Loading proseg… ~35 s');
  assert.equal(h.clock.timers.filter(t => t.fired).length, 5);
  // A run update repeating the same figure (the runs poll) does not restart the count; a smaller one is taken.
  h.state.run = { ...h.state.run, eta_s: 40 }; h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg… ~35 s');
  h.state.run = { ...h.state.run, eta_s: 20 }; h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg… ~20 s');
  h.advance(25000);
  assert.equal(h.overlay().textContent, 'Loading proseg…', 'past the estimate: no number');
  assert.equal(h.clock.timers.filter(t => !t.cleared && !t.fired).length, 0, 'and no more ticking');
});

test('the first frame ends the loading phase; without a phase from the engine nothing changes', () => {
  const h = harness();
  h.state.run = { ...RUN, state: 'running', phase: 'loading', eta_s: null };
  h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg…', 'no estimate: no number');
  assert.equal(h.toggle().textContent, 'Pause'); assert.equal(h.toggle().disabled, false, 'a running start can still be paused ahead');
  h.state.run = { ...h.state.run, phase: 'sampling', ack: { state: 'running' } };
  h.state.frames = [{ iter: 0, phase: 'init', voxel: 2, temperature: 1, changed: 0, t: 1 }]; h.state.current = 0;
  h.render();
  assert.equal(h.overlay().hidden, true);
  assert.equal(h.clock.timers.filter(t => !t.cleared && !t.fired).length, 0);
  // An engine that sends no phase: the old wording.
  const old = harness();
  old.state.run = { ...RUN, state: 'paused' };
  old.render();
  assert.equal(old.overlay().textContent, 'Pausing at the next iteration boundary...');
  assert.equal(old.toggle().textContent, 'Pausing...');
  assert.match(old.$('status').textContent, /^Pausing\.\.\. · /);
});

test('each run counts down its own estimate', () => {
  const h = harness();
  h.state.run = { ...RUN, state: 'paused', phase: 'loading', eta_s: 90 };
  h.render(); h.advance(30001);
  assert.equal(h.overlay().textContent, 'Loading proseg… ~60 s');
  h.state.run = { ...RUN, id: 'r2', state: 'paused', phase: 'loading', eta_s: 40 };
  h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg… ~40 s');
});

test('a new start of proseg counts its own estimate; what the engine says meanwhile shows with it', () => {
  const h = harness();
  h.state.run = { ...RUN, state: 'paused', phase: 'loading', eta_s: 6, pid: 100, started: '2026-10-17T10:00:00+00:00' };
  h.render(); h.advance(4001);
  assert.equal(h.overlay().textContent, 'Loading proseg… ~2 s');
  // Feature selection failed on a small crop: the engine says so, then starts proseg again (a new pid, eta 6 again).
  h.state.run = { ...h.state.run, message: 'Proseg could not start on this region with factorisation; retrying without it…' };
  h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg… ~2 s · Proseg could not start on this region with factorisation; retrying without it…');
  h.state.run = { ...h.state.run, message: '', pid: 101, eta_s: 6 };
  h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg… ~6 s', 'not the first start\'s deadline (it ran out ~2 s later)');
  h.advance(3001);
  assert.equal(h.overlay().textContent, 'Loading proseg… ~3 s');
  // Finish & save before the first frame: still loading, and saying what comes after.
  h.state.run = { ...h.state.run, state: 'finishing', message: 'Saving results from iteration 0…', eta_s: 3 };
  h.render();
  assert.equal(h.overlay().textContent, 'Loading proseg… ~3 s · Saving results from iteration 0…');
});

// proseg's own schedule (2 burn-in, 3 sampling, 2 hill-climb: the standard run ends at iteration 7) as the engine
// describes it, and a frame of each iteration.
const SCHEDULE_INFO = { schedule: { burnin: 2, samples: 3, recorded: 1, hillclimb: 2 },
  plan: [{ phase: 'init', frames: 1 }, { phase: 'burnin', frames: 2 }, { phase: 'upscale', frames: 1 }, { phase: 'sampling', frames: 2 },
    { phase: 'recorded', frames: 1 }, { phase: 'hillclimb', frames: 2 }] };
const frameAt = (iter, phase) => ({ iter, phase, voxel: 1, temperature: 1, changed: 0, t: iter });

test('every run waits on its prior with a calm note, never a question; the note goes once it has sampled', () => {
  const h = harness();
  h.state.run = { ...RUN, state: 'paused', phase: 'paused', sample_kind: 'zarr', info: SCHEDULE_INFO, ack: { state: 'paused', iter: 0, seq: 0 } };
  h.state.frames = [frameAt(0, 'init')]; h.state.current = 0;
  h.render();
  assert.equal(h.$('prior-hint').hidden, false);
  assert.equal(h.$('prior-hint').textContent, "This is proseg's starting point, built from the chosen prior. Press Start to begin sampling.");
  assert.equal(h.toggle().textContent, 'Start', 'nothing has run yet: the button starts it');
  assert.equal(h.toggle().title, 'Start sampling (Space)');
  assert.equal(h.$('hud-sub').textContent, '1 µm voxels · temperature 1.00 · 0 voxels changed', 'temperature in words, not "T"');
  assert.equal(h.fact('Schedule'), null, 'one schedule: nothing to name');
  assert.equal(h.$('hud-total').textContent, '/ 7'); assert.equal(h.$('hud-total').hidden, false);
  // Step: one iteration has run, so "start sampling" is no longer true, also when the user looks back at frame 0.
  h.state.frames = [frameAt(0, 'init'), frameAt(1, 'burnin')];
  h.state.run = { ...h.state.run, ack: { state: 'paused', iter: 1, seq: 1 } };
  h.render();
  assert.equal(h.$('prior-hint').hidden, true);
  assert.equal(h.toggle().textContent, 'Resume', 'it has sampled: Resume');
  // Resume pressed on the prior: the note goes with the click, before proseg's acknowledgement.
  h.state.frames = [frameAt(0, 'init')];
  h.state.run = { ...h.state.run, state: 'running', ack: { state: 'paused', iter: 0, seq: 0 } };
  h.render();
  assert.equal(h.$('prior-hint').hidden, true);
});

test('the end of proseg\'s standard run pauses the run with a note; Resume samples on, open-ended', () => {
  const h = harness();
  const frames = [frameAt(0, 'init'), frameAt(1, 'burnin'), frameAt(2, 'burnin'), frameAt(2, 'upscale'), frameAt(3, 'sampling'),
    frameAt(4, 'sampling'), frameAt(5, 'recorded'), frameAt(6, 'hillclimb'), frameAt(7, 'hillclimb')];
  h.state.frames = frames; h.state.current = 8;
  h.state.run = { ...RUN, state: 'running', phase: 'sampling', info: SCHEDULE_INFO, ack: { state: 'running' } };
  h.render();
  assert.equal(h.$('prior-hint').hidden, true);
  assert.equal(h.$('hud-total').textContent, '/ 7'); assert.equal(h.$('hud-total').hidden, false);
  // proseg holds at the end of its schedule: the engine pauses the run there (schedule_done).
  const classes = node => { const set = new Set(node.className?.split(' ').filter(Boolean)); node.classList = { add: c => set.add(c),
    remove: c => set.delete(c), toggle: (c, on) => { if (on ?? !set.has(c)) set.add(c); else set.delete(c); } }; return set; };
  const toggleClass = classes(h.toggle()), stopClass = classes(h.$('run-stop'));
  h.state.run = { ...h.state.run, state: 'paused', phase: 'paused', schedule_done: { iter: 7, seq: 8 }, ack: { state: 'paused', iter: 7, seq: 8 } };
  h.render();
  assert.equal(h.$('prior-hint').hidden, false);
  // The note says what each button does: Resume gives up proseg's own result for good.
  assert.equal(h.$('prior-hint').textContent, "Proseg's standard run is complete. Finish & save keeps this result. Resume samples on at full temperature, and after that this exact result can't be saved.");
  assert.match(h.$('status').textContent, /^Standard run complete · Iteration 7 · /);
  assert.equal(h.toggle().textContent, 'Resume'); assert.equal(h.toggle().disabled, false);
  assert.ok(stopClass.has('primary') && !toggleClass.has('primary'), 'Finish & save is the main button there, Resume a plain one');
  assert.ok(!stopClass.has('danger'), 'danger is for Abort only');
  assert.equal(h.$('run-stop').disabled, false, 'Finish & save saves proseg\'s own result from here');
  assert.equal(h.$('run-step').disabled, false);
  assert.equal(h.overlay().hidden, true, 'no dialog, nothing over the tissue');
  // Resume: sampling on, past the schedule's end (no total any more); a later pause is an ordinary pause.
  h.state.frames = [...frames, frameAt(8, 'continued')]; h.state.current = 9;
  h.state.run = { ...h.state.run, state: 'running', phase: 'sampling', continued: true, ack: { state: 'running', iter: 8, seq: 9 } };
  h.render();
  assert.equal(h.$('prior-hint').hidden, true); assert.equal(h.$('hud-total').hidden, true);
  assert.ok(toggleClass.has('primary') && !stopClass.has('primary'), 'sampling on: the run button is the main one again');
  h.state.run = { ...h.state.run, state: 'paused', phase: 'paused', ack: { state: 'paused', iter: 8, seq: 9 } };
  h.render();
  assert.equal(h.$('prior-hint').hidden, true); assert.match(h.$('status').textContent, /^Paused · /);
  // Looking back at a frame of the standard run shows its place in it again, and the status line the newest iteration.
  h.state.current = 5; h.render();
  assert.equal(h.$('hud-total').hidden, false);
  assert.match(h.$('status').textContent, /^Paused · Iteration 4 \(newest 8\) · /);
});

test('a run made open-ended before 2026-10-18 shows no total; its timeline phase has no end', () => {
  const h = harness();
  const info = { schedule: { burnin: 100, samples: 60, recorded: 0, hillclimb: 0 }, indefinite: true,
    plan: [{ phase: 'init', frames: 1 }, { phase: 'burnin', frames: 100 }, { phase: 'sampling', frames: 60 }, { phase: 'recorded', frames: null }] };
  h.state.frames = [frameAt(0, 'init'), frameAt(1, 'burnin')]; h.state.current = 1;
  h.state.run = { ...RUN, schedule: 'open', state: 'done', phase: 'done', info, outputs: [] };
  h.render();
  assert.equal(h.$('hud-total').hidden, true);
  assert.equal(scheduleEnd(info), null); assert.equal(scheduleEnd(SCHEDULE_INFO), 7); assert.equal(scheduleEnd(null), null);
});
