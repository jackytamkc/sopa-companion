import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { runButtons } from '../src/run-buttons.mjs';

// Which of the run's buttons a press makes work now (run-buttons.mjs), exactly as the run page draws them (main.js
// renderRun): the words that name Start, Resume or Step (the stage's note at frame 0, Go deeper's first iteration, the
// knob guide) name one only while it works.
const frames = n => Array.from({ length: n }, (_, seq) => ({ seq }));
const none = { start: false, resume: false, step: false, pause: false };

test('the run buttons in every state of a run: on its prior, paused later, pausing, resuming, running, ended, the finish prompt', () => {
  const cases = [
    // On its prior (paused with proseg's ack, frame 0 its only frame): Start and Step.
    ['prior', { state: 'paused', ack: { state: 'paused' } }, 1, false, { start: true, resume: false, step: true, pause: false }],
    // ... with the finish prompt open (Stop & save asked): nothing until it is answered.
    ['prior, finish prompt', { state: 'paused', ack: { state: 'paused' } }, 1, true, none],
    // Paused later: Resume and Step.
    ['paused later', { state: 'paused', ack: { state: 'paused' } }, 5, false, { start: false, resume: true, step: true, pause: false }],
    ['paused later, finish prompt', { state: 'paused', ack: { state: 'paused' } }, 5, true, none],
    // Pausing (proseg has not acknowledged yet) and resuming (its ack still paused): the run button waits; no Step.
    ['pausing', { state: 'paused', ack: { state: 'running' } }, 5, false, none],
    ['pausing on its prior', { state: 'paused', ack: { state: 'running' } }, 1, false, none],
    ['resuming', { state: 'running', ack: { state: 'paused' } }, 5, false, none],
    // Running: Pause alone.
    ['running', { state: 'running', ack: { state: 'running' } }, 5, false, { start: false, resume: false, step: false, pause: true }],
    ['running, finish prompt', { state: 'running', ack: { state: 'running' } }, 5, true, none],
    // Ended, saving its results, preparing: none.
    ['done', { state: 'done' }, 5, false, none], ['stopped', { state: 'stopped' }, 5, false, none], ['failed', { state: 'failed' }, 5, false, none],
    ['finishing', { state: 'finishing', ack: { state: 'paused' } }, 5, false, none], ['preparing', { state: 'preparing' }, 0, false, none],
  ];
  for (const [name, run, n, prompt, want] of cases) assert.deepEqual(runButtons(run, frames(n), { prompt }), want, name);
  // No run (the picker) and no frames list: nothing works.
  assert.deepEqual(runButtons(null, []), none); assert.deepEqual(runButtons(undefined, undefined), none);
  assert.deepEqual(runButtons({ state: 'paused', ack: { state: 'paused' } }, undefined), { start: false, resume: true, step: true, pause: false },
    'a paused run with no frames list held: Resume');
  // The run button is exactly one of Start, Resume and Pause whenever it works.
  for (const [name, run, n, prompt] of cases) {
    const b = runButtons(run, frames(n), { prompt });
    assert.ok([b.start, b.resume, b.pause].filter(Boolean).length <= 1, name);
  }
});

test('renderRun draws the buttons from runButtons, its label by the same rule (Start on the prior, Resume later)', () => {
  const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
  const render = main.slice(main.indexOf('function renderRun() {'), main.indexOf('function renderStainRun('));
  assert.match(render, /const buttons = runButtons\(run, state\.frames, \{ prompt: finishingPrompt \}\);/);
  assert.match(render, /setProp\(toggle, 'disabled', !\(buttons\.start \|\| buttons\.resume \|\| buttons\.pause\)\);/);
  assert.match(render, /setProp\(\$\('run-step'\), 'disabled', !buttons\.step\);/);
  for (const rule of ["const paused = run.state === 'paused' && run.ack?.state === 'paused';", 'const onPrior = paused && state.frames.length === 1',
    "paused ? (onPrior ? 'Start' : 'Resume')"]) assert.ok(render.includes(rule), rule);
  // Go deeper's context and the knob guide's read the same buttons.
  const context = main.slice(main.indexOf('function deeperContext() {'), main.indexOf('function deeperTestCell()'));
  assert.match(context, /buttons: runButtons\(run, state\.frames, \{ prompt: finishingPrompt \}\),/);
  assert.match(main, /buttons: \(\) => runButtons\(state\.run, state\.frames, \{ prompt: finishingPrompt \}\),/);
});
