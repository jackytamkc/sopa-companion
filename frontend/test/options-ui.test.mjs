import test from 'node:test';
import assert from 'node:assert/strict';
import { Options, LiveKnobs, KnobGuide, optionHelp } from '../src/options.js';
import { EXPLAIN_TEXT } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

// A small in-memory DOM for interaction tests; no WebGL, server, or browser process needed.
class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.attributes = new Map();
    this.listeners = new Map(); this.value = ''; this.hidden = false; this.disabled = false;
  }
  set textContent(value) { this.text = value; this.children = []; }
  get textContent() { return (this.text || '') + this.children.map(child => child.textContent).join(''); }
  get firstChild() { return this.children[0]; }
  append(...children) { for (const child of children) { child.parentElement = this; this.children.push(child); } }
  replaceChildren(...children) { this.text = ''; this.children = []; this.append(...children); }
  setAttribute(key, value) { this.attributes.set(key, value); }
  getAttribute(key) { return this.attributes.get(key); }
  removeAttribute(key) { this.attributes.delete(key); }
  addEventListener(event, listener) { this.listeners.set(event, listener); }
  emit(event) { return this.listeners.get(event)?.({ target: this }); }
  checkValidity() { return true; }
  reportValidity() { return true; }
  focus() { document.activeElement = this; }
  closest(tag) { return this.tagName === tag.toUpperCase() ? this : this.parentElement?.closest(tag); }
  querySelectorAll(selector) {
    const tags = selector.toUpperCase().split(',').map(tag => tag.trim());
    return this.children.flatMap(child => [...(tags.includes(child.tagName) ? [child] : []), ...child.querySelectorAll(selector)]);
  }
  click() { if (!this.disabled) return this.onclick?.({ preventDefault() {}, stopPropagation() {} }); }
}
function dom() {
  const storage = new Map();
  globalThis.document = { createElement: tag => new Element(tag), activeElement: null };
  globalThis.localStorage = { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, value) };
  const panel = new Element('details');
  const container = new Element('div'), guideHost = new Element('div'), toggle = new Element('button');
  guideHost.hidden = true; guideHost.id = 'guide';
  panel.append(toggle, guideHost, container);
  return { panel, container, guideHost, toggle };
}
const schema = [
  { flag: 'cell-compactness', label: 'Cell compactness', type: 'float', live: true, default: 0.04,
    min: 0.001, max: 10, what: 'Smaller values favour compact cells.', up: 'More elongated cells.',
    down: 'Rounder cells.', use_when: 'Cells have spikes.', typical: 'Try 0.02 to 0.08; default 0.04.' },
  { flag: 'prior-seg-reassignment-prob', label: 'Cell-body reassignment', type: 'float', live: false,
    default: 0.5, min: 0.000001, max: 0.999999, what: 'Sets how closely cells follow Xenium.' },
  { flag: 'no-diffusion', label: 'No diffusion', type: 'bool', live: true, default: false,
    what: 'Stops further transcript repositioning.', up: 'On: no more repositioning.', down: 'Off: nearby counts are allowed.' },
  { flag: 'temperature', label: 'Temperature', type: 'float', live: true, default: null, min: 0.01, max: 1 },
];
const guideEntries = [{ symptom: 'Spiky cells', explanation: 'Try stronger smoothing.', actions: [
  { flag: 'cell-compactness', direction: 'lower', value: 0.02, live: true, why: 'Rounder cells.' },
  { flag: 'prior-seg-reassignment-prob', direction: 'lower', value: 0.2, live: false, why: 'Follow Xenium.' },
] }, { symptom: 'Settle', explanation: 'Step, inspect, then save.', actions: [
  { flag: 'temperature', direction: 'lower', value: 0.1, live: true, why: 'Less boundary movement.' },
] }];

/** hold.promise, when set: every send waits for it (a change on its way); buttons(): the run buttons that work (the
 *  page's runButtons), none named without it. */
function live(hold = {}, buttons = undefined) {
  const nodes = dom(), sent = [], notices = [];
  let run = { id: 'a', state: 'paused', last: { live: { 'cell-compactness': 0.04, temperature: 1 } } };
  const knobs = new LiveKnobs(nodes.container, async (id, values) => {
    sent.push({ id, values });
    if (hold.promise) await hold.promise;
    run = { ...run, live_values: { ...run.live_values, ...values } };
    knobs.render(run, schema);
  }, message => notices.push(message));
  const guide = new KnobGuide(nodes.guideHost, nodes.toggle, {
    mode: 'live', value: option => knobs.value(option), blocked: option => knobs.blocked(option),
    apply: (action, option) => knobs.apply(option, action.value, true),
    undo: option => knobs.apply(option, undefined, false, true),
    canUndo: option => knobs.session?.history.has(option.flag), ...(buttons ? { buttons } : {}),
  });
  knobs.guide = guide;
  guide.load(guideEntries, schema); knobs.render(run, schema);
  guide.select.value = '0'; guide.select.onchange();
  return { ...nodes, knobs, guide, sent, notices, run };
}
/** The note of a knob guide card (the box under its buttons that says why a press could not go ahead). */
const noteOf = control => control.apply.parentElement.parentElement.children.find(node => node.className === 'missing-note');

test('help cards toggle accessibly, use plain text, and distinguish boolean directions', () => {
  dom();
  const { toggle, card } = optionHelp({ ...schema[0], what: '<b>literal text</b>' });
  assert.equal(card.hidden, true);
  assert.equal(toggle.getAttribute('aria-controls'), card.id);
  toggle.click();
  assert.equal(card.hidden, false);
  assert.equal(toggle.getAttribute('aria-expanded'), 'true');
  assert.match(card.textContent, /What it changes<b>literal text<\/b>/);
  assert.match(card.textContent, /Raise it →More elongated cells/);
  toggle.click(); assert.equal(card.hidden, true);
  assert.match(optionHelp(schema[2]).card.textContent, /Turn it onOn: no more/);
});

test('live rows preserve open cards across snapshots and guide Try/Undo uses the shared sender', async () => {
  const h = live();
  const row = h.container.children[0];
  row.children[1].click();
  assert.equal(row.children[2].hidden, false);
  assert.equal(h.knobs.rows.get(schema[0].flag).input.title, schema[0].what);
  assert.equal(h.guide.controls.length, 1); // Startup-only action has no live Try button.
  assert.match(h.guide.detail.textContent, /Applies to a new run/);
  h.toggle.click(); assert.equal(h.guideHost.hidden, false); assert.equal(h.panel.open, true);
  await h.guide.controls[0].apply.click();
  assert.deepEqual(h.sent, [{ id: 'a', values: { 'cell-compactness': 0.02 } }]);
  assert.match(h.guide.controls[0].status.textContent, /applies from the next iteration/);
  assert.match(h.knobs.rows.get(schema[0].flag).status.textContent, /effective: 0.04/);
  assert.equal(row.children[2].hidden, false);
  await h.guide.controls[0].undo.click();
  assert.deepEqual(h.sent[1], { id: 'a', values: { 'cell-compactness': 0.04 } });
  await h.guide.controls[0].undo.click();                        // nothing left to undo: said, not sent
  assert.equal(h.sent.length, 2); assert.equal(h.guide.controls[0].undo.disabled, false);
  assert.match(noteOf(h.guide.controls[0]).textContent, /Cell compactness: No suggestion to undo yet/);
  assert.deepEqual(h.notices, []);
});


test("the guide's timing and Try's message name Start, Resume or Step only while both of the pair work (the run's buttons, E1)", async () => {
  // The page's runButtons: on the prior (before the first iteration) Start and Step work; paused later, Resume and Step;
  // running, neither (the knob applies at the next iteration by itself); while pausing or with the finish prompt open,
  // none: then no button is named.
  const now = { start: true, resume: false, step: true, pause: false }, h = live({}, () => now);
  const timing = () => h.guide.controls[0].timing.textContent, card = () => h.guide.detail.children.find(c => c.className === 'guide-action');
  assert.equal(timing(), 'Applies from the first iteration: press Start or Step.');
  assert.ok(card().children.some(c => c.className === 'guide-timing' && c === h.guide.controls[0].timing), 'the card\'s own timing line');
  await h.guide.controls[0].apply.click();
  assert.equal(h.guide.controls[0].status.textContent, 'Set to 0.02; applies from the next iteration. Press Start or Step.');
  // Started (running): the words follow the buttons as the page renders them again (renderRun calls update()).
  Object.assign(now, { start: false, step: false, pause: true }); h.guide.update();
  assert.equal(timing(), 'Applies from the next iteration.');
  assert.equal(h.guide.controls[0].status.textContent, 'Set to 0.02; applies from the next iteration.', 'no button named while none moves it on');
  Object.assign(now, { resume: true, step: true, pause: false }); h.guide.update();
  assert.equal(timing(), 'Applies from the next iteration: press Resume or Step.');
  assert.equal(h.guide.controls[0].status.textContent, 'Set to 0.02; applies from the next iteration. Press Resume or Step.');
  // Resume alone (Step waiting for the finish prompt), or Step alone: never half a pair named.
  Object.assign(now, { resume: true, step: false }); h.guide.update(); assert.equal(timing(), 'Applies from the next iteration.');
  Object.assign(now, { start: true, resume: false, step: false }); h.guide.update(); assert.equal(timing(), 'Applies from the next iteration.');
  // A startup-only knob in the live guide applies to a new run, whatever the buttons; a guide with no run says no button.
  assert.match(h.guide.detail.textContent, /Applies to a new run\./);
  const plain = live();
  assert.equal(plain.guide.controls[0].timing.textContent, 'Applies from the next iteration.');
  await plain.guide.controls[0].apply.click();
  assert.equal(plain.guide.controls[0].status.textContent, 'Set to 0.02; applies from the next iteration.');
  for (const text of [timing(), h.guide.controls[0].status.textContent]) assert.deepEqual(bannedWords(text), []);
});

test('trying a suggestion cancels a pending manual debounce and Undo restores that edit', async () => {
  const h = live();
  const input = h.knobs.rows.get(schema[0].flag).input;
  input.value = '0.03'; input.emit('input');
  await h.guide.controls[0].apply.click();
  await new Promise(resolve => setTimeout(resolve, 400));
  assert.equal(h.sent.length, 1);
  await h.guide.controls[0].undo.click();
  assert.deepEqual(h.sent[1].values, { 'cell-compactness': 0.03 });
});

test('guide changes do not leak into another run, and a run that ended says so when Try or Undo is pressed', async () => {
  const h = live();
  await h.guide.controls[0].apply.click();
  h.knobs.render({ ...h.run, state: 'done' }, schema);
  const control = h.guide.controls[0];
  assert.equal(control.apply.disabled, false); assert.equal(control.undo.disabled, false);
  for (const pressed of [control.apply, control.undo]) {
    await pressed.click();
    assert.match(noteOf(control).textContent, /Cell compactness: Live knobs are available while a run is running or paused/);
  }
  assert.equal(h.sent.length, 1, 'nothing more is sent');
  h.knobs.render({ ...h.run, id: 'b' }, schema);
  assert.equal(h.guide.select.value, '');
  assert.equal(h.guide.messages.size, 0);
  assert.equal(h.knobs.session.history.size, 0);
});

test('picker suggestions prefill startup and live options, persist, and honour Reset', async () => {
  const h = dom();
  const options = new Options(h.container);
  options.load(schema);
  const guide = new KnobGuide(h.guideHost, h.toggle, {
    mode: 'picker', value: option => options.value(option), apply: action => options.setValue(action.flag, action.value),
  });
  options.onChange = () => guide.update();
  guide.load(guideEntries, schema);
  guide.select.value = '0'; guide.select.onchange();
  assert.equal(guide.controls.length, 2);
  await guide.controls[1].apply.click();
  assert.deepEqual(options.getValues(), { 'prior-seg-reassignment-prob': 0.2 });
  assert.equal(Number(options.rows.get(schema[1].flag).input.value), 0.2);   // a real input holds strings
  assert.match(guide.controls[1].status.textContent, /Added to Advanced options/);
  const restored = new Options(new Element('div')); restored.load(schema);
  assert.deepEqual(restored.getValues(), options.getValues());
  guide.select.value = '1'; guide.select.onchange();
  assert.equal(guide.controls[0].apply.disabled, false); // Automatic temperature can be prefilled.
  await guide.controls[0].apply.click();
  assert.equal(options.getValues().temperature, 0.1);
  h.container.children[0].click(); // Reset all.
  assert.deepEqual(options.getValues(), {});
});

test("the mixing check's Try it goes through the same session as Live knobs and the knob guide; its Undo puts back the old value exactly", async () => {
  const h = live();
  assert.equal(h.knobs.option('cell-compactness').flag, 'cell-compactness'); assert.equal(h.knobs.option('prior-seg-reassignment-prob'), null, 'startup only');
  // The suggestion's direction (0.04 -> 0.02, lower) is checked against the value now; the knob's row shows the change.
  const previous = await h.knobs.trySetting('cell-compactness', 0.02, 0.04);
  assert.deepEqual(previous, { value: 0.04, automatic: false });
  assert.deepEqual(h.sent, [{ id: 'a', values: { 'cell-compactness': 0.02 } }]);
  assert.match(h.knobs.rows.get('cell-compactness').status.textContent, /Pending until next iteration \(effective: 0.04\)/);
  assert.equal(h.knobs.session.history.get('cell-compactness').value, 0.04, "the knob guide's Undo knows it too");
  await assert.rejects(h.knobs.trySetting('cell-compactness', 0.02, 0.04), /Already at this value/);
  await assert.rejects(h.knobs.trySetting('prior-seg-reassignment-prob', 0.2), /can't change during a run/);
  // Undo with the trial's own old value: exactly that value, and the session forgets the change.
  await h.knobs.undoSetting('cell-compactness', 0.04);
  assert.deepEqual(h.sent[1], { id: 'a', values: { 'cell-compactness': 0.04 } }); assert.equal(h.knobs.session.history.has('cell-compactness'), false);
  // Without a value: the session's own Undo (the value before its last remembered change).
  await h.knobs.trySetting('cell-compactness', 0.02, 0.04);
  await h.knobs.undoSetting('cell-compactness');
  assert.deepEqual(h.sent.at(-1).values, { 'cell-compactness': 0.04 });
  assert.deepEqual(h.notices, []);
});

test("Try it refuses a value already reached (proseg's 32-bit floats too), one beyond it, a knob mid-send and a run that ended", async () => {
  const h = live();
  h.knobs.render({ ...h.run, last: { live: { 'cell-compactness': 0.019999999552965164, temperature: 1 } } }, schema);
  await assert.rejects(h.knobs.trySetting('cell-compactness', 0.02, 0.04), /Already at this value/);
  h.knobs.render({ ...h.run, last: { live: { 'cell-compactness': 0.01, temperature: 1 } } }, schema);
  await assert.rejects(h.knobs.trySetting('cell-compactness', 0.02, 0.04), /Already beyond this suggestion in that direction/);
  h.knobs.render({ ...h.run, last: { live: { 'cell-compactness': 0.04, temperature: 1 } } }, schema);
  const first = h.knobs.trySetting('cell-compactness', 0.02, 0.04);
  await assert.rejects(h.knobs.trySetting('cell-compactness', 0.02, 0.04), /Already at this value/, 'a double click sends once (the value is on its way)');
  await first; assert.equal(h.sent.length, 1);
  h.knobs.render({ ...h.run, state: 'done' }, schema);
  await assert.rejects(h.knobs.trySetting('no-diffusion', true, false), /running or paused/);
});

test('"New run with this setting" fills the advanced options at once: the run\'s own plus the suggested one, the rest back to defaults', () => {
  const h = dom();
  const options = new Options(h.container); options.load(schema);
  options.setValue('prior-seg-reassignment-prob', 0.2);
  const set = options.setAll({ 'cell-compactness': 0.02, temperature: 5, 'not-an-option': 1, 'no-diffusion': false });
  assert.deepEqual(set, ['cell-compactness'], 'an out-of-range value, an unknown flag and a default are left out');
  assert.deepEqual(options.getValues(), { 'cell-compactness': 0.02 });
  assert.equal(Number(options.rows.get('cell-compactness').input.value), 0.02, 'the row shows it');
  const restored = new Options(new Element('div')); restored.load(schema);
  assert.deepEqual(restored.getValues(), { 'cell-compactness': 0.02 }, 'remembered, as any picker option');
  // The suggested setting at proseg's default (R2: back to the standard 0.04): the new run still names it, until its row
  // is edited or reset (then it is an ordinary default again).
  options.setAll({ 'cell-compactness': 0.04, 'no-diffusion': true }, { explicit: ['cell-compactness'] });
  assert.deepEqual(options.getValues(), { 'cell-compactness': 0.04, 'no-diffusion': true });
  const row = options.rows.get('cell-compactness').input; row.value = '0.04'; row.emit('input');
  assert.deepEqual(options.getValues(), { 'no-diffusion': true }, 'edited by hand: an ordinary default');
  options.setAll({ 'cell-compactness': 0.04 }, { explicit: ['cell-compactness'] }); h.container.children[0].click();
  assert.deepEqual(options.getValues(), {}, 'Reset all');
});

test("the mixing check's Try it drops a Live knobs edit still waiting to be sent: Undo puts back the value in effect, never the draft", async () => {
  const h = live();
  const input = h.knobs.rows.get('cell-compactness').input;
  input.value = '0.06'; input.emit('input');                  // typed, not sent yet (350 ms)
  const previous = await h.knobs.trySetting('cell-compactness', 0.02, 0.04);
  assert.deepEqual(previous, { value: 0.04, automatic: false }, 'the value in effect, not 0.06');
  await new Promise(resolve => setTimeout(resolve, 400));
  assert.deepEqual(h.sent.map(s => s.values), [{ 'cell-compactness': 0.02 }], 'the draft is never sent afterwards');
});

test('"New run with this setting" fills the picker without remembering it; Start (save) or leaving (restore) decides', () => {
  const h = dom();
  const options = new Options(h.container); options.load(schema);
  options.setValue('prior-seg-reassignment-prob', 0.2);
  options.setAll({ 'cell-compactness': 0.02 }, { explicit: ['cell-compactness'], save: false });
  assert.deepEqual(options.getValues(), { 'cell-compactness': 0.02 });
  const other = new Options(new Element('div')); other.load(schema);
  assert.deepEqual(other.getValues(), { 'prior-seg-reassignment-prob': 0.2 }, 'the remembered options are unchanged');
  options.restore();
  assert.deepEqual(options.getValues(), { 'prior-seg-reassignment-prob': 0.2 }, 'left without Start: the remembered ones come back');
  options.setAll({ 'cell-compactness': 0.02 }, { save: false }); options.save();
  const started = new Options(new Element('div')); started.load(schema);
  assert.deepEqual(started.getValues(), { 'cell-compactness': 0.02 }, 'a run started with them: remembered from then on');
  options.restore(); assert.deepEqual(options.getValues(), { 'cell-compactness': 0.02 }, 'nothing left to restore');
});

test('options remembered while the picker offered schedule presets load against proseg\'s own defaults, never as a run proseg refuses', () => {
  const h = dom();
  const lengths = [
    { flag: 'burnin-samples', label: 'Burn-in samples', type: 'int', default: 200, min: 0, advanced: true, group: 'Schedule' },
    { flag: 'samples', label: 'Samples', type: 'int', default: 200, min: 0, advanced: true, group: 'Schedule' },
    { flag: 'recorded-samples', label: 'Recorded samples', type: 'int', default: 100, min: 0, advanced: true, group: 'Schedule' },
  ];
  const withLengths = [...lengths, ...schema], load = (saved, list = withLengths) => {
    localStorage.setItem('sopa-companion:options', JSON.stringify(saved));
    const options = new Options(h.container); options.load(list); return options;
  };
  // "Until I stop" (100 burn-in, 60 samples, 0 recorded) remembered what differed from it: 200 burn-in is proseg's own
  // default now, and 30 samples under proseg's 100 recorded is a run it refuses: both go, the rest stays.
  let options = load({ 'burnin-samples': 200, samples: 30, 'cell-compactness': 0.08 });
  assert.deepEqual(options.getValues(), { 'cell-compactness': 0.08 });
  assert.equal(options.value(lengths[1]), 200); assert.equal(options.value(lengths[2]), 100);
  assert.equal(Number(options.rows.get('samples').input.value), 200, 'the row shows proseg\'s default');
  // A schedule proseg accepts stays the user's choice.
  options = load({ 'burnin-samples': 150, samples: 120, 'recorded-samples': 40 });
  assert.deepEqual(options.getValues(), { 'burnin-samples': 150, samples: 120, 'recorded-samples': 40 });
  // More recorded samples than proseg's 200: back to proseg's two defaults; the burn-in stays.
  options = load({ 'burnin-samples': 150, 'recorded-samples': 300 });
  assert.deepEqual(options.getValues(), { 'burnin-samples': 150 });
  // A schema without the schedule lengths loads as it is.
  options = load({ samples: 30, 'cell-compactness': 0.08 }, schema);
  assert.deepEqual(options.getValues(), { 'cell-compactness': 0.08 });
});

test("Try and Undo are never greyed out: a press that cannot go ahead says why under the buttons, by the knob's name", async () => {
  const hold = {}, h = live(hold), control = h.guide.controls[0], note = noteOf(control);
  await control.undo.click();
  assert.equal(note.hidden, false); assert.match(note.textContent, /Cell compactness: No suggestion to undo yet/);
  assert.equal(h.sent.length, 0);
  // A change on its way: the pressed button is busy and says so; pressing it again says the last step is still running,
  // pressing Undo that the knob is being changed.
  let release; hold.promise = new Promise(resolve => { release = resolve; });
  const trying = control.apply.click();
  assert.equal(note.hidden, true, 'a press that goes ahead clears the note');
  assert.equal(control.apply.textContent, 'Sending…'); assert.equal(control.apply.getAttribute('aria-busy'), 'true');
  assert.equal(control.apply.disabled, false);
  await control.apply.click();
  assert.ok(note.textContent.includes(EXPLAIN_TEXT.busy));
  await control.undo.click();
  assert.match(note.textContent, /Cell compactness: Sending change/);
  assert.equal(h.sent.length, 1);
  release(); await trying;
  assert.equal(control.apply.textContent, 'Try'); assert.equal(control.apply.getAttribute('aria-busy'), undefined);
  assert.equal(note.hidden, true, 'the change went through: the note goes');
  assert.match(control.status.textContent, /^Set to 0.02; applies from the next iteration/);
  // The suggestion is reached: Try says so; the status line keeps only what happened.
  hold.promise = null;
  await control.apply.click();
  assert.match(note.textContent, /Cell compactness: Already at this value/);
  assert.doesNotMatch(control.status.textContent, /Already/);
  assert.equal(h.sent.length, 1);
  for (const words of [note.textContent, control.status.textContent]) assert.deepEqual(bannedWords(words), []);
});

test('in the picker, Use in Advanced options is never greyed out: a value already set or one the option refuses is explained', async () => {
  const h = dom();
  const options = new Options(h.container); options.load(schema);
  const guide = new KnobGuide(h.guideHost, h.toggle, {
    mode: 'picker', value: option => options.value(option), apply: action => options.setValue(action.flag, action.value),
  });
  options.onChange = () => guide.update();
  guide.load([{ symptom: 'Cells ignore the stain', explanation: 'Follow Xenium more closely.', actions: [
    { flag: 'prior-seg-reassignment-prob', direction: 'set', value: 2, live: false, why: 'Beyond what the option takes.' },
    { flag: 'prior-seg-reassignment-prob', direction: 'lower', value: 0.2, live: false, why: 'Follow Xenium.' },
  ] }], schema);
  guide.select.value = '0'; guide.select.onchange();
  const [refused, fine] = guide.controls;
  assert.equal(refused.apply.disabled, false);
  await refused.apply.click();
  assert.match(noteOf(refused).textContent, /Cell-body reassignment: Suggestion unavailable/);
  assert.deepEqual(options.getValues(), {});
  await fine.apply.click();
  assert.deepEqual(options.getValues(), { 'prior-seg-reassignment-prob': 0.2 });
  assert.equal(noteOf(fine).hidden, true); assert.match(fine.status.textContent, /Added to Advanced options/);
  await fine.apply.click();
  assert.match(noteOf(fine).textContent, /Cell-body reassignment: Already at this value/);
  assert.equal(fine.apply.getAttribute('aria-busy'), undefined, 'nothing is sent from the picker: never busy');
  for (const control of [refused, fine]) assert.deepEqual(bannedWords(noteOf(control).textContent), []);
});

test('an advanced option holding a value the page cannot use is named for Start, with its range and its input', () => {
  const h = dom();
  const lengths = [{ flag: 'samples', label: 'Samples', type: 'int', default: 200, min: 0, advanced: true, group: 'Schedule' }];
  const options = new Options(h.container); options.load([...lengths, ...schema]);
  assert.deepEqual(options.problems(), []);
  const samples = options.rows.get('samples').input, compactness = options.rows.get('cell-compactness').input;
  assert.equal(samples.id, 'option-samples'); assert.equal(compactness.id, 'option-cell-compactness');
  samples.checkValidity = compactness.checkValidity = () => false;          // e.g. -5 and 20 typed (the browser's check)
  const problems = options.problems();
  assert.deepEqual(problems.map(p => [p.field, p.label, p.message]), [
    ['option-samples', 'Samples', 'Use a whole number of at least 0.'],
    ['option-cell-compactness', 'Cell compactness', 'Use a number from 0.001 to 10.'],
  ]);
  for (const p of problems) assert.deepEqual(bannedWords(`${p.label}: ${p.message}`), []);
});
