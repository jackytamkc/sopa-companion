import test from 'node:test';
import assert from 'node:assert/strict';
import { ClusterSettings, CLUSTER_SETTINGS_TEXT as T, FIELDS, OPEN_KEY, SLOW_MS, fieldText, sameSettings, settingsLine,
  settingsProblems, shown } from '../src/analysis-settings.mjs';
import { EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { bannedWords, QC_TEXT } from '../src/qc-copy.mjs';
import { domFixture } from './dom-fixture.mjs';

// The UMAP pane's Advanced settings (analysis-settings.mjs) in a fake page: the engine's describe as it sends it.
const DEFAULTS = Object.freeze({ resolution: 1, n_neighbors: 15, min_dist: 0.3, min_counts: 10, n_top_genes: 2000 });
const LIMITS = Object.freeze({ resolution: { min: 0.1, max: 3, step: 0.05, whole: false }, n_neighbors: { min: 5, max: 50, step: 1, whole: true },
  min_dist: { min: 0, max: 1, step: 0.05, whole: false }, min_counts: { min: 1, max: 500, step: 1, whole: true },
  n_top_genes: { min: 50, max: 5000, step: 50, whole: true } });
const meta = (settings = {}) => ({ settings: { ...DEFAULTS, ...settings }, defaults: { ...DEFAULTS }, limits: LIMITS,
  changed: !sameSettings({ ...DEFAULTS, ...settings }, DEFAULTS) });
const text = values => Object.fromEntries(Object.entries(values).map(([key, value]) => [key, String(value)]));

function fakeTimers() {
  const timers = [];
  return { timers, setTimeout(fn, ms) { const t = { fn, ms }; timers.push(t); return t; },
    clearTimeout(t) { const i = timers.indexOf(t); if (i >= 0) timers.splice(i, 1); } };
}
/** The disclosure in a fake page (in its body, so explain.mjs finds the fields), loaded with the defaults. */
function panel({ storage = null, loaded = meta() } = {}) {
  const { document } = domFixture(), sent = [], timers = fakeTimers(), slow = [];
  const s = new ClusterSettings({ document, onApply: values => { sent.push(values); }, onSlow: () => slow.push(1), storage, timers });
  s.slowHeard = slow;
  document.body.append(s.root);
  if (loaded) s.load(loaded);
  s.show(true);
  const field = key => document.getElementById(FIELDS.find(f => f.key === key).id);
  return { document, s, sent, timers, field };
}

test('the settings speak plainly: no banned word, no engine key on screen, each tooltip ends with its default', () => {
  for (const [key, words] of Object.entries(T)) assert.deepEqual(bannedWords(words), [], key);
  // The clusters and the map are made from proseg's count of each cell: the fewest transcripts are of that count.
  assert.equal(T.fewestTip, "Cells with fewer transcripts in proseg's count are left out of the clusters and the map. Raise it to leave out small fragments.");
  const { s, field } = panel();
  const nodes = s.root.all(), said = [s.root.textContent, ...nodes.map(node => node.title || '')].join(' ');
  assert.doesNotMatch(said, /n_neighbors|min_dist|min_counts|n_top_genes|n_pcs/);
  assert.deepEqual(bannedWords(said), []);
  for (const f of FIELDS) {
    const tip = `${f.tip} Default: ${shown(DEFAULTS[f.key])}.`;
    assert.equal(field(f.key).title, tip, f.key); assert.equal(field(f.key).parentElement.title, tip, f.key);
    assert.equal(field(f.key).type, 'number'); assert.equal(field(f.key).getAttribute('min'), String(LIMITS[f.key].min));
  }
  assert.equal(field('n_neighbors').title.endsWith('Default: 15.'), true); assert.equal(field('min_dist').title.endsWith('Default: 0.3.'), true);
  assert.match(field('n_top_genes').title, /a panel with fewer genes uses all of them/);
  assert.equal(s.form.getAttribute('novalidate'), '', 'the browser shows no validation bubbles of its own');
  assert.deepEqual([s.apply.type, s.resetButton.type], ['button', 'button']);
  assert.deepEqual([s.apply.getAttribute('data-umap-settings'), s.resetButton.getAttribute('data-umap-settings')], ['apply', 'reset']);
});

test("on a whole sample the fewest transcripts' tooltip says its default is not yet re-measured on proseg's count (E4): before its default, only there", () => {
  // The run's constants.kept (engine/qc_constants.py KEPT): "min_counts" on a whole sample; none on a region's or a
  // tiled region's run.
  const { s, field } = panel();
  s.setKept(['glow', 'where_words', 'start_band', 'min_counts', 'like_depth']);
  assert.equal(field('min_counts').title, `${T.fewestTip} ${QC_TEXT.notRemeasuredShort} Default: 10.`);
  assert.equal(field('min_counts').parentElement.title, field('min_counts').title);
  for (const f of FIELDS.filter(f => f.key !== 'min_counts')) assert.equal(field(f.key).title, `${f.tip} Default: ${shown(DEFAULTS[f.key])}.`, f.key);
  s.load(meta({ min_counts: 20 }));
  assert.equal(field('min_counts').title, `${T.fewestTip} ${QC_TEXT.notRemeasuredShort} Default: 10.`, 'kept across a new describe');
  for (const kept of [['glow', 'where_words'], [], null, 'min_counts']) {
    s.setKept(kept); assert.equal(field('min_counts').title, `${T.fewestTip} Default: 10.`, JSON.stringify(kept));
  }
  // Another run (or none): its own word, before its describe comes.
  s.setKept(['min_counts']); s.reset(); s.setKept(null);
  assert.equal(field('min_counts').title, T.fewestTip);
  assert.equal(QC_TEXT.notRemeasuredShort, "Not yet re-measured on proseg's count for runs like this one.");
  assert.deepEqual(bannedWords(QC_TEXT.notRemeasuredShort), []);
});

test('values read short, the log names changed settings, and a press finds what a field needs', () => {
  assert.deepEqual([shown(0.30000000000000004), shown(15), shown(1), shown(2000)], ['0.3', '15', '1', '2000']);
  assert.equal(settingsLine(DEFAULTS, DEFAULTS), '');
  assert.equal(settingsLine({ ...DEFAULTS, resolution: 1.5, n_neighbors: 30 }, DEFAULTS),
    'resolution 1.5, neighbours 30, spacing 0.3, fewest transcripts 10, genes used 2000 (changed)');
  assert.ok(sameSettings(DEFAULTS, { ...DEFAULTS, min_dist: 0.3 + 1e-12 }) && !sameSettings(DEFAULTS, { ...DEFAULTS, min_counts: 11 }));
  assert.deepEqual(settingsProblems(text({ resolution: 0.1, n_neighbors: 50, min_dist: 0, min_counts: 500, n_top_genes: 5000 }), LIMITS), []);
  assert.deepEqual(settingsProblems(text({ ...DEFAULTS, n_top_genes: 49 }), LIMITS),
    [{ field: 'umap-genes', label: 'Genes used', message: 'Use a whole number from 50 to 5000.' }]);
});

test('a field at its default goes whatever the limit, as the engine takes it (genes used set above the CPU worker\'s limit)', async () => {
  const defaults = { ...DEFAULTS, n_top_genes: 8000 };
  const tooMany = [{ field: 'umap-genes', label: 'Genes used', message: 'Use a whole number from 50 to 5000.' }];
  assert.deepEqual(settingsProblems(text(defaults), LIMITS, defaults), []);
  assert.deepEqual(settingsProblems(text({ ...defaults, n_top_genes: 7000 }), LIMITS, defaults), tooMany);
  assert.deepEqual(settingsProblems(text(defaults), LIMITS), tooMany, 'without the defaults: the limit alone');
  const { s, sent, field } = panel({ loaded: { settings: { ...defaults }, defaults, limits: LIMITS, changed: false } });
  field('resolution').value = '2';
  await s.apply.click();
  assert.deepEqual(sent, [{ ...defaults, resolution: 2 }], 'Apply with genes used left at its default sends');
  assert.equal(s.note.hidden, true);
  s.done();
  await s.resetButton.click();
  assert.deepEqual(sent.at(-1), defaults, 'Reset sends the defaults');
});

test('Apply is never greyed out: a press names each field that needs something, outlines it, says why under it and sends nothing', async () => {
  const { document, s, sent, field } = panel();
  field('resolution').value = '';                  // empty
  field('n_neighbors').value = '15.5';             // a fraction where a whole number goes
  field('min_dist').value = '2';                   // out of range
  field('n_top_genes').value = 'many';             // not a number
  await s.apply.click();
  assert.equal(s.apply.disabled, false); assert.deepEqual(sent, []); assert.equal(s.note.hidden, false);
  const listed = s.note.textContent;
  for (const line of ['Cluster resolution: Use a number from 0.1 to 3.', 'Neighbours per cell: Use a whole number from 5 to 50.',
    'Dot spacing: Use a number from 0 to 1.', 'Genes used: Use a whole number from 50 to 5000.']) assert.ok(listed.includes(line), line);
  assert.ok(!listed.includes(T.fewest));
  for (const key of ['resolution', 'n_neighbors', 'min_dist', 'n_top_genes']) {
    assert.ok(field(key).classList.contains(NEEDS_CLASS), key);
    const row = field(key).parentElement.childNodes;   // the label is a grid: the reason takes its own row under the field
    assert.equal(row.at(-1).className, 'field-note', key); assert.equal(row.at(-2), field(key), key);
  }
  assert.ok(!field('min_counts').classList.contains(NEEDS_CLASS));
  assert.equal(document.activeElement, field('resolution'), 'the first is focused');
  field('resolution').value = '2'; field('n_neighbors').value = '20'; field('min_dist').value = '0.5'; field('n_top_genes').value = '500';
  await s.apply.click();
  assert.deepEqual(sent, [{ ...DEFAULTS, resolution: 2, n_neighbors: 20, min_dist: 0.5, n_top_genes: 500 }]);
  assert.equal(s.note.hidden, true); assert.ok(!field('resolution').classList.contains(NEEDS_CLASS));
});

test('while the new settings are worked on both buttons are busy; each outcome has its line, cleared by a result, a press or an edit', async () => {
  const { s, sent, timers, field } = panel();
  field('resolution').value = '2';
  await s.apply.click();
  assert.deepEqual(sent, [{ ...DEFAULTS, resolution: 2 }]);
  assert.equal(s.isBusy, true);
  assert.deepEqual([s.apply.getAttribute('aria-busy'), s.resetButton.getAttribute('aria-busy')], ['true', 'true']);
  assert.deepEqual([s.apply.textContent, s.resetButton.textContent], [T.applying, T.reset]);
  await s.resetButton.click();                      // a press of either meanwhile: the busy line, nothing filled or sent
  assert.ok(s.note.textContent.includes(EXPLAIN_TEXT.busy)); assert.equal(field('resolution').value, '2');
  await s.apply.click();
  assert.equal(sent.length, 1);
  assert.equal(timers.timers.length, 1); assert.equal(timers.timers[0].ms, SLOW_MS);
  s.done();
  assert.equal(s.isBusy, false); assert.equal(s.apply.getAttribute('aria-busy'), null); assert.equal(s.apply.textContent, T.apply);
  assert.equal(timers.timers.length, 0); assert.equal(s.line.hidden, true);
  for (const kind of ['failed', 'refused', 'slow', 'tooFew', 'unchanged']) {
    await s.apply.click(); assert.equal(s.isBusy, true, kind);
    s[kind](); assert.equal(s.isBusy, false, kind); assert.equal(s.line.textContent, T[kind], kind); assert.equal(s.line.hidden, false, kind);
  }
  for (const kind of ['failed', 'slow', 'tooFew']) { s[kind](); s.resultShown(); assert.equal(s.line.hidden, true, `${kind}: a result clears it`); }
  for (const kind of ['refused', 'unchanged']) { s[kind](); s.resultShown(); assert.equal(s.line.textContent, T[kind], `${kind} stays`); }
  field('min_dist').value = '0.4'; field('min_dist').emit('input');
  assert.equal(s.line.hidden, true, 'an edit clears it');
  s.refused(); await s.apply.click();
  assert.equal(s.line.hidden, true, 'a press clears it');
  timers.timers.at(-1).fn();                        // two minutes with no result of the new settings
  assert.equal(s.line.textContent, T.slow); assert.equal(s.isBusy, false, 'the buttons are free again');
  assert.equal(s.slowHeard.length, 1, 'the page hears of it (its log)');
  s.root.open = false; s.tooFew();
  assert.equal(s.root.open, true, 'too few cells opens the disclosure to say so');
});

test('Reset fills the defaults and sends them, also when the fields hold them already', async () => {
  const { s, sent, field } = panel({ loaded: meta({ resolution: 2, n_top_genes: 500 }) });
  assert.equal(field('resolution').value, '2'); assert.equal(field('n_top_genes').value, '500');
  field('n_neighbors').value = '';
  await s.apply.click();
  assert.ok(field('n_neighbors').classList.contains(NEEDS_CLASS));
  await s.resetButton.click();
  assert.deepEqual(sent, [{ ...DEFAULTS }]);
  assert.deepEqual(FIELDS.map(f => field(f.key).value), ['1', '15', '0.3', '10', '2000']);
  assert.ok(!field('n_neighbors').classList.contains(NEEDS_CLASS), "Apply's outlines go");
  s.done(); s.load(meta());
  await s.resetButton.click();
  assert.deepEqual(sent, [{ ...DEFAULTS }, { ...DEFAULTS }], 'sent again: new runs follow the defaults too');
});

test('the title says "(changed)" while the settings differ; the fields follow the engine unless they hold edits not applied yet', () => {
  const { s, field } = panel({ loaded: null });
  assert.equal(s.root.hidden, true, 'hidden until the settings load');
  s.load({ settings: DEFAULTS }); assert.equal(s.loaded, false, 'nothing without defaults and limits'); assert.equal(s.root.hidden, true);
  s.load(meta());
  assert.equal(s.root.hidden, false); assert.equal(s.titleText.textContent, T.title); assert.equal(s.changed, false);
  assert.equal(field('n_neighbors').getAttribute('max'), '50'); assert.equal(field('n_top_genes').getAttribute('step'), '50');
  s.load(meta({ n_neighbors: 30 }));
  assert.equal(s.titleText.textContent, T.titleChanged); assert.equal(s.changed, true); assert.equal(field('n_neighbors').value, '30');
  field('resolution').value = '2.5';
  assert.equal(s.dirty, true);
  s.load(meta({ n_neighbors: 40 }));                // another window's change, while this one holds an edit
  assert.equal(field('resolution').value, '2.5'); assert.equal(field('n_neighbors').value, '30');
  assert.equal(s.titleText.textContent, T.titleChanged);
  field('resolution').value = '1'; field('n_neighbors').value = '40';   // the fields hold the engine's settings again
  assert.equal(s.dirty, false);
  s.load(meta({ n_neighbors: 40, min_counts: 25 }));   // no edit left: the fields follow
  assert.equal(field('n_neighbors').value, '40'); assert.equal(field('min_counts').value, '25');
  s.show(false); assert.equal(s.root.hidden, true, 'a collapsed or closed pane hides it');
  s.show(true); s.reset();
  assert.equal(s.root.hidden, true); assert.equal(s.loaded, false); assert.equal(field('resolution').value, '');
  assert.equal(s.titleText.textContent, T.title); assert.equal(s.logLine(), '');
});

test('Enter in a field applies; the open state is remembered; storage that throws, or none, is harmless', () => {
  const store = new Map(), storage = { getItem: key => store.get(key) ?? null, setItem: (key, value) => store.set(key, String(value)) };
  const { s, sent, field } = panel({ storage });
  assert.equal(s.root.open, false);
  field('resolution').value = '1.25';
  // A key in a field reaches the form's listener with the field as its target, as the browser's bubbling gives it.
  const enter = s.form.emit('keydown', { key: 'Enter', target: field('resolution'), preventDefault() { this.prevented = true; } });
  assert.equal(enter.prevented, true); assert.deepEqual(sent, [{ ...DEFAULTS, resolution: 1.25 }]);
  s.done(); s.form.emit('keydown', { key: 'a', target: field('resolution') });
  assert.equal(sent.length, 1);
  const submit = s.form.emit('submit', { preventDefault() { this.prevented = true; } });
  assert.equal(submit.prevented, true, 'the form never submits by itself');
  s.root.open = true; s.root.emit('toggle');
  assert.equal(store.get(OPEN_KEY), '1');
  assert.equal(panel({ storage }).s.root.open, true, 'a later page opens it again');
  s.root.open = false; s.root.emit('toggle');
  assert.equal(store.get(OPEN_KEY), '0');
  const blocked = { getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } };
  for (const bad of [blocked, null]) {
    const other = panel({ storage: bad });
    assert.equal(other.s.root.open, false);
    other.s.root.open = true; other.s.root.emit('toggle');
    assert.equal(other.s.root.open, true);
  }
});

test('Enter on a button is that button\'s own press: on Reset it throws the edits away, never applies them', async () => {
  const { s, sent, field } = panel({ loaded: meta({ resolution: 2 }) });
  field('n_neighbors').value = '40';                        // an edit Reset is about to throw away
  s.resetButton.focus();
  for (const button of [s.resetButton, s.apply]) {
    const enter = s.form.emit('keydown', { key: 'Enter', target: button, preventDefault() { this.prevented = true; } });
    assert.equal(enter.prevented, undefined, 'the browser presses the button itself'); assert.deepEqual(sent, []);
  }
  await s.resetButton.click();                              // the press the browser makes of Enter on the focused button
  assert.deepEqual(sent, [{ ...DEFAULTS }]); assert.equal(field('n_neighbors').value, '15');
});

test('a setting with more than six significant digits fills its field exactly: no edit seen, the fields follow, Reset reaches the default', async () => {
  const defaults = { ...DEFAULTS, resolution: 0.6666667 };                  // [analysis] resolution = 0.6666667
  const described = settings => ({ settings: { ...defaults, ...settings }, defaults, limits: LIMITS, changed: !sameSettings({ ...defaults, ...settings }, defaults) });
  const { s, sent, field } = panel({ loaded: described({}) });
  assert.equal(field('resolution').value, '0.6666667'); assert.equal(s.dirty, false, "the engine's own settings are no edit");
  s.load(described({ resolution: 1 / 3, min_dist: 0.1 + 0.2 }));         // a tool's change, at full float precision
  assert.equal(field('resolution').value, String(1 / 3), 'the fields follow it'); assert.equal(field('min_dist').value, String(0.1 + 0.2));
  assert.equal(s.dirty, false);
  field('n_neighbors').value = '20';
  await s.apply.click();                                    // another setting applied: these two stay as they are
  assert.deepEqual(sent, [{ ...defaults, resolution: 1 / 3, min_dist: 0.1 + 0.2, n_neighbors: 20 }]);
  s.done();
  await s.resetButton.click();
  assert.deepEqual(sent.at(-1), defaults); assert.equal(sent.at(-1).resolution, 0.6666667);
  assert.deepEqual([shown(0.6666667), shown(1 / 3)], ['0.666667', '0.333333'], 'words (tooltips, the log) keep six digits');
  assert.deepEqual([fieldText(1), fieldText(0.3), fieldText(2000), fieldText(0.6666667)], ['1', '0.3', '2000', '0.6666667']);
});
