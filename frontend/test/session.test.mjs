import test from 'node:test';
import assert from 'node:assert/strict';
import { bannedWords } from '../src/qc-copy.mjs';
import { SESSION_TEXT, apiSession, sessionName, sampleHeader, windowTitle, sampleStatus, waitingText, chooseText, addWords, addLine, resultsRow }
  from '../src/session.mjs';
import { MARK, sessionId, PrefixedStorage, MemoryStorage, bootStorage } from '../src/storage-boot.mjs';

// The page in a notebook's session: what health.session and the session pushes say, in the
// page's words; and the tab's own storage for a session opened as a link.
const health = (over = {}, sample = {}) => ({ ok: true, session: { mode: 'api', id: '1f3a9c2e',
  sample: { id: 'z1', name: 'SMP02', backed: true, zarr_name: 'SMP02.zarr', view: 'in_place', state: 'ready', progress: { done: 0, total: 0 }, message: null,
    memory_only: [], tools: ['proseg', 'stain'], ...sample },
  tool: null, writes: 'zarr', notebook: { busy: false }, patches: false, link: false, state: 'running', ...over } });

test('every word is plain and none names the notebook\'s variable', () => {
  for (const text of Object.values(SESSION_TEXT)) { assert.deepEqual(bannedWords(text), [], text); assert.doesNotMatch(text, /\bsdata\b/, text); }
});
test('a desktop engine (or an older one) has no session; a notebook\'s engine says its sample, tool and where results go', () => {
  assert.equal(apiSession({ ok: true }), null); assert.equal(apiSession({ ok: true, session: { mode: 'desktop' } }), null); assert.equal(apiSession(null), null);
  const s = apiSession(health({ tool: 'stain', patches: true }));
  assert.equal(s.mode, 'api'); assert.equal(s.tool, 'stain'); assert.equal(s.patches, true); assert.equal(s.sample.name, 'SMP02');
  assert.deepEqual(s.sample.tools, ['proseg', 'stain']); assert.equal(s.state, 'running');
  // A push reads the same: {ev: "session", ...} beside the fields.
  const pushed = apiSession({ ev: 'session', ...health({ state: 'ending', notebook: { busy: true } }).session });
  assert.equal(pushed.state, 'ending'); assert.equal(pushed.notebook.busy, true);
  assert.equal(apiSession(health({ tool: 'other', writes: 'elsewhere' })).tool, null);
  assert.equal(apiSession(health({ writes: 'elsewhere' })).writes, 'memory', 'an unknown writing mode never claims a zarr');
});
test('the header names the bound sample everywhere, never a path; the window title tells two sessions apart', () => {
  assert.deepEqual(sampleHeader(apiSession(health())), { name: 'SMP02', title: 'From your notebook; results go into this object and SMP02.zarr' });
  assert.deepEqual(sampleHeader(apiSession(health({ writes: 'memory' }, { name: 'SMP02 (in memory)', backed: false, zarr_name: null }))),
    { name: 'SMP02 (in memory)', title: 'From your notebook; results go into this object (in memory only)' });
  assert.equal(sampleHeader(apiSession(health({ writes: 'memory_autosave_off' }))).title, SESSION_TEXT.sampleMemory, "auto-save off: memory only");
  assert.equal(windowTitle(apiSession(health())), 'SoPA companion · SMP02'); assert.equal(windowTitle(null), 'SoPA companion');
  assert.equal(chooseText(apiSession(health())), 'Choose a tool for SMP02.');
  assert.equal(waitingText(apiSession(health())), 'Opens when SMP02 is ready');
  assert.equal(waitingText(apiSession(health({}, { name: null }))), 'Opens when your SpatialData is ready');
  assert.equal(sessionName(apiSession(health({}, { name: null }))), 'your SpatialData');
});
test('the first screen says while the SpatialData is prepared, opened, or why it could not be opened', () => {
  assert.equal(sampleStatus(apiSession(health())), null);
  assert.deepEqual(sampleStatus(apiSession(health({}, { state: 'preparing', progress: { done: 2.1e9, total: 6.0e9 } }))),
    { state: 'preparing', text: 'Preparing your SpatialData for the app: 2.1 of 6.0 GB', fraction: 0.35 });
  assert.deepEqual(sampleStatus(apiSession(health({}, { state: 'preparing' }))), { state: 'preparing', text: 'Preparing your SpatialData for the app…' },
    'no size yet: no bar');
  assert.equal(sampleStatus(apiSession(health({}, { state: 'preparing', progress: { done: 7e9, total: 6e9 } }))).fraction, 1, 'never past full');
  assert.deepEqual(sampleStatus(apiSession(health({}, { state: 'opening' }))), { state: 'opening', text: 'Opening SMP02…' });
  assert.deepEqual(sampleStatus(apiSession(health({}, { state: 'failed', message: 'No transcripts and no image the app can place' }))),
    { state: 'failed', text: 'Could not open your SpatialData: No transcripts and no image the app can place' });
});
test('Add to your SpatialData: the dialog\'s words follow where the result goes, and its states in plain words', () => {
  const zarr = apiSession(health()), memory = apiSession(health({ writes: 'memory' }, { backed: false, zarr_name: null })),
    off = apiSession(health({ writes: 'memory_autosave_off' }));
  assert.deepEqual(addWords(zarr, { kind: 'proseg' }), { title: 'Add to your SpatialData?', submit: 'Add', tick: 'Make it the default segmentation (as sopa does)',
    help: "Shapes and a table of proseg's counts, added as sopa's own functions add them. Also written to SMP02.zarr.", saving: 'Adding…' });
  assert.equal(addWords(memory, { kind: 'proseg' }).help, "Shapes and a table of proseg's counts, added as sopa's own functions add them. In memory only: this SpatialData has no zarr.");
  assert.match(addWords(off, { kind: 'stain' }).help, /In memory only: sopa's auto-save is off\.$/); assert.equal(addWords(off, { kind: 'stain' }).submit, 'Add as shapes');
  const run = save => ({ kind: 'proseg', zarr_save: save });
  assert.equal(addLine(zarr, run({ state: 'queued', written_to: '/data/SMP02.zarr' })), 'Written to SMP02.zarr · in your notebook as soon as it is free');
  assert.equal(addLine(memory, run({ state: 'queued', written_to: null })), 'Ready · in your notebook as soon as it is free');
  assert.equal(addLine(apiSession(health({ notebook: { busy: true } })), run({ state: 'queued', written_to: null })), 'Waiting for your notebook: a cell is running');
  assert.equal(addLine(zarr, run({ state: 'saved', added: true, cells: 40070, shapes_key: 'proseg_boundaries', table_key: 'proseg_table' })),
    'Added 40,070 cells as proseg_boundaries + proseg_table');
  assert.equal(addLine(zarr, run({ state: 'failed', message: 'clash', clash: ['proseg_boundaries'], written_to: '/data/SMP02.zarr' })),
    "Not added to your notebook's object: it already has proseg_boundaries (made after you pressed Add). It is in SMP02.zarr.");
  assert.equal(addLine(zarr, run({ state: 'saving', stage: 'building' })), ''); assert.equal(addLine(zarr, run(null)), '');
  assert.deepEqual(resultsRow(run(null)), { label: 'Not added to your SpatialData yet', button: 'Add to your SpatialData…' });
  assert.deepEqual(resultsRow(run({ state: 'saved', shapes_key: 'proseg_boundaries', table_key: 'proseg_table' })),
    { label: 'Added to your SpatialData as proseg_boundaries + proseg_table', button: 'Add again…' });
  // Queued (in the object's zarr, or ready in memory only): on its way, never "Added" before the notebook has it.
  for (const written_to of [null, '/data/SMP02.zarr'])
    assert.deepEqual(resultsRow(run({ state: 'queued', written_to, shapes_key: 'proseg_boundaries', table_key: 'proseg_table' })),
      { label: 'Adding to your SpatialData as proseg_boundaries + proseg_table', button: 'Add again…' });
  assert.deepEqual(resultsRow(run({ state: 'saving' })), { label: 'Not added to your SpatialData yet', button: 'View progress…' });
  assert.equal(resultsRow({ kind: 'stain', zarr_save: { state: 'saving' } }).button, 'View progress…');
  assert.equal(resultsRow({ kind: 'stain', zarr_save: { state: 'queued', shapes_key: 'cellpose_boundaries_1' } }, { outdated: true }).label, SESSION_TEXT.addedOutdated);
});

// ---- the tab's own storage (storage-boot.mjs) ---------------------------------------------------------------------------
test('a session\'s link (api=…) gives the tab storage of its own, remembered by the tab; without it the browser\'s storage stays', () => {
  const tab = new MemoryStorage(), real = new MemoryStorage();
  real.setItem('sopa-companion:panel:qc-panel', 'true');
  const win = { location: { search: '?api=1f3a9c2e' }, sessionStorage: tab, localStorage: real };
  const store = bootStorage(win);
  assert.ok(store instanceof PrefixedStorage); assert.equal(win.localStorage, store, 'localStorage is the session\'s now');
  assert.equal(win.localStorage.getItem('sopa-companion:panel:qc-panel'), null, 'nothing of the browser\'s storage');
  win.localStorage.setItem('sopa-companion:panel:qc-panel', 'false');
  assert.equal(tab.getItem('sc:1f3a9c2e:sopa-companion:panel:qc-panel'), 'false'); assert.equal(real.getItem('sopa-companion:panel:qc-panel'), 'true');
  assert.equal(win.localStorage.length, 1); assert.equal(win.localStorage.key(0), 'sopa-companion:panel:qc-panel');
  win.localStorage.clear(); assert.equal(tab.getItem(MARK), '1f3a9c2e', 'clear() leaves the tab\'s mark');
  // A reload of the tab, its address no longer naming the session: the tab remembers it.
  const again = { location: { search: '' }, sessionStorage: tab, localStorage: real };
  assert.ok(bootStorage(again)); assert.equal(sessionId('', tab), '1f3a9c2e');
  // A desktop page (no api, nothing remembered): untouched.
  const desktop = { location: { search: '?run=r1' }, sessionStorage: new MemoryStorage(), localStorage: real };
  assert.equal(bootStorage(desktop), null); assert.equal(desktop.localStorage, real);
  assert.equal(sessionId('?api=1', null), 'api'); assert.equal(sessionId('?api=../../x', null), 'api', 'never a key from an odd value');
  // A tab whose sessionStorage refuses: kept in memory for the page's life.
  const refusing = { location: { search: '?api=abcd1234' }, get sessionStorage() { throw new Error('SecurityError'); }, localStorage: real };
  const kept = bootStorage(refusing); kept.setItem('k', 'v'); assert.equal(refusing.localStorage.getItem('k'), 'v'); assert.equal(real.getItem('k'), null);
});
