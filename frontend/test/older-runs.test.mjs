import test from 'node:test';
import assert from 'node:assert/strict';
import { domFixture, settle } from './dom-fixture.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { OLDER_TEXT, OlderRunsPrompt, olderRows, olderDone, olderCount, regionWords, whenWords, rowLine, rowTooltip, rowRefusal, rowStatus,
  underWay, POLL_MS, SAVED_MS, GLIDE_MS, GLIDE_FPS } from '../src/older-runs.mjs';

// Runs from before the one-run app: the rows' words, then the prompt against the DOM
// fixture with a virtual clock and a fake engine that answers as engine/runs.py's older rows do (Runs.older: {runs, done,
// count}; a row's answer, stage and message while under way; stage "failed" and its message when it failed; a row
// answered for good leaves `runs` for `done`; Later sets every row aside, `later`, and takes the queued ones off the queue).
const local = (month, day, hour, minute) => new Date(2026, month - 1, day, hour, minute).toISOString();
const proseg = (id, extra = {}) => ({ id, kind: 'proseg', method: 'proseg', state: 'done', iter: 1200, sample: 'z67', sample_name: 'SMP02',
  created: local(10, 14, 15, 42), region: { kind: 'full' }, prior: { cells: { kind: 'shapes', key: 'cellpose_boundaries_1' }, nuclei: null },
  options: { iterations: 2000, 'voxel-size': 2 }, can_add: true, why: 'unsaved', reason: '', detail: '', still_saving: false, answer: null,
  stage: null, message: null, zarr_save: null, later: false, ...extra });
const stain = (id, extra = {}) => ({ id, kind: 'stain', method: 'cellpose', state: 'stopped', sample: 'z65', sample_name: 'SMP01', created: local(10, 9, 10, 5),
  region: { kind: 'box', box: [0, 0, 2000, 1500] }, tiles: { done: 37, of: 120 }, stain: { method: 'cellpose', model: 'cpsam', nucleus: 'DAPI',
    cytoplasm: 'ATP1A1/CD45/E-Cadherin' }, can_add: true, why: 'unsaved', reason: '', detail: '', still_saving: false, answer: null, stage: null,
  message: null, zarr_save: null, later: false, ...extra });

test('every word is plain, and none names the notebook\'s object', () => {
  for (const text of Object.values(OLDER_TEXT)) { assert.deepEqual(bannedWords(text), [], text); assert.doesNotMatch(text, /sdata/); }
});
test('a row reads sample · region · tool · state · time; its tooltip the starting cells and changed settings, or the model and channels', () => {
  assert.equal(rowLine(proseg('a')), 'SMP02 · whole sample · Proseg · Finished at iteration 1,200 · 14 Oct 15:42');
  assert.equal(rowLine(stain('b')), 'SMP01 · box 2.0 × 1.5 mm · Cellpose · 37 of 120 tiles done · 9 Oct 10:05');
  // An earlier version's InstanSeg run (InstanSeg is dormant now): named as what it is, a stain segmentation.
  assert.equal(rowLine(stain('c', { method: 'instanseg', stain: { method: 'instanseg', model: 'fluorescence_nuclei_and_cells', channels: ['DAPI', 'CD45'] } }))
    .split(' · ')[2], 'Stain segmentation');
  assert.equal(rowLine(proseg('d', { zarr_save: { state: 'failed', message: 'Disk full' } })),
    'SMP02 · whole sample · Proseg · Finished at iteration 1,200 · the last save failed · 14 Oct 15:42');
  assert.equal(rowLine({ id: 'e', kind: 'patches', sample_name: 'SMP03', when: local(10, 1, 9, 0) }), 'SMP03 · whole sample · Training patches · 1 Oct 09:00');
  const labels = { iterations: 'Iterations', 'voxel-size': 'Voxel size' }, defaults = { 'voxel-size': 2 };
  assert.equal(rowTooltip(proseg('a'), { label: f => labels[f], defaults: f => defaults[f] }), 'Started from cellpose_boundaries_1 · Iterations 2,000');
  assert.equal(rowTooltip(proseg('a', { prior: { cells: { kind: 'column', column: 'cell_id' }, nuclei: { kind: 'shapes', key: 'nuclei' } }, options: {} })),
    'Started from cell_id cells and nuclei nuclei');
  assert.equal(rowTooltip(stain('b')), 'cpsam · DAPI + ATP1A1/CD45/E-Cadherin');
  // Two runs on one sample and region, a setting apart, the same day: told apart by the time and the tooltip.
  const one = proseg('x', { created: local(10, 14, 15, 42) }), two = proseg('y', { created: local(10, 14, 16, 3), options: { iterations: 3000 } });
  assert.notEqual(rowLine(one), rowLine(two)); assert.notEqual(rowTooltip(one), rowTooltip(two));
});
test('regions, times and the engine\'s answer in a few words', () => {
  assert.equal(regionWords(null), 'whole sample'); assert.equal(regionWords({ kind: 'full' }), 'whole sample');
  assert.equal(regionWords({ kind: 'box', box: [100, 100, 400, 350] }), 'box 300 × 250 µm');
  assert.equal(regionWords({ kind: 'polygon', points: [[0, 0], [2000, 0], [2000, 1000]] }), 'lasso 1.0 mm²');
  assert.equal(regionWords({ kind: 'polygon', points: [[0, 0], [100, 0], [100, 100], [0, 100]] }), 'lasso 10,000 µm²');
  assert.equal(whenWords(local(12, 31, 23, 5)), '31 Dec 23:05'); assert.equal(whenWords('not a time'), ''); assert.equal(whenWords(null), '');
  const older = { runs: [proseg('a'), null, { id: '' }, proseg('b', { later: true })], done: [{ id: 'c', stage: 'saved', message: 'Saved as x' }], count: 1 };
  assert.deepEqual(olderRows(older).map(r => r.id), ['a'], 'a row set aside by Later waits for the next start');
  assert.equal(olderDone(older).get('c').message, 'Saved as x');
  assert.equal(olderCount(older), 1); assert.equal(olderCount(3), 3); assert.equal(olderCount({ runs: [proseg('a')] }), 1); assert.equal(olderCount(undefined), 0);
});
test('a row that cannot be saved says why (the engine\'s own words, else its reason); a row\'s status follows the engine', () => {
  assert.equal(rowRefusal(proseg('a')), null);
  assert.deepEqual(rowRefusal(proseg('a', { can_add: false, why: 'xenium', reason: 'Made on a Xenium folder, which cannot hold its cells.',
    detail: 'Its results are in /runs/a/output: copy them before you discard it.' })),
    { text: 'Made on a Xenium folder, which cannot hold its cells.', title: 'Its results are in /runs/a/output: copy them before you discard it.' });
  assert.equal(rowRefusal(proseg('a', { can_add: false, why: 'sample_gone', sample_path: '/data/SMP02.zarr' })).text, 'SMP02 is no longer where it was.');
  assert.equal(rowRefusal(proseg('a', { can_add: false, why: 'sample_gone', sample_path: '/data/SMP02.zarr' })).title,
    'It was at /data/SMP02.zarr. Put it back and start the app again to save this run.');
  assert.equal(rowRefusal(stain('b', { can_add: false, why: 'filtered' })).text, OLDER_TEXT.filtered);
  assert.equal(rowRefusal(proseg('a', { can_add: false, why: 'patches', folder: '/runs/p' })).title, 'Its boxes are in /runs/p: copy them before you discard it.');
  assert.equal(rowRefusal(proseg('a', { can_add: false, why: 'unknown_code' })).text, OLDER_TEXT.cannot);
  assert.equal(rowRefusal(proseg('a', { can_add: false, why: 'still_saving', still_saving: true })), null, 'still being saved: no refusal, no buttons');
  assert.equal(rowStatus(proseg('a')), '');
  assert.equal(rowStatus(proseg('a', { answer: 'add', stage: 'waiting', message: 'Waiting…' })), 'Waiting…');
  assert.equal(rowStatus(stain('b', { answer: 'add', stage: 'joining' })), 'Joining the finished tiles…');
  assert.equal(rowStatus(proseg('a', { answer: 'add', stage: 'writing', message: 'Saving to the sample…' })), 'Saving to the sample…');
  assert.equal(rowStatus(proseg('a', { answer: 'discard', stage: 'removing' })), 'Discarding…');
  assert.equal(rowStatus(proseg('a', { answer: 'add', stage: 'failed', message: "Couldn't save: Disk full" })), "Couldn't save: Disk full");
  assert.equal(underWay(proseg('a', { answer: 'add', stage: 'failed' })), false, 'a failed answer waits for another');
  assert.equal(underWay(proseg('a', { answer: 'add', stage: 'writing' })), true);
  assert.equal(rowStatus(proseg('a', { zarr_save: { state: 'failed', message: 'No space left' } })), "Couldn't save: No space left");
  assert.equal(rowStatus(proseg('a', { zarr_save: { state: 'saved', shapes_key: 'x' } })), '', 'a listed row is to answer, whatever it saved before');
  assert.equal(rowStatus(proseg('a', { still_saving: true })), 'Still being saved by the last session');
});

// ---- the prompt ---------------------------------------------------------------------------------------------------------
function clock() {
  const c = { t: 0, timers: [], n: 0 };
  c.setTimeout = (fn, ms) => { const id = ++c.n; c.timers.push({ id, at: c.t + ms, fn }); return id; };
  c.clearTimeout = id => { c.timers = c.timers.filter(x => x.id !== id); };
  c.advance = async ms => {
    const end = c.t + ms;
    for (;;) {
      const due = c.timers.filter(x => x.at <= end).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      c.t = due.at; c.timers = c.timers.filter(x => x !== due); due.fn(); await settle(); await settle();
    }
    c.t = end;
  };
  return c;
}
/** The prompt on a fake engine whose rows are `engine.runs` and answered ones `engine.done`; an answer queues itself
 *  (answer, stage "waiting") unless `refuse` throws; `work(id)` and `finishRow(id, stage, message)` move it on. Later
 *  (Runs.answer_all_older "later") sets every row aside: the one at work goes on, the queued ones leave the queue. */
function setup(rows, { refuse = null, answerAll = undefined, reducedMotion = () => false } = {}) {
  const dom = domFixture(), c = clock(), calls = [], notices = [];
  const engine = { runs: rows.map(r => ({ ...r })), done: [] };
  const state = () => ({ runs: engine.runs.map(r => ({ ...r })), done: engine.done.map(r => ({ ...r })), count: engine.runs.filter(r => !r.later).length });
  const queue = (id, answer) => { engine.runs = engine.runs.map(r => r.id === id ? { ...r, answer, stage: 'waiting', message: 'Waiting…', later: false } : r); };
  engine.work = (id, stage, message) => { engine.runs = engine.runs.map(r => r.id === id ? { ...r, stage, message } : r); };
  engine.finishRow = (id, stage, message, zarr_save = null) => {
    const row = engine.runs.find(r => r.id === id);
    engine.runs = engine.runs.filter(r => r.id !== id); engine.done.push({ ...row, stage, message, zarr_save });
  };
  engine.fail = (id, message) => engine.work(id, 'failed', message);
  const prompt = new OlderRunsPrompt({ clock: c, notice: text => notices.push(text), reducedMotion,
    fetchRows: async () => { calls.push(['GET']); return state(); },
    answer: async (id, answer) => { calls.push(['POST', id, answer]); if (refuse) throw refuse(id, answer);
      if (!engine.runs.some(r => r.id === id)) throw Object.assign(new Error('Unknown run'), { status: 404 });
      queue(id, answer); return engine.runs.find(r => r.id === id); },
    answerAll: answerAll ?? (async answer => { calls.push(['POST', null, answer]);
      if (answer === 'later') {
        engine.runs = engine.runs.map(r => ({ ...r, later: true, ...(r.stage === 'waiting' ? { answer: null, stage: null, message: null } : {}) }));
        return state();
      }
      for (const r of engine.runs) if (!r.still_saving && !(r.answer && r.stage !== 'failed') && (answer !== 'add' || r.can_add)) queue(r.id, answer);
      return state(); }),
    label: f => ({ iterations: 'Iterations' })[f], defaults: f => ({ 'voxel-size': 2 })[f] }, dom.document);
  const row = id => prompt.dialog.all().find(n => n.getAttribute?.('data-older-run') === id);
  const button = (id, answer) => (id ? row(id) : prompt.dialog).all().find(n => n.getAttribute?.('data-older-answer') === answer);
  return { ...dom, c, calls, notices, engine, state, prompt, row, button };
}
const shown = node => { for (let n = node; n; n = n.parentElement) if (n.hidden) return false; return true; };

test('the prompt lists every row with its own Save and Discard, the start notice under the lead, and the first Save in focus', async () => {
  const h = setup([proseg('a'), stain('b')]);
  const done = h.prompt.ask(h.state(), { lead: 'A run on SMP03 that was not saved to the sample was removed when the app started.' });
  assert.equal(h.prompt.open, true); assert.equal(h.$('older-title').textContent, 'Save your older runs?');
  assert.equal(h.$('older-lead').textContent, OLDER_TEXT.lead);
  assert.equal(h.$('older-notice').textContent, 'A run on SMP03 that was not saved to the sample was removed when the app started.');
  assert.equal(h.$('older-notice').hidden, false);
  assert.equal(h.document.activeElement, h.button('a', 'add'), 'the first Save');
  assert.equal(h.button('a', 'add').textContent, 'Save to the sample'); assert.equal(h.button('b', 'discard').textContent, 'Discard');
  assert.equal(h.button('a', 'add').title, "Writes its cells into SMP02 under new names, then deletes the run. Nothing is replaced, and it does not become the sample's segmentation.");
  assert.equal(h.button('a', 'discard').title, OLDER_TEXT.discardTitle);
  assert.equal(shown(h.button(null, 'all-add')), true); assert.equal(shown(h.button(null, 'all-discard')), true, 'Save all and Discard all from two rows');
  assert.equal(h.button(null, 'later').textContent, 'Later'); assert.equal(h.button(null, 'later').title, OLDER_TEXT.laterTitle);
  assert.equal(h.row('a').all().find(n => n.className === 'older-line').title, 'Started from cellpose_boundaries_1 · Iterations 2,000');
  // One row: its own title; no Save all.
  const one = setup([proseg('a')]); one.prompt.ask(one.state());
  assert.equal(one.$('older-title').textContent, 'Save your older run?'); assert.equal(shown(one.button(null, 'all-add')), false);
  assert.equal(one.$('older-notice').hidden, true);
  // A Discard: the engine queues it, carries it out and lists the row no more; the last one gone ends the prompt.
  await h.button('a', 'discard').click(); assert.deepEqual(h.calls, [['POST', 'a', 'discard']]);
  assert.equal(shown(h.button('a', 'discard')), false); assert.match(h.row('a').textContent, /Waiting…/);
  h.engine.finishRow('a', 'discarded', 'Discarded'); await h.c.advance(POLL_MS);
  assert.equal(h.row('a'), undefined, 'a discarded row goes at once');
  await h.button('b', 'discard').click(); h.engine.finishRow('b', 'discarded', 'Discarded'); await h.c.advance(POLL_MS); await done;
  assert.equal(h.prompt.open, false);
});
test('a Save is sent once: its buttons go while the engine works on it, its stages show, and its end says what it saved, then it glides out', async () => {
  const h = setup([proseg('a'), proseg('b')]);
  h.prompt.ask(h.state());
  const pressed = h.button('a', 'add').click(); await h.button('a', 'add').click(); await pressed;
  assert.deepEqual(h.calls, [['POST', 'a', 'add']], 'a second press sends nothing');
  assert.equal(shown(h.button('a', 'add')), false); assert.match(h.row('a').textContent, /Waiting…/);
  h.engine.work('a', 'checking', 'Looking for it in the sample…'); await h.c.advance(POLL_MS); assert.match(h.row('a').textContent, /Looking for it in the sample…/);
  h.engine.work('a', 'writing', 'Saving to the sample…'); await h.c.advance(POLL_MS); assert.match(h.row('a').textContent, /Saving to the sample…/);
  h.engine.finishRow('a', 'saved', 'Saved as proseg_boundaries_2 + proseg_table_2', { state: 'saved', shapes_key: 'proseg_boundaries_2', table_key: 'proseg_table_2' });
  await h.prompt.poll();
  assert.match(h.row('a').textContent, /Saved as proseg_boundaries_2 \+ proseg_table_2/, 'it says so a moment before it goes');
  // Then it glides out: a few paints (at most GLIDE_FPS a second), drawn on demand, and nothing after.
  await h.c.advance(SAVED_MS); assert.equal(h.row('a').style.opacity, '0.75'); assert.equal(h.row('a').style.overflow, 'hidden');
  const paints = Math.ceil(GLIDE_MS * GLIDE_FPS / 1000);
  await h.c.advance(Math.ceil(1000 / GLIDE_FPS) * (paints - 1)); assert.equal(h.row('a').style.opacity, '0');
  await h.c.advance(Math.ceil(1000 / GLIDE_FPS)); assert.equal(h.row('a'), undefined);
  assert.equal(h.$('older-title').textContent, 'Save your older run?', 'one row left');
  assert.equal(h.c.timers.length, 1, 'once it has gone only the open prompt\'s poll waits: no paint');
  assert.equal(h.prompt.open, true, 'the other row is still to answer');
  // Saved already (a layer of the sample made from this run): said, and no second copy.
  await h.button('b', 'add').click(); h.engine.finishRow('b', 'saved_already', 'Saved already as proseg_boundaries'); await h.prompt.poll();
  assert.match(h.row('b').textContent, /Saved already as proseg_boundaries/);
  assert.equal(h.prompt.open, true, 'the last row says what it saved before the prompt closes');
  await h.c.advance(SAVED_MS + Math.ceil(1000 / GLIDE_FPS) * paints); assert.equal(h.prompt.open, false, 'then it glides out and the prompt closes');
  // With reduced motion a saved row goes at once, with no glide.
  const r = setup([proseg('a'), proseg('b')], { reducedMotion: () => true }); r.prompt.ask(r.state());
  await r.button('a', 'add').click(); r.engine.finishRow('a', 'saved', 'Saved as proseg_boundaries_2 + proseg_table_2'); await r.prompt.poll();
  assert.match(r.row('a').textContent, /Saved as proseg_boundaries_2/);
  await r.c.advance(SAVED_MS); assert.equal(r.row('a'), undefined);
});
test('a save that fails stays listed with its reason and Try again; Discard stays open', async () => {
  const h = setup([proseg('a')]);
  h.prompt.ask(h.state());
  await h.button('a', 'add').click(); h.engine.fail('a', "Couldn't save: SMP02 is being saved by another process."); await h.c.advance(POLL_MS);
  assert.match(h.row('a').textContent, /Couldn't save: SMP02 is being saved by another process\./);
  assert.equal(h.button('a', 'add').textContent, 'Try again'); assert.equal(shown(h.button('a', 'discard')), true);
  await h.button('a', 'add').click(); assert.deepEqual(h.calls.filter(c => c[0] === 'POST'), [['POST', 'a', 'add'], ['POST', 'a', 'add']], 'Try again sends it again');
  // A refusal of the answer itself (409): on the row, with Try again.
  const g = setup([proseg('b')], { refuse: () => Object.assign(new Error('It is being saved already: wait for it.'), { status: 409 }) });
  g.prompt.ask(g.state()); await g.button('b', 'add').click();
  assert.match(g.row('b').textContent, /Couldn't save: It is being saved already: wait for it\./); assert.equal(g.button('b', 'add').textContent, 'Try again');
  // A save that failed in an earlier version: listed with its reason.
  const f = setup([proseg('c', { zarr_save: { state: 'failed', message: 'No space left' } })]); f.prompt.ask(f.state());
  assert.match(f.row('c').textContent, /Couldn't save: No space left/); assert.equal(f.button('c', 'add').textContent, 'Try again');
});
test('a row that cannot be saved offers Discard only, saying why; one still being saved by the last session has no buttons', () => {
  const h = setup([proseg('a', { can_add: false, why: 'xenium', reason: 'Made on a Xenium folder, which cannot hold its cells.',
    detail: 'Its results are in /var/runs/a/output: copy them before you discard it.' }), proseg('b', { can_add: false, why: 'still_saving', still_saving: true })]);
  h.prompt.ask(h.state());
  assert.equal(shown(h.button('a', 'add')), false); assert.equal(shown(h.button('a', 'discard')), true);
  const why = h.row('a').all().find(n => n.className?.includes('older-why'));
  assert.equal(why.textContent, OLDER_TEXT.xenium); assert.equal(why.title, 'Its results are in /var/runs/a/output: copy them before you discard it.');
  assert.equal(shown(h.button('b', 'add')), false); assert.equal(shown(h.button('b', 'discard')), false);
  assert.match(h.row('b').textContent, /Still being saved by the last session/);
  assert.equal(shown(h.button(null, 'all-add')), false, 'Save all: nothing it could save');
  h.button(null, 'all-add').click(); assert.deepEqual(h.calls, []);
});
test('Save all and Discard all are one request each; an engine without it is answered row by row', async () => {
  const h = setup([proseg('a'), proseg('b'), proseg('c', { can_add: false, why: 'xenium' })]);
  h.prompt.ask(h.state());
  await h.button(null, 'all-add').click();
  assert.deepEqual(h.calls, [['POST', null, 'add']]);
  assert.equal(shown(h.button('a', 'add')), false); assert.equal(shown(h.button('c', 'discard')), true, 'the row it skipped can still be discarded');
  assert.equal(shown(h.button(null, 'all-add')), false, 'nothing left for Save all');
  const calls = [];
  const g = setup([proseg('a'), proseg('b')], { answerAll: async () => { calls.push('all'); throw Object.assign(new Error('Not Found'), { status: 404 }); } });
  g.prompt.ask(g.state()); await g.button(null, 'all-discard').click();
  assert.deepEqual(calls, ['all']); assert.deepEqual(g.calls, [['POST', 'a', 'discard'], ['POST', 'b', 'discard']]);
});
test('Later and Esc: the prompt goes at once and the engine sets the rows aside (one request); a save under way goes on and its end is a notice', async () => {
  const save = { state: 'saved', shapes_key: 'proseg_boundaries_2', table_key: 'proseg_table_2' };
  const h = setup([proseg('a'), proseg('b'), proseg('c')]);
  const done = h.prompt.ask(h.state());
  await h.button(null, 'all-add').click(); h.engine.work('a', 'writing', 'Saving to the sample…'); await h.c.advance(POLL_MS);
  h.button(null, 'later').click(); await done; await settle(); await settle();
  assert.equal(h.prompt.open, false);
  assert.deepEqual(h.calls.filter(c => c[0] === 'POST'), [['POST', null, 'add'], ['POST', null, 'later']], 'Later is one request');
  assert.deepEqual(h.engine.runs.map(r => [r.id, r.later, r.answer]), [['a', true, 'add'], ['b', true, null], ['c', true, null]],
    'the Save all stops after the row under way; the rest wait for the next start');
  h.engine.finishRow('a', 'saved', 'Saved as proseg_boundaries_2 + proseg_table_2', save);
  await h.c.advance(POLL_MS);
  assert.deepEqual(h.notices, ['Saved the older run on SMP02 as proseg_boundaries_2 + proseg_table_2.'], 'the rows set aside before they began say nothing');
  const polls = h.calls.length; await h.c.advance(POLL_MS * 4); assert.equal(h.calls.length, polls, 'nothing asked once it ended');
  // An engine that refuses Later: the prompt goes all the same, and the save under way is followed to its end.
  const g = setup([proseg('a'), proseg('b')], { answerAll: async answer => { g.calls.push(['POST', null, answer]);
    throw Object.assign(new Error('Choose Save to the sample or Discard'), { status: 422 }); } });
  const gone = g.prompt.ask(g.state());
  await g.button('a', 'add').click(); g.engine.work('a', 'writing', 'Saving to the sample…'); await g.c.advance(POLL_MS);
  g.button(null, 'later').click(); await gone; await settle(); await settle();
  assert.equal(g.prompt.open, false); assert.deepEqual(g.calls.filter(c => c[0] === 'POST').at(-1), ['POST', null, 'later']);
  g.engine.finishRow('a', 'saved', 'Saved as proseg_boundaries_2 + proseg_table_2', save); await g.c.advance(POLL_MS);
  assert.deepEqual(g.notices, ['Saved the older run on SMP02 as proseg_boundaries_2 + proseg_table_2.']);
  // A failed save after Esc: said, and listed again next time.
  const f = setup([proseg('a')]); const ended = f.prompt.ask(f.state());
  await f.button('a', 'add').click(); f.engine.work('a', 'writing', 'Saving to the sample…'); await f.c.advance(POLL_MS);
  f.prompt.dialog.emit('cancel'); await ended; await settle(); await settle();
  f.engine.fail('a', "Couldn't save: Disk full"); await f.c.advance(POLL_MS);
  assert.deepEqual(f.notices, ["Couldn't save the older run on SMP02: Disk full. It is listed again next time."]);
});
test('Later in another window: the rows it set aside leave this prompt quietly and it closes; a save under way here still ends in a notice', async () => {
  const h = setup([proseg('a'), proseg('b')]);
  const done = h.prompt.ask(h.state());
  await h.button('a', 'add').click(); h.engine.work('a', 'writing', 'Saving to the sample…'); await h.c.advance(POLL_MS);
  h.engine.runs = h.engine.runs.map(r => ({ ...r, later: true }));      // the other window's Later, as the engine pushes it
  h.prompt.event({ ev: 'older', ...h.state() }); await done;
  assert.equal(h.prompt.open, false); assert.equal(h.row('a'), undefined); assert.equal(h.row('b'), undefined);
  assert.deepEqual(h.notices, [], 'nothing is said saved before it is');
  h.engine.finishRow('a', 'saved', 'Saved as proseg_boundaries_2 + proseg_table_2', { state: 'saved', shapes_key: 'proseg_boundaries_2', table_key: 'proseg_table_2' });
  await h.c.advance(POLL_MS);
  assert.deepEqual(h.notices, ['Saved the older run on SMP02 as proseg_boundaries_2 + proseg_table_2.']);
  assert.deepEqual(olderRows(h.state()), [], 'rows set aside are not to answer');
});
test('rows another window answered go; the pushes keep both windows the same; forty rows keep the title and buttons in view', async () => {
  const rows = Array.from({ length: 40 }, (_, i) => proseg(`r${i}`, { created: local(10, 1 + (i % 28), 9, i) }));
  const h = setup(rows); const done = h.prompt.ask(h.state());
  const list = h.$('older-runs');
  assert.equal(list.children.length, 40); assert.ok(!list.contains(h.$('older-title')) && !list.contains(h.button(null, 'all-add')),
    'the rows scroll in their own box (styles: .older-runs), the title and buttons outside it');
  h.prompt.event({ ev: 'older', runs: rows.slice(1), done: [{ ...rows[0], stage: 'discarded', message: 'Discarded' }] });
  assert.equal(h.row('r0'), undefined); assert.equal(list.children.length, 39);
  // An answer the engine no longer knows (404): the row was answered elsewhere.
  const g = setup([proseg('a'), proseg('b')]);
  g.prompt.ask(g.state()); g.engine.runs = g.engine.runs.filter(r => r.id !== 'a'); await g.button('a', 'discard').click();
  assert.equal(g.row('a'), undefined); assert.equal(g.prompt.open, true);
  // A Save the engine no longer knows (another window discarded the row before this one heard): the row goes quietly,
  // never "Saved"; the last one gone, the prompt closes at once.
  g.engine.runs = []; await g.button('b', 'add').click();
  assert.equal(g.row('b'), undefined, 'no "Saved" for a row this window never saw saved'); assert.equal(g.prompt.open, false);
  h.prompt.event({ ev: 'older', runs: [], done: [] }); await done; assert.equal(h.prompt.open, false);
});
test('the last row saved says what it saved before the prompt closes (then the page goes on); Later meanwhile closes it at once', async () => {
  const save = { state: 'saved', shapes_key: 'proseg_boundaries_2', table_key: 'proseg_table_2' };
  const h = setup([proseg('a')]);
  let resolved = false; const done = h.prompt.ask(h.state()).then(() => { resolved = true; });
  await h.button('a', 'add').click(); h.engine.work('a', 'writing', 'Saving to the sample…'); await h.c.advance(POLL_MS);
  h.engine.finishRow('a', 'saved', 'Saved as proseg_boundaries_2 + proseg_table_2', save); await h.c.advance(POLL_MS);
  assert.equal(h.prompt.open, true); assert.equal(resolved, false, 'the page waits while the row says so');
  assert.match(h.row('a').textContent, /Saved as proseg_boundaries_2 \+ proseg_table_2/);
  assert.equal(h.$('older-title').textContent, 'Save your older run?', 'the title stays while it says so');
  assert.equal(h.button(null, 'later').hidden, false, 'Later still closes it at once');
  await h.c.advance(SAVED_MS); assert.equal(h.prompt.open, true, 'gliding out');
  await h.c.advance(GLIDE_MS + Math.ceil(1000 / GLIDE_FPS)); await done;
  assert.equal(h.prompt.open, false); assert.equal(h.row('a'), undefined); assert.deepEqual(h.notices, [], 'seen in the prompt: no notice as well');
  // With reduced motion: the line, then the prompt closes (no glide).
  const r = setup([proseg('a')], { reducedMotion: () => true }); const ended = r.prompt.ask(r.state());
  await r.button('a', 'add').click(); r.engine.finishRow('a', 'saved', 'Saved as x', { state: 'saved', shapes_key: 'x' }); await r.prompt.poll();
  assert.equal(r.prompt.open, true); await r.c.advance(SAVED_MS); await ended; assert.equal(r.prompt.open, false);
  // Later while the last row says so: the prompt goes at once.
  const l = setup([proseg('a')]); const later = l.prompt.ask(l.state());
  await l.button('a', 'add').click(); l.engine.finishRow('a', 'saved', 'Saved as x', { state: 'saved', shapes_key: 'x' }); await l.prompt.poll();
  l.button(null, 'later').click(); await later; assert.equal(l.prompt.open, false); assert.equal(l.row('a'), undefined);
  await l.c.advance(SAVED_MS + GLIDE_MS + 1000); assert.equal(l.prompt.open, false, 'its timers change nothing once it is closed');
});
