import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { STAIN_TEXT } from '../src/stain.mjs';
import * as proceed from '../src/proceed.mjs';
import * as saveLogic from '../src/zarr-save-state.mjs';
import * as session from '../src/session.mjs';
import { saveFields, validateSaveNames, saveModalState, shouldPromptSave, saveCompleted, saveOutdated, saveable } from '../src/zarr-save-state.mjs';
import { EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

const run = { region:{kind:'full'},progress:{complete:true}, id:'r', state:'done', sample_kind:'zarr', zarr_save:null,
  save_defaults:{shapes_key:'proseg_cells',table_key:'proseg_counts',existing:['cells','table']} };
test('save defaults include sopa default segmentation and require a separate overwrite decision', () => {
  assert.deepEqual(saveFields(run), {shapes_key:'proseg_cells',table_key:'proseg_counts',set_default:true,overwrite:false});
  assert.equal(saveModalState(run,saveFields(run)).canSave,true);
  const resave = {...run,zarr_save:{state:'saved',shapes_key:'custom',table_key:'counts',set_default:false}};
  assert.equal(saveFields(resave).set_default,false);
  assert.equal(saveFields(resave).overwrite,false);
  assert.deepEqual(saveModalState(resave,saveFields(resave)).conflicts,['custom','counts']);
  assert.equal(saveModalState(resave,saveFields(resave)).canSave,false);
});
test('save names allow contract characters, reject paths, whitespace and empty or matching names', () => {
  assert.equal(validateSaveNames({shapes_key:'cells.1-v2_A',table_key:'table-2'}).valid,true);
  for (const name of ['', 'has spaces','foo/bar','../cells','x\\y','<script>','名字','x\n']) {
    assert.equal(validateSaveNames({shapes_key:name,table_key:'table'}).valid,false,name);
  }
  assert.match(validateSaveNames({shapes_key:'same',table_key:'same'}).errors.table_key,/different/);
});
test('either conflicting name requires overwrite and malformed names remain invalid after consent', () => {
  for (const fields of [{shapes_key:'cells',table_key:'new'}, {shapes_key:'new',table_key:'table'}]) {
    const validation=validateSaveNames(fields,run.save_defaults.existing);
    assert.equal(validation.conflicts.length,1); assert.equal(validation.valid,false);
    assert.equal(validateSaveNames({...fields,overwrite:true},run.save_defaults.existing).valid,true);
  }
  assert.equal(validateSaveNames({shapes_key:'bad/name',table_key:'table',overwrite:true},['table']).valid,false);
});
test('a name differing from one in the sample only by case is a field error; Replace is offered for exact names only', () => {
  // The engine refuses a case twin whatever Replace says (Runs.save_to_zarr), even beside the exact name.
  const twin = validateSaveNames({ shapes_key: 'cells', table_key: 'counts', overwrite: true }, ['Cells']);
  assert.match(twin.errors.shapes_key, /^Cells is already in the sample: use a name that differs by more than case\.$/);
  assert.deepEqual(twin.conflicts, []); assert.equal(twin.valid, false, 'never valid, ticked or not');
  const exact = validateSaveNames({ shapes_key: 'Cells', table_key: 'counts' }, ['Cells']);
  assert.deepEqual(exact.errors, {}); assert.deepEqual(exact.conflicts, ['Cells']); assert.equal(exact.valid, false);
  assert.equal(validateSaveNames({ shapes_key: 'Cells', table_key: 'counts', overwrite: true }, ['Cells']).valid, true);
  const both = validateSaveNames({ shapes_key: 'cells', table_key: 'counts', overwrite: true }, ['cells', 'Cells']);
  assert.match(both.errors.shapes_key, /^Cells is already/); assert.equal(both.valid, false);
  const table = validateSaveNames({ shapes_key: 'shapes', table_key: 'TABLE', overwrite: true }, ['table']);
  assert.match(table.errors.table_key, /^table is already/); assert.deepEqual(table.conflicts, []); assert.equal(table.valid, false);
  const stain = validateSaveNames({ shapes_key: 'Cellpose_1', table_key: null, overwrite: true }, ['cellpose_1'], true);
  assert.match(stain.errors.shapes_key, /^cellpose_1 is already/); assert.equal(stain.valid, false);
  assert.deepEqual(bannedWords(twin.errors.shapes_key), []);
});
test('modal disables submission throughout the HTTP request and asynchronous write, then permits failure retry', () => {
  const fields=saveFields(run);
  assert.equal(saveModalState(run,fields,{submitting:true}).canSave,false);
  const saving=saveModalState({...run,zarr_save:{state:'saving',message:'Writing AnnData'}},fields);
  assert.equal(saving.saving,true); assert.equal(saving.canSave,false); assert.equal(saving.message,'Writing AnnData');
  const failed=saveModalState({...run,zarr_save:{state:'failed',message:'Zarr is read-only'}},fields);
  assert.equal(failed.canSave,true); assert.equal(failed.error,'Zarr is read-only');
  assert.equal(saveModalState(run,fields,{error:'HTTP 409: layer exists'}).error,'HTTP 409: layer exists');
  assert.equal(saveModalState({...run,state:'finishing'},fields).canSave,false);
  assert.equal(saveModalState({...run,sample_kind:'xenium'},fields).canSave,false);
});
test('automatic save prompt is only for this session’s Finish & save on an unsaved done zarr run', () => {
  assert.equal(shouldPromptSave(run,true),true);
  assert.equal(shouldPromptSave(run,false),false);
  for(const state of ['preparing','running','paused','finishing','failed','stopped']) assert.equal(shouldPromptSave({...run,state},true),false);
  for(const state of ['saving','saved','failed']) assert.equal(shouldPromptSave({...run,zarr_save:{state}},true),false);
  assert.equal(shouldPromptSave({...run,sample_kind:'xenium'},true),false);
});
test('resave ignores a stale saved event while its new POST is in flight', () => {
  const old={state:'saved',shapes_key:'cells',table_key:'table',at:'before'};
  assert.equal(saveCompleted({...old},old),false);
  assert.equal(saveCompleted({at:'before',table_key:'table',shapes_key:'cells',state:'saved'},old),false);
  assert.equal(saveCompleted({...old,at:'after'},old),true);
  assert.equal(saveCompleted({...old,exclusions_version:1},{...old,exclusions_version:0}),true);
  assert.equal(saveCompleted(old,null),true);
  assert.equal(saveCompleted({state:'failed',message:'Cannot write'},old),true);
  assert.equal(saveCompleted({state:'saving'},null),false);
});

test('saved stain results track both revisions and removed-area versions with legacy zero defaults',()=>{
  const saved={...run,kind:'stain',progress:{revision:4},zarr_save:{state:'saved',result_revision:4}};
  assert.equal(saveOutdated(saved),false);
  assert.equal(saveOutdated({...saved,exclusions:{version:0}}),false);
  assert.equal(saveOutdated({...saved,exclusions:{version:1}}),true);
  const current={...saved,exclusions:{version:1},zarr_save:{...saved.zarr_save,exclusions_version:1}};
  assert.equal(saveOutdated(current),false);
  assert.equal(saveOutdated({...current,exclusions:{version:2,count:0,removed:0}}),true,'undo also changes the saved result');
  assert.equal(saveOutdated({...current,progress:{revision:5}}),true);
  assert.equal(saveOutdated({...current,exclusions:undefined}),true);
  assert.equal(saveOutdated({...saved,exclusions:{version:1},zarr_save:{state:'saved'}}),true);
  for(const state of ['saving','failed'])assert.equal(saveOutdated({...saved,exclusions:{version:1},zarr_save:{state}}),false);
  assert.equal(saveOutdated({...saved,kind:'proseg',exclusions:{version:1}}),false);
  assert.equal(saveOutdated({...saved,zarr_save:null}),false);
  assert.equal(saveOutdated({...saved,zarr_save:{state:'outdated'}}),true);
});

test('results repaint the saved-shapes warning on removal and undo without a new revision',()=>{
  const dom=domFixture();dom.document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const source=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const saved={...run,kind:'stain',progress:{revision:4,resolved:true},exclusions:{version:0},
    zarr_save:{state:'saved',result_revision:4,exclusions_version:0,shapes_key:'cells'}};
  const context=vm.createContext({...proceed,...session,stainExclusions:{pending:false},exclusionsPending:()=>false,$:dom.$,document:dom.document,STAIN_TEXT,saveOutdated,saveable,shown:{},run:saved,
    state:{session:null},setProp:(node,key,value)=>node[key]=value,setText:(node,value)=>node.textContent=value});
  vm.runInContext(source.slice(source.indexOf('function renderResults('),source.indexOf('// ---- the mixing check')),context);
  const render=()=>{vm.runInContext('renderResults(run)',context);return dom.$('results').textContent;};
  assert.match(render(),/Saved as cells/);
  saved.exclusions={version:1};assert.ok(render().includes(STAIN_TEXT.savedOutdated));
  saved.zarr_save={...saved.zarr_save,exclusions_version:1};assert.match(render(),/Saved as cells/);
  saved.exclusions={version:2,count:0,removed:0};assert.ok(render().includes(STAIN_TEXT.savedOutdated));
  const unchanged=dom.$('results').firstChild;render();assert.equal(dom.$('results').firstChild,unchanged);
  // A notebook's session: the same card in the words of your SpatialData.
  context.state.session=session.apiSession({session:{mode:'api',id:'1f3a9c2e',sample:{id:'z1',name:'SMP02',backed:true,zarr_name:'SMP02.zarr',view:'in_place',
    state:'ready',progress:{done:0,total:0},message:null,memory_only:[],tools:['proseg','stain']},tool:null,writes:'zarr',notebook:{busy:false},patches:false,link:false,state:'running'}});
  const button=()=>dom.$('results').all().find(n=>n.tagName==='BUTTON').textContent;
  // On its way: never "Added" before the notebook has it; the line under it says where it is meanwhile.
  saved.zarr_save={state:'queued',written_to:'/data/SMP02.zarr',result_revision:4,exclusions_version:2,shapes_key:'cells'};saved.exclusions={version:2};
  context.shown={};let text=render();
  assert.match(text,/Adding to your SpatialData as cells/);assert.doesNotMatch(text,/Added/);
  assert.match(text,/Written to SMP02\.zarr · in your notebook as soon as it is free/);
  assert.equal(button(),'Add as shapes');
  context.state.session={...context.state.session,notebook:{busy:true}};context.shown={};
  assert.match(render(),/Waiting for your notebook: a cell is running/);
  saved.zarr_save={state:'queued',written_to:null,result_revision:4,exclusions_version:2,shapes_key:'cells'};context.shown={};
  context.state.session={...context.state.session,notebook:{busy:false}};
  text=render();assert.match(text,/Adding to your SpatialData as cells/);assert.match(text,/Ready · in your notebook as soon as it is free/);
  saved.zarr_save={state:'saved',added:true,cells:20,shapes_key:'cells',table_key:null,written_to:'/data/SMP02.zarr',result_revision:4,exclusions_version:2};
  context.shown={};text=render();assert.match(text,/Added to your SpatialData as cells/);assert.doesNotMatch(text,/sample|Saved as|Added 20/);
  // While it is being added the button opens the dialog's progress, in the session's words.
  saved.zarr_save={state:'saving',message:'Writing cell shapes…',result_revision:4,exclusions_version:2};context.shown={};
  text=render();assert.equal(button(),'View progress…');assert.doesNotMatch(text,/save progress|Save as/);
});

// ---- the Save dialog (zarr-save.js) against the DOM fixture. Its imports come from the context; explain.mjs is
// evaluated in the same context, so the document it writes notes into is the fixture's.
const moduleCode = file => readFileSync(new URL(`../src/${file}`, import.meta.url), 'utf8').replace(/^import .*;\n/gm, '').replace(/^export /gm, '');
function saver() {
  const dom = domFixture(), notices = [], posts = [], saved = [], asks = [], cancels = [];
  let request = deferred(), controller, answer = null;
  // The dialog asks for its run as it opens (GET /runs/{id}: the save defaults against the sample's names now); the
  // engine answers when a test gives an answer (answerWith), else never.
  const api = { saveZarr: (id, values) => { posts.push({ id, values }); return request.promise; },
    run: id => { asks.push(id); return answer ? answer.promise : new Promise(() => {}); } };
  const context = vm.createContext({ ...saveLogic, ...session, STAIN_TEXT, document: dom.document, api, queueMicrotask, Intl });
  const Save = vm.runInContext(moduleCode('explain.mjs') + moduleCode('zarr-save.js') + '\nZarrSave', context);
  controller = new Save(run => controller.update(run), message => notices.push(message), run => saved.push(run), null, run => cancels.push(run.id));
  const run = { region:{kind:'full'}, id: 'run', sample_kind: 'zarr', sample_name: 'Tissue', state: 'done', zarr_save: null,
    save_defaults: { shapes_key: 'cells', table_key: 'counts', existing: [] } };
  const submit = dom.$('zarr-submit'), note = dom.$('zarr-note');
  return { ...dom, controller, notices, posts, saved, run, submit, note, asks, cancels, get request() { return request; }, retry() { request = deferred(); },
    answerWith() { answer = deferred(); return answer; } };
}
const busyNow = button => button.getAttribute('aria-busy') === 'true';

test('save automatic prompt is once per finish, waits for other dialogs, and progresses through failure and retry', async () => {
  const h = saver(), blocker = h.document.createElement('dialog'); h.document.body.append(blocker); blocker.showModal();
  h.controller.trackFinish(h.run.id); h.controller.update(h.run); assert.equal(h.controller.dialog.open, false);
  blocker.close(); await settle(); assert.equal(h.controller.dialog.open, true);
  h.controller.dialog.close(); h.controller.update(h.run); assert.equal(h.controller.dialog.open, false);
  h.controller.open(h.run);
  const first = h.submit.click();
  assert.equal(busyNow(h.submit), true); assert.equal(h.submit.textContent, 'Saving…'); assert.equal(h.submit.disabled, false);
  h.request.resolve({ ...h.run, zarr_save: { state: 'saving', message: 'Writing table' } }); await first;
  assert.equal(busyNow(h.submit), true); assert.match(h.$('zarr-progress').textContent, /Writing table/);
  h.controller.update({ ...h.run, zarr_save: { state: 'failed', message: 'Read-only store' } });
  assert.equal(busyNow(h.submit), false); assert.equal(h.submit.textContent, 'Save to zarr');
  assert.equal(h.$('zarr-error').textContent, 'Read-only store');
  h.retry(); const retry = h.submit.click();
  const saved = { ...h.run, zarr_save: { state: 'saved', shapes_key: 'cells', table_key: 'counts', cells: 1234, at: 'new' } };
  h.controller.update(saved); assert.equal(h.controller.dialog.open, false); assert.match(h.notices[0], /Saved 1,234 cells to Tissue as cells \+ counts/);
  h.request.resolve({ ...h.run, zarr_save: { state: 'saving' } }); await retry;
  assert.equal(h.controller.run.zarr_save.state, 'saved', 'late HTTP response cannot overwrite completed event');
  h.controller.update(saved); assert.equal(h.notices.length, 1);
  assert.equal(h.posts.length, 2);
});

test('a save that ends saved tells the page which run wrote into its sample (Layers > Load Other Mask reads its list anew); a failed one does not', async () => {
  const h = saver(), run = { ...h.run, sample: 'z1' };
  h.controller.open(run);
  const first = h.submit.click();
  h.request.resolve({ ...run, zarr_save: { state: 'saving' } }); await first;
  assert.deepEqual(h.saved, []);
  h.controller.update({ ...run, zarr_save: { state: 'failed', message: 'Read-only store' } });
  assert.deepEqual(h.saved, [], 'a failed save wrote nothing');
  h.retry(); const again = h.submit.click();
  h.request.resolve({ ...run, zarr_save: { state: 'saving' } }); await again;
  const done = { ...run, zarr_save: { state: 'saved', shapes_key: 'cells', table_key: 'counts', cells: 12, at: 'now' } };
  h.controller.update(done);
  assert.deepEqual(h.saved.map(r => [r.id, r.sample]), [['run', 'z1']]);
  h.controller.update(done); assert.equal(h.saved.length, 1, 'once per save');
  // A save seen on its way in a run's push (started from another window) counts too.
  const other = { id: 'other', sample: 'z2', kind: 'stain', sample_kind: 'zarr', sample_name: 'Tissue 2', state: 'done', zarr_save: { state: 'saving' } };
  h.controller.update(other); h.controller.update({ ...other, zarr_save: { state: 'saved', shapes_key: 'cellpose_new', table_key: null, cells: 3, at: 'x' } });
  assert.deepEqual(h.saved.map(r => r.sample), ['z1', 'z2']);
});

test('save 409 is inline, overwrite consent resets when names change, and resave ignores old completion', async () => {
  const h = saver(); h.run.save_defaults.existing = ['cells']; h.controller.open(h.run);
  await h.submit.click();                                  // the layer exists: the press asks for consent, posts nothing
  assert.equal(h.posts.length, 0); assert.match(h.note.textContent, /Replace the existing layer: Tick it to replace cells/);
  h.$('zarr-overwrite').checked = true; h.$('zarr-form').oninput({ target: h.$('zarr-overwrite') });
  assert.equal(h.note.hidden, true, 'changing the form clears the note');
  const first = h.submit.click(); assert.equal(h.posts[0].values.overwrite, true);
  h.request.reject(Error('Layer exists (409)')); await first; assert.equal(h.$('zarr-error').textContent, 'Layer exists (409)');
  assert.equal(busyNow(h.submit), false);
  h.$('zarr-table').value = 'new_table'; h.$('zarr-form').oninput({ target: h.$('zarr-table') }); assert.equal(h.$('zarr-overwrite').checked, false);
  const old = { ...h.run, zarr_save: { state: 'saved', shapes_key: 'cells', table_key: 'counts', cells: 1234, at: 'old' } };
  h.controller.open(old); h.$('zarr-overwrite').checked = true; h.controller.render(); h.retry(); const resave = h.submit.click();
  h.controller.update({ ...old, zarr_save: { ...old.zarr_save } }); assert.equal(h.controller.dialog.open, true);
  assert.equal(busyNow(h.submit), true); assert.equal(h.notices.length, 0);
  h.request.resolve({ ...old, zarr_save: { state: 'saving', message: 'Replacing' } }); await resave;
  h.controller.update(old); assert.equal(h.controller.dialog.open, true); assert.equal(busyNow(h.submit), true);
  h.controller.update({ ...old, zarr_save: { ...old.zarr_save, at: 'new' } }); assert.equal(h.controller.dialog.open, false); assert.equal(h.notices.length, 1);
});

test('a stain run with no cells left: Save says there is nothing to save, and asks the engine nothing', async () => {
  const h = saver();
  Object.assign(h.run, { kind: 'stain', state: 'done', progress: { revision: 3, resolved: true, complete: true }, summary: { cells: 0 },
    save_defaults: { shapes_key: 'cellpose_boundaries_1', table_key: null, existing: [] } });
  h.controller.open(h.run);
  await h.submit.click();
  assert.match(h.note.textContent, /There are no cells to save: this run found none, or all were removed/);
  assert.equal(h.posts.length, 0); assert.equal(h.submit.disabled, false, 'never greyed out: the press says why');
  h.run.summary.cells = 3; h.controller.update({ ...h.run });
  const saving = h.submit.click(); assert.equal(h.posts.length, 1, 'cells to save: the press saves');
  h.request.resolve({ ...h.run, zarr_save: { state: 'saving' } }); await saving;
});

test('a resumed stain save releases the dialog even if its saved event was missed', async () => {
  const h = saver();
  Object.assign(h.run, { kind: 'stain', state: 'stopped', progress: { revision: 3, resolved: true, complete: false },
    save_defaults: { shapes_key: 'cells', table_key: null, existing: [] } });
  h.controller.open(h.run);
  assert.equal(h.submit.textContent, STAIN_TEXT.save);
  const posting = h.submit.click();
  const resumed = { ...h.run, state: 'running', progress: { revision: 4, resolved: false, complete: false },
    zarr_save: { state: 'outdated', shapes_key: 'cells', table_key: null, cells: 2, result_revision: 3 } };
  h.controller.update(resumed);
  assert.equal(h.controller.pending.has(h.run.id), false);
  assert.equal(h.controller.submitting, false);
  assert.equal(busyNow(h.submit), false); assert.equal(h.submit.textContent, STAIN_TEXT.save);
  await h.submit.click();                                  // running again: the press says what to do first
  assert.match(h.note.textContent, /Stop the run or let it finish first/); assert.equal(h.posts.length, 1);
  h.request.resolve({ ...h.run, zarr_save: { state: 'saving' } }); await posting;
  assert.equal(h.controller.run.zarr_save.state, 'outdated', 'a late save response must not restore the busy state');
  assert.equal(h.notices.length, 0);
  h.controller.update({ ...resumed, state: 'done', progress: { revision: 10, resolved: true, complete: true } });
  await h.submit.click();
  assert.equal(h.posts.length, 1, 'the existing shapes still require overwrite consent');
  assert.match(h.note.textContent, /Replace the existing layer: Tick it to replace cells, or use a different name/);
  h.$('zarr-overwrite').checked = true; h.controller.render();
  h.submit.click(); assert.equal(h.posts.length, 2); assert.equal(h.posts[1].values.overwrite, true);
});

test('Save is never greyed out: a press with names to fix lists each field, outlines it and moves to the first', async () => {
  const h = saver(); h.controller.open(h.run);
  const shapes = h.$('zarr-shapes'), table = h.$('zarr-table');
  let focused = null; shapes.focus = () => { focused = 'shapes'; };
  shapes.value = ''; table.value = 'my table'; h.$('zarr-form').oninput({ target: shapes });
  assert.equal(h.submit.disabled, false);
  assert.equal(h.$('zarr-shapes-error').textContent, 'Enter a name.', 'the per-field errors stay');
  assert.equal(shapes.getAttribute('aria-invalid'), 'true'); assert.equal(table.getAttribute('aria-invalid'), 'true');
  await h.submit.click();
  assert.equal(h.posts.length, 0); assert.equal(h.note.hidden, false);
  assert.match(h.note.textContent, /Cell shapes layer: Enter a name\./);
  assert.match(h.note.textContent, /Table: Use letters, digits, _ \. - only\./);
  assert.ok(shapes.classList.contains(NEEDS_CLASS) && table.classList.contains(NEEDS_CLASS)); assert.equal(focused, 'shapes');
  shapes.value = 'cells_2'; shapes.emit('input'); h.$('zarr-form').oninput({ target: shapes });    // typing: the browser's input event, then the form's
  assert.equal(h.note.hidden, true); assert.ok(!shapes.classList.contains(NEEDS_CLASS));
  assert.equal(shapes.getAttribute('aria-invalid'), 'false'); assert.equal(table.getAttribute('aria-invalid'), 'true');
  table.value = 'cells_2'; h.$('zarr-form').oninput({ target: table });
  await h.submit.click();
  assert.match(h.note.textContent, /Table: Use a different name for the table\./); assert.equal(h.posts.length, 0);
  table.value = 'counts_2'; h.$('zarr-form').oninput({ target: table });
  h.submit.click();
  assert.deepEqual({ ...h.posts[0].values }, { shapes_key: 'cells_2', table_key: 'counts_2', set_default: true, overwrite: false });
  assert.equal(shapes.disabled, true, 'inputs are disabled only while saving'); assert.equal(h.note.hidden, true);
  await h.submit.click();                                  // a press while saving: still working, nothing posted twice
  assert.ok(h.note.textContent.includes(EXPLAIN_TEXT.busy)); assert.equal(h.posts.length, 1);
  h.request.reject(Error('Disk full')); await settle();
  assert.equal(busyNow(h.submit), false); assert.equal(shapes.disabled, false);
  assert.equal(h.note.hidden, true, 'the save ended: the still-working note goes');
  assert.equal(h.$('zarr-error').textContent, 'Disk full');
});

test('the Save dialog words are plain', async () => {
  const h = saver(); h.run.save_defaults.existing = ['cells']; h.controller.open(h.run);
  h.$('zarr-shapes').value = ''; h.$('zarr-form').oninput({ target: h.$('zarr-shapes') });
  await h.submit.click();
  const words = [h.note.textContent];
  for (const kind of ['stain', 'proseg']) {                 // the dialog stays open while a run starts again
    h.controller.run = { ...h.run, kind, state: 'running' };
    words.push(...h.controller.problems().map(p => p.message));
  }
  assert.deepEqual(words.slice(1), ['Stop the run or let it finish first.', 'Let the run finish first.']);
  h.controller.open(h.run); h.$('zarr-overwrite').checked = true; h.controller.render(); h.submit.click();
  assert.equal(h.submit.textContent, 'Saving…'); words.push(h.submit.textContent);
  for (const text of words) assert.deepEqual(bannedWords(text), [], text);
});

// ---- names taken since the run ended , a stain save of part of the sample ----------------------------------------------
test('a stain save of a box or a lasso never starts as the sample\'s segmentation; a whole-sample one does, and the engine\'s own default wins', () => {
  const stain = (region, defaults = {}) => ({ ...run, kind: 'stain', region, save_defaults: { shapes_key: 'cellpose_boundaries_1', table_key: null, existing: [], ...defaults } });
  assert.equal(saveFields(stain({ kind: 'box', box: [0, 0, 100, 100] })).set_default, false, 'a 100 µm box is not the sample\'s cells');
  assert.equal(saveFields(stain({ kind: 'polygon', points: [[0, 0], [90, 0], [0, 90]] })).set_default, false);
  assert.equal(saveFields(stain({ kind: 'full' })).set_default, true);
  assert.equal(saveFields(stain({ kind: 'box' }, { set_default: true })).set_default, true, 'the engine knows the box covers the sample');
  assert.equal(saveFields(stain({ kind: 'full' }, { set_default: false })).set_default, false);
  assert.equal(saveFields({ ...run, region: { kind: 'box' } }).set_default, false, 'a proseg box is not the whole sample');
  const again = { ...stain({ kind: 'box' }), zarr_save: { state: 'saved', shapes_key: 'cellpose_boundaries_1', set_default: true } };
  assert.equal(saveFields(again).set_default, true, 'a run saved before keeps what was chosen then');
});

test('the Save dialog of a stain run of a box opens with "Make it the sample\'s segmentation" unticked, and posts it so', async () => {
  const h = saver();
  Object.assign(h.run, { kind: 'stain', state: 'done', region: { kind: 'box', box: [0, 0, 100, 100] }, progress: { resolved: true, complete: true },
    save_defaults: { shapes_key: 'cellpose_boundaries_1', table_key: null, existing: ['cells_polygons'] } });
  h.controller.open(h.run);
  assert.equal(h.$('zarr-default').checked, false);
  h.submit.click();
  assert.equal(h.posts[0].values.set_default, false);
});

test('the Save dialog asks for the sample\'s names as it opens: names taken since the run ended are not offered, and typing one offers Replace', async () => {
  const h = saver(), fresh = h.answerWith();
  Object.assign(h.run, { save_defaults: { shapes_key: 'proseg_boundaries', table_key: 'proseg_table', existing: ['cells_polygons'] } });
  h.controller.open(h.run);
  assert.deepEqual(h.asks, ['run'], 'asked once as it opens');
  assert.deepEqual([h.$('zarr-shapes').value, h.$('zarr-table').value], ['proseg_boundaries', 'proseg_table'], 'the names as the run ended, meanwhile');
  // Another run's save took them: the engine works the run's defaults out again.
  fresh.resolve({ ...h.run, save_defaults: { shapes_key: 'proseg_boundaries_2', table_key: 'proseg_table_2',
    existing: ['cells_polygons', 'proseg_boundaries', 'proseg_table'] } });
  await settle();
  assert.deepEqual([h.$('zarr-shapes').value, h.$('zarr-table').value], ['proseg_boundaries_2', 'proseg_table_2']);
  assert.equal(h.$('zarr-overwrite-row').hidden, true, 'nothing to replace');
  // The first run's layer, chosen by hand: Replace is offered, and a press asks for it.
  h.$('zarr-shapes').value = 'proseg_boundaries'; h.$('zarr-table').value = 'proseg_table'; h.$('zarr-form').oninput({ target: h.$('zarr-shapes') });
  assert.equal(h.$('zarr-overwrite-row').hidden, false);
  assert.match(h.$('zarr-warning').textContent, /Already in the sample: proseg_boundaries, proseg_table/);
  await h.submit.click();
  assert.equal(h.posts.length, 0); assert.match(h.note.textContent, /Replace the existing layer: Tick it to replace proseg_boundaries, proseg_table/);
  h.$('zarr-overwrite').checked = true; h.$('zarr-form').oninput({ target: h.$('zarr-overwrite') });
  h.submit.click();
  assert.deepEqual({ ...h.posts[0].values }, { shapes_key: 'proseg_boundaries', table_key: 'proseg_table', set_default: true, overwrite: true });
  // Names typed before the engine answers stay; its names still count for a clash.
  const later = saver(), slow = later.answerWith();
  later.controller.open(later.run);
  later.$('zarr-shapes').value = 'mine'; later.$('zarr-form').oninput({ target: later.$('zarr-shapes') });
  slow.resolve({ ...later.run, save_defaults: { shapes_key: 'cells_2', table_key: 'counts_2', existing: ['cells', 'counts'] } });
  await settle();
  assert.deepEqual([later.$('zarr-shapes').value, later.$('zarr-table').value], ['mine', 'counts_2'],
    'typed by hand: kept; the other field takes its new name (its stale one clashed)');
  assert.deepEqual([...later.controller.view().conflicts], []);
  // A stain run of a box, unticked by its default: the tick follows the engine's answer unless it was changed.
  const box = saver(), answer = box.answerWith();
  Object.assign(box.run, { kind: 'stain', region: { kind: 'box', box: [0, 0, 100, 100] }, progress: { resolved: true, complete: true },
    save_defaults: { shapes_key: 'cellpose_boundaries_1', table_key: null, existing: [] } });
  box.controller.open(box.run);
  answer.resolve({ ...box.run, save_defaults: { shapes_key: 'cellpose_boundaries_2', table_key: null, existing: ['cellpose_boundaries_1'] } });
  await settle();
  assert.deepEqual([box.$('zarr-shapes').value, box.$('zarr-default').checked], ['cellpose_boundaries_2', false]);
});

test('a name clash refused by the engine (409) offers Replace for the names it gives, and a save with Replace goes ahead', async () => {
  const h = saver();
  h.controller.open(h.run);                                  // the engine never answered the ask: the names as the run ended
  assert.equal(h.$('zarr-overwrite-row').hidden, true);
  const first = h.submit.click();
  h.request.reject(Object.assign(new Error('cells and counts already exist in the sample: choose other names or allow replacing'),
    { status: 409, data: { error: 'cells and counts already exist in the sample', clash: ['cells', 'counts'], existing: ['cells', 'counts', 'table'] } }));
  await first; await settle();
  assert.equal(h.$('zarr-overwrite-row').hidden, false, 'Replace is offered');
  assert.match(h.$('zarr-warning').textContent, /Already in the sample: cells, counts/);
  assert.equal(h.$('zarr-error').hidden, true, 'the dialog says it in its own words');
  assert.match(h.note.textContent, /Replace the existing layer: Tick it to replace cells, counts, or use a different name/);
  assert.equal(busyNow(h.submit), false);
  // Another name that the answer listed as taken counts too.
  h.$('zarr-table').value = 'table'; h.$('zarr-form').oninput({ target: h.$('zarr-table') });
  assert.deepEqual([...h.controller.view().conflicts], ['cells', 'table']);
  h.$('zarr-table').value = 'counts'; h.$('zarr-form').oninput({ target: h.$('zarr-table') });
  h.$('zarr-overwrite').checked = true; h.$('zarr-form').oninput({ target: h.$('zarr-overwrite') });
  h.retry(); h.submit.click();
  assert.equal(h.posts.length, 2); assert.equal(h.posts[1].values.overwrite, true);
  // A refusal without names (any other 409) is shown as the engine says it.
  const other = saver(); other.controller.open(other.run);
  const press = other.submit.click();
  other.request.reject(Object.assign(new Error('This run is already being saved'), { status: 409, data: { detail: 'This run is already being saved', status: 409 } }));
  await press;
  assert.equal(other.$('zarr-error').textContent, 'This run is already being saved'); assert.equal(other.$('zarr-overwrite-row').hidden, true);
});

test('range digests make Undo to a saved state current again, including queued API saves',()=>{
  const saved={...run,kind:'stain',progress:{revision:4},exclusions:{version:5,digest:'a'},
    zarr_save:{state:'saved',result_revision:4,exclusions_version:1,exclusions_digest:'a'}};
  assert.equal(saveOutdated(saved),false);
  assert.equal(saveOutdated({...saved,exclusions:{version:6,digest:'b'}}),true);
  assert.equal(saveOutdated({...saved,exclusions:{version:7,digest:'a'}}),false);
  assert.equal(saveOutdated({...saved,zarr_save:{...saved.zarr_save,state:'queued'}}),false);
  assert.equal(saveCompleted({...saved.zarr_save,exclusions_digest:'b'},saved.zarr_save),true);
});
test('Save waits for pending filter changes and checks the kept count again before posting',async()=>{
  const h=saver(),gate=deferred();h.controller.whenSettled=()=>gate.promise;
  h.run={...h.run,kind:'stain',progress:{resolved:true,complete:true},summary:{cells:4}};
  h.controller.open(h.run);const press=h.submit.click();assert.equal(h.posts.length,0);
  h.controller.update({...h.run,summary:{cells:0}});gate.resolve();await press;
  assert.equal(h.posts.length,0);assert.match(h.note.textContent,/no cells to save/);
});

test('Save waits for a pending Undo that brings filtered cells back',async()=>{
  const h=saver(),gate=deferred();h.controller.whenSettled=()=>gate.promise;
  h.run={...h.run,kind:'stain',progress:{resolved:true,complete:true},summary:{cells:0}};
  h.controller.open(h.run);const press=h.submit.click();assert.equal(h.posts.length,0);
  h.controller.update({...h.run,summary:{cells:4}});gate.resolve();await settle();assert.equal(h.posts.length,1);
  h.request.resolve({...h.run,zarr_save:{state:'saving'}});await press;
});
test('closing and reopening the Save dialog cancels a press still waiting for filters',async()=>{
  const h=saver(),gate=deferred();h.controller.whenSettled=()=>gate.promise;
  h.controller.open(h.run);const press=h.submit.click();h.controller.dialog.close();h.controller.open(h.run);
  gate.resolve();await press;assert.equal(h.posts.length,0);
});
test('the dialog: a case twin names its field and offers no Replace; an exact name offers Replace', async () => {
  const h = saver(); h.controller.open({ ...h.run, save_defaults: { shapes_key: 'cells', table_key: 'counts', existing: ['Cells'] } });
  assert.equal(h.$('zarr-overwrite-row').hidden, true); assert.equal(h.$('zarr-warning').hidden, true);
  assert.match(h.$('zarr-shapes-error').textContent, /Cells is already in the sample/);
  await h.submit.click(); assert.deepEqual(h.posts, [], 'nothing is sent the engine would refuse');
  assert.ok(h.$('zarr-shapes').classList.contains(NEEDS_CLASS));
  h.$('zarr-shapes').value = 'Cells'; h.$('zarr-form').emit('input', { target: h.$('zarr-shapes') });
  assert.equal(h.$('zarr-shapes-error').textContent, ''); assert.equal(h.$('zarr-overwrite-row').hidden, false);
  assert.match(h.$('zarr-warning').textContent, /Already in the sample: Cells/);
});
test('in a notebook\'s session the dialog adds to your SpatialData: its words, a clash, Adding…, and where the result is once sent', async () => {
  const h = saver();
  h.controller.setSession(session.apiSession({ session: { mode: 'api', id: '1f3a9c2e', sample: { id: 'z1', name: 'SMP02', backed: true, zarr_name: 'SMP02.zarr',
    view: 'in_place', state: 'ready', progress: { done: 0, total: 0 }, message: null, memory_only: [], tools: ['proseg'] }, tool: null, writes: 'zarr',
    notebook: { busy: false }, patches: false, link: false, state: 'running' } }));
  h.controller.open({ ...h.run, save_defaults: { shapes_key: 'proseg_boundaries', table_key: 'proseg_table', existing: ['proseg_boundaries'] } });
  assert.equal(h.$('zarr-title').textContent, 'Add to your SpatialData?'); assert.equal(h.submit.textContent, 'Add');
  assert.equal(h.$('zarr-default-text').textContent, 'Make it the default segmentation (as sopa does)');
  assert.equal(h.$('zarr-help').textContent, "Shapes and a table of proseg's counts, added as sopa's own functions add them. Also written to SMP02.zarr.");
  assert.equal(h.$('zarr-warning').textContent, 'Already in your SpatialData: proseg_boundaries. Adding will replace this layer.');
  for (const text of [h.$('zarr-title').textContent, h.$('zarr-help').textContent, h.$('zarr-warning').textContent]) assert.doesNotMatch(text, /sample|sdata/);
  h.$('zarr-shapes').value = 'proseg_cells'; h.$('zarr-form').emit('input', { target: h.$('zarr-shapes') });
  const pressed = h.submit.click(); await settle();
  assert.deepEqual(h.posts.map(p => p.values.shapes_key), ['proseg_cells']); assert.equal(h.submit.textContent, 'Adding…');
  h.request.resolve({ ...h.run, zarr_save: { state: 'queued', written_to: '/data/SMP02.zarr', shapes_key: 'proseg_cells', table_key: 'proseg_table' } }); await pressed;
  assert.equal(h.controller.dialog.open, false); assert.deepEqual(h.notices, ['Written to SMP02.zarr · in your notebook as soon as it is free']);
});
test('Not now (or Esc) with no save on its way tells the page, which takes a Finish & save back; a save on its way, or the page closing the dialog, does not', async () => {
  const h = saver();
  h.controller.open(h.run); await h.$('zarr-cancel').click();
  assert.deepEqual(h.cancels, ['run']); assert.equal(h.controller.dialog.open, false);
  h.controller.open(h.run); h.controller.dialog.emit('cancel'); assert.deepEqual(h.cancels, ['run', 'run'], 'Esc is Not now');
  // A save on its way: Not now reads Close, and closing takes nothing back.
  h.controller.open(h.run); const pressed = h.submit.click();
  assert.equal(h.$('zarr-cancel').textContent, 'Close'); await h.$('zarr-cancel').click(); assert.equal(h.cancels.length, 2);
  h.request.resolve({ ...h.run, zarr_save: { state: 'saving' } }); await pressed;
  h.controller.open({ ...h.run, id: 'other', zarr_save: { state: 'saving' } }); h.controller.dialog.emit('cancel'); assert.equal(h.cancels.length, 2);
  // The page closing it (its run gone, the session ended) answers nothing.
  h.controller.open({ ...h.run, id: 'third' }); h.controller.dialog.close(); assert.equal(h.cancels.length, 2);
});
test('in a notebook\'s session a case twin names your SpatialData, never the sample', () => {
  const twin = validateSaveNames({ shapes_key: 'cells', table_key: 'counts' }, ['Cells'], false, { api: true });
  assert.equal(twin.errors.shapes_key, 'Cells is already in your SpatialData: use a name that differs by more than case.');
  const h = saver();
  h.controller.setSession(session.apiSession({ session: { mode: 'api', id: '1f3a9c2e', sample: { id: 'z1', name: 'SMP02', backed: true, zarr_name: 'SMP02.zarr',
    view: 'in_place', state: 'ready', progress: { done: 0, total: 0 }, message: null, memory_only: [], tools: ['proseg'] }, tool: null, writes: 'zarr',
    notebook: { busy: false }, patches: false, link: false, state: 'running' } }));
  h.controller.open({ ...h.run, save_defaults: { shapes_key: 'cells', table_key: 'counts', existing: ['Cells'] } });
  assert.equal(h.$('zarr-shapes-error').textContent, 'Cells is already in your SpatialData: use a name that differs by more than case.');
});
