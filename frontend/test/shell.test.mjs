import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { STAIN_PATCH_TEXT } from '../src/stain-patches.mjs';
import { SHELL_TEXT, SHELL_API_TEXT, TOOLS, toolOf, leaveKind, leaveFailure, decidedAtClose, leaveWords, pending, runLine, LeavePrompt, leftoversNotice, watchClosing,
  closeWindow, nextRevision, CLOSE_CHECK_MS } from '../src/shell.mjs';

// The app shell (shell.mjs): the name users see, the first screen's two tools,
// and closing the app with runs still sampling.
const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
const main = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const between = (from, to) => html.slice(html.indexOf(from), html.indexOf(to, html.indexOf(from)));
const words = part => part.replace(/<!--[\s\S]*?-->/g, ' ').replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
const run = (id, state, extra = {}) => ({ id, state, sample_name: `S-${id}`, last: { iter: 812 }, ...extra });

test('the name users see is "SoPA companion": the page title (the window title), the header brand, the first screen', () => {
  assert.equal(SHELL_TEXT.name, 'SoPA companion');
  assert.match(html, /<title>SoPA companion<\/title>/);
  const header = between('<header', '</header>');
  assert.match(header, /<button id="home" class="brand" type="button" title="Back to the first screen">/);
  assert.equal(words(between('<button id="home"', '</button>')), 'SoPA companion');
  assert.doesNotMatch(header + between('<head>', '</head>'), /Proseg Live/, 'the old name only as a tool now');
  assert.equal(words(between('<section id="tool-choice"', '<div class="tool-cards">')), `${SHELL_TEXT.name} ${SHELL_TEXT.choose}`);
  // The page's static words are shell.mjs's (one place to change them).
  assert.ok(header.includes(`title="${SHELL_TEXT.home}"`)); assert.ok(header.includes(`title="${SHELL_TEXT.closeTitle}">${SHELL_TEXT.close}</button>`));
  assert.ok(html.includes(`title="${SHELL_TEXT.backTitle}">${SHELL_TEXT.back}</button>`));
  // Load-bearing names stay: the page's global, the harness flag, storage keys.
  assert.match(main, /^window\.sopaCompanion = \{/m); assert.match(main, /`sopa-companion:panel:\$\{panel\.id\}`/);
});

test("the first screen offers the two tools in the intended order, in the same words as shell.mjs, then the chosen one's Open sample… and Recent", () => {
  const cards = between('<div class="tool-cards">', '</section>');
  assert.ok(cards.indexOf('id="tool-stain"') >= 0 && cards.indexOf('id="tool-stain"') < cards.indexOf('id="tool-proseg"'), 'Stain segmentation first');
  for (const [tool, { name, about }] of Object.entries(TOOLS)) {
    const card = between(`<button id="tool-${tool}"`, '</button>');
    assert.equal(words(between(`<button id="tool-${tool}"`, '</strong>')), name);
    assert.equal(words(card), `${name} ${about}`);
  }
  assert.equal(TOOLS.stain.name, 'Stain segmentation'); assert.equal(TOOLS.proseg.name, 'Proseg Live');
  // Cellpose by name: InstanSeg is dormant (the engine offers it only behind an internal switch), so no card names it.
  assert.match(TOOLS.stain.about, /^Find cells in the sample's images with Cellpose, clean them up, and make training patches for the Cellpose GUI\.$/);
  assert.doesNotMatch(cards, /instanseg/i);
  assert.match(TOOLS.proseg.about, /^Watch proseg build cells from the transcripts, one iteration at a time, with the ResolVI mixing check and Go deeper\.$/);
  // The chosen tool's part is hidden as the page loads (no tool is remembered) and holds the shared Open sample… and Recent.
  const start = between('<section id="tool-start"', '</div>\n    </div>');
  assert.match(start, /^<section id="tool-start" class="tool-start" hidden>/);
  for (const id of ['tool-back', 'tool-title', 'tool-help', 'empty-open-sample', 'recent-samples']) assert.ok(start.includes(`id="${id}"`), id);
  assert.equal(words(between('<button id="tool-back"', '</button>')), SHELL_TEXT.back);
});

test('every word of the shell is plain: no banned word on the first screen, in the header, the popup or the Training patches tool', () => {
  const texts = [...Object.values(SHELL_TEXT), ...Object.values(SHELL_API_TEXT), ...Object.values(TOOLS).flatMap(t => Object.values(t)), STAIN_PATCH_TEXT.start,
    STAIN_PATCH_TEXT.toolHelp, STAIN_PATCH_TEXT.title, words(between('<div id="empty"', '</main>')), words(between('<header', '</header>'))];
  for (const text of texts) assert.deepEqual(bannedWords(text), [], text);
});

test('the header: Close after the connection dot, never greyed out; the tool named after the brand; the run type never shown', () => {
  const header = between('<header', '</header>');
  assert.match(header, /<span id="connection" class="connection" title="Connecting…"><\/span>\s*<button id="close-app" type="button" title="Close SoPA companion">Close<\/button>\s*$/);
  assert.ok(header.indexOf('id="home"') < header.indexOf('id="tool-name"') && header.indexOf('id="tool-name"') < header.indexOf('id="open-sample"'));
  assert.match(header, /<span id="tool-name" class="tool-name" hidden><\/span>/);
  assert.doesNotMatch(header, /disabled/);
  // The run type stays the page's one record (#run-kind, three values) inside a hidden label.
  assert.match(html, /<label hidden><span data-stain-copy="kind"><\/span><select id="run-kind">\s*<option value="proseg"[^>]*><\/option><option value="stain"[^>]*><\/option><option value="patches"[^>]*><\/option>/);
  // Training patches is a section inside Stain segmentation's picker, holding Patches, Size and Start from.
  const tool = between('<details id="patch-tool"', '</details>');
  assert.ok(between('<div id="stain-picker"', '<div id="proseg-picker"').includes('<details id="patch-tool" class="tool-section">'));
  assert.match(tool, /^<details id="patch-tool" class="tool-section">\s*<summary data-patch-copy="title"><\/summary>/);
  for (const id of ['patch-picker', 'patch-count', 'patch-size', 'patch-start-from']) assert.ok(tool.includes(`id="${id}"`), id);
  assert.equal(STAIN_PATCH_TEXT.title, 'Training patches'); assert.equal(STAIN_PATCH_TEXT.start, 'Make training patches');
});


const cases = [
  ['preparing', {state:'preparing'}, 'none', false, 'not_started'],
  ['loading', {state:'running',frames:0}, 'none', false, 'not_started'],
  ['unstepped', {state:'paused',frames:1,ack:{iter:0}}, 'none', false, 'not_started'],
  ['running', {state:'running',frames:2}, 'ask', true, 'unsaved'],
  ['stepped', {state:'paused',frames:2}, 'ask', true, 'unsaved'],
  ['finishing', {state:'finishing'}, 'ask', true, 'unsaved'],
  ['done', {state:'done'}, 'ask', true, 'unsaved'],
  ['failed', {state:'failed'}, 'none', false, 'ended'],
  ['stopped', {state:'stopped'}, 'none', false, 'ended'],
  ['saving', {state:'done',zarr_save:{state:'saving'}}, 'saving', false, 'saving'],
  ['saved', {state:'done',zarr_save:{state:'saved'}}, 'saved', false, 'saved'],
  ['save failed', {state:'done',zarr_save:{state:'failed'}}, 'ask', true, 'unsaved'],
  ['patches', {kind:'patches'}, 'none', false, 'patches'],
  ['test', {qc_test:{of:'r'}}, 'none', false, 'test'],
  ['stain waiting', {kind:'stain',stain_started:false}, 'none', false, 'not_started'],
  ['stain no tile', {kind:'stain',progress:{done:0}}, 'none', false, 'not_started'],
  ['stain no cells', {kind:'stain',progress:{done:2},summary_model:{cells:0}}, 'none', false, 'no_cells'],
  ['stain filtered', {kind:'stain',progress:{done:2},summary_model:{cells:2},summary:{cells:0}}, 'ask', false, 'filtered'],
  ['stain unjoined', {kind:'stain',state:'failed',progress:{done:2},summary:{cells:2}}, 'ask', true, 'unsaved'],
  ['stain saved', {kind:'stain',progress:{done:2},zarr_save:{state:'saved'}}, 'saved', false, 'saved'],
  ['stain changed', {kind:'stain',progress:{done:2},exclusions:{digest:'b'},zarr_save:{state:'saved',exclusions_digest:'a'}}, 'ask', true, 'changed'],
].map(([name,run,kind,can_add,why])=>({name,run,expect:{kind,can_add,why}}));
const shared = new URL('../../tests/data/leave_cases.json', import.meta.url);
for (const c of existsSync(shared) ? JSON.parse(readFileSync(shared,'utf8')) : cases)
  test(`leave predicate: ${c.name}`,()=>{
    assert.deepEqual(leaveKind(c.run),c.expect);
    if ('decided' in c) assert.equal(decidedAtClose(c.run),c.decided,'decided (a Finish & save closing waits for and saves)');
  });

test('pending includes finished unsaved runs, stain and saves in flight',()=>{
  assert.deepEqual(pending(cases.map(c=>c.run)), cases.filter(c=>['ask','saving'].includes(c.expect.kind)).map(c=>c.run));
  assert.deepEqual(['stain','patches','proseg',undefined].map(toolOf),['stain','stain','proseg','proseg']);
  assert.equal(runLine(run('r','paused',{last:{iter:1250}})), 'Paused at iteration 1,250');
  assert.equal(runLine({kind:'stain',progress:{done:3,total:8}}), 'Cellpose · 3 of 8 tiles done');
});
function popup(answer=async()=>{}) {
  const dom=domFixture(), calls=[];
  const p=new LeavePrompt({answer:async (choice,r)=>{calls.push([choice,r.id]);await answer(choice,r);}},dom.document);
  return {...dom,p,calls};
}
test('every trigger uses the same Save, Discard, Stay prompt, with live progress and retry',async()=>{
  const {p,$,calls,document}=popup();const r=run('a','running',{frames:2});
  for(const reason of ['new','open','home','setting','proceed']){
    p.show(r,{reason});assert.equal($('leave-title').textContent,'Save this run to S-a?');
    assert.deepEqual([$('leave-stay'),$('leave-discard'),$('leave-save')].map(b=>b.textContent),['Stay','Discard','Save to the sample']);
    assert.equal(document.activeElement,$('leave-save'));assert.equal(p.snapshot().reason,reason);
    await $('leave-stay').click();assert.equal(p.open,false);
  }
  // Close: the same prompt in the close popup's words.
  p.show(r,{reason:'close'});assert.equal($('leave-title').textContent,'Close SoPA companion?');
  assert.equal($('leave-lead').textContent,'This run is not saved to S-a yet:');assert.equal($('leave-lead').hidden,false);
  assert.equal($('leave-run').textContent,'Sampling at iteration 812');
  assert.match($('leave-help').textContent,/^Save and close saves it to the sample first \(finishing it if it still samples\)\. Discard and close deletes it\.$/);
  assert.deepEqual([$('leave-stay'),$('leave-discard'),$('leave-save')].map(b=>b.textContent),['Keep working','Discard and close','Save and close']);
  await $('leave-stay').click();assert.equal(p.open,false);
  p.show(r);await $('leave-save').click();assert.deepEqual(calls.at(-1),['add','a']);
  assert.equal(p.open,true,'only gone ends leaving');
  p.update({...r,leaving:{answer:'add',stage:'writing',message:'Writing cells…'}});
  assert.equal($('leave-progress').textContent,'Writing cells…');
  const before=calls.length;await $('leave-discard').click();assert.equal(calls.length,before);
  await $('leave-stay').click();assert.equal(p.open,false);
  p.show({...r,zarr_save:{state:'failed',message:'Disk full'}});
  assert.equal($('leave-save').textContent,'Try again');assert.match($('leave-error').textContent,/Disk full/);
});
test('all filtered cells: Discard and Stay, no Save',async()=>{
  const {p,$,calls}=popup();p.show({id:'a',kind:'stain',progress:{done:2},summary_model:{cells:2},summary:{cells:0}});
  assert.equal($('leave-save').hidden,true);assert.equal($('leave-help').textContent,SHELL_TEXT.filtered);
  await p.choose('add');assert.deepEqual(calls,[]);await p.choose('discard');assert.deepEqual(calls,[['discard','a']]);
});
test('a failed leave request stays visible and can be retried',async()=>{
  const {p,$}=popup(async()=>{throw new Error('Sample moved');});p.show(run('a','done'));
  await p.choose('add');assert.equal(p.open,true);assert.match($('leave-error').textContent,/Sample moved/);
  assert.equal(p.answered,null);assert.equal($('leave-save').textContent,'Try again');
});
test('window close without an answer sends discard; an explicit add stays add',async()=>{
  const {p}=popup(), handlers={},sent=[];
  const win={addEventListener:(name,fn)=>handlers[name]=fn};
  watchClosing(win,p,{sampling:()=>[run('a','done')],beacon:answer=>sent.push(answer)});
  const event={preventDefault(){this.prevented=true;}};handlers.beforeunload(event);
  assert.equal(event.prevented,true);assert.equal(p.snapshot().reason,'close');handlers.pagehide();
  assert.deepEqual(sent,['discard']);await p.choose('add');handlers.pagehide();assert.deepEqual(sent,['discard','add']);
});
test('leftover notices appear once per stamp and persist across reload',()=>{
  const store=new Map(),storage={getItem:key=>store.get(key),setItem:(key,v)=>store.set(key,v)};
  const leftover={stamp:'boot-1',runs:[{sample:'SMP02'}]};
  assert.match(leftoversNotice(leftover,storage),/SMP02/);assert.equal(leftoversNotice(leftover,storage),'');
  store.set('sopa-companion:leftovers:boot-2','seen');assert.equal(leftoversNotice({...leftover,stamp:'boot-2'},storage),'');
  assert.match(leftoversNotice({stamp:'boot-3',runs:[{},{}]},storage),/^2 runs/);
});
test('a window that cannot close reports it after the close check',t=>{
  t.mock.timers.enable({apis:['setTimeout']});let closed=0,stuck=0;
  closeWindow({close:()=>closed++},()=>stuck++);assert.equal(closed,1);
  t.mock.timers.tick(CLOSE_CHECK_MS);assert.equal(stuck,1);
  // The page then says the app has closed and the tab can be closed (main.js appClosed), never that the window cannot
  // close itself.
  assert.deepEqual([SHELL_TEXT.closedTitle,SHELL_TEXT.closedTab],['SoPA companion has closed','You can close this tab.']);
  assert.equal('notClosed' in SHELL_TEXT,false);assert.doesNotMatch(main,/notClosed|cannot close itself/);
});

test('a leave the engine could not carry out is said in the prompt: Couldn\'t discard, save or leave this run, over an older failed save; nothing while one is under way',()=>{
  const {p,$}=popup(),r=run('a','running',{frames:2}),failed=(answer,message='The run has not stopped yet.')=>({answer,message,at:'2026-10-23T12:00:00+00:00'});
  p.show({...r,leave_error:failed('discard')});
  assert.equal($('leave-error').hidden,false);assert.equal($('leave-error').textContent,"Couldn't discard: The run has not stopped yet.");
  assert.equal($('leave-save').textContent,SHELL_TEXT.save,'a Discard that failed is no Save to retry');
  // A Discard that failed after a save that failed: the Discard's own reason.
  p.update({...r,zarr_save:{state:'failed',message:'Disk full'},leave_error:failed('discard')});
  assert.equal($('leave-error').textContent,"Couldn't discard: The run has not stopped yet.");
  p.update({...r,zarr_save:{state:'failed',message:'Disk full'},leave_error:failed('add','Disk full')});
  assert.equal($('leave-error').textContent,"Couldn't save: Disk full");assert.equal($('leave-save').textContent,'Try again');
  // Saved already: only leaving it failed (stopping or removing the run).
  const saved={id:'b',kind:'stain',state:'done',progress:{done:4,revision:1},summary_model:{cells:5},zarr_save:{state:'saved',result_revision:1}};
  p.update({...saved,leave_error:failed('add')});assert.equal($('leave-error').textContent,"Couldn't leave this run: The run has not stopped yet.");
  assert.equal(leaveFailure({...saved,leave_error:failed('add')}),"Couldn't leave this run: The run has not stopped yet.");
  // A leave under way says its progress, never the last one's failure; none failed: nothing.
  p.update({...r,leaving:{answer:'discard',stage:'finishing',message:'Discarding the run…'},leave_error:failed('discard','older')});
  assert.equal($('leave-error').hidden,true);
  p.update(r);assert.equal($('leave-error').hidden,true);assert.equal(leaveFailure(r),'');
  // A notebook's session says Add.
  p.setMode({api:true});p.update({...r,zarr_save:{state:'failed',message:'x'},leave_error:failed('add','x')});
  assert.equal($('leave-error').textContent,"Couldn't add: x");assert.equal(leaveFailure({...r,leave_error:failed('add','x')},{api:true}),"Couldn't add: x");
});
test('a failed Discard is reported as Discard, without turning Save into a retry of it',async()=>{
  const {p,$}=popup(async()=>{throw new Error('The run is still stopping.');});p.show(run('a','done'));
  await p.choose('discard');assert.match($('leave-error').textContent,/Couldn't discard/);
  assert.equal($('leave-save').textContent,SHELL_TEXT.save);
});

test('the built-in cases agree with the engine\'s predicate as the shared fixture pins it',()=>{
  for(const c of cases)assert.deepEqual(leaveKind(c.run),c.expect,c.name);
  // Stable reasons, never shown: a Xenium bundle's run asks with nothing to add; a stain run whose filters were changed
  // after its save asks again (changed); a queued result still on its way to a notebook is a save under way.
  assert.deepEqual(leaveKind({kind:'proseg',sample_kind:'xenium',state:'done'}),{kind:'ask',can_add:false,why:'sample'});
  assert.deepEqual(leaveKind({kind:'proseg',state:'done',zarr_save:{state:'queued',written_to:null}}),{kind:'saving',can_add:false,why:'saving'});
  assert.deepEqual(leaveKind({kind:'proseg',state:'done',zarr_save:{state:'queued',written_to:'SMP02.zarr'}}),{kind:'saved',can_add:false,why:'saved'});
  assert.deepEqual(leaveKind({kind:'proseg',state:'paused',frames:1,ack:{iter:0},steps:1}),{kind:'ask',can_add:true,why:'unsaved'},'a step asked for');
});
test('the words follow the run: finishing first, a stain run\'s tile, a run changed since its save; filters that leave nothing',()=>{
  const words=r=>leaveWords({sample_name:'SMP02',...r});
  assert.equal(words({state:'running',frames:3}).help,'Save finishes it, then writes its cells to SMP02 under new names. Discard deletes it.');
  assert.equal(words({state:'done'}).help,'Save writes its cells to SMP02 under new names. Discard deletes it.');
  assert.equal(words({kind:'stain',state:'running',stain_started:true,progress:{done:2,revision:2},summary:{cells:9}}).help,
    'Save stops it after this tile and writes the cells of its finished tiles to SMP02 under new names. Discard deletes it.');
  assert.equal(words({kind:'stain',state:'stopped',stain_started:true,progress:{done:2,revision:3},summary:{cells:9},zarr_save:{state:'saved',result_revision:2}}).help,
    'Save writes its newest cells to SMP02 under new names. Discard deletes it and keeps what you saved before.');
  const filtered=words({kind:'stain',state:'stopped',progress:{done:2},summary_model:{cells:4},summary:{cells:0}});
  assert.equal(filtered.title,'Leave this run?');assert.equal(filtered.help,SHELL_TEXT.filtered);
  assert.equal(leaveWords({state:'done'},{reason:'setting'}).saveTitle,'Saved, it can be compared with the new run in Compare.');
  // A notebook's session: your SpatialData, never the variable's name.
  const api=leaveWords({sample_name:'SMP02',state:'running',frames:3},{api:true});
  assert.equal(api.title,'Add this run to your SpatialData?');assert.equal(api.save,'Add to your SpatialData');
  assert.equal(api.help,'Add finishes it, then adds its cells under new names. Discard deletes it.');
  const close=leaveWords({sample_name:'SMP02',state:'done'},{reason:'close',api:true});
  assert.equal(close.lead,'This run is not added to your SpatialData yet:');assert.equal(close.save,'Add to your SpatialData and close');
  assert.match(close.help,/Results you already added stay\.$/);
  for(const w of [api,close])for(const text of Object.values(w).filter(v=>typeof v==='string'))assert.doesNotMatch(text,/sdata/);
});
test('Stay waits while a Save or Discard is on its way; a late Discard can never land after it; a Discard under way is never taken back',async()=>{
  const gate=deferred(),sent=[];
  const {p,$}=popup(async(choice)=>{sent.push(choice);if(choice==='discard')await gate.promise;});
  const r=run('a','done');p.show(r);
  const pressed=$('leave-discard').click();
  assert.equal($('leave-progress').textContent,SHELL_TEXT.discarding);assert.equal($('leave-save').textContent,SHELL_TEXT.save,'Save does not say Saving');
  await $('leave-stay').click();p.dialog.emit('cancel');
  assert.deepEqual(sent,['discard'],'Stay and Esc wait for the Discard\'s reply');assert.equal(p.open,true);
  gate.resolve();await pressed;
  p.update({...r,leaving:{answer:'discard',stage:'discarding'}});
  await $('leave-stay').click();assert.deepEqual(sent,['discard'],'the engine is discarding: Stay takes nothing back');
  assert.equal(p.open,true,'only gone ends it');
  // An add on its way: Stay waits for its reply, then cancels the leave in the engine.
  const slow=deferred(),asked=[];const q=popup(async(choice)=>{asked.push(choice);if(choice==='add')await slow.promise;});
  q.p.show(run('b','done'));const add=q.$('leave-save').click();await q.$('leave-stay').click();assert.deepEqual(asked,['add']);
  slow.resolve();await add;q.p.update({...run('b','done'),leaving:{answer:'add',stage:'writing'}});
  await q.$('leave-stay').click();assert.deepEqual(asked,['add','stay']);assert.equal(q.p.open,false);
});
test('every answer carries a newer revision than the one before, also across prompts; the window\'s ✕ names the runs it is for',async()=>{
  const revisions=[];const {p,$}=popup();p.answer=async(choice,r,revision)=>{revisions.push([choice,revision]);};
  p.show(run('a','done'));await $('leave-save').click();p.update({...run('a','done'),leaving:{answer:'add'}});await $('leave-stay').click();
  assert.deepEqual(revisions.map(([c])=>c),['add','stay']);assert.ok(revisions[1][1]>revisions[0][1]);
  const later=nextRevision(Date.now()-60000);assert.ok(later>revisions[1][1],'never older than one given, even with the clock behind');
  assert.ok(nextRevision(later)>later);
  const handlers={},beacons=[],win={addEventListener:(name,fn)=>handlers[name]=fn};
  watchClosing(win,p,{sampling:()=>[run('a','running',{frames:2})],beacon:(answer,runs)=>beacons.push([answer,runs.map(r=>r.id)])});
  p.answered=null;handlers.pagehide();assert.deepEqual(beacons,[['discard',['a']]],'nobody answered: discard, for the run shown');
});
test('the start notice names a save that failed while the app closed and a run still being saved by the last session',()=>{
  const store=new Map(),storage={getItem:key=>store.get(key),setItem:(key,v)=>store.set(key,v)};
  const text=leftoversNotice({stamp:'boot-9',runs:[{sample:'SMP02'},{sample_name:'SMP01',failed:'No space left'},{sample:'SMP03',still_saving:true}]},storage);
  assert.equal(text,'A run on SMP02 that was not saved to the sample was removed when the app started. Saving the run on SMP01 failed when the app '+
    'closed (No space left), so it was removed when the app started. The run on SMP03 was still being saved by the last session when the app started: '+
    'it is removed once that save ends.');
  assert.deepEqual(bannedWords(text),[]);
});
test('closing a notebook\'s session while its result is on its way to the notebook: both answers still end it (the engine completes the Add)',async()=>{
  const {p,$,calls}=popup();p.setMode({api:true});
  const r={id:'a',state:'done',sample_name:'SMP02',zarr_save:{state:'queued',written_to:null,message:'Waiting for your notebook: a cell is running'}};
  assert.equal(leaveKind(r).kind,'saving');
  p.show(r,{reason:'close'});
  assert.equal($('leave-save').textContent,'Add to your SpatialData and close');assert.equal($('leave-progress').hidden,false);
  assert.equal($('leave-progress').textContent,'Waiting for your notebook: a cell is running');
  await $('leave-discard').click();assert.deepEqual(calls,[['discard','a']]);
  // Leaving it any other way waits for the save, as on the desktop.
  const q=popup();q.p.setMode({api:true});q.p.show(r,{reason:'new'});await q.$('leave-discard').click();assert.deepEqual(q.calls,[]);
  assert.equal(q.$('leave-save').textContent,'Adding…');
});
test('a Finish & save not cancelled since, finishing or done, is decided; a save tried, a cancelled dialog, a stain run or a test square is not',()=>{
  const r={kind:'proseg',sample_kind:'zarr',state:'done',finish_requested:true,zarr_save:null};
  assert.equal(decidedAtClose(r),true);assert.equal(decidedAtClose({...r,state:'finishing'}),true);
  for(const other of [{...r,save_cancelled:true},{...r,zarr_save:{state:'failed'}},{...r,state:'paused'},{...r,kind:'stain'},
    {...r,qc_test:{of:'p'}},{...r,sample_kind:'xenium'},{...r,finish_requested:false}])assert.equal(decidedAtClose(other),false,JSON.stringify(other));
});
test('closing, a run whose Finish & save counts as answered is named as being saved whatever the answer, and either button saves it',async()=>{
  const {p,$,calls}=popup();
  const r={id:'a',kind:'proseg',sample_kind:'zarr',sample_name:'SMP02',state:'done',finish_requested:true,zarr_save:null,last:{iter:1200}};
  p.show(r,{reason:'close'});
  assert.equal($('leave-title').textContent,'Close SoPA companion?');
  assert.equal($('leave-lead').textContent,'Being saved already, whatever you choose:');assert.equal($('leave-run').textContent,'Finished at iteration 1,200');
  assert.equal($('leave-help').textContent,'Finish & save was pressed, so it is saved to SMP02 under new names before the app closes.');
  assert.doesNotMatch($('leave-help').textContent,/deletes/,'never says Discard and close deletes it');
  await $('leave-discard').click();assert.deepEqual(calls,[['add','a']],'Discard and close saves it too');assert.equal(p.answered,'add');
  p.update({...r,leaving:{answer:'add',stage:'writing',message:'Saving to the sample…'}});
  assert.equal($('leave-save').textContent,'Saving…');assert.equal($('leave-progress').textContent,'Saving to the sample…');
  // Its Save dialog cancelled since (leave stay): asked as any other run; leaving it any other way asks as before.
  const q=popup();q.p.show({...r,save_cancelled:true},{reason:'close'});
  assert.equal(q.$('leave-lead').textContent,'This run is not saved to SMP02 yet:');await q.$('leave-discard').click();assert.deepEqual(q.calls,[['discard','a']]);
  const n=popup();n.p.show(r,{reason:'new'});assert.equal(n.$('leave-lead').hidden,true);await n.$('leave-discard').click();assert.deepEqual(n.calls,[['discard','a']]);
  // A notebook's session: added whatever the answer, in its words.
  const s=popup();s.p.setMode({api:true});s.p.show(r,{reason:'close'});
  assert.equal(s.$('leave-lead').textContent,'Being added already, whatever you choose:');
  assert.equal(s.$('leave-help').textContent,'Finish & save was pressed, so it is added under new names before the session ends.');
  await s.$('leave-discard').click();assert.deepEqual(s.calls,[['add','a']]);
  for(const text of [SHELL_TEXT.closeDecided,SHELL_TEXT.closeDecidedHelp,SHELL_API_TEXT.closeDecided,SHELL_API_TEXT.closeDecidedHelp])
    assert.deepEqual(bannedWords(text),[]);
});
test('the start notice says how many runs of earlier versions it removed, also with no other leftover',()=>{
  const store=new Map(),storage={getItem:key=>store.get(key),setItem:(key,v)=>store.set(key,v)};
  assert.equal(leftoversNotice({stamp:'boot-o1',runs:[],older_removed:3},storage),
    '3 runs from earlier versions were removed: they were saved already or had nothing to save.');
  assert.equal(leftoversNotice({stamp:'boot-o1',runs:[],older_removed:3},storage),'','once per start');
  assert.equal(leftoversNotice({stamp:'boot-o2',runs:[{sample:'SMP02'}],older_removed:1},storage),
    'A run on SMP02 that was not saved to the sample was removed when the app started. A run from an earlier version was removed: it was saved already or had nothing to save.');
  assert.equal(leftoversNotice({stamp:'boot-o3',runs:[],older_removed:0},storage),'');
  assert.deepEqual(bannedWords(leftoversNotice({stamp:'boot-o4',older_removed:2},storage)),[]);
});
