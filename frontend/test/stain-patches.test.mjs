import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { StainPatches, STAIN_PATCH_TEXT, pixelPoint, boxGesture, canReviewStain, patchSizeNote } from '../src/stain-patches.mjs';
import { PATCH_FOLDER_TEXT } from '../src/patch-folder.mjs';
import { EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { domFixture, deferred, settle } from './dom-fixture.mjs';

const run={id:'r',kind:'stain',state:'done',progress:{done:1,revision:2}};
const value={version:1,size:128,count:8,set:null,run_pixel_to_sample:[0,0,1,1],
  patches:[{id:1,box:[10,10,138,138]},{id:2,box:[200,10,328,138]}],exported:null};
function fixture(t,{open=run}={}) {
  const previous=globalThis.document,previousWindow=globalThis.window,{document}=domFixture();
  globalThis.document=document;globalThis.window={addEventListener(){}};
  document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  document.createElementNS=(_,tag)=>document.createElement(tag);
  const canvas=document.getElementById('stage');
  canvas.setPointerCapture=()=>{};canvas.hasPointerCapture=()=>true;canvas.releasePointerCapture=()=>{};
  canvas.width=800;canvas.height=400;
  const moved=[];
  const stage={canvas,camera:{x:400,y:200,zoom:1},toWorld:(x,y)=>[x,y],toScreen:(x,y)=>[x,y],onChange:fn=>moved.push(fn)};
  const reads=[],writes=[],errors=[],folders=[];
  const read=()=>{const d=deferred();reads.push(d);return d.promise;};
  const write=(op,id,body)=>{const d=deferred();writes.push({op,id,body,...d});return d.promise;};
  // The export dialog's folder listings (patch-folder.mjs): answered by each test.
  const list=(id,path)=>{const d=deferred();folders.push({id,path,...d});return d.promise;};
  const api={stainPatches:read,stainPropose:(...a)=>write('propose',...a),stainPatchBoxes:(...a)=>write('boxes',...a),
    stainPatchExport:(...a)=>write('export',...a),stainPatchOpen:(...a)=>write('open',...a),stainPatchFolders:list};
  const panel=new StainPatches(stage,api,{notice:e=>errors.push(e)});
  t.after(()=>{panel.reset();globalThis.document=previous;globalThis.window=previousWindow;});
  // Returns whether the panel took the gesture (main.js's own region and camera gestures then never see it).
  const fire=(type,x,y)=>{
    let taken=false;
    canvas.emit(type,{button:0,pointerId:1,offsetX:x,offsetY:y,stopImmediatePropagation(){taken=true;}});
    return taken;
  };
  if(open)panel.update(open);
  return {panel,reads,writes,errors,folders,fire,stage,document,camera:()=>moved.forEach(fn=>fn())};
}
// What explain.mjs showed beside a button ('' when nothing is shown).
const note=button=>button.nextSibling?.className==='missing-note' && !button.nextSibling.hidden ? button.nextSibling.textContent : '';
// The box drawn while a gesture is under way, as [x0, y0, x1, y1] (pixels when the geometry is [0, 0, 1, 1]).
const ghost=panel=>{const a=k=>Number(panel.ghost.getAttribute(k));return [a('x'),a('y'),a('x')+a('width'),a('y')+a('height')];};
const buttons=panel=>panel.panel.children.filter(n=>n.tagName==='BUTTON');
// A press the panel should refuse is not awaited: a regression then fails the checks instead of waiting on its request.
const press=async button=>{button.click();await settle();};

test('patch copy is plain and every helper refusal is checked',()=>{
  for(const words of Object.values(STAIN_PATCH_TEXT))assert.deepEqual(bannedWords(words),[],words);
  for(const match of readFileSync(new URL('../../zarr_bridge/stain_patches.py',import.meta.url),'utf8')
    .matchAll(/(?:ValueError|RuntimeError)\("([^"]+)"/g))assert.deepEqual(bannedWords(match[1]),[],match[1]);
  assert.deepEqual(pixelPoint([14,29],[10,20,2,3]),[2,3]);
  assert.deepEqual(boxGesture([1,2,11,12],[3,4],[5,7]),[3,5,13,15]);
  assert.deepEqual(boxGesture(null,[10,20],[5,8]),[5,8,10,20]);
  assert.equal(canReviewStain({...run,state:'paused',paused:true,ack:{state:'running'}}),false);
  assert.equal(canReviewStain({...run,state:'paused',paused:true,ack:{state:'paused'}}),true);
});

test('an export asks how far it is while it runs, says it to the dialog, and stops asking once it answers',async t=>{
  // An export with the model's outlines runs the model on every patch (a minute a patch on a CPU): while its request is
  // open the panel asks the patches' state (GET .../stain/patches "exporting") and hands each answer to the dialog.
  const {panel,reads,writes}=fixture(t);
  reads.shift().resolve(value);await settle();
  panel.exportPollMs=1;
  const heard=[],exporting=panel.exportTo('/data/patches',1,progress=>heard.push(progress));
  await settle();
  assert.equal(writes.at(-1).op,'export');
  for(const done of [0,2]){
    while(!reads.length)await new Promise(r=>setTimeout(r,2));
    reads.shift().resolve({...value,exporting:{done,total:8}});await settle();
  }
  assert.deepEqual(heard,[{done:0,total:8},{done:2,total:8}]);
  writes.at(-1).resolve({...value,version:2,exporting:null});await exporting;
  // Open now: the panel's own reload after an export, and perhaps one question asked before the answer. Each answered
  // "exporting" over many poll periods: a loop still asking would ask again after each and hand the answer on.
  const open=reads.length;let answered=0;
  assert.ok(open>=1 && open<=2,`${open} questions open`);
  for(let round=0;round<10;round++){
    while(reads.length){reads.shift().resolve({...value,version:2,exporting:{done:5,total:8}});answered++;}
    await new Promise(r=>setTimeout(r,5));await settle();
  }
  assert.equal(answered,open,'no question once the export answered');
  assert.deepEqual(heard,[{done:0,total:8},{done:2,total:8}]);
});

test('proposal defaults and numbered boxes persist without idle DOM rebuilding',async t=>{
  const {panel,reads,writes}=fixture(t);
  assert.equal(panel.count.value,'8');assert.equal(panel.size.value,'256');
  reads[0].resolve(value);await settle();
  assert.equal(panel.boxes.children.length,14);
  const nodes=[...panel.boxes.children];panel.apply({...value});assert.deepEqual(panel.boxes.children,nodes);
  const request=panel.propose.click();
  assert.deepEqual(writes[0].body,{version:1,count:8,size:128});
  writes[0].resolve({...value,version:2});await request;
  assert.equal(panel.data.version,2);
});

test('patch size reads in pixels of the selected copy and states its physical width, at the level\'s sdata name',async t=>{
  const h=fixture(t);h.reads[0].resolve({...value,size:256,level:2,factor:4.004,um_per_px:.851});await settle();
  assert.equal(h.panel.resolutionNote.textContent,'256 px = 218 µm across at scale2');
  assert.equal(h.panel.resolutionNote.hidden,false);
  assert.equal(patchSizeNote(256,null),'');
  // Patches cut by an earlier version: named by their level, else by the power of two of their factor; never Full or 1/2.
  assert.equal(patchSizeNote(128,{factor:1,um_per_px:.5}),'128 px = 64 µm across at scale0');
  assert.equal(patchSizeNote(128,{level:1,factor:2,um_per_px:1,label:'1/2'}),'128 px = 128 µm across at scale1');
  assert.equal(patchSizeNote(128,{factor:4,um_per_px:2}),'128 px = 256 µm across at scale2');
  assert.equal(patchSizeNote(128,{level:2,factor:9,um_per_px:4.5}),'128 px = 576 µm across at scale2');
  // Transcripts that seem to be in pixels: their µm are pixels, so no width is claimed (the picker's note too, main.js).
  assert.equal(patchSizeNote(128,{level:1,factor:2,um_per_px:1},{pixelFrame:true}),'');
  assert.deepEqual(bannedWords(h.panel.resolutionNote.textContent),[]);
  h.panel.run={...run,tiling:{cell_px_median:12.4}};h.panel.apply({...h.panel.data,exported:{folder:'/tmp/patches'}});
  assert.match(h.panel.resolutionNote.textContent,/Cells are about 12 px across in these patches\./);
  // The run's own page on such a sample (the engine marks the run pixel_frame): no width either, the cells' size stays.
  h.panel.run={...run,pixel_frame:true,tiling:{cell_px_median:12.4}};h.panel.render();
  assert.equal(h.panel.resolutionNote.textContent,'Cells are about 12 px across in these patches.');
  h.panel.run={...run,pixel_frame:true};h.panel.apply({...h.panel.data,exported:null});
  assert.equal(h.panel.resolutionNote.textContent,'');assert.equal(h.panel.resolutionNote.hidden,true);
});

test('a stain run shows no Training patches before its Start: nothing to cut them from, and no press to refuse',async t=>{
  const {panel,reads}=fixture(t,{open:null});
  // The run as the picker's Start creates it: paused, never started (engine/stain.py stain_started false).
  panel.update({id:'fresh',kind:'stain',state:'paused',paused:true,stain_started:false,progress:{done:0,total:0,revision:0}});
  assert.equal(panel.panel.hidden,true);assert.equal(panel.active,false);
  reads.at(-1)?.resolve({...value,version:0,patches:[]});await settle();
  assert.equal(panel.panel.hidden,true,'its draft read changes nothing');
  // Started (Start pressed: it runs): the panel shows, saying when patches can be made.
  panel.update({id:'fresh',kind:'stain',state:'running',paused:false,stain_started:true,progress:{done:0,total:9,revision:0}});
  assert.equal(panel.panel.hidden,false);assert.equal(panel.hint.textContent,STAIN_PATCH_TEXT.empty);
  // A run of an earlier version (no stain_started) shows it as before.
  panel.update({...run,id:'older'});assert.equal(panel.panel.hidden,false);
});
test('a standalone patch item enables the existing boxes without a single finished tile',async t=>{
  const {panel,reads,writes}=fixture(t);
  panel.update({id:'direct',kind:'patches',state:'done',progress:{done:0,revision:0}});
  reads.at(-1).resolve({...value,version:0});await settle();
  assert.equal(panel.active,true);assert.equal(panel.panel.hidden,false);
  assert.equal(panel.propose.disabled,false);assert.equal(panel.export.disabled,false);
  assert.equal(panel.boxes.children.length,14);
  assert.equal(writes.length,0,'boxes already there: nothing is suggested by itself');
});

test('real pointer events move, drop and add boxes, with input restored after a refusal',async t=>{
  const {panel,reads,writes,fire,errors}=fixture(t);reads[0].resolve(value);await settle();
  fire('pointerdown',50,50);fire('pointermove',60,55);fire('pointerup',60,55);
  assert.deepEqual(writes[0].body.boxes,[[20,15,148,143],[200,10,328,138]]);
  writes[0].resolve(value);await settle();
  fire('pointerdown',131,17);fire('pointerup',131,17);
  assert.deepEqual(writes[1].body.boxes,[[200,10,328,138]]);
  writes[1].resolve(value);await settle();
  await panel.add.click();
  fire('pointerdown',20,20);fire('pointermove',70,80);fire('pointerup',70,80);
  assert.deepEqual(writes[2].body.boxes.at(-1),[20,20,70,80]);
  writes[2].reject(Object.assign(new Error('outside'),{status:422}));await settle();
  assert.equal(panel.busy,false);assert.equal(errors.at(-1),'outside');
  const before=writes.length;fire('pointerdown',500,200);fire('pointerup',550,230);
  assert.equal(writes.length,before,'outside the boxes the normal camera gesture stays available');
});

test('stale reads and writes cannot replace a newer draft or another run',async t=>{
  const {panel,reads,writes}=fixture(t);reads[0].resolve(value);await settle();
  panel.load();const request=panel.propose.click();
  reads[1].resolve({...value,version:0});await settle();assert.equal(panel.data.version,1);
  panel.update({...run,id:'other'});
  assert.equal(panel.propose.getAttribute('aria-busy'),null,'another run is not busy with the last one');
  assert.equal(panel.propose.textContent,STAIN_PATCH_TEXT.propose);
  writes[0].resolve({...value,version:3});await request;
  assert.equal(panel.data,null);assert.equal(panel.run.id,'other');
});

test('free size and all four corners allow bounded non-square boxes',async t=>{
  const {panel,reads,writes,fire}=fixture(t);reads[0].resolve(value);await settle();
  panel.size.value='300';const proposal=panel.propose.click();assert.equal(writes[0].body.size,300);
  writes[0].resolve(value);await proposal;
  fire('pointerdown',138,138);fire('pointermove',160,180);fire('pointerup',160,180);
  assert.deepEqual(writes[1].body.boxes[0],[10,10,160,180]);writes[1].resolve(value);await settle();
  for(const corner of [0,1,2,3]) {
    const box=boxGesture([10,10,138,138],[0,0],[40000,40000],corner);
    assert.ok(box[2]-box[0]<=4096 && box[3]-box[1]<=4096);
  }
  assert.deepEqual(boxGesture([10,10,138,138],[138,138],[12,13],3),[10,10,26,26]);   // pulled from the corner itself
  panel.size.value='1025';panel.render();assert.equal(panel.sizeNote.hidden,false);
  panel.size.value='63';panel.render();assert.equal(panel.propose.disabled,false);
  const before=writes.length;await press(panel.propose);
  assert.equal(writes.length,before);assert.match(note(panel.propose),/Size \(pixels\): a whole number from 64 to 4096/);
});

test('watch renders correction counts, open fallback and resets cleanly on resume',async t=>{
  const {panel,reads,writes}=fixture(t);
  reads[0].resolve({...value,set:'abc',exported:{folder:'/tmp/patches',states:[{id:1,state:'corrected',cells:5}],corrected:1}});
  await settle();
  assert.match(panel.list.textContent,/Patch 1 · corrected · 5 cells/);
  const request=panel.open.click();assert.deepEqual(writes[0].body,{set:'abc'});
  writes[0].resolve({folder:'/tmp/patches',command:'python -m gui first.tif',running:false});await request;
  assert.match(panel.command.textContent,/No desktop display/);
  panel.update({...run,state:'running'});assert.equal(panel.export.disabled,false);
  await press(panel.export);assert.match(note(panel.export),/after pausing or finishing the run/);
  assert.equal(writes.length,1);
  assert.equal(panel.svg.style.display,'none');
  panel.reset();assert.equal(panel.timer!==undefined,true);assert.equal(panel.panel.hidden,true);
  assert.equal(note(panel.export),'','a new run starts without the last run\'s explanations');
});

test('a suggestion with fewer patches than asked says so; a full one says nothing',async t=>{
  const {panel,reads,writes,errors}=fixture(t);
  reads[0].resolve(value);await settle();
  let request=panel.propose.click();
  writes[0].resolve({...value,version:2,message:''});await request;
  assert.deepEqual(errors,[]);
  request=panel.propose.click();
  writes[1].resolve({...value,version:3,message:'Only 2 patches fit in this region.'});await request;
  assert.deepEqual(errors,['Only 2 patches fit in this region.']);
  assert.equal(panel.data.version,3);
});

test('no panel button is ever disabled for missing input: a press names the field and what it needs',async t=>{
  const {panel,reads,writes,document}=fixture(t,{open:{...run,box:[0,0,300,200]}});
  const pressable=()=>{for(const b of buttons(panel))assert.equal(b.disabled,false,b.id);};
  pressable();
  await press(panel.propose);assert.match(note(panel.propose),/still loading/);
  reads[0].resolve({...value,patches:[],version:2});await settle();
  assert.equal(writes.length,0,'emptied on purpose: nothing is suggested by itself');
  panel.count.value='0';panel.size.value='5000';panel.render();pressable();
  await press(panel.propose);
  assert.equal(writes.length,0);
  assert.match(note(panel.propose),/Check these first:/);
  assert.match(note(panel.propose),/Patches: a whole number from 1 to 64/);
  assert.match(note(panel.propose),/Size \(pixels\): a whole number from 64 to 4096/);
  assert.ok(panel.count.classList.contains(NEEDS_CLASS) && panel.size.classList.contains(NEEDS_CLASS));
  assert.equal(panel.count.getAttribute('aria-invalid'),'true');assert.equal(document.activeElement,panel.count);
  panel.count.value='8';panel.size.value='256';     // the region is 300 x 200 pixels: 256 does not fit
  await press(panel.propose);
  assert.equal(note(panel.propose),'Check these first:Size (pixels): at most 200, so each patch fits inside the region');
  assert.ok(!panel.count.classList.contains(NEEDS_CLASS));assert.equal(document.activeElement,panel.size);
  await press(panel.export);assert.match(note(panel.export),/Make training patches or add a box first\./);
  await press(panel.open);assert.match(note(panel.open),/Export patches first\./);
  assert.equal(writes.length,0);pressable();
  panel.size.value='200';const request=panel.propose.click();
  assert.equal(note(panel.propose),'');assert.deepEqual(writes[0].body,{version:2,count:8,size:200});
  writes[0].resolve({...value,version:3,set:'abc',exported:{folder:'/tmp/p',states:[],gui_running:true}});await request;
  assert.equal(note(panel.export),'','an explanation that no longer holds goes once the panel has worked');
  assert.equal(panel.open.textContent,STAIN_PATCH_TEXT.running);pressable();
  const before=writes.length;await press(panel.open);
  assert.equal(writes.length,before);assert.match(note(panel.open),/already open on this folder/);
  panel.update({...run,box:[0,0,300,200],state:'running'});pressable();
  assert.equal(panel.count.disabled,false,'fields are not disabled for a run that is still running');
  for(const b of [panel.propose,panel.add,panel.export]) {
    await press(b);assert.match(note(b),/after pausing or finishing the run/,b.id);
  }
  assert.equal(writes.length,before);
});

test('a running request makes only the pressed button busy, with a label; other presses say the last step is still running',async t=>{
  const {panel,reads,writes,folders}=fixture(t);reads[0].resolve(value);await settle();
  const request=panel.propose.click();
  assert.equal(panel.propose.getAttribute('aria-busy'),'true');assert.equal(panel.propose.textContent,'Suggesting…');
  assert.equal(panel.export.getAttribute('aria-busy'),null);
  assert.equal(panel.count.disabled,true);assert.equal(panel.size.disabled,true);
  for(const b of buttons(panel))assert.equal(b.disabled,false,b.id);
  await press(panel.export);await press(panel.propose);await press(panel.add);
  assert.equal(writes.length,1);assert.equal(panel.adding,false);
  for(const b of [panel.export,panel.propose,panel.add])assert.match(note(b),new RegExp(EXPLAIN_TEXT.busy),b.id);
  writes[0].resolve({...value,version:2});await request;
  assert.equal(panel.propose.getAttribute('aria-busy'),null);assert.equal(panel.propose.textContent,STAIN_PATCH_TEXT.propose);
  assert.equal(panel.count.disabled,false);assert.equal(panel.size.disabled,false);
  for(const b of [panel.export,panel.propose,panel.add])assert.equal(note(b),'','the step is done: its notes go');
  // Export patches asks where first (patch-folder.mjs): it posts nothing itself; the dialog's Export here does, with
  // the version of the press and the folder in its field, and Export patches is busy while that runs.
  const $=id=>globalThis.document.getElementById(id);
  await press(panel.export);
  assert.equal(writes.length,1);assert.equal($('patch-folder').open,true);assert.equal(folders.length,1);
  folders[0].resolve({path:'/tmp/p',exists:false,parent:'/tmp',entries:[],shortcuts:[],new:'Test-scale0',holds:null,
    mixes:null,refused:null});await settle();
  panel.data={...panel.data,version:9};             // a read meanwhile: the export keeps the boxes of the press
  const exporting=$('patch-folder-export').click();
  assert.deepEqual(writes[1].body,{version:2,folder:'/tmp/p'});
  assert.equal(panel.export.textContent,'Exporting…');assert.equal(panel.export.getAttribute('aria-busy'),'true');
  assert.equal($('patch-folder-export').textContent,PATCH_FOLDER_TEXT.exporting);
  writes[1].resolve({...value,version:3,set:'abc',exported:{folder:'/tmp/p',states:[]}});await exporting;await settle();
  assert.equal(panel.export.textContent,STAIN_PATCH_TEXT.export);assert.equal(panel.export.getAttribute('aria-busy'),null);
  assert.equal($('patch-folder').open,false,'exported: the dialog closes');
  assert.equal(panel.folder.textContent,'Saved in /tmp/p');
  const opening=panel.open.click();
  assert.equal(panel.open.textContent,'Opening…');assert.equal(panel.snapshot().busy,true);
  writes[2].resolve({folder:'/tmp/p',command:'python -m gui first.tif',running:true});await opening;
  assert.equal(panel.open.textContent,STAIN_PATCH_TEXT.again);assert.equal(panel.open.getAttribute('aria-busy'),null);
});

test('an export whose folder is gone says so, Open explains it, and Export patches still asks where',async t=>{
  const {panel,reads,writes,folders}=fixture(t);
  const gone={set:'abc',folder:'/tmp/p',missing:true,patches:[],states:[],gui_running:false,corrected:0,cells:0};
  panel.run={...run,tiling:{cell_px_median:12.4}};
  reads[0].resolve({...value,set:'abc',exported:gone});await settle();
  const words=STAIN_PATCH_TEXT.gone.replace('{folder}','/tmp/p');
  assert.equal(panel.folder.textContent,words);
  assert.doesNotMatch(panel.resolutionNote.textContent,/Cells are about/,'no cell size for patches that are gone');
  await press(panel.open);
  assert.equal(writes.length,0);assert.ok(note(panel.open).includes(words),note(panel.open));
  await press(panel.export);
  assert.equal(globalThis.document.getElementById('patch-folder').open,true);assert.equal(folders.length,1);
  assert.equal(writes.length,0,'the dialog asks where; nothing is posted yet');
});

test('a reset closes the export dialog, and a listing it was waiting for shows nothing',async t=>{
  const {panel,reads,folders}=fixture(t);reads[0].resolve(value);await settle();
  await press(panel.export);
  const dialog=globalThis.document.getElementById('patch-folder');
  assert.equal(dialog.open,true);assert.deepEqual(panel.snapshot().folder,
    {open:true,loading:true,exporting:false,path:null,exists:null,refused:null,mixes:null});
  panel.reset();
  assert.equal(dialog.open,false);
  folders[0].resolve({path:'/tmp/late',exists:true,parent:'/tmp',entries:[],shortcuts:[],new:null,holds:null,mixes:null,
    refused:null});await settle();
  assert.equal(panel.folders.snapshot().path,null);
});

test('a draft never suggested is suggested once when its panel opens; an emptied one stays empty',async t=>{
  const {panel,reads,writes,errors}=fixture(t,{open:{...run,id:'item',kind:'patches',progress:{done:0,revision:0}}});
  reads[0].resolve({...value,version:0,patches:[],count:5,size:200});await settle();
  assert.equal(writes.length,1,'the boxes are asked for straight away');
  assert.equal(writes[0].op,'propose');assert.deepEqual(writes[0].body,{version:0,count:5,size:200});
  assert.equal(panel.propose.textContent,'Suggesting…');
  writes[0].reject(Object.assign(new Error('No patches with image detail were found. Try another region.'),{status:422}));
  await settle();
  assert.deepEqual(errors,['No patches with image detail were found. Try another region.']);
  reads.at(-1).resolve({...value,version:0,patches:[],count:5,size:200});await settle();
  assert.equal(writes.length,1,'once: the next reads of the same draft do not ask again');
  const request=panel.propose.click();assert.equal(writes.length,2,'the button suggests again');
  writes[1].resolve({...value,version:1});await request;
  assert.equal(panel.data.version,1);
  panel.update({...run,id:'emptied'});
  reads.at(-1).resolve({...value,version:4,patches:[]});await settle();
  assert.equal(writes.length,2,'boxes emptied on purpose are never filled again by themselves');
});

test('a stain run gets its first boxes only once the user starts working in its panel, and only once',async t=>{
  const {panel,reads,writes}=fixture(t);
  reads[0].resolve({...value,version:0,patches:[]});await settle();
  panel.load();reads.at(-1).resolve({...value,version:0,patches:[]});await settle();
  assert.equal(writes.length,0,'opened and only looked at: no boxes over its outlines');
  panel.panel.emit('focusin',{target:panel.count});                     // the user goes to change the count
  assert.equal(writes.length,1);assert.equal(writes[0].op,'propose');
  writes[0].resolve({...value,version:1});await settle();
  assert.equal(panel.boxes.children.length,14);
  panel.panel.emit('pointerdown',{target:panel.size});assert.equal(writes.length,1,'once');
  // Still running: there is nothing to work on yet, so a press in the panel does not count.
  panel.update({...run,id:'running',state:'running'});reads.at(-1).resolve({...value,version:0,patches:[]});await settle();
  panel.panel.emit('pointerdown',{target:panel.hint});assert.equal(writes.length,1);
  panel.update({...run,id:'running'});reads.at(-1).resolve({...value,version:0,patches:[]});await settle();
  assert.equal(writes.length,1,'finished, and still only looked at');
  panel.panel.emit('pointerdown',{target:panel.hint});
  assert.equal(writes.length,2);writes[1].resolve({...value,version:1});await settle();
  // A first press of one of the panel's buttons does that button's work, never a suggestion besides.
  panel.update({...run,id:'pressed'});reads.at(-1).resolve({...value,version:0,patches:[]});await settle();
  panel.panel.emit('pointerdown',{target:panel.export});await press(panel.export);
  assert.equal(writes.length,2);assert.match(note(panel.export),/Make training patches or add a box first\./);
  panel.panel.emit('focusin',{target:panel.count});assert.equal(writes.length,2);
  const request=panel.propose.click();assert.equal(writes.length,3,'the button suggests');
  writes[2].resolve({...value,version:1});await request;
});

test('the first boxes never explain, outline, focus or scroll by themselves: with a problem they only count as tried',async t=>{
  const {panel,reads,writes,document}=fixture(t,{open:{...run,id:'item',kind:'patches',progress:{done:0,revision:0},box:[0,0,300,200]}});
  let scrolled=0;for(const field of [panel.count,panel.size])field.scrollIntoView=()=>scrolled++;
  reads[0].resolve({...value,version:0,patches:[],count:8,size:256});await settle();      // 256 does not fit in 300 x 200
  assert.equal(writes.length,0);
  for(const b of buttons(panel))assert.equal(note(b),'',b.id);
  assert.ok(!panel.size.classList.contains(NEEDS_CLASS));assert.equal(panel.size.getAttribute('aria-invalid'),null);
  assert.equal(document.activeElement,null);assert.equal(scrolled,0);
  panel.load();reads.at(-1).resolve({...value,version:0,patches:[],count:8,size:256});await settle();
  assert.equal(writes.length,0);assert.equal(note(panel.propose),'','tried once, and still quiet');
  await press(panel.propose);
  assert.match(note(panel.propose),/Size \(pixels\): at most 200, so each patch fits inside the region/);
  assert.equal(document.activeElement,panel.size);assert.equal(scrolled,1,'a real press shows what is wrong');
  panel.size.value='200';const request=panel.propose.click();
  assert.deepEqual(writes[0].body,{version:0,count:8,size:200});
  writes[0].resolve({...value,version:1});await request;
});

test('the first boxes leave count and size open to typing, and keep what was typed meanwhile',async t=>{
  const {panel,reads,writes}=fixture(t);
  reads[0].resolve({...value,version:0,patches:[],count:8,size:128});await settle();
  panel.panel.emit('focusin',{target:panel.count});
  assert.deepEqual(writes[0].body,{version:0,count:8,size:128});
  assert.equal(panel.propose.textContent,STAIN_PATCH_TEXT.suggesting);
  assert.equal(panel.count.disabled,false,'the field the user went to stays theirs');assert.equal(panel.size.disabled,false);
  panel.count.value='12';panel.count.emit('input');
  writes[0].resolve({...value,version:1,count:8,size:128});await settle();
  assert.equal(panel.count.value,'12');
  reads.at(-1).resolve({...value,version:1,count:8,size:128});await settle();
  assert.equal(panel.count.value,'12','what was typed meanwhile stays');
  const request=panel.propose.click();
  assert.deepEqual(writes[1].body,{version:1,count:12,size:128});
  assert.equal(panel.count.disabled,true,'a pressed suggestion holds the fields while it runs');
  writes[1].resolve({...value,version:2,count:12,size:128});await request;
  assert.equal(panel.count.disabled,false);
});

test('a press or click on a box that changes nothing saves nothing',async t=>{
  const {panel,reads,writes,fire}=fixture(t);reads[0].resolve(value);await settle();
  fire('pointerdown',50,50);fire('pointerup',50,50);                              // a click inside a box
  fire('pointerdown',136,136);fire('pointerup',136,136);                          // a click on its corner handle
  fire('pointerdown',60,60);fire('pointermove',90,70);fire('pointerup',60,60);    // dragged, then brought back
  assert.equal(writes.length,0,'the draft keeps its version');assert.equal(panel.data.version,1);
  fire('pointerdown',50,50);fire('pointerup',51,50);
  assert.equal(writes.length,1,'a real move is saved');assert.deepEqual(writes[0].body.boxes[0],[11,10,139,138]);
  writes[0].resolve({...value,version:2});await settle();
});

test('preview boxes stay inside the region: a move or a corner stops at its edges, and a lasso keeps their centres',async t=>{
  const {panel,fire}=fixture(t,{open:null});
  const changes=[];
  panel.showPreview({patches:[{id:1,box:[10,10,138,138]},{id:2,box:[200,10,328,138]}],run_pixel_to_sample:[0,0,1,1],
    extent:[400,300]},{onChange:b=>changes.push(b)});
  fire('pointerdown',50,50);fire('pointermove',-60,-40);
  assert.deepEqual(ghost(panel),[0,0,128,128],'the box shown while dragging stops at the edge too');
  fire('pointerup',-60,-40);
  assert.deepEqual(changes.at(-1),[[0,0,128,128],[200,10,328,138]]);
  fire('pointerdown',250,50);fire('pointerup',900,700);
  assert.deepEqual(changes.at(-1)[1],[272,172,400,300],'the right and bottom edges as well');
  fire('pointerdown',128,128);fire('pointerup',500,450);                          // a corner pulled far outside
  assert.deepEqual(changes.at(-1)[0],[0,0,400,300]);
  // A lasso (the engine's outline, in the boxes' pixels): a box whose centre would leave it stays where it was.
  panel.showPreview({patches:[[10,10,74,74]],run_pixel_to_sample:[0,0,1,1],extent:[400,300],
    outline:[[0,0],[400,0],[0,300]]},{onChange:b=>changes.push(b)});
  const before=changes.length;
  fire('pointerdown',40,40);fire('pointermove',240,190);
  assert.deepEqual(ghost(panel),[10,10,74,74]);
  fire('pointerup',240,190);
  assert.equal(changes.length,before,'kept where it was: nothing changed');
  fire('pointerdown',74,74);fire('pointerup',400,300);                            // its corner, too far for the centre
  assert.equal(changes.length,before);
  fire('pointerdown',40,40);fire('pointerup',140,40);
  assert.deepEqual(changes.at(-1),[[110,10,174,74]],'inside the lasso it moves');
  assert.deepEqual(boxGesture([5,10,128,138],[128,10],[2,10],1,{extent:[400,300]}),[5,10,21,138],
    'pulled across its own corner at the edge, a box keeps its smallest side');
  assert.deepEqual(boxGesture([-5,0,60,60],[0,0],[0,0],-1,{extent:[100,100]}),[-5,0,60,60],'never pushed by a press');
});

test('an open run or item keeps its boxes inside its own region: the window of run.box, and its lasso',async t=>{
  const {panel,reads,writes,fire}=fixture(t,{open:{...run,box:[100,50,500,350]}});
  // Pixels of 2 µm from (100, 50): the run's window is 200 x 150 pixels.
  reads[0].resolve({...value,run_pixel_to_sample:[100,50,2,2],patches:[{id:1,box:[10,10,74,74]}]});await settle();
  fire('pointerdown',150,100);fire('pointerup',900,700);
  assert.deepEqual(writes[0].body.boxes,[[136,86,200,150]]);
  writes[0].resolve({...value,run_pixel_to_sample:[100,50,2,2],patches:[{id:1,box:[136,86,200,150]}],version:2});await settle();
  panel.update({...run,id:'lasso',box:[0,0,400,300],region:{kind:'polygon',points:[[0,0],[400,0],[0,300]],bbox:[0,0,400,300]}});
  reads.at(-1).resolve(value);await settle();
  fire('pointerdown',50,50);fire('pointerup',250,200);                            // its centre would leave the lasso
  assert.equal(writes.length,1,'kept where it was: nothing to save');
  fire('pointerdown',50,50);fire('pointerup',100,50);
  assert.deepEqual(writes[1].body.boxes[0],[60,10,188,138]);
  writes[1].resolve({...value,version:2});await settle();
});

test('boxes a few pixels across on screen move by a press inside them: corners and × only where there is room',async t=>{
  const {panel,fire,stage,camera}=fixture(t,{open:null});
  const changes=[],at=k=>{Object.assign(stage,{toScreen:(x,y)=>[x*k,y*k],toWorld:(x,y)=>[x/k,y/k]});stage.camera.zoom=k;camera();};
  at(5/256);                                                                      // a 256-pixel box is 5 CSS px across
  panel.showPreview({patches:[[0,0,256,256],[512,0,768,256]],run_pixel_to_sample:[0,0,1,1],extent:[4000,4000]},
    {onChange:b=>changes.push(b)});
  assert.equal(panel.boxes.children.length,4,'each box: its outline and number, no corners or ×');
  assert.equal(fire('pointerdown',4,1),true);fire('pointerup',9,6);               // beside its top right corner
  assert.deepEqual(changes.at(-1),[[256,256,512,512],[512,0,768,256]],'moved, not resized or dropped');
  at(30/256);                                                                     // 30 CSS px: corners, still no ×
  assert.equal(panel.boxes.children.length,12);
  fire('pointerdown',45,45);fire('pointerup',50,45);                              // where its × would be taken
  assert.deepEqual(changes.at(-1),[[299,256,555,512],[512,0,768,256]],'moved, not dropped');
  fire('pointerdown',90,30);fire('pointerup',100,40);                             // a corner resizes again
  assert.deepEqual(changes.at(-1),[[299,256,555,512],[512,0,853,341]]);
  at(60/256);                                                                     // room for the × too
  assert.equal(panel.boxes.children.length,14);
  fire('pointerdown',190,10);fire('pointerup',190,10);
  assert.deepEqual(changes.at(-1),[[299,256,555,512]],'its × drops it');
});

test('the picker preview: numbered boxes with the item gestures, kept here and reported after each change',async t=>{
  const {panel,reads,writes,fire,stage,camera}=fixture(t,{open:null});
  const changes=[];
  assert.equal(panel.previewBoxes(),null);assert.equal(panel.snapshot().preview,null);
  panel.showPreview({patches:[{id:1,box:[10,10,138,138]},{id:2,box:[200,10,328,138]}],run_pixel_to_sample:[0,0,1,1]},
    {onChange:boxes=>changes.push(boxes)});
  assert.equal(panel.svg.style.display,'block');assert.equal(panel.boxes.children.length,14);
  assert.deepEqual(panel.previewBoxes(),[[10,10,138,138],[200,10,328,138]]);
  assert.deepEqual(panel.snapshot().preview,panel.previewBoxes());
  assert.equal(fire('pointerdown',50,50),true);fire('pointermove',60,55);
  assert.equal(panel.ghost.getAttribute('visibility'),'visible');
  fire('pointerup',60,55);
  assert.deepEqual(changes,[[[20,15,148,143],[200,10,328,138]]]);
  fire('pointerdown',148,143);fire('pointermove',160,180);fire('pointerup',160,180);     // a corner resizes
  assert.deepEqual(changes.at(-1),[[20,15,160,180],[200,10,328,138]]);
  fire('pointerdown',153,22);fire('pointerup',153,22);                                    // × drops the box
  assert.deepEqual(changes.at(-1),[[200,10,328,138]]);
  assert.deepEqual(panel.previewBoxes(),[[200,10,328,138]]);
  assert.equal(panel.boxes.children.length,7);assert.equal(panel.boxes.children[1].textContent,'1');
  fire('pointerdown',250,50);fire('pointerup',250,50);
  assert.equal(changes.length,3,'a click without a move changes nothing');
  assert.equal(fire('pointerdown',500,300),false,'outside the boxes the picker keeps its own region gesture');
  assert.equal(reads.length+writes.length,0,'nothing is asked of the engine');
  stage.camera.zoom=2;camera();
  assert.equal(panel.svg.getAttribute('viewBox'),'200 100 400 200','the boxes follow the camera');
  panel.clearPreview();
  assert.equal(panel.previewBoxes(),null);assert.equal(panel.snapshot().preview,null);
  assert.equal(panel.boxes.children.length,0);assert.equal(panel.svg.style.display,'none');
  assert.equal(fire('pointerdown',250,50),false);fire('pointerup',250,50);
  assert.equal(changes.length,3,'cleared: no gesture and no report any more');
});

test('a new preview replaces the last one; opening a run clears it and the boxes go back to the engine',async t=>{
  const {panel,reads,writes,fire}=fixture(t,{open:null});
  const changes=[],geometry=[100,50,2,2];
  panel.showPreview({patches:[{id:1,box:[0,0,64,64]}],run_pixel_to_sample:geometry},{onChange:b=>changes.push(b)});
  fire('pointerdown',120,70);
  panel.showPreview({patches:[[10,10,74,74],[80,10,144,74]],run_pixel_to_sample:geometry},{onChange:b=>changes.push(b)});
  fire('pointerup',150,100);
  assert.equal(changes.length,0,'a gesture begun on the old boxes is dropped with them');
  assert.deepEqual(panel.previewBoxes(),[[10,10,74,74],[80,10,144,74]]);
  fire('pointerdown',130,80);fire('pointerup',140,90);
  assert.deepEqual(changes,[[[15,15,79,79],[80,10,144,74]]],'pixel boxes in the preview frame');
  panel.update(run);
  assert.equal(panel.previewBoxes(),null);assert.equal(panel.snapshot().preview,null);
  reads[0].resolve(value);await settle();
  assert.equal(panel.boxes.children.length,14);
  fire('pointerdown',50,50);fire('pointerup',60,55);
  assert.equal(changes.length,1);assert.deepEqual(writes[0].body.boxes,[[20,15,148,143],[200,10,328,138]]);
});

test('the picker keeps its own handles and Shift; the lasso rule is the whole box; a click with Add a box draws nothing',async t=>{
  const {panel,reads,writes,errors,fire,stage}=fixture(t);
  // The picker's region handles and Shift-pan come first: such presses are left to the page.
  const changes=[];
  panel.reset();
  panel.showPreview({patches:[{id:1,box:[100,100,200,200]}],run_pixel_to_sample:[0,0,1,1],extent:[400,300]},
    {onChange:b=>changes.push(b),yields:e=>e.offsetX===150});
  fire('pointerdown',150,150);fire('pointermove',170,150);fire('pointerup',170,150);
  assert.equal(changes.length,0);                               // claimed by the page's handle
  stage.canvas.emit('pointerdown',{button:0,pointerId:1,offsetX:160,offsetY:150,shiftKey:true,stopImmediatePropagation(){}});
  fire('pointermove',180,150);fire('pointerup',180,150);
  assert.equal(changes.length,0);                               // Shift pans
  fire('pointerdown',160,160);fire('pointermove',180,160);fire('pointerup',180,160);
  assert.deepEqual(changes.at(-1),[[120,100,220,200]]);
  // A lasso: a move keeping the centre inside but a corner outside is refused (the rule Start and the engine apply).
  panel.showPreview({patches:[{id:1,box:[0,0,64,64]}],run_pixel_to_sample:[0,0,1,1],extent:[300,300],
    outline:[[0,0],[300,0],[0,300]]},{onChange:b=>changes.push(b)});
  const before=changes.length;
  fire('pointerdown',30,30);fire('pointermove',180,90);fire('pointerup',180,90);
  assert.equal(changes.length,before);                           // centre (182, 92) inside, corner (214, 124) not
  fire('pointerdown',30,30);fire('pointermove',80,40);fire('pointerup',80,40);
  assert.deepEqual(changes.at(-1),[[50,10,114,74]]);
  // An open item: a click with Add a box draws nothing and says how.
  panel.clearPreview();
  panel.update({id:'item',kind:'patches',state:'done',progress:{done:0,revision:0},box:[0,0,400,300]});
  reads.at(-1).resolve({...value,version:2});await settle();
  panel.add.click();
  fire('pointerdown',300,250);fire('pointerup',300,250);
  assert.equal(writes.filter(w=>w.op==='boxes').length,0);
  assert.match(errors.at(-1),/at least 16 pixels/);
});

// ---- the boxes by keyboard (the image's boxes take the pointer only): a row a box, with its × ----------------------------
// A key pressed on a row (or on its ×, given as `target`: the key reaches the row), as the browser gives it; `stopped`: it
// went no further than the row.
const key=(row,name,extra={})=>row.emit('keydown',{key:name,altKey:false,shiftKey:false,ctrlKey:false,metaKey:false,
  preventDefault(){this.prevented=true;},stopPropagation(){this.stopped=true;},...extra});
const rowWords=list=>list.children.map(row=>row.firstChild.textContent);

test('an open run\'s boxes by keyboard: arrows move, Shift and arrows resize, Delete or × drops, each one change to the engine',async t=>{
  const {panel,reads,writes,document}=fixture(t,{open:{...run,box:[0,0,800,400]}});reads[0].resolve(value);await settle();
  const list=panel.rows,first=list.children[0];
  assert.equal(list.hidden,false);assert.equal(list.getAttribute('role'),'list');assert.equal(panel.hint.nextSibling,list);
  assert.deepEqual(rowWords(list),['Patch 1 · 128 × 128 px','Patch 2 · 128 × 128 px']);
  assert.equal(first.getAttribute('tabindex'),'0');assert.equal(first.getAttribute('role'),'listitem');
  assert.equal(first.getAttribute('aria-label'),`Patch 1 · 128 × 128 px. ${STAIN_PATCH_TEXT.keys}`);
  assert.equal(first.title,STAIN_PATCH_TEXT.keys);assert.equal(first.lastChild.getAttribute('aria-label'),'Drop patch 1');
  first.focus();
  assert.ok(key(first,'ArrowRight').prevented);
  assert.deepEqual(writes[0].body,{version:1,boxes:[[26,10,154,138],[200,10,328,138]]});
  key(first,'ArrowRight');assert.equal(writes.length,1,'a key while the engine works on the last change makes none');
  writes[0].resolve({...value,version:2,patches:[{id:1,box:[26,10,154,138]},{id:2,box:[200,10,328,138]}]});await settle();
  assert.equal(list.children[0],first,'the same boxes: the rows are updated in place, the focus stays');
  assert.equal(document.activeElement,first);
  key(first,'ArrowDown',{shiftKey:true});assert.deepEqual(writes[1].body.boxes[0],[26,10,154,154],'Shift: its bottom-right corner');
  writes[1].resolve({...value,version:3,patches:[{id:1,box:[26,10,154,154]},{id:2,box:[200,10,328,138]}]});await settle();
  assert.equal(first.firstChild.textContent,'Patch 1 · 128 × 144 px');
  key(first,'ArrowUp',{altKey:true});assert.deepEqual(writes[2].body.boxes[0],[26,9,154,153],'Alt: by one pixel');
  writes[2].resolve({...value,version:4,patches:[{id:1,box:[26,9,154,153]},{id:2,box:[200,10,328,138]}]});await settle();
  key(first,'ArrowUp');assert.deepEqual(writes[3].body.boxes[0],[26,0,154,144],'a move stops at the region\'s edge');
  writes[3].resolve({...value,version:5,patches:[{id:1,box:[26,0,154,144]},{id:2,box:[200,10,328,138]}]});await settle();
  for(const [name,extra] of [['a',{}],['Enter',{}],['ArrowRight',{ctrlKey:true}],[' ',{target:first.lastChild}]]) {
    const e=key(first,name,extra);assert.equal(e.prevented,undefined,name);assert.equal(e.stopped,undefined,name);
  }
  assert.equal(writes.length,4,'other keys are the page\'s, and Space on the × presses it as ever');
  // Space on the row is the row's: taken (main.js's Space, Resume of the paused run, skips a key taken) and does nothing.
  for(const extra of [{},{shiftKey:true}]){const e=key(first,' ',extra);assert.ok(e.prevented && e.stopped);}
  assert.equal(writes.length,4);
  // Delete drops the box; the focus goes to the row now in its place.
  key(first,'Delete');assert.deepEqual(writes[4].body.boxes,[[200,10,328,138]]);
  writes[4].resolve({...value,version:6,patches:[{id:1,box:[200,10,328,138]}]});await settle();
  assert.deepEqual(rowWords(list),['Patch 1 · 128 × 128 px']);assert.equal(document.activeElement,list.children[0]);
  list.children[0].lastChild.click();assert.deepEqual(writes[5].body.boxes,[],'its × drops it too');
  writes[5].resolve({...value,version:7,patches:[]});await settle();
  assert.equal(list.hidden,true);assert.equal(list.children.length,0);
  // While the run runs again its boxes cannot change: no rows, and keys make nothing.
  writes.length=0;panel.apply({...value,version:8});
  const row=list.children[0];panel.update({...run,box:[0,0,800,400],state:'running'});
  assert.equal(list.hidden,true);key(row,'ArrowRight');assert.equal(writes.length,0);
});

test('the picker preview by keyboard: rows in its Training patches section, each change reported and nothing asked of the engine',async t=>{
  const {panel,reads,writes,document}=fixture(t,{open:null}),changes=[];
  panel.showPreview({patches:[{id:1,box:[10,10,138,138]},{id:2,box:[200,10,328,138]}],run_pixel_to_sample:[0,0,1,1],extent:[800,400]},
    {onChange:b=>changes.push(b)});
  const list=document.getElementById('stain-patch-preview-rows');
  assert.equal(list.parentElement.id,'patch-picker');assert.equal(list.hidden,false);assert.equal(panel.rows.hidden,true);
  assert.deepEqual(rowWords(list),['Patch 1 · 128 × 128 px','Patch 2 · 128 × 128 px']);
  key(list.children[0],'ArrowRight');assert.deepEqual(changes.at(-1),[[26,10,154,138],[200,10,328,138]]);
  key(list.children[0],'ArrowDown',{shiftKey:true});assert.deepEqual(changes.at(-1),[[26,10,154,154],[200,10,328,138]]);
  assert.deepEqual(rowWords(list),['Patch 1 · 128 × 144 px','Patch 2 · 128 × 128 px']);
  // Backspace drops the box and goes no further: with the picker's outline being drawn, it takes back none of its corners.
  const second=list.children[1];second.focus();assert.ok(key(second,'Backspace').stopped);
  assert.deepEqual(changes.at(-1),[[26,10,154,154]]);assert.deepEqual(panel.previewBoxes(),[[26,10,154,154]]);
  assert.equal(document.activeElement,list.children[0],'the last row dropped: the focus goes to the one before');
  list.children[0].lastChild.click();assert.deepEqual(changes.at(-1),[]);assert.equal(list.hidden,true);
  // A lasso: a move that would take a corner out of it is refused, as with the pointer; a smaller step that keeps it is not.
  panel.showPreview({patches:[{id:1,box:[80,80,144,144]}],run_pixel_to_sample:[0,0,1,1],extent:[300,300],
    outline:[[0,0],[300,0],[0,300]]},{onChange:b=>changes.push(b)});
  const before=changes.length;key(list.children[0],'ArrowRight');assert.equal(changes.length,before);
  key(list.children[0],'ArrowRight',{altKey:true});assert.deepEqual(changes.at(-1),[[81,80,145,144]]);
  assert.equal(reads.length+writes.length,0);
  panel.clearPreview();assert.equal(list.hidden,true);assert.equal(list.children.length,0);
});

test('a Cellpose GUI that closed with an error is said once by its pid, and one without its packages leaves the command that installs them',async t=>{
  const {panel,reads,writes,errors}=fixture(t);
  const failed={pid:41,code:1,reason:"KeyError: '$&' in $` x"};
  const status=gui_error=>({...value,set:'abc',exported:{folder:'/tmp/p',states:[],gui_running:false,gui_error}});
  reads[0].resolve(status(failed));await settle();
  assert.deepEqual(errors,[`The Cellpose GUI closed with an error: ${failed.reason}`]);   // its own words, $ and all
  panel.apply(status(failed));panel.apply(status(null));                // the same GUI's again: not said twice
  panel.apply(status({...failed,pid:42,reason:''}));                    // another one, its log empty
  panel.apply(status(failed));                                          // the first one's after another's: still once
  assert.deepEqual(errors.slice(1),[STAIN_PATCH_TEXT.guiClosedPlain]);
  const command="/env/bin/python -m pip install 'cellpose[gui]'";
  const refused=press(panel.open);
  writes.at(-1).reject(Object.assign(new Error('The Cellpose GUI is not installed where Cellpose runs.'),{status:503,data:{command}}));
  await refused;await settle();
  assert.equal(panel.command.textContent,`${STAIN_PATCH_TEXT.install}\n${command}`,'kept under the button to copy');
  const opened=press(panel.open);
  writes.at(-1).resolve({folder:'/tmp/p',command:'env x',running:true});await opened;await settle();
  assert.equal(panel.command.textContent,'','gone once a GUI opens');
});
