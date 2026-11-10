import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import * as geometry from '../src/region.mjs';
import { LassoDraft, mergedMoves } from '../src/lasso.mjs';
import * as priorLogic from '../src/prior-options.mjs';
import * as stain from '../src/stain.mjs';
import { updateNormalizeFields } from '../src/stain-settings.mjs';
import { STAIN_PATCH_TEXT, patchSizeNote } from '../src/stain-patches.mjs';
import { EXPLAIN_TEXT, NEEDS_CLASS } from '../src/explain.mjs';
import { SHELL_TEXT, TOOLS, toolOf, pending, leftoversNotice } from '../src/shell.mjs';
import { olderRows, olderCount } from '../src/older-runs.mjs';
import { apiSession } from '../src/session.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { COMPARE_TEXT } from '../src/compare-copy.mjs';
import { NOTE_TEXT } from '../src/sample-notes.mjs';
import { domFixture, deferred, settle } from './dom-fixture.mjs';

const source = readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
const html = readFileSync(new URL('../index.html',import.meta.url),'utf8');
// explain.mjs evaluated in the page's context, so its notes and outlines use the page's document; sample-notes.mjs too
// (the picker's notes on the sample).
const explainSource = readFileSync(new URL('../src/explain.mjs',import.meta.url),'utf8').replace(/^export /gm,'');
const notesSource = readFileSync(new URL('../src/sample-notes.mjs',import.meta.url),'utf8').replace(/^export /gm,'');
const sample = {id:'sample',kind:'zarr',name:'Offset sample',transcripts:4000000,extent:[100,-200,2100,1800],
  overview:{x0:80,y0:-230,width:1100,height:1050,um_per_px:2},shapes:[{key:'cells'}],limits:{detail_um:1200},
  prior_options:{cells:[{kind:'shapes',key:'cells',label:'Cell boundaries'}],nuclei:[{kind:'column',column:'nuclear',nuclear:true,label:'Nuclear transcripts'}]},
  suggested_prior:{cells:{kind:'shapes',key:'cells'},nuclei:{kind:'column',column:'nuclear',nuclear:true}}};
const plain = value => JSON.parse(JSON.stringify(value));
const models = [
  {name:'cpsam_v2',kind:'builtin',date:null}, {name:'cpdino',kind:'builtin',date:null},
  {name:'cpdino-vitb',kind:'builtin',date:null}, {name:'cpsam',kind:'builtin',date:null},
  {name:'cpsam_20250102',kind:'custom',date:'2025-01-02'},
];

function harness({samples=[],runs=[],search=''}={}) {
  const dom=domFixture();dom.document.body.innerHTML=html;dom.document.querySelectorAll=()=>[];
  const canvas=dom.$('stage'),overlay=dom.$('overlay'),timers=new Map(),delays=new Map(),notices=[],opened=[],created=[],paths=[],pickerLines=[];
  // The Training patches preview: the engine's answers wait until a test gives them; the panel (stain-patches.mjs) keeps
  // the boxes it was shown, as moved or dropped there, and which presses it leaves to the page (`yields`).
  const previews=[],panel={shown:[],clears:0,boxes:null,reset(){},
    showPreview(result,{onChange,yields}={}){this.shown.push(result);this.onChange=onChange;this.yields=yields;this.boxes=result.patches.map(p=>[...p.box]);},
    previewBoxes(){return this.boxes&&this.boxes.map(b=>[...b]);},clearPreview(){this.clears++;this.boxes=null;}};
  let timer=0,timerDelay;
  const strokes=[];
  const g=Object.fromEntries(['setTransform','clearRect','setLineDash','beginPath','closePath','fill','stroke','fillRect','rect','lineTo','moveTo'].map(key=>[key,(...args)=>strokes.push([key,...args])]));
  for(const c of [canvas,overlay]){c.clientWidth=1000;c.clientHeight=800;c.getContext=()=>g;}
  canvas.setPointerCapture=()=>{};canvas.hasPointerCapture=()=>true;canvas.releasePointerCapture=()=>{};
  // The stage caches its CSS size and pixel ratio (main.js never reads layout for them).
  const stage={canvas,scale:1,cssWidth:1000,cssHeight:800,ratio:1,options:{mode:0},orbit:{active:false},set(){},setFrame(){},resize(){},onChange(){},
    toWorld(x,y){return [100+x/this.scale,-200+y/this.scale];},toScreen(x,y){return [(x-100)*this.scale,(y+200)*this.scale];},
    setImage(rect){this.image=rect;},fit(rect){this.fitted=rect;},panBy(x,y){this.pan=[x,y];},zoomAt(){}};
  const state={health:{proseg:true,default_region_um:300},samples:[],runs:[],sample:null,mode:'empty',tool:null,region:null,prior:{cells:null,nuclei:null},frames:[],run:null,current:-1,display:{},hover:0};
  const api={health:async()=>state.health,samples:async()=>samples,runs:async()=>runs,options:async()=>[],optionsGuide:async()=>[],
    sample:async()=>sample,overviewUrl:()=>'/overview.png',shapes:async()=>({polygons:[]}),forgetSample:async id=>paths.push(id),
    image:async url=>({src:url}),          // an image the engine serves, fetched and decoded (api.js image)
    sampleQc:async()=>({state:'learning',eta_s:90}),
    createRun:async(...args)=>{created.push(args);return{id:'new-run'};},
    previewPatches:(sample,region,stain,signal)=>{const job=deferred();previews.push({sample,region,stain,signal,...job});return job.promise;}};
  // The prior's layers, and the sample's segmentation layers: the samples the picker hands their store.
  const priors={reset(){this.vectors=null;},segments:{samples:[],setSample(id){this.samples.push(id);}}};
  const context=vm.createContext({...geometry,LassoDraft,mergedMoves,...priorLogic,...stain,STAIN_PATCH_TEXT,patchSizeNote,updateNormalizeFields,$:dom.$,document:dom.document,window:{devicePixelRatio:1,addEventListener(){}},
    SHELL_TEXT,TOOLS,toolOf,pending,leftoversNotice,
    state,stage,overlay,api,priors,opening:0,ownOpens:0,pickerLoading:false,lastPointer:null,
    fmt:new Intl.NumberFormat('en-GB'),fmt1:new Intl.NumberFormat('en-GB',{maximumFractionDigits:1}),
    ACTIVE:['preparing','running','paused','finishing'],URLSearchParams,AbortController,
    location:{pathname:'/',search},history:{replaceState(){}},localStorage:{getItem:()=>sample.id,setItem(){}},
    Option:function(label,value,defaultSelected=false,selected=false){const el=new dom.Element('option');el.textContent=label;el.value=value;el.selected=selected;return el;},
    Image:class{async decode(){}},requestAnimationFrame:fn=>fn(),setInterval(){},
    setTimeout:(fn,delay)=>{timerDelay=delay;timers.set(++timer,fn);delays.set(timer,delay);return timer;},clearTimeout:id=>{timers.delete(id);delays.delete(id);},
    resetRequests(){},stopReplay(){},clearFrames(){},resetCellHistory(){},updateActiveCells(){},updateTooltip(){},updateCellCard(){},
    setText(el,text){el.textContent=text;},setProp(el,key,value){el[key]=value;},
    analysis:{reset(){},tissueHover(){}},masks:{reset(){}},cellStats:{reset(){}},transcripts:{reset(){}},
    stainView:{reset(){}},stainPatches:panel,tiled:{reset(){}},
    stainExclusions:{reset(){},draw(){}},tissueView:{polygons:[],update(){}},pickerSettings:{refresh(){}},patchSettings:{refresh(){}},
    options:{schema:[],load(){},getValues:()=>({}),problems:()=>[]},knobs:{render(){},guide:{load(){}}},pickerGuide:{load(){}},
    layerPanel:{setChannels(){},save(){},render(){}},timeline:{setFrames(){},setPlan(){},setCurrent(){},refresh(){}},
    setView(){},setMode(){},finishAndSave(){},control(){},setLive(){},startReplay(){},step(){},show(){},
    leave:then=>then(null),notice:message=>notices.push(message),onMessage(){},zarrSave:{pending:new Map()},
    // The older runs' prompt (none here) and a notebook's session (none: the desktop).
    olderPrompt:{open:false,ask:async()=>{}},olderRows,olderCount,apiSession,setSession(session){state.session=session;},
    // The tab's sign-in (page-key.mjs): signed in; the socket and the close prompt a page signed out stops and resets.
    pageKey:{ready:Promise.resolve('k'),reason:()=>'signed-in'},socket:{stop(){}},closePopup:{reset(){}},
    sampleBrowser:{show:path=>paths.push(path)},openRun:async id=>opened.push(id),
    // The mixing check (main.js renderQc and the overlay's areas) has nothing to show while a sample is picked.
    resetQc(){},updateQcLegend(){},qcBins(){},qcOverlay:{draw(){}},qcChip:{visible:false},pickerQc:qc=>pickerLines.push(qc?.state??null),
    // Compare: closed as the picker opens; its header button's words (the real deck), never opened here.
    COMPARE_TEXT,compareView:{closes:[],close(tell){this.closes.push(tell);}},cover(){},openCompare(){opened.push('compare');},
  });
  const slices=[source.slice(source.indexOf('function setSample(sample)'),source.indexOf('function step(delta)')),
    source.slice(source.indexOf('function setupControls()'),source.indexOf('function resizeTranscripts(')),
    source.slice(source.indexOf('function segmented('),source.indexOf('let noticeTimer')),
    source.slice(source.indexOf('function layout()'),source.indexOf('window.sopaCompanion ='))];
  vm.runInContext(explainSource+'\n'+notesSource+'\n'+slices.join('\n'),context);
  return {...dom,state,stage,api,priors,notices,opened,created,paths,strokes,context,pickerLines,previews,panel,run:code=>vm.runInContext(code,context),
    timerDelay:()=>timerDelay,
    tick(){const [id,fn]=[...timers].at(-1);timers.delete(id);return fn();},
    /** The timers set for `ms` (the patch preview's 400 ms) still to fire; fire() runs them, oldest first. */
    pending:ms=>[...delays.values()].filter(d=>d===ms).length,
    fire(ms){const due=[...timers].filter(([id])=>delays.get(id)===ms);for(const [id,fn] of due){timers.delete(id);delays.delete(id);fn();}},
    /** What a press of Start would list: each problem's field id, or its words when it has none. */
    problems(){return plain(vm.runInContext('pickerProblems()',context)).map(p=>p.field||p.message);},
    /** The note a press left beside `id` (explain.mjs), as the user reads it, or '' while none is shown. */
    note(id='start'){const next=dom.$(id).nextSibling;return next?.className==='missing-note'&&!next.hidden?next.textContent:'';},
    pointer(type,x,y,extra={}){canvas.emit(type,{offsetX:x,offsetY:y,pointerId:1,button:0,shiftKey:false,...extra});}};
}

/** Stain segmentation's picker on the sample. `instanseg`: the engine's internal switch on (health lists InstanSeg, which
 *  is dormant otherwise), so the Method choice offers it as well. */
async function stainPicker(saved=null,{instanseg=false}={}) {
  const h=harness(),storage=new Map(saved?[['sopa-companion-stain-model',saved]]:[]);
  h.context.localStorage={getItem:key=>storage.get(key)??null,setItem:(key,value)=>storage.set(key,value)};
  h.api.stainModels=async()=>({models});
  h.api.createStain=async(...args)=>{h.created.push(args);return{id:'new-stain'};};
  h.state.health.stain={available:true,...(instanseg?{methods:{cellpose:true,instanseg:true}}:{})};
  await h.run('enterPick("sample")');h.run('setupControls()');h.run('offerStainMethods()');
  h.state.sample.images=[{key:'image',channels:['DAPI','Boundary']}];
  h.$('run-kind').value='stain';
  h.$('stain-method').value='cellpose';h.$('stain-output').value='cells';
  Object.defineProperty(h.$('stain-channels'),'selectedOptions',{get(){return this.children.filter(o=>o.selected);}});
  h.$('stain-flow').value='.4';h.$('stain-probability').value='0';
  h.run('fillStainImages(state.sample);updatePicker()');
  await new Promise(resolve=>setImmediate(resolve));
  return {...h,storage};
}

test('normalisation controls switch between fixed intensities and percentiles and send an explicit off flag',async()=>{
  const h=await stainPicker();
  h.$('stain-fixed-range').checked=true;h.$('stain-fixed-range').emit('change');
  assert.equal(h.$('stain-low').disabled,true);assert.equal(h.$('stain-intensity-low').disabled,false);
  h.$('stain-intensity-low').value='-2';h.$('stain-intensity-high').value='4095';h.$('stain-intensity-high').emit('input');
  assert.deepEqual(plain(h.run('readStain().normalize.lowhigh')),[-2,4095]);
  assert.equal(h.run('readStain().normalize.percentile'),null);
  h.$('stain-normalize').checked=false;h.$('stain-normalize').emit('change');
  assert.equal(h.run('readStain().normalize.normalize'),false);assert.equal(h.$('stain-intensity-low').disabled,true);
  h.$('stain-normalize').checked=true;h.$('stain-normalize').emit('change');
  h.$('stain-fixed-range').checked=false;h.$('stain-fixed-range').emit('change');
  assert.equal(h.$('stain-low').disabled,false);assert.equal(h.$('stain-intensity-low').disabled,true);
  assert.deepEqual(plain(h.run('readStain().normalize.percentile')),[1,99]);
  assert.equal(h.run('readStain().normalize.lowhigh'),null);
});

test('stain picker defaults to cpsam and groups dated custom models after the installed builtins',async()=>{
  const h=await stainPicker(),select=h.$('stain-model');
  assert.equal(select.value,'cpsam');
  assert.deepEqual(select.children.slice(0,4).map(n=>n.value),models.slice(0,4).map(m=>m.name));
  const group=select.children[4];
  assert.equal(group.tagName,'OPTGROUP');assert.equal(group.label,'Your models');
  assert.equal(group.children[0].textContent,'cpsam_20250102 · 2025-01-02');
  assert.deepEqual(bannedWords(group.label+' '+group.textContent),[]);
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),[]);
});

test('stain picker remembers and submits only the chosen name, and falls back when it disappears',async()=>{
  const h=await stainPicker('cpdino'),select=h.$('stain-model');
  assert.equal(select.value,'cpdino');
  select.value='cpsam_20250102';select.emit('change');
  assert.equal(h.storage.get('sopa-companion-stain-model'),'cpsam_20250102');
  await h.$('start').click();
  assert.equal(h.created[0][2].model,'cpsam_20250102');
  assert.equal('model_path' in h.created[0][2],false);
  assert.deepEqual(h.opened,['new-stain']);
  select.value='';
  await h.run('loadStainModels()');
  assert.equal(select.value,'cpsam_20250102');
  h.api.stainModels=async()=>({models:models.slice(0,4)});
  await h.run('loadStainModels()');
  assert.equal(select.value,'cpsam');
});

test('a custom model trained on patches cut at a smaller copy says which by its name, as the models route sends it',async()=>{
  const h=await stainPicker();
  // The route's entries: `resolution` {level, factor} for a model a patch set's GUI save names, null otherwise.
  h.api.stainModels=async()=>({models:[...models.map(m=>({...m,resolution:null})),
    {name:'cp_learned',kind:'custom',date:'2026-10-21',resolution:{level:1,factor:2}},
    {name:'cp_stored',kind:'custom',date:'2026-10-22',resolution:{level:0,factor:1}}]});
  await h.run('loadStainModels()');
  const group=h.$('stain-model').children[4];
  assert.deepEqual(group.children.map(o=>o.textContent),['cpsam_20250102 · 2025-01-02',
    'cp_learned · 2026-10-21 · learned at scale1','cp_stored · 2026-10-22']);
});

test('the picker names Cellpose\'s settings as Cellpose does, the plain words and help in the label\'s and field\'s tooltip',async()=>{
  const h=harness();await h.run('enterPick("sample")');
  // The page's [data-stain-copy] nodes, as setupControls asks the document for them (stain.mjs stainCopy).
  const copy=h.document.body.all().filter(n=>n.attributes?.has('data-stain-copy'));
  for(const node of copy)node.dataset.stainCopy=node.getAttribute('data-stain-copy');
  h.document.querySelectorAll=selector=>selector==='[data-stain-copy]'?copy:[];
  h.run('setupControls()');
  for(const [id,name,title] of [['stain-flow','flow_threshold','Shape check. Lower values reject more uncertain shapes.'],
    ['stain-probability','cellprob_threshold','Cell threshold. Lower values find more and larger cells.'],['stain-min-size','min_size','Smallest cell']]) {
    const field=h.$(id),label=field.closest('label');
    assert.deepEqual([label.textContent,label.title,field.title],[name,title,title],id);
  }
  assert.equal(h.$('stain-tissue').closest('label').textContent,stain.STAIN_TEXT.tissue,'other controls keep their words');
});

test('stain picker says while models load or cannot be read, never greys Start out, and retries on reopening',async()=>{
  const h=await stainPicker(),pending=deferred(),P=stain.PICKER_TEXT,T=stain.STAIN_TEXT;
  // A first list that comes: the spinner and its words go when it does.
  const first=deferred();
  h.state.stainModels=null;h.api.stainModels=()=>first.promise;h.$('stain-model').replaceChildren();h.run('loadStainModels.key=null');
  const firstLoad=h.run('loadStainModels()');
  assert.equal(h.$('stain-model-loading').hidden,false,'a spinner while the first list loads');
  first.resolve({models});await firstLoad;
  assert.equal(h.$('stain-model-loading').hidden,true,'no spinner once the list came');
  assert.ok(h.$('stain-model').value && !h.$('stain-model').children.some(o=>o.textContent===T.modelsLoading));
  // As a first load, or one after the method changed: no list yet (main.js stain-method's change).
  h.state.stainModels=null;h.api.stainModels=()=>pending.promise;h.$('stain-model').replaceChildren();h.run('loadStainModels.key=null');
  const loading=h.run('loadStainModels()');h.run('updatePicker()');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['stain-model']);
  assert.equal(h.$('picker-hint').textContent,T.modelsLoading,'what the picker waits for still shows');
  // Beside the list a spinner, and in the empty list its words (the first answer can take a while: Cellpose's import).
  const select=h.$('stain-model'),spinner=h.$('stain-model-loading');
  assert.equal(spinner.hidden,false,'a spinner beside the list while it loads');
  assert.deepEqual(select.children.map(o=>[o.textContent,o.value,o.disabled]),[[T.modelsLoading,'',true]]);
  await h.$('start').click();
  assert.ok(h.note().includes(`${T.model}: ${P.modelsLoading}`));assert.equal(h.created.length,0);
  pending.reject(Error('offline'));await loading;
  assert.deepEqual(h.problems(),['stain-model']);
  assert.equal(h.$('picker-hint').textContent,T.modelsError);
  assert.equal(spinner.hidden,true);assert.equal(select.children.length,0,'no list: neither the spinner nor its words');
  await h.$('start').click();assert.ok(h.note().includes(`${T.model}: ${P.modelsError}`));
  h.api.stainModels=async()=>({models});
  await h.run('loadStainModels()');
  assert.deepEqual(h.problems(),[]);assert.equal(h.note(),'','the list goes once nothing is missing');
  assert.equal(spinner.hidden,true);assert.ok(select.children.length>0 && select.value,'the models, one chosen');
  assert.ok(!select.children.some(o=>o.textContent===T.modelsLoading),'and no loading words among them');
  assert.ok(!h.$('stain-model').classList.contains(NEEDS_CLASS));
});

/** Every word a user can reach on the page: each node's own text, an option's value, and the attributes that are read or
 *  shown (titles, labels, placeholders), never an element's id, class or data- name. */
function reachable(root) {
  const texts=[];
  for(const node of root.all()) {
    texts.push(node.text||'',node.title||'',node.placeholder||'',node.label||'',node.alt||'',node.tagName==='OPTION'?node.value:'');
    for(const [key,value] of node.attributes||[])if(!['id','class'].includes(key) && !key.startsWith('data-'))texts.push(value);
  }
  return texts.filter(Boolean);
}

test('InstanSeg is dormant: Cellpose alone, no Method row, no InstanSeg word on the page, and Start sends Cellpose',async()=>{
  const h=await stainPicker(),calls=[];
  h.api.stainModels=async method=>{calls.push(method);return {models};};
  h.state.health.stain.methods={cellpose:true};             // what an engine with the switch off lists
  h.$('stain-method').value='instanseg';                    // a stale choice: never kept
  h.run('offerStainMethods();updateStainMethod()');
  assert.deepEqual(h.$('stain-method').children.map(o=>o.value),['cellpose']);
  assert.equal(h.$('stain-method').value,'cellpose');
  assert.equal(h.$('stain-method').parentElement.hidden,true,'no Method row');
  assert.equal(h.$('stain-instanseg-options').hidden,true,'no Channels, Find or Match cells and nuclei');
  assert.equal(h.$('stain-cellpose-options').hidden,false);
  await h.run('loadStainModels()');
  assert.deepEqual(h.$('stain-model').all().filter(n=>n.tagName==='OPTION').map(o=>o.value),models.map(m=>m.name),
    'Cellpose\'s models only (yours in their group)');
  assert.deepEqual(calls,['cellpose']);
  assert.ok(reachable(h.document.body).some(text=>/Cellpose/.test(text)),'the page\'s words are read');
  assert.deepEqual(reachable(h.document.body).filter(text=>/instanseg/i.test(text)),[]);
  // A press of Start names nothing of InstanSeg either, and sends a Cellpose run.
  h.$('stain-flow').value='11';h.$('stain-flow').emit('input');
  await h.$('start').click();
  assert.doesNotMatch(h.note(),/instanseg/i);
  h.$('stain-flow').value='.4';h.$('stain-flow').emit('input');
  await h.$('start').click();
  assert.equal(h.created.length,1);assert.equal(h.created[0][2].method,'cellpose');
  assert.deepEqual(reachable(h.document.body).filter(text=>/instanseg/i.test(text)),[]);
  // Cellpose's software missing: said in a sentence of its own, never on the Method row the page does not show.
  h.state.health.stain={available:false,methods:{cellpose:false}};
  h.run('updatePicker()');
  assert.deepEqual(plain(h.run('pickerProblems()')).filter(p=>/not available|unavailable/.test(p.message)).map(p=>[p.field,p.message]),
    [[null,stain.STAIN_TEXT.unavailable]]);
  assert.equal(h.$('picker-error').textContent,stain.STAIN_TEXT.unavailable);
});

const instansegModels = ['fluorescence_nuclei_and_cells','single_channel_nuclei','brightfield_nuclei']
  .map(name=>({name,kind:'builtin'}));

test('InstanSeg picker submits channels and output with separate model memory and minimal options',async()=>{
  const h=await stainPicker('cpdino',{instanseg:true}),calls=[];
  h.api.stainModels=async method=>{calls.push(method);return {models:method==='instanseg'?instansegModels:models};};
  h.state.sample.images[0].um_per_px=.2125;
  h.$('stain-method').value='instanseg';h.$('stain-method').emit('change');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.$('stain-model').value,'fluorescence_nuclei_and_cells');
  assert.equal(h.$('stain-cellpose-options').hidden,true);
  assert.equal(h.$('stain-instanseg-options').hidden,false);
  assert.match(h.$('stain-resolution-info').textContent,/0.212 µm per pixel/);
  assert.equal(h.$('start').disabled,false);
  await h.$('start').click();
  assert.deepEqual(plain(h.created[0][2]),{method:'instanseg',model:'fluorescence_nuclei_and_cells',
    image:'image',channels:['DAPI','Boundary'],output:'cells',resolve_cells:false,tissue_only:true,level:0});
  h.$('stain-model').value='single_channel_nuclei';h.$('stain-model').emit('change');
  assert.equal(h.$('stain-output').value,'nuclei');
  assert.equal(h.$('stain-resolve').disabled,true);
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['stain-channels']);
  assert.match(h.run('pickerProblems()[0].message'),/exactly one/);
  h.$('stain-channels').children[1].selected=false;h.$('stain-channels').emit('change');
  assert.deepEqual(h.problems(),[]);
  assert.equal(h.storage.get('sopa-companion-instanseg-model'),'single_channel_nuclei');
  h.$('stain-method').value='cellpose';h.$('stain-method').emit('change');
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.$('stain-model').value,'cpdino');
  assert.deepEqual(calls,['instanseg','cellpose']);
});

test('late model responses cannot cross methods, and a press of Start names an unavailable InstanSeg',async()=>{
  const h=await stainPicker(null,{instanseg:true}),pending=deferred();
  h.api.stainModels=method=>method==='instanseg'?pending.promise:Promise.resolve({models});
  h.$('stain-method').value='instanseg';h.$('stain-method').emit('change');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['stain-model'],'its list is still loading');
  h.$('stain-method').value='cellpose';h.$('stain-method').emit('change');
  await new Promise(resolve=>setImmediate(resolve));
  pending.resolve({models:instansegModels});await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.$('stain-model').value,'cpsam');
  assert.ok(h.$('stain-model').children.length>0);
  assert.equal(h.$('stain-cellpose-options').hidden,false);
  h.state.health.stain.methods={cellpose:true,instanseg:false};
  h.$('stain-method').value='instanseg';h.$('stain-method').emit('change');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['stain-method']);
  assert.equal(h.$('picker-error').textContent,stain.STAIN_TEXT.unavailable);
  await h.$('start').click();
  assert.ok(h.note().includes(`${stain.STAIN_TEXT.method}: ${stain.PICKER_TEXT.unavailable}`));assert.equal(h.created.length,0);
  // Nothing says a list loads for a method whose list cannot load, even when Cellpose's was on its way as it was chosen.
  const slow=deferred(),spinner=h.$('stain-model-loading');h.api.stainModels=()=>slow.promise;
  h.$('stain-method').value='cellpose';h.$('stain-method').emit('change');
  assert.equal(spinner.hidden,false,"Cellpose's list on its way");
  h.$('stain-method').value='instanseg';h.$('stain-method').emit('change');
  assert.equal(spinner.hidden,true,'no spinner for a list that cannot load');
  assert.ok(!h.$('stain-model').children.some(o=>o.textContent===stain.STAIN_TEXT.modelsLoading));
  slow.resolve({models});await new Promise(resolve=>setImmediate(resolve));
  assert.equal(spinner.hidden,true,'nor once the stale list answers');
});

test('model refresh keeps choices and unchanged options, and discovers new GUI models every 15 seconds',async()=>{
  const h=await stainPicker('cpdino',{instanseg:true}),select=h.$('stain-model'),calls=[];
  h.api.stainModels=async method=>{calls.push(method);return {models:method==='instanseg'?instansegModels:models};};
  const first=select.children[0];
  assert.equal(h.run('loadStainModels.pending'),false);
  assert.equal(h.timerDelay(),15000);
  h.tick();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(select.children[0],first);assert.equal(select.value,'cpdino');
  const newer=[...models,{name:'gui-new',kind:'custom',source:'gui',date:'2026-10-20'}];
  h.api.stainModels=async method=>{calls.push(method);return {models:method==='instanseg'?instansegModels:newer};};
  h.tick();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(select.children.at(-1).children.at(-1).value,'gui-new');
  assert.equal(select.value,'cpdino');
  h.$('stain-method').value='instanseg';h.$('stain-method').emit('change');
  await new Promise(resolve=>setImmediate(resolve));
  select.value='single_channel_nuclei';select.emit('change');
  const instansegOption=select.children[0];
  h.tick();await new Promise(resolve=>setImmediate(resolve));
  assert.equal(select.children[0],instansegOption);assert.equal(select.value,'single_channel_nuclei');
  assert.equal(h.timerDelay(),15000);
  assert.deepEqual(calls,['cellpose','cellpose','instanseg','instanseg']);
});

test('boot stays empty with no samples or with recents, and restores only an active or requested run',async()=>{
  for(const samples of [[],[sample]]){
    const h=harness({samples});await h.run('boot()');
    assert.equal(h.state.sample,null);assert.equal(h.state.mode,'empty');assert.equal(h.$('empty').hidden,false);
    assert.equal(h.$('new-run').disabled,false,'never greyed out');assert.deepEqual(h.opened,[]);assert.deepEqual(h.notices,[]);
  }
  const active=harness({runs:[{id:'active',state:'running'}]});await active.run('boot()');assert.deepEqual(active.opened,['active']);
  const wanted=harness({search:'?run=explicit',runs:[{id:'active',state:'running'}]});await wanted.run('boot()');assert.deepEqual(wanted.opened,['explicit']);
});

test('sample overview and fit use the offset geometry; recents reopen by path and forget by id',async()=>{
  const h=harness();await h.run('enterPick("sample")');
  assert.deepEqual(plain(h.stage.image),{x0:80,y0:-230,w:2200,h:2100});
  assert.deepEqual(plain(h.stage.fitted),{x0:100,y0:-200,w:2000,h:2000});
  assert.deepEqual(plain(h.state.prior),sample.suggested_prior);assert.equal(h.$('sample-name').textContent,'Offset sample');
  assert.match(h.$('prior-preview-note').textContent,/transcripts' nuclear column/);
  assert.deepEqual(h.pickerLines,[null,'learning'],'the sample\'s mixing check: cleared, then its state');
  h.state.samples=[{...sample,path:'/data/sample.zarr'}];h.run('renderRecent()');
  const row=h.$('recent-samples').children[0];row.children[0].click();await row.children[1].click();
  assert.deepEqual(h.paths,['/data/sample.zarr','sample']);assert.equal(h.state.samples.length,0);
});

test('rectangle drawing, moving, corner resizing, Shift square, clicking and cancelling stay within extent',async()=>{
  const h=harness();await h.run('enterPick("sample");');h.run('setupPointer()');
  const drag=(a,b,extra={})=>{h.pointer('pointerdown',...a,extra);h.pointer('pointermove',...b,extra);h.pointer('pointerup',...b,extra);};
  drag([50,100],[300,250]);assert.deepEqual(plain(h.state.region.box),[150,-100,400,50]);
  drag([200,170],[250,200]);assert.deepEqual(plain(h.state.region.box),[200,-70,450,80]);
  drag([100,130],[80,110]);assert.deepEqual(plain(h.state.region.box),[180,-90,450,80]);
  drag([500,500],[650,550],{shiftKey:true});assert.deepEqual(plain(h.state.region.box),[600,300,750,450]);
  drag([50,750],[50,750]);assert.deepEqual(plain(h.state.region.box),[100,400,400,700]);
  h.pointer('pointerdown',700,700);h.pointer('pointermove',800,750);h.pointer('pointercancel',800,750);
  assert.deepEqual(plain(h.state.region.box),[100,400,400,700]);
});

test('lassos retain their shape across mode changes, pan with Shift, and allow full samples beyond detail limits',async()=>{
  const h=harness();await h.run('enterPick("sample")');h.run('setupPointer();setupControls();setRegionMode("polygon")');
  const stroke=[[500,100],[800,150],[760,350],[550,400],[500,100]];
  h.pointer('pointerdown',...stroke[0]);for(const p of stroke.slice(1))h.pointer('pointermove',...p);h.pointer('pointerup',...stroke.at(-1));
  const polygon=plain(h.state.region);assert.equal(polygon.kind,'polygon');assert.equal(polygon.points.length,4);
  h.run('setRegionMode("box");setRegionMode("polygon")');assert.deepEqual(plain(h.state.region),polygon);
  h.pointer('pointerdown',500,100,{shiftKey:true});h.pointer('pointermove',550,130,{shiftKey:true});h.pointer('pointerup',550,130,{shiftKey:true});
  assert.deepEqual(plain(h.state.region),polygon);assert.deepEqual(h.stage.pan,[50,30]);
  h.stage.fitted=null;h.run('setRegionMode("full")');assert.equal(h.$('start').disabled,false);assert.match(h.$('picker-mode').textContent,/Whole-sample mode/);
  assert.deepEqual(plain(h.stage.fitted),{x0:100,y0:-200,w:2000,h:2000});
  await h.$('start').click();assert.deepEqual(plain(h.created[0]),['sample',{kind:'full'},sample.suggested_prior,{}],'proseg\'s own schedule, paused on the prior: nothing to choose');
  assert.equal(h.$('schedule'),null);assert.equal(h.$('pause-prior'),null);
  h.state.prior={cells:null,nuclei:null};h.run('updatePicker()');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['prior-choice']);
  assert.equal(h.$('picker-hint').textContent,'','said when Start is pressed, not before');
  h.state.prior=sample.suggested_prior;h.run('setRegion({kind:"box",box:[100,0,119,400]});updatePicker()');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),[stain.PICKER_TEXT.regionSmall]);
});

// The picker's Lasso (main.js setupPointer with lasso.mjs: click corners, or draw by hand). On screen (x, y) is the
// sample's (100 + x, y − 200) µm here.
const LASSO_HELP='Click to place corners or drag to draw; double-click, Enter or click the first point to close; Backspace '+
  'takes back the last corner or stretch; Escape cancels. Shift or right/middle drag to pan.';
/** The Lasso tool on the picker's image (`h`: a picker already open, else a new one), with what a browser sends: a click
 *  (a press and its release on the spot), a drag through `points`, a move with no button down, a double-click (two clicks,
 *  then dblclick) and a key with focus on `target` (the event: defaultPrevented when the outline took the key). */
async function lassoPicker(h=null) {
  if(!h){h=harness();await h.run('enterPick("sample")');}
  h.run('setupPointer();setRegionMode("polygon")');
  const click=(x,y,extra={})=>{h.pointer('pointerdown',x,y,extra);h.pointer('pointerup',x,y,extra);};
  const drag=points=>{h.pointer('pointerdown',...points[0]);for(const p of points.slice(1))h.pointer('pointermove',...p);h.pointer('pointerup',...points.at(-1));};
  const hover=(x,y)=>h.pointer('pointermove',x,y,{button:-1});
  const doubleClick=(x,y,extra={})=>{click(x,y,extra);click(x,y,extra);h.$('stage').emit('dblclick',{offsetX:x,offsetY:y,shiftKey:false,...extra});};
  const key=(k,target=h.$('stage'))=>{
    const e={key:k,target,defaultPrevented:false,metaKey:false,ctrlKey:false,altKey:false,preventDefault(){this.defaultPrevented=true;}};
    for(const fn of h.document.listeners.get('keydown')||[])fn(e);
    return e;
  };
  return {...h,click,drag,hover,doubleClick,key,drawing:()=>h.run('lasso.active'),points:()=>plain(h.state.region.points)};
}
const field={closest:s=>/input/.test(s)?{}:null},onButton={closest:s=>s==='button'?{}:null};   // where focus is

test('Lasso: clicks place corners, drawn with a dashed line on to the pointer; a press on the first point closes them into the region',async()=>{
  const l=await lassoPicker(),g=l.$('overlay').getContext();
  l.click(300,300);l.click(600,300);
  assert.deepEqual(plain(l.state.region),{kind:'polygon',points:[]},'no region until it closes');
  // Drawn as it stands: the corners so far, a dashed line from the last on to the pointer, the first point as a handle.
  l.strokes.length=0;l.hover(600,600);
  const dash=l.strokes.findIndex(s=>s[0]==='setLineDash'&&String(s[1])==='6,4');
  assert.deepEqual(plain(l.strokes.slice(dash+2,dash+4)),[['moveTo',600,300],['lineTo',600,600]]);
  assert.deepEqual(l.strokes.filter(s=>s[0]==='fillRect'),[['fillRect',296,296,8,8]]);assert.equal(g.fillStyle,'#7cc4ff');
  assert.equal(l.drawing(),true);assert.equal(l.document.activeElement,l.$('stage'),'its keys come to the image');
  l.click(600,600);
  // With the pointer on the first point a press would close the outline: the handle is larger and white.
  l.strokes.length=0;l.hover(305,303);
  assert.deepEqual(l.strokes.filter(s=>s[0]==='fillRect'),[['fillRect',294,294,12,12]]);assert.equal(g.fillStyle,'#fff');
  l.click(305,303);
  assert.equal(l.drawing(),false);
  assert.deepEqual(plain(l.state.region),{kind:'polygon',points:[[400,100],[700,100],[700,400]]});
  assert.deepEqual(plain(l.state.lastLasso),plain(l.state.region),'the Lasso tool brings it back');
  l.strokes.length=0;l.run('drawOverlay()');
  assert.ok(l.strokes.some(s=>s[0]==='fill'),'drawn as the region');assert.ok(!l.strokes.some(s=>s[0]==='fillRect'||s[0]==='setLineDash'&&String(s[1])==='6,4'));
  l.run('setRegionMode("polygon")');assert.equal(l.$('region-help').textContent,LASSO_HELP);assert.deepEqual(bannedWords(LASSO_HELP),[]);
});

test('Lasso: one freehand stroke on its own closes when it is released, as the lasso always did; the region stays as it was until then',async()=>{
  const l=await lassoPicker();
  l.run('setRegion({kind:"polygon",points:[[150,-150],[250,-150],[150,-50]]})');const before=plain(l.state.region);
  l.pointer('pointerdown',300,300);
  for(const [x,y] of [[600,300],[600,600],[300,600]]){l.pointer('pointermove',x,y);assert.deepEqual(plain(l.state.region),before,`${x},${y}`);}
  assert.equal(l.drawing(),true);
  l.pointer('pointerup',300,600);
  assert.equal(l.drawing(),false);
  assert.deepEqual(plain(l.state.region),{kind:'polygon',points:[[400,100],[700,100],[700,400],[400,400]]});
});

test('Lasso: a quick freehand stroke keeps its corners: the moves the browser merged into one event all count',async()=>{
  const l=await lassoPicker();
  l.pointer('pointerdown',300,300);
  // The move to (600,300) came merged into the move to (600,600) (a busy frame).
  l.pointer('pointermove',600,600,{clientX:600,clientY:600,getCoalescedEvents:()=>[{clientX:600,clientY:300},{clientX:600,clientY:600}]});
  l.pointer('pointermove',300,600);l.pointer('pointerup',300,600);
  assert.deepEqual(plain(l.state.region),{kind:'polygon',points:[[400,100],[700,100],[700,400],[400,400]]});
});

test('Lasso: corners and freehand stretches mix in one outline, which Enter closes (Enter on a button stays the button\'s)',async()=>{
  const l=await lassoPicker();
  l.click(300,300);assert.equal(l.key('Enter').defaultPrevented,true,'one corner: it says three are needed');
  assert.equal(l.notices.at(-1),stain.PICKER_TEXT.lassoFew);assert.equal(l.drawing(),true);
  l.drag([[600,300],[700,350],[650,450],[700,500]]);   // a stretch after a corner: the outline goes on
  assert.deepEqual(l.points(),[]);assert.equal(l.drawing(),true);
  l.click(300,600);
  assert.equal(l.key('Enter',onButton).defaultPrevented,false);assert.equal(l.drawing(),true);
  assert.equal(l.key('Enter').defaultPrevented,true);assert.equal(l.drawing(),false);
  assert.deepEqual(l.points(),[[400,100],[700,100],[800,150],[750,250],[800,300],[400,400]]);
});

test('Lasso: a double-click closes the outline instead of fitting the view, also on its first point; Shift double-clicks still fit',async()=>{
  const l=await lassoPicker();l.stage.fitted=null;
  l.click(300,300);l.click(600,300);l.click(600,600);l.doubleClick(300,600);
  assert.deepEqual(l.points(),[[400,100],[700,100],[700,400],[400,400]],'the second click\'s corner goes');
  assert.equal(l.stage.fitted,null,'the view stays');assert.equal(l.drawing(),false);
  // On the first point: its first click closes the outline, and the corner its second click placed goes with the double-click.
  l.click(800,800);l.click(1100,800);l.click(1100,1100);l.doubleClick(803,801);
  assert.equal(l.drawing(),false);assert.deepEqual(l.points(),[[900,600],[1200,600],[1200,900]]);
  assert.equal(l.stage.fitted,null);assert.equal(l.run('regionDragging'),false);
  // Two Shift presses pan, so their double-click fits the view; the outline being drawn stays.
  l.click(500,500);l.doubleClick(700,700,{shiftKey:true});
  assert.deepEqual(plain(l.stage.fitted),{x0:100,y0:-200,w:2000,h:2000});assert.deepEqual(plain(l.run('lasso.points()')),[[600,300]]);
});

test('Lasso: a double-click with no outline being drawn fits the view, as everywhere else, and leaves nothing; one whose second click closed the outline does not',async()=>{
  const l=await lassoPicker();l.stage.fitted=null;
  l.doubleClick(500,500);
  assert.deepEqual(plain(l.stage.fitted),{x0:100,y0:-200,w:2000,h:2000});
  assert.equal(l.drawing(),false);assert.equal(l.run('regionDragging'),false);assert.deepEqual(l.notices,[]);
  // The first click 9 px from the first point places a corner, the second 6 px away closes the outline: the view stays.
  l.stage.fitted=null;
  l.click(300,300);l.click(600,300);l.click(600,600);
  l.click(309,300);l.click(306,300);l.$('stage').emit('dblclick',{offsetX:306,offsetY:300,shiftKey:false});
  assert.equal(l.stage.fitted,null);assert.equal(l.drawing(),false);
  assert.deepEqual(l.points(),[[400,100],[700,100],[700,400],[409,100]]);
});

test('Lasso: two corners and a click or double-click on the first point say three are needed: never a sliver region',async()=>{
  const l=await lassoPicker();l.stage.fitted=null;
  l.run('setRegion({kind:"polygon",points:[[150,-150],[250,-150],[150,-50]]})');const before=plain(l.state.region);
  l.click(300,300);l.click(600,300);l.doubleClick(301,300);
  assert.deepEqual(plain(l.state.region),before);assert.equal(l.drawing(),true);assert.equal(l.stage.fitted,null);
  assert.deepEqual(l.notices,[stain.PICKER_TEXT.lassoFew,stain.PICKER_TEXT.lassoFew,stain.PICKER_TEXT.lassoFew]);
  assert.deepEqual(plain(l.run('lasso.points()')),[[400,100],[700,100]]);
  // Three corners in a line enclose nothing: the region stays as it was, and the same words say why.
  l.click(900,300);l.key('Enter');
  assert.deepEqual(plain(l.state.region),before);assert.equal(l.drawing(),false);assert.equal(l.notices.length,4);
});

test('Lasso: Start with an outline still open says to close it first, and never starts on the region before it',async()=>{
  const h=harness();await h.run('enterPick("sample")');h.run('setupControls()');
  const l=await lassoPicker(h);
  l.click(300,300);l.click(600,300);l.click(600,600);l.click(305,303);
  const region=plain(l.state.region);
  l.click(800,800);l.click(1100,800);l.click(1100,1100);
  assert.deepEqual(l.problems(),[stain.PICKER_TEXT.closeLasso]);
  await l.$('start').click();
  assert.equal(l.created.length,0);assert.ok(l.note().includes(`${stain.PICKER_TEXT.region}: ${stain.PICKER_TEXT.closeLasso}`));
  assert.deepEqual(plain(l.state.region),region);
  l.key('Enter');assert.deepEqual(l.problems(),[]);
  await l.$('start').click();assert.equal(l.created.length,1);
  assert.deepEqual(plain(l.created[0][1]),{kind:'polygon',points:[[900,600],[1200,600],[1200,900]]});
});

test('a key a panel or the outline has already taken is not the page\'s as well: an Escape that dropped an outline keeps the selected cell',async()=>{
  const h=harness();await h.run('enterPick("sample")');h.run('setupControls()');
  h.state.mode='run';h.state.run={kind:'stain',state:'done'};h.state.selected=7;
  const escape=taken=>{
    const e={key:'Escape',target:h.$('stage'),defaultPrevented:taken,metaKey:false,ctrlKey:false,altKey:false,preventDefault(){this.defaultPrevented=true;}};
    for(const fn of h.document.listeners.get('keydown')||[])fn(e);
  };
  escape(true);assert.equal(h.state.selected,7);
  escape(false);assert.equal(h.state.selected,0);
});

test('Lasso: Backspace takes back the last corner or the whole last freehand stretch; keys typed in a field stay there',async()=>{
  const l=await lassoPicker(),count=()=>l.run('lasso.points().length');
  l.click(300,300);l.click(600,300);l.drag([[600,450],[650,500],[600,550]]);
  assert.deepEqual(l.points(),[]);assert.equal(count(),5);
  assert.equal(l.key('Backspace').defaultPrevented,true);assert.equal(count(),2,'the stretch at once');
  l.key('Backspace');assert.deepEqual(plain(l.run('lasso.points()')),[[400,100]],'then the corner');
  assert.equal(l.key('Backspace',field).defaultPrevented,false);assert.equal(count(),1);
  l.click(700,500);l.click(300,600);l.key('Enter');
  assert.deepEqual(l.points(),[[400,100],[800,300],[400,400]]);
  // Backspace during a press takes back what the press drew so far; the browser taking the pointer away then drops nothing more.
  l.click(1000,1000);l.pointer('pointerdown',1300,1000);l.pointer('pointermove',1350,1100);l.key('Backspace');
  l.pointer('pointercancel',1350,1100);
  assert.deepEqual(plain(l.run('lasso.points()')),[[1100,800]]);
  // The only corner taken back: no outline left, the region as it was, and the key is the page's again.
  l.key('Backspace');
  assert.equal(l.drawing(),false);assert.equal(l.run('regionDragging'),false);assert.deepEqual(l.points(),[[400,100],[800,300],[400,400]]);
  assert.equal(l.key('Backspace').defaultPrevented,false);
});

test('Lasso: Escape gives up the outline being drawn and the region stays as it was; so do another region tool and another sample',async()=>{
  const l=await lassoPicker();
  l.click(300,300);l.click(600,300);l.click(600,600);l.click(300,300);
  const region=plain(l.state.region);assert.equal(region.points.length,3);
  l.click(800,800);l.click(1100,800);
  l.strokes.length=0;
  assert.equal(l.key('Escape').defaultPrevented,true);assert.equal(l.drawing(),false);assert.deepEqual(plain(l.state.region),region);
  assert.ok(l.strokes.some(s=>s[0]==='fill'),'drawn again as the region');assert.ok(!l.strokes.some(s=>s[0]==='fillRect'),'without the outline');
  assert.equal(l.key('Escape').defaultPrevented,false,'with no outline Escape is the page\'s');
  // A press the browser takes away drops only what it placed.
  l.click(800,800);l.pointer('pointerdown',1100,800);l.pointer('pointercancel',1100,800);
  assert.deepEqual(plain(l.run('lasso.points()')),[[900,600]]);
  l.run('setRegionMode("box")');assert.equal(l.drawing(),false);
  l.run('setRegionMode("polygon")');assert.deepEqual(plain(l.state.region),region);
  // Another sample (here the same one opened again): given up as soon as it starts loading, before it has a region tool.
  l.click(800,800);const opening=l.run('enterPick("sample")');
  assert.equal(l.drawing(),false);l.hover(900,900);await opening;
  assert.equal(l.drawing(),false);assert.equal(l.run('regionDragging'),false);
});

test('shape previews debounce, use only shapes priors, and ignore obsolete responses or zoomed-out views',async()=>{
  const h=harness(),jobs=[];
  h.api.shapes=(id,key,box,signal)=>{const job=deferred();jobs.push({id,key,box,signal,...job});return job.promise;};
  await h.run('enterPick("sample")');assert.equal(jobs.length,0);
  const old=h.tick();assert.equal(jobs.length,1);assert.equal(jobs[0].key,'cells');
  h.state.prior={cells:{kind:'shapes',key:'next'},nuclei:null};h.run('previewPrior()');const fresh=h.tick();
  assert.equal(jobs[0].signal.aborted,true);
  jobs[1].resolve({polygons:[{id:'new',poly:[]}],truncated:true});await fresh;
  jobs[0].resolve({polygons:[{id:'old',poly:[]}]});await old;
  assert.equal(h.priors.vectors.cells[0].id,'new');assert.match(h.$('prior-preview-note').textContent,/20,000/);
  h.$('show-prior').checked=false;h.run('previewPrior()');assert.equal(h.priors.vectors,null);
  h.$('show-prior').checked=true;h.stage.scale=.1;h.run('previewPrior()');assert.match(h.$('prior-preview-note').textContent,/Zoom in/);
  assert.equal(jobs.length,2);
});


test('a refused stain run keeps the engine\'s reason under Start instead of a passing notice',async()=>{
  const h=await stainPicker();
  h.api.createStain=async()=>{throw Object.assign(new Error('This region has no tissue. Untick Tissue only to use it anyway.'),{status:422});};
  assert.equal(h.$('start').disabled,false);
  await h.$('start').click();
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(h.$('picker-error').hidden,false);
  assert.equal(h.$('picker-error').textContent,'This region has no tissue. Untick Tissue only to use it anyway.');
  assert.equal(h.$('start').disabled,false);               // the user can untick Tissue only and start again
  h.$('stain-tissue').checked=false;h.$('stain-tissue').emit('change');h.run('updatePicker()');
  assert.equal(h.$('picker-error').hidden,false,'a change in the picker leaves the reason in view');
  assert.equal(h.$('start').getAttribute('aria-busy'),null);assert.equal(h.$('start').textContent,stain.STAIN_TEXT.start);
});

// ---- Start and New run are never greyed out: a press lists what is missing by field ------------------------------------
/** The Training patches picker: a 300 µm rectangle on an image of 0.5 µm pixels (600 pixels across), complete. */
async function patchPicker() {
  const h=await stainPicker();
  h.state.sample.images[0].um_per_px=.5;
  h.$('patch-start-from').value='model';
  h.$('run-kind').value='patches';h.$('run-kind').emit('change');
  await settle();
  return h;
}
const suggestion=(n,size=64)=>({patches:Array.from({length:n},(_,i)=>({id:i+1,box:[i*size,0,(i+1)*size,size]})),
  run_pixel_to_sample:[100,-50,.5,.5],message:''});

test('Proseg: a press of Start names the missing fields by their labels, outlines them, moves to the first and starts nothing',async()=>{
  const h=harness();await h.run('enterPick("sample")');h.run('setupControls()');
  const P=stain.PICKER_TEXT,choice=h.$('prior-choice');
  let focused=0;choice.focus=()=>{focused++;};
  h.state.prior={cells:null,nuclei:null};h.run('setRegion({kind:"box",box:[100,0,110,400]});updatePicker()');
  assert.equal(h.$('start').disabled,false);
  await h.$('start').click();
  assert.equal(h.created.length,0,'nothing is created');
  assert.ok(h.note().startsWith(EXPLAIN_TEXT.lead));
  assert.ok(h.note().includes(`${P.startFrom}: ${P.firstCells}`));assert.ok(h.note().includes(`${P.region}: ${P.regionSmall}`));
  assert.ok(choice.classList.contains(NEEDS_CLASS));assert.equal(choice.getAttribute('aria-invalid'),'true');assert.equal(focused,1);
  choice.emit('change');                                     // chosen again: its outline goes at once
  assert.ok(!choice.classList.contains(NEEDS_CLASS));assert.deepEqual(h.problems(),[P.regionSmall]);
  h.run('setRegion({kind:"box",box:[100,0,400,400]});updatePicker()');
  assert.equal(h.note(),'','nothing missing: the list goes');
  await h.$('start').click();assert.equal(h.created.length,1);
  // A lasso not drawn yet, and proseg not built: said, also without a field.
  h.run('setRegionMode("polygon")');assert.deepEqual(h.problems(),[P.drawLasso]);
  h.state.health.proseg=false;h.run('updatePicker()');assert.deepEqual(h.problems(),[P.notBuilt,P.drawLasso]);
  assert.equal(h.$('picker-error').textContent,P.notBuilt);
  // What makes it, as health says (proseg_missing): a build (or not said), an install, or the setting that names it.
  for(const [missing,said] of [['install',P.notInstalled],['setting',P.notFound],['build',P.notBuilt],[null,P.notBuilt]]){
    h.state.health.proseg_missing=missing;h.run('updatePicker()');
    assert.deepEqual(h.problems(),[said,P.drawLasso],String(missing));assert.equal(h.$('picker-error').textContent,said);
  }
});

test('Stain segmentation: a press names each setting Cellpose would refuse in the picker\'s words, opening a closed section that holds one',async()=>{
  const h=await stainPicker(),P=stain.PICKER_TEXT,T=stain.STAIN_TEXT,diameter=h.$('stain-diameter');
  let scrolled=0;diameter.scrollIntoView=()=>{scrolled++;};
  h.$('stain-auto').checked=false;h.$('stain-auto').emit('change');
  diameter.value='5';h.$('stain-flow').value='12';h.$('stain-flow').emit('input');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['stain-diameter','stain-flow']);
  await h.$('start').click();
  assert.equal(h.created.length,0);
  assert.ok(h.note().includes(`${T.diameter}: ${P.diameter}`));assert.ok(h.note().includes(`${T.flow}: ${P.flow}`));
  for(const id of ['stain-diameter','stain-flow'])assert.ok(h.$(id).classList.contains(NEEDS_CLASS),id);
  assert.equal(h.document.activeElement,diameter,'the first is focused');assert.equal(scrolled,1);
  diameter.value='30';h.$('stain-flow').value='.4';h.$('stain-niter').value='0';h.$('stain-niter').emit('input');
  assert.equal(h.$('stain-more').open,false);
  await h.$('start').click();
  assert.equal(h.$('stain-more').open,true,'its section opens');assert.ok(h.note().includes(`${T.niter}: ${P.niter}`));
  assert.ok(!h.note().includes(T.diameter),'only what is missing now');
  h.$('stain-niter').value='';h.$('stain-niter').emit('input');
  assert.equal(h.note(),'');
  await h.$('start').click();assert.equal(h.created.length,1);
  // The image's channels: none of this image's, or the same one twice.
  h.$('stain-cytoplasm').value='DAPI';h.$('stain-cytoplasm').emit('input');
  assert.deepEqual(h.problems(),['stain-cytoplasm']);assert.equal(h.run('pickerProblems()[0].message'),P.otherChannel);
});

test('Training patches: a press names the count and the size, with the largest size that fits the region',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT,PT=STAIN_PATCH_TEXT;
  assert.deepEqual(h.problems(),[]);assert.equal(h.$('start').textContent,PT.start,'Make training patches');
  h.$('patch-count').value='0';h.$('patch-size').value='5000';h.$('patch-size').emit('input');
  assert.equal(h.$('start').disabled,false);assert.deepEqual(h.problems(),['patch-count','patch-size']);
  await h.$('start').click();
  assert.ok(h.note().includes(`${PT.count}: ${P.count}`));assert.ok(h.note().includes(`${PT.size}: ${P.size}`));
  assert.equal(h.created.length,0);
  h.$('patch-count').value='4';h.$('patch-size').value='700';h.$('patch-size').emit('input');
  assert.deepEqual(h.problems(),['patch-size']);
  assert.equal(h.run('pickerProblems()[0].message'),P.sizeFits.replace('{largest}','600'),'300 µm at 0.5 µm per pixel');
  h.$('patch-size').value='256';h.run('setRegion({kind:"box",box:[100,0,125,25]});updatePicker()');
  assert.deepEqual(h.problems(),[P.patchRegion],'50 pixels across: the region is what to change');
  // Empty needs no model: a model list still loading is nothing to wait for.
  h.run('setRegion({kind:"box",box:[100,0,400,300]})');h.state.stainModels=null;
  h.$('patch-start-from').value='empty';h.$('patch-start-from').emit('change');
  assert.deepEqual(h.problems(),[]);
  h.$('patch-start-from').value='model';h.$('patch-start-from').emit('change');
  assert.deepEqual(h.problems(),['stain-model']);
});

test('a press while Start is creating says the last step is still running; Start reads Starting… meanwhile and is never greyed out',async()=>{
  const h=await stainPicker(),gate=deferred(),start=h.$('start'),label=start.textContent;
  h.api.createStain=async(...args)=>{h.created.push(args);await gate.promise;return{id:'new-stain'};};
  const first=start.click();
  assert.equal(start.textContent,stain.PICKER_TEXT.starting);assert.equal(start.getAttribute('aria-busy'),'true');
  assert.equal(start.disabled,false);
  await start.click();
  assert.equal(h.created.length,1,'never started twice');assert.ok(h.note().includes(EXPLAIN_TEXT.busy));
  gate.resolve();await first;
  assert.equal(start.textContent,label);assert.equal(start.getAttribute('aria-busy'),null);
  assert.deepEqual(h.opened,['new-stain']);assert.equal(h.note(),'','the note goes with the work');
});

test('Training patches: boxes are suggested 400 ms after the last change, a newer change wins, and Start sends the boxes shown',async()=>{
  const h=await patchPicker(),requested=()=>h.previews.length;
  // The picker opened complete: one request is due, none sent yet; a change meanwhile is folded into it.
  assert.equal(h.pending(400),1);assert.equal(requested(),0);
  h.$('patch-count').value='5';h.$('patch-count').emit('input');
  assert.equal(h.pending(400),1);
  h.fire(400);
  assert.equal(requested(),1);assert.equal(h.previews[0].sample,'sample');assert.deepEqual(plain(h.previews[0].region),plain(h.state.region));
  assert.deepEqual(plain(h.previews[0].stain),plain(h.run('readStain()')),'as Start would send it');assert.equal(h.previews[0].stain.count,5);
  // Another change while it is on its way: that request is cancelled, and its late answer is never shown.
  h.$('patch-count').value='3';h.$('patch-count').emit('input');
  assert.equal(h.previews[0].signal.aborted,true);
  h.fire(400);assert.equal(requested(),2);
  h.previews[0].resolve(suggestion(5));await settle();
  assert.equal(h.panel.shown.length,0);
  assert.equal(h.$('picker-hint').textContent,stain.PICKER_TEXT.suggesting,'said while the boxes are on their way');
  h.previews[1].resolve({...suggestion(3),message:'Found 3 of 5 requested patches.'});await settle();
  assert.equal(h.panel.shown.length,1);assert.deepEqual(plain(h.panel.previewBoxes()),[[0,0,64,64],[64,0,128,64],[128,0,192,64]]);
  assert.equal(h.$('picker-hint').textContent,'Found 3 of 5 requested patches.');
  // Moved and dropped on the image (stain-patches.mjs), then a change the boxes do not depend on (the model): asked again,
  // the same boxes suggested, and what was done to them stays.
  h.panel.boxes=[[10,10,74,74],[128,0,192,64]];h.panel.onChange(h.panel.previewBoxes());
  h.$('stain-model').value='cpdino';h.$('stain-model').emit('change');
  h.fire(400);assert.equal(requested(),3);
  h.previews[2].resolve(suggestion(3));await settle();
  assert.equal(h.panel.shown.length,1,'not drawn again');
  await h.$('start').click();
  assert.deepEqual(plain(h.created[0][2].boxes),[[10,10,74,74],[128,0,192,64]],'exactly the boxes shown');
  assert.equal(h.created[0][2].model,'cpdino');assert.equal(h.created[0][3],'patches');
  // Refused (no tissue there): the reason under Start and no boxes; the next answer takes the reason away.
  h.run('setRegion({kind:"box",box:[1000,1000,1300,1300]});updatePicker()');h.fire(400);
  h.previews[3].reject(Object.assign(new Error('This region has no tissue. Untick Tissue only to use it anyway.'),{status:422}));await settle();
  assert.equal(h.$('picker-error').hidden,false);assert.match(h.$('picker-error').textContent,/no tissue/);
  assert.equal(h.panel.previewBoxes(),null);
  h.$('stain-tissue').checked=false;h.$('stain-tissue').emit('change');h.fire(400);
  assert.equal(h.previews[4].stain.tissue_only,false);
  h.previews[4].resolve(suggestion(3));await settle();
  assert.equal(h.$('picker-error').hidden,true);assert.equal(h.panel.previewBoxes().length,3);
  // Start pressed before a change's 400 ms are up: the boxes are asked for at once, and the item gets those.
  h.$('patch-size').value='128';h.$('patch-size').emit('input');
  const started=h.$('start').click();
  assert.equal(requested(),6);assert.equal(h.previews[5].stain.size,128);assert.equal(h.pending(400),0);
  h.previews[5].resolve(suggestion(2,128));await started;
  assert.deepEqual(plain(h.created.at(-1)[2].boxes),[[0,0,128,128],[128,0,256,128]]);assert.equal(h.created.at(-1)[2].size,128);
  // Something missing: nothing is asked for, and the boxes stay as they are. Another run type, another sample: they go.
  h.$('patch-count').value='';h.$('patch-count').emit('input');
  assert.deepEqual(plain(h.panel.previewBoxes()),[[0,0,128,128],[128,0,256,128]]);assert.equal(h.pending(400),0);
  h.$('patch-count').value='3';h.$('patch-count').emit('input');h.fire(400);h.previews.at(-1).resolve(suggestion(3));await settle();
  h.$('run-kind').value='stain';h.$('run-kind').emit('change');
  assert.equal(h.panel.previewBoxes(),null);assert.equal(h.pending(400),0);
  h.$('run-kind').value='patches';h.$('run-kind').emit('change');await settle();h.fire(400);
  const late=h.previews.at(-1),asked=requested();
  await h.run('enterPick("sample")');
  assert.equal(late.signal.aborted,true,'another sample: the request on its way is cancelled');
  late.resolve(suggestion(3));await settle();assert.equal(h.panel.previewBoxes(),null);
  assert.equal(requested(),asked);
});

test('Training patches: a preview the engine cannot answer (500, 503, no connection) keeps the boxes; the next change or Start asks again, and Start makes nothing when its own ask fails',async()=>{
  const h=await patchPicker(),error=h.$('picker-error'),first=suggestion(2).patches.map(p=>p.box);
  h.fire(400);h.previews[0].resolve(suggestion(2));await settle();
  h.$('patch-count').value='4';h.$('patch-count').emit('input');h.fire(400);
  h.previews[1].reject(Object.assign(new Error('Internal Server Error'),{status:500}));await settle();
  assert.deepEqual(plain(h.panel.previewBoxes()),first,'the boxes stay on the image');assert.equal(h.panel.clears,0);
  assert.equal(error.hidden,true,'only a refusal is shown under Start');assert.equal(h.run('patchPreview.key'),'');
  // Never asked again by itself (the model list's refresh, a box moved): only after a change, or by Start.
  h.run('updatePicker()');h.panel.onChange(h.panel.previewBoxes());
  assert.equal(h.pending(400),0);assert.equal(h.previews.length,2);
  h.$('patch-count').value='5';h.$('patch-count').emit('input');assert.equal(h.pending(400),1);h.fire(400);
  assert.equal(h.previews.length,3);assert.equal(h.previews[2].stain.count,5);
  h.previews[2].reject(new TypeError('Failed to fetch'));await settle();             // no answer at all
  assert.deepEqual(plain(h.panel.previewBoxes()),first);assert.equal(h.pending(400),0);
  // Start asks again for the settings as they stand, and makes the item with the boxes that come.
  const started=h.$('start').click();
  assert.equal(h.previews.length,4);assert.equal(h.previews[3].stain.count,5);
  h.previews[3].resolve(suggestion(5));await started;
  assert.equal(h.created.length,1);assert.deepEqual(plain(h.created[0][2].boxes),plain(suggestion(5).patches.map(p=>p.box)));
  // That ask failing too: the reason under Start, nothing made, Start free, and nothing asked for by itself after it.
  h.$('patch-count').value='6';h.$('patch-count').emit('input');h.fire(400);
  h.previews[4].reject(Object.assign(new Error('The engine is shutting down'),{status:503}));await settle();
  const again=h.$('start').click();
  assert.equal(h.previews.length,6);assert.equal(error.hidden,true,'nothing said while it is on its way');
  h.previews[5].reject(Object.assign(new Error('The engine is shutting down'),{status:503}));await again;
  assert.equal(h.created.length,1,'nothing is made');
  assert.equal(error.hidden,false);assert.equal(error.textContent,stain.PICKER_TEXT.boxesFailed.replace('{reason}','The engine is shutting down'));
  assert.equal(h.$('start').getAttribute('aria-busy'),null);assert.equal(h.pending(400),0);assert.equal(h.previews.length,6);
  assert.equal(h.panel.previewBoxes().length,5,'the last boxes stay');
  // Stain segmentation sends no boxes.
  h.$('run-kind').value='stain';h.$('run-kind').emit('change');
  await h.$('start').click();assert.equal('boxes' in h.created[1][2],false);assert.equal(h.created[1][3],'stain');
});

test('Training patches: a refused preview\'s reason stays under Start, and Start with those settings creates nothing',async()=>{
  const h=await patchPicker(),error=h.$('picker-error');
  const reason='No patches with image detail were found in this region. Choose another region.';
  const refuse=job=>job.reject(Object.assign(new Error(reason),{status:422}));
  h.fire(400);refuse(h.previews[0]);await settle();
  assert.equal(error.hidden,false);assert.equal(error.textContent,reason);assert.equal(h.panel.previewBoxes(),null);
  assert.deepEqual(h.problems(),[],'nothing a field could change');
  await h.$('start').click();
  assert.equal(h.created.length,0,'no item without boxes');assert.equal(h.previews.length,1,'the same settings are not asked for again');
  assert.equal(error.hidden,false);assert.equal(error.textContent,reason,'the reason, under Start');
  assert.equal(h.note(),'');assert.equal(h.$('start').getAttribute('aria-busy'),null);
  h.run('updatePicker()');assert.equal(error.hidden,false,'it stays');
  // A change, and Start pressed before its 400 ms are up: asked at once, refused again, still nothing.
  h.$('patch-count').value='4';h.$('patch-count').emit('input');
  const started=h.$('start').click();
  assert.equal(h.previews.length,2);refuse(h.previews[1]);await started;
  assert.equal(h.created.length,0);assert.equal(error.hidden,false);assert.equal(error.textContent,reason);
  // Another region that has boxes: the reason goes, and Start creates the item with them.
  h.run('setRegion({kind:"box",box:[400,400,700,700]});updatePicker()');h.fire(400);
  h.previews[2].resolve(suggestion(2));await settle();
  assert.equal(error.hidden,true);
  await h.$('start').click();
  assert.equal(h.created.length,1);assert.deepEqual(plain(h.created[0][2].boxes),[[0,0,64,64],[64,0,128,64]]);
});

test('Training patches: moved boxes stay while a field is briefly not valid or the model list cannot be refreshed; a new start clears them',async()=>{
  const h=await patchPicker(),diameter=h.$('stain-diameter'),type=value=>{diameter.value=value;diameter.emit('input');};
  h.$('stain-auto').checked=false;h.$('stain-auto').emit('change');
  h.fire(400);h.previews[0].resolve(suggestion(3));await settle();
  const moved=[[10,10,74,74],[128,0,192,64]],clears=h.panel.clears;
  h.panel.boxes=moved.map(b=>[...b]);h.panel.onChange(h.panel.previewBoxes());   // the first moved, the second dropped
  // 30 → 25 typed: the field passes through 2, which is not valid. Nothing is asked for meanwhile; the boxes stay.
  type('2');
  assert.deepEqual(h.problems(),['stain-diameter']);assert.equal(h.pending(400),0);
  assert.deepEqual(plain(h.panel.previewBoxes()),moved);assert.equal(h.panel.clears,clears);
  type('25');assert.equal(h.pending(400),1);h.fire(400);
  assert.equal(h.previews[1].stain.diameter,25);
  // Gone wrong again while that request is on its way: it is cancelled, and its late answer changes nothing.
  type('2');assert.equal(h.previews[1].signal.aborted,true);
  h.previews[1].resolve(suggestion(4));await settle();
  assert.deepEqual(plain(h.panel.previewBoxes()),moved);
  type('25');h.fire(400);h.previews[2].resolve(suggestion(3));await settle();
  assert.equal(h.panel.shown.length,1,'the same boxes suggested: not drawn again');assert.deepEqual(plain(h.panel.previewBoxes()),moved);
  // The model list cannot be refreshed: named for Start, and the boxes stay; once it loads, the same boxes are kept.
  h.api.stainModels=async()=>{throw new Error('offline');};await h.run('loadStainModels()');
  assert.deepEqual(h.problems(),['stain-model']);assert.deepEqual(plain(h.panel.previewBoxes()),moved);assert.equal(h.panel.clears,clears);
  h.api.stainModels=async()=>({models});await h.run('loadStainModels()');
  assert.deepEqual(h.problems(),[]);h.fire(400);h.previews[3].resolve(suggestion(3));await settle();
  assert.equal(h.panel.shown.length,1);
  await h.$('start').click();
  assert.deepEqual(plain(h.created[0][2].boxes),moved,'Start sends them as moved');assert.equal(h.created[0][2].diameter,25);
  // Another start (Empty): the boxes and what was done to them go; the next answer is drawn afresh.
  h.$('patch-start-from').value='empty';h.$('patch-start-from').emit('change');
  assert.equal(h.panel.previewBoxes(),null);assert.equal(h.panel.clears,clears+1);
  h.fire(400);h.previews[4].resolve(suggestion(3));await settle();
  assert.equal(h.panel.shown.length,2);assert.deepEqual(plain(h.panel.previewBoxes()),plain(suggestion(3).patches.map(p=>p.box)));
});

test('Training patches: the region\'s frame goes to the panel, and a box outside the region is named by its number for Start',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT,named=()=>plain(h.run('pickerProblems()')).map(p=>`${p.label}: ${p.message}`);
  h.fire(400);h.previews[0].resolve({...suggestion(3),extent:[600,600]});await settle();
  assert.deepEqual(plain(h.panel.shown[0].extent),[600,600]);assert.equal(h.panel.shown[0].outline,null);
  // Box 2 ends past the region's right edge: named, outlined nowhere (it is on the image), and Start creates nothing.
  h.panel.boxes=[[0,0,64,64],[560,0,624,64],[128,0,192,64]];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(named(),[`Box 2: ${P.boxOutside}`]);assert.deepEqual(h.problems(),[P.boxOutside]);
  await h.$('start').click();
  assert.equal(h.created.length,0);assert.ok(h.note().includes(`Box 2: ${P.boxOutside}`));
  assert.equal(h.pending(400),0,'a box out of place asks for no new boxes');
  h.panel.boxes[1]=[500,0,564,64];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(h.problems(),[]);assert.equal(h.note(),'','back inside: the list goes');
  // A wider region whose answer suggests the same boxes: kept as moved, now in the wider frame.
  h.run('setRegion({kind:"box",box:[100,0,500,300]});updatePicker()');h.fire(400);
  h.previews[1].resolve({...suggestion(3),extent:[800,600]});await settle();
  assert.equal(h.panel.shown.length,2);assert.deepEqual(plain(h.panel.shown[1].extent),[800,600]);
  assert.deepEqual(plain(h.panel.previewBoxes()),[[0,0,64,64],[500,0,564,64],[128,0,192,64]]);
  // A lasso: its outline comes with the boxes, and a box inside the window but outside the lasso is named.
  h.run('setRegion({kind:"polygon",points:[[100,0],[400,0],[100,300]]});updatePicker()');h.fire(400);
  const outline=[[0,0],[600,0],[0,600]];
  h.previews[2].resolve({...suggestion(2),extent:[600,600],outline});await settle();
  assert.deepEqual(plain(h.panel.shown.at(-1).outline),outline);
  assert.deepEqual(h.problems(),[]);
  h.panel.boxes=[[0,0,64,64],[400,400,464,464]];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(named(),[`Box 2: ${P.boxOutside}`]);
  await h.$('start').click();assert.equal(h.created.length,0);
  h.panel.boxes[1]=[200,200,264,264];h.panel.onChange(h.panel.previewBoxes());
  await h.$('start').click();
  assert.equal(h.created.length,1);assert.deepEqual(plain(h.created[0][2].boxes),[[0,0,64,64],[200,200,264,264]]);
});

test('Training patches: the engine\'s own boxes are never named for Start; a moved box is held to a pixel past the lasso, as the panel holds it',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT;
  // A lasso whose right-angled corner sits in the window's corner: the engine's box there reaches 1.4 pixels past that
  // corner (the engine's pixel of slack is mitred at corners), so the page's rule alone would refuse it.
  const outline=[[.99,.99],[600,.99],[.99,600]],offered=[[0,0,64,64],[100,100,164,164]];
  h.run('setRegion({kind:"polygon",points:[[100,0],[400,0],[100,300]]});updatePicker()');h.fire(400);
  h.previews.at(-1).resolve({patches:offered.map((box,i)=>({id:i+1,box})),run_pixel_to_sample:[100,-50,.5,.5],message:'',extent:[600,600],outline});
  await settle();
  assert.equal(stain.boxInRegion(offered[0],{extent:[600,600],outline}),false,'refused by the rule itself');
  assert.deepEqual(h.problems(),[],'as suggested: not named');
  // Box 2 moved to 1.4 pixels past the long side (x + y = 600.99 there): named; 0.7 pixels past: not.
  h.panel.boxes=[offered[0],[238,237,302,301]];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(h.problems(),[P.boxOutside]);
  h.panel.boxes=[offered[0],[238,236,302,300]];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(h.problems(),[]);
  await h.$('start').click();
  assert.equal(h.created.length,1);assert.deepEqual(plain(h.created[0][2].boxes),[offered[0],[238,236,302,300]]);
});

test('Training patches: no region to suggest boxes in (a lasso not drawn yet, a region too small) takes the boxes and their refusal away; a field being typed keeps them',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT,error=h.$('picker-error');
  h.fire(400);h.previews[0].resolve(suggestion(3));await settle();
  // A field being typed: every problem has its field; the boxes stay, and nothing is asked for meanwhile.
  h.$('patch-count').value='';h.$('patch-count').emit('input');
  assert.deepEqual(h.problems(),['patch-count']);assert.equal(h.panel.previewBoxes().length,3);assert.equal(h.panel.clears,0);
  h.$('patch-count').value='3';h.$('patch-count').emit('input');h.fire(400);h.previews[1].resolve(suggestion(3));await settle();
  // The lasso tool with no lasso drawn yet: no region, so no boxes, and nothing due.
  h.run('setRegionMode("polygon")');
  assert.deepEqual(h.problems(),[P.drawLasso]);assert.equal(h.panel.previewBoxes(),null);assert.equal(h.panel.clears,1);
  assert.equal(h.pending(400),0);
  // Back to the rectangle: asked for again.
  h.run('setRegionMode("box")');assert.equal(h.pending(400),1);h.fire(400);h.previews[2].resolve(suggestion(3));await settle();
  assert.equal(h.panel.previewBoxes().length,3);
  // A rectangle under 20 µm across: the boxes go.
  h.run('setRegion({kind:"box",box:[100,0,110,400]});updatePicker()');
  assert.deepEqual(h.problems(),[P.regionSmall]);assert.equal(h.panel.previewBoxes(),null);assert.equal(h.pending(400),0);
  // Refused for a region (no tissue there), then a region under 64 pixels across, too small for patches: the boxes went
  // with the refusal, and its reason, which was about the region before, goes too.
  h.run('setRegion({kind:"box",box:[1000,1000,1300,1300]});updatePicker()');h.fire(400);
  h.previews[3].reject(Object.assign(new Error('This region has no tissue. Untick Tissue only to use it anyway.'),{status:422}));await settle();
  assert.equal(error.hidden,false);
  h.run('setRegion({kind:"box",box:[1000,1000,1025,1025]});updatePicker()');
  assert.deepEqual(h.problems(),[P.patchRegion]);assert.equal(error.hidden,true);assert.equal(h.run('patchPreview.refused'),'');
  h.run('setRegion({kind:"box",box:[1000,1000,1300,1300]});updatePicker()');assert.equal(h.pending(400),1,'asked for again');
});

test('Training patches: Start pressed while a change waits checks the boxes that come for it, and makes nothing while one is outside the region',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT;
  // A 400 × 300 µm rectangle (800 × 600 pixels); box 2 moved to its right-hand side.
  h.run('setRegion({kind:"box",box:[100,0,500,300]});updatePicker()');h.fire(400);
  h.previews.at(-1).resolve({...suggestion(3),extent:[800,600]});await settle();
  h.panel.boxes=[[0,0,64,64],[700,0,764,64],[128,0,192,64]];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(h.problems(),[]);
  // The rectangle made narrower (300 µm, 600 pixels) and Start pressed at once: the engine suggests the same boxes for it,
  // so box 2 is kept where it was moved, now past the region's edge. Named, as a press names it; nothing is made.
  h.run('setRegion({kind:"box",box:[100,0,400,300]});updatePicker()');
  assert.deepEqual(h.problems(),[],'the boxes on the image are for the wider rectangle: nothing to name yet');
  const started=h.$('start').click();
  h.previews.at(-1).resolve({...suggestion(3),extent:[600,600]});await started;
  assert.deepEqual(plain(h.panel.previewBoxes()),[[0,0,64,64],[700,0,764,64],[128,0,192,64]]);
  assert.equal(h.created.length,0,'nothing is made');assert.ok(h.note().includes(`Box 2: ${P.boxOutside}`));
  assert.equal(h.$('start').getAttribute('aria-busy'),null);
  h.panel.boxes[1]=[500,0,564,64];h.panel.onChange(h.panel.previewBoxes());
  assert.equal(h.note(),'','back inside: the list goes');
  await h.$('start').click();
  assert.deepEqual(plain(h.created[0][2].boxes),[[0,0,64,64],[500,0,564,64],[128,0,192,64]]);
});

test('Training patches: nothing is asked for while the region is drawn, moved or resized; the gesture\'s end asks',async()=>{
  const h=await patchPicker();h.run('setupPointer()');
  // A 400 µm rectangle, on screen from (200, 200) to (600, 600).
  h.run('setRegion({kind:"box",box:[300,0,700,400]});updatePicker()');h.fire(400);h.previews.at(-1).resolve(suggestion(3));await settle();
  let asked=h.previews.length;
  // Moved (a press inside it): nothing due or asked for while it moves, also when the pointer rests longer than 400 ms;
  // the boxes stay meanwhile.
  h.pointer('pointerdown',400,400);
  for(const x of [420,440,460]){h.pointer('pointermove',x,400);assert.equal(h.pending(400),0,String(x));h.fire(400);}
  assert.equal(h.previews.length,asked);assert.equal(h.panel.previewBoxes().length,3);
  h.pointer('pointerup',460,400);
  assert.deepEqual(plain(h.state.region.box),[360,0,760,400]);
  assert.equal(h.pending(400),1,'its end asks');h.fire(400);
  assert.equal(h.previews.length,++asked);assert.deepEqual(plain(h.previews.at(-1).region.box),[360,0,760,400]);
  h.previews.at(-1).resolve(suggestion(3));await settle();
  // A change just before a resize from a corner: what was due waits for the gesture's end too.
  h.$('patch-count').value='4';h.$('patch-count').emit('input');assert.equal(h.pending(400),1);
  h.pointer('pointerdown',660,600);assert.equal(h.pending(400),0);
  h.pointer('pointermove',700,640);h.fire(400);assert.equal(h.previews.length,asked);
  h.pointer('pointerup',700,640);
  assert.deepEqual(plain(h.state.region.box),[360,0,800,440]);
  h.fire(400);assert.equal(h.previews.length,++asked);assert.equal(h.previews.at(-1).stain.count,4);
  h.previews.at(-1).resolve(suggestion(3));await settle();
  // A new rectangle drawn from a point: under 20 µm across while it starts, yet the boxes stay until it ends.
  h.pointer('pointerdown',100,700);h.pointer('pointermove',105,705);
  assert.deepEqual(h.problems(),[stain.PICKER_TEXT.regionSmall]);assert.equal(h.panel.previewBoxes().length,3);
  h.pointer('pointermove',500,1000);h.pointer('pointerup',500,1000);
  assert.deepEqual(plain(h.state.region.box),[200,500,600,800]);h.fire(400);assert.equal(h.previews.length,++asked);
  // A lasso stroke: asked for once it is closed.
  h.run('setRegionMode("polygon")');
  h.pointer('pointerdown',300,300);for(const [x,y] of [[600,300],[600,600],[300,600]]){h.pointer('pointermove',x,y);assert.equal(h.pending(400),0);}
  h.pointer('pointerup',300,600);
  assert.equal(h.state.region.kind,'polygon');assert.equal(h.pending(400),1);h.fire(400);
  assert.equal(h.previews.length,++asked);assert.equal(h.previews.at(-1).region.kind,'polygon');
  // Cancelled (the pointer taken away) while a change was due: the rectangle goes back, and the change is asked for then.
  h.run('setRegionMode("box")');h.fire(400);h.previews.at(-1).resolve(suggestion(3));await settle();
  h.$('patch-count').value='2';h.$('patch-count').emit('input');assert.equal(h.pending(400),1);
  h.pointer('pointerdown',400,400);h.pointer('pointermove',450,450);assert.equal(h.pending(400),0);
  h.pointer('pointercancel',450,450);
  assert.deepEqual(plain(h.state.region.box),[200,500,600,800]);assert.equal(h.pending(400),1);
  h.fire(400);assert.equal(h.previews.at(-1).stain.count,2);assert.equal(h.run('regionDragging'),false);
});

test('Training patches: nothing is asked for while a Lasso outline is drawn, between its presses too; its end asks, and its presses go over the boxes',async()=>{
  const l=await lassoPicker(await patchPicker());
  let asked=l.previews.length;assert.equal(l.pending(400),0);
  l.click(300,300);l.click(600,300);l.hover(600,600);l.fire(400);
  assert.equal(l.pending(400),0);assert.equal(l.previews.length,asked,'nothing between presses');
  assert.deepEqual(l.problems(),[stain.PICKER_TEXT.closeLasso],'Start: the outline is not the region yet');
  l.click(600,600);l.click(305,303);   // closed on its first point
  assert.equal(l.pending(400),1,'its close asks');l.fire(400);assert.equal(l.previews.length,++asked);
  assert.deepEqual(plain(l.previews.at(-1).region),{kind:'polygon',points:[[400,100],[700,100],[700,400]]});
  l.previews.at(-1).resolve(suggestion(3));await settle();assert.deepEqual(l.problems(),[]);
  // A press over a box is the box's until an outline is begun; then each press of the main button places a corner.
  const press=(x,y,extra={})=>l.panel.yields({offsetX:x,offsetY:y,button:0,shiftKey:false,...extra});
  assert.equal(press(400,320),false);
  l.click(800,800);assert.equal(press(400,320),true);assert.equal(press(400,320,{button:2}),false,'a right press pans');
  l.stage.dirty=false;l.hover(900,900);assert.equal(l.stage.dirty,false,'a move draws the outline again, never the image');
  // A change made meanwhile waits, also through a pan (a right drag); Escape ends the outline, the region as it was, and
  // the change is asked for then.
  l.$('patch-count').value='4';l.$('patch-count').emit('input');assert.equal(l.pending(400),0);
  l.pointer('pointerdown',900,900,{button:2});l.pointer('pointermove',950,920,{button:2});l.pointer('pointerup',950,920,{button:2});
  assert.deepEqual(l.stage.pan,[50,20]);assert.equal(l.drawing(),true);assert.equal(l.pending(400),0);
  l.key('Escape');assert.equal(l.pending(400),1);l.fire(400);
  assert.equal(l.previews.length,++asked);assert.equal(l.previews.at(-1).stain.count,4);
  assert.deepEqual(plain(l.previews.at(-1).region.points),[[400,100],[700,100],[700,400]]);
  assert.equal(l.run('regionDragging'),false);
});

test('Training patches: a press on the rectangle\'s corners is the region\'s, even over a box; elsewhere the boxes take it',async()=>{
  const h=await patchPicker();
  h.run('setRegion({kind:"box",box:[300,0,700,400]});updatePicker()');h.fire(400);h.previews.at(-1).resolve(suggestion(3));await settle();
  const yields=h.panel.yields,press=(x,y,extra={})=>yields({offsetX:x,offsetY:y,button:0,shiftKey:false,...extra});
  assert.equal(typeof yields,'function','the panel is told which presses are the page\'s');
  // The rectangle's corners on screen are (200, 200), (600, 200), (200, 600) and (600, 600); a handle reaches 9 pixels.
  for(const [x,y] of [[200,200],[600,200],[200,600],[600,600],[206,205]])assert.equal(press(x,y),true,`${x},${y}`);
  // Inside it (where the boxes are), on a side, past a handle's reach, or a right-button press (a pan): not the region's.
  for(const [x,y,extra] of [[400,400],[200,400],[210,210],[600,600,{button:2}]])assert.equal(press(x,y,extra),false,`${x},${y}`);
  // The page resizes the rectangle from a corner it was given.
  h.run('setupPointer()');h.pointer('pointerdown',600,600);h.pointer('pointermove',650,650);h.pointer('pointerup',650,650);
  assert.deepEqual(plain(h.state.region.box),[300,0,750,450]);
  // A lasso has no handles: a press there is a box's.
  h.run('setRegion({kind:"polygon",points:[[300,0],[700,0],[300,400]]})');assert.equal(press(200,200),false);
});

test('Training patches: the reason under Start stays while Start fetches the boxes of a changed request',async()=>{
  const h=await patchPicker(),error=h.$('picker-error'),reason='This region has no tissue. Untick Tissue only to use it anyway.';
  h.fire(400);h.previews[0].reject(Object.assign(new Error(reason),{status:422}));await settle();
  assert.equal(error.hidden,false);assert.equal(error.textContent,reason);
  // Changed (Tissue only unticked), and Start pressed before its 400 ms are up: its boxes are asked for at once.
  h.$('stain-tissue').checked=false;h.$('stain-tissue').emit('change');assert.equal(h.pending(400),1);
  const started=h.$('start').click();
  assert.equal(h.previews.length,2);assert.equal(h.previews[1].stain.tissue_only,false);
  assert.equal(h.$('start').getAttribute('aria-busy'),'true');
  assert.equal(error.hidden,false,'the reason stays in view while they are on their way');assert.equal(error.textContent,reason);
  h.run('updatePicker()');assert.equal(error.hidden,false);
  h.previews[1].resolve(suggestion(2));await started;
  assert.equal(error.hidden,true,'answered: it goes');assert.equal(h.created.length,1);
  assert.deepEqual(plain(h.created[0][2].boxes),[[0,0,64,64],[64,0,128,64]]);assert.equal(h.created[0][2].tissue_only,false);
  // Start's own refusal (the engine refused to make the item) stays in view the same way.
  h.api.createStain=async()=>{throw Object.assign(new Error(reason),{status:422});};
  await h.$('start').click();assert.equal(error.hidden,false);assert.equal(error.textContent,reason);
  h.$('patch-count').value='2';h.$('patch-count').emit('input');
  const again=h.$('start').click();
  assert.equal(h.previews.length,3);assert.equal(error.hidden,false);assert.equal(error.textContent,reason);
  h.api.createStain=async(...args)=>{h.created.push(args);return{id:'new-stain'};};
  h.previews[2].resolve(suggestion(2));await again;
  assert.equal(h.created.length,2);assert.equal(error.hidden,true);
});

test('Training patches: Start pressed while a model change waits fetches the new boxes and makes the item, naming no box of the earlier ones',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT;
  h.fire(400);h.previews[0].resolve({...suggestion(3),extent:[600,600]});await settle();
  h.panel.boxes=[[0,0,64,64],[560,0,624,64],[128,0,192,64]];h.panel.onChange(h.panel.previewBoxes());
  assert.deepEqual(h.problems(),[P.boxOutside]);
  // Another model: its boxes are due. Those on the image belong to the earlier settings, so none of them is named.
  h.$('stain-model').value='cpdino';h.$('stain-model').emit('change');
  assert.equal(h.pending(400),1);assert.deepEqual(h.problems(),[]);
  const started=h.$('start').click(),fresh=[[0,0,64,64],[200,0,264,64]];
  assert.equal(h.previews.length,2);assert.equal(h.previews[1].stain.model,'cpdino');
  h.previews[1].resolve({patches:fresh.map((box,i)=>({id:i+1,box})),run_pixel_to_sample:[100,-50,.5,.5],message:'',extent:[600,600]});
  await started;
  assert.equal(h.note(),'','no box named');assert.equal(h.created.length,1);
  assert.deepEqual(plain(h.created[0][2].boxes),fresh);assert.equal(h.created[0][2].model,'cpdino');
});

test('Proseg: a press of Start names an advanced option holding a value it cannot use, opening Advanced proseg options at it',async()=>{
  const h=harness();await h.run('enterPick("sample")');h.run('setupControls()');
  // The page's own options (options.js), drawn into this page's Advanced section; the browser's check is `valid`.
  const {Options}=await import('../src/options.js');
  h.Element.prototype.checkValidity=function(){return this.valid!==false;};
  h.Element.prototype.reportValidity=function(){return this.checkValidity();};
  const container=h.$('proseg-options'),options=new Options(container);
  container.querySelectorAll=()=>container.all().filter(n=>['INPUT','SELECT'].includes(n.tagName));
  globalThis.document=h.document;
  try {options.load([{flag:'samples',label:'Samples',type:'int',default:200,min:0,advanced:true,group:'Schedule'}]);}
  finally {delete globalThis.document;}
  h.context.options=options;
  const samples=h.$('option-samples'),advanced=h.$('proseg-advanced');
  assert.deepEqual(h.problems(),[]);assert.equal(advanced.open,false);
  samples.value='-5';samples.valid=false;                     // below its minimum
  assert.deepEqual(h.problems(),['option-samples']);
  await h.$('start').click();
  assert.equal(h.created.length,0,'nothing is started');
  assert.ok(h.note().includes('Samples: Use a whole number of at least 0.'));
  assert.equal(advanced.open,true,'its closed section opens');
  assert.ok(samples.classList.contains(NEEDS_CLASS));assert.equal(h.document.activeElement,samples);
  // After the region, as on the page.
  h.run('setRegion({kind:"box",box:[100,0,110,400]})');await h.$('start').click();
  assert.match(h.note(),/Region: .*Samples: /);
  h.run('setRegion({kind:"box",box:[100,0,400,400]})');
  samples.value='300';samples.valid=true;samples.emit('input');h.run('updatePicker()');
  assert.equal(h.note(),'');
  await h.$('start').click();
  assert.equal(h.created.length,1);assert.deepEqual(plain(h.created[0][3]),{samples:300});
});

test('Start reads Starting… while it works, also when the run type changes meanwhile; then the label of the type chosen',async()=>{
  const h=await patchPicker(),gate=deferred(),start=h.$('start');
  h.api.createStain=async(...args)=>{h.created.push(args);await gate.promise;return{id:'new-stain'};};
  h.fire(400);h.previews[0].resolve(suggestion(2));await settle();
  assert.equal(start.textContent,STAIN_PATCH_TEXT.start);
  const pressed=start.click();await settle();
  assert.equal(h.created.length,1);assert.equal(start.textContent,stain.PICKER_TEXT.starting);
  h.$('run-kind').value='stain';h.$('run-kind').emit('change');
  assert.equal(start.textContent,stain.PICKER_TEXT.starting,'busy: its label stays');assert.equal(start.getAttribute('aria-busy'),'true');
  gate.resolve();await pressed;
  assert.equal(start.getAttribute('aria-busy'),null);
  assert.equal(start.textContent,stain.STAIN_TEXT.start,'the label for the run type chosen now');
});

test('Training patches from Empty: the hidden Cellpose settings are never named; Cellpose\'s defaults are sent',async()=>{
  const h=await patchPicker();
  h.$('stain-auto').checked=false;h.$('stain-auto').emit('change');
  h.$('stain-diameter').value='5';h.$('stain-flow').value='12';h.$('stain-niter').value='0';h.$('stain-flow').emit('input');
  assert.deepEqual(h.problems(),['stain-diameter','stain-flow','stain-niter'],'Model outlines: shown, so named');
  h.$('patch-start-from').value='empty';h.$('patch-start-from').emit('change');
  assert.equal(h.$('stain-tuning').hidden,true);
  assert.deepEqual(h.problems(),[],'Empty: hidden, so never named');
  const started=h.$('start').click();
  h.previews.at(-1).resolve(suggestion(2));await started;
  const sent=plain(h.created[0][2]);
  assert.equal(sent.start_from,'empty');assert.equal(sent.method,'cellpose');
  assert.deepEqual([sent.diameter,sent.flow_threshold,sent.cellprob_threshold,sent.niter,sent.min_size,sent.augment],[null,.4,0,null,15,false]);
  assert.deepEqual(sent.normalize,{normalize:true,lowhigh:null,percentile:[1,99],tile_norm_blocksize:0,sharpen_radius:0,smooth_radius:0,invert:false});
  const {boxes,...settings}=sent;assert.deepEqual(plain(h.previews.at(-1).stain),settings,'as the boxes were asked for');
  // Model outlines again: the fields as the user left them.
  h.$('patch-start-from').value='model';h.$('patch-start-from').emit('change');
  assert.deepEqual(h.problems(),['stain-diameter','stain-flow','stain-niter']);
});

test('Start\'s reasons are read out: the picker\'s error is an alert, and a note keeps its colour in the knob guide',()=>{
  const css=readFileSync(new URL('../src/explain.css',import.meta.url),'utf8');
  assert.match(html,/<p id="picker-error" class="error" role="alert" hidden><\/p>/);
  assert.match(css,/\.missing-note p\{[^}]*color:inherit[^}]*\}/,'the guide greys its paragraphs; a note\'s lead keeps the note\'s colour');
});

test('New run is never greyed out: without a sample a press says to open one; with one it opens the picker',async()=>{
  assert.doesNotMatch(html,/id="new-run"[^>]*disabled/,'not in the page as it loads either');
  assert.doesNotMatch(source,/\$\('new-run'\)\.disabled/,'nothing has to enable it');
  const h=harness();await h.run('boot()');
  assert.equal(h.$('new-run').disabled,false);
  await h.$('new-run').click();
  assert.ok(h.note('new-run').includes(stain.PICKER_TEXT.openSample));assert.equal(h.state.mode,'empty');
  await h.run('enterPick("sample")');
  assert.equal(h.note('new-run'),'','an open sample takes the note away');
  h.state.mode='run';                                      // as from a run of this sample
  await h.$('new-run').click();
  assert.equal(h.state.mode,'pick');assert.deepEqual(h.problems(),[]);
});

// ---- the first screen's two tools and Training patches inside Stain segmentation -----------------------------------------
test('the first screen: two tools and no tool remembered; a card shows its Open sample… and the shared Recent, Back the cards again',async()=>{
  const h=harness({samples:[{...sample,path:'/data/sample.zarr'}]});
  h.context.localStorage={getItem:()=>'stain',setItem(){}};             // whatever is stored, no tool comes back
  await h.run('boot()');
  const shown=id=>!h.$(id).hidden;
  assert.equal(h.state.mode,'empty');assert.equal(h.state.tool,null);
  assert.deepEqual(['tool-choice','tool-start','tool-name','open-sample','new-run'].map(shown),[true,false,false,false,false],
    'the cards first: the header\'s Open sample… and New run wait for a tool');
  assert.equal(h.$('run-kind').parentElement.hidden,true,'the run type is never shown');
  h.$('tool-stain').click();
  assert.equal(h.state.tool,'stain');assert.equal(h.$('run-kind').value,'stain','its picker segments the image');
  assert.deepEqual(['tool-choice','tool-start','tool-name','open-sample','new-run'].map(shown),[false,true,true,true,true]);
  assert.deepEqual([h.$('tool-title').textContent,h.$('tool-help').textContent,h.$('tool-name').textContent],
    [TOOLS.stain.name,TOOLS.stain.open,TOOLS.stain.name],'the header names the tool');
  assert.equal(h.$('recent-samples').children[0].className,'recent-item','Recent is shared by both tools');
  assert.equal(h.document.activeElement,h.$('empty-open-sample'));
  await h.$('new-run').click();assert.ok(h.note('new-run').includes(stain.PICKER_TEXT.openSample));
  h.$('tool-back').click();
  assert.equal(h.note('new-run'),'','the note goes with the hidden New run');
  assert.equal(h.state.tool,null);assert.deepEqual(['tool-choice','tool-start','tool-name','open-sample'].map(shown),[true,false,false,false]);
  assert.equal(h.document.activeElement,h.$('tool-stain'));
  h.$('tool-proseg').click();
  assert.equal(h.$('run-kind').value,'proseg');assert.equal(h.$('tool-name').textContent,'Proseg Live');assert.equal(h.$('tool-help').textContent,TOOLS.proseg.open);
  for(const text of [h.$('tool-help').textContent,TOOLS.stain.open])assert.deepEqual(bannedWords(text),[],text);
});

test('the header\'s Compare is Proseg Live\'s (not on the cards, not in Stain segmentation); without a run a press says to open one',async()=>{
  assert.doesNotMatch(html,/id="compare-button"[^>]*disabled/);
  const h=harness();await h.run('boot()');
  const shown=()=>!h.$('compare-button').hidden;
  assert.equal(shown(),false,'the cards: no tool yet');
  h.$('tool-stain').click();assert.equal(shown(),false,'Stain segmentation compares nothing');
  h.$('tool-back').click();h.$('tool-proseg').click();assert.equal(shown(),true);
  assert.deepEqual([h.$('compare-button').textContent,h.$('compare-button').title],[COMPARE_TEXT.button,COMPARE_TEXT.buttonTooltip]);
  await h.run('enterPick("sample")');
  assert.deepEqual(h.context.compareView.closes,[false],'a view over a run goes as the picker opens');
  assert.deepEqual(h.priors.segments.samples,[null,'sample'],'the picker hands the sample\'s segmentation layers their sample');
  await h.$('compare-button').click();
  assert.ok(h.note('compare-button').includes(COMPARE_TEXT.needsRun));assert.deepEqual(h.opened,[]);
  h.$('home').click();assert.equal(h.note('compare-button'),'','the note goes with the hidden button');
  for(const text of [COMPARE_TEXT.button,COMPARE_TEXT.buttonTooltip,COMPARE_TEXT.needsRun,COMPARE_TEXT.needsCheck])assert.deepEqual(bannedWords(text),[],text);
});

test('a sample opened in a tool enters that tool\'s picker: Stain segmentation segments (Training patches closed), Proseg Live runs proseg',async()=>{
  const h=harness();h.state.health.stain={available:true};h.api.stainModels=async()=>({models});
  await h.run('boot()');h.$('tool-stain').click();
  await h.run('enterPick("sample")');
  assert.equal(h.state.mode,'pick');assert.equal(h.$('run-kind').value,'stain');
  assert.deepEqual([h.$('stain-picker').hidden,h.$('proseg-picker').hidden,h.$('patch-tool').open],[false,true,false]);
  assert.equal(h.$('start').textContent,stain.STAIN_TEXT.start);assert.equal(h.$('tool-name').textContent,'Stain segmentation');
  // Back home and Proseg Live: the same sample, its proseg picker.
  h.$('home').click();
  assert.deepEqual([h.state.mode,h.state.tool,h.state.sample,h.$('tool-choice').hidden,h.$('picker').hidden],['empty',null,null,false,true]);
  h.$('tool-proseg').click();await h.run('enterPick("sample")');
  assert.equal(h.$('run-kind').value,'proseg');
  assert.deepEqual([h.$('stain-picker').hidden,h.$('proseg-picker').hidden,h.$('proseg-advanced').hidden],[true,false,false]);
});

test('Training patches is a tool inside Stain segmentation: open, Start reads Make training patches and the boxes are suggested; closed, Start segments',async()=>{
  const h=await stainPicker(),tool=h.$('patch-tool'),start=h.$('start');
  h.state.sample.images[0].um_per_px=.5;h.$('patch-start-from').value='model';
  assert.deepEqual([tool.open,h.$('patch-picker').hidden,start.textContent],[false,true,stain.STAIN_TEXT.start]);
  tool.open=true;tool.emit('toggle');await settle();
  assert.equal(h.$('run-kind').value,'patches');assert.equal(h.$('patch-picker').hidden,false);
  assert.equal(start.textContent,STAIN_PATCH_TEXT.start);assert.equal(h.pending(400),1,'its boxes are due');
  h.fire(400);h.previews[0].resolve(suggestion(2));await settle();
  assert.equal(h.panel.boxes.length,2,'on the image');
  tool.open=false;tool.emit('toggle');
  assert.deepEqual([h.$('run-kind').value,start.textContent,h.panel.boxes,h.pending(400)],['stain',stain.STAIN_TEXT.start,null,0],
    'closed: the boxes go and Start segments the region');
  tool.emit('toggle');assert.equal(h.$('run-kind').value,'stain','a toggle that changes nothing sets nothing');
  // The harness's way: the run type set on its own opens and closes the section with it, the tool with it.
  h.$('run-kind').value='patches';h.$('run-kind').emit('change');
  assert.deepEqual([tool.open,h.state.tool,start.textContent],[true,'stain',STAIN_PATCH_TEXT.start]);
  h.$('run-kind').value='stain';h.$('run-kind').emit('change');assert.equal(tool.open,false);
  // Start in each: the region segmented, then (the section open again) the item made with the boxes shown.
  await start.click();assert.deepEqual(plain(h.created.map(c=>c[3])),['stain']);
  tool.open=true;tool.emit('toggle');await settle();h.fire(400);h.previews.at(-1).resolve(suggestion(2));await settle();
  await start.click();
  assert.deepEqual(plain(h.created.map(c=>c[3])),['stain','patches']);assert.equal(h.created[1][2].boxes.length,2);
});

test('New run stays in the tool; the brand goes home (the two cards, the sample closed) and a Proseg Live run started there leaves the run type proseg',async()=>{
  const h=harness();h.state.health.stain={available:true};h.api.stainModels=async()=>({models});
  await h.run('boot()');h.$('tool-stain').click();await h.run('enterPick("sample")');
  h.state.mode='run';                                      // as from a run of this sample
  await h.$('new-run').click();
  assert.deepEqual([h.state.mode,h.state.tool,h.$('run-kind').value],['pick','stain','stain'],'New run stays in Stain segmentation');
  // "New run with this setting" (a proseg run's prefill) is a Proseg Live picker whatever the tool was.
  await h.run('enterPick("sample",{region:{kind:"full"},prior:{cells:null,nuclei:null},values:{}})');
  assert.deepEqual([h.state.tool,h.$('run-kind').value,h.$('tool-name').textContent],['proseg','proseg','Proseg Live']);
  h.$('home').click();
  assert.deepEqual([h.state.mode,h.state.tool,h.state.sample,h.$('tool-name').hidden,h.$('new-run').hidden],['empty',null,null,true,true]);
  assert.deepEqual(h.opened,[],'nothing opened or stopped: runs keep going');
});

// ---- what an open found, a sample without transcripts, an engine that does not answer
test('what an open found is one line each under the sample\'s name, with an ⓘ holding the rest; the lines go with the sample', async () => {
  const notes = [
    { code: 'pixel_frame', text: 'These transcripts seem to be in pixels, not micrometres: sizes shown as µm are pixels.', detail: 'Every image has 1.0 µm per pixel.' },
    { code: 'several_points', text: 'This sample has 2 transcript elements: this app uses a_points.', detail: 'Not used: b_points.' },
    { code: 'skipped_images', text: '1 image is left out.', detail: 'he_image: rotated.' },
  ];
  const h = harness(); h.api.sample = async () => ({ ...sample, notes });
  await h.run('boot()'); h.$('tool-proseg').click();
  await h.run('enterPick("sample")');
  const host = h.$('picker-notes');
  assert.equal(host.hidden, false);
  assert.deepEqual(host.children.map(line => [line.getAttribute('data-note'), line.children[0].textContent, line.children[1].title]),
    notes.map(n => [n.code, n.text, n.detail]));
  assert.equal(h.$('picker-sample-info').textContent, '4,000,000 transcripts · 1 shapes layers');
  assert.equal(h.state.tool, 'proseg', 'notes change no tool');
  h.$('home').click();
  assert.equal(host.hidden, true, 'closed with the sample');
  h.api.sample = async () => sample; h.$('tool-proseg').click(); await h.run('enterPick("sample")');
  assert.deepEqual([host.hidden, host.children.length], [true, 0], 'a sample without notes says nothing');
});

test('a sample without transcripts opens in Stain segmentation whichever tool opened it, and the picker says why', async () => {
  const h = harness(); h.state.health.stain = { available: true }; h.api.stainModels = async () => ({ models });
  h.context.fetch = () => new Promise(() => {});           // the stain preview's image: not needed here
  const images = { ...sample, transcripts: null, tools: ['stain'], shapes: [], notes: [], prior_options: { cells: [], nuclei: [] },
    suggested_prior: { cells: null, nuclei: null }, images: [{ key: 'image', channels: ['DAPI'], levels: 1, um_per_px: .5 }] };
  h.api.sample = async () => images;
  await h.run('boot()'); h.$('tool-proseg').click();
  await h.run('enterPick("sample")');
  assert.deepEqual([h.state.mode, h.state.tool, h.$('run-kind').value, h.$('tool-name').textContent], ['pick', 'stain', 'stain', TOOLS.stain.name]);
  assert.deepEqual([h.$('stain-picker').hidden, h.$('proseg-picker').hidden, h.$('compare-button').hidden], [false, true, true]);
  const line = h.$('picker-notes').children[0];
  assert.equal(h.$('picker-notes').hidden, false);
  assert.deepEqual([line.getAttribute('data-note'), line.children[0].textContent, line.children[1].title], ['image_only', NOTE_TEXT.imageOnly, NOTE_TEXT.imageOnlyDetail]);
  assert.equal(h.$('picker-sample-info').textContent, `${NOTE_TEXT.noTranscripts} · 0 shapes layers`);
  assert.equal(h.$('stain-image').value, 'image');
  // Set to proseg from outside (the harness's way): a press of Start says why, and nothing is made.
  h.$('run-kind').value = 'proseg'; h.$('run-kind').emit('change');
  assert.deepEqual(h.problems(), [NOTE_TEXT.imageOnly]);
  await h.$('start').click();
  assert.ok(h.note().includes(NOTE_TEXT.imageOnly)); assert.deepEqual(h.created, []);
  // A sample with transcripts, opened from Proseg Live: its picker, no note.
  h.api.sample = async () => sample; h.$('home').click(); h.$('tool-proseg').click(); await h.run('enterPick("sample")');
  assert.deepEqual([h.state.tool, h.$('run-kind').value, h.$('picker-notes').hidden], ['proseg', 'proseg', true]);
  assert.deepEqual(h.problems(), []);
});

test('the first three requests are given up after 20 s: the first screen says the engine cannot be reached, with Reload', async () => {
  const h = harness({ samples: [{ ...sample, path: '/data/sample.zarr' }] }), signals = [];
  h.api.health = signal => { signals.push(signal); return new Promise(() => {}); };   // never answers
  let reloads = 0; h.context.location.reload = () => reloads++;
  const booting = h.run('boot()');
  await settle();
  assert.equal(h.$('engine-down').hidden, true, 'not before the time is up');
  assert.equal(h.pending(20000), 1, 'one deadline of 20 s');
  h.$('tool-proseg').click();                               // a card chosen while it waits
  h.fire(20000); await booting;
  assert.deepEqual([h.state.tool, h.$('tool-name').hidden, h.$('open-sample').hidden, h.$('compare-button').hidden], [null, true, true, true],
    'nothing to open without the engine');
  assert.deepEqual([h.state.mode, h.$('empty').hidden, h.$('engine-down').hidden, h.$('tool-choice').hidden, h.$('tool-start').hidden],
    ['empty', false, false, true, true], 'said in place of the tools');
  assert.equal(h.$('engine-down').getAttribute('role'), 'alert');
  assert.deepEqual([h.$('engine-down-title').textContent, h.$('engine-down-why').textContent, h.$('engine-reload').textContent],
    [SHELL_TEXT.engineDown, 'It did not answer within 20 seconds.', SHELL_TEXT.reload]);
  assert.equal(signals[0].aborted, true, 'the request is given up');
  assert.equal(h.document.activeElement, h.$('engine-reload'));
  assert.deepEqual(h.notices, []); assert.deepEqual(h.opened, []);
  h.$('engine-reload').click(); assert.equal(reloads, 1);
  for (const text of [SHELL_TEXT.engineDown, SHELL_TEXT.engineLate, SHELL_TEXT.reload]) assert.deepEqual(bannedWords(text), [], text);
  // A refusal is said at once, in the engine's words; nothing waits for the deadline.
  const refused = harness(); refused.api.runs = async () => { throw Object.assign(new Error('A local same-port Origin is required'), { status: 403 }); };
  await refused.run('boot()');
  assert.deepEqual([refused.$('engine-down').hidden, refused.$('engine-down-why').textContent, refused.pending(20000)],
    [false, 'A local same-port Origin is required', 0]);
  // A key the engine refuses (it started again since): the page is not signed in, which is no engine out of reach.
  const signedOut = harness(); signedOut.api.runs = async () => { throw Object.assign(new Error('Not signed in'), { status: 401 }); };
  await signedOut.run('boot()');
  assert.deepEqual([signedOut.$('engine-down').hidden, signedOut.$('engine-down-title').textContent, signedOut.$('engine-down-why').textContent,
    signedOut.$('engine-reload').hidden, signedOut.pending(20000)], [false, SHELL_TEXT.signedOutTitle, SHELL_TEXT.signedOut, true, 0]);
  assert.deepEqual(signedOut.notices, []);
  // A failure with no reason (an error the engine did not expect): the shell's words, not a sample's (none is open).
  const failed = harness(); failed.api.health = async () => { throw Object.assign(new Error('Internal Server Error'), { status: 500 }); };
  await failed.run('boot()');
  assert.deepEqual([failed.$('engine-down').hidden, failed.$('engine-down-why').textContent], [false, SHELL_TEXT.engineFailed]);
  assert.doesNotMatch(SHELL_TEXT.engineFailed, /sample/); assert.deepEqual(bannedWords(SHELL_TEXT.engineFailed), []);
  // In time, nothing of it shows.
  const fine = harness(); await fine.run('boot()');
  assert.deepEqual([fine.$('engine-down').hidden, fine.$('tool-choice').hidden, fine.pending(20000)], [true, false, 0]);
});

test('a sample with transcripts but no image, opened from Stain segmentation, opens in Proseg Live and says why', async () => {
  // It moved to Proseg Live without a word (the engine lists proseg only for it): the picker now says why.
  const h = harness(); h.state.health.stain = { available: true }; h.api.stainModels = async () => ({ models });
  h.context.fetch = () => new Promise(() => {});
  h.api.sample = async () => ({ ...sample, tools: ['proseg'], notes: [], images: [], image_key: null, overview: null });
  await h.run('boot()'); h.$('tool-stain').click();
  await h.run('enterPick("sample")');
  assert.deepEqual([h.state.mode, h.state.tool, h.$('run-kind').value], ['pick', 'proseg', 'proseg']);
  const line = h.$('picker-notes').children[0];
  assert.equal(h.$('picker-notes').hidden, false);
  assert.deepEqual([line.getAttribute('data-note'), line.children[0].textContent, line.children[1].title], ['no_image', NOTE_TEXT.noImage, NOTE_TEXT.noImageDetail]);
});

test('Resolution offers the sample scales by their sdata names, posts the chosen level and warns below ten cell pixels',async()=>{
  const h=await stainPicker();
  // The labels an earlier engine sent: the page names each level itself (scale0, scale1, …).
  h.state.sample.images[0].scales=[{level:0,factor:1,um_per_px:.25,label:'Full'},
    {level:1,factor:2,um_per_px:.5,label:'1/2'},{level:2,factor:4,um_per_px:1,label:'1/4'}];
  h.run('fillResolution()');assert.deepEqual(h.$('stain-level').children.map(o=>o.textContent),['scale0','scale1','scale2']);
  assert.deepEqual(h.$('stain-level').children.map(o=>o.title),['scale0: 0.25 µm per pixel, the image as stored',
    'scale1: 0.5 µm per pixel, 2x coarser than scale0','scale2: 1 µm per pixel, 4x coarser than scale0']);
  h.$('stain-auto').checked=false;h.$('stain-diameter').value='30';h.$('stain-level').value='2';h.$('stain-level').emit('change');
  assert.match(h.$('stain-resolution-info').textContent,/1 µm per pixel · cells about 8 px across/);
  assert.equal(h.$('stain-resolution-warning').hidden,false);assert.deepEqual(h.problems(),[],'warning does not block Start');
  await h.$('start').click();assert.equal(h.created.at(-1)[2].level,2);assert.equal(h.created.at(-1)[2].diameter,30,'pixel settings remain at full size');
});
test('changing images preserves a matching physical resolution within five percent',async()=>{
  const h=await stainPicker();
  h.state.sample.images[0].scales=[{level:0,factor:1,um_per_px:.25,label:'scale0'},{level:1,factor:2,um_per_px:.5,label:'scale1'}];
  h.run('fillResolution()');h.$('stain-level').value='1';h.$('stain-level').emit('change');
  h.state.sample.images.push({key:'other',channels:['DAPI'],um_per_px:.49,scales:[{level:0,factor:1,um_per_px:.49,label:'scale0'}]});
  h.$('stain-image').value='other';h.run('fillStainChannels()');assert.equal(h.$('stain-level').value,'0');
  assert.match(h.$('stain-resolution-info').textContent,/0.49 µm per pixel/);
  h.state.sample.images[1].scales=[{level:0,factor:1,um_per_px:.2,label:'scale0'},{level:1,factor:2,um_per_px:.4,label:'scale1'}];
  h.run('fillResolution()');assert.equal(h.$('stain-level').value,'0','no near match returns to scale0');
});
test('resolution validates the region in level pixels and changes the Training patches preview request',async()=>{
  const h=await patchPicker();
  h.state.sample.images[0].scales=[{level:0,factor:1,um_per_px:.5,label:'scale0'},{level:1,factor:2,um_per_px:1,label:'scale1'}];
  h.run('fillResolution()');const before=plain(h.run('patchRequest()'));
  h.$('stain-level').value='1';h.$('stain-level').emit('change');
  assert.equal(h.run('patchRequest().stain.level'),1);assert.notDeepEqual(plain(h.run('patchRequest()')),before);
  h.run('setRegion({kind:"box",box:[0,0,25,25]});updatePicker()');
  assert.ok(h.problems().includes('stain-level'));
  assert.equal(h.run('pickerProblems().find(p=>p.field==="stain-level").message'),'the region is only 25 pixels across at scale1');
});
test('Training patches at a smaller copy: the size that fits and a region too small for patches count that level\'s pixels',async()=>{
  const h=await patchPicker(),P=stain.PICKER_TEXT;
  h.state.sample.images[0].scales=[{level:0,factor:1,um_per_px:.5},{level:1,factor:2,um_per_px:1}];
  h.run('fillResolution()');h.$('stain-level').value='1';h.$('stain-level').emit('change');
  h.run('setRegion({kind:"box",box:[100,0,400,300]});updatePicker()');        // 300 µm: 300 pixels at scale1, 600 at scale0
  h.$('patch-size').value='500';h.$('patch-size').emit('input');
  assert.deepEqual(h.problems(),['patch-size']);assert.equal(h.run('pickerProblems()[0].message'),P.sizeFits.replace('{largest}','300'));
  h.$('patch-size').value='300';h.$('patch-size').emit('input');assert.deepEqual(h.problems(),[]);
  // 50 µm: 50 pixels at scale1, too small for any patch (64 is the smallest); at scale0 it holds one.
  h.$('patch-size').value='64';h.run('setRegion({kind:"box",box:[100,0,150,50]});updatePicker()');
  assert.deepEqual(h.problems(),[P.patchRegion]);
  h.$('stain-level').value='0';h.$('stain-level').emit('change');assert.deepEqual(h.problems(),[]);
});
test('the Resolution note sizes what Cellpose outlines: cells with a cell body channel, also when InstanSeg\'s Find was left at Nuclei',async()=>{
  const h=await stainPicker();
  Object.assign(h.state.sample,{shapes:[{key:'cell_boundaries',median_area_um2:200},{key:'nucleus_boundaries',median_area_um2:40}],
    suggested_prior:{cells:{kind:'shapes',key:'cell_boundaries'},nuclei:{kind:'shapes',key:'nucleus_boundaries'}}});
  h.state.sample.images[0].scales=[0,1,2].map(level=>({level,factor:2**level,um_per_px:.2125*2**level,um_per_px_y:.2125*2**level}));
  h.run('fillResolution()');h.$('stain-level').value='2';
  h.$('stain-output').value='nuclei';                      // InstanSeg's Find, hidden with Cellpose, as an earlier choice left it
  h.$('stain-cytoplasm').value='Boundary';h.$('stain-level').emit('change');
  assert.equal(h.$('stain-resolution-info').textContent,'0.85 µm per pixel · cells about 19 px across');
  assert.equal(h.$('stain-resolution-warning').hidden,true);
  // No cell body channel: Cellpose outlines nuclei, small at this level.
  h.$('stain-cytoplasm').value='';h.$('stain-level').emit('change');
  assert.equal(h.$('stain-resolution-info').textContent,'0.85 µm per pixel · cells about 8 px across');
  assert.equal(h.$('stain-resolution-warning').hidden,false);
});

test('a sample whose image goes leaves nothing of that image\'s resolution: the readouts, the warning and the patch note are cleared',async()=>{
  const h=await stainPicker();
  h.state.sample.images[0].scales=[0,1,2].map(level=>({level,factor:2**level,um_per_px:.2125*2**level,um_per_px_y:.2125*2**level}));
  Object.assign(h.state.sample,{shapes:[{key:'cell_boundaries',median_area_um2:200}],suggested_prior:{cells:{kind:'shapes',key:'cell_boundaries'}}});
  h.run('fillResolution()');h.$('stain-level').value='2';h.$('stain-level').emit('change');
  h.$('run-kind').value='patches';h.$('patch-size').value='256';h.run('updatePicker()');
  assert.equal(h.$('stain-resolution-info').textContent,'0.85 µm per pixel · cells about 19 px across');
  assert.match(h.$('patch-size-scale').textContent,/^256 px = 218 µm across at scale2$/);
  h.state.sample.images=[];
  h.run('fillStainImages(state.sample);updatePicker()');
  assert.equal(h.$('stain-resolution-info').textContent,'');assert.equal(h.$('stain-resolution-info').title,'');
  assert.equal(h.$('stain-resolution-warning').hidden,true);assert.equal(h.$('stain-level').title,'');
  assert.equal(h.$('stain-level').resolution,null);assert.equal(h.$('patch-size-scale').textContent,'');
});

test('resolution and patch size readouts use the chosen level, and pixel-frame samples make no micrometre claim',async()=>{
  const h=harness();await h.run('boot()');await h.run('enterPick("sample")');
  h.state.sample.images=[{key:'image',channels:['DAPI'],um_per_px:.25,scales:[
    {level:0,factor:1,um_per_px:.25,label:'scale0'},{level:1,factor:2,um_per_px:.5,label:'scale1'}]}];
  h.state.sample.image_key='image';h.run('fillStainImages(state.sample)');h.$('run-kind').value='patches';
  h.$('stain-level').value='1';h.$('stain-level').onchange();h.$('patch-size').value='256';h.run('updatePicker()');
  assert.equal(h.$('patch-size-scale').textContent,'256 px = 128 µm across at scale1');
  h.state.sample.notes=[{code:'pixel_frame'}];h.run('updatePicker()');
  assert.doesNotMatch(h.$('stain-resolution-info').textContent,/µm/);assert.equal(h.$('patch-size-scale').textContent,'');
});
