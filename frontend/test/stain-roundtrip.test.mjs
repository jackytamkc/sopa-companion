import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { domFixture, deferred, settle } from './dom-fixture.mjs';
import { CellposeSettings, TissueView, fillCellposeSettings, updateNormalizeFields, settingsReport, SETTINGS_TEXT } from '../src/stain-settings.mjs';
import { StainExclusions, EXCLUSION_TEXT } from '../src/stain-exclusions.mjs';
import { StainView, stainValues, validStain } from '../src/stain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { encodeVectors } from './vectors-fixture.mjs';

// The PLCV bytes as the engine writes them (engine/vectors.py encode_cells), independent of the page's parsing code:
// header, metadata, ids 1..3, sizes, flags (padded to four bytes) and the channel's brightness when one is asked for.
function cellValuesBuffer(channel=null) {
  const n=3,json=new TextEncoder().encode(JSON.stringify({set:'resolved',areas_version:0,brightness:channel?[channel]:[]}));
  const header=Math.ceil((24+json.length)/4)*4,buffer=new ArrayBuffer(header+24+4+(channel?12:0)),view=new DataView(buffer);
  [0x56434c50,1,header,n,channel?1:0,json.length].forEach((v,i)=>view.setUint32(i*4,v,true));
  new Uint8Array(buffer,24,json.length).set(json);new Uint32Array(buffer,header,n).set([1,2,3]);
  new Float32Array(buffer,header+12,n).set([5,20,80]);
  if(channel)new Float32Array(buffer,header+28,n).set([1,NaN,100]);return buffer;
}

function dom(t) {
  const previous=globalThis.document,fixture=domFixture();
  globalThis.document=fixture.document;
  fixture.document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  t.after(()=>{globalThis.document=previous;});return fixture;
}
const saved={found:true,from:'patch-02_seg.npy',settings:{model:'trained',diameter:23,flow_threshold:.55,
  cellprob_threshold:-1.5,normalize:{percentile:[2,98],tile_norm_blocksize:128,sharpen_radius:1,smooth_radius:0,invert:false}},
  differences:{diameter:{'patch-01_seg.npy':30,'patch-02_seg.npy':23}},unknown_model:null};

test('Cellpose defaults, off normalisation and extra settings have validated create values',t=>{
  const {$}=dom(t),read=()=>stainValues(id=>$(id)?.type==='checkbox'?$(id).checked:$(id)?.value);
  $('stain-image').value='im';$('stain-nucleus').value='DAPI';
  const value=read();assert.equal(value.tissue_only,true);assert.equal(value.niter,null);assert.equal(value.min_size,15);
  assert.equal(value.augment,false);assert.deepEqual(value.normalize,{
    normalize:true,lowhigh:null,percentile:[1,99],tile_norm_blocksize:0,sharpen_radius:0,smooth_radius:0,invert:false});
  const images=[{key:'im',channels:['DAPI']}];assert.ok(validStain(value,images));
  $('stain-normalize').checked=false;$('stain-tissue').checked=false;$('stain-niter').value='25';
  assert.equal(read().normalize.percentile,null);assert.equal(read().tissue_only,false);assert.equal(read().niter,25);
  assert.equal(read().normalize.normalize,false);assert.ok(validStain(read(),images));
  for(const change of [{niter:0},{niter:2.5},{min_size:-2},{min_size:100001},
    {normalize:{...value.normalize,percentile:[99,1]}},{normalize:{...value.normalize,tile_norm_blocksize:1.5}},
    {normalize:{...value.normalize,normalize:'false'}}])
    assert.equal(validStain({...value,...change},images),false);
});

test('GUI settings fill the model and every brightness field and explain differences and unknown models',t=>{
  const {$}=dom(t);$('stain-niter').value='42';
  const lines=fillCellposeSettings($,saved);settingsReport($('stain-settings-report'),saved,lines);
  assert.equal($('stain-model').value,'trained');assert.equal($('stain-auto').checked,false);
  assert.equal($('stain-flow').value,'0.55');assert.equal($('stain-probability').value,'-1.5');
  assert.equal($('stain-diameter').value,'23');assert.equal($('stain-tile-norm').value,'128');
  assert.equal($('stain-low').value,'2');assert.equal($('stain-high').value,'98');assert.equal($('stain-sharpen').value,'1');
  assert.equal($('stain-niter').value,'42','the GUI does not save niter');
  // Reported by Cellpose's own names, as the picker labels the fields.
  assert.deepEqual(lines,['Model: trained','diameter: 23','flow_threshold: 0.55','cellprob_threshold: -1.5',
    'normalize: percentile 2–98','tile_norm_blocksize: 128','sharpen_radius: 1','smooth_radius: 0','invert: off']);
  assert.match($('stain-settings-report').textContent,/different settings/);
  const unknown={...saved,unknown_model:'/old/model',settings:{...saved.settings,model:null,
    normalize:{...saved.settings.normalize,normalize:false,percentile:[2,98],invert:true}}};
  settingsReport($('stain-settings-report'),unknown,fillCellposeSettings($,unknown));
  assert.equal($('stain-model').value,'trained');assert.equal($('stain-normalize').checked,false);
  assert.equal($('stain-invert').checked,true);assert.match($('stain-settings-report').textContent,/current model was kept/);
  for(const words of [...Object.values(SETTINGS_TEXT),...Object.values(EXCLUSION_TEXT),...lines])assert.deepEqual(bannedWords(words),[]);
});

test('imported fixed intensity ranges survive the form and take precedence over percentiles',t=>{
  const {$}=dom(t),read=()=>stainValues(id=>$(id)?.type==='checkbox'?$(id).checked:$(id)?.value);
  $('stain-image').value='im';$('stain-nucleus').value='DAPI';
  const fixed={...saved,settings:{...saved.settings,normalize:{...saved.settings.normalize,lowhigh:[-12.5,4000]}}};
  const lines=fillCellposeSettings($,fixed);
  assert.equal($('stain-normalize').checked,true);assert.equal($('stain-fixed-range').checked,true);
  assert.equal($('stain-more').contains($('stain-intensity-low')),true);
  assert.equal($('stain-intensity-low').value,'-12.5');assert.equal($('stain-intensity-high').value,'4000');
  assert.equal($('stain-low').disabled,true);assert.equal($('stain-high').disabled,true);
  assert.equal($('stain-intensity-low').disabled,false);assert.equal($('stain-intensity-high').disabled,false);
  assert.deepEqual(read().normalize.lowhigh,[-12.5,4000]);assert.equal(read().normalize.percentile,null);
  assert.ok(validStain(read(),[{key:'im',channels:['DAPI']}]));
  assert.ok(lines.includes('lowhigh: -12.5–4000'));assert.ok(lines.includes('normalize: fixed range'));
  for(const words of lines)assert.deepEqual(bannedWords(words),[]);
  $('stain-normalize').checked=false;updateNormalizeFields($);
  assert.equal(read().normalize.normalize,false);assert.equal($('stain-intensity-low').disabled,true);
  $('stain-normalize').checked=true;updateNormalizeFields($);
  assert.deepEqual(read().normalize.lowhigh,[-12.5,4000]);
  $('stain-fixed-range').checked=false;updateNormalizeFields($);
  assert.equal(read().normalize.lowhigh,null);assert.deepEqual(read().normalize.percentile,[2,98]);
  assert.equal($('stain-low').disabled,false);assert.equal($('stain-intensity-low').disabled,true);
  fillCellposeSettings($,{...saved,settings:{...saved.settings,normalize:{...saved.settings.normalize,percentile:null}}});
  assert.equal($('stain-normalize').checked,true,'an absent normalize flag defaults to true');
  assert.equal($('stain-fixed-range').checked,false);assert.equal($('stain-intensity-low').value,'');
  assert.deepEqual(read().normalize.percentile,[1,99]);
});

test('fixed intensity ranges require two finite ordered intensities within the engine bounds',t=>{
  const {$}=dom(t),read=()=>stainValues(id=>$(id)?.type==='checkbox'?$(id).checked:$(id)?.value);
  $('stain-image').value='im';$('stain-nucleus').value='DAPI';$('stain-fixed-range').checked=true;
  for(const [low,high] of [['','10'],['0',''],['x','10'],['0','Infinity'],['10','10'],['10','9'],['-1000000001','10'],['0','1000000001']]){
    $('stain-intensity-low').value=low;$('stain-intensity-high').value=high;
    assert.equal(validStain(read(),[{key:'im',channels:['DAPI']}]),false,`${low}, ${high}`);
  }
});

test('tissue cache is keyed by image, ignores stale reads and hides when switched off',async()=>{
  const jobs=[],draws=[];const view=new TissueView({tissue:(sid,image,signal)=>{
    const d=deferred();jobs.push({sid,image,signal,...d});return d.promise;
  }},()=>draws.push(1),assert.fail);
  view.update('s','a',true);view.update('s','b',true);assert.ok(jobs[0].signal.aborted);
  jobs[0].resolve({polygons:[[[1,2]]]});await settle();assert.deepEqual(view.polygons,[]);
  const polygons=[[[0,0],[10,0],[0,10]]];jobs[1].resolve({polygons});await settle();assert.deepEqual(view.polygons,polygons);
  const before=draws.length;view.update('s','b',true);assert.equal(draws.length,before);
  view.update('s','b',false);assert.deepEqual(view.polygons,[]);
  view.update('s','b',true);assert.deepEqual(view.polygons,polygons);assert.equal(jobs.length,2);
});

const run={id:'r',kind:'stain',state:'done',box:[0,0,200,200],progress:{done:1,revision:9},exclusions:{version:0},summary_model:{cells:100}};
function exclusions(t) {
  const {$,document}=dom(t),canvas=$('stage');
  canvas.setPointerCapture=()=>{};canvas.hasPointerCapture=()=>true;canvas.releasePointerCapture=()=>{};
  const writes=[],reads=[],errors=[],refresh=[],spreads=[];
  const read=()=>{const d=deferred();reads.push(d);return d.promise;};
  const write=(...args)=>{const d=deferred();writes.push({args,...d});return d.promise;};
  const measure=(...args)=>{const d=deferred();spreads.push({args,...d});return d.promise;};
  const panel=new StainExclusions({canvas,scale:1,toWorld:(x,y)=>[x,y],toScreen:(x,y)=>[x,y]},
    {stainExclusions:read,stainExclude:write,stainRestore:write,stainExcludeSize:write,stainBrightness:measure,
      stainExcludeBrightness:write},
    {changed(){},refresh:async(...a)=>refresh.push(a),notice:e=>errors.push(e)});
  const fire=(type,x,y,extra={})=>canvas.emit(type,{button:0,pointerId:1,offsetX:x,offsetY:y,stopImmediatePropagation(){},...extra});
  panel.update(run);t.after(()=>panel.reset());return {panel,reads,writes,spreads,errors,refresh,fire,document};
}

test('lasso posts frame coordinates and version; count and undo update; a click on the area and Delete bring it back',async t=>{
  const {panel,reads,writes,fire,refresh,document}=exclusions(t);
  reads[0].resolve({version:0,areas:[],removed:0});await settle();await panel.button.click();
  fire('pointerdown',10,10);fire('pointermove',80,10);fire('pointermove',80,80);fire('pointerup',10,80);
  assert.equal(writes[0].args[0],'r');assert.equal(writes[0].args[1].version,0);
  assert.equal(writes[0].args[1].points.length,4);
  const data={version:1,areas:[{id:'a',points:writes[0].args[1].points,cells:12}],removed:12};
  writes[0].resolve(data);await settle();assert.equal(panel.total.textContent,'Keeps 88 of 100 cells');
  assert.deepEqual(refresh[0],['r',data]);
  // In the mode, with no outline being drawn, a click inside the area selects it; Delete brings its cells back.
  fire('pointerdown',60,30);fire('pointerup',60,30);assert.equal(panel.snapshot().selected,'a');
  const del={key:'Delete',target:document.body,preventDefault(){this.defaultPrevented=true;},stopImmediatePropagation(){}};
  for(const fn of document.listeners.get('keydown')||[])fn(del);
  assert.equal(del.defaultPrevented,true);assert.deepEqual(writes[1].args,['r','a',1]);
  writes[1].resolve({version:2,areas:[],removed:0});await settle();assert.equal(panel.total.textContent,'Keeps all 100 cells');
  assert.equal(panel.drawing,true,'the mode stays on');assert.deepEqual([panel.snapshot().undo,panel.snapshot().redo],[2,0]);
});

test('exclusions refresh across windows, reject stale writes and end the mode on resume',async t=>{
  const {panel,reads,writes,fire,errors}=exclusions(t);
  reads[0].resolve({version:0,areas:[],removed:0});await settle();
  panel.update({...run,exclusions:{version:3}});assert.equal(reads.length,2);
  reads[1].resolve({version:3,areas:[],removed:0});await settle();await panel.button.click();
  fire('pointerdown',10,10);fire('pointermove',80,10);fire('pointerup',80,80);assert.equal(writes[0].args[1].version,3);
  // A 409: the areas are read again, and another version than the one sent means another window changed them.
  writes[0].reject(Object.assign(Error(),{status:409}));await settle();assert.equal(errors.length,0);
  reads[2].resolve({version:4,areas:[],removed:0});await settle();assert.equal(errors[0],EXCLUSION_TEXT.changed);
  assert.equal(panel.drawing,true,'a refused lasso leaves the mode on');fire('pointerdown',10,10);
  panel.update({...run,state:'running'});fire('pointerup',20,20);assert.equal(writes.length,1);assert.equal(panel.drawing,false);
  assert.equal(panel.button.classList.contains('on'),false);assert.equal(panel.snapshot().outline,0);
  panel.update({...run,id:'other'});reads.at(-1).resolve({version:0,areas:[],removed:0});await settle();
  assert.equal(panel.run.id,'other');
});

test('a lasso drawn while the areas are read waits for them and goes with their version; an older read never wins',async t=>{
  const {panel,reads,writes,fire}=exclusions(t);
  reads[0].resolve({version:0,areas:[],removed:0});await settle();
  panel.update({...run,exclusions:{version:1}});await panel.button.click();
  fire('pointerdown',10,10);fire('pointermove',80,10);fire('pointerup',80,80);
  assert.equal(writes.length,0,'it waits for the areas being read');assert.equal(panel.queued,1);
  reads[1].resolve({version:1,areas:[],removed:0});await settle();assert.equal(writes[0].args[1].version,1);
  writes[0].resolve({version:2,areas:[{id:'new',points:[[10,10],[80,10],[80,80]],cells:5}],removed:5});await settle();
  panel.update({...run,exclusions:{version:1}});reads[2].resolve({version:1,areas:[],removed:0});await settle();
  assert.equal(panel.data.version,2);assert.equal(panel.total.textContent,'Keeps 95 of 100 cells');
});

test('range controls mount in the page containers and commit min and max through the API',async t=>{
  const {panel,reads,writes,document}=exclusions(t);
  reads[0].resolve({version:0,areas:[],cells:100,kept:100,size:null,brightness:{}});await settle();
  const box=panel.sizeChooser.numbers.max;box.value='50';box.emit('blur');
  assert.deepEqual(writes[0].args,['r',{version:0,min:null,max:50}]);
  writes[0].resolve({version:1,areas:[],cells:100,kept:70,size:{min:null,max:50},brightness:{}});await panel.whenSettled();
  assert.equal(panel.total.textContent,'Keeps 70 of 100 cells');
  assert.equal(document.getElementById('stain-exclude-size-apply'),null);
});

test('index.html holds only the containers the range choosers fill: none of the old rows\' fields, Apply, Clear or hidden lines',()=>{
  const html=readFileSync(new URL('../index.html',import.meta.url),'utf8'),at=html.indexOf('<section id="stain-exclusions"');
  const section=html.slice(at,html.indexOf('</section>',at));
  for(const id of ['stain-exclude-size','stain-exclude-bright','stain-exclude-size-apply','stain-exclude-size-clear',
    'stain-exclude-bright-apply','stain-exclude-bright-clear','stain-exclude-small','stain-exclude-dim'])assert.ok(!section.includes(`id="${id}"`),id);
  assert.doesNotMatch(section,/<input|>(Apply|Clear)</);
  assert.match(section,/<div id="stain-exclude-size-row"><\/div>/);
  assert.match(section,/<div id="stain-exclude-bright-row"><select id="stain-exclude-bright-channel" aria-label="Channel"><\/select><\/div>/);
});

test('api.js: the cells\' brightness in a channel (none named: the run\'s default) with its signal, and the rule posted',async()=>{
  const source=readFileSync(new URL('../src/api.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replace(/^export /gm,'');
  const requests=[],window={fetch:async(url,options)=>{requests.push({url,...options});return {ok:true,status:200,json:async()=>({})};},
    dispatchEvent(){}};
  window.fetch.bind=function(){return this;};
  const context=vm.createContext({window,CustomEvent:class {}});
  context.fetch=(...args)=>context.window.fetch(...args);   // api.js requests go through its own observer
  const api=vm.runInContext(`${source}\napi`,context),signal=new AbortController().signal;
  await api.stainBrightness('run 1',null,signal);await api.stainBrightness('run 1','DAPI & co',signal);
  await api.stainExcludeBrightness('run 1',{version:3,channel:'DAPI',min:500,max:null});
  assert.deepEqual(requests.map(r=>[r.method,r.url]),[['GET','/api/v1/runs/run%201/stain/brightness'],
    ['GET','/api/v1/runs/run%201/stain/brightness?channel=DAPI%20%26%20co'],['POST','/api/v1/runs/run%201/stain/exclusions/brightness']]);
  assert.equal(requests[0].signal,signal);assert.equal(requests[1].signal,signal);
  assert.deepEqual(JSON.parse(requests[2].body),{version:3,channel:'DAPI',min:500,max:null});
});

test('area version refetches vectors even when tiles do not change and rejects late old outlines',async t=>{
  const jobs=[],priors={};t.mock.method(globalThis,'fetch',(url,options)=>{const d=deferred();jobs.push({url,options,...d});return d.promise;});
  const view=new StainView({canvas:{width:200,height:200},camera:{x:100,y:100,zoom:1}},priors,{imageView(){}},assert.fail);
  t.after(()=>view.reset());view.update(run);await settle();
  const respond=(i,label)=>jobs[i].resolve({ok:true,arrayBuffer:async()=>encodeVectors({cells:[]},{revision:9,label})});
  respond(0,'before');await settle();view.update({...run,exclusions:{version:1,areas_version:1}});await settle();assert.equal(jobs.length,2);
  view.update({...run,exclusions:{version:2,areas_version:2}});await settle();respond(2,'undo');await settle();respond(1,'removed');await settle();
  assert.equal(priors.vectors.label,'undo');assert.equal(view.shown.exclusions,2);
  assert.equal(view.shown.areas_version,2,'the harness reads which removed areas the outlines shown leave out');
});

test('range commits keep the current vectors; new vector sets reach the cell-values binding',async t=>{
  const jobs=[],seen=[],priors={};t.mock.method(globalThis,'fetch',(url,options)=>{const d=deferred();jobs.push({url,options,...d});return d.promise;});
  const view=new StainView({canvas:{width:200,height:200},camera:{x:100,y:100,zoom:1}},priors,{imageView(){}},assert.fail,vectors=>seen.push(vectors));
  t.after(()=>view.reset());view.update({...run,exclusions:{version:1,areas_version:0}});await settle();
  jobs[0].resolve({ok:true,arrayBuffer:async()=>encodeVectors({cells:[]},{set:'resolved',areas_version:0})});await settle();
  assert.equal(seen[0].set,'resolved');
  view.update({...run,exclusions:{version:2,areas_version:0,size:{min:10,max:50}}});await settle();
  assert.equal(jobs.length,1,'no vector request for a range commit');
  view.reset();assert.equal(seen.at(-1),null);
});

test('PLCV values feed the mounted histogram, live count and set-bound paint without another read during a drag',async t=>{
  const {panel,reads,writes,spreads}=exclusions(t),jobs=[];
  // api.js: stainCells(id, set, channel, signal).
  panel.api.stainCells=(id,set,channel,signal)=>{const d=deferred();jobs.push({id,options:{set,channel},signal,...d});return d.promise;};
  reads[0].resolve({version:0,areas_version:0,areas:[],size:null,brightness:{},cells:3,kept:3});await settle();
  const priors={};panel.priors=priors;panel.setVectors({set:'resolved',areas_version:0});
  assert.deepEqual(jobs[0].options,{set:'resolved',channel:null});jobs[0].resolve(cellValuesBuffer());await settle();
  spreads[0].resolve({channel:'DAPI',channels:['DAPI'],default:'DAPI',p10:1});await settle();
  assert.equal(jobs[1].options.channel,'DAPI');assert.match(panel.brights.textContent,/Measuring DAPI/);
  jobs[1].resolve(cellValuesBuffer('DAPI'));await settle();
  assert.equal(panel.brights.textContent,'');assert.equal(panel.sizeChooser.axis.count,3);
  const slider=panel.brightChooser.sliders.max,axis=panel.brightChooser.axis;
  slider.emit('pointerdown');slider.value=String(axis.position(10)*1000);slider.emit('input');
  assert.equal(writes.length,0);assert.equal(jobs.length,2);assert.equal(panel.total.textContent,'Keeps 2 of 3 cells');
  assert.equal(priors.classes.set,'resolved');assert.deepEqual([...priors.classes.array],[0,0,1]);
  slider.emit('change');assert.equal(writes.length,1);assert.deepEqual(writes[0].args,['r',{version:0,channel:'DAPI',min:null,max:10}]);
});
