import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import { STAIN_TEXT, PICKER_TEXT, CELLPOSE_DEFAULTS, StainView, stainMethods, stainStatus, stainValues, validStain, stainProblems, largestPatch, boxInRegion, stainSizes, stainProgress, stainPreviewImage, resolutionChoices, resolutionReadout, matchResolution, resolutionRegionProblem, levelName, resolutionName, resolutionTitle, modelLabel, stainCopy } from '../src/stain.mjs';
import * as stainModule from '../src/stain.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { boxInPolygon } from '../src/region.mjs';
import { saveFields, saveModalState, saveable } from '../src/zarr-save-state.mjs';
import { deferred, domFixture } from './dom-fixture.mjs';
import { STAIN_PATCH_TEXT } from '../src/stain-patches.mjs';
import { encodeVectors } from './vectors-fixture.mjs';

const pyramid={key:'im',channels:['DAPI'],um_per_px:.2125,scales:[0,1,2,3].map(level=>
  ({level,factor:2**level,um_per_px:.2125*2**level,um_per_px_y:.2125*2**level}))};
test('resolution offers only stored copies by their sdata names and keeps matching physical pixel sizes across images',()=>{
  assert.deepEqual(resolutionChoices(pyramid).map(s=>s.label),['scale0','scale1','scale2','scale3']);
  assert.deepEqual(resolutionChoices({...pyramid,scales:[pyramid.scales[0],pyramid.scales[2]]}).map(s=>s.label),['scale0','scale2']);
  assert.deepEqual(resolutionChoices({um_per_px:1}).map(s=>[s.level,s.label]),[[0,'scale0']]);
  assert.deepEqual(resolutionChoices(pyramid).slice(0,2).map(s=>s.title),['scale0: 0.212 µm per pixel, the image as stored',
    'scale1: 0.425 µm per pixel, 2x coarser than scale0']);
  assert.equal(matchResolution(pyramid,{um_per_px:.44}),1);assert.equal(matchResolution(pyramid,{um_per_px:.5}),0);
  assert.equal(largestPatch(100,100,pyramid,2),117);
  assert.match(resolutionRegionProblem(pyramid,3,[0,0,20,100]).message,/11 pixels across at scale3$/);
  assert.equal(resolutionRegionProblem(pyramid,1,[0,0,20,100]),null);
});
test('every level a five-level pyramid has is offered and checked, scale4 included',()=>{
  const xenium={...pyramid,scales:[0,1,2,3,4].map(level=>({level,factor:2**level,um_per_px:.2125*2**level,
    um_per_px_y:.2125*2**level,label:levelName(level)}))};
  assert.deepEqual(resolutionChoices(xenium).map(s=>s.label),['scale0','scale1','scale2','scale3','scale4']);
  assert.equal(resolutionChoices(xenium)[4].title,'scale4: 3.4 µm per pixel, 16x coarser than scale0');
  const value={...stainValues(id=>({'stain-image':'im','stain-nucleus':'DAPI','stain-auto':true,'stain-flow':'.4','stain-probability':'0'}[id])),level:4};
  assert.ok(validStain(value,[xenium]));assert.deepEqual(stainProblems(value,[xenium]),[]);
  assert.equal(validStain({...value,level:5},[xenium]),false);
  assert.match(resolutionReadout(xenium,{level:4,diameter:30}).warning,/too small/);
  assert.equal(resolutionReadout(xenium,{level:4}).label,'scale4');
  assert.match(resolutionRegionProblem(xenium,4,[0,0,100,100]).message,/29 pixels across at scale4$/);
});
test('a pyramid of other factors still lists its levels by name (3x and 9x: scale1, scale2)',()=>{
  const odd={key:'im',channels:['DAPI'],um_per_px:.3,scales:[1,3,9].map((factor,level)=>({level,factor,um_per_px:.3*factor}))};
  assert.deepEqual(resolutionChoices(odd).map(s=>[s.label,s.factor]),[['scale0',1],['scale1',3],['scale2',9]]);
  assert.deepEqual(resolutionChoices(odd).map(s=>s.title),['scale0: 0.3 µm per pixel, the image as stored',
    'scale1: 0.9 µm per pixel, 3x coarser than scale0','scale2: 2.7 µm per pixel, 9x coarser than scale0']);
});
test('records an earlier version stored are named by their level, else their factor; never Full or 1/2',()=>{
  assert.equal(resolutionName({level:1,factor:2,um_per_px:.425,label:'1/2'}),'scale1');
  assert.equal(resolutionName({level:0,factor:1,label:'Full'}),'scale0');
  assert.equal(resolutionName({level:2,factor:3}),'scale2','the level names it, not the factor');
  assert.equal(resolutionName({factor:4,um_per_px:.85}),'scale2','only a factor: its power of two');
  assert.equal(resolutionName({factor:2.0004}),'scale1');assert.equal(resolutionName({factor:1}),'scale0');
  assert.equal(resolutionName({label:'1/8'}),'scale3');assert.equal(resolutionName({label:'Full'}),'scale0');
  for(const record of [null,undefined,{},{level:-1,factor:0,label:'half'},{level:true}])assert.equal(resolutionName(record),'');
  // A model trained on patches cut at a smaller copy says which, by name; patches cut at scale0 say nothing more.
  assert.equal(modelLabel({name:'mine',date:'2025-01-02',resolution:{level:1,factor:2,label:'1/2'}}),'mine · 2025-01-02 · learned at scale1');
  assert.equal(modelLabel({name:'mine',date:null,resolution:{factor:4}}),'mine · learned at scale2');
  assert.equal(modelLabel({name:'mine',date:'2025-01-02',resolution:{level:0,factor:1}}),'mine · 2025-01-02');
  assert.equal(resolutionTitle({level:2,factor:4.004,um_per_px:.851}),'scale2: 0.851 µm per pixel, 4x coarser than scale0');
  for(const text of Object.values(STAIN_TEXT))assert.doesNotMatch(String(text),/\bFull\b|\b1\/[248]\b/);
});
test('resolution readout uses measured or prior sizes, warns below 10 px, and omits micrometres in pixel frames',()=>{
  assert.equal(resolutionReadout(pyramid,{level:1},{cellPxMedian:16.7}).text,'0.425 µm per pixel · cells about 17 px across');
  assert.match(resolutionReadout(pyramid,{level:1,diameter:19.8}).warning,/too small/);
  assert.equal(resolutionReadout(pyramid,{level:1,diameter:20}).warning,'');
  assert.equal(resolutionReadout(pyramid,{level:1},{pixelFrame:true,cellPxMedian:17}).text,'cells about 17 px across');
  const sample={suggested_prior:{nuclei:{kind:'shapes',key:'saved_result'}},shapes:[{key:'saved_result',median_area_um2:Math.PI*(7.225/2)**2}]};
  assert.equal(resolutionReadout(pyramid,{level:1},{sample}).text,'0.425 µm per pixel · cells about 17 px across');
  assert.match(resolutionReadout(pyramid,{level:2,method:'instanseg'}).warning,/enlarged back/);
  // The menu's tooltip: what the scales are, and one resolution for the training patches and the whole sample.
  for(const words of ['scale0 is the image as stored','a smaller copy the sample already holds','less detail',
    'same resolution for training patches and for segmenting the whole sample','train on scale1 patches, segment at scale1 too'])
    assert.ok(STAIN_TEXT.resolutionHelp.includes(words),words);
  assert.match(STAIN_TEXT.resolutionSingle,/scale0 only/);assert.match(STAIN_TEXT.diameterHelp,/^In pixels of scale0 \(the image as stored\)/);
});
test('the readout sizes what the run outlines: InstanSeg what Find asks for, Cellpose nuclei exactly when it has no cell body channel',()=>{
  const sample={shapes:[{key:'cell_boundaries',median_area_um2:200},{key:'nucleus_boundaries',median_area_um2:40}],
    suggested_prior:{cells:{kind:'shapes',key:'cell_boundaries'},nuclei:{kind:'shapes',key:'nucleus_boundaries'}}};
  const at=settings=>resolutionReadout(pyramid,{level:2,...settings},{sample}).text;
  assert.equal(at({method:'cellpose',cytoplasm:'Boundary',output:'nuclei'}),'0.85 µm per pixel · cells about 19 px across','Find is not Cellpose\'s');
  assert.equal(at({method:'cellpose',cytoplasm:'',output:'cells'}),'0.85 µm per pixel · cells about 8 px across');
  assert.equal(at({method:'instanseg',cytoplasm:'Boundary',output:'nuclei'}),'0.85 µm per pixel · cells about 8 px across');
  assert.equal(at({method:'instanseg',output:'cells'}),'0.85 µm per pixel · cells about 19 px across');
});
test('Cellpose settings are labelled by their Cellpose names; the label\'s and the field\'s tooltip keep the plain words and the help',()=>{
  const dom=domFixture();dom.document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const nodes=dom.document.body.all().filter(n=>n.attributes?.has('data-stain-copy'));
  for(const node of nodes)node.dataset.stainCopy=node.getAttribute('data-stain-copy');   // as the browser's dataset has it
  stainCopy(nodes);
  const native={diameter:'diameter',flow:'flow_threshold',probability:'cellprob_threshold',niter:'niter',minSize:'min_size',
    augment:'augment',normalize:'normalize',fixedRange:'lowhigh',tileNorm:'tile_norm_blocksize',sharpen:'sharpen_radius',
    smooth:'smooth_radius',invert:'invert'};
  // The words each one's label said before ("Shape check (flow_threshold)"): its tooltip now.
  const plain={diameter:'Cell diameter',flow:'Shape check',probability:'Cell threshold',niter:'Steps',minSize:'Smallest cell',
    augment:'Try flipped images',normalize:'Adjust brightness',fixedRange:'Fixed intensity range',tileNorm:'Local brightness size',
    sharpen:'Sharpening',smooth:'Smoothing',invert:'Reverse brightness'};
  const fields={diameter:'stain-diameter',flow:'stain-flow',probability:'stain-probability',niter:'stain-niter',
    minSize:'stain-min-size',augment:'stain-augment',normalize:'stain-normalize',fixedRange:'stain-fixed-range',
    tileNorm:'stain-tile-norm',sharpen:'stain-sharpen',smooth:'stain-smooth',invert:'stain-invert'};
  const byKey=key=>nodes.find(n=>n.dataset.stainCopy===key);
  for(const [key,name] of Object.entries(native)) {
    const label=byKey(key).closest('label'),help=STAIN_TEXT[`${key}Help`];
    assert.equal(byKey(key).textContent,name,key);
    assert.ok(label.contains(dom.$(fields[key])),`${key}: one tooltip for its name and its field`);
    assert.ok(label.title.startsWith(plain[key]),`${key}: ${label.title}`);
    if(help)assert.ok(label.title.endsWith(help),`${key} keeps its help`);
    assert.equal(dom.$(fields[key]).title,label.title,`${key}: the field's own tooltip too, which a screen reader says`);
    assert.deepEqual(bannedWords(label.title),[],label.title);
  }
  assert.equal(byKey('flow').closest('label').title,'Shape check. Lower values reject more uncertain shapes.');
  assert.equal(byKey('probability').closest('label').title,'Cell threshold. Lower values find more and larger cells.');
  assert.equal(byKey('diameter').closest('label').title,
    'Cell diameter. In pixels of scale0 (the image as stored). Cellpose gets it at the chosen resolution.');
  assert.equal(byKey('minSize').closest('label').title,'Smallest cell');
  // The help line stays under its field; controls without a Cellpose name keep their words and get no such tooltip.
  assert.equal(byKey('niterHelp').textContent,STAIN_TEXT.niterHelp);
  for(const key of ['nucleus','cytoplasm','tissue','auto','low','high','intensityLow','intensityHigh']) {
    assert.equal(byKey(key).textContent,STAIN_TEXT[key]);assert.equal(byKey(key).closest('label').title ?? '','',key);
    for(const field of byKey(key).closest('label').children)assert.equal(field.title ?? '','',key);
  }
  for(const node of nodes)assert.doesNotMatch(node.textContent,/Shape check|Cell threshold|\([a-z]+(_[a-z]+)+\)|\(normalisation\)/);
  // A problem names the field as it is labelled.
  const value=stainValues(id=>({'stain-image':'im','stain-nucleus':'DAPI','stain-auto':true,'stain-flow':'11','stain-probability':'0'}[id]));
  assert.deepEqual(stainProblems(value,[pyramid]).map(p=>p.label),['flow_threshold']);
});
test('both methods carry level, and unavailable resolutions have a named field problem',()=>{
  for(const method of ['cellpose','instanseg'])assert.equal(stainValues(id=>({'stain-method':method,'stain-level':'2'}[id])).level,2);
  const value={...stainValues(id=>({'stain-image':'im','stain-nucleus':'DAPI','stain-auto':true,'stain-flow':'.4','stain-probability':'0'}[id])),level:1};
  assert.ok(validStain(value,[pyramid]));
  assert.equal(validStain({...value,level:4},[pyramid]),false);
  assert.deepEqual(stainProblems({...value,level:4},[pyramid]).map(p=>p.field),['stain-level']);
});

test('Adjust brightness off sends no fixed range: one left ticked, empty or out of order never blocks Start', () => {
  const images=[{key:'im',channels:['DAPI']}];
  const values=form=>stainValues(id=>({'stain-image':'im','stain-nucleus':'DAPI',...CELLPOSE_DEFAULTS,'stain-fixed-range':true,...form})[id]);
  // Off: Fixed range and its fields are disabled (stain-settings.mjs updateNormalizeFields) but keep what was typed.
  for(const [low,high] of [['',''],['10','5'],['x','']]) {
    const value=values({'stain-normalize':false,'stain-intensity-low':low,'stain-intensity-high':high});
    assert.equal(value.normalize.lowhigh,null,`${low}, ${high}`);assert.equal(value.normalize.percentile,null);
    assert.deepEqual(stainProblems(value,images,null),[]);assert.ok(validStain(value,images,null));
  }
  // On again: the range kept there counts once more, and what it needs is named on fields that can be edited.
  assert.deepEqual(stainProblems(values({'stain-intensity-low':'','stain-intensity-high':''}),images,null).map(p=>p.field),
    ['stain-intensity-low','stain-intensity-high']);
  assert.deepEqual(values({'stain-intensity-low':'0','stain-intensity-high':'4095'}).normalize.lowhigh,[0,4095]);
});

// main.js's own top-level functions, found by name (their order there does not matter): from `function name(` to the
// first closing brace where the text parses on its own, so JavaScript itself reads past strings, template literals,
// regular expressions and comments (no brace in one of those can end the function early, and none runs off the end).
const mainSource=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
function mainFunction(name) {
  const at=mainSource.search(new RegExp(`^(?:async )?function ${name}\\(`,'m'));
  assert.ok(at>=0,`main.js has function ${name}`);
  for(let i=mainSource.indexOf('}',at);i>=0;i=mainSource.indexOf('}',i+1)) {
    const text=mainSource.slice(at,i+1);
    try {new vm.Script(text);return text;} catch {/* not closed yet */}
  }
  assert.fail(`main.js function ${name} has no end`);
}
/** The picker's stain fields as main.js reads and fills them, against the page's own markup. */
function pickerPage(sample,health=undefined) {
  const dom=domFixture();dom.document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const context=vm.createContext({$:dom.$,...stainModule,STAIN_PATCH_TEXT,state:{sample,health},
    Option:class {constructor(text,value){const option=dom.document.createElement('option');option.textContent=text;option.value=value;return option;}}});
  vm.runInContext(['readStain','stainAvailable','stainModelKey','offerStainMethods','updateStainMethod','fillResolution','renderResolution']
    .map(mainFunction).join('\n'),context);
  return {dom,$:dom.$,context,run:code=>vm.runInContext(code,context)};
}

test('the Resolution control sits under Image, offers the image\'s own copies and goes with Start (main.js)', () => {
  const sample={images:[{...pyramid,scales:resolutionChoices(pyramid)},{key:'twin',channels:['DAPI'],um_per_px:.22,
    scales:[{level:0,factor:1,um_per_px:.22,label:'Full'},{level:1,factor:2,um_per_px:.44,label:'1/2'}]},
    {key:'flat',channels:['DAPI'],um_per_px:.3}]};
  const {$,run}=pickerPage(sample);
  const image=$('stain-image').parentElement,level=$('stain-level').parentElement,rows=image.parentElement.children;
  assert.equal(rows[rows.indexOf(image)+1],level,'the label right under Image');
  $('stain-image').value='im';$('stain-method').value='cellpose';$('stain-auto').checked=true;run('fillResolution()');
  assert.deepEqual($('stain-level').children.map(o=>[o.textContent,o.value]),[['scale0','0'],['scale1','1'],['scale2','2'],['scale3','3']]);
  assert.equal($('stain-level').children[1].title,'scale1: 0.425 µm per pixel, 2x coarser than scale0');
  assert.equal($('stain-level').title,STAIN_TEXT.resolutionHelp);
  $('stain-level').value='1';run('renderResolution()');
  assert.equal($('stain-resolution-info').textContent,'0.425 µm per pixel');
  assert.equal(run('readStain()').level,1,'Start sends the level chosen');
  // A diameter of 30 scale0 pixels is under 10 at scale3: the amber warning, with its reason in the tooltip.
  $('stain-auto').checked=false;$('stain-diameter').value='30';$('stain-level').value='3';run('renderResolution()');
  assert.match($('stain-resolution-info').textContent,/^1\.7 µm per pixel · cells about 4 px across$/);
  assert.equal($('stain-resolution-warning').hidden,false);assert.match($('stain-resolution-warning').textContent,/too small to outline well/);
  for(const words of [$('stain-resolution-info').textContent,$('stain-resolution-warning').textContent,
    $('stain-resolution-warning').title,$('stain-level').title])assert.deepEqual(bannedWords(words),[],words);
  // Another image keeps a copy within 5 % of the same µm per pixel (its labels an earlier engine's: named by level here);
  // one with no such copy goes back to scale0.
  $('stain-level').value='1';run('renderResolution()');$('stain-image').value='twin';run('fillResolution()');
  assert.equal($('stain-level').value,'1');assert.deepEqual($('stain-level').children.map(o=>o.textContent),['scale0','scale1']);
  $('stain-image').value='flat';run('fillResolution()');
  assert.equal($('stain-level').value,'0');assert.deepEqual($('stain-level').children.map(o=>o.textContent),['scale0']);
  assert.equal($('stain-level').title,STAIN_TEXT.resolutionSingle);
  assert.equal($('stain-level').children[0].title,'scale0: 0.3 µm per pixel, the image as stored');
});

test('the picker sizes what the run outlines: nuclei without a cell body channel, cells with one (main.js)', () => {
  // A sample with both boundaries: nuclei 6.8 µm across, cells 16 µm. At scale2 (0.85 µm per pixel) on Auto, a DAPI-only
  // Cellpose run outlines nuclei, 8 px across (warned); with a cell body channel it outlines cells, 19 px (no warning).
  const sample={images:[{...pyramid,scales:resolutionChoices(pyramid)}],
    suggested_prior:{cells:{kind:'shapes',key:'cell_boundaries'},nuclei:{kind:'shapes',key:'nucleus_boundaries'}},
    shapes:[{key:'cell_boundaries',median_area_um2:Math.PI*8**2},{key:'nucleus_boundaries',median_area_um2:Math.PI*3.4**2}]};
  const {$,run}=pickerPage(sample);
  $('stain-image').value='im';$('stain-method').value='cellpose';$('stain-auto').checked=true;run('fillResolution()');
  $('stain-cytoplasm').value='';$('stain-level').value='2';run('renderResolution()');
  assert.equal($('stain-resolution-info').textContent,'0.85 µm per pixel · cells about 8 px across');
  assert.equal($('stain-resolution-warning').hidden,false);assert.match($('stain-resolution-warning').textContent,/too small to outline well/);
  $('stain-cytoplasm').value='Boundary';run('renderResolution()');
  assert.equal($('stain-resolution-info').textContent,'0.85 µm per pixel · cells about 19 px across');
  assert.equal($('stain-resolution-warning').hidden,true);
});

test('direct patches share image controls and models; Empty needs only channels', () => {
  // InstanSeg switched on (the engine's internal switch: health lists it), as its patches below need.
  const {dom,context}=pickerPage({images:[{key:'im',channels:['DAPI','Boundary'],um_per_px:1}]},
    {stain:{available:true,methods:{cellpose:true,instanseg:true}}});
  vm.runInContext('offerStainMethods()',context);
  assert.deepEqual(dom.$('stain-method').children.map(o=>[o.value,o.textContent]),[['cellpose','Cellpose'],['instanseg','InstanSeg']]);
  dom.$('stain-channels').selectedOptions=[];
  for(const [id,value] of Object.entries({'run-kind':'patches','stain-image':'im','stain-nucleus':'DAPI',
    'patch-count':'8','patch-size':'256','stain-cytoplasm':'Boundary','stain-model':'trained','stain-method':'cellpose','patch-start-from':'model'}))dom.$(id).value=value;
  const read=()=>JSON.parse(JSON.stringify(vm.runInContext('readStain()',context)));
  let value=read();
  assert.equal(value.model,'trained');assert.equal(value.count,8);assert.equal(value.size,256);
  assert.equal(value.start_from,'model');assert.equal(value.diameter,null);
  dom.$('stain-auto').checked=false;dom.$('stain-diameter').value='23';dom.$('stain-flow').value='.55';
  dom.$('stain-probability').value='-1.5';dom.$('stain-niter').value='25';dom.$('stain-low').value='2';
  value=read();assert.equal(value.diameter,23);assert.equal(value.flow_threshold,.55);
  assert.equal(value.cellprob_threshold,-1.5);assert.equal(value.niter,25);assert.equal(value.normalize.percentile[0],2);
  dom.$('stain-fixed-range').checked=true;dom.$('stain-intensity-low').value='0';dom.$('stain-intensity-high').value='4095';
  value=read();assert.deepEqual(value.normalize.lowhigh,[0,4095]);assert.equal(value.normalize.percentile,null);
  // Adjust brightness off: the range ticked there stays as typed but is not sent (it would block Start otherwise).
  dom.$('stain-normalize').checked=false;assert.equal(read().normalize.normalize,false);assert.equal(read().normalize.lowhigh,null);
  vm.runInContext('updateStainMethod()',context);
  assert.equal(dom.$('stain-tuning').hidden,false);assert.equal(dom.$('stain-cellpose-options').hidden,false);
  assert.equal(dom.$('stain-method').parentElement.hidden,false,'a choice of two methods: the Method row shows');
  dom.$('stain-method').value='instanseg';dom.$('stain-model').value='fluorescence_nuclei_and_cells';
  vm.runInContext('updateStainMethod()',context);
  value=read();assert.deepEqual(value.channels,['DAPI','Boundary']);
  assert.equal(dom.$('stain-cellpose-options').hidden,false);
  dom.$('stain-model').value='brightfield_nuclei';dom.$('stain-channels').selectedOptions=['red','green','blue'].map(value=>({value}));
  vm.runInContext('updateStainMethod()',context);
  assert.equal(dom.$('stain-instanseg-options').hidden,false);assert.deepEqual(read().channels,['red','green','blue']);
  dom.$('patch-start-from').value='empty';
  vm.runInContext('updateStainMethod()',context);
  value=read();assert.equal(value.method,'cellpose');assert.equal(value.start_from,'empty');
  assert.equal(dom.$('stain-model').parentElement.hidden,true);
  assert.ok(validStain(value,[{key:'im',channels:['DAPI','Boundary']}],null));
  // Empty hides the Cellpose tuning: what was typed there is not read; Cellpose's defaults go instead.
  assert.equal(dom.$('stain-tuning').hidden,true);
  dom.$('stain-diameter').value='5';dom.$('stain-niter').value='0';
  value=read();
  assert.deepEqual([value.diameter,value.flow_threshold,value.cellprob_threshold,value.niter,value.min_size],[null,.4,0,null,15]);
  assert.deepEqual(value.normalize,{normalize:true,lowhigh:null,percentile:[1,99],tile_norm_blocksize:0,sharpen_radius:0,smooth_radius:0,invert:false});
  assert.deepEqual(stainProblems(value,[{key:'im',channels:['DAPI','Boundary']}],null),[]);
  dom.$('patch-start-from').value='model';dom.$('stain-method').value='cellpose';
  value=read();assert.equal(value.diameter,5,'Model outlines read the fields again');assert.equal(value.niter,0);
});

test("Cellpose's defaults are the picker's tuning fields as the page opens, for every one of them, and Cellpose takes them", () => {
  const dom=domFixture();dom.document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  const fields=dom.$('stain-tuning').all().filter(n=>['INPUT','SELECT'].includes(n.tagName));
  assert.deepEqual(fields.map(n=>n.id).sort(),Object.keys(CELLPOSE_DEFAULTS).sort(),'every hidden tuning field has its default');
  for(const field of fields)assert.equal(field.type==='checkbox'?field.checked:field.value,CELLPOSE_DEFAULTS[field.id],field.id);
  const form={'stain-image':'im','stain-nucleus':'DAPI','stain-cytoplasm':'',...CELLPOSE_DEFAULTS};
  const value=stainValues(id=>form[id]);
  assert.deepEqual(stainProblems(value,[{key:'im',channels:['DAPI']}],null),[]);assert.ok(validStain(value,[{key:'im',channels:['DAPI']}],null));
  assert.ok(Object.isFrozen(CELLPOSE_DEFAULTS));
});

test('boxInRegion: a box inside the region\'s image window and, for a lasso, inside the lasso up to a pixel from its edge', () => {
  const extent=[600,400];
  for(const box of [[0,0,64,64],[536,336,600,400],[100,100,356,356]])assert.equal(boxInRegion(box,{extent}),true,String(box));
  for(const box of [[-1,0,63,64],[0,-1,64,63],[537,0,601,64],[0,337,64,401],[700,0,764,64],[10,10,10,74],[20,10,10,74]])
    assert.equal(boxInRegion(box,{extent}),false,String(box));
  assert.equal(boxInRegion([-500,-500,-400,-400],{}),true,'no frame known: the engine checks');
  assert.equal(boxInRegion([5,5,1,1],{}),false,'never an empty box');
  // A right triangle (the lasso's corner at the origin): inside along its two straight sides, outside past the long one.
  const outline=[[0,0],[600,0],[0,400]],frame={extent,outline};
  assert.equal(boxInRegion([0,0,64,64],frame),true);
  // Past the long side (2x + 3y = 1200; a corner (x, y) is (2x + 3y − 1200)/√13 pixels past it): a corner 0.83 pixels past
  // is allowed, 1.11 pixels is not. The distance to the edge counts, not a pixel off each side of the box (√2 along this
  // edge), so the page and the engine agree on it.
  assert.equal(boxInRegion([236,137,300,201],frame),true,'0.83 pixels past');
  assert.equal(boxInRegion([238,136,302,200],frame),false,'1.11 pixels past');
  assert.equal(boxInRegion([-1,-1,63,63],{outline}),false,'past the lasso\'s corner: 1.41 pixels from it');
  assert.equal(boxInRegion([-3,0,61,64],{outline}),false,'3 pixels past a straight side');
  assert.equal(boxInRegion([300,200,364,264],frame),false,'past the long side');
  assert.equal(boxInRegion([200,100,264,164],frame),true);
  // The same rule as the panel's gestures keep (region.mjs boxInPolygon), over many boxes along the long side.
  for(let x=200;x<=320;x+=7)for(let y=100;y<=230;y+=5)
    assert.equal(boxInRegion([x,y,x+64,y+64],frame),boxInPolygon([x,y,x+64,y+64],outline,1),`${x},${y}`);
  // The engine's own boxes are not checked again (it accepted them: its pixel of slack is mitred at the lasso's corners);
  // a box moved from one is.
  const suggested=[[-1,-1,63,63],[238,136,302,200]];
  assert.equal(boxInRegion([-1,-1,63,63],{outline,suggested}),true);
  assert.equal(boxInRegion([238,136,302,200],{...frame,suggested}),true);
  assert.equal(boxInRegion([238,137,302,201],{...frame,suggested}),false,'moved a pixel down');
  assert.equal(boxInRegion([700,0,764,64],{extent,suggested}),false);
  // A U-shaped lasso: a box whose corners are all inside the arms, across the gap between them, is not inside.
  const u=[[0,0],[100,0],[100,300],[200,300],[200,0],[300,0],[300,400],[0,400]];
  assert.equal(boxInRegion([20,100,280,200],{outline:u}),false);
  assert.equal(boxInRegion([20,320,280,390],{outline:u}),true);
  assert.equal(boxInRegion([120,100,180,200],{outline:u}),false,'in the gap: outside');
  // A lasso with a corner poking into the box: not inside.
  const notch=[[0,0],[400,0],[400,400],[200,150],[0,400]];
  assert.equal(boxInRegion([150,100,250,200],{outline:notch}),false);
  assert.equal(boxInRegion([150,20,250,100],{outline:notch}),true);
  assert.equal(boxInRegion([10,10,74,74],{outline:[[0,0],[1,1]]}),true,'not a lasso (two points): only the window counts');
});

test('direct patches show the image without requesting vectors or offering a sample save', () => {
  let reads=0;
  const run={id:'patches',kind:'patches',sample_kind:'zarr',state:'done',box:[0,0,100,100]};
  const view=new StainView({}, {}, {imageView:()=>reads++},()=>assert.fail('outline request'));
  view.reset(run);view.visible=()=>assert.fail('outline view');view.refresh();
  assert.equal(reads,1);assert.equal(saveable(run),false);
});

test('tiny cell diameters are rejected in the form as well as the engine', () => {
  const value={image:'im',nucleus:'DAPI',cytoplasm:null,diameter:30,flow_threshold:.4,cellprob_threshold:0};
  const images=[{key:'im',channels:['DAPI']}];
  for(const diameter of [1,5,9.9]) assert.equal(validStain({...value,diameter},images),false);
  for(const diameter of [null,10,700,1000]) assert.equal(validStain({...value,diameter},images),true);
  const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  assert.match(html,/id="stain-diameter"[^>]*min="10"/);
});

test('completed resolved shapes remain saveable after a bookkeeping failure', () => {
  const run={kind:'stain',sample_kind:'zarr',state:'failed',progress:{resolved:true,complete:true},
    save_defaults:{shapes_key:'cellpose_boundaries_1',table_key:null,existing:[]}};
  assert.ok(saveable(run));
  assert.ok(saveModalState(run,saveFields(run)).canSave);
  assert.equal(saveable({...run,progress:{resolved:false,complete:true}}),false);
  assert.equal(saveable({...run,progress:{resolved:true,complete:false}}),false);
  assert.equal(saveable({...run,state:'running'}),false);
});

test('every stain label, hint and status uses the plain-words vocabulary', () => {
  for (const [key, value] of Object.entries(STAIN_TEXT)) assert.deepEqual(bannedWords(value), [], key);
  // What the picker's Start asks for (main.js pickerProblems), filled in as it is shown.
  for (const [key, value] of Object.entries(PICKER_TEXT)) assert.deepEqual(bannedWords(value.replace('{largest}', '1,320')), [], key);
  for (const file of ['engine/stain.py','engine/stain_patches.py','engine/stain_worker.py',
    'zarr_bridge/stain.py','zarr_bridge/stain_patches.py','zarr_bridge/stain_gui.py','zarr_bridge/stain_instanseg.py'])
    for (const match of readFileSync(new URL(`../../${file}`,import.meta.url),'utf8').matchAll(/(?:RunError|ValueError|RuntimeError)\("([^"]+)"/g))
      assert.deepEqual(bannedWords(match[1]), [], match[1]);
  for (const state of ['running','paused','done','failed','stopped','finishing','preparing'])
    assert.deepEqual(bannedWords(stainStatus({state})), []);
  assert.deepEqual(bannedWords(stainSizes({summary:{area_um2:{median:100,p10:50,p90:200}}})), []);
});

test('the picker offers the engine\'s methods: Cellpose alone while InstanSeg is dormant, with no Method row', () => {
  // Health lists InstanSeg only while the engine's internal switch is on; an engine that lists no methods offers Cellpose.
  for (const health of [undefined, {}, {stain:{available:true}}, {stain:{available:true,methods:{cellpose:true}}},
    {stain:{methods:null}}, {stain:{methods:'instanseg'}}])
    assert.deepEqual(stainMethods(health), ['cellpose'], JSON.stringify(health));
  assert.deepEqual(stainMethods({stain:{methods:{cellpose:false,instanseg:false}}}), ['cellpose','instanseg'], 'listed, even unavailable');
  const images=[{key:'im',channels:['DAPI','Boundary'],um_per_px:1}];
  const off=pickerPage({images},{stain:{available:true,methods:{cellpose:true}}});
  off.$('stain-method').value='instanseg';                 // a stale choice (an earlier page's): back to Cellpose
  off.run('offerStainMethods();updateStainMethod()');
  assert.deepEqual(off.$('stain-method').children.map(o=>o.value),['cellpose']);
  assert.equal(off.$('stain-method').value,'cellpose');
  assert.equal(off.$('stain-method').parentElement.hidden,true,'one method: no Method row');
  assert.equal(off.$('stain-model').parentElement.hidden,false);
  assert.equal(off.$('stain-instanseg-options').hidden,true,'no Channels, Find or Match cells and nuclei row');
  assert.equal(off.run('readStain()').method,'cellpose');
  const on=pickerPage({images},{stain:{available:true,methods:{cellpose:true,instanseg:true}}});
  on.run('offerStainMethods();updateStainMethod()');
  assert.deepEqual(on.$('stain-method').children.map(o=>[o.value,o.textContent]),[['cellpose','Cellpose'],['instanseg','InstanSeg']]);
  assert.equal(on.$('stain-method').parentElement.hidden,false,'switched on: the choice is shown');
  const first=on.$('stain-method').children[0];
  on.run('offerStainMethods()');                            // offered again: the same options, never rebuilt
  assert.equal(on.$('stain-method').children[0],first);
});

test('picker offers Cellpose with named channels and native-size Cellpose auto', () => {
  const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
  // The markup's Method choice holds Cellpose alone; another method comes only from the engine (offerStainMethods).
  assert.match(html, /<select id="stain-method">\s*<option value="cellpose" data-stain-copy="cellpose"><\/option>\s*<\/select>/);
  assert.doesNotMatch(html.replace(/\bid="stain-instanseg-[a-z]+"/g, ''), /instanseg/i, 'no word of the dormant InstanSeg in the markup');
  for (const id of ['run-kind','stain-model','stain-image','stain-nucleus','stain-cytoplasm','stain-auto','stain-diameter','stain-flow','stain-probability'])
    assert.ok(html.includes(`id="${id}"`), id);
  const form = {'stain-image':'im','stain-nucleus':'DAPI','stain-cytoplasm':'','stain-auto':true,'stain-diameter':'30','stain-flow':'.4','stain-probability':'0'};
  const value = stainValues(id=>form[id]), images = [{key:'im',channels:['DAPI','Boundary']}];
  assert.equal(value.diameter, null); assert.equal(value.cytoplasm,null); assert.ok(validStain(value,images));
  assert.equal(value.model, 'cpsam');
  form['stain-model'] = 'trained';
  assert.equal(stainValues(id=>form[id]).model, 'trained');
  assert.ok(validStain(stainValues(id=>form[id]),images,[{name:'trained'}]));
  assert.equal(validStain(stainValues(id=>form[id]),images,[{name:'cpsam'}]),false);
  assert.equal(validStain(value,images,[]),false);
  for (const change of [{nucleus:'absent'}, {cytoplasm:'DAPI'}, {diameter:0}, {diameter:NaN}, {flow_threshold:Infinity}, {cellprob_threshold:21}])
    assert.equal(validStain({...value,...change},images),false);
});

test('progress and pause text distinguish a request from the helper acknowledgement and a partial stop', () => {
  assert.equal(stainProgress({progress:{done:2,total:7}}),'2 / 7');
  assert.equal(stainStatus({state:'running',paused:true}), STAIN_TEXT.pausing);
  assert.equal(stainStatus({state:'paused',paused:true,ack:{state:'paused'}}), STAIN_TEXT.paused);
  assert.equal(stainStatus({state:'paused',paused:false,ack:{state:'paused'}}), STAIN_TEXT.resuming);
  assert.equal(stainStatus({state:'stopped'}), STAIN_TEXT.partial);
  const downloading = stainStatus({state:'running',message:'Downloading the model…'});
  assert.equal(downloading, 'Downloading the model…'); assert.deepEqual(bannedWords(downloading),[]);
  assert.equal(stainStatus({state:'failed',message:'Could not download the model.'}),STAIN_TEXT.failed);
  assert.match(stainSizes({summary:{area_um2:{median:10,p10:5,p90:15}}}), /10.0 µm²/);
});

test('a new stain run waits for Start before loading a model', () => {
  assert.equal(stainStatus({state:'paused',paused:true,stain_started:false,ack:{state:'paused'}}),STAIN_TEXT.waiting);
  assert.match(STAIN_TEXT.waiting,/Press Start to begin segmenting/);
  assert.equal(STAIN_TEXT.start,'Start');
});

test('InstanSeg validates channel counts and nuclei-only outputs without Cellpose controls', () => {
  const images=[{key:'im',channels:['a','b','c','d']}];
  const value={method:'instanseg',model:'fluorescence_nuclei_and_cells',image:'im',
    channels:['a','b','c','d'],output:'cells',resolve_cells:false};
  assert.ok(validStain(value,images));
  assert.ok(validStain({...value,output:'nuclei',resolve_cells:true},images));
  for(const change of [{channels:[]},{channels:['a','a']},{channels:['unknown']},{output:'both'},{resolve_cells:1},
    {model:'single_channel_nuclei'},{model:'brightfield_nuclei'}])
    assert.equal(validStain({...value,...change},images),false);
  assert.ok(validStain({...value,model:'single_channel_nuclei',channels:['a'],output:'nuclei'},images));
  assert.ok(validStain({...value,model:'brightfield_nuclei',channels:['a','b','c'],output:'nuclei'},images));
  assert.equal(stainStatus({state:'running',message:'Downloading model…'}),'Downloading model…');
});

test('the selected image preview keeps its registered bounds and channel intensities', () => {
  const meta={width:2,height:2,x0:10,y0:-20,x1:11,y1:-16}, buffer=Uint16Array.of(10,20,200,400).buffer;
  const view=stainPreviewImage(meta,buffer);
  assert.deepEqual(view.rect,{x0:10,y0:-20,w:1,h:4});
  assert.equal(view.channels[0].low,10);assert.equal(view.channels[0].high,400);
  assert.deepEqual([...view.channels[0].source.data],[10,20,200,400]);
  assert.throws(()=>stainPreviewImage(meta,new ArrayBuffer(2)));
});

test('a partial stain result saves only shapes and always requires an explicit overwrite choice', () => {
  const run = {kind:'stain',sample_kind:'zarr',state:'stopped',progress:{resolved:true},
    save_defaults:{shapes_key:'cellpose_boundaries_1',table_key:null,existing:[]}};
  const fields = saveFields(run);
  assert.ok(saveable(run)); assert.ok(saveModalState(run,fields).canSave);
  assert.equal(saveModalState({...run,state:'running'},fields).canSave,false);
  assert.equal(saveable({...run,progress:{resolved:false}}),false);
  run.save_defaults.existing.push('cellpose_boundaries_1');
  assert.equal(saveModalState(run,fields).canSave,false);
  assert.ok(saveModalState(run,{...fields,overwrite:true}).canSave);
  assert.equal(saveModalState(run,{...fields,shapes_key:'../bad',overwrite:true}).canSave,false);
});

test('outline requests are viewport bounded, cancel stale tiles and clear on a new run', async t => {
  const original = globalThis.fetch, jobs=[];
  globalThis.fetch=(url, options)=>{const d=deferred();jobs.push({url,options,...d});return d.promise;};
  t.after(()=>{globalThis.fetch=original;});
  const stage={canvas:{width:200,height:100},camera:{x:400,y:400,zoom:1}}, priors={};
  const tiled={imageView(){}}, errors=[], view=new StainView(stage,priors,tiled,e=>errors.push(e));
  const run={id:'stain',kind:'stain',box:[0,0,1000,1000],progress:{revision:1,done:1}};
  view.update(run); await Promise.resolve();
  assert.equal(jobs.length,1); assert.match(jobs[0].url,/box=260,330,540,470/);
  view.update({...run,progress:{revision:2,done:2}}); await Promise.resolve();
  assert.ok(jobs[0].options.signal.aborted); assert.equal(jobs.length,2);
  jobs[0].reject(Object.assign(Error(),{name:'AbortError'}));
  view.reset();
  assert.ok(jobs[1].options.signal.aborted);
  jobs[1].reject(Object.assign(Error(),{name:'AbortError'}));
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(errors,[]); assert.equal(priors.vectors,undefined);
});

for(const change of ['refresh','move','camera']) test(`late pan outlines cannot replace the returned viewport (${change})`,async t=>{
  const jobs=[],errors=[],stage={canvas:{width:100,height:100},camera:{x:100,y:100,zoom:1}},priors={};
  t.mock.method(globalThis,'fetch',(url,options)=>{const d=deferred();jobs.push({url,options,...d});return d.promise;});
  const view=new StainView(stage,priors,{imageView(){}},e=>errors.push(e));
  t.after(()=>view.reset());
  const settle=()=>new Promise(resolve=>setImmediate(resolve));
  const respond=(job,label,x)=>job.resolve({ok:true,arrayBuffer:async()=>encodeVectors(
    {cells:[{id:1,poly:[[x,90],[x+10,90],[x,100],[x,90]]}]},{revision:9,label})});
  view.update({id:'completed',kind:'stain',box:[0,0,1000,1000],progress:{revision:9}});
  await settle();respond(jobs[0],'A',95);await settle();
  assert.equal(priors.vectors.label,'A');
  stage.camera.x=800;view.refresh();await settle();
  const requestB=jobs[1];
  stage.camera.x=100;
  if(change!=='camera')view[change]();
  respond(requestB,'B',795);await settle();
  assert.equal(priors.vectors.label,'A');
  assert.deepEqual(view.shown.box,[30,30,170,170]);
  if(change!=='camera')assert.ok(requestB.options.signal.aborted);
  assert.deepEqual(errors,[]);
});

test('the outlines stay shown when what they are handed to (the range choosers) fails, and a new run still opens', async t => {
  const jobs=[],errors=[],logged=[],stage={canvas:{width:100,height:100},camera:{x:50,y:50,zoom:1}},priors={};
  t.mock.method(globalThis,'fetch',(url,options)=>{const d=deferred();jobs.push({url,options,...d});return d.promise;});
  t.mock.method(console,'error',error=>logged.push(error.message));
  const handed=[],view=new StainView(stage,priors,{imageView(){}},e=>errors.push(e),
    vectors=>{handed.push(vectors?.set ?? null);throw TypeError('setVectors is not a function');});
  const settle=()=>new Promise(resolve=>setImmediate(resolve));
  view.reset({id:'a'});   // never throws to the caller (main.js opening a run)
  view.update({id:'a',kind:'stain',box:[0,0,100,100],progress:{revision:3},exclusions:{areas_version:2}});await settle();
  jobs[0].resolve({ok:true,arrayBuffer:async()=>encodeVectors({cells:[{id:1,poly:[[1,1],[9,1],[1,9],[1,1]]}]},
    {revision:3,set:'s3',areas_version:2})});
  await settle();
  assert.equal(priors.vectors.set,'s3');assert.equal(priors.vectorsVisible,true);
  assert.deepEqual(view.shown,{revision:3,exclusions:2,areas_version:2,box:[0,0,100,100]});
  assert.equal(view.failed,null);assert.deepEqual(errors,[],'no "outlines could not be loaded" for a failure that is not theirs');
  assert.deepEqual(handed,[null,'s3']);assert.deepEqual(logged,['setVectors is not a function','setVectors is not a function']);
  view.refresh();await settle();assert.equal(jobs.length,1,'shown outlines are not fetched again');
  view.reset();assert.equal(view.shown,null);
});

test('outlines read while a running run wrote a tile (409) are read again shortly and unsaid; only a failure that stays is said', async t => {
  // A notebook's stain run once had two 409s while it segmented, each said as "could not be loaded".
  const jobs=[],errors=[],stage={canvas:{width:100,height:100},camera:{x:50,y:50,zoom:1}},priors={};
  t.mock.method(globalThis,'fetch',(url,options)=>{const d=deferred();jobs.push({url,options,...d});return d.promise;});
  t.mock.timers.enable({apis:['setTimeout']});
  const view=new StainView(stage,priors,{imageView(){}},e=>errors.push(e));
  const settle=()=>new Promise(resolve=>setImmediate(resolve));
  view.update({id:'a',kind:'stain',box:[0,0,100,100],progress:{revision:3},exclusions:{areas_version:0}});await settle();
  assert.equal(jobs.length,1);
  for(let n=1;n<=stainModule.VECTORS_RETRIES;n++) {
    jobs.at(-1).resolve({ok:false,status:409});await settle();await settle();
    assert.deepEqual(errors,[],`409 number ${n}: nothing said`);assert.ok(!view.failed);
    t.mock.timers.tick(stainModule.VECTORS_RETRY_MS);await settle();
    assert.equal(jobs.length,n+1,`read again after ${stainModule.VECTORS_RETRY_MS} ms`);
  }
  jobs.at(-1).resolve({ok:false,status:409});await settle();await settle();
  assert.deepEqual(errors,[STAIN_TEXT.vectorsError],'a conflict that does not settle is said, once');
  t.mock.timers.tick(10*stainModule.VECTORS_RETRY_MS);await settle();assert.equal(jobs.length,stainModule.VECTORS_RETRIES+1);
  // Another failure (not a conflict) is said at once.
  view.update({id:'a',kind:'stain',box:[0,0,100,100],progress:{revision:4},exclusions:{areas_version:0}});await settle();
  jobs.at(-1).resolve({ok:false,status:500});await settle();await settle();
  assert.equal(errors.length,2);
});

// ---- What Start needs, field by field (a press names the missing field instead of a greyed-out button) ---------------------
const cellposeValue={method:'cellpose',model:'cpsam',image:'im',nucleus:'DAPI',cytoplasm:null,diameter:null,flow_threshold:.4,
  cellprob_threshold:0,tissue_only:true,niter:null,min_size:15,augment:false,
  normalize:{normalize:true,lowhigh:null,percentile:[1,99],tile_norm_blocksize:0,sharpen_radius:0,smooth_radius:0,invert:false}};
const instansegValue={method:'instanseg',model:'fluorescence_nuclei_and_cells',image:'im',channels:['DAPI','Boundary'],
  output:'cells',resolve_cells:false,tissue_only:true};

test('stainProblems names the field of each setting validStain refuses, with the picker\'s label and what it needs', () => {
  const images=[{key:'im',channels:['DAPI','Boundary','R','G','B']}], models=[{name:'cpsam'},{name:'trained'},
    {name:'fluorescence_nuclei_and_cells'},{name:'single_channel_nuclei'},{name:'brightfield_nuclei'}];
  const fields=(value,o)=>stainProblems(value,images,models,o).map(p=>p.field), c=cellposeValue, i=instansegValue;
  const n=change=>({...c,normalize:{...c.normalize,...change}});
  assert.deepEqual(fields(c),[]);assert.deepEqual(fields(i),[]);
  for(const [value,expected] of [
    [{...c,model:'gone'},['stain-model']],[{...c,image:'other'},['stain-image']],[{...c,nucleus:'absent'},['stain-nucleus']],
    [{...c,cytoplasm:'DAPI'},['stain-cytoplasm']],[{...c,cytoplasm:'x'},['stain-cytoplasm']],[{...c,diameter:5},['stain-diameter']],
    [{...c,diameter:NaN},['stain-diameter']],[{...c,flow_threshold:11},['stain-flow']],[{...c,cellprob_threshold:-21},['stain-probability']],
    [{...c,niter:0},['stain-niter']],[{...c,niter:1.5},['stain-niter']],[{...c,min_size:-2},['stain-min-size']],
    [n({lowhigh:[NaN,10]}),['stain-intensity-low']],[n({lowhigh:[10,10]}),['stain-intensity-high']],[n({lowhigh:[0,2e9]}),['stain-intensity-high']],
    [n({percentile:[-1,99]}),['stain-low']],[n({percentile:[1,101]}),['stain-high']],[n({percentile:[50,40]}),['stain-high']],
    [n({percentile:[100,100]}),['stain-low']],[n({tile_norm_blocksize:1.5}),['stain-tile-norm']],[n({sharpen_radius:-1}),['stain-sharpen']],
    [n({smooth_radius:NaN}),['stain-smooth']],[{...c,method:'other'},['stain-method']],[{...c,tissue_only:'yes'},['stain-tissue']],
    [{...c,diameter:5,flow_threshold:11,niter:0},['stain-diameter','stain-flow','stain-niter']],
    [{...i,channels:[]},['stain-channels']],[{...i,channels:['DAPI','DAPI']},['stain-channels']],[{...i,channels:['nope']},['stain-channels']],
    [{...i,model:'single_channel_nuclei',output:'nuclei'},['stain-channels']],[{...i,model:'brightfield_nuclei',output:'nuclei'},['stain-channels']],
    [{...i,model:'single_channel_nuclei',channels:['DAPI']},['stain-output']],[{...i,output:'both'},['stain-output']],
    [{...i,model:'single_channel_nuclei',channels:['DAPI'],output:'nuclei',resolve_cells:true},['stain-resolve']],
  ]) {
    assert.deepEqual(fields(value),expected,JSON.stringify(value));
    assert.equal(validStain(value,images,models),false,JSON.stringify(value));
  }
  const say=(value,o)=>stainProblems(value,images,models,o).map(p=>`${p.label}: ${p.message}`);
  assert.deepEqual(say({...c,diameter:5}),[`${STAIN_TEXT.diameter}: ${PICKER_TEXT.diameter}`]);
  assert.deepEqual(say({...i,model:'single_channel_nuclei',output:'nuclei'}),[`${STAIN_TEXT.channels}: ${PICKER_TEXT.exactlyOne}`]);
  // Training patches read InstanSeg's channels from the nucleus and cell body fields (brightfield keeps its own list).
  const pair={pair:true};
  assert.deepEqual(fields({...i,channels:[]},pair),['stain-nucleus']);
  assert.deepEqual(fields({...i,channels:['DAPI','DAPI']},pair),['stain-cytoplasm']);
  assert.deepEqual(fields({...i,channels:['DAPI','x']},pair),['stain-cytoplasm']);
  assert.deepEqual(say({...i,model:'single_channel_nuclei',output:'nuclei'},pair),[`${STAIN_TEXT.cytoplasm}: ${PICKER_TEXT.oneChannel}`]);
  assert.deepEqual(fields({...i,model:'brightfield_nuclei',output:'nuclei'},pair),['stain-channels']);
  // No list (still loading, or Empty): the model is not checked; an empty list says none were found; no image at all.
  assert.deepEqual(stainProblems({...c,model:'gone'},images,null),[]);
  assert.equal(stainProblems({...c,model:'gone'},images,[])[0].message,PICKER_TEXT.noModels);
  assert.equal(stainProblems(c,[],null)[0].message,PICKER_TEXT.noImage);
});

test('stainProblems is empty exactly when validStain accepts, over thousands of mixed settings', () => {
  let seed=7;
  const rand=()=>(seed=(seed*1103515245+12345)%2147483648)/2147483648, pick=list=>list[Math.floor(rand()*list.length)];
  const images=[{key:'im',channels:['a','b','c']}];
  for(let k=0;k<4000;k++) {
    const value=rand()<.6?{method:pick(['cellpose',undefined,'x']),model:pick(['cpsam','gone']),image:pick(['im','im','no']),
      nucleus:pick(['a','b','z',undefined]),cytoplasm:pick([null,null,'a','b','z',undefined]),diameter:pick([null,5,10,30,1000,1001,NaN,undefined]),
      flow_threshold:pick([0,.4,10,11,-1,NaN]),cellprob_threshold:pick([-20,0,20,21,NaN]),tissue_only:pick([true,false,undefined,'x']),
      niter:pick([null,undefined,0,1,2.5,200]),min_size:pick([undefined,-2,-1,15,100000,100001,1.5]),
      normalize:pick([undefined,{normalize:pick([true,false,undefined,'x']),lowhigh:pick([null,undefined,[0,10],[10,0],[NaN,1],[0,2e9],[1,2,3]]),
        percentile:pick([null,undefined,[1,99],[0,100],[100,100],[50,40],[-1,99],[1,101],[1,NaN]]),tile_norm_blocksize:pick([0,1,1.5,-1,NaN]),
        sharpen_radius:pick([0,1,-1,NaN]),smooth_radius:pick([0,2,-1,NaN])}])} :
      {method:'instanseg',model:pick(['fluorescence_nuclei_and_cells','single_channel_nuclei','brightfield_nuclei','gone']),image:pick(['im','im','no']),
        channels:pick([[],['a'],['a','b'],['a','b','c'],['a','a'],['z'],['a','z'],undefined]),output:pick(['cells','nuclei','both',undefined]),
        resolve_cells:pick([true,false,undefined]),tissue_only:pick([true,false,'x'])};
    const models=pick([null,[],[{name:'cpsam'},{name:'fluorescence_nuclei_and_cells'},{name:'single_channel_nuclei'},{name:'brightfield_nuclei'}]]);
    for(const pair of [false,true])
      assert.equal(stainProblems(value,images,models,{pair}).length===0,validStain(value,images,models),JSON.stringify({value,models,pair}));
  }
});

test('largestPatch: the largest patch side, in the image\'s pixels, that fits the region', () => {
  assert.equal(largestPatch(300,300,{um_per_px:.5}),600);
  assert.equal(largestPatch(300,200,{um_per_px:.2125}),941,'200 µm / 0.2125 µm = 941.2 pixels');
  assert.equal(largestPatch(300,300,{um_per_px:.5,um_per_px_y:1}),300,'its own pixel height when the image has one');
  assert.equal(largestPatch(1.7,10,{um_per_px:.1}),16,'17 × 0.1 µm is just over 1.7 µm in floating point: never one more than fits');
  assert.equal(largestPatch(100,100,null),100);assert.equal(largestPatch(NaN,100,{um_per_px:1}),0);
});
