import test from 'node:test';
import assert from 'node:assert/strict';
import { proceedOffer, proceedPrefill, PROCEED_TEXT } from '../src/proceed.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { priorChoices, choiceIndex } from '../src/prior-options.mjs';
const run={kind:'stain',sample_kind:'zarr',sample_tools:['stain','proseg'],region:{kind:'box',box:[0,0,100,100]},
  progress:{done:4,total:4,complete:true,revision:2},exclusions:{digest:'same',version:9},
  zarr_save:{state:'saved',cells:20,shapes_key:'saved_cells',result_revision:2,exclusions_digest:'same'}};
test('Proceed requires a saved stain result with enough cells and transcripts',()=>{
  assert.equal(proceedOffer(run).ready,true);
  for(const change of [{kind:'proseg'},{kind:'patches'},{sample_kind:'bundle'},
    {zarr_save:null},{zarr_save:{...run.zarr_save,cells:1}},{zarr_save:{...run.zarr_save,state:'saving'}}])
    assert.equal(proceedOffer({...run,...change}).shown,false);
  const only=proceedOffer({...run,sample_tools:['stain']});assert.equal(only.shown,false);assert.equal(only.reason,PROCEED_TEXT.noTranscripts);
});
test('current digests decide readiness, including Undo to the saved filters',()=>{
  assert.equal(proceedOffer({...run,exclusions:{digest:'changed',version:10}}).ready,false);
  assert.equal(proceedOffer({...run,exclusions:{digest:'same',version:11}}).ready,true);
  assert.equal(proceedOffer(run,{pending:true}).reason,PROCEED_TEXT.waiting);
  assert.equal(proceedOffer({...run,progress:{...run.progress,revision:3}}).reason,PROCEED_TEXT.outdated);
  assert.equal(proceedOffer({...run,zarr_save:{...run.zarr_save,state:'outdated'}}).ready,false);
});
test('API queued results are available only when added; range changes and resume make them stale',()=>{
  const queued={...run,zarr_save:{...run.zarr_save,state:'queued',written_to:'/data/sample-1.zarr'}};
  assert.equal(proceedOffer(queued).shown,false);assert.equal(proceedOffer(queued,{api:true}).ready,true);
  assert.equal(proceedOffer({...queued,exclusions:{digest:'changed'}},{api:true}).reason,PROCEED_TEXT.outdatedApi);
  assert.equal(proceedOffer({...queued,progress:{...queued.progress,revision:3}},{api:true}).ready,false);
});
test('API: offered only for cells in the zarr the app reads; added in memory only (sopa\'s auto-save off), a reason instead',()=>{
  const save={...run.zarr_save,written_to:'/data/sample-1.zarr',view_written:null};
  assert.equal(proceedOffer({...run,zarr_save:{...save,state:'queued'}},{api:true}).ready,true,'written into the zarr it reads in place');
  for(const state of ['queued','saved','outdated'])
    assert.deepEqual(proceedOffer({...run,zarr_save:{...save,state,written_to:null}},{api:true}),
      {shown:false,ready:false,reason:PROCEED_TEXT.memoryOnly},state);
  // The app's own copy of the sample: what it took decides, wherever else the cells went.
  assert.equal(proceedOffer({...run,zarr_save:{...save,written_to:null,view_written:true}},{api:true}).ready,true);
  assert.equal(proceedOffer({...run,zarr_save:{...save,view_written:false}},{api:true}).reason,PROCEED_TEXT.notInLayers);
  // The desktop saves into the sample itself: never asked.
  assert.equal(proceedOffer({...run,zarr_save:{...run.zarr_save,written_to:null}}).ready,true);
});
test('the saved shapes alone seed proseg, even if older save data contains a pairing',()=>{
  const prefill=proceedPrefill({...run,zarr_save:{...run.zarr_save,pairing:{fits:true,nuclei:{kind:'shapes',key:'nuclei'}}}});
  assert.deepEqual(prefill,{region:run.region,prior:{cells:{kind:'shapes',key:'saved_cells'},nuclei:null},notes:[]});
  prefill.region.box[0]=10;assert.equal(run.region.box[0],0,'prefill is an independent editable copy');
  const options={cells:[{kind:'shapes',key:'saved_cells',label:'Saved cells'}],nuclei:[{kind:'shapes',key:'nuclei'}]};
  const choices=priorChoices(options,{cells:{kind:'shapes',key:'saved_cells'},nuclei:{kind:'shapes',key:'nuclei'}});
  assert.equal(choiceIndex(choices,prefill.prior),1,'the cells-only choice, even when the sample suggests a pairing');
});
test('partial runs need a usable covered region and explain the smaller area in one note',()=>{
  const partial={...run,progress:{...run.progress,complete:false,done:2},zarr_save:{...run.zarr_save,
    covered:{region:{kind:'box',box:[0,0,40,40]},done:2,of:4,why:'tiles',pieces:2}}};
  assert.equal(proceedOffer(partial).ready,true);assert.equal(proceedOffer(partial).title,PROCEED_TEXT.partialTooltip);
  const prefill=proceedPrefill(partial);assert.deepEqual(prefill.region,partial.zarr_save.covered.region);
  assert.equal(prefill.notes.length,1);assert.equal(prefill.notes[0].text,PROCEED_TEXT.limited);
  assert.match(prefill.notes[0].detail,/2 of 4 tiles/);assert.match(prefill.notes[0].detail,/largest part/);
  assert.equal(proceedOffer({...partial,zarr_save:run.zarr_save}).reason,PROCEED_TEXT.small);
});
test('Proceed words are plain',()=>{for(const value of Object.values(PROCEED_TEXT))assert.deepEqual(bannedWords(value),[]);});
