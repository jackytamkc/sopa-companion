import test from 'node:test';
import assert from 'node:assert/strict';
import {priorSpec,priorIndex} from '../src/prior-options.mjs';

test('prior specs carry only contract fields and retain falsy column sentinels',()=>{
  assert.deepEqual(priorSpec({kind:'shapes',key:'cells',label:'Cell boundaries',count:200}),{kind:'shapes',key:'cells'});
  assert.deepEqual(priorSpec({kind:'column',column:'cell_id',unassigned:0,nuclear:false,label:'Assigned cells'}),{kind:'column',column:'cell_id',unassigned:0,nuclear:false});
  assert.equal(priorSpec(null),null);
});
test('suggested priors match options by identity, independently of display metadata',()=>{
  const choices=[{kind:'shapes',key:'cells',label:'Cells'}, {kind:'column',column:'cell_id',unassigned:0,label:'Transcript assignments'}];
  assert.equal(priorIndex(choices,{kind:'shapes',key:'cells'}),0);
  assert.equal(priorIndex(choices,{kind:'column',column:'cell_id',unassigned:0}),1);
  assert.equal(priorIndex(choices,null),-1);
  assert.equal(priorIndex(choices,{kind:'shapes',key:'missing'}),-1);
});
