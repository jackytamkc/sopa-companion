import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CellValues, decodeCellValues, CELL_VALUES_TEXT } from '../src/stain-cells.mjs';
import { exclusionRanges } from '../src/stain-exclusions.mjs';
import { deferred } from './dom-fixture.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

// PLCV v1 as the engine writes it (engine/vectors.py encode_cells): header, metadata, ids, area, flags, brightness.
// `brightness`: {channel: values}, in the metadata's order; `pad`: bytes of padding after the metadata's own (the format
// allows any multiple of four).
function encode({set='s',areas_version=0,area=[10,20,30],flags=[0,0,0],brightness={},error=null,pad=0}={}) {
  const names=Object.keys(brightness),n=area.length;
  const json=new TextEncoder().encode(JSON.stringify({set,areas_version,channels:['DAPI','Cy5'],default:'DAPI',dtype:'uint16',error,brightness:names}));
  const header=Math.ceil((24+json.length)/4)*4+pad,flagsBytes=Math.ceil(n/4)*4;
  const buffer=new ArrayBuffer(header+8*n+flagsBytes+4*n*names.length),view=new DataView(buffer);
  [0x56434c50,1,header,n,names.length,json.length].forEach((v,i)=>view.setUint32(i*4,v,true));
  new Uint8Array(buffer,24,json.length).set(json);
  new Uint32Array(buffer,header,n).set(area.map((_,i)=>i+1));new Float32Array(buffer,header+4*n,n).set(area);
  new Uint8Array(buffer,header+8*n,n).set(flags);
  names.forEach((name,k)=>new Float32Array(buffer,header+8*n+flagsBytes+4*n*k,n).set(brightness[name].map(v=>v??NaN)));
  return buffer;
}
// Made by the engine itself (engine/vectors.py encode_cells, as GET …/stain/cells answers before packing): five cells of
// set r1:resolved:6:10 at area version 2, the fourth in a removed area, with and without DAPI's brightness (one unknown).
const ENGINE={DAPI:'UExDVgEAAACkAAAABQAAAAEAAACLAAAAeyJzZXQiOiJyMTpyZXNvbHZlZDo2OjEwIiwiYXJlYXNfdmVyc2lvbiI6MiwiY2hhbm5lbHMiOlsiREFQSSIsIkN5NSJdLCJkZWZhdWx0IjoiREFQSSIsImR0eXBlIjoidWludDE2IiwiZXJyb3IiOm51bGwsImJyaWdodG5lc3MiOlsiREFQSSJdfQABAAAAAgAAAAMAAAAEAAAABQAAAP//x0IAAMhCAAAgQQAAyEEAAEhCAAAAAQAAAAAAACBBAACgQQAAwH8AACBCAABIQg==',
  none:'UExDVgEAAACgAAAABQAAAAAAAACFAAAAeyJzZXQiOiJyMTpyZXNvbHZlZDo2OjEwIiwiYXJlYXNfdmVyc2lvbiI6MiwiY2hhbm5lbHMiOlsiREFQSSIsIkN5NSJdLCJkZWZhdWx0IjoiREFQSSIsImR0eXBlIjoidWludDE2IiwiZXJyb3IiOm51bGwsImJyaWdodG5lc3MiOltdfQAAAAEAAAACAAAAAwAAAAQAAAAFAAAA///HQgAAyEIAACBBAADIQQAASEIAAAABAAAAAA=='};
const fromBase64=text=>{const bytes=Buffer.from(text,'base64');return bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);};

test('PLCV reads aligned arrays and refuses unknown, truncated and malformed payloads',()=>{
  const good=encode({brightness:{DAPI:[1,NaN,3]}});const data=decodeCellValues(good);
  assert.deepEqual([...data.area],[10,20,30]);assert.deepEqual([...data.ids],[1,2,3]);assert.equal(data.area.buffer,good);
  assert.ok(Number.isNaN(data.brightness.get('DAPI')[1]));assert.equal(data.dtype,'uint16');
  for(const offset of [0,4,8,12,16,20]) {
    const bad=good.slice(0);new DataView(bad).setUint32(offset,999,true);assert.throws(()=>decodeCellValues(bad),/Malformed/);
  }
  const shuffled=good.slice(0);new Uint32Array(shuffled,decodeCellValues(good).ids.byteOffset,3).set([2,1,3]);
  for(const bad of [null,good.slice(0,20),good.slice(0,-1),encode({set:''}),encode({areas_version:-1}),encode({flags:[2,0,0]}),shuffled])
    assert.throws(()=>decodeCellValues(bad),/Malformed/);
  // More padding after the metadata, as engine/vectors.py decode_cells accepts: the same values. A header that ends inside
  // the metadata, or is not a multiple of four, is refused even when the lengths add up (the arrays moved to start there).
  const padded=decodeCellValues(encode({brightness:{DAPI:[1,NaN,3]},pad:8}));
  assert.deepEqual([...padded.area],[10,20,30]);assert.deepEqual([...padded.brightness.get('DAPI')],[...data.brightness.get('DAPI')]);
  const at=data.ids.byteOffset,bytes=new Uint8Array(good);
  for(const header of [at-4,at+2]) {
    const bad=new Uint8Array(good.byteLength+header-at);
    bad.set(bytes.subarray(0,Math.min(at,header)));bad.set(bytes.subarray(at),header);new DataView(bad.buffer).setUint32(8,header,true);
    assert.throws(()=>decodeCellValues(bad.buffer),/Malformed/,String(header));
  }
});
test("the engine's own payloads decode: ids, sizes, removed-area flags and a channel's brightness, NaN unknown",()=>{
  const data=decodeCellValues(fromBase64(ENGINE.DAPI));
  assert.equal(data.set,'r1:resolved:6:10');assert.equal(data.areas_version,2);assert.equal(data.count,5);
  assert.deepEqual([...data.ids],[1,2,3,4,5]);assert.deepEqual([...data.flags],[0,0,0,1,0]);
  assert.deepEqual([...data.area],[Math.fround(99.99999237),100,10,25,50]);
  assert.deepEqual([...data.brightness.keys()],['DAPI']);assert.deepEqual([...data.brightness.get('DAPI')],[10,20,NaN,40,50]);
  const sizes=decodeCellValues(fromBase64(ENGINE.none));
  assert.equal(sizes.brightness.size,0);assert.deepEqual([...sizes.area],[...data.area]);
});
test('cell classes combine areas, size and per-channel ranges with inclusive float64 thresholds and NaN kept',async()=>{
  const area=[Math.fround(99.99999237),100,10,25,50];
  const values=new CellValues({stainCells:async(_id,_set,channel)=>encode({area,flags:[0,0,0,1,0],
    brightness:channel==='DAPI'?{DAPI:[10,20,NaN,40,50]}:channel==='Cy5'?{Cy5:[0,0,0,0,100]}:{}})});
  await values.load('r','s',0);await values.load('r','s',0,'DAPI');await values.load('r','s',0,'Cy5');
  const rules={size:{min:99.999995,max:100},brightness:{DAPI:{min:10,max:20}}};
  const classes=values.classes(rules);assert.deepEqual([...classes.array],[1,0,1,1,1]);assert.equal(classes.kept,1);
  assert.equal(values.classes(rules),classes,'unchanged rules reuse the generation');
  assert.deepEqual([...values.classes({brightness:{DAPI:{min:15,max:50},Cy5:{min:null,max:99}}}).array],[1,0,0,1,1]);
  assert.equal(values.forSet('old',rules),null,'a stale set paints as kept');
});
test('the shared range cases (tests/fixtures/stain_ranges.json) keep the same cells as the engine',()=>{
  const cases=JSON.parse(readFileSync(new URL('../../tests/fixtures/stain_ranges.json',import.meta.url),'utf8'));
  assert.ok(cases.length>=10);
  for(const c of cases) {
    const values=new CellValues();values.set='s';
    values.data=decodeCellValues(encode({area:c.area.map(v=>v??NaN),flags:c.in_area.map(Number),brightness:c.brightness}));
    for(const [channel,array] of values.data.brightness)values.brightness.set(channel,array);
    const {size,brightness}=exclusionRanges(c.rules);   // a legacy file's minimums read as ranges, as the panel reads them
    const classes=values.classes({size,brightness});
    assert.deepEqual([...classes.array].map(v=>v===0),c.keep,c.name);
    assert.equal(classes.kept,c.keep.filter(Boolean).length,c.name);
  }
});
test('loads coalesce, abort an old set, reject mismatched area versions, and never install late results',async()=>{
  const reads=[],errors=[];const values=new CellValues({stainCells:(...args)=>{const d=deferred();reads.push({...d,args});return d.promise;}},
    {error:e=>errors.push(e)});
  const first=values.load('r','s',0),same=values.load('r','s',0);assert.equal(reads.length,1);
  assert.deepEqual(reads[0].args.slice(0,3),['r','s',null],'api.js stainCells(id, set, channel, signal)');
  const next=values.load('r','new',1);assert.ok(reads[0].args[3].aborted);
  reads[0].resolve(encode());await first;await same;assert.equal(values.data,null);
  reads[1].resolve(encode({set:'new',areas_version:0}));await next;assert.equal(values.data,null);assert.equal(errors.length,1);
  await values.load('r','new',1);assert.equal(reads.length,2,'no retry loop after a failure');
  const retry=values.load('r','new',2);reads[2].resolve(encode({set:'new',areas_version:2}));await retry;
  assert.equal(values.data.areas_version,2);
  // A channel's answer must carry that channel.
  const other=values.load('r','new',2,'DAPI');reads[3].resolve(encode({set:'new',areas_version:2,brightness:{Cy5:[1,2,3]}}));
  await other;assert.equal(values.brightness.has('DAPI'),false);assert.equal(errors.at(-1),CELL_VALUES_TEXT.brightness);
});
test('cells whose brightness could not be read are kept, and the chooser is told why: some of them, or none measured',async()=>{
  const error='The image could not be read to measure DAPI: gone',errors=[];
  const values=new CellValues({stainCells:async(_id,_set,channel)=>encode({brightness:channel==='DAPI'?{DAPI:[NaN,5,NaN]}:
    {Cy5:[NaN,NaN,NaN]},error})},{error:(message,channel)=>errors.push([message,channel])});
  await values.load('r','s',0,'DAPI');
  assert.deepEqual(errors,[[CELL_VALUES_TEXT.unmeasured,'DAPI']]);
  assert.deepEqual([...values.classes({brightness:{DAPI:{min:10,max:null}}}).array],[0,1,0]);
  // Every cell unknown (the run's one image file could not be read): never "Some cells".
  await values.load('r','s',0,'Cy5');
  assert.deepEqual(errors.at(-1),[CELL_VALUES_TEXT.brightness,'Cy5']);assert.equal(values.brightness.get('Cy5').length,3);
  assert.equal(values.classes({brightness:{Cy5:{min:10,max:null}}}).kept,3);
});
test('unmeasured channels keep cells and the HTTP fallback escapes run, set and channel',async t=>{
  const reads=[];t.mock.method(globalThis,'fetch',async(url,options)=>{reads.push({url,options});return {ok:true,arrayBuffer:async()=>encode({set:'a & b',brightness:{'DAPI / Cy5':[NaN,NaN,NaN]}})};});
  const values=new CellValues();await values.load('run / 1','a & b',0,'DAPI / Cy5');
  assert.equal(values.classes({brightness:{missing:{min:100,max:null}}}).kept,3);
  assert.match(reads[0].url,/run%20%2F%201\/stain\/cells\?/);
  const query=new URL(reads[0].url,'http://local').searchParams;assert.equal(query.get('set'),'a & b');assert.equal(query.get('channel'),'DAPI / Cy5');
  values.reset();assert.equal(values.forSet('a & b'),null);
});

test('cell value failure words are plain',()=>{
  for(const words of Object.values(CELL_VALUES_TEXT))assert.deepEqual(bannedWords(words),[]);
});
