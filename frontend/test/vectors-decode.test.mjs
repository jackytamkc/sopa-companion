import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeVectors, polygonIds, VECTORS_QUERY } from '../src/vectors-decode.mjs';
import { polygonSetFromLegacy } from '../src/polygons.mjs';
import { PriorLayers } from '../src/prior-layers.js';
import { layerPath } from '../src/prior-paths.mjs';
import { encodeVectors, largeVectors } from './vectors-fixture.mjs';

const read=name=>readFileSync(new URL(`./fixtures/${name}`,import.meta.url));
const legacy=JSON.parse(read('prior-vectors.json'));   // what GET /runs/{id}/xenium served
const bytes=read('prior-vectors.bin');                  // what the engine's encoder makes of it (tests/test_vectors.py)
const fixture=()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength);

test('the binary decodes into exactly the PolygonSets the viewer makes of the JSON it replaces',()=>{
  const value=decodeVectors(fixture());
  assert.deepEqual(value.source,legacy.source,'metadata the JSON carried is kept');
  for(const name of ['cells','nuclei']) {
    const set=value[name];
    assert.deepEqual(Object.keys(set),['count','offsets','xy','bbox','ids'],'the shared PolygonSet format');
    assert.ok(set.offsets instanceof Uint32Array && set.xy instanceof Float32Array && set.bbox instanceof Float32Array);
    assert.deepEqual(set,polygonSetFromLegacy(legacy[name]),'count, offsets, every coordinate bit for bit, boxes and ids');
    assert.deepEqual(polygonIds(set),legacy[name].map(p=>p.id),'the ids, as the JSON named them');
  }
  assert.equal(value.cells.ids,null,'string ids: no numeric ids, as from the JSON');
  assert.deepEqual([...value.nuclei.ids],[1,2,3],'integer ids are kept as numbers');
  assert.deepEqual([...value.nuclei.bbox.subarray(4,8)],[Infinity,Infinity,-Infinity,-Infinity],'no vertices: meets nothing');
  assert.equal(polygonIds(value.cells)[3],17,'a numeric id among strings stays a number');
});

test('decoding makes views over the response buffer, not copies, and the test encoder matches the engine byte for byte',()=>{
  const buffer=fixture(),value=decodeVectors(buffer);
  for(const name of ['cells','nuclei'])for(const key of ['offsets','xy','bbox'])assert.equal(value[name][key].buffer,buffer,`${name}.${key}`);
  assert.equal(value.nuclei.ids.buffer,buffer);
  const made=encodeVectors({cells:legacy.cells,nuclei:legacy.nuclei},{source:legacy.source});
  assert.deepEqual(new Uint8Array(made),new Uint8Array(fixture()));
  assert.equal(VECTORS_QUERY,'format=bin&v=1','the query the engine route serves the binary for');
});

test('absent layers are null and polygonIds knows only decoded sets',()=>{
  const value=decodeVectors(encodeVectors({cells:[{id:'a',poly:[[1,2],[3,4],[5,1]]}],nuclei:null}));
  assert.equal(value.nuclei,null);assert.equal(value.cells.count,1);assert.equal(value.source,undefined);
  assert.deepEqual(polygonIds(value.cells),['a']);assert.equal(polygonIds({count:0}),null);
  const empty=decodeVectors(encodeVectors({cells:[],nuclei:[]}));
  assert.deepEqual(empty.cells,polygonSetFromLegacy([]));assert.deepEqual(polygonIds(empty.nuclei),[]);
});

test('anything that does not add up exactly is refused',()=>{
  const good=new Uint8Array(fixture()),malformed={message:/^Malformed prior vectors: /};
  for(let k=0;k<good.length;k++)assert.throws(()=>decodeVectors(good.slice(0,k).buffer),malformed,`truncated to ${k} bytes`);
  const longer=new Uint8Array(good.length+4);longer.set(good);
  assert.throws(()=>decodeVectors(longer.buffer),/trailing bytes/);
  const edit=(at,value)=>{const copy=good.slice();new DataView(copy.buffer).setUint32(at,value,true);return copy.buffer;};
  assert.throws(()=>decodeVectors(edit(0,0x4e4f534a)),/not a prior vectors file/);
  assert.throws(()=>decodeVectors(edit(4,2)),/version 2, expected 1/);
  const size=new DataView(good.buffer).getUint32(8,true);
  assert.throws(()=>decodeVectors(edit(size+4,99)),/cells offsets/,'offsets must not decrease');
  assert.throws(()=>decodeVectors(edit(size,1)),/cells offsets/,'offsets start at 0');
  assert.throws(()=>decodeVectors(good),/not an ArrayBuffer/);
});

test('a dense whole sample\'s file decodes in well under the 20 ms that would call for a Worker',async()=>{
  const buffer=largeVectors(),runs=[],seen=[];
  // Telemetry sees the measure through an observer; the page's timeline keeps none of them (one per decode, for as long
  // as the page is open, before).
  const observer=new PerformanceObserver(list=>seen.push(...list.getEntriesByName('plv:vectors-decode')));
  observer.observe({entryTypes:['measure']});
  try {
    for(let k=0;k<5;k++){const t=performance.now();decodeVectors(buffer);runs.push(performance.now()-t);}
    const value=decodeVectors(buffer);
    assert.equal(value.cells.count+value.nuclei.count,225534);assert.equal(value.nuclei.xy.length,2*112887*25);
    assert.ok(Math.min(...runs)<20,`decode took ${Math.min(...runs).toFixed(2)} ms`);
    // The observer's callback is a task of its own: waited for (a busy machine delivers it later than a fixed 20 ms).
    for(const end=performance.now()+2000;seen.length<6&&performance.now()<end;) await new Promise(resolve=>setTimeout(resolve,10));
  } finally {observer.disconnect();}
  assert.equal(seen.length,6);
  assert.equal(seen.at(-1).detail.polygons,225534);assert.equal(seen.at(-1).detail.bytes,buffer.byteLength);
  assert.deepEqual(performance.getEntriesByName('plv:vectors-decode'),[],'cleared from the timeline');
});

test('a metadata block or names table that is not the JSON it must be is refused as malformed vectors',()=>{
  const good=new Uint8Array(fixture()),malformed={message:/^Malformed prior vectors: bad metadata$/};
  const metaAt=20+16*2,length=new DataView(good.buffer).getUint32(16+16*2,true);
  assert.ok(length>=2);
  const flipped=good.slice();flipped[metaAt]=0x00;                       // a bit flip: '{' -> NUL
  assert.throws(()=>decodeVectors(flipped.buffer),malformed,'not a raw SyntaxError the notice would show as is');
  const invalid=good.slice();invalid[metaAt+1]=0xff;                     // not UTF-8
  assert.throws(()=>decodeVectors(invalid.buffer),malformed);
  for(const meta of [[1,2],3,null,'text'])
    assert.throws(()=>decodeVectors(encodeVectors({cells:[],nuclei:null},meta)),malformed,JSON.stringify(meta));
  const value=decodeVectors(encodeVectors({cells:[{id:'a',poly:[[0,0],[1,0],[1,1]]},{id:'b',poly:[[2,2],[3,2],[3,3]]}],nuclei:null}));
  const bbox=value.cells.bbox,table=new Uint8Array(bbox.buffer,bbox.byteOffset+bbox.byteLength+4*value.cells.count,4);
  assert.equal(table[0],0x5b,'the names table: a JSON array');
  table[0]=0x7b;                                                         // '[' -> '{': read lazily, it is no array now
  assert.throws(()=>polygonIds(value.cells),{message:'Malformed prior vectors: bad names'});
});

/** A Path2D stand-in that records its ops (addPath appends the other path's). */
class RecPath {
  constructor(){this.ops=[];}
  moveTo(x,y){this.ops.push(['M',x,y]);} lineTo(x,y){this.ops.push(['L',x,y]);} closePath(){this.ops.push(['Z']);}
  addPath(p){this.ops.push(...p.ops);}
}

test('PriorLayers takes the decoded sets as they are and strokes exactly the paths the JSON gave',()=>{
  // The producer (this decoder) and the consumer (prior-layers.js, prior-paths.mjs) of the shared PolygonSet format.
  const saved=Object.getOwnPropertyDescriptor(globalThis,'document');
  globalThis.document={createElement:()=>({width:0,height:0,getContext:()=>null})};
  try {
    const grid=[];
    for(let j=0;j<30;j++)for(let i=0;i<30;i++){const x=i*20+.25,y=j*20+.5;grid.push({id:`c${i}-${j}`,poly:[[x,y],[x+20,y],[x+20,y+20],[x,y+20],[x,y]]});}
    for(const [json,buffer] of [[legacy,fixture()],[{cells:grid,nuclei:grid.filter((_,k)=>k%7===0),source:{}},null]]) {
      const value=decodeVectors(buffer || encodeVectors({cells:json.cells,nuclei:json.nuclei},{source:json.source}));
      const binary=new PriorLayers({},()=>'rgb(0,0,0)'),old=new PriorLayers({},()=>'rgb(0,0,0)');
      binary.vectors=value;old.vectors=json;
      assert.equal(binary.vectors,value,'installed without a conversion or a copy');
      for(const name of ['cells','nuclei']) {
        assert.equal(binary.vectors[name],value[name]);
        assert.deepEqual(polygonIds(binary.vectors[name]),json[name].map(p=>p.id),'ids still resolve through the installed set');
        for(const [box,zoom] of [[[-1e9,-1e9,1e9,1e9],4],[[100,100,300,250],2],[[-1e9,-1e9,1e9,1e9],1e-3]]) {
          const a=layerPath(binary.vectors[name],box,zoom,RecPath),b=layerPath(old.vectors[name],box,zoom,RecPath);
          assert.deepEqual([a.polygons,a.boxes],[b.polygons,b.boxes]);
          assert.deepEqual(a.path.ops,b.path.ops,`${name} at zoom ${zoom}: the same subpaths, vertex for vertex`);
        }
      }
    }
  } finally {if(saved)Object.defineProperty(globalThis,'document',saved);else delete globalThis.document;}
});
