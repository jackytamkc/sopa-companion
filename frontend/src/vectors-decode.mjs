// Prior vectors: GET /runs/{id}/xenium?format=bin&v=1 is one little-endian binary (docs/ARCHITECTURE.md, "Prior
// vectors") that becomes the shared PolygonSet of polygons.mjs as typed-array views over the response's own buffer: no
// parse, no per-vertex objects, no copy. Decoding is O(polygons) bounds checks: a few ms for a whole sample in Chrome,
// where the old JSON blocked the main thread for hundreds of ms and kept hundreds of MB of heap; the browser inflates
// the zstd transfer off the main thread, so no Worker is needed.
export const VECTORS_QUERY='format=bin&v=1';
const MAGIC=0x56504c50,VERSION=1,LAYERS=['cells','nuclei'];   // MAGIC: 'PLPV' read as a little-endian u32
const LITTLE=new Uint8Array(new Uint16Array([1]).buffer)[0]===1;  // typed arrays use the platform's byte order
const names=new WeakMap();   // PolygonSet -> what polygonIds() needs: {index, table bytes}, then the ids themselves

/** {...meta, cells, nuclei: PolygonSet {count, offsets, xy, bbox, ids} or null} from the binary's ArrayBuffer. ids is
 *  as polygonSetFromLegacy makes it from the JSON: the shapes' numeric ids, or null when they are not all integers
 *  (Xenium's string cell ids: those are polygonIds()). Throws 'Malformed prior vectors: ...' on anything that does
 *  not add up exactly (a truncated or foreign file). */
export function decodeVectors(buffer) {
  const t0=performance.now(),fail=why=>{throw new Error(`Malformed prior vectors: ${why}`);};
  if(!(buffer instanceof ArrayBuffer))fail('not an ArrayBuffer');
  if(!LITTLE)fail('this platform is big-endian');
  if(buffer.byteLength<20)fail('too short');
  const view=new DataView(buffer),u32=at=>view.getUint32(at,true);
  if(u32(0)!==MAGIC)fail('not a prior vectors file');
  if(u32(4)!==VERSION)fail(`version ${u32(4)}, expected ${VERSION}`);
  const size=u32(8),count=u32(12),metaAt=20+16*count;
  if(count!==LAYERS.length || size%4 || size>buffer.byteLength || metaAt>size || metaAt+u32(16+16*count)>size)fail('bad header');
  let meta;
  try {meta=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(new Uint8Array(buffer,metaAt,u32(16+16*count))));}
  catch {fail('bad metadata');}
  if(!meta || typeof meta!=='object' || Array.isArray(meta))fail('bad metadata');
  const value={...meta};
  let at=size;
  const take=(Type,length)=>{
    if(at+length*Type.BYTES_PER_ELEMENT>buffer.byteLength)fail('truncated');
    const array=new Type(buffer,at,length);at+=array.byteLength;return array;
  };
  LAYERS.forEach((name,k)=>{
    const flags=u32(16+16*k),n=u32(20+16*k),vertices=u32(24+16*k),text=u32(28+16*k);
    if(!(flags&1)) {value[name]=null;return;}
    const offsets=take(Uint32Array,n+1),xy=take(Float32Array,2*vertices),bbox=take(Float32Array,4*n),ids=take(Uint32Array,n);
    const table=take(Uint8Array,text+(-text&3)).subarray(0,text);
    if(offsets[0]!==0 || offsets[n]!==vertices)fail(`${name} offsets`);
    for(let i=0;i<n;i++)if(offsets[i+1]<offsets[i])fail(`${name} offsets`);
    const numeric=(flags&2)!==0,set={count:n,offsets,xy,bbox,ids:numeric?ids:null};
    names.set(set,numeric?null:{index:ids,table});value[name]=set;
  });
  if(at!==buffer.byteLength)fail('trailing bytes');
  try {   // cleared at once: observers (telemetry.mjs) have it, and the page's timeline must not grow with every decode
    performance.measure?.('plv:vectors-decode',{start:t0,detail:{bytes:buffer.byteLength,polygons:(value.cells?.count||0)+(value.nuclei?.count||0)}});
    performance.clearMeasures?.('plv:vectors-decode');
  } catch {/* a profiling aid only: never a reason not to show the outlines */}
  return value;
}

/** Each polygon's id as the JSON carried it (a Xenium cell id string, or a number): polygonIds(set)[i]. Made on first
 *  use (drawing never needs them); null for a set this module did not decode. Throws 'Malformed prior vectors: bad
 *  names' when the names table is not the JSON array of ids it must be. */
export function polygonIds(set) {
  if(!names.has(set))return null;
  let value=names.get(set);
  if(value===null)return Array.from(set.ids);
  if(!Array.isArray(value)) {
    let table;
    try {table=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(value.table));} catch {table=null;}
    if(!Array.isArray(table) || value.index.some(k=>k>=table.length))throw new Error('Malformed prior vectors: bad names');
    value=Array.from(value.index,k=>table[k]);names.set(set,value);
  }
  return value;
}
