// A test-side encoder of the prior vectors binary, written from docs/ARCHITECTURE.md ("Prior vectors") and not from
// src/vectors-decode.mjs. vectors-decode.test.mjs checks that it makes the very bytes the engine's encoder made of
// fixtures/prior-vectors.json (fixtures/prior-vectors.bin, which tests/test_vectors.py pins on the Python side).
const LAYERS=['cells','nuclei'];

/** The binary of {cells, nuclei: [{id, poly: [[x, y], ...]}] (the JSON form) or null (absent)} plus meta. */
export function encodeVectors(layers,meta={}) {
  const text=new TextEncoder(),pad=n=>n+(-n&3),described=text.encode(JSON.stringify(meta));
  const records=[],sections=[];
  for(const name of LAYERS) {
    const polys=layers[name];
    if(!polys) {records.push(0,0,0,0);continue;}
    const names=[],index=new Map(),ids=new Uint32Array(polys.length),offsets=new Uint32Array(polys.length+1);
    polys.forEach((p,i)=>{
      if(!index.has(p.id)) {index.set(p.id,names.length);names.push(p.id);}
      ids[i]=index.get(p.id);offsets[i+1]=offsets[i]+p.poly.length;
    });
    const xy=Float32Array.from(polys.flatMap(p=>p.poly.flat()));
    const bbox=Float32Array.from(polys.flatMap(({poly})=>{
      const xs=poly.map(v=>v[0]),ys=poly.map(v=>v[1]);
      return [Math.min(...xs),Math.min(...ys),Math.max(...xs),Math.max(...ys)];   // [] -> +inf, +inf, -inf, -inf
    }));
    // Integer ids a u32 holds are carried as themselves (flag 2, no table); any other ids index a JSON table.
    const numeric=names.length>0 && names.every(v=>Number.isInteger(v) && v>=0 && v<=0xffffffff);
    const table=numeric?new Uint8Array(0):text.encode(JSON.stringify(names));
    if(numeric)ids.forEach((k,i)=>{ids[i]=names[k];});
    records.push(numeric?3:1,polys.length,xy.length/2,table.length);
    sections.push(offsets,xy,bbox,ids,table,new Uint8Array(pad(table.length)-table.length));
  }
  const size=pad(20+16*LAYERS.length+described.length),bytes=sections.reduce((n,s)=>n+s.byteLength,size);
  const out=new Uint8Array(bytes),view=new DataView(out.buffer);
  out.set(text.encode('PLPV'));
  [1,size,LAYERS.length,...records,described.length].forEach((v,k)=>view.setUint32(4+4*k,v,true));
  out.set(described,20+16*LAYERS.length);
  let at=size;
  for(const s of sections) {out.set(new Uint8Array(s.buffer,s.byteOffset,s.byteLength),at);at+=s.byteLength;}
  return out.buffer;
}

/** A dense whole sample's file (225,534 polygons, 5.59 M vertices) made directly, for timing the decoder. */
export function largeVectors(polygons=[112647,112887],perPolygon=25) {
  const layers=polygons.map(n=>({n,vertices:n*perPolygon}));
  const size=20+16*LAYERS.length+4;   // meta "{}" (2 bytes), padded
  const bytes=layers.reduce((s,l)=>s+4*(l.n+1)+8*l.vertices+16*l.n+4*l.n+4,size);
  const buffer=new ArrayBuffer(bytes),view=new DataView(buffer);
  new Uint8Array(buffer).set(new TextEncoder().encode('PLPV'));
  [1,size,LAYERS.length].forEach((v,k)=>view.setUint32(4+4*k,v,true));
  layers.forEach((l,k)=>[1,l.n,l.vertices,2].forEach((v,j)=>view.setUint32(16+16*k+4*j,v,true)));
  view.setUint32(16+16*LAYERS.length,2,true);new Uint8Array(buffer).set([123,125],20+16*LAYERS.length);
  let at=size;
  for(const l of layers) {
    const offsets=new Uint32Array(buffer,at,l.n+1);for(let i=0;i<=l.n;i++)offsets[i]=i*perPolygon;
    at+=4*(l.n+1)+8*l.vertices+20*l.n;
    new Uint8Array(buffer).set([91,93],at);at+=4;   // names "[]"
  }
  return buffer;
}
