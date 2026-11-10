import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeStats, decodeCounted, keptOf, readoutText, readoutTip, READOUT, READOUT_TIPS, READOUT_COUNTS, countName, mostActive, CellStats } from '../src/cellstats.mjs';
import { bannedWords } from '../src/qc-copy.mjs';
import { maskMapping, sampleMasks } from '../src/xenium-helpers.mjs';
test('statistics layout preserves counts and floating centroids',()=>{
  const b=new ArrayBuffer(56);new Uint32Array(b).set([2,3,4,5,6,7,8,9,10,11]);new Float32Array(b,40).set([1.25,NaN,-2.5,4]);
  const s=decodeStats(b);assert.deepEqual([...s.assigned],[6,7]);assert.equal(s.x[0],1.25);assert.ok(Number.isNaN(s.x[1]));assert.equal(s.y[0],-2.5);assert.throws(()=>decodeStats(new ArrayBuffer(27)),/Truncated/);
});
test('most active ranks changes, excludes zero, and bounds list',()=>{
  assert.deepEqual(mostActive({changed:new Uint32Array([0,5,9,5,1])},3),[{id:2,changed:9},{id:1,changed:5},{id:3,changed:5}]);
});
// The previous implementation (a sort per changed cell), kept as the reference for the single-pass one.
const reference=(stats,count=8)=>{
  const best=[];
  stats.changed.forEach((changed,id)=>{if(!changed)return;best.push({id,changed});best.sort((a,b)=>b.changed-a.changed||a.id-b.id);if(best.length>count)best.pop();});
  return best;
};
test('most active equals the sorting reference on random data (ties, zeros, any count)',()=>{
  let seed=12345;const random=()=>((seed=Math.imul(seed^seed>>>15,2246822519)+0x9e3779b9>>>0)/2**32);
  for(let trial=0;trial<400;trial++){
    const n=[0,1,2,7,64,500,4000][trial%7],range=[1,2,5,40,1e3,4e9][trial%6],zeros=random();
    const changed=new Uint32Array(n).map(()=>random()<zeros?0:1+Math.floor(random()*range));
    for(const count of [0,1,2,3,8,13,n,n+3]){
      const stats={changed};
      assert.deepEqual(mostActive(stats,count),reference(stats,count),`n=${n} range=${range} count=${count}`);
    }
    assert.deepEqual(mostActive({changed}),reference({changed}));
    assert.deepEqual(mostActive({changed:[...changed]},5),reference({changed},5));     // plain arrays as well
  }
  // Ascending, descending and constant runs (worst cases for a small sorted buffer).
  for(const changed of [Uint32Array.from({length:3000},(_,i)=>i),Uint32Array.from({length:3000},(_,i)=>3000-i),new Uint32Array(3000).fill(4)])
    assert.deepEqual(mostActive({changed}),reference({changed}));
  assert.deepEqual(mostActive({changed:new Uint32Array([3,1,4])},-1),[]);
  assert.deepEqual(mostActive({changed:new Uint32Array([3,1,4])},2.5),reference({changed:new Uint32Array([3,1,4])},2.5));
});
test('most active is a single pass over a whole sample of cells: it never sorts',()=>{
  let seed=7;const changed=Uint32Array.from({length:40000},()=>(seed=Math.imul(seed,48271)%2147483647)%60);
  const expected=reference({changed}),sort=Array.prototype.sort;let sorts=0,result;
  Array.prototype.sort=function(...args){sorts++;return sort.apply(this,args);};
  try{result=mostActive({changed});}finally{Array.prototype.sort=sort;}
  assert.equal(sorts,0);assert.deepEqual(result,expected);
});
test('Xenium mapping preserves dropped identities as neutral and nearest sampling preserves labels',()=>{
  assert.deepEqual([...maskMapping({1:'b',2:'dropped',3:'a'},['a','b'])],[0,2,0,1]);
  const p=sampleMasks(new Uint32Array([1,2,3,4,5,6,7,8,9,10,11,12]),4,3,2);
  assert.equal(p.step,2);assert.deepEqual([...p.data],[1,3,9,11]);assert.equal(p.ny,2);
});
test('statistics cancels stale fetches and reuses cached frames',async()=>{
  const original=globalThis.fetch,requests=[];
  globalThis.fetch=(url,{signal})=>new Promise(resolve=>requests.push({url,signal,resolve}));
  try {
    const store=new CellStats();store.reset('a');const first=store.load(0),second=store.load(1);
    assert.ok(requests[0].signal.aborted);
    requests[1].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(28)});await second;
    requests[0].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(28)});await first;
    assert.equal(store.current.seq,1);await store.load(1);assert.equal(requests.length,2);
    const stale=store.load(2);store.reset('b');requests[2].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(28)});await stale;assert.equal(store.current,null);
  }finally{globalThis.fetch=original;}
});
test('asking again for the statistics on screen is not a change (no repeated re-render)',async()=>{
  const original=globalThis.fetch;let changes=0;
  globalThis.fetch=async()=>({ok:true,arrayBuffer:async()=>new ArrayBuffer(28)});
  try {
    const store=new CellStats(()=>changes++);store.reset('a');
    await store.load(0);assert.equal(changes,1);
    const cached=store.current;await store.load(0);await store.load(0);assert.equal(changes,1);assert.equal(store.current,cached);
    await store.load(1);assert.equal(changes,2);
    assert.equal(await store.load(0),cached);assert.equal(changes,3);assert.equal(store.current,cached);   // back to a cached frame: a change
  }finally{globalThis.fetch=original;}
});
test('a prefetch fills the cache without touching the frame on screen, and neither request cancels the other',async()=>{
  const original=globalThis.fetch,requests=[];let changes=0;
  globalThis.fetch=(url,{signal})=>new Promise(resolve=>requests.push({seq:Number(url.split('/').at(-1)),signal,resolve}));
  const reply=i=>requests[i].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(28)});
  try {
    const store=new CellStats(()=>changes++);store.reset('a');
    const shown=store.load(0);reply(0);await shown;assert.equal(store.current.seq,0);assert.equal(changes,1);
    const next=store.prefetch(1);assert.equal(store.current.seq,0,'frame 0 keeps its numbers');
    await store.load(0);assert.equal(requests.length,2);assert.equal(requests[1].signal.aborted,false);   // asked again: cached
    reply(1);assert.equal((await next).seq,1);
    assert.equal(store.current.seq,0);assert.equal(changes,1,'a prefetch is no change on screen');
    assert.equal((await store.load(1)).seq,1);assert.equal(store.current.seq,1);assert.equal(changes,2);
    assert.deepEqual(requests.map(r=>r.seq),[0,1],'shown from the cache: one request per frame');
    // Frame 2's statistics are slow; snapshot 3 is on its way: its prefetch leaves frame 2's request alone, and frame
    // 2 asked again (the analysis panel) leaves snapshot 3's alone.
    const slow=store.load(2);store.prefetch(3);store.load(2);
    assert.equal(requests[2].signal.aborted,false);assert.equal(requests[3].signal.aborted,false);assert.equal(requests.length,4);
    reply(2);await slow;assert.equal(store.current.seq,2);
    store.prefetch(4);assert.equal(requests[3].signal.aborted,true,'a newer snapshot supersedes the older prefetch');
    reply(4);await new Promise(r=>setImmediate(r));
    store.load(4);assert.equal(store.current.seq,4);assert.equal(requests.length,5);
  }finally{globalThis.fetch=original;}
});
test('a failed prefetch is no failure: the frame is asked for again once it is on screen',async()=>{
  const original=globalThis.fetch;let changes=0,calls=0,reply={ok:false,status:503};
  globalThis.fetch=async()=>{calls++;return reply;};
  try {
    const store=new CellStats(()=>changes++);store.reset('a');
    assert.equal(await store.prefetch(5),null);assert.equal(store.failed,null);assert.equal(changes,0);
    reply={ok:true,arrayBuffer:async()=>new ArrayBuffer(28)};
    assert.equal((await store.load(5)).seq,5);assert.equal(calls,2);assert.equal(store.current.seq,5);assert.equal(changes,1);
  }finally{globalThis.fetch=original;}
});
test('a failed load names its frame (and notifies) until the next attempt; a superseded one is no failure',async()=>{
  const original=globalThis.fetch;let changes=0,reply={ok:false,status:404};
  globalThis.fetch=async()=>{if(reply==='abort')throw Object.assign(Error('superseded'),{name:'AbortError'});return reply;};
  try {
    const store=new CellStats(()=>changes++);store.reset('a');
    assert.equal(store.failed,null);
    assert.equal(await store.load(3),null);assert.equal(store.failed,3);assert.equal(store.current,null);assert.equal(changes,1);
    reply={ok:true,arrayBuffer:async()=>new ArrayBuffer(28)};
    const again=store.load(3);assert.equal(store.failed,null,'trying again');
    await again;assert.equal(store.current.seq,3);assert.equal(store.failed,null);assert.equal(changes,2);
    reply='abort';await store.load(4);assert.equal(store.failed,null);assert.equal(changes,2);
    reply={ok:false,status:500};await store.load(5);assert.equal(store.failed,5);
    store.reset('b');assert.equal(store.failed,null);
  }finally{globalThis.fetch=original;}
});

// ---- proseg's own count of each cell (a counted frame's counted file, GET /runs/{id}/counted/{seq}) -----------------
test("proseg's count of each cell: little-endian f32 per cell, apart from the statistics' 28-byte layout",()=>{
  const b=new ArrayBuffer(12),v=new DataView(b);[12.5,0,1234.03125].forEach((x,i)=>v.setFloat32(i*4,x,true));
  const kept=decodeCounted(b);assert.ok(kept instanceof Float32Array);assert.deepEqual([...kept],[12.5,0,1234.03125]);
  assert.equal(decodeCounted(new ArrayBuffer(0)).length,0);
  for(const n of [1,2,3,13])assert.throws(()=>decodeCounted(new ArrayBuffer(n)),/^Error: Truncated proseg counts$/);
  // The statistics' own layout is untouched: 28 bytes per cell, a 4-byte multiple that is no counted file's concern.
  assert.throws(()=>decodeStats(new ArrayBuffer(32)),/Truncated cell statistics/);
});
test("a cell's count reads the counted file when the frame has one, else the frame's non-background count; the readout says it in words",()=>{
  const stats={area:Uint32Array.of(3,5),voxels:Uint32Array.of(9,40),assigned:Uint32Array.of(120,7),foreground:Uint32Array.of(80,6)};
  assert.equal(keptOf(stats,0),80);assert.equal(keptOf(stats,1),6);assert.equal(keptOf(stats,2),null);assert.equal(keptOf(stats,-1),null);
  assert.equal(keptOf(null,0),null);
  const counted={...stats,kept:Float32Array.of(91.40625,7)};assert.equal(keptOf(counted,0),91.40625);
  // Whole numbers: what proseg sets aside as background is assigned - kept, never below 0; the count named as it is
  // (H8): the counted file "(averaged)", the frame's own draw "(as saved)".
  assert.equal(readoutText(counted,0,{voxel:2,cluster:'Group K'}),"12 µm² · 91 in proseg's count (averaged) · 29 set aside as background · 9 voxels · Group K");
  assert.equal(readoutText(stats,0,{voxel:0.5,cluster:'Unanalysed'}),"0.8 µm² · 80 in proseg's count (as saved) · 40 set aside as background · 9 voxels · Unanalysed");
  assert.equal(readoutText({...stats,assigned:Uint32Array.of(1200,7),kept:Float32Array.of(1234.6,7)},0,{voxel:1,cluster:''}),
    "3 µm² · 1,235 in proseg's count (averaged) · 0 set aside as background · 9 voxels · ");
  // A count ending in exactly .5 (a sum of 32nds: about one cell in 32): the two whole numbers still add up to what
  // proseg placed in the cell.
  for(const [kept,shown,bg] of [[91.5,'92','28'],[90.5,'91','29'],[91.46875,'91','29'],[119.5,'120','0']])
    assert.equal(readoutText({...stats,kept:Float32Array.of(kept,7)},0,{voxel:2,cluster:'Group K'}),
      `12 µm² · ${shown} in proseg's count (averaged) · ${bg} set aside as background · 9 voxels · Group K`,String(kept));
  assert.equal(READOUT,'{area} µm² · {kept} in {count} · {bg} set aside as background · {voxels} voxels · {cluster}');
  assert.deepEqual(READOUT_COUNTS,{averaged:"proseg's count (averaged)",saved:"proseg's count (as saved)"});
  assert.equal(countName(counted),READOUT_COUNTS.averaged);assert.equal(countName(stats),READOUT_COUNTS.saved);assert.equal(countName(null),READOUT_COUNTS.saved);
  // Its words name proseg's count, never an outline count (the readout's "voxels" is the frame's own word).
  for(const words of [READOUT,...Object.values(READOUT_COUNTS)]){
    assert.deepEqual(bannedWords(words.replaceAll('voxel','')),[]);assert.doesNotMatch(words,/outline|assigned|non-background/);
  }
  // The card's line says in its tooltip which of the two it reads: averaged on a counted frame, else the count proseg
  // saves (one draw: between counted iterations, and on one counted for a pause, the hold, a finish or the end).
  assert.equal(readoutTip(counted),"Proseg's count at this iteration, averaged over its random background draw");
  assert.equal(readoutTip(stats),READOUT_TIPS.saved);assert.equal(readoutTip(null),READOUT_TIPS.saved);
  assert.equal(READOUT_TIPS.saved,"Proseg's count at this iteration as proseg saves it: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.");
  assert.deepEqual(Object.keys(READOUT_TIPS),['kept','saved'],'no "draw" tip: no word says no averaged count was kept');
  for(const tip of Object.values(READOUT_TIPS)){assert.deepEqual(bannedWords(tip),[]);assert.doesNotMatch(tip,/outline/);}
});
/** A store against a fake engine: the statistics of `cells` cells and, for a counted frame, its counted file. */
function countedEngine({cells=2,counted=()=>true,kept=seq=>Float32Array.of(seq+0.25,seq+1.5)}={}){
  const original=globalThis.fetch,asked=[],gates=new Map();
  globalThis.fetch=async(url,{signal})=>{
    const [,route,seq]=/\/(cellstats|counted)\/(\d+)$/.exec(url);asked.push(`${route} ${seq}`);
    if(gates.has(url))await gates.get(url);
    if(signal?.aborted)throw Object.assign(Error('aborted'),{name:'AbortError'});
    if(route==='cellstats')return {ok:true,arrayBuffer:async()=>new ArrayBuffer(28*cells)};
    const values=kept(Number(seq));
    if(values==='network')throw new TypeError('Failed to fetch');   // the engine out of reach
    return values?{ok:true,arrayBuffer:async()=>values.buffer}:{ok:false,status:404};
  };
  return {asked,gates,restore:()=>{globalThis.fetch=original;}};
}
test("a counted frame's statistics come with proseg's count of each cell; a frame without one, or a count that can't be read, keeps the non-background count",async()=>{
  const counted=new Set([1,3,4]),engine=countedEngine({kept:seq=>seq===3?null:seq===4?Float32Array.of(1,2,3):Float32Array.of(seq+0.25,seq+1.5)});
  try {
    let changes=0;const store=new CellStats(()=>changes++,{counted:seq=>counted.has(seq)});store.reset('a');
    const one=await store.load(1);assert.deepEqual([...one.kept],[1.25,2.5]);assert.equal(keptOf(one,1),2.5);assert.equal(changes,1,'one change, with its count');
    assert.deepEqual(engine.asked.slice(0,2).sort(),['cellstats 1','counted 1'],'asked alongside');
    const two=await store.load(2);assert.equal(two.kept,null,'not counted: not asked');assert.equal(keptOf(two,0),0);
    assert.ok(!engine.asked.includes('counted 2'));
    const three=await store.load(3);assert.equal(three.kept,null);assert.equal(store.failed,null,'a count that is not there is no failure');
    const four=await store.load(4);assert.equal(four.kept,null,'one of another run\'s size is never read as this frame\'s');
    assert.equal(new CellStats().counted(5),false,'without the page\'s word nothing is counted');
  }finally{engine.restore();}
});
test("a frame counted for a pause, the hold, a finish or the end reads its own draw: no counted file is asked for, and the readout says as saved",async()=>{
  // Decision A: the engine scores such a frame on the count proseg saves there (f); the readouts read the same, the
  // frame's own draw (cells.zst foreground: Σf), never its counted file (Σe).
  const counted=new Set([3,4]),sampled=new Set([4]),engine=countedEngine();
  try {
    let changes=0;const store=new CellStats(()=>changes++,{counted:seq=>counted.has(seq),sampled:seq=>sampled.has(seq)});store.reset('a');
    const four=await store.load(4);
    assert.deepEqual(engine.asked,['cellstats 4'],'no counted file asked for');assert.equal(four.kept,null);
    four.foreground=Uint32Array.of(11,12);assert.equal(keptOf(four,1),12,'the frame\'s own draw');
    assert.equal(countName(four),"proseg's count (as saved)");assert.equal(readoutTip(four),READOUT_TIPS.saved);
    const three=await store.load(3);assert.deepEqual([...three.kept],[3.25,4.5],'an averaged counted frame: its counted file');
    assert.equal(countName(three),"proseg's count (averaged)");
    store.recount(4);store.recount();for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.ok(!engine.asked.includes('counted 4'),'never asked for, by recount either');
  }finally{engine.restore();}
});
test("a frame held with its counted file and learned since to be counted for the hold drops it: its readouts read the count proseg saves",async()=>{
  // A frame a token counted (averaged: its counted file came with its statistics) counted again for the hold, a finish
  // or the end (its record or the run's sampled list now say so): the engine scores it anew on f, so the readouts do.
  const counted=new Set([5,6]),sampled=new Set(),engine=countedEngine();
  try {
    let changes=0;const store=new CellStats(()=>changes++,{counted:seq=>counted.has(seq),sampled:seq=>sampled.has(seq)});store.reset('a');
    await store.prefetch(6);assert.deepEqual([...store.cache.get(6).kept],[6.25,7.5]);
    const before=await store.load(5);assert.deepEqual([...before.kept],[5.25,6.5]);assert.equal(changes,1);
    sampled.add(5);sampled.add(6);store.recount();
    assert.equal(store.current.kept,null,'the frame on screen: at once');assert.notEqual(store.current,before);assert.equal(changes,2);
    assert.deepEqual(store.current.assigned,before.assigned,'the same statistics');assert.equal(countName(store.current),"proseg's count (as saved)");
    assert.deepEqual([...store.cache.get(6).kept],[6.25,7.5],'another frame held: until it is shown');
    const six=await store.load(6);assert.equal(six.kept,null,'shown: its own draw');assert.equal(changes,3);
    assert.equal(engine.asked.filter(a=>a.startsWith('counted')).length,2,'nothing asked again');
  }finally{engine.restore();}
});
test("a counted file on its way when its frame is learned to be counted for the hold is not kept: the readouts read the count proseg saves",async()=>{
  // S-UIP-3: the file asked for alongside the statistics (frame 8) and the one recount asked for later (frame 9) both
  // land after the page learned their frames are sampled (a counted event, the run's sampled list): neither is kept.
  const counted=new Set([8]),sampled=new Set(),engine=countedEngine();let release8,release9;
  engine.gates.set('/api/v1/runs/a/counted/8',new Promise(r=>{release8=r;}));
  engine.gates.set('/api/v1/runs/a/counted/9',new Promise(r=>{release9=r;}));
  try {
    let changes=0;const store=new CellStats(()=>changes++,{counted:seq=>counted.has(seq),sampled:seq=>sampled.has(seq)});store.reset('a');
    const eight=store.load(8);for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.deepEqual(engine.asked.sort(),['cellstats 8','counted 8'],'asked alongside, while averaged');
    sampled.add(8);release8();
    const stats=await eight;assert.equal(stats.kept,null,'not kept');assert.equal(countName(store.current),"proseg's count (as saved)");
    await store.load(9);counted.add(9);store.recount(9);for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.ok(engine.asked.includes('counted 9'),'asked by recount, while averaged');
    sampled.add(9);release9();for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.equal(store.cache.get(9).kept,null,'not kept by recount either');assert.equal(store.current.kept,null);
    assert.equal(countName(store.current),"proseg's count (as saved)");
  }finally{engine.restore();}
});
test("a frame learned to be counted after its statistics came gets its count then, once; the frame on screen's readouts change with it",async()=>{
  const counted=new Set(),engine=countedEngine();
  try {
    let changes=0;const store=new CellStats(()=>changes++,{counted:seq=>counted.has(seq)});store.reset('a');
    const before=await store.load(7);assert.equal(before.kept,null);assert.equal(changes,1);
    store.recount();await new Promise(r=>setImmediate(r));assert.deepEqual(engine.asked,['cellstats 7'],'not counted yet: nothing asked');
    counted.add(7);store.recount();await new Promise(r=>setImmediate(r));await new Promise(r=>setImmediate(r));
    assert.deepEqual([...store.current.kept],[7.25,8.5]);assert.notEqual(store.current,before);assert.equal(changes,2);
    assert.deepEqual(store.current.assigned,before.assigned,'the same statistics, with the count');
    store.recount();await store.load(7);await new Promise(r=>setImmediate(r));
    assert.equal(engine.asked.filter(a=>a==='counted 7').length,1,'asked once');assert.equal(changes,2);
    // A frame not on screen gets its count into the cache, without a change on screen; one of a run let go meanwhile is
    // dropped.
    await store.prefetch(8);counted.add(8);store.recount(8);await new Promise(r=>setImmediate(r));await new Promise(r=>setImmediate(r));
    assert.equal(changes,2);assert.equal(store.current.seq,7);assert.deepEqual([...(await store.load(8)).kept],[8.25,9.5]);
    counted.add(9);await store.load(9);const gate=new Promise(r=>setTimeout(r,5));
    store.cache.set(10,{...store.current,seq:10,kept:null});counted.add(10);engine.gates.set('/api/v1/runs/a/counted/10',gate);
    store.recount(10);store.reset('b');await gate;await new Promise(r=>setImmediate(r));
    assert.equal(store.cache.size,0);assert.equal(store.current,null);
  }finally{engine.restore();}
});
test("a counted frame whose count can't be had (a 404, a network error) is asked for once, however often its statistics are loaded",async()=>{
  // Callers load the frame on screen several times per frame, and every run event recounts: one request in all.
  for(const answer of [null,'network']){
    const engine=countedEngine({kept:()=>answer});
    try {
      let changes=0;const store=new CellStats(()=>changes++,{counted:()=>true});store.reset('a');
      for(let i=0;i<3;i++)await store.load(6);
      store.recount();store.recount(6);await new Promise(r=>setImmediate(r));await new Promise(r=>setImmediate(r));
      assert.deepEqual(engine.asked.filter(a=>a.startsWith('counted')),['counted 6'],String(answer));
      assert.equal(store.current.kept,null);assert.equal(keptOf(store.current,1),0,'the non-background count');
      assert.equal(store.failed,null,'no failure of the statistics');assert.equal(changes,1);
    }finally{engine.restore();}
  }
});
test("a frame learned to be counted later gets its count only when it has the frame's cells, and another run (or the same one opened again) asks for its own",async()=>{
  // N18: a counted file of another size than the statistics (another run's, or a frame written again) is never read as
  // this frame's, by recount as by the request.
  const counted=new Set(),engine=countedEngine({kept:seq=>seq===5?Float32Array.of(1,2,3):Float32Array.of(seq+0.25,seq+1.5)});
  try {
    let changes=0;const store=new CellStats(()=>changes++,{counted:seq=>counted.has(seq)});store.reset('a');
    await store.load(5);counted.add(5);store.recount();
    for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.deepEqual(engine.asked.filter(x=>x.startsWith('counted')),['counted 5']);
    assert.equal(store.current.kept,null,'three cells against two: not this frame\'s');assert.equal(changes,1);
    // C06: run a's frame 7 was asked for; run b's frame 7 (or run a opened again) is another frame: asked for again.
    counted.add(7);const a=await store.load(7);assert.deepEqual([...a.kept],[7.25,8.5]);
    counted.clear();store.reset('b');
    const b=await store.load(7);assert.equal(b.kept,null,'run b: frame 7 not known to be counted yet');
    counted.add(7);store.recount(7);
    for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.equal(engine.asked.filter(x=>x==='counted 7').length,2,'run b\'s frame 7 asked for its count');
    assert.deepEqual([...store.current.kept],[7.25,8.5],'and shown');
  }finally{engine.restore();}
});
test("a frame learned to be counted while its statistics are on their way is left to their request until it ends, then asked for once",async()=>{
  // C02: their request asked for the count, or not, when it began; recount leaves the frame to it while it runs (the
  // page's redraw at their arrival loads the frame on screen again) and the next recount (a run event, a load) asks once.
  const counted=new Set(),engine=countedEngine();let release;
  engine.gates.set('/api/v1/runs/a/cellstats/7',new Promise(r=>{release=r;}));
  try {
    const asked=()=>engine.asked.filter(x=>x==='counted 7').length,during=[];let store=null;
    store=new CellStats(()=>{store.load(7);during.push(asked());},{counted:seq=>counted.has(seq)});store.reset('a');
    const shown=store.load(7);counted.add(7);store.recount(7);
    assert.equal(asked(),0,'its statistics are not here yet');
    release();const stats=await shown;assert.equal(stats.kept,null,'their request began before it was counted');
    assert.equal(during[0],0,'the redraw at their arrival asks nothing: their request has not ended');
    store.recount();for(let i=0;i<4;i++)await new Promise(r=>setImmediate(r));
    assert.equal(asked(),1);assert.deepEqual([...store.current.kept],[7.25,8.5],'and shown');
    store.recount();await store.load(7);assert.equal(asked(),1,'once');
  }finally{engine.restore();}
});
