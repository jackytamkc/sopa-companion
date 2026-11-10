import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
const source = readFileSync(new URL('../src/main.js', import.meta.url), 'utf8');
const deferred = () => { let resolve, reject; const promise = new Promise((a,b) => { resolve=a; reject=b; }); return { promise, resolve, reject }; };
function harness() {
  const jobs = [], notices = [], marks = [], holds = [];
  const context = vm.createContext({
    // The panels batch (paint-batch.mjs): holds and marks are recorded, marks run nothing here.
    paint: { hold() { const token = { n: holds.length }; holds.push(['hold', token]); return token; }, release(token, now) { holds.push(['release', token, !!now]); } },
    later: new Proxy({}, { get: (_, key) => () => marks.push(key) }), counts: { snapshots: 0 },
    state: { run: { id: 'a' }, frames: [{ seq: 0, iter: 2000 }, { seq: 1, iter: 2005 }], current: 0, live: true, display: { layer: -1 } },
    stage: { frame: { iter: 2000 }, setFrame(frame) { this.frame = frame; } },
    analysis: { frame() {} },
    cellStats: { load() {}, prefetch() {} }, updateActiveCells() {}, noteCounted() {},   // the run's counted frames heard of (main.js)
    timeline: { setCurrent() {} },
    getFrame: () => Promise.resolve({ layers: [], consensus: new Uint32Array(1) }),
    updateHud() {}, updateFacts() {}, updateCellCard() {}, notice: text => notices.push(text),
    setTimeout: fn => { jobs.push(fn); return jobs.length; }, clearTimeout() {}, document: { hidden: false },
  });
  vm.runInContext('let generation=0, requested=-1, retryTimer, covered=false;\n' +
    source.slice(source.indexOf('function resetRequests()'), source.indexOf('function decode(')) +
    source.slice(source.indexOf('let showing = 0;'), source.indexOf('function setLive(on)')), context);
  return { context, jobs, notices, marks, holds, run: code => vm.runInContext(code, context) };
}
test('current and HUD stay on uploaded iteration until new frame succeeds', async () => {
  const h = harness(), pending = deferred();
  h.context.getFrame = () => pending.promise;
  const done = h.run('show(1)');
  assert.equal(h.context.state.current, 0);
  assert.equal(h.context.stage.frame.iter, 2000);
  pending.resolve({ layers: [] });
  await done;
  assert.equal(h.context.state.current, 1);
  assert.equal(h.context.stage.frame.iter, 2005);
});
test('failed upload preserves current and retries the requested frame', async () => {
  const h = harness();
  h.context.stage.setFrame = () => { throw new Error('GPU transient failure'); };
  assert.equal(await h.run('show(1)'), false);
  assert.equal(h.context.state.current, 0);
  assert.equal(h.jobs.length, 1);
  h.context.stage.setFrame = function(frame) { this.frame = frame; };
  h.jobs[0]();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(h.context.state.current, 1);
  assert.equal(h.context.stage.frame.iter, 2005);
});
test('old follow completion cannot suppress or overwrite the new run', async () => {
  const h = harness(), old = deferred(), fresh = deferred();
  h.context.getFrame = () => old.promise;
  const oldFollow = h.run('follow()');
  h.run("resetRequests(); state.run={id:'b'}; state.current=-1;");
  h.context.getFrame = () => fresh.promise;
  const newFollow = h.run('follow()');
  old.resolve({ layers: [] });
  await oldFollow;
  assert.equal(h.context.state.current, -1);
  assert.equal(h.run('followBusy'), true);
  fresh.resolve({ layers: [] });
  await newFollow;
  assert.equal(h.context.state.current, 1);
  assert.equal(h.run('followBusy'), false);
});

test('a follow loop stops when the page is hidden and catches up when it is shown again', async () => {
  const h = harness(), c = h.context, first = deferred(), fetched = [], uploaded = [];
  c.getFrame = seq => { fetched.push(seq); return seq === 1 ? first.promise : Promise.resolve({ layers: [], consensus: new Uint32Array(1) }); };
  c.stage.setFrame = frame => { c.stage.frame = frame; uploaded.push(frame.iter); };
  const following = h.run('follow()');                                   // fetching frame 1 ...
  c.state.frames.push({ seq: 2, iter: 2006 }, { seq: 3, iter: 2007 });    // ... while frames arrive
  c.document.hidden = true;                                              // and the window is minimised
  first.resolve({ layers: [], consensus: new Uint32Array(1) }); await following;
  assert.deepEqual(fetched, [1]); assert.deepEqual(uploaded, [2005]);    // the frame in flight lands; nothing after it
  assert.equal(h.run('followBusy'), false);
  await h.run('follow()'); assert.deepEqual(fetched, [1], 'a hidden page starts no follow either');
  c.document.hidden = false; await h.run('follow()');                    // visibilitychange: the newest frame
  assert.deepEqual(fetched, [1, 3]); assert.equal(c.state.current, 3);
  // Covered by Compare (main.js cover()): the same, and a retry waits for it too.
  c.state.frames.push({ seq: 4, iter: 2008 }); h.run('covered = true');
  await h.run('follow()'); h.run('requested = 4; retryRequested()'); assert.deepEqual(fetched, [1, 3], 'covered: nothing fetched');
  h.run('covered = false'); await h.run('follow()');                    // cover(false): the newest frame
  assert.deepEqual(fetched, [1, 3, 4]); assert.equal(c.state.current, 4);
});

const crop={nx:2,ny:1,x0:4000,y0:-120,voxel:4,iter:2005,consensus:new Uint32Array([1,2]),age:new Uint32Array([3,4]),layers:[]};
test('tiled display uses the returned geometry and consensus without dense prefetches',async()=>{
  const h=harness(),c=h.context;
  c.state.run.mode='tiled';c.state.live=false;c.state.display.layer=2;
  c.getFrame=()=>{throw Error('Dense request on a tiled run');};
  c.tiled={frame:async()=>crop,committed(){}};
  assert.equal(await h.run('show(1)'),true);
  assert.equal(c.stage.frame.x0,4000);assert.equal(c.stage.frame.nx,2);assert.equal(c.stage.frame.voxel,4);
  assert.equal(c.stage.frame.layer,-1);assert.equal(c.stage.frame.labels,crop.consensus);assert.equal(c.state.display.layer,2);
});
test('live tiled following skips frames that arrived while the newest view was loading',async()=>{
  const h=harness(),c=h.context,first=deferred(),seen=[];
  c.state.run.mode='tiled';c.state.current=-1;
  c.getFrame=()=>{throw Error('Dense request on a tiled run');};
  c.tiled={moving:false,committed(){},frame:header=>{seen.push(header.seq);return header.seq===1?first.promise:Promise.resolve({...crop,iter:header.iter});}};
  const following=h.run('follow()');
  c.state.frames.push({seq:2,iter:2006},{seq:3,iter:2007});await h.run('follow()');
  first.resolve(crop);await following;
  assert.deepEqual(seen,[1,3]);assert.equal(c.state.current,3);assert.equal(c.stage.frame.iter,2007);
});
test('a scrub supersedes a slow tiled view without clearing the previous raster or retrying aborts',async()=>{
  const h=harness(),c=h.context,old=deferred();
  c.state.run.mode='tiled';c.state.live=false;
  c.tiled={committed(){},frame:header=>header.seq===1?old.promise:Promise.resolve({...crop,iter:2000})};
  const pending=h.run('show(1)');assert.equal(c.stage.frame.iter,2000);
  await h.run('show(0)');old.reject(Object.assign(Error('superseded'),{name:'AbortError'}));await pending;
  assert.equal(c.state.current,0);assert.equal(c.stage.frame.iter,2000);assert.equal(h.jobs.length,0);assert.deepEqual(h.notices,[]);
});
test('paused run events refresh the displayed window without moving the timeline',async()=>{
  const h=harness(),c=h.context,window=deferred();
  c.state.run={id:'a',mode:'tiled',state:'paused'};c.state.live=false;
  c.zarrSave={update(){}};c.renderRun=()=>{};c.updateLeaving=()=>{};
  c.tiled={windowChanged:()=>true,frame:()=>window.promise,committed(){}};
  h.run('requested=state.current;');
  vm.runInContext(source.slice(source.indexOf('function onMessage(message)'),source.indexOf('// ---- run view')),c);
  h.run('onMessage({ev:"run",run:{id:"a",mode:"tiled",state:"paused",window:{seq:0}}})');
  assert.equal(c.stage.frame.iter,2000);assert.equal(c.state.current,0);
  const layers=[new Uint32Array([1,2])];window.resolve({...crop,iter:2000,layers});
  await new Promise(resolve=>setImmediate(resolve));
  assert.equal(c.stage.frame.layers,layers);assert.equal(c.state.current,0);assert.equal(c.stage.frame.iter,2000);
});
test('under Compare a paused run event fetches no window; nothing is uploaded until the view closes',async()=>{
  const h=harness(),c=h.context,asked=[];
  c.state.run={id:'a',mode:'tiled',state:'paused'};c.state.live=false;
  c.zarrSave={update(){}};c.renderRun=()=>{};c.updateLeaving=()=>{};
  c.tiled={windowChanged:()=>true,frame:header=>{asked.push(header.seq);return Promise.resolve({...crop,iter:2000,layers:[new Uint32Array([1,2])]});},committed(){}};
  h.run('requested=state.current; covered=true;');
  vm.runInContext(source.slice(source.indexOf('function onMessage(message)'),source.indexOf('// ---- run view')),c);
  h.run('onMessage({ev:"run",run:{id:"a",mode:"tiled",state:"paused",window:{seq:0}}})');
  await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(asked,[],'covered: no window asked for');assert.equal(c.stage.frame.layers,undefined);
  // A snapshot already on its way when Compare opened lands without an upload (show() returns false).
  h.run('covered=false');const pending=h.run('show(1)');h.run('covered=true');
  assert.equal(await pending,false);assert.equal(c.state.current,0);assert.equal(c.stage.frame.iter,2000);
});
test('Back from Compare shows a paused whole-sample run\'s window that changed underneath, once, and a refresh asked for meanwhile',async()=>{
  const h=harness(),c=h.context,asked=[],layers=[new Uint32Array([1,2])];
  c.state.run={id:'a',mode:'tiled',state:'paused'};c.state.live=false;c.state.mode='run';
  let changed=true;
  c.tiled={windowChanged:()=>changed,frame:header=>{asked.push(header.seq);return Promise.resolve({...crop,iter:2000,layers});},committed(){changed=false;}};
  const quiet={setCovered(){}};
  Object.assign(c,{testCorner:quiet,qcPanel:quiet,deeper:quiet,qcClient:{frame(){}},updateLeaving(){},zarrSave:{update(){}},renderRun(){}});
  Object.assign(c.stage,quiet);Object.assign(c.timeline,quiet);Object.assign(c.paint,quiet);Object.assign(c.analysis,quiet);
  h.run('let qcCovered=false, tiledMissed=false;');
  vm.runInContext(source.slice(source.indexOf('function cover(on, catchUp = true)'),source.indexOf('/** Compare over the run page')),c);
  vm.runInContext(source.slice(source.indexOf('function onMessage(message)'),source.indexOf('// ---- run view')),c);
  vm.runInContext(source.slice(source.indexOf('async function refreshTiled()'),source.indexOf('function tiledStatus(status)')),c);
  h.run('requested=state.current; cover(true)');
  h.run('onMessage({ev:"run",run:{id:"a",mode:"tiled",state:"paused",window:{seq:0}}})');await h.run('refreshTiled()');
  await new Promise(resolve=>setImmediate(resolve));assert.deepEqual(asked,[],'covered: nothing fetched');
  h.run('cover(false)');await new Promise(resolve=>setImmediate(resolve));await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(asked,[0],'shown once on Back');assert.equal(c.stage.frame.layers,layers);assert.equal(c.state.current,0);
  h.run('cover(true)');h.run('cover(false)');await new Promise(resolve=>setImmediate(resolve));
  assert.deepEqual(asked,[0],'nothing new: nothing asked again');
});

const settle = () => new Promise(resolve => setImmediate(resolve));
test('a snapshot holds the panels until its upload, loads its statistics alongside the frame and releases the hold', async () => {
  const h = harness(), c = h.context, frame = deferred(), stats = deferred(), log = [];
  c.cellStats = { prefetch: seq => { log.push(`stats ${seq}`); return stats.promise; }, load: seq => { log.push(`current ${seq}`); } };
  c.getFrame = seq => { log.push(`frame ${seq}`); return frame.promise; };
  c.stage.setFrame = f => { log.push(`upload ${f.iter}`); c.stage.frame = f; };
  const done = h.run('show(1)');
  assert.deepEqual(log, ['stats 1', 'frame 1'], 'the statistics are asked for before the frame arrives');
  assert.equal(h.holds.length, 1); assert.equal(h.holds[0][0], 'hold');
  frame.resolve({ layers: [], consensus: new Uint32Array(1) }); await settle();
  assert.deepEqual(log, ['stats 1', 'frame 1'], 'the frame waits (briefly) for its statistics: one paint for both');
  stats.resolve({ seq: 1 }); assert.equal(await done, true);
  // They become the statistics on screen with the frame, not before: the frame still shown keeps its readouts.
  assert.deepEqual(log.slice(2), ['upload 2005', 'current 1']);
  assert.deepEqual(h.holds.map(([kind]) => kind), ['hold', 'release']); assert.equal(h.holds[1][1], h.holds[0][1]);
  assert.equal(h.holds[1][2], true, 'released with the upload: the panels are written in this task');
  // The mixing check's result for the new frame is applied in the same batch (qc-client.mjs; never a later result).
  assert.deepEqual(h.marks, ['hud', 'now', 'qc']); assert.equal(c.counts.snapshots, 1);
});

test('slow statistics hold a frame back at most STATS_WAIT_MS; a failed snapshot releases its hold too', async () => {
  const h = harness(), c = h.context;
  c.cellStats = { prefetch: () => new Promise(() => {}), load() {} };      // never arrive
  const done = h.run('show(1)'); await settle();
  assert.equal(c.state.current, 0); assert.equal(h.jobs.length, 1, 'one timer: the statistics wait');
  h.jobs[0](); assert.equal(await done, true); assert.equal(c.state.current, 1);
  c.cellStats = { prefetch() {}, load() {} };
  c.stage.setFrame = () => { throw new Error('GPU transient failure'); };
  assert.equal(await h.run('show(0)'), false);
  assert.deepEqual(h.holds.map(([kind, , now]) => kind + (now ? ' now' : '')), ['hold', 'release now', 'hold', 'release'], 'a failed snapshot writes nothing early');
});
