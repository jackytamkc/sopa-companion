import test from 'node:test';
import assert from 'node:assert/strict';
import {decodeAnalysis,nearestEarlier,nearestLater,labelLookup} from '../src/analysis-helpers.mjs';
import {copy,number,unavailableTooltip,QC_TEXT} from '../src/qc-copy.mjs';
import {keptOf,countName,READOUT_COUNTS} from '../src/cellstats.mjs';
import {domFixture} from './dom-fixture.mjs';
test('analysis decodes odd and even cell counts without aligned float assumptions',()=>{
  for(const n of [0,1,2,4097]){
    const b=new ArrayBuffer(n*14),v=new DataView(b);
    for(let i=0;i<n;i++){v.setInt16(i*2,i%3-1,true);for(let k=0;k<3;k++)v.setFloat32(n*2+(i*3+k)*4,i===0?NaN:i+k/4,true);}
    const {clusters,coords}=decodeAnalysis(b);assert.equal(clusters.length,n);assert.equal(coords.length,n*3);
    if(n){assert.equal(clusters[0],-1);assert.ok(Number.isNaN(coords[0]));}
    if(n>1)assert.equal(coords[5],1.5);
  }
  assert.throws(()=>decodeAnalysis(new ArrayBuffer(13)),/Truncated/);
});
test('lookup reserves background and spans 4096-wide rows',()=>{
  const c=new Int16Array(4096).fill(7);c[4095]=-1;const data=labelLookup(c);
  assert.equal(data.length,8192);assert.equal(data[0],-1);assert.equal(data[4095],7);assert.equal(data[4096],-1);assert.equal(data[8191],-1);
});
test('historical fallback never uses future results, but the oldest after a frame before every one',()=>{
  assert.equal(nearestEarlier([8,1,5],6),5);assert.equal(nearestEarlier([8,1,5],5),5);
  assert.equal(nearestEarlier([8],6),-1);assert.equal(nearestEarlier([],0),-1);
  assert.equal(nearestLater([8,1,5],6),8);assert.equal(nearestLater([8,9,7],0),7);assert.equal(nearestLater([8,1,5],8),-1);
  assert.equal(nearestLater([],0),-1);assert.equal(nearestLater(new Map([[4,0],[2,0]]).keys(),1),2,'any iterable');
});

import vm from 'node:vm';
import {readFileSync} from 'node:fs';
// clusters.js through a data URL (no package.json here: some Node versions would treat .js as CommonJS).
const {createLegend,renderLegend,clusterColor,clusterColorCSS}=await import(`data:text/javascript;base64,${Buffer.from(readFileSync(new URL('../src/clusters.js',import.meta.url),'utf8')).toString('base64')}`);
const controllerSource=readFileSync(new URL('../src/analysis.js',import.meta.url),'utf8').replace(/^import .*;$/gm,'').replace(/^export /gm,'');
const {createPageLog}=await import('../src/log-export.mjs');
function controller(fetch,extra={}){
  const pageLog=createPageLog();
  const ctx=vm.createContext({fetch,AbortController,decodeAnalysis,nearestEarlier,nearestLater,copy,number,unavailableTooltip,keptOf,countName,console,setTimeout,clearTimeout,
    performance,clusterColor,createLegend,harness:null,pageLog,...extra});
  vm.runInContext(controllerSource+'\nthis.Analysis=Analysis;this.applyDelay=applyDelay;this.WITH_SNAPSHOT_MS=WITH_SNAPSHOT_MS;this.COLOURS_KEY=COLOURS_KEY;',ctx);
  const a=Object.create(ctx.Analysis.prototype);
  Object.assign(a,{run:'a',epoch:1,enabled:true,supported:true,results:new Map(),available:[],controls:{},empty:{},context:()=>({seq:10,frames:[]}),frame(){}});
  a.ctx=ctx;a.log=pageLog;return a;
}
const deferred=()=>{let resolve;const promise=new Promise(r=>resolve=r);return {promise,resolve};};
/** A result held of frame `seq` on `count` (3 cells), made before `result` below is (a test run alone by its name starts
 *  before the rest of this file is read). */
const onCount=(seq,count)=>({seq,clusters:new Int16Array(3),coords:new Float32Array(9),summary:{iter:seq*10,clusters:[],count}});
test('a newer analysis waits for the download in flight instead of aborting it (no starvation)',async()=>{
  const jobs=[];
  const a=controller((url,options)=>{const d=deferred();jobs.push({...d,url,signal:options.signal});return d.promise;});
  const first=a.fetchResult(1,{clusters:[]});
  a.fetchResult(2,{clusters:[]});a.fetchResult(3,{clusters:[]});      // arrive while 1 downloads: only the newest waits
  assert.equal(jobs.length,1);assert.equal(jobs[0].signal.aborted,false);
  jobs[0].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});await first;
  await new Promise(r=>setTimeout(r,0));
  assert.equal(jobs.length,2);assert.match(jobs[1].url,/\/3\/cells$/);
  jobs[1].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});await new Promise(r=>setTimeout(r,0));
  assert.deepEqual([...a.results.keys()].sort(),[1,3]);
});
test('a frame counted again for the hold, a finish or the end (another summary.count) drops the result held and fetches it again; the one on screen stays until it is in',async()=>{
  // A token counted frame 7 (its result on the averaged count); the hold counted it again (the engine's result now on the
  // count proseg saves): its ready event says so, and the page asks again; the clusters on screen stay meanwhile.
  const urls=[],gates=[];
  const a=controller(url=>{urls.push(url);const d=deferred();gates.push(d);return d.promise;});
  const answer=()=>gates.shift().resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});
  const settle=async()=>{for(let i=0;i<4;i++)await new Promise(r=>setTimeout(r,0));};
  a.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'averaged'}});answer();await settle();
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/7/cells']);assert.equal(a.results.get(7).summary.count,'averaged');
  const shown=a.applied=a.results.get(7);
  a.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'averaged'}});await settle();
  assert.equal(urls.length,1,'the same count again: nothing asked');
  a.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'sampled'}});await settle();
  assert.equal(urls.length,2,'another count: asked again');assert.equal(a.results.get(7),shown,'the one on screen stays meanwhile');
  assert.equal(shown.stale,true);
  answer();await settle();
  assert.notEqual(a.results.get(7),shown);assert.equal(a.results.get(7).summary.count,'sampled');assert.ok(!a.results.get(7).stale);
  // A result held but not on screen goes at once (asked for again when it is shown); an event without a summary says
  // nothing of its count.
  a.results.set(5,onCount(5,'averaged'));
  a.event({run:'a',state:'ready',seq:5,summary:{clusters:[],count:'sampled'}});await settle();
  assert.equal(a.results.has(5),false);assert.equal(urls.length,2,'not the frame shown: nothing asked now');
  a.event({run:'a',state:'ready',seq:7});await settle();
  assert.equal(a.results.get(7).summary.count,'sampled');assert.equal(urls.length,2);
});
test('a frame counted again while its result of the other count is on its way: what comes is never kept as the frame\'s, and its own is asked for at once when it is shown, else once it is',async()=>{
  // S-UIP-1: frame 7's averaged result is downloading when the hold counts frame 7 again (its event names the count
  // proseg saves there). The download that lands reads the averaged count: kept, it would stand for frame 7 for good.
  const urls=[],gates=[];
  const fetcher=url=>{urls.push(url);const d=deferred();gates.push(d);return d.promise;};
  const answer=(body={})=>gates.shift().resolve({ok:true,json:async()=>body,arrayBuffer:async()=>new ArrayBuffer(14)});
  const settle=async()=>{for(let i=0;i<6;i++)await new Promise(r=>setTimeout(r,0));};
  const still=c=>Object.assign(c,{present(){},badge(){},supported:false});   // updateFrame's own requests only
  const a=still(controller(fetcher));
  a.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'averaged'}});await settle();
  a.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'sampled'}});await settle();
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/7/cells'],'one download at a time');
  answer();await settle();
  assert.equal(a.results.has(7),false,'the averaged one that came is not kept');
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/7/cells','/api/v1/runs/a/analysis/7/cells'],'the frame shown: its own asked for at once');
  a.updateFrame();await settle();assert.equal(urls.length,2,'once');
  answer();await settle();
  assert.equal(a.results.get(7).summary.count,'sampled');assert.equal(a.results.get(7).summary.iter,70);
  // Not the frame shown when it is counted again (frame 3 looked back at shows frame 2's clusters): what comes is not
  // kept either, nothing is asked meanwhile, and its own is asked for once it is shown.
  urls.length=0;
  const b=still(controller(fetcher));b.available=[2];
  b.event({run:'a',state:'ready',seq:9,iter:90,summary:{clusters:[],count:'averaged'}});await settle();
  b.context=()=>({seq:3,frames:[]});
  b.event({run:'a',state:'ready',seq:9,iter:90,summary:{clusters:[],count:'sampled'}});await settle();
  answer();await settle();
  assert.equal(b.results.has(9),false,'not kept');assert.deepEqual(urls,['/api/v1/runs/a/analysis/9/cells'],'nothing asked meanwhile');
  b.context=()=>({seq:9,frames:[]});b.updateFrame();await settle();
  assert.deepEqual(urls.slice(1),['/api/v1/runs/a/analysis/9'],'shown: asked for, its summary first');
  answer({clusters:[],count:'sampled',iter:90});await settle();answer();await settle();
  assert.equal(b.results.get(9).summary.count,'sampled');
  // Shown again before it lands (back to frame 10): asked for again at once, on the count heard last.
  urls.length=0;
  const c=still(controller(fetcher));c.available=[2];
  c.event({run:'a',state:'ready',seq:9,iter:90,summary:{clusters:[],count:'averaged'}});await settle();
  c.context=()=>({seq:3,frames:[]});
  c.event({run:'a',state:'ready',seq:9,iter:90,summary:{clusters:[],count:'sampled'}});await settle();
  c.context=()=>({seq:10,frames:[]});
  answer();await settle();
  assert.equal(c.results.has(9),false);assert.deepEqual(urls,['/api/v1/runs/a/analysis/9/cells','/api/v1/runs/a/analysis/9/cells']);
  answer();await settle();assert.equal(c.results.get(9).summary.count,'sampled');
  // The same count heard again while on its way: what comes is kept.
  urls.length=0;
  const d=still(controller(fetcher));
  d.event({run:'a',state:'ready',seq:5,iter:50,summary:{clusters:[],count:'sampled'}});await settle();
  d.event({run:'a',state:'ready',seq:5,iter:50,summary:{clusters:[],count:'sampled'}});await settle();
  answer();await settle();
  assert.equal(d.results.get(5).summary.count,'sampled');assert.equal(urls.length,1);
  // Waiting its turn behind another frame's download when it is counted again, and no longer shown by then (frame 3
  // looked back at): asked for in its turn on the count named last, never labelled with the one before.
  urls.length=0;
  const e=still(controller(fetcher));e.available=[2];
  e.fetchResult(6,{clusters:[],count:'averaged'});
  e.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'averaged'}});
  e.context=()=>({seq:3,frames:[]});
  e.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'sampled'}});
  answer();await settle();
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/6/cells','/api/v1/runs/a/analysis/7/cells']);
  answer();await settle();
  assert.equal(e.results.get(7).summary.count,'sampled');
});
test('a result on screen of a count its frame no longer has (stale) is asked for again once its frame is shown',async()=>{
  // Following live, the hold counted frame 7 again while frame 9 was the one shown: back on frame 7, its clusters on
  // screen (stale: kept until their successor is in) are asked for again.
  const urls=[],gates=[];
  const a=controller(url=>{urls.push(url);const d=deferred();gates.push(d);return d.promise;});
  Object.assign(a,{present(){},badge(){},supported:false,available:[7,9],context:()=>({seq:7,frames:[]})});
  const shown=a.applied={...onCount(7,'averaged'),stale:true};a.results.set(7,shown);
  a.updateFrame();await new Promise(r=>setTimeout(r,0));
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/7'],'asked for again (its summary first)');assert.equal(a.results.get(7),shown,'shown meanwhile');
  // Counted again while another frame's result is on its way (looked back at frame 7, frame 9's downloading): asked for
  // once that one is in.
  urls.length=0;gates.length=0;
  const b=controller(url=>{urls.push(url);const d=deferred();gates.push(d);return d.promise;});
  Object.assign(b,{present(){},badge(){},supported:false,available:[7],context:()=>({seq:7,frames:[]})});
  const on=b.applied=onCount(7,'averaged');b.results.set(7,on);
  b.fetchResult(9,{clusters:[],count:'averaged'});
  b.event({run:'a',state:'ready',seq:7,iter:70,summary:{clusters:[],count:'sampled'}});
  assert.equal(on.stale,true);assert.deepEqual(urls,['/api/v1/runs/a/analysis/9/cells'],'one at a time');
  gates.shift().resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});for(let i=0;i<4;i++)await new Promise(r=>setTimeout(r,0));
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/9/cells','/api/v1/runs/a/analysis/7/cells'],'then frame 7\'s own, on the count heard');
  gates.shift().resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});for(let i=0;i<4;i++)await new Promise(r=>setTimeout(r,0));
  assert.equal(b.results.get(7).summary.count,'sampled');assert.ok(!b.results.get(7).stale);
});
test('a download aborted by another run never takes what the next one waits for',async()=>{
  // The first run's download lands after the next run's began and queued another frame: that frame is still asked for.
  const urls=[],gates=[];
  const a=controller(url=>{urls.push(url);const d=deferred();gates.push(d);return d.promise;});
  const settle=async()=>{for(let i=0;i<4;i++)await new Promise(r=>setTimeout(r,0));};
  a.fetchResult(1,{clusters:[]});
  a.epoch++;a.run='b';a.controller.abort();a.pending=a.wanted=null;a.results.clear();   // as reset() leaves it
  a.fetchResult(2,{clusters:[]});a.fetchResult(3,{clusters:[]});
  gates.shift().resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});await settle();
  assert.deepEqual(urls,['/api/v1/runs/a/analysis/1/cells','/api/v1/runs/b/analysis/2/cells']);
  gates.shift().resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});await settle();
  assert.deepEqual(urls.slice(2),['/api/v1/runs/b/analysis/3/cells'],'the next run\'s own queue');
  assert.deepEqual([...a.results.keys()],[2]);
});

test('run switch rejects old results and duplicate requests are coalesced',async()=>{
  const d=deferred();let calls=0;const a=controller(()=>{calls++;return d.promise;});
  const pending=a.fetchResult(1,{clusters:[]});await a.fetchResult(1,{clusters:[]});assert.equal(calls,1);
  a.epoch++;a.run='b';d.resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});await pending;
  assert.equal(a.results.size,0);
});
test('future ready event cannot replace the historical view, but stands for a frame looked back at before every analysed one',()=>{
  const a=controller(()=>{throw Error('unexpected fetch');});let requested=null;
  a.fetchResult=seq=>requested=seq;
  a.event({run:'other',state:'ready',seq:4});assert.equal(requested,null);
  a.available=[5];a.event({run:'a',state:'ready',seq:11});assert.equal(requested,null);
  a.event({run:'a',state:'ready',seq:8});assert.equal(requested,8);
  // Frame 10 looked back at with nothing analysed at or before it: the oldest after it (as the engine answers a frame
  // asked for: the nearest with proseg's count), never a later one than that; following live, none.
  const b=controller(()=>{throw Error('unexpected fetch');});requested=null;b.fetchResult=seq=>requested=seq;
  b.event({run:'a',state:'ready',seq:12});assert.equal(requested,12);
  b.event({run:'a',state:'ready',seq:11});assert.equal(requested,11);
  requested=null;b.event({run:'a',state:'ready',seq:14});assert.equal(requested,null,'not the oldest after it');
  b.event({run:'a',state:'ready',seq:9});assert.equal(requested,9,'one at or before it, once there is one');
  const live=controller(()=>{throw Error('unexpected fetch');});requested=null;live.fetchResult=seq=>requested=seq;
  live.context=()=>({seq:10,live:true,frames:[]});
  live.event({run:'a',state:'ready',seq:11});assert.equal(requested,null,'following live it waits for its snapshot');
});

import {pickVoxel} from '../src/analysis-helpers.mjs';
test('3-D picking respects depth, empty space and hidden clusters',()=>{
 const layers=[new Uint32Array([1,0]),new Uint32Array([2,0])];
 assert.equal(pickVoxel([.5,.5,4],[0,0,-1],[2,1,2],layers),2);
 assert.equal(pickVoxel([.5,.5,4],[0,0,-1],[2,1,2],layers,new Set([1]),new Int16Array([0,1])),1);
 assert.equal(pickVoxel([1.5,.5,4],[0,0,-1],[2,1,2],layers),0);
 assert.equal(pickVoxel([-1,.5,.5],[1,0,0],[2,1,2],layers),1);
 assert.equal(pickVoxel([-1,.5,.5],[-1,0,0],[2,1,2],layers),0);
});

test('hovering a cell highlights its whole cluster on the UMAP and in the legend',()=>{
  const a=controller(async()=>{throw Error('no fetch');});
  const calls=[],marks=[];
  a.applied={clusters:Int16Array.from([2,-1,5])};
  a.view={highlight:null,highlightCells(){throw Error('no cell highlight to clear');},highlightCluster:id=>calls.push(id)};
  a.legendView={highlight:id=>marks.push(id)};       // one legend row repainted per change, no scan of the legend
  a.tissueHover(1);a.tissueHover(1);                  // label 1 = cell 0 = cluster 2; repeated hover does not redraw
  assert.deepEqual(calls,[2]);assert.deepEqual(marks,[2]);
  a.tissueHover(2);                                   // a cell that was not analysed clears the highlight
  a.tissueHover(3);a.tissueHover(0);
  assert.deepEqual(calls,[2,null,5,null]);assert.deepEqual(marks,[2,null,5,null]);
});

// ---- coalesced application (whole-sample runs) -------------------------------------------------------------------
test('apply policy: live results of large runs wait for the 1 s window; small runs and seeks never wait',()=>{
  const {applyDelay}=controller(()=>{}).ctx;
  assert.equal(applyDelay(40000,true,300),700);assert.equal(applyDelay(40000,true,1000),0);assert.equal(applyDelay(40000,true,5000),0);
  assert.equal(applyDelay(20000,true,0),1000);assert.equal(applyDelay(19999,true,0),0);   // threshold: 20k cells
  assert.equal(applyDelay(40000,false,0),0);                                             // explicit scrub/seek/paused
});
const result=(seq,cells,summary={})=>({seq,clusters:new Int16Array(cells),coords:new Float32Array(cells*3),
  summary:{iter:seq*10,clusters:[{id:0,size:cells,color:0,label:'A · B'}],ms:{umap:12},...summary}});
/** Controller with fake stage, UMAP view, clock and timers (fully deterministic). */
function harness({live=true,seq=10}={}){
  const {document,Element}=domFixture();globalThis.document=document;
  Element.prototype.remove=function(){if(this.parentElement)this.parentElement.childNodes=this.parentElement.childNodes.filter(c=>c!==this);this.parentElement=null;};
  const timers=[],calls={setAnalysis:[],setHidden:0,setData:[],hide:0,select:[],cluster:[],stage:[]},views=[],state={seq,live};let now=0;
  class View {
    constructor(canvas,handlers){this.handlers=handlers;this.hover=null;this.highlight=null;views.push(this);}
    setData(data){calls.setData.push(data);} hideClusters(){calls.hide++;} highlightCells(){} highlightCluster(id){calls.cluster.push(id);}
    select(id){calls.select.push(id);} resize(){} dispose(){}
  }
  const a=controller(()=>{throw Error('no fetch');},{UmapView:View,performance:{now:()=>now},
    setTimeout:(fn,ms)=>{const t={fn,at:now+ms};timers.push(t);return t;},clearTimeout:t=>{const i=timers.indexOf(t);if(i>=0)timers.splice(i,1);}});
  delete a.frame;
  Object.assign(a,{hidden:new Set(),appliedAt:-Infinity,hoverLabel:0,hoverCluster:null,selectedLabel:0,shownNote:'',coloured:true,tip:{},panel:{render(){}},canvas:{},
    legend:document.createElement('div'),context:()=>({seq:state.seq,live:state.live,frames:[],cells:[]}),
    stage:{options:{},set(o){Object.assign(this.options,o);calls.stage.push(o);},setAnalysis(r){calls.setAnalysis.push(r);},clusterGPU:{setHidden(){calls.setHidden++;}}}});
  const advance=ms=>{const end=now+ms;for(;;){timers.sort((x,y)=>x.at-y.at);if(!timers.length||timers[0].at>end)break;const t=timers.shift();now=t.at;t.fn();}now=end;};
  const offer=r=>{a.results.set(r.seq,r);a.frame();};
  return {a,calls,views,timers,state,advance,offer};
}
test('live results of a large run are coalesced: at most one per second, always the newest, no backlog',()=>{
  const h=harness(),{a,calls}=h,seqs=list=>list.map(r=>r.seq);
  h.offer(result(1,40000));                                  // nothing shown yet: at once
  assert.equal(a.applied.seq,1);assert.deepEqual(seqs(calls.setData),[1]);assert.deepEqual(seqs(calls.setAnalysis),[1]);
  h.advance(300);h.offer(result(2,40000));                   // 0.3 s later: held back, one timer for the window's end
  assert.equal(a.applied.seq,1);assert.equal(h.timers.length,1);assert.equal(h.timers[0].at,1000);
  assert.match(a.logState(),/^on; showing the clusters of iteration 10 \(iteration \? is on screen\), 1 clusters;/);   // the log, not the page
  h.advance(300);h.offer(result(3,40000));a.frame();a.frame();   // a newer one replaces the waiting one; still one timer
  assert.equal(h.timers.length,1);assert.equal(a.applied.seq,1);
  h.advance(399);assert.equal(a.applied.seq,1);
  h.advance(1);                                              // the window has passed: the newest waits for a snapshot ...
  assert.equal(a.applied.seq,1);assert.equal(h.timers.length,1);assert.equal(h.timers[0].at,1000+a.ctx.WITH_SNAPSHOT_MS);
  h.advance(150);a.frame();                                  // ... and is applied in that snapshot's paint (main.js show())
  assert.equal(a.applied.seq,3);assert.deepEqual(seqs(calls.setData),[1,3]);assert.deepEqual(seqs(calls.setAnalysis),[1,3]);
  assert.equal(h.timers.length,0);
  a.frame();a.frame();h.advance(5000);                       // nothing new: no re-upload, no redraw of the UMAP
  assert.deepEqual(seqs(calls.setData),[1,3]);assert.deepEqual(seqs(calls.setAnalysis),[1,3]);
  h.offer(result(4,40000));                                  // > 1 s since the last one: at once
  assert.equal(a.applied.seq,4);assert.equal(h.timers.length,0);
});
test('an explicit seek applies at once and cancels the waiting live result; future results never colour it',()=>{
  const h=harness(),{a,calls}=h;
  h.offer(result(2,40000));h.advance(1000);h.offer(result(5,40000));h.advance(100);h.offer(result(6,40000));
  assert.equal(a.applied.seq,5);assert.equal(h.timers.length,1);
  h.state.live=false;h.state.seq=3;a.frame();                // scrub back to iteration 3 (within the window)
  assert.equal(a.applied.seq,2);assert.equal(a.applyTimer,null);   // (the 150 ms request for iteration 3 remains)
  h.offer(result(7,40000));assert.equal(a.applied.seq,2);    // newer than the frame on screen: never shown there
  h.state.seq=6;a.frame();assert.equal(a.applied.seq,6);     // stepping forward: at once as well
  h.state.live=true;h.state.seq=12;a.frame();                // back to live: newest, but within the window again
  assert.equal(a.applied.seq,6);h.advance(1000);assert.equal(a.applied.seq,6);
  h.advance(a.ctx.WITH_SNAPSHOT_MS);assert.equal(a.applied.seq,7);   // no snapshot came: applied by the fallback
  assert.deepEqual(calls.setData.map(r=>r.seq),[2,5,2,6,7]);
});
test('while following live, a held-back result is applied only with a snapshot, never in a frame of its own',()=>{
  // A whole sample: results downloaded every 600 ms, snapshots every 350 ms (burn-in) or 600 ms (sampling, each result
  // 300 ms after a snapshot), on independent clocks. main.js calls frame() for every snapshot it shows.
  for(const every of [350,600]){
    const h=harness(),{a}=h;let applied=0,inSnapshot=0,inFrame=false;
    const apply=a.apply.bind(a);a.apply=r=>{if(r&&r!==a.applied){applied++;if(inFrame)inSnapshot++;}apply(r);};
    for(let t=0;t<=12000;t+=10){
      if(t%600===300){a.results.set(t,result(t,40000));a.soon();}          // fetchResult's tail: a result is ready
      if(t%every===0){h.state.seq=t;inFrame=true;a.frame();inFrame=false;}  // a snapshot is shown
      h.advance(10);
    }
    assert.ok(applied>=10,`applied ${applied}`);
    assert.equal(inSnapshot,applied,`every ${every} ms: each result was applied in a snapshot's paint`);
  }
});
test('the tissue lookup is uploaded once per result: switching off twice clears it once, on again restores it',()=>{
  const h=harness(),{a,calls}=h,r=result(1,40000);
  h.offer(r);a.off();a.off();a.enabled=true;a.frame();a.frame();
  assert.deepEqual(calls.setAnalysis.map(x=>x?.seq??null),[1,null,1]);assert.equal(a.applied,r);
});
test('small runs apply every result at once',()=>{
  const h=harness();
  for(let seq=1;seq<=5;seq++){h.advance(100);h.offer(result(seq,5000));assert.equal(h.a.applied.seq,seq);}
  assert.equal(h.timers.length,0);assert.equal(h.calls.setData.length,5);
});
test('the view gets the worker\'s centroids and bounds when present and computes nothing itself',()=>{
  const h=harness(),{a,calls}=h,bounds={x:[-1.5,2],y:[0,3]};
  h.offer(result(1,100,{clusters:[{id:0,size:5,color:0,label:'A',centroid:[1,2,0]},{id:3,size:1,color:3,label:'B',centroid:null},
    {id:4,size:2,color:4,label:'C',centroid:[NaN,1,0]},{id:7,size:2,color:7,label:'D',centroid:[.5,-.5]}],bounds}));
  const data=calls.setData[0];
  assert.equal(JSON.stringify([...data.centroids]),'[[0,[1,2,0]],[7,[0.5,-0.5,0]]]');assert.equal(data.bounds,bounds);
  assert.deepEqual(data.palette(3),clusterColor(3));
  assert.equal(JSON.stringify(data.labels),'[{"id":0,"text":"A"},{"id":7,"text":"D"}]');   // only clusters with an anchor
  h.offer(result(2,100));                                    // an older worker sends neither: the view keeps computing them
  assert.equal('centroids' in calls.setData[1],false);assert.equal('bounds' in calls.setData[1],false);
  assert.equal(JSON.stringify(calls.setData[1].labels),'[{"id":0,"text":"A · B"}]');
  h.offer(result(3,100,{bounds:{x:[0,NaN],y:[0,1]}}));assert.equal('bounds' in calls.setData[2],false);
});
test('results update the legend in place and keep the tissue hover, UMAP emphasis and hidden clusters',()=>{
  const h=harness(),{a,calls,views}=h,cluster=(id,size)=>({id,size,color:id,label:`G${id}`});
  const r1=result(1,4,{clusters:[cluster(0,2),cluster(1,1)]});r1.clusters.set([0,1,1,-1]);
  h.offer(r1);
  const rows=()=>a.legend.childNodes[0].childNodes[1].childNodes,[row0,row1]=rows();   // the legend: heading, rows
  a.tissueHover(2);                                          // cell 1 is in cluster 1
  assert.equal(row1.style.background,'#ffffff1f');assert.deepEqual(calls.cluster.slice(-1),[1]);
  row0.click();                                              // hide cluster 0
  assert.deepEqual([...a.hidden],[0]);assert.equal(calls.hide,1);
  const emphasis=calls.cluster.length,setHidden=calls.setHidden;
  const r2=result(2,4,{clusters:[cluster(0,3),cluster(1,1)]});r2.clusters.set([0,1,1,-1]);
  h.advance(10);h.offer(r2);
  assert.equal(rows()[0],row0);assert.equal(rows()[1],row1);  // same nodes: no rebuild
  assert.equal(row0.childNodes[2].textContent,'3');assert.equal(row1.style.background,'#ffffff1f');assert.equal(row0.style.opacity,'.4');
  assert.equal(calls.cluster.length,emphasis);assert.equal(calls.hide,1);   // no extra emphasis passes on the UMAP
  assert.equal(calls.setHidden,setHidden+1);                 // the tissue lookup is new: its hidden clusters are restored
  const r3=result(3,4,{clusters:[cluster(0,3),cluster(2,1)]});r3.clusters.set([0,2,2,-1]);
  h.advance(10);h.offer(r3);                                 // cluster 1 is gone and cell 1 moved to cluster 2
  assert.equal(rows().length,2);assert.equal(rows()[0],row0);assert.equal(rows()[1].dataset.cluster,'2');
  assert.deepEqual(calls.cluster.slice(-1),[2]);assert.equal(rows()[1].style.background,'#ffffff1f');
  assert.equal(views.length,1);
});
test('selection and statistics refreshes only touch the UMAP and the stage on a real change',async()=>{
  const h=harness(),{a,calls}=h,loads=[];
  a.statsStore={load:seq=>{loads.push(seq);return Promise.resolve({seq,assigned:[7,8,9],foreground:[5,6,7]});}};
  h.offer(result(1,3));await new Promise(r=>setImmediate(r));
  a.frame();a.frame();await new Promise(r=>setImmediate(r));
  assert.deepEqual(loads,[10]);                              // once per frame on screen, not on every call
  h.state.seq=11;a.frame();assert.deepEqual(loads,[10,11]);
  const selects=calls.select.length;
  a.selection(3);a.selection(3);a.selection(3);assert.deepEqual(calls.select.slice(selects),[2]);
  a.selection(0);assert.deepEqual(calls.select.slice(selects),[2,null]);
  const stage=calls.stage.length;h.views[0].hover=1;await a.loadStats(11);
  // The hover names the cell's count in proseg's count in the frame on screen: its non-background count until the
  // frame's own count is in, then that one, whole (the store's current statistics, which take it when it comes later),
  // each by its name as the cell readout names it (H8: one draw "as saved", the counted file "averaged").
  assert.equal(calls.stage.length,stage);assert.equal(a.tip.textContent,"cell 1 · A · B · 6 in proseg's count (as saved)");
  a.statsStore.current={seq:11,assigned:[7,8,9],foreground:[5,6,7],kept:Float32Array.of(4.4,5.47,6.2)};a.describe(1);
  assert.equal(a.tip.textContent,"cell 1 · A · B · 5 in proseg's count (averaged)");
  a.statsStore.current={seq:9,assigned:[7,8,9],foreground:[5,6,7],kept:Float32Array.of(1,1,1)};a.describe(1);
  assert.equal(a.tip.textContent,"cell 1 · A · B · 6 in proseg's count (as saved)",'never another frame\'s');
  // Until they are in, "…" names no form of the count: which one comes is not known yet (an averaged counted frame's
  // file may still come), so never "(as saved)" for a number not there (S-UIP-5).
  a.stats=null;a.describe(1);assert.equal(a.tip.textContent,"cell 1 · A · B · … in proseg's count",'until they are in');
  a.stats={seq:11,assigned:[7],foreground:[5],kept:null};a.describe(1);
  assert.equal(a.tip.textContent,"cell 1 · A · B · … in proseg's count",'a cell the statistics do not have: no number, no form');
  a.stats=null;
  a.hover(1);a.hover(1);assert.equal(JSON.stringify(calls.stage.slice(stage)),'[{"hover":2}]');
});

// ---- legend (clusters.js) -----------------------------------------------------------------------------------------
test('legend rows update in place, keep hover and focus, and toggle or solo clusters',()=>{
  const {document,Element}=domFixture();globalThis.document=document;
  Element.prototype.remove=function(){if(this.parentElement)this.parentElement.childNodes=this.parentElement.childNodes.filter(c=>c!==this);this.parentElement=null;};
  const container=document.createElement('div'),hidden=new Set(),hovers=[],toggles=[];
  const legend=createLegend(container,{hidden,onHover:id=>hovers.push(id),onToggle:id=>toggles.push(id)});
  const summary=(iter,list)=>({iter,clusters:list.map(([id,size,label])=>({id,size,color:id,label}))});
  legend.update(summary(1,[[0,10,'A'],[1,20,'B'],[2,30,'C']]));
  const root=container.childNodes[0],list=root.childNodes[1],rows=()=>list.childNodes,[r0,r1,r2]=rows();
  assert.equal(root.childNodes[0].textContent,'Clusters (provisional, iteration 1)');
  assert.equal(list.style.minHeight,'84px');assert.match(list.style.cssText,/max-height:224px/);   // 3 rows of 28 px
  assert.equal(r1.dataset.cluster,'1');assert.equal(r1.childNodes[0].style.background,clusterColorCSS(1));assert.equal(r1.getAttribute('aria-pressed'),'true');
  r1.emit('pointerenter');assert.deepEqual(hovers,[1]);assert.equal(r1.style.background,'#ffffff12');
  legend.update(summary(2,[[0,11,'A'],[1,1234,'B2'],[2,30,'C']]));
  assert.ok(rows().every((row,i)=>row===[r0,r1,r2][i]));    // same nodes, hover kept
  assert.deepEqual(hovers,[1]);assert.equal(r1.style.background,'#ffffff12');
  assert.equal(r1.childNodes[1].textContent,'B2');assert.equal(r1.childNodes[2].textContent,(1234).toLocaleString());
  assert.equal(r1.title,'B2; click to hide/show, Alt/Shift-click to solo');assert.equal(root.childNodes[0].textContent,'Clusters (provisional, iteration 2)');
  legend.highlight(1);assert.equal(r1.style.background,'#ffffff1f');legend.highlight(null);assert.equal(r1.style.background,'#ffffff12');
  r1.emit('pointerleave');assert.deepEqual(hovers,[1,null]);assert.equal(r1.style.background,'transparent');
  r1.click();assert.deepEqual([...hidden],[1]);assert.deepEqual(toggles,[1]);assert.equal(r1.style.opacity,'.4');assert.equal(r1.getAttribute('aria-pressed'),'false');
  r0.onclick({altKey:true});assert.deepEqual([...hidden].sort(),[1,2]);assert.deepEqual(toggles,[1,2]);   // solo cluster 0
  r0.onclick({shiftKey:true});assert.deepEqual([...hidden],[]);                                             // solo again: all shown
  r2.focus();r0.emit('pointerenter');r2.emit('blur');      // a stale blur must not clear the hovered row
  assert.deepEqual(hovers.slice(-2),[2,0]);
  legend.update(summary(3,[[0,11,'A'],[2,30,'C'],[5,4,'E']]));  // cluster 1 gone, 5 new: kept rows survive
  assert.equal(rows().length,3);assert.equal(rows()[0],r0);assert.equal(rows()[1],r2);assert.equal(rows()[2].dataset.cluster,'5');
  assert.equal(r0.style.background,'#ffffff12');
  legend.highlight(5);assert.equal(rows()[2].style.background,'#ffffff1f');
  // Clusters come and go with nearly every result: the legend's height only grows, to 8 rows at most (more scroll inside
  // it), so nothing below it in Layers moves while a run plays.
  legend.update(summary(4,Array.from({length:12},(_,i)=>[i,i+1,`K${i}`])));assert.equal(list.style.minHeight,'224px');
  legend.update(summary(5,[[0,11,'A'],[2,30,'C']]));assert.equal(list.style.minHeight,'224px','never shrinks back');
  assert.equal(rows().length,2);assert.ok(rows().every(row=>row.style.cssText.includes('height:28px')));
  legend.dispose();assert.equal(container.childNodes.length,0);assert.deepEqual(hovers.slice(-1),[null]);
  const cleanup=renderLegend(container,summary(4,[[3,1,'D']]),{});assert.equal(container.childNodes[0].childNodes.length,2);
  cleanup();assert.equal(container.childNodes.length,0);
});

test('a result downloaded while following a live run waits for the next snapshot, or WITH_SNAPSHOT_MS without one',async()=>{
  const timers=[],state={live:true};
  const a=controller(async()=>({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)}),{
    setTimeout:(fn,ms)=>{const t={fn,ms};timers.push(t);return t;},clearTimeout:t=>{if(t)t.cleared=true;}});
  let applied=0;a.frame=()=>{applied++;};a.context=()=>({seq:10,live:state.live,frames:[]});
  await a.fetchResult(1,{clusters:[]});await a.fetchResult(2,{clusters:[]});
  assert.equal(applied,0,'not applied in a paint of its own');
  assert.equal(timers.length,1);assert.equal(timers[0].ms,a.ctx.WITH_SNAPSHOT_MS);
  timers[0].fn();assert.equal(applied,1,'no snapshot came: applied by the fallback');
  const snapshot={soonTimer:timers[0],updateFrame(){this.updated=true;}};
  a.ctx.Analysis.prototype.frame.call(snapshot);                   // main.js shows a snapshot: frame() applies it
  assert.equal(snapshot.updated,true);assert.equal(snapshot.soonTimer,null);assert.equal(timers[0].cleared,true);
  state.live=false;await a.fetchResult(3,{clusters:[]});
  assert.equal(applied,2);assert.equal(timers.length,1,'a paused or scrubbed view applies at once');
});

test('the analysis writes no status on the page: failures and what is shown when looking back go to the log; the empty pane says why',()=>{
  const h=harness(),{a}=h;
  Object.assign(a,{empty:{},controls:{hidden:true},collapsed:false});
  const frames=Array.from({length:20},(_,i)=>({seq:i,iter:10*i}));
  a.context=()=>({seq:h.state.seq,live:h.state.live,frames,cells:[]});
  a.hint();
  assert.equal(a.empty.textContent,'Clusters appear here once the first iteration is analysed.');assert.equal(a.empty.hidden,false);
  a.event({run:'a',state:'running',seq:12,iter:120});a.event({run:'a',state:'running',seq:13,iter:130});
  assert.deepEqual(a.log.lines(),[],'"Analysing iteration n" is neither shown nor logged: every result is');
  a.event({run:'a',state:'error',error:'GPU out of memory'});
  assert.equal(a.empty.textContent,'The clusters could not be computed. The Log button (top right) saves the details.');
  h.offer(result(1,3000,{nclusters:5,ncells:2990,ms:{load:6.2,counts:40,umap:106}}));
  assert.equal(a.empty.hidden,true,'clusters show: nothing to say');assert.equal(a.controls.hidden,false,'the legend shows');
  // Each result's step times are the engine's part of the log (its appendix): the page logs no line per result.
  assert.deepEqual(a.log.lines().map(l=>[l.kind,l.text]),[['clusters','Clusters & UMAP failed: GPU out of memory']]);
  h.state.live=false;h.state.seq=4;a.frame();a.frame();a.frame();   // paused on a later frame: which clusters it shows, once
  assert.deepEqual(a.log.lines().filter(l=>l.kind==='shown').map(l=>l.text),['iteration 40 shows the clusters of iteration 10']);
  h.state.live=true;h.state.seq=12;a.frame();                     // following live: always a little behind, nothing noted
  assert.equal(a.log.lines().filter(l=>l.kind==='shown').length,1);
  a.applied=null;a.collapsed=true;a.hint();assert.equal(a.empty.hidden,true,'a collapsed pane says nothing');
});

test("a run's kind of constants reaches the settings' tooltips: reset hands them the constants kept from before proseg's count, none without a run",async()=>{
  const a=controller(async()=>({ok:true,json:async()=>({available:true,enabled:true,seqs:[]})})),told=[];
  Object.assign(a,{pane:{style:{}},divider:{},canvas:{},collapse:{},split:{clientWidth:1000},width:360,legend:{replaceChildren(){}},
    note:{},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){},frame(){},
    settings:{reset(){told.push('reset');},setKept(kept){told.push(kept);},show(){},load(){}}});
  a.reset('whole',{kept:['min_counts','like_depth']});a.reset('region',{kept:[]});a.reset('old');a.reset(null,{kept:['min_counts']});
  assert.deepEqual(told,['reset',['min_counts','like_depth'],'reset',[],'reset',null,'reset',null]);
});

test('always on: a run opens the pane at once; only an engine that cannot run them, or a tool, closes it',async()=>{
  const replies=[{available:true,enabled:true,seqs:[3]},{available:false,enabled:true,seqs:[]},{available:true,enabled:false,seqs:[]}];
  const asked=[],a=controller(async(url,o)=>{asked.push([url,o.method]);return {ok:true,json:async()=>replies.shift()};});
  const calls=[],settle=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
  Object.assign(a,{pane:{style:{}},divider:{},canvas:{},collapse:{},split:{clientWidth:1000},width:360,legend:{replaceChildren(){}},
    note:{},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){calls.push('view');},frame(){calls.push('frame');}});
  a.reset('r1');
  assert.equal(a.enabled,true);assert.equal(a.pane.hidden,false,'the pane opens with the run: there is no switch');
  await settle();
  assert.deepEqual(a.available,[3]);assert.equal(a.supported,true);assert.deepEqual(calls,['view','frame']);
  assert.deepEqual(asked,[['/api/v1/runs/r1/analysis','GET']],'nothing is switched on: the engine analyses every run');
  a.reset('r2');await settle();
  assert.equal(a.enabled,false);assert.equal(a.pane.hidden,true);assert.equal(a.supported,false);
  assert.match(a.log.lines().at(-1).text,/not available: the analysis worker cannot run \(the engine log says why\)/);
  a.reset('r3');await settle();
  assert.equal(a.enabled,false);assert.match(a.log.lines().at(-1).text,/switched off for this run \(through the engine API\)/);
  a.event({run:'r3',enabled:true,state:'running',seq:0});          // a tool switched it on again
  assert.equal(a.enabled,true);assert.equal(a.pane.hidden,false);
  a.reset();assert.equal(a.enabled,false,'no run: no pane');assert.equal(a.pane.hidden,true);
  // The engine said at start it has no GPU worker environment (/health): a run opens without the pane, which never
  // flashes open and shut.
  a.engineCan=false;a.reset('r4');
  assert.equal(a.enabled,false);assert.equal(a.pane.hidden,true,'closed from the first paint');assert.equal(a.supported,false);
});

test('a run that cannot be clustered keeps the pane closed and says why in one calm line, until a result comes after all',async()=>{
  // A 60 x 60 µm rectangle holds ~20 cells: the engine says so (unavailable / reason too_few_cells) instead of a failure
  // in a 360 px pane for the whole run; an old run without exported transcripts likewise.
  const replies=[{available:true,enabled:true,seqs:[],unavailable:'too_few_cells'},{available:true,enabled:true,seqs:[2],unavailable:null},
    {available:true,enabled:true,seqs:[],unavailable:'no_transcripts'}];
  const asked=[],a=controller(async(url,o)=>{asked.push([url,o.method]);return {ok:true,json:async()=>replies.shift()};});
  const settle=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
  Object.assign(a,{pane:{style:{}},divider:{},canvas:{},collapse:{},split:{clientWidth:1000},width:360,legend:{replaceChildren(){}},
    note:{},empty:{},controls:{},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){},context:()=>({seq:5,live:false,frames:[]})});
  a.reset('small');await settle();
  assert.equal(a.blocked,'too_few_cells');assert.equal(a.pane.hidden,true);
  assert.equal(a.note.textContent,'Too few cells in this region to group them into clusters.');assert.equal(a.note.hidden,false);
  assert.equal(a.controls.hidden,false,'the line shows under Layers > Cells');assert.equal(a.empty.textContent,'','no failure in a pane');
  a.frame();await settle();
  assert.deepEqual(asked.map(([,m])=>m),['GET'],'nothing is asked of the engine for it');
  assert.match(a.logState(),/^cannot run: Too few cells/);assert.match(a.log.lines().at(-1).text,/cannot run for this run: too few cells/);
  a.event({run:'small',state:'error',seq:6,reason:'too_few_cells',error:'TooFewCells: only 20 cells'});
  assert.equal(a.problem,null,'a push with the reason is not a failure either');
  a.fetchResult=()=>{};a.event({run:'small',state:'ready',seq:7,summary:{clusters:[]}});   // cells formed: a result after all
  assert.equal(a.blocked,null);assert.equal(a.note.hidden,true);assert.equal(a.pane.hidden,false);
  a.reset('normal');await settle();assert.equal(a.blocked,null);assert.equal(a.pane.hidden,false);
  a.reset('old');await settle();
  assert.equal(a.note.textContent,'This run was made before clusters were computed live, so it has none.');assert.equal(a.pane.hidden,true);
});

test("a run without proseg's own count says why under Layers > Cells, why a new run has it in the line's tooltip; an old proseg-live says how to update it there",async()=>{
  // The engine's describe `unavailable`: no_counts (made before proseg-live kept proseg's own count), old_proseg (its
  // proseg-live can't keep it). Clusters come from proseg's count alone: no outline count stands in for it.
  const replies=[{available:true,enabled:true,seqs:[],unavailable:'no_counts'},{available:true,enabled:true,seqs:[],unavailable:'old_proseg'},
    {available:true,enabled:true,seqs:[],unavailable:'old_proseg'},{available:true,enabled:true,seqs:[],unavailable:'old_proseg'},
    {available:true,enabled:true,seqs:[],unavailable:'too_few_cells'}];
  const a=controller(async()=>({ok:true,json:async()=>replies.shift()}));
  const settle=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
  Object.assign(a,{pane:{style:{}},divider:{},canvas:{},collapse:{},split:{clientWidth:1000},width:360,legend:{replaceChildren(){}},
    note:{},empty:{},controls:{},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){},context:()=>({seq:5,live:false,frames:[]})});
  const UNAVAILABLE=vm.runInContext('UNAVAILABLE',a.ctx);
  a.reset('before');await settle();
  assert.equal(a.blocked,'no_counts');assert.equal(a.pane.hidden,true);
  assert.equal(a.note.textContent,"This run was made before proseg's own count was kept, so it has no clusters.");
  assert.equal(a.note.title,QC_TEXT.unavailableNoCountsTooltip);
  assert.equal(a.note.title,'This run has no count from proseg to score: it was made before proseg-live kept its count. A new run has one.');
  a.prosegFix='build';a.reset('stale');await settle();
  assert.equal(a.blocked,'old_proseg');assert.equal(a.note.textContent,'Update proseg-live to see clusters.');
  assert.equal(a.note.title,"This proseg-live can't keep proseg's own count. To update it, build it again with native/build.sh.");
  // A setting: in the file the session reads (a notebook's: the user's own, read at launch).
  a.prosegFix='setting';a.notebook=true;a.reset('stale3');await settle();a.notebook=false;
  assert.equal(a.note.title,"This proseg-live can't keep proseg's own count. To update it, point the binary under [proseg] in ~/.config/sopa-companion/config.toml at the new one and launch again.");
  a.prosegFix='a fix not known yet';a.reset('stale2');await settle();
  assert.equal(a.note.textContent,'Update proseg-live to see clusters.');assert.equal(a.note.title,'','no words for a fix the page does not know');
  a.prosegFix='install';a.reset('small');await settle();
  assert.equal(a.blocked,'too_few_cells');assert.equal(a.note.title,'','only an old proseg-live says how to update it');
  for(const reason of ['no_counts','old_proseg']){
    assert.deepEqual(bannedWords(UNAVAILABLE[reason]),[],reason);assert.doesNotMatch(UNAVAILABLE[reason],/outline|:/,reason);
  }
});

test('looking back, clusters of another iteration than the frame on screen say so in the pane\'s head; following live or on their own frame, nothing',()=>{
  const h=harness({live:false,seq:12}),{a}=h;a.snapped=document.createElement('span');a.snapped.hidden=true;
  h.offer(result(10,3));
  assert.equal(a.applied.seq,10);assert.equal(a.snapped.textContent,'At iteration 100');assert.equal(a.snapped.hidden,false);
  // The clusters' own iteration, never the check's (its results are a list of their own).
  assert.equal(a.snapped.title,"Clusters at iteration 100: the nearest with proseg's count.");
  h.state.seq=10;a.frame();assert.equal(a.snapped.hidden,true,'its own frame');assert.equal(a.snapped.textContent,'');assert.equal(a.snapped.title,'');
  h.state.seq=11;h.state.live=true;a.frame();assert.equal(a.snapped.hidden,true,'live: always a little behind the newest frame, nothing to say');
  h.state.live=false;a.frame();assert.equal(a.snapped.hidden,false);
  h.offer(result(130,3));h.state.seq=12;a.frame();assert.equal(a.snapped.textContent,'At iteration 100','never a later one');
  a.off();assert.equal(a.snapped.hidden,true,'switched off: none');
  assert.equal(vm.runInContext('UMAP_TIP',a.ctx),'{cell} · {cluster} · {n} in {count}');
});

test('looking back at a frame with an analysed one before it still on its way, a later result held is never shown meanwhile',()=>{
  // Frame 10 looked back at: frame 8 is analysed (the engine says so) but still downloading, frame 12's result is held.
  // The oldest after a frame stands only for a frame before every analysed one: here 8 is shown once it lands.
  const h=harness({live:false,seq:10}),{a,calls}=h,asked=[];a.fetchResult=seq=>asked.push(seq);
  a.available=[8,12];a.results.set(12,result(12,3));a.frame();
  assert.equal(a.applied??null,null,'not frame 12\'s');assert.deepEqual(calls.setData.map(r=>r.seq),[]);assert.deepEqual(asked,[8]);
  h.offer(result(8,3));assert.equal(a.applied.seq,8);assert.deepEqual(calls.setData.map(r=>r.seq),[8]);
  // Before every analysed frame (here 7), the oldest after it: 8.
  h.state.seq=7;a.frame();assert.equal(a.applied.seq,8);
  a.available=[12];a.results.delete(8);a.frame();assert.equal(a.applied.seq,12,'nothing before it: the oldest after it');
});

test('a run found unable to be clustered takes the badge of a frame looked back at with it, the pane staying for changed settings too',()=>{
  const h=harness({live:false,seq:12}),{a}=h;a.snapped=document.createElement('span');a.snapped.hidden=true;
  Object.assign(a,{pane:{style:{}},divider:{},collapse:{},split:{clientWidth:1000},width:360,note:{},
    settings:{changed:true,tooFew(){this.told=true;},show(){},done(){},reset(){},resultShown(){}}});
  h.offer(result(10,3));assert.equal(a.snapped.textContent,'At iteration 100');
  a.block('too_few_cells');
  assert.equal(a.pane.hidden,false,'changed settings: the pane stays');assert.equal(a.settings.told,true);
  assert.equal(a.snapped.hidden,true);assert.equal(a.snapped.textContent,'');assert.equal(a.snapped.title,'');
  a.block(null);h.offer(result(10,3));assert.equal(a.snapped.textContent,'At iteration 100','shown again with clusters');
  a.block('no_counts');assert.equal(a.pane.hidden,true);assert.equal(a.snapped.hidden,true);assert.equal(a.snapped.textContent,'');
});

test('another run opened goes without the last one\'s badge and the last line\'s tooltip',()=>{
  // N09: looking back at run a showed "At iteration 100" in the pane's head; run b opens before its first frame shows.
  const h=harness({live:false,seq:12}),{a}=h;a.snapped=document.createElement('span');a.snapped.hidden=true;
  Object.assign(a,{pane:{style:{}},divider:{},collapse:{},split:{clientWidth:1000},width:360,note:{},empty:{},controls:{},refresh(){}});
  h.offer(result(10,3));assert.equal(a.snapped.textContent,'At iteration 100');assert.notEqual(a.snapped.title,'');
  a.reset('b');
  assert.deepEqual([a.snapped.hidden,a.snapped.textContent,a.snapped.title],[true,'',''],'no badge of run a\'s clusters');
  // N11: run c could not be clustered (an old proseg-live: how to update it, in the line's tooltip); run d opens on an
  // engine where Clusters & UMAP cannot run at all: the line says where it is set, never with run c's tooltip.
  a.prosegFix='build';a.block('old_proseg');assert.equal(a.note.title,unavailableTooltip('old_proseg',{prosegFix:'build'}));
  a.engineCan=false;a.reset('d');
  assert.equal(a.note.textContent,vm.runInContext('SETUP',a.ctx).desktop);assert.equal(a.note.hidden,false);
  assert.equal(a.note.title,'','the line of where it is set has no tooltip');
});

test('the first clusters ever shown colour the cells by cluster; then the user\'s colours are kept, and Mixing never replaced',()=>{
  const store=new Map(),localStorage={getItem:k=>store.get(k)??null,setItem:(k,v)=>store.set(k,String(v))};
  const page=options=>{const saved=[],a=controller(()=>{},{localStorage});
    Object.assign(a,{coloured:false,stage:{options:{...options},set(o){Object.assign(this.options,o);}},panel:{render(){},save(){saved.push(1);}}});
    a.clusterColours();return {a,saved};};
  let {a,saved}=page({cellColorBy:'identity',transcriptColorBy:'cell'});
  assert.deepEqual(a.stage.options,{cellColorBy:'cluster',transcriptColorBy:'cluster'});assert.equal(saved.length,1,'the panel remembers it');
  assert.equal(store.get(a.ctx.COLOURS_KEY),'1');assert.equal(a.coloured,true);
  ({a,saved}=page({cellColorBy:'identity',transcriptColorBy:'cell'}));   // a later page: the user chose Identity meanwhile
  assert.deepEqual(a.stage.options,{cellColorBy:'identity',transcriptColorBy:'cell'});assert.equal(saved.length,0);
  store.clear();({a}=page({cellColorBy:'mixing',transcriptColorBy:'cell'}));
  assert.deepEqual(a.stage.options,{cellColorBy:'mixing',transcriptColorBy:'cluster'},'Mixing colours stay');
});

// ---- Advanced settings (analysis-settings.mjs) --------------------------------------------------------------------------
import {bannedWords} from '../src/qc-copy.mjs';
import {ClusterSettings,CLUSTER_SETTINGS_TEXT,SLOW_MS} from '../src/analysis-settings.mjs';
const SETTINGS={resolution:1,n_neighbors:15,min_dist:0.3,min_counts:10,n_top_genes:2000};
/** The engine's describe of a settings generation, as GET /analysis, a POST's reply or a settings push carry it. */
const described=(generation,settings={},extra={})=>({available:true,enabled:true,seqs:[],generation,settings:{...SETTINGS,...settings},
  defaults:{...SETTINGS},limits:{},changed:Object.keys(settings).length>0,...extra});
/** The disclosure as the controller sees it, recording what it is told. */
function settingsStub(changed=false){
  const calls=[],stub={changed,isBusy:false,loaded:true,calls,said:name=>calls.filter(c=>c[0]===name).length};
  for(const name of ['load','reset','show','open','done','failed','refused','tooFew','unchanged','resultShown'])stub[name]=(...args)=>calls.push([name,...args]);
  stub.logLine=()=>stub.changed?'resolution 2, neighbours 15, spacing 0.3, fewest transcripts 10, genes used 2000 (changed)':'';
  return stub;
}
const pane=()=>({pane:{style:{}},divider:{},canvas:{},collapse:{},split:{clientWidth:1000},width:360,note:{},controls:{}});
// What the constructor sets (the tests build the controller without it).
const fresh=()=>({gen:null,awaiting:null,missed:null,problem:null});
const plain=value=>JSON.parse(JSON.stringify(value));   // objects made in the controller's context have its prototypes

test('Apply posts the settings with the frame on screen; the clusters shown stay until the first result of the new ones, applied once',async()=>{
  const h=harness({live:false,seq:10}),{a,calls}=h,posted=[];
  Object.assign(a,fresh(),{settings:settingsStub(),gen:0});
  h.offer(Object.assign(result(8,400),{generation:0}));        // frame 10 on screen shows frame 8's clusters
  a.json=async(path,body)=>{posted.push([path,body]);return described(1,{resolution:2});};
  await a.applySettings({...SETTINGS,resolution:2});
  assert.deepEqual(plain(posted),[['',{params:{...SETTINGS,resolution:2},seq:10}]]);
  assert.equal(a.gen,1);assert.equal(a.awaiting,1);assert.equal(a.results.size,0);assert.equal(a.settings.said('load'),1);
  a.frame();
  assert.equal(a.applied.seq,8,'the clusters shown stay');assert.equal(a.settings.said('done'),0,'still busy');
  const flights=calls.setData.length,shown=a.settings.said('resultShown');
  h.offer(Object.assign(result(10,400),{generation:1}));
  assert.equal(a.applied.seq,10);assert.equal(calls.setData.length,flights+1,'one flight to the new layout');
  assert.equal(a.awaiting,null);assert.equal(a.settings.said('done'),1);assert.equal(a.settings.said('resultShown'),shown+1);
  assert.deepEqual(a.log.lines().filter(l=>l.kind==='clusters').map(l=>l.text),['Clusters & UMAP settings applied: the defaults']);
  a.frame();assert.equal(calls.setData.length,flights+1,'applied once');
  // Following live: no frame is named (the next automatic pass takes the settings at once); a reply of the same
  // generation says nothing changed.
  h.state.live=true;posted.length=0;
  a.json=async(path,body)=>{posted.push([path,body]);return described(1,{resolution:2});};
  await a.applySettings({...SETTINGS,resolution:2});
  assert.deepEqual(plain(posted),[['',{params:{...SETTINGS,resolution:2}}]]);assert.equal(a.settings.said('unchanged'),1);assert.equal(a.gen,1);
  // A run with no frame yet: the settings apply from its first clusters, so nothing is awaited.
  h.state.seq=-1;h.state.live=false;posted.length=0;
  a.json=async(path,body)=>{posted.push([path,body]);return described(2,{resolution:2.5});};
  await a.applySettings({...SETTINGS,resolution:2.5});
  assert.deepEqual(plain(posted),[['',{params:{...SETTINGS,resolution:2.5}}]]);assert.equal(a.settings.said('done'),2);
  // Refused: the settings' own line says so, and the log has the status.
  a.json=async()=>{const e=Error('Analysis: HTTP 422');e.status=422;throw e;};
  await a.applySettings({...SETTINGS,resolution:9});
  assert.equal(a.settings.said('refused'),1);assert.equal(a.log.lines().at(-1).text,'Clusters & UMAP settings were not taken: HTTP 422');
});

test('looking back past the clusters shown while new settings are awaited shows none of the earlier ones',()=>{
  const h=harness({live:false,seq:10}),{a}=h;
  Object.assign(a,fresh(),{settings:settingsStub(),gen:0});
  h.offer(Object.assign(result(8,400),{generation:0}));
  a.json=async()=>({});
  a.newGeneration(described(1,{resolution:2}));
  assert.equal(a.applied.seq,8,'frame 10 keeps them');
  h.state.seq=5;a.frame();
  assert.equal(a.applied,null,'frame 5 is earlier than the clusters shown: none until its own come');
});

test('results of earlier settings never show: an older push, a download in flight and an outdated 404 are dropped quietly',async()=>{
  const a=controller(()=>{throw Error('no fetch');});let fetched=null,refreshes=0;
  Object.assign(a,fresh(),{gen:2,available:[8],fetchResult:seq=>fetched=seq,refresh:()=>refreshes++});
  a.event({run:'a',state:'ready',seq:8,generation:1});a.event({run:'a',state:'error',error:'boom',generation:1});
  assert.equal(fetched,null);assert.equal(a.problem,null);
  a.event({run:'a',state:'ready',seq:8,generation:3});        // later settings than the page knows: the engine first
  assert.equal(refreshes,1);assert.equal(fetched,null);
  a.event({run:'a',state:'ready',seq:8,generation:2});assert.equal(fetched,8);
  // A download in flight when the settings change is aborted and dropped.
  const jobs=[],b=controller((url,options)=>{const d=deferred();jobs.push({...d,url,signal:options.signal});return d.promise;});
  Object.assign(b,fresh(),{gen:1,settings:settingsStub()});
  const pending=b.fetchResult(3,{clusters:[]});
  assert.equal(jobs[0].url,'/api/v1/runs/a/analysis/3/cells?generation=1');
  b.newGeneration(described(2,{resolution:2}));
  assert.equal(jobs[0].signal.aborted,true);assert.equal(b.pending,null);
  jobs[0].resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(14)});await pending;
  assert.equal(b.results.size,0);assert.equal(b.problem,null);
  // A 404 of a generation that is not the page's any more: dropped, no failure.
  const asked=[],d=deferred(),c=controller(url=>{asked.push(url);return d.promise;});
  Object.assign(c,fresh(),{gen:1});const missing=c.fetchResult(4);
  assert.deepEqual(asked,['/api/v1/runs/a/analysis/4?generation=1']);
  c.gen=2;d.resolve({ok:false,status:404});await missing;
  assert.equal(c.problem,null);
  // A 404 of the current one: the page asks the engine once (it may have restarted), then it is a failure.
  const e=controller(async()=>({ok:false,status:404}));let again=0;
  Object.assign(e,fresh(),{gen:2,refresh:()=>again++});
  await e.fetchResult(5);assert.equal(again,1);assert.equal(e.problem,null);
  await e.fetchResult(5);assert.equal(again,1);assert.match(e.problem,/HTTP 404/);
});

test("another window's change: the results go, the frame on screen is asked for and the fields follow, with no busy Apply",()=>{
  const h=harness({live:false,seq:10}),{a}=h,posted=[];
  Object.assign(a,fresh(),{settings:settingsStub(),gen:0});
  h.offer(Object.assign(result(10,400),{generation:0}));a.available=[10];
  a.json=async(path,body)=>{posted.push(body);return {};};
  a.event({run:'a',enabled:true,...described(1,{n_neighbors:30}),state:'settings'});
  assert.equal(a.gen,1);assert.equal(a.results.size,0);assert.equal(a.available.length,0);assert.equal(a.awaiting,1);
  assert.equal(a.settings.said('load'),1);assert.equal(a.applied.seq,10,'the clusters shown stay until the new ones');
  h.advance(150);assert.deepEqual(plain(posted),[{seq:10}],'the frame on screen is asked for');
  a.event({run:'a',enabled:true,...described(1,{n_neighbors:30}),state:'settings'});   // the same generation again
  assert.equal(a.settings.said('load'),2);assert.equal(a.awaiting,1);
  a.event({run:'a',state:'error',error:'GPU out of memory',generation:1});              // the new settings failed
  assert.equal(a.settings.said('failed'),1);assert.equal(a.applied.seq,10,'the clusters shown stay');
});

test('too few cells with changed settings keeps the pane and opens its settings; at the defaults the pane closes as before',()=>{
  for(const changed of [true,false]){
    const h=harness(),{a,calls}=h;
    Object.assign(a,pane(),fresh(),{settings:settingsStub(changed)});
    h.offer(result(1,400));
    a.event({run:'a',state:'error',seq:2,reason:'too_few_cells',error:'TooFewCells: only 20 cells'});
    assert.equal(a.blocked,'too_few_cells');assert.equal(a.pane.hidden,!changed,`changed ${changed}`);
    assert.equal(a.settings.said('tooFew'),changed?1:0);
    assert.equal(a.settings.calls.filter(c=>c[0]==='show').at(-1)[1],changed);
    assert.equal(calls.setData.at(-1).clusters.length,changed?0:400,'the dots fade out (with changed settings)');
    assert.equal(a.note.textContent,'Too few cells in this region to group them into clusters.');assert.equal(a.problem,null);
  }
});

/** The controller with the real disclosure (analysis-settings.mjs) on the harness's clock. */
function withSettings(h){
  const {a}=h;
  a.settings=new ClusterSettings({document:globalThis.document,timers:{setTimeout:a.ctx.setTimeout,clearTimeout:a.ctx.clearTimeout},
    onApply:values=>a.applySettings(values),onSlow:()=>a.log.add('clusters','Clusters & UMAP: no result for the new settings after 2 minutes')});
  return a.settings;
}

test('settings just applied that leave too few cells at the defaults (Reset from stricter ones) close the pane and free the buttons',async()=>{
  const h=harness({live:false,seq:10}),{a}=h;
  Object.assign(a,pane(),fresh(),{gen:1});
  const s=withSettings(h);
  s.load(described(1,{min_counts:100}));s.show(true);
  // A run that started with remembered settings too strict for its small region: the pane stays and says so.
  a.event({run:'a',state:'error',seq:3,reason:'too_few_cells',error:'TooFewCells',generation:1});
  assert.equal(a.pane.hidden,false);assert.equal(s.kind,'tooFew');
  a.json=async()=>described(2);
  await s.resetButton.click();
  assert.equal(a.gen,2);assert.equal(a.awaiting,2);assert.equal(a.blocked,null,'a change lifts the hold');assert.equal(s.isBusy,true);
  a.event({run:'a',state:'error',seq:10,reason:'too_few_cells',error:'TooFewCells',generation:2});   // the defaults leave too few too
  assert.equal(a.pane.hidden,true,'at the defaults the pane closes');
  assert.equal(s.isBusy,false,'nothing is left to wait for');
  assert.equal(s.apply.getAttribute('aria-busy'),null);assert.equal(s.apply.textContent,CLUSTER_SETTINGS_TEXT.apply);
  h.advance(SLOW_MS);
  assert.equal(s.kind,'');assert.ok(!a.log.lines().some(l=>/after 2 minutes/.test(l.text)),'nothing turns "slow"');
  // The region holds enough cells later: its map shows under free buttons, and nothing turns "slow" after it either.
  a.fetchResult=seq=>{a.results.set(seq,Object.assign(result(seq,40),{generation:2}));a.frame();};
  a.event({run:'a',state:'ready',seq:10,generation:2,summary:result(10,40).summary});
  assert.equal(a.applied?.seq,10);assert.equal(a.pane.hidden,false);assert.equal(s.isBusy,false);
  h.advance(SLOW_MS);assert.equal(s.line.hidden,true);
});

test('a POST reply that a later change overtook leaves the later settings in the disclosure, and an older push changes nothing',async()=>{
  const h=harness({live:false,seq:10}),{a}=h;
  Object.assign(a,pane(),fresh(),{gen:0});
  const s=withSettings(h);
  s.load(described(0));s.show(true);
  h.offer(Object.assign(result(10,40),{generation:0}));
  const reply=deferred();a.json=()=>reply.promise;          // this window's POST: its reply still on its way
  s.inputs.get('resolution').value='2';
  const pressed=s.apply.click();
  a.event({run:'a',enabled:true,...described(1,{resolution:2}),state:'settings'});     // this window's change, pushed
  a.event({run:'a',enabled:true,...described(2,{resolution:2.5}),state:'settings'});   // then another window's
  reply.resolve(described(1,{resolution:2}));await pressed;
  assert.equal(a.gen,2);assert.equal(s.inputs.get('resolution').value,'2.5');assert.equal(s.meta.settings.resolution,2.5);
  assert.match(s.logLine(),/^resolution 2\.5, /);
  assert.equal(s.isBusy,true,'busy until the first clusters of the settings now in use');
  a.newGeneration(described(1,{resolution:2}));             // a push of the earlier settings, late
  assert.equal(a.gen,2);assert.equal(s.meta.settings.resolution,2.5);assert.equal(s.inputs.get('resolution').value,'2.5');
  h.offer(Object.assign(result(10,40),{generation:2}));
  assert.equal(a.applied.generation,2);assert.equal(s.isBusy,false);
  assert.match(a.log.lines().at(-1).text,/^Clusters & UMAP settings applied: resolution 2\.5, /);
  // An Apply of the settings in use, whose reply (nothing changed, generation 2) comes after another window's change
  // (generation 3): the later settings stay, and nothing says these settings are already in use.
  const again=deferred();a.json=()=>again.promise;
  const unchanged=s.apply.click();
  a.event({run:'a',enabled:true,...described(3,{resolution:1.5}),state:'settings'});
  again.resolve(described(2,{resolution:2.5}));await unchanged;
  assert.equal(a.gen,3);assert.equal(s.inputs.get('resolution').value,'1.5');assert.equal(s.kind,'');
  assert.equal(s.isBusy,true,'busy until the first clusters of generation 3');
  h.offer(Object.assign(result(10,40),{generation:3}));assert.equal(s.isBusy,false);
});

test('refresh adopts the engine\'s generation, a lower one too (a restarted engine), and loads the settings',async()=>{
  const replies=[described(3,{resolution:2},{seqs:[4]}),described(0,{},{seqs:[2]})];
  const a=controller(async()=>({ok:true,json:async()=>replies.shift()}));
  Object.assign(a,pane(),fresh(),{legend:{replaceChildren(){}},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){},frame(){},settings:settingsStub()});
  await a.refresh();
  assert.equal(a.gen,3);assert.deepEqual(a.available,[4]);assert.equal(a.awaiting,null,'a page that opens adopts it');
  assert.equal(a.settings.said('load'),1);
  a.results.set(4,result(4,10));
  await a.refresh();
  assert.equal(a.gen,0);assert.equal(a.results.size,0);assert.deepEqual(a.available,[2]);assert.equal(a.awaiting,0);
});

test('the log says the settings when they differ from the defaults, and while the new ones are awaited',()=>{
  const h=harness(),{a}=h;
  Object.assign(a,fresh(),{settings:settingsStub(false)});h.offer(result(1,400));
  assert.doesNotMatch(a.logState(),/settings/);
  a.settings.changed=true;
  assert.match(a.logState(),/; settings: resolution 2, neighbours 15, spacing 0\.3, fewest transcripts 10, genes used 2000 \(changed\)$/);
  a.awaiting=3;assert.match(a.logState(),/; the new settings' clusters are not shown yet$/);
});

test('where Clusters and the UMAP cannot run the pane stays shut and Layers > Cells says where it is set, on the desktop and in a notebook',async()=>{
  const settle=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
  let SETUP=null;
  for(const notebook of [false,true]){
    const replies=[{available:false,enabled:true,seqs:[]},{available:true,enabled:true,seqs:[],unavailable:'too_few_cells'},
      {available:true,enabled:true,seqs:[]}];
    const a=controller(async()=>({ok:true,json:async()=>replies.shift()}));
    Object.assign(a,pane(),fresh(),{legend:{replaceChildren(){}},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){},frame(){},notebook});
    SETUP=vm.runInContext('SETUP',a.ctx);
    a.reset('r1');await settle();
    assert.equal(a.pane.hidden,true);assert.equal(a.supported,false);
    assert.equal(a.note.textContent,SETUP[notebook?'notebook':'desktop']);assert.equal(a.note.hidden,false);assert.equal(a.controls.hidden,false);
    await a.refresh();                                     // the engine back with its worker: a small region says so
    assert.equal(a.note.textContent,'Too few cells in this region to group them into clusters.');assert.equal(a.note.hidden,false);
    a.reset('r2');await settle();                         // an engine that can cluster: no line
    assert.equal(a.note.hidden,true);assert.equal(a.controls.hidden,true);
    a.engineCan=false;a.reset('r3');                       // said by /health at start: the line at once
    assert.equal(a.note.textContent,SETUP[notebook?'notebook':'desktop']);assert.equal(a.note.hidden,false);
    a.reset();assert.equal(a.note.hidden,true,'no run open: nothing to say');
  }
  for(const words of Object.values(SETUP))assert.deepEqual(bannedWords(words),[],words);
  assert.match(SETUP.desktop,/python under \[analysis\] in the app's config\.toml/);
  assert.match(SETUP.notebook,/python under \[analysis\] in ~\/\.config\/sopa-companion\/config\.toml and launch again/);
  assert.equal(SETUP.desktop,"Clusters and the UMAP can't run here: check python under [analysis] in the app's config.toml (the Log button, top right, says why).");
  assert.equal(SETUP.notebook,"Clusters and the UMAP can't run here: check python under [analysis] in ~/.config/sopa-companion/config.toml and launch again (the Log button, top right, says why).");
});

test('where the engine names the packages its Python lacks, Layers > Cells names them instead',async()=>{
  const settle=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
  for(const notebook of [false,true]){
    const replies=[{available:false,enabled:true,seqs:[],missing:['umap-learn','igraph']},{available:false,enabled:true,seqs:[],missing:[]}];
    const a=controller(async()=>({ok:true,json:async()=>replies.shift()}));
    Object.assign(a,pane(),fresh(),{legend:{replaceChildren(){}},hidden:new Set(),stage:{set(){},setAnalysis(){}},ensureView(){},frame(){},notebook});
    const [MISSING,missingLine,SETUP]=['MISSING','missingLine','SETUP'].map(name=>vm.runInContext(name,a.ctx));
    a.reset('r1');await settle();
    assert.equal(a.pane.hidden,true);assert.equal(a.supported,false);assert.equal(a.note.hidden,false);
    assert.equal(a.note.textContent,'Clusters and the UMAP need umap-learn and igraph, which this Python environment lacks.');
    assert.equal(a.logState(),'not available: this Python environment lacks umap-learn, igraph');
    assert.ok(a.log.lines().some(l=>l.text==='Clusters & UMAP are not available: this Python environment lacks umap-learn, igraph'));
    await a.refresh();                                     // no names: where the setting is
    assert.equal(a.note.textContent,SETUP[notebook?'notebook':'desktop']);
    assert.equal(a.logState(),'not available: the analysis worker cannot run');
    assert.equal(missingLine(['umap-learn']),'Clusters and the UMAP need umap-learn, which this Python environment lacks.');
    assert.equal(missingLine(['scipy','igraph','numba']),'Clusters and the UMAP need scipy, igraph and numba, which this Python environment lacks.');
    assert.equal(missingLine([]),null);assert.equal(missingLine(undefined),null);
    for(const words of [MISSING,missingLine(['scipy','scikit-learn','igraph','umap-learn','numba'])])assert.deepEqual(bannedWords(words),[],words);
  }
});

test('the engine finding mid-session that it cannot run Clusters and the UMAP: the page reads its describe again, once',()=>{
  const a=controller(()=>{throw Error('no fetch');});
  let refreshed=0;a.refresh=()=>{refreshed++;};a.fail=message=>{throw Error(`said as a failure: ${message}`);};
  a.event({run:'other',state:'unavailable'});assert.equal(refreshed,0);
  a.event({run:'a',state:'unavailable',enabled:true});assert.equal(refreshed,1);
  assert.equal(a.supported,true);assert.equal(a.log.lines().length,0,'nothing else: describe says why');
});

test('settings awaited when the run\'s analysis goes off are freed at once, not after the slow line',async()=>{
  // A tool switched the run's analysis off (or the engine found it cannot run) while new settings were awaited: no
  // result of them will come, so the buttons are free again and nothing is awaited.
  const h=harness({live:false,seq:10}),{a}=h;
  Object.assign(a,pane(),fresh(),{settings:settingsStub(),gen:0,enabled:true});
  a.json=async()=>described(1,{resolution:2});
  await a.applySettings({...SETTINGS,resolution:2});
  assert.equal(a.awaiting,1);assert.equal(a.settings.said('done'),0,'busy while the first result is awaited');
  a.event({run:'a',enabled:false,state:'idle'});
  assert.equal(a.enabled,false);assert.equal(a.awaiting,null);assert.equal(a.settings.said('done'),1);
  a.setEnabled(false);assert.equal(a.settings.said('done'),1,'freed once');
});

test('limits the engine gave before its worker said where it runs are read again with the run\'s first result, once',async()=>{
  // On the CPU Genes used stops at 5000: a describe from before the worker's hello (backend null) holds the GPU's limit,
  // so the first result (which comes after the hello) reads the run's describe again; a backend already said asks nothing.
  const settle=async()=>{for(let i=0;i<5;i++)await new Promise(r=>setImmediate(r));};
  const limits=max=>({n_top_genes:{min:50,max,step:50,whole:true}});
  for(const [first,again] of [[null,'cpu'],['gpu',null]]){
    const replies=[described(0,{},{backend:first,limits:limits(10000)}),described(0,{},{backend:again,limits:limits(5000)})];
    const asked=[],a=controller(async(url,o)=>{asked.push([url,o.method]);return {ok:true,json:async()=>replies.shift()};});
    Object.assign(a,pane(),fresh(),{settings:settingsStub(),legend:{replaceChildren(){}},hidden:new Set(),
      stage:{set(){},setAnalysis(){}},ensureView(){},frame(){},fetchResult(){}});
    a.reset('r1');await settle();
    assert.equal(asked.length,1);
    a.event({run:'r1',state:'running',seq:0,generation:0});await settle();
    assert.equal(asked.length,1,'nothing before a result');
    a.event({run:'r1',state:'ready',seq:0,generation:0,summary:{}});await settle();
    const loads=a.settings.calls.filter(c=>c[0]==='load').map(c=>c[1].limits.n_top_genes.max);
    if(first===null){
      assert.deepEqual(asked,[['/api/v1/runs/r1/analysis','GET'],['/api/v1/runs/r1/analysis','GET']]);
      assert.deepEqual(loads,[10000,5000],'the CPU worker\'s limit reaches the field');
    }else{
      assert.equal(asked.length,1,'the worker had said where it runs: nothing to read again');assert.deepEqual(loads,[10000]);
    }
    a.event({run:'r1',state:'ready',seq:1,generation:0,summary:{}});await settle();
    assert.equal(asked.length,first===null?2:1,'once');
  }
});
