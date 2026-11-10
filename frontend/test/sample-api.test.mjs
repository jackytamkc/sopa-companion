import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';

function client() {
  const requests=[],sent=[];
  const context=vm.createContext({window:{fetch() {},dispatchEvent(){}},fetch:async(url,options)=>{
    requests.push({url,...options});return options.method==='DELETE'?{ok:true,status:204}:{ok:true,json:async()=>({id:'a'})};
  },
  // This page's name and its showing (log-export.mjs pageSession, pageShowing), and its key's keepalive POST as the page
  // goes (page-key.mjs send).
  pageSession:{session:'page-1'},pageShowing:{visit:2},pageKey:{send:(url,body,type)=>{sent.push({url,body,type});return true;}}});
  const source=readFileSync(new URL('../src/api.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('export ','');
  const api=vm.runInContext(source+'\napi',context);
  return {api,requests,context,sent};
}
test('sample API encodes paths and shape keys, and forget accepts an empty 204 response',async()=>{
  const h=client();await h.api.browse('/sample data/');await h.api.openSample('/sample data/tissue.zarr');
  await h.api.shapes('sample a','cell shapes',[10,-20,50,80]);assert.equal(await h.api.forgetSample('sample a'),null);
  assert.equal(h.requests[0].url,'/api/v1/browse?path=%2Fsample%20data%2F');
  assert.deepEqual(JSON.parse(h.requests[1].body),{path:'/sample data/tissue.zarr'});
  assert.equal(h.requests[2].url,'/api/v1/samples/sample%20a/shapes/cell%20shapes?box=10,-20,50,80&limit=20000');
  assert.equal(h.requests[3].method,'DELETE');
});
test('run and zarr-save requests send the exact payload and preserve API detail errors',async()=>{
  const h=client(),region={kind:'polygon',points:[[10,-20],[50,-20],[50,80]]},prior={cells:null,nuclei:{kind:'shapes',key:'nuclei'}};
  await h.api.createRun('sample',region,prior,{'no-diffusion':true});
  assert.deepEqual(JSON.parse(h.requests[0].body),{sample:'sample',region,prior,options:{'no-diffusion':true}},'no schedule, no pause choice');
  // No links between runs: a new run never names a run it came from, whatever a caller passes.
  await h.api.createRun('sample',region,prior,{},'run a');
  assert.deepEqual(JSON.parse(h.requests.pop().body),{sample:'sample',region,prior,options:{}});
  const values={shapes_key:'cells',table_key:'counts',set_default:true,overwrite:false};
  await h.api.saveZarr('run a',values);assert.equal(h.requests[1].url,'/api/v1/runs/run%20a/zarr');assert.deepEqual(JSON.parse(h.requests[1].body),values);
  h.context.fetch=async()=>({ok:false,status:409,statusText:'Conflict',json:async()=>({detail:'Layer cells already exists'})});
  await assert.rejects(h.api.saveZarr('a',values),error=>error.status===409&&error.message==='Layer cells already exists');
});
test('the mixing check: a trial and its end go to their endpoints; no previous run is asked for; a refusal carries its status',async()=>{
  const h=client(),body={rule:'R2',area:'A',flag:'cell-compactness',from:0.08,to:0.04};
  // The previous run's line and readout are gone with GET /qc/previous.
  assert.equal(h.api.qcPrevious,undefined);
  await h.api.qcTrial('run a',body);
  await h.api.qcTrialEnd('run a','kept');await h.api.qcTrialEnd('run a','undone','U');
  assert.equal(h.requests[0].url,'/api/v1/runs/run%20a/qc/trial');assert.deepEqual(JSON.parse(h.requests[0].body),body);
  assert.equal(h.requests[1].url,'/api/v1/runs/run%20a/qc/trial/end');assert.deepEqual(JSON.parse(h.requests[1].body),{outcome:'kept'});
  assert.deepEqual(JSON.parse(h.requests[2].body),{outcome:'undone',rule:'U'});
  assert.ok(h.requests.every(r=>!/previous/.test(r.url)));
  h.context.fetch=async()=>({ok:false,status:409,statusText:'Conflict',json:async()=>({detail:'A trial is already running'})});
  await assert.rejects(h.api.qcTrial('a',body),error=>error.status===409);
});
test("the mixing check's per-cell values name its model and, for an iteration, the count its result reads; the start line's name none",async()=>{
  // A frame counted again for the hold, a finish or the end is scored anew on the count proseg saves: the URL names the
  // count (the browser keeps it as immutable), so values of the other count are never served for it.
  const h=client(),urls=[],controller=new AbortController();
  h.context.fetch=async(url,options)=>{urls.push([url,options.signal===controller.signal]);return {ok:true,arrayBuffer:async()=>Uint8Array.of(80,255,20).buffer};};
  assert.deepEqual([...await h.api.qcCells('run a',7,'m 1',controller.signal,'sampled')],[80,255,20]);
  await h.api.qcCells('r1',7,'m1',undefined,'averaged');await h.api.qcCells('r1',7,'m1');await h.api.qcCells('r1','start','m1',undefined,'sampled');
  assert.deepEqual(urls,[['/api/v1/runs/run%20a/qc/7/cells?model=m%201&count=sampled',true],['/api/v1/runs/r1/qc/7/cells?model=m1&count=averaged',false],
    ['/api/v1/runs/r1/qc/7/cells?model=m1',false],['/api/v1/runs/r1/qc/start/cells?model=m1',false]]);
  h.context.fetch=async()=>({ok:false,status:404});
  await assert.rejects(h.api.qcCells('r1',7,'m1',undefined,'averaged'),error=>error.status===404);
});

test('a save refused for a name clash keeps what the engine says besides its words (the names taken and in the sample)',async()=>{
  const h=client(),values={shapes_key:'cells',table_key:'counts',set_default:false,overwrite:false};
  const clash={error:'cells already exists in the sample',clash:['cells'],existing:['cells','table']};
  for(const answer of [clash,{detail:clash,status:409},{detail:'cells already exists in the sample',status:409,clash:['cells'],existing:['cells','table']}]){
    h.context.fetch=async()=>({ok:false,status:409,statusText:'Conflict',json:async()=>answer});
    await assert.rejects(h.api.saveZarr('a',values),error=>error.status===409&&error.message==='cells already exists in the sample'&&
      JSON.stringify(error.data.clash)==='["cells"]'&&JSON.stringify(error.data.existing)==='["cells","table"]');
  }
  // An answer that is not JSON keeps the status text, with no data.
  h.context.fetch=async()=>({ok:false,status:500,statusText:'Internal Server Error',json:async()=>{throw new SyntaxError('not JSON');}});
  await assert.rejects(h.api.sample('a'),error=>error.status===500&&error.message==='Internal Server Error'&&error.data===null);
});
test('a save refused for its names raises no general notice: the Save dialog says it, with Replace',async()=>{
  // The page-wide notice "Request failed (409). Please try again." came up over the dialog that said the clash itself.
  const notices=[],context=vm.createContext({CustomEvent:class{constructor(type,options){this.type=type;this.detail=options?.detail;}}});
  context.window=context;context.dispatchEvent=event=>notices.push(event.detail);    // window.fetch is the page's fetch
  let status=409;
  context.fetch=async()=>({ok:false,status,statusText:String(status),json:async()=>({detail:'cells already exists in the sample',status,clash:['cells'],existing:['cells']})});
  const api=vm.runInContext(readFileSync(new URL('../src/api.js',import.meta.url),'utf8').replace(/^import .*;\n/gm,'').replaceAll('export ','')+'\napi',context);
  const values={shapes_key:'cells',table_key:'counts',set_default:false,overwrite:false};
  for(status of [409,422])await assert.rejects(api.saveZarr('a',values),error=>error.status===status);
  assert.deepEqual(notices,[]);
  status=500;await assert.rejects(api.saveZarr('a',values));
  assert.deepEqual(notices,['Request failed (500). Please try again.'],'a save that failed otherwise still says so');
});
test('the page\'s first three requests take a signal, so the page can give up on them (the engine not answering)',async()=>{
  const h=client(),controller=new AbortController();
  await h.api.health(controller.signal);await h.api.samples(controller.signal);await h.api.runs(controller.signal);
  assert.deepEqual(h.requests.map(r=>[r.url,r.signal===controller.signal]),[['/api/v1/health',true],['/api/v1/samples',true],['/api/v1/runs',true]]);
  await h.api.runs();assert.equal(h.requests.at(-1).signal,undefined,'the runs poll asks without one');
});

// ---- leaving, closing and the runs from earlier versions ----------------------------------------------------------------
test('leaving and closing carry what they answer for: the run ids, a revision and the page; a session\'s Close ends it', async () => {
  const h = client();
  await h.api.leave('run a', 'discard', 1792756625251);
  assert.equal(h.requests[0].url, '/api/v1/runs/run%20a/leave');
  assert.deepEqual(JSON.parse(h.requests[0].body), { answer: 'discard', revision: 1792756625251, page: 'page-1' });
  await h.api.leave('run a', 'stay'); assert.deepEqual(JSON.parse(h.requests[1].body), { answer: 'stay', page: 'page-1' }, 'no revision: the answer and the page');
  await h.api.closing('add', { runs: ['r1'], revision: 7, end: true });
  assert.equal(h.requests[2].url, '/api/v1/closing');
  assert.deepEqual(JSON.parse(h.requests[2].body), { answer: 'add', runs: ['r1'], revision: 7, page: 'page-1', visit: 2, end: true },
    'from the showing that answers: one said before the page was shown again does not count');
  // As the page goes: its goodbye (bye), sent at once as a keepalive POST with the page's key, never through fetch (a
  // page going cannot wait for its key; a beacon cannot carry it).
  assert.equal(h.api.closing('discard', { beacon: true, runs: ['r1'], revision: 8 }), true);
  assert.equal(h.requests.length, 3);
  assert.deepEqual([h.sent[0].url, JSON.parse(h.sent[0].body)],
    ['/api/v1/closing', { answer: 'discard', runs: ['r1'], revision: 8, page: 'page-1', visit: 2, bye: true }]);
  // A goodbye never throws, whatever happens to it: a page that is not signed in (or has no key) sends none.
  h.context.pageKey.send = () => false;
  assert.equal(h.api.closing('discard', { beacon: true }), false);
  h.context.pageKey = null;
  assert.equal(h.api.closing('discard', { beacon: true }), false);
});
test('older runs: listed by GET /runs/older; one answered by its id, all (Later too) with one request', async () => {
  const h = client();
  await h.api.olderRuns(); await h.api.answerOlder('old a', 'add'); await h.api.answerOlder(null, 'discard'); await h.api.answerOlder(null, 'later');
  assert.deepEqual(h.requests.map(r => [r.method, r.url]), [['GET', '/api/v1/runs/older'], ['POST', '/api/v1/runs/older/old%20a'], ['POST', '/api/v1/runs/older'],
    ['POST', '/api/v1/runs/older']]);
  assert.deepEqual(JSON.parse(h.requests[1].body), { answer: 'add' }); assert.deepEqual(JSON.parse(h.requests[2].body), { answer: 'discard' });
  assert.deepEqual(JSON.parse(h.requests[3].body), { answer: 'later' });
});
test("Compare's points name the comparison and the layout they are for; a session's samples are its bound one", async () => {
  const h = client(), urls = [];
  h.context.fetch = async url => { urls.push(url); return { ok: true, arrayBuffer: async () => new ArrayBuffer(10), json: async () => ({ samples: [{ id: 'z1' }] }) }; };
  await h.api.comparePoints('r 1', 2, 'e6be5bd4', 3); await h.api.comparePoints('r1', 0);
  // Each its own: the column's own map (?own=1) and its own clusters, of the same comparison and layout.
  await h.api.comparePoints('r1', 1, 'e6be5bd4', 3, undefined, 'own'); await h.api.comparePoints('r1', 1, 'e6be5bd4', 3, undefined, 'clusters');
  assert.deepEqual(urls, ['/api/v1/runs/r%201/compare/points/2.bin?job=e6be5bd4&layout=3', '/api/v1/runs/r1/compare/points/0.bin',
    '/api/v1/runs/r1/compare/points/1.bin?job=e6be5bd4&layout=3&own=1', '/api/v1/runs/r1/compare/clusters/1.bin?job=e6be5bd4&layout=3']);
  assert.deepEqual(JSON.parse(JSON.stringify(await h.api.samples())), [{ id: 'z1' }], 'a session answers {samples}: the list');
  h.context.fetch = async () => ({ ok: false, status: 409, statusText: 'Conflict', json: async () => ({ detail: 'This comparison was replaced' }) });
  await assert.rejects(h.api.comparePoints('r1', 0, 'old', 1), error => error.status === 409);
});
