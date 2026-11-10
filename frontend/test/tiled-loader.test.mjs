import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { TiledLoader, ViewRequests, viewPlanes } from '../src/tiled-loader.mjs';
import { LayerPanel } from '../src/layer-panel.js';
import { VECTORS_QUERY } from '../src/vectors-decode.mjs';
import { encodeVectors } from './vectors-fixture.mjs';

const VECTORS=encodeVectors({cells:[{id:'c-1',poly:[[10.5,20.25],[30,20],[20,40.75]]},{id:'c-2',poly:[[50,50],[60,50],[55,61]]}],nuclei:null},
  {source:{cells:'xenium_cell_boundaries'}});
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const settle=()=>new Promise(resolve=>setImmediate(resolve));

test('stain image plans shrink from a whole sample to a detailed crop on large zoom changes',()=>{
  const stage={canvas:{width:1000,height:1000},camera:{x:10000,y:10000,zoom:.05}};
  const loader=new TiledLoader(stage,{}, {status(){},channels:()=>[true]});
  loader.reset({id:'stain',kind:'stain',box:[0,0,20000,20000],image:{channels:[{name:'DAPI'}]}});
  const requests=[];
  loader.background=(kind,key)=>{
    const [box,px]=JSON.parse(key);requests.push({box,px});
    loader.image={box,px,sources:[{}]};
  };
  for(const zoom of [.05,.1,.2,1,10]){stage.camera.zoom=zoom;loader.imageView();}
  const image=loader.image,span=image.box[2]-image.box[0];
  assert.ok(span<=140,`100 µm viewport requested ${span} µm`);
  assert.ok(image.px/span>=stage.camera.zoom/2);
  assert.ok(requests.length>=4);
  const plan=loader.plan, count=requests.length;
  stage.camera.x+=5;loader.imageView();
  assert.equal(loader.plan,plan,'a small pan retains the margin');
  assert.equal(requests.length,count);
  stage.camera.zoom=.05;loader.imageView();
  assert.deepEqual(loader.plan.box,[0,0,20000,20000]);
  loader.reset();
});

test('each request kind has one latest operation, shares duplicates, and rejects late aborted responses',async()=>{
  const requests=new ViewRequests(),old=deferred(),fresh=deferred(),signals=[];
  const a=requests.run('frame','old',signal=>{signals.push(signal);return old.promise;});
  const rejected=assert.rejects(a,{name:'AbortError'});
  await settle();
  const b=requests.run('frame','new',signal=>{signals.push(signal);return fresh.promise;});
  assert.equal(b,requests.run('frame','new',()=>{throw Error('duplicate fetch');}));
  await settle();assert.equal(signals[0].aborted,true);assert.equal(signals[1].aborted,false);
  const image=requests.run('image','new',()=>42);assert.equal(await image,42);
  fresh.resolve(2);assert.equal(await b,2);old.resolve(1);await rejected;
  assert.equal(requests.jobs.size,0);
});

test('view planes obey the header ordering and reject malformed bytes',()=>{
  const meta={nx:2,ny:1,x0:3,y0:5,voxel:2,planes:['last_change','consensus']},buffer=new Uint32Array([10,20,1,2]).buffer;
  assert.deepEqual([...viewPlanes({meta,buffer},['consensus','last_change']).consensus],[1,2]);
  assert.throws(()=>viewPlanes({meta,buffer:buffer.slice(4)},['consensus','last_change']),/Truncated/);
});

function harness(t,{image=false,prior=false,shown=null}={}) {
  const calls=[],statuses=[],images=[],errors=[],posts=[];
  const run={id:'a',mode:'tiled',state:'paused',box:[0,0,1000,1000],info:{window_max:160,prior:prior?{format:'tiles',voxel:1}:null},
    image:image?{channels:[{name:'DAPI',max:1000,p_hi:900,hist:[0,1]},{name:'RNA',max:60000}]}:null};
  const stage={canvas:{width:100,height:80},camera:{x:100,y:100,zoom:1},ratio:1,get scale(){return this.camera.zoom;},
    setTranscripts(value){this.transcripts=value;}};
  const priors={setRaster(meta,labels,strength){this.raster={meta,labels,strength};}};
  const frame={seq:0,iter:10,x0:0,y0:0,nx:1000,ny:1000,nz:2,voxel:1,window:null};
  let transcriptStatus=200,delay=null;
  const json=value=>new Response(JSON.stringify(value),{headers:{'Content-Type':'application/json'}});
  const packed=(name,meta,data)=>new Response(data.buffer,{headers:{[name]:JSON.stringify(meta)}});
  t.mock.method(globalThis,'fetch',async(url,options)=>{
    const u=new URL(url,'http://test'),p=u.pathname;calls.push({url:u,options});
    if(delay){const held=delay(url,options);if(held)return held;}
    if(options.method==='POST'){posts.push(JSON.parse(options.body));return json({view:posts.at(-1).box});}
    if(p.endsWith('/xenium'))return u.search===`?${VECTORS_QUERY}`?new Response(VECTORS):new Response('{}',{status:400});
    if(p.includes('/transcripts/view'))return transcriptStatus===413?new Response(JSON.stringify({detail:'too many'}),{status:413}):new Response(new ArrayBuffer(16));
    if(p.includes('/image/view/'))return packed('X-Image',{x0:10,y0:20,x1:200,y1:180,width:2,height:2},new Uint16Array([0,100,200,500]));
    if(p.endsWith('/window')) {
      const w=run.window || frame.window;
      return packed('X-Window',{seq:frame.seq,iter:frame.iter,x0:w.i0,y0:w.j0,nx:w.nx,ny:w.ny,nz:2,voxel:1},new Uint32Array(w.nx*w.ny*2).fill(7));
    }
    const [x0,y0,x1,y1]=u.searchParams.get('box').split(',').map(Number),stride=Number(u.searchParams.get('stride'));
    const nx=Math.ceil((x1-x0)/stride),ny=Math.ceil((y1-y0)/stride),planes=p.includes('/prior/')?['label','permille']:['consensus','last_change'];
    return packed('X-View',{seq:Number(p.match(/frames\/(\d+)/)?.[1]||0),iter:frame.iter,x0,y0,nx,ny,voxel:stride,stride,planes},new Uint32Array(nx*ny*2).fill(1));
  });
  let refreshes=0;
  const loader=new TiledLoader(stage,priors,{refresh(){refreshes++;},status:s=>statuses.push(s),image:(...args)=>images.push(args),error:e=>errors.push(e),
    ...(shown?{channels:meta=>meta.channels.map((_,k)=>!!shown[k])}:{})});
  loader.reset(run);t.after(()=>loader.reset());
  return {loader,run,stage,priors,frame,calls,statuses,images,errors,posts,setDelay(fn){delay=fn;},setTranscriptStatus(v){transcriptStatus=v;},get refreshes(){return refreshes;},
    imageCalls:()=>calls.filter(c=>c.url.pathname.includes('/image/view/'))};
}

test('overview fetches cropped labels, image and prior, without full-frame or transcript requests',async t=>{
  const h=harness(t,{image:true,prior:true});h.stage.camera.zoom=.1;
  const data=await h.loader.frame(h.frame);h.stage.frame=data;h.loader.committed(h.frame,data);await settle();
  assert.equal(data.layers.length,0);assert.equal(h.stage.transcriptsHidden,true);assert.equal(h.priors.vectorsVisible,false);
  assert.equal(h.statuses.at(-1).text,'Zoom in to see transcripts');
  assert.equal(h.calls.some(c=>/transcripts|xenium|\/window$/.test(c.url.pathname)),false);
  assert.ok(h.calls.some(c=>c.url.pathname.endsWith('/frames/0/view')));
  assert.equal(h.images.length,1);assert.deepEqual(h.images[0][0],{x0:10,y0:20,w:190,h:160});
  assert.equal(h.images[0][1].length,2);assert.ok(h.images[0][1][0].data instanceof Uint16Array);
  assert.equal(h.images[0][2].channels[0].p_hi,900);assert.ok(h.priors.raster);
  const count=h.calls.length;await h.loader.frame(h.frame);await settle();assert.equal(h.calls.length,count);
  assert.deepEqual(h.errors,[]);
});

test('a finished run says its 3-D layers exist only where it was viewed',async t=>{
  const h=harness(t);h.run.state='done';const data=await h.loader.frame(h.frame);h.stage.frame=data;h.loader.committed(h.frame,data);await settle();
  assert.match(h.statuses.at(-1).text,/No 3-D layers saved for this area/);assert.equal(h.statuses.at(-1).active,false);
});

test('paused window updates the same consensus in place and old frames use only their own windows',async t=>{
  const h=harness(t),data=await h.loader.frame(h.frame);h.stage.frame=data;h.loader.committed(h.frame,data);await settle();
  assert.equal(data.layers.length,0);assert.match(h.statuses.at(-1).text,/3-D layers follow/);
  h.run.window={seq:0,iter:10,x0:0,y0:0,voxel:1,i0:20,j0:30,nx:160,ny:160,nz:2,file:'window-1.bin.zst'};
  assert.equal(h.loader.windowChanged(h.frame),true);
  const filled=await h.loader.frame(h.frame);h.stage.frame=filled;h.loader.committed(h.frame,filled);
  assert.equal(filled.consensus,data.consensus);assert.equal(filled.layers.length,2);assert.ok(filled.layers[0].every(x=>x===7));
  assert.equal(h.statuses.at(-1).layers,true);assert.equal(h.loader.windowChanged(h.frame),false);
  assert.equal(h.calls.filter(c=>c.url.pathname.endsWith('/frames/0/view')).length,1);
  const historic=await h.loader.frame({...h.frame,seq:1});assert.equal(historic.layers.length,0);
  assert.equal(h.calls.filter(c=>c.url.pathname.endsWith('/window')).length,1);
  h.stage.camera.x=500;const moved=await h.loader.frame(h.frame);assert.equal(moved.layers.length,0);
  assert.equal(h.calls.filter(c=>c.url.pathname.endsWith('/window')).length,1);
});

test('413 transcripts show a hint while keeping 2-D labels; a smaller detail box retries',async t=>{
  const h=harness(t);h.setTranscriptStatus(413);
  const data=await h.loader.frame(h.frame);await settle();
  assert.ok(data.consensus.length);assert.equal(h.stage.transcripts,null);assert.match(h.statuses.at(-1).text,/Too many transcripts/);
  const count=h.calls.filter(c=>c.url.pathname.includes('/transcripts/')).length;
  await h.loader.frame(h.frame);await settle();assert.equal(h.calls.filter(c=>c.url.pathname.includes('/transcripts/')).length,count);
  h.setTranscriptStatus(200);h.stage.camera.zoom=4;await h.loader.frame(h.frame);await settle();
  assert.equal(h.stage.transcripts.byteLength,16);assert.doesNotMatch(h.statuses.at(-1).text,/Too many/);
});

test('pan refresh and view POST debounce independently and repeated frames do not repeat the POST',async t=>{
  t.mock.timers.enable({apis:['setTimeout']});
  const h=harness(t);await h.loader.frame(h.frame);await settle();
  h.loader.move();t.mock.timers.tick(100);h.loader.move();t.mock.timers.tick(119);assert.equal(h.refreshes,0);
  t.mock.timers.tick(1);assert.equal(h.refreshes,1);
  t.mock.timers.tick(80);await settle();assert.equal(h.posts.length,1);
  await h.loader.frame({...h.frame,seq:1});t.mock.timers.tick(400);await settle();assert.equal(h.posts.length,1);
  h.stage.camera.x=600;await h.loader.frame(h.frame);t.mock.timers.tick(299);assert.equal(h.posts.length,1);
  t.mock.timers.tick(1);await settle();assert.equal(h.posts.length,2);
});

test('reset prevents stale rasters and transcripts from being installed in a different run',async t=>{
  const h=harness(t),old=deferred(),tx=deferred();
  h.setDelay(url=>url.includes('/frames/')?old.promise:url.includes('/transcripts/')?tx.promise:null);
  const loading=h.loader.frame(h.frame),rejected=assert.rejects(loading,{name:'AbortError'});
  await settle();h.loader.reset({id:'b',box:h.run.box,info:h.run.info});
  old.resolve(new Response(new Uint32Array(2).buffer,{headers:{'X-View':JSON.stringify({nx:1,ny:1,x0:0,y0:0,voxel:1,planes:['consensus','last_change']})}}));
  tx.resolve(new Response(new ArrayBuffer(16)));await rejected;await settle();
  assert.equal(h.loader.cache.entries.size,0);assert.equal(h.stage.transcripts,undefined);
});

test('returning to a cached crop cancels obsolete transcripts and restores its 413 hint',async t=>{
  const h=harness(t);h.setTranscriptStatus(413);await h.loader.frame(h.frame);await settle();
  const old=deferred();let signal;
  h.setDelay((url,options)=>{if(url.includes('/transcripts/')){signal=options.signal;return old.promise;}});
  h.stage.camera.x=600;await h.loader.frame(h.frame);await settle();assert.match(h.statuses.at(-1).text,/Loading transcripts/);
  h.stage.camera.x=100;await h.loader.frame(h.frame);await settle();
  assert.equal(signal.aborted,true);assert.match(h.statuses.at(-1).text,/Too many transcripts/);
  old.resolve(new Response(new ArrayBuffer(32)));await settle();assert.equal(h.stage.transcripts,null);
  h.stage.camera.zoom=.1;await h.loader.frame(h.frame);h.stage.camera.zoom=1;await h.loader.frame(h.frame);await settle();
  assert.match(h.statuses.at(-1).text,/Too many transcripts/);
});

test('only shown image channels are fetched; turning one on fetches it alone and keeps the loaded ones',async t=>{
  const shown=[true,false],h=harness(t,{image:true,shown});h.stage.camera.zoom=.1;
  const data=await h.loader.frame(h.frame);h.stage.frame=data;h.loader.committed(h.frame,data);await settle();
  assert.deepEqual(h.imageCalls().map(c=>c.url.pathname.split('/').at(-1)),['c0.u16'],'the hidden channel is not fetched');
  const [,sources]=h.images[0];assert.ok(sources[0].data instanceof Uint16Array);assert.equal(sources[1],null);
  shown[1]=true;h.loader.loadImage();await settle();await settle();
  const second=h.imageCalls().at(-1);
  assert.equal(h.imageCalls().length,2);assert.equal(second.url.pathname.split('/').at(-1),'c1.u16');
  assert.equal(second.url.searchParams.get('box'),h.imageCalls()[0].url.searchParams.get('box'),'the same box and size as the image on screen');
  assert.equal(second.url.searchParams.get('px'),h.imageCalls()[0].url.searchParams.get('px'));
  const [rect,both]=h.images[1];assert.equal(both[0],sources[0],'the loaded channel is kept (its texture too)');assert.ok(both[1].data);
  assert.deepEqual(rect,h.images[0][0]);
  h.loader.loadImage();await settle();assert.equal(h.imageCalls().length,2,'nothing is missing: no request');
  shown[0]=shown[1]=false;h.stage.camera.x=900;await h.loader.frame(h.frame);await settle();
  assert.equal(h.imageCalls().length,2,'every channel hidden: nothing fetched, even for a new view');
  assert.deepEqual(h.errors,[]);
});

test('a whole-sample run lists every image channel even when all are saved as hidden; turning one on fetches it',async t=>{
  // The page after openRun(): no channel listed yet, and the user's saved preference from an earlier session: every
  // channel hidden (turning DAPI off, then reloading after a page crash, is enough). Wired as main.js wires them: the
  // loader's channel and image callbacks, the panel's onChannelsChange, and main.js's own loadRunImage().
  const h=harness(t,{image:true});h.stage.camera.zoom=.1;
  const panel=Object.assign(Object.create(LayerPanel.prototype),{channels:[],selected:'cells',render(){},stage:{options:{}},
    priors:{settings:{prior:{},nuclei:{},xenium:{}}},saved:{channels:{DAPI:{on:false},RNA:{on:false}}}});
  h.loader.callbacks.channels=meta=>meta.channels.map((c,k)=>panel.channelOn(c,k));
  h.loader.callbacks.image=(rect,sources,meta)=>{h.images.push([rect,sources,meta]);panel.setChannels(sources.map((s,k)=>panel.channel(meta.channels[k],s,k)));};
  panel.onChannelsChange=()=>h.loader.loadImage();
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8'),context=vm.createContext({state:{run:h.run},layerPanel:panel,tiled:h.loader});
  vm.runInContext(main.slice(main.indexOf('function loadRunImage()'),main.indexOf('function renderChannels(')),context);
  context.loadRunImage();
  assert.deepEqual(panel.channels.map(c=>[c.name,c.on,c.source]),[['DAPI',false,null],['RNA',false,null]],'listed at once, as saved');
  for(let k=0;k<3;k++){const data=await h.loader.frame({...h.frame,seq:k});h.stage.frame=data;h.loader.committed(h.frame,data);await settle();}
  assert.equal(h.imageCalls().length,0,'nothing hidden is fetched');
  panel.channels[1].on=true;panel.save();panel.onChannelsChange();await settle();await settle();   // its eye button
  assert.deepEqual(h.imageCalls().map(c=>c.url.pathname.split('/').at(-1)),['c1.u16']);
  assert.deepEqual(panel.channels.map(c=>[c.name,c.on,!!c.source]),[['DAPI',false,false],['RNA',true,true]]);
  context.loadRunImage();                                    // a later run event: the listed channels stay
  assert.equal(panel.channels[1].on,true);assert.ok(panel.channels[1].source);assert.deepEqual(h.errors,[]);
});

for(const saved of [false,true])test(`stain initialization fetches every visible channel with ${saved?'restored':'default'} settings`,async t=>{
  const h=harness(t,{image:true});
  h.run.kind='stain';h.run.stain={nucleus:'DAPI',cytoplasm:saved?null:'RNA'};
  const panel=Object.assign(Object.create(LayerPanel.prototype),{channels:[],selected:'xenium',render(){},
    saved:{channels:saved?{DAPI:{on:true},RNA:{on:true}}:{}}});
  const state={run:h.run};
  h.stage.setImage=(rect,channels)=>h.images.push([rect,channels]);
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const context=vm.createContext({state,layerPanel:panel,stage:h.stage,priors:h.priors,
    refreshTiled(){},tiledStatus(){},notice:e=>h.errors.push(e),
    TiledLoader:class{constructor(stage,priors,callbacks){h.loader.callbacks=callbacks;return h.loader;}}});
  // The loader's block alone (main.js makes the tissue view right after it; the stain view comes after the removed-cells
  // panel it hands its outlines to).
  vm.runInContext(main.slice(main.indexOf('const tiled = new TiledLoader'),main.indexOf('const tissueView =')),context);
  vm.runInContext(main.slice(main.indexOf('function loadRunImage()'),main.indexOf('function renderChannels(')),context);
  for(let reload=0;reload<2;reload++) {
    h.loader.reset(h.run);panel.setChannels([]);
    const before=h.imageCalls().length;
    context.loadRunImage();await settle();await settle();
    assert.deepEqual(h.imageCalls().slice(before).map(c=>c.url.pathname.split('/').at(-1)),['c0.u16','c1.u16']);
    for(const channel of panel.channels) {
      assert.ok(channel.on,channel.name);
      assert.ok(channel.source?.data,`${channel.name} is visible with pixels`);
    }
    assert.equal(state.image.channels,panel.channels);
  }
  assert.deepEqual(h.errors,[]);
});

test('image requests are no larger than the device-pixel size of the box and respect the stage limits',async t=>{
  const h=harness(t,{image:true});Object.assign(h.stage,{canvas:{width:1500,height:900},camera:{x:500,y:500,zoom:1.4}});
  const data=await h.loader.frame(h.frame);h.stage.frame=data;h.loader.committed(h.frame,data);await settle();
  const call=h.imageCalls()[0],box=call.url.searchParams.get('box').split(',').map(Number),px=Number(call.url.searchParams.get('px'));
  assert.equal(px,Math.ceil(Math.max(box[2]-box[0],box[3]-box[1])*1.4));assert.ok(px<4096,`${px}`);
  const small=harness(t,{image:true});Object.assign(small.stage,{canvas:{width:4000,height:3000},camera:{x:500,y:500,zoom:4},limits:{texture:16384},imageLimit:2048});
  await small.loader.frame(small.frame);await settle();
  assert.equal(Number(small.imageCalls()[0].url.searchParams.get('px')),2048,'after a lost context the stage asks for smaller images');
});

test('snapshots at an unchanged view level ask the stage for no extra redraw',async t=>{
  const h=harness(t);let marks=0;
  Object.defineProperty(h.stage,'dirty',{set(v){if(v)marks++;},get(){return false;},configurable:true});
  h.stage.camera.zoom=.1;await h.loader.frame(h.frame);await settle();
  const after=marks;assert.ok(after>=1,'entering the overview hides transcripts and outlines: one redraw');
  for(let seq=1;seq<5;seq++){await h.loader.frame({...h.frame,seq});await settle();}
  assert.equal(marks,after,'each new snapshot is drawn once, by its upload');
  h.stage.camera.zoom=1;h.loader.notify();assert.equal(marks,after+1,'zooming into detail shows them again');
});

test('the detail level loads the whole run\'s prior vectors once, as typed-array PolygonSets',async t=>{
  const h=harness(t);let marks=0;
  Object.defineProperty(h.stage,'dirty',{set(v){if(v)marks++;},get(){return false;},configurable:true});
  await h.loader.frame(h.frame);await settle();await settle();
  const asked=()=>h.calls.filter(c=>c.url.pathname.endsWith('/xenium'));
  assert.equal(asked().length,1);assert.equal(asked()[0].url.search,`?${VECTORS_QUERY}`,'the binary, not the JSON');
  assert.equal(h.loader.vectorsLoaded,true);assert.ok(marks>=1,'installing them asks for a redraw');
  const {cells,nuclei,source}=h.priors.vectors;
  assert.equal(nuclei,null);assert.deepEqual(source,{cells:'xenium_cell_boundaries'});
  assert.equal(cells.count,2);assert.ok(cells.xy instanceof Float32Array);assert.deepEqual([...cells.offsets],[0,3,6]);
  assert.deepEqual([...cells.bbox.subarray(0,4)],[10.5,20,30,40.75]);
  h.stage.camera.x=600;await h.loader.frame({...h.frame,seq:1});await settle();
  assert.equal(asked().length,1,'fetched once per run, whatever the camera does');assert.deepEqual(h.errors,[]);
});

test('malformed prior vectors are reported once and not asked for again in that run (served immutable)',async t=>{
  const h=harness(t);
  h.setDelay(url=>url.includes('/xenium')?new Response(new ArrayBuffer(12)):null);
  await h.loader.frame(h.frame);await settle();await settle();
  assert.deepEqual(h.errors,['Malformed prior vectors: too short']);
  assert.equal(h.loader.vectorsLoaded,false);assert.equal(h.priors.vectors,undefined);
  const asked=()=>h.calls.filter(c=>c.url.pathname.endsWith('/xenium')).length;
  for(let seq=1;seq<6;seq++){await h.loader.frame({...h.frame,seq});await settle();await settle();}
  assert.equal(asked(),1,'every snapshot at the detail level used to fetch and decode it again');
  assert.equal(h.errors.length,1,'and to raise the notice again');
  h.loader.reset({...h.run,id:'b'});h.setDelay(()=>null);       // another run: asked for
  await h.loader.frame(h.frame);await settle();await settle();
  assert.equal(asked(),2);assert.equal(h.priors.vectors.cells.count,2);
});

test('a refused vectors URL is final, a busy engine is asked again after a wait that doubles',async t=>{
  const h=harness(t);let status=404,now=1000;
  t.mock.method(performance,'now',()=>now);
  h.setDelay(url=>url.includes('/xenium')?new Response(JSON.stringify({detail:`HTTP ${status}`}),{status}):null);
  const asked=()=>h.calls.filter(c=>c.url.pathname.endsWith('/xenium')).length;
  const view=async seq=>{await h.loader.frame({...h.frame,seq});await settle();await settle();};
  await view(0);now+=120000;await view(1);
  assert.equal(asked(),1,'a 404 (another format version, a deleted run) is not asked again');
  h.loader.reset({...h.run,id:'c'});status=503;
  await view(0);assert.equal(asked(),2);
  now+=1000;await view(1);assert.equal(asked(),2,'not before the wait (2 s)');
  now+=1001;await view(2);assert.equal(asked(),3);
  now+=2500;await view(3);assert.equal(asked(),3,'the next wait is 4 s');
  now+=1600;await view(4);assert.equal(asked(),4);
  status=200;h.setDelay(()=>null);now+=9000;await view(5);
  assert.equal(asked(),5);assert.equal(h.loader.vectorsLoaded,true);assert.equal(h.loader.vectorsFailed,null);
  assert.equal(h.errors.length,4,'one notice per failed attempt (the 404, then three 503s)');
});

test('the run-wide vectors of a run that was left are never installed in the next one',async t=>{
  const h=harness(t),late=deferred();
  h.setDelay(url=>url.includes('/xenium')?late.promise:null);
  await h.loader.frame(h.frame);await settle();
  assert.ok(h.loader.requests.jobs.has('vectors'),'run a\'s vectors are on their way');
  h.loader.reset({...h.run,id:'b'});
  late.resolve(new Response(VECTORS));await settle();await settle();
  assert.equal(h.priors.vectors,undefined,'run a\'s outlines are not shown in run b');
  assert.equal(h.loader.vectorsLoaded,false);assert.deepEqual(h.errors,[]);
});
