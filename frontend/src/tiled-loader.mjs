import { visibleBox, covers, intersect, viewLevel, planView, rasterBox, frameWindow, alignWindow, viewKey, imagePixels, needsImage, shownChannels, RasterLRU, IMAGE_MAX_PX } from './tiled-view.mjs';
import { harness } from './harness-hook.mjs';
import { decodeVectors, VECTORS_QUERY } from './vectors-decode.mjs';

const aborted=()=>Object.assign(new Error('View superseded'),{name:'AbortError'});
// proseg's grid starts at the run's first transcript, often inside the run's box: a window or a view raster can only
// cover the part of a box inside that grid (a frame record's whole raster).
const inGrid=(box,frame)=>box && frame?intersect(box,rasterBox(frame)):box;

// One cancellable request per kind; even a transport that completes after abort
// cannot publish its obsolete result. Identical requests share their promise.
export class ViewRequests {
  constructor() {this.jobs=new Map();}
  cancel(kind) {this.jobs.get(kind)?.controller.abort();this.jobs.delete(kind);}
  clear() {for(const kind of this.jobs.keys())this.cancel(kind);}
  run(kind,key,load) {
    const old=this.jobs.get(kind);
    if(old?.key===key)return old.promise;
    this.cancel(kind);
    const job={key,controller:new AbortController()};
    this.jobs.set(kind,job);
    job.promise=Promise.resolve().then(()=>{
      if(job.controller.signal.aborted)throw aborted();
      harness?.tiledRequest(kind);
      return load(job.controller.signal);
    }).then(value=>{
      if(job.controller.signal.aborted || this.jobs.get(kind)!==job)throw aborted();
      return value;
    }).finally(()=>{if(this.jobs.get(kind)===job)this.jobs.delete(kind);});
    return job.promise;
  }
}

async function response(url,signal,body) {
  const r=await fetch(url,{signal,credentials:'same-origin',method:body?'POST':'GET',
    headers:body?{'Content-Type':'application/json'}:{},body:body?JSON.stringify(body):undefined});
  if(!r.ok) {
    let detail=`View: HTTP ${r.status}`;
    try{detail=(await r.json()).detail||detail;}catch{/* optional detail */}
    throw Object.assign(new Error(detail),{status:r.status});
  }
  return r;
}

async function raster(url,header,signal) {
  const r=await response(url,signal),meta=JSON.parse(r.headers.get(header));
  if(!meta)throw new Error(`Missing ${header} geometry`);
  return {meta,buffer:await r.arrayBuffer()};
}

export function viewPlanes({meta,buffer},names) {
  const n=meta.nx*meta.ny;
  if(!Number.isInteger(n) || n<=0 || buffer.byteLength!==n*names.length*4 ||
    ![meta.x0,meta.y0,meta.voxel].every(Number.isFinite) || meta.voxel<=0)throw new Error('Truncated or invalid view raster');
  const all=new Uint32Array(buffer),out={};
  for(const name of names) {
    const k=meta.planes.indexOf(name);
    if(k<0 || k>=names.length)throw new Error(`Missing view plane: ${name}`);
    out[name]=all.subarray(k*n,(k+1)*n);
  }
  return out;
}

export class TiledLoader {
  /** channels(meta) -> [bool]: which image channels are shown (default: the first); only those are fetched. */
  constructor(stage,priors,{refresh,status,image,error,channels}) {
    this.stage=stage;this.priors=priors;this.callbacks={refresh,status,image,error,channels};
    this.requests=new ViewRequests();this.cache=new RasterLRU();this.reset();
  }
  reset(run=null,overview=false) {
    this.generation=(this.generation||0)+1;this.requests.clear();this.cache.clear();
    clearTimeout(this.moveTimer);clearTimeout(this.postTimer);
    this.run=run;this.overview=overview;this.header=null;this.plan=null;this.displayed=null;this.image=null;this.priorKey='';this.transcriptKey='';
    this.postKey='';this.postedKey='';this.vectorsLoaded=false;this.vectorsFailed=null;this.transcriptHint='';this.transcriptResultHint='';this.moving=false;
    this.stage.transcriptsHidden=false;this.priors.vectorsVisible=true;
    this.callbacks.status(null);
  }
  get base() {return `/api/v1/runs/${encodeURIComponent(this.run.id)}`;}
  visible() {return visibleBox(this.stage.camera,this.stage.canvas.width,this.stage.canvas.height);}
  notify() {
    if(!this.run)return;
    const voxel=this.header?.voxel || this.run.info?.burnin_voxel_size || 1;
    const limit=(this.run.info?.window_max || 1600)*voxel,visible=this.visible(),level=this.overview?'overview':viewLevel(visible,limit);
    const layers=level==='detail' && !!this.stage.frame?.layers?.length && covers(rasterBox(this.stage.frame),inGrid(intersect(visible,this.run.box),this.header));
    // Called for every snapshot fetched: a redraw only when what it shows changes (an identical extra frame per
    // snapshot is a frame for a remote display to read back and encode).
    const hidden=level!=='detail';
    if(this.stage.transcriptsHidden!==hidden || this.priors.vectorsVisible!==!hidden) {this.stage.transcriptsHidden=hidden;this.priors.vectorsVisible=!hidden;this.stage.dirty=true;}
    // A finished run's proseg has exited: its 3-D windows are only where the view was while it ran.
    const active=['running','paused','preparing'].includes(this.run.state);
    const text=level==='overview'?'Zoom in to see transcripts':[
      layers?'':active?'3-D layers follow the view…':'No 3-D layers saved for this area',this.transcriptHint,
    ].filter(Boolean).join(' · ');
    this.callbacks.status({level,limit,layers,text,active});
  }
  move() {
    if(!this.run)return;
    this.moving=true;this.notify();clearTimeout(this.moveTimer);
    this.moveTimer=setTimeout(()=>{this.moving=false;this.callbacks.refresh();},120);
  }
  prepare(header) {
    this.header=header;
    const s=this.stage,plan=planView(this.visible(),this.run.box,header.voxel,this.run.info?.window_max || 1600,1/s.scale,s.ratio,this.plan,this.overview);
    const key=viewKey(plan.level,plan.box,plan.stride);
    if(key!==this.plan?.key || header.voxel!==this.plan?.voxel) {
      this.requests.cancel('frame');this.requests.cancel('window');
      for(const kind of ['image','prior','transcripts'])this.requests.cancel(kind);
      this.plan={...plan,key};
    }
    this.notify();
    this.requestWindow();this.loadImage();this.loadPrior();this.loadTranscripts();
    if(plan.level==='detail')this.loadVectors();
    return this.plan;
  }
  query(plan=this.plan) {return `box=${plan.box.join(',')}&stride=${plan.stride}&max_px=${plan.maxPixels}`;}
  valid(generation,plan) {if(this.generation!==generation || this.plan!==plan)throw aborted();}
  async frame(header) {
    const plan=this.prepare(header),generation=this.generation;
    if(!plan.box)return null; // Outside the sample: retain the last raster at its own geometry.
    const key=viewKey(header.seq,plan.box,plan.stride),cached=this.cache.get(key);
    const window=frameWindow(this.run,header),need=inGrid(plan.box,header);
    const windowKey=plan.level==='detail' && window && covers(rasterBox(window),need) ? JSON.stringify([header.seq,window]) : '';
    if(cached)this.requests.cancel('frame');
    if(!windowKey || windowKey===cached?.windowKey)this.requests.cancel('window');
    const view=cached?Promise.resolve(cached):this.requests.run('frame',key,async signal=>{
      const result=await raster(`${this.base}/frames/${header.seq}/view?${this.query(plan)}`,'X-View',signal);
      const planes=viewPlanes(result,['consensus','last_change']);
      return {...result.meta,consensus:planes.consensus,age:planes.last_change,layers:[],bytes:result.buffer.byteLength};
    });
    let windowResult=null;
    if(windowKey && windowKey!==cached?.windowKey)windowResult=this.requests.run('window',windowKey,signal=>
      raster(`${this.base}/frames/${header.seq}/window`,'X-Window',signal)).catch(error=>{
        if(error.name==='AbortError')throw error;
        if(error.status!==404)this.callbacks.error(error.message);
        return null;
      });
    const [data,part]=await Promise.all([view,windowResult]);
    this.valid(generation,plan);
    let layers=windowKey && windowKey===cached?.windowKey?cached.layers:[];
    if(part && part.meta.seq===header.seq && covers(rasterBox(part.meta),need))layers=alignWindow(data,part.meta,part.buffer);
    const value={...data,layers,windowKey,viewKey:key,bytes:data.consensus.byteLength+data.age.byteLength+layers.reduce((n,l)=>n+l.byteLength,0)};
    this.cache.set(key,value);
    return value;
  }
  committed(header,data) {
    this.header=header;this.displayed=data;this.notify();
    // A paused-window event can arrive during the current frame's fetch.
    if(this.windowChanged(header))this.callbacks.refresh();
  }
  windowChanged(header) {
    if(this.plan?.level!=='detail' || !this.plan.box || !header)return false;
    const window=frameWindow(this.run,header);
    return window && covers(rasterBox(window),inGrid(this.plan.box,header)) && JSON.stringify([header.seq,window])!==this.displayed?.windowKey;
  }
  /** `apply` the result of a request kind, or report its error, unless a newer run (or, but for the run-wide vectors,
   *  a newer view) has come meanwhile; `failed(error)` first, when given. */
  background(kind,key,load,apply,failed) {
    const generation=this.generation,plan=this.plan;
    const current=()=>generation===this.generation && (kind==='vectors' || plan===this.plan);
    this.requests.run(kind,key,load).then(value=>{
      if(current())apply(value);
    }).catch(error=>{if(error.name!=='AbortError' && current()){failed?.(error);this.callbacks.error(error.message);}});
  }
  requestWindow() {
    const plan=this.plan,key=plan.level==='detail' && plan.box?JSON.stringify(plan.box):'';
    if(key===this.postKey)return;
    this.postKey=key;clearTimeout(this.postTimer);this.requests.cancel('post');
    if(!key || key===this.postedKey)return;
    this.postTimer=setTimeout(()=>{
      this.background('post',key,signal=>response(`${this.base}/view`,signal,{box:plan.box}),()=>{this.postedKey=key;});
    },300);
  }
  /** Image-only runs share the viewport image loader without a proseg frame or grid. */
  imageView() {
    const visible = this.run && intersect(this.visible(), this.run.box);
    if (!visible) { this.requests.cancel('image'); return; }
    const x = (visible[2]-visible[0])*.2, y = (visible[3]-visible[1])*.2;
    const box = intersect([visible[0]-x, visible[1]-y, visible[2]+x, visible[3]+y], this.run.box);
    const previous = this.plan?.box;
    if (!covers(previous, visible) || previous[2]-previous[0] >= 2*(box[2]-box[0]) ||
      previous[3]-previous[1] >= 2*(box[3]-box[1])) {
      this.requests.cancel('image');
      this.plan = { box };
    }
    this.loadImage();
  }
  /** Largest image to ask for: the engine's 6144, the GPU's texture size and the stage's limit after a lost context. */
  imageCap() {return Math.min(IMAGE_MAX_PX,this.stage.limits?.texture || IMAGE_MAX_PX,this.stage.imageLimit || Infinity);}
  /** Only the channels shown are fetched, each at most at the device-pixel size of the box. A channel turned on over
   *  the image on screen is fetched alone at that image's box and size; channels already loaded keep their data (and
   *  their textures). A new box (pan beyond the margin, a factor-two zoom) fetches the shown channels again. */
  loadImage() {
    const plan=this.plan,meta=this.run?.image,visible=this.run && intersect(this.visible(),this.run.box);
    if(!plan?.box || !visible || !meta?.channels?.length)return;
    const wanted=shownChannels(this.callbacks.channels?.(meta) || meta.channels.map((_,k)=>k===0));
    if(!wanted.length)return;   // every channel hidden: nothing to fetch, what is loaded stays
    const zoom=this.stage.camera.zoom,cap=this.imageCap(),loaded=this.image,fresh=needsImage(loaded,visible,zoom,plan.box,cap);
    const missing=fresh?wanted:wanted.filter(k=>!loaded.sources?.[k]);
    if(!missing.length)return;
    const box=fresh?plan.box:loaded.box,px=fresh?imagePixels(plan.box,zoom,cap):loaded.px,key=JSON.stringify([box,px,wanted]);
    if(this.requests.jobs.get('image')?.key===key)return;
    this.background('image',key,async signal=>{
      // Keep the old textures until every channel of the replacement is ready; the channels load in parallel.
      const parts=await Promise.all(missing.map(k=>raster(`${this.base}/image/view/c${k}.u16?box=${box.join(',')}&px=${px}`,'X-Image',signal)));
      let rect=fresh?null:loaded.rect;
      const sources=meta.channels.map((_,k)=>fresh?null:loaded.sources?.[k] || null);
      parts.forEach(({meta:m,buffer},i)=>{
        if(buffer.byteLength!==m.width*m.height*2)throw new Error('Truncated image view');
        const r=[m.x0,m.y0,m.x1,m.y1];
        if(rect && JSON.stringify(r)!==JSON.stringify(rect))throw new Error('Image channels have different geometry');
        rect=r;sources[missing[i]]={data:new Uint16Array(buffer),width:m.width,height:m.height};
      });
      return {rect,sources};
    },({rect,sources})=>{
      this.callbacks.image({x0:rect[0],y0:rect[1],w:rect[2]-rect[0],h:rect[3]-rect[1]},sources,meta);
      this.image={box,px,rect,sources};
    });
  }
  loadPrior() {
    const plan=this.plan,prior=this.run.info?.prior;
    if(!plan?.box || prior?.format!=='tiles')return;
    const stride=Math.max(1,Math.floor(plan.voxel*plan.stride/prior.voxel));
    const key=viewKey('prior',plan.box,stride);
    if(key===this.priorKey || this.requests.jobs.get('prior')?.key===key)return;
    this.background('prior',key,signal=>raster(`${this.base}/prior/view?${this.query({...plan,stride})}`,'X-View',signal),result=>{
      const {label,permille}=viewPlanes(result,['label','permille']);
      this.priors.setRaster(result.meta,label,permille);this.priorKey=key;
    });
  }
  loadVectors() {
    if(this.vectorsLoaded || this.requests.jobs.has('vectors'))return;
    const failed=this.vectorsFailed;
    if(failed && (failed.final || performance.now()<failed.retryAt))return;
    // The whole run's prior outlines, once: a binary mapped into PolygonSets (vectors-decode.mjs), not a JSON to parse.
    this.background('vectors','vectors',async signal=>decodeVectors(await (await response(`${this.base}/xenium?${VECTORS_QUERY}`,signal)).arrayBuffer()),value=>{
      this.priors.vectors=value;this.vectorsLoaded=true;this.vectorsFailed=null;this.stage.dirty=true;
    },error=>{
      // A file that does not decode, or a URL the engine refuses (another format version, a deleted run), fails the same
      // way every time (it is served immutable): not asked for again in this run. Anything else (the engine busy or
      // restarting) is asked for again at the next detail view after a wait that doubles, up to a minute; until then
      // nothing is fetched and no notice repeats with every snapshot.
      const final=/^Malformed prior vectors/.test(error.message) || [400,404,410,422].includes(error.status);
      const wait=Math.min(60000,2*(failed?.wait || 1000));
      this.vectorsFailed={final,wait,retryAt:performance.now()+wait};
    });
  }
  loadTranscripts() {
    const plan=this.plan;
    if(plan?.level!=='detail' || !plan.box) {this.requests.cancel('transcripts');this.transcriptHint='';return;}
    const key=JSON.stringify(plan.box);
    if(key===this.transcriptKey) {this.transcriptHint=this.transcriptResultHint;this.notify();return;}
    if(this.requests.jobs.get('transcripts')?.key===key)return;
    // Skip requests that would certainly be refused (more than the engine's 3 M limit at the run's mean density).
    const runBox=this.run.box, all=this.run.info?.transcripts || 0, area=(b=>Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]));
    const share=area(runBox)?area(intersect(plan.box,runBox))/area(runBox):1;
    if(all*share>6e6) {
      this.requests.cancel('transcripts');this.stage.setTranscripts(null);this.transcriptKey=key;
      this.transcriptHint=this.transcriptResultHint='Too many transcripts · zoom in to see transcripts';this.notify();return;
    }
    this.transcriptHint='Loading transcripts…';this.notify();
    this.background('transcripts',key,async signal=>{
      try {
        const r=await response(`${this.base}/transcripts/view?box=${plan.box.join(',')}`,signal),bytes=await r.arrayBuffer();
        if(bytes.byteLength%16)throw new Error('Truncated transcript records');
        return {bytes,hint:''};
      } catch(error) {
        if(error.name==='AbortError')throw error;
        return {bytes:null,hint:error.status===413?'Too many transcripts · zoom in to see transcripts':`Transcripts unavailable: ${error.message}`};
      }
    },({bytes,hint})=>{
      this.stage.setTranscripts(bytes);this.transcriptKey=key;this.transcriptHint=this.transcriptResultHint=hint;this.notify();
    });
  }
}
