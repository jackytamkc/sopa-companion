import { projectPrior } from './image-layers.mjs';
import { harness } from './harness-hook.mjs';
import { toPolygonSet, releaseIndex } from './polygons.mjs';
import { paintOutlines, releaseTiles } from './prior-paths.mjs';
import { decodeVectors, VECTORS_QUERY, polygonIds } from './vectors-decode.mjs';
import { SegLayers } from './seg-layers.mjs';
// Screen-resolution vector canvas is composited inside the stage, below masks and points.
//
// The outlines and the prior raster are drawn into a 2-D canvas the size of the stage at the current camera and
// uploaded as one texture. They are drawn again when what they show changes: the vectors, the prior raster, the layer
// settings, the camera or the stage size. While the camera moves (wheel, drag, a fit tween) or the stage is resized
// (the UMAP pane), they are drawn again as often as that costs at most MOTION_SHARE of the main thread, and never when
// one drawing would be a long task by itself (the picker on a whole sample: ~250 ms with the upload on the user's
// software-composited desktop); in between, the last drawing is shown where it belongs under the moving camera
// (linearly sampled, so a zoom neither drops nor blocks its lines), and a fresh one replaces it once the stage has been
// still for its movingUntil (140 ms). A settled view is exactly what one drawing at that camera gives. Each outline
// layer is one path of the polygons near the view (prior-paths.mjs). The picker builds and rasterises that same canvas
// in a worker: its two union strokes can otherwise be deferred until texImage2D, blocking the main thread there.
// Only a complete ImageBitmap replaces the last texture. Detail views (including the prior raster) keep the cheap
// synchronous path, which is also the fallback if the worker or its canvas is unavailable.
//
// The sample's segmentation layers (seg-layers.mjs) are a second raster by the same
// rules (SegRaster): their own canvas, texture, quad and worker, drawn first (under the run's own: Layers lists them
// after Prior shapes), so a change to either never paints or uploads the other, and the run's own canvas is exactly
// what it was. Their strokes are made in the worker on every page (four layers in a detail view can be ~40k polygons:
// a long task on the main thread); without one, on the main thread.
const MOTION_SHARE = .1;      // while moving, drawings may take this share of the main thread (1.6 ms: every frame) ...
const MOTION_TASK_MS = 50;    // ... and none of them may be a long task by itself
const FULL = [-1, 1, 1, -1];   // clip-space corners (left, top, right, bottom) of a raster drawn at the current camera
const VIEW = ['x', 'y', 'zoom', 'width', 'height', 'ratio'];
/** Harness counters this module registers at 0 (metrics.py hook.*): segLayerDraws, each new raster of the segmentation
 *  layers installed (drawn on this thread, or a worker's image uploaded); a raster moved with the camera is none. */
export const COUNTERS = Object.freeze(['segLayerDraws']);
/** performance.measure of one rebuild (user timing 'plv:prior-rebuild'): request through upload (elapsed latency,
 *  including worker time for the picker, not main-thread CPU). Cleared at
 *  once: observers (telemetry.mjs) have it already, and the page's timeline would otherwise grow for as long as it is
 *  open.
 *  The harness hook gets the main thread's part of each rebuild as harness?.priorRebuild(ms, polygons): path building
 *  and stroking on the synchronous path; posting the request and uploading the finished bitmap for the picker's worker,
 *  whose own drawing time goes to harness?.priorWorkerPaint(ms, polygons) instead (off the main thread). */
function measure(start, detail) {
  try { performance.measure?.('plv:prior-rebuild', { start, end: performance.now(), detail }); performance.clearMeasures?.('plv:prior-rebuild'); }
  catch { /* a profiling aid only */ }
}
/** Let the path tiles and grid of a set that is no longer shown go now, not at some later collection. */
function release(set) { if (set) { releaseTiles(set); releaseIndex(set); } }
const sameView = (a,b) => VIEW.every(k=>a[k]===b[k]);

// ---- a cell's starting outline (Go deeper's Show; Stage.setStartOutline) --------------------------------------------
/** The polygon of PolygonSet `set` whose id is `name` (polygonIds: the names the binary carries; else the set's numeric
 *  ids), the largest by area when the name has several parts; -1 for none. One pass over the ids: asked once per cell. */
function polygonNamed(set, name) {
  let ids = null;
  try { ids = polygonIds(set) ?? set?.ids ?? null; } catch { ids = null; }   // a names table that does not decode: none
  if (!ids || name == null || name === '') return -1;
  const text = String(name), number = /^\d+$/.test(text) ? Number(text) : NaN, typed = ArrayBuffer.isView(ids);
  let best = -1, most = -1;
  for (let k = 0; k < set.count; k++) {
    const id = ids[k];
    if (typed ? id !== number : id !== text && !(typeof id === 'number' && String(id) === text)) continue;
    const xy = set.xy, start = set.offsets[k], end = set.offsets[k + 1];
    let area = 0;
    for (let i = start, j = end - 1; i < end; j = i++) area += xy[2 * j] * xy[2 * i + 1] - xy[2 * i] * xy[2 * j + 1];
    if (end - start >= 3 && Math.abs(area) > most) { most = Math.abs(area); best = k; }
  }
  return best;
}
/** The outline of `label`'s voxels in a label raster {labels (row-major, nx x ny), nx, ny, x0, y0, voxel (µm)}: the outer
 *  boundary of its largest 4-connected part along the voxel edges, as the mask draws a cell (two voxels touching at a
 *  corner only are two parts there), its corners only, as a Float32Array of [x, y] µm pairs (a ring: the first point not
 *  repeated); null when the raster holds none of it. */
export function labelOutline(raster, label) {
  const { labels, nx, ny, x0 = 0, y0 = 0, voxel = 1 } = raster ?? {};
  if (!labels || !(label > 0) || !(nx > 0) || !(ny > 0) || labels.length < nx * ny) return null;
  let i0 = Infinity, j0 = Infinity, i1 = -1, j1 = -1;
  for (let j = 0, k = 0; j < ny; j++) for (let i = 0; i < nx; i++, k++) {
    if (labels[k] !== label) continue;
    if (i < i0) i0 = i; if (i > i1) i1 = i; if (j < j0) j0 = j; if (j > j1) j1 = j;
  }
  if (i1 < 0) return null;
  // Its parts in its box (4-connected), the largest kept.
  const w = i1 - i0 + 1, h = j1 - j0 + 1, part = new Int32Array(w * h).fill(-1), stack = [];
  let parts = 0, best = 0, most = 0;
  for (let k = 0; k < w * h; k++) {
    if (part[k] >= 0 || labels[(j0 + Math.floor(k / w)) * nx + i0 + k % w] !== label) continue;
    let size = 0; part[k] = parts; stack.push(k);
    while (stack.length) {
      const q = stack.pop(), qi = q % w, qj = (q - qi) / w; size++;
      for (const [ni, nj] of [[qi - 1, qj], [qi + 1, qj], [qi, qj - 1], [qi, qj + 1]]) {
        const n = nj * w + ni;
        if (ni < 0 || nj < 0 || ni >= w || nj >= h || part[n] >= 0 || labels[(j0 + nj) * nx + i0 + ni] !== label) continue;
        part[n] = parts; stack.push(n);
      }
    }
    if (size > most) { most = size; best = parts; }
    parts++;
  }
  // Its boundary as directed voxel edges with the part on their right (screen axes, y down: clockwise around it); the
  // directions 0 east, 1 south, 2 west, 3 north. Each corner has at most two edges leaving it.
  const inside = (i, j) => i >= 0 && j >= 0 && i < w && j < h && part[j * w + i] === best;
  const V = w + 1, DX = [1, 0, -1, 0], DY = [0, 1, 0, -1], from = [], dirs = [], leaving = new Map();
  const edge = (x, y, d) => { const v = y * V + x, e = dirs.length; from.push(v); dirs.push(d); leaving.has(v) ? leaving.get(v).push(e) : leaving.set(v, [e]); };
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    if (!inside(i, j)) continue;
    if (!inside(i, j - 1)) edge(i, j, 0);
    if (!inside(i + 1, j)) edge(i + 1, j, 1);
    if (!inside(i, j + 1)) edge(i + 1, j + 1, 2);
    if (!inside(i - 1, j)) edge(i, j + 1, 3);
  }
  // Its rings: at a corner with two ways on, the sharper right turn (keeps the part on the right, never crosses itself);
  // the outer boundary is the ring with the largest area (the others go round holes).
  const used = new Uint8Array(dirs.length);
  let ring = null, largest = -1;
  for (let e0 = 0; e0 < dirs.length; e0++) {
    if (used[e0]) continue;
    const corners = [];
    let e = e0, area = 0;
    do {
      used[e] = 1;
      const v = from[e], x = v % V, y = (v - x) / V, d = dirs[e], tx = x + DX[d], ty = y + DY[d];
      area += x * ty - tx * y;
      const ways = leaving.get(ty * V + tx).filter(n => !used[n] || n === e0);
      const next = ways.length > 1 ? [1, 0, 3].map(t => ways.find(n => dirs[n] === (d + t) % 4)).find(n => n != null) : ways[0];
      if (next == null) break;   // cannot happen for a ring of voxel edges; never loops forever
      if (dirs[next] !== d) corners.push(tx, ty);
      e = next;
    } while (e !== e0);
    if (Math.abs(area) > largest) { largest = Math.abs(area); ring = corners; }
  }
  if (!ring || ring.length < 6) return null;
  const out = new Float32Array(ring.length);
  for (let k = 0; k < ring.length; k += 2) { out[k] = x0 + (i0 + ring[k]) * voxel; out[k + 1] = y0 + (j0 + ring[k + 1]) * voxel; }
  return out;
}

/** One stage-size raster of outline layers by the rules above: a 2-D canvas drawn at the camera, uploaded as one texture
 *  and drawn as one full-stage quad, drawn again only when what it shows changes or the camera settles, moved with a
 *  moving camera meanwhile, one worker job at a time, and only a complete worker image replaces the texture. What it
 *  shows is its subclass's: data() the identity of what it shows now (kept in each record: pending, ready, shown),
 *  same(record) whether a record holds that, request() the worker's part of a request and posted() once it went,
 *  paint(view, style, moving) the drawing on this thread, installed() a new raster on screen. */
class OutlineRaster {
  constructor(stage) { this.stage=stage; this.canvas=document.createElement('canvas'); this.hold=false; }
  /** The camera now: {x, y, zoom, width, height, ratio}. */
  view() { const s=this.stage; return {x:s.camera.x,y:s.camera.y,zoom:s.camera.zoom,width:s.canvas.width,height:s.canvas.height,ratio:s.ratio}; }
  /** The camera is moving or the stage is being resized (Stage.emit: within 140 ms of the last change), or a tween runs. */
  moving() { const s=this.stage; return !!s.tween || performance.now()<(s.movingUntil || 0); }
  /** While moving: whether drawing again now keeps within MOTION_SHARE of the main thread and is no long task, by what
   *  the last drawing cost (paint and upload). */
  affordable() { return this.cost!=null && this.cost<=MOTION_TASK_MS && performance.now()-this.paintedAt>=this.cost/MOTION_SHARE; }
  /** Release pending work, including a completed bitmap waiting for the next stage draw. The shown bitmap is kept
   *  separately: it is also the source for a context restore, until replaced or reset. `sent`: what the worker holds. */
  stopWorker() {
    this.worker?.terminate(); this.worker=null; this.sent=null; this.pending=null;
    this.sentClasses=undefined;
    this.ready?.bitmap.close(); this.ready=null;
  }
  closeBitmap() { this.bitmap?.close(); this.bitmap=null; }
  failWorker() { this.workerFailed=true; this.stopWorker(); this.stage.dirty=true; }
  startWorker() {
    if(this.worker)return true;
    if(this.workerFailed || typeof Worker!=='function' || typeof OffscreenCanvas!=='function' || typeof createImageBitmap!=='function')return false;
    try {
      const worker=this.worker=new Worker(new URL('./prior-raster-worker.mjs',import.meta.url),{type:'module'});
      worker.onmessage=({data})=>{
        const pending=this.pending;
        if(this.worker!==worker || !pending || data.id!==pending.id) {data.bitmap?.close();return;}
        if(data.error) {this.failWorker();return;}
        this.pending=null;
        harness?.priorWorkerPaint?.(data.paintMs,data.polygons);   // the worker's drawing: off the main thread
        this.ready={...pending,...data};
        this.stage.dirty=true;
      };
      worker.onerror=event=>{event.preventDefault?.();if(this.worker===worker)this.failWorker();};
      worker.onmessageerror=()=>{if(this.worker===worker)this.failWorker();};
      return true;
    } catch {this.failWorker();return false;}
  }
  /** One in-flight request; after it completes draw() checks the latest state before uploading or requesting again. */
  requestPaint(view,style,moving) {
    if(this.pending)return true;
    const pending={id:this.paintId=(this.paintId||0)+1,start:performance.now(),view,style,...this.data()};
    const request={id:pending.id,view,...this.request(),moving};
    try {
      // Posting clones the request (the vectors once per set) on the main thread: part of the rebuild's main-thread time.
      this.pending=pending; this.worker.postMessage(request); this.posted();
      pending.postMs=performance.now()-pending.start;
      return true;
    } catch {this.failWorker();return false;}
  }
  /** Nothing of it to draw now: no worker, no texture. */
  hide() {
    this.stopWorker();
    if(this.bitmap) {this.closeBitmap();this.shown=null;}
  }
  acceptable(record) {return this.same(record);}
  upToDate(record) {return this.same(record);}
  /** Draw it at `view` with `style` (the settings it is drawn with, as a string), painted by the worker when
   *  `background` (else on this thread): drawn again when what it shows changed, moved with the camera meanwhile. */
  present(view,style,background) {
    if(!background)this.stopWorker();
    let upload=false, clip=FULL, rebuilt=null;
    if(this.ready) {
      const ready=this.ready;this.ready=null;
      if(ready.style===style && this.acceptable(ready) && sameView(ready.view,view)) {
        this.closeBitmap();this.bitmap=ready.bitmap;this.shown=ready;rebuilt=ready;upload=true;
      } else ready.bitmap.close();   // an obsolete camera, size, style or vector set must never replace the texture
    }
    const shown=this.shown;
    const fresh=!shown || shown.style!==style || !this.upToDate(shown) || (!background && this.bitmap);
    const moved=!!shown && !sameView(shown.view,view);
    if(fresh || (moved && !this.hold && (!this.moving() || this.affordable()))) {
      const moving=!fresh && this.moving();
      if(!background || !this.requestPaint(view,style,moving)) {
        this.closeBitmap();rebuilt=this.paint(view,style,moving);upload=true;
      }
    } else if(moved && !this.hold)this.settleLater();
    if(!this.shown)return;   // the first worker image is still being rasterised
    if(!sameView(this.shown.view,view))clip=this.clip(this.shown.view,view);
    const s=this.stage,gl=s.gl,source=this.bitmap || this.canvas;
    if(this.gl!==gl || this.program!==s.imageProgram) {
      upload=true;this.gl=gl;this.program=s.imageProgram;
      const compile=(type,src)=>{const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;};
      this.p=gl.createProgram();
      // uClip: where the raster lies in clip space; FULL (the whole stage, as drawn) once the camera has settled.
      for(const sh of [compile(gl.VERTEX_SHADER,'#version 300 es\nin vec2 aPos;uniform vec4 uClip;out vec2 uv;void main(){uv=aPos;gl_Position=vec4(mix(uClip.x,uClip.z,aPos.x),mix(uClip.y,uClip.w,aPos.y),0,1);}'),compile(gl.FRAGMENT_SHADER,'#version 300 es\nprecision highp float;uniform sampler2D image;in vec2 uv;out vec4 c;void main(){c=texture(image,uv);}')]){gl.attachShader(this.p,sh);gl.deleteShader(sh);}
      gl.bindAttribLocation(this.p,0,'aPos');gl.linkProgram(this.p);this.t=gl.createTexture();this.uClip=gl.getUniformLocation(this.p,'uClip');
      // Clamped: sampled linearly (while moving), a repeating texture would blend the opposite edge into its border.
      gl.bindTexture(gl.TEXTURE_2D,this.t);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    }
    gl.useProgram(this.p);gl.uniform4fv(this.uClip,clip);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.t);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL,false);
    const uploadStart=performance.now();
    if(upload) gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,source);
    if(rebuilt) {this.paintedAt=performance.now();this.cost=rebuilt.cost==null ? this.paintedAt-rebuilt.start : rebuilt.cost+this.paintedAt-uploadStart;}
    // A worker's image: the main thread posted its request and uploads it here (its drawing was reported on arrival).
    if(rebuilt?.bitmap) harness?.priorRebuild((rebuilt.postMs||0)+this.paintedAt-uploadStart,rebuilt.polygons);
    // Drawn as it was painted: texel for pixel. Moved under the camera: sampled linearly (a nearest texel would block
    // the lines up when zooming in, and drop whole rows of them when zooming out).
    const filter=clip===FULL?gl.NEAREST:gl.LINEAR;
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filter);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,filter);
    gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    if(rebuilt) {measure(rebuilt.start,{polygons:rebuilt.polygons,layer:rebuilt.layer});this.installed(rebuilt);}
  }
  /** A new raster is on screen: `rebuilt`, the drawing's record ({paintMs, polygons, ...}). */
  installed() {}
  /** Where the raster drawn at camera `from` lies under camera `to`, in clip space. */
  clip(from,to) {
    const sx=x=>((x-to.x)*to.zoom+to.width/2)/to.width*2-1, sy=y=>1-((y-to.y)*to.zoom+to.height/2)/to.height*2;
    const w=from.width/2/from.zoom,h=from.height/2/from.zoom;
    return [sx(from.x-w),sy(from.y-h),sx(from.x+w),sy(from.y+h)];
  }
  /** Draw again once the stage has been still for its movingUntil; every camera step pushes it back. */
  settleLater() {
    clearTimeout(this.settleTimer);
    this.settleTimer=setTimeout(()=>{this.settleTimer=null;this.stage.dirty=true;},Math.max(0,(this.stage.movingUntil || 0)-performance.now())+1);
  }
}

/** The sample's segmentation layers (seg-layers.mjs: those shown, loaded, at most MAX_SHOWN) on a raster of their own,
 *  drawn under the run's own (PriorLayers draws it first), always by the worker where there is one. Each layer's set is
 *  cloned to the worker once while it is drawn (by an id of its own); the worker lets go of one no longer listed. Its
 *  `segments` store marks the stage dirty on every change (PriorLayers), so it is drawn on demand only. */
class SegRaster extends OutlineRaster {
  constructor(stage,segments) {
    super(stage); this.segments=segments; this.layers=[]; this.ids=new WeakMap(); this.nextId=0;
    for(const name of COUNTERS) harness?.count?.(name,0);
  }
  /** The layers drawn now: [{key, set, style}], front to back. */
  current() { return this.segments?.drawable() ?? []; }
  styleOf(layers) { return JSON.stringify(layers.map(({key,style:o})=>[key,o.color,o.opacity,o.width,!!o.fill])); }
  draw() {
    const layers=this.current();
    if(this.stage.orbit.active || !layers.length) {this.layers=[];this.hide();return;}
    this.layers=layers;
    this.present(this.view(),this.styleOf(layers),this.startWorker());
  }
  data() { return {sets:this.layers.map(l=>l.set)}; }
  same(record) { return record.sets?.length===this.layers.length && record.sets.every((set,k)=>set===this.layers[k].set); }
  idOf(set) { if(!this.ids.has(set))this.ids.set(set,++this.nextId); return this.ids.get(set); }
  request() {
    const layers=this.layers.map(({set,style:o})=>({id:this.idOf(set),color:o.color,opacity:o.opacity,width:o.width,fill:!!o.fill}));
    const missing=this.layers.filter(({set})=>!this.sent?.has(this.idOf(set)));
    return {visible:true,layers,...(missing.length ? {sets:Object.fromEntries(missing.map(({set})=>[this.idOf(set),set]))} : {})};
  }
  posted() { this.sent=new Set(this.layers.map(({set})=>this.idOf(set))); }
  paint(view,style,moving=false) {
    const start=performance.now();
    const layers=this.layers.map(({set,style:o})=>({set,color:o.color,opacity:o.opacity,width:o.width,fill:!!o.fill}));
    const painted=paintOutlines(this.canvas,view,null,{},{moving,layers});
    harness?.priorRebuild(painted.paintMs,painted.polygons);
    this.shown={view,style,...this.data()};
    return {start,...painted};
  }
  installed() { harness?.count?.('segLayerDraws'); }
}

export class PriorLayers extends OutlineRaster {
  constructor(stage, color) {
    super(stage); this.color=color;
    this.settings={prior:{on:true,opacity:.65}, nuclei:{on:false,color:'#ffd18a',opacity:.8,width:1}, xenium:{on:false,color:'#8ecae6',opacity:.7,width:1}};
    this.hold=false;   // the picker keeps the last box's outlines while the next box's load: no drawing of them meanwhile
    // The sample's segmentation layers: their store (main.js sets its sample; the Layers panel lists them) and raster.
    this.segments=new SegLayers();
    this.segRaster=new SegRaster(stage,this.segments);
    this.segments.listen(drawn=>{if(drawn)stage.dirty=true;});   // a list read or a load begun draws nothing
    stage.drawPriors=()=>this.draw();
  }
  /** {cells, nuclei: PolygonSet|null, ...metadata}. Legacy JSON polygon arrays are converted once, when assigned. A set
   *  that is replaced lets go of its path tiles and grid at once (the picker makes one per box). */
  get vectors() { return this._vectors ?? null; }
  set vectors(value) {
    if (value) {
      const cells=toPolygonSet(value.cells), nuclei=toPolygonSet(value.nuclei);
      if (cells!==(value.cells ?? null) || nuclei!==(value.nuclei ?? null)) value={...value,cells,nuclei};
    }
    const old=this._vectors;
    this._vectors=value || null;
    for (const set of [old?.cells,old?.nuclei]) if (set && set!==this._vectors?.cells && set!==this._vectors?.nuclei) release(set);
  }
  reset() { this.stopWorker(); this.closeBitmap(); this.token=(this.token||0)+1; this.vectors=null; this.classes=null; this.prior=null; this.buffer=null; this.raster=null; this.projected=null; this.outline=null; this.vectorsVisible=true; this.shown=null; this.hold=false; this.loaded=null; clearTimeout(this.settleTimer); this.settleTimer=null; this.stage.dirty=true; }
  /** Go deeper's starting outline of a cell (for Stage.setStartOutline): the
   *  prior's own polygon with the cell's `name` (the starting cell as it was drawn; its largest part when it has several),
   *  or, without one, the outline of its `label` in the prior's labels (labelOutline: the start line's cells, as the
   *  engine's where reply at seq "start" reads them; a detail run's whole prior, a whole-sample run's view of it); null
   *  while neither is loaded or holds it. {points: Float32Array of [x, y] µm pairs, closed: true, source: 'prior' |
   *  'start'}. The answer is kept until the vectors or the prior change, so asking again costs nothing. */
  startOutline(name, label=0) {
    const set=this.vectors?.cells ?? null, raster=this.startLabels(), key=`${name ?? ''}|${label>>>0}`, kept=this.outline;
    if(kept && kept.set===set && kept.labels===(raster?.labels ?? null) && kept.key===key) return kept.value;
    let value=null;
    const k=set ? polygonNamed(set,name) : -1;
    if(k>=0) value={points:set.xy.slice(2*set.offsets[k],2*set.offsets[k+1]),closed:true,source:'prior'};
    else if(raster && label>0) { const points=labelOutline(raster,label>>>0); if(points) value={points,closed:true,source:'start'}; }
    this.outline={set,labels:raster?.labels ?? null,key,value};
    return value;
  }
  /** The prior's labels as a raster {labels, nx, ny, x0, y0, voxel} (each column's strongest prior cell, as the prior
   *  layer shows it with every layer), or null before the prior is loaded; a detail run's projected once per prior. */
  startLabels() {
    const m=this.meta;
    if(!m || !(m.nx>0) || !(m.ny>0)) return null;
    let labels=this.raster?.labels ?? null;
    if(!labels && this.buffer) {
      if(this.projected?.buffer!==this.buffer) {
        try { this.projected={buffer:this.buffer,labels:projectPrior(m,this.buffer,-1).labels}; } catch { return null; }   // a truncated prior
      }
      labels=this.projected.labels;
    }
    return labels ? {labels,nx:m.nx,ny:m.ny,x0:m.x0,y0:m.y0,voxel:m.voxel} : null;
  }
  /** Go deeper's hold split (its line's tooltip): how the start held the spot of each of `points` ({x, y, z}: arrays of
   *  one length in µm, a where reply's; z in proseg's own depth units, NaN where the engine gave none) for cell `label`,
   *  from this run's own prior (a detail run's, once load() has its meta and buffer): 1 at the nucleus level, 2 at the
   *  cell level, 0 not known. The spot is the worker's (analysis/worker.py grid_columns and voxel_lookup: the column
   *  floor((x - x0) / voxel) in 32-bit floats, as the engine computes on its float32 records; the layer floor((z -
   *  grid_origin) / grid_spacing) clipped to the layers, from `z` = info.z, 0 and 1 without it; a point with no depth is
   *  placed only in a one-layer prior). Not known: another label there or none, outside the grid, or a level that is
   *  neither or both of round(1000 x nucleus_prob) and round(1000 x cell_prob) of info.prior, each ± 2 (proseg writes
   *  round(1000 x) its stored probability; equal levels can't be told apart). A where reply's places are rounded (x and
   *  y to 0.01 µm, z to 0.001), so a point within half that of a spot's edge can land in the next spot: where that spot
   *  is another cell's or no cell's it is not known, but where it is the same cell's it takes that spot's level, which
   *  may not be its own (only at an edge between the cell's two levels). Returns a Uint8Array, or null for a
   *  whole-sample run (its view of the prior: setRaster, no buffer) or before the prior is in. */
  startHold(points, label, z=null) {
    const m=this.meta, buffer=this.buffer;
    if(!m || !buffer || !(label>0) || !points?.x) return null;
    const nx=m.nx, ny=m.ny, nz=m.nz || 1, n=nx*ny*nz;
    if(!(nx>0) || !(ny>0) || !(nz>0) || buffer.byteLength!==8*n) return null;   // not this geometry's prior (a truncated one)
    const f=Math.fround, all=new Uint32Array(buffer), nucleus=Math.round(1000*m.nucleus_prob), cell=Math.round(1000*m.cell_prob);
    const x0=f(m.x0), y0=f(m.y0), voxel=f(m.voxel);
    const zo=f(Number.isFinite(z?.grid_origin) ? z.grid_origin : 0), zs=f(Number.isFinite(z?.grid_spacing) ? z.grid_spacing : 1);
    const count=points.x.length, out=new Uint8Array(count);
    for(let p=0;p<count;p++) {
      const i=Math.floor(f(f(f(points.x[p])-x0)/voxel)), j=Math.floor(f(f(f(points.y[p])-y0)/voxel));
      if(!(i>=0 && i<nx && j>=0 && j<ny)) continue;
      let k=0;
      if(nz>1) {
        const depth=points.z?.[p];
        if(!Number.isFinite(depth)) continue;
        k=Math.min(nz-1,Math.max(0,Math.floor(f(f(f(depth)-zo)/zs))));
      }
      const v=(k*ny+j)*nx+i;
      if(all[v]!==label) continue;
      const level=all[n+v], atNucleus=Math.abs(level-nucleus)<=2, atCell=Math.abs(level-cell)<=2;   // NaN levels: neither
      out[p]=atNucleus===atCell ? 0 : atNucleus ? 1 : 2;
    }
    return out;
  }
  /** A run in detail mode: its prior vectors (the binary, vectors-decode.mjs: no JSON parsed on the main thread) and
   *  its prior raster, each once per run (openRun and the run's info arriving both ask; a failed one is asked again). */
  async load(run) {
    if(run.mode==='tiled')return; // View rasters and detail-only vectors are supplied by TiledLoader.
    const token=this.token, id=run.id, loaded=this.loaded?.token===token ? this.loaded : (this.loaded={token,vectors:false,prior:false});
    const get=async path=>{const r=await fetch(`/api/v1/runs/${encodeURIComponent(id)}/${path}`); if(!r.ok) throw Error(path); return r;};
    const once=(what,job)=>{loaded[what]=true;return job().catch(error=>{if(this.loaded===loaded)loaded[what]=false;throw error;});};
    const jobs=[];
    if(!loaded.vectors)
      jobs.push(once('vectors',()=>get(`xenium?${VECTORS_QUERY}`).then(r=>r.arrayBuffer()).then(decodeVectors).then(v=>{if(token===this.token){this.vectors=v;this.stage.dirty=true;}})));
    if(run.info?.prior && !loaded.prior)
      jobs.push(once('prior',()=>get('prior').then(r=>r.arrayBuffer()).then(b=>{if(token===this.token){this.meta=run.info.prior;this.buffer=b;this.rebuild();}})));
    await Promise.allSettled(jobs);
  }
  setRaster(meta,labels,strength) {
    this.meta=meta;this.buffer=null;this.raster={labels,strength};this.rebuild();
  }
  rebuild() {
    if(!this.buffer && !this.raster) return;
    const m=this.meta, {labels,strength}=this.raster || projectPrior(m,this.buffer,this.stage.frame?.layer ?? -1);
    const canvas=document.createElement('canvas'); canvas.width=m.nx;canvas.height=m.ny;
    const g=canvas.getContext('2d'), pixels=g.createImageData(m.nx,m.ny);
    const colors=new Map();
    for(let i=0;i<labels.length;i++) if(labels[i]) {
      if(!colors.has(labels[i]))colors.set(labels[i],this.color(labels[i]).match(/\d+/g).map(Number));
      const rgb=colors.get(labels[i]), x=i%m.nx,y=Math.floor(i/m.nx);
      const edge=x===0||y===0||x===m.nx-1||y===m.ny-1||labels[i-1]!==labels[i]||labels[i+1]!==labels[i]||labels[i-m.nx]!==labels[i]||labels[i+m.nx]!==labels[i];
      pixels.data.set([...rgb,Math.round(255*(edge ? .95 : strength[i]/1000))],i*4);
    }
    g.putImageData(pixels,0,0);this.prior=canvas;this.layer=this.stage.frame?.layer;this.stage.dirty=true;
  }
  /** Whether anything of this layer would be drawn now (hidden layers cost no canvas upload and no full-screen quad). */
  visible() {
    const s=this.stage;
    if(this.prior && this.settings.prior.on) return true;
    if(this.vectorsVisible===false) return false;
    return [['xenium',this.vectors?.cells],['nuclei',this.vectors?.nuclei]].some(([key,set])=>set?.count && (this.settings[key].on || !s.info));
  }
  get classes() {return this._classes || null;}
  set classes(value) {
    if(value===this._classes)return;
    if(value && this._classes?.set===value.set && value.gen<this._classes.gen)return;
    this._classes=value;this.classEpoch=(this.classEpoch||0)+1;this.stage.dirty=true;
  }
  data() { return {vectors:this.vectors,prior:this.prior,classes:this.classes,classEpoch:this.classEpoch||0}; }
  same(record) { return record.vectors===this.vectors && record.prior===this.prior; }
  upToDate(record) {return this.same(record) && record.classEpoch===(this.classEpoch||0);}
  /** An image on screen with the cells' classes of its set (the stain range choosers): the harness hears its paint time
   *  and outlines, and how many of the set's cells it keeps and hides (a hidden cell is not drawn at all: none is grey).
   *  Installed images only: one the worker drew for a view or classes no longer wanted is not counted. */
  installed(rebuilt) {
    const classes=this.shown?.classes;
    if(!classes || classes.set==null || classes.set!==this.shown.vectors?.set)return;
    harness?.rangePaint?.(rebuilt.paintMs,rebuilt.polygons,{kept:classes.kept,hidden:classes.cells-classes.kept,grey:0});
  }
  acceptable(record) {
    return this.same(record) && (!this.shown || !this.same(this.shown) || record.classEpoch>=this.shown.classEpoch);
  }
  request() {
    const request={settings:this.settings,visible:this.vectorsVisible!==false,picker:!this.stage.info};
    if(this.sent!==this.vectors)request.vectors={cells:this.vectors?.cells,nuclei:this.vectors?.nuclei,set:this.vectors?.set};
    if(this.sentClasses!==this.classes)request.classes=this.classes;
    return request;
  }
  posted() { this.sent=this.vectors;this.sentClasses=this.classes; }
  draw() {
    this.segRaster?.draw();   // the segmentation layers' raster first: under the run's own
    const s=this.stage;
    if(s.orbit.active || (!this.vectors && !this.prior) || !this.visible()) {this.hide();return;}
    if(this.buffer && this.layer!==s.frame?.layer) this.rebuild();
    const style=JSON.stringify([!!s.info,this.settings,this.vectorsVisible]);
    const background=(!s.info || s.info.kind==='stain') && !(this.prior && this.settings.prior.on) && this.startWorker();
    this.present(this.view(),style,background);
  }
  /** Draw the vectors and the prior raster into the canvas at `view`: {start, polygons stroked, layer: 'xenium+nuclei'}.
   *  `moving`: a drawing while the camera moves, which may reuse the last path of the same tiles (prior-paths.mjs). */
  paint(view,style,moving=false) {
    const s=this.stage,canvas=this.canvas,start=performance.now();
    const painted=paintOutlines(canvas,view,this.vectors,this.settings,{picker:!s.info,visible:this.vectorsVisible!==false,moving,classes:this.classes});
    harness?.priorRebuild(painted.paintMs,painted.polygons);
    if(this.prior && this.settings.prior.on) {
      const g=canvas.getContext('2d'),m=this.meta;g.globalAlpha=this.settings.prior.opacity;g.imageSmoothingEnabled=false;
      g.drawImage(this.prior,m.x0,m.y0,m.nx*m.voxel,m.ny*m.voxel);
    }
    this.shown={view,style,...this.data()};
    return {start,...painted};
  }
  /** The segmentation layers as the page shows them (the harness snapshot's `segmentations`): the store's
   *  {sample, loaded: [{key, name, visible, color, opacity, width, fill, state}], note}, each loaded layer with `drawn`
   *  (its outlines, with its style now, are in the raster on screen). */
  segmentationsShown() {
    const value=this.segments?.snapshot() ?? {sample:null,loaded:[],note:null}, raster=this.segRaster, shown=raster?.shown;
    const current=raster?.current() ?? [];
    const same=!!shown && shown.style===raster.styleOf(current) && shown.sets?.length===current.length && shown.sets.every((set,k)=>set===current[k].set);
    const drawn=new Set(same ? current.map(l=>l.key) : []);
    return {...value,loaded:value.loaded.map(l=>({...l,drawn:drawn.has(l.key)}))};
  }
}
