// Geometry and bounded raster storage for whole-sample views. All coordinates are µm;
// camera zoom and canvas dimensions use device pixels, just like Stage.
export function visibleBox(camera, width, height) {
  const dx=width/camera.zoom/2, dy=height/camera.zoom/2;
  return [camera.x-dx,camera.y-dy,camera.x+dx,camera.y+dy];
}

export function covers(outer, inner, epsilon=1e-5) {
  return !!outer && !!inner && outer[0]<=inner[0]+epsilon && outer[1]<=inner[1]+epsilon &&
    outer[2]>=inner[2]-epsilon && outer[3]>=inner[3]-epsilon;
}

export function intersect(a, b) {
  const box=[Math.max(a[0],b[0]),Math.max(a[1],b[1]),Math.min(a[2],b[2]),Math.min(a[3],b[3])];
  return box[2]>box[0] && box[3]>box[1] ? box : null;
}

export function viewLevel(box, limit) {
  return box[2]-box[0]<=limit && box[3]-box[1]<=limit ? 'detail' : 'overview';
}

export function viewStride(umPerCssPixel, voxel, ratio=1) {
  return Math.max(1,Math.floor(umPerCssPixel/ratio/voxel));
}

export function expandedBox(visible, bounds, factor=1.4, limit=Infinity) {
  const box=intersect(visible,bounds);
  if(!box)return null;
  const out=[];
  for(let k=0;k<2;k++) {
    const size=Math.min(Math.max(box[k+2]-box[k],Math.min((box[k+2]-box[k])*factor,limit)),bounds[k+2]-bounds[k]);
    out[k]=Math.max(bounds[k],Math.min(bounds[k+2]-size,(box[k]+box[k+2]-size)/2));
    out[k+2]=out[k]+size;
  }
  return out;
}

export function planView(visible, bounds, voxel, windowMax, umPerCssPixel, ratio=1, previous=null, overview=false) {
  const limit=windowMax*voxel, level=overview?'overview':viewLevel(visible,limit);
  const stride=level==='detail'?1:viewStride(umPerCssPixel,voxel,ratio);
  // The writer rounds both ends to voxel columns before capping the window.
  // Reserve two columns in the margin so a fractional maximum-size crop is
  // covered by that window. Never shrink the actual visible part of the sample.
  let box=expandedBox(visible,bounds,level==='detail'?1.4:1.5,level==='detail'?limit-2*voxel:Infinity);
  // Retain the margin during small pans. Shrink on a substantial zoom so transcript
  // crops and images don't keep covering a much larger area than the viewport.
  if(box && previous?.level===level && previous.voxel===voxel && previous.stride===stride &&
    covers(previous.box,intersect(visible,bounds)) &&
    (level!=='detail' || (previous.box[2]-previous.box[0]<=limit-2*voxel && previous.box[3]-previous.box[1]<=limit-2*voxel)) &&
    previous.box[2]-previous.box[0]<2*(box[2]-box[0]) && previous.box[3]-previous.box[1]<2*(box[3]-box[1])) box=previous.box;
  // The default server budget (2.4 M) is smaller than a 1600² window. Include
  // rounding to voxel edges, otherwise a detail view can silently become stride 2.
  const maxPixels=box && level==='detail' ? Math.max(2400000,(Math.ceil((box[2]-box[0])/voxel)+1)*(Math.ceil((box[3]-box[1])/voxel)+1)) : 2400000;
  return {level,box,stride,voxel,limit,maxPixels};
}

export function rasterBox(meta) { return [meta.x0,meta.y0,meta.x0+meta.nx*meta.voxel,meta.y0+meta.ny*meta.voxel]; }

export function windowGeometry(frame, window=frame.window) {
  if(!window)return null;
  // Both frame.window and run.window store i0/j0 relative to the full grid.
  const voxel=window.voxel ?? frame.voxel;
  return {...window,voxel,x0:(window.x0 ?? frame.x0)+window.i0*voxel,y0:(window.y0 ?? frame.y0)+window.j0*voxel};
}

export function frameWindow(run, frame) {
  return windowGeometry(frame,run.window?.seq===frame.seq ? run.window : frame.window);
}

export function alignWindow(view, window, buffer) {
  const n=window.nx*window.ny, all=new Uint32Array(buffer);
  if(all.length!==n*window.nz)throw new Error('Truncated window');
  const dx=(window.x0-view.x0)/view.voxel,dy=(window.y0-view.y0)/view.voxel;
  if(Math.abs(window.voxel-view.voxel)>1e-6 || Math.abs(dx-Math.round(dx))>1e-4 || Math.abs(dy-Math.round(dy))>1e-4)throw new Error('Window and view grids do not align');
  const x=Math.round(dx),y=Math.round(dy),x0=Math.max(0,x),y0=Math.max(0,y),x1=Math.min(view.nx,x+window.nx),y1=Math.min(view.ny,y+window.ny);
  const layers=[];
  for(let k=0;k<window.nz;k++) {
    const layer=new Uint32Array(view.nx*view.ny);
    if(x1>x0)for(let row=y0;row<y1;row++) {
      const start=k*n+(row-y)*window.nx+x0-x;
      layer.set(all.subarray(start,start+x1-x0),row*view.nx+x0);
    }
    layers.push(layer);
  }
  return layers;
}

export function viewKey(seq, box, stride) { return JSON.stringify([seq,box,stride]); }

/** The engine's largest image view (server/app.py IMAGE_VIEW_MAX_PX): a whole tissue on a 5120 px wide desktop. */
export const IMAGE_MAX_PX = 6144;

/** Pixels across to request for an image of `box`: its size in the stage's device pixels, so the engine sends one
 *  image pixel per screen pixel (from the coarsest pyramid level still at least that fine, area-averaged down to it).
 *  Bounded by `cap` (the engine's 6144, the GPU's texture size, or less after a lost context) and by the engine's
 *  minimum, 64. */
export function imagePixels(box, devicePxPerUm, cap=IMAGE_MAX_PX) {
  return Math.min(Math.max(64,cap),Math.max(64,Math.ceil(Math.max(box[2]-box[0],box[3]-box[1])*devicePxPerUm)));
}

export function needsImage(image, visible, devicePxPerUm, box=visible, cap=IMAGE_MAX_PX) {
  if(!image || !covers(image.box,visible))return true;
  // Compare pixels per µm, not capped pixel counts over the old box: once a
  // whole-sample image reaches the cap, zooming must still request finer crops.
  const span=b=>Math.max(b[2]-b[0],b[3]-b[1]);
  const current=image.px/span(image.box),needed=imagePixels(box,devicePxPerUm,cap)/span(box);
  return needed>=current*2 || needed<=current/2;
}

/** The image channels to load: the indices of the channels shown (`on[k]` truthy). */
export function shownChannels(on) { return on.flatMap((value,k)=>value?[k]:[]); }

export class RasterLRU {
  constructor(maxBytes=320*2**20,maxEntries=12) {this.maxBytes=maxBytes;this.maxEntries=maxEntries;this.clear();}
  clear() {this.entries=new Map();this.bytes=0;}
  get(key) {
    const value=this.entries.get(key);
    if(value){this.entries.delete(key);this.entries.set(key,value);}
    return value;
  }
  set(key,value) {
    const old=this.entries.get(key);
    if(old){this.bytes-=old.bytes;this.entries.delete(key);}
    if(value.bytes>this.maxBytes)return;
    this.entries.set(key,value);this.bytes+=value.bytes;
    while(this.bytes>this.maxBytes || this.entries.size>this.maxEntries) {
      const [first,item]=this.entries.entries().next().value;
      this.entries.delete(first);this.bytes-=item.bytes;
    }
  }
}
