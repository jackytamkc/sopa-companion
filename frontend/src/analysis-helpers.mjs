export function decodeAnalysis(buffer) {
  if (buffer.byteLength % 14) throw Error('Truncated analysis cells');
  const n = buffer.byteLength / 14, view = new DataView(buffer);
  const clusters = new Int16Array(n), coords = new Float32Array(n * 3);
  for (let i=0;i<n;i++) clusters[i]=view.getInt16(i*2,true);
  for (let i=0;i<n*3;i++) coords[i]=view.getFloat32(n*2+i*4,true);
  return {clusters,coords};
}
export function nearestEarlier(seqs, target) {
  let best=-1;
  for(const seq of seqs) if(seq<=target && seq>best) best=seq;
  return best;
}
/** The oldest of `seqs` after `target` (-1: none): looking back at a frame before the first analysed one, the result
 *  shown is that one's (as the engine answers a frame asked for: the nearest with proseg's count). */
export function nearestLater(seqs, target) {
  let best=-1;
  for(const seq of seqs) if(seq>target && (best<0 || seq<best)) best=seq;
  return best;
}
export function labelLookup(clusters) {
  const data=new Int16Array(4096*Math.max(1,Math.ceil(((clusters?.length||0)+1)/4096))).fill(-1);
  if(clusters) data.set(clusters,1);
  return data;
}

/** First occupied voxel along a ray expressed in voxel coordinates. Event-time picking. */
export function pickVoxel(origin, direction, size, layers, hidden = null, clusters = null) {
  const d=direction.map(v=>Math.abs(v)<1e-12?(v<0?-1e-12:1e-12):v);
  let enter=0,leave=Infinity;
  for(let k=0;k<3;k++){const a=-origin[k]/d[k],b=(size[k]-origin[k])/d[k];enter=Math.max(enter,Math.min(a,b));leave=Math.min(leave,Math.max(a,b));}
  if(leave<=enter)return 0;
  const p=origin.map((v,k)=>Math.max(0,Math.min(size[k]-1,Math.floor(v+d[k]*(enter+1e-7)))));
  const step=d.map(Math.sign),delta=d.map(v=>Math.abs(1/v));
  const next=p.map((v,k)=>(v+(d[k]>0?1:0)-origin[k])/d[k]);
  for(let i=0;i<size[0]+size[1]+size[2];i++){
    const label=layers[p[2]][p[1]*size[0]+p[0]];
    if(label && !hidden?.has(clusters?.[label-1]))return label;
    const k=next[0]<=next[1]&&next[0]<=next[2]?0:next[1]<=next[2]?1:2;
    if(next[k]>=leave)return 0;
    p[k]+=step[k];next[k]+=delta[k];if(p[k]<0||p[k]>=size[k])return 0;
  }
  return 0;
}
