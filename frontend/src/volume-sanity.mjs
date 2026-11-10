// Standalone CPU checks for the volume shader's DDA and perspective depth math.
// Run: node frontend/src/volume-sanity.mjs
import assert from 'node:assert/strict';
function march(origin, dir, size, limit=Infinity) {
  const safe=dir.map(v=>(v<0?-1:1)*Math.max(Math.abs(v),1e-12));
  const a=origin.map((v,i)=>-v/safe[i]);
  const b=origin.map((v,i)=>(size[i]-v)/safe[i]);
  let t=Math.max(0,...a.map((v,i)=>Math.min(v,b[i])));
  const end=Math.min(limit,...a.map((v,i)=>Math.max(v,b[i])));
  if(end<=t)return [];
  const p=origin.map((v,i)=>Math.max(0,Math.min(size[i]-1,Math.floor(v+dir[i]*(t+1e-4)))));
  const step=safe.map(Math.sign), delta=safe.map(v=>Math.abs(1/v));
  const next=p.map((v,i)=>(v+(safe[i]>0?1:0)-origin[i])/safe[i]);
  const segments=[];
  for(let n=0;n<16384;n++) {
    const axis=next[0]<=next[1]&&next[0]<=next[2]?0:next[1]<=next[2]?1:2;
    const finish=Math.min(end,next[axis]);
    assert(finish>=t-1e-8);
    if(finish>t)segments.push({p:[...p],length:finish-t});
    if(finish>=end)break;
    t=next[axis];p[axis]+=step[axis];next[axis]+=delta[axis];
    if(p.some((v,i)=>v<0||v>=size[i]))break;
  }
  return segments;
}
for(const size of [[300,300,4],[1200,1200,4]]) {
  const vertical=march([1.5,1.5,8],[0,0,-1],size);
  assert.equal(vertical.length,4);
  assert.equal(vertical.reduce((v,s)=>v+s.length,0),4);
  assert.equal(march([-1,1.5,2],[1,0,0],size).length,size[0]);
  assert.equal(march([1.5,1.5,8],[0,0,-1],size,5).length,1);
  assert.equal(march([-2,-2,8],[0,0,-1],size).length,0);
  const diagonal=march([-1,-1,-1],[1,1,1],size);
  assert.equal(diagonal.length,4);
}
for(const near of [.001,.1]) for(const far of [100,10000]) for(const z of [1,5,50]) {
  const depth=((far+near)/(far-near)-2*far*near/((far-near)*z)+1)/2;
  const decoded=2*near*far/(far+near-(depth*2-1)*(far-near));
  assert(Math.abs(decoded-z)<1e-6);
}
const opacity=1.4;
assert(Math.abs((1-Math.exp(-opacity))-(1-Math.exp(-opacity/4)**4))<1e-12);
console.log('PASS: DDA bounds, parallel/tied rays, point-depth clipping, depth reconstruction, subdivision-independent opacity');
