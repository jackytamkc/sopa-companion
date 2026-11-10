export function maskMapping(ids, cells) {
  const byId=new Map(cells.map((id,i)=>[String(id),i+1]));
  let max=0;for(const label of Object.keys(ids))max=Math.max(max,Number(label));
  const mapping=new Uint32Array(max+1);
  for(const [label,id] of Object.entries(ids))mapping[Number(label)]=byId.get(String(id))||0;
  return mapping;
}
export function sampleMasks(data,width,height,limit) {
  const step=Math.max(1,Math.ceil(Math.max(width,height)/limit));
  const nx=Math.ceil(width/step),ny=Math.ceil(height/step),out=new Uint32Array(nx*ny);
  for(let y=0;y<ny;y++)for(let x=0;x<nx;x++)out[y*nx+x]=data[Math.min(height-1,y*step)*width+Math.min(width-1,x*step)];
  return {data:out,nx,ny,step};
}
