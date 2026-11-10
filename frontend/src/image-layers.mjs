export const COLORMAPS = ['gray','red','green','blue','cyan','magenta','yellow','viridis','magma','inferno','turbo'];
const anchors = {
  viridis: ['440154','3b528b','21918c','5ec962','fde725'],
  magma: ['000004','51127c','b73779','fc8961','fcfdbf'],
  inferno: ['000004','57106e','bc3754','f98e09','fcffa4'],
  turbo: ['30123b','466be3','28bbec','32f298','a4fc3c','f9ba38','e3440a','7a0403'],
};
const TINTS = {gray:[1,1,1],red:[1,0,0],green:[0,1,0],blue:[0,0,1],cyan:[0,1,1],magenta:[1,0,1],yellow:[1,1,0]};
/** A channel's colormap may also be a colour of its own, '#rrggbb' (a colour picker's value, as segmentation layers
 *  take): black up to that colour, like the named single colours. */
export function isHexColour(value) { return typeof value === 'string' && /^#[0-9a-f]{6}$/i.test(value); }
/** The colour a colormap ends at, as '#rrggbb' (what a colour picker shows for it): a colour of its own as it is, a
 *  named single colour as its hex, a many-colour map (viridis...) its brightest anchor. */
export function colourOf(map) {
  if (isHexColour(map)) return map.toLowerCase();
  const tint = TINTS[map];
  if (tint) return '#' + tint.map(v => (v ? 'ff' : '00')).join('');
  return '#' + (anchors[map] || anchors.viridis).at(-1);
}
export function makeLUT(name) {
  const tint = isHexColour(name) ? [1,3,5].map(i => parseInt(name.slice(i,i+2),16)/255) : TINTS[name];
  const stops = (anchors[name] || anchors.viridis).map(h => [0,2,4].map(i => parseInt(h.slice(i,i+2),16)));
  const out = new Uint8Array(768);
  for(let i=0;i<256;i++) for(let c=0;c<3;c++) {
    const t=i/255*(stops.length-1), j=Math.min(stops.length-2,Math.floor(t));
    out[3*i+c]=tint ? i*tint[c] : stops[j][c]+(stops[j+1][c]-stops[j][c])*(t-j);
  }
  return out;
}
export function downsample16(data,w,h,nw,nh) {
  if(w===nw && h===nh) return data;
  const out=new Uint16Array(nw*nh);
  for(let y=0;y<nh;y++) for(let x=0;x<nw;x++) out[y*nw+x]=data[Math.min(h-1,Math.floor((y+.5)*h/nh))*w+Math.min(w-1,Math.floor((x+.5)*w/nw))];
  return out;
}
export function projectPrior(meta, buffer, layer=-1) {
  const n=meta.nx*meta.ny, count=n*meta.nz;
  if(buffer.byteLength!==count*8) throw new Error('Truncated prior');
  const all=new Uint32Array(buffer), labels=new Uint32Array(n), strength=new Uint32Array(n);
  for(let k=0;k<meta.nz;k++) if(layer<0 || layer===k) for(let i=0;i<n;i++) {
    const p=k*n+i;
    if(all[count+p]>strength[i]) { labels[i]=all[p]; strength[i]=all[count+p]; }
  }
  return {labels,strength};
}
