// PLCV v1 (the engine's encode_cells, engine/vectors.py), little endian: six u32s (magic "PLCV", version, header bytes H,
// cells N, brightness arrays C, JSON bytes), UTF-8 metadata from byte 24 padded to H; then ids u32[N] (1..N: the set's
// order, as the outlines' ids), area f32[N], flags u8[N] padded to four bytes (bit 0: in a removed area), and C brightness
// f32[N] arrays in the order of the metadata's `brightness` channel list. The metadata binds every array to {set,
// areas_version} and may name an `error` for cells whose brightness could not be read (NaN: unknown, kept).
const MAGIC=0x56434c50, VERSION=1;
const LITTLE=new Uint8Array(new Uint16Array([1]).buffer)[0]===1;
export const CELL_VALUES_TEXT = Object.freeze({
  error:"The cells' sizes could not be read. Reopen the run to try again.",
  brightness:'The cells could not be measured in this channel. Reopen the run to try again.',
  unmeasured:'Some cells could not be measured in this channel, so they are kept. Reopen the run to try again.',
});
/** {...metadata, count, ids, area, flags, brightness: Map(channel -> Float32Array)}, views over `buffer`. Throws
 *  'Malformed cell values: …' on anything that does not add up exactly. */
export function decodeCellValues(buffer) {
  const fail=why=>{throw Error(`Malformed cell values: ${why}`);};
  if(!(buffer instanceof ArrayBuffer) || buffer.byteLength<24 || !LITTLE)fail('bad header');
  const view=new DataView(buffer),u32=at=>view.getUint32(at,true);
  if(u32(0)!==MAGIC || u32(4)!==VERSION)fail('unknown format');
  // H: any multiple of four past the metadata, as the engine's own decoder (decode_cells) takes it.
  const header=u32(8),count=u32(12),arrays=u32(16),jsonLength=u32(20),flagsBytes=Math.ceil(count/4)*4;
  if(header%4 || header<24+jsonLength || header+8*count+flagsBytes+4*count*arrays!==buffer.byteLength)fail('bad length');
  let meta;
  try {meta=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(new Uint8Array(buffer,24,jsonLength)));}
  catch {fail('bad metadata');}
  const names=meta?.brightness;
  if(!meta || typeof meta.set!=='string' || !meta.set || !Number.isInteger(meta.areas_version) || meta.areas_version<0 ||
      !Array.isArray(names) || names.length!==arrays || names.some(c=>typeof c!=='string' || !c) ||
      new Set(names).size!==arrays)fail('bad metadata');
  const ids=new Uint32Array(buffer,header,count),area=new Float32Array(buffer,header+4*count,count);
  const flags=new Uint8Array(buffer,header+8*count,count),at=header+8*count+flagsBytes;
  for(let i=0;i<count;i++)if(ids[i]!==i+1 || flags[i]>1)fail('unknown ids or flags');
  return {...meta,count,ids,area,flags,brightness:new Map(names.map((c,k)=>[c,new Float32Array(buffer,at+4*count*k,count)]))};
}
const outside = (value,range) => Number.isFinite(value) &&
  (range?.min!=null && value<range.min || range?.max!=null && value>range.max);

/** One run's per-cell values. Requests are cancellable and deduplicated by set, area version and channel.
 *  API adapter (api.js): stainCells(id, set, channel, signal) -> ArrayBuffer. Without it, the same HTTP endpoint. */
export class CellValues {
  constructor(api, {changed=()=>{},error=()=>{}}={}) {
    Object.assign(this,{api,changed,error});this.gen=0;this.reset();
  }
  reset() {
    for(const job of this.jobs?.values() || [])job.controller.abort();
    this.ticket=(this.ticket||0)+1;this.id=null;this.set=null;this.areasVersion=null;
    this.data=null;this.brightness=new Map();this.jobs=new Map();this.failed=new Set();this.cached=null;
  }
  async load(id,set,areasVersion=0,channel=null) {
    if(!id || !set)return null;
    if(id!==this.id || set!==this.set || areasVersion!==this.areasVersion) {
      this.reset();Object.assign(this,{id,set,areasVersion});
    }
    if(channel===null && this.data || channel!==null && this.brightness.has(channel))return this.data;
    if(this.jobs.has(channel))return this.jobs.get(channel).promise;
    if(this.failed.has(channel))return null;
    const ticket=this.ticket,controller=new AbortController(),job={controller};
    this.jobs.set(channel,job);
    job.promise=(async()=>{
      try {
        let buffer;
        if(this.api?.stainCells)buffer=await this.api.stainCells(id,set,channel,controller.signal);
        else {
          const query=new URLSearchParams({set});if(channel!==null)query.set('channel',channel);
          const response=await fetch(`/api/v1/runs/${encodeURIComponent(id)}/stain/cells?${query}`,{signal:controller.signal});
          if(!response.ok)throw Error(CELL_VALUES_TEXT.error);
          buffer=await response.arrayBuffer();
        }
        if(ticket!==this.ticket || controller.signal.aborted)return null;
        const data=decodeCellValues(buffer);
        if(data.set!==set || data.areas_version!==areasVersion || channel!==null && !data.brightness.has(channel) ||
            this.data && data.count!==this.data.count)throw Error('The cells changed while reading their values.');
        this.data=data;
        if(channel!==null)this.brightness.set(channel,data.brightness.get(channel));
        // Cells whose image could not be read are unknown, hence kept: the chooser's line says so, and says none could be
        // measured when that is so (the engine reads each result file's image on its own).
        if(channel!==null && data.error)
          this.error(data.brightness.get(channel).some(Number.isFinite)?CELL_VALUES_TEXT.unmeasured:CELL_VALUES_TEXT.brightness,channel);
        this.cached=null;return data;
      } catch(e) {
        if(ticket===this.ticket && !controller.signal.aborted && e.name!=='AbortError') {
          this.failed.add(channel);this.error(channel===null?CELL_VALUES_TEXT.error:CELL_VALUES_TEXT.brightness,channel);
        }
        return null;
      } finally {if(ticket===this.ticket){this.jobs.delete(channel);this.changed();}}
    })();
    this.changed();
    return job.promise;
  }
  /** Uint8 class 0 = kept, 1 = hidden. Every comparison promotes the stored float32 to JS Number (float64). */
  classes(rules={}) {
    if(!this.data)return null;
    const key=JSON.stringify([rules.size??null,Object.entries(rules.brightness||{}).sort(([a],[b])=>a.localeCompare(b))]);
    if(this.cached?.key===key)return this.cached.value;
    const {area,flags,count}=this.data,array=new Uint8Array(count),channels=Object.entries(rules.brightness || {});
    let kept=0,inAreas=0,bySize=0,byBrightness=0;
    for(let i=0;i<count;i++) {
      const inArea=!!(flags[i]&1),size=outside(area[i],rules.size);
      const bright=channels.some(([channel,range])=>outside(this.brightness.get(channel)?.[i],range));
      if(inArea)inAreas++;if(size)bySize++;if(bright)byBrightness++;
      array[i]=inArea || size || bright?1:0;if(!array[i])kept++;
    }
    const value={gen:++this.gen,set:this.set,array,kept,cells:count,in_areas:inAreas,by_size:bySize,by_brightness:byBrightness};
    this.cached={key,value};return value;
  }
  forSet(set,rules={}) {return set===this.set?this.classes(rules):null;}
}
