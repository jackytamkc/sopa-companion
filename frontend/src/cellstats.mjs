export function decodeStats(buffer) {
  if (buffer.byteLength % 28) throw Error('Truncated cell statistics');
  const n=buffer.byteLength/28, result={};
  ['area','voxels','assigned','foreground','changed','x','y'].forEach((key,i)=>{
    result[key]=new (i<5?Uint32Array:Float32Array)(buffer,i*n*4,n);
  });
  return result;
}
/** Proseg's count of each cell in a counted frame (its counted file, GET /runs/{id}/counted/{seq}): one little-endian
 *  f32 per cell in label - 1 order, the sum over its genes of proseg's count averaged over its random background draw. */
export function decodeCounted(buffer) {
  if (buffer.byteLength % 4) throw Error('Truncated proseg counts');
  return new Float32Array(buffer,0,buffer.byteLength/4);
}
/** A cell's count in proseg's count (`i`: label - 1): the frame's counted file when it came with the statistics (`kept`:
 *  averaged over its random background draw), else the frame's non-background count (cells.zst foreground: one draw of
 *  it, the count proseg saves; on a frame counted for a pause, the hold, a finish or the end, the very draw its counts
 *  file holds); null for a cell the frame does not have. */
export function keptOf(stats,i) {
  if(!stats||!(i>=0))return null;
  if(stats.kept&&i<stats.kept.length)return stats.kept[i];
  return i<(stats.foreground?.length??0)?stats.foreground[i]:null;
}
/** The cell readout's words (the stage tooltip and the cell card): its area, its count in proseg's count (which one:
 *  countName) and what proseg sets aside as background, both whole and adding up to the transcripts proseg placed in it
 *  (assigned - the whole kept, never below 0), its voxels and its cluster. `voxel` the frame's voxel (µm). */
export const READOUT='{area} µm² · {kept} in {count} · {bg} set aside as background · {voxels} voxels · {cluster}';
/** Which count a cell readout's number is (keptOf's two sources), by name: averaged (the counted file) or as saved (one
 *  draw: the count proseg saves). */
export const READOUT_COUNTS=Object.freeze({averaged:"proseg's count (averaged)",saved:"proseg's count (as saved)"});
export const countName=stats=>stats?.kept?READOUT_COUNTS.averaged:READOUT_COUNTS.saved;
const readoutWhole=new Intl.NumberFormat('en-GB'),readoutTenths=new Intl.NumberFormat('en-GB',{maximumFractionDigits:1});
export function readoutText(stats,i,{voxel=1,cluster=''}={}) {
  const kept=Math.round(keptOf(stats,i)),assigned=stats.assigned[i];
  const values={area:readoutTenths.format(stats.area[i]*voxel*voxel),kept:readoutWhole.format(kept),count:countName(stats),
    bg:readoutWhole.format(Math.max(0,assigned-kept)),voxels:readoutWhole.format(stats.voxels[i]),cluster};
  return READOUT.replace(/\{(\w+)\}/g,(_,name)=>values[name]);
}
/** The cell card's readout line's tooltip: which count its numbers read on this frame, proseg's count averaged over its
 *  background draw (a counted frame: `kept`), else the count proseg saves (keptOf's other source: one draw, also on a
 *  frame counted for a pause, the hold, a finish or the end, whose averaged count is not read: decision A). */
export const READOUT_TIPS=Object.freeze({kept:"Proseg's count at this iteration, averaged over its random background draw",
  saved:"Proseg's count at this iteration as proseg saves it: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table."});
export const readoutTip=stats=>stats?.kept?READOUT_TIPS.kept:READOUT_TIPS.saved;
/** The `count` cells with the most changed columns (changed desc, id asc), zero excluded. One pass over the cells: a
 * value enters the small sorted list only when it beats the list's minimum (tens of thousands of cells per frame in
 * whole-sample runs). */
export function mostActive(stats, count=8) {
  const values=stats.changed,limit=count>=0?Math.floor(count):0,best=[];
  if(!limit)return best;
  let floor=0;   // smallest value in a full list; ids ascend, so an equal value would rank after it
  for(let id=0;id<values.length;id++){
    const changed=values[id];
    if(!changed||best.length===limit&&changed<=floor)continue;
    if(best.length===limit)best.pop();
    let i=best.length;while(i>0&&best[i-1].changed<changed)i--;
    best.splice(i,0,{id,changed});
    if(best.length===limit)floor=best[limit-1].changed;
  }
  return best;
}
/** Per-frame statistics. `current` belongs to the frame on screen (`load`); the next snapshot's statistics are fetched
 * alongside its download (`prefetch`) without touching `current`, so the readouts of the frame still on screen keep
 * their numbers until that snapshot is shown, and neither request cancels the other. A frame the page knows has proseg's
 * own count (`counted(seq)`: its counted file) gets that count with them (`kept`, decodeCounted; null otherwise or when it
 * cannot be read: keptOf then reads the frame's non-background count); one learned to be counted after its statistics
 * came gets it then (recount), once. A frame counted for a pause, the hold, a finish or the end (`sampled(seq)`: the
 * engine scores it on the count proseg saves there) reads the frame's own draw, never its counted file: none is asked
 * for, and one held from before it was learned to be so goes (recount). */
export class CellStats {
  constructor(onChange=()=>{},{counted=()=>false,sampled=()=>false}={}) {this.onChange=onChange;this.counted=counted;this.sampled=sampled;this.reset();}
  /** `failed` is the frame whose statistics could not be loaded by the latest attempt (null when none), so views do
   * not keep showing an older frame's numbers for it. */
  reset(run=null) {
    for(const job of this.jobs?.values()||[])job.controller.abort();
    this.keptController?.abort();this.keptController=new AbortController();
    this.run=run;this.cache=new Map();this.jobs=new Map();this.current=null;this.wanted=null;this.next=null;this.failed=null;this.keptAsked=new Set();
  }
  /** Cancel the requests for frames that are neither `keep` nor `also`. */
  drop(keep,also) {for(const [seq,job] of this.jobs)if(seq!==keep&&seq!==also){job.controller.abort();this.jobs.delete(seq);}}
  /** The statistics of a snapshot on its way, into the cache (shared with a request already made); `current` and the
   *  views are untouched until load(). Supersedes an older prefetch, never the frame on screen's request. */
  prefetch(seq) {
    this.next=seq;this.drop(seq,this.wanted);
    const cached=this.cache.get(seq);
    if(cached)return Promise.resolve(cached);
    return this.jobs.get(seq)?.promise||this.request(seq);
  }
  /** The statistics of the frame now on screen: current at once when cached (or prefetched), else when they arrive. */
  load(seq) {
    this.wanted=seq;this.drop(seq,this.next);
    // Loading the statistics already on screen is not a change (callers ask several times per frame).
    // A frame learned to be counted, or counted for the hold, since its statistics came changes them first (recount):
    // one change on screen.
    if(this.cache.has(seq)){
      this.recount(seq);const cached=this.cache.get(seq);
      if(this.current!==cached){this.current=cached;this.failed=null;this.onChange();}
      return Promise.resolve(cached);
    }
    this.current=null;this.failed=null;
    return this.jobs.get(seq)?.promise||this.request(seq);
  }
  /** Proseg's count of frame `seq`'s cells (decodeCounted), or null: not there, or not readable (never a failure of the
   *  statistics themselves). */
  async keptCount(run,seq,signal) {
    try {
      const r=await fetch(`/api/v1/runs/${encodeURIComponent(run)}/counted/${seq}`,{signal});
      return r.ok?decodeCounted(await r.arrayBuffer()):null;
    } catch { return null; }
  }
  /** The frame's count joins its statistics held here when the page has learned since that the frame is counted (its
   *  counts came after its statistics): asked once per frame; the frame on screen's readouts change with it. A frame
   *  learned since to be counted for a pause, the hold, a finish or the end (sampled) drops the count it holds: its
   *  readouts read the count proseg saves there. */
  recount(seq=this.wanted) {
    const stats=this.cache.get(seq),run=this.run;
    if(stats?.kept&&this.sampled(seq)){
      const next={...stats,kept:null};this.cache.set(seq,next);
      if(this.current===stats){this.current=next;this.onChange();}
      return;
    }
    if(!stats||stats.kept||this.keptAsked.has(seq)||this.jobs.has(seq)||!this.counted(seq)||this.sampled(seq))return;
    this.keptAsked.add(seq);
    this.keptCount(run,seq,this.keptController.signal).then(kept=>{
      if(!kept||kept.length!==stats.assigned.length||this.run!==run||this.cache.get(seq)!==stats||this.sampled(seq))return;
      const next={...stats,kept};this.cache.set(seq,next);
      if(this.current===stats){this.current=next;this.onChange();}
    });
  }
  request(seq) {
    const run=this.run;
    if(!run||seq<0)return Promise.resolve(null);
    const job={controller:new AbortController()},c=job.controller;
    this.jobs.set(seq,job);
    job.promise=(async()=>{
      try {
        // A counted frame's count comes alongside (each request once: a frame learned to be counted later, recount);
        // a frame scored on the count proseg saves (sampled) reads its own draw: no counted file is asked for.
        const counted=this.counted(seq)&&!this.sampled(seq),kept=counted?this.keptCount(run,seq,c.signal):null;
        if(counted)this.keptAsked.add(seq);
        const r=await fetch(`/api/v1/runs/${encodeURIComponent(run)}/cellstats/${seq}`,{signal:c.signal});
        if(!r.ok)throw Error(`Statistics: HTTP ${r.status}`);
        const decoded=decodeStats(await r.arrayBuffer()),counts=await kept;
        const stats={...decoded,seq,kept:counts&&counts.length===decoded.assigned.length&&!this.sampled(seq)?counts:null};
        if(c.signal.aborted||this.run!==run)return null;
        this.cache.set(seq,stats);if(this.cache.size>6)this.cache.delete(this.cache.keys().next().value);
        if(this.wanted===seq){this.current=stats;this.failed=null;this.onChange();}
        return stats;
      } catch(e) {
        // A prefetch that fails is no failure yet: load() tries again once its frame is on screen.
        if(e.name!=='AbortError'&&this.run===run&&this.wanted===seq){this.failed=seq;this.onChange();}
        return null;
      } finally {if(this.jobs.get(seq)===job)this.jobs.delete(seq);}
    })();
    return job.promise;
  }
}
