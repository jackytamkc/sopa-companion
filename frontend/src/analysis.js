import { decodeAnalysis, nearestEarlier, nearestLater } from './analysis-helpers.mjs';
import { UmapView } from './umap.js';
import { clusterColor, createLegend } from './clusters.js';
import { harness } from './harness-hook.mjs';
import { pageLog } from './log-export.mjs';
import { ClusterSettings } from './analysis-settings.mjs';
import { copy, number, unavailableTooltip } from './qc-copy.mjs';
import { keptOf, countName } from './cellstats.mjs';

// Clusters & UMAP are always on (no switch: the engine analyses every active run from its first frame, engine/analysis.py),
// on proseg's own count of each cell, so only frames that have it (counted frames) have clusters: a frame looked back at
// shows the newest analysed one at or before it (before every analysed one, the oldest after it: the engine's answer
// for a frame asked for), with a badge in the pane's head naming its iteration.
// A whole sample (tens of thousands of cells) yields a result every ~0.5 s and applying one (UMAP layout, tissue
// lookup, legend) is not free: while following live, results of large runs are applied at most once per COALESCE_MS,
// always the newest one, and always with a snapshot (present()). Small runs and explicit seeks (scrubbing, stepping, a
// paused run) apply at once.
export const COALESCE_CELLS=20000, COALESCE_MS=1000;
/** Milliseconds to hold a new result back (0 = apply now), `elapsed` ms after the previous one was applied. */
export function applyDelay(cells,live,elapsed){return live&&cells>=COALESCE_CELLS?Math.max(0,COALESCE_MS-elapsed):0;}
// While following a live run, a result that has just been downloaded is applied with the next snapshot (main.js calls
// frame() for every frame it shows), so recolouring the tissue and the UMAP shares that snapshot's paint; a run that
// stops sending snapshots gets it after WITH_SNAPSHOT_MS anyway. What the analysis is doing (which iteration's clusters
// are shown when looking back, failures) is not written on the page: it goes to the log (log-export.mjs, the Log button;
// each result's step times are the engine's part), so a stream of results costs no paint of its own.
export const WITH_SNAPSHOT_MS=1000;
// A run the engine cannot cluster for a reason a new attempt does not fix (engine/analysis.py describe `unavailable`,
// a status's `reason`): the UMAP pane stays closed and one calm line under Layers > Cells says why, instead of a failure.
export const UNAVAILABLE={
  too_few_cells:'Too few cells in this region to group them into clusters.',
  no_transcripts:'This run was made before clusters were computed live, so it has none.',
  // A run without proseg's own count: made before proseg-live kept it, or its proseg-live can't keep it (the line's
  // tooltip says why a new run has it, or how to update proseg-live: qc-copy.mjs unavailableTooltip).
  no_counts:"This run was made before proseg's own count was kept, so it has no clusters.",
  old_proseg:'Update proseg-live to see clusters.',
};
// The UMAP's hover: the cell, its cluster and its count in proseg's count in the frame on screen (whole, "…" until the
// frame's statistics are in), named as the cell readout names it ({count}: cellstats.mjs countName, averaged on a counted
// frame, as saved on one counted for a pause, the hold, a finish or the end and between counted iterations). Its form
// only beside its number: "…" is in plain "proseg's count" (UMAP_COUNT), as which one comes is not known yet.
export const UMAP_TIP="{cell} · {cluster} · {n} in {count}";
const UMAP_COUNT="proseg's count";
// An engine whose analysis worker cannot run (its python off, not there, or its libraries not loading): the pane stays
// shut, and one line under Layers > Cells says where it is set: the app's config.toml on the desktop; in a notebook the
// user's own settings file, which launch() reads (api/launch.py), and then a new launch. When the engine names the
// packages its Python lacks (describe `missing`), the line names them instead (MISSING).
export const SETUP=Object.freeze({
  desktop:"Clusters and the UMAP can't run here: check python under [analysis] in the app's config.toml (the Log button, top right, says why).",
  notebook:"Clusters and the UMAP can't run here: check python under [analysis] in ~/.config/sopa-companion/config.toml and launch again (the Log button, top right, says why).",
});
export const MISSING='Clusters and the UMAP need {names}, which this Python environment lacks.';
/** MISSING with the packages named ("a", "a and b", "a, b and c"); null for none. */
export function missingLine(names){
  const list=Array.isArray(names)?names.filter(name=>typeof name==='string'&&name):[];
  if(!list.length)return null;
  const joined=list.length===1?list[0]:`${list.slice(0,-1).join(', ')} and ${list.at(-1)}`;
  return MISSING.replace('{names}',joined);
}
// The first clusters ever shown in this browser colour the cells and transcripts by cluster (as switching Clusters &
// UMAP on used to); from then on the layer panel remembers the user's colours, so a choice of Identity sticks.
export const COLOURS_KEY='sopa-companion:cluster-colours:v1';
const write=(node,text)=>{if(node.textContent!==text)node.textContent=text;};   // an unchanged text causes no layout
/** The count a result's summary reads (the engine's `count`: "sampled" on a frame counted for a pause, the hold, a
 *  finish or the end; absent: "averaged"). */
const countOf=summary=>summary?.count==='sampled'?'sampled':'averaged';
const finite=v=>Array.isArray(v)&&v.length>=2&&v.every(Number.isFinite);
const count=n=>Number.isFinite(n)?n.toLocaleString('en-GB'):'?';
/** UmapView.setData input. The worker's own aggregates (cluster centroids: medians in UMAP space; bounds: 2-98 % of the
 * embedding) are passed when present so the view never sorts every point for them. A cluster without a centroid has
 * no analysed point, hence no label (a label without an anchor would make the view compute every centroid itself). */
function viewData(result) {
  const summary=result.summary,cs=summary.clusters||[],colors=[];
  for(const c of cs)colors[c.id]=clusterColor(c.color??c.id);
  const data={...result,palette:id=>colors[id],labels:cs.map(c=>({id:c.id,text:c.label||(c.top||[]).slice(0,3).map(g=>g.gene).join(' · ')}))};
  if(cs.some(c=>c.centroid!==undefined)){
    data.centroids=new Map(cs.filter(c=>finite(c.centroid)).map(c=>[c.id,[c.centroid[0],c.centroid[1],c.centroid[2]??0]]));
    data.labels=data.labels.filter(l=>data.centroids.has(l.id));
  }
  const b=summary.bounds;if(b&&finite(b.x)&&finite(b.y))data.bounds=b;
  return data;
}

// Frame display never awaits this controller. Results and requests are run-scoped.
export class Analysis {
  constructor(stage, panel, context, select) {
    Object.assign(this,{stage,panel,context,select});this.results=new Map();this.hidden=new Set();this.epoch=0;this.appliedAt=-Infinity;this.hoverLabel=0;this.hoverCluster=null;this.selectedLabel=0;
    this.coloured=false;this.supported=null;this.missing=[];this.problem=null;this.shownNote='';this.available=[];this.blocked=null;
    // The engine's describe came before its worker said where it runs (`backend` null): its limits are the GPU's
    // until then (Genes used stops lower on the CPU), so they are read again with the run's first result (event).
    this.limitsEarly=false;
    this.covered=false;this.held=false;   // Compare's full-screen view over the page (setCovered): what would be applied waits
    // The run's settings generation as the engine says it (null until known), the generation whose first result the
    // pane waits for after a change (null: none), and the generation whose result went missing once (fetchResult).
    this.gen=null;this.awaiting=null;this.missed=null;
    this.notebook=false;this.setupShown=false;   // a notebook's session (main.js sets it once known): where SETUP says it is set
    // Whether Clusters & UMAP can run on this engine at all (/health `analysis`, main.js sets it once known): a run
    // opens with its pane only when it can be clustered, so the pane never flashes open and shut on such an engine.
    this.engineCan=undefined;
    // Layers > Cells: the clusters' legend, shown once there are clusters (there is no switch to show), or one calm
    // line when the run cannot be clustered (UNAVAILABLE).
    this.controls=document.createElement('div');this.controls.className='analysis-controls';this.controls.hidden=true;
    this.legend=document.createElement('div');this.note=document.createElement('p');this.note.className='muted small analysis-note';this.note.hidden=true;
    this.controls.append(this.legend,this.note);panel.analysisControls=this.controls;panel.render();
    const tissue=document.querySelector('.stage-wrap');this.split=document.createElement('div');this.split.className='view-split';
    tissue.before(this.split);this.split.append(tissue);
    this.divider=document.createElement('div');this.divider.className='umap-divider';this.divider.tabIndex=0;this.divider.setAttribute('role','separator');this.divider.setAttribute('aria-label','Resize UMAP');
    this.pane=document.createElement('section');this.pane.className='umap-pane';
    const head=document.createElement('div');head.className='umap-head';head.append('UMAP');
    // Looking back, clusters of another iteration than the frame on screen (the nearest with proseg's count) say which.
    this.snapped=document.createElement('span');this.snapped.className='muted small umap-snapped';this.snapped.hidden=true;
    this.snapped.style.cssText='margin:0 auto 0 8px;cursor:help';head.append(this.snapped);
    this.collapse=document.createElement('button');this.collapse.textContent='Collapse';head.append(this.collapse);
    this.canvas=document.createElement('canvas');this.canvas.setAttribute('aria-label','UMAP');
    // While the pane has no clusters to show, one plain sentence says why (hint()).
    this.empty=document.createElement('p');this.empty.className='muted small umap-empty';this.empty.hidden=true;
    this.empty.style.cssText='position:absolute;left:12px;right:12px;top:45%;margin:0;text-align:center;pointer-events:none';
    this.tip=document.createElement('div');this.tip.className='umap-tip';
    // Advanced settings under the head: how the cells are grouped and laid out (analysis-settings.mjs). Reading storage
    // can throw where site data is blocked: the disclosure then forgets whether it was open.
    let storage=null;try{storage=globalThis.localStorage;}catch{}
    this.settings=new ClusterSettings({document,storage,onApply:values=>this.applySettings(values),
      onSlow:()=>pageLog.add('clusters','Clusters & UMAP: no result for the new settings after 2 minutes')});
    this.pane.append(head,this.settings.root,this.canvas,this.empty,this.tip);this.split.append(this.divider,this.pane);
    this.width=360;try{this.width=Number(localStorage.getItem('sopa-companion:umap-width'))||360;}catch{}
    this.collapse.onclick=()=>{this.collapsed=!this.collapsed;this.resize();};
    this.divider.onpointerdown=e=>{this.divider.setPointerCapture(e.pointerId);this.drag=true;};
    this.divider.onpointermove=e=>{if(this.drag){this.width=this.split.getBoundingClientRect().right-e.clientX;this.collapsed=false;this.resize();}};
    this.divider.onpointerup=()=>{this.drag=false;try{localStorage.setItem('sopa-companion:umap-width',this.width);}catch{}};
    this.divider.onpointercancel=()=>{this.drag=false;};
    this.divider.onkeydown=e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();this.width+=e.key==='ArrowLeft'?20:-20;this.resize();}};
    pageLog.provide('Clusters & UMAP',()=>this.logState());
    this.resize();
    window.addEventListener('resize',()=>this.resize());
  }
  resize() {
    // Too few cells with changed settings keeps the pane (its settings stay in reach); at the defaults it closes.
    const tuned=this.blocked==='too_few_cells'&&!!this.settings?.changed;
    this.pane.hidden=this.divider.hidden=!this.enabled||(!!this.blocked&&!tuned);
    this.width=Math.max(180,Math.min(this.width,Math.max(180,this.split.clientWidth*.65)));
    this.pane.style.width=(this.collapsed?100:this.width)+'px';this.canvas.hidden=!!this.collapsed;
    this.collapse.textContent=this.collapsed?'Expand':'Collapse';this.hint();this.view?.resize();
    this.settings?.show(this.enabled&&!this.collapsed&&(!this.blocked||tuned));
  }
  /** A run's view: Clusters & UMAP are on at once (the pane opens with the run, unless the engine said they cannot run
   *  there: engineCan); refresh() closes the pane if the engine cannot analyse, a tool switched the run's analysis off,
   *  or the run cannot be clustered (block). `kept`: the constants this kind of run keeps from before proseg's count
   *  (run.constants.kept: the settings' tooltips say so). */
  reset(run=null,{kept=null}={}) {
    this.epoch++;this.controller?.abort();this.metaController?.abort();this.statsController?.abort();this.stats=null;this.statsRequested=null;this.stopTimers();this.appliedAt=-Infinity;
    this.run=run;this.enabled=!!run&&this.engineCan!==false;this.supported=this.engineCan===false?false:null;this.problem=null;this.shownNote='';this.blocked=null;
    this.results.clear();this.available=[];this.hidden.clear();this.applied=null;this.pending=null;this.wanted=null;this.hoverLabel=0;this.hoverCluster=null;this.selectedLabel=0;this.held=false;
    this.gen=null;this.awaiting=null;this.missed=null;this.settings?.reset();this.settings?.setKept?.(run?kept:null);
    this.dropLegend();this.controls.hidden=true;this.note.hidden=true;this.badge(-1,true);
    this.tissue(null);this.stage.set({clusterHover:-1});this.view?.dispose();this.view=null;this.resize();this.setupLine();
    if(run)this.refresh();
  }
  stopTimers(){clearTimeout(this.timer);this.timer=null;clearTimeout(this.applyTimer);this.applyTimer=null;clearTimeout(this.soonTimer);this.soonTimer=null;}
  /** The pane's sentence while it shows no clusters: the first ones are on their way, or the analysis failed and the
   *  log says why. Written only on a change; nothing while clusters show or the pane is collapsed. */
  hint() {
    const text=!this.enabled||this.collapsed||this.applied||this.blocked?'':this.problem?'The clusters could not be computed. The Log button (top right) saves the details.':'Clusters appear here once the first iteration is analysed.';
    write(this.empty,text);if(this.empty.hidden!==!text)this.empty.hidden=!text;
  }
  /** An analysis that failed: into the log, and said in the pane while it has no clusters to show. */
  fail(message) {
    this.problem=String(message);pageLog.add('clusters',`Clusters & UMAP failed: ${this.problem}`);this.hint();
  }
  /** The run cannot be clustered for `reason` (UNAVAILABLE): no pane, no requests, one calm line under Layers > Cells;
   *  null lifts it (a result came after all: the region holds enough cells now). */
  block(reason) {
    const next=reason&&UNAVAILABLE[reason]?reason:null;
    if(next===this.blocked)return;
    this.blocked=next;
    if(next){const awaited=this.awaiting!=null;
      pageLog.add('clusters',`Clusters & UMAP cannot run for this run: ${next.replace(/_/g,' ')}`);this.stopTimers();this.tissue(null);this.applied=null;this.badge(-1,true);this.dropLegend();this.awaiting=null;
      // With changed settings the pane stays: its dots fade out, and the disclosure opens to say why (resize keeps it).
      // Otherwise the pane closes, and settings just applied (Reset among them) leave nothing to wait for: the buttons
      // are free again.
      if(next==='too_few_cells'&&this.settings?.changed){this.view?.setData({clusters:new Int16Array(),coords:new Float32Array()});this.settings?.tooFew();}
      else if(awaited)this.settings?.done();}
    const text=next?UNAVAILABLE[next]:'';
    write(this.note,text);this.note.hidden=!text;this.controls.hidden=!text&&!this.applied;
    // A run without proseg's own count: why a new run has it, or how to update an old proseg-live, in the line's
    // tooltip (main.js sets prosegFix from /health, and notebook for a notebook's session).
    const tip=unavailableTooltip(next,{prosegFix:this.prosegFix,notebook:this.notebook});if(this.note.title!==tip)this.note.title=tip;
    this.resize();
  }
  /** A result has arrived: shown with the next snapshot while following a live run, at once otherwise. */
  soon() {
    if(!this.context().live){this.frame();return;}
    if(this.soonTimer==null){const epoch=this.epoch;this.soonTimer=setTimeout(()=>{this.soonTimer=null;if(epoch===this.epoch)this.frame();},WITH_SNAPSHOT_MS);}
  }
  async json(path='',body,signal) {
    const response=await fetch(`/api/v1/runs/${encodeURIComponent(this.run)}/analysis${path}`,{signal,method:body===undefined?'GET':'POST',headers:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body)});
    if(!response.ok){const error=Error(`Analysis: HTTP ${response.status}`);error.status=response.status;throw error;}return response.json();
  }
  /** The engine's word on this run: whether it can analyse at all (and, where it cannot, the packages its Python
   *  lacks), whether the run's analysis is on (always, unless a tool switched it off) and the frames analysed so far. */
  async refresh() {
    const epoch=this.epoch;this.metaController?.abort();const c=this.metaController=new AbortController();
    try {const meta=await this.json('',undefined,c.signal);if(epoch!==this.epoch||c.signal.aborted)return;
      // Another settings generation than the one shown (another window changed them, or a restarted engine: even a
      // lower one) replaces what the page holds; else the settings fill the disclosure. An engine that says no
      // generation leaves it unknown: nothing is then filtered or asked by generation.
      const g=meta.generation;
      if(Number.isInteger(g)&&this.gen!=null&&g!==this.gen)this.newGeneration(meta,true);
      else{if(this.gen==null&&Number.isInteger(g))this.gen=g;this.settings?.load(meta);}
      this.available=meta.seqs||[];this.supported=meta.available!==false;this.missing=Array.isArray(meta.missing)?meta.missing:[];this.limitsEarly=meta.backend===null;
      if(!this.supported)pageLog.add('clusters',`Clusters & UMAP are not available: ${this.missing.length?'this Python environment lacks '+this.missing.join(', '):'the analysis worker cannot run (the engine log says why)'}`);
      else if(meta.enabled===false)pageLog.add('clusters','Clusters & UMAP are switched off for this run (through the engine API)');
      const on=this.supported&&meta.enabled!==false;
      this.block(on?meta.unavailable??null:null);   // first: nothing is asked of a run that cannot be clustered
      this.setEnabled(on);this.setupLine();
    }catch(e){if(epoch===this.epoch&&e.name!=='AbortError')this.fail(e.message);}
  }
  /** An engine where Clusters & UMAP cannot run: one line under Layers > Cells says what is missing (the packages, when
   *  the engine names them: missingLine) or where it is set (SETUP), while the pane stays shut; nothing once a run can
   *  be clustered, or with no run open. */
  setupLine() {
    const text=this.run&&this.supported===false?(missingLine(this.missing)??SETUP[this.notebook?'notebook':'desktop']):'';
    if(!text&&!this.setupShown)return;
    this.setupShown=!!text;
    if(!text&&this.blocked)return;   // the line under Layers > Cells is block()'s now: why this run cannot be clustered
    write(this.note,text);this.note.hidden=!text;this.controls.hidden=!text&&!this.applied;if(this.note.title)this.note.title='';
  }
  /** The run's settings changed (the engine's push or a POST's reply, else a describe: `any`, which also takes a lower
   *  generation, as a restarted engine's): what is held of the earlier settings is dropped, the download in flight
   *  is aborted, and the clusters on screen stay until the first result of the new ones (present()). The generation
   *  held only fills the disclosure; an earlier one (a push or a reply that a later change overtook) says nothing of
   *  the run's settings now, so nothing changes. */
  newGeneration(meta,any=false) {
    const g=meta?.generation;
    if(!Number.isInteger(g))return;
    if(this.gen!=null&&!any&&g<this.gen)return;
    if(this.gen!=null&&g===this.gen){this.settings?.load(meta);return;}
    this.gen=g;this.controller?.abort();this.pending=this.wanted=null;this.results.clear();this.available=[];this.problem=null;this.awaiting=g;
    this.settings?.load(meta);
    if(this.blocked==='too_few_cells')this.block(null);   // a change lifts the hold: these settings may cluster
    this.frame();
  }
  /** Apply or Reset (analysis-settings.mjs): the settings go to the engine with the frame on screen when the run is not
   *  followed live (that frame is computed first); following live, the next automatic pass takes them at once. */
  async applySettings(values) {
    const epoch=this.epoch,before=this.gen,{seq,live}=this.context();
    let reply;
    try{reply=await this.json('',{params:values,...(live||seq<0?{}:{seq})});}
    catch(e){
      if(epoch!==this.epoch)return;
      this.settings?.refused();pageLog.add('clusters',`Clusters & UMAP settings were not taken: ${e.status?`HTTP ${e.status}`:e.message}`);return;
    }
    if(epoch!==this.epoch)return;
    // The buttons stay busy while the first clusters of the run's new settings are awaited; with no frame yet the
    // settings apply from its first clusters.
    const g=reply?.generation,settled=()=>{if(seq<0||this.awaiting==null)this.settings?.done();};
    // Another change (another window, a tool) came after this one: the page holds its settings already.
    if(Number.isInteger(g)&&this.gen!=null&&g<this.gen){settled();return;}
    if(Number.isInteger(g)&&(before==null||g>before)){
      this.newGeneration(reply);    // the push may have come first: then this only fills the disclosure
      settled();
    }else{this.settings?.load(reply);this.settings?.unchanged();}
  }
  setEnabled(on) {
    this.enabled=on;
    if(on){this.ensureView();this.frame();}else this.off();
    this.resize();
  }
  ensureView() {
    if(this.view)return;
    this.view=new UmapView(this.canvas,{onHover:id=>this.hover(id),onSelect:id=>{this.select(id);this.selection(id===null?0:id+1);}});
    if(this.covered)this.view.setCovered(true);   // made under Compare: its first frame waits too
    if(this.hidden.size)this.view.hideClusters(this.hidden);
    this.selection(this.selectedLabel);
  }
  off(){
    this.controller?.abort();this.stopTimers();this.tissue(null);this.applied=null;this.badge(-1,true);this.controls.hidden=true;this.shownNote='';
    // Settings sent but not applied yet when the run's analysis goes off (a tool switched it off, the engine found it
    // cannot run): no result of them will come, so the buttons are free again at once, not after SLOW_MS.
    if(this.awaiting!=null){this.awaiting=null;this.settings?.done();}
  }
  event(event) {
    if(event.run!==this.run)return;
    // The engine found mid-session that it cannot run Clusters & UMAP (its worker's Python lacks what they need): it
    // says why in its describe, which the pane and the line under Layers > Cells follow.
    if(event.state==='unavailable'){this.refresh();return;}
    if(event.state==='ready'||event.state==='running')this.supported=true;   // the engine analyses: it has its worker
    if(event.state==='ready'&&this.limitsEarly){this.limitsEarly=false;this.refresh();}   // its limits, now it has said hello
    if(event.enabled!==undefined && event.enabled!==this.enabled && this.supported!==false){   // a tool switched it
      pageLog.add('clusters',`Clusters & UMAP switched ${event.enabled?'on':'off'} for this run (through the engine API)`);
      this.setEnabled(event.enabled);if(!this.enabled)return;
    }
    if(!this.enabled)return;
    if(event.state==='settings'){this.newGeneration(event);return;}   // changed here, in another window or by a tool
    // A push of earlier settings is dropped; one of later settings than the page knows asks the engine first. A push
    // without a generation counts as one of the current settings.
    if(Number.isInteger(event.generation)&&Number.isInteger(this.gen)){
      if(event.generation<this.gen)return;
      if(event.generation>this.gen){this.refresh();return;}
    }
    if(event.state==='error'){
      if(UNAVAILABLE[event.reason]){this.block(event.reason);return;}   // not a failure: this run cannot be clustered
      if(this.awaiting!=null)this.settings?.failed();   // the new settings failed: the clusters shown stay
      this.fail(event.summary?.error||event.error||event.message||'Analysis failed');return;
    }
    if(event.state!=='ready')return;                       // "running": nothing to show (the log has every result)
    if(this.blocked){this.block(null);this.ensureView();}  // a result after all: the region holds enough cells now
    if(!this.available.includes(event.seq))this.available.push(event.seq);
    // A frame counted again (the hold, a finish or the end after a token's count of it) is scored on another count: the
    // result held of the other one goes (the one on screen stays until its successor is in: stale), asked for again;
    // one on its way is held to the count named last when it comes (fetchResult), and one waiting its turn is asked for
    // on it, whether its frame is shown or not.
    const held=this.results.get(event.seq);
    if(held&&event.summary&&countOf(held.summary)!==countOf(event.summary)){if(held===this.applied)held.stale=true;else this.results.delete(event.seq);}
    if(event.summary&&event.seq===this.pending)this.heard={summary:event.summary,iter:event.iter};
    else if(event.summary&&event.seq===this.wanted?.seq)this.wanted={seq:event.seq,summary:event.summary,iter:event.iter};
    const {seq,live}=this.context();
    // Future results must never colour a historical frame, but the one shown for a frame before every analysed one.
    if(event.seq===this.shownSeq(this.available,seq,live))this.fetchResult(event.seq,event.summary,event.iter);
  }
  /** The analysed frame shown for frame `seq` among `seqs`: the newest at or before it; looking back (not following
   *  live) at a frame before every analysed one, the oldest after it (the engine answers a frame asked for with the
   *  nearest that has proseg's count). -1: none. */
  shownSeq(seqs,seq,live){const earlier=nearestEarlier(seqs,seq);return earlier>=0||live?earlier:nearestLater(seqs,seq);}
  frame() {
    clearTimeout(this.soonTimer);this.soonTimer=null;
    if(this.covered){this.held=true;return;}   // under Compare: applied once it closes (setCovered)
    try { this.updateFrame(); } catch(e) { this.fail(`the clusters view: ${e.message}`); }
  }
  /** Compare's full-screen view over the page (main.js cover()), or not. Covered, no result is applied (no flight on the
   *  UMAP, no new colours on the tissue, no legend change) and the UMAP draws nothing, while results keep arriving and
   *  are kept; uncovered (`catchUp`: unless another run or the picker opens instead), the newest for the frame on screen
   *  is applied once, and what the UMAP had pending is drawn. */
  setCovered(on,catchUp=true) {
    on=!!on;if(on===this.covered)return;
    this.covered=on;this.view?.setCovered(on);
    if(on)return;
    const held=this.held;this.held=false;
    if(held&&catchUp)this.frame();
  }
  updateFrame() {
    if(!this.enabled||!this.run||this.blocked)return;
    const {seq,live}=this.context();if(seq<0)return;
    this.present(seq,live);this.badge(seq,live);
    const shown=this.shownSeq(this.available,seq,live);
    if(shown>=0&&(!this.results.has(shown)||this.results.get(shown).stale))this.fetchResult(shown);
    if(live){clearTimeout(this.timer);this.timer=null;}
    // A frame not analysed yet is asked for once the engine said it can analyse (a request it would refuse is a notice).
    if(!live && this.supported && !this.results.has(seq)) {
      // A bounded debounce also services continuous replay (which changes every 71 ms).
      this.target=seq;
      if(!this.timer)this.timer=setTimeout(()=>{this.timer=null;const epoch=this.epoch;this.json('',{seq:this.target}).catch(e=>{if(epoch===this.epoch)this.fail(e.message);});},150);
    }
    // This runs several times per frame; the statistics of a frame are asked for once (the frame display shares them).
    if(this.stats?.seq!==seq)this.loadStats(seq);
  }
  /** Shows the newest result at or before the frame on screen (looking back at a frame before every analysed one, the
   *  oldest after it: shownSeq); live results of large runs wait their turn. */
  present(seq,live) {
    // Of the results held: the newest at or before the frame on screen; the oldest after it only when no analysed frame
    // comes before it (one that does is on its way: the later one is never shown meanwhile).
    const earlier=nearestEarlier(this.results.keys(),seq);
    const key=earlier>=0||live||nearestEarlier(this.available,seq)>=0?earlier:nearestLater(this.results.keys(),seq);
    const result=this.results.get(key)||null,applied=this.applied;
    // After a change of settings the clusters shown stay until the first result of the new ones, as long as they belong
    // to a frame at or before the one on screen (a frame looked at earlier than them shows none until its own comes).
    if(!result&&this.awaiting!=null&&applied&&applied.seq<=seq)return;
    const wait=result&&applied&&result!==applied&&applied.seq<=seq?applyDelay(result.clusters.length,live,performance.now()-this.appliedAt):0;
    if(wait>0) {
      // One timer at most, and it applies nothing itself: once the window has passed the result rides on the next
      // snapshot (soon(): main.js calls frame() for it, or WITH_SNAPSHOT_MS later without one), so recolouring the
      // tissue and the UMAP never costs a presented frame of its own. Whatever is newest then is applied: no backlog.
      if(this.applyTimer==null){const epoch=this.epoch;this.applyTimer=setTimeout(()=>{this.applyTimer=null;if(epoch===this.epoch)this.soon();},wait);}
    } else {clearTimeout(this.applyTimer);this.applyTimer=null;this.apply(result);}
    // A frame looked at (paused, scrubbed) that shows an earlier frame's clusters: noted for the log once per change.
    // While following live the clusters are always a little behind the newest frame: nothing to note.
    const note=!live&&this.applied&&this.applied.seq!==seq?`iteration ${this.context().frames?.[seq]?.iter??'?'} shows the clusters of iteration ${this.applied.summary.iter}`:'';
    if(note!==this.shownNote){this.shownNote=note;if(note)pageLog.add('shown',note);}
  }
  /** The pane's badge: looking back (not following live, where the clusters are always a little behind the newest
   *  frame), clusters of another frame than the one on screen name their iteration (snappedBadge), why in its tooltip;
   *  nothing otherwise. Written only on a change. */
  badge(seq,live) {
    if(!this.snapped)return;
    const iter=!live&&this.applied&&this.applied.seq!==seq?this.applied.summary?.iter:null,i=Number.isFinite(iter)?number(iter):null;
    const text=i==null?'':copy('snappedBadge',{i}),title=i==null?'':copy('snappedTooltipClusters',{i});
    write(this.snapped,text);if(this.snapped.hidden!==!text)this.snapped.hidden=!text;if(this.snapped.title!==title)this.snapped.title=title;
  }
  async fetchResult(seq,summary,iter) {
    // A result held of another count than the frame's now (stale) is asked for again; one on its way is held to the
    // count the engine named last for its frame when it comes (heard: event()).
    if(this.pending===seq||this.results.has(seq)&&!this.results.get(seq).stale)return;
    // Single flight: results arrive several times a second, so aborting the download in progress for each newer one
    // would starve the view. The latest request waits and runs next; run switches still abort via the epoch.
    if(this.pending!==null&&this.pending!==undefined){this.wanted={seq,summary,iter};return;}
    // Asked for the settings generation the page holds (when it knows it): a result of earlier settings is never
    // fetched, and one downloaded while the settings changed is dropped.
    const c=this.controller=new AbortController(),epoch=this.epoch,gen=this.gen,query=Number.isInteger(gen)?`?generation=${gen}`:'';this.pending=seq;this.heard=null;
    try {
      if(!summary)summary=await this.json(`/${seq}${query}`,undefined,c.signal);
      if(gen!==this.gen)return;
      const response=await fetch(`/api/v1/runs/${encodeURIComponent(this.run)}/analysis/${seq}/cells${query}`,{signal:c.signal});
      if(!response.ok){const error=Error(`Analysis cells: HTTP ${response.status}`);error.status=response.status;throw error;}
      const data=decodeAnalysis(await response.arrayBuffer());if(epoch!==this.epoch||c.signal.aborted||!this.enabled||gen!==this.gen)return;
      // Counted again while on its way (the hold, a finish or the end): what came reads the count it had, so it is not
      // kept; asked for again at once when it is the frame's clusters to show, else once it is (updateFrame).
      const heard=this.heard;
      if(heard&&countOf(heard.summary)!==countOf(summary)){
        const {seq:at,live}=this.context();
        if(this.shownSeq(this.available,at,live)===seq)this.wanted={seq,...heard};
        return;
      }
      const result={...data,seq,generation:gen,summary:{...summary,iter:iter??summary.iter??this.context().frames[seq]?.iter??seq}};
      this.results.set(seq,result);this.problem=null;
      // Bound retained typed arrays; server index remains available for older scrubs.
      if(this.results.size>12)this.results.delete(this.results.keys().next().value);
      this.soon();
    }catch(e){
      if(epoch!==this.epoch||e.name==='AbortError'||gen!==this.gen)return;   // earlier settings' result: dropped quietly
      // Missing in the current generation: the page's word on it may be stale (a restarted engine): asked once.
      if(e.status===404&&Number.isInteger(gen)&&this.missed!==gen){this.missed=gen;this.refresh();return;}
      this.fail(e.message);
    }
    finally{
      // What waits is this flight's to hand on while it is the one in progress (an aborted one's successor has its own):
      // asked for when its result is missing or stale (a result of another count than an event named is never kept
      // fresh: event() and the check above).
      if(this.controller===c){
        this.pending=null;
        const next=this.wanted;this.wanted=null;
        if(next&&epoch===this.epoch&&this.enabled&&(!this.results.has(next.seq)||this.results.get(next.seq).stale))
          this.fetchResult(next.seq,next.summary,next.iter);
      }
    }
  }
  apply(result) {
    if(this.applied===result)return;this.applied=result;this.appliedAt=performance.now();this.tissue(result);
    if(!result){this.dropLegend();this.view?.setData({clusters:new Int16Array(),coords:new Float32Array()});this.controls.hidden=true;}
    else {
      harness?.count('analysisApplied');
      if(!this.coloured)this.clusterColours();
      this.ensureView();this.view.setData(viewData(result));this.legendFor().update(result.summary);if(this.controls.hidden)this.controls.hidden=false;
      // Each result's step times (the line that used to show under the legend) are in the engine's part of the log.
      this.settings?.resultShown();
      if(this.awaiting!=null&&(result.generation??this.gen)>=this.awaiting){   // the first result of new settings
        this.awaiting=null;this.settings?.done();pageLog.add('clusters',`Clusters & UMAP settings applied: ${this.settings?.logLine()||'the defaults'}`);
      }
    }
    this.hint();
    // The hovered cell may have changed cluster, and tooltips and the cell card name the cluster.
    this.tissueHover(this.hoverLabel);if(this.view?.hover!=null)this.describe(this.view.hover);this.onReadoutChange?.();
  }
  /** Once per browser (COLOURS_KEY): cells and transcripts in their identity colours turn to cluster colours with the
   *  first clusters; the layer panel saves that, and any later choice. Mixing colours are never replaced. */
  clusterColours() {
    this.coloured=true;
    try{if(localStorage.getItem(COLOURS_KEY))return;localStorage.setItem(COLOURS_KEY,'1');}catch{/* no storage: once per page */}
    const o=this.stage.options||{},next={};
    if((o.cellColorBy??'identity')==='identity')next.cellColorBy='cluster';
    if((o.transcriptColorBy??'cell')==='cell')next.transcriptColorBy='cluster';
    if(Object.keys(next).length){this.stage.set(next);this.panel.render();this.panel.save?.();}
  }
  tissue(result) {
    // The tissue's cluster lookup is a texture upload and a redraw: only for a result the stage does not show yet.
    if(this.stageResult===result)return;
    this.stageResult=result;this.stage.setAnalysis(result);if(result)this.stage.clusterGPU?.setHidden(this.hidden);
  }
  legendFor() {
    return this.legendView||=createLegend(this.legend,{hidden:this.hidden,onToggle:()=>this.filters(),
      onHover:id=>{this.stage.set({clusterHover:id??-1});this.view?.highlightCluster(id);}});
  }
  dropLegend(){this.legendView?.dispose();this.legendView=null;this.legend.replaceChildren();}
  filters(){this.stage.clusterGPU?.setHidden(this.hidden);this.stage.dirty=true;this.view?.hideClusters(this.hidden);}
  tissueHover(label){
    // Hovering a cell lights up its whole cluster on the UMAP (one point is too small to find) and its legend row.
    this.hoverLabel=label;
    const cluster=label&&this.applied?this.applied.clusters[label-1]:-1,id=cluster>=0?cluster:null;
    if(id===this.hoverCluster)return;
    this.hoverCluster=id;if(this.view?.highlight!=null)this.view.highlightCells(null);this.view?.highlightCluster(id);
    this.legendView?.highlight(id);
  }
  selection(label){
    // Called on every cell-card refresh: only a different selection (or a new view) wakes the UMAP.
    this.selectedLabel=label||0;
    if(!this.view||this.selectedView===this.view&&this.selectedShown===this.selectedLabel)return;
    this.selectedView=this.view;this.selectedShown=this.selectedLabel;this.view.select(this.selectedLabel?this.selectedLabel-1:null);
  }
  hover(id){
    const label=id===null?0:id+1;if(this.stage.options?.hover!==label)this.stage.set({hover:label});
    this.describe(id);
  }
  describe(id){
    if(id==null){write(this.tip,'');return;}
    const c=this.applied?.clusters[id],cluster=this.applied?.summary.clusters?.find(x=>x.id===c),ctx=this.context();
    // The frame on screen's statistics: the store's own (which take the frame's count when it comes after them), else
    // those loaded here.
    const current=this.statsStore?.current,stats=current?.seq===ctx.seq?current:this.stats?.seq===ctx.seq?this.stats:null,kept=keptOf(stats,id);
    const values={cell:ctx.cells[id]||`cell ${id}`,cluster:cluster?.label||'Unanalysed',n:kept==null?'…':count(Math.round(kept)),count:kept==null?UMAP_COUNT:countName(stats)};
    write(this.tip,UMAP_TIP.replace(/\{(\w+)\}/g,(_,name)=>values[name]));
  }
  async loadStats(seq) {
    const stats=await this.statsStore?.load(seq);
    if(stats?.seq===this.context().seq)this.stats=stats;
    if(this.view?.hover!=null)this.describe(this.view.hover);
    return stats;
  }
  /** The log's line on Clusters & UMAP (log-export.mjs asks when the log is written). */
  logState() {
    if(!this.run)return 'no run open';
    if(this.supported===false)return `not available: ${this.missing.length?'this Python environment lacks '+this.missing.join(', '):'the analysis worker cannot run'}`;
    if(!this.enabled)return 'switched off for this run (through the engine API)';
    if(this.blocked)return `cannot run: ${UNAVAILABLE[this.blocked]}`;
    const {seq,frames}=this.context(),a=this.applied;
    const shown=a?`showing the clusters of iteration ${a.summary.iter}${a.seq!==seq?` (iteration ${frames?.[seq]?.iter??'?'} is on screen)`:''}, ${count(a.summary.clusters?.length)} clusters`:'no clusters shown yet';
    const settings=this.settings?.logLine?.()||'';
    return `on; ${shown}; ${this.results.size} results held by the page, ${this.available.length} iterations analysed`+(this.problem?`; last failure: ${this.problem}`:'')
      +(settings?`; settings: ${settings}`:'')+(this.awaiting!=null?"; the new settings' clusters are not shown yet":'');
  }
}
