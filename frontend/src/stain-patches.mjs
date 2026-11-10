import { problem, guard, busy, clearExplanation, EXPLAIN_TEXT } from './explain.mjs';
import { PatchFolderDialog } from './patch-folder.mjs';
import { boxInPolygon } from './region.mjs';
import { levelName, resolutionName } from './stain.mjs';

export const STAIN_PATCH_TEXT = Object.freeze({
  // A tool inside Stain segmentation's picker: its section, and Start's label while it is open.
  title:'Training patches', start:'Make training patches',
  toolHelp:'Suggested boxes appear on the image. Correct them in the Cellpose GUI to train a model.',
  startFrom:'Start from', model:'Model outlines', blank:'Empty',
  directHelp:'Choose a region, then make training patches.', tooSmall:'Drag across the image to draw a box at least 16 pixels wide and tall.',
  fullHelp:'Patches will be chosen across the sample image.', delete:'Delete this patch item? Exported files will be kept.', count:'Patches', size:'Size (pixels)', propose:'Make training patches',
  add:'Add a box', export:'Export patches', open:'Open in Cellpose GUI', again:'Open again', running:'Cellpose is open',
  hint:'Drag a box to move it or a corner to resize it. Click × to drop it, or choose Add a box and drag. Right-drag pans.',
  large:'Large patches can make the Cellpose GUI slow.',
  sizeHelp:'Choose 1–64 patches and a size from 64 to 4096 pixels that fits inside the region.',
  help:'Correct the outlines in Cellpose and save (Ctrl+S). To train, choose Models > Train new model in Cellpose. Then start a new run here and choose the trained model under Your models.',
  preparing:'Making patches…', empty:'Make patches after pausing or finishing the run.',
  fallback:'No desktop display is available. Open the folder with this command:',
  // The GUI this engine started closed with an error (GET …/patches exported.gui_error): the last line of its log.
  guiClosed:'The Cellpose GUI closed with an error: {reason}', guiClosedPlain:'The Cellpose GUI closed with an error.',
  install:'The Cellpose GUI is not installed where Cellpose runs. Install it with this command, then open it again:',
  changed:'The boxes changed. The latest boxes are now shown; try again.',
  error:'The patches could not be loaded. Try opening the run again.',
  // What a press says instead of a greyed-out button (explain.mjs), and the labels of a button at work.
  countRange:'a whole number from 1 to 64', sizeRange:'a whole number from 64 to 4096',
  sizeFits:'at most {largest}, so each patch fits inside the region',
  regionSmall:'this region is less than 64 pixels across, too small for patches',
  loading:'The patches are still loading. Try again in a moment.', noBoxes:'Make training patches or add a box first.',
  exportFirst:'Export patches first.', guiOpen:'The Cellpose GUI is already open on this folder.',
  suggesting:'Suggesting…', exporting:'Exporting…', opening:'Opening…',
  // Where the last export went (its folder, chosen in the export dialog: patch-folder.mjs), or that it is gone.
  savedIn:'Saved in {folder}', gone:'The exported patches are no longer in {folder}. Export them again to go on.',
  // The boxes as a list the keyboard reaches (the image's boxes take the pointer only): a row a box, with its ×.
  rows:'Patch boxes', row:'Patch {n} · {w} × {h} px', drop:'Drop patch {n}',
  keys:'Arrow keys move it, by 1 pixel with Alt. Shift and arrow keys change its size. Delete drops it.',
});
export const canReviewStain = run => run?.kind==='patches' || run?.kind==='stain' && run.progress?.done>0 &&
  (run.state==='paused' && run.paused && run.ack?.state==='paused' || ['done','stopped','failed'].includes(run.state));
const text=(node,value)=>{if(node.textContent!==value)node.textContent=value;};
const attr=(node,key,value)=>{if(node.getAttribute(key)!==String(value))node.setAttribute(key,value);};
const svgNode=tag=>document.createElementNS('http://www.w3.org/2000/svg',tag);

export const pixelPoint = (world,geometry) => world.map((p,i)=>Math.round((p-geometry[i])/geometry[i+2]));
/** A patch's size across in µm at the level the patches are cut at (its SpatialData name: stain.mjs resolutionName); none
 *  on a sample whose transcripts seem to be in pixels (`pixelFrame`: its µm are pixels there). */
export function patchSizeNote(size, resolution, {pixelFrame=false}={}) {
  if(pixelFrame || !(resolution?.um_per_px>0) || !(size>0))return '';
  return `${size} px = ${Math.round(size*resolution.um_per_px)} µm across at ${resolutionName(resolution) || levelName(0)}`;
}
const clamp=(v,lo,hi)=>Math.min(Math.max(v,lo),hi);
/** The box a gesture leaves: `box` moved (corner -1) or one corner pulled, by as far as the pointer went from `from` to
 *  `to` (so a press without a move changes nothing), or a new box between the two points (box null). With `frame`
 *  ({extent: [width, height] or null, outline or null}: the boxes' pixel window and, for a lasso, its outline in the same
 *  pixels) a box stops at the window's edges, a corner never makes a side under 16 pixels, and a gesture that would take
 *  a box wholly inside the lasso partly out of it leaves the box where it was (boxInPolygon: the rule Start and the
 *  engine apply). A box already past an edge is not pushed back by a press. */
export function boxGesture(box,from,to,corner=-1,frame=null) {
  const edge=frame?frame.extent||[Infinity,Infinity]:null;
  if(!box) {
    const [a,b]=edge?[from,to].map(p=>[clamp(p[0],0,edge[0]),clamp(p[1],0,edge[1])]):[from,to];
    return [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];
  }
  const lo=edge?[Math.min(0,box[0]),Math.min(0,box[1])]:[-Infinity,-Infinity];
  const hi=edge?[Math.max(edge[0],box[2]),Math.max(edge[1],box[3])]:[Infinity,Infinity];
  const d=[to[0]-from[0],to[1]-from[1]];
  let next;
  if(corner>=0) {
    const anchor=[box[corner%2 ? 0 : 2],box[corner<2 ? 3 : 1]],own=[box[corner%2 ? 2 : 0],box[corner<2 ? 1 : 3]];
    const p=[0,1].map(i=>{
      const v=clamp(own[i]+d[i],lo[i],hi[i]);
      const q=anchor[i]+(v<anchor[i]?-1:1)*Math.max(16,Math.min(4096,Math.abs(v-anchor[i])));
      return q>=lo[i] && q<=hi[i]?q:anchor[i]+Math.sign(own[i]-anchor[i])*16;   // no room past the anchor: smallest side
    });
    next=[Math.min(anchor[0],p[0]),Math.min(anchor[1],p[1]),Math.max(anchor[0],p[0]),Math.max(anchor[1],p[1])];
  } else {
    const dx=clamp(d[0],lo[0]-box[0],hi[0]-box[2]),dy=clamp(d[1],lo[1]-box[1],hi[1]-box[3]);
    next=[box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy];
  }
  const outline=frame?.outline;
  return outline && boxInPolygon(box,outline) && !boxInPolygon(next,outline)?box:next;
}
// What a box offers on screen (CSS px): its corners resize it once it is three handle radii across, and its × (a
// DROP-sized corner inside its top right) drops it once that leaves most of the box to take hold of. A smaller box is
// moved by a press anywhere inside it.
const HANDLE=9,DROP=[24,27];

export class StainPatches {
  constructor(stage,api,{notice}={}) {
    Object.assign(this,{stage,api,notice});
    this.exportPollMs=2000;                  // how often a running export is asked how far it is (exportTo)
    const T=STAIN_PATCH_TEXT;
    // Export patches asks every time where this export goes (patch-folder.mjs); the dialog posts through exportTo.
    this.folders=new PatchFolderDialog(api,{notice});
    const make=(tag,value,id)=>{const n=document.createElement(tag);if(value)n.textContent=value;if(id)n.id=id;return n;};
    this.panel=document.getElementById('stain-patches');
    this.panel.append(make('h3',T.title));
    const field=(label,type,id,value)=>{
      const row=make('label'),input=make(type==='select'?'select':'input',null,id);
      if(type!=='select')input.type=type;
      input.value=value;row.append(make('span',label),input);this.panel.append(row);return input;
    };
    this.count=field(T.count,'number','stain-patch-count','8');this.count.min='1';this.count.max='64';
    this.size=field(T.size,'number','stain-patch-size','256');this.size.min='64';this.size.max='4096';
    this.sizeNote=make('p',T.large,'stain-patch-size-note');this.sizeNote.hidden=true;this.panel.append(this.sizeNote);
    this.resolutionNote=make('p','', 'stain-patch-resolution');this.panel.append(this.resolutionNote);
    this.size.oninput=this.count.oninput=()=>this.render();
    // Never greyed out for what is missing: a press says what (explain.mjs). Only a running request keeps one busy.
    const button=(label,id,check,act)=>{const n=make('button',label,id);n.type='button';guard(n,check,act);this.panel.append(n);return n;};
    this.propose=button(T.propose,'stain-patch-propose',()=>this.proposeProblems(),()=>this.suggest());
    this.add=button(T.add,'stain-patch-add',()=>this.problems(),()=>{this.adding=!this.adding;this.render();});
    this.export=button(T.export,'stain-patch-export',()=>this.problems(()=>this.data.patches?.length?[]:[problem(null,'',T.noBoxes)]),
      ()=>this.folders.show({run:this.run,version:this.data.version,
        exportTo:(folder,version,onProgress)=>this.exportTo(folder,version,onProgress),
        words:{changed:T.changed,failed:T.error}}));
    // A GUI without its packages is refused with the command that installs them: kept under the button to copy.
    this.open=button(T.open,'stain-patch-open',()=>this.openProblems(),()=>this.action(this.open,T.opening,
      ()=>this.api.stainPatchOpen(this.run.id,{set:this.data.set}).catch(e=>{this.install=e.data?.command||null;throw e;}),
      result=>{this.install=null;this.launch=result;this.render();this.load();}));
    // The boxes by keyboard (renderRows): the open run's or item's under the hint, the picker's preview in its Training
    // patches section.
    const rows=id=>{const n=make('div',null,id);n.className='patch-rows';n.setAttribute('role','list');
      n.setAttribute('aria-label',T.rows);n.hidden=true;return n;};
    this.hint=make('p',T.hint);this.rows=rows('stain-patch-rows');this.panel.append(this.hint,this.rows,make('p',T.help));
    this.previewRows=rows('stain-patch-preview-rows');document.getElementById('patch-picker')?.append(this.previewRows);
    this.list=make('div',null,'stain-patch-list');this.panel.append(this.list);
    this.folder=make('p',null,'stain-patch-folder');this.command=make('pre',null,'stain-patch-command');
    this.panel.append(this.folder,this.command);
    this.settings=make('button','Use settings from Cellpose','stain-patch-settings');this.settings.type='button';
    this.settingsReason=make('p',null,'stain-patch-settings-reason');this.panel.append(this.settings,this.settingsReason);
    this.svg=svgNode('svg');this.svg.id='stain-patch-boxes';this.svg.setAttribute('aria-hidden','true');
    this.svg.setAttribute('preserveAspectRatio','none');stage.canvas.after(this.svg);
    this.boxes=svgNode('g');this.svg.append(this.boxes);
    this.ghost=svgNode('rect');this.ghost.setAttribute('vector-effect','non-scaling-stroke');this.svg.append(this.ghost);
    this.reset();this.events();
    stage.onChange?.(()=>{if(this.shown)this.move();});      // the picker moves only its own overlay with the camera
    globalThis.window?.addEventListener('focus',()=>this.load());
    // A stain run's first boxes wait until the user starts working in this panel, so boxes never cover results that are
    // only being looked at (training patches are asked for, never pushed on a run). A press of
    // one of its buttons does that button's work instead.
    const working=e=>{
      if(this.suggested || !this.active || this.run?.kind!=='stain')return;
      if(e.target?.closest?.('button')){this.suggested=true;return;}
      this.working=true;this.suggestFirst();
    };
    for(const type of ['pointerdown','focusin'])this.panel.addEventListener(type,working);
  }
  reset() {
    this.folders?.close(true);         // another run: the export dialog goes, and what it was waiting for shows nothing
    this.ticket=(this.ticket||0)+1;this.readTicket=(this.readTicket||0)+1;this.release?.();this.release=null;
    clearTimeout(this.timer);this.controller?.abort();this.run=null;this.data=null;this.launch=null;this.install=null;
    this.suggested=false;
    this.working=false;this.auto=false;
    this.busy=false;this.loading=false;this.adding=false;this.drag=null;this.active=false;this.boxKey=null;this.listKey=null;
    this.clearPreview();this.quiet();
    this.panel.hidden=true;this.svg.style.display='none';this.ghost.setAttribute('visibility','hidden');this.boxes.replaceChildren();
    this.renderRows();
  }
  update(run) {
    if(this.run?.id!==run?.id)this.reset();
    const was=this.active;this.run=run;this.active=canReviewStain(run);
    // A stain run shows the panel once it has started: before its Start it has no cells to cut patches from (the
    // picker's Training patches make them straight from the sample).
    this.panel.hidden=!(run?.kind==='patches' || run?.kind==='stain' && run.stain_started!==false);
    this.svg.style.display=this.active || this.shown?'block':'none';
    if(!this.active && !this.shown){this.drag=null;this.adding=false;this.ghost.setAttribute('visibility','hidden');}
    if(!this.data && !this.loading || this.active && !was)this.load();
    this.move();this.render();this.renderRows();
  }
  schedule() {clearTimeout(this.timer);if(this.run)this.timer=setTimeout(()=>this.load(),2000);}
  async load() {
    if(!this.run || this.busy || this.loading || typeof document!=='undefined' && document.hidden){this.schedule();return;}
    this.loading=true;const ticket=this.ticket,read=++this.readTicket,id=this.run.id;
    this.controller?.abort();this.controller=new AbortController();
    try {
      const data=await this.api.stainPatches(id,this.controller.signal);
      if(ticket!==this.ticket || read!==this.readTicket)return;
      this.apply(data);
    } catch(e) {if(ticket===this.ticket && e.name!=='AbortError')this.notice?.(STAIN_PATCH_TEXT.error);}
    finally {if(ticket===this.ticket && read===this.readTicket){this.loading=false;this.render();this.schedule();this.suggestFirst();}}
  }
  // A draft never suggested (version 0, no boxes) gets its boxes once by itself: a Training patches item's when its
  // panel opens, a stain run's once the user starts working in the panel. Never an explanation,
  // outline, focus or scroll by itself: with a problem it only counts as tried, and Make training patches says what is
  // wrong when pressed. A draft emptied on purpose has a later version.
  suggestFirst() {
    if(this.suggested || !this.active || this.busy || this.data?.version!==0 || this.data.patches?.length)return;
    if(this.run.kind==='stain' && !this.working)return;
    this.suggested=true;
    if(!this.proposeProblems().length)this.suggest(true);
  }
  /** Show the engine's draft; the fields take its count and size when it is a new version, unless `typed` (the boxes
   *  nobody pressed for: what was typed meanwhile stays). */
  apply(data,typed=false) {
    if(!typed && (!this.data || data.version!==this.data.version)) {
      this.count.value=String(data.count);this.size.value=String(data.size);
    }
    this.data=data;this.draw();this.render();
    // Said once for each GUI that closed with an error (by its pid): the button alone only turns back to "Open again".
    const failed=data?.exported?.gui_error;
    if(failed && !(this.guiFailed??=new Set()).has(failed.pid)) {
      this.guiFailed.add(failed.pid);
      this.notice?.(failed.reason?STAIN_PATCH_TEXT.guiClosed.replace('{reason}',()=>failed.reason):STAIN_PATCH_TEXT.guiClosedPlain);
    }
  }
  async action(button,label,request,apply=result=>this.apply(result),auto=false) {
    if(this.busy || !this.run || !this.data)return;
    const ticket=this.ticket;this.readTicket++;this.controller?.abort();this.loading=false;this.busy=true;this.auto=auto;
    this.release=button?busy(button,label):null;this.render();
    try {const result=await request();if(ticket===this.ticket)apply(result);}
    catch(e) {if(ticket===this.ticket)this.notice?.(e.status===409?STAIN_PATCH_TEXT.changed:e.message||STAIN_PATCH_TEXT.error);}
    finally {if(ticket===this.ticket){this.release?.();this.release=null;this.busy=false;this.auto=false;this.quiet();this.render();this.load();}}
  }
  /** The export dialog's Export here (patch-folder.mjs): the boxes of `version` into `folder`, Export patches busy
   *  meanwhile; the result shown as action() shows it. A refusal is thrown back to the dialog, which says it. */
  async exportTo(folder,version,onProgress) {
    if(this.busy || !this.run)throw new Error(EXPLAIN_TEXT.busy);
    const ticket=this.ticket,id=this.run.id;this.readTicket++;this.controller?.abort();this.loading=false;this.busy=true;
    this.release=busy(this.export,STAIN_PATCH_TEXT.exporting);this.render();
    // While it runs, how far it is (`onProgress`: the dialog's words): an export with the model's outlines runs the model
    // on every patch, a minute a patch on a CPU. Asked every exportPollMs; an answer that fails says nothing here.
    const asking={on:!!onProgress,stop:null};
    const pause=()=>new Promise(done=>{const timer=setTimeout(done,this.exportPollMs);asking.stop=()=>{clearTimeout(timer);done();};});
    (async()=>{while(asking.on){await pause();if(!asking.on)return;
      try{const status=await this.api.stainPatches(id);if(asking.on&&status?.exporting)onProgress(status.exporting);}catch{/* the export's own answer says it */}}})();
    try {const result=await this.api.stainPatchExport(id,{version,folder});if(ticket===this.ticket)this.apply(result);return result;}
    finally {asking.on=false;asking.stop?.();
      if(ticket===this.ticket){this.release?.();this.release=null;this.busy=false;this.quiet();this.render();this.load();}}
  }
  /** Make training patches. `auto`: the first boxes, nobody pressed for them; count and size stay open to typing. */
  suggest(auto=false) {
    return this.action(this.propose,STAIN_PATCH_TEXT.suggesting,()=>this.api.stainPropose(this.run.id,{
      version:this.data.version,count:Number(this.count.value),size:Number(this.size.value)}),
      result=>{this.apply(result,auto);if(result?.message)this.notice?.(result.message);},auto);   // e.g. fewer fit than asked
  }
  commit(boxes) {
    if(JSON.stringify(boxes)===JSON.stringify(this.data.patches.map(p=>p.box)))return;      // a press without a change
    return this.action(null,null,()=>this.api.stainPatchBoxes(this.run.id,{version:this.data.version,boxes}));
  }
  /** What a press of a box button finds missing (explain.mjs problems): the running request, a run that is not paused or
   *  finished, the first read, then `more()`. */
  problems(more=()=>[]) {
    if(this.busy)return [problem(null,'',EXPLAIN_TEXT.busy)];
    if(!this.active)return [problem(null,'',STAIN_PATCH_TEXT.empty)];
    if(!this.data)return [problem(null,'',STAIN_PATCH_TEXT.loading)];
    return more();
  }
  proposeProblems() {
    return this.problems(()=>{
      const T=STAIN_PATCH_TEXT,count=Number(this.count.value),size=Number(this.size.value),largest=this.largest(),found=[];
      if(!Number.isInteger(count) || count<1 || count>64)found.push(problem(this.count.id,T.count,T.countRange));
      if(!Number.isInteger(size) || size<64 || size>4096)found.push(problem(this.size.id,T.size,T.sizeRange));
      else if(largest!==null && size>largest)found.push(problem(this.size.id,T.size,largest<64?T.regionSmall:
        T.sizeFits.replace('{largest}',largest.toLocaleString('en-GB'))));
      return found;
    });
  }
  openProblems() {
    const T=STAIN_PATCH_TEXT,exported=this.data?.exported;
    if(this.busy)return [problem(null,'',EXPLAIN_TEXT.busy)];
    if(!this.data)return [problem(null,'',T.loading)];
    if(exported?.missing)return [problem(null,'',T.gone.replace('{folder}',exported.folder))];
    return !exported?[problem(null,'',T.exportFirst)]:exported.gui_running?[problem(null,'',T.guiOpen)]:[];
  }
  /** The run's region in pixels of its boxes' window (the engine's window of run.box, rounded outwards to whole pixels):
   *  [width, height], or null while unknown. */
  windowSize() {
    const box=this.run?.box,g=this.data?.run_pixel_to_sample;
    if(!box || !(g?.[2]>0 && g[3]>0))return null;
    const sides=[0,1].map(i=>Math.ceil((box[i+2]-g[i])/g[i+2]-1e-6));
    return sides.every(Number.isFinite)?sides:null;
  }
  /** The largest patch side inside the run's region, or null while unknown. */
  largest() {const sides=this.windowSize();return sides?Math.min(...sides):null;}
  /** Where a gesture keeps a box (boxGesture): the picker's preview has its window and lasso from the engine; an open run
   *  or item has its own region (run.box, and run.region's lasso) in its boxes' pixels. */
  frame(preview) {
    if(preview)return {extent:this.shown.extent,outline:this.shown.outline};
    const g=this.data.run_pixel_to_sample,region=this.run?.region;
    return {extent:this.windowSize(),outline:region?.kind==='polygon' && region.points?.length>=3?
      region.points.map(([x,y])=>[(x-g[0])/g[2],(y-g[1])/g[3]]):null};
  }
  quiet() {for(const button of [this.propose,this.add,this.export,this.open])clearExplanation(button);}
  /** The picker's suggested boxes before Start, while no run is open: numbered boxes with the item's move / corner / ×
   *  gestures, kept here without engine calls and inside `extent` ([width, height] of the boxes' pixel window) and, for a
   *  lasso, `outline` (the lasso in those pixels). onChange(boxes) follows each gesture that changes them. A press with
   *  Shift, or one yields(event) claims (the region's own handles), is left to the page. */
  showPreview({patches,run_pixel_to_sample,extent,outline},{onChange,yields}={}) {
    this.drag=null;this.ghost.setAttribute('visibility','hidden');this.onPreview=onChange;this.yields=yields||null;
    this.shown={patches:patches.map((p,i)=>({id:i+1,box:[...(p.box||p)]})),run_pixel_to_sample:[...run_pixel_to_sample],
      extent:extent?[...extent]:null,outline:outline?.length>=3?outline.map(p=>[...p]):null};
    this.svg.style.display='block';this.move();
  }
  previewBoxes() {return this.shown?this.shown.patches.map(p=>[...p.box]):null;}
  clearPreview() {
    if(!this.shown)return;
    this.shown=null;this.onPreview=null;this.yields=null;this.drag=null;this.boxKey=null;this.boxes.replaceChildren();
    this.ghost.setAttribute('visibility','hidden');this.svg.style.display=this.active?'block':'none';this.draw();
  }
  changePreview(boxes) {
    if(JSON.stringify(boxes)===JSON.stringify(this.previewBoxes()))return;     // a click without a move
    this.shown={...this.shown,patches:boxes.map((box,i)=>({id:i+1,box}))};   // numbered as the item will be
    this.draw();this.onPreview?.(this.previewBoxes());
  }
  move() {
    if(!this.active && !this.shown)return;
    const {camera:c,canvas}=this.stage,w=canvas.width/c.zoom,h=canvas.height/c.zoom;
    attr(this.svg,'viewBox',`${c.x-w/2} ${c.y-h/2} ${w} ${h}`);
    this.draw();
  }
  rect(node,b,[ox,oy,sx,sy]) {
    for(const [k,v] of Object.entries({x:ox+b[0]*sx,y:oy+b[1]*sy,width:(b[2]-b[0])*sx,height:(b[3]-b[1])*sy}))
      attr(node,k,v);
  }
  /** What box `b` has room for on screen at this camera: its corner handles and its × (see HANDLE and DROP). */
  room(b,[ox,oy,sx,sy]) {
    const a=this.stage.toScreen(ox+b[0]*sx,oy+b[1]*sy),c=this.stage.toScreen(ox+b[2]*sx,oy+b[3]*sy);
    const w=Math.abs(c[0]-a[0]),h=Math.abs(c[1]-a[1]);
    return {corners:Math.min(w,h)>=3*HANDLE,drop:w>=2*DROP[0] && h>=2*DROP[1]};
  }
  draw() {
    this.renderRows();
    const source=this.shown || this.data;
    if(!source?.run_pixel_to_sample)return;
    const key=JSON.stringify([source.patches,source.run_pixel_to_sample,this.stage.camera.zoom]);
    if(key===this.boxKey)return;this.boxKey=key;this.boxes.replaceChildren();
    const g=source.run_pixel_to_sample,[ox,oy,sx,sy]=g, font=14*(globalThis.devicePixelRatio||1)/this.stage.camera.zoom;
    for(const p of source.patches) {
      const rect=svgNode('rect');rect.setAttribute('vector-effect','non-scaling-stroke');this.rect(rect,p.box,g);
      const label=svgNode('text');label.textContent=String(p.id);label.setAttribute('x',ox+p.box[0]*sx+font/3);
      label.setAttribute('y',oy+p.box[1]*sy+font);label.setAttribute('font-size',font);
      this.boxes.append(rect,label);
      const room=this.room(p.box,g);
      if(room.drop) {
        const drop=svgNode('text');drop.textContent='×';drop.setAttribute('x',ox+p.box[2]*sx-font);
        drop.setAttribute('y',oy+p.box[1]*sy+font*1.5);drop.setAttribute('font-size',font);this.boxes.append(drop);
      }
      if(room.corners)for(const [x,y] of [[p.box[0],p.box[1]],[p.box[2],p.box[1]],[p.box[0],p.box[3]],[p.box[2],p.box[3]]]) {
        const handle=svgNode('circle');attr(handle,'cx',ox+x*sx);attr(handle,'cy',oy+y*sy);attr(handle,'r',font/4);
        this.boxes.append(handle);
      }
    }
  }
  /** The boxes as rows the keyboard reaches, "Patch 2 · 128 × 128 px" and a × each: the picker's preview in its own list,
   *  the open run's or item's (while its boxes can change) under the hint. Made again only when the boxes' ids change,
   *  else updated in place; a row that had the focus passes it to the row now in its place (the next one after a drop). */
  renderRows() {
    const T=STAIN_PATCH_TEXT;
    for(const [list,source,preview] of [[this.rows,this.active && !this.shown?this.data:null,false],[this.previewRows,this.shown,true]]) {
      const patches=source?.patches || [],key=JSON.stringify(patches.map(p=>p.id));
      list.hidden=!patches.length;
      if(key!==list.dataset.key) {
        const focused=[...list.children].findIndex(row=>row.contains(document.activeElement));
        list.dataset.key=key;list.replaceChildren(...patches.map(p=>this.row(p.id,preview)));
        if(focused>=0 && patches.length)list.children[Math.min(focused,patches.length-1)].focus({preventScroll:true});
      }
      patches.forEach((p,i)=>{
        const row=list.children[i],words=T.row.replace('{n}',p.id).replace('{w}',p.box[2]-p.box[0]).replace('{h}',p.box[3]-p.box[1]);
        text(row.firstChild,words);attr(row,'aria-label',`${words}. ${T.keys}`);
      });
    }
  }
  row(id,preview) {
    const T=STAIN_PATCH_TEXT,row=document.createElement('div'),drop=document.createElement('button');
    row.className='patch-row';row.setAttribute('role','listitem');row.setAttribute('tabindex','0');row.title=T.keys;
    drop.type='button';drop.textContent='×';drop.setAttribute('aria-label',T.drop.replace('{n}',id));
    drop.onclick=()=>this.editBox(id,preview,null);
    row.append(document.createElement('span'),drop);
    // Arrow keys move the box (16 pixels, 1 with Alt), Shift and arrow keys pull its bottom-right corner, Delete or
    // Backspace drops it: the pointer's gestures (boxGesture), with their edges, smallest side and lasso rule. Space on
    // the row does nothing: it is neither the page's Space (main.js: Resume of the paused run) nor a scroll; on the × it
    // presses the ×. A key taken here goes no further: Backspace would also take back a corner of the picker's outline
    // being drawn (main.js setupPointer).
    row.addEventListener('keydown',e=>{
      if(e.key===' ' && e.target===row && !e.ctrlKey && !e.metaKey && !e.altKey){e.preventDefault();e.stopPropagation();return;}
      const step=e.altKey?1:16,d={ArrowLeft:[-step,0],ArrowRight:[step,0],ArrowUp:[0,-step],ArrowDown:[0,step]}[e.key];
      if(e.ctrlKey || e.metaKey || !d && !['Delete','Backspace'].includes(e.key))return;
      e.preventDefault();e.stopPropagation();
      this.editBox(id,preview,d?box=>boxGesture(box,[0,0],d,e.shiftKey?3:-1,this.frame(preview)):null);
    });
    return row;
  }
  /** Box `id` changed from its row (`change(box)` → the new box; null drops it): to the preview's onChange or the
   *  engine, as a pointer gesture is, and only when one could be made (no gesture under way, the run's boxes open). */
  editBox(id,preview,change) {
    const t=preview?this.shown:this.data;
    if(!t || this.drag || !preview && (!this.active || this.busy || this.shown))return;
    const boxes=change?t.patches.map(p=>p.id===id?change(p.box):p.box):t.patches.filter(p=>p.id!==id).map(p=>p.box);
    if(preview)this.changePreview(boxes);else this.commit(boxes);
  }
  render() {
    if(!this.run)return;
    const T=STAIN_PATCH_TEXT, exported=this.data?.exported;
    this.count.disabled=this.size.disabled=this.busy && !this.auto;   // the first boxes never take a field from the user
    this.sizeNote.hidden=Number(this.size.value)<=1024;
    const cellPx=this.run?.tiling?.cell_px_median;
    // A run on a sample in pixels (`pixel_frame`, from the sample's notes) claims no width, as the picker does not.
    text(this.resolutionNote,[patchSizeNote(Number(this.size.value),this.data?.um_per_px?this.data:this.run?.stain?.resolution,
      {pixelFrame:this.run?.pixel_frame===true}),
      exported && !exported.missing && cellPx>0?`Cells are about ${Math.round(cellPx)} px across in these patches.`:''].filter(Boolean).join(' · '));
    this.resolutionNote.hidden=!this.resolutionNote.textContent;
    attr(this.add,'aria-pressed',this.adding);
    if(this.open.getAttribute('aria-busy')!=='true')text(this.open,exported?.gui_running?T.running:this.launch?T.again:T.open);
    text(this.hint,this.busy?T.preparing:this.active?T.hint:T.empty);
    const key=JSON.stringify(exported?.states);
    if(key!==this.listKey) {
      this.listKey=key;this.list.replaceChildren();
      for(const p of exported?.states||[]) {
        const row=document.createElement('p');row.textContent=`Patch ${p.id} · ${p.state} · ${p.cells??'—'} cells`;this.list.append(row);
      }
    }
    text(this.folder,exported?.folder?(exported.missing?T.gone:T.savedIn).replace('{folder}',exported.folder):'');
    text(this.command,this.install?`${T.install}\n${this.install}`:
      this.launch && !this.launch.running?`${T.fallback}\n${this.launch.command}`:'');
  }
  events() {
    const canvas=this.stage.canvas,stop=e=>{e.preventDefault();e.stopImmediatePropagation();};
    // A gesture acts on the picker's preview (kept here) or else on the open item's draft (saved by the engine).
    const target=preview=>preview?this.shown:this.data;
    const point=(e,g)=>pixelPoint(this.stage.toWorld(e.offsetX,e.offsetY),g);
    canvas.addEventListener('pointerdown',e=>{
      const preview=!!this.shown,t=target(preview);
      if(!preview && (!this.active || this.busy) || !t?.run_pixel_to_sample || e.button!==0 || e.altKey)return;
      if(preview && (e.shiftKey || this.yields?.(e)))return;      // the picker's pan and region handles come first
      const g=t.run_pixel_to_sample,p=point(e,g),[ox,oy,sx,sy]=g,adding=this.adding && !preview;
      // Corners and the × only where a box has room for them on screen (room()); else a press inside moves the box.
      const cornerOf=b=>!this.room(b,g).corners?-1:[[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]].findIndex(([x,y])=>{
        const q=this.stage.toScreen(ox+x*sx,oy+y*sy);return Math.hypot(q[0]-e.offsetX,q[1]-e.offsetY)<=HANDLE;
      });
      const found=!adding && (t.patches.find(b=>cornerOf(b.box)>=0) ||
        t.patches.find(b=>p[0]>=b.box[0] && p[0]<=b.box[2] && p[1]>=b.box[1] && p[1]<=b.box[3]));
      if(!found && !adding)return;
      stop(e);canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);
      const corner=found?cornerOf(found.box):-1;
      const top=found && this.room(found.box,g).drop && this.stage.toScreen(ox+found.box[2]*sx,oy+found.box[1]*sy);
      const drop=corner<0 && top && e.offsetX>=top[0]-DROP[0] && e.offsetY<=top[1]+DROP[1];
      this.drag={from:p,id:found?.id,box:found?.box,drop,corner,preview,frame:this.frame(preview)};
    },true);
    canvas.addEventListener('pointermove',e=>{
      if(!this.drag)return;stop(e);const d=this.drag,g=target(d.preview).run_pixel_to_sample;
      this.rect(this.ghost,boxGesture(d.box,d.from,point(e,g),d.corner,d.frame),g);
      this.ghost.setAttribute('visibility','visible');
    },true);
    canvas.addEventListener('pointerup',e=>{
      if(!this.drag)return;stop(e);const d=this.drag,t=target(d.preview);this.drag=null;
      if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
      this.ghost.setAttribute('visibility','hidden');
      const b=boxGesture(d.box,d.from,point(e,t.run_pixel_to_sample),d.corner,d.frame);
      let boxes=t.patches.map(p=>p.id===d.id?b:p.box);
      if(d.drop)boxes=t.patches.filter(p=>p.id!==d.id).map(p=>p.box);
      else if(!d.id) {
        this.adding=false;
        // A click or a few-pixel drag draws no box: say how instead of sending one the engine refuses.
        if(b[2]-b[0]<16 || b[3]-b[1]<16){this.render();this.notice?.(STAIN_PATCH_TEXT.tooSmall);return;}
        boxes.push(b);
      }
      if(d.preview)this.changePreview(boxes);else this.commit(boxes);
    },true);
    const cancel=()=>{this.drag=null;this.adding=false;this.ghost.setAttribute('visibility','hidden');this.render();};
    for(const event of ['pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>{if(this.drag)cancel();},true);
    document.addEventListener('keydown',e=>{
      if((this.active || this.shown) && e.key==='Escape' && !e.target.closest?.('input,select,textarea') &&
          !document.querySelector('dialog[open]'))cancel();
    });
  }
  snapshot() {return {active:this.active,version:this.data?.version,boxes:this.data?.patches,exported:this.data?.exported,
    busy:this.busy,loading:this.loading,preview:this.previewBoxes(),folder:this.folders.snapshot()};}
}
