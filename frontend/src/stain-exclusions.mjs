import { clampPoint, simplifyPolygon, polygonArea, pointInPolygon, polygonBBox } from './region.mjs';
import { canReviewStain } from './stain-patches.mjs';
import { guard, busy, clearExplanation, problem, EXPLAIN_TEXT } from './explain.mjs';
import { LassoDraft, mergedMoves } from './lasso.mjs';
import { RangeChooser, normaliseRange, sameRange } from './range-chooser.mjs';
import { CellValues } from './stain-cells.mjs';

export const EXCLUSION_TEXT = Object.freeze({
  title:'Remove cells in an area', done:'Done removing',
  hint:'Click corners around cells or drag around them; click the first point, double-click or press Enter to remove '+
    'them, then draw the next. Click an area and press Delete to bring its cells back. Backspace takes back a corner, '+
    'Escape drops the outline and a second Escape stops. Ctrl+Z undoes, Ctrl+Shift+Z redoes. Right-drag pans.',
  idle:'Each area removes whole cells whose centres are inside it. Click an area and press Delete, or use Undo, to '+
    'bring them back.',
  // The help line's tooltip: what a click on an area does, in full.
  helpTip:'A click inside a removed area selects it: where areas overlap, the one on top, and a click again on the '+
    'same spot the one under it. Delete or Backspace then removes that area, one step that Undo takes back: its cells '+
    'come back, apart from those also inside another area or outside the size or brightness range. Escape or a click '+
    'elsewhere lets it go. While an outline is being drawn, clicks add its corners.',
  changed:'The removed cells changed in another window. The latest are shown and the undo history was cleared; '+
    'click an area and press Delete to bring its cells back.',
  failed:'No answer came for the change, so it may or may not have gone through. The latest are shown and the '+
    'undo history was cleared; click an area and press Delete to bring its cells back.',
  error:'The removed areas could not be loaded. Reopen the run to try again.',
  small:'Draw an area with at least three corners.',
  paused:'Pause the run or let it finish first.', removing:'Removing cells…', restoring:'Bringing cells back…',
  waiting:'{n} waiting',
  nothingUndo:'There is nothing to undo yet: Undo covers the changes made in this window since the run was opened. '+
    'Click an area and press Delete to bring its cells back.',
  nothingRedo:'There is nothing to redo.',
  dropped:'{n} later changes were not sent.', droppedOne:'1 later change was not sent.',
  restarted:'The run started again, so {n} changes were not sent.',
  restartedOne:'The run started again, so 1 change was not sent.',
  sizeLabel:'Cell size (µm²)', brightLabel:'Brightness in',
  sizeHelp:'Cells outside the range lose their outlines and are left out of Save as shapes and training patches. '+
    'Drag a handle or type a size; an empty box means no limit.',
  brightHelp:"Each cell's brightest 5 µm² in this channel, in the image's own units. Cells outside the range lose their "+
    'outlines and are left out of Save as shapes and training patches. A cell with no pixel inside its outline is always kept.',
  kept:'Keeps {n} of {m} cells', keptAll:'Keeps all {m} cells', keptOne:'Keeps {n} of 1 cell',
  keptHelp:'Left out: {s} for their size, {b} for their brightness, {a} in removed areas. A cell can be left out for more '+
    'than one reason. Widen a range to bring cells back.',
  inUse:'{channel} · in use',
  // The label beside the area clicked on the image: the cells inside it (some may also be left out for another reason:
  // helpTip), none counted yet for an outline still on its way.
  selected:'{n} cells inside · Delete removes this area', selectedOne:'1 cell inside · Delete removes this area',
  selectedNew:'Delete removes this area',
  brightsNone:'No cell could be measured in {channel}.', measuring:'Measuring {channel} in each cell…',
  measuringFirst:'Measuring the brightness of each cell…',
  measureFailed:'The cells could not be measured in this channel. Reopen the run to try again.',
});
// Inputs whose keys are the user's typing: Ctrl+Z there undoes the typing, never an area.
const TYPED=/^(?:text|search|number|email|url|tel|password|date|datetime-local|month|week|time)$/;
const typing=t=>!!t && (t.isContentEditable || t.tagName==='TEXTAREA' || t.tagName==='INPUT' && TYPED.test(t.type||'text'));
const put=(node,value)=>{if(node.textContent!==value)node.textContent=value;};
// The label beside the area clicked on, in the overlay's words' font (qc-overlay.mjs FONT).
const LABEL_FONT='600 12px "Ubuntu Sans", Inter, system-ui, sans-serif';
const sameRule = sameRange;
const rangeRule = value => {const r=normaliseRange(value);return r.min===null && r.max===null?null:r;};
// Older saved exclusions are read as minimums. Every new write uses the range interface.
export function exclusionRanges(data) {
  return {...data,size:'size' in data?rangeRule(data.size):data.min_area_um2?{min:data.min_area_um2,max:null}:null,
    brightness:data.brightness || (data.min_brightness?{[data.min_brightness.channel]:{min:data.min_brightness.value,max:null}}:{})};
}
/** `reason`, then how many changes waiting their turn went with it, as a sentence of its own. */
const dropping=(reason,n)=>!n?reason:`${/[.!?…]$/.test(reason)?reason:`${reason}.`} `+
  (n===1?EXCLUSION_TEXT.droppedOne:EXCLUSION_TEXT.dropped.replace('{n}',n));

/** Area removals and per-channel ranges share one versioned queue and Undo history. Drag previews never post. */
export class StainExclusions {
  constructor(stage,api,{changed,refresh,notice,priors=null,values=null,onClasses=null,apiMode=false,patches=true}) {
    Object.assign(this,{stage,api,changed,refresh,notice,priors,onClasses});
    const get=id=>document.getElementById(id);
    this.panel=get('stain-exclusions');this.button=get('stain-exclude');this.help=get('stain-exclude-help');
    this.total=get('stain-exclude-total');this.status=get('stain-exclude-status');
    // The areas are acted on where they are, on the image (no list of them here): the help line's tooltip says how,
    // and a status of its own, not shown (stain-exclusions.css), gives a screen reader the label of the area clicked on.
    this.help.title=EXCLUSION_TEXT.helpTip;
    this.said=document.createElement('span');this.said.id='stain-exclude-selected';
    this.said.setAttribute('role','status');this.status.after(this.said);
    // Where the pointer last was on the image with no button held (hover), also for the view moving under it.
    this.pointer=null;stage.onChange?.(()=>this.hover());
    this.undoButton=get('stain-exclude-undo');this.redoButton=get('stain-exclude-redo');
    this.sizes=get('stain-exclude-sizes');this.brights=get('stain-exclude-brights');
    this.channelField=get('stain-exclude-bright-channel');
    const sizeRoot=get('stain-exclude-size-row'),brightRow=get('stain-exclude-bright-row');
    const brightRoot=document.createElement('div');brightRoot.id='stain-exclude-bright-range';
    const brightLabel=document.createElement('label');brightLabel.textContent=EXCLUSION_TEXT.brightLabel;
    brightLabel.setAttribute('for','stain-exclude-bright-channel');brightRow.replaceChildren(brightLabel,this.channelField,brightRoot);
    const help=text=>apiMode?text.replace('Save as shapes','Add as shapes').replace(' and training patches',patches?' and training patches':''):text;
    sizeRoot.title=help(EXCLUSION_TEXT.sizeHelp);brightRow.title=help(EXCLUSION_TEXT.brightHelp);
    const options=kind=>({kind,id:kind==='size'?'stain-exclude-size':'stain-exclude-bright',label:kind==='size'?EXCLUSION_TEXT.sizeLabel:'',
      calm:()=>this.stage.calmMotion || this.stage.reducedMotion?.(),canChange:()=>this.active && !!this.data && (kind==='size' || this.brightChannel()!==null),
      onInput:value=>{this.live={kind,channel:this.brightChannel(),value};this.updateClasses();},
      // Undo, Redo or another window moving the handles to the rule (renderRanges): the range they stand at, in the channel
      // they started in, until they stop.
      onTween:value=>{
        const channel=this.tweens.get(kind)?.channel ?? this.brightChannel();
        if(value)this.tweens.set(kind,{channel,value});else this.tweens.delete(kind);
        if(!this.resetting)this.updateClasses();
      },
      onCommit:value=>this.finishRange(kind,value),onCancel:()=>this.finishRange()});
    this.sizeChooser=new RangeChooser(sizeRoot,options('size'));this.brightChooser=new RangeChooser(brightRoot,options('brightness'));
    this.values=values || new CellValues(api,{changed:()=>{this.renderRanges();this.updateClasses();},
      error:(message,channel)=>{this.valueErrors.set(channel,message);}});
    // Both labels of the mode's button, which stain-exclusions.css makes it as wide as: Undo and Redo never move.
    Object.assign(this.button.dataset,{off:EXCLUSION_TEXT.title,on:EXCLUSION_TEXT.done});
    this.draft=new LassoDraft();
    // Never greyed out: a press says why the areas cannot change yet (render keeps the mode's button busy while they
    // first load), that there is nothing to undo or redo, or what a rule's field needs; it clears what the others said.
    const check=kind=>()=>{this.quiet();return this.problems(kind);};
    // "Done removing" with an outline still open closes it first, so its cells go as the label says (one with too few
    // points to close goes with the mode).
    guard(this.button,check('mode'),()=>{
      if(this.drawing && this.draft.key({key:'Enter'})==='close')this.finish();
      this.drawing?this.stop():this.start();
    });
    guard(this.undoButton,check('undo'),()=>this.queue({kind:'undo'}));
    guard(this.redoButton,check('redo'),()=>this.queue({kind:'redo'}));
    // A channel chosen here is measured (again, if its read failed) and stays when a rule in another channel goes.
    this.channelField.addEventListener('change',()=>{
      this.brightChooser.cancel();this.chosen=this.channel=this.channelField.value || null;this.unmeasured=null;
      this.render();this.updateValues();
    });
    this.reset();this.events();
  }
  reset() {
    this.resetting=true;this.tweens=new Map();this.rulesShown=null;
    this.sizeChooser?.setHidden(true);this.brightChooser?.setHidden(true);this.values?.reset();
    for(const waiter of this.settlers || [])waiter.reject(Error('The run changed.'));
    this.settlers=[];this.settleError=this.readError=null;this.live=null;this.deferredData=null;this.vectors=null;this.valueErrors=new Map();
    this.sizeChooser?.setRange(null);this.brightChooser?.setRange(null);
    if(this.priors)this.priors.classes=null;this.onClasses?.(null);
    this.ticket=(this.ticket||0)+1;this.controller?.abort();this.run=null;this.data=null;this.loading=false;
    this.busy=false;this.sending=null;this.waiting=[];this.undos=[];this.redos=[];
    this.active=false;this.drawing=false;this.pressing=false;this.draft.reset();this.panel.hidden=true;
    // The area clicked on the image (select) and where that click was (pick), the press that may select one.
    this.selected=null;this.picked=null;this.fresh=null;this.tap=null;
    this.stage.canvas.classList.remove('removing','over-area');
    // For the brightness rule: the read on its way and the last answer, the channels it listed and the run's default,
    // the channel chosen here and the one shown.
    this.measuring?.controller.abort();this.measuring=null;this.spread=null;this.unmeasured=null;this.channels=null;
    this.usual=null;this.chosen=null;this.channel=null;
    this.channelField.replaceChildren();this.namesKey=null;
    this.done?.();this.done=null;this.working=null;this.quiet();this.resetting=false;
  }
  quiet() {for(const button of [this.button,this.undoButton,this.redoButton])clearExplanation(button);}
  /** Mode, Undo and Redo remain available; a press explains anything that is still needed. */
  problems(kind='mode') {
    const T=EXCLUSION_TEXT;
    if(kind==='mode' && this.drawing)return [];
    if(!this.active)return [problem(null,'',T.paused)];
    // No areas while a change that went wrong is read again: the last step is still running, nothing is lost.
    if(!this.data)return [problem(null,'',this.loading || this.busy?EXPLAIN_TEXT.busy:T.error)];
    const left=this.planned();
    if(kind==='undo' && !left.undo)return [problem(null,'',T.nothingUndo)];
    if(kind==='redo' && !left.redo)return [problem(null,'',T.nothingRedo)];
    return [];
  }
  brightChannel() {return this.channel ?? this.usual ?? this.run?.stain?.nucleus ?? this.run?.stain?.channels?.[0] ?? null;}
  /** The Undo and Redo steps there will be once the change on its way and those waiting have gone through. */
  planned() {
    let undo=this.undos.length,redo=this.redos.length;
    for(const step of [this.sending,...this.waiting]) {
      if(!step)continue;
      const sent=step===this.sending;          // its entry has already left its list
      if(step.kind==='undo'){if(!sent)undo--;redo++;}
      else if(step.kind==='redo'){if(!sent)redo--;undo++;}
      else {undo++;redo=0;}
    }
    return {undo,redo};
  }
  rule(name,channel=this.brightChannel()) {return (name==='size'?this.data?.size:this.data?.brightness?.[channel])??null;}
  plannedRule(name,channel=this.brightChannel()) {
    const rules=this.plannedRules();return name==='size'?rules.size:rules.brightness[channel]??null;
  }
  plannedRules() {
    const rules={size:this.data?.size??null,brightness:Object.assign(Object.create(null),this.data?.brightness)};
    const undos=[...this.undos],redos=[...this.redos];
    const apply=(entry,undo)=>{
      if(entry?.size)rules.size=undo?entry.size.before:entry.size.after;
      if(entry?.brightness)rules.brightness[entry.brightness.channel]=undo?entry.brightness.before:entry.brightness.after;
    };
    if(this.sending) {
      const {entry,kind}=this.sending,undo=kind==='undo';apply(entry,undo);
      // After the answer, refresh can still be pending; the entry is already back in history then.
      const stack=undo?redos:undos;if(stack.at(-1)!==entry)stack.push(entry);
    }
    for(const step of this.waiting) {
      const undo=step.kind==='undo',entry=step.entry || (undo?undos:redos).pop();
      apply(entry,undo);if(entry)(undo?redos:undos).push(entry);
      if(step.entry)redos.length=0;
    }
    return rules;
  }
  setVectors(vectors) {
    if(vectors?.set!==this.vectors?.set || vectors?.areas_version!==this.vectors?.areas_version)this.valueErrors.clear();
    this.vectors=vectors;this.updateValues();this.updateClasses();
  }
  updateValues() {
    if(!this.active || !this.data || !this.vectors?.set)return;
    const {set,areas_version=0}=this.vectors;
    if(this.data.areas_version!=null && areas_version!==this.data.areas_version)return;
    this.values.load(this.run.id,set,areas_version);
    const channels=new Set([...Object.keys(this.data.brightness || {}),this.brightChannel()]);
    for(const channel of channels)if(channel!==null)this.values.load(this.run.id,set,areas_version,channel);
  }
  liveRules() {
    let {size,brightness}=this.plannedRules();
    for(const [kind,{channel,value}] of this.tweens)if(kind==='size')size=value;else brightness[channel]=value;
    if(this.live?.kind==='size')size=this.live.value;
    if(this.live?.kind==='brightness')brightness[this.live.channel]=this.live.value;
    return {size,brightness};
  }
  updateClasses() {
    const classes=this.values.forSet(this.vectors?.set,this.liveRules());
    if(this.priors)this.priors.classes=classes;this.onClasses?.(classes);
    this.stage.dirty=true;this.renderCount(classes);this.changed();
  }
  /** What the Keeps line counts: the cells' classes as shown on the image (a held or gliding handle's range included)
   *  once their values are in, else the engine's last answer. {data, cells, kept}. */
  counts(classes=this.values.forSet(this.vectors?.set,this.liveRules())) {
    const data=classes || this.data || {},cells=data.cells ?? this.run?.summary_model?.cells ?? this.run?.summary?.cells ?? 0;
    return {data,cells,kept:data.kept ?? Math.max(0,cells-(data.removed || 0))};
  }
  renderCount(classes=null) {
    const T=EXCLUSION_TEXT,{data,cells,kept}=this.counts(classes),f=n=>n.toLocaleString('en-GB');
    put(this.total,(cells===1?T.keptOne:cells && kept===cells?T.keptAll:T.kept).replace('{n}',f(kept)).replace('{m}',f(cells)));
    this.total.title=T.keptHelp.replace('{s}',f(data.by_size || 0)).replace('{b}',f(data.by_brightness || 0)).replace('{a}',f(data.in_areas || 0));
  }
  finishRange(kind,value) {
    if(this.resetting)return;
    this.live=null;
    const deferred=this.deferredData;this.deferredData=null;
    if(deferred) {
      const external=!deferred.own && this.data && deferred.data.version!==this.data.version;
      if(external)this.forget(EXCLUSION_TEXT.changed);
      this.apply(deferred.data,{own:true});
      if(external){this.render();this.updateClasses();this.settled();return;}
    }
    if(kind) {
      const channel=this.brightChannel(),after=rangeRule(value);
      if(!sameRule(after,this.plannedRule(kind,channel)))this.queue({kind,entry:{[kind]:{before:null,after,...kind==='brightness'?{channel}:{}}}});
    }
    this.render();this.updateClasses();this.next();this.settled();
  }
  renderRanges() {
    if(!this.values || this.resetting)return;
    const data=this.values.data,channel=this.brightChannel(),rules=this.plannedRules();
    const hidden=!this.active || data?.count===0;
    this.sizeChooser.setHidden(hidden);this.brightChooser.setHidden(hidden);
    this.sizeChooser.setValues(data?.area,{pixelArea:(this.run?.stain?.resolution?.um_per_px || 0)**2});
    const bright=this.values.brightness.get(channel);
    this.brightChooser.setValues(bright,{integer:data?.integer!==false && !/float|^f[248]$/i.test(data?.dtype || '')});
    // The handles glide to a rule that changed while shown (Undo, Redo, another window), the outlines coming back or
    // going as they pass them (follow); rules first loaded, or another channel's rule shown, change no rule.
    const shown=this.rulesShown,follow=!!shown;this.rulesShown=this.data?{channel}:null;
    this.sizeChooser.setRange(rules.size,{animate:true,follow});
    this.brightChooser.setRange(rules.brightness[channel],{animate:true,follow:follow && shown.channel===channel});
    put(this.sizes,this.valueErrors.get(null) || '');
    const names=[...new Set([...this.channels||[],...Object.keys(rules.brightness),...channel===null?[]:[channel]])];
    const key=JSON.stringify([names,names.map(name=>!!rangeRule(rules.brightness[name]))]);
    if(key!==this.namesKey) {
      this.namesKey=key;this.channelField.replaceChildren(...names.map(name=>Object.assign(document.createElement('option'),
        {value:name,textContent:rangeRule(rules.brightness[name])?EXCLUSION_TEXT.inUse.replace('{channel}',name):name})));
    }
    if(channel!==null)this.channelField.value=channel;
    const reading=this.values.jobs.has(channel) && channel!==null;
    put(this.brights,this.valueErrors.get(channel) || (reading?EXCLUSION_TEXT.measuring.replace('{channel}',channel):
      bright && !this.brightChooser.axis.count?EXCLUSION_TEXT.brightsNone.replace('{channel}',channel):this.brightLine()));
  }
  async whenSettled() {
    const ticket=this.ticket;
    await Promise.all([this.sizeChooser.whenSettled(),this.brightChooser.whenSettled()]);
    if(ticket!==this.ticket)throw Error('The run changed.');
    if(!this.busy && !this.loading && !this.waiting.length) {
      // A change refused, or the areas' last read failed (until a later read answers: load), is said once.
      const error=this.settleError || this.readError;
      if(error){this.settleError=this.readError=null;throw error;}
      // No areas read: shown for review, that read failed; a run that cannot be reviewed yet (running, failed, no
      // tile) has nothing of this window's to settle.
      if(!this.data){if(this.active)throw Error(EXCLUSION_TEXT.error);return null;}
      return this.data;
    }
    return new Promise((resolve,reject)=>this.settlers.push({resolve,reject}));
  }
  settled() {
    if(this.busy || this.loading || this.waiting.length || this.sizeChooser.gesture || this.brightChooser.gesture)return;
    const waiters=this.settlers.splice(0);
    const error=this.settleError || this.readError;
    for(const waiter of waiters)error?waiter.reject(error):waiter.resolve(this.data);
    if(waiters.length)this.settleError=this.readError=null;
  }
  update(run) {
    if(this.run?.id!==run.id)this.reset();
    this.run=run;
    const was=this.active;
    this.active=run.kind==='stain' && ['paused','done','stopped'].includes(run.state) && canReviewStain(run);
    this.panel.hidden=!this.active;
    if(!this.active){this.sizeChooser.setHidden(true);this.brightChooser.setHidden(true);}
    // Running again: the mode ends with the outline being drawn, and the changes not sent yet go (the engine refuses
    // them while the run runs); the notice says how many.
    if(!this.active && (this.drawing || this.draft.active || this.waiting.length)) {
      const n=this.waiting.length;this.drawing=false;this.pressing=false;this.fresh=null;this.draft.reset();this.waiting=[];
      this.changed();
      if(n)this.notice(n===1?EXCLUSION_TEXT.restartedOne:EXCLUSION_TEXT.restarted.replace('{n}',n));
    }
    // An area clicked on stays selected only while the areas can change.
    if(!this.active){this.stage.canvas.classList.remove('over-area');this.select(null);}
    if(run.kind==='stain' && (this.active || run.exclusions?.count>0) &&
        (!this.data || this.active && !was || this.data.version!==(run.exclusions?.version||0) ||
         run.exclusions?.removed!==undefined && this.data.removed!==run.exclusions.removed))this.load();
    if(this.active && !was)this.unmeasured=null;   // shown again: a brightness read that failed is tried again (render)
    this.updateValues();this.render();this.settled();
  }
  start() {this.drawing=true;this.draft.reset();this.render();this.changed();}
  stop() {this.drawing=false;this.pressing=false;this.draft.reset();this.render();this.changed();}
  async load() {
    if(this.loading || this.busy || !this.run || this.run.kind!=='stain')return;
    const ticket=this.ticket,id=this.run.id;this.loading=true;
    const controller=this.controller=new AbortController();
    try {
      const data=await this.api.stainExclusions(id,controller.signal);
      if(ticket!==this.ticket || controller.signal.aborted)return;
      this.readError=null;   // a read answered: one that failed before says nothing any more
      // A run-detail update may arrive while this read is in flight. Older areas than shown are no use; older ones are
      // still shown rather than none (next reads them again before a change goes).
      if(this.data && data.version<(this.run.exclusions?.version||0))return;
      this.apply(data);
    } catch(e) {   // changes waiting for the areas cannot go without them, and their outlines leave the image
      if(ticket===this.ticket && e.name!=='AbortError'){this.notice(EXCLUSION_TEXT.error);this.readError=Error(EXCLUSION_TEXT.error);this.waiting=[];this.changed();}
    }
    finally {if(ticket===this.ticket && this.controller===controller){this.loading=false;this.render();this.next();this.settled();}}
  }
  apply(data,{own=false}={}) {
    data=exclusionRanges(data);
    if(this.sizeChooser.gesture || this.brightChooser.gesture){this.deferredData={data,own};return;}
    if(this.data && data.version<this.data.version)return;
    const changed=JSON.stringify(data)!==JSON.stringify(this.data);
    if(!own && this.data && data.version!==this.data.version && (this.undos.length || this.redos.length))this.forget(EXCLUSION_TEXT.changed);
    // The area clicked on goes when it does (its cells back, from here or another window). One clicked on while on its
    // way is known by the id the engine gave it (next) from then on; until then it stays while it is on its way.
    const id=this.selected?.id ?? this.selected;
    this.selected=id!==null && typeof id!=='object' && !data.areas?.some(a=>a.id===id)?null:id;
    this.data=data;this.updateValues();this.render();if(changed)this.updateClasses();
  }
  /** The read of the cells' brightness wanted now: the run, the channel shown and the run's outlines. */
  readKey() {return JSON.stringify([this.run?.id,this.brightChannel(),this.run?.progress?.revision||0]);}
  /** Read the cells' brightness in the channel shown (render calls it), once the areas are in, so that a rule in effect
   *  names its channel first: when the panel shows and when the channel or the run's outlines (progress.revision)
   *  change, unless that read is shown or on its way already; one that failed waits for one of those. An answer for
   *  another run, channel or outlines is never shown (the ticket, and the read wanted since). */
  async measure() {
    if(!this.active || !this.data)return;
    const channel=this.brightChannel(),id=this.run.id,revision=this.run.progress?.revision||0,key=this.readKey();
    if(key===this.measuring?.key)return;
    this.measuring?.controller.abort();this.measuring=null;                  // a read no longer wanted is let go
    if(key===this.spread?.key || key===this.unmeasured)return;
    const ticket=this.ticket,job=this.measuring={key,channel,controller:new AbortController()};
    try {
      const data=await this.api.stainBrightness(id,channel,job.controller.signal);
      if(ticket!==this.ticket || this.measuring!==job)return;
      // Asked without a channel (the run's default, not known before), the answer names it.
      this.channels=data.channels;this.usual=data.default;this.updateValues();
      this.spread={...data,key:JSON.stringify([id,channel??data.channel,revision])};
    } catch(e) {
      if(ticket===this.ticket && this.measuring===job && e.name!=='AbortError')this.unmeasured=key;
    } finally {
      if(ticket===this.ticket && this.measuring===job){this.measuring=null;this.render();}
    }
  }
  /** The line under the brightness rule: the cells' brightness in the channel shown, or that it is being measured or
   *  could not be. */
  brightLine() {
    const T=EXCLUSION_TEXT,s=this.spread,key=this.readKey();
    const on=this.measuring?.key===key?this.measuring:null;
    if(on)return on.channel===null?T.measuringFirst:T.measuring.replace('{channel}',on.channel);
    if(s?.key===key) {
      return s.p10==null?T.brightsNone.replace('{channel}',s.channel):'';
    }
    return key===this.unmeasured?T.measureFailed:'';
  }
  /** An outline, Delete on an area, a size, a brightness, Undo or Redo: sent now, or after the change on its way (never
   *  two at once, never lost). */
  queue(step) {
    this.settleError=this.readError=null;
    const last=this.waiting.at(-1),name=['size','brightness'].find(k=>step.entry?.[k]);
    if(name && last?.kind===step.kind && (name==='size' || last.entry.brightness.channel===step.entry.brightness.channel))
      last.entry[name].after=step.entry[name].after;
    else this.waiting.push(step);
    this.quiet();this.render();this.updateClasses();this.next();
  }
  /** The outline drawn is closed: its area goes to the engine, and the mode stays on for the next one. */
  finish() {
    const points=simplifyPolygon(this.draft.points(),1.5/this.stage.scale,5000);this.draft.reset();
    if(points.length<3 || polygonArea(points)===0){this.notice(EXCLUSION_TEXT.small);this.changed();return;}
    this.queue({kind:'lasso',entry:{area:{id:null,points},added:true}});
  }
  /** Delete on an area: put it back. When this window changed that area before, its record is the history's own (its id
   *  followed it), so Undo and Redo of both changes keep working; an outline still on its way is its own record, which
   *  has the engine's id by the time this change's turn comes (next). */
  restore(area) {
    const known=area.id===null?area:[...this.undos,...this.redos,this.sending?.entry,...this.waiting.map(s=>s.entry)]
      .find(entry=>entry?.area && entry.area.id===area.id)?.area;
    this.queue({kind:'restore',entry:{area:known || {id:area.id,points:area.points},added:false}});
  }
  /** The outlines on their way or waiting that the engine has not named yet: drawn until it has them, and clicked on as
   *  the areas are (their records: {id: null, points}). */
  pendingAreas() {
    return [this.sending,...this.waiting].filter(s=>s?.kind==='lasso' && s.entry.area.id===null).map(s=>s.entry.area);
  }
  /** Select `area` (null: none), drawn selected with its label; Delete then brings its cells back. An area is kept by
   *  its id, an outline on its way by its record (apply). */
  select(area) {
    const selected=area===null?null:area.id ?? area;
    if(selected===this.selected)return;
    this.selected=selected;this.say();this.changed();
  }
  /** The selected area as the areas hold it now, the outline on its way that was clicked on, or null. */
  selectedArea() {
    const s=this.selected,id=s?.id ?? s;
    if(id!==null && typeof id==='object')return this.pendingAreas().includes(id)?id:null;
    return id===null?null:this.data?.areas?.find(a=>a.id===id) ?? null;
  }
  /** Delete or Backspace on the selected area: its cells come back (restore), and the selection goes. */
  dropSelected() {
    const area=this.selectedArea();this.select(null);
    if(area)this.restore(area);
  }
  /** The areas under `point` (µm), the one on top first: the outlines on their way (drawn over the areas), then the
   *  areas, the last drawn first. Each area's bounds are kept with the areas they were taken from, so a pointer move
   *  tests few outlines. */
  areasAt(point) {
    const areas=this.data?.areas || [],[x,y]=point;
    if(this.bounded!==areas){this.bounded=areas;this.bounds=areas.map(a=>polygonBBox(a.points));}
    const inside=(points,b)=>!!b && x>=b[0] && x<=b[2] && y>=b[1] && y<=b[3] && pointInPolygon(point,points);
    return [...areas.filter((a,i)=>inside(a.points,this.bounds[i])),
      ...this.pendingAreas().filter(a=>inside(a.points,polygonBBox(a.points)))].reverse();
  }
  /** The area a click at `point` selects: the one on top there, which is the selected one when the click is inside it
   *  (it is drawn over the others); a click again on the same spot steps to the next one under it, so that every area
   *  can be reached where areas overlap or one lies inside another. Null outside every area. */
  pick(point) {
    const under=this.areasAt(point),i=under.indexOf(this.selectedArea()),last=this.picked;
    this.picked=point;
    if(i<0)return under[0] ?? null;
    const again=!!last && Math.hypot(point[0]-last[0],point[1]-last[1])*this.stage.scale<=this.draft.closeRadius;
    return under[again?(i+1)%under.length:i];
  }
  /** The press now down began no outline, inside an area: a click selects that area (pointerup), so the corner it
   *  placed is not drawn unless the press turns into a drag. */
  picking() {
    const piece=this.draft.press;
    return this.pressing && !!this.fresh && !!piece && !piece.stroke && this.areasAt(this.fresh).length>0;
  }
  /** The hand where the pointer rests over an area a click would select (stain-exclusions.css .over-area): while the
   *  areas can change, and in the mode only while no outline is being drawn (a click then adds a corner). Again
   *  whenever the areas, the outline or the view change under it. */
  hover() {
    const p=this.pointer,over=!!p && this.active && !(this.drawing && this.draft.active) &&
      this.areasAt(this.stage.toWorld(...p)).length>0;
    this.stage.canvas.classList.toggle('over-area',over);
  }
  /** The words beside the area clicked on: the cells inside it (none counted yet on its way) and what Delete does. */
  labelText(area) {
    const T=EXCLUSION_TEXT,n=area.cells;
    return n==null?T.selectedNew:n===1?T.selectedOne:T.selected.replace('{n}',n.toLocaleString('en-GB'));
  }
  /** The area clicked on, for a screen reader (the status not shown): its label's words, none without one. */
  say() {const area=this.selectedArea();put(this.said,area?this.labelText(area):'');}
  /** The history no longer matches the areas: it goes, with what was waiting, and the notice says so. */
  forget(reason) {
    const n=this.waiting.length;this.undos=[];this.redos=[];this.waiting=[];
    this.notice(dropping(reason,n));
  }
  /** Send the first waiting change once nothing is on its way and the areas are loaded, with their newest version. */
  async next() {
    if(!this.waiting.length){this.settled();return;}
    if(this.busy || this.loading || this.sizeChooser.gesture || this.brightChooser.gesture || !this.data || !this.active || !this.waiting.length)return;
    // The run detail knows a newer version than the areas read here (another window changed them): read them first,
    // once for each change (one still behind then goes, and the engine says whether its version is old).
    const first=this.waiting[0];
    if((this.run.exclusions?.version||0)>this.data.version && !first.read){first.read=true;return this.load();}
    const T=EXCLUSION_TEXT,ticket=this.ticket,id=this.run.id,step=this.waiting.shift(),undo=step.kind==='undo';
    const ids=new Set(this.data.areas.map(a=>a.id));
    // A new change (outline, Delete on an area, size, brightness) brings its own entry; Undo takes the history's last
    // one, Redo the last one undone.
    const entry=step.entry || (undo?this.undos.pop():this.redos.pop()),name=['size','brightness'].find(k=>entry?.[k]);
    let add,send;
    if(name) {
      // A new rule follows the one in effect when its turn comes; Undo puts that one back, Redo the new one again.
      const change=entry[name],rule=this.rule(name,entry[name].channel);
      if(step.kind===name)change.before=rule;
      const from=undo?change.after:change.before,to=undo?change.before:change.after;
      if(!sameRule(from,rule)){this.forget(T.changed);this.render();this.updateClasses();this.settled();return;}
      if(sameRule(to,rule)){this.render();return this.next();}                             // already the rule: nothing to do
      const limits=normaliseRange(to);
      add=!!to;
      if(name==='size')send=version=>this.api.stainExcludeSize(id,{version,...limits});
      else send=version=>this.api.stainExcludeBrightness(id,{version,channel:change.channel,...limits});

    } else {
      // Nothing left to do: Undo or Redo after a change that failed, Delete on an area already back.
      if(!entry || step.kind==='restore' && !ids.has(entry.area.id)){this.render();return this.next();}
      // Undo of an outline and Redo of a Delete take the area away again; Undo of a Delete and Redo of an outline post
      // it again.
      add=undo?!entry.added:!!entry.added;
      if(add===(entry.area.id!==null) || !add && !ids.has(entry.area.id)) {   // that area was changed elsewhere
        this.forget(T.changed);this.render();this.updateClasses();this.settled();return;
      }
      send=version=>add?this.api.stainExclude(id,{version,points:entry.area.points}):
        this.api.stainRestore(id,entry.area.id,version);
    }
    this.busy=true;this.sending={kind:step.kind,entry,add};this.render();
    const version=this.data.version;
    try {
      const data=await send(version);
      if(ticket!==this.ticket)return;
      if(entry.area) {
        const made=add?data.areas.find(a=>!ids.has(a.id)) || data.areas.at(-1):null;
        entry.area.id=made?made.id:null;if(made)entry.area.points=made.points;
      }
      if(undo)this.redos.push(entry);
      else {this.undos.push(entry);if(step.entry)this.redos=[];}          // a new change clears what could be redone
      this.apply(data,{own:true});
      try {await this.refresh(id,data);} catch {/* the runs poll brings the run detail */}
    } catch(e) {
      if(ticket!==this.ticket)return;
      this.sending=null;
      this.settleError=Error(e.message || T.failed);
      // A 409 is another window's change only when the areas, read again, have another version than the one sent;
      // with the same one the engine refused this change (e.g. the run runs again). A read that fails tells nothing.
      let read=null;
      if(e.status===409) {
        try {read=await this.api.stainExclusions(id);} catch {/* a refusal, as far as is known */}
        if(ticket!==this.ticket)return;
      }
      if(e.status===404 || read && read.version!==version || e.status!==409 && (!e.status || e.status>=500)) {
        // Changed elsewhere (404: this area is already back) or nothing known of the outcome (no answer, a server
        // error): the latest areas are shown and the history no longer matches them.
        this.forget(e.status===404 || read?T.changed:T.failed);this.data=null;if(read)this.apply(read);
        try {await this.refresh(id);} catch {if(ticket===this.ticket)this.notice(T.error);}
      } else {
        // Refused (nothing changed): the entry stays where it was, and what was waiting goes, since it counted on it.
        if(!step.entry)(undo?this.undos:this.redos).push(entry);
        const n=this.waiting.length;this.waiting=[];if(read)this.apply(read);
        this.notice(dropping(e.message || T.paused,n));
      }
    } finally {
      if(ticket===this.ticket){this.busy=false;this.sending=null;if(!this.data)this.load();this.render();this.updateClasses();this.next();this.settled();}
    }
  }
  render() {
    const T=EXCLUSION_TEXT;
    // Busy only until the areas first load ('': the label stays). A change on its way never holds up the mode or a
    // press (they wait their turn); the status beside the count says what is being sent.
    const working=this.loading && !this.data && !this.drawing?'':null;
    if(working!==this.working){this.done?.();this.done=working===null?null:busy(this.button,working);this.working=working;}
    if(working===null && !this.problems().length)clearExplanation(this.button);
    // The label says what a press does; the mode on shows in the button's colour (class "on", stain-exclusions.css)
    // and the crosshair on the image.
    put(this.button,this.drawing?T.done:T.title);this.button.classList.toggle('on',this.drawing);
    this.stage.canvas.classList.toggle('removing',this.active && this.drawing);
    put(this.help,this.drawing?T.hint:T.idle);
    this.renderCount(this.values.forSet(this.vectors?.set,this.liveRules()));
    const waiting=this.waiting.length?T.waiting.replace('{n}',this.waiting.length):'';
    put(this.status,this.sending?`${this.sending.add?T.removing:T.restoring}${waiting?` (${waiting})`:''}`:waiting);
    this.say();this.hover();this.renderRanges();this.measure();
  }
  draw(g,path) {
    if(!this.run || this.run.kind!=='stain')return;
    const selected=this.selectedArea();
    g.lineWidth=1;g.strokeStyle='rgba(255,180,140,.5)';g.fillStyle='rgba(255,180,140,.08)';
    for(const area of this.data?.areas||[]){
      if(area!==selected){g.beginPath();path(g,{kind:'polygon',points:area.points});g.fill();g.stroke();}
    }
    // Outlines on their way or waiting: drawn until the engine has them.
    const lassos=[this.sending,...this.waiting].filter(s=>s?.kind==='lasso').map(s=>s.entry.area);
    for(const area of lassos)if(area!==selected){g.beginPath();path(g,{kind:'polygon',points:area.points});g.stroke();}
    // The area clicked on (or the outline on its way): on top, in the same colour more strongly, its label beside it.
    if(selected) {
      g.lineWidth=2;g.strokeStyle='#ffb48c';g.fillStyle='rgba(255,180,140,.24)';
      g.beginPath();path(g,{kind:'polygon',points:selected.points});g.fill();g.stroke();
      this.label(g,selected);
    }
    // The outline being drawn (lasso.mjs preview), as the picker's Lasso draws its own: the points so far, a dashed line
    // on to the pointer, and the first point, white and larger while a press there would close the outline. None for a
    // press that may yet be a click selecting an area (picking).
    const {points,guide,closable}=this.draft.preview();
    if(!points.length || this.picking())return;
    const screen=points.map(p=>this.stage.toScreen(...p));
    g.strokeStyle='#ffb48c';g.lineWidth=2;g.beginPath();screen.forEach((p,i)=>i?g.lineTo(...p):g.moveTo(...p));g.stroke();
    if(guide) {
      g.lineWidth=1;g.setLineDash([6,4]);
      g.beginPath();g.moveTo(...screen.at(-1));g.lineTo(...this.stage.toScreen(...guide));g.stroke();g.setLineDash([]);
    }
    const [x,y]=screen[0],r=closable?6:4;
    g.fillStyle=closable?'#fff':'#ffb48c';g.fillRect(x-r,y-r,2*r,2*r);
  }
  /** Beside the area clicked on: the cells inside it and what Delete does (labelText), on a dark label at the right of
   *  its top (at its left when the image ends there), kept on the image. */
  label(g,area) {
    const text=this.labelText(area);
    const [x0,y0,x1,y1]=polygonBBox(area.points),[left,top]=this.stage.toScreen(x0,y0),[right]=this.stage.toScreen(x1,y1);
    const W=this.stage.cssWidth ?? Infinity,H=this.stage.cssHeight ?? Infinity,h=20;
    g.font=LABEL_FONT;g.textBaseline='middle';g.textAlign='left';
    const w=Math.ceil(g.measureText(text)?.width ?? 0)+12;
    let x=right+8;
    if(x+w>W-4)x=left-8-w;
    x=Math.max(4,Math.min(x,W-4-w));const y=Math.max(4,Math.min(top,H-4-h));
    g.fillStyle='rgba(14,17,22,.88)';g.fillRect(x,y,w,h);
    g.lineWidth=1;g.strokeStyle='#ffb48c';g.strokeRect(x+.5,y+.5,w-1,h-1);
    g.fillStyle='#e6eaf0';g.fillText(text,x+6,y+h/2+.5);
  }
  events() {
    const canvas=this.stage.canvas,stop=e=>{e.preventDefault();e.stopImmediatePropagation();};
    // A pointer event as the outline takes it: its point in the run's box (kept inside it) and on screen.
    const at=e=>[clampPoint(this.stage.toWorld(e.offsetX,e.offsetY),this.run.box),[e.offsetX,e.offsetY]];
    const on=()=>this.active && this.drawing;
    // What the outline answers (lasso.mjs): closed, its area is removed; given up; a close asked for with too few
    // points, which says so; or to be drawn again.
    const answer=result=>{
      if(result==='close')this.finish();
      else if(result==='cancel'){this.draft.reset();this.changed();}
      else if(result==='few'){this.notice(EXCLUSION_TEXT.small);this.changed();}
      else if(result==='changed')this.changed();
    };
    // In the mode every press of the main button on the image is the outline's (a click places a corner, a drag draws
    // freehand), also while a change is on its way and with Shift or Alt held, so none reaches a patch box: these
    // capture listeners run before stain-patches.mjs's (main.js makes this panel first) and stop the press here. With no
    // outline being drawn, a click inside an area selects it instead (pointerup); a drag from there draws as anywhere.
    // Other buttons pan as ever. Out of the mode a press is the page's (a pan) or a patch box's: its click decides.
    canvas.addEventListener('pointerdown',e=>{
      this.tap=null;
      if(e.button!==0)return;
      if(!on()){if(this.active)this.tap={id:e.pointerId,x:e.offsetX,y:e.offsetY,reach:0,up:null};return;}
      stop(e);canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);this.pressing=true;
      this.fresh=this.draft.active?null:this.stage.toWorld(e.offsetX,e.offsetY);
      answer(this.draft.down(...at(e)));
    },true);
    // How far a press out of the mode has gone from where it began: a click only under the outline's drag threshold.
    const reach=e=>{
      const tap=this.tap;
      if(tap?.id===e.pointerId)tap.reach=Math.max(tap.reach,Math.hypot(e.offsetX-tap.x,e.offsetY-tap.y));
      return tap?.id===e.pointerId?tap:null;
    };
    // A drag of the main button is the outline's alone. With no button held the dashed line follows the pointer, the hand
    // shows over an area a click would select (hover), and the page's own pointer work (its tooltip) goes on. A pan's
    // moves (right or middle button) are not passed on: the page pans in its listener after this one, so the point under
    // the pointer here would be a step behind the view; the outline, which learns the view's scale from the moves it is
    // given (lasso.mjs), is told the view moved instead.
    canvas.addEventListener('pointermove',e=>{
      reach(e);
      if(!e.buttons){this.pointer=[e.offsetX,e.offsetY];this.hover();}
      if(!on())return;
      if(!this.pressing && e.buttons){this.draft.viewChanged();return;}
      if(this.pressing)stop(e);
      // A drag takes every move the browser merged into this one (mergedMoves), so a quick stretch keeps its corners.
      let result='none';
      for(const move of this.pressing?mergedMoves(e):[e])result=this.draft.move(...at(move),this.pressing);
      answer(result);
    },true);
    canvas.addEventListener('pointerup',e=>{
      const tap=reach(e);if(tap)tap.up=e;
      if(!this.pressing)return;
      stop(e);this.pressing=false;
      if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
      const fresh=this.fresh;this.fresh=null;
      if(!on())return;
      // A click that began no outline (its one corner, never a stretch) inside an area, or an outline on its way, selects
      // it in place of the corner (pick); any other press lets the selection go.
      const result=this.draft.up(...at(e));
      const area=fresh && result==='changed' && this.draft.points().length===1?this.pick(fresh):null;
      if(area){this.draft.reset();this.select(area);}          // its corner was never drawn (picking)
      else {this.select(null);answer(result);}
      this.hover();                                            // an outline begun: no hand
    },true);
    // A press that ends without a release (a lost pointer) takes back only what it placed or drew; the outline stays.
    for(const name of ['pointercancel','lostpointercapture'])canvas.addEventListener(name,()=>{
      if(!this.pressing)return;
      this.pressing=false;this.fresh=null;if(this.draft.press)answer(this.draft.key({key:'Backspace'}));
    },true);
    // Out of the mode a click inside an area selects it (pick) and one elsewhere lets the selection go, once every
    // listener has had the press: never after a drag (a pan). A press a patch box took (stain-patches.mjs prevents its
    // release's default) is the box's, and lets the selection go too, so that Delete is never taken for the box's.
    // The keys then come to the image, not to a field focused before.
    canvas.addEventListener('click',e=>{
      const tap=this.tap;this.tap=null;
      if(!tap?.up || on() || !this.active)return;
      if(tap.up.defaultPrevented){this.select(null);return;}
      if(tap.reach>=this.draft.dragThreshold)return;
      const area=this.pick(this.stage.toWorld(e.offsetX,e.offsetY));
      this.select(area);
      if(area)canvas.focus({preventScroll:true});
    });
    canvas.addEventListener('pointerleave',()=>{this.pointer=null;canvas.classList.remove('over-area');});
    // In the mode a double-click closes the outline being drawn. With none being drawn it is the page's (it fits the
    // view, as everywhere else), and what its clicks began goes.
    canvas.addEventListener('dblclick',e=>{
      if(!on())return;
      const result=this.draft.doubleClick();
      if(result==='none'){this.changed();return;}
      stop(e);answer(result);
    },true);
    // While the panel is shown; keys typed in a text field or a dialog stay theirs. With an area selected: Delete brings
    // its cells back, and so does Backspace while no outline is being drawn; Escape lets it go, before anything else. In
    // the mode, with an outline being drawn: Enter closes it (not on a focused button: there it presses the button),
    // Backspace takes back its last corner or stretch, Escape drops it; Escape with none leaves the mode. Ctrl+Z undoes,
    // Ctrl+Shift+Z and Ctrl+Y redo, as a press of Undo or Redo would, while the mode is on or the focus is in the panel.
    // A key taken here goes no further: neither the page (main.js: Escape lets the cell go) nor the patch panel (all on
    // the document, after this panel's) acts on it as well.
    document.addEventListener('keydown',e=>{
      if(this.panel.hidden || e.defaultPrevented || typing(e.target) || document.querySelector?.('dialog[open]'))return;
      const key=String(e.key).toLowerCase(),plain=!e.ctrlKey && !e.metaKey && !e.altKey;
      const outline=this.drawing && this.draft.active,take=()=>{e.preventDefault();e.stopImmediatePropagation?.();};
      if(plain && this.selectedArea() && (key==='escape' || key==='delete' || key==='backspace' && !outline)) {
        take();
        if(key==='escape')this.select(null);else this.dropSelected();
      } else if(outline && plain && ['enter','backspace','escape'].includes(key) &&
          !(key==='enter' && e.target?.closest?.('button'))) {
        take();answer(this.draft.key(e));this.hover();          // the outline gone: the hand again over an area
      } else if(key==='escape') {
        if(!this.drawing)return;
        take();this.stop();
      } else if((e.ctrlKey || e.metaKey) && !e.altKey && (key==='z' || key==='y') &&
          (this.drawing || this.panel.contains(e.target))) {
        take();(key==='y' || e.shiftKey?this.redoButton:this.undoButton).click();
      }
    });
  }
  /** For the harness (main.js hook): the mode, the change on its way (busy) and how many wait their turn (queued), the
   *  Undo and Redo steps there are, the areas and their version, the area clicked on the image as shown (selected: its
   *  id; null with none, and for an outline clicked on its way until the engine names it), the points of the outline
   *  being drawn, the size rule (µm², null without one) and the cells it removes, the brightness rules ({channel: {min,
   *  max}}) and the cells they remove, and the Keeps line's counts (kept of cells: live while a handle is held or glides
   *  after Undo or Redo). */
  snapshot() {
    const {cells,kept}=this.counts();
    return {active:this.active,drawing:this.drawing,busy:this.busy,loading:this.loading,queued:this.waiting.length,
      undo:this.undos.length,redo:this.redos.length,version:this.data?.version??null,removed:this.data?.removed??null,
      areas_version:this.data?.areas_version??null,areas:this.data?.areas.length??null,
      selected:this.selectedArea()?.id??null,outline:this.draft.points().length,size:this.data?.size??null,
      small:this.data?.by_size??null,brightness:this.data?.brightness??{},dim:this.data?.by_brightness??null,kept,cells};
  }
  get queued() {return this.waiting.length;}
  /** A change not settled yet (main.js exclusionsPending: Save and Proceed wait, the results card says so): a handle
   *  held, or a change on its way or waiting its turn. */
  get pending() {return !!(this.busy || this.waiting.length || this.sizeChooser?.gesture || this.brightChooser?.gesture);}
}
