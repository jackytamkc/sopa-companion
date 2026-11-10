import { COLORMAPS, colourOf, isHexColour } from './image-layers.mjs';
import { QC_TEXT } from './qc-copy.mjs';
import { STAIN_TEXT } from './stain.mjs';
import { SEG_TEXT } from './seg-layers.mjs';
const $=id=>document.getElementById(id);
const NUMBER=new Intl.NumberFormat('en-GB');
const SEGMENTATIONS=Symbol('segmentations');   // where the Load Other Mask group goes among the rows
const eyeIcon=on=>`<svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true"><path d="M1 10 Q10 0 19 10 Q10 20 1 10Z" fill="none" stroke="currentColor"/><circle cx="10" cy="10" r="3" fill="currentColor"/>${on?'':'<path d="M2 18L18 2" stroke="currentColor" stroke-width="2"/>'}</svg>`;
// Colour by: identity, cluster, or mixing (the mixing check's values). Mixing is offered
// only while the open run's check is on (main.js sets mixingAvailable); the stage colours the cells (stage.js).
export const CELL_COLOR_BY=['identity','cluster','mixing'];
// The display modes (the Cells / Outlines buttons, keys 1 and 2): stage options.mode is the index.
export const DISPLAY_MODES=Object.freeze(['cells','outlines']);
const KEY='sopa-companion:layers:v1';
// A channel's look is kept by its name (it is on, its colours, gamma, opacity, blending: the same stain looks the same in
// every sample), its contrast limits per sample: one sample's limits saturated another's image of the same channel
// name. The samples whose limits are kept, the least recently changed forgotten first.
const CHANNEL_STYLE=['on','gamma','opacity','colormap','blending'];
export const CONTRAST_SAMPLES=50;
const pick=(o,keys)=>Object.fromEntries(keys.filter(k=>o&&typeof o==='object'&&k in o).map(k=>[k,o[k]]));
/** Saved stage settings as the stage can take them: a display mode it no longer has (2, Activity, removed by
 *  an earlier version) is dropped, so the default (Cells) stands. Returns the settings and whether anything was
 *  dropped. */
export function savedStage(saved) {
  const stage=saved&&typeof saved==='object'&&!Array.isArray(saved)?{...saved}:{};
  const dropped='mode' in stage&&!(Number.isInteger(stage.mode)&&stage.mode>=0&&stage.mode<DISPLAY_MODES.length);
  if(dropped)delete stage.mode;
  return {stage,dropped};
}
export class LayerPanel {
  constructor(stage,priors) {
    this.stage=stage;this.priors=priors;this.channels=[];this.selected='cells';this.mixingAvailable=false;this.mixingWhy=null;
    try{const saved=JSON.parse(localStorage.getItem(KEY)||'{}');this.saved=saved&&typeof saved==='object'&&!Array.isArray(saved)?saved:{};}catch{this.saved={};}
    // Limits saved by channel name (by an earlier version) are left out: each sample's image opens with its own until
    // changed there.
    const channels=this.saved.channels&&typeof this.saved.channels==='object'?this.saved.channels:{};
    this.saved.channels=Object.fromEntries(Object.entries(channels).map(([name,c])=>[name,pick(c,CHANNEL_STYLE)]));
    const contrast=this.saved.contrast;this.saved.contrast=contrast&&typeof contrast==='object'&&!Array.isArray(contrast)?contrast:{};
    this.sample=null;   // the sample whose image the channels show (main.js openRun): their limits are kept under it
    const {stage:settings,dropped}=savedStage(this.saved.stage);
    if(this.saved.stage)this.saved.stage=settings;
    stage.set(settings);
    for(const key of ['prior','nuclei','xenium']) Object.assign(priors.settings[key],this.saved[key]||{});
    this.priorChosen=this.saved.priorChosen===true;
    // The sample's segmentation layers (PriorLayers' store, seg-layers.mjs): a group of rows after the run's own outlines.
    this.segments=priors.segments ?? null;this.segAdd=false;this.segHosts=[];this.segSample=this.segments?.sample ?? null;
    this.segEl=document.createElement('div');this.segEl.className='seg-group';this.segEl.setAttribute('data-seg-group','');
    this.segments?.listen(()=>this.segChanged());
    const panel=$('display-panel');
    this.list=document.createElement('div');this.list.className='layer-list';
    this.controls=document.createElement('div');this.controls.className='layer-controls';
    this.cellControls=document.createElement('div');
    for(const id of ['mode','trails','fill','layers-row']) {const el=$(id);this.cellControls.append(id==='trails'||id==='fill'?el.parentElement:el);}
    $('transcripts').parentElement.hidden=true;$('gain').parentElement.hidden=true;$('channels').hidden=true;
    panel.append(this.list,this.controls,this.cellControls);
    $('transcripts').checked=stage.options.transcripts;$('trails').checked=stage.options.trails;$('fill').value=stage.options.fill;
    panel.addEventListener('input',()=>this.save());panel.addEventListener('change',()=>this.save());
    this.render();
    if(dropped)this.save();   // the settings saved with the removed mode, migrated once
  }
  save() {
    const o=this.stage.options;
    this.saved.stage=Object.fromEntries(['cells','transcripts','markOffFootprint','transcriptAlpha','transcriptSize','transcriptColorBy','cellColorBy','mode','fill','trails'].map(k=>[k,o[k]]));
    for(const k of ['prior','nuclei','xenium']) this.saved[k]={...this.priors.settings[k]};
    this.saved.priorChosen=this.priorChosen;
    this.saved.channels ||= {};this.saved.contrast ||= {};
    for(const c of this.channels) {
      this.saved.channels[c.name]=pick(c,CHANNEL_STYLE);
      if(c.sample==null)continue;
      const limits={...this.saved.contrast[c.sample],[c.name]:{low:c.low,high:c.high,depth:c.depth}};
      delete this.saved.contrast[c.sample];this.saved.contrast[c.sample]=limits;   // the most recent last
    }
    for(const sample of Object.keys(this.saved.contrast).slice(0,-CONTRAST_SAMPLES))delete this.saved.contrast[sample];
    try{localStorage.setItem(KEY,JSON.stringify(this.saved));}catch{/* optional */}
    this.onChange?.();
  }
  /** The image's channels (each new view image of a whole-sample, stain or patches run brings new ones): a channel
   *  selected stays selected by its name; any other row selected (a segmentation layer, a Xenium mask) stays as it is. */
  setChannels(channels) {if(typeof this.selected==='object'&&this.channels.includes(this.selected))this.selected=channels.find(c=>c.name===this.selected.name)||'cells';this.channels=channels;this.render();}
  /** Whether channel k is shown: as listed now, else as saved, else only the first channel. */
  channelOn(meta,k) {const c=this.channels.find(c=>c.name===meta.name);return c?!!c.on:(this.saved.channels?.[meta.name]?.on ?? k===0);}
  /** source: an image, {data: Uint16Array, width, height}, or null for a channel of a whole-sample view that is not
   *  loaded (hidden channels are not fetched; such views are always 16-bit). Its look as saved for its name; its contrast
   *  limits as saved for this sample (`sample`, which the channel keeps), else the image's own statistics. */
  channel(meta,source,k) {
    const depth=!source || source.data?16:8,max=depth===16?(meta.max||65535):255;
    const style=pick(this.saved.channels?.[meta.name],CHANNEL_STYLE),limits=this.sample==null?null:this.saved.contrast?.[this.sample]?.[meta.name];
    const c={...meta,source,depth,max,low:depth===16?(meta.p_lo??0):0,high:depth===16?(meta.p_hi??max):255,gamma:1,opacity:1,colormap:['gray','yellow','green','magenta'][k%4],blending:'additive',...style,on:this.channelOn(meta,k),sample:this.sample};
    if(limits?.depth===depth&&Number.isFinite(limits.low)&&Number.isFinite(limits.high)){c.low=limits.low;c.high=limits.high;}
    c.depth=depth;c.low=Math.max(0,Math.min(max,c.low));c.high=Math.max(c.low,Math.min(max,c.high));return c;
  }
  autoPrior(first) {if(!this.priorChosen && this.priors.settings.prior.on!==first){this.priors.settings.prior.on=first;this.stage.dirty=true;this.render();}}
  /** Whether Colour by may be Mixing (the open run's check is on and can run). The check is off by default: Mixing is
   *  then not listed at all (no reason given), unless the cells are coloured by it right now (main.js puts their colours
   *  back). On but unable to run, it is listed, disabled, with `why` in plain words as its tooltip. The cell controls are
   *  drawn again only on a change. */
  setMixingAvailable(on,why=null) {on=!!on;why=on?null:why||null;if(this.mixingAvailable===on&&this.mixingWhy===why)return;this.mixingAvailable=on;this.mixingWhy=why;if(this.selected==='cells')this.render();}
  /** The cells' Colour by choices now: Mixing only while it is offered (setMixingAvailable). */
  cellColourChoices() {return this.mixingAvailable||this.mixingWhy||this.stage.options.cellColorBy==='mixing'?CELL_COLOR_BY:CELL_COLOR_BY.filter(c=>c!=='mixing');}
  render() {
    this.list.replaceChildren();
    const zarr=this.sampleKind==='zarr';
    if(zarr && Object.values(this.masks?.settings||{}).includes(this.selected))this.selected='cells';
    if(this.segments?.isLayer(this.selected) && !this.segments.layers.includes(this.selected))this.selected='cells';   // taken off the list
    const stain=['stain','patches'].includes(this.runKind);
    if(this.runKind==='patches' && typeof this.selected==='string' && this.channels.length)this.selected=this.channels[0];
    if(stain && typeof this.selected==='string' && this.selected!=='xenium')this.selected='xenium';
    const rows=[...(this.runKind==='patches'?[]:stain?[[STAIN_TEXT.outlines,'xenium',this.priors.settings.xenium.color]]:[['Transcripts','transcripts','#c5cbd5'],['Cells','cells','#df9ebc'],['Proseg prior','prior','#b6a2d5'],[zarr?'Prior nuclei':'Xenium nuclei','nuclei',this.priors.settings.nuclei.color],[zarr?'Prior shapes':'Xenium cells','xenium',this.priors.settings.xenium.color],...Object.entries(zarr?{}:this.masks?.settings||{}).map(([k,o])=>[k==='cellMasks'?'Xenium cell masks':'Xenium nucleus masks',o,'#aaa'])]),SEGMENTATIONS,...this.channels.map(c=>[c.name,c,c.colormap])];
    this.segShown=false;
    for(const entry of rows) {
      if(entry===SEGMENTATIONS) {if(this.segments?.sample){this.list.append(this.segEl);this.segShown=true;}continue;}
      const [name,key,color]=entry;
      const row=document.createElement('div');row.className='layer-row';
      const eye=document.createElement('button');eye.className='layer-eye';
      const on=typeof key==='object'?key.on:['cells','transcripts'].includes(key)?this.stage.options[key]:this.priors.settings[key].on;
      eye.innerHTML=eyeIcon(on);eye.setAttribute('aria-label',`Toggle ${name}`);eye.setAttribute('aria-pressed',String(on));
      eye.onclick=()=>{if(typeof key==='object'){key.on=!on;if(Object.values(this.masks?.settings||{}).includes(key)&&key.on)this.masks.load();}else if(['cells','transcripts'].includes(key)){this.stage.set({[key]:!on});if(key==='transcripts')$('transcripts').checked=!on;}else{this.priors.settings[key].on=!on;if(key==='prior')this.priorChosen=true;}this.stage.dirty=true;this.save();this.render();if(this.channels.includes(key))this.onChannelsChange?.();};
      const select=document.createElement('button');select.className='layer-name';select.textContent=name;select.setAttribute('aria-pressed',String(this.selected===key));
      const ramp=COLORMAPS.includes(color)||(typeof key==='object'&&key.colormap===color&&isHexColour(color));   // a channel's own colour is a ramp too
      const sw=document.createElement('i');sw.style.background=ramp?`linear-gradient(90deg,#111,${color==='gray'?'white':color==='viridis'?'#fde725':color==='magma'||color==='inferno'||color==='turbo'?'#fc8961':color})`:color;
      select.prepend(sw);select.onclick=()=>{this.selected=key;this.render();};row.append(eye,select);this.list.append(row);if(key==='cells'&&this.analysisControls)this.list.append(this.analysisControls);
    }
    this.renderSegs();   // the group (when shown) and its copy in the picker (emptied without a sample)
    if(!zarr && (this.masks?.loading||this.masks?.error)){const note=document.createElement('small');note.textContent=this.masks.loading?'Loading Xenium masks...':this.masks.error;this.list.append(note);}
    const key=this.selected;this.controls.replaceChildren();this.cellControls.hidden=key!=='cells';
    const range=(label,value,min,max,step,fn)=>{const input=document.createElement('input');input.type='range';Object.assign(input,{min,max,step,value});input.oninput=()=>{fn(Number(input.value));this.stage.dirty=true;this.save();};this.field(label,input);};
    const select=(label,value,choices,fn,disabled=new Set())=>{const el=document.createElement('select');for(const item of choices){const opt=document.createElement('option');opt.value=item;opt.textContent=item==='cell'?'Cell':item==='identity'?'Identity':item==='cluster'?'Cluster':item==='mixing'?QC_TEXT.mixing:item;if(disabled.has(item)){opt.disabled=true;opt.title=this.mixingWhy||QC_TEXT.mixingOffTooltip;}el.append(opt);}el.value=value;el.onchange=()=>{fn(el.value);this.stage.dirty=true;this.save();};this.field(label,el);return el;};
    if(key==='cells'){select('Colour by',this.stage.options.cellColorBy,this.cellColourChoices(),v=>this.onColorBy?this.onColorBy(v):this.stage.set({cellColorBy:v}),this.mixingAvailable?new Set():new Set(['mixing']));}
    else if(key==='transcripts'){range('Size (0.5–12 px; [ / ])',Math.log2(this.stage.options.transcriptSize),-1,Math.log2(12),.01,v=>this.stage.set({transcriptSize:2**v}));const mark=document.createElement('input');mark.type='checkbox';mark.checked=this.stage.options.markOffFootprint;mark.onchange=()=>{this.stage.set({markOffFootprint:mark.checked});this.save();};this.field('Mark off-footprint transcripts',mark);range('Opacity',this.stage.options.transcriptAlpha,0,1,.01,v=>this.stage.set({transcriptAlpha:v}));select('Colour by',this.stage.options.transcriptColorBy,['cell','cluster'],v=>this.stage.set({transcriptColorBy:v}));}
    else if(this.segments?.layers.includes(key)) {
      // A segmentation layer: its colour, opacity, line width, outlines or filled, and Remove (the store keeps the style).
      const layer=key,o=layer.style,store=this.segments;
      const color=document.createElement('input');color.type='color';color.value=o.color;color.oninput=()=>store.setStyle(layer,{color:color.value});this.field(SEG_TEXT.colour,color);
      range(SEG_TEXT.opacity,o.opacity,0,1,.01,v=>store.setStyle(layer,{opacity:v}));range(SEG_TEXT.width,o.width,.5,5,.1,v=>store.setStyle(layer,{width:v}));
      const fill=document.createElement('select');
      for(const [value,label] of [['outline',SEG_TEXT.outlines],['fill',SEG_TEXT.filled]]){const opt=document.createElement('option');opt.value=value;opt.textContent=label;fill.append(opt);}
      fill.value=o.fill?'fill':'outline';fill.onchange=()=>store.setStyle(layer,{fill:fill.value==='fill'});this.field(SEG_TEXT.showAs,fill);
      const remove=document.createElement('button');remove.type='button';remove.className='seg-remove';remove.textContent=SEG_TEXT.remove;remove.title=SEG_TEXT.removeTitle;remove.setAttribute('data-action','seg-remove');
      remove.onclick=()=>store.remove(layer);this.controls.append(remove);   // its change selects the run's own again (segChanged)
    }
    else if(Object.values(this.masks?.settings||{}).includes(key)) {range('Opacity',key.opacity,0,1,.01,v=>key.opacity=v);const outline=document.createElement('input');outline.type='checkbox';outline.checked=key.outline;outline.onchange=()=>{key.outline=outline.checked;this.stage.dirty=true;};this.field('Outline only',outline);}
    else if(typeof key==='object') {
      const c=key;this.contrast(c);
      range('Gamma',c.gamma,.2,3,.01,v=>c.gamma=v);
      // The colormap list, or a colour of its own as segmentation layers take (HEX or RGB in the picker): black to it.
      select('Colormap',c.colormap,isHexColour(c.colormap)?[...COLORMAPS,c.colormap]:COLORMAPS,v=>{c.colormap=v;this.render();});
      const colour=document.createElement('input');colour.type='color';colour.value=colourOf(c.colormap);colour.title='Any colour (HEX or RGB): the channel goes from black to it';
      colour.oninput=()=>{c.colormap=colour.value;this.stage.dirty=true;this.save();};colour.onchange=()=>this.render();this.field('Colour',colour);
      range('Opacity',c.opacity,0,1,.01,v=>c.opacity=v);select('Blending',c.blending,['additive','translucent'],v=>c.blending=v);
      const note=document.createElement('small');note.textContent=c.depth===16?'16-bit intensity':'8-bit PNG fallback';this.controls.append(note);
    } else if(this.priors.settings[key]) {
      const o=this.priors.settings[key];range('Opacity',o.opacity,0,1,.01,v=>o.opacity=v);
      if(key!=='prior'){const color=document.createElement('input');color.type='color';color.value=o.color;color.oninput=()=>{o.color=color.value;this.stage.dirty=true;this.save();};this.field('Colour',color);range('Line width',o.width,.5,5,.1,v=>o.width=v);}
    }
  }
  field(label,input){const row=document.createElement('label');row.className='layer-field';row.append(document.createTextNode(label),input);this.controls.append(row);}
  /** The store changed (a layer loaded, failed, listed, taken off or restyled; the sample's list read; another sample):
   *  the group drawn again, or the whole panel when the selected layer left the list or the group comes or goes. Another
   *  sample closes Add a mask… (its list was the last sample's; opened again, it reads this one's). */
  segChanged() {
    const s=this.segments;
    if(s.sample!==this.segSample){this.segSample=s.sample;this.segAdd=false;}
    if((s.isLayer(this.selected) && !s.layers.includes(this.selected)) || !!s.sample!==this.segShown)this.render();
    else this.renderSegs();
  }
  /** Also draw the group, without the selected layer's fields, into `host` (the run picker, where Layers is not shown). */
  mountSegmentations(host) {if(host && !this.segHosts.includes(host)){this.segHosts.push(host);this.fillSegs(host,true);}}
  renderSegs() {if(this.segShown)this.fillSegs(this.segEl,false);for(const host of this.segHosts)this.fillSegs(host,true);}
  /** Layers > Load Other Mask in `group`: a row per layer listed (its eye; its colour and name over
   *  its key, or how its load goes; Try again after a failure), then Add a mask… and, while it is open, the
   *  sample's others (as its list names them, with their keys and shapes), and the note of what could not be done (a
   *  fifth shown). `compact` (the picker): the names select nothing. */
  fillSegs(group,compact) {
    const store=this.segments;group.replaceChildren();
    if(!store?.sample)return;
    const el=(tag,className,text)=>{const node=document.createElement(tag);if(className)node.className=className;if(text!=null)node.textContent=text;return node;};
    group.append(el('div','seg-head',SEG_TEXT.group));
    for(const layer of store.layers) {
      const o=layer.style,row=el('div','layer-row seg-row');row.setAttribute('data-seg-key',layer.key);
      const eye=el('button','layer-eye');eye.innerHTML=eyeIcon(o.on);eye.setAttribute('aria-label',`Toggle ${layer.name}`);eye.setAttribute('aria-pressed',String(o.on));
      eye.onclick=()=>store.show(layer,!o.on);   // the store's change draws the group (and the stage) again
      const name=el(compact?'span':'button','layer-name'),sw=el('i'),label=el('span','seg-label');
      if(!compact){name.setAttribute('aria-pressed',String(this.selected===layer));name.onclick=()=>{this.selected=layer;this.render();};}
      sw.style.background=o.color;label.append(el('span','',layer.name));
      if(layer.name!==layer.key)label.append(el('small','',layer.key));
      if(['loading','failed'].includes(layer.state))label.append(el('small','seg-state',layer.state==='loading'?SEG_TEXT.loading:SEG_TEXT.failed));
      const count=store.count(layer);
      name.title=`${layer.name} (${layer.key})${count!=null?`: ${NUMBER.format(count)} ${SEG_TEXT.shapes}`:''}${layer.error?`. ${SEG_TEXT.failed}: ${layer.error}`:''}`;
      name.append(sw,label);row.append(eye,name);
      if(layer.state==='failed'){const retry=el('button','seg-retry',SEG_TEXT.retry);retry.setAttribute('data-action','seg-retry');retry.onclick=()=>store.retry(layer);row.append(retry);}
      group.append(row);
    }
    const addRow=el('div','layer-row seg-add-row'),add=el('button','layer-name seg-add',SEG_TEXT.add);
    add.setAttribute('data-action','seg-add');add.setAttribute('aria-expanded',String(this.segAdd));
    add.onclick=()=>{this.segAdd=!this.segAdd;if(this.segAdd)store.loadCatalogue(true);this.renderSegs();};   // read anew at each opening: a segmentation saved since is offered
    addRow.append(add);group.append(addRow);
    if(this.segAdd) {
      const box=el('div','seg-add-list'),known=store.catalogue,items=store.addable();
      if(!known || known.state==='loading')box.append(el('p','',SEG_TEXT.reading));
      else if(known.state==='failed'){const retry=el('button','seg-retry',SEG_TEXT.retry);retry.setAttribute('data-action','seg-list-retry');retry.onclick=()=>store.loadCatalogue();box.append(el('p','',SEG_TEXT.readFailed),retry);}
      else if(!known.items.length)box.append(el('p','',SEG_TEXT.none));
      else if(!items.length)box.append(el('p','',SEG_TEXT.allAdded));
      else for(const item of items) {
        const pick=el('button','seg-pick'),about=[item.name && item.name!==item.key?item.key:'',Number.isFinite(item.count)?`${NUMBER.format(item.count)} ${SEG_TEXT.shapes}`:''].filter(Boolean).join(' · ');
        pick.setAttribute('data-seg-add-key',item.key);pick.append(el('span','',item.name || item.key));if(about)pick.append(el('small','',about));
        pick.onclick=()=>{this.segAdd=false;if(!store.add(item.key))this.renderSegs();};   // added: the store's change draws the group again
        box.append(pick);
      }
      group.append(box);
    }
    if(store.note){const note=el('small','seg-note',store.note);note.setAttribute('role','status');group.append(note);}
  }
  contrast(c) {
    const title=document.createElement('div');title.className='muted small';title.textContent='Contrast limits';this.controls.append(title);
    const wrap=document.createElement('div');wrap.className='contrast';const canvas=document.createElement('canvas');canvas.width=256;canvas.height=55;wrap.append(canvas);
    const g=canvas.getContext('2d'),hist=c.hist||[],peak=Math.max(1,...hist.map(n=>Math.log1p(n)));g.fillStyle='#637389';hist.forEach((n,i)=>{const h=55*Math.log1p(n)/peak;g.fillRect(i,55-h,1,h);});
    const fields=document.createElement('div');fields.className='contrast-values';const inputs=[];
    for(const [key,label] of [['low','Minimum'],['high','Maximum']]) {
      const slider=document.createElement('input'),number=document.createElement('input');slider.type='range';number.type='number';
      for(const input of [slider,number]){input.min=0;input.max=c.max;input.step=1;input.value=c[key];input.setAttribute('aria-label',`Contrast ${label}`);input.oninput=()=>{if(!input.checkValidity())return;const v=Number(input.value);c[key]=key==='low'?Math.min(v,c.high):Math.max(v,c.low);for(const [el,k] of inputs)el.value=c[k];this.stage.dirty=true;this.save();};inputs.push([input,key]);}
      wrap.append(slider);fields.append(number);
    }
    const auto=document.createElement('button');auto.textContent='Auto';auto.onclick=()=>{c.low=c.depth===16?(c.p_lo??0):0;c.high=c.depth===16?(c.p_hi??c.max):255;for(const [el,k]of inputs)el.value=c[k];this.stage.dirty=true;this.save();};fields.append(auto);this.controls.append(wrap,fields);
  }
}
