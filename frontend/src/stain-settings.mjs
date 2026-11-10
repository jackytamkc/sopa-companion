import { guard, explain, busy, clearExplanation, problem } from './explain.mjs';
import { matchResolution, resolutionChoices, levelName, STAIN_TEXT } from './stain.mjs';

export const SETTINGS_TEXT = Object.freeze({
  use:'Use settings from Cellpose', missing:'No corrected patch saved by the Cellpose GUI yet for this sample.',
  loading:'Checking Cellpose settings…', error:'Could not read Cellpose settings. Reopen the picker to try again.',
  differences:'Some patches have different settings. The newest file was used.',
  unknown:'Model not in the list; the current model was kept:',
  newRun:'Opens the picker with these settings for a new run or patch item.',
});

export function updateNormalizeFields(get) {
  const enabled=get('stain-normalize').checked, fixed=get('stain-fixed-range').checked;
  get('stain-fixed-range').disabled=!enabled;
  get('stain-low').disabled=get('stain-high').disabled=!enabled || fixed;
  get('stain-intensity-low').disabled=get('stain-intensity-high').disabled=!enabled || !fixed;
}

/** Only fields returned by the GUI endpoint are copied; run-only choices stay as the user set them. Each is reported by
 *  the picker's own name for it (Cellpose's names; the resolution by its level's name). */
export function fillCellposeSettings(get, result, image=null) {
  const s=result.settings, lines=[], T=STAIN_TEXT;
  const field=(id,label,value)=>{get(id).value=String(value);lines.push(`${label}: ${value}`);};
  if(s.model)field('stain-model','Model',s.model);
  // Pixel-valued settings are already converted by the engine. Only the image copy is matched here.
  const resolution=result.resolution, level=get('stain-level');
  if(level && image && resolution?.um_per_px>0) {
    const matched=matchResolution(image,resolution), choice=resolutionChoices(image).find(s=>s.level===matched);
    if(choice?.um_per_px>0 && Math.abs(choice.um_per_px/resolution.um_per_px-1)<=.05) {
      level.value=String(matched);lines.push(`Resolution: ${choice.label} (the patches')`);
    } else lines.push('Resolution: the current choice was kept; this image has no matching copy.');
  } else if(level && image && resolutionChoices(image).some(choice=>choice.level===s.level)) {
    level.value=String(s.level);lines.push(`Resolution: ${levelName(s.level)} (the patches')`);
  }
  const auto=s.diameter==null || s.diameter==='auto';
  get('stain-auto').checked=auto;get('stain-diameter').disabled=auto;
  if(!auto)get('stain-diameter').value=String(s.diameter);
  lines.push(`${T.diameter}: ${auto?'Auto':s.diameter}`);
  field('stain-flow',T.flow,s.flow_threshold);
  field('stain-probability',T.probability,s.cellprob_threshold);
  const n=s.normalize;
  get('stain-normalize').checked=n.normalize!==false;
  get('stain-fixed-range').checked=n.lowhigh!=null;
  get('stain-intensity-low').value=String(n.lowhigh?.[0] ?? '');
  get('stain-intensity-high').value=String(n.lowhigh?.[1] ?? '');
  get('stain-low').value=String(n.percentile?.[0] ?? 1);get('stain-high').value=String(n.percentile?.[1] ?? 99);
  updateNormalizeFields(get);
  lines.push(`${T.normalize}: ${n.normalize===false?'off':n.lowhigh!=null?'fixed range':`percentile ${(n.percentile || [1,99]).join('–')}`}`);
  if(n.lowhigh!=null)lines.push(`${T.fixedRange}: ${n.lowhigh.join('–')}`);
  field('stain-tile-norm',T.tileNorm,n.tile_norm_blocksize);
  field('stain-sharpen',T.sharpen,n.sharpen_radius);
  field('stain-smooth',T.smooth,n.smooth_radius);
  get('stain-invert').checked=n.invert;lines.push(`${T.invert}: ${n.invert?'on':'off'}`);
  return lines;
}

export function settingsReport(node,result,lines) {
  node.replaceChildren();
  const add=(tag,words)=>{const n=document.createElement(tag);n.textContent=words;node.append(n);return n;};
  add('p',`From ${result.from}`);
  const list=add('ul','');
  for(const words of lines){const li=document.createElement('li');li.textContent=words;list.append(li);}
  if(Object.keys(result.differences || {}).length) {
    add('p',SETTINGS_TEXT.differences);
    for(const [key,files] of Object.entries(result.differences))
      add('p',`${key}: ${Object.entries(files).map(([file,value])=>`${file} = ${JSON.stringify(value)}`).join('; ')}`);
  }
  if(result.unknown_model)add('p',`${SETTINGS_TEXT.unknown} ${result.unknown_model}`);
}

/** The button is never greyed out: a press reads the newest file again and uses it, or says that the
 *  GUI has saved none yet. Reads on their own (sample change, window focus) only keep `data` current. */
export class CellposeSettings {
  constructor(api,button,reason,use) {
    Object.assign(this,{api,button,reason,use});this.ticket=0;
    button.textContent=SETTINGS_TEXT.use;button.disabled=false;
    const missing=[problem(null,'',SETTINGS_TEXT.missing)];
    guard(button,()=>this.sample?[]:missing,async()=>{
      const sample=this.sample,done=busy(button,SETTINGS_TEXT.loading);
      try {
        const data=await this.refresh(sample);
        clearExplanation(button);                     // e.g. "still working" from a second press meanwhile
        if(this.sample!==sample)return;
        if(data?.found)await this.use(data);else if(data)explain(button,missing);
      } finally {done();}
    });
  }
  async refresh(sample) {
    const ticket=++this.ticket;
    this.controller?.abort();this.controller=new AbortController();
    if(this.sample!==sample){this.data=null;this.sample=sample;}
    clearExplanation(this.button);this.reason.textContent='';
    if(!sample)return;
    try {
      const data=await this.api.cellposeSettings(sample,this.controller.signal);
      if(ticket!==this.ticket)return;
      return this.data=data;
    } catch(e) {if(ticket===this.ticket && e.name!=='AbortError')this.reason.textContent=SETTINGS_TEXT.error;}
  }
}

export class TissueView {
  constructor(api,changed,notice) {Object.assign(this,{api,changed,notice});this.cache=new Map();this.key=null;this.polygons=[];}
  update(sample,image,on) {
    const key=on && sample && image ? JSON.stringify([sample,image]) : null;
    if(key===this.key)return;
    this.key=key;this.controller?.abort();this.polygons=this.cache.get(key)?.polygons || [];this.changed();
    if(!key || this.cache.has(key))return;
    const controller=this.controller=new AbortController();
    this.api.tissue(sample,image,controller.signal).then(data=>{
      if(controller.signal.aborted || this.key!==key)return;
      this.cache.set(key,data);this.polygons=data.polygons;this.changed();
    }).catch(e=>{if(e.name!=='AbortError' && this.key===key)this.notice('The tissue outline could not be loaded.');});
  }
}
