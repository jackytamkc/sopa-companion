import { explain, clearExplanation, problem } from './explain.mjs';

export const RANGE_TEXT = Object.freeze({
  noLimit:'no limit', sizeMin:'Smallest cell size', sizeMax:'Largest cell size',
  brightMin:'Lowest brightness', brightMax:'Highest brightness',
  sizeNumber:'type a number from 0 to 1,000,000, or leave it empty for no limit',
  brightNumber:'type a number from −1,000,000,000 to 1,000,000,000, or leave it empty for no limit',
  order:'make the smallest value no more than the largest',
});
export const normaliseRange = range => ({min:range?.min===0?null:range?.min??null,max:range?.max??null});
export const sameRange = (a,b) => (a?.min??null)===(b?.min??null) && (a?.max??null)===(b?.max??null);
const clamp = (v,lo,hi) => Math.max(lo,Math.min(hi,v));
export function roundRangeValue(value, kind='size', integer=true) {
  return kind==='size'?(value<10?Math.round(value*10)/10:Math.round(value)):
    integer?Math.round(value):Number(value.toPrecision(3));
}
const roundAbove = value => {
  if(!(value>0))return 1;
  const power=10**Math.floor(Math.log10(value));
  return Math.ceil(value/power)*power;
};
/** A fixed axis for one cell set: up to 256 bins; size is log10, brightness linear. Missing values have no bar. */
export function rangeHistogram(values=[], {kind='size',pixelArea=.1}={}) {
  const finite=Array.from(values).filter(Number.isFinite).sort((a,b)=>a-b);
  const floor=Math.max(.1,Number.isFinite(pixelArea)?pixelArea:.1);
  const min=kind==='size'?floor:Math.min(0,finite[0]??0);
  const rank=(finite.length-1)*.999,lo=finite[Math.floor(rank)],hi=finite[Math.ceil(rank)];
  const percentile=finite.length?lo+(hi-lo)*(rank%1):0;
  const max=kind==='size'?Math.max(min*10,roundAbove(finite.at(-1) || 1)):
    Math.max(min+1,roundAbove(percentile));
  const position=value=>clamp(kind==='size'?Math.log10(Math.max(min,value)/min)/Math.log10(max/min):(value-min)/(max-min),0,1);
  const value=position=>kind==='size'?min*(max/min)**clamp(position,0,1):min+clamp(position,0,1)*(max-min);
  const edges=kind==='size'?[...new Set(Array.from({length:257},(_,i)=>Math.ceil(value(i/256)/floor)*floor))]:
    Array.from({length:257},(_,i)=>value(i/256));
  const bins=new Uint32Array(edges.length-1);
  const bin=v=>{
    let lo=0,hi=edges.length-1;
    while(lo+1<hi){const mid=(lo+hi)>>1;if(v<edges[mid])hi=mid;else lo=mid;}
    return Math.min(bins.length-1,lo);
  };
  for(const v of finite)bins[bin(v)]++;
  return {bins,edges,bin,min,max,position,value,count:finite.length};
}

/** Small contrast-style chooser. onInput is local, once per frame; onCommit is once per completed gesture.
 *  setRange waits out a held thumb before applying another window's state. cancel/hide never commit. onTween hears the
 *  range the handles stand at while setRange moves them to a changed rule (null once they stop), never a commit. */
export class RangeChooser {
  constructor(root, {kind='size',id=root.id || `range-${kind}`,label='',onInput=()=>{},onCommit=()=>{},onCancel=()=>{},
    onTween=()=>{},canChange=()=>true,calm=()=>false,integer=true,pixelArea=.1}={}) {
    Object.assign(this,{root,kind,onInput,onCommit,onCancel,onTween,canChange,calm,integer,pixelArea});
    this.value=this.committed=normaliseRange(null);this.waiters=[];this.sliders={};this.numbers={};
    const make=(tag,cls)=>{const n=document.createElement(tag);if(cls)n.className=cls;return n;};
    this.wrap=make('div','contrast');this.canvas=make('canvas');this.canvas.width=256;this.canvas.height=55;
    this.wrap.append(this.canvas);this.fields=make('div','contrast-values');root.classList.add('range-chooser');
    const heading=make('div','range-label');heading.textContent=label;heading.hidden=!label;
    root.replaceChildren(heading,this.wrap,this.fields);
    for(const end of ['min','max']) {
      const label=RANGE_TEXT[`${kind==='size'?'size':'bright'}${end==='min'?'Min':'Max'}`];
      const slider=this.sliders[end]=make('input'), number=this.numbers[end]=make('input');
      slider.type='range';slider.min='0';slider.max='1000';slider.step='1';slider.id=`${id}-${end}-handle`;
      number.type='number';number.step='any';number.min=kind==='size'?'0':'-1000000000';number.max=kind==='size'?'1000000':'1000000000';
      number.id=`${id}-${end}`;number.placeholder=RANGE_TEXT.noLimit;
      for(const n of [slider,number])n.setAttribute('aria-label',label);
      this.wrap.append(slider);this.fields.append(number);
      slider.addEventListener('pointerdown',e=>{
        if(e.button!=null && e.button!==0)return;
        if(!this.begin('pointer')){e.preventDefault();return;}
        slider.setPointerCapture?.(e.pointerId);
      });
      slider.addEventListener('input',()=>{
        if(!this.gesture && !this.begin('input')){this.draw();return;}
        const pos=Number(slider.value)/1000;
        let value=(end==='min'?pos<=0:pos>=1)?null:roundRangeValue(this.axis.value(pos),kind,this.integer);
        if(value!==null)value=clamp(value,Number(number.min),Number(number.max));
        const other=this.value[end==='min'?'max':'min'];
        if(value!==null && other!==null)value=end==='min'?Math.min(value,other):Math.max(value,other);
        this.value=normaliseRange({...this.value,[end]:value});this.draw();this.schedule();
        if(this.gesture==='key'){clearTimeout(this.keyTimer);this.keyTimer=setTimeout(()=>this.finish(),400);}
      });
      slider.addEventListener('change',()=>{if(this.gesture!=='key')this.finish();});
      slider.addEventListener('pointerup',()=>{if(this.gesture==='pointer')this.finish();});
      for(const event of ['pointercancel','lostpointercapture','blur'])slider.addEventListener(event,()=>this.cancel());
      slider.addEventListener('keydown',e=>{
        if(e.key==='Escape'){e.preventDefault();this.cancel();return;}
        if(!e.ctrlKey && !e.metaKey && !e.altKey && /^(Arrow|Page|Home$|End$)/.test(e.key))this.begin('key');
      });
      slider.addEventListener('keyup',e=>{if(/^(Arrow|Page|Home$|End$)/.test(e.key) && this.gesture==='key')this.finish();});
      number.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();this.commitBox(end,true);}});
      number.addEventListener('blur',()=>this.commitBox(end,false));
    }
    const unit=make('span');unit.textContent=kind==='size'?'µm²':'';this.fields.append(unit);
    this.axis=rangeHistogram([],{kind,pixelArea});this.draw();
    this.canvas.addEventListener('pointermove',e=>{
      const width=this.canvas.getBoundingClientRect?.().width || 256;
      const i=this.axis.bin(this.axis.value(e.offsetX/width)),lo=this.axis.edges[i],hi=this.axis.edges[i+1];
      this.canvas.title=`${this.axis.bins[i]} cells, ${Number(lo.toPrecision(3))}–${Number(hi.toPrecision(3))}${kind==='size'?' µm²':''}`;
    });
    this.blur=()=>{this.cancel();this.stopAnimation();};this.visibility=()=>{if(document.hidden)this.blur();};
    globalThis.window?.addEventListener?.('blur',this.blur);
    document.addEventListener('visibilitychange',this.visibility);
  }
  setValues(values, {integer=this.integer,pixelArea=this.pixelArea}={}) {
    if(this.values===values && this.pixelArea===pixelArea && this.integer===integer)return;
    this.stopAnimation();   // its positions belong to the old axis: the handles go straight to their place on the new one
    this.values=values;this.integer=integer;this.pixelArea=pixelArea;
    this.axis=rangeHistogram(values,{kind:this.kind,pixelArea});this.draw();
  }
  /** Show `range`. `animate`: the handles glide there over 200 ms (none with calm or reduced motion); `follow` as well
   *  (the rule itself changed: Undo, Redo, another window): onTween hears the range they stand at, the one before at once
   *  and null when they stop, so the outlines come back or go as the handles pass them; the bars follow too. */
  setRange(range, {animate=false,follow=false}={}) {
    const value=normaliseRange(range);
    if(this.gesture){if(this.pending || !sameRange(value,this.committed))this.pending={value,animate,follow};return;}
    if(sameRange(value,this.value)){this.committed=value;return;}
    this.stopAnimation();
    const before=this.value,from=this.positions();this.value=this.committed=value;this.draw();
    if(animate && !this.calm() && !globalThis.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches && globalThis.requestAnimationFrame) {
      const to=this.positions(),start=performance.now();
      const step=now=>{
        const part=clamp((now-start)/200,0,1),at={};
        for(const end of ['min','max']){at[end]=from[end]+(to[end]-from[end])*part;this.sliders[end].value=String(1000*at[end]);}
        this.animation=part<1?requestAnimationFrame(step):null;
        if(!follow)return;
        this.bars(at);
        if(part<1)this.onTween(this.rangeAt(at));else {this.following=false;this.onTween(null);}
      };
      this.animation=requestAnimationFrame(step);
      if(follow){this.following=true;this.bars(from);this.onTween({...before});}
    }
  }
  /** The range at axis positions {min, max} (0 to 1): an end at its end of the axis is no limit. */
  rangeAt(at) {return {min:at.min<=0?null:this.axis.value(at.min),max:at.max>=1?null:this.axis.value(at.max)};}
  begin(kind) {
    if(this.root.hidden || !this.canChange())return false;
    if(!this.gesture){this.stopAnimation();this.gesture=kind;this.before={...this.value};}
    return true;
  }
  schedule() {
    if(this.frame!=null)return;
    if(!globalThis.requestAnimationFrame){this.onInput({...this.value});return;}
    this.frame=requestAnimationFrame(()=>{this.frame=null;this.onInput({...this.value});});
  }
  flush() {if(this.frame!=null){globalThis.cancelAnimationFrame?.(this.frame);this.frame=null;this.onInput({...this.value});}}
  finish() {
    if(!this.gesture)return;
    clearTimeout(this.keyTimer);this.flush();
    const changed=!sameRange(this.value,this.before);this.gesture=null;
    if(this.pending) {const next=this.pending;this.pending=null;this.setRange(next.value,next);this.onCancel({...this.value});}
    else if(changed){this.committed={...this.value};this.onCommit({...this.value});}
    else this.onCancel({...this.value});
    this.resolve();
  }
  cancel() {
    if(!this.gesture)return;
    clearTimeout(this.keyTimer);if(this.frame!=null)globalThis.cancelAnimationFrame?.(this.frame);this.frame=null;
    this.gesture=null;this.value=this.committed=this.pending?.value || this.before;this.pending=null;this.draw();
    this.onCancel({...this.value});this.resolve();
  }
  commitBox(end, focus) {
    const box=this.numbers[end],text=String(box.value).trim();
    if(!this.canChange()){this.draw();return;}
    const raw=text==='' && !box.validity?.badInput?null:Number(text);
    let message=!Number.isFinite(raw) && raw!==null || box.validity?.badInput || raw!==null &&
      (raw<Number(box.min) || raw>Number(box.max))?RANGE_TEXT[this.kind==='size'?'sizeNumber':'brightNumber']:null;
    const value=normaliseRange({...this.value,[end]:raw===null?null:roundRangeValue(raw,this.kind,this.integer)});
    if(!message && value.min!==null && value.max!==null && value.min>value.max)message=RANGE_TEXT.order;
    if(message) {
      if(!focus)this.draw();
      explain(box,[problem(box.id,box.getAttribute('aria-label'),message)],{focus});return;
    }
    clearExplanation(box);
    if(sameRange(value,this.value)){this.draw();return;}
    this.value=this.committed=value;this.draw();this.onInput({...value});this.onCommit({...value});
  }
  positions() {return {min:this.value.min===null?0:this.axis.position(this.value.min),max:this.value.max===null?1:this.axis.position(this.value.max)};}
  draw() {
    const positions=this.positions();
    for(const end of ['min','max']) {
      this.sliders[end].value=String(positions[end]*1000);this.sliders[end].hidden=!this.axis.count;
      this.sliders[end].setAttribute('aria-valuetext',this.value[end]===null?RANGE_TEXT.noLimit:String(this.value[end]));
      this.numbers[end].value=this.value[end]===null?'':String(this.value[end]);
    }
    this.bars(positions);
  }
  /** The histogram, its bars between the handles at `positions` lit. */
  bars(positions) {
    const g=this.canvas.getContext?.('2d');if(!g)return;
    g.clearRect(0,0,256,55);const peak=Math.max(1,...Array.from(this.axis.bins,Math.log1p));
    this.axis.bins.forEach((n,i)=>{
      const h=55*Math.log1p(n)/peak,lo=this.axis.position(this.axis.edges[i]),hi=this.axis.position(this.axis.edges[i+1]),at=(lo+hi)/2;
      g.fillStyle=at<positions.min || at>positions.max?'#39414d':'#637389';g.fillRect(lo*256,55-h,(hi-lo)*256,h);
    });
  }
  setHidden(hidden) {if(hidden)this.cancel();this.root.hidden=hidden;if(hidden)this.stopAnimation();}
  stopAnimation() {
    if(this.animation!=null)globalThis.cancelAnimationFrame?.(this.animation);this.animation=null;
    if(this.following){this.following=false;this.onTween(null);}   // the outlines go straight to the rule shown
  }
  whenSettled() {return this.gesture?new Promise(resolve=>this.waiters.push(resolve)):Promise.resolve();}
  resolve() {for(const resolve of this.waiters.splice(0))resolve();}
  destroy() {
    this.cancel();this.stopAnimation();globalThis.window?.removeEventListener?.('blur',this.blur);
    document.removeEventListener?.('visibilitychange',this.visibility);
  }
}
