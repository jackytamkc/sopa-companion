import test from 'node:test';
import assert from 'node:assert/strict';
import { domFixture } from './dom-fixture.mjs';
import { RangeChooser, rangeHistogram, roundRangeValue, RANGE_TEXT } from '../src/range-chooser.mjs';
import { bannedWords } from '../src/qc-copy.mjs';

function setup(t,options={}) {
  const dom=domFixture(),old=globalThis.document;globalThis.document=dom.document;
  const root=dom.document.createElement('div');root.id='size';dom.document.body.append(root);
  const live=[],commits=[],cancelled=[];
  const chooser=new RangeChooser(root,{onInput:r=>live.push(r),onCommit:r=>commits.push(r),onCancel:r=>cancelled.push(r),...options});
  chooser.setValues(new Float32Array([1,2,4,8,16,32,64,128]));
  t.after(()=>{chooser.destroy();globalThis.document=old;});
  return {...dom,chooser,live,commits,cancelled,min:chooser.sliders.min,max:chooser.sliders.max};
}
function move(input,value) {input.value=String(value);input.emit('input');}
test('size uses a log axis; brightness uses a linear percentile axis and skips NaN',()=>{
  const size=rangeHistogram([1,10,100,NaN]);assert.ok(Math.abs(size.position(10)-2/3)<1e-12);
  assert.ok(Math.abs(size.value(size.position(10))-10)<1e-12);assert.equal(size.count,3);
  const bright=rangeHistogram([-2,0,10,10000,NaN],{kind:'brightness'});
  assert.equal(bright.min,-2);assert.equal(bright.max,10000);assert.equal(bright.bins.reduce((a,b)=>a+b),4);
  assert.equal(rangeHistogram([NaN]).count,0);
  assert.deepEqual([roundRangeValue(3.14),roundRangeValue(15.8),roundRangeValue(123.45,'brightness',false)],[3.1,16,123]);
});

test('size histogram bins span at least one run pixel area',()=>{
  const pixelArea=.425**2,axis=rangeHistogram([1,2,20,50,1000],{pixelArea});
  for(let i=1;i<axis.edges.length;i++)assert.ok(axis.edges[i]-axis.edges[i-1]>=pixelArea-1e-10);
  assert.equal(axis.bins.reduce((a,b)=>a+b),5);
});

test('Undo moves handles only for 200 ms, and calm or reduced motion skips the animation',t=>{
  const frames=new Map();let id=0,now=0,calm=false;
  const old=globalThis.requestAnimationFrame,oldCancel=globalThis.cancelAnimationFrame;
  globalThis.requestAnimationFrame=fn=>{frames.set(++id,fn);return id;};globalThis.cancelAnimationFrame=n=>frames.delete(n);
  t.mock.method(performance,'now',()=>now);
  t.after(()=>{globalThis.requestAnimationFrame=old;globalThis.cancelAnimationFrame=oldCancel;});
  const h=setup(t,{calm:()=>calm});h.chooser.setRange({min:4,max:64},{animate:true});
  assert.equal(frames.size,1);
  const tick=time=>{now=time;const [[n,fn]]=frames;frames.delete(n);fn(now);};
  tick(100);assert.equal(frames.size,1);tick(200);assert.equal(frames.size,0);
  assert.equal(h.live.length,0,'handle motion does not make preview or commit work');
  calm=true;h.chooser.setRange({min:null,max:null},{animate:true});assert.equal(frames.size,0);
});

test('a changed rule (follow): the range the moving handles stand at, the one before at once, null when they stop; the bars follow',t=>{
  const frames=new Map();let id=0,now=0;
  const old=globalThis.requestAnimationFrame,oldCancel=globalThis.cancelAnimationFrame;
  globalThis.requestAnimationFrame=fn=>{frames.set(++id,fn);return id;};globalThis.cancelAnimationFrame=n=>frames.delete(n);
  t.mock.method(performance,'now',()=>now);
  t.after(()=>{globalThis.requestAnimationFrame=old;globalThis.cancelAnimationFrame=oldCancel;});
  const tweens=[],h=setup(t,{onTween:r=>tweens.push(r)}),tick=time=>{now=time;const [[n,fn]]=frames;frames.delete(n);fn(now);};
  // Lit bars (between the handles) as the canvas is drawn.
  let lit=0;h.chooser.canvas.getContext=()=>({clearRect(){lit=0;},fillRect(){if(this.fillStyle==='#637389')lit++;}});
  h.chooser.setRange({min:4,max:64});assert.deepEqual(tweens,[],'no motion, nothing to follow');
  const narrow=lit;now=0;h.chooser.setRange({min:null,max:null},{animate:true,follow:true});
  assert.deepEqual(tweens,[{min:4,max:64}]);assert.equal(lit,narrow,'the bars start where the handles are');
  tick(100);const mid=tweens.at(-1);
  assert.ok(mid.min>h.chooser.axis.min && mid.min<4 && mid.max>64 && mid.max<h.chooser.axis.max,JSON.stringify(mid));
  assert.ok(lit>narrow);const half=lit;
  tick(200);assert.equal(tweens.at(-1),null);assert.equal(frames.size,0);assert.ok(lit>half);
  assert.equal(h.live.length+h.commits.length,0,'never a preview or a commit');
  // Without follow (another channel's rule shown): the handles move, nothing is heard.
  tweens.length=0;h.chooser.setRange({min:4,max:64},{animate:true});tick(300);tick(500);assert.deepEqual(tweens,[]);
  // Stopped by a press on a handle, or by the cells' values changing: null at once, no frame left.
  for(const [min,stop] of [[2,()=>h.min.emit('pointerdown')],[8,()=>h.chooser.setValues(new Float32Array([3,30,300]))]]) {
    tweens.length=0;h.chooser.cancel();h.chooser.setRange({min,max:null},{animate:true,follow:true});
    assert.equal(tweens.length,1);stop();assert.deepEqual(tweens.slice(1),[null]);assert.equal(frames.size,0);
  }
  // Another window's rule while a handle is held: followed from where the drag let go.
  h.chooser.cancel();tweens.length=0;h.min.emit('pointerdown');move(h.min,600);
  h.chooser.setRange({min:8,max:16},{animate:true,follow:true});assert.deepEqual(tweens,[]);
  const held={...h.chooser.value};h.min.emit('change');assert.deepEqual(tweens,[held]);assert.equal(h.commits.length,0);
  tick(now+200);assert.equal(tweens.at(-1),null);assert.deepEqual(h.chooser.value,{min:8,max:16});
});

test('keyboard edits commit after 400 ms idle when no keyup arrives',t=>{
  let idle; t.mock.method(globalThis,'setTimeout',(fn,ms)=>{assert.equal(ms,400);idle=fn;return 1;});
  t.mock.method(globalThis,'clearTimeout',()=>{});
  const h=setup(t);h.min.emit('keydown',{key:'ArrowRight'});move(h.min,500);
  assert.equal(h.commits.length,0);idle();assert.equal(h.commits.length,1);
  h.min.emit('keyup',{key:'ArrowRight'});assert.equal(h.commits.length,1);
});
test('a drag updates locally, and release commits exactly once',t=>{
  const h=setup(t);h.min.emit('pointerdown');move(h.min,500);move(h.min,600);
  assert.equal(h.live.length,2);assert.equal(h.commits.length,0);
  h.min.emit('change');h.min.emit('pointerup');h.min.emit('lostpointercapture');
  assert.equal(h.commits.length,1);assert.deepEqual(h.commits[0],h.live.at(-1));
});
test('frames coalesce live inputs and no idle animation loop starts',t=>{
  const frames=new Map();let id=0;
  t.mock.method(globalThis,'setTimeout',()=>assert.fail('pointer input needs no timer'));
  const old=globalThis.requestAnimationFrame,oldCancel=globalThis.cancelAnimationFrame;
  globalThis.requestAnimationFrame=fn=>{frames.set(++id,fn);return id;};globalThis.cancelAnimationFrame=n=>frames.delete(n);
  t.after(()=>{globalThis.requestAnimationFrame=old;globalThis.cancelAnimationFrame=oldCancel;});
  const h=setup(t);assert.equal(frames.size,0);h.min.emit('pointerdown');move(h.min,300);move(h.min,400);
  assert.equal(frames.size,1);assert.equal(h.live.length,0);
  const [[n,fn]]=frames;frames.delete(n);fn();assert.equal(h.live.length,1);assert.equal(frames.size,0);
  h.min.emit('change');assert.equal(h.commits.length,1);assert.equal(frames.size,0);
});
for(const event of ['pointercancel','lostpointercapture','blur'])test(`${event} reverts a held thumb without a commit`,t=>{
  const h=setup(t);h.chooser.setRange({min:2,max:64});h.min.emit('pointerdown');move(h.min,600);h.min.emit(event);
  assert.deepEqual(h.chooser.value,{min:2,max:64});assert.equal(h.commits.length,0);assert.equal(h.cancelled.length,1);
});
test('hiding a panel cancels a drag, and whenSettled waits for that cancellation',async t=>{
  const h=setup(t);h.min.emit('pointerdown');move(h.min,500);let done=false;
  const pending=h.chooser.whenSettled().then(()=>done=true);await Promise.resolve();assert.equal(done,false);
  h.chooser.setHidden(true);await pending;assert.equal(h.commits.length,0);assert.equal(done,true);
});
test('another window waits for release and replaces the held value without a stale commit',t=>{
  const h=setup(t);h.min.emit('pointerdown');move(h.min,600);
  h.chooser.setRange({min:1,max:4});assert.notEqual(h.chooser.value.min,1);
  h.min.emit('change');assert.deepEqual(h.chooser.value,{min:1,max:4});assert.equal(h.commits.length,0);
});

test('the newest external range wins even if it restores the value before the drag',t=>{
  const h=setup(t);h.chooser.setRange({min:2,max:64});h.min.emit('pointerdown');move(h.min,600);
  h.chooser.setRange({min:10,max:20});h.chooser.setRange({min:2,max:64});h.min.emit('change');
  assert.deepEqual(h.chooser.value,{min:2,max:64});assert.equal(h.commits.length,0);
});
test('arrow repeats form one change on keyup, including native change events',t=>{
  const h=setup(t);h.min.emit('keydown',{key:'ArrowRight'});
  for(const n of [300,400,500]){move(h.min,n);h.min.emit('change');}
  assert.equal(h.commits.length,0);h.min.emit('keyup',{key:'ArrowRight'});assert.equal(h.commits.length,1);
});
test('empty and zero minimum mean no limit; maximum zero is a real bound',t=>{
  const h=setup(t),{min,max}=h.chooser.numbers;
  h.chooser.setRange({min:1,max:10});min.value='0';min.emit('blur');assert.equal(min.value,'');assert.equal(h.commits[0].min,null);
  max.value='0';max.emit('keydown',{key:'Enter'});assert.equal(h.commits[1].max,0);
  max.value='';max.emit('blur');assert.equal(h.commits[2].max,null);
});

test('a handle stays within the accepted bounds even if the histogram includes larger values',t=>{
  const h=setup(t);h.chooser.setValues([10,1e9]);h.min.emit('pointerdown');move(h.min,999);h.min.emit('change');
  assert.equal(h.commits[0].min,1e6);
});
test('an invalid blur restores the committed value without taking focus; Enter keeps the typed correction',t=>{
  const h=setup(t),min=h.chooser.numbers.min;h.chooser.setRange({min:2,max:4});
  min.focus=min.scrollIntoView=()=>assert.fail('blur moved focus');min.value='9';min.emit('blur');
  assert.equal(min.value,'2');assert.equal(min.getAttribute('aria-invalid'),'true');assert.equal(h.commits.length,0);
  let focused=false;min.focus=()=>focused=true;min.scrollIntoView=()=>{};min.value='9';min.emit('keydown',{key:'Enter'});
  assert.equal(min.value,'9');assert.ok(focused);assert.equal(h.commits.length,0);
});
test('unknown brightness hides handles but keeps number boxes usable',t=>{
  const h=setup(t,{kind:'brightness',integer:false});h.chooser.setValues(new Float32Array([NaN,NaN]));
  assert.ok(h.min.hidden && h.max.hidden);h.chooser.numbers.max.value='12.345';h.chooser.numbers.max.emit('blur');
  assert.deepEqual(h.commits,[{min:null,max:12.3}]);
});
test('range words are plain',()=>{for(const word of Object.values(RANGE_TEXT))assert.deepEqual(bannedWords(word),[]);});

test('histogram canvas uses log-height bars and changes their colour with the range',t=>{
  const h=setup(t),bars=[],g={clearRect(){},fillRect(...rect){bars.push({rect,color:this.fillStyle});}};
  h.chooser.canvas.getContext=()=>g;h.chooser.setRange({min:10,max:50});
  assert.ok(bars.some(b=>b.color==='#637389'));assert.ok(bars.some(b=>b.color==='#39414d'));
  assert.ok(bars.every(b=>b.rect[2]>=0 && b.rect[3]>=0 && b.rect[3]<=55));
  h.chooser.canvas.emit('pointermove',{offsetX:120});assert.deepEqual(bannedWords(h.chooser.canvas.title),[]);
});
