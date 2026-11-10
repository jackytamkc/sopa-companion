import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import { readFileSync } from 'node:fs';
import { fitSize } from '../src/robustness.mjs';
import { downsample16 } from '../src/image-layers.mjs';
import { domFixture } from './dom-fixture.mjs';

const source=readFileSync(new URL('../src/stage.js',import.meta.url),'utf8');
const context=vm.createContext({fitSize,downsample16,Uint8Array,document:{createElement(){throw Error('no canvas needed');}}});
vm.runInContext('this.ImageStage=class {'+source.slice(source.indexOf('  setImage('),source.indexOf('  setChannels('))+'}',context);

test('tiled image replacement keeps the old textures until the new ones are uploaded, and asks the GPU nothing',()=>{
  const deleted=[],uploads=[],params=[],old={rect:{x0:0,y0:0,w:10,h:10},channels:[{source:{width:1,height:1,data:new Uint16Array(1)}}]},texture={old:true};
  let id=0,fail=false;
  const gl={LINEAR:9729,LINEAR_MIPMAP_LINEAR:9987,TEXTURE_MIN_FILTER:10241,createTexture:()=>({id:++id}),deleteTexture:t=>deleted.push(t),bindTexture(){},pixelStorei(){},
    texImage2D(...args){if(fail)throw new RangeError('out of memory');assert.equal(deleted.includes(texture),false);uploads.push(args);},
    generateMipmap(){throw Error('a view needs no mipmaps');},getError(){throw Error('synchronous getError');},isContextLost(){throw Error('not needed');},
    texParameteri(target,key,value){params.push([key,value]);}};
  const stage=new context.ImageStage();Object.assign(stage,{gl,image:old,textures:{channels:[texture]},limits:{texture:4096},norm16:{R16_EXT:1},orbit:{enabled:false},mipmapped:new WeakSet()});
  const rect={x0:100,y0:200,w:300,h:400},channels=[{max:1000,source:{width:2,height:1,data:new Uint16Array([0,1000])}}];
  fail=true;assert.throws(()=>stage.setImage(rect,channels,true),/previous image remains/);
  assert.equal(stage.image,old);assert.equal(stage.textures.channels[0],texture);assert.equal(deleted.includes(texture),false);
  fail=false;stage.setImage(rect,channels,true);
  assert.equal(stage.image.rect,rect);assert.equal(stage.image.channels,channels);assert.ok(deleted.includes(texture));assert.equal(uploads.length,1);
  assert.ok(params.some(([key,value])=>key===10241&&value===9729),'sampled LINEAR');
  assert.ok(!params.some(([,value])=>value===9987),'no mipmap filter');
});

test('channels not loaded get no texture, and an unchanged channel keeps its texture when another one arrives',()=>{
  let id=0;const deleted=[],uploads=[];
  const gl={createTexture:()=>({id:++id}),deleteTexture:t=>deleted.push(t),bindTexture(){},pixelStorei(){},texImage2D(...args){uploads.push(args);},
    generateMipmap(){},texParameteri(){}};
  const stage=new context.ImageStage();Object.assign(stage,{gl,image:null,textures:{channels:[]},limits:{texture:4096},norm16:{R16_EXT:1},orbit:{enabled:false},mipmapped:new WeakSet()});
  const rect={x0:0,y0:0,w:4,h:2},dapi={width:2,height:1,data:new Uint16Array([1,2])};
  stage.setImage(rect,[{source:dapi,on:true},{source:null,on:false}],true);
  assert.equal(uploads.length,1);assert.equal(stage.textures.channels.length,2);assert.equal(stage.textures.channels[1],null);
  const first=stage.textures.channels[0];
  stage.setImage({...rect},[{source:dapi,on:true},{source:{width:2,height:1,data:new Uint16Array([3,4])},on:true}],true);
  assert.equal(uploads.length,2,'only the new channel is uploaded');assert.equal(stage.textures.channels[0],first);assert.deepEqual(deleted,[]);
  stage.setImage({x0:5,y0:0,w:4,h:2},[{source:dapi,on:true},{source:null,on:false}],true);   // a new view: uploaded again
  assert.equal(uploads.length,3);assert.equal(deleted.length,2);
});

test('u16 image views upload a typed fallback without norm16 while retaining intensity controls',()=>{
  const gl={createTexture:()=>({}),deleteTexture(){},bindTexture(){},pixelStorei(){},
    texImage2D(...args){this.data=args.at(-1);},generateMipmap(){},texParameteri(){}};
  const stage=new context.ImageStage();Object.assign(stage,{gl,textures:{channels:[]},limits:{texture:4096},norm16:null,orbit:{enabled:false},mipmapped:new WeakSet()});
  const channel={max:1000,low:50,high:800,source:{width:3,height:1,data:new Uint16Array([0,500,1000])}};
  stage.setImage({x0:0,y0:0,w:3,h:1},[channel],true);
  assert.deepEqual([...gl.data],[0,128,255]);assert.equal(stage.image.channels[0].high,800);
});

test('tiled status disables 3-D and z slices while layers follow the view, and restores dense controls',()=>{
  const dom=domFixture();dom.document.body.innerHTML=readFileSync(new URL('../index.html',import.meta.url),'utf8');
  dom.$('view').querySelector=()=>dom.$('view').children.find(b=>b.dataset.value==='3d');
  for(const name of ['2-D','z1','z2']){const b=dom.document.createElement('button');b.textContent=name;dom.$('layers').append(b);}
  const stage={orbit:{enabled:true}},switches=[];
  const main=readFileSync(new URL('../src/main.js',import.meta.url),'utf8');
  const c=vm.createContext({$:dom.$,stage,state:{display:{layer:1}},fmt1:new Intl.NumberFormat('en-GB',{maximumFractionDigits:1}),setView:(on,animate)=>{switches.push([on,animate]);stage.orbit.enabled=on;}});
  vm.runInContext(main.slice(main.indexOf('function tiledStatus('),main.indexOf('function phaseLabel(')),c);
  vm.runInContext('tiledStatus({level:"overview",limit:1600,layers:false,text:"Zoom in to see transcripts"})',c);
  const button=dom.$('view').querySelector();assert.equal(button.disabled,true);assert.equal(button.title,'Zoom in to ≤ 1.6 mm to see this part in 3-D');
  assert.deepEqual(switches,[[false,false]]);assert.equal(dom.$('layers').children[1].disabled,true);
  vm.runInContext('tiledStatus({level:"detail",limit:1600,layers:true,text:""})',c);
  assert.equal(button.disabled,false);assert.equal(dom.$('layers').children[2].getAttribute('aria-pressed'),'true');
  vm.runInContext('tiledStatus(null)',c);assert.equal(button.disabled,false);assert.equal(dom.$('tiled-hint').hidden,true);
});
