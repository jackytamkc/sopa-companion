import { compile, XENIUM_VS, XENIUM_FS } from './stage.js';
import { maskMapping, sampleMasks } from './xenium-helpers.mjs';
export class XeniumMasks {
  constructor(stage,onChange) {
    this.stage=stage;this.onChange=onChange;
    this.settings={cellMasks:{on:false,opacity:.45,outline:false},nucleusMasks:{on:false,opacity:.45,outline:false}};
    stage.drawXenium=()=>this.draw();
  }
  reset(run=null) {
    this.controller?.abort();this.dispose();this.run=run;this.meta=null;this.planes=null;this.loading=false;this.error='';
    for(const o of Object.values(this.settings))o.on=false;
    this.onChange?.();
  }
  dispose(){if(this.gl){for(const t of this.textures||[])this.gl.deleteTexture(t);if(this.program)this.gl.deleteProgram(this.program.program);}this.textures=null;this.program=null;}
  async load() {
    if(!this.run||this.meta||this.loading)return;
    const c=this.controller=new AbortController(),run=this.run;this.loading=true;this.error='';this.onChange?.();
    const get=async path=>{const r=await fetch(`/api/v1/runs/${encodeURIComponent(run)}/${path}`,{signal:c.signal});if(!r.ok)throw Error(`Xenium masks: HTTP ${r.status}`);return r;};
    try {
      const [meta,cells]=await Promise.all([(async()=> (await get('xenium/masks')).json())(),(async()=> (await get('cells')).json())()]);
      const bytes=await (await get('xenium/masks.bin')).arrayBuffer(),n=meta.width*meta.height;
      if(bytes.byteLength!==n*8)throw Error('Truncated Xenium masks');
      if(c.signal.aborted||this.run!==run)return;
      this.meta=meta;this.planes=[new Uint32Array(bytes,n*4,n),new Uint32Array(bytes,0,n)];this.mapping=maskMapping(meta.ids,cells);
      this.stage.dirty=true;
    }catch(e){if(e.name!=='AbortError')this.error=e.message;}
    finally{if(this.controller===c){this.loading=false;this.onChange?.();}}
  }
  pick(x,y) {
    if(!this.meta||this.stage.orbit.active)return [];
    const m=this.meta,ix=Math.floor((x-m.x0)/m.um_per_px),iy=Math.floor((y-m.y0)/m.um_per_px);
    if(ix<0||iy<0||ix>=m.width||iy>=m.height)return [];
    return [...new Set(['cellMasks','nucleusMasks'].flatMap((key,i)=>this.settings[key].on?[m.ids[this.planes[i][iy*m.width+ix]]]:[]).filter(Boolean))];
  }
  draw() {
    const s=this.stage,g=s.gl,m=this.meta;
    if(!m||s.orbit.active||!Object.values(this.settings).some(o=>o.on))return;
    if(this.gl!==g||this.stageProgram!==s.maskProgram||!this.textures){
      this.dispose();this.gl=g;this.stageProgram=s.maskProgram;this.program=compile(g,XENIUM_VS,XENIUM_FS);
      this.sampled=this.planes.map(p=>sampleMasks(p,m.width,m.height,s.limits.texture));
      this.textures=this.sampled.map(p=>{const t=g.createTexture();s.integerTexture(t,p.nx,p.ny,p.data);return t;});
      this.mapWidth=Math.min(s.limits.texture,this.mapping.length);const rows=Math.ceil(this.mapping.length/this.mapWidth),data=new Uint32Array(this.mapWidth*rows);data.set(this.mapping);
      const t=g.createTexture();s.integerTexture(t,this.mapWidth,rows,data);this.textures.push(t);
    }
    const {program,uniforms:u}=this.program;g.useProgram(program);s.clusterGPU.bind(u,s.options,s.options.cellColorBy);
    g.enable(g.BLEND);g.blendFunc(g.SRC_ALPHA,g.ONE_MINUS_SRC_ALPHA);s.orbit.uniforms(g,u);
    g.uniform3f(u.uView,s.camera.x,s.camera.y,s.camera.zoom);g.uniform2f(u.uCanvas,s.canvas.width,s.canvas.height);
    g.uniform1f(u.uPxPerUm,s.camera.zoom);g.uniform1f(u.uEdgePx,s.options.edgePx*s.ratio);
    g.uniform1i(u.uHasAge,0);g.uniform1i(u.uTrails,0);g.uniform1ui(u.uHover,0);g.uniform1ui(u.uSelected,0);
    g.activeTexture(g.TEXTURE4);g.bindTexture(g.TEXTURE_2D,this.textures[2]);g.uniform1i(u.uMapping,4);g.uniform1i(u.uMapWidth,this.mapWidth);
    g.activeTexture(g.TEXTURE1);g.bindTexture(g.TEXTURE_2D,s.textures.age);g.uniform1i(u.uAge,1);
    ['cellMasks','nucleusMasks'].forEach((key,i)=>{
      const o=this.settings[key],p=this.sampled[i];if(!o.on)return;
      g.activeTexture(g.TEXTURE0);g.bindTexture(g.TEXTURE_2D,this.textures[i]);g.uniform1i(u.uLab,0);
      g.uniform4f(u.uRect,m.x0,m.y0,m.width*m.um_per_px,m.height*m.um_per_px);
      g.uniform3f(u.uGeom,m.x0,m.y0,m.um_per_px*p.step);g.uniform2i(u.uSize,p.nx,p.ny);
      g.uniform1i(u.uMode,o.outline?1:0);g.uniform1f(u.uFill,.42);g.uniform1f(u.uOpacity,o.opacity);
      g.drawArrays(g.TRIANGLE_STRIP,0,4);
    });
  }
}
