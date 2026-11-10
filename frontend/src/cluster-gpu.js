import { clusterColor } from './clusters.js';
import { labelLookup } from './analysis-helpers.mjs';
export const CLUSTER_GLSL = `
precision highp isampler2D;
uniform isampler2D uClusters;
uniform sampler2D uClusterStyle;
uniform bool uClusterEnabled, uClusterColor;
uniform int uClusterHover, uClusterCount;
uniform uint uLinkedHover;
int clusterOf(uint label) {
  if (!uClusterEnabled || label==0u || label>=uint(uClusterCount)) return -1;
  return texelFetch(uClusters,ivec2(int(label)%4096,int(label)/4096),0).r;
}
vec4 clusterStyle(uint label) {
  int c=clusterOf(label);
  return c<0 ? vec4(.4,.443,.494,1) : texelFetch(uClusterStyle,ivec2(c%4096,c/4096),0);
}
float clusterWeight(uint label) {
  if(!uClusterEnabled) return 1.;
  return clusterStyle(label).a * (uClusterHover<0 || clusterOf(label)==uClusterHover ? 1. : .12);
}
vec3 displayColor(uint label) {
  vec3 c=uClusterEnabled && uClusterColor ? clusterStyle(label).rgb : cellColor(label);
  return label!=0u && label==uLinkedHover ? mix(c,vec3(1),.65) : c;
}
`;
export class ClusterGPU {
  constructor(gl) { this.gl=gl;this.lookup=gl.createTexture();this.style=gl.createTexture();this.setData(null); }
  upload(texture,internal,format,type,data,rows) {
    const g=this.gl;g.bindTexture(g.TEXTURE_2D,texture);g.pixelStorei(g.UNPACK_ALIGNMENT,1);
    g.texImage2D(g.TEXTURE_2D,0,internal,4096,rows,0,format,type,data);
    g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MIN_FILTER,g.NEAREST);g.texParameteri(g.TEXTURE_2D,g.TEXTURE_MAG_FILTER,g.NEAREST);
  }
  setData(result) {
    this.result=result;const g=this.gl,data=labelLookup(result?.clusters);this.count=(result?.clusters.length||0)+1;
    this.upload(this.lookup,g.R16I,g.RED_INTEGER,g.SHORT,data,data.length/4096);this.setHidden(new Set());
  }
  setHidden(hidden) {
    this.hidden=hidden;
    const g=this.gl, cs=this.result?.summary.clusters||[];
    let max=0;for(const c of cs) max=Math.max(max,c.id);
    const data=new Uint8Array(4096*Math.max(1,Math.ceil((max+1)/4096))*4);
    for(const c of cs) {const rgb=clusterColor(c.color??c.id);for(let k=0;k<3;k++)data[c.id*4+k]=Math.round(rgb[k]*255);data[c.id*4+3]=hidden.has(c.id)?10:255;}
    this.upload(this.style,g.RGBA8,g.RGBA,g.UNSIGNED_BYTE,data,data.length/16384);
  }
  bind(u,options,mode) {
    const g=this.gl;g.activeTexture(g.TEXTURE5);g.bindTexture(g.TEXTURE_2D,this.lookup);g.uniform1i(u.uClusters,5);
    g.activeTexture(g.TEXTURE6);g.bindTexture(g.TEXTURE_2D,this.style);g.uniform1i(u.uClusterStyle,6);
    g.uniform1i(u.uClusterEnabled,!!this.result);g.uniform1i(u.uClusterColor,mode==='cluster');g.uniform1i(u.uClusterCount,this.count);
    g.uniform1i(u.uClusterHover,options.clusterHover??-1);g.uniform1ui(u.uLinkedHover,options.hover||0);
  }
}
