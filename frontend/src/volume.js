import { fitSize } from './robustness.mjs';
import { PROJECT } from './scene3d.js';

export const VOLUME_VS = `#version 300 es
in vec2 aPos;
out vec2 uv;
void main() { uv = aPos; gl_Position = vec4(aPos * 2.0 - 1.0, 0, 1); }
`;

/** GLSL float literal (always with a point). */
const glf = v => (Number.isInteger(v) ? v.toFixed(1) : String(v));
/** Go deeper's focus in the volume (`focus`: stage.js's FOCUS_* values {lit, grey, sourceGrey, dark, sourceDark, face,
 *  white, hit, veil, sourceVeil, veilColor (a GLSL vec3)}): each id's tier, once per change of id along the ray (0
 *  background or no focus, 1 the window's cell, 2 its other lit cells, the rows it names, 3 every other cell); a colour
 *  greyed and darkened toward the veil's colour by its tier, so it loses brightness as well as colour (no tier changes an
 *  alpha, so nothing turns see-through); and what the ray's colour took from tier 1 and tier 2 (hit1, hit2), which the
 *  per-pixel veil reads. With uFocusOn false every tier is 0: no colour, alpha or pixel changes. */
const focusDecl = f => `uniform bool uFocusOn;
uniform uint uFocusCell;
uniform uint uFocusLit[${f.lit}];
uniform float uFocusDim;
float hit1 = 0., hit2 = 0.;
int tierOf(uint id) {
  if (!uFocusOn || id == 0u) return 0;
  if (id == uFocusCell) return 1;
  for (int k = 0; k < ${f.lit}; k++) if (id == uFocusLit[k]) return 2;
  return 3;
}
vec3 tierColor(vec3 c, int tier) {
  if (tier < 2) return c;
  c = mix(c, vec3(dot(c, vec3(0.2126, 0.7152, 0.0722))), (tier == 2 ? ${glf(f.sourceGrey)} : ${glf(f.grey)}) * uFocusDim);
  return mix(c, ${f.veilColor}, (tier == 2 ? ${glf(f.sourceDark)} : ${glf(f.dark)}) * uFocusDim);
}
// What the ray's colour took since a0 (acc.a then; a1 now) goes to that tier's sum; a0 moves on to a1.
void credit(int tier, inout float a0, float a1) {
  if (tier == 1) hit1 += a1 - a0; else if (tier == 2) hit2 += a1 - a0;
  a0 = a1;
}
`;
/** The volume's fragment shader on the shared colour functions `colors`; with `focus` (above) Go deeper's focus
 *  variant, the plain shader plus its lines (each behind uFocusOn): the tiers' colours, a face of the window's cell
 *  `face` x as strong (up to opaque) and `white` of the way to white, and, after the points' composite and the box's
 *  lines, the veil over every pixel whose ray met less than `hit` of the cell (`sourceVeil` where it met that much of a
 *  source, else `veil`). */
export function volumeFragment(colors, focus = null) {
  const f = focus, on = text => (f ? text : '');
  return `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2DArray;
precision highp usampler2D;
in vec2 uv;
out vec4 outColor;
uniform vec2 uCanvas, uZ;
uniform vec3 uGeom;
uniform ivec3 uSize;
uniform usampler2DArray uLayers;
uniform usampler2D uAge;
uniform sampler2D uScene, uDepth, uPoints;
uniform bool uHasAge, uTrails, uCells;
uniform int uMode;
uniform float uIter, uPulse, uOpacity, uFog, uBox;
${PROJECT}
${colors}
${on(f && focusDecl(f))}uint labelAt(ivec3 p) {
  if (any(lessThan(p, ivec3(0))) || any(greaterThanEqual(p, uSize))) return 0u;
  return texelFetch(uLayers, p, 0).r;
}
void add(inout vec4 acc, vec3 c, float a) { acc += (1.-acc.a) * vec4(c*a,a); }
void main() {
  vec3 bg = texture(uScene, uv).rgb;
  vec3 ray = normalize(-uBack + uRight * (uv.x*2.-1.) * (uCanvas.x/uCanvas.y)/uFocal
                            + uUp * (uv.y*2.-1.)/uFocal);
  vec3 eye = uCentre + uBack*uDistance;
  // DDA coordinates are voxel indices; t remains distance in rendered micrometres.
  float height = max(uHeight, .0001);
  vec3 spacing = vec3(uGeom.z, uGeom.z, max(uZ.y*height,.00001));
  vec3 origin = (vec3(eye.x,-eye.y,eye.z) - vec3(uGeom.xy,uZ.x*height))/spacing;
  vec3 dir = vec3(ray.x,-ray.y,ray.z)/spacing;
  // Avoid parallel-axis NaNs without changing the signs of negative directions.
  vec3 safeDir = mix(vec3(1),vec3(-1),lessThan(dir,vec3(0))) * max(abs(dir),vec3(1e-12));
  vec3 a = -origin/safeDir, b = (vec3(uSize)-origin)/safeDir;
  vec3 lo = min(a,b), hi = max(a,b);
  float enter = max(max(lo.x,lo.y),lo.z), leave = min(min(hi.x,hi.y),hi.z);
  float depth = texture(uDepth,uv).r;
  float viewDepth = 2.*uNear*uFar/(uFar+uNear-(depth*2.-1.)*(uFar-uNear));
  float stop = depth < 1. ? viewDepth/max(dot(ray,-uBack),1e-6) : 1e20;
  float t = max(enter,0.), end = min(leave,stop);
  vec4 acc = vec4(0);
  if (end > t && uCells) {
    ivec3 p = clamp(ivec3(floor(origin+dir*(t+1e-4))),ivec3(0),uSize-1);
    ivec3 stepDir = ivec3(sign(safeDir));
    vec3 delta = abs(1./safeDir);
    vec3 next = (vec3(p)+step(vec3(0),safeDir)-origin)/safeDir;
    vec3 normal = lo.x >= lo.y && lo.x >= lo.z ? vec3(-sign(dir.x),0,0)
      : lo.y >= lo.z ? vec3(0,sign(dir.y),0) : vec3(0,0,-sign(dir.z));
    uint previous = enter >= 0. ? 0u : labelAt(p);
${on('    int tierWas = tierOf(previous);   // the tier of previous, looked up once per change of id\n')}    // A line can cross at most nx+ny+nz voxels. Supports grids up to 4096 square.
    for (int i=0; i<16384; ++i) {
      uint id = labelAt(p);
${on(`      int tier = id == previous ? tierWas : tierOf(id);
      // What this step gives the ray goes to its cell's tier (a face leaving a cell for the background: that cell's).
      int gives = id != 0u ? tier : tierWas;
      float given = acc.a;
`)}      float age = uHasAge ? float(texelFetch(uAge,p.xy,0).r) : 0.;
      float since = uIter-age;
      float recent = age>0. && since>=0. ? exp(-since/1.1) : 0.;
      vec3 base = displayColor(id);
${on('      base = tierColor(base, tier);\n')}      float fog = exp(-uFog * max(0.,t-uDistance*.5)/max(uDistance,1.));
      vec3 fogColor = vec3(.043,.055,.075);
      if (id != previous && (id!=0u || previous!=0u)) {
        vec3 face = id!=0u ? base : displayColor(previous);
${on('        if (id==0u) face = tierColor(face, tierWas);\n')}        float light = .38 + .55*max(0.,dot(normal,normalize(vec3(-.4,.6,1))))
          + .22*max(0.,dot(normal,normalize(vec3(.7,-.4,.3))))
          + .14*pow(1.-abs(dot(normal,ray)),3.);
        bool contact = id!=0u && previous!=0u;
        face = mix(face,vec3(1),contact ? .32 : .10)*light;
        if (uTrails) face += vec3(.55,.38,.12)*recent*(.55+.45*uPulse);
        float faceAlpha = (uMode==1 ? (contact ? .65 : .26) : (contact ? .38 : .15)*min(1.,uOpacity))*clusterWeight(id!=0u?id:previous);
${on(`        // A face of the window's cell: stronger (up to opaque) and whiter, so its shape reads first.
        if (gives == 1) { face = mix(face,vec3(1),${glf(f?.white ?? 0)}*uFocusDim); faceAlpha = min(1.,faceAlpha*mix(1.,${glf(f?.face ?? 1)},uFocusDim)); }
`)}        add(acc,mix(fogColor,face,fog),faceAlpha);
      }
      float crossing = min(min(next.x,next.y),next.z);
      float finish = min(end,crossing);
      if (id!=0u && uMode!=1) {
        if (uTrails) base = mix(base,vec3(1,.97,.78),recent*(.55+.45*uPulse)*.65);
        // Normalize by slab thickness: opacity does not depend on voxel subdivision.
        float alpha = 1.-exp(-uOpacity * max(0.,finish-t)/max(spacing.z*float(uSize.z),.001));
        add(acc,mix(fogColor,base*.72,fog),alpha*clusterWeight(id));
      }
      if (finish>=end || acc.a>.995) {
        if (end==leave && leave<=stop && id!=0u)
          add(acc,mix(fogColor,base*.8,fog),(uMode==1 ? .26 : .12*min(1.,uOpacity))*clusterWeight(id!=0u?id:previous));
${on('        credit(gives, given, acc.a);\n')}        break;
      }
${on('      credit(gives, given, acc.a); tierWas = tier;\n')}      previous=id;
      if (next.x<=next.y && next.x<=next.z) { p.x+=stepDir.x; t=next.x; next.x+=delta.x; normal=vec3(-float(stepDir.x),0,0); }
      else if (next.y<=next.z) { p.y+=stepDir.y; t=next.y; next.y+=delta.y; normal=vec3(0,float(stepDir.y),0); }
      else { p.z+=stepDir.z; t=next.z; next.z+=delta.z; normal=vec3(0,0,-float(stepDir.z)); }
      if (any(lessThan(p,ivec3(0))) || any(greaterThanEqual(p,uSize))) {
        if (previous!=0u) add(acc,mix(fogColor,${f ? 'tierColor(displayColor(previous),tier)' : 'displayColor(previous)'}*.8,fog),(uMode==1 ? .26 : .12*min(1.,uOpacity))*clusterWeight(id!=0u?id:previous));
${on('        credit(tier, given, acc.a);\n')}        break;
      }
    }
  }
  vec3 color = acc.rgb+(1.-acc.a)*bg;
  // Screen-space thin box edges at the entry and exit surfaces.
  if (leave>max(enter,0.)) {
    float line=0.;
    for (int j=0;j<2;j++) {
      float hit=j==0 ? enter : leave;
      vec3 q=origin+dir*hit;
      vec3 dist=min(abs(q),abs(vec3(uSize)-q))*spacing;
      float second=min(max(dist.x,dist.y),min(max(dist.x,dist.z),max(dist.y,dist.z)));
      float pixel=max(hit,1.)*2./(uFocal*uCanvas.y);
      if (hit>=0. && hit<=stop+.01) line=max(line,1.-smoothstep(pixel*.4,pixel*1.3,second));
    }
    color=mix(color,vec3(.42,.52,.65),line*uBox);
  }
  // Points are unlit premultiplied emission, separate from the opaque scene.
  // Bound the volume tint rather than multiplying by its transmittance: even
  // points inside a dense cell retain at least 88% of their original colour.
  // This screen-space approximation avoids a second ray march per point.
  vec4 points=texture(uPoints,uv);
  vec3 tint=mix(vec3(1),clamp(acc.rgb/max(acc.a,.0001),0.,1.),.12*acc.a);
  color=color*(1.-points.a)+points.rgb*tint;
${on(`  // Go deeper's focus: every pixel whose ray meets (almost) none of the window's cell goes toward the veil's colour,
  // cells, image and transcripts alike (a source's a little), so nothing behind a dimmed cell shows through it.
  if (uFocusOn && uCells) color = mix(color, ${f?.veilColor}, (hit1 >= ${glf(f?.hit ?? 0)} ? 0. : hit2 >= ${glf(f?.hit ?? 0)} ? ${glf(f?.sourceVeil ?? 0)} : ${glf(f?.veil ?? 0)}) * uFocusDim);
`)}  outColor=vec4(color,1);
}`; }

// Scene, transparent points (sharing scene depth) and resolved volume, one set per size. The 3-D view renders at
// reduced resolution while the camera moves and at full resolution at rest: both sets are kept (at most SETS), so
// starting or ending a motion allocates nothing. Completeness is checked synchronously once per context, for the first
// set (the formats never change); a later allocation that fails loses the context instead (Stage rebuilds it).
const SETS = 2;
export class VolumeTargets {
  constructor(gl, limits) { this.limit=Math.min(limits?.texture || gl.getParameter(gl.MAX_TEXTURE_SIZE), limits?.renderbuffer || gl.getParameter(gl.MAX_RENDERBUFFER_SIZE)); this.gl=gl; this.width=0; this.height=0; this.sets=new Map(); this.checked=false; this.allocations=0; }
  resize(width,height) {
    ({ width, height } = fitSize(width, height, this.limit));
    if (width===this.width && height===this.height) return;
    const key=`${width}x${height}`;
    let set=this.sets.get(key);
    if (set) this.sets.delete(key);   // re-inserted below: the map stays in least recently used order
    else {
      set=this.allocate(width,height);
      for (const [k,old] of this.sets) { if (this.sets.size<SETS) break; this.release(old); this.sets.delete(k); }
    }
    this.sets.set(key,set);
    Object.assign(this,set);
  }
  allocate(width,height) {
    const gl=this.gl, set={width,height,resources:[]};
    this.allocations++;
    const texture=(depth=false)=>{
      const t=gl.createTexture(); set.resources.push(['deleteTexture',t]);
      gl.bindTexture(gl.TEXTURE_2D,t);
      gl.texImage2D(gl.TEXTURE_2D,0,depth?gl.DEPTH_COMPONENT24:gl.RGBA8,width,height,0,depth?gl.DEPTH_COMPONENT:gl.RGBA,depth?gl.UNSIGNED_INT:gl.UNSIGNED_BYTE,null);
      for (const p of [gl.TEXTURE_MIN_FILTER,gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D,p,gl.NEAREST);
      for (const p of [gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D,p,gl.CLAMP_TO_EDGE);
      return t;
    };
    set.color=texture(); set.depth=texture(true); set.points=texture(); set.result=texture();
    const target=(color,depth)=>{
      const f=gl.createFramebuffer(); set.resources.push(['deleteFramebuffer',f]); gl.bindFramebuffer(gl.FRAMEBUFFER,f);
      gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,color,0);
      if(depth) gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,depth,0);
      if(!this.checked && gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE) { this.release(set); gl.bindFramebuffer(gl.FRAMEBUFFER,null); throw new Error('Incomplete volume framebuffer'); }
      return f;
    };
    set.scene=target(set.color,set.depth); set.pointScene=target(set.points,set.depth); set.output=target(set.result);
    this.checked=true;
    gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    return set;
  }
  release(set) { for (const [kind,value] of set.resources) this.gl[kind](value); set.resources=[]; }
}
