import { pickVoxel } from './analysis-helpers.mjs';
import { ClusterGPU, CLUSTER_GLSL } from './cluster-gpu.js';
import { makeLUT, downsample16, isHexColour } from './image-layers.mjs';
import { fitSize, rasterSupported } from './robustness.mjs';
import { VOLUME_VS, volumeFragment, VolumeTargets } from './volume.js';
import { Orbit, PROJECT } from './scene3d.js';
import { pointVertex, POINT_FS } from './transcripts.js';
import { SuspectGlow } from './glow.mjs';
import { harness } from './harness-hook.mjs';

/**
 * stage.set options (no additional UI):
 * volumeOpacity: 1.4 optical density through the slab; scales with fill / 0.42.
 * transcriptAlpha: 0.7, transcriptSize: 1.8 device px (both views).
 * 3-D transcripts are unlit, gently depth-faded and composited over the volume with mild tinting.
 * volumeMovingScale: 0.6 of drawing-buffer resolution during camera motion.
 * volumeFog: 0.08 distance fog strength; volumeBox: 0.22 box visibility (0 hides).
 * Existing transcripts, stretch, mode (0 Cells/1 Outlines), fill,
 * gain, trails, edgePx, hover and selected retain their API.
 */
// WebGL2 stage: a micrometre camera, the morphology image underlay and the live proseg label raster.
//
// The mask is drawn per screen pixel from the integer label texture, so outlines keep a constant on-screen width at
// any zoom, cell colours are a pure function of the cell id (stable across iterations), and the "last changed at
// iteration" raster drives the change trails without keeping any history on the GPU.
//
// Redraw policy (every presented frame is read back, composited and encoded in software when the app is viewed through
// a remote desktop): the stage draws only when something changed, and one new iteration is one
// frame. Camera motion (pan, zoom, tweens, orbit) runs at the display rate. The change highlight of a new iteration is
// one static frame, held until the next snapshot replaces it, in whole-sample runs and whenever iterations arrive
// faster than every STATIC_BELOW_MS; only an occasional iteration of a small run (a step, the first after a pause)
// fades, in at most FADE_FRAMES frames (flashPlan). Nothing is scheduled while the page is hidden, drawing never reads
// layout (the CSS size is cached by resize()), and no per-frame or per-update path waits on the GPU process: no
// synchronous GL query (getError, getParameter, readPixels, checkFramebufferStatus, ...) runs outside initialisation.
// A failed GPU allocation loses the context; the webglcontextlost/webglcontextrestored handlers rebuild everything.

export const FADE_MS = 450;            // the short fade of an occasional new iteration in a small run: uPulse 1 -> 0
export const FADE_FRAMES = 4;          // ... drawn in at most four frames: 1, 2/3, 1/3, 0
export const STATIC_BELOW_MS = 1500;   // iterations closer than this show their highlight as one static frame
export const LOSS_WINDOW_MS = 60000;   // a context lost again within this long has the image size limit halved

/** How the change highlight of a new iteration is shown, given the time since the previous one arrived (ms) and whether
 *  the run is a whole-sample (tiled) run. A stream (every live whole-sample run; any run, replay or scrub that brings
 *  iterations less than STATIC_BELOW_MS apart) shows it as one static frame: the highlight stays until the next
 *  snapshot replaces it, so a snapshot costs exactly one frame. An occasional iteration of a small run fades in
 *  FADE_FRAMES steps. */
export function flashPlan(interval, tiled = false) {
  return tiled || interval < STATIC_BELOW_MS ? { duration: Infinity, fade: false } : { duration: FADE_MS, fade: true };
}
/** uPulse `elapsed` ms after the iteration arrived: 1 for a static highlight; a fade steps 1, 2/3, 1/3, then 0, so a
 *  redraw for another reason (hover, camera) in between shows the same flash as the step before it. */
export function flashValue(plan, elapsed) {
  if (!plan || !(elapsed < plan.duration)) return 0;
  if (!plan.fade) return 1;
  const steps = FADE_FRAMES - 1;
  return 1 - Math.min(steps, Math.floor(Math.max(0, elapsed) / (plan.duration / steps))) / steps;
}
/** When the flash next changes (ms after it started), after a draw `drawn` ms after it; null when it never will. */
export function flashNext(plan, drawn) {
  if (!plan?.fade || !(drawn < plan.duration)) return null;
  const step = plan.duration / (FADE_FRAMES - 1);
  return Math.min(plan.duration, (Math.floor(Math.max(0, drawn) / step) + 1) * step);
}

// Colour by: Mixing. The mixing check scores every cell on proseg's own count: one byte per cell,
// floor(share x 250) of its RNA that seems to come from neighbours, 255 = not judged (under 100 transcripts). The bytes
// go into a per-label R8 texture in the cluster lookup's 4096-wide layout (texel (label % 4096, label / 4096)), on
// texture units the other programs leave free, and the colour functions shared by the 2-D mask, the Xenium masks, the
// 3-D volume and the transcripts (COLORS) colour a cell by its step on a one-hue ramp: little or none (< 10 %), some
// (10-30 %), a lot (>= 30 %: the cells the headline counts). Not judged is the unmapped grey, drawn in the 2-D view at
// about half opacity in diagonal stripes fixed to the screen (fill and outline), so that colour is not the only cue.
//
// A new result cross-fades from the one on screen (two textures and a mix uniform) only when the previous result came a
// second or more before (a whole-sample run's ~1 per second, paused, a scrub): TINT_FRAMES frames over TINT_MS, on the
// steps the overlay's per-bin layer uses too. Faster results (a small run's ~10 per second) land in the snapshot's own
// frame.
export const MIXING_CODES = Object.freeze({ scale: 250, some: 25, mixed: 75, unjudged: 255 });
export const MIXING_COLORS = Object.freeze({ little: '#6b3f63', some: '#b2477a', mixed: '#f08cbc', unjudged: '#5c6573', accent: '#d55181' });
export const MIX_WIDTH = 4096;     // the cluster lookup's row length
export const MIX_UNITS = Object.freeze([7, 8]);   // texture units: 0-4 are the frame's, the volume's, the points' and the
                                                  // Xenium masks', 5-6 the cluster lookup and style (cluster-gpu.js)
export const TINT_MS = 300;        // a result's cross-fade lasts this long ...
export const TINT_FRAMES = 3;      // ... in this many frames (at 0, 150 and 300 ms: 1/3, 2/3, 1) ...
export const TINT_AFTER_MS = 1000; // ... and happens only this long or more after the previous result

/** The step of a cell's mixing byte: 'little' | 'some' | 'mixed' | 'unjudged' (also for no byte). */
export function mixingStep(code) {
  if (code == null || !(code < MIXING_CODES.unjudged)) return 'unjudged';
  return code >= MIXING_CODES.mixed ? 'mixed' : code >= MIXING_CODES.some ? 'some' : 'little';
}
/** One byte per cell in label - 1 order (as /runs/{id}/qc/{seq}/cells serves them) laid out for the shader: texel
 *  `label` of a MIX_WIDTH-wide texture; the background (label 0) and the padding read "not judged". */
export function packMixing(bytes) {
  if (!bytes) return null;
  const count = bytes.length + 1, rows = Math.max(1, Math.ceil(count / MIX_WIDTH));
  const data = new Uint8Array(rows * MIX_WIDTH).fill(MIXING_CODES.unjudged);
  data.set(bytes, 1);
  return { data, rows, count };
}
/** How the transcripts are coloured: in 2-D as the cells (Layers > Cells > Colour by); in 3-D by their own choice
 *  (Layers > Transcripts > Colour by: 'cell' | 'cluster'), where 'cell' follows the cells' Mixing colours (the panel's
 *  Show on the cells colours the cells in every view, their transcripts included). */
export function pointColourBy(options, threeD) {
  if (!threeD) return options.cellColorBy;
  return options.transcriptColorBy === 'cell' && options.cellColorBy === 'mixing' ? 'mixing' : options.transcriptColorBy;
}
/** The cross-fade of a result `since` ms after the previous one ({duration, frames}), or null when it lands at once:
 *  under TINT_AFTER_MS, for a first result (since Infinity) or with reduced motion. */
export function tintPlan(since, reduced = false) {
  return !reduced && since >= TINT_AFTER_MS && since < Infinity ? { duration: TINT_MS, frames: TINT_FRAMES } : null;
}
/** The new result's weight `elapsed` ms into the cross-fade: 1/3, 2/3, then 1 (a redraw for another reason in between
 *  shows the step before it); 1 without one. */
export function tintValue(plan, elapsed) {
  if (!plan || !(elapsed < plan.duration)) return 1;
  const step = plan.duration / (plan.frames - 1);
  return Math.min(plan.frames, Math.floor(Math.max(0, elapsed) / step) + 1) / plan.frames;
}
/** When the cross-fade next changes (ms after it started), after a draw `drawn` ms after it; null once complete. */
export function tintNext(plan, drawn) {
  if (!plan || !(drawn < plan.duration)) return null;
  const step = plan.duration / (plan.frames - 1);
  return Math.min(plan.duration, (Math.floor(Math.max(0, drawn) / step) + 1) * step);
}
/** Go deeper's outlines (Stage.setFocus): at most this many source neighbours and same-kind neighbours, and a source's
 *  outline strength for its share of the cell's RNA (35 % at 1 %, full from 40 %). */
export const FOCUS_SOURCES = 5, FOCUS_SAME = 10;
export const focusAlpha = share => 0.35 + 0.65 * Math.max(0, Math.min(1, (Number.isFinite(share) ? share : 0) / 0.4));
/** Go deeper's "Show where on the image" (Stage.setSpill): the window's genes as dots
 *  at their transcripts' places, one colour per gene (at most SPILL_COLORS), about SPILL_UM wide on the image and never
 *  under SPILL_MIN_PX nor over SPILL_MAX_PX CSS px, in 2-D; meanwhile every cell but the window's cell and its sources
 *  (at most SPILL_LIT together) is drawn at SPILL_DIM of its strength and the usual transcripts are hidden. */
export const SPILL_UM = 0.6, SPILL_MIN_PX = 4, SPILL_MAX_PX = 14, SPILL_DIM = 0.12, SPILL_LIT = 6, SPILL_COLORS = 9;
/** Go deeper's focus: while its window is open on a cell the cells it involves
 *  (at most FOCUS_LIT: the cell and its shown neighbour rows, the set Show lights) stay lit and everything else
 *  greys: a veil over the image, the Xenium masks and the priors (FOCUS_VEIL of the way to VEIL_COLOR, with a soft edge
 *  FOCUS_SOFT_PX CSS px wide outside the lit cells), every other proseg cell FOCUS_GREY of the way to grey at
 *  FOCUS_CELL_ALPHA of its alpha, and its transcripts at FOCUS_TX_ALPHA. Of the lit cells the window's own is the
 *  clearest (its white outline FOCUS_CELL_BOLD_PX CSS px wider) and the rows it names come second: FOCUS_SOURCE_GREY of
 *  the way to grey at FOCUS_SOURCE_ALPHA of their alpha, their transcripts FOCUS_SOURCE_TX_GREY and FOCUS_SOURCE_TX_ALPHA,
 *  the veil FOCUS_SOURCE_VEIL over them. In 3-D the volume greys the cells by the same tiers and darkens them, its rows
 *  FOCUS_3D_SOURCE_DARK and every other cell FOCUS_3D_DARK of the way to VEIL_COLOR, without making any of them
 *  see-through (a face of the window's cell FOCUS_3D_FACE x as strong, up to opaque, and FOCUS_3D_WHITE of the way to
 *  white), and lays the veil over every pixel whose ray meets less than FOCUS_3D_HIT of that cell (FOCUS_SOURCE_VEIL
 *  where it meets a source); the transcripts grey as in 2-D. Each cell against its own look without the focus (its
 *  colour sets how bright it is): the window's cell keeps it (in 3-D a little brighter, its faces lit), its rows about
 *  half to two thirds of theirs, every other cell a quarter to a half (first values, measured on the stage in software
 *  WebGL). It eases in over FOCUS_IN_MS (on open and on a new cell) and out over FOCUS_OUT_MS (on close), in
 *  FOCUS_FRAMES frames; at once with reduced or calm motion, and in 3-D. */
export const FOCUS_LIT = SPILL_LIT, FOCUS_VEIL = 0.62, FOCUS_SOFT_PX = 2.5, FOCUS_GREY = 0.6, FOCUS_CELL_ALPHA = 0.22, FOCUS_TX_ALPHA = 0.15, FOCUS_TX_GREY = 0.85;
export const FOCUS_SOURCE_GREY = 0.35, FOCUS_SOURCE_ALPHA = 0.6, FOCUS_SOURCE_TX_GREY = 0.35, FOCUS_SOURCE_TX_ALPHA = 0.55, FOCUS_SOURCE_VEIL = 0.4;
export const FOCUS_CELL_BOLD_PX = 1.5, FOCUS_3D_FACE = 2, FOCUS_3D_WHITE = 0.35, FOCUS_3D_SOURCE_DARK = 0.25, FOCUS_3D_DARK = 0.45, FOCUS_3D_HIT = 0.02;
export const FOCUS_IN_MS = 320, FOCUS_OUT_MS = 220, FOCUS_FRAMES = 4, VEIL_COLOR = '#0a0b0e';
/** Go deeper's camera move in 3-D (Stage.fit3d): at most this many passes to fit its box. */
export const FIT_PASSES = 4;
/** Its suspect transcripts glowing (Stage.setGlow, glow.mjs): the window's ordinary dots at this share of
 *  their opacity meanwhile, so the glow leads; and (0.5, so the halos stand out on bright stain) the lit cells' image
 *  FOCUS_INNER of
 *  the way to VEIL_COLOR inside their outlines, which stay bright. The inner veil
 *  eases in and out with the glow as the focus does (focusPlan), at once with reduced or calm motion. */
export const GLOW_DOTS = 0.5, FOCUS_INNER = 0.5;
/** The focus's ease from `from` to `to` (0-1): {from, to, duration, frames}, or null when it lands at once (no change,
 *  reduced or calm motion, 3-D). */
export function focusPlan(from, to, reduced = false) {
  if (reduced || !(Math.abs(to - from) > 1e-3)) return null;
  return { from, to, duration: to > from ? FOCUS_IN_MS : FOCUS_OUT_MS, frames: FOCUS_FRAMES };
}
/** The focus's strength `elapsed` ms into its ease: four smoothed steps (at 0, 1/3, 2/3 and all of its duration), so a
 *  redraw for another reason in between shows the step before it; the end value once it is over. */
export function focusValue(plan, elapsed) {
  if (!(elapsed < plan.duration)) return plan.to;
  const step = plan.duration / (plan.frames - 1), t = Math.min(plan.frames, Math.floor(Math.max(0, elapsed) / step) + 1) / plan.frames;
  return plan.from + (plan.to - plan.from) * t * t * (3 - 2 * t);
}
/** When the ease next changes (ms after it started), after a draw `drawn` ms after it; null once it is over. */
export function focusNext(plan, drawn) { return tintNext(plan, drawn); }
const glslColor = hex => `vec3(${[1, 3, 5].map(i => (parseInt(hex.slice(i, i + 2), 16) / 255).toFixed(4)).join(', ')})`;
// Part of COLORS: it also runs in the transcripts' vertex shader, so nothing here may read gl_FragCoord.
const MIXING_GLSL = `
uniform highp sampler2D uMixFrom, uMixTo;   // per label, the cluster lookup's layout: floor(share x 250), 255 = not judged
uniform bool uMixing;                       // this program colours cells by mixing (Colour by: Mixing)
uniform float uMixT;                        // the cross-fade from uMixFrom (0) to uMixTo (1)
uniform ivec2 uMixCount;                    // labels each of them covers (0: no result yet)
float mixStripes = 0.;                      // how much of the last cell coloured is "not judged" (mixingFill)
int mixingCode(highp sampler2D codes, int count, uint label) {
  if (int(label) >= count) return 256;      // no result for this cell
  return int(texelFetch(codes, ivec2(int(label) % ${MIX_WIDTH}, int(label) / ${MIX_WIDTH}), 0).r * 255. + .5);
}
vec3 mixingStep(int code) {
  return code >= ${MIXING_CODES.unjudged} ? ${glslColor(MIXING_COLORS.unjudged)} : code >= ${MIXING_CODES.mixed} ? ${glslColor(MIXING_COLORS.mixed)}
    : code >= ${MIXING_CODES.some} ? ${glslColor(MIXING_COLORS.some)} : ${glslColor(MIXING_COLORS.little)};
}
vec3 mixingColor(uint label) {
  int a = mixingCode(uMixFrom, uMixCount.x, label), b = mixingCode(uMixTo, uMixCount.y, label);
  mixStripes = mix(a == ${MIXING_CODES.unjudged} ? 1. : 0., b == ${MIXING_CODES.unjudged} ? 1. : 0., uMixT);
  return mix(mixingStep(a), mixingStep(b), uMixT);
}
`;

const QUAD_VS = `#version 300 es
in vec2 aPos;
uniform vec4 uRect;    // world x0, y0, width, height (µm)
uniform vec3 uView;    // camera centre x, y (µm) and device pixels per µm
uniform vec2 uCanvas;  // drawing-buffer size (device px)
uniform float uPlane;
${PROJECT}
out vec2 vWorld;
out vec2 vUv;
void main() {
  vec2 world = uRect.xy + aPos * uRect.zw;
  vWorld = world;
  vUv = aPos;
  vec2 px = (world - uView.xy) * uView.z + 0.5 * uCanvas;
  vec2 clip = px / uCanvas * 2.0 - 1.0;
  gl_Position = u3D ? project(vec3(world, uPlane)) : vec4(clip.x, -clip.y, 0.0, 1.0);
}`;

const IMAGE_FS = `#version 300 es
precision highp float;
uniform sampler2D uChannel, uLut;
uniform vec2 uContrast;
uniform float uGamma, uOpacity, uScale;
uniform bool u3D;
in vec2 vUv;
out vec4 outColor;
void main() {
  float raw = texture(uChannel, vUv).r * uScale;
  float v = pow(clamp((raw-uContrast.x)/max(uContrast.y-uContrast.x,0.0001),0.,1.), 1./uGamma);
  vec3 c = texture(uLut, vec2((v*255.+.5)/256., .5)).rgb;
  outColor = vec4(c * (u3D ? .3 : 1.), uOpacity);
}`;

export const MASK_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uLab;     // label per voxel column (0 = background)
uniform usampler2D uAge;     // iteration at which the column's consensus label last changed (0 = never)
uniform bool uHasAge;
uniform vec3 uGeom;          // raster origin x, y (µm) and voxel size (µm)
uniform ivec2 uSize;         // raster width, height (voxels)
uniform float uPxPerUm;      // device px per µm
uniform float uEdgePx;       // outline width (device px)
uniform int uMode;           // 0 cells, 1 outlines
uniform float uFill;
uniform float uIter;         // iteration of the frame shown
uniform bool uTrails;
uniform float uPulse;        // 1 right after a new frame arrives, decays to 0
uniform uint uHover, uSelected;
// Go deeper's window (qc-deeper.js, Stage.setFocus): its cell, the neighbours its RNA seems to come from (their outline's
// strength) and the neighbours that look like its own kind; uFocusHot (a row hovered or pinned) is drawn uFocusExtra wider.
uniform bool uFocusOn;
uniform uint uFocusCell, uFocusHot, uFocusSrc[5], uFocusSame[10];
uniform float uFocusA[5], uFocusExtra;
// ... and the cells it lights (its cell and its shown neighbour rows): the cell keeps its look, its outline uFocusBold
// wider; the rows grey a little (second); every other cell greys by uFocusDim (0-1, eased); all set only with uFocusOn.
uniform uint uFocusLit[${FOCUS_LIT}];
uniform float uFocusDim, uFocusBold;
// Its "Show where on the image" (Stage.setSpill): the cells that stay lit (its cell and sources); every other cell fades.
uniform bool uSpillOn;
uniform uint uSpillLit[${SPILL_LIT}];
in vec2 vWorld;
out vec4 outColor;

uint labelAt(ivec2 c) {
  if (c.x < 0 || c.y < 0 || c.x >= uSize.x || c.y >= uSize.y) return 0u;
  return texelFetch(uLab, c, 0).r;
}

vec3 hsv(float h, float s, float v) {
  vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return v * mix(vec3(1.0), k, s);
}

vec3 identityColor(uint id) {
  uint h = id * 2654435761u;
  h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
  float hue = fract(float(id) * 0.6180339887 + float(h & 255u) / 1400.0);
  float sat = 0.52 + 0.33 * float((h >> 8) & 255u) / 255.0;
  float val = 0.82 + 0.18 * float((h >> 16) & 255u) / 255.0;
  return hsv(hue, sat, val);
}
${MIXING_GLSL}
// A cell's colour when it is not coloured by cluster: displayColor() (CLUSTER_GLSL) falls back to this, so its mixing
// branch reaches every view that colours cells (2-D, 3-D, transcripts, Xenium masks) with displayColor's own linked
// hover highlight.
vec3 cellColor(uint id) { return uMixing ? mixingColor(id) : identityColor(id); }

${CLUSTER_GLSL}
// Not judged, under Colour by: Mixing (mixStripes, set by the cell's displayColor): diagonal stripes fixed to the
// screen, about half the opacity on average. They thin the fill and the outline alike, so the cue stays in Outlines
// mode (a striped outline reads as dashed). 1 for every other cell and colour mode: nothing else changes.
float mixingFill() {
  if (mixStripes <= 0.) return 1.;
  return mix(1., mod(gl_FragCoord.x + gl_FragCoord.y, 8.) < 4. ? .7 : .25, mixStripes);
}
// Go deeper: 1 = the window's cell (a white outline, also once the selection is cleared), 2 = a neighbour its RNA seems to
// come from (a magenta outline as strong as its share, a), 3 = a neighbour that looks like its own kind (a grey dotted
// outline), 0 = none. The outlines never touch a fill (the Mixing colours use the fills); the focus also greys every
// cell it does not light (focusLit: on col after the trails) and takes the selection's dim back as it does. With the
// window closed (uFocusOn false) nothing below changes a pixel.
int focusOf(uint id, out float a) {
  a = 0.;
  if (!uFocusOn || id == 0u) return 0;
  if (id == uFocusCell) return 1;
  for (int k = 0; k < 5; k++) if (id == uFocusSrc[k]) { a = uFocusA[k]; return 2; }
  for (int k = 0; k < 10; k++) if (id == uFocusSame[k]) return 3;
  return 0;
}
// Go deeper's focus: whether a cell stays as it is while every other one greys (never the background).
bool focusLit(uint id) {
  for (int k = 0; k < ${FOCUS_LIT}; k++) if (id == uFocusLit[k]) return id != 0u;
  return false;
}
// Go deeper's dots on the image: whether a cell stays lit (the window's cell or one of its sources; never the background).
bool spillLit(uint id) {
  for (int k = 0; k < ${SPILL_LIT}; k++) if (id == uSpillLit[k]) return id != 0u;
  return false;
}

void main() {
  vec2 v = (vWorld - uGeom.xy) / uGeom.z;
  ivec2 c = ivec2(floor(v));
  if (c.x < 0 || c.y < 0 || c.x >= uSize.x || c.y >= uSize.y) discard;
  uint id = texelFetch(uLab, c, 0).r;
  vec2 f = fract(v);
  float vpx = uGeom.z * uPxPerUm;
  float w = min(uEdgePx, 0.34 * vpx);
  float focusA = 0.;
  int focus = focusOf(id, focusA);
  if (focus == 2 && id == uFocusHot) w = min(uEdgePx + uFocusExtra, 0.45 * vpx);
  else if (focus == 1) w = min(uEdgePx + uFocusBold, 0.45 * vpx);   // the window's cell: the clearest outline
  vec4 d = vec4(f.x, 1.0 - f.x, f.y, 1.0 - f.y) * vpx;   // device px to the left, right, top, bottom voxel edge

  bool edge = false;
  uint other = 0u;
  if (d.x < w) { uint n = labelAt(c + ivec2(-1, 0)); if (n != id) { edge = true; other = n; } }
  if (d.y < w) { uint n = labelAt(c + ivec2(1, 0));  if (n != id) { edge = true; other = n; } }
  if (d.z < w) { uint n = labelAt(c + ivec2(0, -1)); if (n != id) { edge = true; other = n; } }
  if (d.w < w) { uint n = labelAt(c + ivec2(0, 1));  if (n != id) { edge = true; other = n; } }
  if (!edge && d.x < w && d.z < w && labelAt(c + ivec2(-1, -1)) != id) edge = true;
  if (!edge && d.y < w && d.z < w && labelAt(c + ivec2(1, -1)) != id) edge = true;
  if (!edge && d.x < w && d.w < w && labelAt(c + ivec2(-1, 1)) != id) edge = true;
  if (!edge && d.y < w && d.w < w && labelAt(c + ivec2(1, 1)) != id) edge = true;

  float age = uHasAge ? float(texelFetch(uAge, c, 0).r) : 0.0;
  float since = uIter - age;
  float recent = (age > 0.0 && since >= 0.0) ? exp(-since / 1.1) : 0.0;     // this iteration and the last few

  vec4 col = vec4(0.0);
  if (id != 0u) {
    vec3 base = displayColor(id);
    float fill = uMode == 0 ? uFill : 0.0;
    float edgeAlpha = 0.95;
    bool dimmed = uSelected != 0u && id != uSelected && !uSpillOn;   // Go deeper's dots on: their fading below instead
    float lift = uFocusOn ? uFocusDim : 0.;   // Go deeper's focus greys the other cells itself: the selection's dim gives way
    if (id == uHover) fill = min(1.0, fill + 0.22);
    if (dimmed) { fill *= mix(0.3, 1.0, lift); edgeAlpha *= mix(0.45, 1.0, lift); }
    float stripes = mixingFill();
    fill *= stripes; edgeAlpha *= stripes;
    col = vec4(base, fill);
    if (edge) {
      vec3 edgeColor = other != 0u ? mix(base, vec3(1.0), 0.35) : mix(base, vec3(1.0), 0.15);
      if (id == uSelected) edgeColor = vec3(1.0);
      if (focus == 1) edgeColor = vec3(1.0);
      else if (focus == 2) { edgeColor = vec3(0.8353, 0.3176, 0.5059); edgeAlpha = max(focusA, 0.35); }
      else if (focus == 3 && mod(gl_FragCoord.x + gl_FragCoord.y, 6.) < 3.) { edgeColor = vec3(0.6039, 0.6431, 0.6980); edgeAlpha = 0.9; }
      col = vec4(edgeColor, edgeAlpha);
    }
  }
  if (uTrails && recent > 0.01) {
    float glow = recent * (0.55 + 0.45 * uPulse);
    vec3 hot = vec3(1.0, 0.97, 0.78);
    col = vec4(mix(col.rgb, hot, glow), max(col.a, 0.85 * glow));
  }
  // Go deeper's focus: every cell it does not light (and the trails' glow, so a cell changing next door stays grey) is
  // pulled toward grey and faded, as far as the ease has gone; the lit cells but its own (the rows it names: second),
  // a little of the way, fill and outline alike.
  if (uFocusOn && !focusLit(id)) col = vec4(mix(col.rgb, vec3(dot(col.rgb, vec3(0.2126, 0.7152, 0.0722))), ${FOCUS_GREY} * uFocusDim), col.a * (1.0 - ${(1 - FOCUS_CELL_ALPHA).toFixed(2)} * uFocusDim));
  else if (uFocusOn && id != uFocusCell) col = vec4(mix(col.rgb, vec3(dot(col.rgb, vec3(0.2126, 0.7152, 0.0722))), ${FOCUS_SOURCE_GREY} * uFocusDim), col.a * (1.0 - ${(1 - FOCUS_SOURCE_ALPHA).toFixed(2)} * uFocusDim));
  if (uSpillOn && !spillLit(id)) col.a *= ${SPILL_DIM};   // Go deeper's dots on the image: every other cell nearly fades away
  col.a *= clusterWeight(id);
  if (col.a < 0.003) discard;
  outColor = col;
}`;

// The colour functions shared with the 3-D volume and the transcripts (vertex shader): identity, mixing, cluster (the
// cluster functions come last, with the line that ends them).
export const COLORS = MASK_FS.slice(MASK_FS.indexOf('vec3 hsv'),
  MASK_FS.indexOf(CLUSTER_GLSL) + CLUSTER_GLSL.length + 1);

// Go deeper's dots (Stage.setSpill): a crisp disc per transcript, its rim darkened so it reads over any cell colour, in
// 2-D. aPos: x, y (µm), the gene's colour slot, the dot's opacity.
export const SPILL_VS = `#version 300 es
precision highp float;
in vec4 aPos;
uniform vec2 uCanvas;
uniform vec3 uView;
uniform vec3 uSpillSize;                  // the dot's width on the image (µm), its least and most size (device px)
uniform vec3 uSpillColors[${SPILL_COLORS}];
uniform float uSpillAlpha;                // 1, or GLOW_DOTS while the suspects glow over the dots
out vec4 vColor;
out float vSize;
void main() {
  vec2 clip = (aPos.xy - uView.xy) * uView.z / uCanvas * 2.0;
  gl_Position = vec4(clip.x, -clip.y, 0, 1);
  vSize = clamp(uSpillSize.x * uView.z, uSpillSize.y, uSpillSize.z);
  gl_PointSize = vSize;
  vColor = vec4(uSpillColors[clamp(int(aPos.z + .5), 0, ${SPILL_COLORS - 1})], aPos.w * uSpillAlpha);
}`;
export const SPILL_FS = `#version 300 es
precision highp float;
in vec4 vColor;
in float vSize;
out vec4 outColor;
void main() {
  float r = length(gl_PointCoord * 2.0 - 1.0) * vSize * .5, edge = vSize * .5;   // device px from the centre, to the rim
  float alpha = 1. - smoothstep(edge - 1., edge, r);
  if (alpha < .02) discard;
  float rim = clamp(vSize * .12, .6, 1.4);   // a dark ring about a pixel wide, never most of a small dot
  outColor = vec4(mix(vColor.rgb, vec3(0.043, 0.055, 0.075), .85 * smoothstep(edge - rim - .6, edge - rim + .4, r)), vColor.a * alpha);
}`;

// Go deeper's veil (Stage.setFocus): one quad over the whole drawing buffer, drawn last in 2-D (after the image,
// the Xenium masks, the priors, the proseg mask and the transcripts or Go deeper's dots). Outside the lit cells (the frame's labels, so its edge follows their
// outlines; the background and everything outside the raster included) it lays VEIL_COLOR at uVeil, which darkens and
// greys what is under it; inside the window's cell (uCell) nothing changes, inside its other lit cells (the rows it
// names) it lays uSource, a lighter veil, so they read second. Its edge is soft over uSoftPx device px: the distance
// from the pixel to the nearest lit voxel among the eight around its own (one texel tap each, only where that voxel is
// within reach), so it never looks cut out and costs one tap away from the edges.
// While the suspects glow the same program also lies inside the lit cells, in a pass of its own under the
// dots (uVeil 0, uSource 0, uInner the strength): there it lays VEIL_COLOR at uInner. Inside a lit cell it never covers
// the cells' outlines, which it finds as the mask does (a label change within the outline's width of the voxel's edge,
// the hot source's and the window's cell's wider), so they stay bright; the dots and the glow (its own canvas) are drawn
// over it.
export const VEIL_VS = `#version 300 es
in vec2 aPos;
uniform vec3 uView;    // camera centre x, y (µm) and device pixels per µm
uniform vec2 uCanvas;  // drawing-buffer size (device px)
out vec2 vWorld;
void main() {
  vWorld = uView.xy + (aPos - 0.5) * uCanvas / uView.z;
  gl_Position = vec4(aPos.x * 2.0 - 1.0, 1.0 - aPos.y * 2.0, 0.0, 1.0);
}`;
export const VEIL_FS = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform usampler2D uLab;       // the frame's label per voxel column (0 = background)
uniform vec3 uGeom;            // raster origin x, y (µm) and voxel size (µm)
uniform ivec2 uSize;           // raster width, height (voxels)
uniform float uVoxelPx;        // device px per voxel
uniform float uSoftPx;         // the soft edge outside the lit cells (device px)
uniform uint uLit[${FOCUS_LIT}];
uniform float uVeil;           // how far everything else goes to the veil's colour (0-1; 0: left as it is)
// While the suspects glow: how far the lit cells go to it inside their outlines (0-1; 0: untouched), and their
// outlines' width as the mask draws them (device px; uHot, a hot source, uHotPx; the window's cell uBoldPx wider; 0
// with the cells hidden).
uniform float uInner, uEdgePx, uHotPx, uBoldPx;
uniform uint uHot;
// The window's cell, and how far its other lit cells (the rows it names) go to the veil's colour (0-1; 0 in the inner
// pass).
uniform uint uCell;
uniform float uSource;
in vec2 vWorld;
out vec4 outColor;
uint labelAt(ivec2 c) {
  if (c.x < 0 || c.y < 0 || c.x >= uSize.x || c.y >= uSize.y) return 0u;
  return texelFetch(uLab, c, 0).r;
}
bool litId(uint id) {
  if (id == 0u) return false;
  for (int k = 0; k < ${FOCUS_LIT}; k++) if (id == uLit[k]) return true;
  return false;
}
bool lit(ivec2 c) { return litId(labelAt(c)); }
void main() {
  vec2 v = (vWorld - uGeom.xy) / uGeom.z;
  ivec2 c = ivec2(floor(v));
  vec2 f = v - floor(v);
  uint id = labelAt(c);
  if (litId(id)) {
    float strength = id == uCell ? uInner : max(uInner, uSource);
    if (strength <= 0.) discard;
    // Its outline stays bright: the mask's own edge test (MASK_FS), as wide as the mask draws it.
    float w = id == uHot ? min(uHotPx, 0.45 * uVoxelPx) : id == uCell ? min(uEdgePx + uBoldPx, 0.45 * uVoxelPx) : min(uEdgePx, 0.34 * uVoxelPx);
    vec4 d = vec4(f.x, 1.0 - f.x, f.y, 1.0 - f.y) * uVoxelPx;   // device px to the left, right, top, bottom voxel edge
    if ((d.x < w && labelAt(c + ivec2(-1, 0)) != id) || (d.y < w && labelAt(c + ivec2(1, 0)) != id)
      || (d.z < w && labelAt(c + ivec2(0, -1)) != id) || (d.w < w && labelAt(c + ivec2(0, 1)) != id)
      || (d.x < w && d.z < w && labelAt(c + ivec2(-1, -1)) != id) || (d.y < w && d.z < w && labelAt(c + ivec2(1, -1)) != id)
      || (d.x < w && d.w < w && labelAt(c + ivec2(-1, 1)) != id) || (d.y < w && d.w < w && labelAt(c + ivec2(1, 1)) != id)) discard;
    outColor = vec4(${glslColor(VEIL_COLOR)}, strength);
    return;
  }
  if (uVeil <= 0.) discard;
  float reach = uSoftPx / max(uVoxelPx, 1e-6), near = reach;   // in voxels
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
    float d = length(vec2(dx < 0 ? f.x : dx > 0 ? 1. - f.x : 0., dy < 0 ? f.y : dy > 0 ? 1. - f.y : 0.));
    if ((dx != 0 || dy != 0) && d < near && lit(c + ivec2(dx, dy))) near = d;
  }
  outColor = vec4(${glslColor(VEIL_COLOR)}, uVeil * smoothstep(0., 1., near / reach));
}`;
const hexRgb = hex => (/^#[0-9a-f]{6}$/i.test(hex) ? [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255) : [0.6, 0.64, 0.7]);

/** Colour by: Mixing on the GPU: two R8 textures (`from`, the result on screen before the newest; `to`, the newest), on
 *  MIX_UNITS, and the cross-fade weight `t` between them. Uploads use the first of its own units, so they never
 *  disturb another program's bindings, and ask the GPU nothing (no query). */
export class MixingTextures {
  constructor(gl) {
    this.gl = gl; this.textures = [gl.createTexture(), gl.createTexture()]; this.counts = [0, 0]; this.to = 1; this.t = 1;
    this.upload(0, null); this.upload(1, null);
  }
  /** `packed` (packMixing) into texture `slot`; null: no result (one row, every cell "no result"). */
  upload(slot, packed) {
    const g = this.gl, data = packed?.data || new Uint8Array(MIX_WIDTH).fill(MIXING_CODES.unjudged);
    g.activeTexture(g.TEXTURE0 + MIX_UNITS[0]); g.bindTexture(g.TEXTURE_2D, this.textures[slot]); g.pixelStorei(g.UNPACK_ALIGNMENT, 1);
    g.texImage2D(g.TEXTURE_2D, 0, g.R8, MIX_WIDTH, data.length / MIX_WIDTH, 0, g.RED, g.UNSIGNED_BYTE, data);
    g.texParameteri(g.TEXTURE_2D, g.TEXTURE_MIN_FILTER, g.NEAREST); g.texParameteri(g.TEXTURE_2D, g.TEXTURE_MAG_FILTER, g.NEAREST);
    g.texParameteri(g.TEXTURE_2D, g.TEXTURE_WRAP_S, g.CLAMP_TO_EDGE); g.texParameteri(g.TEXTURE_2D, g.TEXTURE_WRAP_T, g.CLAMP_TO_EDGE);
    this.counts[slot] = packed?.count || 0;
  }
  /** The newest result: `fade` keeps the one on screen as `from` (and starts t at 0); otherwise it replaces it. */
  set(packed, fade = false) {
    if (fade) this.to = 1 - this.to;
    this.upload(this.to, packed);
    this.t = fade ? 0 : 1;
  }
  bind(u, on) {
    const g = this.gl, from = 1 - this.to;
    g.activeTexture(g.TEXTURE0 + MIX_UNITS[0]); g.bindTexture(g.TEXTURE_2D, this.textures[from]); g.uniform1i(u.uMixFrom, MIX_UNITS[0]);
    g.activeTexture(g.TEXTURE0 + MIX_UNITS[1]); g.bindTexture(g.TEXTURE_2D, this.textures[this.to]); g.uniform1i(u.uMixTo, MIX_UNITS[1]);
    g.uniform1i(u.uMixing, on ? 1 : 0); g.uniform1f(u.uMixT, this.t); g.uniform2i(u.uMixCount, this.counts[from], this.counts[this.to]);
  }
}
/** The cluster lookup (cluster-gpu.js) plus the mixing textures: every program built on COLORS binds both through this
 *  one call (Stage's, the Xenium masks'), so its mixing samplers always have their own textures. `mode`: the colour
 *  choice of what is drawn ('identity' | 'cluster' | 'mixing'; transcripts: 'cell' | 'cluster' | 'mixing'). */
export class ColourGPU extends ClusterGPU {
  constructor(gl) { super(gl); this.mixing = new MixingTextures(gl); }
  bind(u, options, mode) { super.bind(u, options, mode); this.mixing.bind(u, mode === 'mixing'); }
}

export function compile(gl, vs, fs) {
  const program = gl.createProgram();
  for (const [type, source] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
    gl.attachShader(program, shader);
    gl.deleteShader(shader);
  }
  gl.bindAttribLocation(program, 0, 'aPos');
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  const uniforms = {};
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < count; i++) {
    const name = gl.getActiveUniform(program, i).name;
    uniforms[name] = gl.getUniformLocation(program, name);
  }
  return { program, uniforms };
}

export class Stage {
  constructor(canvas) {
    this.canvas = canvas;
    this.camera = { x: 0, y: 0, zoom: 1 };   // zoom: device px per µm
    this.orbit = new Orbit(this);
    this.options = { markOffFootprint: true, cells: true, cellColorBy: 'identity', transcriptColorBy: 'cell', volumeOpacity: 1.4, transcriptAlpha: 0.7, transcriptSize: 1.8, volumeMovingScale: 0.6, volumeFog: 0.08, volumeBox: 0.22, transcripts: true, stretch: 1, mode: 0, fill: 0.42, gain: 1, trails: true, edgePx: 1.6, hover: 0, selected: 0 };
    this.image = null;    // {rect, channels: [{source, color, on}]}
    this.frame = null;    // {nx, ny, x0, y0, voxel, iter, labels, age}
    this.flash = null;    // the newest iteration's flash: {at, duration, fade} (flashPlan)
    this.flashDue = null; // ms after flash.at when the flash next needs a redraw; null once settled
    this.tiled = false;   // a whole-sample run: every new iteration is a static highlight frame
    this.drawCount = 0;   // frames drawn (telemetry)
    this.losses = [];     // when the context was lost recently
    this.mixing = { bytes: null, packed: null, at: -Infinity };   // Colour by: Mixing, the newest result (setMixing)
    this.tint = null;     // its cross-fade: {at, duration, frames} (tintPlan)
    this.tintDue = null;  // ms after tint.at when the cross-fade next needs a redraw; null once complete
    this.calmMotion = false;        // the mixing check's "calm motion" switch: results land at once
    this.animators = new Set();     // other canvases animated in the stage's frames (addAnimator)
    this.focus = null;              // Go deeper's focus as set (setFocus); null with its window closed
    this.focusLevel = 0;            // its strength drawn now (0-1, eased)
    this.focusFade = null;          // its ease: {from, to, at, duration, frames} (focusPlan)
    this.focusFadeDue = null;       // ms after focusFade.at when the ease next needs a redraw; null once over
    this.focusOut = null;           // a closed focus, drawn while it eases out
    // Its suspect transcripts glowing (setGlow): their own canvas between this one and the overlay. Made here so its
    // counter reads 0 from the start; it joins the animators only while it has something to show.
    this.glowLayer = new SuspectGlow(this, { dot: () => this.spillDotPx() });
    this.innerLevel = 0;            // the veil inside the lit cells while the suspects glow: its strength drawn now (0-1)
    this.innerFade = null;          // its ease: {from, to, at, duration, frames} (focusPlan)
    this.innerFadeDue = null;       // ms after innerFade.at when the ease next needs a redraw; null once over
    this.startOutline = null;       // Go deeper's starting outline of its cell (setStartOutline), drawn on the overlay
    this.listeners = new Set();
    this.dirty = true;
    this.tween = null;
    // A GPU allocation that fails (out of memory) loses the context instead of being polled with getError. Restoring
    // rebuilds every texture from the data kept here; a context lost again soon after halves the size of the images
    // (the largest uploads), so a loop of losses converges instead of repeating. Halved from the size actually in use
    // (imageSize: a whole-sample view is at most 4096 px, a detail image ~5.6k), not from the GPU's maximum (32768 on
    // large GPUs), so the very next restore uploads smaller images.
    canvas.addEventListener('webglcontextlost', (e) => {
      e.preventDefault(); this.gl = null; this.sleep(); this.glowLayer?.hide(); harness?.count('contextLosses');
      const now = performance.now();
      this.losses = [...this.losses.filter(t => now - t < LOSS_WINDOW_MS), now];
      const inUse = Math.min(this.imageLimit || Infinity, this.limits?.texture || Infinity, this.imageSize || 4096);
      if (this.losses.length > 1) this.imageLimit = Math.max(1024, Math.floor(inUse / 2));
      this.onContextLost?.(this.losses.length);
    });
    canvas.addEventListener('webglcontextrestored', () => {
      try { this.init(); this.onContextRestored?.(null); } catch (error) { this.onContextRestored?.(error); }
    });
    // Nothing animates while the page is hidden, nor while a full-screen view covers it (setCovered: Compare); showing
    // it again draws the current state once.
    this.covered = false;
    document.addEventListener('visibilitychange', () => { if (document.hidden) this.sleep(); else this.dirty = true; });
    this.init();
    new ResizeObserver(entries => this.resize(entries.at(-1))).observe(canvas);
    const watchRatio = () => matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`)
      .addEventListener('change', () => { this.resize(); watchRatio(); }, { once: true });
    watchRatio();
    this.resize();
    this.wake();
  }

  get dirty() { return this._dirty; }
  set dirty(value) { this._dirty = value; if (value) this.wake(); }
  /** Draw on the next animation frame. */
  wake() {
    if (this.raf != null || document.hidden || this.covered) return;
    clearTimeout(this.passiveTimer); this.passiveTimer = null;
    this.raf = requestAnimationFrame(t => { this.raf = null; this.tick(t); this.plan(); });
  }
  sleep() { cancelAnimationFrame(this.raf); this.raf = null; clearTimeout(this.passiveTimer); this.passiveTimer = null; }
  /** A full-screen view over the page (main.js: Compare's dialog) covers the stage (`on`) or no longer does: covered,
   *  nothing is drawn or scheduled, its animators (Go deeper's glow) included, as while the page is hidden; uncovered,
   *  the current state is drawn once. */
  setCovered(on) {
    on = !!on;
    if (on === this.covered) return;
    this.covered = on;
    if (on) this.sleep(); else this.dirty = true;
  }
  /** After a frame: camera motion continues at the display rate; a settling flash, a mixing cross-fade and the
   *  animators' frames wake the stage only when due. */
  plan() {
    if (!this.gl || document.hidden || this.covered) return;
    const now = performance.now();
    if (this.tween || this.orbit.transition || (this.refines() && (this.wasMoving || now < (this.movingUntil || 0)))) { this.wake(); return; }
    if (this.raf != null || this.passiveTimer != null) return;
    const due = this.nextDue(now);
    if (due == null) return;
    this.passiveTimer = setTimeout(() => { this.passiveTimer = null; this.wake(); }, Math.max(0, due - now));
  }
  /** The earliest of the flash's next step, the cross-fade's next step, the focus's and its inner veil's next steps and
   *  the animators' next frames (ms), or null. */
  nextDue(now) {
    let due = this.flashDue != null && this.flashing() ? this.flash.at + this.flashDue : Infinity;
    if (this.tintDue != null && this.tinting()) due = Math.min(due, this.tint.at + this.tintDue);
    if (this.focusFadeDue != null && this.focusFade) due = Math.min(due, this.focusFade.at + this.focusFadeDue);
    if (this.innerFadeDue != null && this.innerFade) due = Math.min(due, this.innerFade.at + this.innerFadeDue);
    for (const animator of this.animators || []) { const t = animator.due(now); if (t != null) due = Math.min(due, t); }
    return due < Infinity ? due : null;
  }
  /** Something else that animates with the stage (the mixing check's per-bin layer and marks on the 2-D overlay, Go
   *  deeper's arrows): {due(now): when its next frame is due (performance.now() ms) or null, frame(now): draw it,
   *  returning false when that frame was not motion}. Its frames are drawn in the stage's animation frames (the stage
   *  redraws only when it has a reason of its own), so both canvases change together and nothing is scheduled while
   *  every animator is idle. Every tick asks the animators, so a stage.set() that changes what one draws reaches it in
   *  that frame. Call replan() when an animator's schedule changes. */
  addAnimator(animator) { this.animators ??= new Set(); this.animators.add(animator); this.replan(); }
  removeAnimator(animator) { this.animators?.delete(animator); }
  /** Plan the next wake-up again (an animator's schedule changed); draws nothing by itself. */
  replan() {
    if (!this.gl || document.hidden || this.covered || this.raf != null) return;
    clearTimeout(this.passiveTimer); this.passiveTimer = null; this.plan();
  }
  /** The flash is visible only with trails on a frame that has its change ages. */
  flashing() { const o = this.options; return !!(o.trails && this.frame?.age && (o.cells || this.orbit.active)); }
  /** Mixing colours are in use (cells, Xenium masks or transcripts). */
  mixingShown() { const o = this.options; return o.cellColorBy === 'mixing' || o.transcriptColorBy === 'mixing'; }
  /** A cross-fade is on screen. */
  tinting() { return !!this.tint && this.mixingShown(); }
  reducedMotion() {
    try { return !!(this.motionQuery ??= globalThis.matchMedia?.('(prefers-reduced-motion: reduce)'))?.matches; } catch { return false; }
  }
  /** The 3-D volume renders at reduced resolution while the camera moves, then once more at full resolution. */
  refines() { return this.orbit.active && !!this.frame?.layers?.length; }

  init() {
    // No preserveDrawingBuffer: nothing reads the canvas back, and every draw repaints it completely.
    const gl = this.canvas.getContext('webgl2', { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    // Synchronous queries happen here, once per context, and never while drawing or uploading.
    this.norm16 = gl.getExtension('EXT_texture_norm16');
    this.luts = new Map();
    this.mipmapped = new WeakSet();   // image textures that have their mipmap chain
    this.limits = { texture: gl.getParameter(gl.MAX_TEXTURE_SIZE), layers: gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS), renderbuffer: gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) };
    const points = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);   // Go deeper's dots stay under the largest point size
    this.limits.point = Number(points?.[1]) > 1 ? Number(points[1]) : 64;
    try {
      const debug = gl.getExtension('WEBGL_debug_renderer_info');
      this.renderer = String(gl.getParameter(debug ? debug.UNMASKED_RENDERER_WEBGL : gl.RENDERER) || '');
    } catch { this.renderer = ''; }
    const hiddenClusters=this.clusterGPU?.hidden;
    this.clusterGPU = new ColourGPU(gl);
    this.clusterGPU.setData(this.analysisResult);
    if(hiddenClusters)this.clusterGPU.setHidden(hiddenClusters);
    if (this.mixing?.packed) this.clusterGPU.mixing.set(this.mixing.packed);   // a restored context: the newest, at once
    this.tint = null; this.tintDue = null;
    this.imageProgram = compile(gl, QUAD_VS, IMAGE_FS);
    this.maskProgram = compile(gl, QUAD_VS, MASK_FS);
    // The volume greys and darkens the cells by Go deeper's focus (its tiers; with no focus every pixel as without it).
    this.volumeProgram = compile(gl, VOLUME_VS, volumeFragment(COLORS, { lit: FOCUS_LIT, grey: FOCUS_GREY, sourceGrey: FOCUS_SOURCE_GREY,
      dark: FOCUS_3D_DARK, sourceDark: FOCUS_3D_SOURCE_DARK, face: FOCUS_3D_FACE, white: FOCUS_3D_WHITE, hit: FOCUS_3D_HIT, veil: FOCUS_VEIL,
      sourceVeil: FOCUS_SOURCE_VEIL, veilColor: glslColor(VEIL_COLOR) }));
    this.volumeTargets = new VolumeTargets(gl, this.limits);
    this.pointProgram = compile(gl, pointVertex(COLORS), POINT_FS);
    // Go deeper's focus: the same transcripts with the cells it does not light faded (its rows' a little), a program of
    // their own so that with the window closed the transcripts' program and its uniforms are exactly those of before.
    this.pointFocusProgram = compile(gl, pointVertex(COLORS, { lit: FOCUS_LIT, alpha: FOCUS_TX_ALPHA, grey: FOCUS_TX_GREY,
      sourceAlpha: FOCUS_SOURCE_TX_ALPHA, sourceGrey: FOCUS_SOURCE_TX_GREY }), POINT_FS);
    this.veilProgram = compile(gl, VEIL_VS, VEIL_FS);
    this.pointVao = gl.createVertexArray();
    this.pointBuffer = gl.createBuffer();
    this.spillProgram = compile(gl, SPILL_VS, SPILL_FS);
    this.spillVao = gl.createVertexArray();
    this.spillBuffer = gl.createBuffer();
    this.layerTexture = gl.createTexture();
    this.layerShape = '';
    if (this.transcripts) this.setTranscripts(this.transcripts);
    const vao = this.quadVao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 0, 1, 1, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.textures = { lab: gl.createTexture(), age: gl.createTexture(), consensus: gl.createTexture(), blank: gl.createTexture(), channels: [] };
    for (const t of [this.textures.lab, this.textures.age]) this.integerTexture(t, 1, 1, new Uint32Array(1));
    gl.bindTexture(gl.TEXTURE_2D, this.textures.blank);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array(1));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    // A restored context: upload what was on screen (the snapshot even when the image cannot be prepared).
    let failure = null;
    if (this.image) try { this.setImage(this.image.rect, this.image.channels, false, this.image.mipmaps); } catch (error) { failure = error; }
    if (this.frame) this.setFrame(this.frame, false);
    if (this.spill) this.uploadSpill();   // Go deeper's dots again
    this.dirty = true;
    if (failure) throw failure;
  }

  integerTexture(texture, width, height, data) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32UI, width, height, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // ---- content -------------------------------------------------------------------------------------------
  /** channels: [{source: HTMLImageElement | {data: Uint16Array, width, height} | null, on, ...}] covering rect {x0, y0, w,
   *  h} µm. A channel without a source (a whole-sample view loads only the visible channels) gets no texture.
   *  keepPrevious: the textures on screen stay until the replacement is uploaded; a channel whose source object is
   *  unchanged over the same rect keeps its texture (turning a channel on uploads only that channel).
   *  mipmaps: the whole-run image of a detail run is zoomed far out; view-sized images (replaced on every pan, never
   *  shown much smaller than they were fetched in 2-D) are sampled LINEAR with no mipmap chain to build, except in 3-D,
   *  where the tilted ground plane shows them up to ~4x smaller (mipmapImage). No GL query here: a failed allocation
   *  loses the context and webglcontextrestored uploads this.image again. */
  setImage(rect, channels, keepPrevious = false, mipmaps = !keepPrevious) {
    const previous = { image: this.image, textures: this.textures?.channels || [] };
    this.image = rect ? { rect, channels, mipmaps } : null;
    const gl = this.gl;
    if (!gl) return;
    const sameRect = keepPrevious && !!rect && !!previous.image && ['x0', 'y0', 'w', 'h'].every(k => previous.image.rect[k] === rect[k]);
    const reusable = new Map();
    if (sameRect) previous.image.channels.forEach((c, k) => { if (c.source && previous.textures[k]) reusable.set(c.source, previous.textures[k]); });
    if (!keepPrevious) { for (const t of previous.textures) if (t) gl.deleteTexture(t); this.textures.channels = []; }
    const textures = [], created = [], chain = mipmaps || this.orbit.enabled;
    let largest = 0;   // the largest texture side uploaded: a second lost context halves the image limit from it
    try {
    for (const channel of this.image ? channels : []) {
      let source = channel.source;
      if (!source) { textures.push(null); continue; }
      const width = source.naturalWidth || source.width, height = source.naturalHeight || source.height;
      const size = fitSize(width, height, Math.min(this.limits.texture, this.imageLimit || Infinity));
      largest = Math.max(largest, size.width, size.height);
      const kept = reusable.get(source);
      if (kept) { reusable.delete(source); if (chain) this.mipmap(kept); textures.push(kept); continue; }
      const t = gl.createTexture();
      created.push(t);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
      if (source.data && this.norm16) {
        const data = downsample16(source.data, width, height, size.width, size.height);
        gl.texImage2D(gl.TEXTURE_2D, 0, this.norm16.R16_EXT, size.width, size.height, 0, gl.RED, gl.UNSIGNED_SHORT, data);
      } else if(source.data) {
        // Browsers without EXT_texture_norm16 still accept tiled u16 images.
        // Keep the channel's original intensity scale and histogram for controls.
        const data=downsample16(source.data,width,height,size.width,size.height);
        const scale=channel.max || 65535;
        gl.texImage2D(gl.TEXTURE_2D,0,gl.R8,size.width,size.height,0,gl.RED,gl.UNSIGNED_BYTE,Uint8Array.from(data,v=>Math.min(255,Math.round(v/scale*255))));
      } else {
        if (size.width !== width || size.height !== height) {
          const canvas = document.createElement('canvas');
          canvas.width = size.width; canvas.height = size.height;
          canvas.getContext('2d').drawImage(source, 0, 0, size.width, size.height);
          source = canvas;
        }
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, gl.RED, gl.UNSIGNED_BYTE, source);
      }
      if (chain) { gl.generateMipmap(gl.TEXTURE_2D); this.mipmapped.add(t); }
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, chain ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      textures.push(t);
    }
    } catch (error) {
      // Only a CPU-side failure lands here (downsampling or converting the pixels): the image on screen stays.
      for (const t of created) gl.deleteTexture(t);
      if (keepPrevious) this.image = previous.image; else this.image = null;
      this.dirty = true;
      throw new Error(keepPrevious ? 'The image view could not be prepared. The previous image remains displayed.'
        : 'The morphology image could not be prepared. The image has been hidden; segmentation remains available.');
    }
    this.textures.channels = textures; this.imageSize = largest;
    if (keepPrevious) for (const t of previous.textures) if (t && !textures.includes(t)) gl.deleteTexture(t);
    this.dirty = true;
  }

  /** A channel texture's mipmap chain, built once: GPU work, no query. */
  mipmap(texture) {
    if (!texture || this.mipmapped.has(texture)) return;
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture); gl.generateMipmap(gl.TEXTURE_2D);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    this.mipmapped.add(texture);
  }
  /** 3-D shows the image as a tilted ground plane, 1.7-3.8x smaller than a whole-sample view image was fetched: every
   *  channel texture on screen gets its mipmap chain (once), so the image neither aliases nor shimmers while orbiting
   *  (a shimmering image is also the worst content for a remote display's encoder). Called on the switch to 3-D;
   *  images uploaded while in 3-D get theirs in setImage. */
  mipmapImage() {
    if (!this.gl || !this.image) return;
    for (const t of this.textures.channels) this.mipmap(t);
    this.dirty = true;
  }

  setChannels(channels) {
    if (this.image) { this.image.channels = channels; this.dirty = true; }
  }

  /** frame: {nx, ny, x0, y0, voxel, iter, labels: Uint32Array(nx*ny), age: Uint32Array|null}. The upload is never
   *  followed by a GL query (getError would wait for the GPU process, which is exactly what stalls a page whose frames
   *  are back-pressured by a remote display): an allocation that fails loses the context, and restoring it uploads
   *  this.frame again. isContextLost() reads a flag kept by the browser, it does not ask the GPU process. */
  setFrame(frame, pulse = true) {
    if (frame && (!this.gl || this.gl.isContextLost())) throw new Error('The graphics context is unavailable. Retrying the snapshot.');
    if (frame && !rasterSupported(frame, this.limits)) {
      this.frame = null;
      this.dirty = true;
      const error = new Error(`Cannot display label raster ${frame.nx} × ${frame.ny} × ${frame.layers?.length || 0}: GPU supports ${this.limits.texture} × ${this.limits.texture} and ${this.limits.layers} layers. Use a smaller region or larger voxels.`);
      error.permanent = true;
      throw error;
    }
    if (frame && this.gl) {
      const old = { lab: this.textures.lab, age: this.textures.age, layers: this.layerTexture, shape: this.layerShape };
      this.textures.lab = this.gl.createTexture(); this.textures.age = this.gl.createTexture();
      this.layerTexture = this.gl.createTexture(); this.layerShape = '';
      try {
      this.integerTexture(this.textures.lab, frame.nx, frame.ny, frame.labels);
      const gl = this.gl, layers = frame.layers || [], shape = `${frame.nx}:${frame.ny}:${layers.length}`;
      if (layers.length) {
        gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.layerTexture);
        if (shape !== this.layerShape) {
          gl.texImage3D(gl.TEXTURE_2D_ARRAY, 0, gl.R32UI, frame.nx, frame.ny, layers.length, 0, gl.RED_INTEGER, gl.UNSIGNED_INT, null);
          gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
          gl.texParameteri(gl.TEXTURE_2D_ARRAY, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
          this.layerShape = shape;
        }
        layers.forEach((data, k) => gl.texSubImage3D(gl.TEXTURE_2D_ARRAY, 0, 0, 0, k, frame.nx, frame.ny, 1, gl.RED_INTEGER, gl.UNSIGNED_INT, data));
      } else {
        // An active sampler must be complete even when the shader takes the
        // consensus branch while a tiled view waits for its 3-D window.
        gl.bindTexture(gl.TEXTURE_2D_ARRAY,this.layerTexture);
        gl.texImage3D(gl.TEXTURE_2D_ARRAY,0,gl.R32UI,1,1,1,0,gl.RED_INTEGER,gl.UNSIGNED_INT,new Uint32Array(1));
        gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MIN_FILTER,gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D_ARRAY,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      }
      if (frame.age) this.integerTexture(this.textures.age, frame.nx, frame.ny, frame.age);
      if (gl.isContextLost()) throw new Error('The graphics context was lost. Retrying; the previous iteration remains displayed.');
      gl.deleteTexture(old.lab); gl.deleteTexture(old.age); gl.deleteTexture(old.layers);
      } catch (error) {
        this.gl.deleteTexture(this.textures.lab); this.gl.deleteTexture(this.textures.age); this.gl.deleteTexture(this.layerTexture);
        this.textures.lab = old.lab; this.textures.age = old.age; this.layerTexture = old.layers; this.layerShape = old.shape;
        throw error;
      }
    }
    // The 2-D view usually shows the consensus itself: share its texture instead of uploading the raster twice.
    this.consensusIsLabels = !!frame && (frame.consensus || frame.labels) === frame.labels;
    if(frame && this.gl && !this.consensusIsLabels) this.integerTexture(this.textures.consensus, frame.nx, frame.ny, frame.consensus);
    this.frame = frame;
    if (!frame) { this.flash = null; this.flashDue = null; }   // a new run starts without a highlight
    else if (pulse) {
      const now = performance.now();
      this.flash = { at: now, ...flashPlan(now - (this.flash?.at ?? -Infinity), this.tiled) };
      this.flashDue = 0;
    }
    this.dirty = true;
  }

  setTranscripts(bytes) {
    this.transcripts = bytes;
    if (this.gl) {
      const gl = this.gl;
      gl.bindVertexArray(this.pointVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, bytes || 0, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 16, 0);
    }
    this.dirty = true;
  }

  drawTranscripts() {
    const f = this.frame, gl = this.gl, layers=f?.layers || [];
    if (!this.options.transcripts || this.transcriptsHidden || !this.transcripts || !this.info?.z || !f) return;
    const focus = this.pointFocusProgram ? this.focusShown() : null;   // Go deeper's focus, in 2-D and 3-D
    const { program, uniforms: u } = focus ? this.pointFocusProgram : this.pointProgram;
    gl.useProgram(program); this.clusterGPU.bind(u,this.options,pointColourBy(this.options,this.orbit.active)); gl.bindVertexArray(this.pointVao);
    if (focus) { gl.uniform1uiv(u['uFocusLit[0]'], focus.lit); gl.uniform1ui(u.uFocusCell, focus.cell); gl.uniform1f(u.uFocusDim, this.focusLevel); }
    gl.enable(gl.BLEND);
    // Store premultiplied colour and coverage in the transparent 3-D point target.
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(u.uCanvas, this.canvas.width, this.canvas.height);
    gl.uniform3f(u.uView, this.camera.x, this.camera.y, this.camera.zoom);
    this.orbit.uniforms(gl, u);
    gl.uniform3f(u.uGeom, f.x0, f.y0, f.voxel);
    gl.uniform3i(u.uSize, f.nx, f.ny, layers.length);
    gl.uniform1i(u.uHasLayers,!!layers.length);
    gl.uniform2f(u.uZ, this.info.z.grid_origin, this.info.z.grid_spacing);
    gl.uniform1i(u.uLayer, f.layer ?? -1);
    gl.uniform1f(u.uTranscriptAlpha, this.options.transcriptAlpha);
    gl.uniform1f(u.uTranscriptSize, this.options.transcriptSize * (this.renderScale || 1));
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D_ARRAY, this.layerTexture);
    gl.uniform1i(u.uLayers, 2);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, this.consensusIsLabels ? this.textures.lab : this.textures.consensus);
    gl.uniform1i(u.uConsensus, 3); gl.uniform1i(u.uMarkOffFootprint, this.options.markOffFootprint);
    gl.drawArrays(gl.POINTS, 0, this.transcripts.byteLength / 16);
    gl.bindVertexArray(this.quadVao);
  }

  setAnalysis(result) { this.analysisResult=result; this.clusterGPU?.setData(result); this.dirty=true; }

  /** Colour by: Mixing. `bytes`: the result to show, one byte per cell in label - 1 order (floor(share x 250), 255 = not
   *  judged; /runs/{id}/qc/{seq}/cells), or null (no result: every cell grey, no stripes). Applied in the next frame
   *  drawn: call it in the task that applies its snapshot and they change together. It cross-fades from the result on
   *  screen (tintPlan) when the previous one came TINT_AFTER_MS or more before, mixing colours are shown, and neither
   *  reduced motion nor the calm switch is on; otherwise it lands at once. `at`: the result's time (performance.now()
   *  ms: its cross-fade's frames start there); `motion` false: it lands at once (a late result
   *  for the paused frame on screen: qc-client.mjs view.still). Returns the cross-fade plan, or null. */
  setMixing(bytes, { at = performance.now(), motion = true } = {}) {
    const m = this.mixing;
    bytes = bytes || null;
    if (bytes === m.bytes) return null;
    const plan = bytes && m.bytes && this.mixingShown() ? tintPlan(at - m.at, !motion || this.calmMotion || this.reducedMotion()) : null;
    m.bytes = bytes; m.packed = packMixing(bytes); m.at = bytes ? at : -Infinity;
    this.tint = plan ? { ...plan, at } : null; this.tintDue = plan ? 0 : null;
    if (this.gl && this.clusterGPU) this.clusterGPU.mixing.set(m.packed, !!plan);
    if (this.mixingShown()) this.dirty = true;
    return plan;
  }
  /** Go deeper (qc-deeper.js): `focus` {cell, sources:
   *  [[label, share]], same: [labels], hot, lit: [labels], dim} outlines the window's cell in white (also once the
   *  selection is cleared), the neighbours its RNA seems to come from in magenta as strong as their share (35-100 %; at
   *  most 5), the neighbours that look like its own kind in dotted grey (at most 10), and `hot` (a row hovered or pinned)
   *  1 px wider; and it lights `lit` (the cell first, each once, at most FOCUS_LIT; by default the cell and its sources)
   *  while everything else greys by `dim` (0-1, by default 1): the veil, the other cells, their transcripts (FOCUS_*);
   *  of the lit cells the window's own is the clearest (its outline bolder) and the others come second. The outlines and
   *  the veil are 2-D; in 3-D the volume and the transcripts take the same tiers (drawVolume, drawTranscripts). The
   *  greying eases in on open and on a new cell (from none), out on close, and between two strengths (focusPlan); at once
   *  with reduced or calm motion or in 3-D. `motion` (the window's own) is not used here. null: none, and once its ease
   *  is over the stage draws exactly as without the window. Redraws only when it changed; returns whether it did. */
  setFocus(focus) {
    const f = focus && focus.cell > 0 ? focus : null;
    const src = new Uint32Array(FOCUS_SOURCES), alpha = new Float32Array(FOCUS_SOURCES), same = new Uint32Array(FOCUS_SAME);
    const lit = new Uint32Array(FOCUS_LIT);
    let dim = 0;
    if (f) {
      (f.sources || []).slice(0, FOCUS_SOURCES).forEach(([label, share], k) => { src[k] = label >>> 0; alpha[k] = focusAlpha(share); });
      (f.same || []).slice(0, FOCUS_SAME).forEach((label, k) => { same[k] = label >>> 0; });
      const shown = Array.isArray(f.lit) ? f.lit : (f.sources || []).map(([label]) => label);
      [...new Set([f.cell, ...shown].map(l => l >>> 0).filter(l => l > 0))].slice(0, FOCUS_LIT).forEach((label, k) => { lit[k] = label; });
      dim = Number.isFinite(f.dim) ? Math.max(0, Math.min(1, f.dim)) : 1;
    }
    const key = f ? `${f.cell}|${src.join(',')}|${[...alpha].map(a => a.toFixed(3)).join(',')}|${same.join(',')}|${f.hot >>> 0}|${lit.join(',')}|${dim.toFixed(3)}` : '';
    if (key === (this.focusKey ?? '')) return false;
    this.focusKey = key;
    const before = this.focusShown();
    this.focus = f ? { cell: f.cell >>> 0, hot: (f.hot || 0) >>> 0, src, alpha, same, lit, dim } : null;
    // The ease: from none for a new cell, else from the strength on screen (a closing focus is drawn until it is over).
    const from = f && before?.cell !== this.focus.cell ? 0 : (this.focusLevel ?? 0);
    const plan = focusPlan(from, dim, this.calmMotion || this.reducedMotion() || !!this.orbit?.active);
    this.focusFade = plan ? { ...plan, at: performance.now() } : null; this.focusFadeDue = plan ? 0 : null;
    this.focusLevel = plan ? focusValue(plan, 0) : dim;
    this.focusOut = !f && plan ? before : null;
    this.dirty = true;
    return true;
  }
  /** The focus drawn now: the one set, or a closed one while it eases out; null when none. */
  focusShown() { return this.focus ?? (this.focusOut && this.focusLevel > 0 ? this.focusOut : null); }
  /** Go deeper's suspect transcripts glowing (glow.mjs SuspectGlow): `glow` {key, points:
   *  Float32Array of [x µm, y µm, p] per transcript (p: how likely it is a neighbour's, 0-1), depth: Float32Array of
   *  each one's depth (the transcripts' normalised z; NaN for none) or null, giver: [x, y] µm or null (the side the glow
   *  ripples in from)} or null. On their own canvas over this one, in 2-D and 3-D (there at their depth, never hidden by
   *  the volume), breathing at most 12 times a second while seen (the page shown, the context alive), drawn once per
   *  change with reduced or calm motion; the window's dots step back to GLOW_DOTS meanwhile, and while the focus is on
   *  the lit cells dim inside their outlines (FOCUS_INNER, 2-D), easing in and out with the glow. The same key again: no
   *  work. Returns whether it changed. */
  setGlow(glow) {
    const layer = this.glowLayer ??= new SuspectGlow(this, { dot: () => this.spillDotPx() });
    const was = layer.count > 0, changed = layer.set(glow ?? null), has = layer.count > 0;
    if (changed && was !== has) {
      this.easeInner(has ? 1 : 0);
      if (this.spill || this.focusShown()) this.dirty = true;   // the dots' opacity, the lit cells' inner veil
    }
    return changed;
  }
  /** The inner veil's ease to `to` (0-1) from the strength drawn now, as the focus's (focusPlan): at once with reduced
   *  or calm motion, in 3-D, or with no focus drawn (nothing to ease on screen). */
  easeInner(to) {
    const from = this.innerLevel ?? 0, plan = focusPlan(from, to, this.calmMotion || this.reducedMotion() || !!this.orbit?.active || !this.focusShown());
    this.innerFade = plan ? { ...plan, at: performance.now() } : null; this.innerFadeDue = plan ? 0 : null;
    this.innerLevel = plan ? focusValue(plan, 0) : to;
  }
  /** Go deeper's starting outline of its cell while its Show is on: `outline`
   *  {points: Float32Array of the polygon's [x, y] µm pairs, closed (true unless false)} or null. It is drawn dashed
   *  white on the overlay, over the stage and its glow, by qc-overlay.mjs (QcOverlay reads stage.startOutline in the
   *  refresh frame this asks for, in 2-D); the stage's own drawing never changes for it. main.js takes the outline from
   *  prior-layers.js (PriorLayers.startOutline: the prior's polygon of the cell's name, else its starting labels traced).
   *  At least two points, all finite, else none. The same points again: no work. Returns whether it changed. */
  setStartOutline(outline) {
    const p = outline?.points, ok = p instanceof Float32Array && p.length >= 4 && p.every(Number.isFinite);
    const points = ok ? p.subarray(0, p.length & ~1) : null, closed = outline?.closed !== false, was = this.startOutline;
    if (!points ? !was : was && was.closed === closed && was.points.length === points.length && points.every((v, i) => v === was.points[i])) return false;
    this.startOutline = points ? { points: points.slice(), closed, count: points.length / 2 } : null;
    this.replan();   // the overlay's animator draws it (a refresh frame, not motion)
    return true;
  }
  /** The glow as shown, for the harness's snapshot: {on (set and on screen), count (its transcripts), fps (its repaints
   *  a second while it breathes, 0 when static or off)}. */
  get glow() { return this.glowLayer ? this.glowLayer.state() : { on: false, count: 0, fps: 0 }; }
  /** Go deeper's dots' width now (CSS px): SPILL_UM on the image, between SPILL_MIN_PX and SPILL_MAX_PX (nor over the
   *  GPU's largest point), as drawSpill draws them. */
  spillDotPx() {
    const r = this.ratio, low = SPILL_MIN_PX * r, high = Math.max(low, Math.min(SPILL_MAX_PX * r, this.limits?.point || 64));
    return Math.max(low, Math.min(high, SPILL_UM * this.camera.zoom)) / r;
  }
  /** Go deeper's "Show where on the image" (qc-deeper.js; SPILL_*): `spill` {key (its answer), cell, lit: [labels] (the
   *  cell's sources), points: Float32Array of x, y (µm), colour slot, opacity per dot (in draw order), colors: ['#rrggbb',
   *  ...] per slot} or null. The cell and `lit` keep their look, every other cell fades, the usual transcripts are hidden
   *  and the dots drawn over the cells; 2-D only. null: none, and the stage draws exactly as without it. The dots are
   *  uploaded once per change (and again on a restored context); redraws only when it changed; returns whether it did. */
  setSpill(spill) {
    const s = spill && spill.cell > 0 && spill.points instanceof Float32Array ? spill : null;
    const lit = new Uint32Array(SPILL_LIT), colors = new Float32Array(3 * SPILL_COLORS);
    if (s) {
      [...new Set([s.cell, ...(s.lit || [])].map(l => l >>> 0).filter(l => l > 0))].slice(0, SPILL_LIT).forEach((label, k) => { lit[k] = label; });
      (s.colors || []).slice(0, SPILL_COLORS).forEach((hex, k) => colors.set(hexRgb(hex), 3 * k));
    }
    const key = s ? `${s.key ?? ''}|${lit.join(',')}|${s.points.length}|${(s.colors || []).join(',')}` : '';
    if (key === (this.spillKey ?? '')) return false;
    this.spillKey = key || null;   // off again: as it was before the first press (the harness compares the state)
    this.spill = s ? { cell: s.cell >>> 0, lit, colors, points: s.points, count: Math.floor(s.points.length / 4) } : null;
    if (this.spill && this.gl) this.uploadSpill();
    this.dirty = true;
    return true;
  }
  uploadSpill() {
    const gl = this.gl;
    gl.bindVertexArray(this.spillVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.spillBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.spill.points, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 16, 0);
    gl.bindVertexArray(this.quadVao ?? null);
  }
  /** The dots (setSpill), over the cells in 2-D; never under SPILL_MIN_PX nor over SPILL_MAX_PX CSS px (nor the GPU's
   *  largest point). */
  drawSpill() {
    const s = this.spill, gl = this.gl;
    if (!s?.count || !this.spillProgram) return;
    const { program, uniforms: u } = this.spillProgram, r = this.ratio;
    gl.useProgram(program); gl.bindVertexArray(this.spillVao);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform2f(u.uCanvas, this.canvas.width, this.canvas.height);
    gl.uniform3f(u.uView, this.camera.x, this.camera.y, this.camera.zoom);
    gl.uniform3f(u.uSpillSize, SPILL_UM, SPILL_MIN_PX * r, Math.max(SPILL_MIN_PX * r, Math.min(SPILL_MAX_PX * r, this.limits?.point || 64)));
    gl.uniform3fv(u['uSpillColors[0]'], s.colors);
    gl.uniform1f(u.uSpillAlpha, this.glowLayer?.count > 0 ? GLOW_DOTS : 1);   // the suspects glow over them: they step back
    gl.drawArrays(gl.POINTS, 0, s.count);
    gl.bindVertexArray(this.quadVao);
  }
  /** Go deeper's veil (setFocus): everything drawn so far outside the lit cells goes FOCUS_VEIL x the focus's strength of
   *  the way to VEIL_COLOR, with a soft edge FOCUS_SOFT_PX CSS px wide, and the lit cells but the window's own
   *  FOCUS_SOURCE_VEIL x that strength inside their outlines; 2-D, drawn last. With `inner` (0-1) the other pass instead
   *  (while the suspects glow; drawn under the dots): the lit cells go that far inside their outlines, which stay as the
   *  mask drew them (its outline width; none with the cells hidden), and nothing else changes. */
  drawVeil(focus, inner = 0) {
    const f = this.frame, gl = this.gl, { program, uniforms: u } = this.veilProgram, r = this.ratio;
    gl.useProgram(program);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.uniform3f(u.uView, this.camera.x, this.camera.y, this.camera.zoom);
    gl.uniform2f(u.uCanvas, this.canvas.width, this.canvas.height);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.textures.lab); gl.uniform1i(u.uLab, 0);
    gl.uniform3f(u.uGeom, f.x0, f.y0, f.voxel); gl.uniform2i(u.uSize, f.nx, f.ny);
    gl.uniform1f(u.uVoxelPx, f.voxel * this.camera.zoom); gl.uniform1f(u.uSoftPx, FOCUS_SOFT_PX * r);
    gl.uniform1uiv(u['uLit[0]'], focus.lit); gl.uniform1f(u.uVeil, inner > 0 ? 0 : FOCUS_VEIL * this.focusLevel);
    const edge = this.options.cells ? this.options.edgePx * r : 0;   // the mask's uEdgePx; its hot source 1 CSS px wider
    gl.uniform1f(u.uInner, inner); gl.uniform1f(u.uEdgePx, edge); gl.uniform1f(u.uHotPx, edge > 0 ? edge + r : 0);
    gl.uniform1f(u.uBoldPx, edge > 0 ? FOCUS_CELL_BOLD_PX * r : 0);   // the window's cell's outline, as the mask draws it
    gl.uniform1ui(u.uHot, focus.hot && focus.src.includes(focus.hot) ? focus.hot : 0);
    gl.uniform1ui(u.uCell, focus.cell); gl.uniform1f(u.uSource, inner > 0 ? 0 : FOCUS_SOURCE_VEIL * this.focusLevel);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }
  /** The mixing byte shown for `label` (the newest result's), or null (background, no result). */
  mixingCode(label) { const p = this.mixing?.packed; return p && label > 0 && label < p.count ? p.data[label] : null; }

  set(options) {
    // Options are primitives: an unchanged value needs no redraw (hover and selection are set very often).
    if (!Object.keys(options).some(key => this.options[key] !== options[key])) return;
    Object.assign(this.options, options);
    this.dirty = true;
  }

  // ---- camera --------------------------------------------------------------------------------------------
  /** Cache the canvas's CSS size and pixel ratio: called by the ResizeObserver (with its entry), on device-pixel-ratio
   *  changes and by layout changes. Nothing else reads layout, so drawing never forces a synchronous layout. */
  resize(entry) {
    const box = entry?.contentRect || this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(box.width * dpr)), height = Math.max(1, Math.round(box.height * dpr));
    this.cssWidth = box.width; this.cssHeight = box.height; this.pixelRatio = width / Math.max(1, box.width);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
      this.dirty = true;
      this.emit();
    }
  }

  /** Device px per CSS px (cached by resize). */
  get ratio() { return this.pixelRatio || 1; }

  /** CSS pixels <-> µm */
  toWorld(x, y) {
    const r = this.ratio, c = this.camera;
    return [(x * r - this.canvas.width / 2) / c.zoom + c.x, (y * r - this.canvas.height / 2) / c.zoom + c.y];
  }
  toScreen(x, y) {
    const r = this.ratio, c = this.camera;
    return [((x - c.x) * c.zoom + this.canvas.width / 2) / r, ((y - c.y) * c.zoom + this.canvas.height / 2) / r];
  }
  /** CSS px per µm */
  get scale() { return this.camera.zoom / this.ratio; }

  /** Fit the world rect `rect` {x0, y0, w, h} (µm) to the stage, `margin` (a share of it) around; in 3-D the view
   *  first goes back to the orbit's own angle. With `view` {box: [x0, y0, x1, y1] µm, free: [x0, y0, x1, y1] CSS px} in
   *  3-D (Go deeper's camera move: main.js deeperFly) the angle, the turn and the slab stay as the user left them, and
   *  the camera comes to fit3d's: `box` in the middle of `free`, all of it inside. A 420 ms tween; at once without
   *  `animate`. */
  fit(rect, animate = true, margin = 0.06, view = null) {
    if (!rect) return;
    const keep = this.orbit.enabled && Array.isArray(view?.box) && Array.isArray(view?.free);
    if (this.orbit.enabled && !keep) this.orbit.switch(true, animate);
    const w = this.canvas.width, h = this.canvas.height;
    const zoom = Math.min(w / (rect.w * (1 + 2 * margin)), h / (rect.h * (1 + 2 * margin)));
    const target = keep ? this.fit3d(view.box, view.free, zoom) : { x: rect.x0 + rect.w / 2, y: rect.y0 + rect.h / 2, zoom };
    if (!animate) { this.camera = target; this.tween = null; this.dirty = true; this.emit(); return; }
    this.dirty = true;
    this.tween = { from: { ...this.camera }, to: target, start: performance.now(), duration: 420 };
  }
  /** The camera {x, y, zoom} that shows `box` (µm, on the image and up to the slab's top) in the middle of `free` (CSS px)
   *  as the orbit looks now, every corner of it inside: from `zoom` (the 2-D fit's), at most FIT_PASSES passes, each
   *  turning the camera so the image's point under free's middle is the box's middle, projecting the box's 8 corners
   *  (Orbit.project) and zooming out until they all fit (a corner behind the eye halves the zoom); the last pass's
   *  camera, never closer than the 2-D fit. */
  fit3d(box, free, zoom) {
    const o = this.orbit, B = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], F = [(free[0] + free[2]) / 2, (free[1] + free[3]) / 2];
    const half = [(free[2] - free[0]) / 2, (free[3] - free[1]) / 2];
    let camera = null;
    for (let pass = 0; pass < FIT_PASSES; pass++) {
      const g = o.ground(F[0], F[1], { x: B[0], y: B[1], zoom });
      camera = { x: 2 * B[0] - g[0], y: 2 * B[1] - g[1], zoom };
      const corners = [];
      for (const x of [box[0], box[2]]) for (const y of [box[1], box[3]]) for (const z of [0, 1]) corners.push(o.project(x, y, z, camera));
      if (corners.some(p => !p)) { zoom *= 0.5; continue; }
      const s = Math.min(...corners.map(p => Math.min(half[0] / Math.max(Math.abs(p.x - F[0]), 1e-9), half[1] / Math.max(Math.abs(p.y - F[1]), 1e-9))));
      if (s >= 1) break;
      zoom *= Math.max(0.25, 0.98 * s);
    }
    return camera;
  }

  zoomAt(cssX, cssY, factor) {
    const world = (x, y) => this.orbit.active ? this.orbit.ground(x, y) : this.toWorld(x, y);
    const [wx, wy] = world(cssX, cssY);
    const c = this.camera;
    c.zoom = Math.min(400, Math.max(0.002, c.zoom * factor));
    const [nx, ny] = world(cssX, cssY);
    c.x += wx - nx;
    c.y += wy - ny;
    this.tween = null;
    this.dirty = true;
    this.emit();
  }

  panBy(dxCss, dyCss) {
    const r = this.ratio;
    this.camera.x -= dxCss * r / this.camera.zoom;
    this.camera.y -= dyCss * r / this.camera.zoom;
    this.tween = null;
    this.dirty = true;
    this.emit();
  }

  onChange(listener) { this.listeners.add(listener); }
  emit() { this.movingUntil = performance.now() + 140; this.wake(); for (const l of this.listeners) l(); }

  /** The label under a world point in the shown frame (0 = none). */
  labelAt(x, y) {
    const f = this.frame;
    if (!f) return 0;
    const i = Math.floor((x - f.x0) / f.voxel), j = Math.floor((y - f.y0) / f.voxel);
    if (i < 0 || j < 0 || i >= f.nx || j >= f.ny) return 0;
    return f.labels[j * f.nx + i];
  }

  pickLabel(x,y) {
    if(!this.orbit.active){const p=this.toWorld(x,y);return this.labelAt(p[0],p[1]);}
    const f=this.frame;if(!f?.layers?.length)return 0;
    const b=this.orbit.basis(),distance=this.canvas.height/(2*Math.tan(Math.PI/8)*this.camera.zoom);
    const sx=(x*this.ratio-this.canvas.width/2)/this.camera.zoom,sy=-(y*this.ratio-this.canvas.height/2)/this.camera.zoom;
    const height=Math.max(.0001,(this.info?.z?.thickness_um||1)*1.5*this.options.stretch*this.orbit.amount);
    const spacing=[f.voxel,f.voxel,(this.info?.z?.grid_spacing||1/f.layers.length)*height];
    const eye=[this.camera.x+b.back[0]*distance,this.camera.y-b.back[1]*distance,b.back[2]*distance];
    const base=[f.x0,f.y0,(this.info?.z?.grid_origin||0)*height];
    const origin=eye.map((v,k)=>(v-base[k])/spacing[k]);
    const direction=b.back.map((v,k)=>(-v*distance+b.right[k]*sx+b.up[k]*sy)*(k===1?-1:1)/spacing[k]);
    return pickVoxel(origin,direction,[f.nx,f.ny,f.layers.length],f.layers,this.clusterGPU.hidden,this.analysisResult?.clusters);
  }

  // ---- drawing -------------------------------------------------------------------------------------------
  tick(time) {
    this.orbit.tick(time);
    let redraw = this._dirty;
    if (this.tween) {
      const { from, to, start, duration } = this.tween;
      const t = Math.min(1, (time - start) / duration), e = 1 - Math.pow(1 - t, 3);
      const zoom = Math.exp(Math.log(from.zoom) + (Math.log(to.zoom) - Math.log(from.zoom)) * e);
      this.camera = { x: from.x + (to.x - from.x) * e, y: from.y + (to.y - from.y) * e, zoom };
      if (t >= 1) this.tween = null;
      redraw = true;
      this.emit();
    }
    const moving = time < (this.movingUntil || 0);
    if (moving !== this.wasMoving && this.refines()) redraw = true;
    this.wasMoving = moving;
    // The flash and the cross-fade are sampled by the wall clock: a due step is drawn even if this frame's timestamp is
    // a little early.
    const now = performance.now(), elapsed = this.flash ? now - this.flash.at : Infinity;
    if (this.flashDue != null && elapsed >= this.flashDue && this.flashing()) redraw = true;
    const tinted = this.tint ? now - this.tint.at : Infinity, tintStep = this.tintDue != null && tinted >= this.tintDue && this.tinting();
    if (tintStep) redraw = true;
    if (this.clusterGPU?.mixing) this.clusterGPU.mixing.t = tintValue(this.tint, tinted);
    // Go deeper's focus easing in or out: its strength for this frame's draw; and the veil inside its lit cells, as the
    // suspects' glow comes or goes.
    const faded = this.focusFade ? now - this.focusFade.at : Infinity;
    if (this.focusFadeDue != null && faded >= this.focusFadeDue) redraw = true;
    if (this.focusFade) this.focusLevel = focusValue(this.focusFade, faded);
    const inside = this.innerFade ? now - this.innerFade.at : Infinity;
    if (this.innerFadeDue != null && inside >= this.innerFadeDue) redraw = true;
    if (this.innerFade) this.innerLevel = focusValue(this.innerFade, inside);
    // The animators (the overlay's per-bin layer and marks, Go deeper's arrows) draw their due frames in this same
    // animation frame. A frame that only the mixing check's motion asks for (not a result's own) is counted for the
    // harness; an animator's frame that is not motion (frame() returns false: a refresh after an option changed) is not.
    let motion = tintStep && this.tintDue > 0;
    for (const animator of this.animators || []) {
      const due = animator.due(now);
      if (due != null && due <= now && animator.frame(now) !== false) motion = true;
    }
    if (motion) harness?.count('qcAnimFrames');
    if (redraw && this.gl) {
      this.draw(flashValue(this.flash, elapsed));
      this.flashDue = flashNext(this.flash, elapsed);
      this.tintDue = tintNext(this.tint, tinted);
      if (this.focusFade) {
        this.focusFadeDue = focusNext(this.focusFade, faded);
        if (this.focusFadeDue == null) { this.focusFade = null; if (!(this.focusLevel > 0)) this.focusOut = null; }   // over
      }
      if (this.innerFade) { this.innerFadeDue = focusNext(this.innerFade, inside); if (this.innerFadeDue == null) this.innerFade = null; }
    }
  }

  drawVolume(pulse) {
    const gl=this.gl, f=this.frame, o=this.options, targets=this.volumeTargets;
    const {program,uniforms:u}=this.volumeProgram;
    gl.bindFramebuffer(gl.FRAMEBUFFER,targets.output);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND); gl.depthMask(false);
    gl.useProgram(program); this.clusterGPU.bind(u,o,o.cellColorBy); gl.bindVertexArray(this.quadVao);
    gl.uniform2f(u.uCanvas,this.canvas.width,this.canvas.height);
    this.orbit.uniforms(gl,u);
    gl.uniform3f(u.uGeom,f.x0,f.y0,f.voxel);
    gl.uniform3i(u.uSize,f.nx,f.ny,f.layers.length);
    gl.uniform2f(u.uZ,this.info?.z?.grid_origin || 0,this.info?.z?.grid_spacing || 1/f.layers.length);
    for (const [unit,name,texture,target] of [
      [0,'uLayers',this.layerTexture,gl.TEXTURE_2D_ARRAY],
      [1,'uAge',this.textures.age,gl.TEXTURE_2D],
      [2,'uScene',targets.color,gl.TEXTURE_2D],
      [3,'uDepth',targets.depth,gl.TEXTURE_2D],
      [4,'uPoints',targets.points,gl.TEXTURE_2D]]) {
      gl.activeTexture(gl.TEXTURE0+unit); gl.bindTexture(target,texture); gl.uniform1i(u[name],unit);
    }
    gl.uniform1i(u.uCells,o.cells); gl.uniform1i(u.uHasAge,!!f.age); gl.uniform1i(u.uTrails,o.trails);
    gl.uniform1i(u.uMode,o.mode); gl.uniform1f(u.uIter,f.iter); gl.uniform1f(u.uPulse,pulse);
    gl.uniform1f(u.uOpacity,Math.max(0,o.volumeOpacity*o.fill/.42));
    gl.uniform1f(u.uFog,Math.max(0,o.volumeFog)); gl.uniform1f(u.uBox,o.cells ? o.volumeBox : 0);
    // Go deeper's focus in 3-D: its cell, its lit cells and how far the ease has gone (no focus: every pixel as before).
    const focus = this.focusShown();
    gl.uniform1i(u.uFocusOn, focus ? 1 : 0);
    if (focus) { gl.uniform1ui(u.uFocusCell, focus.cell); gl.uniform1uiv(u['uFocusLit[0]'], focus.lit); gl.uniform1f(u.uFocusDim, this.focusLevel); }
    gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER,targets.output);
    gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER,null);
    gl.blitFramebuffer(0,0,targets.width,targets.height,0,0,this.canvas.width,this.canvas.height,gl.COLOR_BUFFER_BIT,gl.LINEAR);
    gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.depthMask(true);
  }

  draw(pulse) {
    const gl = this.gl;
    this.dirty = false;
    this.drawCount++; harness?.count('stageDraws');
    const volume = this.orbit.active && this.frame?.layers?.length;
    this.renderScale = volume && this.wasMoving ? Math.max(.25, Math.min(1, this.options.volumeMovingScale)) : 1;
    if (volume) {
      this.volumeTargets.resize(Math.max(1, Math.round(this.canvas.width*this.renderScale)), Math.max(1, Math.round(this.canvas.height*this.renderScale)));
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.volumeTargets.scene);
    } else gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, volume ? this.volumeTargets.width : this.canvas.width, volume ? this.volumeTargets.height : this.canvas.height);
    gl.clearColor(0.043, 0.055, 0.075, 1);
    gl.depthMask(true);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.bindVertexArray(this.quadVao);
    if (this.orbit.active) gl.enable(gl.DEPTH_TEST); else gl.disable(gl.DEPTH_TEST);
    const view = [this.camera.x, this.camera.y, this.camera.zoom];
    const canvas = [this.canvas.width, this.canvas.height];

    if (this.image && this.textures.channels.length) {
      const { program, uniforms: u } = this.imageProgram;
      gl.useProgram(program);
      gl.disable(gl.BLEND);
      const r = this.image.rect;
      gl.uniform4f(u.uRect, r.x0, r.y0, r.w, r.h);
      gl.uniform3fv(u.uView, view);
      gl.uniform2fv(u.uCanvas, canvas);

      this.orbit.uniforms(gl, u);
      gl.uniform1f(u.uPlane, this.info?.z?.grid_origin || 0);
      gl.enable(gl.BLEND); gl.depthMask(false);
      for (let k = this.image.channels.length - 1; k >= 0; k--) {
        const c = this.image.channels[k];
        if (!c.on || !this.textures.channels[k]) continue;   // off, or not loaded yet (a view loads visible channels)
        const map = c.colormap || 'gray';
        if (!this.luts.has(map)) {
          if (isHexColour(map)) {   // a colour picker sends many: drop the colours of their own no channel uses any more
            const used = new Set(this.image.channels.map(ch => ch.colormap));
            for (const [name, old] of this.luts) if (isHexColour(name) && !used.has(name)) { gl.deleteTexture(old); this.luts.delete(name); }
          }
          const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB8,256,1,0,gl.RGB,gl.UNSIGNED_BYTE,makeLUT(map));
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
          gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
          this.luts.set(map,t);
        }
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,this.textures.channels[k]);
        gl.uniform1i(u.uChannel,0);
        gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,this.luts.get(map)); gl.uniform1i(u.uLut,1);
        gl.uniform2f(u.uContrast,c.low ?? 0,c.high ?? 255);
        gl.uniform1f(u.uScale,c.source.data ? (this.norm16 ? 65535 : c.max || 65535) : 255);
        gl.uniform1f(u.uGamma,c.gamma ?? 1); gl.uniform1f(u.uOpacity,c.opacity ?? 1);
        gl.blendFunc(gl.SRC_ALPHA,c.blending === 'translucent' ? gl.ONE_MINUS_SRC_ALPHA : gl.ONE);
        gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
      }
      gl.depthMask(true);
    }

    if (volume) {
      // Share the image's depth for occlusion, but never let translucent points
      // write depth: their soft sprite footprints must not cut holes in the volume.
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.volumeTargets.pointScene);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.depthMask(false);
      this.drawTranscripts();
    } else if (this.orbit.active) this.drawTranscripts();
    if (volume) { this.drawVolume(pulse); return; }
    gl.depthMask(false);
    const f = this.frame;
    this.drawXenium?.();
    this.drawPriors?.();
    const focus = this.orbit.active ? null : this.focusShown();   // Go deeper's outlines and veil: 2-D (3-D: drawVolume)
    if (f && this.options.cells) {
      const { program, uniforms: u } = this.maskProgram;
      const o = this.options;
      gl.useProgram(program);
      this.clusterGPU.bind(u,o,o.cellColorBy);
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform4f(u.uRect, f.x0, f.y0, f.nx * f.voxel, f.ny * f.voxel);
      gl.uniform3fv(u.uView, view);
      gl.uniform2fv(u.uCanvas, canvas);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.textures.lab);
      gl.uniform1i(u.uLab, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, this.textures.age);
      gl.uniform1i(u.uAge, 1);
      gl.uniform1i(u.uHasAge, f.age ? 1 : 0);
      gl.uniform3f(u.uGeom, f.x0, f.y0, f.voxel);
      gl.uniform2i(u.uSize, f.nx, f.ny);
      gl.uniform1f(u.uPxPerUm, this.camera.zoom);
      gl.uniform1f(u.uEdgePx, o.edgePx * this.ratio);
      gl.uniform1i(u.uMode, o.mode);
      gl.uniform1f(u.uFill, o.fill);
      gl.uniform1f(u.uIter, f.iter);
      gl.uniform1i(u.uTrails, o.trails ? 1 : 0);
      gl.uniform1f(u.uPulse, pulse);
      gl.uniform1ui(u.uHover, o.hover >>> 0);
      gl.uniform1ui(u.uSelected, this.orbit.active ? 0 : o.selected >>> 0);
      gl.uniform1i(u.uFocusOn, focus ? 1 : 0);   // Go deeper's outlines and greying
      if (focus) {
        gl.uniform1ui(u.uFocusCell, focus.cell); gl.uniform1ui(u.uFocusHot, focus.hot); gl.uniform1f(u.uFocusExtra, this.ratio);
        gl.uniform1uiv(u['uFocusSrc[0]'], focus.src); gl.uniform1fv(u['uFocusA[0]'], focus.alpha); gl.uniform1uiv(u['uFocusSame[0]'], focus.same);
        gl.uniform1uiv(u['uFocusLit[0]'], focus.lit); gl.uniform1f(u.uFocusDim, this.focusLevel); gl.uniform1f(u.uFocusBold, FOCUS_CELL_BOLD_PX * this.ratio);
      }
      const spill = this.orbit.active ? null : this.spill;   // and its dots' fading: 2-D only
      gl.uniform1i(u.uSpillOn, spill ? 1 : 0);
      if (spill) gl.uniform1uiv(u['uSpillLit[0]'], spill.lit);
      this.orbit.uniforms(gl, u);
      if (!this.orbit.active) {
        gl.uniform1f(u.uPlane, 0);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    }
    // While the suspects glow the lit cells dim inside their outlines too, under the dots: the image there gives way
    // to the glow (its own canvas) and the dots, which keep their colour.
    const inner = focus && f && this.veilProgram ? FOCUS_INNER * this.focusLevel * (this.innerLevel || 0) : 0;
    if (inner > 0) this.drawVeil(focus, inner);
    if (!this.orbit.active) { if (this.spill) this.drawSpill(); else this.drawTranscripts(); }   // Go deeper's dots instead
    // The veil last, over all of it: dense transcripts pile up, so fading each dot alone hardly shows (measured in a
    // browser: 0.94); laid over the drawing it darkens whatever sits outside the lit cells.
    if (focus && f && this.focusLevel > 0 && this.veilProgram) this.drawVeil(focus);
    gl.depthMask(true);
  }
}

export const XENIUM_VS = QUAD_VS;
export const XENIUM_FS = MASK_FS.replace('uniform bool uHasAge;', 'uniform bool uHasAge; uniform usampler2D uMapping; uniform int uMapWidth; uniform float uOpacity;')
  .replace('vec3 base = displayColor(id);', 'uint mapped = texelFetch(uMapping, ivec2(int(id)%uMapWidth,int(id)/uMapWidth),0).r; vec3 base = mapped == 0u ? vec3(.55) : displayColor(mapped);')
  .replace('col.a *= clusterWeight(id);', 'uint mapped = texelFetch(uMapping, ivec2(int(id)%uMapWidth,int(id)/uMapWidth),0).r; col.a *= uOpacity * (mapped == 0u ? 1. : clusterWeight(mapped));');
