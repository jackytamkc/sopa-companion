// Go deeper's suspect transcripts, glowing (Stage.setGlow): the transcripts proseg counts for the window's cell that
// it reads as likely from a neighbour, each a soft hot-pink halo (about GLOW_HALO x the window's dot, GLOW_HALO_MIN_PX to
// GLOW_HALO_MAX_PX CSS px across) under a bright white-pink core, as strong as its p (P: proseg's share of it for the
// cell times how likely it came from a neighbour), breathing between GLOW_LOW and full over GLOW_PERIOD_MS, the breath
// rippling in from the giver's side (none when the window names no giver), while the stage dims the lit cells' image
// under it (Stage.setGlow). The one thing on the stage that moves by itself, on purpose (the breathing draws the eye):
// at most GLOW_FPS repaints a second, and only while it is set and seen, in 2-D and 3-D.
//
// It draws on its own 2-D canvas between the stage and the overlay, sized and placed to the box of the glowing points on
// screen plus their halo, so a repaint changes only that patch (a remote display reads back and encodes what changed).
// In 3-D each point stands at its depth, placed as the transcripts are drawn (Orbit.project, the shaders' own
// projection), its halo and core scaled as their dots are by distance; over the WebGL canvas, the volume never hides it;
// a point without a depth, or behind the eye, is not drawn there. Its frames come through the stage's scheduler
// (Stage.addAnimator: it joins only while it has points), in the stage's animation frames, so a camera move or an orbit
// redraws it with the stage and the overlay. They are neither stage draws (the stage stays still) nor the mixing check's
// motion (frame() returns false): the harness counts them as glowFrames. Nothing is scheduled while it is off, the page
// hidden or the WebGL context lost; with reduced or calm motion it is drawn once per change, at full strength.
import { harness as hook } from './harness-hook.mjs';

export const GLOW_FPS = 12;              // repaints a second at most while it breathes
export const GLOW_PERIOD_MS = 1600;      // one breath: from GLOW_LOW up to full and back
export const GLOW_LOW = 0.6;             // the breath between 60 % and 100 %
export const GLOW_RIPPLE = 0.5;          // the far side of the points breathes this share of a breath after the giver's side
// The halo across (CSS px): about 6 x the dot, 30-56 px (at 4 x, 16-36 px, the suspects read as small pink dots on
// bright stain, only +0.08 brighter there).
export const GLOW_HALO = 6, GLOW_HALO_MIN_PX = 30, GLOW_HALO_MAX_PX = 56;
export const GLOW_CORE = 0.8;            // the core across: this share of the dot (at least GLOW_CORE_MIN_PX)
export const GLOW_CORE_MIN_PX = 4;
export const GLOW_CORE_LOW = 0.85;       // the core breathes from this to full: the transcript's place stays plain
export const GLOW_COLOR = '#ff4fa3';     // hot magenta-pink, apart from the neighbours' #d55181
export const COUNTERS = Object.freeze(['glowFrames']);
const [R, G, B] = [1, 3, 5].map(i => parseInt(GLOW_COLOR.slice(i, i + 2), 16));
const pink = a => `rgba(${R},${G},${B},${a})`;
/** The halo's and the core's radial falloffs ([place from the centre 0 to the rim 1, colour]), exported for the test
 *  corner's glow to match. The halo's falloff is wide and bright (halos add up where they overlap, so a suspect reads
 *  at any zoom; a crowd may bloom toward white); the core is whiter. */
export const HALO_STOPS = Object.freeze([[0, pink(1)], [0.18, pink(0.8)], [0.42, pink(0.38)], [0.7, pink(0.1)], [1, pink(0)]].map(Object.freeze));
export const CORE_STOPS = Object.freeze([[0, 'rgba(255,255,255,1)'], [0.5, 'rgba(255,218,236,1)'], [0.8, pink(0.9)], [1, pink(0)]].map(Object.freeze));

/** How strong the glow is `t` ms after it began (GLOW_LOW-1, starting low), for a point `phase` (a share of a breath)
 *  behind the giver's side. */
export function glowBreath(t, phase = 0) {
  return GLOW_LOW + (1 - GLOW_LOW) * (0.5 - 0.5 * Math.cos(2 * Math.PI * (t / GLOW_PERIOD_MS - phase)));
}
/** Each point's phase (0 to GLOW_RIPPLE) by its place along the line from the giver to the points' middle, the giver's
 *  side first; all 0 without a giver or when the points do not spread along that line. `points`: [x, y, p] triples. */
export function ripplePhases(points, giver) {
  const n = Math.floor(points.length / 3), out = new Float32Array(n);
  if (!giver || !n) return out;
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) { cx += points[3 * i]; cy += points[3 * i + 1]; }
  const dx = cx / n - giver[0], dy = cy / n - giver[1], len = Math.hypot(dx, dy);
  if (!(len > 1e-6)) return out;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    out[i] = ((points[3 * i] - giver[0]) * dx + (points[3 * i + 1] - giver[1]) * dy) / len;
    lo = Math.min(lo, out[i]); hi = Math.max(hi, out[i]);
  }
  for (let i = 0; i < n; i++) out[i] = hi - lo > 1e-6 ? (out[i] - lo) / (hi - lo) * GLOW_RIPPLE : 0;
  return out;
}
/** A dot's size at `depth` in 3-D against its size at the eye's distance `dist` from what it looks at (transcripts.js's
 *  rule for the transcripts' dots: clamp(sqrt(dist / depth), 0.85, 1.5)). */
export function depthScale(dist, depth) { return Math.max(0.85, Math.min(1.5, Math.sqrt(dist / depth))); }
/** The halo's and the core's size (CSS px across) around the window's dot of `dot` CSS px. */
export function glowSizes(dot) {
  const d = Number.isFinite(dot) && dot > 0 ? dot : 6;
  return { halo: Math.max(GLOW_HALO_MIN_PX, Math.min(GLOW_HALO_MAX_PX, GLOW_HALO * d)), core: Math.max(GLOW_CORE_MIN_PX, GLOW_CORE * d) };
}
/** The box {x, y, w, h} (whole CSS px) that holds the points on screen `xy` ([x, y] pairs, CSS px) with `reach` px around
 *  each, inside the stage (`width` x `height`); null when none of it is on screen. */
export function glowBox(xy, reach, width, height) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i + 1 < xy.length; i += 2) {
    x0 = Math.min(x0, xy[i]); x1 = Math.max(x1, xy[i]); y0 = Math.min(y0, xy[i + 1]); y1 = Math.max(y1, xy[i + 1]);
  }
  x0 = Math.max(0, Math.floor(x0 - reach)); y0 = Math.max(0, Math.floor(y0 - reach));
  x1 = Math.min(width, Math.ceil(x1 + reach)); y1 = Math.min(height, Math.ceil(y1 + reach));
  return x1 > x0 && y1 > y0 ? { x: x0, y: y0, w: x1 - x0, h: y1 - y0 } : null;
}

/**
 * The glow of one window. new SuspectGlow(stage, {dot, now, reducedMotion, createCanvas, harness}): `stage` the Stage
 * (camera, toScreen, ratio, cssWidth/cssHeight, canvas, gl, orbit, calmMotion, reducedMotion(), addAnimator and
 * removeAnimator); dot() the window's dot now (CSS px, Stage.spillDotPx); createCanvas() a canvas for the glow and its
 * sprites (tests; by default the page's). set(glow) with {key, points: Float32Array of [x µm, y µm, p], depth:
 * Float32Array of each point's depth (the transcripts' normalised z; NaN for none) or null, giver} or null.
 */
export class SuspectGlow {
  constructor(stage, { dot = () => 6, now = () => performance.now(), reducedMotion = null, createCanvas = null, harness = hook } = {}) {
    this.stage = stage; this.dot = dot; this.now = now; this.prefersReduced = reducedMotion; this.createCanvas = createCanvas;
    this.harness = harness; this.key = ''; this.points = null; this.depth = null; this.phases = null; this.count = 0; this.canvas = null;
    this.shown = false; this.dirty = false; this.start = 0; this.drawnAt = -Infinity; this.drawnView = ''; this.drawnBreathing = null;
    this.sprites = new Map();
    for (const name of COUNTERS) this.harness?.count(name, 0);
  }
  /** The glowing transcripts (those with finite places and p > 0; p over 1 counts as 1), each with its depth (`depth`,
   *  in the points' order: NaN where there is none, then it is drawn in 2-D only), or null. The same key, number of
   *  points and giver again: no work. Returns whether it changed. */
  set(glow) {
    const g = glow && glow.points instanceof Float32Array ? glow : null;
    const giver = g && Array.isArray(g.giver) && Number.isFinite(g.giver[0]) && Number.isFinite(g.giver[1]) ? [+g.giver[0], +g.giver[1]] : null;
    const key = g ? `${g.key ?? ''}|${g.points.length}|${giver ? giver.join(',') : ''}` : '';
    if (key === this.key) return false;
    this.key = key;
    const kept = [], depths = [], depth = g?.depth && typeof g.depth.length === 'number' ? g.depth : null;
    for (let i = 0; g && i + 2 < g.points.length; i += 3) {
      const x = g.points[i], y = g.points[i + 1], p = g.points[i + 2], z = depth ? Number(depth[i / 3]) : NaN;
      if (Number.isFinite(x) && Number.isFinite(y) && p > 0) { kept.push(x, y, Math.min(1, p)); depths.push(Number.isFinite(z) ? z : NaN); }
    }
    if (!kept.length) {
      this.points = this.depth = this.phases = null; this.count = 0; this.hide();
      this.stage.removeAnimator?.(this);
      return true;
    }
    this.points = Float32Array.from(kept); this.depth = Float32Array.from(depths); this.count = kept.length / 3;
    this.phases = ripplePhases(this.points, giver);
    this.start = this.now(); this.dirty = true;
    this.stage.addAnimator?.(this);   // plans its first paint, in the stage's next frame
    return true;
  }
  /** Hide it at once (off, a lost context); the next paint shows it again. */
  hide() { this.conceal(); this.drawnView = ''; }
  conceal() {
    if (this.canvas && this.canvas.style.display !== 'none') this.canvas.style.display = 'none';
    this.shown = false;
  }
  /** It can be drawn: set, with the stage's context alive (in 2-D and 3-D). */
  active() { return !!this.points && !!this.stage.gl; }
  /** It breathes (else it is drawn once per change, at full strength): neither reduced nor calm motion. */
  breathing() {
    if (this.stage.calmMotion) return false;
    if (this.prefersReduced) return !this.prefersReduced();
    return !this.stage.reducedMotion?.();
  }
  /** What places it on screen: a change means a new box and a repaint (in 3-D the orbit and the slab's height too). */
  view() {
    const s = this.stage, c = s.camera, o = s.orbit;
    return `${c.x},${c.y},${c.zoom},${s.cssWidth},${s.cssHeight},${s.ratio},${this.dot()},${o?.active ? 1 : 0},${o?.angle},${o?.yaw},${o?.amount},`
      + `${s.options?.stretch},${s.info?.z?.thickness_um}`;
  }

  // ---- the stage's animator -----------------------------------------------------------------------------------------
  /** When its next frame is due (performance.now() ms), or null: at once for a change, a camera move or to hide it; then
   *  every 1000 / GLOW_FPS ms on a fixed grid while it breathes and is on screen; never while the page is hidden. */
  due(now = this.now()) {
    if (globalThis.document?.hidden) return null;
    if (!this.active()) return this.shown ? now : null;
    if (this.dirty || this.view() !== this.drawnView || this.breathing() !== this.drawnBreathing) return now;
    if (!this.shown || !this.drawnBreathing) return null;
    const step = 1000 / GLOW_FPS;
    return this.start + (Math.floor((this.drawnAt - this.start) / step + 1e-6) + 1) * step;
  }
  /** One frame, in the stage's animation frame: not the mixing check's motion (stage.js counts only that). */
  frame(now = this.now()) { this.paint(now); return false; }
  /** Where each point shows now: [index, x, y (CSS px), its size's scale] in draw order (its own), in 2-D every point at
   *  scale 1 (Stage.toScreen); in 3-D (the orbit active) those with a depth in front of the eye, placed by
   *  Orbit.project and scaled by depthScale. */
  placed() {
    const s = this.stage, o = s.orbit?.active ? s.orbit : null, out = [], dist = o ? o.distance() : 0;
    for (let i = 0; i < this.count; i++) {
      const x = this.points[3 * i], y = this.points[3 * i + 1];
      if (!o) { const [sx, sy] = s.toScreen(x, y); out.push([i, sx, sy, 1]); continue; }
      const z = this.depth?.[i], p = Number.isFinite(z) ? o.project(x, y, z) : null;
      if (p) out.push([i, p.x, p.y, depthScale(dist, p.depth)]);
    }
    return out;
  }
  /** Draw it as it is at `now` (or hide it): returns whether it drew. */
  paint(now = this.now()) {
    this.dirty = false;
    if (!this.active()) { this.hide(); return false; }
    const s = this.stage, ratio = s.ratio || 1;
    this.drawnView = this.view(); this.drawnBreathing = this.breathing(); this.drawnAt = now;
    const at = this.placed(), xy = new Float32Array(2 * at.length);
    at.forEach(([, x, y], j) => { xy[2 * j] = x; xy[2 * j + 1] = y; });
    const { halo, core } = glowSizes(this.dot()), most = at.reduce((m, p) => Math.max(m, p[3]), 1);
    const box = glowBox(xy, halo * most / 2 + 1, s.cssWidth, s.cssHeight);
    if (!box) { this.conceal(); return false; }   // every point off screen (or none placed in 3-D): nothing until the view moves
    const c = this.place(box, ratio), g = c.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height);
    g.setTransform(ratio, 0, 0, ratio, -box.x * ratio, -box.y * ratio);
    const t = now - this.start, breathing = this.drawnBreathing, halos = this.sprite('halo', halo, ratio), cores = this.sprite('core', core, ratio);
    g.globalCompositeOperation = 'lighter';   // where halos overlap they add up, as light does
    for (const [i, x, y, k] of at) {
      g.globalAlpha = this.points[3 * i + 2] * (breathing ? glowBreath(t, this.phases[i]) : 1);
      g.drawImage(halos, x - halo * k / 2, y - halo * k / 2, halo * k, halo * k);
    }
    g.globalCompositeOperation = 'source-over';
    for (const [i, x, y, k] of at) {   // the cores breathe less: the transcript's place stays plain
      g.globalAlpha = this.points[3 * i + 2] * (breathing ? GLOW_CORE_LOW + (1 - GLOW_CORE_LOW) * glowBreath(t, this.phases[i]) : 1);
      g.drawImage(cores, x - core * k / 2, y - core * k / 2, core * k, core * k);
    }
    g.globalAlpha = 1;
    this.shown = true;
    this.harness?.count('glowFrames');
    return true;
  }
  /** The canvas sized and placed to `box` (CSS px of the stage), made on first use between the stage and the overlay. */
  place(box, ratio) {
    let c = this.canvas;
    if (!c) {
      c = this.canvas = this.createCanvas ? this.createCanvas() : globalThis.document.createElement('canvas');
      c.id = 'glow'; c.setAttribute?.('aria-hidden', 'true');
      // Screen blending: the glow only ever adds light to what is under it, so a suspect brightens on bright stain too
      // (laid over normally, pink on bright yellow came out darker).
      c.style.cssText = 'position:absolute;left:0;top:0;width:0;height:0;pointer-events:none;display:none;mix-blend-mode:screen';
      const at = this.stage.canvas;
      at?.parentNode?.insertBefore?.(c, at.nextSibling ?? null);
    }
    const w = Math.max(1, Math.round(box.w * ratio)), h = Math.max(1, Math.round(box.h * ratio));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    const style = c.style, put = (name, value) => { if (style[name] !== value) style[name] = value; };
    put('left', `${box.x}px`); put('top', `${box.y}px`); put('width', `${box.w}px`); put('height', `${box.h}px`); put('display', 'block');
    return c;
  }
  /** A halo or core of `size` CSS px drawn once (a radial falloff) and kept for the zoom levels in use. */
  sprite(kind, size, ratio) {
    const px = Math.max(2, Math.ceil(size * ratio)), key = `${kind}|${px}`;
    let s = this.sprites.get(key);
    if (s) return s;
    if (this.sprites.size >= 8) this.sprites.clear();
    s = this.createCanvas ? this.createCanvas() : typeof OffscreenCanvas === 'function' ? new OffscreenCanvas(px, px) : globalThis.document.createElement('canvas');
    s.width = px; s.height = px;
    const g = s.getContext('2d'), r = px / 2, fill = g.createRadialGradient(r, r, 0, r, r, r);
    for (const [at, colour] of kind === 'halo' ? HALO_STOPS : CORE_STOPS) fill.addColorStop(at, colour);
    g.fillStyle = fill; g.fillRect(0, 0, px, px);
    this.sprites.set(key, s);
    return s;
  }
  /** {on (set and on screen), count (the transcripts that glow), fps (repaints a second while it breathes; 0 static)}. */
  state() { const on = !!this.points && this.shown; return { on, count: this.count, fps: on && this.breathing() ? GLOW_FPS : 0 }; }
}
