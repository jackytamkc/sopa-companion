// Orbit camera expressed as an orthonormal basis; shared by quads and points.
export const PROJECT = `
uniform bool u3D;
uniform vec3 uCentre, uRight, uUp, uBack;
uniform float uDistance, uFocal, uHeight, uNear, uFar;
vec4 project(vec3 p) {
  vec3 d = vec3(p.x, -p.y, p.z * uHeight) - uCentre;
  float depth = uDistance - dot(d, uBack);
  return vec4(dot(d, uRight) * uFocal / (uCanvas.x / uCanvas.y),
              dot(d, uUp) * uFocal,
              (uFar + uNear) / (uFar - uNear) * depth - 2.0 * uFar * uNear / (uFar - uNear), depth);
}`;

export class Orbit {
  constructor(stage) { this.stage = stage; this.angle = 0; this.yaw = 0; this.amount = 0; this.enabled = false; }
  switch(on, animate = true) {
    this.enabled = on;
    if (on) this.stage.mipmapImage?.();   // the tilted image is shown smaller than it was fetched
    this.transition = { start: performance.now(), from: [this.angle, this.yaw, this.amount], to: [on ? Math.PI * 55 / 180 : 0, 0, on ? 1 : 0], duration: animate && !matchMedia('(prefers-reduced-motion: reduce)').matches ? 600 : 0 };
    this.tick(performance.now());
  }
  get active() { return this.enabled || !!this.transition || this.amount > 0; }
  tick(time) {
    const t = this.transition;
    if (!t) return;
    const p = t.duration ? Math.min(1, (time - t.start) / t.duration) : 1, e = p * p * (3 - 2 * p);
    [this.angle, this.yaw, this.amount] = t.from.map((v, i) => v + (t.to[i] - v) * e);
    if (p === 1) this.transition = null;
    this.stage.dirty = true;
    this.stage.emit();
  }
  basis() {
    const c = Math.cos(this.angle), s = Math.sin(this.angle), cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
    return { right: [cy, sy, 0], up: [-sy * c, cy * c, s], back: [sy * s, -cy * s, c] };
  }
  /** The eye's distance from the point it looks at (µm) for `camera` (its zoom: device px per µm there): uDistance. */
  distance(camera = this.stage.camera) { return this.stage.canvas.height / (2 * Math.tan(Math.PI / 8) * camera.zoom); }
  /** How high (µm) the top of the slab stands as drawn: the physical thickness, gently exaggerated (the slider
   *  multiplies this baseline), as far as the orbit has come in: uHeight, by which a normalised depth z is raised. */
  height() {
    const s = this.stage, thickness = s.info?.z?.thickness_um || 1, stretch = 1.5 * s.options.stretch;
    return thickness * stretch * this.amount;
  }
  uniforms(gl, u) {
    const s = this.stage, f = s.frame, b = this.basis();
    const width = f ? f.nx * f.voxel : 300;
    const thickness = s.info?.z?.thickness_um || 1;
    const stretch = 1.5 * s.options.stretch;
    const distance = this.distance();
    gl.uniform1i(u.u3D, this.active);
    gl.uniform3f(u.uCentre, s.camera.x, -s.camera.y, 0);
    gl.uniform3fv(u.uRight, b.right); gl.uniform3fv(u.uUp, b.up); gl.uniform3fv(u.uBack, b.back);
    gl.uniform1f(u.uDistance, distance); gl.uniform1f(u.uFocal, 1 / Math.tan(Math.PI / 8));
    gl.uniform1f(u.uHeight, this.height());
    gl.uniform1f(u.uNear, Math.max(0.001, distance / 10000));
    gl.uniform1f(u.uFar, distance + width * 20 + thickness * stretch * 4);
  }
  /** Where a point (x, y µm; z the transcripts' normalised depth, 0 on the image) shows on the stage in this view, as
   *  the shaders draw it (PROJECT with uniforms()' values; `camera` the stage's by default): {x, y} CSS px and its
   *  `depth` (µm along the view, the shaders' w), or null when it lies at or behind the near plane (uNear). The exact
   *  inverse of ground() on the image, and the 2-D toScreen() with the orbit at rest. */
  project(x, y, z = 0, camera = this.stage.camera) {
    const s = this.stage, b = this.basis(), dist = this.distance(camera), focal = 1 / Math.tan(Math.PI / 8);
    const d = [x - camera.x, camera.y - y, z * this.height()];
    const depth = dist - (d[0] * b.back[0] + d[1] * b.back[1] + d[2] * b.back[2]);
    if (!(depth > Math.max(0.001, dist / 10000))) return null;
    const W = s.canvas.width, H = s.canvas.height, r = s.ratio || 1;   // device px; W / r is the stage's CSS width
    const nx = (d[0] * b.right[0] + d[1] * b.right[1] + d[2] * b.right[2]) * focal / (W / H) / depth;
    const ny = (d[0] * b.up[0] + d[1] * b.up[1] + d[2] * b.up[2]) * focal / depth;
    return { x: (nx + 1) / 2 * W / r, y: (1 - ny) / 2 * H / r, depth };
  }
  /** The point on the image (z 0, µm) under the stage's CSS px (x, y) in this view, seen from `camera` (the stage's by
   *  default). */
  ground(x, y, camera = this.stage.camera) {
    const s = this.stage, b = this.basis(), d = this.distance(camera);
    const sx = (x * s.ratio - s.canvas.width / 2) / camera.zoom;
    const sy = -(y * s.ratio - s.canvas.height / 2) / camera.zoom;
    const ray = b.back.map((v, i) => -v * d + b.right[i] * sx + b.up[i] * sy);
    const t = -b.back[2] * d / ray[2];
    return [camera.x + b.back[0] * d + ray[0] * t, camera.y - b.back[1] * d - ray[1] * t];
  }
  drag(dx, dy, pan) {
    if (this.transition) return;
    const s = this.stage;
    if (pan) {
      const x = s.canvas.clientWidth / 2, y = s.canvas.clientHeight / 2;
      const a = this.ground(x, y), b = this.ground(x + dx, y + dy);
      s.camera.x += a[0] - b[0]; s.camera.y += a[1] - b[1];
    } else {
      this.yaw -= dx * 0.006;
      this.angle = Math.max(0, Math.min(Math.PI * 0.45, this.angle + dy * 0.006));
    }
    s.dirty = true; s.emit();
  }
}
