import test from 'node:test';
import assert from 'node:assert/strict';
import { Orbit, PROJECT } from '../src/scene3d.js';
import { Stage } from '../src/stage.js';

// The orbit camera's projection on the page (scene3d.js Orbit.project): where a point shows on the stage in 3-D, as
// the shaders draw it (PROJECT, with the values uniforms() sets), so Go deeper's glow stands where the transcripts are
// drawn. It is the exact inverse of Orbit.ground on the image, the 2-D toScreen with the orbit at rest, and a raised
// point stands higher on the screen.

/** A stage of `width` x `height` device px at pixel ratio `ratio`, its camera at (300, 200) µm and `zoom`, a 15 µm slab;
 *  the Stage's own toScreen and its real Orbit at `deg` degrees, turned by `yaw`, its slab at `amount`. */
function stageAt({ deg = 55, yaw = 0, zoom = 2, amount = 1, width = 1600, height = 1000, ratio = 1, stretch = 1 } = {}) {
  const stage = Object.create(Stage.prototype);
  Object.assign(stage, { canvas: { width, height }, pixelRatio: ratio, cssWidth: width / ratio, cssHeight: height / ratio,
    camera: { x: 300, y: 200, zoom }, info: { z: { thickness_um: 15 } }, options: { stretch }, frame: null });
  stage.orbit = new Orbit(stage);
  Object.assign(stage.orbit, { enabled: amount > 0, angle: deg * Math.PI / 180, yaw, amount });
  return stage;
}
const ANGLES = [0, 30, 55], YAWS = [0, 0.6, 2.0], ZOOMS = [0.5, 2];

test('project undoes ground on the image: any stage pixel, any angle, turn and zoom', () => {
  for (const deg of ANGLES) for (const yaw of YAWS) for (const zoom of ZOOMS) for (const ratio of [1, 1.5]) {
    const s = stageAt({ deg, yaw, zoom, ratio }), W = s.canvas.width / ratio, H = s.canvas.height / ratio;
    for (const [px, py] of [[W / 2, H / 2], [10, 20], [W - 3, 7], [W * 0.3, H * 0.8], [W - 1, H - 1]]) {
      const [x, y] = s.orbit.ground(px, py), p = s.orbit.project(x, y, 0);
      const name = `${deg} deg, yaw ${yaw}, zoom ${zoom}, ratio ${ratio}: ${px},${py}`;
      assert.ok(p, name);
      assert.ok(Math.abs(p.x - px) < 1e-6 && Math.abs(p.y - py) < 1e-6, `${name} -> ${p.x},${p.y}`);
      assert.ok(p.depth > 0, name);
    }
  }
});

test('ground and project take another camera than the stage\'s (the 3-D fit tries cameras without moving it)', () => {
  const s = stageAt({ deg: 55, yaw: 0.6 }), cam = { x: 900, y: -40, zoom: 0.7 }, before = { ...s.camera };
  const [x, y] = s.orbit.ground(123, 456, cam), p = s.orbit.project(x, y, 0, cam);
  assert.ok(Math.abs(p.x - 123) < 1e-6 && Math.abs(p.y - 456) < 1e-6);
  assert.deepEqual(s.camera, before, 'the stage\'s own camera untouched');
  const q = s.orbit.project(x, y, 0);
  assert.ok(Math.hypot(q.x - 123, q.y - 456) > 1, 'its own camera sees it elsewhere');
});

test('at rest (the 2-D view: no angle, no turn, no slab) project is the stage\'s own toScreen', () => {
  for (const zoom of ZOOMS) for (const ratio of [1, 2]) {
    const s = stageAt({ deg: 0, yaw: 0, amount: 0, zoom, ratio });
    for (const [x, y] of [[300, 200], [0, 0], [512.25, -33.5], [310, 190]]) for (const z of [0, 0.5, 1]) {
      const p = s.orbit.project(x, y, z), [sx, sy] = s.toScreen(x, y);
      assert.ok(Math.abs(p.x - sx) < 1e-6 && Math.abs(p.y - sy) < 1e-6, `${x},${y},${z} at zoom ${zoom}, ratio ${ratio}: ${p.x},${p.y} against ${sx},${sy}`);
    }
  }
});

test('a raised point under the middle of the stage keeps its x and stands higher, at 55 degrees for every turn', () => {
  for (const yaw of YAWS) for (const stretch of [1, 3]) {
    const s = stageAt({ deg: 55, yaw, stretch }), W = s.canvas.width, H = s.canvas.height, [x, y] = s.orbit.ground(W / 2, H / 2);
    const low = s.orbit.project(x, y, 0), high = s.orbit.project(x, y, 1), mid = s.orbit.project(x, y, 0.5);
    assert.ok(Math.abs(high.x - W / 2) < 1e-6 && Math.abs(low.x - W / 2) < 1e-6, `yaw ${yaw}: x kept`);
    assert.ok(high.y < mid.y && mid.y < low.y, `yaw ${yaw}: higher on the screen (${low.y} -> ${mid.y} -> ${high.y})`);
    assert.ok(high.depth < low.depth, 'nearer the eye');
  }
  // The slab's height is the shaders' uHeight: thickness x 1.5 x stretch x amount.
  const s = stageAt({ amount: 0.5, stretch: 2 });
  assert.equal(s.orbit.height(), 15 * 1.5 * 2 * 0.5);
});

test('project is the shaders\' PROJECT with uniforms()\'s values: the same place and depth on the stage', () => {
  // The shader's own lines, read from PROJECT: the projection the transcripts, the image and the volume share.
  assert.ok(PROJECT.includes('vec3 d = vec3(p.x, -p.y, p.z * uHeight) - uCentre;') && PROJECT.includes('float depth = uDistance - dot(d, uBack);'));
  assert.ok(PROJECT.includes('return vec4(dot(d, uRight) * uFocal / (uCanvas.x / uCanvas.y),\n              dot(d, uUp) * uFocal,'));
  for (const deg of [10, 55, 81]) for (const yaw of [0, 0.6, -2.6]) {
    const s = stageAt({ deg, yaw, zoom: 1.3, ratio: 1.25, stretch: 2 }), u = {};
    s.orbit.uniforms({ uniform1i: (k, v) => { u[k] = v; }, uniform1f: (k, v) => { u[k] = v; }, uniform3f: (k, ...v) => { u[k] = v; }, uniform3fv: (k, v) => { u[k] = v; } },
      Object.fromEntries(['u3D', 'uCentre', 'uRight', 'uUp', 'uBack', 'uDistance', 'uFocal', 'uHeight', 'uNear', 'uFar'].map(k => [k, k])));
    const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2], W = s.canvas.width, H = s.canvas.height;
    for (const [x, y, z] of [[300, 200, 0], [320, 150, 1], [250, 260, 0.4], [301, 199, 0.9]]) {
      const d = [x - u.uCentre[0], -y - u.uCentre[1], z * u.uHeight - u.uCentre[2]];
      const w = u.uDistance - dot(d, u.uBack), cx = dot(d, u.uRight) * u.uFocal / (W / H) / w, cy = dot(d, u.uUp) * u.uFocal / w;
      const p = s.orbit.project(x, y, z), name = `${deg} deg, yaw ${yaw}: ${x},${y},${z}`;
      assert.ok(Math.abs(p.depth - w) < 1e-9, name);
      assert.ok(Math.abs(p.x - (cx + 1) / 2 * W / 1.25) < 1e-6 && Math.abs(p.y - (1 - cy) / 2 * H / 1.25) < 1e-6, name);
    }
    // Behind the near plane: no place (the shaders clip it).
    const eye = s.orbit.basis().back.map(v => v * u.uDistance);
    assert.equal(s.orbit.project(s.camera.x + eye[0], s.camera.y - eye[1], eye[2] / u.uHeight), null, `${deg} deg: the eye itself`);
    assert.ok(Math.abs(u.uNear - Math.max(0.001, u.uDistance / 10000)) < 1e-12);
  }
});
