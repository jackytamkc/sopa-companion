import test from 'node:test';
import assert from 'node:assert/strict';
import { Stage } from '../src/stage.js';
import { VolumeTargets } from '../src/volume.js';
import { PriorLayers } from '../src/prior-layers.js';
import { TiledLoader } from '../src/tiled-loader.mjs';
import { fitSize } from '../src/robustness.mjs';
import { fakeLoop } from './fake-loop.mjs';

// Synchronous WebGL calls: each one waits for the GPU process, which stalls the page whenever presenting is
// back-pressured (a remote display that encodes every frame in software). Allowed at initialisation only.
const SYNC = ['getError', 'getParameter', 'readPixels', 'checkFramebufferStatus', 'getShaderParameter', 'getProgramParameter',
  'getActiveUniform', 'getUniformLocation', 'getExtension', 'getShaderInfoLog', 'getProgramInfoLog', 'finish', 'getBufferSubData',
  'clientWaitSync', 'getQueryParameter', 'getSyncParameter', 'getTexParameter', 'getFramebufferAttachmentParameter'];

/** A WebGL2 context that records every call; constants are distinct numbers, objects are fresh tokens. `max`: what
 *  every getParameter answers (MAX_TEXTURE_SIZE: 32768 on recent desktop GPUs, 16384 llvmpipe, 8192 SwiftShader). */
function fakeGL(log, max = 4096) {
  const constants = new Map(), constant = key => { if (!constants.has(key)) constants.set(key, 1000 + constants.size); return constants.get(key); };
  return new Proxy({}, { get(_, key) {
    if (typeof key !== 'string') return undefined;
    if (/^[A-Z0-9_]+$/.test(key)) return constant(key);
    return (...args) => {
      log.push(key);
      if (key === 'getParameter') return max;
      if (key === 'getShaderParameter' || key === 'getProgramParameter') return args[1] === constant('ACTIVE_UNIFORMS') ? 0 : true;
      if (key === 'checkFramebufferStatus') return constant('FRAMEBUFFER_COMPLETE');
      if (key === 'getExtension' || key === 'isContextLost') return key === 'getExtension' ? null : false;
      return { id: log.length };
    };
  } });
}
function stageHarness(max = 4096) {
  const loop = fakeLoop(), log = [], listeners = {};
  const canvas = { width: 0, height: 0, addEventListener(type, fn) { listeners[type] = fn; }, getContext: () => fakeGL(log, max),
    getBoundingClientRect: () => ({ width: 1000, height: 800 }) };
  const stage = loop.with(() => new Stage(canvas));
  return { loop, log, listeners, stage, canvas };
}
const frame = iter => ({ nx: 4, ny: 2, x0: 0, y0: 0, voxel: 1, iter, labels: new Uint32Array(8), consensus: null, age: new Uint32Array(8), layers: [] });
const synchronous = log => log.filter(name => SYNC.includes(name));

test('snapshots, image views and draws make no synchronous GL call; init makes them once', () => {
  const h = stageHarness();
  assert.ok(synchronous(h.log).length > 0, 'initialisation queries the limits and compiles');
  h.log.length = 0;
  const channels = [{ on: true, source: { width: 3, height: 2, data: new Uint16Array(6) } }, { on: false, source: null }];
  h.loop.with(() => {
    h.stage.tiled = true;
    for (let k = 0; k < 5; k++) h.stage.setFrame(frame(k));
    h.stage.setImage({ x0: 0, y0: 0, w: 4, h: 2 }, channels, true);
    h.stage.setImage({ x0: 1, y0: 0, w: 4, h: 2 }, channels, true);
    h.stage.panBy(5, 3); h.stage.zoomAt(10, 10, 1.5);
  });
  h.loop.run(2000);
  assert.ok(h.stage.drawCount >= 1);
  assert.ok(h.log.includes('texImage2D') && h.log.includes('drawArrays'));
  assert.deepEqual(synchronous(h.log), []);
  assert.equal(h.log.filter(n => n === 'generateMipmap').length, 0, 'view-sized images build no mipmap chain');
});

test('a lost context is restored with everything on screen; losses in quick succession halve the image size', () => {
  const h = stageHarness(), lost = [], restored = [];
  h.stage.onContextLost = n => lost.push(n); h.stage.onContextRestored = e => restored.push(e);
  const image = { on: true, source: { width: 4096, height: 2, data: new Uint16Array(8192) } };   // at the 4096-px cap
  h.loop.with(() => { h.stage.setFrame(frame(1)); h.stage.setImage({ x0: 0, y0: 0, w: 8, h: 4 }, [image]); });
  h.loop.run(100);
  h.loop.at(200, () => h.listeners.webglcontextlost({ preventDefault() {} }));
  h.loop.run(300);
  assert.equal(h.stage.gl, null); assert.deepEqual(lost, [1]); assert.equal(h.stage.imageLimit, undefined);
  assert.throws(() => h.loop.with(() => h.stage.setFrame(frame(2))), /graphics context is unavailable/);
  h.log.length = 0;
  h.loop.at(400, () => h.listeners.webglcontextrestored());
  h.loop.run(500);
  assert.deepEqual(restored, [null]); assert.ok(h.stage.gl);
  assert.equal(h.stage.frame.iter, 1); assert.equal(h.stage.image.channels[0], image);
  assert.ok(h.log.filter(n => n === 'texImage2D').length >= 3, 'the snapshot and the image were uploaded again');
  assert.ok(h.log.includes('drawArrays'), 'and drawn');
  h.loop.at(10000, () => h.listeners.webglcontextlost({ preventDefault() {} }));   // lost again within a minute
  h.loop.run(10100);
  assert.deepEqual(lost, [1, 2]); assert.equal(h.stage.imageLimit, 2048);
  h.loop.at(10200, () => h.listeners.webglcontextlost({ preventDefault() {} }));
  h.loop.run(10300);
  assert.equal(h.stage.imageLimit, 1024);
});

test('two quick losses halve the image size in use, not the GPU maximum: the next upload and request are smaller', () => {
  for (const max of [32768, 16384, 8192]) {
    for (const [side, view] of [[4096, true], [5647, false]]) {   // a whole-sample view (at most 4096 px); a 1.2 mm region
      const h = stageHarness(max), loader = Object.create(TiledLoader.prototype); loader.stage = h.stage;
      const lose = t => {
        h.loop.at(t, () => h.listeners.webglcontextlost({ preventDefault() {} })); h.loop.run(t + 5);
        h.loop.at(t + 10, () => h.listeners.webglcontextrestored()); h.loop.run(t + 20);
      };
      h.loop.with(() => h.stage.setImage({ x0: 0, y0: 0, w: 10, h: 10 }, [{ on: true, source: { width: side, height: 2, data: new Uint16Array(side * 2) } }], view));
      assert.equal(h.stage.imageSize, side);
      lose(1000); assert.equal(h.stage.imageLimit, undefined, 'one loss changes nothing');
      lose(5000);
      const half = Math.floor(side / 2);
      assert.equal(h.stage.imageLimit, half, `${max}: halved from the ${side} px in use`);
      assert.equal(h.stage.imageSize, fitSize(side, 2, half).width, 'the restored image is uploaded smaller at once');
      assert.equal(loader.imageCap(), Math.min(6144, half), 'and whole-sample views ask for no more');
      lose(9000); assert.equal(h.stage.imageLimit, Math.max(1024, Math.floor(half / 2)));
    }
  }
});

test('3-D shows view images with a mipmap chain: built on the switch for the image on screen, once, with no query', () => {
  const h = stageHarness(), count = name => h.log.filter(n => n === name).length;
  const channels = [{ on: true, source: { width: 64, height: 32, data: new Uint16Array(2048) } }, { on: false, source: null }];
  h.loop.with(() => h.stage.setImage({ x0: 0, y0: 0, w: 4, h: 2 }, channels, true));
  assert.equal(count('generateMipmap'), 0, '2-D: a view image is never shown much smaller than it was fetched');
  h.log.length = 0;
  h.loop.with(() => h.stage.orbit.switch(true, false));
  assert.equal(count('generateMipmap'), 1, 'the tilted ground plane is minified up to ~4x: the image on screen gets its chain');
  h.loop.with(() => { h.stage.orbit.switch(false, false); h.stage.orbit.switch(true, false); });
  assert.equal(count('generateMipmap'), 1, 'once per texture');
  const more = [channels[0], { on: true, source: { width: 64, height: 32, data: new Uint16Array(2048) } }];
  h.loop.with(() => h.stage.setImage({ x0: 0, y0: 0, w: 4, h: 2 }, more, true));   // a channel turned on in 3-D
  assert.equal(count('generateMipmap'), 2, 'the new channel is uploaded with its chain; the kept one is not rebuilt');
  h.loop.with(() => h.stage.setImage({ x0: 1, y0: 0, w: 4, h: 2 }, more, true));   // a pan: a new view image
  assert.equal(count('generateMipmap'), 4);
  assert.deepEqual(synchronous(h.log), []);
});

test('3-D render targets are kept for both the moving and the resting size: switching allocates and checks nothing', () => {
  const log = [], gl = fakeGL(log), targets = new VolumeTargets(gl, { texture: 4096, renderbuffer: 4096 });
  const count = name => log.filter(n => n === name).length;
  targets.resize(1000, 800);
  assert.equal(count('texImage2D'), 4); assert.equal(count('checkFramebufferStatus'), 3);
  targets.resize(600, 480);
  assert.equal(count('texImage2D'), 8); assert.equal(count('checkFramebufferStatus'), 3, 'checked once per context');
  const before = log.length;
  for (let k = 0; k < 10; k++) { targets.resize(1000, 800); targets.resize(600, 480); }
  assert.equal(log.length, before, 'motion starts and stops reuse both sets');
  assert.equal(targets.width, 600); assert.equal(targets.allocations, 2);
  targets.resize(1200, 900);   // the window was resized: the least recently used set goes
  assert.equal(targets.allocations, 3); assert.equal(count('deleteTexture'), 4); assert.equal(count('deleteFramebuffer'), 3);
  assert.equal(targets.sets.size, 2); assert.deepEqual([...targets.sets.keys()], ['600x480', '1200x900']);
  const broken = new VolumeTargets(new Proxy(fakeGL([]), { get: (t, k) => k === 'checkFramebufferStatus' ? () => 0 : t[k] }), { texture: 64, renderbuffer: 64 });
  assert.throws(() => broken.resize(10, 10), /Incomplete volume framebuffer/);
});

test('hidden prior and outline layers cost no canvas upload and no full-screen quad', () => {
  const log = [], gl = fakeGL(log), stage = { gl, imageProgram: {}, orbit: { active: false }, canvas: { width: 100, height: 80 }, camera: { x: 0, y: 0, zoom: 1 }, ratio: 1, info: {}, frame: null };
  const priors = Object.create(PriorLayers.prototype);
  Object.assign(priors, { stage, settings: { prior: { on: false, opacity: .6 }, nuclei: { on: false, color: '#fff', opacity: 1, width: 1 }, xenium: { on: false, color: '#fff', opacity: 1, width: 1 } },
    vectors: { cells: [{ poly: [[0, 0], [1, 0], [1, 1]] }], nuclei: [] }, prior: { width: 2 }, meta: { x0: 0, y0: 0, nx: 2, ny: 2, voxel: 1 }, vectorsVisible: true });
  assert.equal(priors.visible(), false);
  priors.draw();
  assert.deepEqual(log, [], 'nothing drawn, nothing uploaded');
  priors.settings.xenium.on = true; assert.equal(priors.visible(), true);
  priors.vectorsVisible = false; assert.equal(priors.visible(), false, 'outlines are hidden in a zoomed-out whole-sample view');
  priors.settings.prior.on = true; assert.equal(priors.visible(), true);
  stage.info = null; priors.settings.prior.on = false; priors.vectorsVisible = true; priors.settings.xenium.on = false;
  assert.equal(priors.visible(), true, 'the picker previews the chosen outlines');
});
