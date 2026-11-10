import test from 'node:test';
import assert from 'node:assert/strict';
import { visibleBox, covers, intersect, viewLevel, viewStride, expandedBox, planView, rasterBox, windowGeometry, frameWindow, alignWindow, viewKey, imagePixels, needsImage, shownChannels, RasterLRU } from '../src/tiled-view.mjs';

test('level depends on both visible sides and the current frame voxel size',()=>{
  assert.equal(viewLevel([100,-200,1700,1400],1600),'detail');
  assert.equal(viewLevel([0,0,1601,10],1600),'overview');
  assert.equal(viewLevel([0,0,10,1601],1600),'overview');
  const box=[0,0,2000,1800],bounds=[0,0,11000,11000];
  assert.equal(planView(box,bounds,2,1600,3).level,'detail');
  assert.equal(planView(box,bounds,1,1600,3).level,'overview');
  assert.equal(planView(box,bounds,2,1600,3,1,null,true).level,'overview'); // Fit also starts small tiled runs in overview.
});

test('camera geometry and stride account for device pixel ratio',()=>{
  const camera={x:100,y:-200,zoom:2};
  assert.deepEqual(visibleBox(camera,2000,1000),[-400,-450,600,50]);
  assert.equal(viewStride(8,2,1),4);
  assert.equal(viewStride(8,2,2),2);
  assert.equal(viewStride(.2,1,2),1);
  assert.equal(viewStride(5.9,2,1),2);
});

test('detail boxes retain a margin but fit the window and offset run bounds',()=>{
  const bounds=[100,-200,10100,9800];
  assert.deepEqual(expandedBox([100,0,1100,1000],bounds),[100,-200,1500,1200]);
  assert.deepEqual(expandedBox([200,0,1700,1500],bounds,1.4,1600),[150,-50,1750,1550]);
  assert.deepEqual(expandedBox([10000,9700,10500,10200],bounds),[9960,9660,10100,9800]);
  assert.equal(expandedBox([0,0,50,100],bounds),null);
  const p=planView([0,0,11000,11000],bounds,1,1600,10);
  assert.deepEqual(p.box,bounds);assert.equal(p.stride,10);
});

test('small pans reuse the box; leaving its margin, zooming deeply or changing voxel size replaces it',()=>{
  const bounds=[0,0,10000,10000],a=planView([4000,4000,5000,5000],bounds,1,1600,1);
  const b=planView([4010,4010,5010,5010],bounds,1,1600,1,1,a);
  assert.equal(b.box,a.box);
  const c=planView([4400,4400,5400,5400],bounds,1,1600,1,1,b);
  assert.notDeepEqual(c.box,b.box);assert.ok(covers(c.box,[4400,4400,5400,5400]));
  const d=planView([4490,4490,4510,4510],bounds,1,1600,.02,1,b);
  assert.equal(d.box[2]-d.box[0],28);
  assert.notEqual(planView([4010,4010,5010,5010],bounds,2,1600,1,1,a).box,a.box);
});

test('detail budget retains stride one for the entire maximum window',()=>{
  const p=planView([2000.1,2000.1,3600.1,3600.1],[0,0,10000,10000],1,1600,2);
  assert.equal(p.stride,1);assert.equal(p.maxPixels,1601**2);
  assert.ok(p.maxPixels>2400000);
  const near=planView([2000.1,2000.1,3500.1,3500.1],[0,0,10000,10000],1,1600,2);
  assert.ok(covers(near.box,[2000.1,2000.1,3500.1,3500.1]));
  assert.ok(Math.floor(near.box[2])-Math.floor(near.box[0])+1<=1600);
});

test('coverage intersects the sample and tolerates floating point rounding only',()=>{
  assert.ok(covers([0,0,100,100],[1,2,100.0000001,90]));
  assert.ok(!covers([0,0,100,100],[1,2,100.1,90]));
  assert.ok(!covers(null,[0,0,1,1]));assert.ok(!covers([0,0,1,1],null));
  assert.deepEqual(intersect([0,0,100,100],[10,-30,120,90]),[10,0,100,90]);
  assert.equal(intersect([0,0,1,1],[1,0,2,1]),null);
});

test('paused and stored windows use full-grid offsets, and old frames never borrow the current window',()=>{
  const f={seq:3,x0:100.5,y0:-200.5,voxel:2,window:{i0:10,j0:20,nx:100,ny:80}};
  assert.deepEqual(rasterBox(windowGeometry(f)),[120.5,-160.5,320.5,-.5]);
  const paused={seq:3,x0:100.5,y0:-200.5,voxel:2,i0:30,j0:40,nx:100,ny:80};
  assert.equal(frameWindow({window:paused},f).x0,160.5);
  assert.equal(frameWindow({window:{...paused,seq:4}},f).x0,120.5);
  assert.equal(frameWindow({window:paused},{...f,seq:2,window:null}),null);
});

test('window layers align rows at differing origins and pad uncovered view edges with zero',()=>{
  const view={x0:100.5,y0:-20,nx:4,ny:3,voxel:2};
  const window={x0:102.5,y0:-18,nx:2,ny:2,nz:2,voxel:2};
  const buffer=new Uint32Array([1,2,3,4,5,6,7,8]).buffer;
  const layers=alignWindow(view,window,buffer);
  assert.deepEqual([...layers[0]],[0,0,0,0,0,1,2,0,0,3,4,0]);
  assert.deepEqual([...layers[1]],[0,0,0,0,0,5,6,0,0,7,8,0]);
  const cut=alignWindow({...view,x0:104.5,y0:-16,nx:1,ny:1},window,buffer);
  assert.deepEqual(cut.map(l=>[...l]),[[4],[8]]);
  assert.throws(()=>alignWindow(view,window,buffer.slice(4)),/Truncated/);
  assert.throws(()=>alignWindow(view,{...window,voxel:1},buffer),/align/);
  assert.throws(()=>alignWindow(view,{...window,x0:103},buffer),/align/);
});

test('image coverage and factor-two resolution changes control refetching, capped at 4096 px',()=>{
  const box=[0,0,1000,800],image={box,px:1000};
  assert.equal(imagePixels(box,2),2000);assert.equal(imagePixels(box,8),6144);
  assert.equal(needsImage(image,[10,10,990,790],1.9),false);
  assert.equal(needsImage(image,[10,10,990,790],2),true);
  assert.equal(needsImage(image,[10,10,990,790],.5),true);
  assert.equal(needsImage(image,[-1,0,900,700],1),true);
  assert.equal(needsImage({box,px:4096},box,10),false);
  const full={box:[0,0,11000,11000],px:4096},detail=[5000,5000,5500,5500];
  assert.equal(needsImage(full,detail,2,[4900,4900,5600,5600]),true);
});

test('raster keys separate frames, fractional boxes and strides; LRU enforces bytes and entry count',()=>{
  const a=viewKey(1,[.1,2,3,4],1),b=viewKey(1,[.2,2,3,4],1);
  assert.notEqual(a,b);assert.notEqual(a,viewKey(2,[.1,2,3,4],1));assert.notEqual(a,viewKey(1,[.1,2,3,4],2));
  assert.equal(a,viewKey(1,[.1,2,3,4],1));
  const cache=new RasterLRU(10,3);
  cache.set(a,{bytes:4});cache.set(b,{bytes:4});cache.get(a);cache.set('c',{bytes:4});
  assert.equal(cache.get(b),undefined);assert.equal(cache.bytes,8);
  cache.set(a,{bytes:6});assert.equal(cache.bytes,10);
  cache.set('huge',{bytes:11});assert.equal(cache.get('huge'),undefined);assert.equal(cache.bytes,10);
  cache.clear();for(let i=0;i<4;i++)cache.set(i,{bytes:1});
  assert.equal(cache.entries.size,3);assert.equal(cache.get(0),undefined);assert.equal(cache.bytes,3);
});

test('an image request is the device-pixel size of its box, never the cap when the stage is smaller', () => {
  // A 1500 x 900 device-pixel stage showing 3000 x 1800 µm (0.5 px/µm); the fetched box adds a 1.5x margin.
  const camera = { x: 5000, y: 5000, zoom: .5 }, visible = visibleBox(camera, 1500, 900), bounds = [0, 0, 11000, 11000];
  const plan = planView(visible, bounds, 2, 1600, 1 / camera.zoom, 1, null, true);
  const px = imagePixels(plan.box, camera.zoom);
  assert.equal(px, Math.ceil(Math.max(plan.box[2] - plan.box[0], plan.box[3] - plan.box[1]) * camera.zoom));
  assert.ok(px <= 1.5 * 1500 + 1 && px < 4096, `${px} px`);
  assert.equal(imagePixels([0, 0, 1000, 800], 2, 1024), 1024, 'a smaller cap: the GPU texture size or a reduced limit');
  assert.equal(imagePixels([0, 0, 10, 10], .5), 64, 'the engine never sends less than 64 px');
  assert.equal(needsImage({ box: [0, 0, 1000, 800], px: 1024 }, [10, 10, 990, 790], 2, [0, 0, 1000, 800], 1024), false, 'the cap is not a reason to refetch');
});

test('only the channels shown are loaded', () => {
  assert.deepEqual(shownChannels([true, false, 1, 0]), [0, 2]);
  assert.deepEqual(shownChannels([false, false]), []);
  assert.deepEqual(shownChannels([]), []);
});
