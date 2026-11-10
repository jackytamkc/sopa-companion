import test from 'node:test';
import assert from 'node:assert/strict';
import { makeLUT, COLORMAPS, downsample16, projectPrior } from '../src/image-layers.mjs';
test('all requested LUTs provide 256 RGB texels; primary ramps preserve endpoints',()=>{
  for(const name of COLORMAPS)assert.equal(makeLUT(name).length,768);
  assert.deepEqual([...makeLUT('gray').slice(-3)],[255,255,255]);
  assert.deepEqual([...makeLUT('cyan').slice(-3)],[0,255,255]);
  assert.deepEqual([...makeLUT('red').slice(0,3)],[0,0,0]);
});
test('CPU resize preserves uint16 intensities and rectangular geometry',()=>{
  const data=new Uint16Array([0,1000,2000,3000,4000,50000,60000,65535]);
  assert.equal(downsample16(data,4,2,4,2),data);
  assert.deepEqual([...downsample16(data,4,2,2,1)],[50000,65535]);
});
test('prior projection selects the strongest label; Z slicing retains its own prior',()=>{
  const meta={nx:2,ny:1,nz:2};
  const data=new Uint32Array([1,2,3,4,500,800,800,500]).buffer;
  const p=projectPrior(meta,data);
  assert.deepEqual([...p.labels],[3,2]);assert.deepEqual([...p.strength],[800,800]);
  assert.deepEqual([...projectPrior(meta,data,1).labels],[3,4]);
  assert.throws(()=>projectPrior(meta,data.slice(4)),/Truncated/);
});
