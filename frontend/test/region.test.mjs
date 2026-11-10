import test from 'node:test';
import assert from 'node:assert/strict';
import { normalizeBox, clampBox, clampPoint, rectangleFromPoints, polygonArea, polygonBBox, pointInPolygon,
  simplify, simplifyPolygon, regionBBox, regionArea, estimatedTranscripts, modeForBBox, boxInPolygon } from '../src/region.mjs';

const extent = [-400, 120, 1600, 1120];
test('rectangles normalize all drag directions and clip to an offset extent', () => {
  assert.deepEqual(normalizeBox([50, 600, -600, -20]), [-600, -20, 50, 600]);
  assert.deepEqual(clampBox([50, 600, -600, -20], extent), [-400, 120, 50, 600]);
  assert.deepEqual(clampPoint([-500, 1400], extent), [-400, 1120]);
  assert.deepEqual(rectangleFromPoints([100, 500], [-200, 300], extent), [-200, 300, 100, 500]);
});
test('moving against any edge preserves rectangle dimensions; oversized rectangles fit the extent', () => {
  assert.deepEqual(clampBox([1550, 1000, 1850, 1200], extent, true), [1300, 920, 1600, 1120]);
  assert.deepEqual(clampBox([-500, 20, -200, 220], extent, true), [-400, 120, -100, 320]);
  assert.deepEqual(clampBox([-900, -900, 9000, 9000], extent, true), extent);
});
test('Shift rectangles stay square, including reversed drags and extent edges', () => {
  assert.deepEqual(rectangleFromPoints([0, 500], [-100, 300], extent, true), [-200, 300, 0, 500]);
  assert.deepEqual(rectangleFromPoints([1500, 1000], [2000, 1100], extent, true), [1500, 1000, 1600, 1100]);
  assert.deepEqual(rectangleFromPoints([-350, 200], [-600, -100], extent, true), [-400, 150, -350, 200]);
});
test('shoelace area is absolute, translation stable, and accepts closed or open rings', () => {
  const triangle = [[-200, 120], [100, 120], [100, 320]];
  assert.equal(polygonArea(triangle), 30000);
  assert.equal(polygonArea(triangle.toReversed()), 30000);
  assert.equal(polygonArea([...triangle, triangle[0]]), 30000);
  assert.equal(polygonArea(triangle.map(([x,y]) => [x+1e10,y+1e10])), 30000);
  assert.equal(polygonArea([]), 0);
  assert.deepEqual(polygonBBox(triangle), [-200,120,100,320]);
  assert.equal(polygonBBox([]), null);
});
test('point containment uses even-odd fill for concave and self-intersecting rings, including edges', () => {
  const concave = [[0,0],[100,0],[100,30],[30,30],[30,100],[0,100]];
  assert.equal(pointInPolygon([20,80], concave), true);
  assert.equal(pointInPolygon([80,80], concave), false);
  assert.equal(pointInPolygon([30,50], concave), true);
  const bow = [[0,0],[100,100],[0,100],[100,0]];
  assert.equal(pointInPolygon([50,20], bow), true);
  assert.equal(pointInPolygon([10,50], bow), false);
  assert.equal(pointInPolygon([50,50], bow), true);
  assert.equal(pointInPolygon([0,0], []), false);
});
test('Douglas–Peucker drops subpixel noise while preserving significant bends and endpoints', () => {
  const line = [[0,0],[1,.1],[2,0],[3,2],[4,0],[5,0]];
  assert.deepEqual(simplify(line,.2), [[0,0],[2,0],[3,2],[4,0],[5,0]]);
  assert.deepEqual(simplify(line,3), [[0,0],[5,0]]);
  assert.equal(line.length,6);
});
test('closed lasso simplification keeps its shape and caps complex strokes at 2000 vertices', () => {
  const ring = [[0,0],[50,.1],[100,0],[100,50],[100,100],[50,100],[0,100],[0,50],[0,0]];
  assert.deepEqual(simplifyPolygon(ring,1.5), [[0,0],[100,0],[100,100],[0,100]]);
  const noisy = Array.from({length:6000},(_,i) => {
    const a=i/6000*Math.PI*2,r=100+(i%2)*20; return [r*Math.cos(a),r*Math.sin(a)];
  });
  const result = simplifyPolygon(noisy,.001);
  assert.ok(result.length>=3 && result.length<=2000);
  assert.ok(polygonArea(result)>20000);
  assert.equal(noisy.length,6000);
  assert.deepEqual(simplifyPolygon([[1,2],[1,2],[1,2]]), [[1,2]]);
});
test('all region types derive world bounds and actual area, not polygon bbox area', () => {
  const polygon = {kind:'polygon',points:[[0,120],[1000,120],[1000,1120]]};
  assert.deepEqual(regionBBox(polygon,extent), [0,120,1000,1120]);
  assert.equal(regionArea(polygon,extent),500000);
  assert.equal(estimatedTranscripts(4000000,polygon,extent),1000000);
  assert.deepEqual(regionBBox({kind:'full'},extent),extent);
  assert.equal(regionArea({kind:'full'},extent),2000000);
  assert.equal(estimatedTranscripts(4000000,{kind:'full'},extent),4000000);
  assert.equal(regionArea({kind:'box',box:[100,220,0,120]},extent),10000);
  assert.equal(regionArea(null,extent),0);
  assert.equal(estimatedTranscripts(100,{kind:'full'},[0,0,0,10]),0);
});
test('detail threshold is inclusive and compares each bbox side', () => {
  assert.equal(modeForBBox([100,-300,1300,900],1200),'detail');
  assert.equal(modeForBBox([100,-300,1301,900],1200),'tiled');
  assert.equal(modeForBBox([100,-300,1300,901],1200),'tiled');
  assert.equal(modeForBBox([100,-300,600,200],400),'tiled');
});


test('boxInPolygon: the whole box inside the lasso, a pixel of slack at its edge, concave notches caught', () => {
  const triangle = [[0, 0], [300, 0], [0, 300]];
  assert.equal(boxInPolygon([0, 0, 64, 64], triangle), true);
  assert.equal(boxInPolygon([150, 60, 214, 124], triangle), false);     // its centre is inside, a corner is not
  assert.equal(boxInPolygon([0, 0, 150.5, 150.5], triangle), true);     // corner on the slanted edge, within the slack
  assert.equal(boxInPolygon([0, 0, 152, 152], triangle), false);        // 1.4 units past it
  const notched = [[0, 0], [190, 0], [200, 100], [210, 0], [400, 0], [400, 400], [0, 400]];
  assert.equal(boxInPolygon([150, 50, 250, 150], notched), false);      // all four corners inside, the notch cuts in
  assert.equal(boxInPolygon([150, 120, 250, 220], notched), true);
  assert.equal(boxInPolygon([0, 0, 10, 10], [[0, 0], [1, 1]]), false);
  assert.equal(boxInPolygon([5, 5, 5, 9], triangle), false);
});
