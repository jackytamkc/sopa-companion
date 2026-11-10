import test from 'node:test';
import assert from 'node:assert/strict';
import * as areasModule from '../src/qc-areas.mjs';

const { pointInOutline, outlineBounds, flyTarget, areaOutline } = areasModule;

// Where a flagged area of the mixing check lies: no area is drawn since 2026-10-22 (no flagged area on screen), so the
// page only finds its outline, the cells inside it and where to fly the camera for it.

const square = (x, y, s) => [[x, y], [x + s, y], [x + s, y + s], [x, y + s]];
const close = (a, b, eps = 1e-6) => Math.abs(a - b) <= eps;

test('inside an outline, never in a concave notch; the outline\'s box', () => {
  const ell = [[0, 0], [20, 0], [20, 10], [10, 10], [10, 30], [0, 30]];
  assert.equal(pointInOutline(5, 5, ell), true); assert.equal(pointInOutline(5, 25, ell), true);
  assert.equal(pointInOutline(15, 20, ell), false, 'the notch'); assert.equal(pointInOutline(-1, 5, ell), false);
  assert.deepEqual(outlineBounds(square(10, 20, 30)), [10, 20, 40, 50]);
});

test('the camera flies to an area at least 300 µm across, padded, centred on it', () => {
  const rect = (r, want) => assert.ok(Object.keys(want).every(k => close(r[k], want[k], 1e-9)), JSON.stringify(r));
  rect(flyTarget(square(1000, 2000, 100)), { x0: 1050 - 195, y0: 2050 - 195, w: 390, h: 390 });
  rect(flyTarget([[0, 0], [800, 0], [800, 400], [0, 400]]), { x0: -120, y0: -60, w: 1040, h: 520 });
  rect(flyTarget(square(0, 0, 10), { minSide: 50, pad: 0 }), { x0: -20, y0: -20, w: 50, h: 50 });
});

test("an area's outline: its own, else the result's area of the same letter; finite points only, none under 3", () => {
  const areas = [{ L: 'A', outline: square(0, 0, 100) }, { L: 'B', outline: square(500, 0, 50) }];
  assert.deepEqual(areaOutline({ L: 'Z', outline: square(9, 9, 1) }, areas), square(9, 9, 1), 'its own first');
  assert.deepEqual(areaOutline({ L: 'B' }, areas), square(500, 0, 50), 'a card names its area by its letter only');
  assert.equal(areaOutline({ L: 'C' }, areas), null, 'not in the result shown');
  assert.equal(areaOutline({ L: 'A' }, null), null); assert.equal(areaOutline(null, areas), null); assert.equal(areaOutline({}, areas), null);
  assert.deepEqual(areaOutline({ L: 'A', outline: [[0, 0], [NaN, 3], [10, 0], [10, 10], null] }), [[0, 0], [10, 0], [10, 10]], 'its finite points');
  assert.equal(areaOutline({ L: 'A', outline: [[0, 0], [1, 1]] }), null, 'fewer than 3 points');
});

test('nothing of an area on screen is left: no state of areas drawn, their motion, styles or chips', () => {
  assert.deepEqual(Object.keys(areasModule).sort(), ['areaOutline', 'flyTarget', 'outlineBounds', 'pointInOutline']);
});
