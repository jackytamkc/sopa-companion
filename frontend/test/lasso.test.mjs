import test from 'node:test';
import assert from 'node:assert/strict';
import { LassoDraft, mergedMoves } from '../src/lasso.mjs';

// A pointer on a view like the stage's: world = (screen - offset) / scale, here 2 screen px per world unit, so points
// coming back in world units are half the screen positions. Gestures are given in screen px; the draft gets both, as a
// host forwards them. zoom() is the wheel (the world point under the pointer stays put), pan() a right-drag.
class Hand {
  constructor(draft = new LassoDraft(), scale = 2, offset = [0, 0]) { Object.assign(this, {draft, scale, offset}); }
  world([x, y]) { return [(x - this.offset[0]) / this.scale, (y - this.offset[1]) / this.scale]; }
  down(p) { return this.draft.down(this.world(p), p); }
  move(p, pressed = false) { return this.draft.move(this.world(p), p, pressed); }
  up(p) { return this.draft.up(this.world(p), p); }
  click(p) { this.down(p); return this.up(p); }
  drag(path) { this.down(path[0]); for (const p of path.slice(1, -1)) this.move(p, true); return this.up(path.at(-1)); }
  zoom(factor, [x, y]) {
    const [wx, wy] = this.world([x, y]);
    this.scale *= factor; this.offset = [x - wx * this.scale, y - wy * this.scale];
  }
  pan(dx, dy) { this.offset = [this.offset[0] + dx, this.offset[1] + dy]; }
}
const key = name => ({key: name});
const triangle = [[0, 0], [200, 0], [200, 200]];       // three corners, in screen px

test('clicks place corners where they were pressed, in world units; Enter closes the outline', () => {
  const hand = new Hand(), draft = hand.draft;
  assert.equal(draft.active, false);
  assert.equal(hand.down([20, 20]), 'changed');
  assert.equal(hand.up([20, 20]), 'changed');
  assert.equal(draft.active, true);
  hand.click([220, 20]); hand.click([220, 220]);
  assert.deepEqual(draft.points(), [[10, 10], [110, 10], [110, 110]]);
  assert.equal(draft.key(key('Enter')), 'close');
  assert.deepEqual(draft.points(), [[10, 10], [110, 10], [110, 110]], 'the host takes the points after the close');
  draft.reset();
  assert.equal(draft.active, false);
  assert.deepEqual(draft.points(), []);
});

test('a drag draws freehand: a point every screen px once past the drag threshold, ending at the release', () => {
  const hand = new Hand(), draft = hand.draft;
  hand.click([0, 0]);                                    // a corner first, so the stretch does not close on release
  hand.down([100, 100]);
  hand.move([101, 100], true); hand.move([102, 101], true);   // under 3 px from the press: still a click
  assert.deepEqual(draft.points(), [[0, 0], [50, 50]]);
  hand.move([103, 100], true);                           // 3 px: a drag, and its first new point
  hand.move([103.5, 100], true);                         // under 1 px from the last point: none
  hand.move([104, 100], true); hand.move([104, 110], true);
  assert.equal(hand.up([100, 110]), 'changed');
  assert.deepEqual(draft.points(), [[0, 0], [50, 50], [51.5, 50], [52, 50], [52, 55], [50, 55]]);
  // A press released 3 px or more away is a drag also with no move between.
  hand.down([200, 200]);
  assert.equal(hand.up([210, 200]), 'changed');
  assert.deepEqual(draft.points().slice(-2), [[100, 100], [105, 100]]);
});

test('jitter under the drag threshold is a click: one corner where the press was', () => {
  const hand = new Hand(), draft = hand.draft;
  hand.down([50, 50]); hand.move([51, 51], true); hand.move([52, 50.5], true);
  assert.equal(hand.up([51.5, 51.5]), 'changed');
  assert.deepEqual(draft.points(), [[25, 25]]);
  // Measured from the press, not from the last move: creeping 1 px at a time is a drag at 3 px.
  hand.down([80, 80]);
  for (const x of [81, 82, 83]) hand.move([x, 80], true);
  hand.up([83, 80]);
  assert.deepEqual(draft.points(), [[25, 25], [40, 40], [41.5, 40]]);
});

test('a lone freehand stroke closes on release, as the lasso did; a stretch after another piece does not', () => {
  const ring = [[0, 0], [100, 0], [100, 100], [0, 100]];
  let hand = new Hand();
  assert.equal(hand.drag(ring), 'close');
  assert.deepEqual(hand.draft.points(), [[0, 0], [50, 0], [50, 50], [0, 50]]);
  hand = new Hand();                                     // after a corner
  hand.click([300, 300]);
  assert.equal(hand.drag(ring), 'changed');
  assert.equal(hand.draft.points().length, 5);
  hand = new Hand();                                     // a stroke with too few points stays, to build on
  assert.equal(hand.drag([[0, 0], [10, 0]]), 'changed');
  assert.deepEqual(hand.draft.points(), [[0, 0], [5, 0]]);
  assert.equal(hand.drag([[10, 10], [0, 10]]), 'changed', 'a second stretch');
  assert.equal(hand.draft.key(key('Enter')), 'close');
  assert.deepEqual(hand.draft.points(), [[0, 0], [5, 0], [5, 5], [0, 5]]);
  hand = new Hand(new LassoDraft({minPoints: 4}));      // minPoints holds for this close too
  assert.equal(hand.drag([[0, 0], [100, 0], [100, 100]]), 'changed');
});

test('corners and freehand stretches mix in drawing order; a press on the first point closes the outline', () => {
  const hand = new Hand(), draft = hand.draft;
  hand.click([0, 0]);
  assert.equal(hand.drag([[100, 0], [150, 50], [100, 100]]), 'changed');
  hand.click([0, 100]);
  assert.deepEqual(draft.points(), [[0, 0], [50, 0], [75, 25], [50, 50], [0, 50]]);
  hand.move([3, 4]);
  assert.equal(hand.down([3, 4]), 'close');              // 5 px from the first point
  assert.deepEqual(draft.points(), [[0, 0], [50, 0], [75, 25], [50, 50], [0, 50]], 'the closing press adds nothing');
  draft.reset();
  assert.equal(hand.up([3, 4]), 'none', 'its release finds no press');
  assert.equal(draft.active, false);
});

test('closing needs minPoints points every way; the first point is closeRadius screen px wide', () => {
  let hand = new Hand(), draft = hand.draft;
  hand.click([0, 0]); hand.click([200, 0]);
  assert.equal(draft.key(key('Enter')), 'few');
  assert.equal(draft.points().length, 2);
  // Two points: a click on the first places nothing (a corner there would close a sliver at the next press).
  assert.equal(hand.down([0, 0]), 'changed');
  assert.equal(hand.up([0, 0]), 'few');
  assert.deepEqual(draft.points(), [[0, 0], [100, 0]]);
  hand.click([200, 200]);
  hand.move([8, 0]);
  assert.equal(hand.down([8, 0]), 'close', '8 px away');
  draft.reset();
  for (const p of triangle) hand.click(p);
  assert.equal(hand.down([9, 0]), 'changed', '9 px away');
  hand = new Hand(new LassoDraft({minPoints: 4, closeRadius: 20})), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  assert.equal(draft.key(key('Enter')), 'few');
  assert.equal(hand.down([15, 0]), 'changed', 'three points of four');
  assert.equal(hand.up([15, 0]), 'few');
  hand.click([0, 200]);
  assert.equal(hand.down([15, 0]), 'close', '15 px away, within 20');
  // With no outline these are the page's.
  hand = new Hand(), draft = hand.draft;
  assert.equal(draft.key(key('Enter')), 'none');
  assert.equal(draft.doubleClick(), 'none');
});

test('a corner on the first point of an outline too small to close never makes a sliver', () => {
  // Two corners, then a double-click (or two clicks) on the first point: before, the first click placed a corner there
  // and the second closed [A, B, A'], an outline with no area.
  const hand = new Hand(), draft = hand.draft;
  hand.click([0, 0]); hand.click([200, 0]);
  assert.equal(hand.click([1, 1]), 'few');
  assert.equal(hand.click([1, 1]), 'few');
  assert.equal(draft.doubleClick(), 'few');
  assert.deepEqual(draft.points(), [[0, 0], [100, 0]]);
  assert.equal(draft.key(key('Enter')), 'few');
  // A drag from the first point still draws.
  assert.equal(hand.drag([[0, 0], [0, 100], [100, 200]]), 'changed');
  assert.equal(draft.points().length, 5);
});

test('a first click that slips is a short stretch to build on, not an outline closed on its release', () => {
  // A few px of drift makes a stretch of three points; on its own it used to close at once, as a sliver.
  const hand = new Hand(), draft = hand.draft;
  assert.equal(hand.drag([[100, 100], [104, 100], [106, 102]]), 'changed');
  assert.equal(draft.active, true);
  hand.click([300, 100]); hand.click([300, 300]);
  assert.equal(draft.key(key('Enter')), 'close');
  // A stroke that goes 3 × closeRadius (24 px) from its press is a lasso drawn in one go.
  const lone = new Hand();
  assert.equal(lone.drag([[0, 0], [10, 0], [20, 5], [22, 6]]), 'changed', '22.8 px from the press');
  lone.draft.reset();
  assert.equal(lone.drag([[0, 0], [10, 0], [20, 5], [24, 8]]), 'close', '25.3 px from the press');
});

test('a double-click closes the outline, dropping the corner its second click placed', () => {
  let hand = new Hand(), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  hand.click([0, 200]); hand.click([1, 201]);            // the browser's two clicks, then its dblclick
  assert.equal(draft.doubleClick(), 'close');
  assert.deepEqual(draft.points(), [[0, 0], [100, 0], [100, 100], [0, 100]]);
  // Too few points: the repeat goes all the same, and the outline goes on ('few': the host says it needs three).
  hand = new Hand(), draft = hand.draft;
  hand.click([0, 0]); hand.click([200, 0]); hand.click([201, 0]);
  assert.equal(draft.doubleClick(), 'few');
  assert.deepEqual(draft.points(), [[0, 0], [100, 0]]);
  // A double-click on the first point: its first click closes, its second starts an outline that the double-click
  // takes away again (also when that second press wobbled into a short stretch).
  for (const wobble of [0, 4]) {
    hand = new Hand(), draft = hand.draft;
    for (const p of triangle) hand.click(p);
    assert.equal(hand.down([1, 1]), 'close');
    draft.reset();
    assert.equal(hand.up([1, 1]), 'none');
    hand.down([1, 1]); hand.up([1 + wobble, 1]);
    assert.equal(draft.active, true);
    assert.equal(draft.doubleClick(), 'changed');
    assert.equal(draft.active, false, `wobble ${wobble}`);
    assert.equal(draft.doubleClick(), 'none');
  }
  // A second press that wobbled into a short stretch is the double-click's all the same: one corner, where it began.
  hand = new Hand(), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  hand.click([0, 200]); hand.down([0, 200]); hand.up([4, 200]);
  assert.equal(draft.doubleClick(), 'close');
  assert.deepEqual(draft.points(), [[0, 0], [100, 0], [100, 100], [0, 100]]);
  // A stretch drawn before the double-click stays.
  hand = new Hand(), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  hand.drag([[0, 200], [0, 240]]);
  hand.click([60, 260]); hand.click([61, 260]);
  assert.equal(draft.doubleClick(), 'close');
  assert.deepEqual(draft.points().slice(3), [[0, 100], [0, 120], [30, 130]]);
});

test('a double-click with no outline being drawn is the page\'s, and leaves nothing behind', () => {
  // The habit from the rest of the app: a double-click fits the view. Its two clicks began an outline, which goes.
  for (const wobble of [0, 2, 5]) {
    const hand = new Hand(), draft = hand.draft;
    hand.down([50, 50]); hand.up([50 + wobble, 50]); hand.click([51, 50]);
    assert.equal(draft.doubleClick(), 'none', `wobble ${wobble}`);
    assert.equal(draft.active, false);
  }
  // One whose second press closed the outline (its first landed 9 px from the first point and placed a corner) has
  // done its work: no fit of the view, and the corner stays in the closed outline.
  const hand = new Hand(), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  hand.click([9, 0]);
  assert.equal(hand.down([6, 0]), 'close');
  const closed = draft.points();
  draft.reset();
  assert.equal(hand.up([6, 0]), 'none');
  assert.equal(draft.doubleClick(), 'changed');
  assert.equal(draft.active, false);
  assert.deepEqual(closed, [[0, 0], [100, 0], [100, 100], [4.5, 0]]);
});

test('Backspace takes back the last corner, or the whole last freehand stretch', () => {
  const hand = new Hand(), draft = hand.draft;
  hand.click([0, 0]);
  hand.drag([[100, 0], [120, 40], [100, 80]]);
  hand.click([0, 80]);
  assert.deepEqual(draft.points(), [[0, 0], [50, 0], [60, 20], [50, 40], [0, 40]]);
  assert.equal(draft.key(key('Backspace')), 'changed');
  assert.deepEqual(draft.points(), [[0, 0], [50, 0], [60, 20], [50, 40]]);
  assert.equal(draft.key(key('Backspace')), 'changed');
  assert.deepEqual(draft.points(), [[0, 0]]);
  assert.equal(draft.key(key('Backspace')), 'changed');
  assert.equal(draft.active, false);
  assert.equal(draft.key(key('Backspace')), 'none');
  // The stretch being drawn goes too, and the rest of that press draws nothing.
  hand.click([0, 0]);
  hand.down([100, 0]); hand.move([120, 40], true);
  assert.equal(draft.points().length, 3);
  assert.equal(draft.key(key('Backspace')), 'changed');
  hand.move([140, 60], true);
  assert.equal(hand.up([140, 60]), 'none');
  assert.deepEqual(draft.points(), [[0, 0]]);
});

test('Escape gives up an outline in progress and is left to the page otherwise', () => {
  const hand = new Hand(), draft = hand.draft;
  assert.equal(draft.key(key('Escape')), 'none');
  hand.click([0, 0]);
  for (const name of ['a', 'Delete', 'Tab']) assert.equal(draft.key(key(name)), 'none', name);
  assert.equal(draft.points().length, 1);
  assert.equal(draft.key(key('Escape')), 'cancel');
  draft.reset();
  assert.equal(draft.active, false);
  assert.equal(draft.key(key('Escape')), 'none');
  hand.down([0, 0]); hand.move([50, 0], true);           // also in the middle of a drag
  assert.equal(draft.key(key('Escape')), 'cancel');
  draft.reset();
  assert.equal(hand.up([60, 0]), 'none');
  assert.deepEqual(draft.points(), []);
});

test('preview: the outline, a guide to the pointer, and the first point, closable within closeRadius', () => {
  const hand = new Hand(), draft = hand.draft;
  assert.equal(hand.move([10, 10]), 'none', 'no outline: nothing to draw');
  assert.deepEqual(draft.preview(), {points: [], guide: null, closable: false, first: null});
  hand.click([0, 0]); hand.click([200, 0]);
  assert.equal(hand.move([4, 2]), 'changed');
  assert.deepEqual(draft.preview(), {points: [[0, 0], [100, 0]], guide: [2, 1], closable: false, first: [0, 0]},
    'two points cannot close');
  hand.click([200, 200]);
  hand.move([6, 0]);
  assert.deepEqual(draft.preview(), {points: [[0, 0], [100, 0], [100, 100]], guide: [3, 0], closable: true,
    first: [0, 0]});
  hand.move([10, 0]);
  assert.equal(draft.preview().closable, false);
  hand.down([60, 60]); hand.move([30, 30], true); hand.move([2, 2], true);
  const drawing = draft.preview();
  assert.deepEqual(drawing.points.slice(-3), [[30, 30], [15, 15], [1, 1]], 'the stretch being drawn is shown');
  assert.equal(drawing.closable, false, 'not while the button is down');
  hand.up([2, 2]);
  assert.equal(draft.preview().closable, true, 'a press here would close');
});

test('the first point closes where it is on screen now, after a zoom or pan between corners', () => {
  // The wheel, at the last corner: the first point moves from (0, 0) to (100, 100) on screen.
  let hand = new Hand(), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  hand.zoom(0.5, [200, 200]);
  hand.move([150, 150]); hand.move([2, 2]);
  assert.equal(draft.preview().closable, false, 'where the first point was');
  hand.move([104, 103]);
  assert.equal(draft.preview().closable, true);
  assert.equal(hand.down([104, 103]), 'close');
  // A right-drag pan carries the image with the pointer: resting on the first point, the pointer stays on it, here
  // from (4, 2) to (104, 52). The host forwards the pan's moves, whose world point moves only by rounding (here along
  // the pan, so only its size tells it from a real move), or it does not, and the pointer is nudged afterwards.
  for (const forwarded of [true, false]) {
    hand = new Hand(), draft = hand.draft;
    for (const p of triangle) hand.click(p);
    hand.move([4, 2]);
    for (let step = 1; step <= 5; step++) {
      hand.pan(20, 10);
      if (forwarded) draft.move([2 + 2e-12 * step, 1 + 1e-12 * step], [4 + 20 * step, 2 + 10 * step], false);
    }
    const at = forwarded ? [104, 52] : [106, 51];
    if (!forwarded) hand.move(at);
    assert.equal(draft.preview().closable, true, `forwarded ${forwarded}`);
    assert.equal(hand.down(at), 'close', `forwarded ${forwarded}`);
  }
  // Any other change of view (a fit): a move or two later the first point is measured as the view is now.
  hand = new Hand(), draft = hand.draft;
  for (const p of triangle) hand.click(p);
  hand.move([300, 300]);
  hand.scale = 1; hand.offset = [100, 100];
  hand.move([150, 150]); hand.move([120, 130]);
  assert.equal(hand.down([106, 105]), 'close', '7.8 px from the first point at (100, 100)');
});

test('a pan the host does not pass on teaches no scale (viewChanged)', () => {
  // The pointer rests at (95, 100), a right-drag pans 200 px exactly along x, and the next move goes 7 px further the
  // same way. That pair spans the pan: its world step is 207 px of screen against a 7 px world step, a wrong scale.
  const run = told => {
    const hand = new Hand(new LassoDraft(), 1), draft = hand.draft;
    hand.click([100, 100]); hand.click([300, 100]); hand.click([300, 300]);
    hand.move([95, 100]);
    hand.pan(200, 0);
    if (told) draft.viewChanged();
    hand.move([302, 100]);                               // the first point is now at (300, 100) on screen
    return [draft.preview().closable, hand.down([302, 100])];
  };
  assert.deepEqual(run(true), [true, 'close']);
  assert.notDeepEqual(run(false), [true, 'close'], 'without it the draft learns a wrong scale');
});

test('before a move has taught the view, the first point is where it was placed on screen', () => {
  let hand = new Hand(new LassoDraft({minPoints: 1})), draft = hand.draft;
  hand.click([40, 40]);
  assert.equal(draft.preview().closable, true);
  assert.equal(hand.down([40.5, 40]), 'close');
  // A press the host clamped to the edge of its extent teaches nothing (its world point did not follow the screen).
  hand = new Hand(new LassoDraft({minPoints: 1})), draft = hand.draft;
  hand.click([40, 40]);
  assert.equal(draft.down([30, 20], [100, 60]), 'changed', '63 px away on screen');
});

test('a press whose release never came ends where it is; a release or double-click with no press does nothing', () => {
  const hand = new Hand(), draft = hand.draft;
  assert.equal(hand.up([5, 5]), 'none');
  assert.equal(draft.doubleClick(), 'none');
  hand.down([0, 0]); hand.move([40, 0], true);
  hand.down([100, 100]); hand.up([100, 100]);
  assert.deepEqual(draft.points(), [[0, 0], [20, 0], [50, 50]]);
});

test('mergedMoves: the moves a busy frame merged into one event, oldest first, placed by their client offset from it', () => {
  const e = {offsetX: 50, offsetY: 60, clientX: 150, clientY: 260,
    getCoalescedEvents: () => [{clientX: 110, clientY: 230}, {clientX: 130, clientY: 250}, {clientX: 150, clientY: 260}]};
  assert.deepEqual(mergedMoves(e), [{offsetX: 10, offsetY: 30}, {offsetX: 30, offsetY: 50}, {offsetX: 50, offsetY: 60}]);
  const plain = {offsetX: 5, offsetY: 6};
  assert.deepEqual(mergedMoves(plain), [plain], 'an event without merged moves: itself');
  const one = {...plain, clientX: 1, clientY: 2, getCoalescedEvents: () => [{clientX: 1, clientY: 2}]};
  assert.equal(mergedMoves(one)[0], one, 'one merged move is the event itself');
});
