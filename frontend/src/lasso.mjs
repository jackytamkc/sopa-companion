// An area drawn on the image as corners, freehand, or both, whichever feels natural. A click places a corner and a drag
// draws a freehand stretch, in any mix; a press on the first point, a double-click or Enter closes the area. One
// freehand stroke on its own closes when it is released, as the lasso always did. Backspace takes back the last corner
// or the whole last stretch, Escape gives the area up. A close asked for with fewer than minPoints points closes
// nothing, and a click on the first point then places nothing either: a corner on top of the first point would close a
// sliver. A double-click with no outline being drawn is the page's (it fits the view), and leaves nothing behind.
//
// LassoDraft only keeps that state: it listens to nothing and draws nothing. The host forwards presses of the main
// button, moves, releases, double-clicks and keys (right and middle buttons stay the page's, to pan) and draws
// preview(). Points are in the host's world units, clamped by the host; screen positions (px) decide what is a click, a
// drag and a close. Each call answers 'close' (take points(), then reset()), 'cancel' (reset()), 'changed' (draw again),
// 'few' (a close asked for too early: say that it needs minPoints points, and draw again) or 'none' (nothing happened;
// a key or a double-click is left to the page).

const distance = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const copy = p => [p[0], p[1]];
const size = draft => draft.pieces.reduce((n, piece) => n + piece.points.length, 0);

/** The pointer is at `world`, `screen`. On the way the view's scale (screen px per world unit) is learnt: two events
 *  in a row under the same view give it exactly, |Δscreen| = scale × |Δworld| on each axis. A zoom or pan between
 *  them, or a point the host clamped, breaks that, and such a pair teaches nothing (a pan carries the world along with
 *  the pointer, so Δworld is only rounding). With it, the first point is measured where it is on screen now. */
function track(draft, world, screen) {
  const last = draft.pointer;
  draft.pointer = {world: copy(world), screen: copy(screen)};
  if (!last || last.stale) return;
  const sx = Math.abs(screen[0] - last.screen[0]), sy = Math.abs(screen[1] - last.screen[1]);
  const wx = Math.abs(world[0] - last.world[0]), wy = Math.abs(world[1] - last.world[1]);
  const ds = Math.hypot(sx, sy), dw = Math.hypot(wx, wy);
  const rounding = 1e-9 * (Math.abs(world[0]) + Math.abs(world[1]) + 1);
  if (ds >= 1 && dw > rounding && Math.abs(sx * wy - sy * wx) <= 1e-3 * ds * dw) draft.scale = ds / dw;
}

/** Whether `world`, `screen` is on the outline's first point: within closeRadius on screen as the view is now, or,
 *  before any move has taught the scale, of where the first point was placed. */
function onFirst(draft, world, screen) {
  const first = draft.pieces[0];
  if (!first) return false;
  const away = draft.scale ? distance(world, first.points[0]) * draft.scale : distance(screen, first.start);
  return away <= draft.closeRadius;
}
const closable = (draft, world, screen) => size(draft) >= draft.minPoints && onFirst(draft, world, screen);

/** The pointer moved with the button down: under dragThreshold from the press it is still a click; once past it, the
 *  press is a freehand stretch that gains a point every screen px. `reach` is how far from the press it has been. */
function extend(press, world, screen, threshold) {
  press.reach = Math.max(press.reach, distance(screen, press.start));
  if (!press.stroke && press.reach < threshold) return;
  press.stroke = true;
  if (distance(screen, press.end) < 1) return;
  press.points.push(copy(world));
  press.end = copy(screen);
}

export class LassoDraft {
  constructor({closeRadius = 8, dragThreshold = 3, minPoints = 3} = {}) {
    Object.assign(this, {closeRadius, dragThreshold, minPoints});   // screen px, screen px, points
    this.pointer = null;   // the last event's {world, screen}: the guide's end
    this.scale = 0;        // screen px per world unit, learnt from the pointer (track); 0 until known
    // Presses are numbered, also across outlines, so a double-click can tell what its own two presses did: the number
    // of the last press, and of the last press that closed an outline.
    this.presses = 0;
    this.closedBy = -Infinity;
    this.reset();
  }
  /** Forget the outline: after 'close' once its points are taken, and after 'cancel'. The view's scale stays. */
  reset() {
    // The outline in drawing order: corners (one point) and freehand stretches. Each piece keeps the screen positions
    // of its first and last point as they were placed (a drag is measured from the first, the next freehand point and a
    // double-click's second corner from the last) and the number of the press that placed it (n).
    this.pieces = [];
    this.press = null;     // the piece the button is down on
  }
  /** An outline is in progress. */
  get active() { return this.pieces.length > 0; }
  /** The outline's points so far, in world units (a new array; the points themselves are the draft's). */
  points() { return this.pieces.flatMap(piece => piece.points); }

  /** A press of the main button. On the first point of an outline with minPoints points it closes the outline and adds
   *  nothing; elsewhere it starts a piece, a corner unless it moves (move). On the first point of an outline with
   *  fewer, a click places nothing (up: 'few') and a drag from there draws as anywhere else. */
  down(world, screen) {
    track(this, world, screen);
    this.press = null;                                   // a press whose release never came ends where it is
    const n = ++this.presses;
    if (closable(this, world, screen)) { this.closedBy = n; return 'close'; }
    this.press = {points: [copy(world)], stroke: false, start: copy(screen), end: copy(screen), reach: 0, n,
                  early: onFirst(this, world, screen)};
    this.pieces.push(this.press);
    return 'changed';
  }
  /** The pointer moved; `pressed`: with the main button still down since down(). Otherwise only the guide follows. */
  move(world, screen, pressed) {
    track(this, world, screen);
    if (pressed && this.press) extend(this.press, world, screen, this.dragThreshold);
    return this.active ? 'changed' : 'none';
  }
  /** The press ends: a click leaves its corner where it was pressed, a drag its stretch up to here. An outline that is
   *  one freehand stretch and nothing else closes here, once the stroke has gone 3 × closeRadius from its press (a
   *  click that slipped is no outline: it stays, a corner to build on). A click on the first point of an outline too
   *  small to close is taken back: 'few' once the outline has two points or more (with one, it is the second click of
   *  a double-click as often as not, and says nothing). */
  up(world, screen) {
    track(this, world, screen);
    const press = this.press;
    if (!press) return 'none';
    extend(press, world, screen, this.dragThreshold);   // also a press that moved with no move() between
    this.press = null;
    if (press.early && !press.stroke) {
      this.pieces.pop();                                 // the press is always the last piece
      return size(this) > 1 ? 'few' : 'changed';
    }
    if (press.stroke && this.pieces.length === 1 && press.points.length >= this.minPoints &&
        press.reach >= 3 * this.closeRadius) { this.closedBy = press.n; return 'close'; }
    return 'changed';
  }
  /** A double-click, called as the browser fires it: right after its two presses (and releases). When one of them closed
   *  the outline, its work is done: what the second placed after the close goes, and the page does not take the
   *  double-click either ('changed'). When it began the outline, none was being drawn: what it placed goes and the
   *  double-click is the page's ('none', e.g. to fit the view). Otherwise its two presses make one corner, where the
   *  first was pressed (also when one wobbled into a short stretch), and the outline closes ('few' with too few points). */
  doubleClick() {
    const first = this.presses - 1, mine = piece => piece.n >= first;
    this.press = null;
    if (this.closedBy >= first) {
      this.closedBy = -Infinity;
      this.pieces = this.pieces.filter(piece => !mine(piece));
      return 'changed';
    }
    if (this.pieces.every(mine)) { this.reset(); return 'none'; }
    const placed = this.pieces.find(mine);
    this.pieces = this.pieces.filter(piece => !mine(piece));
    if (placed) this.pieces.push({...placed, points: [placed.points[0]], stroke: false, end: placed.start});
    return size(this) >= this.minPoints ? 'close' : 'few';
  }
  /** The view moved under the pointer without the host passing that move on (a pan): the next pair of events spans two
   *  views and teaches no scale. */
  viewChanged() { if (this.pointer) this.pointer.stale = true; }
  /** Enter closes the outline once it has minPoints points ('few' before); Backspace takes back the last corner or the
   *  whole last freehand stretch (also one being drawn); Escape gives the outline up ('cancel'). Other keys, and these
   *  with no outline, are the page's ('none'). */
  key(event) {
    const key = event?.key;
    if (!this.active) return 'none';
    if (key === 'Enter') return size(this) >= this.minPoints ? 'close' : 'few';
    if (key === 'Escape') return 'cancel';
    if (key !== 'Backspace') return 'none';
    if (this.pieces.pop() === this.press) this.press = null;   // the rest of that press draws nothing
    return 'changed';
  }
  /** What the host draws: the outline placed so far (`points`), the pointer (`guide`, in world units; null with no
   *  outline) for a dashed line from the last point, and the `first` point, `closable` when a press at the pointer
   *  would close the outline (highlight it then). */
  preview() {
    const points = this.points(), pointer = this.active ? this.pointer : null;
    const close = !!pointer && !this.press && closable(this, pointer.world, pointer.screen);
    return {points, guide: pointer ? copy(pointer.world) : null, closable: close, first: points[0] || null};
  }
}

/** Where a pointer move went, oldest first, as {offsetX, offsetY}: each move the browser merged into this event while
 *  the page was busy (getCoalescedEvents), then the event's own. A host passes them all to move() while the button is
 *  down, or a quick drag cuts its corners. A merged move is placed by its client offset from the event, which needs
 *  nothing else of it. */
export function mergedMoves(e) {
  const merged = typeof e.getCoalescedEvents === 'function' ? e.getCoalescedEvents() : [];
  if (merged.length < 2) return [e];
  return merged.map(m => ({offsetX: e.offsetX + m.clientX - e.clientX, offsetY: e.offsetY + m.clientY - e.clientY}));
}
