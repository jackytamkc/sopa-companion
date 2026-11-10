// One paint per change. A remote desktop reads back, composites and encodes in software every frame the
// browser presents, so the panels a new snapshot changes (HUD, facts, status line, timeline, runs list, most active
// cells, cell card) are written together, in the animation frame in which the stage draws that snapshot, instead of
// spreading the same information over several frames as each piece arrives.
//
// mark(key, fn) queues fn for the next batch (the latest fn per key, once), run on the next animation frame. A snapshot
// on its way holds the batch: show() takes a hold before it fetches, and the task that uploads the frame releases it
// with release(token, true), which writes everything queued meanwhile (the new frame's status, its timeline, its
// statistics) at once, in that task, so it is painted with the stage's next frame even when the next snapshot's fetch
// starts (and holds) right away. A hold expires after HOLD_MS, so a slow fetch delays the panels at most that long.
// Urgent marks (pointer interaction) are not held. ride(key, fn) runs fn with the next batch without asking for one (a
// throttled timeline redraw joins a snapshot's frame).
export const HOLD_MS = 1000;   // a whole-sample view takes ~0.1-0.5 s to fetch; a slower one lets the panels go first

export class PaintBatch {
  constructor({ hold = HOLD_MS } = {}) {
    this.holdMs = hold; this.jobs = new Map(); this.riders = new Map(); this.holds = new Map();
    this.raf = null; this.timer = null; this.urgent = false; this.flushes = 0; this.covered = false;
  }
  /** A full-screen view over the page (main.js: Compare's dialog), or no longer: covered, no batch is asked for and
   *  none is written (the jobs wait, the latest per key, as while the page is hidden; a snapshot released with its
   *  upload writes nothing either); uncovered, what waited runs in the next batch. */
  setCovered(on) {
    this.covered = !!on;
    if (this.covered && this.raf != null) { cancelAnimationFrame(this.raf); this.raf = null; }
    if (!this.covered) this.schedule();
  }
  /** Run `fn` in the next batch; `urgent` skips any hold (the user is interacting). */
  mark(key, fn, urgent = false) {
    this.riders.delete(key); this.jobs.set(key, fn);
    if (urgent) this.urgent = true;
    this.schedule();
  }
  /** Run `fn` with the next batch, whenever that is, without scheduling one. */
  ride(key, fn) { if (!this.jobs.has(key)) this.riders.set(key, fn); }
  /** Hold the batch until release(token) (or HOLD_MS): a snapshot is on its way. */
  hold() { const token = {}; this.holds.set(token, performance.now() + this.holdMs); return token; }
  /** The snapshot has arrived (`now`: and was uploaded in this task, so write the panels with it) or was dropped. */
  release(token, now = false) {
    this.holds.delete(token);
    if (now) { this.flush(); return; }
    if (this.held) return;
    clearTimeout(this.timer); this.timer = null; this.schedule();
  }
  /** Whether an unexpired hold remains (expired ones are dropped). */
  get held() {
    const now = performance.now();
    for (const [token, until] of this.holds) { if (until > now) return true; this.holds.delete(token); }
    return false;
  }
  get pending() { return this.jobs.size > 0; }
  schedule() {
    if (this.raf != null || !this.jobs.size || this.covered) return;
    if (!this.urgent && this.held) { this.wait(); return; }
    this.raf = requestAnimationFrame(() => {
      this.raf = null;
      if (!this.urgent && this.held) { this.wait(); return; }
      this.flush();
    });
  }
  /** Check again when the last current hold expires (release() comes first when the snapshot arrives in time). */
  wait() {
    if (this.timer != null) return;
    const until = Math.max(...this.holds.values());
    this.timer = setTimeout(() => { this.timer = null; this.schedule(); }, Math.max(0, until - performance.now()));
  }
  /** Run everything queued now, in the order first queued; one failing job does not stop the others. Covered, nothing
   *  runs: it stays queued for the batch once uncovered. */
  flush() {
    if (this.covered || (!this.jobs.size && !this.riders.size)) return;
    const jobs = [...this.jobs.values(), ...this.riders.values()];
    this.jobs.clear(); this.riders.clear(); this.urgent = false; this.flushes++;
    for (const fn of jobs) { try { fn(); } catch (error) { console.error(error); } }
  }
}
