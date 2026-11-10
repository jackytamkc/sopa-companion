// The iteration timeline: proseg's schedule as phase bands, per-iteration statistics, the frame on screen and the
// live edge. Click or drag anywhere to look at that iteration.
//
// Drawn on demand, never in a loop: pointer interaction on the next animation frame, data changes (new frames, the
// live playhead, the growing axis) at most every REDRAW_MS, nothing while the page is hidden. Frame statistics are
// reduced once per frame as frames arrive, and drawing never reads layout (a ResizeObserver supplies the size).
// With a PaintBatch (main.js) the timeline draws in the page's batch: a new snapshot's playhead and statistics appear in
// the frame that shows the snapshot, and a data redraw held back by REDRAW_MS joins the next batch that runs anyway.
// The hover readout names a live knob change by a vetted name (qc-copy.mjs knobLine, KNOB_NAMES) with the values as Live
// knobs shows them and the plain change: "Cell compactness 0.04 → 0.02 (tighter shapes)"; any other knob by its plain
// change ("held closer to Xenium: 1 → 4", a switch "drift off"): never the raw proseg flag, nor a Live
// knobs label outside a Setting line ("Prior weight" holds a banned word). `knobLabel` stays for callers.
import { harness } from './harness-hook.mjs';
import { QC_TEXT, copy, knobLine, sameSetting, hoverKey } from './qc-copy.mjs';

export const PHASES = {
  init: { label: 'Initial', color: '#5c6573', help: 'The starting cells, from the outlines or the column the run started from, before any sampling.' },
  burnin: { label: 'Burn-in', color: '#7b5cc4', help: 'Coarse voxels. Cells grow and shrink quickly while the expression model settles.' },
  upscale: { label: 'Finer voxels', color: '#c49a5c', help: 'Voxels are split into finer ones, so the boundaries can refine at full resolution.' },
  sampling: { label: 'Sampling', color: '#3d8fd1', help: 'Sampling at full resolution: every boundary can still move.' },
  recorded: { label: 'Recorded', color: '#2ba886', help: 'Sampling continues, and these iterations are averaged into the final result.' },
  hillclimb: { label: 'Hill-climb', color: '#d9a441', help: 'The temperature is lowered so the outlines settle into the final segmentation.' },
  continued: { label: 'Continued', color: '#4fb8c9', help: 'Sampling goes on after proseg\'s standard run, at full temperature; these iterations are averaged in too.' },
};

/** The last iteration of a run's schedule (burn-in + samples + hill-climb: where proseg's standard run ends and the run
 *  pauses), or null without the run's info or for a run made open-ended (runs of older versions could be set to sample
 *  until stopped: a phase of their plan has no end). */
export function scheduleEnd(info) {
  const s = info?.schedule;
  if (!s || (info.plan || []).some(part => part.frames === null)) return null;
  const end = s.burnin + s.samples + s.hillclimb;
  return Number.isFinite(end) ? end : null;
}

/** The phases the timeline draws: proseg's plan, then, once the run samples on past the end of its schedule (frames of
 *  phase "continued"), an open-ended "continued" phase after it. */
export function withContinued(plan, frames) {
  if (frames?.at(-1)?.phase !== 'continued' || plan.some(part => part.frames === null)) return plan;
  return [...plan, { phase: 'continued', frames: null }];
}

const fmt = new Intl.NumberFormat('en-GB');
export const REDRAW_MS = 100;   // data-driven redraws: ~10 fps while following a live run

const finite = Number.isFinite;
export const TEMPERATURE_TOLERANCE = 1e-3;   // within 0.1 % of the schedule's temperature is the schedule's (float32)
/** What proseg's own temperature schedule needs, from a run (engine/runlog.py temperature_schedule): {burnin, samples}
 *  of its info's schedule, and `temperature` when one was set at the start (Advanced options: it holds for every
 *  iteration); null without the info. */
export function temperatureSchedule(run) {
  const s = run?.info?.schedule;
  if (!s) return null;
  const fixed = run.options?.temperature;
  return { burnin: s.burnin, samples: s.samples, ...(finite(fixed) ? { temperature: fixed } : {}) };
}
/** The temperature proseg's own schedule gives a frame's iteration (proseg 3.2.0): 1 in burn-in and sampling; in the
 *  hill-climb cooling^k at its k-th iteration (k from 0, cooling = 0.01^(1 / burn-in iterations)); 1 again past the
 *  end of the standard run ("continued"); a temperature set at the start everywhere; null when it cannot be told. */
export function scheduleTemperature(frame, schedule) {
  if (!schedule) return null;
  if (finite(schedule.temperature)) return schedule.temperature;
  if (['init', 'burnin', 'upscale', 'sampling', 'recorded', 'continued'].includes(frame?.phase)) return 1;
  const { burnin, samples } = schedule, it = frame?.iter;
  if (frame?.phase !== 'hillclimb' || !finite(burnin) || !finite(samples) || !finite(it) || burnin <= 0) return null;
  return Math.exp(Math.log(0.01) / burnin * Math.max(0, it - burnin - samples - 1));
}
/** The schedule's own temperature moves, never a knob change (neither marked nor listed as one): the frame's temperature
 *  is the one proseg's schedule gives its iteration (`schedule`: temperatureSchedule's). A temperature set live is a knob
 *  change wherever it is set, in the hill-climb or right after the end of the standard run too. Without a schedule: the
 *  hill-climb and the step after it count as the schedule's (the old rule). `previous`: the frame before with knob values. */
export function scheduledTemperature(frame, previous, schedule = null) {
  const expected = scheduleTemperature(frame, schedule), t = frame?.live?.temperature ?? frame?.temperature;
  if (expected == null || !finite(t)) return frame?.phase === 'hillclimb' || previous?.phase === 'hillclimb';
  return Math.abs(t - expected) <= TEMPERATURE_TOLERANCE * Math.max(expected, 1e-3);
}
const knobKey = (live, skipTemperature) =>
  JSON.stringify(Object.entries(live).filter(([flag]) => !(skipTemperature && flag === 'temperature')).sort(([a], [b]) => a.localeCompare(b)));

/** Per-frame statistics for drawing, reduced once per frame as frames arrive (received frames never change). */
export function reduceFrames(r, frames, schedule = null) {
  if (!r || r.frames !== frames || r.n > frames.length || (r.n && frames[r.n - 1] !== r.last) || r.schedule !== schedule)
    r = { frames, n: 0, last: null, changed: [], loglik: [], tick: [], maxChanged: 1, lo: Infinity, hi: -Infinity, liveFrame: null, thinned: false, schedule };
  for (let i = r.n; i < frames.length; i++) {
    const f = frames[i], changed = Math.max(0, f.changed || 0), loglik = Number.isFinite(f.loglik) ? f.loglik : NaN;
    r.changed.push(changed); r.loglik.push(loglik); r.maxChanged = Math.max(r.maxChanged, changed);
    if (loglik === loglik) { r.lo = Math.min(r.lo, loglik); r.hi = Math.max(r.hi, loglik); }
    let tick = false;   // a live knob changed at this frame
    if (f.live) {
      const skip = scheduledTemperature(f, r.liveFrame, schedule);
      tick = i > 0 && knobKey(f.live, skip) !== (r.liveFrame ? knobKey(r.liveFrame.live, skip) : ''); r.liveFrame = f;
    }
    r.tick.push(tick);
    if (i > 0 && f.iter - frames[i - 1].iter > 1) r.thinned = true;
  }
  r.n = frames.length; r.last = frames[r.n - 1] ?? null;
  return r;
}

/** Pixel columns of the statistics (frame i at x = left + i * scale): each keeps the range of its frames. */
export function columnsOf(r, left, scale) {
  const columns = [], ticks = [];
  let c = null;
  for (let i = 0; i < r.n; i++) {
    const x = Math.floor(left + i * scale), changed = r.changed[i], loglik = r.loglik[i];
    if (c?.x !== x) columns.push(c = { x, min: changed, max: changed, lo: Infinity, hi: -Infinity });
    else { c.min = Math.min(c.min, changed); c.max = Math.max(c.max, changed); }
    if (loglik === loglik) { c.lo = Math.min(c.lo, loglik); c.hi = Math.max(c.hi, loglik); }
    if (r.tick[i] && ticks.at(-1) !== x) ticks.push(x);
  }
  return { columns, ticks };
}

// The mixing check's lane: a thin strip under the chart with its own small
// scale, never a second axis on the chart. It is drawn inside the canvas's own height (app.css: 116 px), the chart above
// it LANE_H px shorter while it shows: the dock, and so the stage, never changes size for the check (the stage's visual
// parity holds with the check on). A canvas under LANE_MIN_HEIGHT has no room for it. Its scale fits the line once
// cells have formed and the start, so a change of a point or two shows; the forming stretch is faint (it rises by
// itself); the newest value is written at the line's end, and the labels sit where the line is not (never over its
// newest stretch).
export const LANE_H = 30, LANE_MIN_HEIGHT = 100, LANE_COLOR = '#d55181', LANE_PAD = 2;

/** The lane's line per pixel column (point seq at x = left + seq * scale): each keeps its range and its last value, and
 *  whether it is still the forming stretch (its last point at or before `formedIter`). */
export function laneColumns(points, left, scale, formedIter = null) {
  const columns = [];
  let c = null;
  for (const p of points) {
    if (!Number.isFinite(p.per100) || !Number.isFinite(p.seq)) continue;
    const x = Math.floor(left + p.seq * scale), forming = Number.isFinite(formedIter) && Number.isFinite(p.iter) && p.iter <= formedIter;
    if (c?.x !== x) columns.push(c = { x, min: p.per100, max: p.per100, last: p.per100, ...(forming ? { forming } : {}) });
    else { c.min = Math.min(c.min, p.per100); c.max = Math.max(c.max, p.per100); c.last = p.per100; if (!forming) delete c.forming; }
  }
  return columns;
}
/** The lane's scale [lo, hi]: the settled columns (else all) and the start, LANE_PAD points around them, at least 4 wide. */
export function laneRange(columns, start) {
  const settled = columns.filter(c => !c.forming), use = settled.length ? settled : columns;
  const values = [...use.flatMap(c => [c.min, c.max]), ...(Number.isFinite(start) ? [start] : [])];
  if (!values.length) return [0, 5];
  let lo = Math.max(0, Math.floor(Math.min(...values) - LANE_PAD)), hi = Math.min(100, Math.ceil(Math.max(...values) + LANE_PAD));
  if (hi - lo < 4) { hi = Math.min(100, lo + 4); lo = Math.max(0, hi - 4); }
  return [lo, hi];
}

/** The open-ended axis follows the frame count: one more iteration snaps (no redraw loop), larger jumps glide. */
export function nextDomain(domain, target, dt) {
  if (!domain || Math.abs(target - domain) <= Math.max(0.1, 0.02 * domain)) return target;
  return domain + (target - domain) * (1 - Math.exp(-Math.max(0, dt) / 120));
}

export class Timeline {
  constructor(canvas, { onScrub, batch = null, knobLabel = null }) {
    this.canvas = canvas;
    this.batch = batch; this.drawCount = 0;
    this.knobLabel = typeof knobLabel === 'function' ? knobLabel : () => null;
    this.readout = document.createElement('div');
    this.readout.className = 'tooltip timeline-readout';
    this.readout.hidden = true;
    document.body.append(this.readout);
    this.onScrub = onScrub;
    this.frames = [];
    this.plan = this.basePlan = [];
    this.current = -1;
    this.hover = -1;
    this.pad = { left: 12, right: 12, top: 4, band: 18, bottom: 16 };
    this.width = canvas.clientWidth; this.height = canvas.clientHeight;
    this.viewport = window.innerWidth;
    this.drawnAt = -Infinity; this.raf = null; this.timer = null;
    new ResizeObserver(entries => {
      const box = entries.at(-1).contentRect;
      this.width = box.width; this.height = box.height; this.invalidate(true);
    }).observe(canvas);
    window.addEventListener('resize', () => { this.viewport = window.innerWidth; });
    document.addEventListener('visibilitychange', () => { if (!document.hidden && this.dirty) this.schedule(); });
    this.covered = false;   // a full-screen view over the page (setCovered): nothing drawn until it goes
    let dragging = false;
    // The canvas's position in the viewport comes from the pointer itself: placing the readout reads no layout.
    const track = (e) => { this.origin = [e.clientX - e.offsetX, e.clientY - e.offsetY]; };
    canvas.addEventListener('pointerdown', (e) => {
      if (!this.total) return;
      track(e);
      dragging = true;
      canvas.setPointerCapture(e.pointerId);
      this.scrub(e);
    });
    canvas.addEventListener('pointermove', (e) => {
      track(e);
      this.setHover(this.seqAt(e.offsetX));
      if (dragging) this.scrub(e);
    });
    canvas.addEventListener('pointerup', () => { dragging = false; });
    canvas.addEventListener('pointercancel', () => { dragging = false; });
    canvas.addEventListener('pointerleave', () => this.setHover(-1));
    this.invalidate(true);
  }

  /** plan: [{phase, frames}] in schedule order (an open "continued" phase joins it when the run samples on past it);
   *  run: the run (its info's schedule and options: proseg's own temperature moves are not marked as knob changes). */
  setPlan(plan, run = null) {
    this.basePlan = plan || []; this.plan = withContinued(this.basePlan, this.frames); this.domain = 0;
    this.temperatures = temperatureSchedule(run); this.invalidate(true);
  }
  /** The mixing lane: {points: [{seq, iter, per100, shown, single, count}] ascending by seq, version (changes with the
   *  points), start (the start line, per 100: proseg's count at iteration 0, named so) | null, formedIter (the iteration
   *  cells have formed by: the stretch before it is faint and outside the scale)}, or null (the check is off). Redrawn
   *  with the data, at the timeline's own pace. */
  setQc(lane) {
    const key = lane ? `${lane.version}:${lane.start}:${lane.points?.length}:${lane.formedIter}` : '';
    if (key === this.qcKey) return;
    this.qcKey = key; this.qcBySeq = null;
    this.qc = lane ? { points: lane.points || [], start: lane.start ?? null, formedIter: lane.formedIter ?? null } : null;
    if (!lane) this.laneTexts = [];
    this.invalidate();
  }
  /** Every word the lane drew last (the harness checks them for banned words). */
  qcTexts() { return [...(this.laneTexts ?? [])]; }
  /** The lane's point at frame `seq` (its own result only), or null. */
  qcPointAt(seq) {
    if (!this.qc) return null;
    this.qcBySeq ??= new Map(this.qc.points.map(p => [p.seq, p]));
    return this.qcBySeq.get(seq) ?? null;
  }
  setFrames(frames) { this.frames = frames; this.plan = withContinued(this.basePlan, frames); this.reduce(); this.invalidate(); }
  /** The frame on screen; `urgent` when the user moved it (scrub, step, replay) rather than a live update. */
  setCurrent(seq, urgent = false) { if (seq !== this.current) { this.current = seq; this.invalidate(urgent); } }
  setHover(seq) { if (seq !== this.hover) { this.hover = seq; this.invalidate(true); } }
  refresh() { this.invalidate(true); }
  /** Covered by a full-screen view (main.js: Compare's dialog), or no longer: covered, it draws nothing, as while the
   *  page is hidden (new frames and results only mark it dirty); uncovered, it draws what changed meanwhile, once. */
  setCovered(on) {
    this.covered = !!on;
    if (!this.covered && this.dirty) this.schedule();
  }

  /** Redraw: pointer interaction (urgent) on the next animation frame, data changes at most every REDRAW_MS. */
  invalidate(urgent = false) {
    this.dirty = true;
    this.urgent ||= urgent;
    this.schedule();
  }
  schedule() {
    if (this.raf != null || document.hidden || this.covered) return;
    const wait = this.urgent ? 0 : this.drawnAt + REDRAW_MS - performance.now();
    if (this.batch) {
      const run = () => { clearTimeout(this.timer); this.timer = null; if (this.dirty && !document.hidden && !this.covered) this.draw(); };
      if (wait > 0) {
        this.batch.ride('timeline', run);
        if (this.timer == null) this.timer = setTimeout(() => { this.timer = null; this.schedule(); }, wait);
      } else { clearTimeout(this.timer); this.timer = null; this.batch.mark('timeline', run, this.urgent); }
      return;
    }
    if (wait > 0) {
      if (this.timer == null) this.timer = setTimeout(() => { this.timer = null; this.schedule(); }, wait);
      return;
    }
    clearTimeout(this.timer); this.timer = null;
    this.raf = requestAnimationFrame(() => { this.raf = null; if (this.dirty && !this.covered) this.draw(); });
  }

  reduce() { return this.reduction = reduceFrames(this.reduction, this.frames, this.temperatures ?? null); }
  get thinned() { return this.reduce().thinned; }
  get bands() {
    const r = this.reduce(), open = this.openEnded;
    if (!r.thinned) return this.plan;
    if (r.bands?.n !== r.n || r.bands.open !== open) {
      const list = [];
      for (const frame of this.frames) {
        const last = list.at(-1);
        if (last?.phase === frame.phase) last.frames++;
        else list.push({ phase: frame.phase, frames: 1 });
      }
      if (open && list.length) list.at(-1).frames = null;
      r.bands = { n: r.n, open, list };
    }
    return r.bands.list;
  }

  get openEnded() { return this.plan.some(p => p.frames === null); }

  get total() {
    const planned = this.thinned ? this.frames.length : this.plan.reduce((n, p) => n + p.frames, 0);
    return this.openEnded ? (this.domain || Math.max(planned + 60, this.frames.length * 1.1)) : Math.max(planned, this.frames.length);
  }

  x(seq, total = this.total) {
    const w = this.width - this.pad.left - this.pad.right;
    return this.pad.left + (total > 1 ? seq / (total - 1) : 0) * w;
  }

  seqAt(px) {
    const total = this.total, w = this.width - this.pad.left - this.pad.right;
    if (!total) return -1;
    return Math.round(Math.min(1, Math.max(0, (px - this.pad.left) / w)) * (total - 1));
  }

  scrub(e) {
    const seq = Math.min(this.seqAt(e.offsetX), this.frames.length - 1);
    if (seq >= 0) this.onScrub(seq);
  }

  draw() {
    const now = performance.now(), dt = Math.min(250, now - this.drawnAt);
    this.dirty = false; this.urgent = false; this.drawnAt = now; this.drawCount++; harness?.count('timelineDraws');
    const r = this.reduce();
    if (this.openEnded) {
      const target = Math.max(this.plan.reduce((n, p) => n + (p.frames || 0), 0) + 60, this.frames.length * 1.1);
      this.domain = nextDomain(this.domain, target, dt);
      if (this.domain !== target) this.invalidate();
    }
    const canvas = this.canvas, ratio = window.devicePixelRatio || 1;
    const W = this.width, H = this.height;
    if (canvas.width !== Math.round(W * ratio) || canvas.height !== Math.round(H * ratio)) {
      canvas.width = Math.round(W * ratio);
      canvas.height = Math.round(H * ratio);
    }
    const g = canvas.getContext('2d');
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    g.clearRect(0, 0, W, H);
    const total = this.total, x = (seq) => this.x(seq, total);
    if (!total) { this.hideReadout(); return; }
    const { top, band, bottom } = this.pad;
    const lane = this.qc && H >= LANE_MIN_HEIGHT ? this.qc : null, laneH = lane ? LANE_H : 0;
    const chartTop = top + band + 6, chartBottom = H - bottom - laneH, chartH = chartBottom - chartTop;
    const step = total > 1 ? x(1) - x(0) : 1;
    const latest = this.frames.length - 1;

    // Phase bands.
    let start = 0;
    g.font = '600 10px "Ubuntu Sans", system-ui, sans-serif';
    g.textBaseline = 'middle';
    for (const part of this.bands) {
      const meta = PHASES[part.phase] || { label: part.phase, color: '#5c6573' };
      const x0 = x(start) - step / 2, x1 = part.frames === null ? W - this.pad.right : x(start + part.frames - 1) + step / 2;
      g.fillStyle = meta.color;
      if (part.frames === null) {
        const fade = g.createLinearGradient(x0, 0, x1, 0);
        fade.addColorStop(0, meta.color); fade.addColorStop(0.7, meta.color); fade.addColorStop(1, 'transparent');
        g.fillStyle = fade;
      }
      g.globalAlpha = 0.85;
      g.fillRect(x0, top, Math.max(1, x1 - x0 - 1), band);
      g.globalAlpha = 0.07;
      g.fillRect(x0, chartTop, Math.max(1, x1 - x0 - 1), chartH);
      g.globalAlpha = 1;
      const label = part.label || meta.label;
      if (x1 - x0 > g.measureText(label).width + 10) {
        g.fillStyle = '#0e1116';
        g.fillText(label.toUpperCase(), x0 + 6, top + band / 2 + 0.5);
      }
      start += part.frames;
    }

    // Statistics: voxel changes (area) and log-likelihood (line), each scaled to what has arrived so far. Columns are
    // rebuilt only when the frame count, the width or the axis changes: pointer movement draws O(width) points.
    const key = `${r.n}:${W}:${total}`;
    if (r.key !== key) {
      Object.assign(r, columnsOf(r, this.pad.left, total > 1 ? (W - this.pad.left - this.pad.right) / (total - 1) : 0));
      r.key = key;
    }
    const { columns, ticks, maxChanged, lo, hi } = r;
    const changeY = value => chartBottom - Math.sqrt(value / maxChanged) * chartH * 0.9;
    if (columns.length) {
      const fill = g.createLinearGradient(0, chartTop, 0, chartBottom);
      fill.addColorStop(0, 'rgba(255,180,84,.30)'); fill.addColorStop(1, 'rgba(255,180,84,.04)');
      g.fillStyle = fill; g.beginPath(); g.moveTo(columns[0].x, chartBottom);
      for (const c of columns) g.lineTo(c.x, changeY(c.max));
      g.lineTo(columns.at(-1).x, chartBottom); g.closePath(); g.fill();
      g.strokeStyle = 'rgba(255,180,84,.85)'; g.lineWidth = 1.2; g.beginPath();
      for (let i = 0; i < columns.length; i++) {
        const c = columns[i];
        if (i) g.lineTo(c.x, changeY(c.max)); else g.moveTo(c.x, changeY(c.max));
        g.lineTo(c.x, changeY(c.min));
      }
      g.stroke();
    }
    if (hi > lo) {
      const y = value => chartBottom - (value - lo) / (hi - lo) * chartH * 0.92 - 2;
      g.strokeStyle = '#7cc4ff'; g.lineWidth = 1.6; g.beginPath();
      let pen = false;
      for (const c of columns) {
        if (!Number.isFinite(c.lo)) { pen = false; continue; }
        if (pen) g.lineTo(c.x, y(c.lo)); else g.moveTo(c.x, y(c.lo));
        g.lineTo(c.x, y(c.hi)); pen = true;
      }
      g.stroke();
    }
    g.fillStyle = '#e6eaf0';
    for (const tx of ticks) g.fillRect(tx, top + band - 4, 1, 7);

    // Not yet computed: darken the future.
    if (latest < total - 1) {
      const xl = x(Math.max(0, latest)) + step / 2;
      g.fillStyle = 'rgba(14,17,22,.6)';
      g.fillRect(xl, this.openEnded ? chartTop : top, W - xl, chartBottom - (this.openEnded ? chartTop : top));
      if (latest >= 0) {
        g.fillStyle = '#ff5f5f';
        g.beginPath();
        g.arc(x(latest), top + band + 3, 3, 0, Math.PI * 2);
        g.fill();
      }
    }

    // Axis ticks.
    g.fillStyle = '#7d8796';
    g.font = '10px "Ubuntu Sans", system-ui, sans-serif';
    g.textBaseline = 'alphabetic';
    for (const tick of axisTicks(this.frames, total, niceStep(total, W / 70), r.thinned)) {
      g.fillRect(x(tick.seq), chartBottom, 1, 3);
      if (tick.label != null) g.fillText(fmt.format(tick.label), x(tick.seq) + 3, H - 3 - laneH);
    }
    // Legend, on a backdrop so it never collides with the curves.
    g.font = '10px "Ubuntu Sans", system-ui, sans-serif';
    const legend = [['voxels changed per iteration (√ scale)', '#ffb454'], ['log-likelihood', '#7cc4ff']];
    const lw = Math.max(...legend.map(([t]) => g.measureText(t).width)) + 30;
    g.fillStyle = 'rgba(21,26,33,.85)';
    g.fillRect(W - lw - 14, chartTop, lw + 4, 30);
    legend.forEach(([text, color], i) => {
      g.fillStyle = color;
      g.fillRect(W - lw - 8, chartTop + 9 + i * 13, 14, 2);
      g.fillStyle = '#c5ccd5';
      g.fillText(text, W - lw + 10, chartTop + 13 + i * 13);
    });

    // Hover and the frame on screen.
    if (this.hover >= 0 && this.hover <= latest && this.hover !== this.current) {
      g.fillStyle = 'rgba(255,255,255,.25)';
      g.fillRect(Math.round(x(this.hover)), top, 1, chartBottom - top);
    }
    if (this.current >= 0) {
      const cx = Math.round(x(this.current));
      g.fillStyle = '#ffffff';
      g.fillRect(cx - 1, top, 2, chartBottom - top);
      g.beginPath();
      g.moveTo(cx - 5, top);
      g.lineTo(cx + 5, top);
      g.lineTo(cx, top + 6);
      g.fill();
    }
    if (lane) this.drawLane(g, lane, total, H - laneH, H, latest);
    if (this.hover >= 0 && this.hover <= latest) this.showReadout(this.hover, total);
    else this.hideReadout();
  }

  /** The mixing lane between y0 and y1: its line (one colour; the forming stretch faint), the start line dashed, the
   *  newest value at the line's end, the legend and the start's label where the line is not. */
  drawLane(g, lane, total, y0, y1, latest) {
    const W = this.width, left = this.pad.left, right = W - this.pad.right, x = seq => this.x(seq, total);
    const key = `${this.qcKey}:${W}:${total}`;
    if (this.laneKey !== key) {
      this.laneCols = laneColumns(lane.points, left, total > 1 ? (right - left) / (total - 1) : 0, lane.formedIter);
      this.laneSpan = laneRange(this.laneCols, lane.start); this.laneKey = key;
    }
    const cols = this.laneCols, [lo, hi] = this.laneSpan, top = y0 + 7, bottom = y1 - 5;
    const y = v => bottom - Math.min(1, Math.max(0, (v - lo) / (hi - lo))) * (bottom - top);
    g.fillStyle = '#2a313c'; g.fillRect(left, y0 + 1, right - left, 1);
    g.font = '10px "Ubuntu Sans", system-ui, sans-serif'; g.textBaseline = 'middle';
    if (Number.isFinite(lane.start)) {
      const sy = Math.round(y(lane.start)) + 0.5;
      g.strokeStyle = 'rgba(154,164,178,.7)'; g.lineWidth = 1; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(left, sy); g.lineTo(right, sy); g.stroke(); g.setLineDash([]);
    }
    const texts = [];
    let lastX = null;
    if (cols.length) {
      const stroke = (part, faint) => {
        if (!part.length) return;
        g.globalAlpha = faint ? .35 : 1; g.strokeStyle = LANE_COLOR; g.lineWidth = 1.4; g.beginPath();
        part.forEach((c, i) => { if (i) g.lineTo(c.x, y(c.max)); else g.moveTo(c.x, y(c.max)); if (c.min !== c.max) g.lineTo(c.x, y(c.min)); });
        g.stroke(); g.globalAlpha = 1;
      };
      const settled = cols.findIndex(c => !c.forming);
      stroke(settled < 0 ? cols : cols.slice(0, settled + 1), true);
      if (settled >= 0) stroke(cols.slice(settled), false);
      const last = cols.at(-1), point = lane.points.at(-1), ly = y(last.last);
      g.fillStyle = LANE_COLOR; g.beginPath(); g.arc(last.x, ly, 2.5, 0, Math.PI * 2); g.fill();
      // The newest value at the line's end: right of the dot, or left of it near the right edge.
      const value = (Number.isInteger(point?.shown) ? point.shown : Math.round(last.last)).toLocaleString('en-GB'), vw = g.measureText(value).width;
      const vx = last.x + 5 + vw <= right ? last.x + 5 : last.x - 5 - vw;
      g.fillStyle = 'rgba(21,26,33,.85)'; g.fillRect(vx - 2, ly - 6, vw + 4, 12); g.fillStyle = '#e6eaf0'; g.fillText(value, vx, ly);
      texts.push(value); lastX = Math.max(last.x, vx + vw);
    }
    // The future is dark, as on the chart; the playhead runs through the lane.
    if (latest < total - 1) { const xl = x(Math.max(0, latest)) + (total > 1 ? (x(1) - x(0)) / 2 : 0); g.fillStyle = 'rgba(14,17,22,.6)'; g.fillRect(xl, y0 + 2, W - xl, y1 - y0 - 2); }
    if (this.current >= 0) { g.fillStyle = '#ffffff'; g.fillRect(Math.round(x(this.current)) - 1, y0 + 2, 2, y1 - y0 - 2); }
    // The legend and the start's label, on backdrops, where the line is not: past its newest value (the part not
    // computed yet) when there is room, else at the left end (over the faint forming stretch, never the newest one).
    const legend = QC_TEXT.timelineLegend, lw = g.measureText(legend).width + 26, ly = y0 + (y1 - y0) / 2 + 1;
    const startLabel = Number.isFinite(lane.start) ? copy('startReference', { s: Math.round(lane.start).toLocaleString('en-GB') }) : null;
    const sw = startLabel ? g.measureText(startLabel).width + 8 : 0;
    const after = lastX == null ? left : lastX + 12, lx = after + lw + 4 + (sw ? sw + 8 : 0) <= right ? after : left;
    g.fillStyle = 'rgba(21,26,33,.85)'; g.fillRect(lx, ly - 7, lw + 4, 14);
    g.fillStyle = LANE_COLOR; g.fillRect(lx + 6, ly - 1, 14, 2);
    g.fillStyle = '#c5ccd5'; g.fillText(legend, lx + 24, ly);
    texts.push(legend);
    if (startLabel) {
      const sx = Math.min(lx + lw + 12, right - sw), sy = y(lane.start);
      g.fillStyle = 'rgba(21,26,33,.85)'; g.fillRect(sx, sy - 6, sw, 12); g.fillStyle = '#9aa4b2'; g.fillText(startLabel, sx + 4, sy);
      texts.push(startLabel);
    }
    this.laneTexts = texts;
  }

  hideReadout() { if (!this.readout.hidden) this.readout.hidden = true; }

  /** The hovered iteration's details. Its text (and measured size) change only with the hovered frame. */
  showReadout(seq, total) {
    const f = this.frames[seq], box = this.readout, qc = this.qcPointAt(seq);
    if (!f) { this.hideReadout(); return; }
    if (this.readoutFrame !== f || this.readoutQc !== qc || box.hidden) {
      const phase = PHASES[f.phase]?.label || f.phase;
      const shown = qc && (Number.isInteger(qc.shown) ? qc.shown : Number.isFinite(qc.per100) ? Math.round(qc.per100) : null);
      const lines = [`Iteration ${f.iter} · ${phase}`,
        Number.isFinite(f.temperature) ? `Temperature ${f.temperature.toFixed(3)}` : '',
        `${fmt.format(f.changed ?? 0)} voxels changed`,
        Number.isFinite(f.loglik) ? `log-likelihood ${f.loglik.toExponential(4)}` : '',
        // A live result's number smoothed, a saved iteration's as saved, else one iteration's own (as the chart's hover).
        shown != null ? copy(hoverKey(qc, 'timelineHover'), { n: fmt.format(shown) }) : ''].filter(Boolean);
      let before;
      for (let i = seq - 1; i >= 0; i--) {
        if (this.frames[i].live) { before = this.frames[i]; break; }
      }
      if (before && f.live) {
        const previous = before.live, skip = scheduledTemperature(f, before, this.temperatures ?? null);
        for (const [flag, value] of Object.entries(f.live)) {
          if (skip && flag === 'temperature') continue;   // the hill-climb's own cooling: its line above says it
          if (!sameSetting(value, previous[flag])) lines.push(knobLine(flag, previous[flag], value, this.knobLabel(flag)));
        }
      }
      box.textContent = lines.join('\n');
      box.hidden = false;
      this.readoutFrame = f; this.readoutQc = qc;
      this.readoutSize = [box.offsetWidth, box.offsetHeight];
    }
    const [left, top] = this.origin || [0, 0], [w, h] = this.readoutSize;
    const px = `${Math.round(Math.max(4, Math.min(left + this.x(seq, total) + 10, this.viewport - w - 4)))}px`;
    const py = `${Math.round(Math.max(4, top - h - 6))}px`;
    if (box.style.left !== px) box.style.left = px;
    if (box.style.top !== py) box.style.top = py;
  }
}

/** The axis's ticks ({seq: where, label: the iteration written there, or null}): at round iterations (0, 20, ... 200,
 *  220 ...), each at the first snapshot whose iteration reaches it, so the labels and proseg's phases agree (the extra
 *  finer-voxel snapshot shifted every later label by one: 219, 239 ... 299, 399); ahead of the newest snapshot, one per
 *  iteration, marks without a label. A thinned run (past 2,000 iterations only every 5th or more is kept) keeps evenly
 *  spaced snapshots, each labelled with its own iteration. */
export function axisTicks(frames, total, every, thinned = false) {
  const out = [];
  if (thinned) {
    for (let s = 0; s < total; s += every) out.push({ seq: s, label: frames[s]?.iter ?? null });
    return out;
  }
  const n = frames.length, lastIter = n ? frames[n - 1].iter : 0;
  for (let target = 0, i = 0; ; target += every) {
    while (i < n && !(frames[i].iter >= target)) i++;
    const seq = i < n ? i : n ? n - 1 + (target - lastIter) : target;
    if (seq >= total) break;
    out.push({ seq, label: i < n ? frames[i].iter : null });
  }
  return out;
}

function niceStep(total, slots) {
  const raw = total / Math.max(1, slots);
  for (const s of [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000]) if (s >= raw) return s;
  return 1000 * Math.ceil(raw / 1000);
}
