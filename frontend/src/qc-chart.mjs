import { QC_TEXT as C, copy, number, changeLabel, settingValue, hoverKey, KNOB_NAMES } from './qc-copy.mjs';

export const QC_COLORS = Object.freeze({
  little: '#6b3f63', some: '#b2477a', mixed: '#f08cbc', accent: '#d55181',
  unjudged: '#5c6573', ink: '#e6eaf0', muted: '#9aa4b2', grid: '#2a313c',
  band: 'rgba(138,148,163,.20)', forming: 'rgba(138,148,163,.10)',
});
const finite = Number.isFinite;
const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
export const FORMING_PHASES = Object.freeze(['init', 'burnin', 'burn-in', 'burn_in']);

/** Rounding boundaries are n ± .5, with another .3 required before moving.
 * Historical frames should use their stored shown value, never the live history.
 */
export function shownInteger(value, previous = null) {
  if (!finite(value)) return null;
  value = clamp(value, 0, 100);
  if (!Number.isInteger(previous)) return Math.round(value);
  if (value >= previous + .8 - 1e-9) return clamp(Math.floor(value + .2 + 1e-9), 0, 100);
  if (value <= previous - .8 + 1e-9) return clamp(Math.ceil(value - .2 - 1e-9), 0, 100);
  return clamp(previous, 0, 100);
}

export function linearScale(domain, range) {
  const [d0, d1] = domain, [r0, r1] = range, span = d1 - d0;
  const scale = value => span ? r0 + (value - d0) / span * (r1 - r0) : (r0 + r1) / 2;
  scale.invert = value => r1 === r0 ? (d0 + d1) / 2 : d0 + (value - r0) / (r1 - r0) * span;
  scale.domain = [...domain]; scale.range = [...range];
  return scale;
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count), power = 10 ** Math.floor(Math.log10(raw || 1));
  const factor = raw / power;
  return (factor <= 1 ? 1 : factor <= 2 ? 2 : factor <= 5 ? 5 : 10) * power;
}
export function niceTicks(min, max, count = 4) {
  if (!finite(min) || !finite(max) || max < min) return [];
  if (min === max) return [min];
  const step = niceStep(max - min, count), ticks = [];
  for (let n = Math.ceil(min / step); n * step <= max + step * 1e-9; n++) ticks.push(Number((n * step).toPrecision(12)));
  return ticks;
}

/** Sorted copies; invalid rates are gaps, not zeroes. Last duplicate iteration wins. */
export function seriesPoints(series = []) {
  const byIteration = new Map();
  for (const p of series ?? []) if (finite(p?.iter) && p.iter >= 0) byIteration.set(p.iter, { ...p,
    per100: finite(p.per100) ? clamp(p.per100, 0, 100) : null,
  });
  return [...byIteration.values()].sort((a, b) => a.iter - b.iter);
}

/** The trial band "expected without the change": flat, never a sloped
 *  line (a line over the last results gave false verdicts in 68 % of the no-change trials). Centre: the mean of the
 *  live line over the 50 iterations before the change, taken only from 20 iterations after cells formed (burn-in) on,
 *  with at least 10 results (else no band); half-width: max(2 SD, 1 in 100); drawn for at most 300 iterations, as long
 *  as verdicts are given. The verdict is the engine's (depth-matched, on its own smoothed values): the page draws the
 *  engine's band when the trial carries one, and this one only as a stand-in for drawing. */
export const TRIAL_BAND = Object.freeze({ history: 50, settle: 20, minHistory: 10, mult: 2, floor: 1, horizon: 300 });
const noteOf = (trial, key) => { const notes = trial?.notes ?? []; return Array.isArray(notes) ? notes.includes(key) : !!notes[key]; };
/** A running trial whose line can be judged against a band (not changed while cells formed, not mixed up with another
 *  change, not without history before it, not ended). Drift (R3) is judged like any other change: proseg's count shows
 *  it; only an older engine's "no verdict" note still means no band. */
export function canPredict(trial) {
  return !!trial && !trial.forming && !trial.mixed_up && !['forming', 'mixed_up', 'no_history', 'no_verdict', 'last_stage'].some(n => noteOf(trial, n))
    && !trial.outcome && !trial.ended;
}
export function flatBand(series, trial = {}, { endIter, formedIter = null } = {}) {
  if (!finite(trial.start_iter) || !canPredict(trial)) return null;
  const all = seriesPoints(series), from = trial.start_iter - TRIAL_BAND.history;
  const settled = finite(formedIter) ? formedIter + TRIAL_BAND.settle : -Infinity;
  const before = all.filter(p => finite(p.per100) && !p.single && p.iter < trial.start_iter && p.iter >= from && p.iter >= settled);
  if (before.length < TRIAL_BAND.minHistory) return null;
  const centre = before.reduce((n, p) => n + p.per100, 0) / before.length;
  const sd = Math.sqrt(before.reduce((n, p) => n + (p.per100 - centre) ** 2, 0) / (before.length - 1));
  const halfWidth = Math.max(TRIAL_BAND.mult * sd, TRIAL_BAND.floor);
  const newest = all.at(-1)?.iter ?? trial.start_iter, end = Math.min(trial.start_iter + TRIAL_BAND.horizon, Math.max(trial.start_iter, endIter ?? newest));
  return { centre, sd, halfWidth, from: trial.start_iter, to: end, points: bandPoints(centre - halfWidth, centre + halfWidth, trial.start_iter, end) };
}
const bandPoints = (low, high, from, to) => [from, to].filter((iter, i, a) => i === 0 || iter !== a[0])
  .map(iter => ({ iter, expected: clamp((low + high) / 2, 0, 100), low: clamp(low, 0, 100), high: clamp(high, 0, 100) }));
/** The engine's band in whichever form it comes: points [{iter, low, high}], or a flat {low, high} | {centre (or
 *  expected), half_width} from its first iteration (`from_iter`, else the trial's start) to `to_iter` (else the newest
 *  point, at most 300 iterations on). Null when it has none. */
export function trialBand(trial, newestIter) {
  const band = trial?.band;
  if (Array.isArray(band)) return safeBand(band);
  if (!band || typeof band !== 'object') return null;
  const centre = finite(band.centre) ? band.centre : band.expected, half = finite(band.half_width) ? band.half_width : band.halfWidth;
  const low = finite(band.low) ? band.low : finite(centre) && finite(half) ? centre - half : null;
  const high = finite(band.high) ? band.high : finite(centre) && finite(half) ? centre + half : null;
  const from = finite(band.from_iter) ? band.from_iter : trial.start_iter;
  if (!finite(low) || !finite(high) || low > high || !finite(from)) return null;
  const to = finite(band.to_iter) ? band.to_iter : Math.min(from + TRIAL_BAND.horizon, Math.max(from, finite(newestIter) ? newestIter : from));
  return safeBand(bandPoints(low, high, from, to));
}

function safeBand(points = []) {
  return points.filter(p => finite(p?.iter) && finite(p.low) && finite(p.high) && p.low <= p.high)
    .map(p => ({ ...p, low: clamp(p.low, 0, 100), high: clamp(p.high, 0, 100) })).sort((a, b) => a.iter - b.iter);
}

/** The chart's layout. Once cells have formed (points past `formingUntil`), the y range fits the settled points and
 *  the start ±3 points (the comparison that matters is readable), and the forming stretch is marked `forming` (drawn
 *  faint, the rise of cells growing from their nuclei is not "getting worse"); before that it starts at 0. The start's
 *  reference line is named after what it is, proseg's count at iteration 0 ("Iteration 0: 12", never "start 12" nor
 *  the cells the run started from). */
export const SETTLED_PAD = 3;
export function layoutChart({ series = [], start = null, formingUntil, changes = [], trial = null,
  width = 264, height = 186, xDomain, yDomain, selectedIter = null, kind = 'chart', padding } = {}) {
  width = Math.max(48, finite(width) ? width : 264); height = Math.max(14, finite(height) ? height : 186);
  const spark = kind === 'sparkline', lane = kind === 'timeline';
  const pad = padding ?? (spark ? { left: 2, right: 2, top: 2, bottom: 2 }
    : lane ? { left: 26, right: 12, top: 6, bottom: 18 } : { left: 26, right: 12, top: 14, bottom: 44 });
  const plot = { left: pad.left, right: Math.max(pad.left + 1, width - pad.right), top: pad.top,
    bottom: Math.max(pad.top + 1, height - pad.bottom) };
  const raw = seriesPoints(series);
  const startValue = finite(start) ? start : start?.per100;
  // Cells form during init and burn-in; the band ends at the first later point.
  if (formingUntil == null && raw.some(p => FORMING_PHASES.includes(p.phase))) {
    formingUntil = raw.find(p => p.phase && !FORMING_PHASES.includes(p.phase))?.iter ?? raw.at(-1)?.iter;
  }
  // The trial's band: the engine's when the trial carries one (an engine trial always says `band`: null when it has
  // none, and then none is drawn); the flat band drawn from the line (flatBand) only for a trial that does not say.
  let prediction = null, band = [];
  if (trial && canPredict(trial)) {
    band = trialBand(trial, raw.at(-1)?.iter) ?? [];
    if (!('band' in trial)) { prediction = flatBand(raw, trial, { formedIter: formingUntil }); band = safeBand(prediction?.points); }
  }
  const xmax = Math.max(1, ...[...raw, ...band].map(p => p.iter), ...changes.map(p => finite(p.iter) ? p.iter : 0));
  const xd = xDomain && xDomain[1] > xDomain[0] ? [...xDomain] : [0, xmax];
  const ys = raw.map(p => p.per100).filter(finite);
  if (finite(startValue)) ys.push(clamp(startValue, 0, 100));
  for (const p of band) ys.push(p.high, p.low);
  const settled = finite(formingUntil) ? raw.filter(p => p.iter > formingUntil && finite(p.per100)) : [];
  let yd = yDomain && yDomain[1] > yDomain[0] ? [...yDomain] : null;
  if (!yd) {
    if (spark) {
      const lo = Math.min(...ys, ys.length ? Infinity : 0), hi = Math.max(...ys, ys.length ? -Infinity : 1);
      yd = [Math.max(0, lo - .5), Math.min(100, hi + .5)];
    } else if (settled.length && kind === 'chart') {
      const values = [...settled.map(p => p.per100),
        ...band.flatMap(p => [p.low, p.high])];
      if (finite(startValue)) values.push(clamp(startValue, 0, 100));
      const lo = Math.max(0, Math.floor(Math.min(...values) - SETTLED_PAD)), hi = Math.min(100, Math.ceil(Math.max(...values) + SETTLED_PAD));
      yd = [lo, Math.max(hi, lo + 2)];
    } else {
      const high = Math.max(5, ...ys), step = niceStep(high, lane ? 2 : 3);
      yd = [0, Math.min(100, Math.ceil((high + .1) / step) * step)];
    }
  }
  const x = linearScale(xd, [plot.left, plot.right]), y = linearScale(yd, [plot.bottom, plot.top]);
  const project = p => ({ ...p, x: x(p.iter), y: finite(p.per100) ? y(p.per100) : null });
  const inDomain = p => p.iter >= xd[0] && p.iter <= xd[1];
  const points = raw.filter(inDomain).map(project);
  if (finite(formingUntil)) for (const p of points) p.forming = p.iter <= formingUntil;
  const formingBand = finite(formingUntil) && formingUntil > xd[0] ? {
    x0: x(xd[0]), x1: x(Math.min(xd[1], formingUntil)), y0: plot.top, y1: plot.bottom, label: C.chartForming,
  } : null;
  const ticks = changes.filter(p => finite(p.iter) && inDomain(p)).map(p => ({ ...p, x: x(p.iter), label: changeLabel(p) })).filter(t => t.label);
  const projectedBand = band.filter(inDomain).map(p => ({ ...p, x: x(p.iter), yLow: y(p.low), yHigh: y(p.high) }));
  return { kind, width, height, plot, x, y, xDomain: xd, yDomain: yd, points,
    hoverPoints: points.filter(p => finite(p.y)),
    // Iterations are whole numbers: no half-iteration ticks (a short run's [0, 1] would read "0 1 1").
    xTicks: spark ? [] : niceTicks(...xd, Math.max(1, Math.floor((plot.right - plot.left) / 65))).filter(Number.isInteger)
      .map(value => ({ value, x: x(value), label: number(value) })),
    yTicks: spark ? [] : niceTicks(...yd, lane ? 2 : 3).map(value => ({ value, y: y(value), label: number(value, value % 1 ? 1 : 0) })),
    start: finite(startValue) ? { value: startValue, y: y(clamp(startValue, 0, 100)), label: copy('startReference', { s: number(startValue) }) } : null,
    formingBand, changes: ticks, band: projectedBand, prediction,
    selected: finite(selectedIter) ? points.find(p => p.iter === selectedIter && finite(p.y)) ?? null : null,
  };
}

export function layoutSparkline(series, options = {}) { return layoutChart({ ...options, series, width: 48, height: 14, kind: 'sparkline' }); }
export function layoutTimeline(options = {}) { return layoutChart({ height: 58, ...options, kind: 'timeline' }); }

/** CSS-pixel x coordinate, closest judged result, ties choose the earlier frame. */
export function hoverAt(layout, px, { maxDistance = Infinity } = {}) {
  const points = layout.hoverPoints;
  if (!finite(px) || px < layout.plot.left || px > layout.plot.right || !points.length) return null;
  let lo = 0, hi = points.length;
  while (lo < hi) { const mid = (lo + hi) >>> 1; if (points[mid].x < px) lo = mid + 1; else hi = mid; }
  const right = points[Math.min(lo, points.length - 1)], left = points[Math.max(0, lo - 1)];
  const point = px - left.x <= right.x - px + 1e-9 ? left : right;
  if (Math.abs(point.x - px) > maxDistance) return null;
  const change = layout.changes.filter(c => c.iter <= point.iter).sort((a, b) => a.iter - b.iter).at(-1) ?? null;
  let text = copy(hoverKey(point), { i: number(point.iter), n: number(point.per100, 1), j: number(point.judged) });
  // The change with its values, under the Live knobs name where that name may be shown ("Cell compactness 0.04 → 0.02
  // (tighter shapes)"), else its plain name ("held closer to Xenium (1 → 4)").
  if (change && change.value === undefined) text = copy('chartAfterPlain', { hover: text, setting: change.label, k: number(change.iter) });
  else if (change) {
    const values = { hover: text, setting: change.label, from: settingValue(change.from), to: settingValue(change.value), k: number(change.iter) };
    text = KNOB_NAMES[change.flag] ? copy('chartAfterKnob', { ...values, knob: KNOB_NAMES[change.flag] }) : copy('chartAfter', values);
  }
  return { ...point, change, text };
}

/** The design tokens, resolved once per element: reading computed style on every update would make the browser
 *  recalculate style inside the snapshot's paint batch. */
const resolved = new WeakMap();
export function chartColors(element) {
  if (element && typeof element === 'object' && resolved.has(element)) return resolved.get(element);
  const style = globalThis.getComputedStyle?.(element);
  const read = (name, fallback) => style?.getPropertyValue?.(name)?.trim() || fallback;
  const colors = { ...QC_COLORS, ink: read('--ink-1', QC_COLORS.ink), muted: read('--ink-2', QC_COLORS.muted),
    grid: read('--hair', QC_COLORS.grid),
    unjudged: read('--unmapped', QC_COLORS.unjudged), accent: read('--qc-accent', QC_COLORS.accent) };
  if (element && typeof element === 'object') resolved.set(element, colors);
  return colors;
}

function path(ctx, points, stroke) {
  ctx.beginPath(); let connected = false;
  for (const p of points) {
    if (!finite(p.y)) { connected = false; continue; }
    if (connected) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y);
    connected = true;
  }
  ctx.strokeStyle = stroke; ctx.stroke();
}

/** An explicit draw only: no event listeners, timers, or animation frames.
 * segmentProgress reveals the newest segment and moves its dot in the same draw.
 */
export function renderChart(canvas, layout, { dpr = globalThis.window?.devicePixelRatio ?? globalThis.devicePixelRatio ?? 1,
  colors = QC_COLORS, segmentProgress = 1, endFrom = null } = {}) {
  dpr = finite(dpr) && dpr > 0 ? dpr : 1;
  const pixelWidth = Math.max(1, Math.round(layout.width * dpr)), pixelHeight = Math.max(1, Math.round(layout.height * dpr));
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const cssWidth = `${layout.width}px`, cssHeight = `${layout.height}px`;   // unchanged: no style write, no recalc
  if (canvas.style.width !== cssWidth) canvas.style.width = cssWidth;
  if (canvas.style.height !== cssHeight) canvas.style.height = cssHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) return layout;
  ctx.setTransform(pixelWidth / layout.width, 0, 0, pixelHeight / layout.height, 0, 0);
  ctx.clearRect(0, 0, layout.width, layout.height);
  const { plot, kind } = layout, spark = kind === 'sparkline';
  ctx.save();
  ctx.font = '10px system-ui'; ctx.lineWidth = 1; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
  if (layout.formingBand && !spark) {
    const b = layout.formingBand;
    ctx.fillStyle = colors.forming; ctx.fillRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
    ctx.save(); ctx.beginPath(); ctx.rect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0); ctx.clip();
    ctx.strokeStyle = colors.grid; ctx.beginPath();
    for (let x = b.x0 - layout.height; x < b.x1; x += 7) { ctx.moveTo(x, b.y1); ctx.lineTo(x + layout.height, b.y0); }
    ctx.stroke(); ctx.restore();
  }
  if (layout.band.length && !spark) {
    ctx.fillStyle = colors.band; ctx.beginPath();
    layout.band.forEach((p, i) => i ? ctx.lineTo(p.x, p.yHigh) : ctx.moveTo(p.x, p.yHigh));
    for (const p of [...layout.band].reverse()) ctx.lineTo(p.x, p.yLow);
    ctx.closePath(); ctx.fill();
  }
  if (!spark) {
    ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const tick of layout.yTicks) {
      ctx.fillStyle = colors.muted; ctx.fillText(tick.label, plot.left - 5, tick.y);
      ctx.strokeStyle = colors.grid; ctx.beginPath(); ctx.moveTo(plot.left, tick.y); ctx.lineTo(plot.right, tick.y); ctx.stroke();
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const tick of layout.xTicks) { ctx.fillStyle = colors.muted; ctx.fillText(tick.label, tick.x, plot.bottom + 5); }
  }
  ctx.save(); ctx.beginPath(); ctx.rect(plot.left - 3, plot.top - 3, plot.right - plot.left + 6, plot.bottom - plot.top + 6); ctx.clip();
  if (layout.start) {
    ctx.setLineDash([4, 4]); ctx.strokeStyle = colors.muted; ctx.beginPath();
    ctx.moveTo(plot.left, layout.start.y); ctx.lineTo(plot.right, layout.start.y); ctx.stroke(); ctx.setLineDash([]);
  }
  if (!spark) for (const tick of layout.changes) {
    ctx.strokeStyle = colors.muted; ctx.beginPath(); ctx.moveTo(tick.x, plot.bottom); ctx.lineTo(tick.x, plot.top); ctx.stroke();
  }
  const progress = clamp(segmentProgress, 0, 1), points = layout.points.map(p => ({ ...p }));
  let last = points.at(-1);
  if (!layout.selected && last && finite(last.y) && progress < 1) {
    const from = endFrom ?? points.at(-2);
    if (from && finite(from.y)) { last.x = from.x + (last.x - from.x) * progress; last.y = from.y + (last.y - from.y) * progress; }
  }
  ctx.lineWidth = spark ? 1.4 : 1.8;
  // The forming stretch faint (cells growing from their nuclei raise the number by themselves), the rest at full strength.
  const settledFrom = points.findIndex(p => !p.forming);
  if (!spark && settledFrom !== 0 && points.some(p => p.forming)) {
    ctx.save(); ctx.globalAlpha = .35; ctx.setLineDash([3, 3]);
    path(ctx, settledFrom < 0 ? points : points.slice(0, settledFrom + 1), colors.accent); ctx.restore();
    if (settledFrom > 0) path(ctx, points.slice(settledFrom), colors.accent);
  } else path(ctx, points, colors.accent);
  if (layout.selected) {
    last = { ...layout.selected };
    if (endFrom && finite(endFrom.y) && progress < 1) {
      last.x = endFrom.x + (last.x - endFrom.x) * progress;
      last.y = endFrom.y + (last.y - endFrom.y) * progress;
    }
  }
  if (last && finite(last.y)) {
    ctx.fillStyle = colors.accent; ctx.beginPath(); ctx.arc(last.x, last.y, spark ? 1.5 : 3, 0, Math.PI * 2); ctx.fill();
  }
  ctx.restore();
  if (!spark) {
    // The start's label at the left end, above its line (below it when there is no room): never on the line's end.
    ctx.fillStyle = colors.muted; ctx.textAlign = 'left';
    if (layout.start) {
      const above = layout.start.y - 3 - 10 >= plot.top;
      ctx.textBaseline = above ? 'bottom' : 'top';
      ctx.fillText(layout.start.label, plot.left + 3, above ? layout.start.y - 3 : layout.start.y + 3);
    }
    ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
    if (last && finite(last.per100) && finite(last.y)) {
      // The value beside the dot: to its left, or to its right while the dot is too near the axis labels. A whole number
      // is the engine's own (`shown`, as the headline: moved only past a boundary by 0.3), never a second rounding.
      const label = layout.band.length ? number(last.per100, 1) : number(Number.isInteger(last.shown) ? last.shown : last.per100);
      const left = last.x - 6 - (ctx.measureText(label)?.width ?? 0) >= plot.left;
      ctx.fillStyle = colors.ink; ctx.textBaseline = 'top'; ctx.textAlign = left ? 'right' : 'left';
      ctx.fillText(label, left ? last.x - 6 : last.x + 6, clamp(last.y + 5, plot.top, plot.bottom - 10));
      ctx.fillStyle = colors.muted; ctx.textAlign = 'right';
    }
    if (kind === 'chart') {
      ctx.textBaseline = 'top'; ctx.fillText(C.iteration, plot.right, layout.height - 11);
      ctx.textAlign = 'left';
      if (layout.formingBand) ctx.fillText(C.chartForming, plot.left, plot.bottom + 20);
      // Avoid overlapping change labels at panel width. Every tick remains hoverable.
      let edge = layout.formingBand ? plot.left + 78 : -Infinity;
      for (const tick of layout.changes) {
        const width = ctx.measureText(tick.label).width, x = clamp(tick.x - width / 2, plot.left, Math.max(plot.left, plot.right - width));
        if (x >= edge + 5) { ctx.fillText(tick.label, x, plot.bottom + 20); edge = x + width; }
      }
    }
  }
  ctx.restore();
  return layout;
}

export const renderSparkline = renderChart;
export const renderTimeline = renderChart;
