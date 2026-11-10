import { QC_TEXT as C, copy, number, likeForLike, heldClause, HEADLINE_MIN_JUDGED, ROUGH_JUDGED } from './qc-copy.mjs';
import { layoutSparkline, renderSparkline, chartColors, shownInteger, FORMING_PHASES } from './qc-chart.mjs';

// The stage chip under the HUD (number, change since the start and a 48 x 14 px
// sparkline; click opens the panel). Static: one synchronous draw per changed state, no scheduled frames. The change is
// the engine's like-for-like one (the summary's `change`), with an arrow only past its threshold and
// never the difference of two rounded numbers; a word held a few seconds keeps the size it was decided on (likeForLike).
// Its tooltip names the count the number reads (chipCountAveraged, or chipCountSaved on an iteration counted for a
// pause, the hold, a finish or the end, chipCountSavedChange there with a change beside it: each part's count), from
// which iteration a live number is smoothed (chipSmoothed), and what the change itself reads while its word is held
// (changeHeld).

/** States the chip shows. Off (the check's default: nothing is learned until it is switched on), unavailable and not
 *  yet known keep the stage clear, with no words of their own: the panel says why. */
export const CHIP_STATES = Object.freeze(['waiting', 'learning', 'waiting-start', 'live', 'paused', 'scrubbing', 'finished', 'failed']);
const integer = point => Number.isInteger(point?.shown) && point.shown >= 0 && point.shown <= 100 ? point.shown : shownInteger(point?.per100);

/** The chip's text, tooltip and sparkline from the panel's state (qc-panel.js fromView). */
export function chipModel(state = {}) {
  const status = state.enabled === false ? 'off' : state.state ?? 'loading';
  const visible = CHIP_STATES.includes(status);
  const point = state.atStart ? (Number.isFinite(state.start?.per100) ? state.start : null) : state.summary;
  let text = '', tooltip = C.openPanel, shown = null;
  if (status === 'waiting') text = C.chipWaiting;
  else if (status === 'learning') {
    const of = Math.max(1, state.progress?.of ?? 50), step = state.progress?.step ?? 0;
    text = copy('chipLearning', { pct: number(Math.max(0, Math.min(100, step / of * 100))) });
  } else if (status === 'waiting-start') text = C.chipStart;
  else if (status === 'failed') text = C.chipFailed;
  else if (visible && !Number.isFinite(point?.per100)) text = C.chipFrame;
  // Too few judged cells for a number (one judged cell of three read "0 in 100").
  else if (visible && Number.isFinite(point.judged) && point.judged < HEADLINE_MIN_JUDGED) text = C.chipTooFew;
  else if (visible) {
    shown = integer(point);
    // While cells form (their phase, or the start line's areas still shown) the number rises by itself: no arrow.
    const forming = !state.atStart && (FORMING_PHASES.includes(point.phase) || state.areaView === 'start');
    const start = state.atStart || state.startSource === 'nuclei' ? null : state.start;
    const change = Number.isFinite(start?.per100) ? likeForLike(point.change) : null;
    const values = { n: number(shown), d: change?.d ?? '' };
    // ... and with no start line (a start from nuclei, alone or with their cells) nothing to compare with later either.
    if (forming) { text = copy('chipForming', values); tooltip = copy(Number.isFinite(state.start?.per100) ? 'chipTooltipForming' : 'chipTooltipFormingNoStart', values); }
    else if (!change) { text = copy('chipNumber', values); tooltip = `${copy('timelineHover', values)}. ${C.openPanel}`; }
    else {
      // "like for like" beside the raw number, and a hollow arrow for a change another learning of the model could
      // read the other way (the panel's "by this model's reading").
      const way = change.direction === 'more' ? 'More' : change.direction === 'fewer' ? 'Fewer' : null;
      text = copy(way ? `chip${way}${change.light ? 'Light' : ''}` : 'chipSame', values);
      tooltip = copy(way ? `chipTooltip${way}${change.light ? 'Light' : ''}` : 'chipTooltipSame', values);
      // Its thresholds are this kind of run's constants: said while not yet re-measured on proseg's count.
      if ((point.constants ?? state.summary?.constants)?.measured === false) tooltip = `${tooltip} ${C.notRemeasured}`;
    }
    // A live number: from which iteration, smoothed; a word held a few seconds: what the change itself reads now (an own
    // or saved change is never held); then which count it reads: on a saved iteration each part's (the number the count
    // proseg saves, a change beside it the averaged count, like for like).
    const smoothed = !state.atStart && point.single === false, saved = !state.atStart && point.count === 'sampled';
    tooltip = [tooltip, smoothed && Number.isFinite(point.iter) ? copy('chipSmoothed', { i: number(point.iter) }) : '', !forming && change ? heldClause(point.change) : '',
      saved ? (!forming && change ? C.chipCountSavedChange : C.chipCountSaved) : C.chipCountAveraged].filter(Boolean).join(' ');
    // Under 300 judged cells the number moves a lot by chance: its tooltip says so first.
    if (Number.isFinite(point.judged) && point.judged < ROUGH_JUDGED) tooltip = `${C.roughNumber} ${tooltip}`;
    // Looking back, a result of another iteration than the frame on screen (the nearest with proseg's count) says which,
    // after the frame on screen's own iteration.
    if (!state.atStart && Number.isFinite(state.snapped)) tooltip = `${copy('snappedTooltipMixing', { i: number(state.snapped) })} ${tooltip}`;
    if (status === 'scrubbing') tooltip = `${copy('notNewest', { i: number(Number.isFinite(state.frameIter) ? state.frameIter : point.iter) })} ${tooltip}`;
  }
  const series = state.series ?? [];
  const upTo = ['scrubbing', 'paused'].includes(status) && point && !state.atStart ? series.filter(p => p.iter <= point.iter) : series;
  return { status, visible, text, tooltip, shown,
    sparkline: shown == null || !upTo.length ? null : layoutSparkline(upTo, { start: state.atStart ? null : state.start }) };
}

/** Static HUD button: onClick() normally opens and scrolls to the panel. setState(state) with qc-panel.js's state. */
export class QcChip {
  constructor(host, { onClick = () => {} } = {}) {
    this.host = host; this.onClick = onClick;
    const doc = host.ownerDocument ?? globalThis.document;
    this.root = doc.createElement('button'); this.root.type = 'button'; this.root.className = 'qc-chip';
    this.label = doc.createElement('span'); this.canvas = doc.createElement('canvas');
    this.canvas.setAttribute('aria-hidden', 'true'); this.canvas.className = 'qc-sparkline';
    this.root.append(this.label, this.canvas); host.append(this.root);
    this.colors = chartColors(this.canvas);   // read once, now: never a style recalculation in a result's paint batch
    this.root.onclick = () => { if (!this.disposed) this.onClick(this.state); };
    this.setState({ state: 'loading' });
  }
  get visible() { return !!this.model?.visible; }
  setState(state) {
    if (this.disposed) return;
    const key = JSON.stringify({ state: state.state, enabled: state.enabled, atStart: state.atStart, start: state.start?.per100,
      startJudged: state.start?.judged, startSource: state.startSource, progress: state.progress, seq: state.summary?.seq,
      shown: state.summary?.shown, per100: state.summary?.per100, judged: state.summary?.judged, iter: state.summary?.iter,
      phase: state.summary?.phase, change: state.summary?.change ?? null, count: state.summary?.count ?? null, single: state.summary?.single ?? null,
      areaView: state.areaView, snapped: state.snapped ?? null,
      frameIter: state.frameIter ?? null, series: state.seriesKey ?? state.series?.length });
    if (key === this.key) return;
    this.key = key; this.state = state; this.model = chipModel(state);
    this.root.dataset.state = this.model.status;
    if (this.host.hidden !== !this.model.visible) this.host.hidden = !this.model.visible;
    if (this.label.textContent !== this.model.text) this.label.textContent = this.model.text;
    this.root.title = this.model.tooltip; this.root.setAttribute('aria-label', `${this.model.text}. ${this.model.tooltip}`);
    this.canvas.hidden = !this.model.sparkline;
    if (this.model.sparkline) renderSparkline(this.canvas, this.model.sparkline, { colors: this.colors ??= chartColors(this.canvas) });
  }
  dispose() {
    if (this.disposed) return;
    this.disposed = true; this.root.onclick = null; this.root.remove();
  }
}
