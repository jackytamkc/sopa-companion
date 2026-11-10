import { QC_TEXT as C, RULE_COPY, copy, number, duration, list, startName, settingValue, changeLabel, bannedWords, reasonText, unavailableTooltip,
  knobName, knobChange, settingLine, settingShort, cardKey, tryingLabel, triedLabel, likeForLike, verdictLine, heldClause, countWord, coverageOf,
  retentionOf, retentionLine, retentionTooltip, retentionSentence, learnedLine, HOW_KEYS, sameSetting, HEADLINE_MIN_JUDGED, LOW_COVERAGE, ROUGH_JUDGED,
  testReasonText, testReasonGroup, TEST_RUN_REFUSALS } from './qc-copy.mjs';
import { layoutChart, renderChart, chartColors, hoverAt, shownInteger, FORMING_PHASES, TRIAL_BAND } from './qc-chart.mjs';
import { problem, EXPLAIN_TEXT } from './explain.mjs';

// The Mixing check panel: one switch in its header; the headline ("11 in 100"
// "cells look mixed") with an ⓘ that says what it means (a model's estimate, not a count of boundary errors; blind to
// mixing between cells of the same kind), what it rests on (`coverage`, read on proseg's own count, averaged over its
// random background draw) and the saved proseg results' own number; a badge when the result shown is another
// iteration's than the frame on screen (`snapped`: the nearest with proseg's count); one verdict
// line, the like-for-like change since the start beside the start's number (the engine's `change`, an arrow only past
// its threshold; only for a run that started from cells alone); the transcripts the cells kept only when that moves the
// number by itself (`retention`); the chart
// over the run, "Show on the cells" and "How is this estimated?" (five points and how the model was learned). No flagged
// area is shown (no list, letter or "No area stands out"): the engine's areas still feed its suggestions and trials,
// whose words say "where mixing stands out". Words on screen are a number, a few-word verdict and labels; explanations
// are in tooltips. Above them all, the selected cell's row: "Go deeper ›" and its mixing line (setCell), and under it
// "Test a setting here…", whose test is shown in a corner of the stage (testModel). It renders a plain state object
// (fromView: from qc-client.mjs's view) and owns only the nodes inside its <details>; every action is a callback.
// Which count a number reads is said where it is shown (the summary's `count`): the basis line names it ("proseg's count
// (averaged)", or "(as saved)" on an iteration counted for a pause, the hold, a finish or the end, whose own numbers the
// engine scores on the count proseg saves there), and the headline's ⓘ says which iteration its number is from and
// whether it is smoothed (live: the newest proseg has counted, with the ones before it) or its own (looking back at a
// result judged alone, or a resting run's newest: qc-client.mjs restSeq); the verdict's tooltip says when its word is
// held a few seconds (change.held) and what the change itself reads now.
// Suggestions (options.suggestions, which main.js sets): one suggestion card at a time with its "More ideas", the
// just appeared slides in once (ENTRANCE). Motion is change-driven only (the number counts to a new integer over
// NUMBER_MS, the newest line segment draws on over SEGMENT_MS) and only when the state allows it (state.motion: updates
// at least ~1 s apart), never with reduced motion (or the calm-motion switch), a collapsed panel, a hidden page or one
// Compare covers (setCovered). Its frames fall on the steps of the stage's own motion for the same result
// (MOTION_STEP_MS: the cell tint's 150 ms steps), so one result costs at most 4 paints in all.
export const PANEL_STATES = Object.freeze(['off', 'waiting', 'learning', 'waiting-start', 'live', 'paused', 'scrubbing', 'finished', 'unavailable', 'failed']);
export const MOTION_STEP_MS = 150;
export const NUMBER_MS = 450;
export const SEGMENT_MS = 300;
export const MAX_MOTION_FRAMES = 3;
export const CHOSEN_GRACE_MS = 10000;    // an idea picked under "More ideas" stays shown this long when a result omits it
export const CHART_HEIGHT = 132;
export const JUDGE_MIN = 100;
const finite = Number.isFinite;
const noop = () => {};
const valueText = value => finite(value) ? String(value) : '';   // a number as a field holds it ('' for none)
const keyOf = value => JSON.stringify(value);
const setText = (node, text) => { if (node.textContent !== text) node.textContent = text; };
const setHidden = (node, hidden) => { if (node.hidden !== hidden) node.hidden = hidden; };
const safeText = (value, fallback) => typeof value === 'string' && !bannedWords(value).length && !/resolvi/i.test(value) ? value : fallback;
/** The ⓘ icon (inline SVG, one stroke weight, drawn in the text's colour by qc.css; the test corner's own,
 *  qc-test-corner.mjs ICONS.info). */
const INFO_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true" focusable="false"><circle cx="8" cy="8" r="6.25"></circle>'
  + '<path d="M8 7.25v4M8 4.75v.01"></path></svg>';
/** A run with no start line: one that started from nuclei only, or from cells with their nuclei (the engine's
 *  start_kind "paired": proseg's count at iteration 0 holds the nuclei alone); an engine that names no start_kind, by
 *  where the cells started alone. */
const noStart = state => state.startSource === 'nuclei' || (typeof state.startKind === 'string' && state.startKind !== 'cells');
/** A run whose cells form from nuclei: started from nuclei only, or from cells with their nuclei (start_kind "paired");
 *  a start from cells only has none, and a start not known yet is said in words true for every start. */
const fromNuclei = state => state.startSource === 'nuclei' || state.startKind === 'nuclei' || state.startKind === 'paired';
/** The not-re-measured caveat (notRemeasured) for a number resting on this kind of run's constants (the arrow, the
 *  trial's verdict, the chip's change) while the shown summary says they are not yet re-measured on proseg's count
 *  (constants.measured false: engine/qc_constants.py); '' otherwise. */
export const notRemeasuredOf = (point, state) => (point?.constants ?? state?.summary?.constants)?.measured === false ? C.notRemeasured : '';
/** Cells still forming (their phase, or the engine's area_view "start": the start line's areas until burn-in + 20): the
 *  number rises by itself, so it is not compared with the start yet. */
export const formingPoint = (point, state) => !!point && !state.atStart && (FORMING_PHASES.includes(point.phase) || state.areaView === 'start');
const dateText = value => { const d = new Date(value); return Number.isNaN(d.getTime()) ? C.missing : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' }); };
/** The headline's integer: the engine's own (moved only past a boundary by 0.3), else the value rounded. */
export const headlineInteger = point => Number.isInteger(point?.shown) && point.shown >= 0 && point.shown <= 100 ? point.shown : shownInteger(point?.per100);
/** The headline ⓘ's first line (H2, H10): which iteration the number is from and how: the start line's own (iteration
 *  0), a result's own number (single: a saved iteration's, one judged alone, a resting run's newest), or a smoothed one,
 *  live (`status` "live": the newest proseg has counted, with the 0.3 margin of its whole number) or looked back at as
 *  it was shown live; '' for a result that says neither. */
export function numberLine(point, status, atStart = false) {
  if (atStart) return copy('numberOwn', { i: number(0) });
  if (!point || !finite(point.iter)) return '';
  if (point.single === true) return copy('numberOwn', { i: number(point.iter) });
  return point.single === false ? copy(status === 'live' ? 'numberLive' : 'numberPast', { i: number(point.iter) }) : '';
}

/** The run's view with the engine's word on the check as a whole (GET /health `qc`: {available, state, reason}):
 *  where Clusters and the UMAP cannot run (reason "analysis"), nothing is ever scored here, so the panel says so
 *  whatever the sample's own state, with the switch off as on (it stays pressable: a press turns the run's check on and
 *  the panel says the same). The check's other reasons (no graphics card it can use, its packages) come with each run's
 *  own answer (sample_reason), newer than the page's /health. */
export function withEngine(view, qc) {
  if (!view || qc?.available !== false || qc?.reason !== 'analysis') return view;
  return { ...view, sampleState: 'unavailable', sampleReason: 'analysis',
    ...(view.enabled ? { status: 'unavailable', reason: 'analysis' } : {}) };
}

/** The panel's state from qc-client.mjs's view and what the page knows (sample name, colours on, a click on its way, how
 *  to update an old proseg-live: prosegFix, and in which session: notebook). */
export function fromView(view, page = {}) {
  if (!view) return { state: 'loading', ...(page.failed ? { failed: true } : {}) };
  // Not the frame's own iteration (it changes with every snapshot: the panel would render each time), but while the result
  // shown is another iteration's (`snapped`: only looking back), the frame's, which the status line names then.
  return { state: view.status, enabled: view.enabled, runId: view.runId, model: view.model, summary: view.summary, start: view.start,
    startSource: view.startSource, startKind: view.startKind ?? null, snapped: view.snapped ?? null,
    ...(view.snapped != null ? { frameIter: view.frameIter ?? null } : {}), atStart: view.atStart, reason: view.reason, progress: view.progress, eta_s: view.eta_s,
    shared: view.shared, relearn: view.relearn, requested: view.requested, restarting: view.restarting, failed: !!view.failed,
    sampleState: view.sampleState ?? null, sampleReason: view.sampleReason ?? null, room: view.room ?? null,
    series: view.line, seriesKey: view.lineVersion, changes: view.ticks, formingUntil: view.formingUntil,
    // The result's areas (the engine's, never shown) for the suggestion cards' own numbers and "Look at
    // its cells", and the engine's area_view (cells still form while it is "start").
    areas: view.areas, areaView: view.areaView, batch: view.batch,
    // The model, for "How is this estimated?" (no other number from outside this app).
    runState: view.runState, sampleModel: view.sampleModel,
    suggestion: view.suggestion ?? null, moreIdeas: view.moreIdeas ?? [], trial: view.trial ?? null, trialEnded: view.trialEnded ?? null,
    tiled: view.tiled, motion: view.motion, still: view.still, ...page };
}

// The suggestion cards.
/** Shape knobs: "New run with this setting" is their main action once cells have formed (a live change then shows only
 *  after ~1,000 iterations); *Try it* stays the main action while cells form. */
export const SHAPE_RULES = Object.freeze(['R2', 'R4', 'R5']);
/** Rules whose card changes a setting (U puts back the old value). */
export const SETTING_RULES = Object.freeze(['U', 'R2', 'R3', 'R4', 'R5']);
/** The card's entrance: it slides and fades in once, on the stage's 150 ms steps (opacity, px down), then stays. */
export const ENTRANCE = Object.freeze([[.35, 8], [.75, 3], [1, 0]]);
/** The classes of the paragraphs under a card's Why? (suggestionModel `more`), by kind. */
const MORE_CLASSES = Object.freeze({ detail: 'qc-card-detail', cost: 'qc-cost', timing: 'qc-timing', evidence: 'qc-card-evidence',
  footer: 'qc-small qc-muted qc-card-footer', count: 'qc-small qc-muted qc-card-count' });
/** The timing lines whose figures come from the outline-era tests (H5: their tooltip says so); R5's untested line has
 *  none. */
const OUTLINE_TIMING = Object.freeze(['timingEarly', 'timingLater', 'timingLaterNoTime']);
const lettersOf = s => Array.isArray(s?.areas) ? s.areas.filter(L => typeof L === 'string') : typeof s?.area === 'string' ? [s.area] : [];
const flagOn = (s, key) => s?.[key] === true || s?.[key.replace(/[A-Z]/g, c => `_${c.toLowerCase()}`)] === true;
const actionName = a => ({ new_run: 'new-run', newRun: 'new-run', 'new-run': 'new-run', try: 'try', inspect: 'inspect' })[a] ?? null;
const verdictKey = v => {
  const s = String(v ?? '').toLowerCase().replace(/[\s-]+/g, '_');
  return ['less', 'less_mixed'].includes(s) ? 'less' : ['more', 'more_mixed'].includes(s) ? 'more'
    : ['no_change', 'no_clear_change', 'no_change_after', 'none_after', 'unclear', 'done'].includes(s) ? 'none' : null;
};
const noteOn = (item, key) => {
  const notes = item?.notes ?? [];
  return item?.[key] === true || (Array.isArray(notes) ? notes.includes(key) : !!notes?.[key]);
};

/** A suggestion (the engine's summary `suggestion`, engine/qc_rules.py evaluate: {id, rule, areas, flag?, from?, to?,
 *  confidence, timing "early" | "later", actions, main, later_s?} and the numbers its sentence needs: v, growth, rose, k,
 *  tried, changed, before_per100, after_per100) as the card's words. In view (short cards) only the title, one short
 *  why (`why`), the Setting line (`setting`, its hint in its ⓘ: `settingHint`), a short confidence and the buttons; the
 *  rest is `more`, the paragraphs under Why? in order ({kind, text}: `detail` the why's other sentences, R2's "the gain
 *  may be small" ('' when an area rose), then `cost`, `timing`, `evidence` and the `footer`). The why uses its areas'
 *  own numbers (growth a fraction: .18 = 18 %; a card for several areas gives their range, and R5 their squeezed cells
 *  summed) and calls them "where mixing stands out", never by their letters (no area is shown, nor a "for area A"
 *  line); the Setting line names the knob as Live knobs does (`knobLabel(flag)`, the page's option schema), a switch as
 *  its checkbox; U's `from` is the value now and `to` the one Undo puts back. `phase`: the phase of the result shown
 *  (the timing line and the main action, when the card does not say); `ended`: the run has ended (a new run is all that
 *  is left, for every setting but U); `laterTime`: what 1,000 iterations take in this run (the card's own later_s
 *  first); `inspect` false: the page cannot move the view to an area (no "Look at its cells"). Figures from the tests
 *  made while the check counted the RNA inside the drawn outlines say so in their tooltips (H5, outlineEraTooltip:
 *  `confidenceTip`, each `more` paragraph's `tip`): a card of RULE_COPY era "outline" on its sized confidence line and
 *  its tests under Why?, the shape cost and timing lines on every card. A card of a result scored on the count proseg
 *  saves (the summary's count "sampled": a pause, the hold, a finish or the end) ends its Why? with what its numbers
 *  read (cardAveraged: the averaged count, smoothed, as every card's areas). */
export function suggestionModel(suggestion, state = {}, { knobLabel = null, phase = null, ended = false, laterTime = null, inspect = true } = {}) {
  const rule = RULE_COPY[suggestion?.rule];
  if (!rule) return null;
  const id = suggestion.rule, summary = state.summary ?? {}, letters = lettersOf(suggestion);
  const known = state.areas ?? summary.areas ?? [], areas = letters.map(L => known.find(a => a.L === L)).filter(Boolean);
  const area = areas[0] ?? {};
  const label = (typeof knobLabel === 'function' ? knobLabel(suggestion.flag) : null) ?? suggestion.label ?? null;
  const toggle = typeof suggestion.to === 'boolean' || typeof suggestion.from === 'boolean';
  // U: "after you changed Cell compactness" (the knob's Live knobs name, else "this setting": its Setting line names it);
  // a switch by its plain change ("after you turned one-piece cells off"). R7 names what was tried ("You tried tighter
  // shapes here").
  const knob = safeText(knobName(label) ?? (suggestion.flag === 'cell-compactness' ? C.settingCompactness : null), null);
  const changed = suggestion.changed ?? { flag: suggestion.flag, from: suggestion.to, to: suggestion.from };
  const turned = id === 'U' && toggle ? changeLabel({ flag: changed.flag, from: changed.from, to: changed.to }) : '';
  const named = id === 'U' ? knob ?? C.thisSetting : id === 'R7' && suggestion.tried ? triedLabel(suggestion.tried) : changeLabel(suggestion);
  // The areas' own numbers: one area's, or the range (growth) or the sum (squeezed cells) of every area the card names.
  const own = finite(suggestion.growth) ? suggestion.growth : area.growth;
  const growths = (areas.length > 1 ? areas.map(a => a.growth) : [own]).filter(finite).map(g => Math.round(Math.abs(g) * 100));
  const low = growths.length ? Math.min(...growths) : null, high = growths.length ? Math.max(...growths) : null;
  const squeezed = a => finite(a.tiny) ? a.tiny - (finite(a.tiny_start) ? a.tiny_start : 0) : a.shrunk;
  const tinies = areas.length > 1 ? areas.map(squeezed).filter(finite) : [];
  const tiny = tinies.length > 1 ? tinies.reduce((n, k) => n + k, 0) : finite(suggestion.k) ? suggestion.k : squeezed(area);
  const values = { setting: named || C.missing, where: letters.length ? C.standsOut : C.inTissue, change: turned || C.missing,
    g: low == null ? C.missing : low === high ? number(low) : `${number(low)}–${number(high)}`, k: number(tiny),
    v: settingValue(suggestion.v ?? suggestion.from),
    a: number(suggestion.before_per100 ?? suggestion.expected ?? area.start_per100, 1), b: number(suggestion.after_per100 ?? suggestion.now ?? area.per100, 1) };
  // A card that offers no setting (R1, R6, R7: RULE_COPY `own`) keeps its own line, what was seen or where to look, never
  // the engine's "unlikely to help"; the others take the engine's confidence when it is one the cards use.
  const engineConfidence = Object.hasOwn(C, suggestion.confidence) && Object.values(RULE_COPY).some(r => r.confidence === suggestion.confidence);
  const confidence = rule.own ? C[rule.confidence] : rule.sized && (!engineConfidence || suggestion.confidence === rule.confidence) ? C[rule.sized]
    : engineConfidence ? C[suggestion.confidence] : C[rule.confidence];
  const hasSetting = SETTING_RULES.includes(id) && typeof suggestion.flag === 'string';
  const startupOnly = flagOn(suggestion, 'startupOnly');
  const forming = ['early', 'later'].includes(suggestion.timing) ? suggestion.timing === 'early' : FORMING_PHASES.includes(phase);
  const shape = SHAPE_RULES.includes(id);
  if (finite(suggestion.later_s) && suggestion.later_s > 0) laterTime = duration(suggestion.later_s);
  // The actions this card offers (the engine names them): Try it only while the run can change, New run for a shape
  // knob, a setting that cannot change during a run, or any setting once the run has ended (U excepted: nothing to put
  // back then); the main one first.
  const canTry = hasSetting && id !== 'U' && !ended && !startupOnly;
  const canNew = hasSetting && id !== 'U' && (shape || startupOnly || ended);
  // "Look at its cells" (the engine's `inspect`: R1, R6, R7, whose cause is not known): the view moves to its area,
  // where a cell can be looked at with Go deeper; only for a card with an area, and only on a page that can move there
  // (`inspect` false: the page has no area to fly to).
  const canInspect = inspect !== false && letters.length > 0;
  let actions = Array.isArray(suggestion.actions) ? suggestion.actions.map(actionName).filter(Boolean) : ['try', 'new-run'];
  if (ended && canNew && !actions.includes('new-run')) actions.push('new-run');
  actions = actions.filter((a, i) => actions.indexOf(a) === i && ((a === 'try' && canTry) || (a === 'new-run' && canNew) || (a === 'inspect' && canInspect)));
  const main = suggestion.main ?? suggestion.primary, wanted = actionName(main);
  // The engine's main action wins; R1's is its starting outlines (a button of its own): none of these is then the main.
  const primary = actions.includes(wanted) ? wanted : main === 'show_start' ? null
    : actions.includes('new-run') && (ended || !forming || !actions.includes('try')) ? 'new-run' : actions.find(a => a !== 'inspect') ?? actions[0] ?? null;
  actions.sort((a, b) => (b === primary) - (a === primary));
  const laterKey = laterTime ? 'timingLater' : 'timingLaterNoTime', later = laterTime ? copy('timingLater', { t: laterTime }) : C.timingLaterNoTime;
  const setting = hasSetting ? settingShort({ flag: suggestion.flag, from: suggestion.from, to: suggestion.to, label }) : null;
  // R2 says what is true of its areas: grown past the start, else got more mixed; and when none of them rose, that the
  // gain may be small there (under Why?).
  const rose = areas.length ? areas.some(a => a.tag === 'rose') : suggestion.rose === true;
  const body = id === 'R2' ? (finite(high) && high > 10 ? 'ruleR2Body' : 'ruleR2BodyRose') : id === 'U' && turned ? 'ruleUBodySwitch' : rule.body;
  const why = copy(body, values);
  const detail = id === 'R2' ? (rose ? '' : C.ruleR2NotRose) : rule.more ? copy(rule.more) : null;
  // Try it's tooltip: a switch by what it does ("Turns drift on ..."), never "Sets Switch off ... to off".
  const tooltip = !hasSetting ? null : toggle
    ? copy('tryTooltipSwitch', { change: changeLabel({ flag: suggestion.flag, from: suggestion.from, to: suggestion.to }) || C.missing,
      back: changeLabel({ flag: suggestion.flag, from: suggestion.to, to: suggestion.from }) || C.missing }) + (id === 'R3' ? ` ${C.driftActsLater}` : '')
    : copy('tryTooltip', { label: safeText(knobName(label), C.thisSetting), value: settingValue(suggestion.to), old: settingValue(suggestion.from) });
  const whole = !letters.length;
  const footer = !hasSetting ? C.suggestionFooterInfo : ended ? (whole ? C.suggestionFooterEndedWhole : C.suggestionFooterEnded)
    : primary === 'new-run' ? C.suggestionFooterNewRun : id === 'R3' ? C.suggestionFooterDrift : whole ? C.suggestionFooterWhole : C.suggestionFooter;
  const cost = ['R2', 'R4'].includes(id) ? C.shapeCost : null;
  // The timing test was of cell shapes: R5 (the hold) was never tested during a run.
  const timingKey = shape && hasSetting && !ended ? (id === 'R5' ? 'timingUntested' : forming ? 'timingEarly' : laterKey) : null;
  const timing = timingKey === 'timingLater' ? later : timingKey ? C[timingKey] : null;
  const evidence = rule.why ? copy(rule.why) : null;
  const outline = rule.era === 'outline' ? C.outlineEraTooltip : '';
  // Under Why?, in the order the card once showed them in view: the why's other sentences, the cost, the timing, the
  // tests behind it, the footer's caveats, then (a result scored on the count proseg saves) what its numbers read. Which
  // paragraphs a card has depends on its rule, on whether the run ended and on the result's count only (R2's detail is
  // there, empty, when an area rose), so new numbers never rebuild it.
  const more = [detail == null ? null : { kind: 'detail', text: detail }, cost && { kind: 'cost', text: cost, tip: C.outlineEraTooltip },
    timing && { kind: 'timing', text: timing, tip: OUTLINE_TIMING.includes(timingKey) ? C.outlineEraTooltip : '' },
    evidence && { kind: 'evidence', text: evidence, tip: outline }, { kind: 'footer', text: footer },
    summary.count === 'sampled' && { kind: 'count', text: C.cardAveraged }].filter(Boolean);
  return { source: suggestion, rule, id, key: cardKey(suggestion), title: copy(rule.title),
    why, detail, setting: setting?.text ?? null, settingHint: setting?.hint ?? null, flag: hasSetting ? suggestion.flag : null, confidence,
    confidenceTip: rule.sized && confidence === C[rule.sized] ? outline : '', cost, timing, evidence, more, tooltip,
    hasSetting, actions, primary, undo: id === 'U' && hasSetting && !ended, showStart: id === 'R1', footer,
    // The area "Look at its cells" (and R1's starting outlines) moves the view to: the card's first, its outline the
    // result's (never drawn).
    inspectArea: letters.length ? (known.find(a => a.L === letters[0]) ?? { L: letters[0] }) : null };
}

/** A trial row {kind: 'area' | 'whole' | 'depth', area?, expected, now} ("before"/"after" too); one decimal when they
 *  differ by less than 2 (2.7 "Headline rules"). An area's row is "Where mixing stands out", never its letter. */
export function trialRowText(row) {
  const expected = finite(row.expected) ? row.expected : row.before, now = finite(row.now) ? row.now : row.after;
  const kind = ['depth', 'transcripts'].includes(row.kind) ? 'depth' : ['whole', 'tissue'].includes(row.kind) ? 'whole' : 'area';
  const digits = kind !== 'depth' && finite(expected) && finite(now) && Math.abs(expected - now) < 2 ? 1 : 0;
  return copy(kind === 'depth' ? 'trialDepth' : kind === 'whole' ? 'trialWhole' : 'trialArea', { a: number(expected, digits), b: number(now, digits) });
}

/** A trial row's arrow ('down' | 'up' | '': an arrow only for a clear change, never for any difference of two
 *  numbers): on the row the engine judged (the area's own when its verdict is the area's, else the whole tissue's),
 *  and only while that verdict is less or more mixed; transcripts per cell (a fact, not a verdict) from a 5 % change on. */
export function trialRowDirection(trial, row) {
  const expected = finite(row?.expected) ? row.expected : row?.before, now = finite(row?.now) ? row.now : row?.after;
  if (!finite(expected) || !finite(now)) return '';
  const kind = ['depth', 'transcripts'].includes(row.kind) ? 'depth' : ['whole', 'tissue'].includes(row.kind) ? 'whole' : 'area';
  if (kind === 'depth') return expected > 0 && Math.abs(now - expected) >= 0.05 * expected ? (now < expected ? 'down' : 'up') : '';
  const verdict = verdictKey(trial?.verdict);
  if (!['less', 'more'].includes(verdict) || kind !== (trial?.on === 'area' ? 'area' : 'whole')) return '';
  return verdict === 'less' ? 'down' : 'up';
}

/** Whether a trial's line is a verdict (less or more mixed, or no change after its iterations): one resting on the
 *  trial minima and band, this kind of run's constants, never a note's line (forming, too few, the last stage...). */
export const trialVerdictReached = trial => !!trial && !['forming', 'mixed_up', 'no_history', 'no_band', 'no_matched'].some(key => noteOn(trial, key))
  && (['less', 'more'].includes(verdictKey(trial.verdict)) || (verdictKey(trial.verdict) === 'none' && !noteOn(trial, 'last_stage')));

/** The engine judges a setting change; the page explains its verdict and any limits. */
export function trialVerdictText(trial, { k = 0, laterTime = null, formed = false } = {}) {
  if (!trial) return '';
  if (noteOn(trial, 'forming')) return formed ? C.trialFormedPast : C.trialForming;
  if (noteOn(trial, 'mixed_up')) return C.trialMixedUp;
  if (noteOn(trial, 'no_history') || noteOn(trial, 'no_band')) return C.trialNoHistory;
  if (noteOn(trial, 'no_matched')) return C.trialNoMatched;
  const verdict = verdictKey(trial.verdict), area = trial.on === 'area' && (trial.area ?? lettersOf(trial)[0] ?? null) != null;
  if (verdict === 'less') return area ? C.trialLessArea : C.trialLess;
  if (verdict === 'more') return area ? C.trialMoreArea : C.trialMore;
  if (noteOn(trial, 'last_stage')) return C.trialLastStage;
  const later = trial.timing !== 'early', steps = number(finite(trial.iterations) ? trial.iterations : Math.min(k, TRIAL_BAND.horizon));
  if (verdict === 'none') return !later ? copy('trialNoChangeAfter', { k: steps })
    : laterTime ? copy('trialNoChangeAfterLater', { k: steps, t: laterTime }) : copy('trialNoChangeAfterLaterNoTime', { k: steps });
  if (later) return laterTime ? copy('trialPendingLater', { t: laterTime }) : C.trialPendingLaterNoTime;
  return copy('trialNoChange', { timing: laterTime ? copy('timingLater', { t: laterTime }) : C.timingLaterNoTime });
}
/** Whether a trial's line (trialVerdictText) gives the timing figures of the outline-era tests (H5: its tooltip says
 *  so): a change made after cells formed waiting or done without a clear change, and the timing line of one made while
 *  they formed; never a verdict or a note's line. */
export function trialVerdictOutline(trial) {
  if (!trial || ['forming', 'mixed_up', 'no_history', 'no_band', 'no_matched'].some(key => noteOn(trial, key))) return false;
  const verdict = verdictKey(trial.verdict);
  if (verdict === 'less' || verdict === 'more' || noteOn(trial, 'last_stage')) return false;
  return verdict !== 'none' || trial.timing !== 'early';
}

// "Test a setting here…": does a setting take the pink suspects of this cell, and of nearby cells with the same pattern,
// out of their cells beyond what it takes of their own RNA at the same spot? The engine fixes the suspects before any run
// (GET /runs/{id}/qc/test/pool previews them, with why a test can't run here, what it would wait for first and where it
// was not measured), then runs the area afresh to the end of the schedule, as now and with the value chosen, all at once,
// and reads proseg's own file of each run. Its answer (result v2, engine/qc_rule.py) says whether the suspects come
// back, whether they are clearly fewer, what it cost, where they went and this cell alone, and where it was not measured
// (in tooltips); the page says it in the deck's words, never working a verdict out again.
// Nothing is suggested: every value starts empty, beside the run's value now. The test is shown in a corner of the stage
// (the page's qc-test-corner.mjs) from the model this panel hands it (QcPanel testModel); the corner's buttons come back
// through testAction. The panel keeps "Test a setting here…" and one line while a test is open; Go deeper's own "Test a
// setting here…" opens the same form for its cell (openTestForm).
/** The settings a test can change, in one order whatever the run started from (engine/qc_test.py FLAGS): three tried
 *  here, then six more of proseg's own options (twins through frontend/test/fixtures/qc-test-settings.json). */
export const TEST_FLAGS = Object.freeze(['cell-compactness', 'prior-weight', 'diffusion-probability', 'burnin-voxel-size',
  'ab-nihlo-bubble-prob', 'no-diffusion', 'diffusion-sigma-near', 'density-bins', 'voxel-layers']);
/** Whether a setting was tried in a test yet: far drift and the six were not (the form says so, and what each is in
 *  its tooltip). */
export const TEST_TRIED = Object.freeze(Object.fromEntries(TEST_FLAGS.map(f => [f, f === 'cell-compactness' || f === 'prior-weight'])));
/** How the form offers each: a typed number, a typed whole number, one of the run's own choices (the starting grid's,
 *  from the pool reply) or a switch (drift: the engine's 1 = off, 0 = on). */
export const TEST_KINDS = Object.freeze({ ...Object.fromEntries(TEST_FLAGS.map(f => [f, 'number'])), 'burnin-voxel-size': 'choice',
  'no-diffusion': 'switch', 'density-bins': 'whole', 'voxel-layers': 'whole' });
/** The values a test takes where they are not the setting's own bounds (Live knobs', or the pool reply's): far drift 0.1
 *  to 0.3. */
export const TEST_RANGES = Object.freeze({ 'diffusion-probability': Object.freeze([0.1, 0.3]) });
/** The settings said with a unit, those that do nothing while drift is off, and those proseg reads only at its start
 *  (their value now is the pool reply's: Live knobs does not hold them). */
export const TEST_UNITS = Object.freeze({ 'burnin-voxel-size': 'µm', 'diffusion-sigma-near': 'µm' });
export const TEST_DRIFTING = Object.freeze(['diffusion-probability', 'diffusion-sigma-near']);
export const TEST_STARTUP = Object.freeze(['burnin-voxel-size', 'density-bins', 'voxel-layers']);
/** A test's time against one at proseg's 4 depth layers (the engine's layer_factor, measured). */
export const TEST_LAYER_COST = 0.0226;
export const layerFactor = layers => Math.max(1, 1 + TEST_LAYER_COST * ((finite(layers) ? layers : 4) - 4));
/** Each setting's name, its tooltip and its words while drift is off (where it does nothing), by the deck's keys. */
const TEST_WORDS = Object.freeze({
  'cell-compactness': ['settingCompactness', null], 'prior-weight': ['testHold', null], 'diffusion-probability': ['testFarDrift', 'testFarDriftTip'],
  'burnin-voxel-size': ['testGrid', 'testGridTip'], 'ab-nihlo-bubble-prob': ['testGaps', 'testGapsTip'], 'no-diffusion': ['testDrift', 'testDriftTip'],
  'diffusion-sigma-near': ['testNearDrift', 'testNearDriftTip'], 'density-bins': ['testCrowding', 'testCrowdingTip'],
  'voxel-layers': ['testLayers', 'testLayersTip'] });
const DRIFT_OFF_WORDS = Object.freeze({ 'diffusion-probability': 'testDriftOff', 'diffusion-sigma-near': 'testNearDriftOff' });
/** A test's states (GET /runs/{id}/qc/test), and those while it is on its way: waiting for what the app does or waits
 *  for first (the run's last iteration, the mixing check, another run), then its runs at work. */
export const TEST_STATES = Object.freeze(['waiting', 'preparing', 'running', 'comparing', 'done', 'failed', 'cancelled']);
export const TEST_ACTIVE = Object.freeze(['waiting', 'preparing', 'running', 'comparing']);
/** The corner's phases: the form, the start on its way, then the test's own states. */
export const TEST_PHASES = Object.freeze(['form', 'starting', ...TEST_STATES]);
/** What the corner's buttons and fields send back (testAction): the form's setting and value, Run, Cancel, the answers
 *  (New run with this, Keep my setting, Run again) and Close. */
export const TEST_ACTIONS = Object.freeze(['setting', 'value', 'run', 'cancel', 'new', 'keep', 'again', 'close']);
/** The actions that change the form only (no engine call: never busy). */
const FORM_ACTIONS = Object.freeze(['setting', 'value']);
/** How a run started: from cells only, cells paired with nuclei, or nuclei only. */
export const TEST_STARTS = Object.freeze(['cells', 'paired', 'nuclei']);
/** Runs on each side until the engine says (engine/qc_rule.py CONSTANTS runs_per_side, by the start's nearest measured
 *  row): 4 for a start from cells only, 3 from nuclei and cells or nuclei only. */
export const TEST_RUNS_PER_SIDE = Object.freeze({ paired: 3, cells: 4, nuclei: 3 });
export const TEST_POLL_MS = 1000;        // the test is asked for at most once a second
export const TEST_POOL_SHOW_MS = 250;    // "Finding nearby cells…" only once the pool has been asked for this long
export const TEST_POOL_WAIT_S = 5;       // a pool still being worked out is asked for again, the engine waiting up to this
export const TEST_POOL_AGAIN_MS = 2000;  // a wait listed (what a press would wait for first) is asked about again this often
export const TEST_POOL_RUN_WAIT_MS = 30000;   // Run waits at most this long for a pool on its way, then starts all the same
export const PROSEG_HOLD = 1;            // proseg's own Prior weight where a run does not set it (native/proseg main.rs)
/** The section's one line while the test is open, by what the test is doing. */
const TEST_STATUS = Object.freeze({ open: 'testStatusOpen', waiting: 'testStatusWaiting', running: 'testStatusRunning', done: 'testStatusDone',
  stopped: 'testStatusStopped' });
/** The form's value field (explain.mjs outlines it by id; the corner gives its field this name). */
export const TEST_FIELDS = Object.freeze({ flag: 'qc-test-flag', value: 'qc-test-value' });
/** The went bars' parts, in their order (result.went per side: shares of the suspects). */
export const TEST_WENT = Object.freeze(['stay', 'neighbour', 'other', 'background']);
const WENT_WORDS = Object.freeze({ stay: 'testWentStay', neighbour: 'testWentNeighbour', other: 'testWentOther', background: 'testWentBackground' });
/** The cost rows, in their order (result.cost). */
export const TEST_COST = Object.freeze([['own', 'own_same_spot', 'testCostOwn'], ['all', 'own_all', 'testCostAll'],
  ['perCell', 'transcripts_per_cell', 'testCostPerCell']]);
/** A setting by its name in the form: compactness by Live knobs' own name (the deck holds the word only so), the
 *  others by their plain names. */
export const testName = flag => (TEST_WORDS[flag] ? C[TEST_WORDS[flag][0]] : null);
/** A tested value in words: a number as Live knobs shows it, with its unit ("1 µm"); the switch "drift off" / "drift on"
 *  (`form` 'short': "off" / "on"); null without a value. */
export function testValueText(flag, value, form = 'long') {
  if (!finite(value)) return null;
  if (TEST_KINDS[flag] === 'switch') return value === 1 ? (form === 'short' ? C.testSwitchOff : C.testDriftOffWord) : (form === 'short' ? C.testSwitchOn : C.testDriftOnWord);
  return TEST_UNITS[flag] ? copy('testUm', { v: settingValue(value) }) : settingValue(value);
}
const testNowText = (now, flag = null) => finite(now) ? copy('testNow', { v: testValueText(flag, now, 'short') ?? settingValue(now) }) : '';
/** A value typed in the form as a number, or null (blank, or not a number). */
const typedValue = text => String(text ?? '').trim() !== '' && finite(Number(text)) ? Number(text) : null;
const lowerFirst = text => text ? text[0].toLowerCase() + text.slice(1) : text;
const signed = n => n > 0 ? `+${number(n)}` : n < 0 ? `−${number(-n)}` : number(0);

/** The engine's test (the object itself, or one under `test`), or null when it is none. */
export function testRecord(value) {
  const test = value && typeof value === 'object' && value.test && typeof value.test === 'object' ? value.test : value;
  return test && typeof test === 'object' && TEST_STATES.includes(test.state) ? test : null;
}

/** How a run started: the test's own `start` when it says, else from the run's record by the engine's own
 *  rule (engine/qc_rules.py start_of): 'nuclei' without starting cells; 'cells' when every transcript inside a starting
 *  cell seeded it (prepared.seeded_from, zarr_bridge/helper.py: no nuclei chosen, or a pairing whose nuclei flagged none
 *  there); else 'paired' (cells that grow from their nuclei: a Xenium bundle's own, a sample's cells with nuclei). Null
 *  without a run. */
export function testStart(run, test = null) {
  const own = test?.start ?? test?.info?.start;
  if (TEST_STARTS.includes(own)) return own;
  if (!run || typeof run !== 'object') return null;
  const prior = run.prior;
  if (run.sample_kind !== 'zarr' || !prior || typeof prior !== 'object') return 'paired';
  if (!prior.cells) return 'nuclei';
  const seeded = run.prepared?.seeded_from;
  if (seeded != null) return seeded === 'cells' ? 'cells' : 'paired';
  return prior.nuclei ? 'paired' : 'cells';
}

/** How firmly a run holds its starting cells (the harness's snapshot): {value: the Prior weight proseg started with (the
 *  run's options, else proseg's own PROSEG_HOLD), effective: the weight it printed and applies for that value (the run's
 *  info.json `prior_weight`)}, else the test's own `hold` {value, effective}; null when neither says. */
export function holdOf(run, test = null) {
  const printed = run?.info?.prior_weight;
  if (finite(printed) && printed > 0) {
    const asked = run.options && typeof run.options === 'object' ? run.options['prior-weight'] : null;
    return { value: finite(asked) ? asked : PROSEG_HOLD, effective: printed };
  }
  const own = test?.hold ?? test?.info?.hold;
  return own && typeof own === 'object' && finite(own.value) && finite(own.effective) && own.effective > 0
    ? { value: own.value, effective: own.effective } : null;
}

/** Whether drift is switched off in the run (its options with the live values set since): far drift then does nothing,
 *  and the engine refuses it. */
export const driftOff = run => ({ ...(run?.options ?? {}), ...(run?.live_values ?? {}) })['no-diffusion'] === true;

/** The engine's reasons ({code, ...data}: a pool reply's, a refused start's, a test's), each code once, in the order the
 *  engine lists them, in the deck's words (qc-copy.mjs testReasonText, `fallback` for a code the page does not know: a
 *  refusal's unless said): [{code, text}]. */
export function testReasons(list, fallback = C.testReasonUnknown) {
  const seen = new Set(), out = [];
  for (const reason of Array.isArray(list) ? list : []) {
    const code = typeof reason?.code === 'string' ? reason.code : null;
    if (!code || seen.has(code)) continue;
    seen.add(code); out.push({ code, text: testReasonText(reason, fallback) });
  }
  return out;
}
/** What a test waits for first (a pool reply's `waits`, a waiting test's `waiting`) and where it was not measured (a
 *  pool reply's, a test's and an answer's `caveats`), as testReasons gives them, each with its own words for a code the
 *  page does not know. */
export const testWaits = list => testReasons(list, C.testWaitUnknown);
export const testCaveats = list => testReasons(list, C.testCaveatUnknown);
/** Reasons' codes, each once in their order (the hooks: data-waits, data-caveats, the harness's qc.test). */
export const reasonCodes = list => testReasons(list).map(r => r.code);

/** The one line a press of Run says while the engine's refusals are listed (testReasons' [{code, text}]): one about the
 *  run or its sample (qc-copy.mjs TEST_RUN_REFUSALS: no cell of it can be tested), else, by their groups (testReasonGroup),
 *  one that leaves this cell nothing to test, else a test of this run on its way (it passes by itself); '' without a
 *  reason. */
export function testNotHere(reasons) {
  const codes = (Array.isArray(reasons) ? reasons : []).map(r => r?.code), groups = new Set(codes.map(testReasonGroup));
  return codes.some(code => TEST_RUN_REFUSALS.includes(code)) ? C.testNotHereRun : groups.has('cant') ? C.testNotHereCant
    : groups.has('first') ? C.testNotHereFirst : '';
}
// A press of Run while the engine's refusals are listed: kept with what a press could not do (it goes as they go), and
// said as testNotHere's one line, never as a problem to check (formModel).
const NOT_HERE = Object.freeze(problem(null, '', ''));

/** The form's pool line from the pool the engine found ({cells, suspects, need, reached}: the preview's): "Nearby cells
 *  with the same pattern: 23 cells, 187 suspects", what a test needs in its tooltip, its numbers kept as data (`state`
 *  'reached', or 'short': the test takes it all the same, and its answer says so); null without one. */
export function poolLine(pool) {
  if (!pool || typeof pool !== 'object' || ![pool.cells, pool.suspects, pool.need].every(Number.isInteger)) return null;
  const values = { c: number(pool.cells), n: number(pool.suspects) };
  return { state: pool.reached === false || pool.suspects < pool.need ? 'short' : 'reached',
    text: copy(pool.cells === 1 ? 'testPoolOne' : 'testPool', values), tooltip: copy('testPoolTooltip', { k: number(pool.need) }),
    cells: pool.cells, suspects: pool.suspects, need: pool.need };
}

/** The areas a test runs (engine/qc_pool.py area_of): the region as it is, a square around the cell, or another window
 *  around it (a region narrower than the square, cut to it). */
export const TEST_AREAS = Object.freeze(['as_is', 'square', 'crop']);
const AREA_WORDS = Object.freeze({ as_is: 'testAreaRegion', square: 'testAreaSquare', crop: 'testAreaCrop' });
/** The area a test runs ({rule: TEST_AREAS, box: [x0, y0, x1, y1] µm}) in words: "Area: your whole region, 300 × 300
 *  µm", "Area: a 500 × 500 µm square around this cell" or "Area: 500 × 180 µm around this cell"; '' without one. */
export function areaText(area) {
  const b = area?.box;
  if (!TEST_AREAS.includes(area?.rule) || !Array.isArray(b) || b.length !== 4 || !b.every(finite)) return '';
  return copy(AREA_WORDS[area.rule], { w: number(Math.round(b[2] - b[0])), h: number(Math.round(b[3] - b[1])) });
}

/** The runs a test makes, in words: "Your run waits · 3 runs as now, 3 with 0.02 · about 45 s" (`perSide` on each
 *  side; `value` the one tested, before one is typed "this value"; `seconds` the engine's estimate, the line without
 *  it when none); '' without the runs per side. */
export function runsText(perSide, value = null, seconds = null, flag = null) {
  if (!Number.isInteger(perSide) || perSide <= 0) return '';
  const values = { a: number(perSide), b: number(perSide), v: finite(value) ? testValueText(flag, value) : C.testThisValue };
  return finite(seconds) && seconds > 0 ? copy('testRuns', { ...values, t: duration(seconds) }) : copy('testRunsNoTime', values);
}

/** The engine's estimate (s) for a test of `flag` at `value` before the POST: the pool reply's, times the depth layers'
 *  cost of `value` over the run's own (`now`) in a test of depth layers; else as given. */
export function testEstimate(seconds, flag, value, now) {
  if (!finite(seconds)) return seconds ?? null;
  return flag === 'voxel-layers' && finite(value) ? Math.round(seconds * layerFactor(value) / layerFactor(now)) : seconds;
}

/** The hold a test of it is against, from the pool reply's `hold` ({value, seq, ...}): its value at the frame tested once
 *  the reply knows that frame (`seq`: F, the run's last iteration), else null (a reply made before the run reached F,
 *  seq null: the engine then checks the value now, and so does the form). */
export const holdAtEnd = hold => hold && typeof hold === 'object' && Number.isInteger(hold.seq) && finite(hold.value) ? hold.value : null;

/** What a press of Run the test still needs on the page, as explain.mjs problems (never a greyed-out button): a cell,
 *  and a number within the setting's bounds (TEST_RANGES, else Live knobs' `setting` {value (the run's now), min, max})
 *  that is not the value the test is against: the run's value now, but the hold's at the frame tested once the pool
 *  reply knows that frame (`hold`: the reply's, {value, seq, ...}; holdAtEnd: F's value, the run's last iteration, as
 *  the engine checks it: a hold set live since then is another value, said as the one it used then; a reply made before
 *  F leaves the value now, as the engine's check does), none while the reply is still to come (`holdDue`: the pool
 *  reply on its way, checked again once it came); far drift only while drift is on in the run (`driftOff`). Whatever
 *  else stands in the way is the engine's reason, which the form lists (the run's state, the mixing check's, the
 *  cell's, the pool's). */
export function testProblems({ label = 0, text = '', flag = null, setting = null, driftOff = false, hold = null, holdDue = false,
  choices = null, flat = false } = {}) {
  const problems = [];
  if (!label) problems.push(problem(null, '', C.testNoCell));
  const kind = TEST_KINDS[flag] ?? 'number';
  const range = TEST_RANGES[flag], min = range ? range[0] : finite(setting?.min) ? setting.min : null;
  const max = range ? range[1] : finite(setting?.max) ? setting.max : null;
  const value = String(text ?? '').trim() === '' ? NaN : Number(text);
  const now = finite(setting?.value) ? setting.value : null, atEnd = flag === 'prior-weight' ? holdAtEnd(hold) : null;
  const against = atEnd ?? (flag === 'prior-weight' && holdDue ? null : now);
  const field = msg => problem(TEST_FIELDS.value, C.testValueLabel, msg);
  if (!finite(value)) problems.push(field(kind === 'choice' || kind === 'switch' ? C.testChoose : C.testValueMissing));
  else if (kind === 'switch' && value !== 0 && value !== 1) problems.push(field(C.testChoose));
  else if (kind === 'choice' && Array.isArray(choices) && !choices.some(c => sameSetting(c, value))) problems.push(field(C.testChoose));
  else if (kind === 'whole' && (!Number.isInteger(value) || (min != null && value < min) || (max != null && value > max))) {
    problems.push(field(copy('testWhole', { min: min == null ? C.missing : number(min), max: max == null ? C.missing : number(max) })));
  } else if (kind === 'number' && ((min != null && value < min) || (max != null && value > max))) {
    problems.push(field(copy('testRange', { min: min == null ? C.missing : settingValue(min), max: max == null ? C.missing : settingValue(max) })));
  } else if (against != null && sameSetting(value, against)) {
    problems.push(field(atEnd != null && !(now != null && sameSetting(atEnd, now)) ? C.testValueSameEnd : C.testValueSame));
  } else if (DRIFT_OFF_WORDS[flag] && driftOff) problems.push(problem(null, '', C[DRIFT_OFF_WORDS[flag]]));
  else if (flag === 'voxel-layers' && flat) problems.push(problem(null, '', C.testFlat));
  return problems;
}

/** A share (0-1, the engine's to 4 decimals) as a percentage to one decimal, so that a half rounds up as written
 *  (0.565: 56.5, then 57), never down by the float's own error. */
export const tenths = x => Math.round(1000 * x) / 10;
/** A share as a whole percentage (tenths, then rounded). */
export const percent = x => Math.round(tenths(x));
/** Two shares (0-1) as whole percentages, with one decimal where the two would read the same though they differ: a
 *  verdict never stands beside two equal numbers it did not see. Missing: the dash. */
export function percentPair(a, b) {
  const tie = finite(a) && finite(b) && tenths(a) !== tenths(b) && percent(a) === percent(b);
  const text = x => finite(x) ? number(tie ? tenths(x) : percent(x), tie ? 1 : 0) : C.missing;
  return [text(a), text(b)];
}
/** Shares as whole percentages that add up to 100 (largest remainder: the parts' floors, then a point each to the
 *  largest remainders, the first on a tie), in their own order; null without a share to show. */
export function wholePercents(shares) {
  const values = shares.map(v => finite(v) && v > 0 ? v : 0), sum = values.reduce((a, v) => a + v, 0);
  if (!(sum > 0)) return null;
  const raw = values.map(v => 100 * v / sum), out = raw.map(Math.floor);
  const order = raw.map((v, i) => [v - out[i], i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (let k = 0, left = 100 - out.reduce((a, v) => a + v, 0); k < left; k++) out[order[k % order.length][1]]++;
  return out;
}

/** The answer's tooltip (the deck's testCalibration), every number from the test's own row of measured numbers
 *  (result.constants.rates: its start and its area), as whole numbers: how often identical runs come out "clearly fewer"
 *  ({fa} on average, {fw} in the worst of {areas} areas; at least 1 where the rate is above 0), how often a drop of the
 *  suspects alone by {drop} % is caught ({pw} in 10; from this pool's own share as now, come_back.mean, to that share
 *  less the drop: the "to" worked out from the "from" shown, so that the sentence's own sum holds, 61 % → 46 % rather
 *  than 45 % for 60.5 %), and how often a drop of the suspects and their own RNA alike still says it ({sf} in 100); with
 *  testCalibrationSettings where the run's settings are not proseg's defaults (parent_defaults false). '' without the
 *  numbers. */
export function calibrationText(result) {
  const rates = result?.constants?.rates;
  if (!rates || typeof rates !== 'object') return '';
  const per100 = x => finite(x) && x >= 0 ? (x > 0 ? Math.max(1, Math.round(100 * x)) : 0) : null;
  const fa = per100(rates.fp_mean), fw = per100(rates.fp_worst), sf = per100(rates.shrink_fire);
  const pw = finite(rates.power) ? Math.round(10 * rates.power) : null, drop = finite(rates.drop) && rates.drop > 0 && rates.drop < 1 ? rates.drop : null;
  const mean = result.come_back?.mean, from = finite(mean) ? percent(mean) : null;
  if ([fa, fw, sf, pw, drop, from].some(v => v == null) || !Number.isInteger(rates.areas)) return '';
  const text = copy('testCalibration', { fa: number(fa), fw: number(fw), areas: number(rates.areas), drop: number(Math.round(100 * drop)),
    from: number(from), to: number(Math.round(from * (1 - drop))), pw: number(pw), sf: number(sf) });
  return result.parent_defaults === false ? `${text} ${C.testCalibrationSettings}` : text;
}
/** The short pools' own numbers (E3, the deck's testCalibrationShort), for a test with fewer suspects than its numbers
 *  are for (its caveats hold too_few) where its row has them (result.constants.short_rates {fp_mean, fp_worst, power,
 *  drop, tests}), rounded as calibrationText rounds them; {need} the suspects its numbers are for (the pool's k); '' else. */
export function calibrationShortText(result) {
  const rates = result?.constants?.short_rates, caveats = Array.isArray(result?.caveats) ? result.caveats : [];
  if (!rates || typeof rates !== 'object' || !caveats.some(c => (typeof c === 'string' ? c : c?.code) === 'too_few')) return '';
  const per100 = x => finite(x) && x >= 0 ? (x > 0 ? Math.max(1, Math.round(100 * x)) : 0) : null;
  const fa = per100(rates.fp_mean), fw = per100(rates.fp_worst), pw = finite(rates.power) ? Math.round(10 * rates.power) : null;
  const drop = finite(rates.drop) && rates.drop > 0 && rates.drop < 1 ? rates.drop : null, need = result.pool?.k;
  if ([fa, fw, pw, drop].some(v => v == null) || !Number.isInteger(rates.tests) || !Number.isInteger(need)) return '';
  return copy('testCalibrationShort', { need: number(need), n: number(rates.tests), fa: number(fa), fw: number(fw), drop: number(Math.round(100 * drop)),
    pw: number(pw) });
}

/** The word an earlier answer gave, as Run again's line says it under the new one ("clearly fewer", "clearly fewer (own
 *  RNA not separated)", "not clear"; its first letter lower-cased), or null when it judged nothing (they came and went,
 *  or too few). */
export function earlierWord(result) {
  if (result?.verdict === 'clearly_fewer') return lowerFirst(result.rule === 'suspects' ? C.testFewerAloneYes : C.testFewerYes);
  return result?.verdict === 'not_clear' ? lowerFirst(C.testFewerNo) : null;
}

/** The hold values a test's runs were given (the test record's or its answer's `hold`: {value (the run's at the frame
 *  tested), effective, scale, seq, given: null | {scale, as_now, tried}}: proseg scales the hold by its area's
 *  transcripts per cell, so the runs are given the value that holds as firmly there as the run's does): in a test of the
 *  hold (`flag` prior-weight) the runs as now's for the run's own and those with the value's for the value tried
 *  (`value`); in a test of another setting every run's, which all keep the run's hold; '' until the engine gave them (a
 *  test that waits, or a record from before them). */
export function holdGiven(hold, flag, value) {
  const given = hold?.given;
  if (!given || typeof given !== 'object' || !finite(given.as_now) || !finite(hold.value)) return '';
  const values = { a: settingValue(given.as_now), v: settingValue(hold.value) };
  if (flag !== 'prior-weight') return copy('testHowHoldGivenAll', values);
  const tried = finite(given.tried) && finite(value) ? copy('testHowHoldTried', { b: settingValue(given.tried), t: settingValue(value) }) : '';
  return [copy('testHowHoldGiven', values), tried].filter(Boolean).join(' ');
}

/** A test that is done, its answer in words part by part (result v2: engine/qc_rule.py answer), as the corner draws
 *  it; null without an answer the page reads (v 2). The engine's verdict and its reasons are taken as they are: only the
 *  chosen rule's verdict is drawn (result.rule, result.fewer; fewer.forms stay the record's), and none is derived again
 *  from numbers. {verdict, rule, why, back {word: yes | no, title, answer, text, tip, info}, fewer {word: fewer | unclear,
 *  why: yes | apart | gap, shrink (the one sentence: clearly fewer on the excess with the cell shrinking), title, head,
 *  text, whyText, tip}, cost {title, rows [{key: own | all | perCell, label, value, change}], shrink, text}, went {title,
 *  sides [{side: A | B, parts [{key, label, share, percent}], text}]}, alone {text}, toofew {why, text, tip}, calibration,
 *  caveats [{code, text}]}: the parts the verdict has (result v2: too few has no runs to read, they came and went has no
 *  "fewer"), each null otherwise; too few: toofew alone (with no alone line). Where the test was not measured (the
 *  answer's `caveats`: judged with the nearest measured row's numbers) is said in tooltips only, one a line: after how
 *  often the test is wrong (the fewer part's ⓘ, `fewer.tip`), else in the back part's own ⓘ (`back.info`), and for too
 *  few in its line's (`toofew.tip`). */
export function testReadout(test) {
  const r = test?.result;
  if (!r || typeof r !== 'object' || r.v !== 2 || typeof r.verdict !== 'string') return null;
  const value = finite(r.setting?.value) ? r.setting.value : test.value, v = finite(value) ? testValueText(r.setting?.flag ?? test.flag, value) : C.missing;
  const rule = r.rule === 'suspects' ? 'suspects' : 'excess', excess = rule === 'excess', fires = r.verdict === 'clearly_fewer';
  const calibration = calibrationText(r), pool = r.pool && typeof r.pool === 'object' ? r.pool : {};
  const caveats = testCaveats(r.caveats), said = caveats.map(x => x.text).join('\n');
  const out = { verdict: r.verdict, rule, why: typeof r.why === 'string' ? r.why : null, back: null, fewer: null, cost: null, went: null,
    alone: null, toofew: null, calibration, caveats };
  if (r.verdict === 'too_few') {
    const why = ['pool_short', 'cell_lost', 'no_controls'].includes(r.why) ? r.why : 'pool_short';
    out.toofew = { why, text: why === 'cell_lost' ? C.testTooFewCell : why === 'no_controls' ? C.testNoControls
      : copy('testTooFew', { n: finite(pool.suspects) ? number(pool.suspects) : C.missing, k: finite(pool.k) ? number(pool.k) : C.missing }), tip: said };
    return out;
  }
  const back = r.come_back, f = r.fewer && typeof r.fewer === 'object' ? r.fewer : null, shrinking = r.cost?.shrinking === true;
  if (back && typeof back === 'object') {
    const held = Array.isArray(back.held) ? back.held : [];
    const numbers = finite(back.mean) ? copy('testBack', { n: finite(pool.suspects) ? number(pool.suspects) : C.missing, p: number(percent(back.mean)),
      runs: held.map(h => finite(h) ? number(percent(h)) : C.missing).join(' · ') || C.missing }) : '';
    out.back = { word: back.passed === true ? 'yes' : 'no', title: C.testBackTitle, answer: back.passed === true ? C.testBackYes : C.testBackNo,
      text: back.passed === true ? numbers : C.testBackNot, tip: back.passed === true ? '' : numbers, info: f ? '' : said };
  }
  if (f) {
    const one = fires && excess && shrinking, why = fires ? 'yes' : f.why_not === 'gap' ? 'gap' : 'apart';
    const [a, b] = percentPair(f.suspects?.mean?.a, f.suspects?.mean?.b), [c, d] = percentPair(f.own?.mean?.a, f.own?.mean?.b);
    out.fewer = { word: fires ? 'fewer' : 'unclear', why, shrink: one, title: excess ? C.testFewerTitle : C.testFewerTitleAlone,
      head: one ? C.testFewerYesShrink : fires ? (excess ? C.testFewerYes : C.testFewerAloneYes) : C.testFewerNo,
      text: excess ? copy('testFewerText', { v, a, b, c, d }) : copy('testFewerTextAlone', { v, a, b }),
      whyText: why === 'yes' ? (excess ? C.testFewerWhyYes : C.testFewerAloneWhyYes) : why === 'gap' ? C.testFewerWhyGap
        : excess ? C.testFewerWhyApart : C.testFewerAloneWhyApart,
      tip: [calibration, said, calibrationShortText(r), holdGiven(r.hold, r.setting?.flag ?? test.flag, value)].filter(Boolean).join('\n') };
  }
  const cost = r.cost;
  if (cost && typeof cost === 'object') {
    const rows = TEST_COST.map(([key, field, label]) => {
      const x = cost[field] && typeof cost[field] === 'object' ? cost[field] : {};
      const [a, b] = key === 'perCell' ? [x.a, x.b].map(n => finite(n) ? number(Math.round(n)) : C.missing) : percentPair(x.a, x.b);
      return { key, label: C[label], value: copy(key === 'perCell' ? 'testCostCount' : 'testCostShare', { a, b }),
        change: finite(x.change) ? copy('testCostChange', { c: signed(Math.round(100 * x.change)) }) : '' };
    });
    const said = shrinking && !out.fewer?.shrink;
    out.cost = { title: C.testCostTitle, rows, shrink: said, text: said ? C.testShrink : '' };
  }
  const went = r.went;
  if (went && typeof went === 'object') {
    const sides = ['A', 'B'].map(side => {
      const shares = TEST_WENT.map(key => went[side]?.[key]), percents = wholePercents(shares);
      if (!percents) return null;
      const parts = TEST_WENT.map((key, i) => ({ key, label: C[WENT_WORDS[key]], share: finite(shares[i]) ? shares[i] : 0, percent: percents[i] }));
      return { side, parts, text: parts.map(p => copy('testWentPart', { label: p.label, p: number(p.percent) })).join(' · ') };
    }).filter(Boolean);
    out.went = sides.length ? { title: C.testWentTitle, sides } : null;
  }
  const n = r.cell?.suspects;
  if (Number.isInteger(n) && n >= 0) out.alone = { text: n === 1 ? C.testAloneOne : copy('testAlone', { n: number(n) }) };
  return out;
}

/** A test on its way as the corner shows it, from its progress (the run's push and the test's own record: {iter, of,
 *  runs, done, sides {A, B}, marks, phase, retry}): `text` the engine's phase in words (the runs prepared or starting;
 *  while they sample the slowest run's iteration of the schedule's end and the part of the schedule it is in, settling
 *  to marks[0], refining to marks[1], then final settling; the runs writing their results, how many are done; the
 *  engine reading them), `tooltip` the schedule's parts, `retry` the line said once the runs started again without
 *  factorisation; the numbers as they are (iter capped at `of`). */
export function testProgress(test) {
  const p = test?.progress && typeof test.progress === 'object' ? test.progress : {};
  const of = Number.isInteger(p.of) && p.of > 0 ? p.of : null;
  const marks = Array.isArray(p.marks) && p.marks.length === 2 && p.marks.every(Number.isInteger) ? [...p.marks] : null;
  const iter = finite(p.iter) ? Math.max(0, of ? Math.min(of, p.iter) : p.iter) : 0;
  const side = s => finite(p.sides?.[s]) ? Math.max(0, of ? Math.min(of, p.sides[s]) : p.sides[s]) : null;
  const runs = Number.isInteger(p.runs) && p.runs > 0 ? p.runs : null, done = Number.isInteger(p.done) && p.done >= 0 ? p.done : 0;
  const phase = test?.state === 'comparing' ? 'reading' : ['preparing', 'loading', 'sampling', 'writing', 'reading'].includes(p.phase) ? p.phase
    : test?.state === 'running' ? 'sampling' : 'preparing';
  const part = !marks || iter < marks[0] ? 'settling' : iter < marks[1] ? 'refining' : 'final';
  const word = C[{ settling: 'testPhaseSettling', refining: 'testPhaseRefining', final: 'testPhaseFinal' }[part]];
  const text = phase === 'preparing' ? C.testPreparing : phase === 'loading' ? C.testLoading : phase === 'reading' ? C.testReading
    : phase === 'writing' && runs ? copy('testWriting', { d: number(done), n: number(runs) })
      : copy('testIteration', { i: number(iter), n: of ? number(of) : C.missing, phase: word });
  const tooltip = marks && of ? copy('testProgressTooltip', { a: number(marks[0]), b: number(marks[1]), n: number(of) }) : '';
  return { iter, of, runs, done, sides: { A: side('A'), B: side('B') }, marks, phase, part, text, tooltip, retry: p.retry === true ? C.testRetry : null };
}

/** A test waiting for what the app does or waits for first (its `waiting`: the engine's waits now, in its order), as the
 *  corner says it: `text` the first one's words ("Starts once your run reaches its last iteration"); once none holds,
 *  what the engine does before the runs start: it works out the cell's pool (testPoolAsking, "Finding nearby cells with
 *  the same pattern…": a whole sample's takes minutes); `tooltip` every one, a line each, when there are more; `codes`
 *  theirs. */
export function testWaiting(test) {
  const waits = testWaits(test?.waiting);
  return { text: waits[0]?.text ?? C.testPoolAsking, tooltip: waits.length > 1 ? waits.map(w => w.text).join('\n') : '', codes: waits.map(w => w.code) };
}

/**
 * QcPanel owns only nodes inside its <details>. No fetches, key bindings or app state.
 *
 * host: the page's <details class="panel"> (main.js remembers it open or closed), or any element (a <details> is made
 * inside it, remembered with options.storage). state: fromView(...) (see above); suggestion, moreIdeas, trial,
 * spread) and, optionally, laterTime (what 1,000 iterations take in this run, e.g. "3 min": the card's and the trial's
 * own later_s come first).
 * Callbacks (all optional): onToggle(enabled), onShowCells(enabled, context?) (context {start: true, area, areas,
 * suggestion}: R1's "Show the outlines of ... here", the starting outlines on and the view moved to its area), onRetry(),
 * onReload() (the run's check could not be reached: ask again), onCalmMotion(enabled) (the page keeps the choice and
 * passes it back as state.calmMotion), onFrame() (each motion frame painted), and for the suggestions
 * onTry(suggestion), onNewRun(suggestion), onUndo(trial | U suggestion), onKeep(trial | U suggestion),
 * onWhy(suggestion, open), onIdea(suggestion), onCard(suggestion) (a card has just been shown) and
 * onInspect(suggestion, area) ("Look at its cells": the view moved to its area, Go deeper on its most mixed cell;
 * without it the card offers no such button). Options: storage (null: none), storageKey, reducedMotion, suggestions
 * (the card and readout), knobLabel(flag) (the Live knobs label of a proseg option: the card's Setting line), width,
 * scheduler {now, setTimeout, clearTimeout, requestAnimationFrame, cancelAnimationFrame}, bindDeeper(button, label)
 * (the selected cell's "Go deeper ›": its words, state and click, qc-deeper.js bindButton; without it the row has no
 * button). "Test a setting here…" (renderTest; without onTestStart the row offers none), the page's calls:
 * testSetting(flag) -> {label (Live knobs), value (the run's now), min, max} | null; onTestPreview(label, {wait, signal})
 * -> the pool reply (GET /runs/{id}/qc/test/pool for cell label - 1: its pool, area, runs per side and reasons);
 * onTestStart(label, flag, value) -> the test (POST {cell, flag, value}; a refusal rejects with its status, and a 409
 * with the engine's reasons in `data.reasons`); onTestFetch() -> the test, rejecting with status 404 when there is none
 * (GET); onTestCancel(test) and onTestAnswer(test) (Keep or Close) settle once the run is as it was before the test,
 * rejecting with `plain` words to show; onTestNewRun(test) (New run with this: the page leaves the run for a new one
 * with the setting from the start; the engine ends the test as the run is left); onTestShow() (the form was opened for
 * a cell from elsewhere, Go deeper: the page shows this panel). setTestRun(run) hands over the run's record (its id,
 * kind, state, schedule_done, pushed `qc_test`, how it started: prior, prepared, sample_kind; its options, live values
 * and info). onTestCorner(model) shows the test in the stage's corner on every change of it (testModel; model.open false
 * closes it), the corner's buttons come back through testAction(action, payload).
 */
export class QcPanel {
  constructor(host, options = {}) {
    this.host = host; this.options = options; this.doc = host.ownerDocument ?? globalThis.document;
    this.width = options.width ?? 218; this.height = CHART_HEIGHT; this.state = {}; this.listeners = [];
    this.clock = options.scheduler ?? {
      now: () => globalThis.performance?.now() ?? Date.now(),
      setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms), clearTimeout: id => globalThis.clearTimeout(id),
      requestAnimationFrame: fn => globalThis.requestAnimationFrame(fn), cancelAnimationFrame: id => globalThis.cancelAnimationFrame(id),
    };
    try { this.storage = options.storage === undefined ? globalThis.localStorage : options.storage; } catch { this.storage = null; }
    this.storageKey = options.storageKey ?? 'sopa-companion:qc:open';
    const own = host.tagName !== 'DETAILS';
    this.root = own ? this.el('details', 'qc-panel') : host;
    if (!own) this.root.classList?.add('qc-panel');
    if (own) { let saved = null; try { saved = this.storage?.getItem(this.storageKey); } catch { /* optional */ } this.root.open = saved == null ? true : saved === 'true'; }
    const header = this.el('summary', 'qc-header'); header.append(this.el('span', 'qc-title', C.title));
    this.toggle = this.button(C.on, 'toggle', () => this.call('onToggle', !(this.state.enabled ?? false)));
    this.toggle.className = 'qc-switch'; this.toggle.setAttribute('role', 'switch'); this.toggle.setAttribute('aria-label', C.title);
    this.toggle.title = C.switchTooltip;
    header.append(this.toggle); this.root.append(header);
    this.body = this.el('div', 'qc-body'); this.subtitle = this.el('p', 'qc-subtitle', C.subtitle);
    // The selected cell (setCell): its name, "Go deeper ›" and its mixing line, first in the body, so nothing above it
    // changes while it shows; the button sits above the line, so a line that changes or goes never moves it under the
    // pointer. Built once and updated in place, never by setState: a press lands on the button it was aimed at, and
    // keyboard focus stays on it.
    this.cellRow = this.el('section', 'qc-cell'); this.cellRow.hidden = true; this.cellLabel = 0;
    const cellHead = this.el('header', 'qc-cell-head'); this.cellSwatch = this.el('i'); this.cellName = this.el('span', 'qc-cell-name');
    cellHead.append(this.cellSwatch, this.cellName);
    this.cellLine = this.el('p', 'qc-small qc-muted qc-card-line'); this.cellLine.hidden = true;
    this.deeperButton = this.el('button', 'qc-button qc-deeper-open'); this.deeperButton.type = 'button';
    this.deeperButton.hidden = typeof options.bindDeeper !== 'function';
    this.cellRow.append(cellHead, this.deeperButton, this.cellLine);
    // "Test a setting here…" in a section of its own right under the row, drawn as its continuation (qc.css), then one
    // line while its test is open in the stage's corner (renderTest, testModel). The row keeps its three nodes.
    this.testBox = this.el('section', 'qc-test'); this.testBox.hidden = true; this.testBox.dataset.state = 'none';
    this.testRunId = null; this.testRunState = null; this.testRun = null; this.testRec = null; this.testForm = null; this.testParts = null;
    this.testTimer = null; this.testFetching = false; this.testWanted = false; this.testFetchedAt = -Infinity;
    this.covered = false;   // a full-screen view over the page (setCovered): no test asked for, nothing moving meanwhile
    this.testDismissed = new Set(); this.cellNameText = '';
    // The form's pool (askPool: the engine's preview of the form's cell) and how many times a pool was asked for (each
    // ask numbered: a refused start's reasons give way only to a preview asked after them), the reasons a refused start
    // gave, and the earlier answer Run again said under the new one.
    this.testPool = null; this.poolAsks = 0; this.testRefused = null; this.testEarlier = null;
    // The corner's side: a start on its way, an answer on its way (its action), what a press could not do (problems for
    // Run, a message for the others), and the model last handed over.
    this.testStarting = false; this.testBusy = null; this.testShownProblems = []; this.testMessage = null;
    this.cornerKey = keyOf({ open: false }); this.cornerOpen = false; this.cornerModel = { open: false };
    this.body.append(this.cellRow, this.testBox, this.subtitle);
    this.status = this.el('div', 'qc-status'); this.status.setAttribute('role', 'status');
    // The number and its label, with an ⓘ that says what it is, what it rests on and the saved results'
    // number (written per result: setState); one verdict line (the like-for-like change since the start, the start's
    // number beside it) or the start's number alone; the retention line only when it moves the number by itself.
    this.headline = this.el('div', 'qc-headline');
    const line = this.el('div', 'qc-number-line'); this.numberNode = this.el('strong', 'qc-number');
    const label = this.el('span', 'qc-number-label', C.numberLabel);
    this.numberInfo = this.info(C.numberTooltip);
    line.append(this.numberNode, label, this.numberInfo); this.headline.append(line);
    // The result shown is another iteration's than the frame on screen (looking back: the nearest with proseg's count).
    this.snappedNode = this.el('p', 'qc-small qc-muted qc-snapped'); this.snappedNode.hidden = true;
    this.changeNode = this.el('p', 'qc-change'); this.startNode = this.el('p', 'qc-small qc-muted qc-start');
    // A number from few cells says so in view, short; the whole basis is in the ⓘ.
    this.roughNode = this.el('p', 'qc-small qc-muted qc-rough'); this.roughNode.hidden = true;
    // How many transcripts the cells kept since the start, when that by itself moves the number.
    this.retentionNode = this.el('p', 'qc-small qc-muted qc-retention'); this.retentionNode.hidden = true;
    this.retentionText = this.el('span'); this.retentionInfo = this.info('');
    this.retentionNode.append(this.retentionText, this.retentionInfo);
    this.headline.append(this.snappedNode, this.changeNode, this.startNode, this.roughNode, this.retentionNode);
    this.notices = this.el('div', 'qc-notices');
    // The chart's title is its ⓘ's tooltip, in the corner above the plot (with what its dashed line is, while it has one).
    this.chartBox = this.el('figure', 'qc-chart');
    const caption = this.el('figcaption', 'qc-chart-caption'); this.chartInfo = this.info(C.chartTitle); caption.append(this.chartInfo); this.chartBox.append(caption);
    this.canvas = this.el('canvas', 'qc-chart-canvas'); this.canvas.setAttribute('role', 'img'); this.canvas.setAttribute('aria-label', C.chartTitle);
    this.canvas.tabIndex = 0;
    this.readout = this.el('p', 'qc-chart-readout'); this.readout.hidden = true; this.readout.setAttribute('aria-live', 'polite');
    this.chartLegend = this.el('div', 'qc-chart-legend'); this.chartBox.append(this.canvas, this.chartLegend, this.readout);
    this.showCells = this.button(C.showCells, 'show-cells', () => this.call('onShowCells', !this.state.showingCells));
    this.showCells.title = C.showCellsTooltip;
    // The suggestion card and trial readout (options.suggestions); without the option the slot stays empty and
    // they show now (the harness's snapshot).
    this.slot = this.el('section', 'qc-slot'); this.slot.dataset.slot = 'suggestions'; this.slot.hidden = true;
    this.suggestions = this.el('section', 'qc-suggestions'); this.trial = this.el('section', 'qc-trial');
    this.slot.append(this.suggestions, this.trial);
    this.shown = { card: null, trial: null }; this.entering = null;
    this.finished = this.el('section', 'qc-finished');
    // The page keeps the calm-motion choice (it stops the stage's motion too) and passes it back in the state.
    this.calm = this.button(C.calmMotion, 'calm-motion', () => this.call('onCalmMotion', !this.state.calmMotion));
    this.calm.title = C.calmMotionTooltip;
    this.how = this.el('details', 'qc-how'); this.how.append(this.el('summary', '', C.howTitle));
    this.howBody = this.el('div'); this.how.append(this.howBody);
    this.estimateNote = this.el('p', 'qc-small qc-muted', C.estimate);
    this.body.append(this.status, this.headline, this.notices, this.chartBox, this.showCells,
      this.slot, this.finished, this.calm, this.estimateNote, this.how);
    this.root.append(this.body);
    if (own) host.append(this.root);
    // The chart's design tokens are read now, once: a computed-style read inside the first result's paint batch would
    // make the browser recalculate the page's style there (a noticeable pause on a whole sample).
    this.colors = chartColors(this.canvas);
    this.listen(this.root, 'toggle', () => {
      if (own) try { this.storage?.setItem(this.storageKey, String(this.root.open)); } catch { /* optional */ }
      this.stopMotion(true); this.stopEntrance(true); if (this.root.open) this.redraw();
    });
    // Hover and keys read the pointer's own offset: no layout read.
    this.listen(this.canvas, 'pointermove', event => {
      if (!this.layout) return;
      const x = finite(event.offsetX) ? event.offsetX : (event.clientX ?? 0) - (this.canvas.getBoundingClientRect?.().left ?? 0);
      this.showHover(hoverAt(this.layout, x));
    });
    this.listen(this.canvas, 'pointerleave', () => this.showHover(null));
    // The suggestion card is not rebuilt while one of its buttons is pressed (the click lands on the card it was aimed
    // at) or has keyboard focus; the deferred change follows once it is let go, after the click.
    const releaseCard = () => { if (!this.cardPressed) return; this.cardPressed = false; if (this.cardStale) this.clock.setTimeout(() => this.refreshCard(), 0); };
    this.listen(this.suggestions, 'pointerdown', () => { this.cardPressed = true; });
    for (const type of ['pointerup', 'pointercancel', 'pointerleave']) this.listen(this.suggestions, type, releaseCard);
    this.listen(this.suggestions, 'focusin', () => { if (!this.cardPressed) this.cardFocused = true; });
    this.listen(this.suggestions, 'focusout', () => { this.cardFocused = false; if (this.cardStale) this.clock.setTimeout(() => this.refreshCard(), 0); });
    this.listen(this.doc, 'visibilitychange', () => { if (!this.doc.hidden) this.scheduleTestPoll(); });
    this.listen(this.canvas, 'blur', () => this.showHover(null));
    this.listen(this.canvas, 'keydown', event => {
      if (!this.layout || !['ArrowLeft', 'ArrowRight', 'Home', 'End', 'Escape'].includes(event.key)) return;
      event.preventDefault(); if (event.key === 'Escape') { this.showHover(null); return; }
      const points = this.layout.hoverPoints; if (!points.length) return;
      let i = points.findIndex(p => p.iter === this.hoverIter);
      i = event.key === 'Home' ? 0 : event.key === 'End' ? points.length - 1 : Math.max(0, Math.min(points.length - 1, (i < 0 ? points.length - 1 : i) + (event.key === 'ArrowLeft' ? -1 : 1)));
      this.showHover(hoverAt(this.layout, points[i].x));
    });
    this.media = globalThis.matchMedia?.('(prefers-reduced-motion: reduce)');
    if (this.media) this.listen(this.media, 'change', () => { if (this.reducedMotion) { this.stopMotion(true); this.stopEntrance(true); } });
    this.listen(this.doc, 'visibilitychange', () => { if (this.doc.hidden) { this.stopMotion(true); this.stopEntrance(true); } });
    // The chart follows its box's width (the side column, its scrollbar) from the observer's own size: no layout read.
    if (globalThis.ResizeObserver) {
      this.observer = new ResizeObserver(entries => { const width = entries.at(-1)?.contentRect?.width; if (width > 0) this.resize(width); });
      this.observer.observe(this.chartBox);
    }
    this.setState({ state: 'loading' });
  }

  el(tag, className = '', text = '') {
    const el = this.doc.createElement(tag); if (className) el.className = className; if (text) el.textContent = text; return el;
  }
  listen(node, type, fn) { node.addEventListener?.(type, fn); this.listeners.push([node, type, fn]); }
  call(name, ...args) { if (!this.disposed) (this.options[name] ?? noop)(...args); }
  button(text, action, callback) {
    const button = this.el('button', 'qc-button', text); button.type = 'button'; button.dataset.action = action;
    // In the panel's <summary> a click would also open or close it: the switch does only its own job.
    button.onclick = event => { event?.stopPropagation?.(); event?.preventDefault?.(); if (!button.disabled && !this.disposed) callback(); };
    return button;
  }
  section(node, key, build) {
    key = keyOf(key); if (node._qcKey === key) return;
    node._qcKey = key; node.replaceChildren(); build(node);
  }
  paragraph(parent, text, className = 'qc-small') { if (text) parent.append(this.el('p', className, text)); }
  /** An ⓘ beside the thing it explains (an inline SVG icon): its words are its tooltip and its
   *  accessible name, and it takes the keyboard so they can be read. */
  info(text) {
    const info = this.el('span', 'qc-info'); info.tabIndex = 0; info.setAttribute('role', 'img'); info.innerHTML = INFO_ICON;
    this.tip(info, text); return info;
  }
  tip(node, text) {
    if (node.title === text) return;
    node.title = text;
    if (node.className === 'qc-info') node.setAttribute('aria-label', text);   // an ⓘ is named by its words
  }
  get reducedMotion() { return this.options.reducedMotion === true || this.media?.matches === true || this.state.calmMotion === true; }
  showHover(result) { this.hoverIter = result?.iter; setHidden(this.readout, !result); setText(this.readout, result?.text ?? ''); }

  /** The headline's point: the start line at frame 0 (none for a run with no start line), else the result shown. */
  point(state) {
    const start = state.start ?? state.summary?.start;
    if (state.atStart) return !noStart(state) && finite(start?.per100) ? { ...start, iter: 0 } : null;
    return state.summary && finite(state.summary.per100) ? state.summary : null;
  }

  /** The headline is shown, its meaning in its ⓘ (main.js updateCellRow: the selected cell's line then leaves out what
   *  the ⓘ says, qc-copy.mjs cardLine `note`, and keeps it in its own tooltip). */
  get headlineShown() { return !this.headline.hidden; }

  /** The selected cell's row (main.js updateCellRow, every frame): cell {label, name, color, line, tip, loading}, or
   *  null (no cell selected in a run on a sample). `line`: its mixing line ('' while the check has none for it), `tip`
   *  that line's tooltip. `loading`: the result shown has the cell's own values on the way; a line in the row keeps its
   *  place meanwhile, invisible, so nothing under the row moves up and back when they land; a row without one gets none
   *  ("Go deeper ›" sits above the line, which never moves it). Only what changed is written. */
  setCell(cell) {
    if (this.disposed) return;
    setHidden(this.cellRow, !cell);
    const held = !!cell?.loading && !cell.line && !this.cellLine.hidden;
    if (!held) {
      setText(this.cellLine, cell?.line ?? ''); setHidden(this.cellLine, !cell?.line);
      const tip = cell?.line ? cell.tip ?? '' : '';
      if (this.cellLine.title !== tip) this.cellLine.title = tip;
    }
    const visibility = held ? 'hidden' : '';
    if (this.cellLine.style.visibility !== visibility) this.cellLine.style.visibility = visibility;
    // The test form goes with the selection (a test on its way stays, naming its own cell; so does a form of its own, Go
    // deeper's or a test run again whose start was refused, until another cell is selected).
    if (!cell) {
      this.cellLabel = 0; this.cellNameText = '';
      if (!this.testForm?.cell) this.testForm = null;
      this.renderTest(); return;
    }
    const fresh = cell.label !== this.cellLabel, name = String(cell.name ?? '');
    this.cellLabel = cell.label; this.cellNameText = name;
    if (fresh && this.testForm?.cell && this.testForm.cell.label !== cell.label) this.testForm = { ...this.testForm, cell: null };
    if (fresh) this.cellSwatch.style.background = cell.color ?? '';
    setText(this.cellName, name);
    if (this.cellRow.dataset.cell !== name) this.cellRow.dataset.cell = name;   // data, not copy: a word check leaves it out
    if (!this.deeperButton.hidden) this.options.bindDeeper(this.deeperButton, cell.label);
    this.renderTest();
  }

  setState(state) {
    if (this.disposed) return;
    const key = keyOf({ ...state, series: undefined, cells: undefined }), width = this.width;
    if (key === this.stateKey && width === this.keyWidth) return;
    const oldLayout = this.layout, oldShown = this.displayed, newRun = this.runKey !== `${state.runId}:${state.model}`;
    this.stopMotion(false); this.state = state; this.stateKey = key; this.keyWidth = width; this.runKey = `${state.runId}:${state.model}`;
    const status = state.state ?? 'loading', on = PANEL_STATES.includes(status) ? status : 'loading';
    this.root.dataset.state = on;
    const enabled = state.enabled ?? null;
    // Before the engine has said how it is, the switch says neither on nor off (it cannot be clicked yet).
    setText(this.toggle, enabled == null ? '' : enabled ? C.on : C.offLabel);
    this.toggle.setAttribute('aria-checked', enabled == null ? 'mixed' : String(!!enabled));
    this.toggle.disabled = enabled == null;
    const showing = ['live', 'paused', 'scrubbing', 'finished'].includes(on);
    const point = showing ? this.point(state) : null;
    // Coverage over all of the run's cells: no headline from too few judged cells (one of three once read "0 in 100");
    // the notices below say how many there are.
    const coverage = coverageOf(point);
    const hasNumber = !!point && (finite(point.judged) ? point.judged : HEADLINE_MIN_JUDGED) >= HEADLINE_MIN_JUDGED;
    const target = hasNumber ? headlineInteger(point) : null;
    const forming = formingPoint(point, state);
    this.target = target;
    this.renderStatus(on, point);
    setHidden(this.headline, !hasNumber);
    const start = noStart(state) ? null : state.start ?? state.summary?.start ?? null;
    const startValue = start?.per100;
    // The chart's ⓘ says what its dashed line is while it has one: proseg's count at iteration 0.
    this.tip(this.chartInfo, finite(startValue) ? `${C.chartTitle}\n${C.chartStartTooltip}` : C.chartTitle);
    if (hasNumber) {
      // Which count the result reads (decision A): on an iteration counted for a pause, the hold, a finish or the end,
      // the count proseg saves there (the frame's own numbers); its change and retention still compare the averaged
      // count like for like, which their tooltips say then.
      const saved = !state.atStart && point.count === 'sampled', count = state.atStart ? C.countAveraged : countWord(point);
      // The like-for-like change since the start (the engine's `change`, its direction only past its threshold, its
      // size unrounded to one decimal), said as like for like (each cell against itself at the start: the start's
      // number, iteration 0's, is in its tooltip, H13); while cells form the number rises by itself: no comparison (the
      // notice says when to compare); without a like-for-like change none either (two levels alone also differ by the
      // transcripts the cells lost): the start's number alone then.
      const change = !state.atStart && !forming && finite(startValue) ? likeForLike(point.change) : null;
      const s = number(startValue), verdict = verdictLine(change);
      setText(this.changeNode, verdict); setHidden(this.changeNode, !verdict);
      const startTip = [finite(startValue) ? `${copy('start', { s })}.` : '', C.startTooltip].filter(Boolean).join(' ');
      // Its two numbers on the line (the change's own: the cells judged now and at the start, each cut to the same
      // depth) say what they are after how it compares (likeForLikeNumbers); its word held a few seconds (the engine's
      // Steady: change.held) says what the change itself reads now; a live change says it is smoothed; its thresholds
      // (repeat runs', the model's) are this kind of run's constants: not yet re-measured on proseg's count while the
      // summary says so (notRemeasured), as the headline's ⓘ.
      this.tip(this.changeNode, change ? [startTip, C.changeTooltip, change.now != null ? C.likeForLikeNumbers : '', heldClause(point.change),
        point.single === false ? C.changeSmoothed : '', change.light ? C.changeLightTooltip : '', saved ? C.likeForLikeSaved : '',
        notRemeasuredOf(point, state)].filter(Boolean).join('\n') : '');
      const startText = !verdict && !state.atStart && finite(startValue) ? copy('startShort', { s }) : '';
      setText(this.startNode, startText); setHidden(this.startNode, !startText); this.tip(this.startNode, startText ? startTip : '');
      // Based on how many of all the run's cells, else (results from before cells were counted) the judged cells and
      // their typical transcripts: in the ⓘ, naming the count; a number from fewer than ROUGH_JUDGED cells says so in
      // view too.
      const basis = coverage ? copy(coverage.judged < ROUGH_JUDGED ? 'basisCoverageFew' : 'basisCoverage', { count, j: number(coverage.judged), n: number(coverage.cells) })
        : !state.atStart && finite(start?.median_tx) ? copy('basis', { count, j: number(point.judged), t: number(point.median_tx), t0: number(start.median_tx) })
          : copy('basisNoStart', { count, j: number(point.judged), t: number(point.median_tx) });
      const rough = coverage && coverage.judged < ROUGH_JUDGED ? copy('roughCells', { j: number(coverage.judged), n: number(coverage.cells) }) : '';
      setText(this.roughNode, rough); setHidden(this.roughNode, !rough);
      // The transcripts all cells kept since the start, in view when that by itself moves the number, else in the ⓘ;
      // the start line against itself says nothing.
      const retention = state.atStart ? null : retentionOf(point), retained = retentionLine(retention);
      setText(this.retentionText, retained); setHidden(this.retentionNode, !retained);
      this.tip(this.retentionInfo, [retentionTooltip(retention), retained && saved ? C.retentionSaved : ''].filter(Boolean).join(' '));
      const batch = state.batch ?? point.batch_per100;
      // First which iteration the number is from and how (H10); which count it reads, after the basis; and while this
      // kind of run's thresholds are not yet re-measured on proseg's count (the summary's constants:
      // engine/qc_constants.py), that too.
      const constants = point.constants ?? state.summary?.constants ?? null;
      const sentence = retained ? '' : retentionSentence(retention);
      this.tip(this.numberInfo, [numberLine(point, on, state.atStart), C.numberTooltip, C.meaning, `${basis}.`, saved ? C.basisTooltipSaved : C.basisTooltip,
        constants?.measured === false ? C.notRemeasured : '', sentence && saved ? `${sentence} ${C.retentionSaved}` : sentence,
        finite(batch) ? copy('earlierInfo', { b: number(batch) }) : ''].filter(Boolean).join('\n'));
    }
    // Looking back, a result of another iteration than the frame on screen (the nearest with proseg's count) says so.
    const snapped = hasNumber && !state.atStart && finite(state.snapped) ? state.snapped : null;
    const snappedText = snapped == null ? '' : copy('snappedBadge', { i: number(snapped) });
    setText(this.snappedNode, snappedText); setHidden(this.snappedNode, !snappedText);
    this.tip(this.snappedNode, snapped == null ? '' : copy('snappedTooltipMixing', { i: number(snapped) }));
    const notices = [];
    if (showing && point) {
      // A run with no start line: nothing to compare with; "from their nuclei" only where cells form from them.
      if (forming) notices.push(finite(startValue) ? C.forming : fromNuclei(state) ? C.formingNoStart : C.formingShort);
      // Too few judged cells for a number, or most cells too small to judge: from the counts over all the run's cells,
      // else (older results) from the typical judged cell.
      if (!hasNumber) {
        notices.push(coverage ? copy('tooFewCount', { j: number(coverage.judged), n: number(coverage.cells) }) : C.tooFew);
        // The transcripts all cells kept since the start, in view without a number too (it lives in the headline,
        // which is hidden then).
        const kept = state.atStart ? null : retentionOf(point);
        if (kept) notices.push(copy('retentionShort', { x: number(Math.round(kept.ratio * 100)) }));
      }
      else if (coverage ? coverage.share < LOW_COVERAGE : finite(point.median_tx) && point.median_tx < JUDGE_MIN) notices.push(C.lowDepth);
      // No notice about areas (their outlines, a single snapshot's areas, the start line's dashed ones): none is
      // shown.
    }
    if (showing && state.startSource === 'nuclei') notices.push(C.nucleiStart);
    else if (showing && state.startKind === 'paired') notices.push(C.nucleiCellsStart);
    if (showing && state.overview && point) notices.push(C.overviewNote);
    this.section(this.notices, notices, parent => notices.forEach(text => this.paragraph(parent, text)));
    // The chart: the line over the run with the start reference, the forming band and plain change ticks.
    const series = state.series ?? [];
    setHidden(this.chartBox, !showing || !series.length);
    let chartChanged = false;
    if (!this.chartBox.hidden) {
      const chartKey = keyOf([state.seriesKey ?? series.length, series.length, series.at(-1)?.seq, series.at(-1)?.per100, startValue,
        state.formingUntil, (state.changes ?? []).length, ['scrubbing', 'paused'].includes(on) ? point?.iter : null, width,
        this.options.suggestions ? [state.trial] : null]);
      chartChanged = this.chartKey !== chartKey;
      if (chartChanged) {
        this.chartKey = chartKey;
        // The start's reference line is named after what it is, proseg's count at iteration 0 ("Iteration 0: 12"), as
        // the timeline's (never after the cells the run started from).
        this.layout = layoutChart({ series, start: start ?? null, formingUntil: state.formingUntil, changes: state.changes ?? [],
          trial: this.options.suggestions ? state.trial : null,
          selectedIter: ['scrubbing', 'paused'].includes(on) && !state.atStart ? point?.iter : null, width, height: this.height });
        this.colors ??= chartColors(this.canvas);
        this.section(this.chartLegend, [this.layout.band.length > 0], parent => {
          if (this.layout.band.length) { const label = this.el('span', 'qc-band-key', C.chartExpected); label.title = C.chartExpectedTooltip; parent.append(label); }
        });
        if (this.hoverIter != null) this.showHover(hoverAt(this.layout, this.layout.x(this.hoverIter)));
      }
    } else { this.layout = null; this.chartKey = null; this.showHover(null); }
    // "Show on the cells" whenever Colour by: Mixing is offered (Layers, key M): also while the sample learns, when the
    // legend says the colours come with the first result. Switched off (the default) the panel is its heading, its switch
    // and two short sentences: no subtitle, no Show on the cells, no estimate note or explanation, which speak of results
    // there are none of; before the engine has said, or while the check cannot run, the status says it all.
    const off = state.enabled === false || on === 'off';
    const offered = state.enabled === true && (showing || ['waiting', 'learning', 'waiting-start'].includes(on));
    this.root.classList?.toggle('qc-off', off);
    // The subtitle's question and the estimate note at the foot say what the headline's ⓘ says once it shows: then only
    // that (little clutter).
    setHidden(this.subtitle, off || hasNumber); setHidden(this.estimateNote, state.enabled !== true || hasNumber); setHidden(this.how, off);
    setHidden(this.showCells, !offered);
    if (this.showCells.disabled !== off) this.showCells.disabled = off;
    const cellsTitle = off ? C.mixingOffTooltip : C.showCellsTooltip;
    if (this.showCells.title !== cellsTitle) this.showCells.title = cellsTitle;
    this.showCells.setAttribute('aria-pressed', String(!off && !!state.showingCells));
    if (this.reducedMotion) this.stopEntrance(true);   // the calm-motion switch, passed back in the state
    this.renderSlot(state, on, point, newRun);
    this.renderFinished(on, point, startValue, forming);
    this.renderHow(state);
    setHidden(this.calm, !state.motionBudgetExceeded);
    this.calm.setAttribute('aria-pressed', String(!!state.calmMotion));
    // Motion: the number counts to a new integer; the newest segment draws on. Only when allowed and visible.
    const canMove = !newRun && state.motion === true && !this.reducedMotion && this.root.open && !this.doc.hidden && !this.covered &&
      ['live', 'scrubbing', 'paused', 'finished'].includes(on) && !state.restarting;
    const moveNumber = canMove && finite(oldShown) && finite(target) && oldShown !== target;
    const oldEnd = oldLayout?.selected ?? oldLayout?.points.at(-1), newEnd = this.layout?.selected ?? this.layout?.points.at(-1);
    const moveLine = canMove && chartChanged && finite(oldEnd?.y) && finite(newEnd?.y) && (oldEnd.iter !== newEnd.iter || oldEnd.per100 !== newEnd.per100);
    if (moveNumber || moveLine) {
      this.motion = { startTime: this.clock.now(), from: oldShown, to: target, number: moveNumber, line: moveLine,
        endFrom: this.layout?.selected ? oldEnd : (oldEnd.iter === newEnd.iter ? oldEnd : null), frame: 0,
        duration: moveNumber ? NUMBER_MS : SEGMENT_MS };
      if (chartChanged && !moveLine) this.draw();
      this.paintMotion(0); this.scheduleMotion();
    } else {
      this.displayNumber(target);
      if (chartChanged) this.draw();
    }
    this.renderTest();   // the test's form asks for its pool again once the check is switched (askPool's key)
  }

  renderStatus(status, point) {
    // The room learning needs on the graphics card (state.room): its numbers are in the waiting and too-small words.
    const state = this.state, plain = reason => reasonText(status, reason, state.room);
    const room = state.room ? [state.room.need_mb, state.room.free_mb, state.room.total_mb, state.room.genes] : null;
    this.section(this.status, [status, state.progress, state.eta_s, state.reason, state.shared, state.relearn, state.requested,
      state.restarting, point?.iter, state.atStart, state.startSource, state.startKind, state.failed, status === 'off' ? state.sampleState : null,
      status === 'off' ? state.sampleReason : null, status === 'unavailable' ? [state.prosegFix ?? null, !!state.notebook] : null,
      status === 'scrubbing' ? state.frameIter ?? null : null, room], parent => {
      let text = null;
      if (status === 'loading' && state.failed) {         // the run's check could not be reached: said, and asked again
        this.paragraph(parent, C.loadFailed);
        parent.append(this.button(C.tryAgain, 'reload', () => this.call('onReload')));
      } else if (status === 'off') {                    // what turning it on does, and what it costs on this sample
        this.paragraph(parent, C.off, 'qc-small');
        if (state.sampleState === 'ready') this.paragraph(parent, C.offLearned, 'qc-small qc-muted');
        else if (state.sampleState === 'off') this.paragraph(parent, C.offNotLearned, 'qc-small qc-muted');
        // A sample the check cannot learn, or could not: said with the switch off too (it said nothing, and turning
        // it on only to read why).
        else if (state.sampleState === 'unavailable') this.paragraph(parent, reasonText('unavailable', state.sampleReason), 'qc-small qc-muted');
        else if (state.sampleState === 'failed') this.paragraph(parent, copy('failed', { reason: reasonText('failed', state.sampleReason, state.room) }), 'qc-small qc-muted');
      }
      else if (status === 'waiting') {
        this.paragraph(parent, plain(state.reason));
        if (state.shared) this.paragraph(parent, C.cardShared);
      } else if (status === 'waiting-start') text = C.waitingStart;
      else if (status === 'learning') {
        const of = Math.max(1, state.progress?.of ?? 50), step = Math.max(0, Math.min(of, state.progress?.step ?? 0));
        const line = finite(state.eta_s) ? copy(of === 50 ? 'learning' : 'learningProgress', { k: number(step), of: number(of), t: duration(state.eta_s) })
          : copy('learningNoTime', { k: number(step), of: number(of) });
        this.paragraph(parent, line);
        const progress = this.el('progress', 'qc-progress'); progress.max = of; progress.value = step; progress.setAttribute('aria-label', line); parent.append(progress);
        this.paragraph(parent, state.relearn ? copy('softwareChanged', { t: duration(state.eta_s) }) : C.learningNote);
        if (state.shared) this.paragraph(parent, C.cardShared);
      } else if (status === 'unavailable') {
        // A run without proseg's own count: why a new run has it, or how to update an old proseg-live, in the line's
        // tooltip (unavailableTooltip).
        this.paragraph(parent, plain(state.reason));
        const tip = unavailableTooltip(state.reason, { prosegFix: state.prosegFix, notebook: state.notebook });
        if (tip) parent.lastChild.title = tip;
      } else if (status === 'failed') {
        this.paragraph(parent, copy('failed', { reason: plain(state.reason) }));
        parent.append(this.button(C.tryAgain, 'retry', () => this.call('onRetry')));
      } else if (['live', 'paused', 'scrubbing', 'finished'].includes(status)) {
        if (state.restarting) text = copy('workerRestarting', { i: number(point?.iter) });
        else if (state.atStart) text = noStart(state) ? null : copy('startHeading');
        else if (!point) text = C.waitingFrame;
        // The frame on screen's iteration (the badge names the result's when it is another's).
        else if (status === 'scrubbing') text = copy('notNewest', { i: number(finite(state.frameIter) ? state.frameIter : point.iter) });
      }
      this.paragraph(parent, text);
      if (state.requested && ['waiting', 'learning'].includes(status)) this.paragraph(parent, C.requested);
    });
  }

  /** The card's change deferred while one of its buttons was pressed or focused. */
  refreshCard() {
    if (this.disposed || !this.cardStale || this.cardPressed || this.cardFocused) return;
    this.stateKey = null; this.setState(this.state);
  }

  /** The suggestions (options.suggestions): the suggestion card and the trial readout in the slot, the comparison with
   *  (nothing slides in then). */
  renderSlot(state, status, point, newRun = false) {
    const on = this.options.suggestions === true;
    setHidden(this.slot, !on);
    if (!on) {
      if (this.slot._qcEmpty !== true) { this.suggestions.replaceChildren(); this.trial.replaceChildren(); this.slot._qcEmpty = true; }
      this.shown = { card: null, trial: null };
      return;
    }
    this.slot._qcEmpty = false;
    if (newRun) { this.chosenIdea = null; this.whyOpen = null; this.moreOpen = false; this.cardKey = null; this.trialKey = null; }
    const card = this.renderSuggestions(point, status, newRun), trial = this.renderTrial(point, status, newRun);
    this.shown = { card, trial };
  }

  /** The one card (the highest-ranked idea, never while a trial runs, never for a frame that is not the
   *  newest), about 25 words in view (short cards):
   *  its title, one short why (no "for area A" line: "cells where mixing stands out"), the Setting line with its hint in an ⓘ, a short
   *  confidence and the buttons; everything else it said (the why's other sentences, the cost, the timing, the tests
   *  behind it, the footer's caveats) folds under Why? (suggestionModel `more`), and while paused Try it says in its
   *  tooltip that it applies when the run resumes. The other ideas under "More ideas". A card is known by the engine's
   *  id (its rule and change, never its areas), so areas joining or leaving it change its words only; an idea picked
   *  under "More ideas" stays picked (and shown for CHOSEN_GRACE_MS when a result omits it). Built once per card and
   *  action set, its words then updated in place, and never rebuilt while one of its buttons is pressed or has focus: a
   *  click lands on the card it was aimed at. A card that appears slides in once, but not for a result that lands while
   *  paused or finished (state.still). Returns what it shows (the harness). */
  renderSuggestions(point, status, newRun) {
    const state = this.state, running = !!state.trial && !state.trial.outcome, ended = status === 'finished';
    let ideas = [state.suggestion, ...(state.moreIdeas ?? [])].filter(s => s && RULE_COPY[s.rule]);
    if (ended) ideas = ideas.filter(s => s.rule !== 'U');           // nothing to put back in a run that has ended
    ideas = ideas.filter((s, i) => ideas.findIndex(t => cardKey(t) === cardKey(s)) === i);
    let chosen = this.chosenIdea ? ideas.find(s => cardKey(s) === this.chosenIdea) ?? null : null;
    if (chosen) { this.chosenLast = chosen; this.chosenMissing = null; }
    else if (this.chosenIdea) {                                       // picked, and missing from this result
      const now = this.clock.now();
      this.chosenMissing ??= now;
      if (this.chosenLast && !running && now - this.chosenMissing < CHOSEN_GRACE_MS) { chosen = this.chosenLast; ideas = [...ideas, chosen]; }
      else { this.chosenIdea = null; this.chosenLast = null; this.chosenMissing = null; }
    }
    // Rule R0: no card before iteration 10, nor for an area not confirmed; the run's own areas come at burn-in + 20,
    // so while cells form and nothing is offered the slot says it is too early (not beside the words of
    // a trial just kept or undone: its card comes back once it has led again).
    const tooEarly = !!point && !ended && (point.phase === 'init' || (finite(point.iter) && point.iter < 10)
      || (!ideas.length && !state.trialEnded && formingPoint(point, state)));
    const hidden = !point || running || state.atStart || !['live', 'paused', 'finished'].includes(status) || (!ideas.length && !tooEarly);
    setHidden(this.suggestions, hidden);
    // A card hidden for a moment (a trial, a scrub) and shown again is the same card: only one that went slides in anew.
    if (hidden) {
      this.suggestions._qcKey = null; this.cardParts = null; this.cardPressed = this.cardFocused = false;
      if (!ideas.length) this.cardKey = null;
      return null;
    }
    if (tooEarly) {
      const early = fromNuclei(state) ? C.tooEarlyNuclei : C.tooEarly;   // "from their nuclei" only where cells form from them
      this.section(this.suggestions, ['early', early], parent => { parent.append(this.el('h3', '', C.suggestions)); this.paragraph(parent, early); });
      this.cardParts = null; return null;
    }
    chosen ??= ideas[0];
    // "Look at its cells" only where the page can move the view there (options.onInspect).
    const context = { knobLabel: this.options.knobLabel, phase: point.phase, ended, laterTime: state.laterTime ?? null,
      inspect: typeof this.options.onInspect === 'function' };
    const model = suggestionModel(chosen, state, context);
    const extras = ideas.filter(s => s !== chosen);
    const structure = [model.key, model.actions, model.primary, model.undo, model.showStart, !!model.setting, !!model.settingHint,
      model.more.map(p => p.kind), extras.map(cardKey), ended];
    const fresh = this.suggestions._qcKey !== keyOf(structure);
    if (fresh && this.cardParts && (this.cardPressed || this.cardFocused)) {
      // A button of the card shown is pressed or has focus: the card stays as it is until it is let go.
      this.cardStale = true;
      return this.cardShown ?? null;
    }
    this.cardStale = false;
    this.section(this.suggestions, structure, parent => {
      parent.append(this.el('h3', '', C.suggestions));
      const card = this.el('article', 'qc-suggestion-card'); card.dataset.rule = chosen.rule;
      // In view: the title, the why, the Setting line (its words, then the ⓘ with its hint), the confidence and the
      // buttons; under Why? (.qc-evidence, folded) one paragraph per `more` entry.
      const parts = this.cardParts = { card, buttons: [], title: this.el('h4'), why: this.el('p', 'qc-card-why'),
        setting: model.setting ? this.el('p', 'qc-setting') : null, settingText: model.setting ? this.el('span') : null,
        settingInfo: model.setting && model.settingHint ? this.info('') : null,
        confidence: this.el('p', 'qc-confidence'), more: this.el('div', 'qc-evidence'),
        lines: model.more.map(p => this.el('p', MORE_CLASSES[p.kind])) };
      if (parts.setting) parts.setting.append(...[parts.settingText, parts.settingInfo].filter(Boolean));
      parts.more.hidden = true; parts.more.append(...parts.lines);
      card.append(...[parts.title, parts.why, parts.setting, parts.confidence].filter(Boolean));
      const actions = this.el('div', 'qc-actions'), keep = button => { parts.buttons.push(button); actions.append(button); };
      if (model.undo) {
        keep(this.button(C.undoIt, 'undo-suggestion', () => this.call('onUndo', parts.model.source)));
        keep(this.button(C.keep, 'keep-suggestion', () => this.call('onKeep', parts.model.source)));
      }
      if (model.showStart) {
        // The starting outlines on, and the view moved to the card's area (its outline the result's, never drawn).
        keep(this.button(copy('showStartOutlines', { start: startName(state.startSource) }), 'show-start',
          () => this.call('onShowCells', true, { start: true, areas: lettersOf(parts.model.source),
            area: parts.model.inspectArea ?? { L: lettersOf(parts.model.source)[0] }, suggestion: parts.model.source })));
      }
      for (const action of model.actions) {
        const button = this.button(action === 'try' ? C.try : action === 'inspect' ? C.inspect : C.newRun, action,
          () => action === 'inspect' ? this.inspect(parts.model) : this.call(action === 'try' ? 'onTry' : 'onNewRun', parts.model.source));
        // The main action stands out: New run the app's primary button (filled), Try it its accent outline (qc.css).
        if (action === model.primary) { button.dataset.primary = ''; if (action === 'new-run') button.className += ' primary'; }
        keep(button);
      }
      // Why? on every card: each has at least its footer's caveats under it.
      const why = this.button(C.why, 'why', () => {
        const open = parts.more.hidden; this.whyOpen = open ? parts.model.key : null;
        setHidden(parts.more, !open); why.setAttribute('aria-expanded', String(open)); this.call('onWhy', parts.model.source, open);
      });
      why.className += ' qc-text-button'; parts.whyButton = why; actions.append(why);
      card.append(actions, parts.more); parent.append(card);
      if (extras.length) {
        const more = this.el('details', 'qc-more'); more.open = !!this.moreOpen;
        more.addEventListener?.('toggle', () => { this.moreOpen = more.open; });   // goes with the element
        more.append(this.el('summary', '', copy('moreIdeas', { k: number(extras.length) })));
        for (const idea of extras) {
          const candidate = suggestionModel(idea, state, context);
          more.append(this.button(candidate.title, 'idea', () => {
            this.chosenIdea = cardKey(idea); this.chosenLast = idea; this.chosenMissing = null; this.whyOpen = null;
            this.call('onIdea', idea); this.cardPressed = this.cardFocused = false; this.stateKey = null; this.setState(this.state);
          }));
        }
        parent.append(more);
      }
    });
    const parts = this.cardParts; parts.model = model;
    setText(parts.title, model.title); setText(parts.why, model.why);
    if (parts.setting) { setText(parts.settingText, model.setting); if (parts.settingInfo) this.tip(parts.settingInfo, model.settingHint); }
    setText(parts.confidence, model.confidence); if (parts.confidence.title !== model.confidenceTip) parts.confidence.title = model.confidenceTip;
    model.more.forEach((p, i) => { setText(parts.lines[i], p.text); setHidden(parts.lines[i], !p.text); if (parts.lines[i].title !== (p.tip ?? '')) parts.lines[i].title = p.tip ?? ''; });
    const open = this.whyOpen === model.key;
    setHidden(parts.more, !open); parts.whyButton.setAttribute('aria-expanded', String(open));
    // Nothing is clicked twice: every action waits while one is on its way (state.acting); New run says why it cannot
    // Paused, Try it says that it applies when the run resumes (its trial readout says it in view once pressed).
    const still = ['paused', 'finished'].includes(status);
    for (const button of parts.buttons) {
      const action = button.dataset.action, disabled = !!state.acting;
      if (button.disabled !== disabled) button.disabled = disabled;
      const title = action === 'try' ? (status === 'paused' ? `${model.tooltip} ${C.tryPaused}` : model.tooltip) : action === 'inspect' ? C.inspectTooltip
        : action === 'show-start' ? copy('showStartTooltip', { start: startName(state.startSource) })
          : action === 'new-run' ? (still ? C.newRunTooltipStill : C.newRunTooltip) : '';
      if (title && button.title !== title) button.title = title;
    }
    // A card that has just appeared slides in once; the same card with new numbers stays put; a card that lands with a
    // result while paused or finished lands without motion.
    if (model.key !== this.cardKey) {
      this.cardKey = model.key;
      if (fresh) { this.call('onCard', model.source); if (!newRun && state.still !== true) this.entrance(parts.card); }
    }
    this.cardShown = { rule: model.id, areas: lettersOf(chosen), flag: model.flag, to: chosen.to ?? null, actions: model.actions, primary: model.primary };
    return this.cardShown;
  }
  /** "Look at its cells" (the engine's `inspect`): the page's own way (options.onInspect(suggestion, area): the view
   *  moves to the card's area, no outline drawn, and Go deeper opens on its most mixed cell); each press moves it again. */
  inspect(model) {
    const area = model?.inspectArea;
    if (area) this.call('onInspect', model.source, area);
  }
  /** "Trying {setting} since iteration {i}": what is watched, the expected -> now rows, the
   *  engine's verdict in plain words, its notes, and Keep / Undo while it runs (and, for a shape knob, New run with this
   *  setting: the readout's own advice for a clear answer); after Keep, Undo or the run's end, what was tried, in the
   *  past ("Tried ... from iteration i to j (undone)"), for a few iterations. Updated in place, so its buttons survive
   *  the results that come meanwhile. */
  renderTrial(point, status, newRun) {
    const state = this.state, trial = state.trial ?? state.trialEnded ?? null;
    const hidden = !point || !trial || state.atStart || !['live', 'paused', 'scrubbing', 'finished'].includes(status);
    setHidden(this.trial, hidden);
    if (hidden) { this.trial._qcKey = null; this.trialParts = null; this.trialKey = null; return null; }
    // Whether it watches an area too: said as "where mixing stands out", never by its letter.
    const area = (trial.area ?? lettersOf(trial)[0] ?? null) != null, started = finite(trial.start_iter);
    const outcome = trial.outcome ?? null, live = !outcome && ['live', 'paused'].includes(status);
    const shape = SHAPE_RULES.includes(trial.rule) && typeof trial.flag === 'string';
    const runEnded = outcome === 'ended' || (status === 'finished' && !outcome);
    // A run with no start line gets no verdict, ever: its readout says so (why in the tooltip), with no rows, count,
    // area or loss line that would read as a comparison or a wait for one.
    const none = noStart(state);
    const rows = none ? [] : (trial.rows ?? []).filter(row => row && typeof row === 'object');
    const structure = [trial.id ?? null, outcome, rows.map(row => [row.kind, row.area]), live, status === 'paused' && live && !started, runEnded, live && shape];
    const fresh = this.trial._qcKey !== keyOf(structure);
    this.section(this.trial, structure, parent => {
      const parts = this.trialParts = { title: this.el('h4'), watching: this.el('p', 'qc-small'), columns: this.el('p', 'qc-small qc-muted', C.trialColumns),
        rows: [], verdict: this.el('p', 'qc-verdict'), small: this.el('p', 'qc-small'), depth: this.el('p', 'qc-small'), buttons: [] };
      parts.verdict.setAttribute('aria-live', 'polite');   // a verdict that comes is read out
      parent.append(parts.title, parts.watching);
      if (rows.length) {
        const box = this.el('div', 'qc-trial-rows'); parent.append(parts.columns, box);
        for (const row of rows) {
          const line = this.el('div', 'qc-trial-row'), text = this.el('span', 'qc-trial-text'), glyph = this.el('span', 'qc-row-change');
          glyph.setAttribute('aria-hidden', 'true'); line.append(text, glyph); box.append(line); parts.rows.push({ text, glyph });
        }
      }
      parent.append(parts.verdict, parts.small, parts.depth);
      if (runEnded) this.paragraph(parent, C.trialEnded);
      if (live) {
        const actions = this.el('div', 'qc-actions');
        for (const [text, action, name] of [[C.keep, 'keep', 'onKeep'], [C.undo, 'undo', 'onUndo']]) {
          const button = this.button(text, action, () => this.call(name, parts.trial)); parts.buttons.push(button); actions.append(button);
        }
        if (shape) {           // the readout's advice ("for a clear answer, start a new run with this setting"), as a button
          const button = this.button(C.newRun, 'new-run', () => this.call('onNewRun', parts.trial)); parts.buttons.push(button); actions.append(button);
        }
        parent.append(actions);
        if (status === 'paused' && !started) this.paragraph(parent, C.tryPaused, 'qc-small qc-paused-note');
      }
    });
    const parts = this.trialParts; parts.trial = trial;
    const setting = tryingLabel(trial) || C.missing, k = started && finite(point.iter) ? Math.max(0, point.iter - trial.start_iter) : 0;
    const i = number(trial.start_iter), j = number(finite(trial.endedIter) ? trial.endedIter : point.iter);
    setText(parts.title, !started ? copy('trialTitleWaiting', { setting }) : outcome === 'kept' ? copy('trialTitleKept', { setting, i })
      : outcome === 'undone' ? copy('trialTitleUndone', { setting, i, j }) : outcome === 'ended' ? copy('trialTitleEnded', { setting, i })
        : copy('trialTitle', { setting, i }));
    // What is watched, counting on: only while it runs.
    const watching = started && !outcome && !none;
    setText(parts.watching, !watching ? '' : copy(area ? 'trialWatching' : 'trialWatchingWhole', { k: number(k) }));
    setHidden(parts.watching, !watching);
    rows.forEach((row, n) => {
      const { text, glyph } = parts.rows[n];
      setText(text, trialRowText(row));
      const direction = trialRowDirection(trial, row);
      if (glyph.dataset.direction !== direction) glyph.dataset.direction = direction;
      setHidden(glyph, !direction);
    });
    // The verdict line: the engine's verdict in plain words while it runs (and as it was when its run ended); after Keep
    // or Undo, what was done.
    const laterTime = finite(trial.later_s) && trial.later_s > 0 ? duration(trial.later_s) : state.laterTime ?? null;
    const name = knobName(typeof this.options.knobLabel === 'function' ? this.options.knobLabel(trial.flag) : null);
    const toggle = typeof trial.from === 'boolean' || typeof trial.to === 'boolean';
    // A switch says what putting it back does ("Put back drift off."): its name may read backwards ("Switch off ...: on").
    const back = toggle ? changeLabel(knobChange(trial.flag, trial.to, trial.from) ?? {}) : '';
    const undone = back ? copy('trialUndoneChange', { change: back })
      : name && !bannedWords(name).length && !toggle ? copy('trialUndone', { label: name, old: settingValue(trial.from) })
      : copy('trialUndoneValue', { old: settingValue(trial.from) });
    setText(parts.verdict, outcome === 'kept' ? C.trialKept : outcome === 'undone' ? undone
      : started ? (none ? C.trialNoStart : trialVerdictText(trial, { k, laterTime, formed: !formingPoint(point, state) })) : '');
    // A verdict rests on the trial minima and band (this kind of run's constants): the caveat while not re-measured; a
    // line giving the timing figures says they come from the outline-era tests (H5).
    this.tip(parts.verdict, started && !outcome && none ? C.trialNoStartTooltip : started && !outcome
      ? [trialVerdictReached(trial) ? notRemeasuredOf(point, state) : '', trialVerdictOutline(trial) ? C.outlineEraTooltip : ''].filter(Boolean).join('\n') : '');
    setHidden(parts.verdict, !(outcome === 'kept' || outcome === 'undone') && !started);
    const small = !outcome && !none && noteOn(trial, 'small_area') && area ? C.trialSmall : '';
    setText(parts.small, small); setHidden(parts.small, !small);
    // The transcripts the cells lost (the engine's whole %, given from 5 % on): the comparison allows for it.
    const depth = !outcome && !none && finite(trial.depth_loss) && trial.depth_loss >= 1 ? copy('trialDepthNote', { x: number(Math.round(trial.depth_loss)) }) : '';
    setText(parts.depth, depth); setHidden(parts.depth, !depth);
    for (const button of parts.buttons) {
      const disabled = !!state.acting;
      if (button.disabled !== disabled) button.disabled = disabled;
      if (button.dataset.action === 'new-run') {
        const title = status === 'paused' ? C.newRunTooltipStill : C.newRunTooltip;
        if (button.title !== title) button.title = title;
      }
    }
    const key = `${trial.id}:${outcome}`;
    if (key !== this.trialKey) { const before = this.trialKey; this.trialKey = key; if (fresh && before === null && !outcome && !newRun) this.entrance(this.trial); }
    return { id: trial.id ?? null, rule: trial.rule ?? null, flag: trial.flag ?? null, from: trial.from ?? null, to: trial.to ?? null,
      start_iter: started ? trial.start_iter : null, verdict: trial.verdict ?? null, outcome };
  }

  /** A card or readout that has just appeared slides and fades in once (ENTRANCE, on the stage's 150 ms steps: at most
   *  3 paints), never with reduced motion or the calm-motion switch, in a collapsed panel, a hidden or covered page; it
   *  never loops and nothing is left scheduled afterwards. */
  entrance(node) {
    this.stopEntrance(true);
    if (!node || this.reducedMotion || !this.root.open || this.doc.hidden || this.covered || this.slot.hidden) return;
    const paint = step => { const [opacity, y] = ENTRANCE[step]; node.style.opacity = step === ENTRANCE.length - 1 ? '' : String(opacity);
      node.style.transform = step === ENTRANCE.length - 1 ? '' : `translateY(${y}px)`; };
    const motion = this.entering = { node, step: 0, startTime: this.clock.now(), timer: null, frame: null };
    paint(0);
    const next = () => {
      motion.timer = this.clock.setTimeout(() => {
        motion.timer = null;
        motion.frame = this.clock.requestAnimationFrame(() => {
          motion.frame = null; if (this.entering !== motion || this.disposed) return;
          paint(++motion.step); this.call('onFrame');
          if (motion.step < ENTRANCE.length - 1) next(); else this.entering = null;
        });
      }, Math.max(0, motion.startTime + (motion.step + 1) * MOTION_STEP_MS - this.clock.now()));
    };
    next();
  }
  stopEntrance(finish = false) {
    const motion = this.entering; this.entering = null;
    if (!motion) return;
    if (motion.timer != null) this.clock.clearTimeout(motion.timer);
    if (motion.frame != null) this.clock.cancelAnimationFrame(motion.frame);
    if (finish) { motion.node.style.opacity = ''; motion.node.style.transform = ''; }
  }

  /** A run that ended: finished, or stopped by the user or the engine ("Stopped at iteration ..."); one that ended while
   *  cells still formed is not compared with the start, nor is one with no start line (which never names a start, nor
   *  a comparison to come). The whole tissue's numbers only: no line names a flagged area, nor says none stood out (no
   *  area is shown). */
  renderFinished(status, point, startValue, forming = false) {
    const state = this.state, done = !['stopped', 'failed'].includes(state.runState);
    setHidden(this.finished, status !== 'finished' || !point || state.atStart);
    // Too few judged cells for a number: the run's end without one.
    const few = !!point && finite(point.judged) && point.judged < HEADLINE_MIN_JUDGED;
    this.section(this.finished, [this.finished.hidden, point?.iter, headlineInteger(point), startValue, done, forming, few], parent => {
      if (this.finished.hidden) return;
      const i = number(point.iter);
      if (few) this.paragraph(parent, copy(done ? 'finishedFew' : 'stoppedFew', { i }));
      else if (forming && !noStart(state)) this.paragraph(parent, copy(done ? 'finishedForming' : 'stoppedForming', { i }));
      else if (!finite(startValue)) this.paragraph(parent, copy(done ? 'finishedNoStart' : 'stoppedNoStart', { i, n: number(headlineInteger(point)) }));
      else this.paragraph(parent, copy(done ? 'finished' : 'stoppedAt', { i, n: number(headlineInteger(point)), s: number(startValue) }));
    });
  }

  /** "How is this estimated?": five short points (the model, how it reads a cell, its blind spots, like
   *  for like, the count it reads: proseg's own), then how the model was learned (qc-copy.mjs learnedLine: its cells,
   *  when and how long, a small table's passes; model.provenance). Nothing in it names files outside the app. */
  renderHow(state) {
    const model = state.sampleModel ?? null, provenance = state.summary?.provenance ?? model?.provenance ?? null;
    this.section(this.howBody, [model?.cells, model?.created, model?.seconds, model?.table, provenance], parent => {
      const steps = this.el('ol');
      for (const key of HOW_KEYS) steps.append(this.el('li', '', C[key]));
      parent.append(steps);
      const learned = learnedLine(model, provenance, { date: dateText(model?.created), t: duration(model?.seconds) });
      if (learned) { const line = this.el('p', 'qc-how-learned', learned.text); if (learned.tooltip) line.title = learned.tooltip; parent.append(line); }
    });
  }

  // ---- "Test a setting here…", shown in the stage's corner ----------
  /** The run a test belongs to (main.js: every run update, and the run's reset): its record, with its id, kind, state,
   *  the end of its schedule (`schedule_done`), the `qc_test` {state, progress} its detail and pushes carry while a test
   *  exists, how it started (testStart), its options, live values and info. A test this page does not hold, or one in
   *  another state, is asked for (scheduleTestPoll: at most once a second); a push's progress is taken as it comes. A run
   *  update without `qc_test` says nothing either way (the runs list may leave it out): a test on its way is asked for
   *  each second, and only the engine's 404 ends it. */
  setTestRun(run) {
    if (this.disposed) return;
    const id = run?.id ?? null;
    if (id !== this.testRunId) this.resetTest(id);
    this.testRunState = run?.state ?? null; this.testRun = run && typeof run === 'object' ? run : null;
    const push = run?.qc_test, test = this.testRec;
    if (push && typeof push === 'object' && TEST_STATES.includes(push.state)) {
      if (test?.state === push.state) {
        if (push.progress && typeof push.progress === 'object' && keyOf(push.progress) !== keyOf(test.progress ?? null)) this.testRec = { ...test, progress: push.progress };
      } else this.scheduleTestPoll(true);
    }
    this.renderTest();
  }
  /** Another run (or none): its test is none of this one's, and nothing more is asked for the old one. */
  resetTest(id = null) {
    if (this.testTimer != null) this.clock.clearTimeout(this.testTimer);
    this.testTimer = null; this.testRunId = id; this.testRun = null; this.testRec = null; this.testForm = null; this.testWanted = false;
    this.testFetchedAt = -Infinity; this.testDismissed = new Set(); this.testStarting = false; this.testBusy = null;
    this.testShownProblems = []; this.testMessage = null; this.testRefused = null; this.testEarlier = null; this.dropPool();
  }
  /** A setting as the page has it ({label (Live knobs), value (the run's now), min, max}: options.testSetting), or {}. */
  testSetting(flag) {
    const setting = typeof this.options.testSetting === 'function' ? this.options.testSetting(flag) : null;
    return setting && typeof setting === 'object' ? setting : {};
  }
  /** A test can be asked for cell `label` (a label; the selected cell's by default): the page wires a test (onTestStart),
   *  the run is a proseg run, whatever its state, and there is a cell. Neither the run's state nor the mixing check's
   *  hides the entry: they are reasons, which the form lists (the engine's). */
  testable(label = this.cellLabel) {
    return typeof this.options.onTestStart === 'function' && this.testRunId != null && (this.testRun?.kind ?? 'proseg') === 'proseg'
      && Number.isInteger(label) && label > 0;
  }
  /** How the test's run started: the test's own word, else the run's record (testStart). */
  testStartKind() { return testStart(this.testRun, this.testRec); }
  /** The cell a form is for: its own (Go deeper's, or a test run again whose start was refused), else the selected one. */
  testCell() {
    return this.testForm?.cell ?? (this.cellLabel ? { label: this.cellLabel, name: this.cellNameText } : null);
  }
  /** What the section shows: the test (its state) while there is one, else for a cell a test can be asked for the form
   *  while it is open, else "Test a setting here…"; else nothing. */
  testMode() {
    if (this.testRec) return this.testRec.state;
    const cell = this.testCell();
    if (!cell || !this.testable(cell.label)) return 'none';
    return this.testForm ? 'form' : 'entry';
  }

  /** The section: "Test a setting here…", or one line while its test is open in the stage's corner (the
   *  corner holds the form, the progress and the answer), each built once and written in place; the form's pool asked
   *  for while the form shows (askPool); then the corner's model (emitCorner). Its data-state is the mode (the harness
   *  reads it). */
  renderTest() {
    if (this.disposed) return;
    const mode = this.testMode(), box = this.testBox, view = mode === 'none' ? 'none' : mode === 'entry' ? 'entry' : 'open';
    if (box._qcKey !== view) {
      box._qcKey = view; box.replaceChildren(); this.testParts = null;
      if (view === 'entry') this.buildTestEntry(box);
      else if (view === 'open') this.buildTestStatus(box);
    }
    if (box.dataset.state !== mode) box.dataset.state = mode;
    setHidden(box, mode === 'none');
    if (view === 'open') {
      const phase = mode === 'waiting' ? 'waiting' : TEST_ACTIVE.includes(mode) ? 'running' : mode === 'done' ? 'done'
        : TEST_STATES.includes(mode) ? 'stopped' : 'open';
      const status = this.testParts.status;
      setText(status, C[TEST_STATUS[phase]]);
      if (status.dataset.phase !== phase) status.dataset.phase = phase;
    }
    if (mode === 'form') this.askPool(); else if (!this.testStarting) this.dropPool();
    this.emitCorner();
  }
  buildTestEntry(box) {
    const open = this.el('button', 'qc-button qc-test-open', C.testOpen); open.type = 'button'; open.dataset.action = 'test-open';
    open.title = C.testOpenTooltip;
    open.onclick = event => { event?.stopPropagation?.(); this.openTestForm(); };
    box.append(open);
  }
  /** While a test is open: its phase in a few words (the corner holds the rest). */
  buildTestStatus(box) {
    const status = this.el('p', 'qc-small qc-test-status'); status.title = C.testStatusTooltip; status.setAttribute('role', 'status');
    this.testParts = { status }; box.append(status);
  }
  /** "Test a setting here…": the form, in the corner, for the selected cell or for `cell` ({label, name}: Go deeper's
   *  own, opened from its window): the first setting of the one order every start has, its value empty beside the run's
   *  value now (nothing is suggested); the pool is asked for at once. Opened for a cell from elsewhere, the page shows
   *  this panel too (onTestShow). A test that is open stays: its corner shows it. */
  openTestForm(cell = null) {
    if (this.disposed) return;
    const own = cell && Number.isInteger(cell.label) && cell.label > 0 ? { label: cell.label, name: String(cell.name ?? '') } : null;
    if (!own && !this.cellLabel) return;
    if (!this.testRec) {
      this.testForm = { flag: TEST_FLAGS[0], text: '', cell: own };
      this.testShownProblems = []; this.testMessage = null; this.testEarlier = null;
      this.renderTest();
    }
    if (own) this.call('onTestShow');
  }
  /** How firmly the run holds its starting cells (holdOf, from its record, else its test's word). */
  testHold() { return holdOf(this.testRun, this.testRec); }
  /** A setting as the form offers it (the corner's choice): its name, the run's value now, whether a test tried it (far
   *  drift: "Not tried here yet.", what it does in its tooltip: the form's only notes), its bounds (TEST_RANGES, else
   *  Live knobs'). Never a value: nothing is suggested. `from`: the value the test compares with (its Setting line's
   *  first), the value now but in a test of the hold its value at the frame tested once the pool reply knows that
   *  frame (holdAtEnd); one set live since then keeps "now" and says in its tooltip what the test compares with
   *  (`nowTip`, testNowEnd). */
  testEntry(flag) {
    const setting = this.testSetting(flag), range = TEST_RANGES[flag], tried = TEST_TRIED[flag] !== false, kind = TEST_KINDS[flag] ?? 'number';
    const replied = this.testReplySettings()?.[flag] ?? null;
    // The value now: Live knobs' (the switch's as 1 / 0), but the pool reply's for those proseg reads at its start.
    const knob = typeof setting.value === 'boolean' ? (setting.value ? 1 : 0) : finite(setting.value) ? setting.value : null;
    const now = TEST_STARTUP.includes(flag) ? (finite(replied?.now) ? replied.now : null) : knob ?? (finite(replied?.now) ? replied.now : null);
    const atEnd = flag === 'prior-weight' ? holdAtEnd(this.testReplyHold()) : null, since = atEnd != null && !(now != null && sameSetting(atEnd, now));
    const [, tipKey] = TEST_WORDS[flag] ?? [null, null], driftIsOff = driftOff(this.testRun);
    const refusal = DRIFT_OFF_WORDS[flag] && driftIsOff ? C[DRIFT_OFF_WORDS[flag]] : replied?.refusal?.code === 'flat' ? C.testFlat : '';
    const choices = kind === 'switch' ? [1, 0] : kind === 'choice' && Array.isArray(replied?.choices) ? replied.choices.filter(finite) : null;
    const pick = (a, b, c) => (finite(a) ? a : finite(b) ? b : finite(c) ? c : null);
    return { flag, name: testName(flag), kind, unit: TEST_UNITS[flag] ?? null, now, nowText: testNowText(now, flag),
      nowTip: since ? copy('testNowEnd', { v: settingValue(atEnd) }) : '', from: atEnd ?? now, tried,
      note: refusal || (tried ? '' : C.testNotTriedNote), tip: tried || !tipKey ? '' : copy(tipKey, { v: settingValue(replied?.voxel ?? 1) }),
      choices: choices && choices.map(v => ({ value: v, text: testValueText(flag, v, 'short'), now: now != null && sameSetting(v, now) })),
      refusal: refusal || null,
      min: range ? range[0] : pick(replied?.min, setting.min), max: range ? range[1] : pick(replied?.max, setting.max) };
  }
  /** A test's Setting line: Live knobs' words for a live setting ("Setting: Cell compactness 0.04 → 0.02 …", the switch as
   *  its tick), the deck's own for one proseg reads at its start, which Live knobs does not hold ("Setting: Starting grid
   *  2 µm → 1 µm"); '' without both values. */
  testSettingLine(flag, from, to) {
    if (!finite(from) && !finite(to)) return settingLine({ flag, from: C.missing, to: C.missing, label: this.testSetting(flag).label ?? null }) ?? '';
    if (TEST_STARTUP.includes(flag)) {
      return copy('testSettingStart', { name: C[TEST_WORDS[flag][0] + 'Plain'] ?? testName(flag), old: testValueText(flag, from) ?? C.missing,
        new: testValueText(flag, to) ?? C.missing });
    }
    const switched = TEST_KINDS[flag] === 'switch' && finite(from) && finite(to);
    return settingLine({ flag, from: switched ? from === 1 : finite(from) ? from : C.missing, to: switched ? to === 1 : finite(to) ? to : C.missing,
      label: this.testSetting(flag).label ?? null }) ?? '';
  }
  /** The form's pool reply's `settings` for its cell ({flag: {now, min, max, choices, voxel, refusal}}), or null. */
  testReplySettings() {
    const cell = this.testCell(), pool = this.testPool?.label === cell?.label ? this.testPool : null, settings = pool?.reply?.settings;
    return settings && typeof settings === 'object' ? settings : null;
  }
  /** The run's hold as the form's pool reply for its cell says it (`hold`: {value, seq, ...}, F's once the reply knows
   *  F, else the value when the reply was made, seq null), or null before a reply said it. */
  testReplyHold() {
    const cell = this.testCell(), pool = this.testPool?.label === cell?.label ? this.testPool : null, hold = pool?.reply?.hold;
    return hold && typeof hold === 'object' ? hold : null;
  }
  /** What Run still needs (testProblems): the hold against its value at the frame tested once the form's pool reply
   *  knows that frame (its `hold` with F's seq), against the value now when the reply was made before F (as the engine
   *  checks it then), not checked while no reply has said it yet (Run checks again once it came: startTest); a reply
   *  that never says it leaves the run's value now. */
  testCheck() {
    const form = this.testForm ?? {}, cell = this.testCell(), pool = this.testPool?.label === cell?.label ? this.testPool : null;
    const hold = this.testReplyHold();
    const entry = this.testEntry(form.flag);
    return testProblems({ label: cell?.label ?? 0, text: form.text, flag: form.flag, setting: { value: entry.now, min: entry.min, max: entry.max },
      driftOff: driftOff(this.testRun), hold, holdDue: !hold && !!pool && (pool.asking || pool.state === 'asking'),
      choices: entry.kind === 'choice' ? entry.choices?.map(c => c.value) ?? null : null,
      flat: this.testReplySettings()?.['voxel-layers']?.refusal?.code === 'flat' });
  }
  /** The form after one of its fields changed (testAction): another setting (its field empty again: what was typed was
   *  for the other), or the value typed. A change of either makes it another test than an earlier one (no repeat line). */
  formChange(form, action, payload) {
    if (action === 'setting') return TEST_FLAGS.includes(payload) && payload !== form.flag ? { ...form, flag: payload, text: '' } : form;
    return action === 'value' ? { ...form, text: String(payload ?? '') } : form;
  }

  // The form's pool: the engine's preview of the cell (onTestPreview: GET /runs/{id}/qc/test/pool: its refusals, what a
  // test would wait for first and where it was not measured), asked for once per run, cell, end of the run's schedule
  // (its frame) and switch of the mixing check; "Finding…" only once it took TEST_POOL_SHOW_MS. One still being worked
  // out ("working") is asked for again, the engine waiting up to TEST_POOL_WAIT_S (its cheap fields shown meanwhile).
  // While a wait is listed (the run's last iteration, the check learning or scoring, Go deeper's details, another run:
  // most pass with nothing on the page to say so) it is asked about again every TEST_POOL_AGAIN_MS, so that Run's tooltip
  // says what holds now (a preview with a wait does no pool work), as is a refused start's (recheckPool). A preview asked
  // after a refused start replaces its reasons: the form lists the engine's newest word. {key, label, state: 'asking' |
  // 'ready' | 'failed', reply, asking (a request on its way), ask (the number of the last ask), shown ("Finding…" due),
  // askedAt, timers, waiters (Run waiting for it)}.
  askPool() {
    const cell = this.testCell();
    if (!cell || typeof this.options.onTestPreview !== 'function') return;
    const key = keyOf([this.testRunId, cell.label, this.testRun?.schedule_done?.seq ?? null, this.state.enabled === true]);
    if (this.testPool?.key === key) return;
    this.dropPool();
    if (this.testRefused && this.testRefused.label !== cell.label) this.testRefused = null;
    const pool = this.testPool = { key, label: cell.label, state: 'asking', reply: null, asking: false, ask: 0, shown: false, askedAt: -Infinity,
      show: null, again: null, controller: null, waiters: [] };
    pool.show = this.clock.setTimeout(() => {
      pool.show = null;
      if (this.testPool === pool && pool.state === 'asking') { pool.shown = true; this.renderTest(); }
    }, TEST_POOL_SHOW_MS);
    this.fetchPool(pool, 0);
  }
  /** Ask the engine for `pool` (waiting up to `wait` seconds for its answer): the reply kept, the state it says. A ready
   *  reply asked after a refused start for its cell replaces that start's reasons (the engine's newer word), and a
   *  press's line (testNotHere) goes once the engine offers the test again. */
  fetchPool(pool, wait) {
    if (pool.again != null) { this.clock.clearTimeout(pool.again); pool.again = null; }
    let controller = null;
    try { controller = typeof globalThis.AbortController === 'function' ? new globalThis.AbortController() : null; } catch { controller = null; }
    const ask = pool.ask = ++this.poolAsks;
    pool.controller = controller; pool.asking = true; pool.askedAt = this.clock.now();
    const done = () => { pool.asking = false; pool.controller = null; };
    Promise.resolve().then(() => this.options.onTestPreview(pool.label, { wait, signal: controller?.signal })).then(reply => {
      if (this.testPool !== pool || this.disposed) return;
      done();
      if (reply && typeof reply === 'object' && reply.state === 'working') {
        // Still being worked out (also a pool asked about again: the engine let its answer go): its cheap fields
        // meanwhile, Run waiting, and asked again (never sooner than TEST_POOL_SHOW_MS after the last ask: an engine that
        // answers "working" at once is not asked in a loop).
        pool.reply = reply; pool.state = 'asking';
        const delay = Math.max(0, pool.askedAt + TEST_POOL_SHOW_MS - this.clock.now());
        pool.again = this.clock.setTimeout(() => { pool.again = null; if (this.testPool === pool) this.fetchPool(pool, TEST_POOL_WAIT_S); }, delay);
        this.renderTest(); return;
      }
      if (!reply || typeof reply !== 'object' || reply.state !== 'ready') { pool.state = 'failed'; pool.reply = reply && typeof reply === 'object' ? reply : null; }
      else {
        pool.state = 'ready'; pool.reply = reply;
        // A refused start's reasons stand until a preview asked after them (one on its way then may be older); offered,
        // nothing the page knows of stands in the way any more.
        if (this.testRefused?.label === pool.label && ask > this.testRefused.asks) this.testRefused = null;
        if (reply.offer !== false && !this.testRefused && this.testShownProblems.includes(NOT_HERE)) {
          this.testShownProblems = this.testShownProblems.filter(p => p !== NOT_HERE);
        }
        if ((Array.isArray(reply.waits) && reply.waits.length > 0) || (Array.isArray(reply.reasons) && reply.reasons.some(r => r?.wait === true))) {
          pool.again = this.clock.setTimeout(() => { pool.again = null; if (this.testPool === pool) this.fetchPool(pool, 0); }, TEST_POOL_AGAIN_MS);
        }
      }
      this.settlePool(pool); this.recheckPool(pool); this.renderTest();
    }, error => {
      if (this.testPool !== pool || this.disposed) return;
      done();
      if (error?.name === 'AbortError') return;
      if (pool.state !== 'ready') pool.state = 'failed';   // a re-ask that failed keeps the answer it had
      this.settlePool(pool); this.recheckPool(pool); this.renderTest();
    });
  }
  /** A refused start whose reasons pass by themselves (wait: true) or that named a test of this run on its way (busy):
   *  its cell's pool is asked about again TEST_POOL_AGAIN_MS later, once nothing else is on its way or due for it, and
   *  that preview's answer replaces the refusal's reasons (fetchPool). Once a preview was asked after the refusal, the
   *  pool's own re-asks take over. */
  recheckPool(pool = this.testPool) {
    const refused = this.testRefused;
    if (!pool || this.testPool !== pool || !refused?.recheck || refused.label !== pool.label || pool.asking || pool.again != null
      || pool.ask > refused.asks) return;
    pool.again = this.clock.setTimeout(() => { pool.again = null; if (this.testPool === pool) this.fetchPool(pool, 0); }, TEST_POOL_AGAIN_MS);
  }
  /** Run's wait for the pool ends (its answer came, or none will). */
  settlePool(pool, value = pool) {
    if (value === pool && (pool.asking || pool.again != null && pool.state !== 'ready')) return;
    const waiters = pool.waiters; pool.waiters = [];
    for (const waiter of waiters) { if (waiter.timer != null) this.clock.clearTimeout(waiter.timer); waiter.resolve(value); }
  }
  /** The form's pool, once it is not on its way any more (Run waits for it, TEST_POOL_RUN_WAIT_MS at most: the start
   *  works the pool out itself); `fresh`: a refusal (the preview's, or a refused start's for this cell) is asked about
   *  again first (the check switched on, another run paused meanwhile: the engine says what holds now). Null without
   *  one. */
  poolNow(fresh = false) {
    const pool = this.testPool;
    if (!pool) return Promise.resolve(null);
    const refused = (pool.state === 'ready' && pool.reply?.state === 'ready' && pool.reply.offer === false) || this.testRefused?.label === pool.label;
    if (fresh && !pool.asking && refused) this.fetchPool(pool, 0);
    if (!pool.asking && !(pool.again != null && pool.state !== 'ready')) return Promise.resolve(pool);
    return new Promise(resolve => {
      const waiter = { resolve, timer: null };
      waiter.timer = this.clock.setTimeout(() => { waiter.timer = null; pool.waiters = pool.waiters.filter(w => w !== waiter); resolve(null); },
        TEST_POOL_RUN_WAIT_MS);
      pool.waiters.push(waiter);
    });
  }
  /** The form's pool goes (the form closed, another run): nothing more is asked for it, and Run stops waiting for it. */
  dropPool() {
    const pool = this.testPool;
    if (!pool) return;
    this.testPool = null;
    for (const timer of [pool.show, pool.again]) if (timer != null) this.clock.clearTimeout(timer);
    try { pool.controller?.abort?.(); } catch { /* the request ends on its own */ }
    this.settlePool(pool, null);
  }
  /** The reasons the form lists: a refused start's for its cell (the engine's, then), else the pool's refusals. */
  formReasons() {
    const cell = this.testCell(), pool = this.testPool;
    if (this.testRefused && this.testRefused.label === cell?.label) return testReasons(this.testRefused.reasons);
    return pool && pool.label === cell?.label && pool.state === 'ready' && pool.reply?.state === 'ready' ? testReasons(pool.reply.reasons) : [];
  }

  /** The corner's buttons and fields: 'setting' (payload: the flag), 'value' (payload: the text typed), 'run', 'cancel' (a
   *  test on its way), 'new' (New run with this), 'keep' (Keep my setting), 'again' (the same test once more), 'close'
   *  (the form; or ends the test as Cancel, Keep or dismissing a test that stopped would). Never refused silently: a press
   *  while another is on its way says so (explain.mjs busy), Run says what it still needs. Settles once the action went
   *  through or was refused (the model then says why). */
  async testAction(action, payload) {
    if (this.disposed || !TEST_ACTIONS.includes(action)) return;
    if (FORM_ACTIONS.includes(action)) {
      if (!this.testForm) return;
      const form = this.formChange(this.testForm, action, payload);
      if (form !== this.testForm) { this.testForm = form; this.testEarlier = null; }
      this.testShownProblems = []; this.testMessage = null; this.renderTest();
      return;
    }
    if (this.testStarting || this.testBusy) { this.testMessage = EXPLAIN_TEXT.busy; this.renderTest(); return; }
    const test = this.testRec;
    if (action === 'run') return test ? undefined : this.startTest();
    if (action === 'again') return this.againTest();
    if (action === 'close' && !test) {
      this.testForm = null; this.testShownProblems = []; this.testMessage = null; this.testRefused = null; this.testEarlier = null; this.renderTest();
      return;
    }
    if (!test) return;
    const kind = action !== 'close' ? action : TEST_ACTIVE.includes(test.state) ? 'cancel' : test.state === 'done' ? 'keep' : 'close';
    // Only what the test's state offers: Cancel while it is on its way, the answers for its answer, Close once it stopped.
    const offered = { cancel: TEST_ACTIVE.includes(test.state), new: test.state === 'done', keep: test.state === 'done',
      close: ['failed', 'cancelled'].includes(test.state) }[kind];
    if (offered) await this.answerTest(kind);
  }
  /** Run the test (its check passed): it waits for the pool on its way, and asks nothing when the engine said why a test
   *  can't run here (its refusals stay listed); else asked for the form's cell (onTestStart: POST {cell, flag, value}),
   *  also while the engine's waits hold (the test then waits for them: state waiting) and wherever it was not measured;
   *  the corner then follows it. What Run still needs (testProblems) and a refusal go to the form in plain words: the
   *  engine's reasons (a 409 that gives them, or one about a test of this run on its way: busy) listed as the pool's are;
   *  a value the engine refuses names the value (422); a cell not in the newest iteration (404). */
  async startTest() {
    const form = this.testForm, cell = this.testCell(), run = this.testRunId;
    if (!form || !cell) return;
    const problems = this.testCheck();
    if (problems.length) { this.testShownProblems = problems; this.renderTest(); return; }
    this.testStarting = true; this.testShownProblems = []; this.testMessage = null; this.renderTest();
    try {
      const pool = await this.poolNow(true);
      if (run !== this.testRunId || this.disposed) return;
      if (pool?.label === cell.label && pool.state === 'ready' && pool.reply?.state === 'ready' && pool.reply.offer === false) {
        this.testRefused = null; this.testShownProblems = [NOT_HERE];
        return;
      }
      // The hold against its value at the frame tested, now that the pool reply has said it (or will not).
      const late = this.testCheck();
      if (late.length) { this.testShownProblems = late; return; }
      const test = testRecord(await this.options.onTestStart(cell.label, form.flag, Number(form.text)));
      if (run !== this.testRunId || this.disposed) return;
      this.testForm = null; this.testStarting = false; this.testMessage = null; this.testRefused = null;
      if (test && this.testEarlier?.pending) this.testEarlier = { ...this.testEarlier, id: test.id, pending: false };
      // The answer is the test as it stands: the next ask a second after it.
      if (test) { this.testFetchedAt = this.clock.now(); this.applyTest(test); } else { this.renderTest(); this.scheduleTestPoll(true); }
    } catch (error) {
      if (run !== this.testRunId || this.disposed) return;
      const status = error?.status, reasons = Array.isArray(error?.data?.reasons) ? error.data.reasons : [];
      if (status === 409) {
        // The engine's reasons, listed in the form until a preview asked after them says otherwise (fetchPool); one
        // without reasons, or naming a test of this run on its way (busy: pressed in another window): that reason, and
        // the test there is asked for. Reasons that pass by themselves, and busy, are asked about again a little later
        // (recheckPool).
        const busy = !reasons.length || reasons.some(r => r?.code === 'busy');
        this.testRefused = { label: cell.label, reasons: reasons.length ? reasons : [{ code: 'busy' }], asks: this.poolAsks,
          recheck: busy || reasons.some(r => r?.wait === true) };
        this.testShownProblems = [NOT_HERE];
        if (busy) this.scheduleTestPoll(true);
        this.recheckPool();
      } else {
        this.testShownProblems = [status === 422 ? problem(TEST_FIELDS.value, C.testValueLabel, C.testValueRefused)
          : problem(null, '', status === 404 ? C.testCellMissing : C.testStartFailed)];
      }
    } finally {
      if (this.testStarting) { this.testStarting = false; this.renderTest(); }
    }
  }
  /** Cancel (a test on its way), New run with this or Keep my setting (its answer), Close (a test that stopped): the page
   *  puts the run back as it was before the test (resumed only if it was running then, which a run waiting at its end
   *  never was) and the engine forgets the test; the corner then closes and the section offers a new one. New run with
   *  this leaves the run for a new one with the setting (the engine ends the test as the run is left): nothing is ended
   *  here. A refusal says why (the page's plain words, else the deck's) and leaves the test as it is. `busy`: the action
   *  the corner shows on its way; `then`: what follows at once on success, before the corner hears of it (againTest).
   *  Resolves with whether it went through. */
  async answerTest(kind, { busy = kind, then = null } = {}) {
    const test = this.testRec, run = this.testRunId;
    if (!test) return false;
    this.testBusy = busy; this.testMessage = null; this.renderTest();
    try {
      if (kind === 'new') { await this.options.onTestNewRun?.(test); return true; }
      if (kind === 'cancel') await this.options.onTestCancel?.(test);
      else await this.options.onTestAnswer?.(test);
      if (run !== this.testRunId || this.disposed) return false;
      if (test.id != null) this.testDismissed.add(test.id);
      if (this.testRec === test || (test.id != null && this.testRec?.id === test.id)) this.testRec = null;
      then?.();
      return true;
    } catch (error) {
      if (run !== this.testRunId || this.disposed) return false;
      this.testMessage = safeText(error?.plain, kind === 'cancel' ? C.testCancelFailed : C.testAnswerFailed);
      return false;
    } finally {
      if (!this.disposed) { this.testBusy = null; this.renderTest(); }
    }
  }
  /** Run again: the answered test put away as Keep does (the run as it was before it), then the same setting and value
   *  tested again for the same cell (the engine fixes its pool again); the corner stays open in between, and the new
   *  answer says the earlier one's word under it (testRepeat, testRepeatNote: each test is a new chance of a false
   *  "clearly fewer"). A refused start leaves the form open for that cell, saying why. */
  async againTest() {
    const test = this.testRec;
    if (!test || test.state !== 'done' || !Number.isInteger(test.cell) || !TEST_FLAGS.includes(test.flag) || !finite(test.value)) return;
    const earlier = { v: settingValue(test.value), word: earlierWord(test.result), id: null, pending: true };
    const form = { flag: test.flag, text: valueText(test.value), cell: { label: test.cell + 1, name: String(test.cell_name ?? '') } };
    const kept = await this.answerTest('keep', { busy: 'again', then: () => { this.testForm = form; this.testStarting = true; this.testEarlier = earlier; } });
    if (!kept) return;
    this.testStarting = false;
    await this.startTest();
  }
  /** The engine's test (null: none), unless this page has answered it already. What a press could not do goes with the
   *  test (or the state) it was said for. */
  applyTest(test) {
    const before = this.testRec;
    this.testRec = test && !(test.id != null && this.testDismissed.has(test.id)) ? test : null;
    if (this.testRec?.id !== before?.id || this.testRec?.state !== before?.state) this.testMessage = null;
    if (this.testRec) { this.testForm = null; this.testRefused = null; }
    this.renderTest(); this.scheduleTestPoll();
  }
  /** Ask for the test (onTestFetch): each TEST_POLL_MS while it is on its way, and once when a push or a refusal says
   *  there is one this page does not hold (`soon`); never more than once a second, nothing while the page is hidden or
   *  covered. */
  scheduleTestPoll(soon = false) {
    if (soon) this.testWanted = true;
    if (this.disposed || this.testTimer != null || this.testFetching || this.doc.hidden || this.covered || this.testRunId == null
      || typeof this.options.onTestFetch !== 'function' || !(this.testWanted || TEST_ACTIVE.includes(this.testRec?.state))) return;
    const wait = Math.max(0, this.testFetchedAt + TEST_POLL_MS - this.clock.now());
    this.testTimer = this.clock.setTimeout(() => { this.testTimer = null; this.fetchTest(); }, wait);
  }
  /** A full-screen view covers the page (main.js: Compare's dialog), or no longer does: covered, as while the page is
   *  hidden, the test is not asked for and nothing moves (a number counting or a card sliding in lands at once; none
   *  starts); uncovered, the test is asked for again when it is on its way. */
  setCovered(on) {
    this.covered = !!on;
    if (this.covered && this.testTimer != null) { this.clock.clearTimeout(this.testTimer); this.testTimer = null; }
    if (this.covered) { this.stopMotion(true); this.stopEntrance(true); }
    else this.scheduleTestPoll();
  }
  async fetchTest() {
    const run = this.testRunId;
    if (this.disposed || run == null || this.testFetching) return;
    this.testWanted = false; this.testFetching = true; this.testFetchedAt = this.clock.now();
    try {
      const test = testRecord(await this.options.onTestFetch());
      if (run === this.testRunId && !this.disposed) this.applyTest(test);
    } catch (error) {
      // None (404): a test on its way was cancelled elsewhere (said, with Close); an answer shown was answered elsewhere.
      const held = this.testRec;
      if (run === this.testRunId && !this.disposed && error?.status === 404) this.applyTest(held && TEST_ACTIVE.includes(held.state) ? { ...held, state: 'cancelled' } : null);
    } finally {
      this.testFetching = false; this.scheduleTestPoll();
    }
  }

  /** The test as the stage's corner shows it, {open: false} while there is none: {open, phase (TEST_PHASES), run, cell
   *  {label, name}, start, hold, form {settings [{flag, name, now, nowText, tried, note, tip, min, max}], flag, value,
   *  problems [text], notHere (a press while the refusals are listed: testNotHere's line, else ''), field ('value' |
   *  null), line (its Setting line), pool {state: 'asking' | 'reached' | 'short' | 'failed', text, tooltip, cells,
   *  suspects, need} | null, area {rule, text} | null, runs (the runs line), how (the runs line's tooltip: the How line),
   *  reasons (the refusals listed), waits (what a press would wait for first), caveats (where the test was not measured):
   *  each [{code, text}]} | null, test {flag, name, value, valueText, now, nowText, line, rule, area, square, perSide,
   *  runs, runsText, how, cancelTooltip} | null, waiting (testWaiting's, while the test waits) | null, progress
   *  (testProgress, while its runs work) | null, result (testReadout's, with `repeat` {text, note} | null after Run
   *  again and the answers' words `new`, `keep`, `again` {text, tooltip}) | null, message, busy (the action on its way),
   *  busyText}. `square` is the test's square (µm) once the engine gives it (drawn on the stage while the test exists).
   *  Its words are the deck's, so the corner adds only its own labels. */
  testModel() {
    const mode = this.testMode();
    if (mode === 'none' || mode === 'entry') return { open: false };
    const test = this.testRec, form = this.testForm, start = this.testStartKind();
    const phase = test ? test.state : this.testStarting ? 'starting' : 'form';
    const cell = test ? { label: Number.isInteger(test.cell) ? test.cell + 1 : null, name: test.cell_name ?? null } : this.testCell();
    const model = { open: true, phase, run: this.testRunId, cell: { label: cell?.label ?? null, name: cell?.name || null }, start,
      hold: this.testHold(), form: null, test: null, waiting: null, progress: null, result: null, message: null, busy: null, busyText: null };
    if (!test && form) model.form = this.formModel(form, start);
    // The setting under test: the test's own, or the form's while its start is on its way.
    const tested = test ?? (phase === 'starting' && form ? { flag: form.flag, value: typedValue(form.text), current: this.testSetting(form.flag).value } : null);
    if (tested) model.test = this.testedModel(tested, test, start);
    if (test?.state === 'waiting') model.waiting = testWaiting(test);
    else if (test && TEST_ACTIVE.includes(test.state)) model.progress = testProgress(test);
    if (test?.state === 'done') {
      const r = testReadout(test), earlier = this.testEarlier;
      model.result = { ...(r ?? { verdict: null, rule: null, why: null, back: null, fewer: null, cost: null, went: null, alone: null, toofew: null, calibration: '',
        caveats: [] }),
        repeat: earlier && !earlier.pending && earlier.id === test.id
          ? { text: earlier.word ? copy('testRepeat', { v: earlier.v, word: earlier.word }) : copy('testRepeatNone', { v: earlier.v }), note: C.testRepeatNote } : null,
        new: { text: C.testNewRun, tooltip: C.testNewRunTooltip }, keep: { text: C.testKeep, tooltip: C.testKeepTooltip },
        again: { text: C.testRunAgain, tooltip: C.testRunAgainTooltip } };
    }
    // What happened to a test that stopped (the engine's reason when it is plain: "Your run ended, so the test stopped"),
    // and what a press could not do.
    const reason = test && ['failed', 'cancelled'].includes(test.state) ? safeText(test.message, '') : '';
    model.message = [test?.state === 'failed' ? C.testFailed : test?.state === 'cancelled' ? C.testCancelled : '',
      reason ? copy('testReason', { reason }) : '', this.testMessage ?? ''].filter(Boolean).join(' ') || null;
    const busy = this.testStarting ? 'run' : this.testBusy;
    if (busy) { model.busy = busy; model.busyText = busy === 'cancel' ? C.testCancelling : ['run', 'again'].includes(busy) ? C.testStarting : C.testWorking; }
    return model;
  }
  /** The form as the corner shows it (testModel's `form`): the three settings in their order, the one chosen and what
   *  was typed, the problems of the last press, its Setting line (from the value the test compares with: testEntry's
   *  `from`); the pool's line ("Finding…" once due, the pool, or that it could not be counted), the area and the runs
   *  from the pool's reply (the runs per side the start has before it) with how they are made (the How line: the runs
   *  line's tooltip), the refusals (why a test can't run here), and what a press would wait for first and where the
   *  test was not measured (Run's tooltip: the pool reply's waits and caveats, those known so far). */
  formModel(form, start) {
    const settings = TEST_FLAGS.map(flag => this.testEntry(flag)), chosen = settings.find(s => s.flag === form.flag);
    const cell = this.testCell(), pool = this.testPool?.label === cell?.label ? this.testPool : null, reply = pool?.reply ?? null;
    const line = !pool ? null : pool.state === 'failed' ? { state: 'failed', text: C.testPoolFailed, cells: null, suspects: null, need: null }
      : pool.state === 'asking' ? (pool.shown ? { state: 'asking', text: C.testPoolAsking, cells: null, suspects: null, need: null } : null)
        : poolLine(reply?.pool);
    const area = areaText(reply?.area), perSide = Number.isInteger(reply?.runs_per_side) ? reply.runs_per_side : TEST_RUNS_PER_SIDE[start] ?? null;
    // At a run's end counted for the hold (the pool reply's count "sampled": F's), the test reads the averaged count while
    // the panel and the pink dots read the count proseg saves there: the pool line's tooltip and each listed reason's say
    // so (testAveraged); at an averaged F they read the same, and nothing is said. Where the averaged count gives the cell
    // no suspect that can be tested but the count proseg saves does (the caveat saved_suspects: its suspects are taken
    // from there), they say that instead (testAveragedSaved).
    const savedSuspects = Array.isArray(reply?.caveats) && reply.caveats.some(c => c?.code === 'saved_suspects');
    const averaged = reply?.count === 'sampled' && finite(reply.iter) ? copy(savedSuspects ? 'testAveragedSaved' : 'testAveraged', { i: number(reply.iter) }) : '';
    if (averaged && line?.tooltip) line.tooltip = `${line.tooltip} ${averaged}`;
    const shown = this.testShownProblems.filter(p => p !== NOT_HERE), reasons = this.formReasons().map(r => averaged ? { ...r, tip: averaged } : r);
    const fields = shown.some(p => p.field === TEST_FIELDS.value) ? ['value'] : [];
    return { settings, flag: form.flag, value: String(form.text ?? ''), problems: shown.map(p => p.label ? `${p.label}: ${p.message}` : p.message),
      notHere: this.testShownProblems.includes(NOT_HERE) ? testNotHere(reasons) : '',
      field: fields[0] ?? null, line: this.testSettingLine(form.flag, chosen?.from, typedValue(form.text)),
      pool: line, area: area ? { rule: reply.area.rule, text: area } : null,
      runs: runsText(perSide, typedValue(form.text), testEstimate(reply?.estimate_s, form.flag, typedValue(form.text), chosen?.now), form.flag),
      how: `${C.testHow} ${C.testHowHold}`, reasons, waits: testWaits(reply?.waits), caveats: testCaveats(reply?.caveats) };
  }
  /** The setting under test as the corner names it (testModel's `test`): its name (as the form's choice names it), value,
   *  the run's value now and its Setting line; the area it runs (the test's, else the pool's) and its square; the runs
   *  (on each side, in all, in words, how they are made) and Cancel's tooltip naming them (while the test waits: none
   *  made yet). */
  testedModel(tested, test, start) {
    const now = finite(tested.current) ? tested.current : null, value = finite(tested.value) ? tested.value : null;
    const reply = this.testPool?.reply ?? null, area = test ? test.area : reply?.area;
    const perSide = Number.isInteger(test?.runs_per_side) ? test.runs_per_side : Number.isInteger(reply?.runs_per_side) ? reply.runs_per_side
      : TEST_RUNS_PER_SIDE[start] ?? null;
    return { flag: tested.flag ?? null, name: testName(tested.flag), value, valueText: value == null ? C.missing : testValueText(tested.flag, value),
      now, nowText: testNowText(now, tested.flag), line: this.testSettingLine(tested.flag, now, value),
      rule: TEST_AREAS.includes(area?.rule) ? area.rule : null, area: areaText(area), square: test ? this.testSquare : null,
      perSide, runs: perSide ? 2 * perSide : null, runsText: runsText(perSide, value, finite(test?.estimate_s) ? test.estimate_s
        : testEstimate(reply?.estimate_s, tested.flag, value, now), tested.flag),
      how: [C.testHow, C.testHowHold, holdGiven(test?.hold, tested.flag, value)].filter(Boolean).join(' '),
      cancelTooltip: test?.state === 'waiting' ? C.testCancelWaitingTooltip : perSide ? copy('testCancelTooltip', { n: number(2 * perSide) }) : '' };
  }
  /** Hands the corner its model when it changed (onTestCorner), and {open: false} once when the test closes. */
  emitCorner() {
    const model = this.testModel(), key = keyOf(model);
    if (key === this.cornerKey) return;
    const was = this.cornerOpen;
    this.cornerKey = key; this.cornerOpen = !!model.open; this.cornerModel = model;
    if (model.open || was) this.call('onTestCorner', model);
  }
  /** The model last handed to the corner (the page's snapshot). */
  get testCornerModel() { return this.cornerModel; }
  /** The test as shown (the harness's snapshot, main.js qc.test), or null while the section shows nothing: its mode
   *  (entry, form, or the test's state), the test's id, its cell (its Xenium id), setting, value and the run's value;
   *  its progress (iter, of, runs, done); the corner's phase while it is open, whether it is, how the run started; the
   *  area's rule; the reasons by their codes: the refusals (the form's; a waiting test's that ended on them), the waits
   *  (the form's preview's; a waiting test's now) and the caveats (the preview's; the test's, its answer's once done);
   *  the pool ({cells, suspects, need}: the test's, else the form's preview); the engine's verdict and rule; how firmly
   *  the run holds its starting cells (holdOf). */
  get testShown() {
    const mode = this.testBox.hidden ? 'none' : this.testBox.dataset.state;
    if (mode === 'none') return null;
    const test = this.testRec, form = this.testForm, p = test?.progress && typeof test.progress === 'object' ? test.progress : {};
    const reply = !test && this.testPool?.label === this.testCell()?.label ? this.testPool.reply : null;
    const pool = test?.pool && typeof test.pool === 'object' ? { cells: test.pool.cells ?? null, suspects: test.pool.suspects ?? null, need: test.pool.k ?? null }
      : reply?.pool && typeof reply.pool === 'object' ? { cells: reply.pool.cells ?? null, suspects: reply.pool.suspects ?? null, need: reply.pool.need ?? null } : null;
    const area = (test ? test.area : reply?.area)?.rule;
    return { mode, id: test?.id ?? null, cell: test ? test.cell_name ?? null : this.testCell()?.name || null, flag: test?.flag ?? form?.flag ?? null,
      value: test ? test.value ?? null : form ? typedValue(form.text) : null,
      current: test ? test.current ?? null : form ? this.testEntry(form.flag).now ?? null : null,
      iter: p.iter ?? null, of: p.of ?? null, runs: p.runs ?? null, done: p.done ?? null,
      phase: this.cornerModel.open ? this.cornerModel.phase : null, corner: this.cornerOpen, start: this.testStartKind(),
      area: TEST_AREAS.includes(area) ? area : null, reasons: reasonCodes(test ? test.reasons : this.formReasons()),
      waits: reasonCodes(test ? test.waiting : reply?.waits), caveats: reasonCodes(test ? test.result?.caveats ?? test.caveats : reply?.caveats), pool,
      verdict: typeof test?.result?.verdict === 'string' ? test.result.verdict : null, rule: typeof test?.result?.rule === 'string' ? test.result.rule : null,
      hold: this.testHold() };
  }
  /** The square of the test this panel holds (µm, [x0, y0, x1, y1]: the area when it is a square around the cell), or
   *  null (drawn on the stage, dashed, while the test exists; the corner's model carries it too, test.square). */
  get testSquare() {
    const square = this.testRec?.square;
    return Array.isArray(square) && square.length === 4 && square.every(finite) ? [...square] : null;
  }

  displayNumber(value) {
    this.displayed = value;
    setText(this.numberNode, finite(value) ? copy('number', { n: number(Math.round(value)) }) : C.missing);
  }
  draw(segmentProgress = 1, endFrom = null) {
    if (!this.layout || !this.root.open || this.chartBox.hidden || this.disposed) return;
    renderChart(this.canvas, this.layout, { colors: this.colors ?? chartColors(this.canvas), segmentProgress, endFrom });
  }
  redraw() { this.draw(); }
  /** Every word the chart draws on its canvas now (the harness checks them for banned words). */
  chartTexts() {
    const l = this.layout;
    if (!l || this.chartBox.hidden) return [];
    return [l.start?.label, l.formingBand ? C.chartForming : null, C.iteration, ...l.changes.map(c => c.label)].filter(Boolean);
  }
  paintMotion(elapsed) {
    const m = this.motion; if (!m) return;
    const t = Math.min(1, elapsed / NUMBER_MS), eased = 1 - (1 - t) ** 3;
    this.displayNumber(m.number ? m.from + (m.to - m.from) * eased : m.to);
    if (m.line) this.draw(Math.min(1, elapsed / SEGMENT_MS), m.endFrom);
  }
  scheduleMotion() {
    const m = this.motion; if (!m) return;
    // Timers ask for a paint on the stage's own steps (MOTION_STEP_MS), at most MAX_MOTION_FRAMES after the result's own.
    const next = Math.min(m.duration, (m.frame + 1) * MOTION_STEP_MS);
    this.timer = this.clock.setTimeout(() => {
      this.timer = null;
      if (!this.motion || this.disposed) return;
      this.frame = this.clock.requestAnimationFrame(() => {
        this.frame = null; if (!this.motion || this.disposed) return;
        const elapsed = this.clock.now() - m.startTime; m.frame++;
        this.paintMotion(elapsed); this.call('onFrame');
        if (elapsed >= m.duration) this.stopMotion(false);
        else if (m.frame >= MAX_MOTION_FRAMES) this.stopMotion(true);
        else this.scheduleMotion();
      });
    }, Math.max(0, m.startTime + next - this.clock.now()));
  }
  stopMotion(finish = false) {
    if (this.timer != null) this.clock.clearTimeout(this.timer);
    if (this.frame != null) this.clock.cancelAnimationFrame(this.frame);
    this.timer = this.frame = null;
    const motion = this.motion; this.motion = null;
    if (finish && motion) { this.displayNumber(this.target); if (motion.line) this.draw(); }
  }
  /** The chart box's width (CSS px, from the observer): the chart is laid out again at that width. */
  resize(width) {
    if (this.disposed || !(width > 0) || Math.abs(width - this.width) <= 0.5) return;
    this.width = width; this.stopMotion(true); this.chartKey = null; this.setState({ ...this.state });
  }
  open({ scroll = true } = {}) {
    if (this.disposed) return;
    if (!this.root.open) this.root.open = true;
    if (this.host.tagName !== 'DETAILS') try { this.storage?.setItem(this.storageKey, 'true'); } catch { /* optional */ }
    this.draw();
    if (scroll) this.root.scrollIntoView?.({ block: 'nearest', behavior: 'auto' });
  }
  dispose() {
    if (this.disposed) return;
    if (this.cornerOpen) { this.cornerOpen = false; this.cornerModel = { open: false }; this.call('onTestCorner', this.cornerModel); }
    this.stopMotion(false); this.stopEntrance(false); this.disposed = true; this.observer?.disconnect();
    if (this.testTimer != null) this.clock.clearTimeout(this.testTimer);
    this.dropPool();
    for (const [node, type, fn] of this.listeners) node.removeEventListener?.(type, fn);
    this.listeners = []; if (this.root !== this.host) this.root.remove(); else this.root.replaceChildren();
  }
}

