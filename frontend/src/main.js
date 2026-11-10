// First: a notebook session's tab keeps its own storage, set up before anything reads storage (storage-boot.mjs).
import { MARK } from './storage-boot.mjs';
import { CellStats, mostActive, readoutText, readoutTip } from './cellstats.mjs';
import { runButtons } from './run-buttons.mjs';
import { XeniumMasks } from './xenium-masks.js';
import { Analysis } from './analysis.js';
import { clusterColorCSS } from './clusters.js';
import { LayerPanel } from './layer-panel.js';
import { PriorLayers } from './prior-layers.js';
// SoPA companion's page; in its Proseg Live tool, pick a region, start proseg and watch every sampler iteration live.
import { Options, LiveKnobs, KnobGuide } from './options.js';
import { api, LiveSocket } from './api.js';
// The tab's sign-in (index.html loads it first: the same module): its key, the word that it is signed out, and whether
// this tab is the app's own Chrome window.
import { pageKey, SIGNED_OUT, appWindow } from './page-key.mjs';
import { Stage } from './stage.js';
import { TranscriptLoader } from './transcripts.js';
import { Timeline, PHASES, scheduleEnd } from './timeline.js';
import { SampleBrowser } from './sample-browser.js';
import { sampleTools, sampleNotes, renderNotes, openFailure, NOTE_TEXT } from './sample-notes.mjs';
import { ZarrSave } from './zarr-save.js';
import { PROCEED_TEXT, proceedOffer, proceedPrefill } from './proceed.mjs';
import { saveable, saveOutdated } from './zarr-save-state.mjs';
import { clampBox, clampPoint, rectangleFromPoints, simplifyPolygon, regionBBox, regionArea, estimatedTranscripts, modeForBBox } from './region.mjs';
import { LassoDraft, mergedMoves } from './lasso.mjs';
import { priorChoices, choiceIndex } from './prior-options.mjs';
import { TiledLoader } from './tiled-loader.mjs';
import { PaintBatch } from './paint-batch.mjs';
import { Telemetry } from './telemetry.mjs';
import { harness } from './harness-hook.mjs';
import { polygonCount } from './polygons.mjs';
import { QcClient } from './qc-client.mjs';
import { QcPanel, fromView, withEngine } from './qc-panel.js';
import { QcChip } from './qc-chip.mjs';
import { QcOverlay, MixingLegend, LEGEND_PLACE, fetchBins, DeeperArrows } from './qc-overlay.mjs';
import { QC_TEXT, copy, duration, cellLine, cardLine, knobName, triedLabel, settingValue } from './qc-copy.mjs';
import { QcDeeper } from './qc-deeper.js';
import { TestCorner } from './qc-test-corner.mjs';
import { CompareView } from './compare.mjs';
import { COMPARE_TEXT } from './compare-copy.mjs';
import { pointInOutline, flyTarget, areaOutline } from './qc-areas.mjs';
import { modelLabel, STAIN_TEXT, PICKER_TEXT, CELLPOSE_DEFAULTS, StainView, stainMethods, stainProgress, stainSizes, stainStatus, stainValues, validStain, stainProblems, largestPatch, boxInRegion, stainPreviewImage, stainCopy, levelName, resolutionName, resolutionChoices, matchResolution, resolutionReadout, resolutionRegionProblem } from './stain.mjs';
import { problem, explain, guard, busy, clearExplanation } from './explain.mjs';
import { SHELL_TEXT, TOOLS, toolOf, leaveKind, leaveFailure, decidedAtClose, pending, LeavePrompt, leftoversNotice, watchClosing, closeWindow, nextRevision, CLOSE_CHECK_MS } from './shell.mjs';
import { OlderRunsPrompt, olderRows, olderCount } from './older-runs.mjs';
import { SESSION_TEXT, apiSession, sampleHeader, windowTitle, sampleStatus, waitingText, chooseText, resultsRow, addLine } from './session.mjs';
import { STAIN_PATCH_TEXT, StainPatches, patchSizeNote } from './stain-patches.mjs';
import { CellposeSettings, TissueView, fillCellposeSettings, updateNormalizeFields, settingsReport, SETTINGS_TEXT } from './stain-settings.mjs';
import { StainExclusions } from './stain-exclusions.mjs';
// A session that ended unexpectedly: its flag set as the page starts, the popup at boot (globalThis.sopaCompanionCrash).
import './crash-report.mjs';

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat('en-GB');
const fmt1 = new Intl.NumberFormat('en-GB', { maximumFractionDigits: 1 });
const ACTIVE = ['preparing', 'running', 'paused', 'finishing'];
const CACHE_BYTES = 320 * 2 ** 20;

const state = {
  health: null, samples: [], sample: null, mode: 'empty', tool: null,   // tool: 'stain' | 'proseg', null on the two cards
  stainModels: null, stainModelError: '',
  region: null, lastRectangle: null, lastLasso: null, prior: { cells: null, nuclei: null }, runs: [],
  run: null, frames: [], cells: [], image: null,
  current: -1, live: true, replay: null,
  display: { mode: 'cells', layer: -1 },
  hover: 0, selected: 0,
};
// What the panels show now: a new frame (every ~0.5 s in a whole-sample run) rewrites only what changed.
const shown = { knobs: '', results: '', buttons: '', activeRun: null, activeStats: null, activeNames: null, active: '', card: null, tip: null };
// One paint per snapshot (paint-batch.mjs): what a new frame, its statistics or a run event change in the panels is
// written in the page's next batch, and a snapshot on its way holds that batch until it is uploaded, so the stage, the
// HUD, the facts, the status line, the timeline, the most active cells change in one frame. User
// actions keep writing at once.
const paint = new PaintBatch();
// The overlay is to be drawn again by the paint batch (overlaySoon: Go deeper's arrows changed in its apply), unless drawn
// before it.
let overlayWanted = false;
const counts = { snapshots: 0 };   // frames shown (telemetry)
let covered = false;   // Compare's full-screen view is over the run page (cover(): nothing underneath draws or asks)
const later = {
  hud: () => paint.mark('hud', updateHud),
  // "This iteration": the facts first (they show or hide the panel), then the cell card and the most active cells.
  now: () => paint.mark('now', () => { updateFacts(); updateCellCard(); updateActiveCells(); }),
  // The run's panels; the mixing check's test square follows the run's state and its pushed qc_test with them.
  run: () => paint.mark('run', () => { renderRun(); qcPanel.setTestRun?.(state.mode === 'run' ? state.run : null); }),
  tooltip: () => paint.mark('tooltip', () => { if (lastPointer) updateTooltip(lastPointer); }),
  // The mixing check: the result for the frame shown (never a later one), with the snapshot (qc-client.mjs).
  qc: () => paint.mark('qc', () => qcClient.frame()),
};
const HISTORY_MS = 2000;   // a selected cell's whole-run history is refetched at most this often while frames arrive
// A sparkline canvas changes width without any window resize when the side column's scrollbar appears or goes (a
// panel opens, closes or hides): the observer hands over its new size (no layout read) and the sparklines are redrawn.
const sparkSizes=typeof ResizeObserver==='function'?new ResizeObserver(entries=>{
  let stale=false;
  for(const {target,contentRect:{width,height}} of entries){
    if(target.size?.[0]===width&&target.size?.[1]===height)continue;
    target.size=width&&height?[width,height]:null;stale=true;   // collapsed: measured again when it is drawn
  }
  const parts=shown.card;
  if(stale&&parts?.history)for(const c of parts.charts)if(c.canvas.size)sparkline(c.canvas,parts.history.iter,parts.history[c.key],c.color);
}):null;

const options = new Options($('proseg-options'));
const knobs = new LiveKnobs($('live-knobs'), async (id, values) => {
  if (state.run?.id !== id || !['running', 'paused'].includes(state.run.state)) throw new Error('Select an active run to change live knobs.');
  globalThis.sopaCompanionLog?.add('action', `live knobs: ${Object.entries(values || {}).map(([flag, v]) => `${flag} ${v ?? 'automatic'}`).join(', ')}`);
  const run = await api.control(id, 'set', undefined, values);
  if (state.run?.id === id) { state.run = run; renderRun(); }
}, notice);
const pickerGuide = new KnobGuide($('options-guide'), $('options-guide-toggle'), {
  mode: 'picker', value: option => options.value(option),
  apply: action => options.setValue(action.flag, action.value),
});
options.onChange = () => pickerGuide.update();
knobs.guide = new KnobGuide($('live-guide'), $('live-guide-toggle'), {
  mode: 'live', value: option => knobs.value(option), blocked: option => knobs.blocked(option),
  buttons: () => runButtons(state.run, state.frames, { prompt: finishingPrompt }),
  apply: (action, option) => knobs.apply(option, action.value, true),
  canUndo: option => knobs.session?.history.has(option.flag),
  undo: option => knobs.apply(option, undefined, false, true),
});
window.addEventListener('fetch-notice', e => notice(e.detail));
const stage = makeStage();
// A lost GPU context (an allocation that failed) is restored by the stage itself; the view is shown again from there.
stage.onContextLost = () => notice('The graphics context was lost; restoring the view…');
stage.onContextRestored = error => {
  if (error) notice(String(error.message || error));
  else if (stage.imageLimit) notice(`The view was restored; images now load at most ${fmt.format(stage.imageLimit)} px across.`);
  retryRequested();
};
const transcripts = new TranscriptLoader(stage);
const priors = new PriorLayers(stage, swatch);
const layerPanel = new LayerPanel(stage, priors);
const tiled = new TiledLoader(stage, priors, {
  refresh: refreshTiled, status: tiledStatus, error: notice,
  // Whole-sample views fetch only the channels shown; turning one on fetches it (layerPanel.onChannelsChange).
  channels: meta => meta.channels.map((channel, k) => layerPanel.channelOn(channel, k)),
  image: (rect, sources, meta) => {
    const channels=sources.map((source,k)=>layerPanel.channel(meta.channels[k],source,k));
    stage.setImage(rect,channels,true);
    state.image={meta,channels};layerPanel.setChannels(channels);
  },
});
const tissueView = new TissueView(api, () => drawOverlay(), notice);
// Made before the patch panel: its capture listeners on the stage then run first, so in the remove mode a press on the
// image draws a lasso and never reaches a patch box (stain-exclusions.mjs events).
const stainExclusions = new StainExclusions(stage, api, {priors,
  ranges:{size:$('stain-exclude-size-row'),brightness:$('stain-exclude-bright-row')}, changed:()=>{ drawOverlay(); if (state.run) later.run(); },notice,
  classes:value=>{ priors.classes=value; stage.dirty=true; },
  refresh:async (id,data)=>{
    if(state.run?.id!==id)return;
    if(data) {
      state.run={...state.run,exclusions:{...data,count:data.areas?.length ?? data.count ?? 0},
        ...(Number.isFinite(data.kept) ? {summary:{...state.run.summary,cells:data.kept}} : {})};
      stainView.update(state.run);
    }
    const run=await api.run(id);
    if(state.run?.id===id)onMessage({ev:'run',run});
  }});
// The stain view hands each set of outlines it draws to the removed-cells panel (made just above): its range choosers
// read those cells' sizes and brightness, fill their histograms and hide the outlines outside a range live.
const stainView = new StainView(stage, priors, tiled, notice, vectors => stainExclusions.setVectors(vectors));
const stainPatches = new StainPatches(stage, api, {notice});
// One tool on the image at a time: turning the remove mode on ends Add a box, and Add a box ends the mode.
$('stain-exclude').addEventListener('click', () => {
  if (stainExclusions.drawing && stainPatches.adding) { stainPatches.adding = false; stainPatches.render(); }
});
$('stain-patches').addEventListener('click', e => {
  if (e.target?.id === 'stain-patch-add' && stainPatches.adding && stainExclusions.drawing) stainExclusions.stop();
});
const useCellposeSettings = async result => {
  const ticket=opening;
  await loadStainModels();
  if(ticket!==opening || state.mode!=='pick' || $('stain-method').value!=='cellpose')return;
  // The patches' resolution matched to this image's own levels (stain-settings.mjs), never their level number as is.
  const lines=fillCellposeSettings($,result,state.sample?.images?.find(image => image.key === $('stain-image').value) ?? null);
  renderResolution();
  settingsReport($('stain-settings-report'),result,lines);updatePicker();
};
const pickerSettings = new CellposeSettings(api,$('stain-use-settings'),$('stain-settings-reason'),useCellposeSettings);
stainPatches.settings.title=SETTINGS_TEXT.newRun;
const patchSettings = new CellposeSettings(api,stainPatches.settings,stainPatches.settingsReason,async result=>{
  const run=state.run;if(!run)return;
  await leave(async () => {
    $('run-kind').value=run.kind;$('stain-method').value='cellpose';
    await enterPick(run.sample);
    if(state.mode!=='pick' || state.sample?.id!==run.sample)return;
    $('stain-image').value=run.stain.image;fillStainChannels();
    $('stain-level').value=String(run.stain.level ?? run.stain.resolution?.level ?? 0);
    if(run.stain.nucleus)$('stain-nucleus').value=run.stain.nucleus;
    if(run.stain.cytoplasm)$('stain-cytoplasm').value=run.stain.cytoplasm;
    if(run.kind==='patches')$('patch-start-from').value='model';
    setRegion(run.region);updateRunKind();await loadStainModels();
    if(state.mode==='pick' && state.sample?.id===run.sample){useCellposeSettings(result);loadStainPreview();}
  }, 'settings');
});
layerPanel.onChannelsChange = () => { if (state.run?.mode === 'tiled' || ['stain','patches'].includes(state.run?.kind)) tiled.loadImage(); };
const masks = new XeniumMasks(stage, () => layerPanel.render());
layerPanel.masks=masks;layerPanel.render();
// The sample's segmentation layers (Layers > Load Other Mask) also in the picker's card, where Layers
// is not shown: their rows and Add a mask… (no fields), drawn on the overview as on a run.
layerPanel.mountSegmentations($('picker-segmentations'));
// Go deeper's arrows wait for the centres; its basis line (proseg's own count of the cell) for the frame's statistics,
// which carry proseg's count of each cell when the frame is counted (frameCounted), unless it was counted for a pause,
// the hold, a finish or the end (frameSampled: the count proseg saves there, the frame's own draw, is read then).
const cellStats=new CellStats(()=>{later.now();later.tooltip();deeper.statsChanged();if(deeperArrows.missing)deeperGeometry();if(qcInspecting)qcInspectDue();},
  {counted:seq=>frameCounted(seq),sampled:seq=>frameSampled(seq)});
const analysis = new Analysis(stage, layerPanel, () => ({seq:state.current, live:state.live && state.run?.state==='running', frames:state.frames, cells:state.cells}), async id => {
  state.selected=id===null?0:id+1;stage.set({selected:state.selected});updateCellCard();
  if(id===null)return;
  const run=state.run?.id,seq=state.current;
  await analysis.loadStats(seq);
  if(state.run?.id!==run||state.current!==seq||state.selected!==id+1)return;
  let x=analysis.stats?.seq===seq?analysis.stats.x[id]:NaN,y=analysis.stats?.seq===seq?analysis.stats.y[id]:NaN;
  if(Number.isFinite(x)&&Number.isFinite(y)){
    stage.tween=null;stage.camera.x=x;stage.camera.y=y;
    if(state.run.mode==='tiled') {
      tiled.overview=false;
      const side=Math.min(600,(state.run.info?.window_max || 1600)*(state.frames[seq]?.voxel || 1)*.7);
      stage.camera.zoom=Math.max(stage.camera.zoom,Math.max(stage.canvas.width,stage.canvas.height)/side);
      if(stage.orbit.enabled)setView(false,false);
    }
    stage.dirty=true;stage.emit();
  }
});
analysis.statsStore=cellStats;
// A new UMAP view gets Go deeper's rings too, and the test corner the new clusters' colours.
analysis.onReadoutChange=()=>{later.now();later.tooltip();deeperRings();testCorner.recolor();};
const overlay = $('overlay');
// The timeline's readout names a knob change as Live knobs does (its label), never by the raw proseg flag.
const timeline = new Timeline($('timeline'), { onScrub: (seq) => { stopReplay(); setLive(false); show(seq); }, batch: paint, knobLabel: flag => knobLabel(flag) });
// The mixing check: its panel (the one switch), the stage chip, the per-bin layer on
// the overlay and the Mixing legend (qc-overlay.mjs), all from the result qc-client.mjs picks for the frame shown. No
// flagged area is drawn or listed (they added little on screen): a card's
// "Look at its cells" and R1's "Show the outlines of ... here" move the view to its area (flyToArea), nothing drawn.
const qcOverlay = new QcOverlay(stage, { redraw: () => drawOverlay() });
const mixingLegend = new MixingLegend($('stage').parentElement);
const qcPanel = new QcPanel($('qc-panel'), {
  storage: null,   // a details.panel: setupControls() remembers it open or closed, like the other panels
  onToggle: on => { globalThis.sopaCompanionLog?.add('action', `mixing check switched ${on ? 'on' : 'off'}`); qcClient.setEnabled(on); },
  onRetry: () => qcClient.retry(),
  onInspect: (suggestion, area) => qcInspect(suggestion, area),   // "Look at its cells" (R1, R6, R7: the cause is not known)
  onShowCells: (on, context) => context?.start ? showStartOutlines(context.area) : setMixingColours(on),
  onReload: () => qcClient.refresh(), onCalmMotion: on => setCalmMotion(on),
  onFrame: () => harness?.count('qcAnimFrames'),
  // The suggestion card, the try loop and the new run with the suggested setting.
  suggestions: true, knobLabel: flag => knobLabel(flag),
  onTry: suggestion => qcTry(suggestion), onKeep: item => qcKeep(item), onUndo: item => qcUndo(item),
  onNewRun: suggestion => newRunWithSetting(suggestion), onCard: () => harness?.count('qcSuggestionsShown'),
  bindDeeper: (button, label) => deeper.bindButton(button, label),   // the selected cell's "Go deeper ›" (updateCellRow)
  // "Test a setting here…" under the row (and in Go deeper: qcTestOpen); the setting as Live knobs has it, and the
  // test's calls (qcTestPreview, qcTestStart and the rest, below the suggestions' actions).
  testSetting: flag => qcTestSetting(flag), onTestPreview: (label, options) => qcTestPreview(label, options),
  onTestStart: (label, flag, value) => qcTestStart(label, flag, value),
  onTestFetch: () => qcTestFetch(), onTestCancel: test => qcTestEnd(test), onTestAnswer: test => qcTestEnd(test),
  onTestNewRun: test => qcTestNewRun(test), onTestShow: () => openQcPanel(),
  // The test is shown in the stage's corner (its square on the stage while it exists).
  onTestCorner: model => { testCorner.show(model); qcTestSquare(model); },
});
const qcChip = new QcChip($('qc-chip'), { onClick: () => openQcPanel() });
const qcClient = new QcClient({ api, batch: paint, context: qcContext, render: renderQc, notice, harness,
  // What new runs start with (pickerQc); switched on, Compare's note "switch it on" goes (a press opens it now).
  onSwitch: on => { qcNewRuns = on; if (on) clearExplanation($('compare-button')); } });
// Go deeper: one cell's RNA explained in a movable window inside the stage
// (qc-deeper.js; "Go deeper ›" in the mixing check's row for the selected cell, or D), its arrows on the overlay
// (qc-overlay.mjs DeeperArrows), its outlines on the stage (stage.setFocus) and its rings in the UMAP
// (UmapView.ringCells). The window applies its results in the check's paint batch (renderQc), at most one a second while
// live.
const deeperArrows = new DeeperArrows(stage, { redraw: () => drawOverlay(), onNote: kind => deeper.setStageNote(kind) });
const deeper = new QcDeeper($('stage').parentElement, { api, batch: paint, harness, context: deeperContext,
  name: label => cellName(label), group: label => deeperGroup(label), swatch: label => swatch(label),
  startCode: label => qcClient.view?.enabled ? qcClient.startValue(label) : null,
  screenOf: label => { const c = deeperCentre(label); return c ? stage.toScreen(c[0], c[1]) : null; },
  onFocus: focus => deeperFocus(focus), onSelect: label => deeperSelect(label), onFind: label => deeperFind(label),
  onSwitchOn: () => { globalThis.sopaCompanionLog?.add('action', 'mixing check switched on (Go deeper)'); qcClient.setEnabled(true); },
  onRetryLearning: () => qcClient.retry(), onChange: () => updateCellCard(), onBox: box => deeperBox(box),
  onSpill: spill => deeperSpill(spill), onFly: (rect, animate, view) => deeperFly(rect, animate, view),
  onGlow: glow => stage.setGlow?.(glow ?? null),   // its suspects glowing (glow.mjs)
  onStartOutline: start => deeperStartOutline(start),   // its cell's starting outline while Show is on
  onTest: (label, name) => qcTestOpen(label, name),     // "Test a setting here…" beside Show on the image
  log: text => globalThis.sopaCompanionLog?.add('deeper', text) });
// The test corner (qc-test-corner.mjs): the panel's test of a setting, drawn in the stage's bottom
// left. The panel keeps the test's state machine and hands the corner a model on every change (onTestCorner); the
// corner's presses go back to it, its views come from GET /runs/{id}/qc/test/view, and the Mixing legend moves out of
// its way (legendSpot: beside it at the stage's bottom, else hidden while it is open); its cells are drawn in the
// stage's colours (stageColour).
const testCorner = new TestCorner($('stage').parentElement, { hook: harness,
  fetchView: id => api.qcTestView(id),
  onAction: (action, payload) => qcPanel.testAction(action, payload),
  onBox: box => {
    if (box) qcBoxes.set(testCorner.root, { left: box.left, bottom: box.bottom, w: box.w, h: box.h }); else qcBoxes.delete(testCorner.root);
    legendCorner = box ? { box: { ...box }, node: testCorner.root } : null;
    placeLegend();   // the legend placed for it
    later.hud();     // and the stage's note, at the same corner, gives way to it while it is open (updateHud)
  },
  colorOf: label => stageColour(label),
  reducedMotion: () => stage.reducedMotion(), calmMotion: () => qcCalm,
  stage: () => ({ width: stage.cssWidth, height: stage.cssHeight }) });
// Compare (compare.mjs): the run's cells beside the sample's other segmentations, their UMAPs and
// their mixing side by side, in a full-screen modal <dialog> over the run page (state.mode stays 'run'; the page's keys
// rest while a dialog is open). While it is open the page underneath draws and asks for nothing (`covered`, cover()),
// as while the page is hidden, and it catches up when the view closes (Back, Esc), or the view goes with the run
// (openRun, enterPick).
const compareView = new CompareView($('compare-host'), { hook: harness,
  fetchStatus: id => api.compareStatus(id), fetchPoints: (id, index, job, layout, what) => api.comparePoints(id, index, job, layout, undefined, what),
  start: (id, columns, seq) => api.compareStart(id, columns, seq), listSegmentations: id => api.segmentations(id),
  learnDetails: id => api.learnSampleDetail(id),
  reducedMotion: () => stage.reducedMotion(), calmMotion: () => qcCalm,
  onClose: () => cover(false) });
/** The run page covered by Compare (`on`), or shown again. Covered, nothing underneath draws, moves or asks: the stage
 *  and its glow, the timeline, the test corner, the mixing panel (its test, its counting number and sliding cards), the
 *  UMAP pane (a new result is held: no flight) and Go deeper's window (no motion, no change), live follow and the panels
 *  a new frame changes, the mixing check's results and the runs poll, as while the page is hidden. Shown again
 *  (`catchUp`: unless another run or the picker opens instead), the mixing check's result as it stands, the newest
 *  clusters and Go deeper's answer for the frame on screen, the panels and the newest frame when live. */
function cover(on, catchUp = true) {
  on = !!on;
  if (on === covered) return;
  covered = on;
  stage.setCovered(on); timeline.setCovered(on); testCorner.setCovered(on); qcPanel.setCovered(on); paint.setCovered(on);
  analysis.setCovered(on, catchUp); deeper.setCovered(on, catchUp);
  if (on) return;
  const missed = qcCovered, tiledWas = tiledMissed; qcCovered = false; tiledMissed = false;
  if (!catchUp || state.mode !== 'run' || !state.run) return;
  if (missed) qcClient.frame();
  later.run(); later.hud();
  if (state.live) { if (!document.hidden) follow(); }
  else if (requested >= 0 && requested !== state.current) retryRequested();   // a frame asked for meanwhile
  // A whole-sample run's window that changed underneath (a paused run's run event, a view refresh): shown now.
  if (state.run.mode === 'tiled' && state.current >= 0 && requested === state.current &&
      (tiledWas || tiled.windowChanged?.(state.frames[state.current]))) show(state.current, { pulse: false });
}
/** Compare over the run page (the header's button, once the run's mixing check is on): the frame on screen goes with
 *  it, so the run's column is that frame. */
function openCompare() {
  const run = state.run, frame = state.frames[state.current];
  globalThis.sopaCompanionLog?.add('action', 'compare opened');
  cover(compareView.open({ run: run.id, seq: Number.isInteger(frame?.seq) ? frame.seq : null,
    iteration: frame?.iter ?? null, sampleId: run.sample, box: run.box, kept: run.constants?.kept ?? null }));
}
layerPanel.onColorBy = value => colourBy(value);
const sampleBrowser = new SampleBrowser(sample => {
  state.samples = [{ id: sample.id, name: sample.name, path: sample.path, kind: sample.kind }, ...state.samples.filter(s => s.id !== sample.id)];
  renderRecent(); enterPick(sample);
});
// Every navigation waits for the same engine-owned leave operation. In a notebook's session Close asks with the same
// prompt, and its answer ends the session (answerClose).
let leaving = null;
const closePopup = new LeavePrompt({ log: text => globalThis.sopaCompanionLog?.add('action', text),
  answer: (answer, run, revision) => state.session && closePopup.reason === 'close' ? answerClose(answer, run, revision) : answerLeave(answer, revision) });
// Runs older versions made and never saved: listed at the first start, before anything opens.
const olderPrompt = new OlderRunsPrompt({ fetchRows: () => api.olderRuns(), answer: (id, answer) => api.answerOlder(id, answer),
  answerAll: answer => api.answerOlder(null, answer), notice: text => notice(text), reducedMotion: () => stage.reducedMotion(),
  label: flag => knobLabel(flag), defaults: flag => options.schema.find(o => o.flag === flag)?.default });
const closing = { hold: !harness, popup: closePopup };
// The window's ✕ with a run pending: the browser holds the window and the prompt asks; going anyway sends the answer
// given, else discard, for the run this window shows (never one it did not show: the first screen, a picker, behind the
// older runs' prompt) and with a revision (a newer answer, or a run started since, wins). A notebook's session that is
// closing or ended has its answer already (Close's, with end): nothing more is sent.
watchClosing(window, closePopup, { sampling: () => state.ended || !state.run ? [] : pending([state.run]),
  hold: () => closing.hold && !state.ended, ask: () => { if (state.run && !olderPrompt.open) pressClose(); },
  beacon: (answer, runs) => api.closing(answer, { beacon: true, runs: runs.map(run => run.id), revision: nextRevision() }) });
const zarrSave = new ZarrSave(run => onMessage({ ev: 'run', run }), notice, run => {
  priors.segments.sampleSaved(run.sample); state.revealProceed = run.id; later.run();
}, () => settleExclusions(), run => saveDeclined(run));

async function settleExclusions() {
  if (state.run?.kind !== 'stain') return;
  if (stainExclusions.whenSettled) {
    let timer;
    try {
      const settled = await Promise.race([stainExclusions.whenSettled(), new Promise((_, reject) => {
        timer = setTimeout(() => reject(stillWaiting()), 5000);
      })]);
      if (settled === false) throw stillWaiting();
    } finally { clearTimeout(timer); }
  }
  else if (stainExclusions.busy || stainExclusions.queued || stainExclusions.waiting?.length) throw stillWaiting();
}
/** Cell changes still on their way after five seconds (`waiting`: the one settle failure that holds a leave). */
const stillWaiting = () => Object.assign(new Error(PROCEED_TEXT.waiting), { waiting: true });
function leave(then, reason = 'leave') {
  if (leaving) return leaving.promise;
  if (!state.run) return Promise.resolve(then(null));
  const run = state.run;
  let resolve; const promise = new Promise(done => { resolve = done; });
  const work = leaving = { id: run.id, last: run, then, reason, resolve, promise, requested: false };
  (async () => {
    // Cell changes on their way land first, so the prompt saves or discards them too; one still going after five seconds
    // keeps the window on the run (the notice says to wait). A read or change that failed never does: the panel said so
    // when it happened, and what the engine holds is what the prompt saves or discards.
    try { await settleExclusions(); } catch (error) { if (error.waiting) throw error; }
    if (leaving !== work) return;
    if (state.run?.id !== run.id) { leaving = null; work.resolve(false); return; }
    work.last = state.run;
    if (reason === 'proceed' && !proceedOffer(state.run, { api: !!state.session, pending: exclusionsPending() }).ready) throw new Error(PROCEED_TEXT.outdated);
    const kind = leaveKind(state.run);
    work.notice = kind.why === 'not_started' && state.run.kind !== 'stain' ? SHELL_TEXT.notStarted :
      state.run.kind === 'patches' && stainPatches.data?.patches?.length && !stainPatches.data?.exported ? SHELL_TEXT.patchesDiscarded : '';
    if (state.run.leaving) { work.requested = work.taken = true; work.before = failureKey(state.run.leave_error); closePopup.show(state.run, { reason }); }
    else if (['ask', 'saving'].includes(kind.kind)) { work.waiting = kind.kind === 'saving'; closePopup.show(state.run, { reason }); }
    else await answerLeave(kind.kind === 'saved' ? 'add' : 'discard');
  })().catch(error => { if (leaving === work) { leaving = null; work.resolve(false); notice(error.message); } });
  return promise;
}
async function answerLeave(answer, revision = nextRevision()) {
  const work = leaving;
  if (!work) return;
  if (answer === 'stay') {
    if (work.requested || work.last.leaving) {
      const run = await api.leave(work.id, 'stay', revision);
      if (leaving !== work) return;
      if (run && state.run?.id === work.id) onMessage({ev:'run',run});
    }
    leaving = null; work.resolve(false); return;
  }
  // The failure the run carries now is an earlier leave's (`before`): a snapshot read before this leave was taken that
  // arrives after its answer still carries it, and never ends this one (updateLeaving).
  work.requested = true; work.waiting = false; work.before = failureKey((state.run?.id === work.id ? state.run : work.last)?.leave_error);
  try {
    const run = await api.leave(work.id, answer, revision);
    if (leaving === work && run && state.run?.id === work.id) onMessage({ ev:'run', run });
  } catch (error) {
    if (leaving !== work) return;
    work.requested = work.taken = false;
    if (error.status === 404) runGone(work.id); else throw error;
  }
}
/** Which leave failure a run carries: its own id (an older engine's record, by its time). */
const failureKey = error => error?.id ?? error?.at;
function updateLeaving(run) {
  const work = leaving;
  if (work?.id !== run.id) return;
  work.last = run;
  // `taken`: the engine has the leave under way (its answer said so). The run back from then on with none under way and
  // the leave's failure (`leave_error`: Runs._leave keeps the run) means it is over: the prompt, if open, says why and
  // asks again; else the leave ends here, so every way out works again, and the notice says why.
  if (work.requested && run.leaving) work.taken = true;
  else if (work.taken && !run.leaving && run.leave_error && !(work.before && failureKey(run.leave_error) === work.before)) {
    work.taken = false;
    if (!closePopup.open) { leaving = null; work.resolve(false); notice(leaveFailure(run, { api: !!state.session })); return; }
  }
  if (closePopup.open) closePopup.update(run);
  if (work.waiting && leaveKind(run).kind === 'saved') answerLeave('add').catch(error => {
    closePopup.error = error.message; closePopup.update(run);
  });
}
/** Not now (or Esc) in the Save dialog of a run whose Finish & save counts as answered: the engine is told with
 *  `leave stay`, so closing the app no longer saves it whatever the answer (it asks, and nobody answering discards). A
 *  leave under way is decided anyway and is never withdrawn from here. */
function saveDeclined(run) {
  if (!decidedAtClose(run) || run.leaving) return;
  globalThis.sopaCompanionLog?.add('action', 'Save dialog after Finish & save: Not now');
  api.leave(run.id, 'stay', nextRevision()).then(answer => { if (answer?.id && state.run?.id === answer.id) onMessage({ ev: 'run', run: answer }); },
    () => { /* a run gone meanwhile, or an engine closing: nothing to take back */ });
}
function runGone(id) {
  state.runs = state.runs.filter(run => run.id !== id);
  if (state.run?.id !== id && leaving?.id !== id) return;
  const work = leaving?.id === id ? leaving : null, run = work?.last || state.run;
  if (work) leaving = null;
  closePopup.reset(); zarrSave.cancelFinish?.(id); zarrSave.pending.delete(id);
  if (zarrSave.run?.id === id) { zarrSave.dialog?.close(); zarrSave.run = null; }
  if (work) {
    state.run = null;
    Promise.resolve().then(() => work.then(run)).then(value => { if (work.notice) notice(work.notice); work.resolve(value ?? true); }, error => { notice(error.message); work.resolve(false); });
  } else {
    enterPick(run.sample); notice(SHELL_TEXT.closedElsewhere);
  }
}
const socket = new LiveSocket({
  connection: (ok) => { if (ok) { retryRequested(); if(['stain','patches'].includes(state.run?.kind)){stainPatches.load();stainView.failed=null;stainView.refresh();} else if(state.run){analysis.refresh();qcClient.refresh();deeper.refresh();} } $('connection').classList.toggle('ok', ok); $('connection').title = ok ? 'Connected to the engine' : 'Reconnecting…';
    if (!ok && state.session) sessionGone(); },
  message: onMessage,
});
// The engine no longer takes this tab's key (page-key.mjs: a request or the socket refused it): the page says so, in
// place of everything else. A page that goes says goodbye on its socket as well (the engine closes once the last page
// that was open said goodbye), and one the browser brings back connects again.
window.addEventListener(SIGNED_OUT, event => signedOut(event.detail));
window.addEventListener('pagehide', () => socket.goodbye?.());
window.addEventListener('pageshow', event => { if (event?.persisted && !state.closed && !state.signedOut && state.ended !== 'ended') socket.back?.(); });
// ---- a notebook's session (session.mjs) ---------------------------------------------------------------------------
/** The session as the engine says it (health.session, then {ev: "session"} pushes); null on the desktop. */
function setSession(session) {
  state.session = session;
  if (!session) return;
  // A card pressed while the sample loaded (openSessionTool): opened below once it is ready, dropped if it failed.
  const waited = ['ready', 'failed'].includes(session.sample.state) ? state.sessionWait : null;
  if (waited) state.sessionWait = null;
  closePopup.setMode({ api: true }); zarrSave.setSession?.(session);
  document.title = windowTitle(session);
  setSample(state.sample);          // the header names the bound sample on every screen
  renderTool();
  if (session.state === 'ending') { sessionEnded(true); return; }
  // The card pressed meanwhile, else tool=: that tool's picker on the bound sample, once it is ready and nothing else is
  // open (boot does tool= first). The press is the user's newer choice, so tool= never opens after it.
  const tool = session.sample.state !== 'ready' ? null : waited || (!state.sessionTool && session.tool);
  if (state.booted && tool && state.mode === 'empty' && !state.run && !state.ended) {
    state.sessionTool = true; openSessionTool(tool);
  }
  if (state.run) { shown.results = ''; later.run(); }   // "Waiting for your notebook" comes and goes with it
}
/** The socket closed in a session: after its ending, the session has ended; else, when the engine no longer answers.
 *  A page signed out says that instead (signedOut). */
function sessionGone() {
  if (state.ended === 'ended' || state.signedOut) return;
  if (state.ended === 'closing' || state.session?.state === 'ending') { sessionEnded(false); return; }
  api.health().catch(() => sessionEnded(false));
}
/** The session is closing (`closing`) or has ended: its screen takes the page's place, with no Reload (a new session
 *  starts from the notebook); nothing asks the engine any more once it has ended. */
function sessionEnded(closing = false) {
  if (state.ended === 'ended' || closing && state.ended === 'closing') return;
  const first = !state.ended;
  state.ended = closing ? 'closing' : 'ended';
  if (!closing) socket.stop?.();
  if (first) {
    closePopup.reset(); if (zarrSave.dialog?.open) zarrSave.dialog.close(); if (olderPrompt.open) olderPrompt.finish();
    leaving = null; state.sessionWait = null; enterPick(null);
  }
  state.engineDown = true; state.tool = null; state.mode = 'empty';
  setText($('engine-down-title'), closing ? SESSION_TEXT.closing : SESSION_TEXT.endedTitle);
  setText($('engine-down-why'), closing ? '' : SESSION_TEXT.ended);
  setProp($('engine-reload'), 'hidden', true);
  layout();
}

// ---- frames: fetch, decode, cache ---------------------------------------------------------------------------
const cache = new Map();      // seq -> {consensus, age, layers: [Uint32Array]}
const inflight = new Map();   // seq -> Promise
let cacheBytes = 0;
let generation = 0, requested = -1, retryTimer;
let pickerLoading = false;
function resetRequests() { generation++; showing++; followBusy = false; requested = -1; clearTimeout(retryTimer); if(state.run?.mode==='tiled')tiled.reset(); }
function retryRequested() {
  if (state.run && requested >= 0 && !covered) show(requested, { pulse: false });   // under Compare: when it closes
}


function decode(header, buffer) {
  const plane = header.nx * header.ny, all = new Uint32Array(buffer);
  if (all.length !== plane * (2 + header.nz)) throw new Error(`Frame ${header.seq} is truncated`);
  const layers = [];
  for (let k = 0; k < header.nz && all.length >= (2 + k + 1) * plane; k++) layers.push(all.subarray((2 + k) * plane, (3 + k) * plane));
  return { consensus: all.subarray(0, plane), age: all.subarray(plane, 2 * plane), layers, bytes: buffer.byteLength };
}

function getFrame(seq) {
  if (cache.has(seq)) {
    const value = cache.get(seq);
    cache.delete(seq); cache.set(seq, value);    // most recently used
    return Promise.resolve(value);
  }
  if (inflight.has(seq)) return inflight.get(seq);
  const gen = generation, runId = state.run.id, header = state.frames[seq];
  const promise = api.frame(runId, seq).then((buffer) => {
    const value = decode(header, buffer);
    if (gen === generation && state.run?.id === runId) {
      cache.set(seq, value);
      cacheBytes += value.bytes;
      for (const [key, old] of cache) {
        if (cacheBytes <= CACHE_BYTES || key === seq) break;
        cache.delete(key); cacheBytes -= old.bytes;
      }
    }
    return value;
  }).finally(() => { if (inflight.get(seq) === promise) inflight.delete(seq); });
  inflight.set(seq, promise);
  return promise;
}

function clearFrames() {
  cache.clear(); inflight.clear(); cacheBytes = 0;
}

let showing = 0;
const STATS_WAIT_MS = 150;   // a frame waits at most this long for its statistics, so both appear in one paint
/** `promise` settled, or `ms` passed (null for no promise). */
function settled(promise, ms) {
  if (typeof promise?.then !== 'function') return null;
  let timer;
  return Promise.race([promise.catch(() => {}), new Promise(resolve => { timer = setTimeout(resolve, ms); })]).finally(() => clearTimeout(timer));
}
async function show(seq, { pulse = true } = {}) {
  if (!state.run || seq < 0 || seq >= state.frames.length) return;
  requested = seq;
  clearTimeout(retryTimer);
  const ticket = ++showing, gen = generation;
  // Everything this snapshot changes is painted with it: the panels' batch waits for the upload, and the frame's
  // statistics (most active cells, cell card, tooltip) download alongside the frame instead of after it. They become
  // current only when the frame is shown: until then the frame on screen keeps its own numbers.
  const hold = paint.hold();
  let committed = false;
  try {
    const stats = cellStats.prefetch(seq);
    let data;
    try {
      if(state.run.mode==='tiled') {tiled.run=state.run;data=await tiled.frame(state.frames[seq]);}
      else data = await getFrame(seq);
      if (ticket !== showing || gen !== generation) return false;
      if(!data)return false;
      await settled(stats, STATS_WAIT_MS);
      if (ticket !== showing || gen !== generation) return false;
      // Compare opened while it downloaded: nothing is uploaded underneath; cover(false) asks for the frame again.
      if (covered) return false;
      const h = state.run.mode==='tiled'?data:state.frames[seq], layer = state.run.mode==='tiled' && !data.layers.length?-1:state.display.layer;
      stage.setFrame({ nx: h.nx, ny: h.ny, x0: h.x0, y0: h.y0, voxel: h.voxel, iter: h.iter,
        consensus: data.consensus, layers: data.layers, layer, labels: layer >= 0 && data.layers[layer] ? data.layers[layer] : data.consensus, age: data.age }, pulse);
    } catch (error) {
      if (ticket === showing && gen === generation && error.name!=='AbortError') {
        notice(String(error.message || error));
        if (!stage.frame) { state.current = -1; timeline.setCurrent(-1); later.hud(); later.now(); }
        if (!error.permanent) retryTimer = setTimeout(() => { if (gen === generation && ticket === showing) retryRequested(); }, 1000);
      }
      return false;
    }
    state.current = seq; counts.snapshots++;
    cellStats.load(seq);   // prefetched: current now, in this paint (or when a slow download arrives)
    if(state.run.mode==='tiled')tiled.committed(state.frames[seq],data);
    analysis.frame();
    timeline.setCurrent(seq, !state.live);   // a live playhead redraws at the timeline's own pace
    later.hud(); later.now(); later.qc();
    if(state.run.mode!=='tiled')for (let k = 1; k <= 3; k++) if (seq + k < state.frames.length && !state.live) getFrame(seq + k).catch(() => {});
    return committed = true;
  } finally {
    // The stage draws the upload on the next animation frame: the panels are written now, in this task, so they are
    // painted in that same frame even if follow() starts the next snapshot (and holds the batch) right away.
    paint.release(hold, committed);
  }
}

// Live mode follows the newest frame, but never queues: if frames outpace the network, intermediate ones are skipped.
// A hidden page fetches and uploads nothing, also when it is hidden while following: the visibilitychange handler
// catches up with the newest frame when it is shown again. Nor does a page Compare covers (cover(): when it closes).
let followBusy = false;
async function follow() {
  if (!state.live || followBusy || !state.frames.length || document.hidden || covered) return;
  if(state.run?.mode==='tiled' && tiled.moving)return;
  const gen = generation;
  followBusy = true;
  try {
    let target;
    do {
      target = state.frames.length - 1;
      if (target !== state.current) await show(target);
    } while (gen === generation && state.live && !document.hidden && !covered && state.frames.length - 1 !== target && !(state.run?.mode==='tiled' && tiled.moving));
  } finally {
    if (gen === generation) followBusy = false;
  }
}

function setLive(on) {
  state.live = on;
  $('live').setAttribute('aria-pressed', String(on));
  if (on) { stopReplay(); follow(); }
}

function stopReplay() {
  if (state.replay) { clearInterval(state.replay); state.replay = null; $('replay').setAttribute('aria-pressed', 'false'); }
}

function startReplay() {
  if (state.replay) return stopReplay();
  setLive(false);
  if (state.current >= state.frames.length - 1) show(0);
  $('replay').setAttribute('aria-pressed', 'true');
  let busy = false;
  state.replay = setInterval(async () => {
    if (busy || covered) return;   // under Compare the replay waits (it goes on when the view closes)
    const next = state.current + 1;
    if (next >= state.frames.length) {
      stopReplay();
      if (ACTIVE.includes(state.run?.state)) setLive(true);
      return;
    }
    busy = true;
    if(state.run?.mode!=='tiled')for (let k = 1; k <= 6; k++) if (next + k < state.frames.length) getFrame(next + k).catch(() => {});
    await show(next);
    busy = false;
  }, 1000 / 14);
}

// ---- socket ---------------------------------------------------------------------------------------------------
function onMessage(message) {
  if (message.ev === 'analysis') { analysis.event(message); if (message.state === 'unavailable') healthAgain(); return; }
  if (message.ev === 'qc') { if (state.mode === 'pick' && message.run === undefined && message.sample === state.sample?.id) pickerQc(message); qcClient.event(message); return; }
  if (message.ev === 'deeper') { deeper.event(message); return; }   // Go deeper's results, errors and the details' learning
  if (message.ev === 'gone') { runGone(message.run); return; }
  if (message.ev === 'older') { olderPrompt.event(message); return; }                // the same older runs in every window
  if (message.ev === 'session') { setSession(apiSession(message)); return; }          // a notebook's session changed
  if (message.ev === 'run') {
    const before = state.run, run = message.run;
    if (before?.id === run.id && before.kind === 'stain' && (before.exclusions?.version || 0) > (run.exclusions?.version || 0))
      message = { ...message, run: { ...run, exclusions: before.exclusions, summary: before.summary } };
    zarrSave.update(message.run); updateLeaving(message.run); awayNotice(message.run);
  }
  if (message.ev === 'runs') {
    // A test square's own two runs (their meta's qc_test names it) are the engine's, never listed.
    state.runs = message.runs.filter(run => !run.qc_test?.of)
      .map(run => run.id === state.run?.id ? { ...run, frames: Math.max(run.frames || 0, state.frames.length) } : run);
    for (const run of state.runs) if (run.id !== state.run?.id) { zarrSave.update(run); updateLeaving(run); }
    const current = state.runs.find(run => run.id === state.run?.id);
    if (current) onMessage({ ev: 'run', run: current });
    // One run at a time: a run that appears while this window shows none was started in another window, and is followed.
    // Never this window's own: its Start or a run it opens (ownOpens), or the run its start-up opens (until booted).
    if (!state.run && state.health?.one_run && state.booted && !ownOpens && !leaving && !olderPrompt.open && !state.ended && state.runs.length) {
      const id = state.runs[0].id;
      if (followingRun !== id) { followingRun = id; openRun(id).then(() => notice(SHELL_TEXT.startedElsewhere), error => notice(error.message)).finally(() => { followingRun = null; }); }
    }
  } else if (message.ev === 'counted' && state.run && message.run === state.run.id) {
    // A watch from the frames the page holds (a reconnect, or the run just opened): every counted one among them, its
    // counts line may have come after its record (proseg_counts.recent names the 64 newest alone), and those counted for
    // a pause, the hold, a finish or the end (`sampled`).
    if (Array.isArray(message.seqs)) for (const seq of message.seqs) if (Number.isInteger(seq)) countedSeqs.add(seq);
    if (Array.isArray(message.sampled)) for (const seq of message.sampled) if (Number.isInteger(seq)) sampledSeqs.add(seq);
    cellStats.recount?.();
  } else if (message.ev === 'frames' && state.run && message.run === state.run.id) {
    if (message.from > state.frames.length) { socket.watch(state.run.id, state.frames.length); return; }
    const fresh = message.frames.slice(state.frames.length - message.from);
    if (!fresh.length) return;
    for (const frame of fresh) state.frames.push(frame);
    timeline.setFrames(state.frames);
    state.run.last = state.frames.at(-1);
    state.run.frames = state.frames.length;
    updateLeaving(state.run);
    if (covered) return;   // under Compare nothing shows the new frame: cover() catches up when it closes
    // The panels change with the frame that follow() is about to show (one paint), or in the next batch.
    later.run();
    // A hidden page fetches and uploads nothing; it catches up with the newest frame when shown again.
    if (state.live) { if (!document.hidden) follow(); } else later.hud();
  } else if (message.ev === 'run' && state.run && message.run.id === state.run.id) {
    const before = state.run;
    state.run = message.run;
    // The frame on screen counted since its statistics came (proseg_counts.recent): its count joins them.
    noteCounted(state.run); cellStats.recount?.();
    if (['stain','patches'].includes(state.run?.kind)) {
      showStain(state.run);
      if (!before.image && state.run.image) loadRunImage();
      if (before.state !== state.run.state || before.paused !== state.run.paused || before.ack?.state !== state.run.ack?.state) renderRun();
      else later.run();
      return;
    }
    if(state.run.mode==='tiled') {
      tiled.run=state.run;
      // A paused run's new window (its 3-D layers over a moved view) is fetched now, or when Compare closes (cover()).
      if(!covered && requested===state.current && tiled.windowChanged(state.frames[state.current]))show(state.current,{pulse:false});
    }
    if (!before.info && state.run.info) onInfo();
    if (!before.image?.channels?.length && state.run.image?.channels?.length) loadRunImage();
    // A new state (paused, finishing, done...) shows at once and may change how analysis results apply; other run
    // events (preparation progress, the runs poll every 1.5 s) only join the next batch.
    if (before.state !== state.run.state || before.ack?.state !== state.run.ack?.state) { renderRun(); analysis.frame(); qcClient.frame(); }
    else later.run();
    if (before.state !== 'done' && state.run.state === 'done') revealResults(state.run);
  }
}
/** The run on screen was paused while no window showed it (the engine pauses a page's runs when the page went without
 *  closing: a crash, a sleeping computer, a lost connection): the page says why, once for each such pause. */
const awaySaid = new Set();
function awayNotice(run) {
  if (!run?.paused_away || run.id !== state.run?.id) return;
  const pause = `${run.id} ${run.paused_away}`;
  if (awaySaid.has(pause)) return;
  awaySaid.add(pause); notice(SHELL_TEXT.awayPaused);
}

// ---- run view -------------------------------------------------------------------------------------------------
let opening = 0, followingRun = null;
let ownOpens = 0;   // this window's Start and openRun still on their way (onMessage never follows a run meanwhile)
async function openRun(runId) {
  const ticket = ++opening;
  // Compare goes with the run it was over, and what a press of its button said.
  compareView.close(false); cover(false, false); clearExplanation($('compare-button'));
  // Nothing due or on its way is asked for while the run loads. The picker's boxes go once it has loaded (below): a run
  // that cannot be opened leaves them, and the picker's next change or Start asks for boxes again.
  cancelPatchPreview(); patchPreview.key = '';
  globalThis.sopaCompanionLog?.add('action', `opened run ${runId}`);
  resetRequests();
  stopReplay();
  let run, frames;
  ownOpens++;          // until the run is this window's (state.run below)
  try {
    run = await api.run(runId);
    if (ticket !== opening) return;
    ({ frames } = await api.frames(runId));
    if (ticket !== opening) return;
  } finally { ownOpens--; }
  clearFrames();
  resetRequests();
  setView(false, false);
  stage.info = run.info;
  transcripts.reset(run.id);
  layerPanel.sampleKind = run.sample_kind;
  layerPanel.sample = run.sample ?? null;   // the image's contrast limits are remembered for this sample (layer-panel.js)
  masks.reset(run.sample_kind === 'zarr' ? null : run.id);cellStats.reset(run.id);countedSeqs=new Set();sampledSeqs=new Set();noteCounted(run);resetCellHistory();lastPointer=null;
  resetPreview();
  priors.reset();
  priors.segments.setSample(run.sample_kind === 'zarr' ? run.sample : null);   // its segmentation layers
  stainView.reset();
  clearPatchPreview(); stainPatches.reset(); stainExclusions.reset(); tissueView.update(null,null,false);
  layerPanel.runKind = run.kind;
  if (!['stain','patches'].includes(run?.kind)) priors.load(run);
  state.display.layer = Math.min((run.info?.layers || 1)-1, layerPanel.saved.layer ?? -1);
  state.mode = 'run';
  state.run = run;
  if (run.leaving) leave(() => enterPick(null));
  setTool(toolOf(run.kind));   // the header names the run's tool, and New run stays in it
  tiled.reset(run.mode==='tiled' || ['stain','patches'].includes(run?.kind)?run:null,true);
  // The clusters' settings say which of their constants this kind of run keeps from before proseg's count
  // (run.constants.kept: "min_counts" on a whole sample).
  analysis.reset(['stain','patches'].includes(run?.kind) ? null : run.id, { kept: run.constants?.kept ?? null });
  resetQc(['stain','patches'].includes(run?.kind) ? null : run);
  if (['stain','patches'].includes(run?.kind)) {
    stage.info = { kind: 'stain' };
    priors.settings.xenium.on = true;
    layerPanel.selected = 'xenium';
    stainView.reset(run);
  }
  state.frames = frames;
  state.cells = [];
  state.image = null;
  renderChannels([]);
  renderLayers(run.info?.layers || 1);
  state.current = -1;
  state.selected = state.hover = 0;
  stage.set({ selected: 0, hover: 0 });
  stage.tiled = run.mode === 'tiled';   // whole-sample runs show every new iteration as one static highlight frame
  stage.setFrame(null, false);
  timeline.setPlan(run.info?.plan || [], run);   // the run: proseg's temperature schedule for the knob marks
  timeline.setFrames(state.frames);
  timeline.setCurrent(-1);
  history.replaceState(null, '', '?run=' + encodeURIComponent(runId));
  setSample(run.sample ? { id: run.sample, name: run.sample_name, kind: run.sample_kind } : null);
  socket.watch(runId, frames.length);
  layout();
  shown.knobs = shown.results = '';   // layout() showed the knobs panel: render the run's own state again
  const [x0, y0, x1, y1] = run.box;
  stage.fit({ x0, y0, w: x1 - x0, h: y1 - y0 }, false, 0.03);
  stage.setImage(null, []);
  loadRunImage();
  if (run.info && !['stain','patches'].includes(run?.kind)) onInfo();
  if (['stain','patches'].includes(run?.kind)) showStain(run);
  renderRun();
  zarrSave.update(run);
  state.live = ACTIVE.includes(run.state);
  $('live').setAttribute('aria-pressed', String(state.live));
  if (frames.length) await show(frames.length - 1, { pulse: false });
  if (ticket === opening && state.live) follow();
  if (ticket === opening) awayNotice(state.run);
}

function onInfo() {
  const info = state.run.info;
  stage.info = info;
  stage.dirty = true;
  if(state.run.mode!=='tiled')transcripts.load();
  priors.load(state.run);
  timeline.setPlan(info.plan || [], state.run);
  const id = state.run.id;
  api.cells(id).then((cells) => { if (state.run?.id === id) state.cells = cells; }).catch(() => {});
  renderLayers(info.layers || 1);
}

function loadRunImage() {
  const run = state.run, meta = run.image;
  if(run.mode==='tiled' || ['stain','patches'].includes(run?.kind)) {
    // Every channel of the run is listed at once, loaded or not (a whole-sample view fetches only the channels that
    // are on): a channel saved as hidden stays in the panel, and turning it on fetches it (onChannelsChange).
    if(meta?.channels?.length && !layerPanel.channels.length)layerPanel.setChannels(meta.channels.map((c,k)=>{
      const channel=layerPanel.channel(c,null,k);
      if(['stain','patches'].includes(run?.kind) && layerPanel.saved.channels?.[c.name]?.on===undefined)
        channel.on=(run.stain.channels || [run.stain.nucleus,run.stain.cytoplasm]).includes(c.name);
      return channel;
    }));
    tiled.run=run;
    if (['stain','patches'].includes(run?.kind)) tiled.imageView(); else tiled.loadImage();
    return;
  }
  if (!meta?.channels?.length) { state.image = null; return; }
  const runId = run.id, gen = generation;
  Promise.all(meta.channels.map(async (channel,k) => {
    let source;
    if (stage.norm16) {
      try {
        const response=await fetch(api.imageUrl(runId,channel.file16 || `c${k}.u16`));
        if(!response.ok) throw Error('uint16 unavailable');
        const bytes=await response.arrayBuffer();
        if(bytes.byteLength!==meta.width*meta.height*2) throw Error('Truncated image');
        source={data:new Uint16Array(bytes),width:meta.width,height:meta.height};
      } catch { /* PNG fallback */ }
    }
    if(!source)source=await api.image(api.imageUrl(runId,channel.file));
    const c=layerPanel.channel(channel,source,k);
    if(!source.data) {
      // A CPU canvas: reading pixels back from a GPU canvas would wait for the GPU process.
      const canvas=document.createElement('canvas');canvas.width=source.naturalWidth;canvas.height=source.naturalHeight;
      const g=canvas.getContext('2d',{willReadFrequently:true});g.drawImage(source,0,0);const rgba=g.getImageData(0,0,canvas.width,canvas.height).data;
      c.hist=Array(256).fill(0);for(let i=0;i<rgba.length;i+=4)c.hist[rgba[i]]++;
    }
    return c;
  })).then(channels=>{
    if(gen!==generation || state.run?.id!==runId)return;
    state.image={meta,channels};
    stage.setImage({x0:meta.x0,y0:meta.y0,w:meta.width*meta.um_per_px,h:meta.height*(meta.um_per_px_y ?? meta.um_per_px)},channels);
    layerPanel.setChannels(channels);
  }).catch(()=>{if(gen===generation)notice('The morphology image could not be loaded; the masks are still shown.');});
}
function renderChannels() { layerPanel.setChannels([]); }

function renderLayers(count) {
  $('layers-row').hidden = stage.orbit.enabled || count < 2;
  const group = $('layers');
  group.textContent = '';
  const options = [[-1, '2-D'], ...Array.from({ length: count }, (_, k) => [k, `z${k + 1}`])];
  for (const [value, label] of options) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.title = value < 0 ? "Transcript-weighted projection of all z layers (proseg's consensus)" : `Voxel layer ${value + 1} of ${count} (bottom to top)`;
    button.setAttribute('aria-pressed', String(state.display.layer === value));
    button.onclick = () => {
      state.display.layer = value;
      layerPanel.saved.layer=value; layerPanel.save();
      for (const b of group.children) b.setAttribute('aria-pressed', String(b === button));
      if (state.current >= 0) show(state.current, { pulse: false });
    };
    group.append(button);
  }
  if(state.run?.mode==='tiled' && !['stain','patches'].includes(state.run?.kind))tiled.notify();
}

let tiledMissed = false;   // a whole-sample view refresh asked for under Compare: cover(false) shows the frame again
async function refreshTiled() {
  if(state.run?.mode!=='tiled')return;
  if(covered){tiledMissed=true;return;}
  const seq=state.live?state.frames.length-1:requested>=0?requested:state.current;
  await show(seq,{pulse:false});
  if(state.live)follow();
}

function tiledStatus(status) {
  // Called for every camera move and every frame: write only what changed.
  const hint=$('tiled-hint'),text=status?.text || '';
  if(hint.hidden!==!text)hint.hidden=!text;
  if(hint.textContent!==text)hint.textContent=text;
  const available=!status || status.layers;
  const button=$('view').querySelector('[data-value="3d"]');
  if(button.disabled!==!available)button.disabled=!available;
  const title=status?.level==='overview'?`Zoom in to ≤ ${fmt1.format(status.limit/1000)} mm to see this part in 3-D`:
    status && !status.layers?(status.active?'3-D layers follow the view…':'This finished run has 3-D layers only where it was viewed'):'3-D view (V)';
  if(button.title!==title)button.title=title;
  // An active run's layers follow a moved view a moment later (its window is posted, then written): 3-D stays, the stage
  // draws the consensus until they come. Only where none can come (the overview, a finished run) does 2-D return.
  if(!available && stage.orbit.enabled && !(status.active && status.level==='detail'))setView(false,false);
  for(const b of $('layers').children) {
    const disabled=!!status && b.textContent!=='2-D' && !available;
    if(b.disabled!==disabled)b.disabled=disabled;
    const pressed=String(b.textContent===(available && state.display.layer>=0?`z${state.display.layer+1}`:'2-D'));
    if(status && b.getAttribute('aria-pressed')!==pressed)b.setAttribute('aria-pressed',pressed);
  }
}

function phaseLabel(phase) { return PHASES[phase]?.label || phase || '—'; }

function rate() {
  const f = state.frames, n = f.length;
  if (n < 3) return null;
  const k = Math.min(12, n - 1);
  const dt = f[n - 1].t - f[n - 1 - k].t;
  return dt > 0 ? (f[n - 1].iter - f[n - 1 - k].iter) / dt : null;
}

// ---- DOM writes only on change: a per-frame update must not invalidate style or layout for an unchanged value ----
/** The stage, or (no WebGL2: often a remote or virtual display) the window says it cannot draw, and why, in place of the
 *  page; the error goes on (the page's log keeps it) and nothing else of the page starts. Only the app's own Chrome
 *  window has chrome_flags to change (page-key.mjs appWindow: known at once, before the engine is asked anything);
 *  any other browser gets its own words. */
function makeStage() {
  try { return new Stage($('stage')); } catch (error) {
    const box = $('no-webgl');
    if (box) {
      setText($('no-webgl-title'), SHELL_TEXT.noWebglTitle); setText($('no-webgl-why'), SHELL_TEXT.noWebglWhy);
      setText($('no-webgl-help'), appWindow() ? SHELL_TEXT.noWebglHelp : SHELL_TEXT.noWebglHelpBrowser); box.hidden = false;
    }
    throw error;
  }
}
function setText(el, text) { text = text == null ? '' : String(text); if (el.textContent !== text) el.textContent = text; }
function setProp(el, key, value) { if (el[key] !== value) el[key] = value; }
function setAttr(el, key, value) { if (el.getAttribute(key) !== value) el.setAttribute(key, value); }
/** A collapsed side panel is not rendered: its contents are brought up to date when it opens. */
function panelOpen(id) { const panel = $(id); return panel.open && !panel.hidden; }

/** The prior's own layer shows by itself at frame 0 (unless the user chose), but not over the cells' Mixing colours:
 *  there they show where Xenium's cells already look mixed. */
function autoPrior() { layerPanel.autoPrior(state.current === 0 && stage.options.cellColorBy !== 'mixing'); }
/** Paused by itself at the end of proseg's standard run (its schedule: the engine's `schedule_done`), not sampled on since. */
function heldAtEnd(run) { return !!run?.schedule_done && !run.continued && run.state === 'paused' && run.ack?.state === 'paused'; }
/** The calm note on the stage (the #prior-hint card) while a run waits for the user, never a dialog: paused on its prior
 *  before it has sampled (every run starts there: the run button says Start, and the note names it only while a press
 *  works: runButtons), or held at the end of proseg's standard run, where it says what each button does: Resume gives
 *  up proseg's own result for good (it samples on at temperature 1, averaging the new iterations in, and only the
 *  newest state can be saved from then on); '' otherwise. */
function stageNote(run) {
  if (heldAtEnd(run)) return "Proseg's standard run is complete. Finish & save keeps this result. Resume samples on at full temperature, and after that this exact result can't be saved.";
  if (run.state === 'paused' && run.ack?.state === 'paused' && state.current === 0 && state.frames.length === 1)
    return `This is proseg's starting point, built from ${run.sample_kind === 'zarr' ? 'the chosen prior' : 'the Xenium segmentation'}.`
      + (runButtons(run, state.frames, { prompt: finishingPrompt }).start ? ' Press Start to begin sampling.' : '');
  return '';
}
function updateHud() {
  const run = state.run;
  if (!run) return;
  if (['stain','patches'].includes(run?.kind)) return;
  const h = state.frames[state.current];
  autoPrior();
  const note = legendCorner ? '' : stageNote(run);   // the test corner open over it: the corner says the run waits
  setProp($('prior-hint'), 'hidden', !note);
  setText($('prior-hint'), note);
  setProp($('hud'), 'hidden', !h);
  if (h) {
    const phase = PHASES[h.phase], dot = $('hud-dot'), color = phase?.color || '#5c6573';
    setText($('hud-phase'), phaseLabel(h.phase));
    if (dot.phaseColor !== color) { dot.phaseColor = color; dot.style.background = color; }
    dot.classList.toggle('pulse', state.live && run.state === 'running' && run.ack?.state === 'running');
    setText($('hud-iter'), fmt.format(h.iter));
    // "/ 450": the iteration out of proseg's standard run while within it; past its end (sampling on) it is open-ended.
    const end = scheduleEnd(run.info);
    setText($('hud-total'), end == null ? '' : '/ ' + fmt.format(end));
    setProp($('hud-total'), 'hidden', end == null || h.iter > end);
    const parts = [`${h.voxel} µm voxels`, `temperature ${h.temperature.toFixed(2)}`, `${fmt.format(h.changed)} voxels changed`];
    if (!state.live) parts.push(state.current === state.frames.length - 1 ? 'latest' : `${fmt.format(state.frames.length - 1 - state.current)} behind live`);
    setText($('hud-sub'), parts.join(' · '));
  }
  const r = rate();
  // Iterations, as the HUD and the log name them (never snapshot numbers: they count differently, from 0 and past the
  // finer-voxel step); looking back, the newest one too.
  const newest = state.frames.at(-1)?.iter;
  const where = h ? `Iteration ${fmt.format(h.iter)}` + (state.current < state.frames.length - 1 && newest != null ? ` (newest ${fmt.format(newest)})` : '') : 'Waiting for the first iteration';
  if (run.state === 'done') {
    setText($('status'), `Finished - ${fmt.format(state.frames.at(-1)?.iter ?? run.last?.iter ?? 0)} iterations - results saved`);
    return;
  }
  if (['stopped', 'failed'].includes(run.state)) {
    setText($('status'), `${run.state === 'failed' ? 'Failed' : 'Stopped'}${run.message ? ' - ' + run.message : ''}`);
    return;
  }
  setText($('status'), (run.phase === 'loading' && !h ? 'Loading proseg… · ' : run.state === 'paused' ? (run.ack?.state === 'paused' ? (heldAtEnd(run) ? 'Standard run complete · ' : 'Paused · ') : 'Pausing... · ') : '') + where + ` · ${fmt1.format(run.state === 'running' ? r || 0 : 0)} iterations/s · ${fmt1.format(state.frames.at(-1)?.t || 0)} s elapsed`);
}

/** A definition list, updated in place when its terms are unchanged (each new frame only changes values). */
function facts(target, rows) {
  rows = rows.filter(([, v]) => v !== undefined && v !== null && v !== '');
  const nodes = target.children;
  if (nodes.length === 2 * rows.length && rows.every(([k], i) => nodes[2 * i].textContent === k)) {
    rows.forEach(([, v], i) => setText(nodes[2 * i + 1], v));
    return;
  }
  target.textContent = '';
  for (const [k, v] of rows) {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    target.append(dt, dd);
  }
}

function updateFacts() {
  const h = state.frames[state.current];
  setProp($('now-panel'), 'hidden', !h);
  if (!h || !panelOpen('now-panel')) return;
  const pct = (v) => (v == null || !h.transcripts ? null : `${fmt1.format((100 * v) / h.transcripts)} %`);
  facts($('facts'), [
    ['Phase', phaseLabel(h.phase)],
    ['Iteration', fmt.format(h.iter)],
    ['Temperature', h.temperature.toFixed(3)],
    ['Voxel size', `${h.voxel} µm × ${h.nz} layers`],
    ['Cells', `${fmt.format(h.cells)}${h.visible !== h.cells ? ` (${fmt.format(h.visible)} visible in 2-D)` : ''}`],
    ['Assigned', pct(h.assigned)],
    ['Not background', pct(h.foreground)],
    ['Voxels changed', fmt.format(h.changed)],
    ['Log-likelihood', h.loglik == null ? 'not yet updated' : h.loglik.toExponential(5)],
    ['Iteration time', h.ms ? `${fmt.format(Math.round(h.ms.morphology + h.ms.repositioning + h.ms.parameters))} ms` : null],
    ['Elapsed', `${fmt1.format(h.t)} s`],
  ]);
  setText($('phase-help'), PHASES[h.phase]?.help || '');
}

// While proseg reads its inputs (run.phase 'loading': started, no frame yet; up to a minute or two for a whole sample)
// the overlay says so
// and counts down the engine's estimate (run.eta_s: the seconds left when it was sent), once a second: the one overlay
// change that is not the user's or a new iteration's, for the minute or so a whole sample loads. Within one start of
// proseg the earliest estimate stands, so a run update that repeats the figure does not restart the count; a new start
// (the engine retrying without factorisation: a new pid) counts its own estimate. Past it, the overlay just says loading.
let loadingEta = null, loadingTimer = null;
function loadingText(run) {
  const now = performance.now(), start = `${run.id} ${run.pid ?? ''} ${run.started ?? ''}`;
  if (Number.isFinite(run.eta_s) && run.eta_s >= 0 && (loadingEta?.start !== start || now + run.eta_s * 1000 < loadingEta.until))
    loadingEta = { start, until: now + run.eta_s * 1000 };
  const left = loadingEta?.start === start ? loadingEta.until - now : 0;
  if (left <= 0) return 'Loading proseg…';
  const seconds = Math.ceil(left / 1000);
  loadingTimer = setTimeout(() => later.run(), left - (seconds - 1) * 1000 + 1);   // just after the shown number changes
  return `Loading proseg… ~${seconds} s`;
}

function renderRun() {
  const run = state.run;
  if (!run) return;
  if (['stain','patches'].includes(run?.kind)) { renderStainRun(run); return; }
  const active = ACTIVE.includes(run.state);
  const paused = run.state === 'paused' && run.ack?.state === 'paused';
  const pausing = run.state === 'paused' && !paused;
  const resuming = run.state === 'running' && run.ack?.state === 'paused';
  const loading = run.phase === 'loading' && !state.frames.length;   // nothing to pause or resume yet
  clearTimeout(loadingTimer);
  const toggle = $('run-toggle');
  // On its prior, before anything has run, the run button starts the run; held at the end of proseg's standard run,
  // Finish & save is the main button (it keeps proseg's own result) and Resume a plain one: a deliberate choice.
  const onPrior = paused && state.frames.length === 1, held = heldAtEnd(run);
  setText(toggle, loading && pausing ? 'Loading…' : pausing ? 'Pausing...' : resuming ? 'Resuming...' : paused ? (onPrior ? 'Start' : 'Resume') : 'Pause');
  setProp(toggle, 'title', onPrior ? 'Start sampling (Space)' : 'Pause or resume proseg (Space)');
  toggle.classList.toggle('primary', !held); $('run-stop').classList.toggle('primary', held);
  const sampling = ['running', 'paused'].includes(run.state);
  // Which buttons a press makes work (run-buttons.mjs: the words that name a button read the same).
  const buttons = runButtons(run, state.frames, { prompt: finishingPrompt });
  setProp(toggle, 'disabled', !(buttons.start || buttons.resume || buttons.pause));
  setProp($('run-step'), 'disabled', !buttons.step);
  setProp($('run-stop'), 'disabled', !sampling || finishingPrompt);
  // The knob guide's timing and Try message name Start, Resume or Step only while they work.
  const buttonsKey = JSON.stringify(buttons);
  if (buttonsKey !== shown.buttons) { shown.buttons = buttonsKey; knobs.guide?.update?.(); }
  setProp($('run-abort'), 'disabled', !active);
  // The live knobs depend on the run's state and values and on the newest frame's effective values, not on every frame.
  const last = (run.last?.seq ?? -1) > (state.frames.at(-1)?.seq ?? -1) ? run.last : state.frames.at(-1) || run.last;
  const knobsKey = () => JSON.stringify([knobs.key, run.id, run.state, last?.live, run.live_values, run.options, options.schema.length]);
  if (knobsKey() !== shown.knobs) { knobs.render({ ...run, last }, options.schema); shown.knobs = knobsKey(); }
  for (const b of $('speed').children) {
    setAttr(b, 'aria-pressed', String(b.dataset.value === (run.speed || 'max')));
    setProp(b, 'disabled', !sampling);
  }
  const live = $('live');
  live.classList.toggle('ended', !active);
  setText(live.lastChild, active ? 'Live' : 'Latest');
  setProp(live, 'title', active ? 'Follow new iterations as they arrive (L)' : 'Show the final iteration (L)');
  const info = run.info;
  const [x0, y0, x1, y1] = run.box;
  facts($('run-facts'), [
    ['Sample', run.sample_name],
    ['Region', `${fmt.format(Math.round(x1 - x0))} × ${fmt.format(Math.round(y1 - y0))} µm`],
    ['Origin', `${fmt.format(Math.round(x0))}, ${fmt.format(Math.round(y0))} µm`],
    ['Transcripts', run.transcripts ? fmt.format(run.transcripts) + (info ? ` (${fmt.format(info.transcripts)} modelled)` : '') : null],
    ['Cells', info ? fmt.format(info.cells) : null],
    ['Genes', info ? fmt.format(info.genes) : null],
    // Finished runs: the results card below names the folder, so the state stays short.
    ['State', (loading ? 'Loading proseg…' : pausing ? 'Pausing...' : resuming ? 'Resuming...' : run.ack?.state && sampling ? run.ack.state : run.state)
      + (run.message && run.state !== 'done' ? ` — ${run.message}` : '')],
  ]);
  setProp($('run-panel'), 'hidden', false);
  setProp($('run-error'), 'hidden', run.state !== 'failed');
  setText($('run-error'), run.state === 'failed' ? run.message : '');
  renderResults(run);
  const overlayState = $('hud-state');
  const busy = pausing || (finishingPrompt && !paused) || run.state === 'finishing' || run.state === 'preparing' || (ACTIVE.includes(run.state) && !state.frames.length);
  // While loading, what the engine says meanwhile (a retry without factorisation, Finish & save asked for, the disk
  // guard's pause) shows with the countdown.
  const text = busy ? (loading && !finishingPrompt ? [loadingText(run), run.message].filter(Boolean).join(' · ') : pausing || (finishingPrompt && !paused) ? 'Pausing at the next iteration boundary...' : run.state === 'finishing' ? 'Saving results...' : run.message || 'Starting proseg…')
    : run.state === 'failed' && !state.frames.length ? 'Proseg failed: ' + run.message : '';
  const spinning = overlayState.firstElementChild?.className === 'spinner';
  if (!text) setProp(overlayState, 'hidden', true);
  else if (overlayState.hidden || overlayState.textContent !== text || spinning !== busy) {
    overlayState.hidden = false;
    // A changing busy text (the countdown, preparation progress) replaces only its words, not the spinner element:
    // no DOM is rebuilt for it once a second (the spinner is static: nothing animates by itself).
    if (busy && spinning && overlayState.childNodes.length === 2) overlayState.lastChild.textContent = text;
    else if (busy) { overlayState.innerHTML = '<span class="spinner"></span>'; overlayState.append(document.createTextNode(text)); }
    else overlayState.textContent = text;
  }
  updateHud();
  const index = state.runs.findIndex(r => r.id === run.id);
  const summary = { ...run, frames: Math.max(run.frames || 0, state.frames.length) };
  if (index >= 0) state.runs[index] = summary; else state.runs.unshift(summary);
}

function showStain(run) {
  stainExclusions.update(run);stainPatches.update(run);
  tissueView.update(run.sample,run.stain?.image,run.stain?.tissue_only!==false);
  if(patchSettings.sample!==run.sample)patchSettings.refresh(run.sample);
  priors.vectorsVisible=run.kind==='stain';stainView.update(run);
}

function resolutionFact(run) {
  const scale = run.stain?.resolution;
  if (!(scale?.um_per_px > 0)) return null;
  const cells = run.tiling?.cell_px_median;
  return `${resolutionName(scale) || levelName(0)} · ${Number(scale.um_per_px.toPrecision(3))} µm per pixel` +
    (cells > 0 ? ` · cells about ${Math.round(cells)} px across${cells < 10 ? ': too small to outline well' : ''}` : '');
}

function renderStainRun(run) {
  const patches=run.kind==='patches';
  const active = ['preparing', 'running', 'paused'].includes(run.state);
  const resumable = ['stopped', 'failed', 'done'].includes(run.state) && !run.progress?.complete;
  const toggle = $('run-toggle');
  const waiting=!patches && run.stain_started===false;
  $('run-toggle').hidden=$('run-stop').hidden=patches;
  setText(toggle, waiting ? STAIN_TEXT.start : run.paused || resumable ? STAIN_TEXT.resume : STAIN_TEXT.pause);
  setProp(toggle, 'title', toggle.textContent);
  setProp(toggle, 'disabled', !active && !resumable);
  toggle.classList.toggle('primary', active || resumable);
  setText($('run-stop'), STAIN_TEXT.stop);
  setProp($('run-stop'), 'title', STAIN_TEXT.stopping);
  setProp($('run-stop'), 'disabled', !active);
  facts($('run-facts'), patches ? [[STAIN_TEXT.sample, run.sample_name], [STAIN_PATCH_TEXT.startFrom,
    run.start_from==='empty'?STAIN_PATCH_TEXT.blank:STAIN_PATCH_TEXT.model],
    ...(run.start_from==='empty'?[]:[[STAIN_TEXT.method,STAIN_TEXT[run.stain.method]],[STAIN_TEXT.model,run.stain.model]])] : [[STAIN_TEXT.method, STAIN_TEXT[run.stain?.method || 'cellpose']], [STAIN_TEXT.model, run.stain?.model || 'cpsam'],
    [STAIN_TEXT.sample, run.sample_name],
    ['Resolution', resolutionFact(run)],
    [STAIN_TEXT.progress, stainProgress(run)], [run.progress?.resolved ? STAIN_TEXT.cells : STAIN_TEXT.tileCells,
      fmt.format(run.summary?.cells || 0)],
    [STAIN_TEXT.sizes, stainSizes(run)],
    [STAIN_TEXT.lastTile, run.last_tile ? `${run.last_tile.seconds.toFixed(1)} s` : null],
    [STAIN_TEXT.state, stainStatus(run)],
    ...(run.tissue?.tiles_skipped>0 ? [[STAIN_TEXT.tissue,`${run.tissue.tiles_skipped} tiles skipped outside the tissue`]] : [])]);
  $('run-panel').hidden = false;
  $('run-error').hidden = run.state !== 'failed';
  setText($('run-error'), run.message || STAIN_TEXT.failed);
  $('hud-state').hidden = true;
  $('prior-hint').hidden = patches || !waiting && !!run.progress?.resolved;
  setText($('prior-hint'), waiting ? STAIN_TEXT.waiting : run.progress?.resolved ? '' : STAIN_TEXT.interim);
  setText($('status'), patches ? STAIN_PATCH_TEXT.title : `${STAIN_TEXT.title} · ${stainProgress(run)} · ${stainStatus(run)}`);
  renderResults(run);
  const index = state.runs.findIndex(r => r.id === run.id);
  if (index >= 0) state.runs[index] = run; else state.runs.unshift(run);
}

let highlightTimer = null;
function revealResults(run) {
  $('run-panel').open = true;
  const box = $('results');
  requestAnimationFrame(() => {
    if (state.run?.id !== run.id || state.run.state !== 'done') return;
    // An instant scroll and a static highlight: a smooth scroll or a fading highlight is dozens of frames to encode.
    box.scrollIntoView({ block: 'nearest' });
    box.classList.add('highlight');
    clearTimeout(highlightTimer); highlightTimer = setTimeout(() => box.classList.remove('highlight'), 2000);
  });
}

function renderResults(run) {
  if(run.kind==='patches'){
    if($('results').children.length)$('results').replaceChildren();
    setProp($('results'),'hidden',true);return;
  }
  setProp($('retry-note'), 'hidden', !run.retried_without_factorization);
  setText($('retry-note'), 'Proseg retried without factorization.');
  const box = $('results'), key = JSON.stringify([run.id, run.state, run.sample_kind, run.zarr_save, run.message, run.progress, run.stain, run.exclusions?.digest, run.exclusions?.version, exclusionsPending(),
    state.session?.notebook.busy ?? null, state.session?.writes ?? null]);
  if (key === shown.results) return;   // unchanged (renderRun runs for every new frame)
  shown.results = key;
  box.hidden = run.state !== 'done' && !saveable(run);
  box.replaceChildren();
  if (box.hidden) return;
  if (['stain','patches'].includes(run?.kind)) {
    const model = document.createElement('p'); model.textContent = `${STAIN_TEXT.model}: ${run.stain?.model || 'cpsam'}`;
    box.append(model);
  }
  if (run.sample_kind === 'zarr') {
    const save = run.zarr_save, row = document.createElement('div'); row.className = 'save-sample';
    // A notebook's session: Add to your SpatialData (session.mjs), with where the result is now.
    const added = state.session ? resultsRow(run, { outdated: saveOutdated(run) }) : null, label = document.createElement('span');
    label.textContent = added ? added.label : saveOutdated(run) ? STAIN_TEXT.savedOutdated : ['saved', 'queued'].includes(save?.state) ?
      `Saved as ${save.shapes_key}${save.table_key ? ' + ' + save.table_key : ''}` : `Not saved to ${run.sample_name} yet`;
    row.append(label);
    const line = state.session && save?.state !== 'saved' ? addLine(state.session, run) : '';   // where it is until it is added
    if (line) { const where = document.createElement('p'); where.className = 'muted small'; where.textContent = line; row.append(where); }
    if (save?.state === 'saving' || save?.state === 'failed') {
      const message = document.createElement('p'); message.className = save.state === 'saving' ? 'save-progress' : 'error';
      if (save.state === 'saving') { const spinner = document.createElement('span'); spinner.className = 'spinner'; message.append(spinner); }
      message.append(document.createTextNode(save.message || (save.state === 'saving' ? 'Writing shapes and table…' : 'Saving failed. Try again.'))); row.append(message);
    }
    const button = document.createElement('button'); button.type = 'button'; button.className = 'primary';
    button.textContent = ['saved', 'queued'].includes(save?.state) ? 'Save again…' : save?.state === 'saving' ? 'View save progress…' : 'Save to sample…';
    if (['stain','patches'].includes(run?.kind)) button.textContent = STAIN_TEXT.save;
    if (added) button.textContent = added.button;      // a session's words, its saving one too ("View progress…")
    button.onclick = () => zarrSave.open(state.run); row.append(button); box.append(row);
  }
  const offer = proceedOffer(run, { api: !!state.session, pending: exclusionsPending() });
  if (offer.shown) {
    const button = document.createElement('button'); button.id = 'proceed'; button.type = 'button';
    button.textContent = PROCEED_TEXT.button; button.title = offer.title; button.className = offer.ready ? 'primary' : '';
    guard(button, () => [], () => proceedToProseg(state.run));
    box.append(button);
    if (state.revealProceed === run.id) { state.revealProceed = null; requestAnimationFrame(() => { if (state.run?.id === run.id) button.scrollIntoView({ block:'nearest' }); }); }
  } else if (offer.reason) {
    const reason = document.createElement('p'); reason.className = 'muted small'; reason.textContent = offer.reason; box.append(reason);
  }
}

function exclusionsPending() { return !!(stainExclusions.pending || stainExclusions.busy || stainExclusions.queued || stainExclusions.waiting?.length); }

// ---- the mixing check -----------------------------------------------------------------------------------------------
// qc-client.mjs picks the result for the frame shown (never a later one) and calls renderQc in the snapshot's paint
// batch (later.qc() in show()), or at once for the user's own actions: the panel, the chip, the timeline lane, the cell
// colours, the overlay's per-bin layer and the cell readouts change together.
let qcShownCells = null, qcBinsKey = null, qcBinsSeq = null, qcReadout = null;
let qcCovered = false;   // a result rendered under Compare waits: cover() renders the one shown when the view closes
let qcPreviousColour = null;   // the cells' Colour by before Mixing, as this page saw it (null: none seen here)
let qcCalm = false;   // the panel's "calm motion" switch (shown only when the motion costs too much): the page's choice
// The check's switch for new runs, as far as this page knows: the engine starts every new run with the last choice made
// (its answer about the picker's sample, GET /samples/{id}/qc `new_runs`; then qc-client.mjs onSwitch: this page's own
// switch, or one flipped in another window while this page watched); null before any: off, as the check ships. Only
// while it is on does the picker say how the sample's check stands (pickerQc).
let qcNewRuns = null;
// The boxes over the stage (the HUD, the Mixing chip and legend, Go deeper's window, the test corner): the legend keeps
// clear of them while the test corner is open (placeLegend).
const qcBoxes = new Map();
let legendCorner = null;      // the test corner's box while it is open (its onBox) and its node: the legend's place (placeLegend)
function qcContext() {
  const o = stage.options;
  return { run: state.run, frames: state.frames, seq: state.current, live: state.live && state.run?.state === 'running',
    wantCells: o.cellColorBy === 'mixing' || o.transcriptColorBy === 'mixing' };
}
// The engine found mid-session that Clusters and the UMAP cannot run here (a run's analysis went "unavailable"): what
// /health says now reaches the mixing check's panel (withEngine) and the next run's pane (engineCan), as at the start.
let healthAsked = null;
function healthAgain() {
  healthAsked ??= api.health().then(health => {
    state.health = health; analysis.engineCan = health?.analysis !== false; analysis.prosegFix = health?.proseg_fix ?? null; qcClient.frame();
  }).catch(() => {}).finally(() => { healthAsked = null; });
}
function resetQc(run) {
  deeper.reset(run); deeperStartOutline(null); if (deeperArrows.clear()) drawOverlay();   // Go deeper closes with its run
  qcClient.reset(run); qcOverlay.clear(); qcInspecting = null; qcBinsKey = qcBinsSeq = qcReadout = null;
  if (qcShownCells !== null) { qcShownCells = null; stage.setMixing?.(null); }
  // The timeline's mixing lane is drawn inside the timeline's own height (timeline.js LANE_H): the dock, and so the
  // stage, never changes size for the check.
  timeline.setQc(null); qcClient.frame();
  // The mixing check's row goes with its run: the selection is still the old run's here (the calls above may have drawn
  // the row for it), and a new run shows its panel before its first frame.
  qcPanel.setCell(null); qcPanel.setTestRun?.(run ?? null);   // the old run's test square too (nothing more asked for it)
}
function renderQc(view) {
  if (covered) { qcCovered = true; return; }   // nothing of the check moves under Compare (cover())
  view = withEngine(view, state.health?.qc);   // nothing is scored where Clusters and the UMAP cannot run: said so
  const t0 = performance.now(), running = state.mode === 'run' && !!state.run, on = running && view.enabled === true;
  const o = stage.options;
  // Colour by: Mixing is offered while the check can show values, and listed (disabled, saying why) while it is on but
  // cannot run; switched off it is not listed. Off (or unable to run), the cells go back to their previous colours
  // (which renders again).
  const usable = on && !['unavailable', 'failed'].includes(view.status);
  layerPanel.setMixingAvailable(usable, on ? QC_TEXT.mixingCantRun : null);
  if (running && !usable && view.status !== 'loading' && (o.cellColorBy === 'mixing' || o.transcriptColorBy === 'mixing')) { setMixingColours(false); return; }
  qcOverlay.visible = on;
  // The zoom and the 3-D view are told by the Mixing legend beside the stage (it follows the camera): not the panel.
  // A click on its way (every action waits), whether another run can start and the repeat-run spread that
  // applies (whole samples: 20,000 cells or more, as the engine's qc_rules.repeat_mode).
  const panelState = fromView(running ? view : null, { sampleName: state.run?.sample_name ?? null, showingCells: o.cellColorBy === 'mixing',
    calmMotion: qcCalm, acting: qcActing, prosegFix: state.health?.proseg_fix ?? null, notebook: !!state.session,
    large: (state.run?.info?.cells ?? 0) >= 20000 });
  qcPanel.setState(panelState); qcChip.setState(panelState);
  // Paused or finished, a result that arrives for the frame on screen lands without motion (view.still:
  // nothing moves while paused apart from the transition a scrub causes); the cell tint and the per-bin layer
  // cross-fade only a second or more after the previous result (tintPlan).
  const motion = !view.still, cells = on ? view.cells ?? null : null;
  if (cells !== qcShownCells) { qcShownCells = cells; stage.setMixing?.(cells, { at: t0, motion }); }
  // The per-bin layer's usual level: the start line's (fixed for the run), so its colours change only where the tissue
  // changed; a run from nuclei only has none: the level shown. The result's areas are not handed over: none is drawn.
  const redraw = qcOverlay.setLevel({ anchorPer100: view.start?.per100 ?? null, tissuePer100: view.summary?.shown ?? view.summary?.per100 ?? null });
  timeline.setQc(usable ? { points: view.line, version: view.lineVersion, start: view.start?.per100 ?? null, formedIter: view.formingUntil ?? null } : null);
  if (redraw) drawOverlay(); else { qcBins(); updateQcLegend(); }
  // The cell tooltip and card follow the values shown.
  // Which result the cell values are from, too: a readout that withheld an earlier result's value shows the
  // result's own as soon as it arrives.
  const readout = `${view.status}:${on}:${view.atStart}:${view.summary?.seq}:${!!view.cells}:${view.startCells}:${view.cellsSeq}:${view.cellsCurrent}`;
  if (readout !== qcReadout) { qcReadout = readout; if (lastPointer) updateTooltip(lastPointer); if (state.selected) updateCellCard(); }
  if (qcInspecting) qcInspectDue();
  harness?.qcShown(running ? view.status : null, 0);   // the areas drawn: none
  try { performance.measure?.('plv:qc-apply', { start: t0, detail: { seq: view.summary?.seq ?? null } }); performance.clearMeasures?.('plv:qc-apply'); }
  catch { /* a profiling aid only */ }
  deeper.changed();   // Go deeper: the frame on screen or the check changed (its own throttle: one update a second live)
}
/** Colour by: Mixing on or off (the panel's Show on the cells, key M, Layers > Cells > Colour by). */
function setMixingColours(on) {
  const o = stage.options;
  if (on) {
    if (!layerPanel.mixingAvailable) return;
    if (o.cellColorBy !== 'mixing') qcPreviousColour = o.cellColorBy || 'identity';
    stage.set({ cellColorBy: 'mixing' });
  } else {
    // Back to the colour before Mixing. One this page never saw (Mixing kept in the saved settings from an earlier
    // session, on a run whose check is off: it is off by default): the clusters' colours, which every run has now.
    const back = qcPreviousColour && qcPreviousColour !== 'mixing' ? qcPreviousColour : 'cluster';
    stage.set({ ...(o.cellColorBy === 'mixing' ? { cellColorBy: back } : {}), ...(o.transcriptColorBy === 'mixing' ? { transcriptColorBy: 'cell' } : {}) });
  }
  testCorner.recolor();   // its cells in the stage's colours (stageColour)
  if (state.run) autoPrior();
  layerPanel.save(); layerPanel.render();
  qcClient.changed(true);   // the values the colours need, the legend, the panel's button
}
/** The panel's "calm motion" switch: results land at once on the stage, the overlay and the panel; the suspects' glow
 *  (the stage's and the test corner's) holds still, or breathes again, at once. */
function setCalmMotion(on) {
  qcCalm = !!on; stage.calmMotion = qcCalm; qcOverlay.calm = qcCalm;
  stage.replan?.(); testCorner.refresh(); compareView.refresh();
  qcClient.changed(true);
}
/** Layers > Cells > Colour by. */
function colourBy(value) {
  globalThis.sopaCompanionLog?.add('action', `colour by ${value}`);
  if (value === 'mixing') { setMixingColours(true); return; }
  qcPreviousColour = value; stage.set({ cellColorBy: value }); testCorner.recolor(); if (state.run) autoPrior(); qcClient.changed(true);
}
/** A card's area in view (its "Look at its cells", R1's "Show the outlines of ... here"): the camera moves there, at
 *  least 300 µm across (qc-areas.mjs flyTarget), out of 3-D. Nothing is drawn for it and nothing stays selected (no area
 *  is shown): each press moves the view again. Its
 *  outline is the card's own, else the result shown's area of the same letter; without one the view stays. */
function flyToArea(area) {
  const outline = areaOutline(area, qcClient.view?.areas);
  if (!outline || state.mode !== 'run') return;
  if (stage.orbit.enabled) setView(false, false);
  if (state.run?.mode === 'tiled') tiled.overview = false;
  stage.fit(flyTarget(outline), true, 0);
}
/** "Show Xenium's cells outlines here": the starting cells' outlines on, and the card's area in view. */
function showStartOutlines(area) {
  priors.settings.xenium.on = true; stage.dirty = true; layerPanel.save(); layerPanel.render();
  flyToArea(area);
}
function openQcPanel() { $('qc-panel').hidden = false; qcPanel.open(); }
/** "Look at its cells" on a card whose cause is not known (the engine's `inspect`: R1, R6, R7): the view
 *  moved to the card's area (flyToArea), then its most mixed judged cell selected with Go deeper open on it, so its genes
 *  can be checked against the image and the marker genes. The result shown's own cell values and the frame's cell
 *  statistics are needed: when they are not in yet they are asked for, and the cell is picked when they arrive (within
 *  INSPECT_WAIT_MS; the view moved alone otherwise). */
const INSPECT_WAIT_MS = 5000;
let qcInspecting = null;   // {area, until}: a "Look at its cells" waiting for the values or statistics it needs
function qcInspect(suggestion, area) {
  if (!area?.L || state.mode !== 'run') return;
  flyToArea(area);
  globalThis.sopaCompanionLog?.add('action', `looked at the cells where mixing stands out${suggestion?.rule ? ` (${suggestion.rule})` : ''}`);
  qcInspecting = { area, until: performance.now() + INSPECT_WAIT_MS };
  if (!qcClient.view?.cellsCurrent) qcClient.cellReading(1);   // the result shown's own values (a readout's request)
  qcInspectDue();
}
/** A waiting "Look at its cells": its cell, once the values and statistics are in (next tick: never inside a render). */
function qcInspectDue() {
  const ask = qcInspecting;
  if (!ask) return;
  if (performance.now() > ask.until || state.mode !== 'run') { qcInspecting = null; return; }
  const label = mostMixedCell(ask.area);
  if (!label) return;
  qcInspecting = null;
  setTimeout(() => { if (state.mode !== 'run') return; deeperSelect(label); deeper.open(label, { from: null }); }, 0);
}
/** The most mixed judged cell (the highest byte below 255) whose centre lies inside the area's outline (its own, else the
 *  result shown's area of the same letter), in the result shown and the frame on screen; 0 while that result's own
 *  values or the frame's cell statistics are not in. */
function mostMixedCell(area) {
  const s = cellStats.current, v = qcClient.view, outline = areaOutline(area, v?.areas);
  if (!outline || !s?.x || s.seq !== state.current || !v?.cells || !v.cellsCurrent) return 0;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of outline) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
  let best = 0, top = -1;
  for (let i = 0, n = Math.min(v.cells.length, s.x.length); i < n; i++) {
    const code = v.cells[i], x = s.x[i], y = s.y[i];
    if (code >= 255 || code <= top || !(x >= x0 && x <= x1 && y >= y0 && y <= y1)) continue;
    if (pointInOutline(x, y, outline)) { best = i + 1; top = code; }
  }
  return best;
}

// ---- the mixing check's suggestions: the try loop and a new run with the setting
// Nothing changes without a click: *Try it* sets the knob through the Live knobs session (its row shows the change, the
// knob guide's Undo knows it), then registers the trial; Undo puts back the old value the same way; Keep only ends the
// trial. One action at a time (qcActing): a double click never sends twice. A paused run takes the change when it is
// resumed or stepped (the card says so).
let qcActing = false;
/** The Live knobs label of a proseg flag (the option schema), or null. */
function knobLabel(flag) { return options.schema.find(o => o.flag === flag)?.label ?? null; }
/** The plain words for a refusal: the Live knobs session's own reasons in the deck's words (qc-copy.mjs), anything else
 *  (the engine's, the browser's "Failed to fetch") as the general `fallback`. */
const SESSION_REFUSALS = Object.freeze({ 'Already at this value.': 'tryAlready', 'Already beyond this suggestion in that direction.': 'tryBeyond',
  "This setting can't change during a run.": 'tryNotLive', 'Live knobs are available while a run is running or paused.': 'tryNotNow',
  'Select an active run first.': 'tryNotNow', 'Wait for the pending change to this knob.': 'tryBusy', 'Sending change…': 'tryBusy',
  'Wait for the current value before trying this suggestion.': 'tryWaiting', 'Waiting for the current value so Undo can restore it.': 'tryWaiting',
  'There is no suggestion to undo for this knob.': 'undoNothing' });
const plainError = (error, fallback) => !error?.status && SESSION_REFUSALS[error?.message] ? QC_TEXT[SESSION_REFUSALS[error.message]] : fallback;
async function qcAct(work) {
  if (qcActing || !state.run) return;
  qcActing = true; qcClient.changed(true);
  try { await work(state.run.id); }
  finally { qcActing = false; qcClient.changed(true); }
}
/** The click's run is still the one open (a run opened meanwhile is never changed or answered for it). */
const sameRun = run => state.run?.id === run && qcClient.runId === run;
function qcTry(suggestion) {
  return qcAct(async run => {
    let previous;
    try { previous = await knobs.trySetting(suggestion.flag, suggestion.to, suggestion.from); }
    catch (error) { notice(plainError(error, QC_TEXT.tryFailed)); return; }
    if (!sameRun(run)) return;
    const area = Array.isArray(suggestion.areas) ? suggestion.areas[0] ?? null : suggestion.area ?? null;
    try { await qcClient.startTrial({ rule: suggestion.rule, area, flag: suggestion.flag, from: previous?.value ?? suggestion.from ?? null, to: suggestion.to }); }
    catch {
      // The check cannot watch it: the setting goes back as it was (nothing changes without a readout to undo it).
      try { if (sameRun(run)) await knobs.undoSetting(suggestion.flag, previous?.value); notice(QC_TEXT.trialFailedUndone); }
      catch { notice(QC_TEXT.trialFailed); }
    }
  });
}
/** Keep: a trial ends where it is; an "Undo your last change?" card is answered (it goes). */
function qcKeep(item) {
  return qcAct(async run => {
    try { await (isTrial(item) ? qcClient.endTrial('kept', run) : qcClient.answerUndoCard(item, 'kept', run)); }
    catch { notice(QC_TEXT.trialEndFailed); }
  });
}
/** Undo: the old value goes back through the Live knobs session (the trial's `from`; an "Undo your last change?" card's
 *  `to`), then the trial ends as undone (the card is answered), both for the run the click was on. */
function qcUndo(item) {
  return qcAct(async run => {
    const trial = isTrial(item);
    try { await knobs.undoSetting(item.flag, trial ? item.from : item.to); }
    catch (error) { notice(plainError(error, QC_TEXT.undoFailed)); return; }
    if (!sameRun(run)) return;
    try { await (trial ? qcClient.endTrial('undone', run) : qcClient.answerUndoCard(item, 'undone', run)); }
    catch { notice(QC_TEXT.trialEndFailed); }
  });
}
const isTrial = item => !!item && item.rule !== 'U';   // a trial's rule is the card it came from; U is never tried
/** Copy the settings before leaving; there is no link to the run that supplied them. */
function newRunWithSetting(suggestion) {
  return qcAct(async () => {
    const run = state.run;
    if (!run?.sample) return;
    const live = Object.fromEntries(Object.entries(run.live_values || {}).filter(([flag]) => flag !== 'temperature'));
    const values = { ...(run.options || {}), ...live };
    if (typeof suggestion?.flag === 'string' && suggestion.to !== undefined) values[suggestion.flag] = suggestion.to;
    const explicit = typeof suggestion?.flag === 'string' ? [suggestion.flag] : [];
    await leave(async () => {
      const filled = await new Promise(resolve => {
        enterPick(run.sample, { region: run.region, prior: run.prior, values, explicit }, { filled: resolve }).then(resolve, () => resolve(false));
      });
      if (state.mode !== 'pick' || state.sample?.id !== run.sample) return;
      notice(!filled ? QC_TEXT.newRunFailed : filled.priorMissing ? QC_TEXT.newRunPriorMissing : copy('newRunReady', { setting: triedLabel(suggestion ?? {}) || QC_TEXT.missing }));
    }, 'setting');
  });
}
async function proceedToProseg(run) {
  const button = $('proceed'); if (!button || proceedToProseg.busy) return;
  proceedToProseg.busy = true; const done = busy(button, 'Opening…'), id = run.id;
  try {
    await settleExclusions();
    if (state.run?.id !== id) return;
    run = state.run;
    const offer = proceedOffer(run, { api: !!state.session, pending: exclusionsPending() });
    if (!offer.ready) { explain(button, [problem(null, '', offer.reason)]); return; }
    let detail, prefill = proceedPrefill(run);
    for (let attempt = 0; attempt < 2; attempt++) {
      detail = await api.sample(run.sample);
      if (state.run?.id !== id) return;
      if (choiceIndex(priorChoices(detail.prior_options, detail.suggested_prior), prefill.prior) >= 0) break;
      if (attempt) { explain(button, [problem(null, '', PROCEED_TEXT.missing)]); return; }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
    if (!proceedOffer(state.run, { api: !!state.session, pending: exclusionsPending() }).ready || state.run.zarr_save.shapes_key !== run.zarr_save.shapes_key) {
      explain(button, [problem(null, '', PROCEED_TEXT.outdated)]); return;
    }
    await leave(async last => {
      // The engine may have saved under another free name while leaving. Read its final key and the sample again.
      const next = proceedPrefill(last);
      try { detail = await api.sample(last.sample); }
      catch (error) { await enterPick(null); notice(error.message); return false; }
      if (choiceIndex(priorChoices(detail.prior_options, detail.suggested_prior), next.prior) < 0) {
        await enterPick(detail);
        state.prior = { cells:null, nuclei:null }; $('prior-choice').value = '-1'; updatePicker();
        notice(PROCEED_TEXT.missing); return false;
      }
      return enterPick(detail, next);
    }, 'proceed');
  } catch (error) { notice(error.message); }
  finally { proceedToProseg.busy = false; done(); }
}
// ---- "Test a setting here…" in the mixing check and in Go deeper, for a cell -----------------------------------------
// The panel keeps the test's state machine and the stage's corner shows its form, progress and answer (qc-panel.js
// testModel, qc-test-corner.mjs); here its calls reach the engine. The engine keeps the run waiting while it tests;
// Keep, Cancel and Close put it back as it was when the test's runs started (resumed only if it was running then: the
// test's own `was_running`, the engine's newest word first, else what this page saw when it started it; a run waiting at
// its end never was); New run with this leaves the run for a new one with the setting from the start (the engine ends
// the test as the run is left).
const qcTestRunning = new Map();   // test id -> the run was running when this page started that test
/** A setting as Live knobs has it: {label, value (the run's now, null before it is known), min, max}, or null. */
function qcTestSetting(flag) {
  const option = options.schema.find(o => o.flag === flag);
  return option ? { label: option.label ?? null, value: knobs.value(option) ?? null, min: option.min ?? null, max: option.max ?? null } : null;
}
/** Go deeper's "Test a setting here…": the panel's form for the window's cell, in the corner. */
function qcTestOpen(label, name = null) {
  if (state.mode !== 'run' || !(label > 0)) return;
  qcPanel.openTestForm?.({ label, name: name ?? cellName(label) });
}
/** The pool a test of the cell would test and why a test can't answer there (GET /runs/{id}/qc/test/pool for cell
 *  label - 1, the engine waiting up to `wait` seconds for a pool being worked out); none without a run (not a 404:
 *  nothing is gone). Quiet: it sits under /qc. */
function qcTestPreview(label, { wait = 0, signal } = {}) {
  harness?.count('qcTestPoolFetches');
  return state.mode === 'run' && state.run ? api.qcTestPool(state.run.id, label - 1, { wait, signal }) : Promise.reject(new Error('no run'));
}
/** The start: {cell (label - 1), flag, value}, nothing else (the engine fixes the suspects and their neighbours). */
async function qcTestStart(label, flag, value) {
  const run = state.run;
  if (state.mode !== 'run' || !run) throw new Error('no run');
  globalThis.sopaCompanionLog?.add('action', `test a setting around ${cellName(label)}: ${flag} ${value}`);
  const test = await api.qcTestStart(run.id, { cell: label - 1, flag, value });
  const id = (test?.test ?? test)?.id;
  if (id != null) qcTestRunning.set(id, run.state === 'running');
  harness?.count('qcTests');
  return test;
}
/** The run's test (GET), at most once a second (the panel's pace); none without a run (not a 404: nothing is gone). */
function qcTestFetch() {
  harness?.count('qcTestFetches');
  return state.mode === 'run' && state.run ? api.qcTest(state.run.id) : Promise.reject(new Error('no run'));
}
/** Cancel, Close or Keep: the engine forgets the test (DELETE cancels one on its way, waiting or with its runs deleted;
 *  the engine keeps it as cancelled, so a second DELETE dismisses it; one that ended is dismissed; a test gone already
 *  is fine), then the run resumes if it was running when the test's runs started. The engine's word on that is its
 *  newest DELETE reply's (the test as it ended): a test this page last saw waiting may have started its runs since,
 *  pausing a run that sampled on past its end; else the page's record's. Rejects with `plain` words for the panel. */
async function qcTestEnd(test) {
  const run = state.run, fail = text => Object.assign(new Error(text), { plain: text });
  if (state.mode !== 'run' || !run || !test) return;
  const going = ['waiting', 'preparing', 'running', 'comparing'].includes(test.state);
  globalThis.sopaCompanionLog?.add('action', `test a setting: ${going ? 'cancelled' : test.state === 'done' ? 'kept the setting' : 'closed'}`);
  let ended = null;
  for (let times = going ? 2 : 1; times > 0; times--) {
    try { const reply = await api.qcTestCancel(run.id); ended = reply?.test ?? reply ?? ended; }
    catch (error) { if (error?.status === 404) break; throw fail(going ? QC_TEXT.testCancelFailed : QC_TEXT.testAnswerFailed); }
  }
  const said = [ended?.was_running, test.was_running].find(v => typeof v === 'boolean');
  const resume = said ?? qcTestRunning.get(test.id) === true;
  if (resume && state.run?.id === run.id && state.run.state === 'paused') {
    try { const resumed = await api.control(run.id, 'resume'); if (state.run?.id === run.id) { state.run = resumed; renderRun(); qcPanel.setTestRun?.(resumed); } }
    catch { notice(QC_TEXT.testResumeFailed); }
  }
  qcTestRunning.delete(test.id);
}
/** New run with this: the picker filled in as this run had it (its region, starting cells and settings) with the
 *  tested setting at the tested value from the start, as a suggestion's New run (newRunWithSetting); leaving the run
 *  ends its test (the engine cancels or dismisses it as the run is left). */
function qcTestNewRun(test) {
  if (!test || typeof test.flag !== 'string' || !Number.isFinite(test.value)) return Promise.resolve();
  globalThis.sopaCompanionLog?.add('action', `test a setting: new run with ${test.flag} ${test.value}`);
  // The drift switch goes to its option as a tick (the test's 1 = drift off): the options form refuses a number there.
  const tick = value => (test.flag === 'no-diffusion' && Number.isFinite(value) ? value === 1 : value);
  return newRunWithSetting({ flag: test.flag, to: tick(test.value), from: tick(test.current) });
}
/** The test's square on the stage while the test exists (qc-overlay.mjs setTestSquare draws it dashed in
 *  the stage's next frame): the engine's square [x0, y0, x1, y1] µm from the panel's model (test.square), gone with the
 *  test (the corner's {open: false}). */
function qcTestSquare(model) {
  const square = model?.open && Array.isArray(model.test?.square) ? model.test.square : null;
  qcOverlay.setTestSquare?.(square ? { box: square } : null);
}
/** The picker filled in as a run of this sample had it: its region (rectangle, lasso or the full sample), its starting
 *  cells and its options (`values`; `explicit`: the suggested setting, named even at its default), not remembered as
 *  the picker's options. Returns {priorMissing} (the run's starting cells are not offered: "None" shows), or false when
 *  it could not be filled in. */
function prefillPicker(prefill) {
  try { return fillPicker(prefill); } catch { return false; }
}
function fillPicker({ region, prior, values, explicit = [] }) {
  const sample = state.sample;
  const select = $('prior-choice'), choices = state.priorChoices || [];
  const index = choiceIndex(choices, prior), priorMissing = index < 0 && !!(prior?.cells || prior?.nuclei);
  select.value = String(index); state.prior = index >= 0 ? { ...choices[index].prior } : { cells: null, nuclei: null };
  select.title = select.options?.[select.selectedIndex]?.textContent || '';
  if (values) options.setAll(values, { explicit, save: false });
  if (region?.kind === 'box' && Array.isArray(region.box)) state.lastRectangle = { kind: 'box', box: [...region.box] };
  if (region?.kind === 'polygon' && Array.isArray(region.points)) state.lastLasso = { kind: 'polygon', points: region.points.map(p => [...p]) };
  setRegionMode(['box', 'polygon', 'full'].includes(region?.kind) ? region.kind : 'box');
  const advanced = $('picker').querySelector?.('details.advanced');
  if (advanced && values) advanced.open = true;   // the suggested setting is in view
  resetPreview(); previewPrior(); updatePicker();
  return { priorMissing };
}
/** The per-bin layer of the result shown (Colour by: Mixing, zoomed out over a whole sample), fetched once per result
 *  and count (a frame counted again for the hold, a finish or the end is another result: its grid is asked for again);
 *  at frame 0 the start line's own (GET /runs/{id}/qc/start/bins). */
function qcBins() {
  const run = state.run?.id, seq = qcClient.view?.atStart ? 'start' : qcClient.applied?.seq;
  if (!run || seq == null || !qcOverlay.wantsBins()) return;
  const applied = qcClient.applied, count = seq === 'start' ? 'averaged' : applied?.summary?.count ?? applied?.point?.count ?? 'averaged';
  const key = `${run}:${seq}:${qcClient.model}:${count}`;
  if (key === qcBinsKey) return;
  qcBinsKey = key;
  // A later result's layer never stays on an earlier frame while this one's loads (the start line counts as frame 0).
  const order = s => s === 'start' ? 0 : s;
  if (qcBinsSeq != null && order(qcBinsSeq) > order(seq)) { qcBinsSeq = null; if (qcOverlay.setBins(null)) drawOverlay(); }
  const motion = !qcClient.view?.still;
  fetchBins(run, seq).then(grid => {
    if (qcBinsKey !== key || state.run?.id !== run) return;
    qcBinsSeq = grid ? seq : null;
    if (qcOverlay.setBins(grid, { motion })) drawOverlay();
  }).catch(() => {});
}
/** The picker's one line about the opened sample's mixing check, only while the check
 *  is on for new runs (qcNewRuns: the remembered choice): ready, learning (once), waiting for the graphics card, or not
 *  learned yet, when Start will learn it first (a few minutes of GPU 0, once: said before Start, never a surprise).
 *  Nothing otherwise: the check is off by default and a sample is learned only once a user switches it on (the run's
 *  panel says why it cannot run). */
function pickerQc(sample) {
  if (['stain','patches'].includes($('run-kind').value)) { $('picker-qc').hidden = true; return; }
  const t = sample?.eta_s ?? sample?.estimate_s, on = qcNewRuns === true;
  const text = !on ? '' : sample?.state === 'ready' ? QC_TEXT.pickerReady : sample?.state === 'waiting' ? QC_TEXT.pickerWaiting
    : sample?.state === 'learning' ? (Number.isFinite(t) ? copy('pickerLearning', { t: duration(t) }) : QC_TEXT.pickerLearningNoTime)
    : sample?.state === 'off' ? QC_TEXT.pickerLearnFirst : '';
  setText($('picker-qc'), text); setProp($('picker-qc'), 'hidden', !text);
}
function updateQcLegend() {
  const view = qcClient.view, on = state.mode === 'run' && !!state.run && view?.enabled === true, o = stage.options;
  // One row per colour shade; zoomed out, the patches' own key with the ramp's numbers (the overlay's,
  // qc-overlay.mjs); without values the cells are plain grey and the legend says why. No outline key: no area is drawn.
  mixingLegend.update({ mixing: on && (o.cellColorBy === 'mixing' || o.transcriptColorBy === 'mixing'), overview: on && qcOverlay.overview(),
    view3d: !!stage.orbit.enabled, stops: qcOverlay.binStops(), noValues: on && !qcShownCells });
  placeLegend();
}
/** Where the Mixing legend goes while the test corner is open (moved above the corner it met the Mixing chip at the
 *  top left). `corner` the corner's box ({left, top, bottom, w, h} CSS px in the stage) or
 *  null; `size` the legend's own ({w, h}, else LEGEND_PLACE's most); `room` the stage's {width, height}; `avoid` the
 *  other boxes over the stage ({x, y, w, h} from its top left: the HUD, the chip, Go deeper's window). Its own place at
 *  the bottom left when the corner leaves it free (or is closed); else just right of the corner at the stage's bottom,
 *  when it fits there inside the stage and clear of the corner and of every box in `avoid`; else null: hidden while the
 *  corner is open. Returns {left, bottom} (CSS px from the stage's bottom left). */
function legendSpot(corner, size, room, avoid = []) {
  const P = LEGEND_PLACE, W = room?.width || 0, H = room?.height || 0;
  const w = size?.w > 0 ? size.w : P.width, h = size?.h > 0 ? size.h : P.height;
  const at = (left, bottom) => ({ x: left, y: H - bottom - h, w, h });
  const meets = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
  const own = { left: P.left, bottom: P.bottom };
  if (!corner || !(corner.w > 0 && corner.h > 0)) return own;
  const c = { x: corner.left, y: Number.isFinite(corner.top) ? corner.top : H - corner.bottom - corner.h, w: corner.w, h: corner.h };
  if (!meets(at(own.left, own.bottom), c)) return own;
  const right = { left: Math.round(c.x + c.w + P.gap), bottom: P.bottom }, r = at(right.left, right.bottom);
  const inside = r.x + r.w <= W - P.left && r.y >= 0;
  return inside && !meets(r, c) && !avoid.some(b => meets(r, b)) ? right : null;
}
/** The Mixing legend where legendSpot puts it, at once (no motion), and its box in qcBoxes moved with it. Hidden by the
 *  corner it keeps its own `hidden` (MixingLegend.update's) and its size (`off`: hidden by the corner). */
function placeLegend() {
  const node = mixingLegend.root, style = node?.style, place = qcBoxes.get(node);
  if (!style) return;
  const H = stage.cssHeight, avoid = [];
  for (const [other, p] of qcBoxes) {
    if (other === node || other === legendCorner?.node || !other || other.hidden || !p.w || !p.h) continue;
    avoid.push({ x: p.left, y: p.bottom == null ? p.top : H - p.bottom - p.h, w: p.w, h: p.h });
  }
  const spot = legendSpot(legendCorner?.box ?? null, place, { width: stage.cssWidth, height: H }, avoid);
  const left = spot?.left ?? LEGEND_PLACE.left, bottom = spot?.bottom ?? LEGEND_PLACE.bottom, visibility = spot ? '' : 'hidden';
  if (style.left !== `${left}px`) style.left = `${left}px`;
  if (style.bottom !== `${bottom}px`) style.bottom = `${bottom}px`;
  if ((style.visibility ?? '') !== visibility) style.visibility = visibility;
  if (place) Object.assign(place, { left, bottom, off: !spot });
}
if (typeof ResizeObserver === 'function') {
  const watch = (node, place) => {
    if (!node) return;
    qcBoxes.set(node, { ...place, w: 0, h: 0 });
    new ResizeObserver(entries => {
      const box = entries.at(-1)?.borderBoxSize?.[0], rect = entries.at(-1)?.contentRect;
      Object.assign(qcBoxes.get(node), { w: box?.inlineSize ?? rect?.width ?? 0, h: box?.blockSize ?? rect?.height ?? 0 });
      placeLegend();
    }).observe(node);
  };
  watch($('hud'), { left: 12, top: 12 }); watch($('qc-chip'), { left: 12, top: 118 }); watch(mixingLegend.root, { left: 12, bottom: 12 });
}

// ---- cell hover and selection ---------------------------------------------------------------------------------
function cellName(label) { return state.cells[label - 1] || `cell ${label - 1}`; }

function swatch(label) {
  // The same colour the shader computes for this cell.
  const id = label >>> 0;
  let h = Math.imul(id, 2654435761) >>> 0;
  h = (h ^ (h >>> 15)) >>> 0; h = Math.imul(h, 2246822519) >>> 0; h = (h ^ (h >>> 13)) >>> 0;
  const hue = ((id * 0.6180339887) % 1 + (h & 255) / 1400) % 1;
  const sat = 0.52 + 0.33 * ((h >>> 8) & 255) / 255, val = 0.82 + 0.18 * ((h >>> 16) & 255) / 255;
  const k = (n) => { const x = Math.abs(((hue * 6 + n) % 6) - 3) - 1; return val * (1 - sat + sat * Math.min(1, Math.max(0, x))); };
  return `rgb(${[k(0), k(4), k(2)].map((v) => Math.round(v * 255)).join(',')})`;
}
/** A cell of this run in the stage's colour (the test corner's cells): its cluster's under Colour by:
 *  Cluster (the stage's grey for a cell without one), else its own (swatch: the shader's identity colour; Colour by:
 *  Mixing too, whose colours the corner does not show). */
function stageColour(label) {
  const result = analysis.applied;
  if (stage.options.cellColorBy !== 'cluster' || !result) return swatch(label);
  const c = result.clusters?.[label - 1], s = Number.isInteger(c) && c >= 0 ? result.summary?.clusters?.find(x => x.id === c) : null;
  return s ? clusterColorCSS(s.color ?? s.id) : '#66717e';
}

let lastPointer=null;
function clusterLabel(label) {
  const result=analysis.applied,c=result?.clusters[label-1];
  return result?.summary.clusters.find(x=>x.id===c)?.label || 'Unanalysed';
}
/** The counted frames of the run on screen the page has heard of from its run's newest counted frames
 *  (proseg_counts.recent, the 64 newest, with every run event) and, on each watch, from every counted frame it holds
 *  ({ev: "counted"}: a page away while more than 64 were counted): a frame's counts may come after its record did,
 *  which is pushed once, so such a frame stays known once it has left that list (noteCounted). The same for the frames
 *  counted for a pause, the hold, a finish or the end (proseg_counts.sampled, the counted event's `sampled`: the count
 *  proseg saves there, which the engine scores them on): a frame counted by a token may be counted again for the hold
 *  after its record came. */
let countedSeqs = new Set(), sampledSeqs = new Set();
function noteCounted(run) {
  const recent = run?.proseg_counts?.recent, sampled = run?.proseg_counts?.sampled;
  if (Array.isArray(recent)) for (const seq of recent) if (Number.isInteger(seq)) countedSeqs.add(seq);
  if (Array.isArray(sampled)) for (const seq of sampled) if (Number.isInteger(seq)) sampledSeqs.add(seq);
}
/** Whether frame `seq` of the run on screen has proseg's own count of each cell: its record names its counted file, or
 *  the run's newest counted frames list it, now or since the run was opened. */
function frameCounted(seq) {
  return !!state.frames[seq]?.counted_file || countedSeqs.has(seq)
    || (Array.isArray(state.run?.proseg_counts?.recent) && state.run.proseg_counts.recent.includes(seq));
}
/** Whether frame `seq` of the run on screen was counted for a pause, the hold, a finish or the end (its record's count
 *  "sampled", or the run's sampled frames list it, now or since the run was opened): its numbers read the count proseg
 *  saves there. */
function frameSampled(seq) {
  return state.frames[seq]?.count === 'sampled' || sampledSeqs.has(seq)
    || (Array.isArray(state.run?.proseg_counts?.sampled) && state.run.proseg_counts.sampled.includes(seq));
}
/** A cell's line in the tooltip and the card: its area, its count in proseg's count (the counted file's on counted frames,
 *  else the frame's non-background count, with the same words) and what proseg sets aside as background, its voxels and
 *  its cluster (cellstats.mjs readoutText). */
function cellReadout(label) {
  const stats=cellStats.current,h=state.frames[state.current],i=label-1;
  if(stats?.seq!==state.current||i>=stats.area.length)return 'Statistics unavailable';
  return readoutText(stats,i,{voxel:h.voxel,cluster:clusterLabel(label)});
}
/** The card's readout line's tooltip: which count it reads on the frame on screen (cellstats.mjs readoutTip); '' while
 *  its statistics are not there. */
function cellReadoutTip(label) {
  const stats=cellStats.current;
  return stats?.seq===state.current&&label-1<stats.area.length?readoutTip(stats):'';
}
function updateTooltip(event) {
  lastPointer={offsetX:event.offsetX,offsetY:event.offsetY};
  const tip=$('tooltip');
  if(state.mode!=='run'||!stage.frame){setProp(tip,'hidden',true);return;}
  const label=stage.pickLabel(event.offsetX,event.offsetY);
  if(label!==state.hover||stage.options.hover!==label){state.hover=label;stage.set({hover:label});analysis.tissueHover(label);}
  const xenium=masks.pick(...stage.toWorld(event.offsetX,event.offsetY));
  // The mixing check: the cell's own line (its value in the result shown); no area line (none is drawn).
  const code=label&&qcClient.view?.enabled?qcClient.cellValue(label):null;
  // Go deeper: a source neighbour under the pointer (its cell or its arrow) lifts its row and lights its outline; the
  // tooltip says how much of the window's cell seems to come from it (or that it looks like the same kind).
  deeper.hover(deeperArrows.hit(event.offsetX,event.offsetY)||label);
  const deepLine=label?deeper.tooltipLine(label):null;
  if(!label&&!xenium.length){setProp(tip,'hidden',true);return;}
  // Built when it shows another cell; a new frame's numbers (or statistics arriving) under a resting pointer only
  // change its text, in place. Its size is read only when the text changed, not for every pointer move, and the stage's
  // size is cached by the stage.
  const lines=[...(label?[cellName(label),cellReadout(label)]:[]),...(code!=null?[cellLine(code)]:[]),...(deepLine?[deepLine]:[]),...(xenium.length?[`Xenium: ${xenium.join(', ')}`]:[])];
  // The mixing check's own lines (the cell's, Go deeper's) are its copy: the harness checks them for banned words.
  const qcLines=new Set([...(code!=null?[lines.indexOf(cellLine(code))]:[]),...(deepLine?[lines.indexOf(deepLine)]:[])]);
  const subject=`${label}:${xenium.join(',')}:${code!=null}:${deepLine??''}`;
  let changed=tip.hidden||shown.tip?.subject!==subject;
  if(changed){
    tip.replaceChildren(...lines.map((text,i)=>{const line=document.createElement(label&&!i?'b':'div');line.textContent=text;if(qcLines.has(i))line.dataset.qcCopy='';return line;}));
    tip.hidden=false;setProp(tip.style,'maxWidth','360px');
  } else lines.forEach((text,i)=>{const line=tip.children[i];if(line.textContent!==text){line.textContent=text;changed=true;}});
  if(changed)shown.tip={subject,size:[tip.offsetWidth,tip.offsetHeight]};
  const [w,h]=shown.tip.size;
  setProp(tip.style,'left',`${Math.round(Math.max(8,Math.min(event.offsetX+16,stage.cssWidth-w-8)))}px`);
  setProp(tip.style,'top',`${Math.round(Math.max(8,Math.min(event.offsetY+16,stage.cssHeight-h-8)))}px`);
}
let cellHistory=null,historyController=null,historyKey='',historyCell='',historyPending=false,historyError='',historyAt=-Infinity,historyTimer=null;
function resetCellHistory(){historyController?.abort();clearTimeout(historyTimer);historyTimer=null;cellHistory=null;historyKey='';historyCell='';historyPending=false;historyError='';}
function loadCellHistory(label) {
  const cell=`${state.run?.id}:${label}`,key=`${cell}:${state.frames.length}`;
  if(historyKey===key||(historyCell===cell&&historyPending))return;
  // The history spans the whole run, so every new frame extends it: while frames stream, refetch it every HISTORY_MS.
  const wait=historyCell===cell?historyAt+HISTORY_MS-performance.now():0;
  if(wait>0){if(historyTimer==null)historyTimer=setTimeout(()=>{historyTimer=null;later.now();},wait);return;}
  if(historyCell!==cell)cellHistory=null;
  historyController?.abort();const c=historyController=new AbortController();historyCell=cell;historyKey=key;historyPending=true;historyError='';historyAt=performance.now();
  fetch(`/api/v1/runs/${encodeURIComponent(state.run.id)}/cellhistory?cell=${label-1}&points=400`,{signal:c.signal})
    .then(r=>{if(!r.ok)throw Error(`History: HTTP ${r.status}`);return r.json();})
    .then(data=>{if(!c.signal.aborted&&historyKey===key)cellHistory=data;})
    .catch(e=>{if(e.name!=='AbortError'&&historyKey===key)historyError=e.message;})
    .finally(()=>{if(!c.signal.aborted&&historyKey===key){historyPending=false;later.now();}});
}
function updateCellCard() {
  analysis.selection(state.selected);
  deeper.select(state.mode==='run'?state.selected:0);deeperRings();   // Go deeper follows the selection (also collapsed)
  updateCellRow();   // the mixing check's row: the cell's mixing line and "Go deeper ›"
  const card=$('cell-card'),label=state.selected;
  // No run (a leave lets it go before the picker resets the panels, which come back here): no card.
  if(!label||state.mode!=='run'||!state.run){setProp(card,'hidden',true);resetCellHistory();if(shown.card)sparkSizes?.disconnect();shown.card=null;return;}
  if(!panelOpen('now-panel'))return;   // collapsed: refreshed when the panel opens
  loadCellHistory(label);setProp(card,'hidden',false);
  // The structure changes with the cell and its history's state; a new frame only updates the numbers and sparklines.
  const history=cellHistory,note=history?'':historyPending?'Loading cell history…':historyError||'History unavailable';
  // The card is built once per cell: its name, its readout and its history. The history part (a note while it loads,
  // then the two sparklines) sits in a box of its own and is rebuilt alone when the history lands.
  const structure=`${state.run.id}:${label}`;
  let parts=shown.card;
  if(parts?.structure!==structure){
    card.replaceChildren();sparkSizes?.disconnect();
    const header=document.createElement('header'),dot=document.createElement('i'),name=document.createElement('span');
    dot.style.background=swatch(label);header.append(dot,name);
    const line=document.createElement('div');line.className='muted';
    const tail=document.createElement('div');tail.className='cell-card-history';
    card.append(header,line,tail);
    parts=shown.card={structure,name,line,tail,note:null,charts:[]};
  }
  if(parts.note!==note){
    for(const c of parts.charts)sparkSizes?.unobserve?.(c.canvas);
    parts.note=note;parts.charts=[];parts.history=null;parts.marks=null;
    const nodes=[];
    if(note){const small=document.createElement('small');small.textContent=note;nodes.push(small);}
    else for(const [key,title,color] of [['area_um2','Area (µm²)',swatch(label)],['transcripts','Assigned transcripts','#7bd9ea']]){
      const titleEl=document.createElement('small');titleEl.textContent=title+' · whole run';const canvas=document.createElement('canvas');nodes.push(titleEl,canvas);
      parts.charts.push({canvas,key,color});sparkSizes?.observe(canvas);
    }
    parts.tail.replaceChildren(...nodes);
  }
  setText(parts.name,cellName(label));setText(parts.line,cellReadout(label));setProp(parts.line,'title',cellReadoutTip(label));
  const marks=`${state.frames.at(-1)?.iter}:${state.frames[state.current]?.iter}`;
  if(history&&(parts.history!==history||parts.marks!==marks)){
    parts.history=history;parts.marks=marks;
    for(const c of parts.charts)sparkline(c.canvas,history.iter,history[c.key],c.color);
  }
}
/** The mixing check's row for the selected cell (qc-panel.js setCell): "Go deeper ›" and its mixing line, for any
 *  selected cell of a run on a sample, also while the check is off (the window then offers the check's switch). */
function updateCellRow() {
  const label=state.mode==='run'&&state.run?.sample?state.selected:0;
  if(!label){qcPanel.setCell(null);return;}
  if(!panelOpen('qc-panel'))return;   // collapsed: refreshed when the panel opens (its values are asked for only then)
  // The mixing check: this cell now and in the starting cells (its values load on demand). While the panel shows its
  // headline, the meaning line under it says what the number is not: the cell's line keeps that in its tooltip only.
  const v=qcClient.view,code=v?.enabled?qcClient.cellValue(label):null,note=!qcPanel.headlineShown;
  const line=code==null?'':cardLine(code,qcClient.startValue(label),v.startSource,{note});
  // While the result shown has its own values still on the way (they are withheld, never an earlier result's), a line
  // in the row keeps its place, invisible, so nothing under it moves up and back when they land; only where
  // they come (a result or the start line shown), never while the sample waits or learns (no stand-in: an empty box).
  const loading=!line&&!!v?.enabled&&['live','paused','scrubbing','finished'].includes(v.status)&&!!(v.summary||v.atStart&&v.start);
  qcPanel.setCell({label,name:cellName(label),color:swatch(label),line,tip:note||line===QC_TEXT.cellUnjudged?'':QC_TEXT.cardNote,loading});
}
function sparkline(canvas,iters,values,color) {
  if(!canvas.size?.[0])canvas.size=[canvas.clientWidth,canvas.clientHeight];   // measured once per layout, not per frame
  const ratio=window.devicePixelRatio||1,[w,h]=canvas.size;
  if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);}
  const g=canvas.getContext('2d');g.setTransform(ratio,0,0,ratio,0,0);g.clearRect(0,0,w,h);
  if(!iters?.length||!values?.length)return;
  const total=Math.max(1,state.frames.at(-1)?.iter||iters.at(-1)),max=Math.max(1,...values);
  g.strokeStyle=color;g.lineWidth=1.6;g.beginPath();
  iters.forEach((iter,i)=>{const x=iter/total*(w-4)+2,y=h-3-values[i]/max*(h-8);if(i)g.lineTo(x,y);else g.moveTo(x,y);});g.stroke();
  const x=(state.frames[state.current]?.iter||0)/total*(w-4)+2;g.fillStyle='#fff';g.fillRect(x,0,1,h);
}
/** The window was resized (or zoomed: a new pixel ratio): sparklines measure their canvases again. Width changes
 * within the page reach them through sparkSizes. */
function resizeCellCard() {
  if(!shown.card)return;
  for(const c of shown.card.charts)c.canvas.size=null;
  shown.card.history=null;updateCellCard();
}
// ---- Go deeper: the page's side of the window (qc-deeper.js) --------------------------------------------------------
/** What the window reads of the page, and the open test corner's box (TestCorner.box: {left, top, bottom, w,
 *  h} CSS px in the stage, or null), which its camera move keeps clear of as it does of the window; whether a test of a
 *  setting can be asked for its cell (`testable`: the mixing check's testable() for the selected cell, else the
 *  window's own), and how the start held a cell's points (`startHold`: the start layers of a detail run, PriorLayers
 *  startHold on the run's depth grid; null where there are none). `startKind`: how the run started (the engine's
 *  start_kind: "cells" has a start line of proseg's count at iteration 0, "paired" and "nuclei" none). `buttons`: which run
 *  buttons a press makes work now (run-buttons.mjs: its words name one only then). `counted` and `sampled`: whether a
 *  frame has proseg's count of its own (frameCounted), and whether it reads the count proseg saves there (frameSampled:
 *  an answer held on its averaged count from before it was counted again is asked for again). */
function deeperContext() {
  const run = state.mode === 'run' ? state.run : null;
  return { run, frames: state.frames, seq: state.current, live: !!(state.live && run?.state === 'running'), selected: state.selected,
    buttons: runButtons(run, state.frames, { prompt: finishingPrompt }),
    cells: state.cells, qc: run ? qcClient.view : null, startSource: qcClient.startSource, startKind: qcClient.startKind ?? null, calm: qcCalm, view3d: !!stage.orbit.enabled,
    stage: { width: stage.cssWidth, height: stage.cssHeight }, counted: seq => frameCounted(seq), sampled: seq => frameSampled(seq),
    prosegFix: state.health?.proseg_fix ?? null, notebook: !!state.session,
    cellStats: (label, seq) => deeperCellStats(label, seq), cornerBox: () => testCorner.box(),
    testable: !!run && qcPanel.testable?.(deeperTestCell()) === true,
    startHold: (points, label) => (run ? priors.startHold?.(points, label, run.info?.z ?? null) : null) ?? null };
}
/** The cell Go deeper's "Test a setting here…" tests: the selected one, else the window's own (its focus, once it
 *  showed one: a window left open on its cell). */
function deeperTestCell() {
  try { return state.selected || deeperShown?.cell || 0; } catch { return state.selected || 0; }   // (before the page set up)
}
/** The live cell group of `label` (Clusters & UMAP): {label, color (its palette index), iter}, or null. */
function deeperGroup(label) {
  const result = analysis.applied, c = result?.clusters?.[label - 1];
  if (!(c >= 0)) return null;
  const cluster = result.summary?.clusters?.find(x => x.id === c);
  const text = cluster ? cluster.label || (cluster.top || []).slice(0, 3).map(g => g.gene).join(' · ') : '';
  return text ? { label: text, color: cluster.color ?? c, iter: result.summary?.iter ?? null } : null;
}
/** A cell's centre (µm) in the statistics of the frame on screen, or null. */
function deeperCentre(label) {
  const s = cellStats.current, i = label - 1;
  if (!s || s.seq !== state.current || !s.x || i < 0 || i >= s.x.length) return null;
  const x = s.x[i], y = s.y[i];
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
}
/** Proseg's own count of a cell in counted frame `seq` (Go deeper's basis tooltip, beside the count of the result's
 *  frame: what it reads as background is assigned - kept): {assigned, kept} from that frame's statistics held here (the
 *  frame on screen's, or one in the cache), not rounded: on a frame counted for a pause, the hold, a finish or the end
 *  (frameSampled) the count proseg saves there, the frame's own draw (cells.zst foreground: what keptOf reads once the
 *  statistics hold no counted file, Σf), as Go deeper's answer of that frame reads; else the frame's counted file,
 *  averaged over its random background draw, or null while they are not in or came without that file (one draw would
 *  not be the averaged count). */
function deeperCellStats(label, seq = state.current) {
  const s = seq === state.current && cellStats.current?.seq === seq ? cellStats.current : cellStats.cache?.get(seq), i = label - 1;
  if (!s || s.seq !== seq || i < 0 || i >= (s.assigned?.length ?? 0)) return null;
  if (frameSampled(seq)) return i < (s.foreground?.length ?? 0) ? { assigned: s.assigned[i], kept: s.foreground[i] } : null;
  if (!s.kept || i >= s.kept.length) return null;
  return { assigned: s.assigned[i], kept: s.kept[i] };
}
/** A cell's diameter (µm) from its area in the frame on screen (10 µm when not known). */
function deeperDiameter(label) {
  const s = cellStats.current, h = state.frames[state.current], i = label - 1;
  const area = s?.seq === state.current && i >= 0 && i < (s.area?.length ?? 0) ? s.area[i] * (h?.voxel ?? 1) ** 2 : 0;
  return area > 0 ? 2 * Math.sqrt(area / Math.PI) : 10;
}
let deeperShown = null;   // the window's focus as last handed to the stage, the overlay and the UMAP
/** The window's focus: the outlines on the stage (its cell, its sources, its same-kind neighbours), the cells that stay
 *  lit while every other one greys out (lit, dim), the arrows on the overlay, the rings in the UMAP. */
function deeperFocus(focus) {
  deeperShown = focus ?? null;
  stage.setFocus?.(focus ? { cell: focus.cell, sources: focus.sources.map(s => [s.label, s.share]), same: focus.same, hot: focus.hot,
    lit: focus.lit, dim: focus.dim, motion: !!focus.motion } : null);
  deeperRings();
  deeperGeometry(!!focus?.motion, !!focus?.apply);
}
/** The arrows' ends from the statistics of the frame on screen: with each update of the window (at most one a second
 *  live), or once those statistics arrive when they were missing. Changed, they are drawn at once, but for the window's
 *  own apply (`apply`: a result painted), whose arrows are drawn with the page's next paint batch (overlaySoon): a row
 *  or an arrow under the pointer, a pin and Show on the image never wait for a snapshot on its way. */
function deeperGeometry(motion = false, apply = false) {
  const f = deeperSpillOn ? null : deeperShown;   // the dots on the image set the arrows aside (they would cover them)
  const changed = deeperArrows.set(f ? { cell: f.cell, target: deeperCentre(f.cell), diameter: deeperDiameter(f.cell), hot: f.hot,
    sources: f.sources.map(s => ({ ...s, from: deeperCentre(s.label) })) } : null, { motion });
  if (changed && state.mode === 'run') { if (apply) overlaySoon(); else drawOverlay(); }
}
/** The overlay drawn again with the page's next paint batch, once however often it is asked for before then, and not
 *  at all when something draws it first (drawOverlay clears the wish): the window's apply stays light (P5). */
function overlaySoon() {
  overlayWanted = true;
  paint.mark('overlay', () => { if (overlayWanted) drawOverlay(); });
}
/** The UMAP's rings: the window's cell (when the selection is not on it) and its sources. */
function deeperRings() {
  const f = deeperShown;
  analysis.view?.ringCells?.(f ? [...(f.cell !== state.selected ? [[f.cell - 1, 1]] : []), ...f.sources.slice(0, 5).map(s => [s.label - 1, 2])] : null);
}
/** A neighbour's "›" or "Back to …" in the window: the selection moves with it. */
function deeperSelect(label) {
  if (state.mode !== 'run' || !(label > 0)) return;
  state.selected = label; stage.set({ selected: label }); updateCellCard();
}
/** Find: fly to the window's cell (the stage's 420 ms tween). */
function deeperFind(label) {
  const c = deeperCentre(label);
  if (!c) return;
  if (state.run?.mode === 'tiled') tiled.overview = false;
  const side = Math.max(40, deeperDiameter(label) * 6);
  stage.fit({ x0: c[0] - side / 2, y0: c[1] - side / 2, w: side, h: side }, true, 0);
}
let deeperSpillOn = false;   // Go deeper's "Show on the image" is on (its dots on the stage)
/** "Show on the image": the dots and the fading on the stage (stage.setSpill, 2-D), the arrows set aside meanwhile; the
 *  suspects' glow lives on that view, so it goes with the dots (the window turns it off itself too: onGlow(null)). */
function deeperSpill(spill) {
  const was = deeperSpillOn; deeperSpillOn = !!spill;
  stage.setSpill?.(spill ?? null);
  if (!spill) { stage.setGlow?.(null); deeperStartOutline(null); }
  if (was !== deeperSpillOn) deeperGeometry();
}
let deeperStart = null;   // Go deeper's cell whose starting outline is asked for while its Show is on ({cell, name}), or null
/** Its cell's starting outline, dashed white over the stage while Show is on (stage.setStartOutline,
 *  drawn by qc-overlay.mjs): the prior's polygon of the cell's name, else its starting labels traced (prior-layers.js
 *  startOutline); asked again from drawOverlay until the prior's vectors or labels are in. */
function deeperStartOutline(start = deeperStart) {
  deeperStart = start && start.cell > 0 ? { cell: start.cell, name: start.name ?? cellName(start.cell) } : null;
  stage.setStartOutline?.(deeperStart ? priors.startOutline(deeperStart.name, deeperStart.cell) : null);
}
/** Its one camera move to the cell and its sources (the stage's 420 ms tween; at once with reduced or calm motion); in
 *  3-D `view` {box (µm), free (CSS px)}: the stage frames the box in the part of the stage the window leaves free,
 *  turning nothing (stage.fit). */
function deeperFly(rect, animate = true, view = null) {
  if (state.mode !== 'run' || !rect) return;
  if (state.run?.mode === 'tiled') tiled.overview = false;
  stage.fit(rect, animate, 0, view);
}
/** The window's box over the stage: the Mixing legend keeps clear of it beside the test corner (null: closed). */
function deeperBox(box) {
  if (box) qcBoxes.set(deeper.root, { left: box.left, top: box.top, w: box.w, h: box.h }); else qcBoxes.delete(deeper.root);
  placeLegend();
}
function updateActiveCells() {
  let list=$('active-cells');
  if(!list){list=document.createElement('div');list.id='active-cells';$('cell-card').after(list);}
  const run=state.mode==='run'?(state.run?.id??null):null;
  setProp(list,'hidden',!run);
  if(shown.activeRun!==run){
    shown.activeRun=run;shown.activeStats=shown.activeNames=null;shown.active='';
    const title=document.createElement('strong');title.textContent='Most active cells';list.replaceChildren(...(run?[title]:[]));
  }
  if(!run||!panelOpen('now-panel'))return;   // collapsed: refreshed when the panel opens
  // The shown list stays until this frame's statistics arrive (no flicker), and each statistics set is ranked once
  // (this runs several times per frame). The buttons are reused: a new frame usually only changes their text.
  const stats=cellStats.current;
  if(stats?.seq!==state.current){
    // This frame's statistics could not be loaded: an older frame's ranking must not stand for it.
    if(cellStats.failed===state.current&&shown.active){shown.active='';shown.activeStats=null;list.replaceChildren(list.children[0]);}
    return;
  }
  if(stats===shown.activeStats&&state.cells===shown.activeNames)return;
  shown.activeStats=stats;shown.activeNames=state.cells;
  const top=mostActive(stats),key=JSON.stringify(top.map(({id,changed})=>[id,changed,cellName(id+1)]));
  if(key===shown.active)return;
  shown.active=key;
  let buttons=[...list.children].slice(1);
  if(buttons.length!==top.length){
    const title=document.createElement('strong');title.textContent='Most active cells';list.replaceChildren(title);
    buttons=top.map(()=>{
      const button=document.createElement('button');button.type='button';button.style.cssText='display:block;width:100%;text-align:left;margin-top:3px';
      button.onclick=()=>analysis.select(button.cell);list.append(button);return button;
    });
  }
  top.forEach(({id,changed},i)=>{buttons[i].cell=id;setText(buttons[i],`${cellName(id+1)} · ${fmt.format(changed)} changed`);});
}

// ---- region picking -------------------------------------------------------------------------------------------
function setSample(sample) {
  state.sample = sample;
  // In a notebook's session the header names the bound sample everywhere, never a path (its view may be a scratch copy).
  const header = state.session ? sampleHeader(state.session) : null;
  $('sample-name').textContent = header ? sample?.name || header.name : sample?.name || '';
  $('sample-name').title = header ? header.title : sample?.path || '';
  if (sample) clearExplanation($('new-run'));   // a press without a sample said to open one: it is open now
}

function renderRecent() {
  const list = $('recent-samples'); list.replaceChildren();
  if (!state.samples.length) { const note = document.createElement('p'); note.className = 'muted small'; note.textContent = 'No recent samples yet.'; list.append(note); }
  for (const sample of state.samples) {
    const row = document.createElement('div'); row.className = 'recent-item';
    const open = document.createElement('button'); open.type = 'button'; open.className = 'recent-open';
    const name = document.createElement('strong'), path = document.createElement('small'); name.textContent = sample.name; path.textContent = sample.path;
    open.append(name, path); open.onclick = () => sampleBrowser.show(sample.path);
    const forget = document.createElement('button'); forget.type = 'button'; forget.className = 'recent-forget'; forget.textContent = '×';
    forget.title = 'Forget this sample (files are kept)'; forget.setAttribute('aria-label', `Forget ${sample.name}`);
    forget.onclick = async () => {
      forget.disabled = true;
      try { await api.forgetSample(sample.id); state.samples = state.samples.filter(s => s.id !== sample.id); renderRecent(); }
      catch (error) { notice(error.message); forget.disabled = false; }
    };
    row.append(open, forget); list.append(row);
  }
}

// ---- the first screen's two tools (shell.mjs) ------------------------------------------------------
/** The tool the page is in: 'stain', 'proseg', or null on the first screen's two cards (no tool is remembered: a bare
 *  reload lands there). The picker's run type (#run-kind, the one record its readers and the harness use) follows it:
 *  Proseg Live runs proseg; Stain segmentation keeps its own two (segment, or make Training patches while that tool is
 *  open). Returns whether the run type changed (the picker then shows that tool's fields). */
function setTool(tool) {
  state.tool = Object.hasOwn(TOOLS, tool ?? '') ? tool : null;
  const changed = !!state.tool && toolOf($('run-kind').value) !== state.tool;
  if (changed) { $('run-kind').value = state.tool === 'stain' ? 'stain' : 'proseg'; updateRunKind(); }
  renderTool();
  return changed;
}
/** The header's tool name, and the first screen: the two cards, or the chosen tool's Open sample… and Recent. On the
 *  cards the header's own Open sample… and New run wait for a tool to be chosen. */
function renderTool() {
  // An engine that did not answer the page's first requests (boot), or a notebook's session that ended: the first screen
  // says so, in place of the tools.
  const down = !!state.engineDown, api = !!state.session, tool = down ? null : TOOLS[state.tool] ?? null;
  // A notebook's session: the two cards open the bound sample (no Open sample…, Recent or ‹ Both tools).
  const cards = state.mode === 'empty' && (!tool || api);
  setProp($('engine-down'), 'hidden', !down);
  setText($('tool-name'), tool?.name ?? ''); setProp($('tool-name'), 'hidden', !tool);
  setProp($('tool-choice'), 'hidden', !cards || down); setProp($('tool-start'), 'hidden', !tool || api);
  setText($('tool-title'), tool?.name ?? ''); setText($('tool-help'), tool?.open ?? '');
  setText($('tool-choose'), api ? chooseText(state.session) : SHELL_TEXT.choose);
  // A notebook's SpatialData loading: the static spinner (nothing animates by itself), its words and, while it is
  // prepared and its size known, a bar of the part done; a failure in their place, with no spinner.
  const status = api ? sampleStatus(state.session) : null, loading = !!status && status.state !== 'failed';
  setText($('session-status-text'), status?.text ?? ''); setProp($('session-status'), 'hidden', !status);
  if (status) $('session-status').dataset.state = status.state;
  setProp($('session-spinner'), 'hidden', !loading);
  setProp($('session-progress'), 'hidden', !(status?.fraction >= 0));
  if (status?.fraction >= 0) setProp($('session-progress'), 'value', status.fraction);
  // A card pressed meanwhile waits for it (openSessionTool): its spinner and words, until setSession opens it.
  for (const name of Object.keys(TOOLS)) {
    const waiting = loading && state.sessionWait === name, card = $(`tool-${name}`);
    if (waiting) setAttr(card, 'aria-busy', 'true'); else card.removeAttribute('aria-busy');
    setProp($(`tool-${name}-wait`), 'hidden', !waiting);
    setText($(`tool-${name}-wait-text`), waiting ? waitingText(state.session) : '');
  }
  for (const id of ['tool-back', 'empty-open-sample', 'recent']) setProp($(id), 'hidden', api);
  setProp($('open-sample'), 'hidden', cards || api); setProp($('new-run'), 'hidden', cards);
  if (cards) clearExplanation($('new-run'));   // what a press said beside it goes with it
  // Compare is Proseg Live's (a run's cells read by the mixing check): not on the cards nor in Stain segmentation.
  setProp($('compare-button'), 'hidden', state.tool !== 'proseg');
  if (state.tool !== 'proseg') clearExplanation($('compare-button'));
}
/** A card of the first screen (null: back to both): that tool's Open sample… and Recent; a sample opened there enters
 *  the tool's picker. */
function chooseTool(tool) {
  globalThis.sopaCompanionLog?.add('action', tool ? `tool: ${TOOLS[tool].name}` : 'tool: back to both');
  if (state.session && tool) { openSessionTool(tool); return; }
  setTool(tool);
  $(state.tool ? 'empty-open-sample' : 'tool-stain').focus?.();
}
/** A card in a notebook's session: that tool's picker on the bound sample. While the sample is still prepared or opened
 *  the card waits for it, loading (renderTool), and setSession opens it once the sample is ready; for a sample that could
 *  not be opened, or a tool it cannot use, the press says why (never greyed out). */
function openSessionTool(tool) {
  const session = state.session, button = $(`tool-${tool}`), status = sampleStatus(session);
  if (status?.state === 'failed') { explain(button, [problem(null, '', status.text)]); return Promise.resolve(false); }
  if (status) {
    for (const name of Object.keys(TOOLS)) clearExplanation($(`tool-${name}`));
    state.sessionWait = tool; renderTool(); return Promise.resolve(false);
  }
  if (session.sample.tools.length && !session.sample.tools.includes(tool)) {
    explain(button, [problem(null, '', tool === 'proseg' ? NOTE_TEXT.imageOnly : NOTE_TEXT.noImage)]); return Promise.resolve(false);
  }
  clearExplanation(button); setTool(tool);
  return enterPick(session.sample.id);
}
/** The brand leaves the run before returning to the two tools. */
function goHome() {
  globalThis.sopaCompanionLog?.add('action', 'back to the first screen');
  return leave(() => { state.tool = null; return enterPick(null); }, 'home');
}

/** The run picker for `sample` (an id or its detail). `prefill`: a run of this sample to fill it in from ("New run with
 *  this setting": {region, prior, values}), applied in the same step that first lets Start go ahead, before
 *  the overview image loads; returns what prefillPicker did (and calls `filled` with it at once, before the image).
 *  Without one, the remembered options come back if a fill-in was left unstarted. */
async function enterPick(sample, prefill = null, { filled: onFilled = null } = {}) {
  const ticket = ++opening, sampleId = typeof sample === 'string' ? sample : sample?.id;
  state.proceedNotes = [];
  // Compare goes with the run it was over, and what a press of its button said.
  compareView.close(false); cover(false, false); clearExplanation($('compare-button'));
  if (!prefill?.values) options.restore?.();
  pickerSettings.sample=null;pickerSettings.refresh(null);$('stain-settings-report').replaceChildren();
  resetRequests(); resetPreview(); stopReplay();
  clearPatchPreview(); stainView.reset(); stainPatches.reset(); stainExclusions.reset(); tissueView.update(null,null,false); tiled.reset(); layerPanel.runKind = null;
  state.overviewImage = null;
  // The picker of the tool chosen (a run's own once it was opened); "New run with this setting" is a proseg run.
  if (prefill) state.tool = 'proseg';
  if (sampleId) setTool(state.tool || toolOf($('run-kind').value));
  analysis.reset(); resetQc(null); masks.reset(); cellStats.reset(); countedSeqs = new Set(); sampledSeqs = new Set(); resetCellHistory(); lastPointer = null;
  pickerLoading = !!sampleId;
  setSample(null);
  dropLasso(); state.region = state.lastRectangle = state.lastLasso = null;
  state.prior = { cells: null, nuclei: null };
  state.mode = sampleId ? 'pick' : 'empty'; state.run = null; state.frames = []; state.current = -1;
  clearFrames();
  knobs.render(null, options.schema);
  setView(false, false); transcripts.reset(); priors.reset(); priors.segments.setSample(null); layerPanel.setChannels([]);
  stage.info = null; stage.setImage(null, []); stage.setFrame(null, false); stage.set({ hover: 0, selected: 0 });
  $('prior-hint').hidden = true;
  timeline.setFrames([]); timeline.setPlan([]); timeline.setCurrent(-1);
  history.replaceState(null, '', location.pathname);
  $('picker-error').hidden = true; clearExplanation($('start'));
  for (const id of ['picker-name', 'picker-sample-info', 'picker-size', 'picker-area', 'picker-mode', 'prior-preview-note']) $(id).textContent = '';
  renderNotes($('picker-notes'), []);
  $('prior-choice').replaceChildren();
  $('picker-hint').textContent = sampleId ? 'Loading the sample…' : '';
  layout(); updateActiveCells(); pickerQc(null);
  if (!sampleId) return false;
  const valid = () => ticket === opening && state.mode === 'pick';
  let filled = false;
  // The sample's mixing check: one line while the check is on for new runs (as shipped, opening a sample learns
  // nothing), kept up by its events. The engine's answer says what new runs start with (`new_runs`, its remembered
  // choice): newer than anything this page heard before, and the only word a freshly loaded page has.
  api.sampleQc(sampleId).then(qc => {
    if (typeof qc?.new_runs === 'boolean') qcNewRuns = qc.new_runs;
    if (valid()) pickerQc(qc);
  }).catch(() => {});
  try {
    const detail = typeof sample === 'object' && sample.extent ? sample : await api.sample(sampleId);
    if (!valid()) return false;
    setSample({ ...detail, id: sampleId });
    priors.segments.setSample(detail.kind === 'zarr' ? sampleId : null);   // its segmentation layers, in the picker too
    // A sample opens in a tool it offers: one without transcripts (images only) in Stain segmentation, whichever tool it
    // was opened from. What the open found (transcripts in pixels, several transcript elements, images left out, z
    // planes, images only: the sample detail's `notes`) is one line each under its name, the rest in an ⓘ.
    const tools = sampleTools(detail, Object.keys(TOOLS));
    if (!tools.includes(state.tool)) setTool(tools[0]);
    renderNotes($('picker-notes'), sampleNotes(detail));
    fillStainImages(detail);
    if (['stain','patches'].includes($('run-kind').value)) loadStainPreview();
    $('picker-name').textContent = detail.name;
    $('picker-sample-info').textContent = `${detail.transcripts == null ? NOTE_TEXT.noTranscripts : `${fmt.format(detail.transcripts)} transcripts`} · ${detail.shapes?.length || 0} shapes layers`;
    {
      // One prior (the user's pick among everything the sample offers); the suggested one first selected.
      const select = $('prior-choice'), choices = state.priorChoices = priorChoices(detail.prior_options, detail.suggested_prior);
      choices.forEach((choice, i) => select.append(new Option(choice.label, String(i))));
      const index = Math.max(0, choiceIndex(choices, detail.suggested_prior));
      select.value = String(choices.length ? index : -1);
      state.prior = choices.length ? { ...choices[index].prior } : { cells: null, nuclei: null };
      select.title = select.options?.[select.selectedIndex]?.textContent || '';   // the whole layer name
    }
    $('show-prior').checked = true;
    const [x0, y0, x1, y1] = detail.extent;
    const size = Math.min(state.health?.default_region_um || 300, x1 - x0, y1 - y0);
    state.lastRectangle = { kind: 'box', box: clampBox([(x0+x1-size)/2, (y0+y1-size)/2, (x0+x1+size)/2, (y0+y1+size)/2], detail.extent) };
    pickerLoading = false; setRegionMode('box');
    if (prefill) { filled = prefillPicker(prefill); state.proceedNotes = prefill.notes || []; renderNotes($('picker-notes'), [...sampleNotes(detail), ...state.proceedNotes]); onFilled?.(filled); }   // no await since the sample loaded: Start never went ahead before this
    const focus = prefill && regionBBox(state.region, detail.extent);
    if (focus) stage.fit({ x0:focus[0], y0:focus[1], w:focus[2]-focus[0], h:focus[3]-focus[1] }, true);
    else stage.fit({ x0, y0, w: x1 - x0, h: y1 - y0 }, false);
    const ov = detail.overview;
    if (ov) {
      try {
        const img = await api.image(api.overviewUrl(sampleId));
        if (!valid()) return filled;
        state.overviewImage = { rect:{ x0: ov.x0, y0: ov.y0, w: ov.width * ov.um_per_px, h: ov.height * (ov.um_per_px_y ?? ov.um_per_px) }, channels:[{ source: img, color: [1.15, 1.25, 1.5], on: true }] };
        loadStainPreview();
      } catch { if (valid()) notice('The overview image could not be loaded. You can still choose a region and prior.'); }
    }
  } catch (error) {
    if (!valid()) return filled;
    pickerLoading = false; $('picker-hint').textContent = '';
    $('picker-error').hidden = false; $('picker-error').textContent = openFailure(error).text;   // one plain line, as the open dialog says it
  }
  return filled;
}

function clearProceedNotes() {
  if (!state.proceedNotes?.length) return;
  state.proceedNotes = []; renderNotes($('picker-notes'), sampleNotes(state.sample));
}
function setRegion(region) {
  clearProceedNotes();
  state.region = region;
  if (region.kind === 'box') state.lastRectangle = region;
  if (region.kind === 'polygon') state.lastLasso = region;
}

function setRegionMode(kind) {
  if (!state.sample?.extent) return;
  dropLasso();   // a tool chosen (again) starts afresh
  setRegion(kind === 'full' ? { kind } : kind === 'box' ? state.lastRectangle : state.lastLasso || { kind: 'polygon', points: [] });
  if(kind==='full')fitView();
  for (const button of $('region-mode').children) button.setAttribute('aria-pressed', String(button.dataset.value === kind));
  $('region-help').textContent = kind === 'box' ? 'Drag a rectangle; drag inside to move, corners to resize. Shift = square. Right/middle drag to pan.' :
    kind === 'polygon' ? 'Click to place corners or drag to draw; double-click, Enter or click the first point to close; Backspace ' +
      'takes back the last corner or stretch; Escape cancels. Shift or right/middle drag to pan.' : 'The whole sample will be segmented. Zoom in to inspect the prior.';
  updatePicker(); drawOverlay();
}

/** What Start needs before it can go ahead, for the run type chosen: explain.mjs problems, each naming its field as the
 *  picker labels it (none: Start goes ahead). Start is never greyed out for them: a press lists them. */
function pickerProblems() { return [...settingProblems(), ...boxProblems()]; }
/** Why proseg-live cannot run, by what makes it (health's proseg_missing): a build, an install, or the setting. */
const prosegMissing = () => ({install: PICKER_TEXT.notInstalled, setting: PICKER_TEXT.notFound})[state.health?.proseg_missing] ?? PICKER_TEXT.notBuilt;
/** Everything but the Training patches boxes themselves: the picker's fields and region, and what it is waiting for. */
function settingProblems() {
  const sample = state.sample, P = PICKER_TEXT;
  if (pickerLoading) return [problem(null, '', P.loading)];
  if (state.mode !== 'pick' || !sample?.extent) return [problem(null, '', P.openSample)];
  const box = regionBBox(state.region, sample.extent), w = box ? box[2] - box[0] : 0, h = box ? box[3] - box[1] : 0;
  // An outline still being drawn is not the region yet: Start would otherwise go ahead with the one before it.
  const region = drawingLasso() ? P.closeLasso : state.region?.kind === 'polygon' && state.region.points.length < 3 ? P.drawLasso :
    !box ? P.drawRegion : w < 20 || h < 20 ? P.regionSmall : '';
  const stain = ['stain','patches'].includes($('run-kind').value);
  // A proseg run on a sample without transcripts (only set from outside: the picker opens it in Stain segmentation).
  if (!stain && !sampleTools(sample, Object.keys(TOOLS)).includes('proseg')) return [problem(null, '', NOTE_TEXT.imageOnly)];
  const found = stain ? stainPickerProblems(region ? null : [w, h]) : [
    ...(state.health?.proseg ? [] : [problem(null, '', prosegMissing())]),
    ...(state.prior.cells || state.prior.nuclei ? [] : [problem('prior-choice', P.startFrom, state.priorChoices?.length ? P.firstCells : P.noFirstCells)]),
  ];
  // Proseg's advanced options come after the region, as on the page (a press opens their section to the first one).
  const after = stain ? [] : options.problems();
  return region ? [...found, problem(null, P.region, region), ...after] : [...found, ...after];
}
/** Training patches: each of `boxes` (by default those on the image, when they were suggested for the picker as it stands)
 *  that is not inside the region they were suggested for (the preview's frame: the region's image window and its lasso),
 *  by the number the image shows on it. A box as the engine suggested it is never named: the engine accepted it. */
function boxProblems(boxes = null) {
  if (!boxes) boxes = $('run-kind').value === 'patches' && previewMatches() ? stainPatches.previewBoxes() : null;
  const frame = { ...patchPreview.frame, suggested: patchPreview.offered };
  return (boxes || []).flatMap((box, i) => boxInRegion(box, frame) ? [] :
    [problem(null, PICKER_TEXT.box.replace('{n}', i + 1), PICKER_TEXT.boxOutside)]);
}
/** Stain segmentation and Training patches: the patches' count and size (`size`: the region's [w, h] in µm, null while the
 *  region has a problem of its own), the method and model list as they stand, then whatever validStain() rejects, by
 *  field (stain.mjs stainProblems). */
function stainPickerProblems(size) {
  const T = STAIN_TEXT, P = PICKER_TEXT, PT = STAIN_PATCH_TEXT, found = [], last = [];
  const patches = $('run-kind').value === 'patches', empty = patches && $('patch-start-from').value === 'empty';
  const value = readStain(), images = state.sample.images, image = images?.find(i => i.key === value.image);
  // Sizes in pixels of the level chosen (stain.mjs): a level the image does not have is stainProblems' to name.
  const small = size && resolutionRegionProblem(image, value.level, [0, 0, ...size]);
  if (small) last.push(small);
  if (patches) {
    if (!(Number.isInteger(value.count) && value.count >= 1 && value.count <= 64)) found.push(problem('patch-count', PT.count, P.count));
    if (!(Number.isInteger(value.size) && value.size >= 64 && value.size <= 4096)) found.push(problem('patch-size', PT.size, P.size));
    else if (size && image) {
      const largest = largestPatch(...size, image, value.level);
      if (largest < 64) last.push(problem(null, P.region, P.patchRegion));
      else if (value.size > largest) found.push(problem('patch-size', PT.size, P.sizeFits.replace('{largest}', fmt.format(largest))));
    }
  }
  const available = empty || stainAvailable(), listed = available && !empty && !state.stainModelError ? state.stainModels : null;
  // Said on the Method choice when there is one to make; with Cellpose alone (no Method row) in a sentence of its own.
  if (!available) found.push(stainMethods(state.health).length > 1 ? problem('stain-method', T.method, P.unavailable) :
    problem(null, '', T.unavailable));
  else if (!empty && state.stainModelError) found.push(problem('stain-model', T.model, P.modelsError));
  else if (!empty && !state.stainModels) found.push(problem('stain-model', T.model, P.modelsLoading));
  found.push(...stainProblems(value, images, listed, { pair: patches && value.method === 'instanseg' && value.model !== 'brightfield_nuclei' }));
  if (!found.length && !validStain(value, images, listed)) found.push(problem(null, '', T.settingsError));   // never sent as it is
  return [...found, ...last];
}

function updatePicker() {
  if (state.sample) renderResolution();
  const sample = state.sample, settings = settingProblems();
  if (!settings.length && !boxProblems().length) clearExplanation($('start'));   // what a press listed goes once nothing is missing
  patchPreviewDue(settings);   // a box moved out of the region holds back no suggestion
  if (state.mode !== 'pick' || !sample?.extent) return;
  const box = regionBBox(state.region, sample.extent), w = box ? box[2] - box[0] : 0, h = box ? box[3] - box[1] : 0;
  const area = regionArea(state.region, sample.extent), count = estimatedTranscripts(sample.transcripts, state.region, sample.extent);
  $('picker-size').textContent = box ? `${fmt.format(Math.round(w))} × ${fmt.format(Math.round(h))} µm` : 'Draw a lasso on the sample';
  const estimate = count >= 1e6 ? `${fmt1.format(count / 1e6)} M` : count >= 1e3 ? `${fmt1.format(count / 1e3)} k` : fmt.format(count);
  $('picker-area').textContent = box ? `${area >= 1e6 ? fmt1.format(area / 1e6) + ' mm²' : fmt.format(Math.round(area)) + ' µm²'} · ≈ ${estimate} transcripts` : '';
  $('picker-mode').textContent = !box ? '' : modeForBBox(box, sample.limits?.detail_um) === 'detail' ?
    'Full detail: every iteration, 2-D and 3-D, everywhere' : 'Whole-sample mode: the live view streams the part you zoom into';
  if (['stain','patches'].includes($('run-kind').value)) {
    const patches=$('run-kind').value==='patches', empty=patches && $('patch-start-from').value==='empty';
    $('picker-mode').textContent = patches ? STAIN_PATCH_TEXT.directHelp : STAIN_TEXT.tiles;
    $('picker-area').textContent = box ? `${area >= 1e6 ? fmt1.format(area / 1e6) + ' mm²' : fmt.format(Math.round(area)) + ' µm²'}` : '';
    if (state.region?.kind === 'full') $('region-help').textContent = patches ? STAIN_PATCH_TEXT.fullHelp : STAIN_TEXT.fullHelp;
    $('patch-size-note').hidden=Number($('patch-size').value)<=1024;
    $('patch-size-scale').textContent = patchSizeNote(Number($('patch-size').value), $('stain-level').resolution,
      { pixelFrame: sample?.notes?.some(note => note.code === 'pixel_frame') });
    const available = empty || stainAvailable();
    // Only what the picker is waiting for, or what the suggested boxes found (e.g. fewer fit than asked): what Start
    // still needs is listed when it is pressed (pickerProblems).
    $('picker-hint').textContent = (!empty && available && (state.stainModelError || (!state.stainModels && STAIN_TEXT.modelsLoading))) ||
      (patches && patchPreview.controller && PICKER_TEXT.suggesting) ||
      (patches && patchPreview.shown && patchPreview.shown === patchPreview.key && patchPreview.message) || '';
    // Only its own message: a refusal the Start handler shows (e.g. no tissue in the region) must stay visible.
    if (!available) { $('picker-error').hidden = false; $('picker-error').textContent = STAIN_TEXT.unavailable; }
    else if ($('picker-error').textContent === STAIN_TEXT.unavailable) $('picker-error').hidden = true;
    return;
  }
  $('picker-hint').textContent = '';
  if (!state.health?.proseg) { $('picker-error').hidden = false; $('picker-error').textContent = prosegMissing(); }
}

// ---- Training patches: the suggested boxes on the image before Start ------------------------------------------------
// While Start would go ahead, the engine suggests the boxes for the picker's region and settings (POST
// /samples/{id}/patches/preview: nothing is written) PATCH_PREVIEW_MS after the last change; a newer change cancels an
// older request. The boxes stay on the image until the next answer, and can be moved, resized or dropped there
// (stain-patches.mjs showPreview), inside the region's frame the answer gives. An answer that suggests the same boxes again
// (after a change they do not depend on, such as the model, or a field that was briefly not valid) keeps what was done to
// them. Start creates the item with exactly the boxes shown, when they were suggested for the picker as it stands.
const PATCH_PREVIEW_MS = 400;
// key: the request asked for (or due); shown: the request whose boxes are on the image, `suggested` the engine's answer
// for it (its boxes, `offered`) and `frame` their region ({extent, outline}); refused: the request the engine refused, with
// its reason (`refusal`); failed: the request the engine could not answer (`failure`: why), asked again only after a change
// or by Start.
const patchPreview = { key: '', timer: 0, controller: null, pending: null, shown: '', suggested: '', offered: null, message: '',
  frame: null, refused: '', refusal: '', failed: '', failure: '' };
let regionDragging = false;   // a region gesture in progress (regionGesture): no boxes are asked for until it ends
/** What the picker would ask for now: its sample and region, and the settings Start would send. */
function patchRequest() { return { sample: state.sample?.id, region: state.region, stain: readStain() }; }
/** Whether the boxes on the image were suggested for the picker as it stands (`settled`: with no change or request still
 *  waiting for boxes of its own). */
function previewMatches(settled = false) {
  return !!patchPreview.shown && (!settled || !patchPreview.timer && !patchPreview.controller) &&
    patchPreview.shown === JSON.stringify(patchRequest());
}
/** updatePicker: asked for again after a change. A field still being typed or not valid yet (typing 25 over 30 passes
 *  through 2; a model list that could not be refreshed: every problem has its field) only holds the request back, and the
 *  boxes on the image stay as they are. No region to suggest boxes in (none drawn, a lasso not drawn yet, a region too
 *  small: a problem without a field) takes them away. While the region is being drawn, moved or resized nothing is asked
 *  for: the gesture's end asks. */
function patchPreviewDue(problems) {
  if (state.mode !== 'pick' || $('run-kind').value !== 'patches') { clearPatchPreview(); return; }
  if (regionDragging) return;
  if (problems.some(p => !p.field)) { clearPatchPreview(); return; }
  const request = patchRequest(), key = JSON.stringify(request);
  if (key !== patchPreview.failed) patchPreview.failed = '';    // changed since it failed: it may be asked for again
  if (problems.length) { cancelPatchPreview(); patchPreview.key = ''; return; }
  if (key === patchPreview.key || key === patchPreview.failed) return;
  cancelPatchPreview(); patchPreview.key = key;
  patchPreview.timer = setTimeout(() => askPatchPreview(request, key), PATCH_PREVIEW_MS);
}
function cancelPatchPreview() {
  clearTimeout(patchPreview.timer); patchPreview.controller?.abort();
  patchPreview.timer = 0; patchPreview.controller = patchPreview.pending = null;
}
/** No boxes, nothing asked for and nothing remembered: another run type or start, another sample, a run opened, no region
 *  to suggest boxes in. A preview's refusal under Start goes with them (Start's own stays). */
function clearPatchPreview() {
  cancelPatchPreview();
  if (patchPreview.shown) stainPatches.clearPreview();
  if (patchPreview.refusal && $('picker-error').textContent === patchPreview.refusal) $('picker-error').hidden = true;
  Object.assign(patchPreview, { key: '', shown: '', suggested: '', offered: null, message: '', frame: null, refused: '', refusal: '',
    failed: '', failure: '' });
}
/** setupPointer: a region gesture (a lasso being drawn; the rectangle drawn, moved or resized) begins or ends. While it
 *  lasts nothing is asked for, and what was due waits too: the gesture's end (its updatePicker) asks for the region it
 *  leaves. A Lasso outline (lasso) is one gesture from its first point until it closes or is given up, whatever presses
 *  and pans come between. */
function regionGesture(on) {
  regionDragging = on || lasso.active;
  if (regionDragging && patchPreview.timer) { cancelPatchPreview(); patchPreview.key = ''; }
}
function askPatchPreview(request, key) {
  cancelPatchPreview(); patchPreview.key = key;
  const controller = patchPreview.controller = new AbortController(), current = () => patchPreview.controller === controller;
  patchPreview.pending = api.previewPatches(request.sample, request.region, request.stain, controller.signal)
    .then(result => { if (current()) showPatchPreview(result, key); }, error => { if (current()) refusePatchPreview(error, key); })
    .finally(() => { if (current()) patchPreview.controller = patchPreview.pending = null; });
  updatePicker();   // "Suggesting boxes…" meanwhile
}
function showPatchPreview(result, key) {
  const suggested = JSON.stringify([result.patches, result.run_pixel_to_sample]);
  // The region's frame, in the boxes' pixels: its image window [width, height] and, for a lasso, the lasso. The panel keeps
  // moved boxes inside it; a box outside it is named when Start is pressed (boxProblems).
  const frame = { extent: result.extent ?? null, outline: result.outline ?? null };
  const kept = patchPreview.shown && suggested === patchPreview.suggested && stainPatches.previewBoxes();
  // A press on the region's own handles is the page's, even over a box (regionHandle).
  if (!kept || JSON.stringify(frame) !== JSON.stringify(patchPreview.frame))
    stainPatches.showPreview({ patches: kept ? kept.map((box, i) => ({ id: i + 1, box })) : result.patches,
      run_pixel_to_sample: result.run_pixel_to_sample, ...frame }, { onChange: () => updatePicker(), yields: regionHandle });
  Object.assign(patchPreview, { controller: null, pending: null, shown: key, suggested, offered: result.patches.map(p => p.box),
    message: result.message || '', frame, failed: '', failure: '' });
  // An earlier preview's refusal goes once one is answered (Start's own refusal stays).
  if (patchPreview.refusal && $('picker-error').textContent === patchPreview.refusal) $('picker-error').hidden = true;
  patchPreview.refused = patchPreview.refusal = '';
  updatePicker();
}
function refusePatchPreview(error, key) {
  if (error.name === 'AbortError') return;
  const reason = String(error.message || error);
  if (error.status !== 422) {
    // Not a refusal: the engine could not answer (busy, failing) or could not be reached. The boxes on the image stay; this
    // request is asked for again after the next change, or by Start (previewedBoxes), never by itself.
    Object.assign(patchPreview, { controller: null, pending: null, key: '', failed: key, failure: reason });
    updatePicker();
    return;
  }
  // Refused as creating the item would be (e.g. no tissue in the region), or for what only the boxes need (no patches
  // with image detail): the boxes go, the engine's reason is under Start, like Start's, and a press of Start with these
  // settings says it again instead of creating an item without boxes.
  if (patchPreview.shown) stainPatches.clearPreview();
  Object.assign(patchPreview, { controller: null, pending: null, shown: '', suggested: '', offered: null, message: '', frame: null,
    refused: key, refusal: reason, failed: '', failure: '' });
  $('picker-error').hidden = false; $('picker-error').textContent = reason;
  updatePicker();
}
/** The boxes Start sends with a Training patches item: those shown for exactly `request`. Asked for at once when a change
 *  still waits for them, or when none are shown for it and the engine did not refuse it (e.g. its request failed); a
 *  request on its way is waited for. Null when the engine refused it (Start says why and creates nothing) or the request
 *  was cancelled meanwhile; throws when the engine could not answer (Start says so and creates nothing). */
async function previewedBoxes(request) {
  const key = JSON.stringify(request), waiting = patchPreview.key === key && !!patchPreview.controller;
  if (!waiting && (patchPreview.timer || patchPreview.shown !== key && patchPreview.refused !== key)) askPatchPreview(request, key);
  await patchPreview.pending;
  if (patchPreview.shown === key) return stainPatches.previewBoxes();
  if (patchPreview.failed === key) throw Error(PICKER_TEXT.boxesFailed.replace('{reason}', () => patchPreview.failure));
  return null;
}

function regionPath(g, region, box) {
  if (region?.kind === 'polygon') {
    region.points.forEach(([x, y], i) => { const p = stage.toScreen(x, y); if (i) g.lineTo(...p); else g.moveTo(...p); }); g.closePath();
  } else if (box) {
    const [a, b] = stage.toScreen(box[0], box[1]), [c, d] = stage.toScreen(box[2], box[3]); g.rect(a, b, c - a, d - b);
  }
}

/** `view`: the view may have changed (the scale bar, the prior's outlines and the legend follow it); false when only the
 *  overlay's drawing did, as for each pointer move while a Lasso outline is drawn (in the stain picker previewPrior would
 *  have the whole image drawn again each time). */
function drawOverlay(view = true) {
  overlayWanted = false;   // drawn now: a paint batch's overlaySoon has nothing left to do
  // The overlay covers the stage exactly, so it takes the stage's cached size and pixel ratio (no layout read).
  const ratio = stage.ratio, W = stage.cssWidth, H = stage.cssHeight;
  if (overlay.width !== stage.canvas.width || overlay.height !== stage.canvas.height) { overlay.width = stage.canvas.width; overlay.height = stage.canvas.height; }
  const g = overlay.getContext('2d'); g.setTransform(ratio, 0, 0, ratio, 0, 0); g.clearRect(0, 0, W, H);
  if (view) { updateScale(); previewPrior(); updateQcLegend(); }
  const hideScale = stage.orbit.active || state.mode === 'empty';
  if ($('scale').hidden !== hideScale) $('scale').hidden = hideScale;
  // Go deeper: its arrows' note (2-D, zoomed in), the window inside the stage; the test corner kept inside it too.
  if (state.mode === 'run') { deeperArrows.sync(); deeper.layout(); testCorner.layout(); }
  if (view && state.mode === 'run' && deeperStart && !stage.startOutline) deeperStartOutline();   // the prior's vectors or labels came in after Show
  if (stage.orbit.active) return;
  g.strokeStyle='rgba(180,220,190,.4)';g.lineWidth=1;g.setLineDash([]);
  for(const points of tissueView.polygons){g.beginPath();regionPath(g,{kind:'polygon',points});g.stroke();}
  if(state.mode==='run')stainExclusions.draw(g,regionPath);
  if (state.mode === 'pick' && state.sample?.extent) {
    g.setLineDash([]);
    const region = state.region, box = regionBBox(region, state.sample.extent);
    if (lasso.active) { drawLasso(g, region, box); return; }
    if (!box) return;
    if (region.kind === 'box') {
      g.fillStyle = 'rgba(8,10,14,.55)'; g.beginPath(); g.rect(0, 0, W, H); regionPath(g, region, box); g.fill('evenodd');
    }
    g.beginPath(); regionPath(g, region, box);
    if (region.kind === 'polygon') { g.fillStyle = 'rgba(124,196,255,.13)'; g.fill('evenodd'); }
    g.strokeStyle = '#7cc4ff'; g.lineWidth = 2; g.stroke();
    if (region.kind === 'box') {
      g.fillStyle = '#7cc4ff';
      for (const p of [[box[0],box[1]],[box[2],box[1]],[box[0],box[3]],[box[2],box[3]]]) { const [x,y] = stage.toScreen(...p); g.fillRect(x-4,y-4,8,8); }
    }
  } else if (state.mode === 'run' && state.run) {
    g.strokeStyle = 'rgba(230,234,240,.45)'; g.lineWidth = 1; g.setLineDash([6, 4]);
    g.beginPath(); regionPath(g, state.run.region, state.run.box); g.stroke(); g.setLineDash([]);
    qcOverlay.draw(g); qcBins();   // the mixing check's per-bin layer and the marks, in world µm
    deeperArrows.draw(g);   // Go deeper's arrows from the window's sources to its cell, over the layer
  }
}
/** The Lasso outline being drawn (lasso.mjs preview): its points so far, a dashed line on to the pointer, and its first
 *  point, white and larger while a press there would close it. The region stays as it was until the outline closes
 *  (Escape keeps it): a thin line meanwhile. */
function drawLasso(g, region, box) {
  if (box) { g.strokeStyle = 'rgba(124,196,255,.55)'; g.lineWidth = 1; g.beginPath(); regionPath(g, region, box); g.stroke(); }
  const { points, guide, closable } = lasso.preview(), path = points.map(p => stage.toScreen(...p));
  g.strokeStyle = '#7cc4ff'; g.lineWidth = 2; g.beginPath();
  path.forEach((p, i) => i ? g.lineTo(...p) : g.moveTo(...p)); g.stroke();
  if (guide) {
    g.lineWidth = 1; g.setLineDash([6, 4]);
    g.beginPath(); g.moveTo(...path.at(-1)); g.lineTo(...stage.toScreen(...guide)); g.stroke(); g.setLineDash([]);
  }
  const [x, y] = path[0], r = closable ? 6 : 4;
  g.fillStyle = closable ? '#fff' : '#7cc4ff'; g.fillRect(x - r, y - r, 2 * r, 2 * r);
}

let previewTimer, previewKey = '', previewTicket = 0, previewController, previewShown = '';
function resetPreview() {
  clearTimeout(previewTimer); previewController?.abort(); previewTicket++; previewKey = ''; priors.hold = false;
}
function previewPrior() {
  if (state.mode !== 'pick' || !state.sample?.extent) { resetPreview(); return; }
  if (['stain','patches'].includes($('run-kind').value)) { resetPreview(); priors.vectors = null; stage.dirty = true; return; }
  const sample = state.sample, id = sample.id, prior = state.prior;
  const columnNotes = ['cells','nuclei'].filter(kind => prior[kind]?.kind === 'column').map(kind => `${kind === 'cells' ? 'Cells' : 'Nuclei'} assigned from the transcripts' ${prior[kind].column} column`);
  const shapes = ['cells','nuclei'].filter(kind => prior[kind]?.kind === 'shapes');
  const a = stage.toWorld(0,0), b = stage.toWorld(stage.cssWidth,stage.cssHeight);
  const visible = [Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];
  const zoomed = Math.max(visible[2]-visible[0],visible[3]-visible[1]) < 3000;
  const note = text => { $('prior-preview-note').textContent = [...columnNotes, text].filter(Boolean).join(' · '); };
  if (!$('show-prior').checked || !zoomed || !shapes.length) {
    resetPreview(); previewShown = ''; if (priors.vectors) { priors.vectors = null; stage.dirty = true; }
    note($('show-prior').checked && shapes.length && !zoomed ? 'Zoom in to see prior outlines.' : ''); return;
  }
  const box = clampBox(visible, sample.extent), shown = JSON.stringify([id, prior]), key = JSON.stringify([id, box.map(v => Math.round(v)), prior]);
  if (key === previewKey) return;
  resetPreview(); previewKey = key; const ticket = previewTicket, controller = previewController = new AbortController();
  // A new box keeps the outlines on screen (they move with the camera) until its own arrive; another prior clears them.
  // Held meanwhile: drawing the old box again at the new camera would be thrown away when the new one arrives (on a
  // whole sample each drawing is a long task on the user's desktop).
  if (shown !== previewShown && priors.vectors) { priors.vectors = null; stage.dirty = true; }
  priors.hold = !!priors.vectors;
  note('Loading prior outlines…');
  previewTimer = setTimeout(async () => {
    const keys = [...new Set(shapes.map(kind => prior[kind].key))];
    const results = await Promise.allSettled(keys.map(key => api.shapes(id, key, box, controller.signal)));
    if (ticket !== previewTicket || state.mode !== 'pick' || state.sample?.id !== id) return;
    // Only the layers chosen as shapes priors are stroked (PriorLayers converts JSON polygons to PolygonSets).
    const vectors = { cells: null, nuclei: null }; let truncated = false, error = '';
    for (let i = 0; i < keys.length; i++) {
      const result = results[i];
      if (result.status === 'fulfilled') {
        for (const kind of shapes) if (prior[kind].key === keys[i]) vectors[kind] = result.value.polygons;
        truncated ||= result.value.truncated;
      } else if (result.reason.name !== 'AbortError') error = result.reason.message;
    }
    priors.vectors = vectors; priors.hold = false; previewShown = shown; stage.dirty = true;
    note(error ? `Prior outlines unavailable: ${error}` : truncated ? 'Showing up to 20,000 outlines per layer; zoom in for more.' : '');
  }, 200);
}

function updateScale() {
  const pxPerUm = stage.scale;
  const target = 110 / pxPerUm;
  const steps = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  const um = steps.find((s) => s >= target * 0.6) || 5000;
  const width = `${Math.round(um * pxPerUm)}px`, label = um >= 1000 ? `${um / 1000} mm` : `${um} µm`;
  if ($('scale-bar').style.width !== width) $('scale-bar').style.width = width;   // camera moves only
  if ($('scale-label').textContent !== label) $('scale-label').textContent = label;
}

// ---- interaction ----------------------------------------------------------------------------------------------
// The Lasso tool's outline while it is drawn (lasso.mjs): clicks place corners and drags draw freehand, in any mix, and
// a press on the first point, a double-click or Enter makes it the region; one freehand stroke on its own does when it
// is released, as the lasso always did. Until then the region stays as it was. setupPointer forwards the pointer and
// keys, drawOverlay draws it.
const lasso = new LassoDraft();
/** An outline is being drawn in the picker (one left behind when a run opened is dropped by enterPick). */
function drawingLasso() { return lasso.active && state.mode === 'pick'; }
/** What setupPointer does with an answer of the outline (lasso.mjs): 'close' makes it the region, simplified as a traced
 *  lasso always was (one left with no area keeps the region as it was, and says what it needs, as 'few' does), 'cancel'
 *  gives it up, 'changed' draws it again. Its end (regionGesture) asks for what the picker held back meanwhile. */
function lassoResult(result) {
  if (result === 'none') return;
  if (result === 'few') { notice(PICKER_TEXT.lassoFew); result = 'changed'; }
  if (result === 'close') {
    const points = simplifyPolygon(lasso.points(), 1.5 / stage.scale, 2000);
    if (points.length >= 3 && regionArea({ kind: 'polygon', points }) > 0) setRegion({ kind: 'polygon', points });
    else notice(PICKER_TEXT.lassoFew);
  }
  if (result !== 'changed') lasso.reset();
  const drawing = regionDragging;
  regionGesture(false); drawOverlay(false);
  if (regionDragging !== drawing) updatePicker();
}
/** The outline being drawn is given up, the region staying as it was: another sample, or a region tool chosen. */
function dropLasso() {
  if (!lasso.active) return;
  lasso.reset(); regionGesture(false);
}
/** The region rectangle's corner a press is on, as setupPointer resizes from it: its index in [top left, top right, bottom
 *  left, bottom right], or -1. */
function regionCorner(e) {
  const b = state.region?.kind === 'box' ? state.region.box : null;
  return !b ? -1 : [[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]].findIndex(p => {
    const [x, y] = stage.toScreen(...p); return Math.hypot(x - e.offsetX, y - e.offsetY) <= 9;
  });
}
/** A press the page acts on before the picker's boxes (stain-patches.mjs showPreview `yields`): one on the region's own
 *  handles, the rectangle's corners, which setupPointer resizes it from, and any while a Lasso outline is drawn (its
 *  corners may go over a box). A press with Shift the panel leaves to the page itself. */
function regionHandle(e) {
  return state.mode === 'pick' && !!state.sample?.extent && e.button === 0 && !stage.orbit.active && (regionCorner(e) >= 0 || lasso.active);
}
function setupPointer() {
  const canvas = $('stage'), wrap = canvas.parentElement;
  let drag = null;
  const cancel = () => {
    // A Lasso press the browser takes away drops what it placed or drew (as Backspace); the rest of the outline stays.
    if (drag?.kind === 'lasso') { drag = null; lassoResult(lasso.key({ key: 'Backspace' })); return; }
    if (drag?.before && drag.sample === state.sample?.id && state.mode === 'pick') setRegion(drag.before);
    drag = null; regionGesture(false); wrap.classList.remove('grabbing'); drawOverlay(); updatePicker();
  };
  // A pointer event as the Lasso outline takes it: its point in the sample (kept inside it) and on screen.
  const at = e => [clampPoint(stage.toWorld(e.offsetX, e.offsetY), state.sample.extent), [e.offsetX, e.offsetY]];
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('wheel', e => {
    if(state.run?.mode==='tiled' && e.deltaY<0)tiled.overview=false;
    e.preventDefault(); stage.zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0016)));
  }, { passive: false });
  // A double-click closes the Lasso outline being drawn. With none being drawn it fits the view, as everywhere else, and
  // what its two clicks began goes; so it does with Shift (two pans) and outside the Lasso.
  canvas.addEventListener('dblclick', e => {
    const lassoing = state.mode === 'pick' && state.region?.kind === 'polygon' && !e.shiftKey;
    const result = lassoing ? lasso.doubleClick() : 'none';
    if (result !== 'none') return lassoResult(result);
    if (lassoing) lassoResult('cancel');
    fitView();
  });
  canvas.addEventListener('pointerdown', e => {
    e.preventDefault(); canvas.setPointerCapture(e.pointerId);
    if (stage.orbit.active) { drag = { kind: 'orbit', moved: 0, x: e.offsetX, y: e.offsetY, pan: e.button !== 0 || e.shiftKey }; return; }
    const point = stage.toWorld(e.offsetX, e.offsetY), region = state.region;
    const pan = state.mode !== 'pick' || !state.sample?.extent || e.button !== 0 || region?.kind === 'full' || (e.shiftKey && region?.kind !== 'box');
    if (pan) drag = { kind: 'pan', x: e.offsetX, y: e.offsetY, moved: 0 };
    else {
      const before = region, base = { before, sample: state.sample.id, x: e.offsetX, y: e.offsetY, moved: 0 };
      if (region.kind === 'polygon') {
        // The Lasso: a corner, or a freehand stretch once it moves, or on the outline's first point its end. Its keys
        // then come to the image, never to a button pressed before.
        canvas.focus({ preventScroll: true });
        const result = lasso.down(...at(e));
        drag = result === 'changed' ? { ...base, kind: 'lasso' } : null;
        lassoResult(result);
      } else {
        const b = region.box, corners = [[b[0],b[1]],[b[2],b[1]],[b[0],b[3]],[b[2],b[3]]];
        const corner = regionCorner(e);   // also where the picker's boxes leave a press to the page (regionHandle)
        const inside = point[0] >= b[0] && point[0] <= b[2] && point[1] >= b[1] && point[1] <= b[3];
        drag = corner >= 0 ? { ...base, kind: 'resize', anchor: corners[3-corner] } :
          inside ? { ...base, kind: 'move', anchor: point } : { ...base, kind: 'draw', anchor: clampPoint(point, state.sample.extent) };
        regionGesture(true);
      }
    }
    wrap.classList.toggle('grabbing', drag?.kind === 'pan');
  });
  const move = e => {
    if (!drag) { if (drawingLasso()) lassoResult(lasso.move(...at(e), false)); updateTooltip(e); return; }   // the outline's guide
    // The picker changed under a region gesture (another sample, run or region tool): it ends here, and what it held back
    // is asked for now.
    if (drag.sample && (state.mode !== 'pick' || drag.sample !== state.sample?.id || state.region.kind !== drag.before.kind)) {
      drag = null; wrap.classList.remove('grabbing'); lasso.reset(); regionGesture(false); updatePicker(); return;
    }
    $('tooltip').hidden = true; lastPointer = null;
    if (drag.kind === 'orbit') {
      drag.moved += Math.abs(e.offsetX-drag.x)+Math.abs(e.offsetY-drag.y);
      stage.orbit.drag(e.offsetX-drag.x,e.offsetY-drag.y,drag.pan); drag.x=e.offsetX; drag.y=e.offsetY; return;
    }
    if (drag.kind === 'pan') {
      stage.panBy(e.offsetX-drag.x,e.offsetY-drag.y);
      drag.moved += Math.abs(e.offsetX-drag.x)+Math.abs(e.offsetY-drag.y); drag.x=e.offsetX; drag.y=e.offsetY;
      if (drawingLasso()) lasso.move(...at(e), false);   // the pan drew it again, its guide still on the pointer
      return;
    }
    if (drag.kind === 'lasso') {   // every move the browser merged into this one, so a quick stretch keeps its corners
      let result = 'none';
      for (const m of mergedMoves(e)) result = lasso.move(...at(m), true);
      lassoResult(result); return;
    }
    drag.moved = Math.max(drag.moved, Math.hypot(e.offsetX-drag.x,e.offsetY-drag.y));
    if (drag.moved < 3) return;
    const point = clampPoint(stage.toWorld(e.offsetX,e.offsetY),state.sample.extent);
    if (drag.kind === 'move') {
      const dx=point[0]-drag.anchor[0],dy=point[1]-drag.anchor[1],b=drag.before.box;
      setRegion({kind:'box',box:clampBox([b[0]+dx,b[1]+dy,b[2]+dx,b[3]+dy],state.sample.extent,true)});
    } else setRegion({kind:'box',box:rectangleFromPoints(drag.anchor,point,state.sample.extent,e.shiftKey)});
    drawOverlay(); updatePicker();
  };
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', e => {
    if (drag?.sample) move(e);
    const was = drag; drag = null; wrap.classList.remove('grabbing');
    regionGesture(false);   // the updatePicker below asks for the region the gesture leaves (a Lasso outline goes on)
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    if (was?.kind === 'lasso') lassoResult(lasso.up(...at(e)));   // a corner, a stretch, or a stroke on its own: the region
    else if (was?.sample && was.sample === state.sample?.id && state.mode === 'pick') {
      if (['draw','move'].includes(was.kind) && was.moved < 3) {
        const [x,y] = clampPoint(stage.toWorld(e.offsetX,e.offsetY),state.sample.extent),ex=state.sample.extent;
        const size = Math.min(state.health?.default_region_um || 300,ex[2]-ex[0],ex[3]-ex[1]);
        setRegion({kind:'box',box:clampBox([x-size/2,y-size/2,x+size/2,y+size/2],ex,true)});
      }
      drawOverlay(); updatePicker();
    }
    if (['pan','orbit'].includes(was?.kind) && was.moved < 4 && state.mode === 'run' && e.button === 0) {
      // A click selects the cell under it (no area chip to click).
      state.selected = state.selected === state.hover ? 0 : state.hover; stage.set({selected:state.selected}); updateCellCard();
    }
  });
  canvas.addEventListener('pointercancel',cancel);
  canvas.addEventListener('lostpointercapture',() => { if (drag) cancel(); });
  // While an outline is drawn: Enter closes it, Backspace takes back its last corner or stretch, Escape gives it up (the
  // region stays as it was); the rest of a press under way then draws nothing. Keys typed in a field or a dialog, and
  // Enter on a button, stay theirs.
  document.addEventListener('keydown', e => {
    if (e.defaultPrevented || !drawingLasso() || document.querySelector('dialog[open]') || e.target.closest('input, select, textarea') ||
      e.metaKey || e.ctrlKey || e.altKey || e.key === 'Enter' && e.target.closest('button')) return;
    const result = lasso.key(e);
    if (result === 'none') return;
    e.preventDefault();
    if (drag?.kind === 'lasso') drag = null;
    lassoResult(result);
  });
  canvas.addEventListener('pointerleave',() => {
    $('tooltip').hidden=true; lastPointer=null; deeper.hover(0);   // a source's hover highlight goes with the pointer
    if(state.hover){state.hover=0;stage.set({hover:0});analysis.tissueHover(0);}
  });
  stage.onChange(() => {drawOverlay();if(['stain','patches'].includes(state.run?.kind)){stainPatches.move();stainView.move();}else if(state.run?.mode==='tiled')tiled.move();});
}

function fitView() {
  if (state.mode === 'run' && state.run) {
    if(state.run.mode==='tiled') {tiled.overview=true;if(stage.orbit.enabled)setView(false,false);}
    const [x0,y0,x1,y1] = state.run.box; stage.fit({x0,y0,w:x1-x0,h:y1-y0},true,.03);
  } else if (state.sample?.extent) {
    const [x0,y0,x1,y1] = state.sample.extent; stage.fit({x0,y0,w:x1-x0,h:y1-y0});
  }
}

function step(delta) {
  if (!state.frames.length) return;
  stopReplay();
  setLive(false);
  show(Math.min(state.frames.length - 1, Math.max(0, state.current + delta)), { pulse: true });
}

async function control(action, speed) {
  if (!state.run) return;
  const id = state.run.id;
  // The user's run actions go into the log (log-export.mjs), with the iteration on screen; named as the button said
  // (Resume on a run still on its prior is its Start button: renderRun).
  const run0 = state.run, onPrior = run0.state === 'paused' && run0.ack?.state === 'paused' && state.frames.length === 1;
  const named = action === 'speed' ? `speed ${speed}` : action === 'resume' && onPrior ? 'start' : action;
  globalThis.sopaCompanionLog?.add('action', `${named} (at iteration ${state.frames[state.current]?.iter ?? '-'})`);
  try {
    const run = await api.control(id, action, speed);
    if (state.run?.id !== id) return;
    state.run = run;
    renderRun();
  } catch (error) {
    notice(String(error.message || error));
  }
}

let finishingPrompt = false;
async function finishAndSave() {
  if (state.run?.kind === 'stain') { await control('stop'); return; }
  if (finishingPrompt || !['running', 'paused'].includes(state.run?.state)) return;
  const gen = generation;
  const id = state.run.id, wasRunning = state.run.state === 'running';
  finishingPrompt = true;
  renderRun();
  const deadline = performance.now() + 20000;
  const withinDeadline = async promise => {
    let timer;
    try { return await Promise.race([promise, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error('Pause acknowledgement timed out after 20 seconds. Finish was not sent. Check the run state and try again.')), Math.max(0, deadline - performance.now()));
    })]); } finally { clearTimeout(timer); }
  };
  try {
    if (wasRunning) {
      const run = await withinDeadline(api.control(id, 'pause'));
      if (gen !== generation || state.run?.id !== id) return;
      state.run = run;
      renderRun();
    }
    stopReplay();
    setLive(false);
    while (true) {
      const run = await withinDeadline(api.run(id));
      if (gen !== generation || state.run?.id !== id) return;
      onMessage({ ev: 'run', run });
      if (run.ack?.state === 'paused') break;
      if (!['running', 'paused'].includes(run.state)) throw new Error('The run ended before pause was acknowledged. Finish was not sent.');
      await withinDeadline(new Promise(resolve => setTimeout(resolve, 200)));
    }
    const ack = state.run.ack;
    const batch = await withinDeadline(api.frames(id, state.frames.length));
    if (gen !== generation) return;
    onMessage({ ev: 'frames', run: id, ...batch });
    if (state.frames[ack.seq]?.iter !== ack.iter) throw new Error('The acknowledged snapshot is not available yet. Finish was not sent; try again.');
    const displayed = await withinDeadline(show(ack.seq, { pulse: false }));
    if (gen !== generation) return;
    if (!displayed || state.current !== ack.seq) throw new Error('Could not display the acknowledged snapshot. Finish was not sent; try again.');
    // Let the uploaded snapshot paint before opening the blocking confirmation dialog.
    await withinDeadline(new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
    if (gen !== generation || state.current !== ack.seq || state.run.ack?.state !== 'paused' || state.run.ack.seq !== ack.seq) return;
    const iteration = ack.iter, end = state.run.schedule_done?.iter;
    // Held at the end of proseg's standard run nothing is asked: nothing samples there, its note says what Finish & save
    // keeps, and finishing there is what proseg itself does. Past that end the question says how far past it the run is
    // (the harness accepts the question by its first words).
    const past = state.run.continued && Number.isFinite(end) && iteration > end ? iteration - end : 0;
    const held = heldAtEnd(state.run);
    const accepted = held || confirm(`Stop sampling now and save proseg's results from iteration ${iteration}${past ? ` (${fmt.format(past)} iteration${past === 1 ? '' : 's'} past proseg's standard run)` : ''}?`);
    globalThis.sopaCompanionLog?.add('action', `Finish & save at iteration ${iteration}: ${held ? 'the end of the standard run, nothing asked' : accepted ? 'confirmed' : 'declined'}`);
    if (accepted || (wasRunning && await carriesOn(id))) {
      if (accepted) zarrSave.trackFinish(id);
      let run;
      try { run = await api.control(id, accepted ? 'finish' : 'resume'); }
      catch (error) { if (accepted) zarrSave.cancelFinish(id); throw error; }
      zarrSave.update(run);
      if (gen === generation && state.run?.id === id) { state.run = run; renderRun(); }
    }
  } catch (error) {
    notice(String(error.message || error));
  } finally {
    finishingPrompt = false;
    renderRun();
  }
}
/** Finish & save declined on a run that was sampling: it samples on, unless it reached the end of proseg's standard run
 *  while the question was open (the run as the engine has it now: held there with its note, where Resume would sample
 *  on past the end, which nobody asked for). */
async function carriesOn(id) { return !heldAtEnd(await api.run(id)); }

function setupControls() {
  for (const node of document.querySelectorAll('[data-patch-copy]')) node.textContent = STAIN_PATCH_TEXT[node.dataset.patchCopy];
  $('patch-start-from').onchange=()=>{clearPatchPreview();updateRunKind();updatePicker();loadStainPreview();};
  for(const id of ['patch-count','patch-size'])$(id).oninput=()=>updatePicker();
  stainCopy(document.querySelectorAll('[data-stain-copy]'));   // Cellpose's settings by Cellpose's names, plain words as tooltips
  // The run type (hidden) is set by the tool chosen and the Training patches section; set on its own (the harness's way)
  // the tool and the section follow it.
  $('run-kind').onchange = () => { clearPatchPreview(); updateRunKind(); setTool(toolOf($('run-kind').value)); resetPreview(); priors.reset(); previewPrior(); updatePicker(); loadStainPreview(); };
  $('patch-tool').addEventListener('toggle', patchToolToggled);
  $('stain-image').onchange = () => { fillStainChannels(); updatePicker(); loadStainPreview(); };
  $('stain-method').onchange = () => {
    loadStainModels.ticket = (loadStainModels.ticket || 0) + 1; loadStainModels.pending = false;
    loadStainModels.key = null;
    $('stain-model').replaceChildren(); $('stain-model').value = ''; state.stainModels = null; state.stainModelError = '';
    showModelsLoading(false);     // the previous method's load is stale now; the new one says it loads, if it can
    updateStainMethod(); loadStainModels(); updatePicker(); loadStainPreview();
  };
  $('stain-model').onchange = () => {
    try { localStorage.setItem(stainModelKey(), $('stain-model').value); } catch { /* optional storage */ }
    updateStainMethod(); updatePicker(); loadStainPreview();
  };
  $('stain-channels').onchange = () => { updatePicker(); loadStainPreview(); };
  for (const id of ['stain-output', 'stain-resolve']) $(id).onchange = () => updatePicker();
  for (const id of ['stain-nucleus', 'stain-cytoplasm', 'stain-diameter', 'stain-flow', 'stain-probability'])
    $(id).oninput = () => { updatePicker(); if(id==='stain-nucleus')loadStainPreview(); };
  for(const id of ['stain-niter','stain-min-size','stain-low','stain-high','stain-intensity-low','stain-intensity-high','stain-tile-norm','stain-sharpen','stain-smooth'])$(id).oninput=updatePicker;
  for(const id of ['stain-augment','stain-invert'])$(id).onchange=updatePicker;
  for(const id of ['stain-normalize','stain-fixed-range'])$(id).onchange=()=>{updateNormalizeFields($);updatePicker();};
  $('stain-tissue').onchange=()=>{updatePicker();loadStainPreview();};
  $('stain-level').onchange = () => { renderResolution(); updatePicker(); };
  $('stain-auto').onchange = () => { $('stain-diameter').disabled = $('stain-auto').checked; updatePicker(); };
  for (const panel of document.querySelectorAll('details.panel')) {
    const key = `sopa-companion:panel:${panel.id}`;
    try { const saved = localStorage.getItem(key); if (saved !== null) panel.open = saved === 'true'; } catch { /* optional storage */ }
    panel.addEventListener('toggle', () => { try { localStorage.setItem(key, String(panel.open)); } catch { /* optional storage */ } });
  }
  $('open-sample').onclick = $('empty-open-sample').onclick = () => leave(async () => {
    if (state.mode === 'run') await enterPick(state.sample);
    sampleBrowser.show();
  }, 'open');
  // The first screen and the header (shell.mjs): the two tools' cards, Back to both, the brand (home) and Close.
  $('tool-stain').onclick = () => chooseTool('stain'); $('tool-proseg').onclick = () => chooseTool('proseg');
  $('tool-back').onclick = () => chooseTool(null);
  $('engine-reload').onclick = () => location.reload();   // the engine did not answer (boot): the page starts again
  $('home').onclick = goHome;
  $('close-app').onclick = pressClose;
  // Never greyed out (explain.mjs): pressed without a sample, New run says to open one.
  guard($('new-run'), () => state.sample ? [] : [problem(null, '', PICKER_TEXT.openSample)], () => leave(() => enterPick(state.sample.id), 'new'));
  // Compare, never greyed out either: without a run, or with the run's mixing check off, a press
  // says why (it reads every column with the check's model).
  $('compare-button').textContent = COMPARE_TEXT.button; $('compare-button').title = COMPARE_TEXT.buttonTooltip;
  guard($('compare-button'), () => state.mode !== 'run' || !state.run ? [problem(null, '', COMPARE_TEXT.needsRun)]
    : qcClient.view?.enabled !== true ? [problem(null, '', COMPARE_TEXT.needsCheck)] : [], () => openCompare());
  $('fit').onclick = fitView;
  segmented($('region-mode'), setRegionMode);
  $('prior-choice').onchange = () => {
    clearProceedNotes();
    const select = $('prior-choice'), choices = state.priorChoices || [];
    select.title = select.options?.[select.selectedIndex]?.textContent || '';
    const choice = choices[Number(select.value)];
    state.prior = choice ? { ...choice.prior } : { cells: null, nuclei: null };
    resetPreview(); previewPrior(); updatePicker();
  };
  $('show-prior').onchange = () => { resetPreview(); previewPrior(); };
  // Start is never greyed out either: a press lists what is missing (pickerProblems), with a closed section that holds one
  // opened; only the request on its way keeps it busy ("Starting…").
  guard($('start'), () => {
    const found = pickerProblems();
    for (const p of found) { const section = p.field && $(p.field)?.closest?.('details'); if (section && !section.open) section.open = true; }
    return found;
  }, async () => {
    const done = busy($('start'), PICKER_TEXT.starting), ticket = opening, kind = $('run-kind').value;
    ownOpens++;          // the engine pushes the new run before its answer reaches this press: it is this window's own
    try {
      const sample = state.sample.id, region = state.region;
      let run;
      if (['stain','patches'].includes(kind)) {
        // Training patches: the boxes on the image as they are now (moved, resized or dropped).
        const request = patchRequest(), stain = request.stain, boxes = kind === 'patches' ? await previewedBoxes(request) : null;
        if (ticket !== opening) return;
        // Refused for these settings when the boxes were suggested (e.g. no patches with image detail): that reason stays
        // under Start and nothing is created.
        if (kind === 'patches' && patchPreview.refused === JSON.stringify(request)) {
          $('picker-error').hidden = false; $('picker-error').textContent = patchPreview.refusal;
          return;
        }
        // The boxes as they arrived for this press (a change was still waiting for its own): one moved outside the region
        // they are for, e.g. kept from an earlier answer for a larger region, is named as a press names it; nothing is made.
        const outside = boxes ? boxProblems(boxes) : [];
        if (outside.length) { explain($('start'), outside); return; }
        $('picker-error').hidden = true;
        run = await api.createStain(sample, region, boxes ? { ...stain, boxes } : stain, kind);
      } else {
        $('picker-error').hidden = true;
        // proseg's own schedule, paused on the prior until Resume: nothing to choose.
        run = await api.createRun(sample, region, state.prior, options.getValues());
      }
      options.save?.();     // a run started with them: the picker's options from now on (a fill-in too)
      if (ticket === opening) await openRun(run.id);
    } catch (error) {
      if (ticket === opening) { $('picker-error').hidden = false; $('picker-error').textContent = String(error.message || error); }
      else notice(String(error.message || error));
    } finally {
      ownOpens--; done(); startLabel();   // the run type may have changed meanwhile
      if (ticket === opening) updatePicker();   // else a run or sample opening sets the page up (nothing asked for here)
    }
  });
  segmented($('view'), (value) => { globalThis.sopaCompanionLog?.add('action', `view ${value === '3d' ? '3-D' : '2-D'}`); setView(value === '3d'); });
  $('transcripts').onchange = () => { stage.set({ transcripts: $('transcripts').checked }); layerPanel.save(); layerPanel.render(); };
  $('z-stretch').oninput = () => {
    const value = 2 ** Number($('z-stretch').value);
    stage.set({ stretch: value });
    $('z-stretch-value').textContent = `×${fmt1.format(1.5 * value)} real`;
  };
  segmented($('mode'), (value) => { globalThis.sopaCompanionLog?.add('action', `display ${value}`); setMode(value); });
  segmented($('speed'), (value) => control('speed', value));
  $('trails').onchange = () => stage.set({ trails: $('trails').checked });
  $('fill').oninput = () => stage.set({ fill: Number($('fill').value) });
  $('gain').oninput = () => stage.set({ gain: Number($('gain').value) });
  $('run-toggle').onclick = () => control(['stain','patches'].includes(state.run?.kind) ?
    (state.run.paused || !ACTIVE.includes(state.run.state) ? 'resume' : 'pause') : state.run?.state === 'paused' ? 'resume' : 'pause');
  $('run-step').onclick = () => { setLive(true); control('step'); };
  $('run-stop').onclick = finishAndSave;
  $('run-abort').onclick = () => { if (confirm('Abort this run without saving results?')) control('abort'); $('run-abort').parentElement.open = false; };
  $('first').onclick = () => { stopReplay(); setLive(false); show(0); };
  $('prev').onclick = () => step(-1);
  $('next').onclick = () => step(1);
  $('last').onclick = () => { stopReplay(); show(state.frames.length - 1); };
  $('live').onclick = () => setLive(!state.live);
  $('replay').onclick = () => startReplay();
  // A collapsed "This iteration" panel is not updated for every frame: bring it up to date when it opens.
  $('now-panel').addEventListener('toggle', () => { if ($('now-panel').open) { updateFacts(); updateCellCard(); updateActiveCells(); } });
  $('qc-panel').addEventListener('toggle', () => { if ($('qc-panel').open) updateCellRow(); });   // its row for the cell likewise
  document.addEventListener('keydown', (e) => {
    // A key a panel or the picker's outline has taken (e.g. Escape dropping an outline) is not the page's as well.
    if (e.defaultPrevented || document.querySelector('dialog[open]') || e.target.closest('input, select, textarea') || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.target.closest('button') && [' ', 'Enter'].includes(e.key)) return;
    if (state.run?.kind === 'patches' && !['f', 'Escape'].includes(e.key)) return;
    if (['stain','patches'].includes(state.run?.kind) && ![' ', 'f', 'Escape'].includes(e.key)) return;
    // Held at the end of proseg's standard run, Space and S do not leave it (that gives up proseg's own result for good):
    // the note says so, and leaving takes a click on Resume or Step.
    const keys = {
      ' ': () => heldAtEnd(state.run) ? flashNote() : state.run && ACTIVE.includes(state.run.state) && $('run-toggle').click(),
      s: () => heldAtEnd(state.run) ? flashNote() : state.run?.state === 'paused' && $('run-step').click(),
      ArrowLeft: () => step(e.shiftKey ? -10 : -1), ArrowRight: () => step(e.shiftKey ? 10 : 1),
      Home: () => $('first').click(), End: () => $('last').click(),
      v: () => setView(!stage.orbit.enabled),
      x: () => $('transcripts').click(),
      '[': () => resizeTranscripts(1/1.2), ']': () => resizeTranscripts(1.2),
      l: () => $('live').click(), r: () => startReplay(), f: () => fitView(),
      1: () => setMode('cells'), 2: () => setMode('outlines'),
      t: () => { $('trails').checked = !$('trails').checked; stage.set({ trails: $('trails').checked }); layerPanel.save(); },
      // Colour by: Mixing (the panel's Show on the cells); while it cannot be shown, the key says why in plain words.
      m: () => { if (stage.options.cellColorBy === 'mixing' || layerPanel.mixingAvailable) setMixingColours(stage.options.cellColorBy !== 'mixing'); else if (state.mode === 'run' && state.run) notice(layerPanel.mixingWhy || QC_TEXT.mixingOffTooltip); },
      Escape: () => { state.selected = 0; stage.set({ selected: 0 }); updateCellCard(); },
      // Go deeper: open or close the window for the selected cell (also while the mixing check is collapsed; Esc closes
      // it only while focus is inside it).
      d: () => deeper.toggle(state.selected, { from: null }),
    };
    const action = keys[e.key] || keys[e.key.toLowerCase?.()];
    if (action && state.mode === 'run') { e.preventDefault(); action(); }
  });
}

function resizeTranscripts(factor){stage.set({transcriptSize:Math.max(.5,Math.min(12,stage.options.transcriptSize*factor))});layerPanel.save();layerPanel.render();}

/** A key that would leave the end of proseg's standard run: the note stands out for a moment (a static highlight, two
 *  frames: nothing animates by itself) so the click it asks for is the way on. */
let noteTimer = null;
function flashNote() {
  const note = $('prior-hint');
  if (note.hidden) return;
  note.classList.add('attention');
  clearTimeout(noteTimer); noteTimer = setTimeout(() => note.classList.remove('attention'), 1200);
}

function setView(on, animate = true) {
  if(on && state.run?.mode==='tiled' && $('view').querySelector('[data-value="3d"]').disabled)return;
  stage.orbit.switch(on, animate);
  stage.set({ hover: 0 });
  state.hover = 0;
  $('tooltip').hidden = true;
  $('z-stretch-row').hidden = !on;
  $('layers-row').hidden = on || (state.run?.info?.layers || 1) < 2;
  for (const b of $('view').children) b.setAttribute('aria-pressed', String((b.dataset.value === '3d') === on));
  drawOverlay();
}

/** Cells (filled, key 1) or Outlines (key 2): stage options.mode 0 or 1. Any other name (a mode that no longer exists)
 *  shows Cells. */
function setMode(value) {
  if (value !== 'outlines') value = 'cells';
  state.display.mode = value;
  stage.set({ mode: value === 'outlines' ? 1 : 0 });
  layerPanel.save();
  for (const b of $('mode').children) b.setAttribute('aria-pressed', String(b.dataset.value === value));
}

function segmented(group, onChange) {
  group.addEventListener('click', (e) => {
    const button = e.target.closest('button');
    if (!button || button.disabled) return;
    for (const b of group.children) b.setAttribute('aria-pressed', String(b === button));
    onChange(button.dataset.value);
  });
}

let noticeTimer = null;
function notice(text) {
  // A page signed out, or closed, says so on its first screen: nothing more is said over it.
  if (state.signedOut || state.closed) return;
  // Over Compare (a modal dialog, in the top layer) the notice shows inside its view, else where the page keeps it.
  const box = $('notice'), home = notice.home ??= box.parentElement, host = covered ? compareView.root : home;
  if (host && box.parentElement !== host) host.append(box);
  box.hidden = false;
  box.textContent = text;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => { if (box.textContent === text) box.hidden = true; }, 5000);
}

// ---- telemetry: every 10 s while a run is open, how this page renders (telemetry.mjs; no cell ids or paths) --------
const telemetry = new Telemetry({
  context: () => state.mode === 'run' && state.run ? {
    mode: state.run.mode === 'tiled' ? 'tiled' : 'detail', state: state.run.state, cells: state.run.info?.cells ?? null,
    live: !!state.live, umap: !!analysis.enabled, view: stage.orbit?.enabled ? '3d' : '2d',
    level: state.run.mode === 'tiled' ? tiled.plan?.level ?? null : null,
    stage: [stage.canvas.width, stage.canvas.height], renderer: stage.renderer || null,
  } : null,
  counters: () => ({ stage: stage.drawCount, umap: analysis.view?.drawCount, timeline: timeline.drawCount, batches: paint.flushes, snapshots: counts.snapshots }),
}).start();

function layout() {
  const picking = state.mode === 'pick', running = state.mode === 'run';
  const stain = running && ['stain','patches'].includes(state.run?.kind);
  $('app').classList.toggle('setup', !running);
  $('picker').hidden = !picking;
  $('empty').hidden = state.mode !== 'empty';
  renderTool();   // the first screen's cards or the chosen tool, and the header's tool name
  $('dock').hidden = !running;
  $('hud').hidden = !running || state.current < 0;
  $('hud-state').hidden = true;
  $('now-panel').hidden = !running || state.current < 0;
  $('run-panel').hidden = !running;
  $('display-panel').hidden = !running;
  $('fit').hidden = state.mode === 'empty';
  $('knobs-panel').hidden = !running || stain;
  $('qc-panel').hidden = !running || stain;
  $('qc-chip').hidden = !running || !qcChip.visible;   // in a run: as its state says (qc-chip.mjs)
  for (const button of $('view').children) button.disabled = !running;
  for (const id of ['view', 'mode', 'run-step', 'timeline']) $(id).hidden = stain;
  for (const id of ['transcripts', 'trails', 'fill', 'speed', 'first', 'run-abort']) $(id).parentElement.hidden = stain;
  if (!stain) { $('run-toggle').hidden=$('run-stop').hidden=false; setText($('run-stop'), 'Finish & save'); setProp($('run-stop'), 'title', "Finish sampling and save proseg's results from here"); }
  $('stage').parentElement.classList.toggle('picking', picking);
  stage.resize();
  drawOverlay();
  requestAnimationFrame(() => timeline.refresh());
}

function readStain() {
  const patches=$('run-kind').value==='patches', empty=patches && $('patch-start-from').value==='empty';
  // Empty runs no model: its hidden Cellpose tuning is not read but sent as Cellpose's defaults (never named by Start).
  const value=stainValues(id => id === 'stain-channels' ? [...$(id).selectedOptions].map(o => o.value) :
    id==='stain-method' && empty ? 'cellpose' : empty && Object.hasOwn(CELLPOSE_DEFAULTS, id) ? CELLPOSE_DEFAULTS[id] :
    ['stain-auto', 'stain-resolve', 'stain-tissue', 'stain-augment', 'stain-normalize', 'stain-fixed-range', 'stain-invert'].includes(id) ? $(id).checked : $(id).value);
  if(patches) {
    Object.assign(value,{start_from:$('patch-start-from').value,count:Number($('patch-count').value),size:Number($('patch-size').value)});
    if(value.method==='instanseg' && value.model!=='brightfield_nuclei')value.channels=[$('stain-nucleus').value,$('stain-cytoplasm').value].filter(Boolean);
    if(value.method==='instanseg') Object.assign(value,{output:value.model==='fluorescence_nuclei_and_cells'?'cells':'nuclei',resolve_cells:false});
  }
  value.level = Number($('stain-level').value || 0);
  return value;
}
function stainAvailable() {
  return state.health?.stain?.methods?.[$('stain-method').value] ?? state.health?.stain?.available;
}
/** The Method choice holds the methods the engine offers, each by its name (stain.mjs stainMethods: Cellpose alone
 *  unless the engine's internal switch turns InstanSeg on); a choice it no longer holds goes back to the first. */
function offerStainMethods() {
  const select = $('stain-method'), offered = stainMethods(state.health);
  if ([...select.children].map(option => option.value).join() !== offered.join())
    select.replaceChildren(...offered.map(method => new Option(STAIN_TEXT[method], method)));
  if (!offered.includes(select.value)) select.value = offered[0];
}
function stainModelKey() {
  return $('stain-method').value === 'instanseg' ? 'sopa-companion-instanseg-model' : 'sopa-companion-stain-model';
}
function updateStainMethod() {
  const patches=$('run-kind').value==='patches', empty=patches && $('patch-start-from').value==='empty';
  const instanseg = !empty && $('stain-method').value === 'instanseg';
  $('patch-picker').hidden=!patches;
  // The Method row only when there is a choice (offerStainMethods): with Cellpose alone it is never shown.
  $('stain-method').parentElement.hidden=empty || stainMethods(state.health).length<2;
  $('stain-model').parentElement.hidden=empty;
  $('stain-tuning').hidden=empty || instanseg;
  if(!instanseg && !empty && state.mode==='pick' && pickerSettings.sample!==state.sample?.id)refreshCellposeSettings();
  const brightfield=instanseg && $('stain-model').value==='brightfield_nuclei';
  $('stain-cellpose-options').hidden = instanseg && (!patches || brightfield);
  $('stain-instanseg-options').hidden = !instanseg || patches && !brightfield;
  $('stain-instanseg-tuning').hidden=patches;
  const dual = !instanseg || !$('stain-model').value || $('stain-model').value === 'fluorescence_nuclei_and_cells';
  $('stain-output').disabled = $('stain-resolve').disabled = !dual;
  if (!dual) { $('stain-output').value = 'nuclei'; $('stain-resolve').checked = false; }
  else if (updateStainMethod.dual === false) $('stain-output').value = 'cells';
  updateStainMethod.dual = dual;
  renderResolution();
}
function updateRunKind() {
  const stain = ['stain','patches'].includes($('run-kind').value), patches = $('run-kind').value === 'patches';
  $('stain-picker').hidden = !stain;
  $('proseg-picker').hidden = $('proseg-advanced').hidden = stain;
  if ($('patch-tool').open !== patches) $('patch-tool').open = patches;   // its toggle then finds the run type set
  startLabel();
  if (stain) { $('picker-qc').hidden = true; updateStainMethod(); if($('patch-start-from').value!=='empty' || $('run-kind').value!=='patches')loadStainModels(); }
  else if (state.region?.kind === 'full') $('region-help').textContent = 'The whole sample will be segmented. Zoom in to inspect the prior.';
  $('picker-error').hidden = true;
}
/** Training patches, a tool inside Stain segmentation: its section opened or closed sets the run
 *  type, as the old Run type choice did (the boxes are suggested while it is open, and Start makes the item). */
function patchToolToggled() {
  const kind = $('run-kind').value, want = $('patch-tool').open ? 'patches' : 'stain';
  if (kind === 'proseg' || kind === want) return;   // set from the run type itself (updateRunKind), or not in Stain segmentation
  globalThis.sopaCompanionLog?.add('action', `Training patches ${want === 'patches' ? 'opened' : 'closed'}`);
  $('run-kind').value = want; $('run-kind').onchange();
}
/** Start's label for the run type chosen; while Start is busy its "Starting…" stays, and this runs again when it ends. */
function startLabel() {
  if ($('start').getAttribute('aria-busy') !== 'true') $('start').textContent = $('run-kind').value === 'patches' ? STAIN_PATCH_TEXT.start : STAIN_TEXT.start;
}
function loadStainModels() {
  const method=$('stain-method').value;
  if(loadStainModels.pending===method)return loadStainModels.promise;
  return loadStainModels.promise=fetchStainModels();
}
/** While the model list loads for the first time, or again after the method changed (its first answer can take a while:
 *  the first import of Cellpose and PyTorch on a new install): a static spinner beside the list and, in the empty list,
 *  its words; both go when the list comes, or cannot be read. A refresh of a list already shown says nothing. */
function showModelsLoading(on) {
  const select = $('stain-model'), spinner = $('stain-model-loading'), shown = showModelsLoading.option;
  if (spinner) spinner.hidden = !on;
  if (on && !select.children.length) {
    const option = new Option(STAIN_TEXT.modelsLoading, '', true, true);
    option.disabled = true; select.append(option); showModelsLoading.option = option;
  } else if (!on && shown) {
    if (Array.from(select.children).includes(shown)) shown.remove();
    showModelsLoading.option = null;
  }
}
async function fetchStainModels() {
  const method = $('stain-method').value;
  if (loadStainModels.pending === method || !stainAvailable()) return;
  const ticket = (loadStainModels.ticket || 0) + 1;
  loadStainModels.ticket = ticket;
  loadStainModels.pending = method; state.stainModelError = '';
  if (!state.stainModels) showModelsLoading(true);
  try {
    const result = await api.stainModels(method);
    if (ticket !== loadStainModels.ticket || method !== $('stain-method').value) return;
    state.stainModels = result.models;
    const select = $('stain-model');
    let preferred = select.value;
    if (!preferred) try { preferred = localStorage.getItem(stainModelKey()); } catch { /* optional storage */ }
    const key = JSON.stringify([method, result.models]);
    if (loadStainModels.key !== key) {
      loadStainModels.key = key; select.replaceChildren();
      for (const model of result.models.filter(m => m.kind === 'builtin')) select.append(new Option(model.name, model.name));
      const custom = result.models.filter(m => m.kind === 'custom');
      if (custom.length) {
        const group = document.createElement('optgroup'); group.label = STAIN_TEXT.yourModels;
        for (const model of custom) group.append(new Option(modelLabel(model), model.name));
        select.append(group);
      }
    }
    const fallback = method === 'instanseg' ? 'fluorescence_nuclei_and_cells' : 'cpsam';
    select.value = result.models.some(m => m.name === preferred) ? preferred :
      result.models.some(m => m.name === fallback) ? fallback : result.models[0]?.name || '';
    updateStainMethod();
  } catch {
    if (ticket !== loadStainModels.ticket || method !== $('stain-method').value) return;
    state.stainModels = []; state.stainModelError = STAIN_TEXT.modelsError;
  } finally {
    if (ticket === loadStainModels.ticket) {
      loadStainModels.pending = false;
      showModelsLoading(false);
      updatePicker();
      clearTimeout(loadStainModels.timer);
      loadStainModels.timer = setTimeout(refreshStainModels, 15000);
    }
  }
}
function refreshStainModels() {
  if($('run-kind').value==='patches' && $('patch-start-from').value==='empty')return;
  if (state.mode === 'pick' && ['stain','patches'].includes($('run-kind').value) && !document.hidden) loadStainModels();
}
function fillResolution() {
  const image = state.sample?.images?.find(image => image.key === $('stain-image').value), select = $('stain-level');
  // The image's pyramid levels by their SpatialData names (stain.mjs resolutionChoices: scale0 alone for an image without
  // them), keeping the physical resolution chosen before when this image has it (matchResolution), else scale0.
  select.replaceChildren(...resolutionChoices(image).map(scale => Object.assign(new Option(scale.label, String(scale.level)), { title: scale.title })));
  select.value = String(matchResolution(image, select.resolution));
  renderResolution();
}
function renderResolution() {
  const select = $('stain-level'), image = state.sample?.images?.find(image => image.key === $('stain-image').value);
  const scales = resolutionChoices(image), scale = scales.find(scale => String(scale.level) === select.value);
  if (!scale) {
    // No image (a sample without one), or no level of it: nothing of an earlier image's resolution stays on screen, nor
    // in the patch note (updatePicker), which reads select.resolution.
    select.resolution = null; select.title = '';
    $('stain-resolution-info').textContent = $('stain-resolution-info').title = '';
    $('stain-resolution-warning').textContent = $('stain-resolution-warning').title = ''; $('stain-resolution-warning').hidden = true;
    return;
  }
  select.resolution = scale;
  // The shared readout (stain.mjs resolutionReadout): the size of what the run outlines at this level, which is nuclei
  // for Cellpose without a cell body channel or InstanSeg asked for nuclei (the sample's matching boundaries), else cells.
  // Find (#stain-output) is InstanSeg's alone: Cellpose never reads it, even when an earlier InstanSeg choice left it.
  const method = $('stain-method').value;
  const readout = resolutionReadout(image, { level: scale.level, method,
    diameter: method === 'cellpose' && !$('stain-auto').checked ? Number($('stain-diameter').value) : null,
    cytoplasm: $('stain-cytoplasm').value, output: method === 'instanseg' ? $('stain-output').value : null }, { sample: state.sample });
  $('stain-resolution-info').textContent = readout.text; $('stain-resolution-info').title = readout.title;
  $('stain-resolution-warning').textContent = readout.warning; $('stain-resolution-warning').hidden = !readout.warning;
  $('stain-resolution-warning').title = readout.warningTitle || '';
  select.title = scales.length === 1 ? STAIN_TEXT.resolutionSingle : STAIN_TEXT.resolutionHelp;
}
function fillStainImages(sample) {
  const select = $('stain-image'); select.replaceChildren();
  for (const image of sample.images || []) select.append(new Option(image.key, image.key));
  select.value = sample.image_key || sample.images?.[0]?.key || '';
  fillStainChannels(); updateRunKind();
}
function fillStainChannels() {
  fillResolution();
  const names = state.sample?.images?.find(i => i.key === $('stain-image').value)?.channels || [];
  $('stain-nucleus').replaceChildren(); $('stain-cytoplasm').replaceChildren();
  $('stain-cytoplasm').append(new Option(STAIN_TEXT.none, ''));
  for (const name of names) for (const id of ['stain-nucleus', 'stain-cytoplasm']) $(id).append(new Option(name, name));
  $('stain-nucleus').value = names.find(n => /dapi|hoechst/i.test(n)) || names[0] || '';
  $('stain-cytoplasm').value = '';
  $('stain-channels').replaceChildren();
  for (const name of names) $('stain-channels').append(new Option(name, name, true, true));
  updateStainMethod();
}

function refreshCellposeSettings() {
  if(document.hidden)return;
  if(state.mode==='pick' && ['stain','patches'].includes($('run-kind').value) && $('stain-method').value==='cellpose')pickerSettings.refresh(state.sample?.id);
  if(['stain','patches'].includes(state.run?.kind))patchSettings.refresh(state.run.sample);
}
async function loadStainPreview() {
  tissueView.update(state.sample?.id,$('stain-image').value,
    state.mode==='pick' && ['stain','patches'].includes($('run-kind').value) && $('stain-tissue').checked);
  if(state.mode==='pick' && pickerSettings.sample!==state.sample?.id) {
    $('stain-settings-report').replaceChildren();refreshCellposeSettings();
  }
  loadStainPreview.controller?.abort();
  if (!['stain','patches'].includes($('run-kind').value)) {
    if(state.overviewImage)stage.setImage(state.overviewImage.rect,state.overviewImage.channels);
    return;
  }
  const sample = state.sample, settings = readStain();
  const image = sample?.images?.find(i=>i.key===settings.image);
  const channel = image?.channels.indexOf(settings.method === 'instanseg' ? settings.channels[0] : settings.nucleus);
  if(!image || channel<0){stage.setImage(null, []);return;}
  const controller = loadStainPreview.controller = new AbortController(), ticket=opening;
  try {
    const response=await fetch(`/api/v1/samples/${encodeURIComponent(sample.id)}/image/${encodeURIComponent(image.key)}/c${channel}.u16?box=${sample.extent.join(',')}&px=1024`,{signal:controller.signal});
    if(!response.ok)throw Error(STAIN_TEXT.previewError);
    const value=stainPreviewImage(JSON.parse(response.headers.get('X-Image')),await response.arrayBuffer());
    if(controller.signal.aborted || ticket!==opening || state.mode!=='pick' || !['stain','patches'].includes($('run-kind').value))return;
    stage.setImage(value.rect,value.channels);
  } catch(error) {
    if(error.name!=='AbortError' && ticket===opening && state.mode==='pick')notice(STAIN_TEXT.previewError);
  }
}

// ---- closing the app ---------------------------------------------------------------------------
function pressClose() {
  if (state.ended) return closeApp();
  if (state.session) return closeSession();
  return leave(() => { enterPick(null); closeApp(); }, 'close');
}
/** Close: the page says goodbye (the engine closes once the last page said goodbye), then closes its window; in a tab
 *  the page may not close, it says the app has closed (appClosed), or, with the app still open in another tab (the
 *  engine asked first), that this tab is closed. A page not signed in closed nothing of the app: it closes its window
 *  if it may, and its screen goes on saying how to open the app again. */
function closeApp() {
  if (state.signedOut) return closeWindow(window);
  const others = otherPages();
  socket.goodbye?.(); closeWindow(window, () => { closePopup.reset(); others.then(appClosed); });
}
/** How many pages but this one the engine has open now (GET /health presence.pages; this tab's own socket, while it is
 *  open, is one of them); 0 when it does not say within CLOSE_CHECK_MS (an engine going already). */
function otherPages() {
  if (state.ended) return Promise.resolve(0);
  const mine = socket.connected?.() ? 1 : 0, controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CLOSE_CHECK_MS);
  return api.health(controller.signal).then(health => {
    const pages = health?.presence?.pages;
    return Number.isInteger(pages) ? Math.max(0, pages - mine) : 0;
  }, () => 0).finally(() => clearTimeout(timer));
}
/** The app has closed, in a tab the page could not close: the first screen says so (with `others` pages still open: the
 *  app goes on there, only this tab is closed), with nothing more to press (Reload hidden, never greyed out), and the
 *  socket is not opened again. */
function appClosed(others = 0) {
  state.closed = true; socket.stop?.(); state.engineDown = true; state.mode = 'empty'; state.tool = null;
  setText($('engine-down-title'), others ? SHELL_TEXT.closedHereTitle : SHELL_TEXT.closedTitle);
  setText($('engine-down-why'), others ? SHELL_TEXT.closedHere : SHELL_TEXT.closedTab);
  setProp($('engine-reload'), 'hidden', true); layout();
}
/** A notebook's session: Close ends the session. A run whose result is not added yet is asked
 *  about first (Add to your SpatialData and close, Discard and close, Keep working); the engine then adds it or discards
 *  it as the session closes (POST /closing {answer, end: true}), and the page says the session has ended. A leave being
 *  asked or carried out (New run, the brand…) keeps its own prompt, as on the desktop: the window's ✕ never turns it into
 *  this one, whose Keep working would leave that leave unanswered. */
async function closeSession() {
  if (closePopup.open && closePopup.reason === 'close') return;
  if (leaving) return leaving.promise;
  try { await settleExclusions(); } catch { /* the run as it stands */ }
  const run = state.run && pending([state.run]).length ? state.run : null;
  if (!run) { try { await answerClose('discard', null, nextRevision()); } catch { /* said in the notice */ } return; }
  closePopup.show(run, { reason: 'close' });
}
async function answerClose(answer, run, revision) {
  if (answer === 'stay') return;
  globalThis.sopaCompanionLog?.add('action', `close the session: ${answer}`);
  try { await api.closing(answer, { end: true, runs: run ? [run.id] : [], revision }); }
  catch (error) { if (!run) notice(error.message); throw error; }
  closePopup.reset(); sessionEnded(true);
}

// ---- boot -----------------------------------------------------------------------------------------------------
// The page's first three requests (the engine's health, the recent samples, the runs) are given up after BOOT_MS: one
// that never answered left an empty first screen with no way out. The first screen then says so, with
// Reload, in place of the tools.
const BOOT_MS = 20000;
/** The answers to the requests `asks(signal)` starts, in order; rejected when one fails or when they have not all
 *  answered within `ms` (`late`: true). Either way the requests still on their way are cancelled. */
function answeredWithin(asks, ms) {
  const controller = new AbortController(), answers = Promise.all(asks(controller.signal));
  let timer = 0;
  const late = new Promise((_, reject) => { timer = setTimeout(() => reject(Object.assign(new Error('no answer'), { late: true })), ms); });
  return Promise.race([answers, late]).finally(() => { clearTimeout(timer); controller.abort(); });
}
/** The engine did not answer the first requests, or refused them: the first screen says so (with what it answered, if
 *  anything) and offers Reload; the tools wait for it. A page signed out says that instead (signedOut). */
function engineDown(error) {
  if (state.signedOut) return;
  globalThis.sopaCompanionLog?.add('action', `the engine did not answer: ${error?.late ? `nothing within ${BOOT_MS / 1000} s` : error?.message || error}`);
  state.engineDown = true; state.mode = 'empty'; state.tool = null;   // a card chosen meanwhile: nothing to open now
  setText($('engine-down-title'), SHELL_TEXT.engineDown);
  const failure = openFailure(error);      // its words; an answer with none: the shell's (no sample is involved yet)
  setText($('engine-down-why'), error?.late ? SHELL_TEXT.engineLate.replace('{seconds}', BOOT_MS / 1000) :
    failure.text === NOTE_TEXT.engineFailed ? SHELL_TEXT.engineFailed : failure.text);
  setText($('engine-reload'), SHELL_TEXT.reload);
  layout();
  $('engine-reload').focus?.();
}
/** This tab is not signed in (page-key.mjs; `reason` as its reason()): it has no key, its link was used already or too
 *  late ('expired'), or the engine refused its key (it started again since, with a new one). The first screen says so
 *  in place of the tools, with how to open the app again (a notebook's session: from the notebook); nothing more is
 *  asked of the engine, and there is nothing to press (Reload would sign nothing in: hidden, never greyed out). Once. */
function signedOut(reason) {
  if (state.signedOut) return;
  state.signedOut = reason || 'none';
  globalThis.sopaCompanionLog?.add('action', `the page is not signed in (${state.signedOut})`);
  socket.stop?.();
  closePopup.reset(); if (zarrSave.dialog?.open) zarrSave.dialog.close(); if (olderPrompt.open) olderPrompt.finish();
  if (state.mode !== 'empty') { leaving = null; enterPick(null); }
  const deck = notebookTab() ? SESSION_TEXT : SHELL_TEXT, used = state.signedOut === 'expired';
  state.engineDown = true; state.mode = 'empty'; state.tool = null;
  setText($('engine-down-title'), used ? deck.linkUsedTitle : deck.signedOutTitle);
  setText($('engine-down-why'), used ? deck.linkUsed : deck.signedOut);
  setProp($('engine-reload'), 'hidden', true);
  layout();
}
/** Whether this tab is a notebook session's: the engine said so, or the tab's link named the session (storage-boot.mjs
 *  keeps that in the tab, since the page rewrites its address). */
function notebookTab() {
  if (state.session) return true;
  try { if (new URLSearchParams(location.search).has('api')) return true; } catch { /* no address */ }
  try { return !!sessionStorage.getItem(MARK); } catch { return false; }
}
async function boot() {
  setupPointer();
  setupControls();
  setMode(stage.options.mode === 1 ? 'outlines' : 'cells');
  layout(); renderRecent();
  window.addEventListener('resize', () => { drawOverlay(); resizeCellCard(); });
  // A hidden page neither polls nor follows new frames; it shows the newest one when it is visible again.
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {refreshStainModels();refreshCellposeSettings();}
    if (!document.hidden && state.mode === 'run') {
      if(['stain','patches'].includes(state.run?.kind)){stainPatches.load();stainView.refresh();}else if(state.live)follow();
    }
  });
  window.addEventListener('focus', refreshStainModels);
  window.addEventListener('focus', refreshCellposeSettings);
  // Signed in first (page-key.mjs): a tab without a key asks the engine nothing and says why; an engine that could not
  // be reached to sign in is said as any engine that does not answer (a failed fetch).
  if (!await pageKey.ready) {
    if (pageKey.reason() === 'unreachable') engineDown(new TypeError('no answer')); else signedOut(pageKey.reason());
    return;
  }
  try {
    [state.health, state.samples, state.runs] = await answeredWithin(signal => [api.health(signal), api.samples(signal), api.runs(signal)], BOOT_MS);
    state.runs = state.runs.filter(run => !run.qc_test?.of);   // a test square's own runs are never opened or listed
  } catch (error) {
    // A key the engine refused (it started again since): signed out (page-key.mjs says so too), not unreachable.
    if (error?.status === 401) signedOut('refused'); else engineDown(error);
    return;
  }
  // A notebook's session (health.session): the bound sample, Add to your SpatialData, Close ends the session.
  setSession(apiSession(state.health));
  offerStainMethods();
  // The last session ended unexpectedly (the engine's or this page's): a popup offers the log saved then
  // (crash-report.mjs). Under ?harness=1 nothing is asked or shown (its scenarios press on the page, which the popup
  // would cover): a scenario shows it with window.sopaCompanionCrash.check().
  globalThis.sopaCompanionCrash?.check({ api, quiet: !!harness });
  // An engine without the GPU worker's environment: runs open without the UMAP pane (it never flashes open and shut),
  // and Layers > Cells says where that environment is set: the app's config.toml, or a notebook's own settings.
  analysis.engineCan = state.health?.analysis !== false;
  analysis.notebook = !!state.session;
  analysis.prosegFix = state.health?.proseg_fix ?? null;   // how to update an old proseg-live (its line's tooltip)
  const [schemaResult, guideResult] = await Promise.allSettled([api.options(), api.optionsGuide()]);
  if (schemaResult.status === 'fulfilled') {
    options.load(schemaResult.value);
  } else {
    $('proseg-options').textContent = 'Options could not be loaded. Reload to retry.';
  }
  const guide = guideResult.status === 'fulfilled' ? guideResult.value : [];
  pickerGuide.load(guide, options.schema);
  knobs.guide.load(guide, options.schema);
  renderRecent();
  let refreshingRuns = false;
  setInterval(async () => {
    if (document.hidden || covered || refreshingRuns || state.ended || state.signedOut || state.closed) return;
    refreshingRuns = true;
    const watched = state.run?.id;
    try {
      if(watched) {
        const id=watched,run=await api.run(id);
        if(state.run?.id===id)onMessage({ev:'run',run});
      } else onMessage({ ev: 'runs', runs: await api.runs() });
    } catch (error) { if (error.status === 404 && state.run?.id === watched) runGone(watched); }
    finally { refreshingRuns = false; }
  }, 1500);
  const wanted = new URLSearchParams(location.search).get('run');
  // Runs older versions made and never saved: listed before anything opens, the start notice in the prompt's lead;
  // Later (or the last row answered) lets the page go on. A notebook's session starts empty: never there.
  const leftovers = leftoversNotice(state.health?.leftovers);
  if (!state.session && olderCount(state.health?.older) > 0) {
    let older = state.health.older;
    try { older = await api.olderRuns(); } catch { /* an engine without GET /runs/older: health's own rows */ }
    if (olderRows(older).length) await olderPrompt.ask(older, { lead: leftovers });
    else if (leftovers) notice(leftovers);
  } else if (leftovers) notice(leftovers);
  state.booted = true;
  if (state.ended || state.signedOut) return;
  if (wanted) {
    try { return await openRun(wanted); }
    catch (error) { if (error.status !== 404) throw error; history.replaceState(null, '', location.pathname); }
  }
  {
    const current = state.health?.one_run ? state.runs[0] : state.runs.find(run => ACTIVE.includes(run.state));
    if (current) return openRun(current.id);
  }
  // tool=: a notebook's session opens that tool's picker on its sample (once it is ready: setSession waits for it).
  const session = state.session;
  if (session?.tool && session.sample.state === 'ready' && !state.sessionTool) { state.sessionTool = true; return openSessionTool(session.tool); }
  state.mode = 'empty';
  layout();
}

window.sopaCompanion = { state, stage, timeline, paint, telemetry, qc: qcClient, deeper, stain: stainPatches, exclusions:stainExclusions, stainView,
  testCorner,   // the test of a setting in the stage's corner (snapshot(), and the model the panel last gave it)
  compare: compareView,   // Compare's view (snapshot(), texts())
  closing,   // {hold: the window's ✕ asks the browser to hold the window (off under ?harness=1), popup}
  // Every word the mixing check draws on a canvas (the timeline lane, the panel's chart, Go deeper's arrows), and the test
  // corner's and Compare's: the harness checks them for banned words with the page's own text.
  qcTexts: () => [...timeline.qcTexts?.() ?? [], ...qcPanel.chartTexts?.() ?? [], ...deeper.texts(), ...deeperArrows.texts(),
    ...testCorner.texts(), ...compareView.texts()],
  // The Live knobs names a Setting line may hold exactly (qc-copy.mjs settingChecked: the harness takes them out first).
  knobNames: () => (options.schema || []).filter(o => o.live).map(o => knobName(o.label)).filter(Boolean) };
// What the page shows, for the development harness's snapshot (harness-hook.mjs: only with ?harness=1; reads only).
harness?.provide(() => {
  const run = state.run, layers = priors.settings, count = polygonCount, qcView = qcClient.view;
  return {
    page: state.mode, framesShown: counts.snapshots, stage: [stage.canvas.width, stage.canvas.height],
    // Go deeper's suspects glowing on the stage ({on, count, fps}: glow.mjs).
    glow: stage.glow ?? null,
    // Compare's view ({open, state, columns [{key, state, mixed_per100, like100_per100,
    // points}], area, colourBy, kinds, ...}: CompareView.snapshot) and the sample's segmentation layers ({sample, note,
    // loaded: [{key, name, visible, color, opacity, width, fill, state, drawn}]}: PriorLayers.segmentationsShown).
    compare: compareView.snapshot(), segmentations: priors.segmentationsShown(),
    // Go deeper's starting outline as the overlay draws it and the veil inside the lit cells (0-1).
    startOutline: stage.startOutline ? { count: stage.startOutline.count, closed: stage.startOutline.closed } : null,
    innerVeil: stage.innerLevel ?? 0,
    // The tool the page is in ('stain' | 'proseg'; null on the first screen's two cards) and the close popup ({open, runs,
    // answered, sent}: shell.mjs LeavePrompt.snapshot).
    tool: state.tool ?? null, closing: closePopup.snapshot(),
    // The older runs' prompt ({open, runs, answered}: older-runs.mjs) and a notebook's session (null on the
    // desktop: {state, sample (its state), ended: null | 'closing' | 'ended'}).
    older: olderPrompt.snapshot(),
    session: state.session ? { state: state.session.state, sample: state.session.sample.state, ended: state.ended ?? null } : null,
    // Signed in or not (null, else page-key.mjs's reason: 'none' | 'expired' | 'refused'), and closed by Close in a tab
    // the page could not close.
    signedOut: state.signedOut ?? null, closed: !!state.closed,
    run: run && { id: run.id, state: run.state, mode: run.mode === 'tiled' ? 'tiled' : 'detail', ackState: run.ack?.state ?? null,
      frames: state.frames.length, current: state.current, iter: state.frames[state.current]?.iter ?? null, live: !!state.live },
    level: run?.mode === 'tiled' ? tiled.plan?.level ?? null : null,
    // The UMAP pane, and its Advanced settings: the run's settings generation, that of the clusters shown (a change
    // has landed once they agree), whether the settings differ from the defaults, Apply busy, the clusters shown.
    umap: { open: !!analysis.enabled, width: analysis.enabled ? parseFloat(analysis.pane.style.width) || 0 : 0,
      generation: analysis.gen ?? null, applied: analysis.applied?.generation ?? null, changed: !!analysis.settings?.changed,
      busy: !!analysis.settings?.isBusy, nclusters: analysis.applied?.summary?.nclusters ?? null },
    tiled: run?.mode === 'tiled' ? { vectorsLoaded: !!tiled.vectorsLoaded, pending: [...tiled.requests.jobs.keys()] } : null,
    stain: ['stain','patches'].includes(run?.kind)?stainPatches.snapshot():null,
    // Removed areas while their panel is shown: the remove mode (drawing), the change on its way (busy) and how many
    // wait their turn (queued), the Undo and Redo steps there are, and the areas' version and cells removed.
    exclusions: run?.kind === 'stain' && stainExclusions.active ? stainExclusions.snapshot() : null,
    // The run picker: what Start still needs (each problem's field id, or, when it has no field, its
    // words as a press lists them: "Box 2: move it inside the region"), how many Training patches boxes on the image were
    // suggested for the picker as it stands (0 while a change waits for its own), the boxes on the image (or null), and
    // whether Start is busy. The harness waits on these, never on a greyed-out Start.
    picker: state.mode === 'pick' ? { kind: $('run-kind').value,
      problems: pickerProblems().map(p => p.field || (p.label ? `${p.label}: ${p.message}` : p.message)),
      preview: previewMatches(true) ? (stainPatches.previewBoxes()?.length ?? 0) : 0, boxes: stainPatches.previewBoxes() ?? null,
      busy: $('start').getAttribute('aria-busy') === 'true' } : null,
    priors: { cells: count(priors.vectors?.cells), nuclei: count(priors.vectors?.nuclei),
      drawn: !!priors.vectors && priors.shown?.vectors === priors.vectors && priors.vectorsVisible !== false &&
        (!['stain','patches'].includes(run?.kind) || priors.vectors.revision === run.progress?.revision),
      layers: { prior: !!layers.prior.on, nuclei: !!layers.nuclei.on, xenium: !!layers.xenium.on } },
    // The mixing check as shown: the panel's state, the result's frame and iteration (never later than the frame on
    // screen, but looking back at a frame before every result: `later`), the iteration its badge names (`snapped`: the
    // result of another frame, looking back), its headline, the result's area letters (the engine's) and the areas drawn
    // (none: 0), whether the cells are coloured by it, and the suggestion card and readouts.
    qc: run && qcView ? { state: qcView.status, enabled: qcView.enabled ?? null, seq: qcView.summary?.seq ?? null,
      iter: qcView.summary?.iter ?? null, snapped: qcView.snapped ?? null, later: !!qcView.later,
      shown: qcView.atStart ? (Number.isFinite(qcView.start?.per100) ? Math.round(qcView.start.per100) : null) : qcView.summary?.shown ?? null,
      atStart: !!qcView.atStart, start: Number.isFinite(qcView.start?.per100) ? qcView.start.per100 : null,
      areas: (qcView.areas || []).map(a => a.L), areaView: qcView.areaView ?? null, drawn: 0,
      colours: stage.options.cellColorBy === 'mixing', cells: !!qcView.cells,
      // Which result the cell values are from (a readout shows only the result shown's own).
      cellsSeq: qcView.cellsSeq ?? null, cellsCurrent: !!qcView.cellsCurrent,
      // The suggestions as the panel shows them: the card ({rule, areas, flag, to, actions, primary}), the trial ({id,
      // rule, flag, from, to, start_iter, verdict, outcome}) and its verdict.
      card: qcPanel.shown?.card ?? null, trial: qcPanel.shown?.trial ?? null,
      acting: qcActing,
      // The test of a setting as shown ({mode, id, cell, flag, value, current, iter, of, runs, done, phase, corner, start,
      // area, reasons, pool, verdict, rule, hold}: qc-panel.js testShown), only while it shows.
      ...(qcPanel.testShown ? { test: qcPanel.testShown } : {}),
      // Go deeper's window ({open, cell, seq, iter, state, code, shown, parts, learned, foreign, sources, small, rough, ...}).
      deeper: deeper.snapshot(),
      // The test corner ({open, phase, frames, shown {control, test}, glow {on, fps}, ...}: qc-test-corner.mjs).
      testCorner: testCorner.snapshot(),
      // The test square on the overlay ([x0, y0, x1, y1] µm while a test exists, else null: QcOverlay.testSquare).
      testSquare: qcOverlay.testSquare ?? null } : null,
  };
});
boot().catch(error => notice(error.message));
