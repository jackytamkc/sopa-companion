/** All Mixing check display copy. Templates use named values, never HTML.
 * The two technical exceptions are the exact Live knobs label in a
 * Setting line, and the method name inside the collapsed explanation only.
 */
export const BANNED_WORDS = Object.freeze([
  'diffusion', 'posterior', 'ELBO', 'latent', 'amortis', 'amortiz', 'Procrustes',
  'Dirichlet', 'encoder', 'mixture', 'kNN', 'Leiden', 'likelihood', 'contamination',
  'softmax', 'epoch', 'embedding', 'inference', 'Spearman', 'Jaccard', 'Wilson',
  'hysteresis', 'centroid', 'quantile', 'calibrat', 'AUROC', 'voxel', 'prior',
]);

/** Case-insensitive substring checks intentionally also catch the listed stems. */
export function bannedWords(text) {
  const lower = String(text ?? '').toLowerCase();
  return BANNED_WORDS.filter(word => lower.includes(word.toLowerCase()));
}

export const QC_TEXT = Object.freeze({
  title: 'Mixing check',
  // The switch is remembered: new runs start the way it is left (engine/qc.py configure, prefs.json).
  switchTooltip: 'Turn the mixing check on or off. New runs start the way you leave it.',
  subtitle: "Do cells hold RNA that looks like a neighbour's?",
  // The score is a model's reading of each cell's own transcripts, not where they came from, nor "now"; and its
  // neighbour part holds RNA like a neighbour of ANY kind, the same kind included: the model "reads" RNA as coming from
  // neighbouring cells, never "another kind of cell's".
  off: "Off. Turn it on to see, for every cell, how much of its RNA the model reads as coming from neighbouring cells.",
  // The off panel's second line, as the sample stands (GET /runs/{id}/qc sample_state): learned, or learned first.
  offLearned: 'This sample is already learned: results appear within seconds.',
  offNotLearned: "This sample hasn't been learned yet: turning it on learns it first (a few minutes, once; the run carries on meanwhile).",
  learningNote: 'This happens once per sample. Later runs on this sample start at once.',
  // Learning waits only for room on its graphics card (enough free memory, however busy the card is), not for it to be
  // free.
  waiting: 'Waiting for room on the graphics card…',
  // Why learning waits (GET /samples/{id}/qc reason): gpu_busy and gpu_memory are `waiting` above.
  waitingQueue: 'Waiting for its turn to learn this sample…',
  waitingLock: 'Waiting while another sample is being learned on this computer…',
  cardShared: 'The graphics card is shared with another job, so this may take a few minutes longer.',
  waitingStart: "Waiting for proseg's starting cells…",
  waitingFrame: 'Waiting for the next iteration…',
  requested: 'This iteration will be checked as soon as the sample has been learned.',
  unavailableTable: 'The mixing check needs a table of cells with transcript counts in this sample (for example xenium_table).',
  unavailableCounts: "This sample's cell table doesn't hold whole transcript counts, so the mixing check can't learn from it.",
  // What is missing on this computer (GET /health qc.reason, a sample's reason), never naming how the check works:
  // its packages and [qc] python ("software"), Clusters and the UMAP, which score every frame, and [analysis] python
  // ("analysis"), or the graphics card it learns on: none this computer can use ("no_gpu"), a Mac's GPU this PyTorch
  // can't use on its macOS ("apple_gpu": PyTorch 2.15 needs macOS 15), a PyTorch without its CUDA build ("torch_cpu"), a
  // driver too old for that PyTorch ("driver"). Each reads as one sentence with one colon inside
  // Go deeper's line too (qc-deeper-copy.mjs cantRunText).
  unavailableSoftware: "The mixing check needs scvi-tools and scviva-tools, in this Python environment or in one named as python under [qc] in the app's config.toml.",
  unavailableAnalysis: "The mixing check needs Clusters and the UMAP, which can't run with the python under [analysis] in the app's config.toml.",
  unavailableGpu: "The mixing check learns on an NVIDIA graphics card or a Mac's own GPU, and this computer has none it can use.",
  unavailableAppleGpu: "The mixing check learns on this Mac's GPU, which failed a short test with this computer's PyTorch (the Log says why; on macOS 14, installing the app again brings PyTorch 2.14, which passes it).",
  unavailableTorchCpu: "The mixing check learns on an NVIDIA graphics card, which this computer's PyTorch can't use until its CUDA build is installed.",
  unavailableDriver: "The mixing check learns on an NVIDIA graphics card, whose driver is too old for this computer's PyTorch and needs updating.",
  unavailableNoTranscripts: "This finished run didn't keep the positions of its transcripts, which the mixing check needs.",
  unavailablePositions: "This sample's cell table doesn't say where its cells lie on the transcripts, so the mixing check can't learn from it.",
  // A run without proseg's own count (the engine's no_counts: made before proseg-live kept it; old_proseg: its proseg-live
  // can't keep it), each with a tooltip wherever it shows (unavailableTooltip): the first's why a new run has it, the
  // second's how to update it, by how this app finds the program (GET /health proseg_fix: PROSEG_FIXES), the setting in
  // the file the session reads (a notebook's: the user's own, read at launch).
  unavailableNoCounts: "This run was made before proseg's own count was kept. Open a new run to see clusters and the mixing check.",
  unavailableNoCountsTooltip: 'This run has no count from proseg to score: it was made before proseg-live kept its count. A new run has one.',
  unavailableOldProseg: 'Update proseg-live to see clusters and the mixing check.',
  fixBuild: 'build it again with native/build.sh',
  fixInstall: "install it as the README's Install section says",
  fixSetting: "point the binary under [proseg] in the app's config.toml at the new one",
  fixSettingNotebook: 'point the binary under [proseg] in ~/.config/sopa-companion/config.toml at the new one and launch again',
  failureMemory: 'the graphics card ran out of memory',
  failureSettle: 'the learning did not settle',
  failureTimeout: 'it took longer than its time limit',
  failureTable: 'the cell table could not be read',
  failureFewCells: 'it has too few cells to learn from',
  failureGpu: 'it could not run on the graphics card set aside for it',
  failureSmallCard: "the graphics card has too little memory for this sample's panel",
  failureUnexpected: 'an unexpected error (details in the engine log)',
  tryAgain: 'Try again',
  // A refused request, in plain words (the engine's own message never reaches the page).
  switchFailed: "Couldn't switch the mixing check for this run. Try again.",
  retryFailed: "Couldn't start learning this sample again. Try again.",
  pickerReady: 'Mixing check: ready for this sample.',
  pickerLearningNoTime: 'Mixing check: learning this sample now (once).',
  pickerWaiting: 'Mixing check: waiting for the graphics card.',
  // The check is on for new runs (the remembered choice) and this sample has never been learned: Start learns it first.
  pickerLearnFirst: 'Mixing check: on. Starting learns this sample first (a few minutes, once).',
  numberLabel: 'cells look mixed',
  numberTooltip: "A cell looks mixed when the model reads 30 % or more of its RNA as coming from neighbouring cells. Only cells with 100 or more transcripts in proseg's count are judged.",
  // What the number is, one hover away in the headline's ⓘ (on screen a number, a verdict and labels); its
  // blind spot both ways (between cells of the same kind it can miss mixing, or show some it can't back up: about 1 in 5
  // cells that look mixed).
  meaning: "A model's estimate of how mixed each cell's RNA looks, not a count of boundary errors. Between cells of the same kind it can't tell RNA apart, so it can miss mixing there or show mixing it can't back up.",
  // The change since the start is the like-for-like one: each cell against itself at the start, both cut down to the
  // same number of transcripts. The verdict line says it in a few words (verdictLine); this tooltip says how.
  changeTooltip: "Like for like: each cell is compared with itself at the start, both cut down to the same number of transcripts, so cells that only lost or gained transcripts don't count as a change. It says fewer or more only when the gap is bigger than repeat runs give, and keeps its word a few seconds when the gap moves back across that line, so it doesn't flicker.",
  // After it, when the verdict line shows its two numbers (verdict…At): what they are, and why they need not be the
  // headline's or the start line's.
  likeForLikeNumbers: "Its two numbers: of every 100 cells judged both now and at the start, how many look mixed, each cell cut to the same number of transcripts on both sides. So they can differ from the number above and from iteration 0's.",
  // Below the model's own threshold: another learning of the model
  // could read it the other way.
  changeLightTooltip: 'The gap is smaller than another learning of the model could give, so it holds for this model only.',
  // What the retention line does to the number above, in its ⓘ.
  retentionUpTooltip: 'That by itself raises the number above: cells that hold more transcripts look more mixed.',
  retentionDownTooltip: 'That by itself lowers the number above: cells that hold fewer transcripts look less mixed.',
  // The start line is proseg's count at iteration 0; the arrow (the engine's like-for-like change) compares each frame
  // with the cells proseg started with, their RNA kept as proseg keeps it in that frame.
  startTooltip: "The start: proseg's count at iteration 0. The arrow compares like for like: the cells proseg started with, their RNA kept as proseg keeps it now, so a change shows what the boundaries and drift did.",
  earlierTooltipPlain: "The proseg cells already saved in this sample, read on proseg's own count.",
  // While cells form, in words true for every start: only a start from cells alone has a start line, and it has no
  // nuclei, so its cells form from the starting cells.
  forming: 'Cells are still forming, so this number rises at first. Compare it with the start once they settle.',
  // ... and in a run with no start line (it started from nuclei, alone or with their cells): nothing to compare with.
  formingNoStart: 'Cells are still forming from their nuclei, so this number rises at first.',
  // ... and in one with no start line that is not known to have started from nuclei (its start line not in yet).
  formingShort: 'Cells are still forming, so this number rises at first.',
  tooFew: "Too few cells with 100 or more transcripts in proseg's count yet.",
  lowDepth: "Most cells here have fewer than 100 transcripts in proseg's count, so only a few can be judged.",
  nucleiStart: 'This run started from nuclei only, so there is no starting number to compare with.',
  // A run that started from cells with their nuclei (the engine's start_kind "paired"): proseg's count at iteration 0
  // holds the nuclei alone, so there is no start line either.
  nucleiCellsStart: "This run started from cells with their nuclei, and proseg's count at iteration 0 holds only the nuclei, so there is no starting number to compare with.",
  // The headline's ⓘ after the basis: which count it reads (the averaged count, never "expected"), and that the saved
  // table holds one draw of it; then, while this kind of run's thresholds are not yet re-measured on proseg's count
  // (the summary's constants.measured false), that too. The retention line's ⓘ ends with what it compares.
  basisTooltip: "Proseg's count at this iteration, averaged over its random background draw. The saved table holds one draw of it, so its check can differ by a cell or two in 100.",
  // ... and on an iteration counted for a pause, the hold, a finish or the end (the summary's count "sampled"), the count
  // proseg saves there, the frame's own numbers (decision A); true also once the run has moved on.
  basisTooltipSaved: "Proseg's count at this iteration as proseg saves it: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.",
  // The count a number reads, by name: in the basis line ({count}) and wherever a number of the check is said.
  countAveraged: "proseg's count (averaged)",
  countSaved: "proseg's count (as saved)",
  notRemeasured: "Its thresholds are not yet re-measured on proseg's count for runs like this one.",
  // A constant of the worker or Go deeper kept from the outline-era measurements for this kind of run (the run's
  // constants.kept: engine/qc_constants.py KEPT), in the tooltip of what rests on it, short.
  notRemeasuredShort: "Not yet re-measured on proseg's count for runs like this one.",
  retentionBasis: "Proseg's count now, against the cells proseg started with, kept the same way.",
  // On a saved iteration the arrow and the retention still compare like for like on the averaged count, the start's
  // side and the iteration's alike (decision A): said in their tooltips.
  likeForLikeSaved: "The change since the start reads proseg's count averaged over its random background draw on both sides, like for like, not the count as saved.",
  retentionSaved: "Both sides read proseg's count averaged over its random background draw, not the count as saved.",
  // While the run plays the headline and its change are smoothed over the iterations (the engine's EMA), and its word is
  // held a few seconds (qc_rules.Steady: change.held, direction_now): said in their tooltips, the change itself in words.
  changeSmoothed: 'While the run plays the change is smoothed like the number above.',
  changeNowSame: 'about the same as at the start', changeNowMore: 'more than at the start', changeNowFewer: 'fewer than at the start',
  // The verdict line while its word is held (change.held): the words of its line with numbers (verdict…At), without
  // them: a held block's two numbers are those its word compared (engine qc_rules.Steady), an earlier result's than the
  // number above, so the word is said alone until it follows the change again (changeHeld, in its tooltip, says what
  // the change itself reads).
  verdictSameHeld: 'About the same, like for like',
  verdictMoreHeld: '▲ More, like for like',
  verdictFewerHeld: '▼ Fewer, like for like',
  verdictMoreLightHeld: '△ More, like for like',
  verdictFewerLightHeld: '▽ Fewer, like for like',
  // The chip's tooltip names the count its number reads; on an iteration counted for a pause, the hold, a finish or the
  // end with a change since the start beside the number, each part's count: the number the count proseg saves, the
  // change the averaged count on both sides, like for like (decision A).
  chipCountAveraged: "Read on proseg's count averaged over its random background draw.",
  chipCountSaved: 'Read on the count proseg saves at this iteration.',
  chipCountSavedChange: "The number reads the count proseg saves at this iteration; the change since the start reads proseg's count averaged over its random background draw, on both sides.",
  // The figures of the shape cards, their timing and their tests were measured while the check counted the RNA inside the
  // drawn outlines (RULE_COPY `era` "outline"): said in their tooltips, the figures kept.
  outlineEraTooltip: "From tests made while the mixing check counted the RNA inside the drawn outlines, before it read proseg's count; not measured again yet.",
  // A card on a saved iteration: its areas and numbers are the averaged count's, smoothed (the live path's), said last
  // under its Why?.
  cardAveraged: "Its numbers read proseg's count averaged over its random background draw, smoothed as the run played, not the count as saved.",
  chartTitle: 'Over this run · per 100 cells',
  // The chart's ⓘ (its title) says what its dashed line is: the start, proseg's count at iteration 0.
  chartStartTooltip: "The dashed line: proseg's count at iteration 0.",
  chartForming: 'cells forming',
  chartExpected: 'expected without the change',
  chartExpectedTooltip: 'Where this line would probably be if nothing had changed, judged from the iterations just before.',
  tighterShapes: 'tighter shapes',
  looserShapes: 'looser shapes',
  standardShapes: 'standard shapes',
  // Prior weight's changes by what they do whatever the run started from (its Live knobs label reads "higher
  // = the starting cells hold their transcripts harder"; the starting cells may be Cellpose's, never "closer to Xenium").
  holdNuclei: 'firmer hold',
  looserHold: 'looser hold',
  driftOn: 'drift on',
  driftOff: 'drift off',
  driftDistance: 'drift distance',
  driftSearchStep: 'drift search step',
  onePieceOn: 'one-piece cells on',
  onePieceOff: 'one-piece cells off',
  cooler: 'cooler',
  warmer: 'warmer',
  moreAttempts: 'more boundary attempts',
  fewerAttempts: 'fewer boundary attempts',
  expressionVariability: 'expression variability',
  timelineLegend: 'cells that look mixed, per 100 (estimate)',
  mixing: 'Mixing',
  mixingOffTooltip: 'Turn the mixing check on to colour the cells by mixing.',
  mixingCantRun: "The mixing check can't run for this run, so the cells can't be coloured by it. Its panel says why.",
  showCells: 'Show on the cells (M)',
  showCellsTooltip: "Colour each cell by how much of its RNA the model reads as coming from neighbouring cells.",
  // The stage's legend (qc-overlay.mjs MixingLegend): a short title and one row per shade, each a few words (what each
  // shade means); no note line. The shade is a cell's share of RNA the model reads as
  // coming from neighbouring cells (the cell's tooltip and the panel's ⓘ say "estimate").
  legendTitle: 'RNA from neighbours',
  legendLittle: 'Under 10 %',
  legendSome: '10–30 %',
  legendMixed: '30 % or more (looks mixed)',
  legendUnjudged: 'Too few transcripts',
  overviewNote: 'At this zoom the colours show groups of cells, not single cells.',
  // The overview layer's own key, the same way: a short title and its three rows (the share of cells that look mixed in
  // each patch, against the tissue's usual level), no notes.
  overviewTitle: 'Cells that look mixed, per patch',
  legendNoValues: 'Not checked yet: the colours come with the first result',
  cellLittle: 'Little or no mixing seen (estimate)',
  cellUnjudged: 'Too few transcripts to judge mixing',
  // Beside a cell's own number (its line in the mixing check), what that number is not (and its blind spot both ways).
  cardNote: "A model's estimate, not proof of a boundary error. Between cells of the same kind it can't tell RNA apart.",
  chipWaiting: 'Mixing check: waiting for the graphics card',
  // The chip's tooltip under 300 judged cells.
  roughNumber: 'Based on few cells, so this number is rough.',
  chipTooFew: 'Mixing check: too few cells with 100 or more transcripts yet',
  // No flagged area is shown: no outline, letter, list or legend line. The
  // engine still flags them for its suggestions and trials, whose words name the place as "where mixing stands out" and
  // never by a letter.
  // The cards' short whys say it too ("cells where mixing stands out grew ..."): with no area on screen, "here" would
  // point at nothing.
  standsOut: 'where mixing stands out', inTissue: 'in the tissue',
  // A refusal of Try it or Undo by the Live knobs session (options.js), in the deck's words (qc-panel's errorText).
  tryAlready: 'This setting is already at that value.',
  tryBeyond: 'This setting is already past that value.',
  tryNotLive: "This setting can't change during a run.",
  tryNotNow: 'Settings can change only while the run is running or paused.',
  tryBusy: 'Wait for the last change to this setting to go through.',
  tryWaiting: "Wait a moment: this setting's current value is not known yet.",
  undoNothing: 'There is no earlier value to put back.',
  suggestions: 'What you could try',
  likely: 'Likely to help',
  // R2's and R3's expected size (about 1 in 100 in tests), on their confidence line: a new run that then reads "about
  // the same" (the region's "clearly" is 1.0 or more) is not a broken promise. Short: the tests behind the number are
  // the card's Why? (ruleR2Why, ruleR3Why).
  likelySize: 'Likely to help: about 1 in 100',
  might: 'Might help a little',
  rough: 'Rough guess',
  seen: 'Seen in this run',
  // A card that offers no setting (R1, R6): it says where to look, never that a setting cannot help (the engine's
  // words never claim that either). The engine's own "unlikely" is not used for any card any more.
  closerLook: 'Worth a closer look',
  suggestionFooter: 'Changes the whole run from the next iteration, not just where mixing stands out. An estimate from a model, not a verdict.',
  // A card for the whole tissue: no "not just where mixing stands out"; drift acts only once cells have formed.
  suggestionFooterWhole: 'Changes the whole run from the next iteration. An estimate from a model, not a verdict.',
  suggestionFooterDrift: 'Changes the whole run once cells have formed. An estimate from a model, not a verdict.',
  // "New run with this setting" as the main action says what it does where it is seen (not only in its tooltip).
  suggestionFooterNewRun: 'New run with this setting pauses this run and runs this region again with the setting from the start, to compare the two on the chart. An estimate from a model, not a verdict.',
  // A run that has ended can only be followed by a new run: the card's footer says what that changes. A card that
  // changes nothing (probably the tissue, unlikely to help) keeps only the honesty half.
  suggestionFooterEnded: 'A new run with this setting changes the whole region, not just where mixing stands out. An estimate from a model, not a verdict.',
  suggestionFooterEndedWhole: 'A new run with this setting changes the whole region. An estimate from a model, not a verdict.',
  suggestionFooterInfo: 'An estimate from a model, not a verdict.',
  thisSetting: 'this setting',
  // The trial's title names the change as something tried ("Trying a firmer hold to the starting cells since iteration
  // 412"): the tick labels "firmer hold" / "looser hold" alone do not say what holds.
  tryingHoldCloser: 'a firmer hold to the starting cells', tryingHoldLooser: 'a looser hold to the starting cells',
  // R0's slot while cells form, true for every start (a start from cells only has no nuclei); "from their nuclei" only
  // for a start from nuclei, alone or with their cells.
  tooEarly: 'Too early to suggest anything: cells are still forming.',
  tooEarlyNuclei: 'Too early to suggest anything: cells are still forming from their nuclei.',
  // When a change is best made, and what a shape change costs (the template timingLater names the time).
  timingEarly: 'Best tried now, while cells are still forming: in a test, a change at this stage gave about two thirds of the full effect.',
  timingLaterNoTime: 'Cells have formed, so a change now mostly stops them growing; in a test it showed only after about 1,000 iterations, at half the full effect.',
  // R5 (Prior weight): the timing test was of cell shapes only; a change of the hold during a run was never tested.
  timingUntested: 'Not tested when changed during a run; for a clear answer, start a new run with this setting.',
  shapeCost: 'This changes the whole run. In tests, cells everywhere got 6–15 % smaller and kept 4–12 % fewer transcripts.',
  try: 'Try it',
  newRun: 'New run with this setting',
  // The engine's `inspect` (R1, R6, R7: the cause is not known): the view moves to the place (no outline is drawn)
  // and Go deeper opens on its most mixed cell (main.js qcInspect).
  inspect: 'Look at its cells',
  inspectTooltip: "Moves the view to where mixing stands out and opens Go deeper on its most mixed cell, to see which of its genes look like a neighbour's.",
  why: 'Why?',
  undoIt: 'Undo it',
  keep: 'Keep',
  undo: 'Undo',
  tryPaused: 'Applies when you Resume or Step.',
  // The trial's rows compare like for like (engine/qc_rules.py Watch.readout: "now" at the same depth).
  trialColumns: 'expected → now, like for like',
  trialExpected: 'expected',
  trialNow: 'now',
  wholeTissue: 'Whole tissue',
  transcriptsPerCell: 'Transcripts per cell',
  trialLess: 'Less mixed than expected, by more than the usual ups and downs.',
  trialMore: 'More mixed than expected. Consider Undo.',
  // The verdict of an area of 300 or more judged cells is its own: said for where mixing stands out, never
  // by its letter (no area is shown).
  trialLessArea: 'Less mixed than expected where mixing stands out, by more than the usual ups and downs.',
  trialMoreArea: 'More mixed than expected where mixing stands out. Consider Undo.',
  trialSmall: 'The cells where mixing stands out are too few to judge on their own; the whole tissue is the better guide.',
  trialForming: "Changed while cells are still forming, so this line rises by itself and can't show the change on its own. For a clear answer, start a new run with this setting.",
  // Changed while cells formed, read once they have: the line has settled, but nothing from before the change can say
  // where it would be.
  trialFormedPast: "Changed while cells were still forming, so there is nothing from before the change to compare it with here. For a clear answer, start a new run with this setting.",
  trialMixedUp: "Another setting changed during this trial, so this test can't tell which change did what.",
  // The full schedule's last stage ("Hill-climb") moves the line by itself: the test stops there.
  trialLastStage: 'The run moved on to its final settling stage, which moves this line by itself, so this test stops here.',
  trialPendingLaterNoTime: 'No clear change yet: a change made after cells formed often needs about 1,000 iterations.',
  // The change came too soon after cells formed for the 50 iterations before it to say where the line would be.
  trialNoHistory: 'There are too few iterations from before the change to judge it here. For a clear answer, start a new run with this setting.',
  // A run with no start line (it started from nuclei, alone or with their cells): no trial ever gets a verdict there,
  // said in a few words, why in the tooltip (no row, count or wait for a verdict that never comes).
  trialNoStart: 'No verdict on this kind of run',
  trialNoStartTooltip: "This run started from nuclei, alone or with their cells: proseg's count at iteration 0 holds only the nuclei, so there is no like-for-like start to judge a change against. For a clear answer, start a new run with this setting.",
  // The engine's note `no_matched`: the results around the change lack the like-for-like numbers a verdict needs.
  trialNoMatched: "These results can't be compared like for like, so this test can't judge the change here. For a clear answer, start a new run with this setting.",
  trialEnded: 'The run ended during this trial.',
  trialKept: 'Kept. The chart marks where it started.',
  // Notices after a click that did not go through (the engine's own words never reach the page).
  tryFailed: "Couldn't change the setting. Try again.",
  trialFailed: "The setting changed, but the mixing check couldn't start watching it.",
  // ... and so the setting was put back as it was (nothing stays changed without a readout to undo it from).
  trialFailedUndone: "The mixing check couldn't start watching this change, so the old setting was put back.",
  trialEndFailed: "The mixing check couldn't take your choice. Try again.",
  undoFailed: "Couldn't put the old setting back. Try again.",
  newRunFailed: "Couldn't set up the new run. Try again.",
  newRunTooltip: 'Opens a new run of the same region, starting cells and settings, with this setting from the start. This run is saved to the sample or discarded first.',
  // The same action while paused or finished.
  newRunTooltipStill: 'Opens a new run of the same region, starting cells and settings, with this setting from the start. This run is saved to the sample or discarded first.',
  // The run's starting cells are not among those the picker offers now: the picker says None until they are chosen.
  newRunPriorMissing: "New run set up, but this run's starting cells are not offered for this sample: choose them before you press Start.",
  estimate: 'An estimate from a model, not a measured error.',
  howTitle: 'How is this estimated?',
  // Five short points, then the provenance line (howLearned), at most 90 words in all; the method is named here only.
  // Nothing here or anywhere in this deck names files outside the app, nor claims the model is anyone's own.
  how1: "A ResolVI model learned this sample's kinds of cells once.",
  how2: "It estimates how much of each cell's RNA looks like its neighbours'.",
  how3: "It can't tell same-kind neighbours apart; truly overlapping cells look mixed even with good outlines.",
  how4: "Cells under 100 transcripts aren't judged; changes are compared like for like, at equal transcript counts.",
  how5: "It reads proseg's own count of each cell: the transcripts proseg places in it, less those it reads as background.",
  // The model's table in plain words (never a zarr key on screen): the provenance line's tooltip (howTable).
  tableXenium: "Xenium's cells", tableOther: "the sample's other cell table",
  howPasses: 'A small table takes more passes over its cells.',
  // Additional state/accessibility labels where the deck leaves the wording open.
  on: 'on', offLabel: 'off', live: 'Live', paused: 'Paused', finishedLabel: 'Finished',
  // The stage chip has no words for off or unavailable: it is hidden then (qc-chip.mjs CHIP_STATES).
  unavailable: 'Mixing check unavailable.',
  chipStart: 'Mixing check: waiting for the starting cells',
  chipFrame: 'Mixing check: waiting for the next iteration',
  chipFailed: 'Mixing check: could not learn this sample',
  openPanel: 'Click to open the mixing check.',
  calmMotion: 'Calm motion', calmMotionTooltip: 'Use still updates to keep this view responsive.',
  loadFailed: "Couldn't reach the mixing check for this run. It tries again by itself.",
  iteration: 'iteration', missing: '—', automatic: 'automatic', info: 'ⓘ',
  xeniumStart: "Xenium's cells", otherStart: 'the starting cells',
  and: ' and ', comma: ', ',
  settingCompactness: 'Cell compactness', settingDrift: 'Switch off the transcript drift model',
  // A switch in a Setting line, as it is done in Live knobs (a checkbox): 'Setting: untick "..." (drift on)'.
  tick: 'tick', untick: 'untick',
  // Drift is modelled only once cells have formed (the Try it tooltip of R3).
  driftActsLater: 'It acts once cells have formed.',
  // R2 for an area that did not get more mixed: its gain may be small.
  ruleR2NotRose: 'Where mixing stands out, it has not risen since the start, so the gain may be small.',
  // farDrift and gapProposals without their value (the timeline's readout gives the values beside).
  farDriftPlain: 'far drift', gapProposalsPlain: 'gap proposals',
  ruleUTitle: 'Undo your last change?',
  // What was seen, and that the cause is not known (never claim a setting cannot fix something).
  ruleR7Title: "The setting you tried didn't help here",
  ruleR1Title: 'Mixed from the start: cause not known',
  ruleR2Title: 'Go back to the standard cell shape',
  ruleR3Title: 'Turn transcript drift back on',
  ruleR4Title: 'Try tighter cell shapes',
  ruleR5Title: 'Hold cells closer to their nuclei',
  ruleR6Title: 'More mixed, though cells did not grow',
  // A card's why is one short sentence in view; the rest of what it said is the card's Why? (`rule${R}More`, then its
  // cost, timing, evidence and footer).
  ruleR3Body: 'With drift off, cells kept 7-17 % fewer transcripts in tests.',
  // Drift shows in proseg's count, so a drift trial is judged live like any other.
  ruleR3More: 'Drift lets proseg give RNA that drifted a little back to the cell it came from. It starts acting once cells have formed.',
  ruleR6Body: 'The outlines may not be what changed: RNA may be spreading in the tissue.',
  ruleR6More: "Where mixing stands out, cells got more mixed without growing or shrinking. The cause isn't known; check the tissue image and marker genes there.",
  // R1's and R7's other sentences (their first is in view: QC_TEMPLATES ruleR1Body, ruleR7Body).
  ruleR1More: "The cause isn't known: cells may lie very close together or on top of each other, different kinds of cells may interleave, or they may be at the edge of the tissue. Their RNA may include a neighbour's, so check the tissue image and marker genes before naming their cell type.",
  ruleR7More: "Other settings were not tried here. Its cells' RNA may include a neighbour's: check marker genes before naming their cell type.",
  // The tests behind the cards (three samples and one long test), said in words: no sample is named and none of a
  // tested sample's own levels is given, only what was seen in how many samples, and an effect's size where the card's
  // confidence line rests on it (R2, R3: "about 1 in 100") or it is a cost (R3's transcripts, R5's smaller cells: the
  // reason R5 is offered at all). R4: one clear drop of three, of which about half remains like for like (the rest came
  // from cells holding fewer transcripts), and its gentler step untested. R5: the firmer hold changed nothing in one
  // sample and raised mixing a little in the other two. R3: repeat runs of two regions of one sample on proseg's count
  // (500 µm, started from cells only: where its size, the confidence line's "about 1 in 100", was measured, said first),
  // drift switched back on after cells formed, and two tests of its cost on that sample (about 7 % in those regions, 17 %
  // in the other); a trial on a region of a few hundred cells cannot show its size (its like-for-like part, about 0.6 in
  // 100, is a third to a half of such a region's band).
  ruleR4Why: 'Tested on three samples. Set from the start, tighter shapes clearly lowered mixed-looking cells in one, barely in another and not in the third. Like for like, about half of that drop remains: the rest came from cells holding fewer transcripts. Set early in a run they did about two thirds as much; set after cells had formed they changed nothing for 100 iterations and about half as much after 1,000. A gentler change, to 0.03, has not been tested.',
  ruleR2Why: 'In a test, looser shapes (0.08) gave about 1 more mixed-looking cell in 100, and more neighbour-like RNA, in two runs.',
  ruleR3Why: "In repeat runs of two regions of one sample, started from cells only, switching drift off gave about 1 more mixed-looking cell in 100, and more neighbour-like RNA, in every run, and about 7 % fewer transcripts per cell (17 % in another test of that sample). Switched back on after cells had formed, it lowered mixing again, half to two thirds of that like for like. Trying it on a region of a few hundred cells reads no clear change: such a region can't show a change this small.",
  ruleR5Why: 'In a test, a weaker hold (a quarter as strong) made cells about 14 % smaller, so a firmer one may keep small cells from being squeezed. In tests on three samples, holding cells closer to their nuclei (×4) never lowered mixing: no change in one, a little more in the other two. Treat this as a rough guess.',
  // "Test a setting here…" under the selected cell's row and in Go deeper: fresh runs of the area (your region as it is,
  // or a 500 µm square around the cell, cut to the region where it is narrower), as now and with the value chosen, to
  // the end of the schedule; whether the suspects of this cell and of nearby cells with the same pattern stay in their
  // cells, beyond what their own RNA at the same spot does. Its words describe the runs and the places, never an
  // outcome; the answer's words follow the engine's verdict as it is (result v2: engine/qc_rule.py), and nothing is
  // suggested. No word here says "look-alike", "similar", "same kind" or "like for like" (each means something else in
  // this panel), and "hold" stands only in the hold's own name and in testHowHold (what the runs are given of it).
  testOpen: 'Test a setting here…',
  testOpenTooltip: 'Opens a corner of the image where fresh runs of this area go to the end of the schedule, as now and with a value you choose (3 of each, or 4 for a start from cells only), to see whether the suspects of this cell and of nearby cells with the same pattern stay in their cells. About a minute; your run waits meanwhile.',
  testValueLabel: 'Value to test',
  // Prior weight by its plain name in the choice (its Live knobs name stands only in a Setting line): the hold to the
  // starting cells, whatever they are (Xenium's, or Cellpose's for a start from cells only). Far drift: proseg's share
  // of transcripts that drift far, offered at 0.1 to 0.3 and not tried in a test yet (its note and tooltip are the
  // form's only notes: nothing in it says what a setting may do).
  testHold: 'Hold to the starting cells',
  testFarDrift: 'Far drift',
  testFarDriftNote: 'Not tried here yet.',
  testFarDriftTip: "Of the drift proseg allows, the share with the far distance, 4 µm by default, from where a transcript was seen; the rest has the short drift distance, 1 µm by default. Proseg's default share is 0.2.",
  // Six more of proseg's own settings, each by its plain name (its unit in brackets) and what it is in its tooltip
  // (testGridTip, with the run's own block size, is a template), each "Not tried here yet."; a startup-only one's name
  // without its unit for the Setting line (testSettingStart). The switch's words: "on" / "off" beside its buttons,
  // "drift on" / "drift off" in sentences.
  testNotTriedNote: 'Not tried here yet.',
  testGrid: 'Starting grid (µm)',
  testGridPlain: 'Starting grid',
  testGaps: 'Background gaps',
  testGapsTip: "The share of proseg's tries at moving a cell's edge that offer the spot to no cell, a gap, instead of to the cell next to it. 0: never; 1: always. Proseg's default is 0.05.",
  testDrift: 'Drift (on or off)',
  testDriftTip: "Whether proseg may count a transcript for a cell a little way from where it was seen. Off: every transcript counts only for the cell at its own spot, or for none. Proseg's default is on.",
  testNearDrift: 'Short drift distance (µm)',
  testNearDriftTip: "How far most drift reaches: the usual distance, in µm, a transcript may lie from a cell and still count for it. Far drift sets the share of rarer, longer moves. Proseg's default is 1 µm.",
  testCrowding: 'Crowding levels',
  testCrowdingPlain: 'Crowding levels',
  testCrowdingTip: "How many levels of crowding proseg sorts the tissue into, from sparse to dense, by how many transcripts lie close by. Each level has its own background amount of every gene. Proseg's default is 5.",
  testLayers: 'Depth layers',
  testLayersPlain: 'Depth layers',
  testLayersTip: "How many layers proseg cuts the section's depth into. A cell has its own outline in every layer. Proseg's default is 4; a sample without depth gets one layer. Each layer over 4 adds about 2 % to the test's time.",
  testSwitchOn: 'on',
  testSwitchOff: 'off',
  testDriftOnWord: 'drift on',
  testDriftOffWord: 'drift off',
  testChoose: 'choose one of the values shown.',
  // The form's pool line while the engine finds the pool (GET /runs/{id}/qc/test/pool), or when it could not.
  testPoolAsking: 'Finding nearby cells with the same pattern…',
  testPoolFailed: "Couldn't count the nearby cells with the same pattern. The test counts them again when it starts.",
  // A reason the page has no words for (a newer engine's), by its kind: a refusal (listed under "Can't test here:"), a
  // wait (Run's tooltip, a waiting test's line), a caveat (Run's tooltip, the answer's).
  testReasonUnknown: "The test can't run here",
  testWaitUnknown: 'Starts as soon as it can',
  testCaveatUnknown: 'Not measured yet here',
  // A press of Run while the engine's refusals are listed: nothing is asked of the engine, the reasons stay listed, and
  // one line says what holds (qc-panel.js testNotHere): a reason about the run or its sample (TEST_RUN_REFUSALS: no
  // cell of it can be tested); else one that leaves this cell nothing to test (a code the page does not know too); else
  // a test of this run on its way.
  testNotHereRun: "This run can't be tested.",
  testNotHereCant: "This cell can't be tested.",
  testNotHereFirst: 'Wait until it ends.',
  // Far drift with drift switched off in the run (the engine refuses it too: 422).
  testDriftOff: 'Far drift does nothing while drift is switched off in this run.',
  testNearDriftOff: 'Short drift distance does nothing while drift is switched off in this run.',
  // Depth layers on a sample without depth (the engine refuses it too: 422).
  testFlat: 'This sample has no depth: proseg makes one layer whatever is set.',
  // Before a value is typed, the runs line names it so (testRuns' {v}).
  testThisValue: 'this value',
  // The How line: the runs line's tooltip, in the form and while the runs work. How the runs are made, never what they
  // may show (that your run waits is in the runs line itself). With testHowHold and the values given (testHowHoldGiven
  // or testHowHoldGivenAll, testHowHoldTried) the tooltip stays within a tooltip's 75 words.
  testHow: "Every run goes afresh from your run's starting cells to the end of the schedule, all at once; those with the value change only this setting.",
  // After it, always: proseg scales the hold by the input's transcripts per cell, so the runs are given the value whose
  // strength in their area is your run's (or, for a hold tried, the value's at your run's scale); what they are given in
  // testHowHoldGiven and testHowHoldTried (the test record's hold.given) once the engine has given it.
  testHowHold: 'Proseg makes the hold stronger where cells hold more transcripts, so each run gets a value as strong as in your run. Only the value given to proseg changes.',
  // The test lives in a corner of the image; the panel keeps one line about it.
  testStatusOpen: 'Testing a setting: open',
  testStatusWaiting: 'Testing a setting: waiting to start',
  testStatusRunning: 'Testing a setting: running',
  testStatusDone: 'Testing a setting: answer ready',
  testStatusStopped: 'Testing a setting: stopped',
  testStatusTooltip: 'The test is shown in a corner of the image.',
  // While it runs: what the engine is doing (progress.phase), and the schedule's three parts (testIteration's {phase}).
  testPreparing: 'Preparing the runs…',
  testLoading: 'Starting the runs…',
  testReading: 'Reading the runs…',
  testPhaseSettling: 'settling', testPhaseRefining: 'refining', testPhaseFinal: 'final settling',
  testRetry: "One run couldn't start, so every run started again without one start-up step.",
  testStarting: 'Starting…',
  testCancelling: 'Cancelling…',
  testWorking: 'Working…',
  // Cancel while the test waits for what comes first (no run of its own yet): the engine pauses again a run the press
  // resumed (one that reached its end meanwhile waits there), and a mixing check the press switched on stays on (the
  // switch is remembered).
  testCancelWaitingTooltip: 'Stops the test before its runs start. Your run is paused again if the test resumed it; the mixing check stays on.',
  // The answer (result v2), part by part: do they come back (the gate), are they fewer (the rule the engine chose:
  // the excess over their own RNA at the same spot, or the suspects alone), what it cost, where they went, this cell
  // alone; or too few to judge.
  testBackTitle: 'Do they come back?',
  testBackYes: 'Yes', testBackNo: 'No',
  testBackNot: 'These come and go between runs: proseg already moves them on its own. Nothing to judge.',
  testFewerTitle: 'Fewer than their own RNA?', testFewerTitleAlone: 'Fewer suspects?',
  testFewerYes: 'Clearly fewer', testFewerAloneYes: 'Clearly fewer (own RNA not separated)', testFewerNo: 'Not clear',
  // "Clearly fewer" on the excess while the cells shrank: both in one sentence (the cost part then does not repeat it).
  testFewerYesShrink: 'Clearly fewer, mostly by shrinking the cell: their own RNA at the same spot dropped too',
  // Why, by the rule's own reason (fewer.why_not: "apart", "gap", or none when it fired).
  testFewerWhyYes: 'In every run with it, the suspects drop further than their own RNA at the same spot, beyond what identical runs give.',
  testFewerAloneWhyYes: 'Every run with it is below every run as now, by more than identical runs differ.',
  testFewerWhyApart: 'Not every run with it shows the suspects dropping further than their own RNA at the same spot.',
  testFewerAloneWhyApart: 'Not every run with it is below every run as now.',
  testFewerWhyGap: 'The difference is no wider than identical runs give.',
  testTooFewCell: "This cell's suspects weren't found in every run, so nothing was judged.",
  testNoControls: "Their own RNA at the same spot wasn't found in every run, so nothing was judged.",
  testCostTitle: 'What it cost',
  testCostOwn: 'Own RNA at the same spot', testCostAll: 'All their own RNA', testCostPerCell: 'Transcripts per cell here',
  testShrink: 'Mostly by shrinking the cell',
  testWentTitle: 'Where the suspects went',
  testWentStay: 'stay', testWentNeighbour: 'to the neighbour', testWentOther: 'another cell', testWentBackground: 'background',
  testAloneOne: 'This cell alone: 1 suspect: too few to judge one cell.',
  // Added to the answer's tooltip when the run's settings are not proseg's defaults (the test was measured with them).
  testCalibrationSettings: "Measured with proseg's default settings; this run has others.",
  // The answers: none is the main one (nothing is suggested).
  testNewRun: 'New run with this',
  testNewRunTooltip: 'Opens a new run of the same region, starting cells and settings, with this value from the start. This run is saved to the sample or discarded first.',
  testKeep: 'Keep my setting',
  testKeepTooltip: 'Leaves your settings as they are. Your run goes on as it was before the test.',
  testRunAgain: 'Run again',
  testRunAgainTooltip: 'Runs the same test again, with fresh runs.',
  // After Run again, under the new answer.
  testRepeatNote: 'Each test is a new chance of a false "clearly fewer"; a repeat is not a stronger answer.',
  testFailed: "The test couldn't finish. Your settings are unchanged.",
  testCancelled: 'The test was cancelled. Your settings are unchanged.',
  // What a press of Run the test still needs (explain.mjs lists "Value to test: type a number."): never a greyed button.
  testValueMissing: 'type a number.',
  testValueSame: "that is your run's value now: choose another.",
  // The hold is tested against its value at the run's last iteration (the pool reply's `hold`: the engine's), which a
  // hold set live since then makes another value than now's.
  testValueSameEnd: "that was your run's value at its last iteration: choose another.",
  testValueRefused: "this value can't be used for this setting.",
  testNoCell: 'Click a cell on the image first.',
  // The engine's refusals and failures, in plain words (its own words never reach the page; its reasons have the
  // deck's: TEST_REASON_WORDS).
  testCellMissing: "This cell isn't in the run's newest iteration: click it again.",
  testStartFailed: "Couldn't start the test. Try again.",
  testCancelFailed: "Couldn't cancel the test. Try again.",
  testAnswerFailed: "Couldn't do that. Try again.",
  testResumeFailed: "Your run couldn't be resumed: press Resume.",
});

export const QC_TEMPLATES = Object.freeze({
  learning: "Learning this sample's cell types… {k} of 50 · about {t} left",
  learningProgress: "Learning this sample's cell types… {k} of {of} · about {t} left",
  learningNoTime: "Learning this sample's cell types… {k} of {of}",
  seconds: '{n} s', minutes: '{n} min',
  // Relearning also means the sample's other table was saved again, not only new software.
  softwareChanged: 'Something this check depends on changed since this sample was learned, so it is learning again (about {t}).',
  failed: "Couldn't learn this sample: {reason}.",
  failedWithAction: "Couldn't learn this sample: {reason}. [Try again]",
  // What learning this sample needs on its graphics card, which grows with the panel (GET /samples/{id}/qc room:
  // engine/qc.py room_needed_mb), and what is free now; a card smaller than that in all can never learn it (gpu_small).
  waitingRoom: 'Waiting for room on the graphics card: learning this sample needs about {need} free ({genes} genes), and {free} is free now.',
  failureSmallCardRoom: "its {genes} genes need about {need} free on the graphics card, which has {total} in all: it needs a graphics card with more memory",
  // On a Mac (room.mac): its GPU uses the Mac's own memory, the cells' counts in it too.
  failureSmallMacRoom: "its {genes} genes and their counts need about {need} on this Mac's GPU, which may use {total} in all: it needs a Mac with more memory",
  gigabytes: '{n} GB',
  workerRestarting: 'Mixing check paused while the graphics worker restarts; showing iteration {i}.',
  pickerLearning: 'Mixing check: learning this sample now (about {t}, once).',
  number: '{n} in 100',
  // The headline's ⓘ, first: which iteration its number is from and how (H2, H10): live, the newest result smoothed
  // with the ones before it (the engine's EMA; its whole number moves only 0.3 past the halfway point: SHOWN_MARGIN);
  // looking back, as it was shown live; an iteration's own number (a rest frame's own, or one judged alone).
  numberLive: "From iteration {i}, the newest proseg has counted, smoothed with the iterations before it; its whole number changes only once it is 0.3 past the halfway point, so it doesn't flicker. Paused, it shows one iteration's own number.",
  numberPast: 'Iteration {i}, smoothed with the iterations before it, as it was shown live.',
  numberOwn: "Iteration {i}'s own number, not smoothed.",
  // The verdict's word held a few seconds (change.held) while the change itself reads {now} (changeNowSame/More/Fewer of
  // change.direction_now: the smoothed change's own word), in the verdict's tooltip and the chip's.
  changeHeld: "This word is held a few seconds so it doesn't flicker; the change itself now reads {now}.",
  // The headline's one verdict line: the like-for-like change (change.matched, unrounded, to one
  // decimal) only when it is bigger than repeat runs give (change.meaningful), said as like for like (it compares each
  // cell with itself at the start, not the start's number in the line above: H13); a hollow arrow when it is smaller
  // than another learning of the model could give (as the chip). The start's number and "by this model's reading" are in
  // its tooltip.
  verdictSame: 'Same as at the start, like for like',
  verdictMore: '▲ {d} more, like for like',
  verdictFewer: '▼ {d} fewer, like for like',
  verdictMoreLight: '△ {d} more, like for like',
  verdictFewerLight: '▽ {d} fewer, like for like',
  // ... with the change's own two numbers (the summary's change.per100_now and per100_start, to one decimal): the cells
  // judged both now and at the start, each cut to the same depth, so the line says what it compares where the headline
  // (on the count proseg saves, at a pause, the hold, a finish or the end) and the start line read other numbers.
  verdictSameAt: 'About the same, like for like: {now} now, {start} at the start',
  verdictMoreAt: '▲ More, like for like: {now} now, {start} at the start',
  verdictFewerAt: '▼ Fewer, like for like: {now} now, {start} at the start',
  verdictMoreLightAt: '△ More, like for like: {now} now, {start} at the start',
  verdictFewerLightAt: '▽ Fewer, like for like: {now} now, {start} at the start',
  // ... and with no verdict to give (cells forming, none yet), the start's number alone.
  startShort: 'At the start: {s} in 100',
  // The start line is proseg's count at iteration 0, named so (never after the cells it started from: H3).
  start: "At the start (proseg's count at iteration 0): {s} in 100",
  startHeading: 'At the start: iteration 0',
  // The proseg cells saved in the sample earlier (a table holding proseg's own count); in the headline's ⓘ
  // (earlierInfo).
  earlier: "Proseg's cells saved in this sample: {b} in 100",
  earlierInfo: "Proseg's cells saved in this sample: {b} in 100.",
  earlierTooltip: "The proseg cells already saved in this sample ({table}), read on proseg's own count.",
  // "judged" read as "judged mixed": the cells it is based on are named instead ("estimate" is the (i) and the footer's).
  // {count}: which count it reads (countAveraged, or countSaved on an iteration counted for a pause, the hold, a finish
  // or the end).
  basis: 'Based on {count}: {j} cells with 100 or more transcripts · typical cell {t} ({t0} at the start)',
  basisNoStart: 'Based on {count}: {j} cells with 100 or more transcripts · typical cell {t}',
  // Judged coverage over all of the run's cells (`coverage`) and transcript retention (`retention`).
  basisCoverage: 'Based on {count}: {j} of {n} cells (those with 100 or more transcripts)',
  // Under 300 judged cells (the fewest any verdict rests on) the number moves a
  // lot by chance (6 points either way at 30 cells): said beside it.
  basisCoverageFew: 'Based on {count}: only {j} of {n} cells (those with 100 or more transcripts), so this number is rough',
  // ... said in view too, short, while the rest of the basis is in the headline's ⓘ.
  roughCells: 'Rough: based on only {j} of {n} cells',
  tooFewCount: 'Too few cells with 100 or more transcripts yet ({j} of {n}).',
  // Transcript retention (all cells' transcripts now over at the start),
  // one number either way ("123 %" when cells gained some). In view only when it moves the number by itself (under 90 %
  // or from 110 %: the usual case, cells that gained transcripts), short, what it does in its ⓘ; else in
  // the headline's ⓘ, as a sentence.
  retentionShort: 'Cells hold {x} % of their start RNA',
  retention: 'Cells hold {x} % of the transcripts they started with',
  notNewest: 'Iteration {i} (not the newest).',
  // The result shown is another iteration's than the frame on screen (looking back: the nearest with proseg's count,
  // the engine's snapping), a short badge beside the number and the clusters, why in its tooltip; each pane names its
  // own result's iteration (the check's and the clusters' results are lists of their own: a check switched on mid-run
  // or a thinned series can show another iteration than the clusters).
  snappedBadge: 'At iteration {i}',
  snappedTooltipMixing: "The mixing check at iteration {i}: the nearest with proseg's count.",
  snappedTooltipClusters: "Clusters at iteration {i}: the nearest with proseg's count.",
  // An old proseg-live's line (unavailableOldProseg): how to update it ({fix}: fixBuild, fixInstall or fixSetting).
  unavailableOldProsegTooltip: "This proseg-live can't keep proseg's own count. To update it, {fix}.",
  pausedFrame: 'Paused · iteration {i}',
  // The start's reference line, named after what it is: proseg's count at iteration 0 ("start 12" read as the line's
  // own start).
  startReference: 'Iteration 0: {s}',
  // The chart's hover: a live result's number smoothed (single false), a saved iteration's as saved (count "sampled"),
  // else plain (one iteration's own number).
  chartHover: 'Iteration {i} · {n} in 100 cells look mixed (of {j} with 100 or more transcripts)',
  chartHoverSmoothed: 'Iteration {i} · {n} in 100 cells look mixed, smoothed (of {j} with 100 or more transcripts)',
  chartHoverSaved: 'Iteration {i} · {n} in 100 cells look mixed, as saved (of {j} with 100 or more transcripts)',
  chartAfter: '{hover} · after {setting} ({from} → {to}), iteration {k}',
  chartAfterKnob: '{hover} · after {knob} {from} → {to} ({setting}), iteration {k}',
  chartAfterPlain: '{hover} · after {setting}, iteration {k}',
  chartAfterSuffix: '… · after {setting} (iteration {k})',
  farDrift: 'far drift {v}', gapProposals: 'gap proposals {v}',
  timelineHover: '{n} in 100 cells look mixed',
  timelineHoverSmoothed: '{n} in 100 cells look mixed, smoothed',
  timelineHoverSaved: '{n} in 100 cells look mixed, as saved',
  // The model's reading of a cell's own RNA, never where it came from nor "now" (the cell's number is the result shown's
  // own: qc-client.mjs cellValue), and of neighbouring cells of ANY kind; cardLine adds what the number is not
  // (cardNote).
  cellMixed: 'Looks mixed · the model reads about {p} % of its RNA as coming from neighbouring cells (estimate)',
  cellSome: 'Some mixing · the model reads about {p} % as coming from neighbouring cells (estimate)',
  cellCard: 'Mixing: the model reads about {p} % of its RNA as coming from neighbouring cells ({q} % at iteration 0).',
  cellCardNow: 'Mixing: the model reads about {p} % of its RNA as coming from neighbouring cells.',
  // A reading from an earlier result than the one shown (qc-client.mjs cellReading), when a readout shows one at all.
  cellCardFrom: 'Mixing at iteration {i}: the model read about {p} % of its RNA as coming from neighbouring cells ({q} % at iteration 0).',
  cellCardFromNow: 'Mixing at iteration {i}: the model read about {p} % of its RNA as coming from neighbouring cells.',
  // The chip: the arrow only for a like-for-like change bigger than repeat runs give, to one decimal, and said to be like
  // for like beside the raw number; a hollow arrow when another learning of the model could read it the other way (the
  // panel's "by this model's reading").
  chipMore: 'Mixing {n} in 100 · ▲{d} like for like', chipFewer: 'Mixing {n} in 100 · ▼{d} like for like',
  chipMoreLight: 'Mixing {n} in 100 · △{d} like for like', chipFewerLight: 'Mixing {n} in 100 · ▽{d} like for like',
  // "Same as at the start" alone; its tooltip (chipTooltipSame) keeps "like for like".
  chipSame: 'Mixing {n} in 100 · same as at the start',
  // While cells form the number rises by itself: no comparison with the start yet.
  chipForming: 'Mixing {n} in 100 · cells forming',
  chipTooltipForming: '{n} in 100 cells look mixed; cells are still forming, so compare with the start once they settle. Click to open the mixing check.',
  chipTooltipFormingNoStart: '{n} in 100 cells look mixed; cells are still forming, so this number rises at first. Click to open the mixing check.',
  chipNumber: 'Mixing {n} in 100', chipLearning: 'Mixing check: learning this sample… {pct} %',
  chipTooltipMore: '{n} in 100 cells look mixed; {d} more than at the start, like for like. Click to open the mixing check.',
  chipTooltipFewer: '{n} in 100 cells look mixed; {d} fewer than at the start, like for like. Click to open the mixing check.',
  // A hollow arrow: the gap is under the model's own threshold (as changeLightTooltip says it).
  chipTooltipMoreLight: '{n} in 100 cells look mixed; {d} more than at the start, like for like. The gap is smaller than another learning of the model could give, so it holds for this model only. Click to open the mixing check.',
  chipTooltipFewerLight: '{n} in 100 cells look mixed; {d} fewer than at the start, like for like. The gap is smaller than another learning of the model could give, so it holds for this model only. Click to open the mixing check.',
  chipTooltipSame: '{n} in 100 cells look mixed; about the same as at the start, like for like. Click to open the mixing check.',
  // A live number in the chip: from which iteration, smoothed (as the headline's numberLive).
  chipSmoothed: 'From iteration {i}, smoothed while the run plays.',
  // The patches' key on the stage: one row per shade, against the tissue's usual level (qc-overlay.mjs binStops).
  overviewLow: 'Up to {a} in 100 (usual here)',
  overviewMid: 'About {b} in 100',
  overviewHigh: '{c} in 100 or more',
  // R1's button: the starting cells' outlines on, and the view moved to the place its card is about (no outline of that
  // place is drawn: its tooltip says where the view goes).
  showStartOutlines: 'Show the outlines of {start} here',
  showStartTooltip: 'Shows the outlines of {start} and moves the view to where mixing stands out.',
  setting: 'Setting: {label} {old} → {new}',
  // The Live knobs label's own hint after the values ("Cell compactness 0.04 → 0.02 (lower = rounder cells)").
  settingHint: 'Setting: {label} {old} → {new} ({hint})',
  // A switch as it is done in Live knobs (its checkbox), with its plain change: a label named for what it switches off
  // ("Switch off the transcript drift model") would read backwards as "on → off".
  settingTick: 'Setting: {verb} "{label}" ({change})',
  settingTickPlain: 'Setting: {verb} "{label}"',
  // Exact Live knobs wording: the only banned-word exception in this registry.
  settingHold: 'Setting: Prior weight {old} → {new}',
  // A suggestion card's Setting line (short cards): the Live knobs name and
  // the values alone ("Cell compactness 0.08 → 0.04"), its hint in the line's ⓘ; a switch as its checkbox, its plain
  // change in the ⓘ (settingShort). settingChecked knows these lines as Setting lines.
  settingShort: '{label} {old} → {new}',
  settingTickOn: 'Tick "{label}"', settingTickOff: 'Untick "{label}"',
  tryTooltip: 'Sets {label} to {value} for the whole run from the next iteration. Undo puts back {old}.',
  // A switch by its plain change ("Turns drift on ...; Undo puts back drift off."), never "Sets Switch off ... to off".
  tryTooltipSwitch: 'Turns {change} for the whole run from the next iteration. Undo puts back {back}.',
  // Its facts kept and shortened (the card's New run says the rest): {t} = the time 1,000 iterations take in this run
  // (a few minutes on a small region).
  timingLater: 'Cells have formed, so a change now mostly stops them growing; in a test it showed only after about 1,000 iterations (about {t} here), at half the full effect.',
  moreIdeas: 'More ideas ({k})',
  trialTitle: 'Trying {setting} since iteration {i}',
  // Asked for, not in a frame yet (paused: it applies when you Resume or Step).
  trialTitleWaiting: 'Trying {setting} from the next iteration',
  // After Undo, Keep or the run's end: what was tried, in the past (no "Watching ... so far" counting on).
  trialTitleUndone: 'Tried {setting} from iteration {i} to {j} (undone)',
  trialTitleKept: 'Kept {setting} since iteration {i}',
  trialTitleEnded: 'Tried {setting} from iteration {i} until the run ended',
  // A switch tried: "Trying with drift on since iteration 47".
  tryingWith: 'with {change}',
  trialWatching: 'Watching where mixing stands out and the whole tissue · {k} iterations so far',
  trialWatchingWhole: 'Watching the whole tissue · {k} iterations so far',
  trialArea: 'Where mixing stands out: {a} → {b} in 100',
  trialWhole: 'Whole tissue: {a} → {b} in 100',
  trialDepth: 'Transcripts per cell: {a} → {b}',
  trialNoChange: 'No clear change yet. {timing}',
  trialNoChangeAfter: 'No clear change after {k} iterations.',
  // Changed after cells formed (such a change shows only after ~1,000 iterations, never
  // within the 300 a trial is judged over): said while it waits and when its window closes.
  trialPendingLater: 'No clear change yet: a change made after cells formed often needs about 1,000 iterations (about {t} here).',
  trialNoChangeAfterLater: 'No clear change after {k} iterations. A change made after cells formed often needs about 1,000 (about {t} here); for a clear answer, start a new run with this setting.',
  trialNoChangeAfterLaterNoTime: 'No clear change after {k} iterations. A change made after cells formed often needs about 1,000; for a clear answer, start a new run with this setting.',
  // The loss is a fact; the rows compare like for like, so it does not move them by itself (under pure loss the
  // like-for-like change stays at about 0, where a re-weighting by depth took out only part of it).
  trialDepthNote: 'Cells also lost {x} % of their transcripts. The rows above compare like for like, so that loss alone does not move them.',
  trialUndone: 'Put back {label} {old}. Applies from the next iteration.',
  // A switch: its plain change ("Put back drift off."), else "Put back {label}: on."; a knob whose Live knobs name may
  // not stand outside a Setting line (Prior weight): its value alone.
  trialUndoneChange: 'Put back {change}. Applies from the next iteration.',
  trialUndoneSwitch: 'Put back {label}: {old}. Applies from the next iteration.',
  trialUndoneValue: 'Put back the old value, {old}. Applies from the next iteration.',
  // "New run with this setting": the picker is filled in; {setting} is the plain change ("tighter shapes").
  newRunReady: 'New run set up: the same region, starting cells and settings, with {setting} from the start. Press Start to run it.',
  // A live knob change as the app's timeline says it (knobLine): the Live knobs name, the values as Live knobs shows them
  // and the plain change; never the raw flag.
  knobLine: '{knob} {from} → {to}', knobLineNamed: '{knob} {from} → {to} ({change})',
  knobSwitch: '{knob}: {from} → {to}', knobSwitchNamed: '{knob}: {from} → {to} ({change})',
  knobSet: '{knob}: {to}', knobPlain: '{change}: {from} → {to}', knobOther: 'Another setting changed: {from} → {to}',
  // The whole tissue's numbers only: no line names a flagged area. The start is proseg's count at iteration 0.
  finished: 'Finished at iteration {i}: {n} in 100 cells looked mixed ({s} at iteration 0).',
  // A run stopped by the user or by the engine, and one that ended while cells formed.
  stoppedAt: 'Stopped at iteration {i}: {n} in 100 cells looked mixed ({s} at iteration 0).',
  // ... and a run with no start line (it started from nuclei, alone or with their cells): no start number beside it.
  finishedNoStart: 'Finished at iteration {i}: {n} in 100 cells looked mixed.',
  stoppedNoStart: 'Stopped at iteration {i}: {n} in 100 cells looked mixed.',
  finishedForming: "Finished at iteration {i}, while cells were still forming, so it can't be compared with the start yet.",
  // Ended with too few cells of 100 or more transcripts for a number.
  finishedFew: 'Finished at iteration {i}, with too few cells of 100 or more transcripts for a number.',
  stoppedFew: 'Stopped at iteration {i}, with too few cells of 100 or more transcripts for a number.',
  stoppedForming: "Stopped at iteration {i}, while cells were still forming, so it can't be compared with the start yet.",
  resultFact: 'Mixing (estimate): {n} in 100 at iteration {i} (start {s})',
  // "How is this estimated?"'s provenance line, with the passes over the cells when a small table took
  // more than usual (provenance more_epochs_small_table).
  howLearned: 'Learned from {cells} cells on {date}, in {t}.',
  howLearnedPasses: 'Learned in {e} passes over {cells} cells on {date}, in {t}.',
  howTable: 'The cells it learned from: {table}.',
  // The rows compare like for like (the band's centre plus the matched change since the change), not the headline.
  ruleUBody: 'Like for like, mixing rose from {a} to {b} in 100 after you changed {setting}.',
  // A switch by its plain change ("after you turned one-piece cells off").
  ruleUBodySwitch: 'Like for like, mixing rose from {a} to {b} in 100 after you turned {change}.',
  // Only after a trial that made the area more mixed (engine/qc.py `_remember_tried`: "no clear change" is no evidence).
  // What was seen of that one setting; never that no setting could help: that other settings were not tried is the
  // first sentence of its Why? (QC_TEXT ruleR7More).
  ruleR7Body: 'You tried {setting} here and mixing rose rather than dropped.',
  // An area that stood out already at the start and did not grow (persistence says nothing about whether a setting
  // would help; areas at the tissue's edge look more mixed): the possible causes are its Why? (ruleR1More).
  ruleR1Body: 'These cells already looked mixed at iteration 0, and proseg has not made them more mixed.',
  // What is true of the place only (the shapes may have been set so from the start): grown since the start, or got more
  // mixed. {where} is "where mixing stands out" for a card about flagged areas (never their letters; the
  // trial readout calls them so too), "in the tissue" for one about the whole tissue. One short sentence each, the
  // numbers kept.
  ruleR2Body: 'Shapes are looser than standard ({v}); cells {where} grew about {g} % bigger.',
  ruleR2BodyRose: 'Shapes are looser than standard ({v}); cells {where} got more mixed in this run.',
  ruleR4Body: 'Cells {where} grew about {g} % bigger and seem to reach into their neighbours.',
  ruleR5Body: 'Small cells {where} are squeezed by their neighbours: {k} have shrunk to almost nothing.',
  // "Test a setting here…" (qc-panel.js). {v} a setting's value as Live knobs shows it; the setting's name stands
  // beside a sentence (the corner's header), never in it.
  testNow: 'now {v}',
  testUm: '{v} µm',
  testWhole: 'use a whole number from {min} to {max}.',
  testSettingStart: 'Setting: {name} {old} → {new}',
  testGridTip: "The blocks proseg starts from: each goes to the starting cell (its nucleus, in a start with nuclei) with the most transcripts in it, and a starting cell that gets none does not start. Cells settle on this grid first, then each block splits into final {v} µm blocks. Only whole multiples of {v} µm work; proseg's default is 2 µm.",
  // ... its tooltip in a test of the hold set live since the run's last iteration: the test compares with the value
  // the run had then (the pool reply's `hold`, F's), as its runs as now are given.
  testNowEnd: "The test compares your value with {v}, your run's value at its last iteration.",
  testRange: 'use a number from {min} to {max}.',
  testReason: 'Reason: {reason}',
  // The form: the pool the engine found (the clicked cell and nearby cells with the same pattern, until their suspects
  // reach what a test needs: that number in the line's tooltip, a pool short of it tested all the same), the area its
  // runs re-run (the region as it is, a square around the cell, or a window around it cut to a narrower region), and the
  // runs.
  testPool: 'Nearby cells with the same pattern: {c} cells, {n} suspects',
  testPoolOne: 'Nearby cells with the same pattern: 1 cell, {n} suspects',
  testPoolTooltip: 'The test was measured with {k} suspects or more. With fewer it runs all the same, and its answer says so.',
  testAreaRegion: 'Area: your whole region, {w} × {h} µm',
  testAreaSquare: 'Area: a {w} × {h} µm square around this cell',
  testAreaCrop: 'Area: {w} × {h} µm around this cell',
  testRuns: 'Your run waits · {a} runs as now, {b} with {v} · about {t}',
  testRunsNoTime: 'Your run waits · {a} runs as now, {b} with {v}',
  // While it runs: the slowest run's iteration of the schedule's end, and which part of the schedule it is in.
  testIteration: 'Iteration {i} of {n} · {phase}',
  testWriting: 'Writing their results · {d} of {n} runs done',
  testProgressTooltip: "Every run starts from your run's starting cells and goes to the end of the schedule: settling to iteration {a}, refining to {b}, then final settling to {n}. The bars show the slowest run as now and the slowest with the value.",
  testCancelTooltip: 'Stops the test and deletes its {n} runs. Your run goes on as it was before the test.',
  // The hold values the runs were given (testHowHold says why): in a test of the hold, the runs as now's {a} for the
  // run's own {v} at the frame tested and those with the value's {b} for the value tried {t}; in a test of another
  // setting every run's {a} (all of them keep your hold). In the How tooltip while the test runs and in the answer's
  // fewer ⓘ.
  testHowHoldGiven: 'Here the runs as now are given {a} for your {v}.',
  testHowHoldGivenAll: 'Here every run is given {a} for your {v}.',
  testHowHoldTried: 'Those with the value are given {b} for {t}.',
  // The answer's parts (QC_TEXT has their titles and the words without numbers). Shares are whole percentages, with one
  // decimal where two would read the same.
  testBack: 'As now, these {n} suspects stay in their cells in {p} % of cases (runs {runs} %).',
  testFewerText: 'With {v}: suspects {a} → {b} %, their own RNA at the same spot {c} → {d} %.',
  testFewerTextAlone: 'With {v}: suspects {a} → {b} %.',
  testTooFew: 'Too few suspects were left to judge once the runs were read: {n} of {k}.',
  testCostShare: '{a} → {b} %',
  testCostCount: '{a} → {b}',
  testCostChange: '{c} %',
  testWentPart: '{label} {p} %',
  testAlone: 'This cell alone: {n} suspects: too few to judge one cell.',
  // The answer's tooltip: how often identical runs say "clearly fewer", how often a drop of the suspects alone is caught,
  // and how often a drop of the suspects with their own RNA still says it, all from the test's own row of measured
  // numbers (result.constants.rates: its start and area).
  testCalibration: 'About {fa} in 100 tests of identical runs come out "clearly fewer" ({fw} in 100 in the worst of {areas} areas). A drop of the suspects alone by {drop} % ({from} % → {to} %), their own RNA at the same spot unchanged, is caught about {pw} times in 10. When the suspects and their own RNA drop alike, it still says "clearly fewer" about {sf} in 100 times. Measured on one sample.',
  // A test with fewer suspects than its numbers are for (the caveat too_few), where its row has numbers of its own for
  // such tests (result.constants.short_rates): after the caveat, what they are.
  testCalibrationShort: 'Tests with fewer than {need} suspects, like this one, were measured apart, on only {n} test cells of one sample: about {fa} in 100 tests of identical runs came out "clearly fewer" ({fw} in 100 at worst), and a drop of the suspects alone by {drop} % was caught about {pw} times in 10.',
  // At a saved F (the pool reply's count "sampled"), the test reads the averaged count (decision A) while the panel and
  // the pink dots read the count proseg saves there: said in the pool line's tooltip and in each listed reason's.
  testAveraged: 'The test judges the cell and picks its suspects on proseg\'s count averaged over its random background draw; the panel and the pink dots at iteration {i} show the count proseg saves there, so the two can differ a little.',
  // ... and where the averaged count gives the cell no suspect that can be tested while its pink dots glow on the count
  // proseg saves (the caveat saved_suspects: the engine takes them from there), that instead.
  testAveragedSaved: "The test judges the cell on proseg's count averaged over its random background draw, which gives it none that can be tested, so they are taken from the count proseg saves at iteration {i}, as the pink dots show.",
  // After Run again: the earlier answer's word ({word}: testFewerYes, testFewerAloneYes or testFewerNo, its first letter
  // lower-cased), or nothing judged (it came and went, or too few).
  testRepeat: 'Earlier test with {v}: {word}.',
  testRepeatNone: 'Earlier test with {v}: nothing judged.',
});

/** Missing values fail early rather than leaking braces or "undefined" to the UI. */
export function copy(key, values = {}) {
  if (Object.hasOwn(QC_TEXT, key)) return QC_TEXT[key];
  if (!Object.hasOwn(QC_TEMPLATES, key)) throw new RangeError(`Unknown QC copy key: ${key}`);
  return QC_TEMPLATES[key].replace(/\{([^}]+)\}/g, (_, name) => {
    if (!Object.hasOwn(values, name) || values[name] == null) throw new TypeError(`Missing QC copy value: ${key}.${name}`);
    return String(values[name]);
  });
}

export function startName(source) { return source === 'xenium' ? QC_TEXT.xeniumStart : QC_TEXT.otherStart; }
// One formatter per number of decimals, made once: toLocaleString builds a new Intl.NumberFormat on every call (about
// 40 µs each), which made Go deeper's window spend ~50 ms on a cell with a thousand genes (its rows and tooltips).
const NUMBER_FORMATS = new Map();
export function number(value, digits = 0) {
  if (!Number.isFinite(value)) return QC_TEXT.missing;
  let format = NUMBER_FORMATS.get(digits);
  if (!format) {
    format = new Intl.NumberFormat('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits });
    NUMBER_FORMATS.set(digits, format);
  }
  return format.format(value);
}
export function duration(seconds) {
  return Number.isFinite(seconds) ? copy(seconds >= 60 ? 'minutes' : 'seconds', { n: Math.max(1, Math.ceil(seconds / (seconds >= 60 ? 60 : 1))) }) : QC_TEXT.missing;
}
export function list(items) {
  return items.length < 2 ? String(items[0] ?? '') : items.slice(0, -1).join(QC_TEXT.comma) + QC_TEXT.and + items.at(-1);
}
export function settingValue(value) {
  return value == null ? QC_TEXT.automatic : typeof value === 'boolean' ? QC_TEXT[value ? 'on' : 'offLabel']
    : typeof value === 'number' ? String(Number(value.toPrecision(6))) : String(value);
}

// Tighter shapes is a "Rough guess" (clear in one of three samples). `sized`: the confidence line with the size the
// tests measured (R2; R3, drift, measured on proseg's count: about 1 in 100). The cards that offer no setting say where
// to look (R1, R6) or what was seen (U, R7), never that a setting is unlikely to help: `own` = the card's own line wins
// over the engine's. `body` is the card's one short why in view, `more` the rest of it (R1, R3, R6, R7), first under
// its Why? fold. `era`: when its figures (the sized confidence line, the tests under Why?) were measured: "outline" while
// the check counted the RNA inside the drawn outlines (R2, R4, R5: their tooltips say so, outlineEraTooltip), "proseg"
// on proseg's count (R3), null for a card with no figure of its own.
export const RULE_COPY = Object.freeze(Object.fromEntries([
  ['U', 'seen', null, 'tighterShapes'], ['R7', 'seen', null, 'tighterShapes', null, true],
  ['R1', 'closerLook', null, 'tighterShapes', null, true], ['R2', 'likely', 'ruleR2Why', 'standardShapes', 'likelySize', false, 'outline'],
  ['R3', 'likely', 'ruleR3Why', 'driftOn', 'likelySize', false, 'proseg'], ['R4', 'rough', 'ruleR4Why', 'tighterShapes', null, false, 'outline'],
  ['R5', 'rough', 'ruleR5Why', 'holdNuclei', null, false, 'outline'], ['R6', 'closerLook', null, 'tighterShapes', null, true],
].map(([rule, confidence, why, setting, sized = null, own = false, era = null]) => [rule, Object.freeze({ title: `rule${rule}Title`, body: `rule${rule}Body`,
  more: Object.hasOwn(QC_TEXT, `rule${rule}More`) ? `rule${rule}More` : null, confidence, why, setting, sized, own, era })])));

/** The engine's reasons (GET /samples/{id}/qc, GET /runs/{id}/qc) -> QC_TEXT keys. An
 *  unknown reason gets the general text of its state: the engine's own words never reach the page. */
export const WAITING_REASONS = Object.freeze({ gpu_busy: 'waiting', gpu_memory: 'waiting', lock: 'waitingLock', queue: 'waitingQueue' });
export const FAILURE_REASONS = Object.freeze({ oom: 'failureMemory', not_settled: 'failureSettle', timeout: 'failureTimeout',
  unreadable: 'failureTable', too_few_cells: 'failureFewCells', gpu: 'failureGpu', gpu_small: 'failureSmallCard',
  unexpected: 'failureUnexpected' });
export const UNAVAILABLE_REASONS = Object.freeze({ no_table: 'unavailableTable', not_counts: 'unavailableCounts',
  software: 'unavailableSoftware', analysis: 'unavailableAnalysis', no_transcripts: 'unavailableNoTranscripts',
  positions: 'unavailablePositions', no_gpu: 'unavailableGpu', apple_gpu: 'unavailableAppleGpu', torch_cpu: 'unavailableTorchCpu',
  driver: 'unavailableDriver',
  no_counts: 'unavailableNoCounts', old_proseg: 'unavailableOldProseg' });
/** The room learning needs on the graphics card, in its reason's words (GET /samples/{id}/qc room: {need_mb, free_mb,
 *  total_mb, genes}): while it waits (gpu_memory), how much it needs and how much is free now; a card too small for
 *  the panel (gpu_small), how much it needs and how much the card has. null without the numbers each needs. */
export function roomText(state, reason, room) {
  if (!room || !Number.isFinite(room.need_mb) || !Number.isFinite(room.genes)) return null;
  const gb = mb => copy('gigabytes', { n: number(mb / 1024, 1) });
  if (state === 'waiting' && reason === 'gpu_memory' && Number.isFinite(room.free_mb)) {
    return copy('waitingRoom', { need: gb(room.need_mb), free: gb(room.free_mb), genes: number(room.genes) });
  }
  if (state === 'failed' && reason === 'gpu_small' && Number.isFinite(room.total_mb)) {
    return copy(room.mac ? 'failureSmallMacRoom' : 'failureSmallCardRoom',
      { need: gb(room.need_mb), total: gb(room.total_mb), genes: number(room.genes) });
  }
  return null;
}
export function reasonText(state, reason, room = null) {
  const said = roomText(state, reason, room);
  if (said) return said;
  const map = state === 'waiting' ? WAITING_REASONS : state === 'failed' ? FAILURE_REASONS : state === 'unavailable' ? UNAVAILABLE_REASONS : null;
  const key = map?.[reason] ?? { waiting: 'waiting', failed: 'failureUnexpected', unavailable: 'unavailable' }[state];
  return key ? QC_TEXT[key] : '';
}
/** How to update an old proseg-live (GET /health proseg_fix: how this app finds the program, locate.py) -> QC_TEXT keys. */
export const PROSEG_FIXES = Object.freeze({ build: 'fixBuild', install: 'fixInstall', setting: 'fixSetting' });
/** The tooltip of an old proseg-live's line (unavailableOldProseg): how to update it, by the engine's `fix` (a setting:
 *  in the file a notebook's session reads, `notebook`); '' for a fix the page has no words for (none said rather than a
 *  wrong one). */
export function oldProsegTooltip(fix, notebook = false) {
  if (typeof fix !== 'string' || !Object.hasOwn(PROSEG_FIXES, fix)) return '';
  return copy('unavailableOldProsegTooltip', { fix: QC_TEXT[fix === 'setting' && notebook ? 'fixSettingNotebook' : PROSEG_FIXES[fix]] });
}
/** The tooltip of a run's line that the check and the clusters can't run on it (an unavailable reason), wherever it
 *  shows: a run without proseg's own count says why a new run has it (no_counts), an old proseg-live how to update it
 *  (oldProsegTooltip); '' for the other reasons, whose lines say it all. */
export function unavailableTooltip(reason, { prosegFix = null, notebook = false } = {}) {
  return reason === 'no_counts' ? QC_TEXT.unavailableNoCountsTooltip : reason === 'old_proseg' ? oldProsegTooltip(prosegFix, notebook) : '';
}

/** What a test of a setting can't do here, waits for first, or was not measured for (the engine's reasons,
 *  engine/qc_pool.py REASON_WORDS: {code, words, ...data}), in the order the engine lists them and the page's own words
 *  for each code: the same words (the shared fixture frontend/test/fixtures/qc-test-reasons.json, which both sides'
 *  tests read), with their numbers ({max_um}, {min_um}, {have}, {need}) filled in from the reason's data. The run's and
 *  the mixing check's own reasons come first (ending to other_run), then the cell's (edge to no_controls), then the
 *  pool's (too_few: the test's numbers, its row's rates, are those of pools that reached {need} suspects). */
export const TEST_REASON_WORDS = Object.freeze({
  ending: 'This run is being saved',
  ended: 'This run ended before its last iteration',
  no_counts: "Made before proseg's own count was kept",
  old_proseg: 'Update proseg-live to test a setting',
  not_at_end: 'Starts once your run reaches its last iteration',
  past_end: 'Not measured yet for a run that sampled on past its end',
  approx_targets: "Suspects placed approximately: proseg's exact shares at your run's end are no longer available",
  targets_now: 'Suspects taken where your run is now, not at its end',
  whole_sample: 'Not measured yet for runs of 20,000 cells or more',
  tiled_box: 'Not measured yet for regions over {max_um} µm',
  schedule: 'Not measured yet for a changed schedule',
  nuclei_start: 'Not measured yet for runs started from nuclei only',
  lasso: 'Not measured yet for lasso regions',
  area_narrow: 'Not measured yet for regions over 500 µm long and under 500 µm wide',
  square_not_measured: 'Not measured yet for cells-only starts on regions over 500 µm',
  region_small: 'Not measured yet for regions under {min_um} µm',
  check_unavailable: "The mixing check can't run on this sample",
  check_off: 'Switches the mixing check on first',
  check_wait: 'Starts once the mixing check has learned this sample',
  check_scoring: "Starts once the mixing check has scored your run's last iteration",
  details: "Starts once Go deeper's details are learned for this sample",
  settings: 'Not measured yet with these mixing check settings',
  count_basis: "Not measured yet with suspects from proseg's own count",
  busy: 'A test of this run is on its way',
  other_run: 'Starts once the other run pauses or ends',
  edge: "Not measured yet for cells this close to your region's edge",
  not_judged: 'This cell has too few transcripts to be judged',
  no_suspects: 'No suspects to test in this cell: none glows, or none could be placed',
  saved_suspects: "Suspects taken from proseg's count as saved: its averaged count gives none that can be tested",
  low_share: "Doesn't look mixed overall: its pink dots are tested anyway",
  weak_suspects: 'Its suspects are weak: none of its pink dots looks likely',
  same_type: "The neighbours this cell's RNA seems to come from all look like its own kind",
  pattern_none: 'Not measured yet for suspects that sit where no cell started',
  pattern_neighbour: "Not measured yet for suspects that sit inside a neighbour's starting outline",
  pattern_other: "Not measured yet for suspects that sit inside another cell's starting outline",
  pattern_mixed: 'Not measured yet for suspects that sit in different places',
  no_primary: "This cell's suspects all come from neighbours at the area's edge",
  no_controls: 'No own RNA at the same spot to compare its suspects with',
  too_few: 'Its numbers are for tests with {need} suspects or more: {have} here',
});
/** Each reason's kind (engine/qc_pool.py KIND), which says where the page shows it: a refusal (nothing to test, or the
 *  test can't run, a run without proseg's own count among them: the pool reply's and a refused start's `reasons`,
 *  listed in the form, and Run asks nothing), a wait (what the app does or waits for first: `waits`, in Run's tooltip; a
 *  press is taken and the test waits, its line saying for what), or a caveat (the test runs where it was not measured,
 *  with its suspects taken as it could, or with fewer suspects than its numbers are for: `caveats`, in Run's tooltip and
 *  the answer's, never on screen). */
const REFUSALS = ['ending', 'ended', 'no_counts', 'old_proseg', 'check_unavailable', 'busy', 'not_judged', 'no_suspects', 'same_type', 'no_primary',
  'no_controls'];
const WAITS = ['not_at_end', 'check_off', 'check_wait', 'check_scoring', 'details', 'other_run'];
export const TEST_REASON_KINDS = Object.freeze(Object.fromEntries(Object.keys(TEST_REASON_WORDS).map(code =>
  [code, REFUSALS.includes(code) ? 'refuse' : WAITS.includes(code) ? 'wait' : 'caveat'])));
/** A reason's words ({code, ...data}: the engine's), or `fallback` for a code the page does not know (a newer
 *  engine's; the engine's own words never reach the page): by default a refusal's, testReasonUnknown (a wait's and a
 *  caveat's: testWaitUnknown, testCaveatUnknown). A number reads as the engine writes it (300, not 300.0; 74, not
 *  74.00); a value the reason lacks reads "—", never braces. */
export function testReasonText(reason, fallback = QC_TEXT.testReasonUnknown) {
  const code = typeof reason === 'string' ? reason : reason?.code;
  if (typeof code !== 'string' || !Object.hasOwn(TEST_REASON_WORDS, code)) return fallback;
  const data = reason && typeof reason === 'object' ? reason : {};
  return TEST_REASON_WORDS[code].replace(/\{([^}]+)\}/g, (_, name) => {
    const v = data[name];
    return typeof v === 'number' && Number.isFinite(v) ? String(Number(v.toPrecision(6))) : typeof v === 'string' && v ? v : QC_TEXT.missing;
  });
}
/** The corner lists the refusals in two groups, in this order, each under its heading (qc-test-corner.mjs), the
 *  engine's order kept inside each: what passes by itself first (a test of this run on its way; a wait too, were one
 *  ever listed), and what leaves the test nothing to measure for this cell or can't run on this run or sample (a code the
 *  page does not know too). */
export const TEST_REASON_GROUPS = Object.freeze(['first', 'cant']);
/** The refusals about the run or its sample rather than the cell: while one holds, no cell of the run can be tested. */
export const TEST_RUN_REFUSALS = Object.freeze(['ending', 'ended', 'check_unavailable', 'no_counts', 'old_proseg']);
export function testReasonGroup(code) {
  return code === 'busy' || (typeof code === 'string' && Object.hasOwn(TEST_REASON_KINDS, code) && TEST_REASON_KINDS[code] === 'wait')
    ? 'first' : 'cant';
}

/** A live knob change (a frame's `live` values: flag -> value) as one of the plain change labels of the chart
 *  ({key, value}), or null for a flag with no plain name. Values may be float32 (0.03999999910593033). KNOB_NAMES:
 *  the Live knobs label of a knob whose name may appear beside its values (only names with no banned word). */
export const KNOB_NAMES = Object.freeze({ 'cell-compactness': 'Cell compactness', temperature: 'Temperature' });
export const STANDARD_COMPACTNESS = 0.04;
export function knobChange(flag, from, to) {
  const change = plainChange(flag, from, to);
  return change && { ...change, from };
}
function plainChange(flag, from, to) {
  const near = (a, b) => Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));
  const up = Number.isFinite(from) && Number.isFinite(to) ? to > from : null;
  switch (flag) {
    case 'cell-compactness': return { key: near(to, STANDARD_COMPACTNESS) ? 'standardShapes' : up === false ? 'tighterShapes' : 'looserShapes', value: to };
    case 'prior-weight': return { key: up === false ? 'looserHold' : 'holdNuclei', value: to };
    case 'no-diffusion': return { key: to ? 'driftOff' : 'driftOn', value: to };
    case 'diffusion-probability': return { key: 'farDrift', value: to };
    case 'diffusion-sigma-near': case 'diffusion-sigma-far': case 'diffusion-sigma-z': return { key: 'driftDistance', value: to };
    case 'diffusion-proposal-sigma': case 'diffusion-proposal-sigma-z': return { key: 'driftSearchStep', value: to };
    case 'enforce-connectivity': return { key: to ? 'onePieceOn' : 'onePieceOff', value: to };
    case 'temperature': return { key: up === true ? 'warmer' : 'cooler', value: to };
    case 'morphology-steps-per-iter': return { key: up === false ? 'fewerAttempts' : 'moreAttempts', value: to };
    case 'ab-nihlo-bubble-prob': return { key: 'gapProposals', value: to };
    case 'dispersion': return { key: 'expressionVariability', value: to };
    default: return null;
  }
}

/** key is one of the plain change labels above; raw flag names are never labels (an unknown change has none: ''). A
 *  suggestion or a trial ({flag, from, to}) is named by its own change (0.04 -> 0.02: "tighter shapes"), else by its
 *  rule's. */
export function changeLabel(change = {}) {
  if (change.setting && !bannedWords(change.setting).length && !/resolvi/i.test(change.setting)) return change.setting;
  const to = change.to !== undefined ? change.to : change.value;
  const own = !change.key && change.flag && to !== undefined ? plainChange(change.flag, change.from, to) : null;
  const key = change.key ?? own?.key ?? RULE_COPY[change.rule]?.setting;
  return key && (Object.hasOwn(QC_TEXT, key) || ['farDrift', 'gapProposals'].includes(key))
    ? copy(key, { v: settingValue(change.value ?? change.to) }) : '';
}

/** A change as something tried, for the R7 card and the new run's notice ("tighter shapes", "a firmer hold to the
 *  starting cells", a switch by its plain change: "drift on"). */
export function triedLabel(change = {}) {
  const to = change.to !== undefined ? change.to : change.value;
  const key = change.flag && to !== undefined ? plainChange(change.flag, change.from, to)?.key : null;
  return key === 'holdNuclei' ? QC_TEXT.tryingHoldCloser : key === 'looserHold' ? QC_TEXT.tryingHoldLooser : changeLabel(change);
}
/** The same for the trial's title ("Trying tighter shapes since ..."): a switch "with drift on". */
export function tryingLabel(change = {}) {
  const to = change.to !== undefined ? change.to : change.value, label = triedLabel(change);
  return label && (typeof to === 'boolean' || typeof change.from === 'boolean') ? copy('tryingWith', { change: label }) : label;
}

/** A Live knobs label ("Cell compactness (lower = rounder cells)") as a name ("Cell compactness") and the hint it carries
 *  ("lower = rounder cells"; null when there is none, or when it holds a banned word: "(no diffusion)"). */
export function knobName(label) {
  return typeof label === 'string' && label.trim() ? label.replace(/\s*\([^()]*\)\s*$/, '').trim() || null : null;
}
export function knobHint(label) {
  const hint = typeof label === 'string' ? /\(([^()]+)\)\s*$/.exec(label)?.[1]?.trim() : null;
  return hint && !bannedWords(hint).length ? hint : null;
}
/** Two setting values are the same (proseg reports knobs as 32-bit floats: 0.02 comes back as 0.019999999552965164). */
export const sameSetting = (a, b) => a === b || (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b)));

/** A suggestion's Setting line: the Live knobs name, the values as Live knobs shows
 *  them, and the label's own hint ("Setting: Cell compactness 0.04 → 0.02 (lower = rounder cells)"); a switch as it is
 *  done in Live knobs, its checkbox, with its plain change ('Setting: untick "Switch off the transcript drift model"
 *  (drift on)'). Without the page's label, the names this registry knows. Never the raw flag. */
export function settingLine({ flag, from, to, label } = {}) {
  const known = { 'cell-compactness': QC_TEXT.settingCompactness, 'no-diffusion': QC_TEXT.settingDrift, 'prior-weight': 'Prior weight' };
  const name = knobName(label) ?? known[flag] ?? null, hint = knobHint(label);
  const values = { old: settingValue(from), new: settingValue(to) };
  if (!name) return null;
  if (name === 'Prior weight' && !hint) return copy('settingHold', values);
  if (typeof to === 'boolean') {
    const change = changeLabel({ flag, from, to }), verb = to ? QC_TEXT.tick : QC_TEXT.untick;
    return change ? copy('settingTick', { verb, label: name, change }) : copy('settingTickPlain', { verb, label: name });
  }
  return hint ? copy('settingHint', { label: name, hint, ...values }) : copy('setting', { label: name, ...values });
}
/** A suggestion card's short Setting line: {text, hint}, the text the
 *  Live knobs name and the values alone ("Cell compactness 0.08 → 0.04"), a switch its checkbox ('Untick "Switch off
 *  the transcript drift model"'); the hint, for the line's ⓘ, the Live knobs label's own ("lower = rounder cells"), a
 *  switch's plain change ("drift on"), else null. Without the page's label, the names this registry knows; null for a
 *  knob it cannot name (never the raw flag). */
export function settingShort({ flag, from, to, label } = {}) {
  const known = { 'cell-compactness': QC_TEXT.settingCompactness, 'no-diffusion': QC_TEXT.settingDrift, 'prior-weight': 'Prior weight' };
  const name = knobName(label) ?? known[flag] ?? null;
  if (!name) return null;
  if (typeof to === 'boolean') return { text: copy(to ? 'settingTickOn' : 'settingTickOff', { label: name }), hint: changeLabel({ flag, from, to }) || null };
  return { text: copy('settingShort', { label: name, old: settingValue(from), new: settingValue(to) }), hint: knobHint(label) };
}
const escaped = text => text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
/** Whether a whole line is a card's short Setting line for the knob `name` (settingShort: its templates, the values any
 *  word). */
const shortSettingLine = (text, name) => ['settingShort', 'settingTickOn', 'settingTickOff'].some(key => new RegExp(`^\\s*${QC_TEMPLATES[key]
  .split(/(\{\w+\})/).map(part => part === '{label}' ? escaped(name) : /^\{\w+\}$/.test(part) ? '\\S+' : escaped(part)).join('')}\\s*$`).test(text));
/** The exact Live knobs names a Setting line may hold (the deck's one exception to its banned words: "Prior weight"),
 *  taken out of such a line before it is checked (the node tests and the harness check every other word). A Setting
 *  line starts "Setting:", or is a card's short one, whole (settingShort: "Prior weight 1 → 2"). */
export function settingChecked(line, names = []) {
  const text = String(line ?? '');
  const known = [...names, 'Prior weight'].filter(Boolean).sort((a, b) => b.length - a.length);
  const name = /^\s*Setting:/.test(text) ? known.find(n => text.includes(n)) : known.find(n => shortSettingLine(text, n));
  return name ? text.replace(name, '') : text;
}

/** A live knob change as the app's timeline readout says it: a vetted name (KNOB_NAMES: no banned or jargon
 *  word; a Live knobs label may stand only in a Setting line) with the values as Live knobs shows them and the plain
 *  change ("Cell compactness 0.04 → 0.02 (tighter shapes)"); any other knob by its plain change ("firmer hold: 1 → 2",
 *  a switch "drift off"); the raw flag never. `from` undefined: the value it was set to ("—" before). */
export function knobLine(flag, from, to) {
  const name = KNOB_NAMES[flag] ?? null, own = knobChange(flag, from, to);
  const change = own && ['farDrift', 'gapProposals'].includes(own.key) ? QC_TEXT[`${own.key}Plain`] : changeLabel(own ?? {});
  const values = { from: from === undefined ? QC_TEXT.missing : settingValue(from), to: settingValue(to) };
  const toggle = typeof to === 'boolean' || typeof from === 'boolean';
  if (name && from === undefined) return copy('knobSet', { knob: name, to: values.to });
  if (name) return change ? copy(toggle ? 'knobSwitchNamed' : 'knobLineNamed', { knob: name, change, ...values })
    : copy(toggle ? 'knobSwitch' : 'knobLine', { knob: name, ...values });
  if (change && toggle) return change;
  return change ? copy('knobPlain', { change, ...values }) : copy('knobOther', values);
}

/** A suggestion's identity: the engine's own (`id`, engine/qc_rules.py card_id: its rule and the change it offers, for
 *  R7 the change tried, for U the change it answers), else the same from its fields; never its areas, which may join
 *  or leave the same card. */
export function cardKey(s) {
  if (!s) return '';
  if (typeof s.id === 'string' && s.id) return s.id;
  const change = s.rule === 'R7' && s.tried && typeof s.tried === 'object' ? s.tried : s;
  return JSON.stringify([s.rule, change.flag ?? null, change.to ?? null, s.rule === 'U' ? s.changed?.from ?? null : null]);
}

/** One cell's value as the worker codes it (floor(share x 250); 255 = not judged): its share, or null. */
export const NOT_JUDGED = 255, MIXED_CODE = 75, SOME_CODE = 25;
export const cellShare = code => Number.isInteger(code) && code >= 0 && code < NOT_JUDGED ? Math.min(1, (code + 0.5) / 250) : null;   // its range's middle
/** A share as a whole percentage, rounded down so that it agrees with the steps: "some" (10-30 %) never reads 30 %, a
 *  cell that looks mixed (30 % or more) never reads 29 %. */
export const sharePercent = share => number(Math.floor(share * 100 + 1e-9));
/** The cell tooltip's line; thresholds as the colours and the headline. */
export function cellLine(code) {
  const share = cellShare(code);
  if (share == null) return QC_TEXT.cellUnjudged;
  const p = sharePercent(share);
  return code >= MIXED_CODE ? copy('cellMixed', { p }) : code >= SOME_CODE ? copy('cellSome', { p }) : QC_TEXT.cellLittle;
}
/** The selected cell's line in the mixing check: the cell in the result shown, and at the start (proseg's count at
 *  iteration 0, named so whatever the cells it started from: `startSource` is no longer said) when that cell was judged
 *  there, then what the number is not (cardNote, in view beside the number). `iter`: the reading is an earlier result's
 *  than the one shown (qc-client.mjs cellReading), said with its iteration, never as the result shown's. `note: false`
 *  leaves cardNote out, where the mixing check's meaning line under its headline says it in view (main.js updateCellRow). */
export function cardLine(code, startCode, startSource, { iter = null, note = true } = {}) {
  const now = cellShare(code), start = cellShare(startCode);
  if (now == null) return QC_TEXT.cellUnjudged;
  const p = sharePercent(now), from = Number.isFinite(iter) ? { i: number(iter) } : null;
  const line = start == null ? copy(from ? 'cellCardFromNow' : 'cellCardNow', { p, ...from })
    : copy(from ? 'cellCardFrom' : 'cellCard', { p, q: sharePercent(start), ...from });
  return note ? `${line} ${QC_TEXT.cardNote}` : line;
}

// ---- the engine's like-for-like change, coverage, retention and the model's provenance --------------------------------
/** The fewest cells with 100 or more transcripts the headline is given for (one judged cell of three once read "0 in
 *  100"); fewer, and the panel says how many there are. */
export const HEADLINE_MIN_JUDGED = 30;
/** Under this share of all cells with 100 or more transcripts, the panel warns that only a few can be judged. */
export const LOW_COVERAGE = 0.5;
/** The model's own threshold for a like-for-like change, when the engine's change does not carry one: a region's
 *  (engine/qc_constants.py matched_model, the engine's mode when it knows no other). */
export const MODEL_THRESHOLD = 2.3;
/** Under this transcript retention the retention line says that the loss by itself lowers the headline; from
 *  RETENTION_HIGH on, that the gain by itself raises it (cells usually gain transcripts). */
export const RETENTION_LOW = 0.9, RETENTION_HIGH = 1.1;
/** Under this many judged cells the basis line says the number is rough (F2_depth.local_version's minimum support). */
export const ROUGH_JUDGED = 300;
const finiteNumber = Number.isFinite;

/** A like-for-like change as the engine judged it (the summary's `change`: {matched (unrounded
 *  points), threshold (repeat runs: the sampler), model_threshold, meaningful, direction, per100_now and per100_start
 *  (the two numbers behind matched, unrounded), held (its word held a few seconds: engine qc_rules.Steady)}): its
 *  direction only when it is meaningful (never from rounded numbers, never below the threshold), `light` when it is
 *  smaller than the model's own threshold ("by this model's reading"), `d` its size to one decimal, `held`, and `now`
 *  and `start` its two numbers to one decimal (null unless both are there, a summary stored before them having none,
 *  and while its word is held: they are then those the word compared, an earlier result's than the headline's). A held
 *  word keeps the size it was decided on: its `d` and `light` are those of the two numbers its word compared (the held
 *  block's per100_now and per100_start), never the change's own size now, which may sit inside the threshold (a held
 *  block stored before the numbers came keeps the change's own). Null without a like-for-like change. */
export function likeForLike(change, modelThreshold = null) {
  const matched = change?.matched;
  if (!finiteNumber(matched)) return null;
  // No verdict from the engine (engine/qc_rules.py change_block: cells still forming, too few cells for any verdict,
  // `enough` false: 300 judged and 400 paired; or no verdict yet, `direction` null): nothing is claimed either way.
  if (change.forming === true || change.enough === false || change.direction === null) return null;
  const threshold = finiteNumber(change.threshold) ? change.threshold : null;
  const said = ['fewer', 'more', 'same'].includes(change.direction) ? change.direction : null;
  const meaningful = typeof change.meaningful === 'boolean' ? change.meaningful
    : said ? said !== 'same' : threshold != null && Math.abs(matched) >= threshold;
  // The engine's own word wins: a live verdict may be held steady a few seconds after the change crossed back (engine
  // qc_rules.Steady), and its direction is then the one shown, not the sign of `matched`.
  const direction = said ? (meaningful || said === 'same' ? said : 'same')
    : !meaningful || matched === 0 ? 'same' : matched < 0 ? 'fewer' : 'more';
  const model = finiteNumber(change.model_threshold) ? change.model_threshold : finiteNumber(modelThreshold) ? modelThreshold : MODEL_THRESHOLD;
  const held = change.held === true, numbers = finiteNumber(change.per100_now) && finiteNumber(change.per100_start), both = !held && numbers;
  const size = held && numbers ? Math.abs(change.per100_now - change.per100_start) : Math.abs(matched);
  return { matched, direction, light: direction !== 'same' && size < model, d: number(size, 1),
    threshold, modelThreshold: model, held, now: both ? number(change.per100_now, 1) : null,
    start: both ? number(change.per100_start, 1) : null };
}
/** The headline's verdict line for a like-for-like change since the start (likeForLike), said as like for like with
 *  what it compares, its own two numbers ("▼ Fewer, like for like: 6.8 now, 8.6 at the start", "About the same, like
 *  for like: …", a hollow arrow under the model's own threshold): the cells judged now and at the start, each cut to
 *  the same depth, which need not be the headline's number (on the count proseg saves at a pause, the hold, a finish or
 *  the end) nor the start line's (iteration 0's: in the line's tooltip, with what the two are: likeForLikeNumbers). A
 *  word held a few seconds says the same words without the numbers ("▼ Fewer, like for like": its tooltip says it is
 *  held and what the change itself reads); a change without them (a summary stored before them) says its size alone
 *  ("▼ 2.1 fewer, like for like", "Same as at the start, like for like"); '' without a change. */
export function verdictLine(change) {
  if (!change) return '';
  const at = !change.held && change.now != null && change.start != null ? { now: change.now, start: change.start } : null;
  const form = change.held ? 'Held' : at ? 'At' : '';
  if (change.direction === 'same') return form ? copy(`verdictSame${form}`, at ?? {}) : copy('verdictSame');
  const key = `verdict${change.direction === 'fewer' ? 'Fewer' : 'More'}${change.light ? 'Light' : ''}`;
  return form ? copy(`${key}${form}`, at ?? {}) : copy(key, { d: change.d });
}
/** The words of a change's direction now (the keys of change.direction_now: the change's own word while the word shown
 *  is held), for changeHeld's {now}. */
export const CHANGE_NOW = Object.freeze({ same: 'changeNowSame', more: 'changeNowMore', fewer: 'changeNowFewer' });
/** The clause a held word adds to its tooltips (the verdict line's, the chip's): what the change itself reads now; ''
 *  for a change whose word is its own (never held: an own or saved change, or one under no hold). */
export function heldClause(change) {
  const now = change?.held === true ? CHANGE_NOW[change.direction_now] : null;
  return now ? copy('changeHeld', { now: QC_TEXT[now] }) : '';
}
/** Which count a point or summary reads, by name ({count} of the basis line): the count proseg saves on an iteration
 *  counted for a pause, the hold, a finish or the end (`count` "sampled"), else the averaged count. */
export const countWord = point => point?.count === 'sampled' ? QC_TEXT.countSaved : QC_TEXT.countAveraged;
/** The hover's words for a point of the line (the chart's `chartHover`, the timeline lane's `timelineHover`): a live
 *  result's number smoothed with the ones before it (single false), a saved iteration's as saved (count "sampled": the
 *  count proseg saves there), else one iteration's own number, plain. */
export const hoverKey = (point, base = 'chartHover') => point?.count === 'sampled' ? `${base}Saved` : point?.single === false ? `${base}Smoothed` : base;

/** Judged coverage over all of the run's cells (`coverage {cells, judged, empty, under20, under100, judged_share}`,
 *  else the worker's own counts on a summary or the start line: `ncells` (the start's `cells`), `judged`, `empty`,
 *  `under20`, `under100`): null when the run's cells are not counted. */
export function coverageOf(item) {
  if (!item || typeof item !== 'object') return null;
  const own = item.coverage && typeof item.coverage === 'object' ? item.coverage : null, c = own ?? item;
  const cells = own ? (finiteNumber(own.cells) ? own.cells : own.ncells) : finiteNumber(item.ncells) ? item.ncells : item.cells;
  const judged = finiteNumber(c.judged) ? c.judged : item.judged;
  if (!finiteNumber(cells) || !finiteNumber(judged) || cells <= 0) return null;
  const share = finiteNumber(c.judged_share) ? c.judged_share : judged / cells;
  const count = key => finiteNumber(c[key]) ? c[key] : null;
  return { cells, judged, share, empty: count('empty'), under20: count('under20'), under100: count('under100') };
}
/** Transcript retention (`retention {ratio, start_tx, tx}`): proseg's count of all cells now over the cells it started
 *  with, kept the same way (the frame's start block), or null. */
export function retentionOf(item) {
  const r = item?.retention;
  if (finiteNumber(r?.ratio) && r.ratio >= 0) return { ratio: r.ratio, tx: finiteNumber(r.tx) ? r.tx : null, start: finiteNumber(r.start_tx) ? r.start_tx : null };
  return null;
}
/** Which way the transcripts the cells kept move the headline by themselves: 'down' under RETENTION_LOW, 'up' from
 *  RETENTION_HIGH on, else null (no claim). */
export function retentionMoves(retention) {
  if (!retention) return null;
  return retention.ratio < RETENTION_LOW ? 'down' : retention.ratio >= RETENTION_HIGH ? 'up' : null;
}
/** The retention line in view ("Cells hold 115 % of their start RNA": a whole percent) and its ⓘ, only when it moves
 *  the number (retentionMoves), else '' (the headline's ⓘ says it as a sentence: retentionSentence). Either ends with
 *  what it compares (retentionBasis). */
export function retentionLine(retention) {
  return retentionMoves(retention) ? copy('retentionShort', { x: number(Math.round(retention.ratio * 100)) }) : '';
}
export function retentionTooltip(retention) {
  const moves = retentionMoves(retention);
  return moves ? `${moves === 'up' ? QC_TEXT.retentionUpTooltip : QC_TEXT.retentionDownTooltip} ${QC_TEXT.retentionBasis}` : '';
}
export function retentionSentence(retention) {
  return retention ? `${copy('retention', { x: number(Math.round(retention.ratio * 100)) })}. ${QC_TEXT.retentionBasis}` : '';
}

/** A model's learning table in plain words (never a zarr key such as "xenium_table" on screen): Xenium's cells, else
 *  the sample's other cell table. */
export function tableName(key) {
  if (typeof key !== 'string' || !key) return null;
  return /^xenium(_table)?$/i.test(key) ? QC_TEXT.tableXenium : QC_TEXT.tableOther;
}

/** "How is this estimated?": its five points, in order. */
export const HOW_KEYS = Object.freeze(['how1', 'how2', 'how3', 'how4', 'how5']);

/** The provenance line under the five points: how many cells the model learned from, on which day and in
 *  how long (`date` and `t` as the page writes them), with the passes over the cells when a small table took more
 *  than usual (model.provenance difference more_epochs_small_table);
 *  its tooltip names the table in plain words and why the passes. `model` is the sample's model
 *  ({cells, table}), `provenance` the model's ({table, epochs_run, differences, ...}) or null for a model learned
 *  before it was recorded. Null without the model's cell count. Go deeper's "How is this worked out?" can end with the
 *  same line. */
export function learnedLine(model, provenance, { date, t } = {}) {
  if (!finiteNumber(model?.cells)) return null;
  const passes = Array.isArray(provenance?.differences) && provenance.differences.includes('more_epochs_small_table')
    && finiteNumber(provenance.epochs_run) ? provenance.epochs_run : null;
  const text = passes != null ? copy('howLearnedPasses', { cells: number(model.cells), e: number(passes), date: date ?? QC_TEXT.missing, t: t ?? QC_TEXT.missing })
    : copy('howLearned', { cells: number(model.cells), date: date ?? QC_TEXT.missing, t: t ?? QC_TEXT.missing });
  const key = typeof provenance?.table === 'string' && provenance.table ? provenance.table : model.table;
  const table = tableName(key);
  const tooltip = [table ? copy('howTable', { table }) : '', passes != null ? QC_TEXT.howPasses : ''].filter(Boolean).join(' ');
  return { text, tooltip };
}
