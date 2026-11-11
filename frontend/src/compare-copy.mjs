// Compare: every word of the full-screen view that sets the run's cells and the
// sample's other segmentations side by side, each column's UMAP on top and its mixing below. Templates use named values,
// never HTML. Short words on screen; what they mean goes into tooltips (the counting rule, the area, the transfer caveat,
// the like-for-like rule). Each column is read on its own count, which its head names (proseg's for proseg's cells,
// Xenium's for Xenium's cells, else the transcripts inside its outlines); proseg's sets background aside, said once in
// the counting rule's tooltip. The check's banned-word rule (qc-copy.mjs BANNED_WORDS), Go deeper's plain-word list and the
// honesty rules of the mixing check hold for every string (frontend/test/compare.test.mjs). No word sets one
// column against another (no "more", "less", "fewer", no arrows: the mixing check's thresholds were set on repeat runs of
// one segmentation, not on different segmentations), and a kind is never called a cell type (the kinds are the mixing
// model's). The columns' names (the engine's) and the sample's shapes keys come as data. The Map switch: One map (every
// column laid out together) or Each its own (each laid out and grouped alone, lined up with the yardstick column: the
// run's when it is a column, else the first); Differences and the clusters' numbers are set beside that column, and say
// "under", "over" or "about the same", never a verdict.
import { bannedWords } from './qc-copy.mjs';

export { bannedWords };

export const COMPARE_TEXT = Object.freeze({
  // ---- the page's header button (button#compare-button, main.js): never greyed out, a press that cannot open says why
  button: 'Compare',
  buttonTooltip: "Set this run's cells beside the sample's other segmentations: their UMAPs and their mixing, side by side.",
  needsCheck: 'Compare reads every column with the ResolVI mixing check: switch it on for this run first.',
  needsRun: "Open a run first: Compare sets its cells beside the sample's other segmentations.",
  // ---- the header
  title: 'Compare',
  sub: 'UMAP and mixing, side by side',
  add: 'Add',
  addTooltip: "Add one of this sample's segmentations as a column (4 columns at the most).",
  addTitle: 'Add a column',
  addFull: '4 columns at the most: take one out first.',
  addNone: 'Every segmentation of this sample is a column already.',
  addLoading: "Reading this sample's segmentations…",
  addFailed: "Couldn't list this sample's segmentations.",
  colourBy: 'Colour by',
  byKind: 'Kind',
  byKindTooltip: 'Each cell in the colour of its kind: one colour is one kind in every column.',
  byMixing: 'Mixing',
  byMixingTooltip: 'Each cell in the colour of how much of its RNA the mixing model (ResolVI) reads as from neighbours, as on the tissue.',
  // The Map switch (remembered in this browser) and the third colouring it brings: Clusters on each column's own map,
  // Differences on the one map.
  mapBy: 'Map',
  mapOne: 'One map',
  mapOneTooltip: "Every column's cells laid out together: the same place is the same kind of cell in every column.",
  mapOwn: 'Each its own',
  byClusters: 'Clusters',
  byDifferences: 'Differences',
  // The yardstick column in the templates ({ref}): the run's when it is a column, else the first.
  refRunColumn: "the run's column",
  refFirstColumn: 'the first column',
  refRun: "the run's",
  refFirst: "the first column's",
  back: 'Back',
  backTooltip: 'Back to the run, which kept going meanwhile (Esc).',
  remove: 'Take this column out',
  keepOne: 'One column at least stays.',
  tryAgain: 'Try again',
  // ---- the area all columns cover, and how far the work is
  areaFinding: 'Finding the area all of them cover…',
  smallCover: "Under a tenth of another column's area",
  smallCoverTooltip: 'Every column is read on the area all of them cover, so the others are cut down to its part.',
  starting: 'Starting…',
  working: 'Working…',
  stepQueued: 'Waiting its turn…',
  stepOutlines: "Reading each column's outlines…",
  stepArea: 'Finding where every column has cells…',
  stepCounting: "Reading each column's counts…",
  stepScoring: 'Reading the mixing and laying out the UMAPs…',
  stepKinds: 'Sorting the cells into kinds…',
  stepUmap: 'Laying out the UMAPs…',
  retrying: 'No answer yet: asking again…',
  unreachable: "The engine isn't answering.",
  failed: "Couldn't compare.",
  // ---- a column
  runNameNoIter: 'This run',
  sourceXenium: 'Xenium',
  sourceImported: 'Imported',
  sourceCellpose: 'Cellpose',
  // A segmentation the dormant InstanSeg saved earlier: an ordinary saved stain segmentation (the engine names it so).
  sourceInstanseg: 'Stain segmentation',
  sourceProseg: 'proseg (this app)',
  columnWorking: 'Working…',
  columnWaiting: 'Waiting its turn…',
  columnCounting: 'Reading its counts…',
  columnFailed: "Couldn't work this column out.",
  // Which count a column is read on (the engine's column `count`), under its name; in its tooltip, what that count is
  // (then countingBias while a column is on proseg's count: COUNT_TOOLTIPS). Proseg's count by whose it is (H9): the
  // run's column by its form (averaged, or as saved on an iteration counted for a pause, the hold, a finish or the end),
  // a table proseg saved in the sample as that table (one draw of its count).
  countProsegAveraged: "Proseg's count (averaged)",
  countProsegSaved: "Proseg's count (as saved)",
  countProsegTable: "Proseg's saved table",
  countXenium: "Xenium's count",
  countOutlines: 'Transcripts inside its outlines',
  countProsegTooltip: "This column reads proseg's own count of each cell at this iteration, averaged over its random background draw.",
  countProsegSavedTooltip: "This column reads proseg's own count of each cell at this iteration as proseg saves it: one random background draw, not averaged.",
  countProsegTableTooltip: 'This column reads a table proseg saved in this sample: one random background draw of its count, as proseg saves it.',
  countXeniumTooltip: "This column reads Xenium's own count of each cell, from its cell table.",
  countOutlinesTooltip: 'This segmentation has no count of its own: this column counts the transcripts inside its outlines.',
  umapWaiting: 'Waiting for the UMAP…',
  umapNone: 'Too few cells for a UMAP.',
  umapFailed: "Couldn't load this UMAP.",
  layoutFailed: "Couldn't lay out the UMAPs this time.",
  ownNone: 'Too few cells for a map of its own.',
  ownFailed: "Couldn't lay out its own map this time.",
  shownTooltip: 'The UMAPs hold the cells with at least 100 transcripts in their count in the area, 60,000 at the most in all: a column with too many for its share shows an even pick of them.',
  // ---- a column's mixing
  mixedTail: 'in 100 cells look mixed',
  transfer: "The mixing model learned from one of this sample's cell tables; every other segmentation is read by it as it stands, so a small gap between columns may come from the model rather than the outlines.",
  sideBySide: "The columns stand side by side with no verdict: how far apart two segmentations must be to differ hasn't been measured.",
  // The bias of proseg's count, after the counting rule when a column is on it.
  countingBias: "Proseg's count sets background aside, and that alone takes some mixing out of its columns.",
  // Whose draw the run column's cut to 100 transcripts starts from (the engine's like_from, H6): proseg's own (its
  // saved count, on an iteration that has it), else one the app draws from proseg's count.
  likeFromProseg: "Each cut starts from proseg's own draw at this iteration, the count it saves.",
  likeFromApp: "Proseg kept no draw at this iteration, so each cut starts from a random draw the app makes from proseg's count.",
  rough: 'rough',
  roughTooltip: 'Under 300 cells with at least 100 transcripts in their count in the area, so this number is rough.',
  histTitle: 'RNA from neighbours, per cell (%)',
  histTooltip: "How the cells with at least 100 transcripts in their count spread by the share of their RNA the mixing model reads as from neighbours. The colours are the tissue's: under 10 %, 10–30 %, 30 % or above (looks mixed).",
  statCells: 'cells in the area',
  statCellsTooltip: "This column's cells whose outline has its middle in the area.",
  statMedian: 'median transcripts',
  statMedianTooltip: "The median of this column's transcripts per cell in the area, in its own count.",
  statUnder: 'under 100 transcripts',
  statUnderTooltip: 'Cells with under 100 transcripts are not judged: too few to read.',
  // ---- the key under the columns
  kindsTitle: 'Kinds',
  kindsTooltip: "Groups of cells the mixing model tells apart, each named by three genes it makes most. One colour is one kind in every column: each column's cells take the kind nearest to them.",
  kindsNeeded: "Colours by kind come with the mixing model's details, which this sample hasn't learned yet.",
  kindsComing: 'The kinds come with the results.',
  kindsUnavailable: "The kinds couldn't be worked out for this sample's model, so the cells are coloured by mixing.",
  // The engine's kinds_reason "busy": the sample was being saved, so the model's kinds could not be read this time (a new
  // Compare makes them).
  kindsBusy: "The kinds couldn't be read while this sample was being saved, so the cells are coloured by mixing. Open Compare again once the save is done.",
  learn: 'Learn the details',
  learnTooltip: 'Learns them once, on the graphics card (a few minutes); the mixing numbers stay as they are.',
  learning: 'Learning the details: a few minutes, once. Open Compare again when it is done.',
  learnFailed: "Couldn't start learning the details.",
  mixingTitle: 'RNA from neighbours',
  mixLittle: 'Under 10 %',
  mixSome: '10–30 %',
  mixMixed: '30 % or above (looks mixed)',
  mixUnjudged: 'Too few transcripts',
  clustersTitle: 'Clusters',
  // Differences' five steps: a column's share of cells around a place over the yardstick's share there.
  diffTitle: 'Differences',
  diffThird: '⅓ or under',
  diffHalf: 'About ½',
  diffSame: 'About the same',
  diffTwice: 'About 2×',
  diffThrice: '3× or over',
  noKind: 'No kind',
  hoverMixed: 'looks mixed',
  missing: '–',
  sep: ' · ',
});

// The histogram's bins (% of a cell's RNA from neighbours), as the engine counts them (0-2, 2-5, 5-10,
// 10-20, 20-30, 30-50, 50+).
export const BIN_LABELS = Object.freeze(['0–2', '2–5', '5–10', '10–20', '20–30', '30–50', '50+']);

export const COMPARE_TEMPLATES = Object.freeze({
  runName: 'This run · iteration {i}',
  // A column's second line: its key and what made it ("cellpose_test · Cellpose cp4_20261021").
  colSub: '{key} · {source}',
  withModel: '{source} {model}',
  // A segmentation in Add's menu: its key, what made it and how many outlines it holds.
  outlines: '{n} outlines',
  areaAll: 'On the area all of them cover: {size}',
  areaOne: 'On the area it covers: {size}',
  areaTooltip: 'Every number and UMAP here uses only the cells in the {n} squares of 100 µm that hold at least 3 cells of every column. A cell counts when the middle of its outline lies in one.',
  mm: '{w} × {h} mm',
  um: '{w} × {h} µm',
  progress: '{step} {done} of {of}',
  failedWhy: "Couldn't compare: {why}",
  umapLabel: "UMAP of {name}'s cells",
  shown: '{n} of {m} cells shown',
  mixed: '{n} in 100 cells look mixed',
  // The counting rule (in the number's ⓘ): each column on its own count, and for a segmentation with none of its own the
  // run's own input filters, as the engine counted it (its `rule`): quality {q} or above (20 without one); a run that
  // left out genes of its own says so, its pattern never shown. With a column on proseg's count, the bias of that count
  // follows (countingBias, COMPARE_TEXT).
  counting: "Each column uses its own count: proseg's for proseg's cells, Xenium's for Xenium's cells, and for a segmentation with no count of its own each transcript of quality {q} or above, of the mixing model's genes, inside its outlines (in 2-D).",
  countingExcluded: "Each column uses its own count: proseg's for proseg's cells, Xenium's for Xenium's cells, and for a segmentation with no count of its own each transcript of quality {q} or above, of the mixing model's genes except the ones this run left out, inside its outlines (in 2-D).",
  mixedTooFew: 'Too few cells to say ({j} with at least 100 transcripts)',
  mixedTooltip: 'Of the {j} cells in the area with at least 100 transcripts in their count, those the mixing model reads as holding 30 % or above of their RNA from neighbours.',
  like: 'At 100 transcripts each: {n} in 100',
  likeTooltip: "Like for like: each of the {j} cells with at least 100 transcripts in their count cut down to exactly 100 of them, picked at random, three times over, and the share that looks mixed averaged, so cells of every size read alike. Levels change when cells are cut down: hold this line against the other columns' same line, not against the line above.",
  bar: '{bin} %: {p} in 100 of these cells ({n})',
  percent: '{p} %',
  hoverShare: 'About {p} % of its RNA from neighbours',
  // Each its own and its Clusters; Differences on the one map ({ref}: the yardstick column, refRunColumn or
  // refFirstColumn; in diffTip refRun or refFirst).
  mapOwnTooltip: "Each column's cells laid out alone and grouped alone, as if it were analysed by itself; every map is drawn at one size, then turned or flipped to line up with {ref}, never reshaped.",
  byClustersTooltip: "Each column's own groups of cells, found on its own map. A group whose genes match one in {ref} takes its number and colour; any other group has its own.",
  byDifferencesTooltip: "Where each column's cells gather on the one map, set beside {ref}: each dot is coloured by its column's share of cells around it over the share {ref} has there.",
  clustersKeyTooltip: "Each column's own groups, numbered as in {ref} where their genes match. Point at a number to light it in every column.",
  diffKeyTooltip: "A column's share of cells around a place over the share {ref} has there: grey about the same, blue under, orange over. The dots of {ref} are all grey.",
  clusterKey: '{n}',
  clusterKeyTooltip: 'Cluster {n}: in {where}',
  clusterTip: 'Cluster {n}',
  diffTip: 'Share of cells here: {r}× {ref}',
});

/** A string of the deck: missing values fail early rather than leaking braces or "undefined" to the page. */
export function compare(key, values = {}) {
  if (Object.hasOwn(COMPARE_TEXT, key)) return COMPARE_TEXT[key];
  if (!Object.hasOwn(COMPARE_TEMPLATES, key)) throw new RangeError(`Unknown compare copy key: ${key}`);
  return COMPARE_TEMPLATES[key].replace(/\{([^}]+)\}/g, (_, name) => {
    if (!Object.hasOwn(values, name) || values[name] == null) throw new TypeError(`Missing compare copy value: ${key}.${name}`);
    return String(values[name]);
  });
}

/** Every string the view can show from this deck, templates filled with sample values (the banned-word tests). */
export function allTexts() {
  const sample = { i: '450', key: 'cellpose_test', source: COMPARE_TEXT.sourceCellpose, model: 'cp4_20261021', size: '2.1 × 1.4 mm', n: '13',
    w: '2.1', h: '1.4', step: COMPARE_TEXT.stepCounting, done: '2', of: '5', why: 'the run has no frame yet', name: 'Xenium cells', m: '41,775',
    j: '4,987', p: '12', bin: BIN_LABELS[3], q: '20', ref: COMPARE_TEXT.refRunColumn, where: 'This run · iteration 450, Xenium cells',
    r: '2.1' };
  return [...Object.values(COMPARE_TEXT), ...BIN_LABELS, ...Object.keys(COMPARE_TEMPLATES).map(key => compare(key, sample))];
}
