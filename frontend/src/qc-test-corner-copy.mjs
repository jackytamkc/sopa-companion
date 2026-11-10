// The test corner: every word of the floating view of a setting test that is the corner's own.
// Templates use named values, never HTML. Short words on screen; what they mean goes into tooltips. The check's
// banned-word rule (qc-copy.mjs BANNED_WORDS), Go deeper's plain-word list and the mixing check's honesty rules
// hold for every string (frontend/test/qc-test-corner.test.mjs). The suspects are transcripts proseg counted for the
// cell that likely came from a neighbour (the model splits a cell's RNA gene by gene, so one by one each is only
// "likely"); the views say how much of each proseg still counts for the cell in each run, and the cell's number is
// proseg's own count, never the transcripts inside its outline. The form's lines, the progress, the answer's words and
// the answers' buttons come from the panel's model (qc-panel.js, its words qc-copy.mjs's), not from here.
import { bannedWords } from './qc-copy.mjs';

export { bannedWords };

export const CORNER_TEXT = Object.freeze({
  // ---- the frame
  titleForm: 'Test a setting',
  titleRunning: 'Testing a setting',
  titleDone: 'Test result',
  titleEnded: 'Test stopped',
  drag: 'Drag to move · double-click to put it back',
  minimise: 'Make smaller',
  restore: 'Make bigger',
  close: 'Close',
  closeTooltip: 'Your run goes on as it was before the test.',
  // ---- choosing: one setting, its value; the pool, the area and the runs (the panel's lines); why a test can't run
  // here (the engine's refusals, in the panel's words), in two groups (qc-copy.mjs testReasonGroup), each under its
  // heading with what it means in the heading's tooltip. What a press would wait for first and where the test was not
  // measured follow Run's own tooltip (the panel's words).
  settingLabel: 'Setting',
  valueLabel: 'Value',
  // A setting offered as values to pick: the buttons' group, and the run's own one (not to be picked).
  valueChoiceLabel: 'Values',
  choiceNowTooltip: "Your run's value now",
  reasonsFirst: 'First:',
  reasonsFirstTooltip: 'Wait for this to end before the test can run.',
  reasonsCant: "Can't test here:",
  reasonsCantTooltip: "The test has nothing to measure for this cell, or it can't run on this run or sample.",
  run: 'Run the test',
  runTooltip: 'Starts the test: fresh runs of this area to the end of the schedule, as now and with this value. Your run waits meanwhile.',
  // What a press of Run still needs on the page (a cell, a value within the setting's range, ...): the engine's refusals
  // are never listed under it (a press while they are listed says one line of the panel's).
  problemsLead: 'Check these first:',
  // ---- watching: one run as now, one with the value (the first of each side), side by side
  asNow: 'As now',
  asNowLabel: 'A run as now, around this cell',
  testLabel: 'A run with the value, around this cell',
  waiting: 'Waiting for the first iteration…',
  noFrames: 'No iterations were kept on this page.',
  notFound: 'Not in this run',
  notFoundTooltip: "This cell's name isn't among this run's cells, so its numbers can't be shown here.",
  // A teal ring: a suspect proseg keeps under a fifth of for the cell (MOVED_BELOW; "Suspects kept" adds its share in).
  suspectsTooltip: 'Transcripts proseg counted for this cell that likely came from a neighbour. Glowing: proseg still counts it for this cell in this run (brighter: a larger share). Teal ring: proseg keeps under a fifth of it for this cell here.',
  totalTooltip: "Proseg's count of this cell in this run, averaged over its random background draw.",
  // A view whose run's newest counted iteration came without proseg's export of the view: the shares from its labels;
  // while the test runs a later iteration may bring the export, once its runs ended none comes (approxDoneTooltip).
  approxTooltip: "Approximate in this view: proseg's exact share comes with a later iteration.",
  approxDoneTooltip: "Approximate in this view: placed from this run's own layers, as no exact share of proseg's came for it.",
  // A view whose run holds the cell but placed none of its suspects (no share of proseg's known for any of them there).
  notPlaced: 'Suspects not placed in this view',
  notPlacedTooltip: "Proseg's share of these suspects isn't known in this view's iteration, so none is drawn or counted here.",
  starting: 'Starting…',
  cancel: 'Cancel',
  cancelTooltip: 'Stops the test and deletes its runs. Your run goes on as it was before the test.',
  // The views' key (the cards' colours: the cell white, neighbours magenta, the glow pink, left the cell teal), and what
  // the other cells' colours are: the image's when the view names each cell's place in your run, else one of their own.
  keyCell: 'This cell',
  keyCellTooltip: "White: this cell's outline in each run, matched by its name.",
  colorsStage: 'Every other cell has the colour it has on the image.',
  colorsOwn: 'Every other cell has a colour of its own.',
  keyNeighbours: 'Neighbours',
  keyNeighboursTooltip: 'Magenta: the neighbours its RNA seems to come from, matched by their names.',
  keySuspect: 'Likely from neighbours',
  keySuspectTooltip: 'Glowing: proseg still counts it for this cell.',
  keyMoved: 'Mostly left the cell',
  keyMovedTooltip: 'Teal ring: proseg keeps under a fifth of it for this cell in that run.',
  // A suspect the test runs couldn't tell apart from another transcript of its gene at its place (engine/qc_test.py
  // match: two layers at one spot can be two cells'): a small grey ring, left out of the counts.
  unsureTooltip: "Another transcript of the same gene lies at the same spot, a little deeper or shallower in the tissue, and the test runs couldn't tell which of the two this one is. Each is a small grey ring, left out of the counts.",
  // The views at the answer could not be had (the engine's last views, asked a few times): what they show may be older.
  stale: "Couldn't get the views at the answer: they may show an earlier iteration.",
  staleTooltip: 'The page asked again a few times and got no reply. The answer is from the runs as they ended; only these views may be older.',
  // ---- the answer (its words are the panel's): Replay and the scrubber under the views
  again: 'Run again',
  againTooltip: 'Runs the same test again, with fresh runs.',
  replay: 'Replay',
  replayStop: 'Stop',
  replayTooltip: 'Plays the iterations this page kept, from the first.',
  scrubLabel: 'Iterations kept on this page',
  scrubTooltip: 'Drag to look back through the iterations this page kept while the test ran (the test runs themselves are deleted).',
  // The ⓘ of the answer's verdict on fewer suspects: how often the test is wrong, then where it was not measured (its
  // label naming what its tooltip holds, one or both); with no such verdict (they came and went), an ⓘ of its own by the
  // first part's word (the panel's tooltip words).
  aboutLabel: 'How often this test is wrong',
  aboutCaveatsLabel: 'How often this test is wrong, and where it was not measured',
  caveatsLabel: 'Where this test was not measured',
  // ---- a test that stopped (the panel's own words come first)
  failed: "The test couldn't finish. Your settings are unchanged.",
  cancelled: 'The test was cancelled. Your settings are unchanged.',
  missing: '–',
  sep: ' · ',
});

export const CORNER_TEMPLATES = Object.freeze({
  withValue: 'With {v}',
  now: 'now {v}',
  // Of the suspects each run could tell apart, how many proseg still counts for the cell there (its shares added up).
  suspectsInside: 'Suspects kept: {a} of {n}',
  // How many suspects the test runs couldn't tell apart (left out of "of {n}" above).
  unsure: "{n} couldn't be told apart from another transcript at the same place",
  transcripts: "Proseg's count: {t}",
  scrubAt: 'Iteration {a} · {b}',
  sub: '{setting} · {cell}',
});

/** A string of the deck: missing values fail early rather than leaking braces or "undefined" to the page. */
export function corner(key, values = {}) {
  if (Object.hasOwn(CORNER_TEXT, key)) return CORNER_TEXT[key];
  if (!Object.hasOwn(CORNER_TEMPLATES, key)) throw new RangeError(`Unknown test corner copy key: ${key}`);
  return CORNER_TEMPLATES[key].replace(/\{([^}]+)\}/g, (_, name) => {
    if (!Object.hasOwn(values, name) || values[name] == null) throw new TypeError(`Missing test corner copy value: ${key}.${name}`);
    return String(values[name]);
  });
}

/** Every string the corner can show from this deck, templates filled with sample values (the banned-word tests). */
export function allTexts() {
  const sample = { v: '0.08', a: '12', b: '5', n: '160', t: '519', setting: 'Far drift', cell: 'ffaaaaaa-1' };
  return [...Object.values(CORNER_TEXT), ...Object.keys(CORNER_TEMPLATES).map(key => corner(key, sample))];
}
