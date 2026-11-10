import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  DEEP_TEXT, DEEP_TEMPLATES, deep, allTexts, headlinePercent, stepKey, startChange, barLabels, chipText, largestRemainder, rowTooltip,
  sharePct, foldText, sameKindCountText, partlyText, howItems, learnedLine, dateText, frameTag, cantRunText, detailFailedText,
  lookAlikeLine, partPercent, startLine, startTip, glowTip, noOutlineLine, START_PARTS, underFloor, count, LIKE_NOW,
} from '../src/qc-deeper-copy.mjs';
import { bannedWords, QC_TEXT, QC_TEMPLATES, cardLine } from '../src/qc-copy.mjs';
import { PLACES } from '../src/qc-deeper.js';

// The check's own strings the window reuses: each must stay one of the check's.
const REUSED = ['learningProgress', 'learning', 'learningNoTime', 'waiting', 'waitingQueue', 'waitingLock', 'cardShared',
  'tryAgain', 'xeniumStart', 'otherStart', 'failed', 'softwareChanged'];

// Go deeper's copy deck (with the shorter window
// and its later words): plain words only, under the check's own banned-word rule,
// "ResolVI" only in "How is this worked out?" item 1, and never the user's own files.
const words = text => String(text).split(/\s+/).filter(t => /[\p{L}\p{N}]/u.test(t)).length;

test('every Go deeper string passes the banned-word rule, expanded or not; "ResolVI" stands only in deepHow1; no notebook anywhere', () => {
  // Each explanation is split into a short line and its tooltip (117 and 88 keys when it was written).
  assert.ok(Object.keys(DEEP_TEXT).length >= 100 && Object.keys(DEEP_TEMPLATES).length >= 80);
  for (const [key, text] of Object.entries({ ...DEEP_TEXT, ...DEEP_TEMPLATES })) {
    assert.deepEqual(bannedWords(text), [], key);
    assert.doesNotMatch(text, /\b(band|depth-matched|compactness|posterior|ambient RNA|contaminat)/i, `${key}: plain words`);
    if (/resolvi/i.test(text)) assert.equal(key, 'deepHow1', key);
  }
  for (const text of allTexts()) {
    assert.deepEqual(bannedWords(text), [], text);
    // The user's own files are never named (no notebook, no notebook's numbers).
    assert.doesNotMatch(text, /notebook|part\s*3|\byour\s+(?:own\s+)?(?:file|notebook|analysis)/i, text);
  }
  for (const key of REUSED) assert.ok(Object.hasOwn(QC_TEXT, key) || Object.hasOwn(QC_TEMPLATES, key), `${key}: the check's own string`);
  // "ambient" appears once, explained, in the loose RNA's tooltip (the user's own word for it).
  assert.match(DEEP_TEXT.deepLooseTooltip, /often called ambient or background RNA/);
});

test("the window works on any sample: the model learned from the sample's own cells, never \"Xenium's cells\"; no sample, tissue or cell type named", () => {
  // The model learned from the sample's own cell table (Xenium's, a CosMx sample's cells, ...): the neighbours' guide
  // says the sample's cells; the start is named by its own source ({start}: startName), never by the deck itself.
  assert.equal(DEEP_TEXT.deepLearnedNote, "The model learned which neighbours each of the sample's cells borrows from; proseg keeps the same cells, so this carries that over and adjusts it for distance.");
  for (const [key, text] of Object.entries({ ...DEEP_TEXT, ...DEEP_TEMPLATES })) {
    assert.doesNotMatch(text, /Xenium/, `${key}: Xenium's cells only through {start}`);
    assert.doesNotMatch(text, /\b[A-Z]{2}\d{2}\b|liver|hepat/i, `${key}: no sample code, tissue or cell type`);
  }
  for (const text of allTexts()) assert.doesNotMatch(text, /\b[A-Z]{2}\d{2}\b|liver|hepat/i, text);
});

test('templates fill every named value, and a missing value or an unknown key fails early', () => {
  assert.deepEqual(Object.keys(DEEP_TEXT).filter(k => k in DEEP_TEMPLATES), []);
  for (const [key, pattern] of Object.entries(DEEP_TEMPLATES)) {
    const names = [...new Set([...pattern.matchAll(/\{([^}]+)\}/g)].map(m => m[1]))];
    const values = Object.fromEntries(names.map((n, i) => [n, `V${i}`]));
    const text = deep(key, values);
    assert.doesNotMatch(text, /\{[^}]+\}|undefined|null/, key);
    for (const v of Object.values(values)) assert.ok(text.includes(v), `${key}: ${v}`);
    for (const name of names) { const missing = { ...values }; delete missing[name]; assert.throws(() => deep(key, missing), TypeError, `${key}.${name}`); }
  }
  assert.throws(() => deep('nope'), RangeError);
});

test('the words: labels and a few-word verdicts on screen, the explanations kept in their tooltips', () => {
  assert.equal(DEEP_TEXT.deepButton, 'Go deeper ›'); assert.equal(DEEP_TEXT.deepButtonOpen, 'Go deeper (open)');
  // The subtitle is the cell's name's tooltip; the pinned note is four words with the sentence in its tooltip.
  assert.equal(DEEP_TEXT.deepSubtitle, 'Where its RNA seems to come from (estimate)');
  assert.equal(DEEP_TEXT.deepPinned, 'Last cell you clicked'); assert.match(DEEP_TEXT.deepPinnedTooltip, /click another cell/);
  // The headline: "29 %" + "from neighbours", the card's meaning and the blind spot in its tooltip.
  assert.equal(DEEP_TEXT.deepNumberLabel, 'from neighbours');
  assert.match(DEEP_TEXT.deepNumberTooltip, /^The share of its RNA the model reads as coming from neighbouring cells: an estimate on proseg's count, not a count of outline errors\. Between cells of the same kind it can't tell RNA apart/);
  for (const key of ['deepNumberLabel', 'deepNumberTooltip', 'deepStepsTooltip', 'deepNeighboursTooltip', 'deepOwnTooltip']) {
    assert.doesNotMatch(DEEP_TEXT[key], /another kind of cell|of a different kind/i, `${key}: the whole share is never "another kind"`);
  }
  // "Like for like" is in the step line's tooltip, and on the line itself when it states a change or "same as at the
  // start" ("start 1 %" beside the small line can read the other way); never on the start's own number.
  assert.match(DEEP_TEXT.deepStepsTooltip, /like for like/); assert.doesNotMatch(DEEP_TEXT.deepUpFromStart, /like for like/);
  assert.equal(deep('deepStepLike', { step: DEEP_TEXT.deepSome, change: DEEP_TEXT.deepSameAsStart }), 'Some mixing · same as at the start · like for like');
  assert.equal(deep('deepStepChange', { step: DEEP_TEXT.deepSome, change: '1 % at the start' }), 'Some mixing · 1 % at the start');
  // The tag rule is "4 times as much in the giving neighbour's kind", not "rarely".
  assert.equal(DEEP_TEXT.deepForeignLabel, 'Neighbour genes');
  assert.equal(DEEP_TEMPLATES.deepForeignHeading, "Genes far more typical of a neighbour's kind, so they probably come from neighbours ({k} here).");
  assert.match(DEEP_TEXT.deepTagForeignTooltip, /at least 4 times as much as this cell's own kind/);
  assert.equal(DEEP_TEXT.deepSourcesHeading, 'Neighbours'); assert.match(DEEP_TEXT.deepSourcesGuide, /a rough guide/);
  assert.equal(DEEP_TEXT.deepShowSameKind, 'Same kind as this cell: no gene tells its RNA apart', 'a same-kind row\'s Show explains');
  assert.equal(DEEP_TEXT.deepWhereShow, 'Show on the image');
  assert.equal(DEEP_TEXT.deepBlindSpot, 'Between cells of the same kind the model can\'t tell RNA apart: it can miss RNA from such a neighbour, or show some it can\'t back up. "Little from neighbours" doesn\'t mean "correct".');
  assert.equal(DEEP_TEXT.deepEstimate, 'An estimate, not a measurement');
  // Every count is proseg's own (the decision of 2026-10-30): the basis says it and which count (averaged, or as
  // saved: deepSavedWords below), its tooltip what proseg sets aside as background (averaged over its random background
  // draw: never "expected"); the second bar that set it against the outline, and the saved results' line (an
  // outline-era comparison), are gone.
  assert.equal(deep('deepBasisShort', { n: '682' }), "Proseg's count (averaged): 682");
  assert.equal(deep('deepBasis', { bg: '165' }), 'The transcripts proseg places in this cell, less 165 it reads as background, averaged over its random background draw. The saved table holds one draw of this count.');
  assert.equal(DEEP_TEXT.deepBasisNoBg, 'The transcripts proseg places in this cell, less those it reads as background, averaged over its random background draw. The saved table holds one draw of this count.');
  assert.equal(deep('deepBasisStart', { n: '161' }), "At the start: 161, proseg's count at iteration 0.");
  for (const gone of ['deepProsegTooltip', 'deepProsegKept', 'deepProsegBackground', 'deepProsegMoved', 'deepSaved', 'deepSavedTooltip'])
    assert.ok(!Object.hasOwn(DEEP_TEXT, gone) && !Object.hasOwn(DEEP_TEMPLATES, gone), gone);
  for (const text of allTexts()) assert.doesNotMatch(text, /inside its outline|moved out, net|\bexpected\b|a little cleaner/i, text);
  assert.equal(DEEP_TEXT.deepEmpty, 'Proseg counts no transcripts for this cell in this iteration.');
  assert.equal(DEEP_TEXT.deepTooFewLater, 'This fills in by itself once proseg counts 100 or more for this cell.');
  assert.equal(deep('deepTooFew', { n: '61' }), 'Too few transcripts to split: proseg counts 61 for this cell in this iteration, in the genes the model knows, and at least 100 are needed.');
  // The glow: of what proseg counts for the cell; its caveats when the shares are approximate here, or none.
  assert.equal(DEEP_TEXT.deepGlowTooltip, 'Pink: proseg counts it for this cell, and it likely came from a neighbour. Brighter: likelier. Pink just outside the outline: proseg moved it in.');
  // The dashed white line only where the stage draws it (2-D, a run from cells): never in 3-D, where it is not drawn.
  assert.equal(DEEP_TEXT.deepGlowStartOutline, "The dashed white line is this cell's starting outline.");
  assert.equal(glowTip(true), `${DEEP_TEXT.deepGlowTooltip} ${DEEP_TEXT.deepGlowStartOutline}`); assert.equal(glowTip(), DEEP_TEXT.deepGlowTooltip);
  assert.equal(DEEP_TEXT.deepGlowApprox, "Approximate here: go to the newest iteration or pause for proseg's exact share.");
  // No share held (a tiled frame's view window misses the cell, or the frame's counts are gone): one tooltip true of
  // both, never "the layers of your view only" (a detail frame has all its layers).
  assert.equal(DEEP_TEXT.deepGlowNone, "Proseg's share of these transcripts isn't available for this iteration: go to the newest iteration or pause to see pink dots here.");
  assert.doesNotMatch(DEEP_TEXT.deepGlowNone, /layers|your view|\bpast\b/);
  // Nothing held there: the line on screen gives no verdict, the why in its tooltip (deepGlowNone).
  assert.equal(DEEP_TEXT.deepGlowNoneShort, 'No pink dots for this iteration');
  // "Where they sit" reads proseg's count of the cell (the shares summed), never where transcripts lie alone.
  assert.equal(deep('deepWhereNear', { n: '31', of: '39', m: '21', mof: '80' }), "31 of the 39 proseg counts for this cell lie within 2 µm of it (the rest of this cell's RNA: 21 of 80).");
  assert.equal(deep('deepWhereNearAlone', { n: '31', of: '39' }), '31 of the 39 proseg counts for this cell lie within 2 µm of it.');
  assert.equal(deep('deepWhereFewIn', { n: '4' }), 'Proseg counts 4 for this cell.');
  // The floor is the engine's FEW_IN_CELL, 4 on proseg's count (what 5 transcripts are worth in it).
  assert.equal(DEEP_TEXT.deepWhereFewTooltip, 'Proseg counts fewer than 4 of them for this cell: too few to tell where they sit.');
  assert.equal(deep('deepWhereDensity', { here: '29', giver: '41', around: 'under 1' }), "Per 100 µm², in proseg's count: 29 in this cell · 41 in it · under 1 in the cells around.");
  assert.match(DEEP_TEXT.deepWhereEdgeTooltip, /^Of what proseg counts for this cell, far more of this gene lies within 2 µm of this neighbour/);
  assert.match(DEEP_TEXT.deepWhereSpreadTooltip, /^What proseg counts of it here is spread through the cell/);
  for (const key of ['deepWhereFewTooltip', 'deepWhereEdgeTooltip', 'deepWhereSpreadTooltip']) assert.doesNotMatch(DEEP_TEXT[key], /transcripts (here )?lie/, key);
  // A run without a start line (only a start from cells alone has one): its first frame's words; the start line's own
  // count and note are proseg's at iteration 0 (no "cells hold only their nuclei": such a start has no start line).
  // Paused on its prior, the run button says Start (main.js renderRun): Start or Step moves it on, never Resume.
  assert.equal(DEEP_TEXT.deepNucleiOnlyRun, 'This run started from nuclei only, so the first iteration has nothing to show yet: press Start or Step.');
  assert.equal(DEEP_TEXT.deepNucleiCellsRun, "This run started from cells with their nuclei, and proseg's count at iteration 0 holds only the nuclei, so the first iteration has nothing to show yet: press Start or Step.");
  assert.equal(DEEP_TEXT.deepNoStartRun, 'The first iteration has nothing to show for this run: press Start or Step.');
  for (const key of ['deepNucleiOnlyRun', 'deepNucleiCellsRun', 'deepNoStartRun']) assert.doesNotMatch(DEEP_TEXT[key], /Resume/, key);
  // Every other run (it ended, saves its results or samples on by itself, or is paused later with its first iteration
  // scrubbed back to, where Resume leaves the view): the same words, no button named.
  assert.equal(DEEP_TEXT.deepNucleiOnlyRunEnded, 'This run started from nuclei only, so the first iteration has nothing to show.');
  assert.equal(DEEP_TEXT.deepNucleiCellsRunEnded, "This run started from cells with their nuclei, and proseg's count at iteration 0 holds only the nuclei, so the first iteration has nothing to show.");
  assert.equal(DEEP_TEXT.deepNoStartRunEnded, 'The first iteration has nothing to show for this run.');
  for (const key of ['deepNucleiOnlyRunEnded', 'deepNucleiCellsRunEnded', 'deepNoStartRunEnded']) {
    assert.doesNotMatch(DEEP_TEXT[key], /Resume|Step|\byet\b/, key);
  }
  assert.equal(deep('deepTooFewStart', { n: '59', start: "Xenium's cells" }), 'Too few transcripts to split: proseg counts 59 for this cell at iteration 0, in the genes the model knows, and at least 100 are needed.');
  assert.equal(deep('deepNuclei', { start: "Xenium's cells" }), "At the first iteration proseg's cells are those Xenium's cells drew, and its count already sets part of their RNA aside as background.");
  // Pink dots near none of the neighbour that would be named: their source is likely a cell the outlines miss.
  assert.equal(deep('deepSourceNoOutline', { kind: 'GZFN · GZQN' }), 'Likely from a GZFN · GZQN cell with no outline here');
  assert.equal(DEEP_TEXT.deepSourceNoOutlinePlain, 'Likely from a cell with no outline here');
  assert.equal(deep('deepSourceNoOutlineTooltip', { cell: 'cell-f' }),
    'None of these pink dots lies near cell-f: they more likely come from a cell of its kind that has no outline here.');
  assert.deepEqual(noOutlineLine({ count: 13, likely: 13, near: 0, places: { own: 13, none: 0, giver: 0, other: 0, unknown: 0 } }, { name: 'cell-f', kind: ['GZFN', 'GZQN', 'GZRN', 'GZSN'] }),
    { text: 'Likely from a GZFN · GZQN · GZRN cell with no outline here · 13 inside its own starting outline',
      tip: `None of these pink dots lies near cell-f: they more likely come from a cell of its kind that has no outline here. ${DEEP_TEXT.deepGlowTooltip}` });
  // The glow's line: how many likely come from neighbours, then where they sat at the start (startLine, below); the
  // muted "Most were inside its starting outline: proseg kept them" is gone with its tooltip.
  assert.equal(startLine({ count: 14, likely: 12, places: { own: 9, none: 3, giver: 0, other: 0, unknown: 0 } }, DEEP_TEXT.deepGlowAll),
    '12 likely from neighbours · 9 inside its own starting outline · 3 where no cell started');
  for (const gone of ['deepGlowStartMost', 'deepGlowStartMostTooltip', 'deepGlowStart', 'deepGlowStartOne', 'deepGlowStartNone', 'deepGlowLikely', 'deepGlowNear', 'deepGlowMaybe'])
    assert.ok(!Object.hasOwn(DEEP_TEXT, gone) && !Object.hasOwn(DEEP_TEMPLATES, gone), gone);
  for (const text of allTexts()) assert.doesNotMatch(text, /\bmost were inside|proseg kept them/i, text);
  assert.match(glowTip(true), /The dashed white line is this cell's starting outline\.$/); assert.doesNotMatch(DEEP_TEXT.deepGlowTooltip, /dashed/);
  // The same-kind count is a sentence of the Neighbours label's tooltip; its own top genes, of its kind's.
  assert.equal(deep('deepOwnTop', { genes: 'GYHNN 6 · GXEN 4' }), 'Its top genes here, mostly its own: GYHNN 6 · GXEN 4.');
  // The glow's line says "likely" (a rough guide built on the per-gene split), never that it knows each transcript.
  assert.equal(deep('deepStartLikely', { who: DEEP_TEXT.deepGlowAll, n: '9' }), '9 likely from neighbours');
  assert.match(DEEP_TEXT.deepWhereIntro, /rough guide/); assert.match(DEEP_TEXT.deepGlowTooltip, /likely came from a neighbour/);
  // The folded genes are never "one transcript each".
  assert.equal(deep('deepTableRest', { k: '3,646', tx: '7,292' }), 'And 3,646 more genes holding 7,292 transcripts in all, not listed one by one.');
});

test('the short lines are short: notes at most 8 words, the look-alike line at most 10, labels a few', () => {
  for (const key of ['deepStartUnjudged', 'deepFloor', 'deepFew', 'deepRough', 'deepSourcesNone', 'deepSourcesSpread', 'deepForeignNone',
    'deepGlowSameKind', 'deepOwnGenesNone', 'deep3d', 'deepZoom', 'deepSourceNoOutlinePlain', 'deepGlowNoneShort']) assert.ok(words(DEEP_TEXT[key]) <= 8, `${key}: ${DEEP_TEXT[key]}`);
  for (const key of ['deepOutside', 'deepSameKindShare', 'deepNucleiShort', 'deepGlowUnplaced', 'deepSourcesMore'])
    assert.ok(words(DEEP_TEMPLATES[key]) <= 8, `${key}: ${DEEP_TEMPLATES[key]}`);
  for (const key of ['deepLookAlike', 'deepLookAlikeAll']) assert.ok(words(DEEP_TEXT[key]) <= 10, key);
  // The basis: two words, which count in brackets (one or two) and its number; the line naming no neighbour: eight words
  // and its genes; each place adds at most five words to the line (its head, three).
  for (const key of ['deepBasisShort', 'deepBasisShortSaved']) assert.ok(words(DEEP_TEMPLATES[key].replace(/\{[^}]+\}/g, '')) <= 4, key);
  assert.ok(words(DEEP_TEMPLATES.deepWhereSnapped.replace(/\{[^}]+\}/g, '')) <= 3, 'deepWhereSnapped');
  assert.ok(words(DEEP_TEMPLATES.deepSourceNoOutline.replace(/\{[^}]+\}/g, '')) <= 8, 'deepSourceNoOutline');
  assert.ok(words(DEEP_TEMPLATES.deepStepLike.replace(/\{[^}]+\}/g, '')) <= 6, 'deepStepLike');
  for (const key of ['deepStartLikely', 'deepStartMaybe']) assert.ok(words(DEEP_TEMPLATES[key].replace(/\{[^}]+\}/g, '')) <= 3, key);
  for (const [, key] of START_PARTS) assert.ok(words(DEEP_TEMPLATES[key].replace(/\{[^}]+\}/g, '')) <= 5, key);
  for (const key of ['deepNumberLabel', 'deepForeignLabel', 'deepSourcesHeading', 'deepWhereHeading', 'deepEstimate', 'deepPinned'])
    assert.ok(words(DEEP_TEXT[key]) <= 5, key);
});

test('the numbers: the card\'s percentage, the steps, the change since the start within 3 points', () => {
  for (const code of [0, 4, 24, 25, 74, 75, 116, 249, 250]) assert.equal(headlinePercent(code), /(\d+)\s?%/.exec(cardLine(code, null, 'xenium'))[1]);
  assert.deepEqual([0, 24, 25, 74, 75, 250, 255].map(stepKey), ['deepLittle', 'deepLittle', 'deepSome', 'deepSome', 'deepMixed', 'deepMixed', null]);
  // The change since the start is the worker's like-for-like one (the cell against itself at the same depth), up or
  // down only past its band; without it, the start's number and no direction.
  assert.equal(startChange(116, 2, { points: 44.1, band: 3 }).text, 'more than at the start');
  assert.equal(startChange(2, 116, { points: -44, band: 3 }).text, 'less than at the start');
  assert.equal(startChange(116, 2, { points: 2.2, band: 3 }).text, 'same as at the start', 'a raw 45-point rise that is 2 points like for like: what depth alone gave');
  assert.equal(startChange(116, 2, { points: 4.5, band: 6.1 }).text, 'same as at the start', 'inside the draws\' own spread');
  // The worker's own word wins: kept near its band as last shown (start_word), it does not flip from frame to frame.
  assert.equal(startChange(116, 2, { points: 4.5, band: 6.1, direction: 'more', held: true }).key, 'deepUpFromStart');
  assert.equal(startChange(116, 2, { points: 44.1, band: 3, direction: 'same' }).key, 'deepSameAsStart');
  assert.equal(startChange(2, 116, { points: -2, band: 3, direction: 'fewer' }).key, 'deepDownFromStart');
  assert.equal(startChange(116, 2, { points: 4.5, band: 3, direction: 'sideways' }).key, 'deepUpFromStart', 'an unknown word: its points');
  assert.equal(startChange(116, 2).text, '1 % at the start', 'no like-for-like comparison: no direction');
  // A direction like for like never stands beside the start's own number, which compares other depths: a cell at 8 %
  // whose start byte read 10 % and that reads more like for like says "more than at the start", not "up from 10 %".
  const eight = Math.round(0.08 * 250), ten = Math.round(0.10 * 250);
  assert.equal(deep('deepStepLike', { step: DEEP_TEXT[stepKey(eight)], change: startChange(eight, ten, { points: 6, band: 5 }).text }),
    'Little or no mixing seen · more than at the start · like for like');
  for (const key of ['deepUpFromStart', 'deepDownFromStart', 'deepSameAsStart']) assert.doesNotMatch(textOf(key), /\d|%/, key);
  // Whether the step line says "like for like": for the worker's comparison, never for the start's number.
  assert.deepEqual([startChange(116, 2, { points: 44.1, band: 3 }).like, startChange(116, 2, { points: 2.2, band: 3 }).like,
    startChange(2, 116, { points: -44, band: 3 }).like, startChange(116, 2).like], [true, true, true, false]);
  assert.equal(startChange(60, 55, null).key, 'deepStartWas');
  assert.deepEqual(startChange(60, 255), { unjudged: true }); assert.equal(startChange(60, null), null); assert.equal(startChange(255, 3), null);
  // Under the headline when most of the RNA from neighbours looks like its own kind: a few words, and in the tooltip
  // the different kind's part of all its RNA, rounded down, or that nothing backs the number up.
  assert.deepEqual(lookAlikeLine(143, 0.81), { text: "Mostly its own kind: can't be checked gene by gene",
    tip: "Most of the RNA from neighbours looks like this cell's own kind: only about 10 % of its RNA looks like a different kind of cell, which genes can back up. The rest can't be checked." });
  assert.match(lookAlikeLine(117, 0.92).tip, /only about 3 % of its RNA/);
  assert.deepEqual(lookAlikeLine(85, 1), { text: "All its own kind: can't be checked gene by gene",
    tip: "All of the RNA from neighbours looks like this cell's own kind, so no gene backs it up: treat the 34 % as uncertain." });
  assert.equal(lookAlikeLine(143, 0.49), null); assert.equal(lookAlikeLine(255, 0.9), null);
});

test('the step line says what it compares with its two numbers, the cell\'s share now and at the start at the same depth (P2); an answer without them keeps its words', () => {
  // The worker's start_change carries the cell's share from neighbours now and at the start (percent, 2 decimals, both
  // at the matched depth): the change said short, its numbers whole percents for the step line.
  const less = startChange(116, 102, { points: -4.6, band: 3, direction: 'fewer', now: 38.4, start: 43.0 });
  assert.deepEqual(less, { key: 'deepLikeLess', text: '▼ less', like: true, at: { now: '38', start: '43' } });
  assert.equal(deep('deepStepLikeAt', { step: DEEP_TEXT.deepMixed, change: less.text, ...less.at }), 'Looks mixed · ▼ less, like for like: 38 % now, 43 % at the start');
  assert.deepEqual(startChange(116, 2, { points: 44.1, band: 3, now: 46.2, start: 2.05 }).at, { now: '46', start: '2' });
  assert.equal(startChange(116, 2, { points: 44.1, band: 3, now: 46.2, start: 2.05 }).key, 'deepLikeMore');
  assert.equal(startChange(116, 102, { points: 2.2, band: 3, now: 44.6, start: 42.4 }).text, 'about the same', 'inside its band');
  assert.equal(startChange(116, 102, { points: 4.5, band: 6.1, direction: 'more', held: true, now: 44, start: 39.5 }).key, 'deepLikeMore', "the worker's word wins");
  // A word held a few seconds (the engine keeps a followed cell's word while its answers' own has moved on): said short
  // without numbers, whatever numbers the answer carries (here 42 % now against 43 % at the start, under "more"; the
  // engine's are those the word compared, an earlier answer's).
  const held = startChange(116, 102, { points: -1.2, band: 3, direction: 'more', held: true, now: 41.8, start: 43 });
  assert.deepEqual(held, { key: 'deepLikeMore', text: '▲ more', like: true, held: true, own: 'same' }, 'with what this answer alone reads (F2)');
  assert.equal(deep('deepStepLikeHeld', { step: DEEP_TEXT.deepMixed, change: held.text }), 'Looks mixed · ▲ more, like for like');
  assert.equal(DEEP_TEMPLATES.deepStepLikeHeld, '{step} · {change}, like for like', 'the words of the line with numbers, without them');
  assert.ok(DEEP_TEMPLATES.deepStepLikeAt.startsWith(`${DEEP_TEMPLATES.deepStepLikeHeld}: `));
  // Without both numbers (an answer kept from before them): as before, no `at`.
  for (const like of [{ points: -4.6, band: 3 }, { points: -4.6, band: 3, now: 38.4 }, { points: -4.6, band: 3, now: null, start: 43 }]) {
    const old = startChange(116, 102, like);
    assert.equal(old.key, 'deepDownFromStart'); assert.equal(old.at, undefined);
  }
  // The words exactly, plain, none banned; the new direction words name no number.
  const words = { deepLikeMore: '▲ more', deepLikeLess: '▼ less', deepLikeSame: 'about the same',
    deepLikeNumbersTooltip: "Its two numbers: this cell's share from neighbours now and as proseg started it, both cut to the same number of transcripts. So they can differ from the number above and from the small line's start." };
  for (const [key, text] of Object.entries(words)) { assert.equal(DEEP_TEXT[key], text, key); assert.deepEqual(bannedWords(text), [], key); }
  assert.equal(DEEP_TEMPLATES.deepStepLikeAt, '{step} · {change}, like for like: {now} % now, {start} % at the start');
  assert.match(deep('deepStepLikeAt', { step: DEEP_TEXT.deepSome, change: DEEP_TEXT.deepLikeSame, now: '12', start: '11' }), /like for like/);
});

test('at a rest frame the word since the start is that iteration\'s own, with its own two numbers; while the run plays a held word says so, with what this answer alone reads (F2)', () => {
  // The audit's cell at a pause (it. 432): the engine held "fewer" for the followed cell, its numbers an earlier
  // answer's (26.51 / 32.04), while this answer's own points (-2.09) sit inside the band: about the same.
  const heldAtPause = { points: -2.09, band: 5, draws: 0, now_tx: 401.3, start_tx: 405.3, direction: 'fewer', held: true, now: 26.51, start: 32.04 };
  // At rest (the answer on the count proseg saves): its own word; the numbers are not the ones its points compare, so
  // none stand beside it (never an earlier answer's under this iteration's word).
  assert.deepEqual(startChange(116, 102, heldAtPause, { rest: true }), { key: 'deepSameAsStart', text: 'same as at the start', like: true });
  // ... the same answer with its own numbers (the engine lets a held word go at a rest frame; or the worker kept a word
  // near the band, its numbers its own): its own word with them, as the panel's verdict line at rest.
  const own = { ...heldAtPause, direction: 'same', held: false, now: 29.48, start: 31.57 };
  assert.deepEqual(startChange(116, 102, own, { rest: true }), { key: 'deepLikeSame', text: 'about the same', like: true, at: { now: '29', start: '32' } });
  const kept = { points: -3.1, band: 5, direction: 'fewer', held: true, now: 28.9, start: 32 };
  assert.deepEqual(startChange(116, 102, kept, { rest: true }), { key: 'deepLikeSame', text: 'about the same', like: true, at: { now: '29', start: '32' } });
  assert.deepEqual(startChange(116, 102, { points: -10.67, band: 5, direction: 'fewer', held: false, now: 20.85, start: 31.52 }, { rest: true }).at,
    { now: '21', start: '32' }, 'a word that is its own: as before');
  assert.equal(startChange(116, 102, { points: 7.2, band: 5, direction: 'same', held: true, now: 40.1, start: 32.9 }, { rest: true }).key, 'deepLikeMore',
    'its own word on its points, whatever word was held');
  // While the run plays: the held word, without numbers, and this answer's own word for the step's tooltip.
  assert.deepEqual(startChange(116, 102, heldAtPause), { key: 'deepLikeLess', text: '▼ less', like: true, held: true, own: 'same' });
  assert.deepEqual(startChange(116, 102, { ...kept, points: -5.4 }), { key: 'deepLikeLess', text: '▼ less', like: true, held: true, own: 'fewer' });
  // An answer kept from before the numbers: the words as before, a held one with its own word for the tooltip.
  assert.deepEqual(startChange(116, 102, { points: 2.4, band: 5, direction: 'more', held: true }), { key: 'deepUpFromStart', text: 'more than at the start', like: true, own: 'same' });
  assert.deepEqual(startChange(116, 102, { points: 2.4, band: 5, direction: 'more', held: true }, { rest: true }), { key: 'deepSameAsStart', text: 'same as at the start', like: true });
  // The held word's tooltip, exactly, with each of the answer's own words: the words the page picks for each (LIKE_NOW).
  assert.equal(DEEP_TEMPLATES.deepLikeHeld, "This word is held from earlier answers so it doesn't flicker; this answer alone reads {now}, like for like.");
  assert.deepEqual(LIKE_NOW, { same: 'deepLikeSame', more: 'deepUpFromStart', fewer: 'deepDownFromStart' });
  assert.deepEqual(['same', 'more', 'fewer'].map(w => deep('deepLikeHeld', { now: DEEP_TEXT[LIKE_NOW[w]] })), [
    "This word is held from earlier answers so it doesn't flicker; this answer alone reads about the same, like for like.",
    "This word is held from earlier answers so it doesn't flicker; this answer alone reads more than at the start, like for like.",
    "This word is held from earlier answers so it doesn't flicker; this answer alone reads less than at the start, like for like."]);
  // The word at rest is the band word by the worker's own rule on the points it sends (analysis/worker.py start_word with
  // no word kept: "more" at the band or past it, "fewer" at minus the band or below, else "same"), which the engine also
  // says at rest (it keeps no word there): one word in both halves.
  for (const [points, key] of [[5, 'deepLikeMore'], [-5, 'deepLikeLess'], [4.99, 'deepLikeSame'], [-4.99, 'deepLikeSame'], [0, 'deepLikeSame']]) {
    assert.equal(startChange(116, 102, { points, band: 5, direction: 'same', now: 30 + points, start: 30 }, { rest: true }).key, key, `points ${points}`);
  }
});

test('the bar\'s labels: whole numbers that add up to 100, loose RNA "< 1 %" below 1 %', () => {
  const a = barLabels(116, [0.539, 0.46, 0.001]);
  assert.deepEqual([a.own, a.nb, a.loose, a.looseUnder], [54, 46, 0, true]);
  assert.deepEqual(a.texts, { own: 'Own 54 %', nb: 'Neighbours 46 %', loose: 'Loose < 1 %' });
  const b = barLabels(50, [0.78, 0.2, 0.02]);
  assert.deepEqual([b.own, b.nb, b.loose], [78, 20, 2]); assert.equal(b.texts.loose, 'Loose 2 %');
  const edge = barLabels(250, [0, 0.99, 0.02]);
  assert.equal(edge.own + edge.nb + edge.loose, 100, 'never more than 100 between them');
  assert.ok(Math.abs(a.widths.reduce((s, w) => s + w, 0) - 1) < 1e-12);
});

test('chips round down, rows say their share in a few characters, whole numbers in tooltips add up (largest remainder)', () => {
  assert.equal(chipText('GXPN', 31.1, 39), 'GXPN 31 of 39'); assert.equal(chipText('GYCNNN', 2.5, 3), 'GYCNNN 2 of 3');
  assert.equal(chipText('GYPNNN', 1.4, 2), 'GYPNNN 1 of 2'); assert.equal(chipText('X', 2.9999999, 3), 'X 3 of 3', 'a float a hair under a whole number');
  assert.deepEqual(largestRemainder([7.9, 31.1, 0], 39), [8, 31, 0]); assert.deepEqual(largestRemainder([1.5, 1.5, 0], 3), [2, 1, 0]);
  assert.equal(largestRemainder([0.4, 0.4, 0.2], 1).reduce((a, b) => a + b, 0), 1); assert.deepEqual(largestRemainder([0, 0], 0), [0, 0]);
  assert.equal(rowTooltip('GYL', 11, 5.1, 5.9), 'GYL: 11 transcripts · about 5 its own, 6 from neighbours (estimate)');
  assert.equal(rowTooltip('GXPN', 39, 7.9, 31.1, 'ffbbbbbb-1'), 'GXPN: 39 transcripts · about 8 its own, 31 from neighbours (most from ffbbbbbb-1) (estimate)');
  assert.equal(sharePct(0.46), '46 %'); assert.equal(sharePct(0.004), '< 1 %');
  // Parts of the headline are rounded DOWN and never above it ("about 34 %" once stood under 33 %).
  assert.equal(sharePct(0.339), '33 %'); assert.equal(sharePct(0.339, 33), '33 %');
  assert.equal(sharePct(0.3405, 33), '33 %', 'a row rounded up to 0.1 by the worker never passes the headline');
  assert.deepEqual([partPercent(0.459999), partPercent(0.46), partPercent(0.99, 46), partPercent(Number.NaN)], [45, 46, 46, 0]);
  for (let code = 0; code < 250; code++) {
    const head = Number(headlinePercent(code)), share = (code + 0.999) / 250;   // the most a single source can hold
    assert.ok(partPercent(share, head) <= head, `${code}`);
  }
  assert.equal(foldText(9, 0.004), '+9 more under 1 %'); assert.equal(foldText(6, 0.02), '+6 more · about 2 %');
  assert.equal(foldText(1, 0.03), '+1 more · about 3 %'); assert.equal(foldText(1, 0.001), '+1 more under 1 %');
  assert.equal(sameKindCountText(6, 10), '6 of its 10 nearest neighbours look the same kind.');
  assert.equal(sameKindCountText(1, 10), '1 of its 10 nearest neighbours looks the same kind.'); assert.equal(sameKindCountText(0, 10), null);
  assert.equal(partlyText(2), '2 of these neighbours are new since the model learned this cell, so their shares are a rougher guide.');
  assert.equal(partlyText(0), null);
});

test('"How is this worked out?": four short points and the provenance line, the method named once, no notebook', () => {
  const model = { cells: 40000, created: '2026-10-21T10:00:00Z', seconds: 420 };
  const items = howItems(model);
  assert.equal(items.length, 5);
  assert.deepEqual(items.slice(0, 4), [DEEP_TEXT.deepHow1, DEEP_TEXT.deepHow2, DEEP_TEXT.deepHow3, DEEP_TEXT.deepHow4]);
  assert.equal(items[4], 'Learned from 40,000 cells on 21 Oct 2026, in 7 min.', 'keeps ui.qc_provenance_shown\'s numbers on screen');
  assert.match(DEEP_TEXT.deepHow1, /^A ResolVI model splits each cell's RNA into three parts/);
  assert.match(DEEP_TEXT.deepHow2, /a gene only a neighbour's kind makes goes mostly to that neighbour/);
  assert.match(DEEP_TEXT.deepHow3, /adjusted for distance \(a rougher guide for cells the model didn't learn\)/);
  assert.match(DEEP_TEXT.deepHow4, /The glow places this split by position.*A rough guide\.$/);
  for (const item of items) assert.ok(words(item) <= 30, item);
  assert.ok(items.reduce((s, t) => s + words(t), 0) <= 110, 'four short points and the line');
  // A model that does not say when or from how many cells: the four points alone.
  assert.equal(howItems(null).length, 4); assert.equal(howItems({ cells: 5, created: 'not a date' }).length, 4);
  assert.equal(learnedLine({ cells: 1200, created: '2026-10-01T08:00:00Z' }), 'Learned from 1,200 cells on 1 Oct 2026.');
  assert.equal(learnedLine({ cells: 0, created: '2026-10-01T08:00:00Z', seconds: 3 }), null);
  assert.deepEqual([dateText(null), dateText(''), dateText('x')], [null, null, null]);
  for (const line of items) assert.deepEqual(bannedWords(line), [], line);
});

test('the frame tags, the run that cannot be checked and the failed details, in plain words', () => {
  assert.equal(frameTag({ kind: 'live', iter: 1201 }), 'Iteration 1,201 · live'); assert.equal(frameTag({ kind: 'paused', iter: 7 }), 'Paused · iteration 7');
  assert.equal(frameTag({ kind: 'older', iter: 7 }), 'Iteration 7 (not the newest)'); assert.equal(frameTag({ kind: 'finished', iter: 1201 }), 'Finished · iteration 1,201');
  assert.equal(frameTag({ kind: 'stopped', iter: 9 }), 'Stopped · iteration 9'); assert.equal(frameTag({ kind: 'start', startSource: 'xenium' }), "At the start (Xenium's cells)");
  assert.equal(frameTag({ kind: 'earlier', iter: 501, asked: 901 }), 'From iteration 501 · working out iteration 901…');
  assert.equal(frameTag({ kind: 'snapped', iter: 1201 }), "At iteration 1,201: the nearest with proseg's count");
  assert.equal(cantRunText(QC_TEXT.unavailableTable), "Go deeper can't run for this run: the mixing check needs a table of cells with transcript counts in this sample (for example xenium_table).");
  // What this computer lacks for the check reads as one sentence, with one colon, inside Go deeper's line.
  const lines = [
    [QC_TEXT.unavailableSoftware, "Go deeper can't run for this run: the mixing check needs scvi-tools and scviva-tools, in this Python environment or in one named as python under [qc] in the app's config.toml."],
    [QC_TEXT.unavailableAnalysis, "Go deeper can't run for this run: the mixing check needs Clusters and the UMAP, which can't run with the python under [analysis] in the app's config.toml."],
    [QC_TEXT.unavailableGpu, "Go deeper can't run for this run: the mixing check learns on an NVIDIA graphics card or a Mac's own GPU, and this computer has none it can use."],
    [QC_TEXT.unavailableAppleGpu, "Go deeper can't run for this run: the mixing check learns on this Mac's GPU, which failed a short test with this computer's PyTorch (the Log says why; on macOS 14, installing the app again brings PyTorch 2.14, which passes it)."],
    [QC_TEXT.unavailableTorchCpu, "Go deeper can't run for this run: the mixing check learns on an NVIDIA graphics card, which this computer's PyTorch can't use until its CUDA build is installed."],
    [QC_TEXT.unavailableDriver, "Go deeper can't run for this run: the mixing check learns on an NVIDIA graphics card, whose driver is too old for this computer's PyTorch and needs updating."],
  ];
  for (const [reason, line] of lines) {
    assert.equal(cantRunText(reason), line);
    assert.equal(line.split(':').length, 2, line);
    assert.deepEqual(bannedWords(line), [], line);
  }
  assert.equal(detailFailedText('detail_mismatch'), "Couldn't add the details: the new learning didn't match the old one exactly, so nothing was changed.");
  assert.equal(detailFailedText('oom', QC_TEXT.failureMemory), "Couldn't add the details: the graphics card ran out of memory.");
});

// ---- Go deeper's line: where the glowing transcripts sat at the start (Phase 1b), its tooltip and the start's hold ------
// The new or rewritten strings, the hold's own last.
const LINE_KEYS = ['deepStartLikely', 'deepStartMaybe', 'deepStartOwn', 'deepStartNone', 'deepStartGiver', 'deepStartOther', 'deepStartUnknown',
  'deepStartNear'];
const HOLD_KEYS = ['deepStartHold', 'deepStartHoldSome', 'deepStartHoldCells'];
const textOf = key => DEEP_TEXT[key] ?? DEEP_TEMPLATES[key];

test('the line: how many likely, then where they sat at the start, a part per place in a fixed order, in the engine\'s own words; places only', () => {
  assert.deepEqual(START_PARTS.map(([place]) => place), [...PLACES], 'own, none, giver, other, unknown: the classes\' own order');
  const all = { own: 8, none: 1, giver: 1, other: 1, unknown: 1 };
  assert.equal(startLine({ count: 15, likely: 12, places: all }, DEEP_TEXT.deepGlowAll),
    "12 likely from neighbours · 8 inside its own starting outline · 1 where no cell started · 1 inside a neighbour's starting outline · "
    + "1 inside another cell's starting outline · 1 unknown at the start");
  // A place no transcript sat in is left out; the order never changes.
  assert.equal(startLine({ count: 12, likely: 12, places: { own: 0, none: 3, giver: 0, other: 9, unknown: 0 } }, DEEP_TEXT.deepGlowAll),
    "12 likely from neighbours · 3 where no cell started · 9 inside another cell's starting outline");
  // Before the start line's reply is in (or with none: a run from nuclei only): the head alone.
  assert.equal(startLine({ count: 12, likely: 12, places: null }, DEEP_TEXT.deepGlowAll), '12 likely from neighbours');
  // A row's Show: that neighbour's; none likely: "maybe", every glowing one counted; nothing glowing: the line says so.
  assert.equal(startLine({ count: 3, likely: 3, places: { own: 2, none: 1, giver: 0, other: 0, unknown: 0 } }, 'ggaaaaaa-1'),
    '3 likely from ggaaaaaa-1 · 2 inside its own starting outline · 1 where no cell started');
  assert.equal(startLine({ count: 4, likely: 0, places: { own: 3, none: 0, giver: 1, other: 0, unknown: 0 } }, DEEP_TEXT.deepGlowAll),
    "4 maybe from neighbours · 3 inside its own starting outline · 1 inside a neighbour's starting outline");
  assert.equal(startLine({ count: 0, likely: 0, places: null }, 'ffbbbbbb-1'), 'None looks likely from ffbbbbbb-1');
  assert.equal(startLine(null, DEEP_TEXT.deepGlowAll), 'None looks likely from neighbours');
  // Whole numbers as the window writes them (en-GB thousands).
  assert.equal(startLine({ count: 2000, likely: 1234, places: { own: 1234, none: 0, giver: 0, other: 0, unknown: 0 } }, DEEP_TEXT.deepGlowAll),
    '1,234 likely from neighbours · 1,234 inside its own starting outline');
  // The places say what the test's reasons say of the same classes (engine/qc_pool.py REASON_WORDS, through the
  // engine's own fixture): where no cell started, inside a neighbour's, inside another cell's starting outline.
  const reasons = Object.fromEntries(JSON.parse(readFileSync(new URL('./fixtures/qc-test-reasons.json', import.meta.url), 'utf8')).map(e => [e.code, e.words]));
  for (const [key, code] of [['deepStartNone', 'pattern_none'], ['deepStartGiver', 'pattern_neighbour'], ['deepStartOther', 'pattern_other']]) {
    const words = DEEP_TEMPLATES[key].replace('{n} ', '');
    assert.ok(reasons[code].endsWith(`sit ${words}`), `${key}: "${words}" in "${reasons[code]}"`);
  }
  // Facts about places, never an outcome or advice.
  for (const key of LINE_KEYS) assert.doesNotMatch(textOf(key), /\b(fewer|clearly|less|more mixed|better|worse|wrong|should|kept|try|test)\b/i, key);
});

test('the line\'s tooltip: the glow\'s own words, how many sit at the shared edge, and the start\'s hold of those inside its own outline, by level', () => {
  const places = { own: 9, none: 3, giver: 0, other: 0, unknown: 0 };
  assert.equal(startTip({ count: 12, likely: 12, near: 0, places: null, levels: null }), DEEP_TEXT.deepGlowTooltip);
  assert.equal(startTip({ count: 12, likely: 12, near: 7, places, levels: null }),
    `${DEEP_TEXT.deepGlowTooltip} At the shared edge, within 2 µm of their neighbour: 7 of the likely ones.`);
  // The hold split, for a detail run (the start's own layers): of those inside its own starting outline, by level.
  const split = startTip({ count: 12, likely: 12, near: 0, places, levels: { nucleus: 5, cell: 4, unknown: 0 } });
  assert.equal(split, `${DEEP_TEXT.deepGlowTooltip} Inside its own starting outline, the start held 5 at the nucleus level and 4 at the cell level. `
    + 'A start from cells only holds the whole starting cell at the nucleus level.');
  assert.match(startTip({ count: 12, likely: 12, near: 0, places, levels: { nucleus: 5, cell: 3, unknown: 1 } }),
    /the start held 5 at the nucleus level and 3 at the cell level; for 1 it could not be told\./);
  // Nothing to split: none inside its own outline, no levels (a whole-sample run), or none whose level could be told.
  for (const [p, levels] of [[{ ...places, own: 0 }, { nucleus: 0, cell: 0, unknown: 0 }], [places, null], [places, { nucleus: 0, cell: 0, unknown: 9 }]])
    assert.doesNotMatch(startTip({ count: 12, likely: 12, near: 0, places: p, levels }), /nucleus level/);
  // "Maybe": the shared edge is said of likely ones only.
  assert.equal(startTip({ count: 4, likely: 0, near: 2, places: null, levels: null }), DEEP_TEXT.deepGlowTooltip);
  // Where the stage draws the cell's starting outline, the dashed line is named right after what the glow is.
  assert.equal(startTip({ count: 12, likely: 12, near: 7, places, levels: null }, true),
    `${DEEP_TEXT.deepGlowTooltip} ${DEEP_TEXT.deepGlowStartOutline} At the shared edge, within 2 µm of their neighbour: 7 of the likely ones.`);
  assert.equal(noOutlineLine({ count: 3, likely: 2, near: 0 }, { name: 'cell-b-1', kind: [] }, true).tip,
    `None of these pink dots lies near cell-b-1: they more likely come from a cell of its kind that has no outline here. ${glowTip(true)}`);
});

test('the line\'s words and the hold\'s pass the banned-word rule, hold no clash word, say "hold" only in the hold\'s own strings, never "compactness"', () => {
  for (const key of [...LINE_KEYS, ...HOLD_KEYS]) {
    const text = textOf(key);
    assert.ok(text, key);
    assert.deepEqual(bannedWords(text), [], key);
    assert.doesNotMatch(text, /look-alike|similar|same kind|like for like/i, `${key}: no clash word`);
    assert.doesNotMatch(text, /compactness/i, key);
    if (!HOLD_KEYS.includes(key)) assert.doesNotMatch(text, /\b(hold|held)/i, `${key}: "hold" only in the hold's own strings`);
  }
  for (const key of HOLD_KEYS) assert.match(textOf(key), /\b(hold|held)/i, key);
  // Expanded as the window shows them.
  const glow = { count: 14, likely: 12, near: 7, places: { own: 9, none: 1, giver: 1, other: 0, unknown: 1 }, levels: { nucleus: 5, cell: 3, unknown: 1 } };
  for (const text of [startLine(glow, DEEP_TEXT.deepGlowAll), startTip(glow), startTip(glow, true)]) {
    assert.deepEqual(bannedWords(text), [], text);
    assert.ok(allTexts().includes(text), `${text}: listed with every string the window can show`);
  }
  // "Test a setting here…" beside Show on the image: the check's own words and tooltip, listed too.
  assert.equal(QC_TEXT.testOpen, 'Test a setting here…');
  for (const text of [QC_TEXT.testOpen, QC_TEXT.testOpenTooltip]) assert.ok(allTexts().includes(text), text);
});

// ---- the round of 2026-10-31: the count proseg saves, the pink dots in 3-D, proseg's count in "Where they sit" ----------
// The new and changed strings, exactly as the round's words table has them.
const ROUND = {
  deepBasisShort: "Proseg's count (averaged): {n}",
  deepBasisShortSaved: "Proseg's count (as saved): {n}",
  deepBasisSaved: 'The transcripts proseg places in this cell, less {bg} it reads as background, as proseg saves them: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.',
  deepBasisNoBgSaved: 'The transcripts proseg places in this cell, less those it reads as background, as proseg saves them: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.',
  deepStartSaved: "Since the start: both sides read proseg's count averaged over its random background draw, like for like, not the count as saved.",
  deepWhere3d: "The genes' coloured dots show in the 2-D view.",
  deepWhereSaved: 'Read on the count proseg saves at this iteration: one random background draw, not averaged.',
  deepWhereSnapped: 'Dots at iteration {i}',
  deepWhereSnappedTooltip: "Proseg's exact shares at iteration {i}, the iteration it counted before this one: here this iteration's shares would only be approximate.",
  deepWhereDensity: "Per 100 µm², in proseg's count: {here} in this cell · {giver} in it · {around} in the cells around.",
  deepWhereDensityAround: "Per 100 µm², in proseg's count: {here} in this cell · {around} in the cells around.",
  deepWhereLayerTooltip: "In proseg's count it is about as common in the cells around as in this cell: it looks like the tissue's own RNA at this spot, which no outline removes.",
  deepWhereIntoLine: "Inside the neighbour's outline, proseg counts for this cell: {genes}",
  deepWhereInto: '{gene}: {n}, {near} of them within 2 µm of this cell.',
  deepWhereIntoBase: "Other transcripts inside the neighbour's outline: proseg counts {of} of them for this cell, {n} within 2 µm of this cell.",
  deepWhereIntoTooltip: "Genes the model reads as this cell's own that lie inside this neighbour's outline, and how many of them proseg counts for this cell. Many counted here, close to this cell, mean the drawn outline reaches over RNA proseg gives this cell.",
  deepNuclei: "At the first iteration proseg's cells are those {start} drew, and its count already sets part of their RNA aside as background.",
  deepNucleiShort: 'Cells as {start} drew them',
  deepNotRemeasured: "Not yet re-measured on proseg's count for runs like this one.",
};

test('the round\'s words: each key exactly as written, plain words, a template where it takes a value', () => {
  for (const [key, text] of Object.entries(ROUND)) {
    assert.equal(textOf(key), text, key);
    assert.equal(Object.hasOwn(DEEP_TEMPLATES, key), /\{[^}]+\}/.test(text), `${key}: a template only with a value to fill`);
    assert.deepEqual(bannedWords(text), [], key);
    assert.doesNotMatch(text, /\bexpected\b/i, `${key}: never "expected"`);
  }
  // The averaged count's words stay as they were (the saved ones are new beside them).
  assert.equal(DEEP_TEXT.deepBasisNoBg, 'The transcripts proseg places in this cell, less those it reads as background, averaged over its random background draw. The saved table holds one draw of this count.');
  assert.equal(DEEP_TEMPLATES.deepBasis, 'The transcripts proseg places in this cell, less {bg} it reads as background, averaged over its random background draw. The saved table holds one draw of this count.');
  // The saved count's words hold in every state of the run (no "if the run finishes here"): a frame scrubbed back to
  // after the run went on still is the count proseg saves at that iteration.
  for (const key of ['deepBasisSaved', 'deepBasisNoBgSaved']) assert.doesNotMatch(textOf(key), /\bif\b|would/, key);
  // "Inside the neighbour's outline" is where they lie; what is counted is proseg's (never the outline's count of the cell).
  assert.match(ROUND.deepWhereIntoLine, /proseg counts for this cell/); assert.match(ROUND.deepWhereIntoTooltip, /how many of them proseg counts for this cell/);
  assert.doesNotMatch(ROUND.deepWhereIntoTooltip, /could mean|positions/);
  // The cells around are counted by proseg, per their own footprints: no "tissue around" left in the density words.
  for (const key of ['deepWhereDensity', 'deepWhereDensityAround', 'deepWhereLayerTooltip']) assert.match(textOf(key), /in the cells around/, key);
  // Every new string is among those the window can show (the banned-word walk covers them).
  const all = allTexts();
  for (const key of ['deepBasisNoBgSaved', 'deepStartSaved', 'deepWhereSaved', 'deepNotRemeasured', 'deepWhere3d']) assert.ok(all.includes(DEEP_TEXT[key]), key);
  assert.ok(all.includes(deep('deepWhereSnapped', { i: '1,201' })) && all.includes(deep('deepBasisShortSaved', { n: '682' })));
});

test('a count said beside its floor reads under it: one decimal, rounded down, only where it would round up to the floor', () => {
  // "At least 100 are needed": 99.71875 (a cell's averaged count, in 32nds) is 99.7, never 100; 3.6875 of "fewer than 4"
  // is 3.6. Everything else is rounded once, as every count (count): 99.4 is 99, 100.4 is 100, 2.469 is 2.
  assert.deepEqual([99.71875, 99.5, 99.96875, 3.6875, 3.5].map(n => underFloor(n, n > 10 ? 100 : 4)), ['99.7', '99.5', '99.9', '3.6', '3.5']);
  assert.deepEqual([99.4, 100, 100.4, 61, 3.4, 4, 0, 2.469].map(n => underFloor(n, n > 10 ? 100 : 4)), ['99', '100', '100', '61', '3', '4', '0', '2']);
  for (const n of [99.4, 100.4, 61, 3.4, 2.469]) assert.equal(underFloor(n, 100), count(n), String(n));
  // Never at or over the floor's own number when under it, whatever the 32nds.
  for (let k = 0; k < 100 * 32; k++) {
    const n = k / 32, said = Number(underFloor(n, 100).replace(/,/g, ''));
    assert.ok(said < 100, `${n} reads ${said}`); assert.ok(Math.abs(said - n) < 0.6, `${n} reads ${said}`);
  }
  assert.equal(underFloor(Number.NaN, 100), count(Number.NaN));
});
