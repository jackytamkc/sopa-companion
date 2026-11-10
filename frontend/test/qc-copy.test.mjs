import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { BANNED_WORDS, bannedWords, QC_TEXT, QC_TEMPLATES, RULE_COPY, WAITING_REASONS, FAILURE_REASONS, UNAVAILABLE_REASONS,
  copy, startName, number, duration, settingValue, list, changeLabel, knobChange, reasonText, cellShare, cellLine, cardLine,
  knobName, knobHint, settingLine, settingShort, settingChecked, knobLine, sameSetting, cardKey, tryingLabel, triedLabel, KNOB_NAMES,
  likeForLike, verdictLine, heldClause, countWord, hoverKey, coverageOf, retentionOf, retentionLine, retentionMoves, retentionTooltip, retentionSentence, learnedLine, HOW_KEYS,
  HEADLINE_MIN_JUDGED, LOW_COVERAGE, MODEL_THRESHOLD, RETENTION_LOW, RETENTION_HIGH, ROUGH_JUDGED, tableName, TEST_REASON_WORDS,
  TEST_REASON_KINDS, TEST_RUN_REFUSALS, testReasonText, PROSEG_FIXES, oldProsegTooltip, unavailableTooltip } from '../src/qc-copy.mjs';

// Every key of the deck: its strings, and its templates as functions of their values (the deck's own copy()).
const QC_COPY = Object.freeze({ ...QC_TEXT, ...Object.fromEntries(Object.keys(QC_TEMPLATES).map(key => [key, values => copy(key, values)])) });

test('every QC string passes the banned-word rule, including expanded templates', () => {
  assert.ok(Object.keys(QC_TEXT).length >= 150);
  assert.ok(Object.keys(QC_TEMPLATES).length >= 80);
  for (const [key, text] of Object.entries({ ...QC_TEXT, ...QC_TEMPLATES })) {
    // The spec explicitly permits the exact Live knobs label only in Setting:.
    const checked = key === 'settingHold' ? text.replace('Setting: Prior weight', 'Setting:') : text;
    assert.ok(!/\b(band|depth-matched|compactness)\b/i.test(text) || /^Setting: |^Cell compactness$/.test(text) || key === 'settingCompactness', `${key}: plain words`);
    assert.deepEqual(bannedWords(checked), [], key);
    if (/resolvi/i.test(text)) assert.equal(key, 'how1', `${key}: the method is named in "How is this estimated?" only`);
    // Never the user's own files or notebooks, in any string of the deck, shown or
    // not; nor a sample's code, a tissue or a cell type (the cards' tests are said in words).
    assert.doesNotMatch(text, /notebook|part\s*3|your (own )?files?\b/i, key);
    assert.doesNotMatch(text, /\b[A-Z]{2}\d{2}\b|liver|hepat/i, `${key}: no sample code, tissue or cell type`);
  }
  for (const [key, pattern] of Object.entries(QC_TEMPLATES)) {
    const values = Object.fromEntries([...pattern.matchAll(/\{([^}]+)\}/g)].map(m => [m[1], '7']));
    const expanded = copy(key, values);
    assert.deepEqual(bannedWords(key === 'settingHold' ? expanded.replace('Prior weight', '') : expanded), [], key);
  }
});

test('every template fills every named placeholder, with no silent missing values', () => {
  assert.deepEqual(Object.keys(QC_TEXT).filter(k => k in QC_TEMPLATES), []);
  for (const [key, pattern] of Object.entries(QC_TEMPLATES)) {
    const names = [...new Set([...pattern.matchAll(/\{([^}]+)\}/g)].map(m => m[1]))];
    const values = Object.fromEntries(names.map((n, i) => [n, `VALUE_${i}`]));
    const text = QC_COPY[key](values);
    assert.equal(typeof text, 'string', key); assert.doesNotMatch(text, /\{[^}]+\}|undefined|null/, key);
    for (const value of Object.values(values)) assert.ok(text.includes(value), `${key}: ${value}`);
    for (const name of names) {
      const missing = { ...values }; delete missing[name];
      assert.throws(() => copy(key, missing), TypeError, `${key}.${name}`);
    }
  }
  assert.throws(() => copy('not_a_copy_key'), RangeError);
  assert.equal(copy('title'), 'Mixing check');
});

test('the ban includes every term in the spec, stems, and case variants', () => {
  // The spec's list ("Banned in every QC string" in the mixing check's design note, which stays with the development
  // copy): kept here so that the test runs wherever the app's tests do, and checked against the note where it is.
  const expected = ['diffusion', 'posterior', 'ELBO', 'latent', 'amortis', 'amortiz', 'Procrustes', 'Dirichlet', 'encoder',
    'mixture', 'kNN', 'Leiden', 'likelihood', 'contamination', 'softmax', 'epoch', 'embedding', 'inference', 'Spearman',
    'Jaccard', 'Wilson', 'hysteresis', 'centroid', 'quantile', 'calibrat', 'AUROC', 'voxel', 'prior'];
  const note = new URL('../../docs/research/RESOLVI_QC.md', import.meta.url);
  if (existsSync(note)) {
    const line = readFileSync(note, 'utf8').match(/\*\*Banned in every QC string\*\*[^:]*: (.*)\./)[1];
    assert.deepEqual(line.split(', ').flatMap(word => word.split('/')), expected, 'the note and this list agree');
  }
  assert.deepEqual(BANNED_WORDS, expected);
  for (const term of BANNED_WORDS) assert.ok(bannedWords(term.toUpperCase()).includes(term));
  assert.deepEqual(bannedWords('Calibrated AMORTISED amortization'), ['amortis', 'amortiz', 'calibrat']);
  assert.deepEqual(bannedWords('Cells are still forming. No clear change yet.'), []);
});

test('the full deck covers states, cell cards, trials, rules and explanation', () => {
  const required = ['off', 'waiting', 'cardShared', 'waitingStart', 'waitingFrame', 'unavailableTable', 'unavailableCounts', 'unavailableSoftware',
    'softwareChanged', 'failedWithAction', 'workerRestarting', 'pickerReady', 'pickerLearning', 'pickerWaiting',
    'numberTooltip', 'startTooltip', 'earlierTooltip', 'basis', 'forming', 'tooFew', 'lowDepth', 'nucleiStart', 'notNewest',
    'chartTitle', 'chartHover', 'chartAfterSuffix', 'chartExpectedTooltip', 'timelineLegend', 'timelineHover',
    'legendTitle', 'legendLittle', 'legendSome', 'legendMixed', 'legendUnjudged', 'legendNoValues', 'overviewNote',
    'overviewTitle', 'overviewLow', 'overviewMid', 'overviewHigh',
    'cellMixed', 'cellSome', 'cellLittle', 'cellUnjudged', 'cellCard', 'chipMore', 'chipFewer', 'chipSame', 'chipLearning', 'chipWaiting',
    'showStartOutlines', 'showStartTooltip', 'standsOut', 'inTissue', 'setting',
    'likely', 'might', 'rough', 'seen', 'suggestionFooter', 'tooEarly', 'timingEarly', 'timingLater', 'shapeCost',
    'tryTooltip', 'tryPaused', 'moreIdeas', 'trialTitle', 'trialWatching', 'trialArea', 'trialWhole', 'trialDepth', 'trialLess', 'trialMore',
    'trialLessArea', 'trialMoreArea',
    'trialNoChange', 'trialNoChangeAfter', 'trialForming', 'trialSmall', 'trialDepthNote', 'trialMixedUp', 'trialUndone', 'trialKept',
    'newRunTooltip', 'finished', 'finishedNoStart', 'stoppedNoStart',
    'resultFact', 'howTitle', ...HOW_KEYS, 'howLearned', 'howLearnedPasses',
    // Proseg's own count: which count the headline reads, a run without it, the start with nuclei, the snapped result.
    'basisTooltip', 'notRemeasured', 'retentionBasis', 'nucleiCellsStart', 'unavailableNoCounts', 'unavailableOldProseg',
    'unavailableOldProsegTooltip', 'fixBuild', 'fixInstall', 'fixSetting', 'snappedBadge', 'snappedTooltipMixing', 'snappedTooltipClusters',
    // The summary's verdict line, the start alone, the retention line and its tooltips.
    'verdictSame', 'verdictMore', 'verdictFewer', 'verdictMoreLight', 'verdictFewerLight', 'startShort', 'retentionShort', 'retentionUpTooltip',
    'retentionDownTooltip', 'roughCells', 'earlierInfo',
    // ... and the test of a setting: the panel's line while a test is open, the form's lines, the progress, the answer.
    'testOpen', 'testOpenTooltip', 'testFarDrift', 'testFarDriftNote', 'testFarDriftTip', 'testPool', 'testPoolOne', 'testPoolTooltip', 'testPoolAsking',
    'testPoolFailed', 'testAreaRegion', 'testAreaSquare', 'testAreaCrop', 'testRuns', 'testIteration', 'testRetry', 'testStatusOpen', 'testStatusWaiting',
    'testStatusRunning', 'testStatusDone', 'testCancelWaitingTooltip', 'testReasonUnknown', 'testWaitUnknown', 'testCaveatUnknown',
    'testStatusStopped', 'testProgressTooltip', 'testBack', 'testFewerText', 'testCalibration', 'testRepeat', 'testRepeatNote', 'testNewRun', 'testKeep',
    'testRunAgain', 'testRunAgainTooltip'];
  for (const key of required) assert.ok(key in QC_COPY, key);
  for (const rule of Object.values(RULE_COPY)) {
    assert.ok(rule.title in QC_COPY); assert.ok(rule.body in QC_COPY); assert.ok(rule.confidence in QC_COPY);
    if (rule.why) assert.ok(rule.why in QC_COPY);
    if (rule.more) assert.ok(rule.more in QC_COPY);
  }
  assert.equal(QC_TEXT.ruleR4Title, 'Try tighter cell shapes');
  assert.equal(QC_TEXT.ruleR3Title, 'Turn transcript drift back on');
});

test('copy helpers keep frequencies, RNA shares and setting values readable', () => {
  assert.equal(startName('xenium'), "Xenium's cells"); assert.equal(startName('cells'), 'the starting cells');
  assert.equal(number(3188), '3,188'); assert.equal(number(10.6, 1), '10.6'); assert.equal(number(null), '—');
  assert.equal(duration(61), '2 min'); assert.equal(duration(30), '30 s');
  assert.equal(settingValue(.039999999), '0.04'); assert.equal(settingValue(null), 'automatic');
  assert.equal(settingValue(false), 'off'); assert.equal(list(['A', 'C']), 'A and C');
  assert.equal(changeLabel({ key: 'farDrift', value: .1 }), 'far drift 0.1');
  assert.equal(changeLabel({ rule: 'R5' }), 'firmer hold');
  assert.equal(changeLabel({ setting: 'posterior drift', flag: 'no-diffusion' }), '', 'a banned or unknown change has no label (no tick)');
});

test('every reason the engine gives has plain words; an unknown one gets its state\'s general words, never the engine\'s', () => {
  // The mixing check's states: waiting (queue, gpu_busy, gpu_memory, lock), failed (the trainer's reasons and the
  // engine's), unavailable (no_table, not_counts, software, and a run's no_transcripts).
  for (const reason of ['queue', 'gpu_busy', 'gpu_memory', 'lock']) assert.ok(WAITING_REASONS[reason], reason);
  for (const reason of ['gpu', 'too_few_cells', 'unreadable', 'oom', 'not_settled', 'timeout', 'unexpected']) assert.ok(FAILURE_REASONS[reason], reason);
  for (const reason of ['no_table', 'not_counts', 'software', 'analysis', 'no_transcripts', 'positions', 'no_gpu', 'apple_gpu',
    'torch_cpu', 'driver']) assert.ok(UNAVAILABLE_REASONS[reason], reason);
  // What the check is missing names its own setting (GET /health qc.reason): [qc] python, or Clusters and the UMAP's
  // [analysis]; or the graphics card it learns on, never naming the method.
  assert.match(reasonText('unavailable', 'software'), /python under \[qc\] in the app's config\.toml/);
  assert.match(reasonText('unavailable', 'analysis'), /python under \[analysis\] in the app's config\.toml/);
  for (const reason of ['software', 'analysis']) assert.deepEqual(bannedWords(reasonText('unavailable', reason)), [], reason);
  const missing = {
    software: "The mixing check needs scvi-tools and scviva-tools, in this Python environment or in one named as python under [qc] in the app's config.toml.",
    analysis: "The mixing check needs Clusters and the UMAP, which can't run with the python under [analysis] in the app's config.toml.",
    no_gpu: "The mixing check learns on an NVIDIA graphics card or a Mac's own GPU, and this computer has none it can use.",
    apple_gpu: "The mixing check learns on this Mac's GPU, which failed a short test with this computer's PyTorch (the Log says why; on macOS 14, installing the app again brings PyTorch 2.14, which passes it).",
    torch_cpu: "The mixing check learns on an NVIDIA graphics card, which this computer's PyTorch can't use until its CUDA build is installed.",
    driver: "The mixing check learns on an NVIDIA graphics card, whose driver is too old for this computer's PyTorch and needs updating.",
  };
  for (const [reason, text] of Object.entries(missing)) {
    assert.equal(reasonText('unavailable', reason), text, reason);
    assert.deepEqual(bannedWords(text), [], reason);
    assert.doesNotMatch(text, /resolvi|:/i, `${reason}: no method named, no colon of its own`);
  }
  assert.equal(QC_TEXT.unavailableGpu, missing.no_gpu); assert.equal(QC_TEXT.unavailableTorchCpu, missing.torch_cpu);
  assert.equal(QC_TEXT.unavailableAppleGpu, missing.apple_gpu);
  assert.equal(QC_TEXT.unavailableDriver, missing.driver);
  assert.doesNotMatch(reasonText('unavailable', 'analysis'), /\[qc\]/); assert.doesNotMatch(reasonText('unavailable', 'software'), /\[analysis\]/);
  assert.equal(reasonText('waiting', 'lock'), 'Waiting while another sample is being learned on this computer…');
  assert.equal(reasonText('waiting', 'queue'), 'Waiting for its turn to learn this sample…');
  assert.equal(reasonText('waiting', 'gpu_memory'), QC_TEXT.waiting);
  // Learning waits only for room on its graphics card (enough free memory, however busy: a design choice, 2026-10-21).
  assert.equal(QC_TEXT.waiting, 'Waiting for room on the graphics card…'); assert.deepEqual(bannedWords(QC_TEXT.waiting), []);
  // Too few cells: under 200, too few with enough transcripts, or a fit that saw no mixing anywhere (a small table).
  assert.equal(reasonText('failed', 'too_few_cells'), 'it has too few cells to learn from');
  // A table whose cells cannot be placed on the transcripts (the engine's "positions"): said, not "unavailable" alone.
  assert.equal(reasonText('unavailable', 'positions'), "This sample's cell table doesn't say where its cells lie on the transcripts, so the mixing check can't learn from it.");
  assert.deepEqual(bannedWords(QC_TEXT.unavailablePositions), []);
  assert.equal(reasonText('failed', 'gpu'), 'it could not run on the graphics card set aside for it');
  // A finished run without its transcripts' file: never said as though the check counted them itself (it reads
  // proseg's own count).
  assert.equal(reasonText('unavailable', 'no_transcripts'), "This finished run didn't keep the positions of its transcripts, which the mixing check needs.");
  assert.doesNotMatch(QC_TEXT.unavailableNoTranscripts, /\bcount/i); assert.deepEqual(bannedWords(QC_TEXT.unavailableNoTranscripts), []);
  assert.equal(reasonText('failed', 'some new reason'), QC_TEXT.failureUnexpected);
  assert.equal(reasonText('unavailable', undefined), QC_TEXT.unavailable);
  assert.equal(reasonText('waiting', 'x'), QC_TEXT.waiting);
  for (const map of [WAITING_REASONS, FAILURE_REASONS, UNAVAILABLE_REASONS]) for (const key of Object.values(map)) assert.ok(QC_TEXT[key], key);
  assert.equal(QC_TEXT.requested, 'This iteration will be checked as soon as the sample has been learned.');
  assert.equal(QC_TEXT.cardShared, 'The graphics card is shared with another job, so this may take a few minutes longer.');
});

test('build.copy_updates keep their facts, with their units said, and a relearn names what changed', () => {
  assert.equal(QC_TEXT.shapeCost, 'This changes the whole run. In tests, cells everywhere got 6–15 % smaller and kept 4–12 % fewer transcripts.');
  assert.equal(QC_TEXT[RULE_COPY.R4.confidence], 'Rough guess');
  assert.equal(QC_TEXT.timingEarly, 'Best tried now, while cells are still forming: in a test, a change at this stage gave about two thirds of the full effect.');
  // timing_later's facts (mostly stops growth; shows after ~1,000 iterations, {t} here; half the effect), shortened: the
  // card's New run with this setting says the rest.
  assert.equal(copy('timingLater', { t: '2.7 min' }), 'Cells have formed, so a change now mostly stops them growing; in a test it showed only after about 1,000 iterations (about 2.7 min here), at half the full effect.');
  // The cards' tests (their Why?) in words: what was seen in how many samples, never which sample nor its own levels;
  // an effect's size only where the card's confidence line rests on it (R2, R3) or it is a cost (R3, R5).
  assert.match(QC_TEXT.ruleR4Why, /^Tested on three samples\. Set from the start, tighter shapes clearly lowered mixed-looking cells in one, barely in another and not in the third\. Like for like, about half of that drop remains: .* A gentler change, to 0\.03, has not been tested\.$/);
  assert.equal(QC_TEXT.ruleR2Why, 'In a test, looser shapes (0.08) gave about 1 more mixed-looking cell in 100, and more neighbour-like RNA, in two runs.');
  // R3 measured on proseg's count (2026-10-31): where its size was measured said first (repeat runs of two regions of
  // one sample, started from cells only: the confidence line's "about 1 in 100" holds there, so the card says so),
  // about 1 more mixed-looking cell in 100 with drift off in every repeat run, the cost in those regions and in the
  // other test of that sample (the body's 7-17 %), and what a trial on a small region can show.
  assert.equal(QC_TEXT.ruleR3Why, "In repeat runs of two regions of one sample, started from cells only, switching drift off gave about 1 more mixed-looking cell in 100, and more neighbour-like RNA, in every run, and about 7 % fewer transcripts per cell (17 % in another test of that sample). Switched back on after cells had formed, it lowered mixing again, half to two thirds of that like for like. Trying it on a region of a few hundred cells reads no clear change: such a region can't show a change this small.");
  assert.equal(QC_TEXT.ruleR3Body, 'With drift off, cells kept 7-17 % fewer transcripts in tests.');
  assert.match(QC_TEXT.ruleR5Why, /^In a test, a weaker hold \(a quarter as strong\) made cells about 14 % smaller, so a firmer one may keep small cells from being squeezed\./);
  assert.match(QC_TEXT.ruleR5Why, /never lowered mixing: no change in one, a little more in the other two\. Treat this as a rough guess\.$/);
  // The first repeat-run strings and numbers went (shown nowhere, and stale against the engine's like-for-like spread);
  // "Unlikely to help" can no longer be shown either.
  for (const key of ['repeatRegion', 'repeatWhole', 'unlikely']) assert.ok(!(key in QC_TEXT), key);
  // The notebook's number and every sentence that named a notebook went with it.
  for (const key of ['how6', 'how7', 'how7Plain', 'how8', 'how1Plain', 'how1Unknown', 'howCompare', 'howDiff', 'diffSingleTable', 'diffMoreEpochs',
    'diffMoreEpochsPlain', 'diffOtherTable', 'diffOtherTablePlain', 'diffOther', 'notRecorded']) assert.ok(!(key in QC_COPY), `${key} is gone`);
  // The calibrations of the outline-era count went with it: the whole-sample start note, the saved results' "usually a
  // little cleaner", drift's "this view can't show it" (drift shows in proseg's count) and its "saved results" confidence.
  for (const key of ['startNote', 'finishedSaved', 'finishedSavedTooltip', 'trialR3', 'likelySaved']) assert.ok(!(key in QC_COPY), `${key} is gone`);
  assert.equal(copy('softwareChanged', { t: '2 min' }), 'Something this check depends on changed since this sample was learned, so it is learning again (about 2 min).');
});

test("the cards' tests and the test form's notes name no sample, tissue or cell type and give no tested sample's own levels", () => {
  // What a card's Why? and the form's notes say of the tests behind them, in words: "clearly lowered ... in one, barely
  // in another and not in the third", never a sample's code beside its levels (as in "XX00: 20.0 → 18.0"), a list of
  // each sample's change ("+0.9, +0.4") or two runs' own numbers ("0.1 and 0.2"); the one cell worked out and the test
  // from cells only are named by what they were, never by a tissue or a cell type.
  const whys = Object.values(RULE_COPY).map(rule => rule.why).filter(Boolean);
  assert.deepEqual(whys.sort(), ['ruleR2Why', 'ruleR3Why', 'ruleR4Why', 'ruleR5Why']);
  for (const key of [...whys, 'testFarDriftNote', 'testFarDriftTip']) {
    const text = QC_TEXT[key];
    assert.doesNotMatch(text, /\b[A-Z]{2}\d{2}\b|\bsample [A-Z]\b/, `${key}: no sample named`);
    assert.doesNotMatch(text, /\d\s*→\s*\d|[+−]\d|\b\d+\.\d+ and \d+\.\d+\b|\bwhole sample\b/, `${key}: no tested sample's own levels: ${text}`);
    assert.doesNotMatch(text, /liver|hepat/i, `${key}: no tissue or cell type`);
  }
  // The form's only notes are far drift's (nothing in it says what a setting may do: nothing is suggested); the notes
  // the form once offered with a value are gone.
  assert.equal(QC_TEXT.testFarDriftNote, 'Not tried here yet.');
  for (const key of ['testShapesNote', 'testShapesNoteCells', 'testHoldNote', 'testHoldLooserNote', 'testShapesLooserNote', 'testShapesLooserTip',
    'testHoldAimTip']) assert.ok(!(key in QC_COPY), `${key} is gone`);
  // The tested settings' own values stay (they are the app's, not a sample's): 0.08, 0.03, ×4, a quarter.
  assert.match(QC_TEXT.ruleR2Why, /looser shapes \(0\.08\)/); assert.match(QC_TEXT.ruleR5Why, /\(×4\)/);
});

test("the mixing check's sources and tests quote no one and keep no review notes or machine details, nor a sample's code in the page's code", () => {
  // Design decisions keep their reasons, said plainly; who asked is said at most as "on request", never in
  // their words, never a reviewer, a finding's number, this computer's layout or (in the page's code) a sample's code,
  // tissue or cell type. The patterns are built from parts, so this file never matches itself.
  const read = path => readFileSync(new URL(path, import.meta.url), 'utf8');
  const sources = ['qc-copy.mjs', 'qc-panel.js', 'qc-client.mjs', 'qc-deeper-copy.mjs', 'qc-test-corner-copy.mjs', 'qc-test-corner.mjs']
    .map(name => [`src/${name}`, read(`../src/${name}`), true]);
  const tests = ['qc-panel', 'qc-client', 'qc-deeper-copy', 'qc-deeper-stage', 'qc-deeper', 'qc-deeper-where-stage', 'qc-deeper-where', 'qc-test-corner', 'qc-test']
    .map(name => [`test/${name}.test.mjs`, read(`./${name}.test.mjs`), false]);
  const rules = [
    [new RegExp(['the owner: "', '\\(owner[:,)]', 'owner, 20\\d\\d', 'kinda', 'Too much text', 'trim the suggestion'].join('|')), 'a quote of the owner', false],
    [new RegExp(['Co' + 'dex', 'Cla' + 'ude', '\\breview' + 'er', 'the ' + 'review of', '\\(review' + ',', 'finding \\d', '\\bC[34] F\\d', 'UX-\\d'].join('|'), 'i'), 'review notes', false],
    [new RegExp(['No' + 'Machine', '\\bGPU [01]\\b', '/ho' + 'me/', '/m' + 'nt/', 'fuse' + 'blk', '\\d{2,3}-core'].join('|'), 'i'), 'a machine detail', false],
    [new RegExp(['\\bD' + 'U\\d{2}\\b', 'li' + 'ver', 'hepat' + 'ocyte'].join('|'), 'i'), "a sample's code, tissue or cell type", true],
  ];
  for (const [name, text, page] of [...sources, ...tests]) {
    const lines = text.split('\n');
    for (const [pattern, what, pageOnly] of rules) {
      if (pageOnly && !page) continue;
      const at = lines.findIndex(line => pattern.test(line));
      assert.equal(at, -1, `${name}:${at + 1}: ${what}: ${lines[at]?.trim().slice(0, 120)}`);
    }
  }
});

test('knob changes have plain names (the chart and the timeline); cells say how mixed they look as the colours do', () => {
  const label = (...a) => changeLabel(knobChange(...a) ?? {});
  assert.equal(label('cell-compactness', 0.03999999910593033, 0.02), 'tighter shapes');
  assert.equal(label('cell-compactness', 0.04, 0.08), 'looser shapes');
  assert.equal(label('cell-compactness', 0.02, 0.03999999910593033), 'standard shapes');
  // The hold by what it does whatever the run started from (Cellpose's cells too), never "closer to Xenium".
  assert.equal(label('prior-weight', 1, 4), 'firmer hold'); assert.equal(label('prior-weight', 4, 1), 'looser hold');
  assert.deepEqual(knobChange('cell-compactness', 0.04, 0.02), { key: 'tighterShapes', value: 0.02, from: 0.04 }, 'with the value before (the hover)');
  assert.equal(label('no-diffusion', false, true), 'drift off'); assert.equal(label('no-diffusion', true, false), 'drift on');
  assert.equal(label('diffusion-probability', .2, .1), 'far drift 0.1'); assert.equal(label('diffusion-sigma-far', 4, 6), 'drift distance');
  assert.equal(label('diffusion-proposal-sigma', 4, 2), 'drift search step'); assert.equal(label('enforce-connectivity', false, true), 'one-piece cells on');
  assert.equal(label('temperature', 1, .5), 'cooler'); assert.equal(label('temperature', .5, 1), 'warmer');
  assert.equal(label('morphology-steps-per-iter', 4000, 8000), 'more boundary attempts'); assert.equal(label('ab-nihlo-bubble-prob', .05, .1), 'gap proposals 0.1');
  assert.equal(label('dispersion', null, 2), 'expression variability'); assert.equal(knobChange('some-new-flag', 1, 2), null);
  // One byte per cell: floor(share x 250), 255 = not judged; 75 and up looks mixed (>= 30 %), 25 and up some (>= 10 %).
  assert.equal(cellShare(255), null); assert.equal(cellShare(null), null); assert.equal(cellShare(0), .002);
  assert.equal(cellLine(255), 'Too few transcripts to judge mixing');
  // The model's reading of a cell's own RNA, never where it came from as a fact nor "now"; and of neighbouring cells of
  // ANY kind: "another kind of cell's" was false for about 1 in 5 cells that look mixed, whose RNA from neighbours looks
  // like their own kind. The card says beside the number what it is not.
  assert.equal(cellLine(75), 'Looks mixed · the model reads about 30 % of its RNA as coming from neighbouring cells (estimate)');
  assert.equal(cellLine(74), 'Some mixing · the model reads about 29 % as coming from neighbouring cells (estimate)', '"some" never reads 30 % (the threshold of "looks mixed")');
  assert.equal(cellLine(25), 'Some mixing · the model reads about 10 % as coming from neighbouring cells (estimate)');
  assert.equal(cellLine(24), 'Little or no mixing seen (estimate)');
  const note = "A model's estimate, not proof of a boundary error. Between cells of the same kind it can't tell RNA apart.";
  assert.equal(QC_TEXT.cardNote, note);
  // The start is proseg's count at iteration 0, named so whatever the cells the run started from (H3).
  assert.equal(cardLine(80, 40, 'xenium'), `Mixing: the model reads about 32 % of its RNA as coming from neighbouring cells (16 % at iteration 0). ${note}`);
  assert.equal(cardLine(80, 40, 'other'), cardLine(80, 40, 'xenium'));
  assert.equal(cardLine(80, 255, 'other'), `Mixing: the model reads about 32 % of its RNA as coming from neighbouring cells. ${note}`);
  assert.equal(cardLine(80, null, 'other'), `Mixing: the model reads about 32 % of its RNA as coming from neighbouring cells. ${note}`);
  assert.equal(cardLine(255, 40, 'xenium'), 'Too few transcripts to judge mixing');
  // A reading from an earlier result than the one shown names its iteration (never passed off as the result shown's).
  assert.equal(cardLine(80, 40, 'xenium', { iter: 1200 }), `Mixing at iteration 1,200: the model read about 32 % of its RNA as coming from neighbouring cells (16 % at iteration 0). ${note}`);
  assert.equal(cardLine(80, null, 'xenium', { iter: 20 }), `Mixing at iteration 20: the model read about 32 % of its RNA as coming from neighbouring cells. ${note}`);
  // The mixing check's row leaves the note out while the panel's meaning line, under its headline, says it in view.
  assert.equal(cardLine(80, 40, 'xenium', { note: false }), 'Mixing: the model reads about 32 % of its RNA as coming from neighbouring cells (16 % at iteration 0).');
  assert.equal(cardLine(80, null, 'xenium', { iter: 20, note: false }), 'Mixing at iteration 20: the model read about 32 % of its RNA as coming from neighbouring cells.');
  assert.equal(cardLine(255, 40, 'xenium', { note: false }), 'Too few transcripts to judge mixing');
  for (const code of [0, 24, 25, 74, 75, 249]) {
    const text = cellLine(code) + cardLine(code, code, 'xenium');
    assert.doesNotMatch(text, /seems to come from|comes from a neighbour|another kind of cell|different kind|\bnow\b/, `${code}: no provenance words, no kind claimed`);
    if (code >= 25) assert.match(text, /the model reads/, `${code}: the model's reading`);
  }
  // Every cell-level string and the legend and tooltip: the whole share is never "another kind of cell's".
  for (const key of ['off', 'numberTooltip', 'showCellsTooltip', 'legendTitle', 'meaning', 'cardNote', 'how2']) {
    assert.doesNotMatch(QC_TEXT[key], /another kind of cell|of a different kind/i, key);
  }
  assert.match(QC_TEXT.meaning, /can miss mixing there or show mixing it can't back up/, 'the blind spot both ways');
  for (const code of [0, 24, 25, 74, 75, 249, 255]) assert.deepEqual(bannedWords(cellLine(code) + cardLine(code, code, 'xenium')), []);
});

// Every Live knobs label (engine/proseg_options.py, live: True), read from the schema file itself.
const optionsSource = readFileSync(new URL('../../engine/proseg_options.py', import.meta.url), 'utf8');
const LIVE_LABELS = Object.fromEntries([...optionsSource.matchAll(/\{'flag': '([^']+)',[^{}]*?'label': '([^']+)'[^{}]*?'live': True/gs)].map(m => [m[1], m[2]]));

test('knobs are named as Live knobs names them in a Setting line, a switch as its checkbox; elsewhere only vetted names; never a raw flag', () => {
  const labels = { 'cell-compactness': 'Cell compactness (lower = rounder cells)', 'prior-weight': 'Prior weight (higher = the starting cells hold their transcripts harder)',
    'no-diffusion': 'Switch off the transcript drift model (no diffusion)', temperature: 'Temperature (lower = boundaries settle)',
    'diffusion-sigma-far': 'Long drift distance, µm (diffusion sigma far)' };
  assert.equal(knobName(labels['cell-compactness']), 'Cell compactness'); assert.equal(knobHint(labels['cell-compactness']), 'lower = rounder cells');
  assert.equal(knobName(labels['no-diffusion']), 'Switch off the transcript drift model'); assert.equal(knobHint(labels['no-diffusion']), null, 'a hint with a banned word is left out');
  assert.equal(knobName(''), null); assert.equal(knobName(null), null); assert.equal(knobHint('No hint'), null);
  // The card's Setting line.
  assert.equal(settingLine({ flag: 'cell-compactness', from: 0.03999999910593033, to: 0.02, label: labels['cell-compactness'] }),
    'Setting: Cell compactness 0.04 → 0.02 (lower = rounder cells)');
  assert.equal(settingLine({ flag: 'prior-weight', from: 1, to: 2, label: labels['prior-weight'] }), 'Setting: Prior weight 1 → 2 (higher = the starting cells hold their transcripts harder)');
  // A switch named for what it switches off would read backwards ("on → off"): it is said as it is done, its checkbox.
  assert.equal(settingLine({ flag: 'no-diffusion', from: true, to: false, label: labels['no-diffusion'] }), 'Setting: untick "Switch off the transcript drift model" (drift on)');
  assert.equal(settingLine({ flag: 'no-diffusion', from: false, to: true, label: labels['no-diffusion'] }), 'Setting: tick "Switch off the transcript drift model" (drift off)');
  assert.equal(settingLine({ flag: 'some-switch', from: true, to: false, label: 'Some switch' }), 'Setting: untick "Some switch"', 'no plain change');
  assert.equal(settingLine({ flag: 'cell-compactness', from: 0.04, to: 0.02 }), 'Setting: Cell compactness 0.04 → 0.02', 'without the page\'s label: the names it knows');
  assert.equal(settingLine({ flag: 'prior-weight', from: 1, to: 2 }), 'Setting: Prior weight 1 → 2');
  assert.equal(settingLine({ flag: 'some-new-flag', from: 1, to: 2 }), null, 'never the raw flag');
  // A suggestion card's Setting line (shorter cards, 2026-10-22): the name and the values
  // alone, the hint (a switch: its plain change) for the line's ⓘ.
  assert.deepEqual(settingShort({ flag: 'cell-compactness', from: 0.08, to: 0.03999999910593033, label: labels['cell-compactness'] }),
    { text: 'Cell compactness 0.08 → 0.04', hint: 'lower = rounder cells' });
  assert.deepEqual(settingShort({ flag: 'prior-weight', from: 1, to: 2, label: labels['prior-weight'] }),
    { text: 'Prior weight 1 → 2', hint: 'higher = the starting cells hold their transcripts harder' });
  assert.deepEqual(settingShort({ flag: 'no-diffusion', from: true, to: false, label: labels['no-diffusion'] }),
    { text: 'Untick "Switch off the transcript drift model"', hint: 'drift on' });
  assert.deepEqual(settingShort({ flag: 'no-diffusion', from: false, to: true, label: labels['no-diffusion'] }),
    { text: 'Tick "Switch off the transcript drift model"', hint: 'drift off' });
  assert.deepEqual(settingShort({ flag: 'some-switch', from: true, to: false, label: 'Some switch' }), { text: 'Untick "Some switch"', hint: null }, 'no plain change');
  assert.deepEqual(settingShort({ flag: 'prior-weight', from: 1, to: 2 }), { text: 'Prior weight 1 → 2', hint: null }, 'without the page\'s label');
  assert.equal(settingShort({ flag: 'some-new-flag', from: 1, to: 2 }), null, 'never the raw flag');
  // The timeline's change readout: a vetted name only (KNOB_NAMES), else the plain change; "—" for no value.
  assert.equal(knobLine('cell-compactness', 0.03999999910593033, 0.019999999552965164), 'Cell compactness 0.04 → 0.02 (tighter shapes)');
  assert.equal(knobLine('prior-weight', 1, 4, labels['prior-weight']), 'firmer hold: 1 → 4', 'its Live knobs name holds a banned word');
  assert.equal(knobLine('no-diffusion', false, true, labels['no-diffusion']), 'drift off', 'a switch by its plain change');
  assert.equal(knobLine('temperature', 1, 0.5, labels.temperature), 'Temperature 1 → 0.5 (cooler)');
  assert.equal(knobLine('diffusion-sigma-far', 4, 6, labels['diffusion-sigma-far']), 'drift distance: 4 → 6');
  assert.equal(knobLine('ab-nihlo-bubble-prob', 0.05, 0.1), 'gap proposals: 0.05 → 0.1', 'the value once, beside the change');
  assert.equal(knobLine('cell-compactness', undefined, 0.02), 'Cell compactness: 0.02', 'no value before');
  assert.equal(knobLine('some-new-flag', undefined, 2), 'Another setting changed: — → 2', 'absent before: not "automatic"');
  assert.equal(knobLine('some-new-flag', 1, 2), 'Another setting changed: 1 → 2');
  // Every live option of the schema: the readout of a change of it and its Setting line hold no banned word, no jargon
  // outside the exact Live knobs name a Setting line may hold, and never the raw flag.
  assert.ok(Object.keys(LIVE_LABELS).length >= 10 && LIVE_LABELS['prior-weight'] === labels['prior-weight']);
  const names = Object.values(LIVE_LABELS).map(knobName);
  for (const [flag, label] of Object.entries(LIVE_LABELS)) {
    for (const [from, to] of [[1, 2], [2, 1], [undefined, 2], [false, true], [true, false], [null, 0.5]]) {
      const line = knobLine(flag, from, to, label);
      assert.deepEqual(bannedWords(line), [], `${flag}: ${line}`); assert.ok(!line.includes(flag), line);
      assert.ok(!/\b(band|depth-matched|compactness)\b/i.test(line) || KNOB_NAMES[flag], `${flag}: plain words (${line})`);
      const setting = settingLine({ flag, from, to, label });
      if (setting) { assert.deepEqual(bannedWords(settingChecked(setting, names)), [], setting); assert.ok(!setting.includes(`--${flag}`), setting); }
      // ... and a card's short Setting line, and its hint (in the ⓘ, never with the name: no exception for it).
      const short = settingShort({ flag, from, to, label });
      assert.ok(short, `${flag}: named`);
      assert.deepEqual(bannedWords(settingChecked(short.text, names)), [], short.text); assert.ok(!short.text.includes(flag), short.text);
      assert.deepEqual(bannedWords(short.hint ?? ''), [], `${flag}: ${short.hint}`);
    }
  }
  assert.equal(settingChecked('Setting: Prior weight 1 → 2 (higher = closer to Xenium)'), 'Setting:  1 → 2 (higher = closer to Xenium)');
  assert.equal(settingChecked('A sentence about the Prior weight'), 'A sentence about the Prior weight', 'only a Setting line may hold it');
  // A card's short Setting line is one, whole: the name with its two values, or a switch's checkbox; nothing else.
  assert.equal(settingChecked('Prior weight 1 → 2'), ' 1 → 2'); assert.equal(settingChecked('  Prior weight 0.5 → automatic '), '   0.5 → automatic ');
  assert.equal(settingChecked('Untick "Prior weight"'), 'Untick ""');
  for (const line of ['Prior weight 1 → 2 and more', 'The Prior weight 1 → 2', 'Prior weight is 2', 'Prior weight 1 → 2 (higher = closer to Xenium)', 'Untick "Prior weight" now'])
    assert.equal(settingChecked(line), line, line);
  assert.equal(sameSetting(0.019999999552965164, 0.02), true); assert.equal(sameSetting(0.02, 0.03), false); assert.equal(sameSetting(true, true), true);
  assert.equal(sameSetting(undefined, 1), false);
});

test('a suggestion names its own change; cards are known by the engine\'s id (never their areas); the repeat-run numbers are calibration.json\'s', () => {
  assert.equal(changeLabel({ rule: 'R2', flag: 'cell-compactness', from: 0.08, to: 0.04 }), 'standard shapes');
  assert.equal(changeLabel({ rule: 'R4', flag: 'cell-compactness', from: 0.04, to: 0.02 }), 'tighter shapes');
  assert.equal(changeLabel({ rule: 'R3', flag: 'no-diffusion', from: true, to: false }), 'drift on');
  assert.equal(changeLabel({ rule: 'R5', flag: 'prior-weight', from: 1, to: 2 }), 'firmer hold');
  // Something tried (the trial's title, R7, the new run's notice): a switch "with drift on", the hold "a firmer hold".
  assert.equal(tryingLabel({ flag: 'no-diffusion', from: true, to: false }), 'with drift on');
  assert.equal(tryingLabel({ flag: 'prior-weight', from: 1, to: 2 }), 'a firmer hold to the starting cells');
  assert.equal(tryingLabel({ flag: 'cell-compactness', from: 0.04, to: 0.02 }), 'tighter shapes');
  // ... and as something tried elsewhere (R7, the new run's notice): a switch by its plain change, never "with with".
  assert.equal(triedLabel({ flag: 'no-diffusion', from: true, to: false }), 'drift on');
  assert.equal(triedLabel({ flag: 'prior-weight', from: 1, to: 2 }), 'a firmer hold to the starting cells');
  assert.equal(copy('newRunReady', { setting: triedLabel({ flag: 'no-diffusion', from: true, to: false }) }).includes('with drift on from the start'), true);
  assert.equal(cardKey({ id: 'R4:cell-compactness:0.02', rule: 'R4', areas: ['A'] }), cardKey({ id: 'R4:cell-compactness:0.02', rule: 'R4', areas: ['A', 'C'] }));
  assert.equal(cardKey({ rule: 'R4', areas: ['A'], flag: 'cell-compactness', to: 0.02 }), cardKey({ rule: 'R4', areas: ['C', 'A'], flag: 'cell-compactness', to: 0.02 }),
    'without the engine\'s id: the same, from its change');
  assert.notEqual(cardKey({ rule: 'R4', flag: 'cell-compactness', to: 0.02 }), cardKey({ rule: 'R2', flag: 'cell-compactness', to: 0.04 }));
  assert.notEqual(cardKey({ rule: 'R7', tried: { flag: 'cell-compactness', to: 0.02 } }), cardKey({ rule: 'R7', tried: { flag: 'prior-weight', to: 2 } }));
  assert.equal(cardKey(null), '');
  // The calibration, in the measuring harness's own folder (SOPA_COMPANION_HARNESS_DIR): checked only where it is
  // set and holds the file (an installed copy has neither).
  const harness = process.env.SOPA_COMPANION_HARNESS_DIR;
  const file = harness ? join(harness, 'resolvi-prep', 'phase0', 'calibration.json') : null;
  if (file && existsSync(file)) {
    const calibration = JSON.parse(readFileSync(file, 'utf8')).build;
    assert.equal(calibration.copy_updates.R4_confidence, QC_TEXT[RULE_COPY.R4.confidence]);
  }
  // The suggestions' and trials' words, in plain English, each through the banned-word test above (they are in
  // the registry); the raw flag is no longer shown anywhere (the Setting line's tooltip is gone).
  for (const key of ['suggestionFooterEnded', 'suggestionFooterEndedWhole', 'suggestionFooterInfo', 'suggestionFooterWhole', 'suggestionFooterDrift',
    'suggestionFooterNewRun', 'trialNoHistory', 'trialEnded', 'tryFailed', 'trialFailed', 'trialFailedUndone', 'trialEndFailed',
    'undoFailed', 'newRunFailed', 'newRunPriorMissing', 'newRunTooltipStill', 'settingHint', 'settingTick',
    'settingTickPlain', 'trialTitleWaiting', 'trialTitleUndone', 'trialTitleKept', 'trialTitleEnded', 'trialUndoneValue',
    'newRunReady', 'knobLine', 'knobLineNamed', 'knobSwitch', 'knobSwitchNamed', 'knobSet', 'knobPlain', 'knobOther',
    'trialFormedPast', 'trialLastStage', 'trialPendingLater', 'trialPendingLaterNoTime', 'trialNoChangeAfterLater',
    'trialNoChangeAfterLaterNoTime', 'trialLessArea', 'trialMoreArea', 'ruleUBodySwitch', 'ruleR2BodyRose',
    'ruleR2NotRose', 'likelySize', 'timingUntested', 'tryTooltipSwitch', 'tryingWith', 'driftActsLater', 'tryAlready', 'tryBeyond', 'tryNotLive',
    'tryNotNow', 'tryBusy', 'tryWaiting', 'undoNothing',
    // 2026-10-22 (shorter cards): the short Setting line, the why's other sentences.
    'settingShort', 'settingTickOn', 'settingTickOff', 'ruleR1More', 'ruleR3More', 'ruleR6More', 'ruleR7More']) assert.ok(key in QC_COPY, key);
  for (const key of ['settingFlag', 'settingSwitch', 'settingSwitchNamed', 'ruleR4Idea', 'ruleR6Idea']) assert.ok(!(key in QC_COPY), `${key} is gone`);
  // The cards' confidence lines, short (their tests are under Why?); drift (R3) is judged live, and "Likely to help:
  // about 1 in 100" as R2 since its size was measured on proseg's count (the engine's CONFIDENCE).
  assert.equal(QC_TEXT.likelySize, 'Likely to help: about 1 in 100');
  assert.equal(RULE_COPY.R3.confidence, 'likely'); assert.equal(QC_TEXT[RULE_COPY.R3.confidence], 'Likely to help');
  assert.equal(RULE_COPY.R3.sized, 'likelySize'); assert.equal(RULE_COPY.R3.sized, RULE_COPY.R2.sized);
});


test('a change is the engine\'s like-for-like one, said only past its threshold, sized unrounded, "by this model\'s reading" under the model\'s own', () => {
  assert.equal(likeForLike(null), null); assert.equal(likeForLike({ raw: -1.7 }), null, 'no like-for-like change: no change at all (a raw one rewards lost transcripts)');
  // The engine's verdict decides; the page never subtracts two rounded numbers (12.49 -> 12.51 was "▲1").
  // One short verdict line that says what it compares (H13): like for like, each cell with itself at the start, never
  // beside the start's own number (iteration 0's, which the line does not compare with: in its tooltip); "by this
  // model's reading" in its tooltip (changeTooltip, changeLightTooltip), a light change by its hollow arrow, as the chip.
  const tiny = likeForLike({ raw: 0.02, matched: 0.02, threshold: 0.7, model_threshold: 0.6, meaningful: false, direction: 'same' });
  assert.equal(tiny.direction, 'same'); assert.equal(verdictLine(tiny), 'Same as at the start, like for like');
  assert.equal(verdictLine(tiny, '16'), 'Same as at the start, like for like', 'no start number, whatever is passed');
  const fewer = likeForLike({ matched: -1.84, threshold: 0.3, model_threshold: 0.6, meaningful: true, direction: 'fewer' });
  assert.deepEqual([fewer.direction, fewer.light, fewer.d], ['fewer', false, '1.8']);
  assert.equal(verdictLine(fewer), '▼ 1.8 fewer, like for like');
  const light = likeForLike({ matched: 0.44, threshold: 0.3, model_threshold: 0.6, meaningful: true, direction: 'more' });
  assert.deepEqual([light.direction, light.light], ['more', true]);
  assert.equal(verdictLine(light), '△ 0.4 more, like for like');
  assert.equal(verdictLine(likeForLike({ matched: -0.45, threshold: 0.3, meaningful: true })), '▽ 0.5 fewer, like for like',
    `without the engine's model threshold: ${MODEL_THRESHOLD} (as measured on the model's points)`);
  assert.equal(verdictLine(likeForLike({ matched: 1.2, threshold: 0.3, model_threshold: 0.6, meaningful: true, direction: 'more' })), '▲ 1.2 more, like for like');
  assert.match(QC_TEXT.changeTooltip, /^Like for like: each cell is compared with itself at the start/);
  assert.match(QC_TEXT.changeLightTooltip, /holds for this model only\.$/);
  // Without `meaningful`: the engine's direction, else its threshold; never a direction from a change under it.
  assert.equal(likeForLike({ matched: -0.9, threshold: 1.0 }).direction, 'same');
  assert.equal(likeForLike({ matched: -1.2, threshold: 1.0 }).direction, 'fewer');
  assert.equal(likeForLike({ matched: -1.2, direction: 'same' }).direction, 'same');
  assert.equal(likeForLike({ matched: 0, meaningful: true }).direction, 'same');
  assert.equal(verdictLine(null), '');
  // A live verdict held steady by the engine (qc_rules.Steady): its own word, though the
  // change has just crossed back under its threshold (and never a direction its sign would give instead).
  const held = likeForLike({ matched: -0.97, threshold: 1.0, meaningful: true, direction: 'fewer', direction_now: 'same', held: true });
  assert.deepEqual([held.direction, held.d], ['fewer', '1.0']);
  assert.equal(likeForLike({ matched: 0.2, threshold: 1.0, meaningful: true, direction: 'fewer', held: true }).direction, 'fewer');
  assert.equal(likeForLike({ matched: -1.2, threshold: 1.0, meaningful: false, direction: 'fewer' }).direction, 'same', 'never a direction the engine did not call meaningful');
  // The chip says "like for like" beside a change it shows, with a hollow arrow under the model's own threshold;
  // "same as at the start" alone, its tooltip keeping "like for like".
  assert.equal(copy('chipFewer', { n: '12', d: '1.0' }), 'Mixing 12 in 100 · ▼1.0 like for like');
  assert.equal(copy('chipMoreLight', { n: '12', d: '0.4' }), 'Mixing 12 in 100 · △0.4 like for like');
  assert.equal(copy('chipSame', { n: '12' }), 'Mixing 12 in 100 · same as at the start');
  assert.match(copy('chipTooltipSame', { n: '12' }), /about the same as at the start, like for like\./);
  // The Undo card's numbers are like for like too.
  assert.match(copy('ruleUBody', { a: '15.0', b: '16.6', setting: 'tighter shapes' }), /^Like for like, mixing rose from 15\.0 to 16\.6 in 100/);
  // No verdict from the engine (engine/qc_rules.py change_block): too few cells, cells still forming, or none yet.
  assert.equal(likeForLike({ matched: -2.1, threshold: 2.4, meaningful: false, direction: null, enough: false }), null, 'not "about the same": nothing');
  assert.equal(likeForLike({ matched: -2.1, threshold: 0.7, meaningful: false, direction: null, enough: true, forming: true }), null);
  assert.equal(likeForLike({ matched: -2.1, threshold: 0.7, meaningful: false, direction: null }), null);
  assert.equal(likeForLike({ matched: null, threshold: 1, meaningful: false, direction: null }), null);
  // Every chip word, both ways, through the banned words and the plain-words rule; every verdict line says what it
  // compares (like for like) in 6-8 words, and no start number.
  for (const key of ['chipMore', 'chipFewer', 'chipMoreLight', 'chipFewerLight', 'chipTooltipMoreLight', 'chipTooltipFewerLight', 'chipTooltipSame']) {
    const text = copy(key, { d: '1.2', n: '11' });
    assert.deepEqual(bannedWords(text), [], key); assert.match(text, /like for like/, key); assert.doesNotMatch(text, /depth-matched|matched/i, key);
  }
  for (const key of ['verdictSame', 'verdictMore', 'verdictFewer', 'verdictMoreLight', 'verdictFewerLight']) {
    const text = copy(key, { d: '1.2', s: '16' }), n = text.split(' ').length;
    assert.deepEqual(bannedWords(text), [], key); assert.match(text, /like for like$/, key); assert.ok(n >= 6 && n <= 8, `${key}: a few words (${n})`);
    assert.doesNotMatch(QC_TEMPLATES[key], /\{s\}/, `${key}: no start number`);
  }
  // E2: a hollow arrow (under the model's own threshold) says, in the chip's tooltip, that it holds for this model only.
  assert.equal(copy('chipTooltipMoreLight', { n: '11', d: '0.4' }), '11 in 100 cells look mixed; 0.4 more than at the start, like for like. The gap is smaller than another learning of the model could give, so it holds for this model only. Click to open the mixing check.');
  assert.equal(copy('chipTooltipFewerLight', { n: '11', d: '0.4' }), '11 in 100 cells look mixed; 0.4 fewer than at the start, like for like. The gap is smaller than another learning of the model could give, so it holds for this model only. Click to open the mixing check.');
  for (const key of ['chipTooltipMoreLight', 'chipTooltipFewerLight']) assert.ok(QC_TEMPLATES[key].includes(QC_TEXT.changeLightTooltip), `${key}: changeLightTooltip's clause`);
});

test('the verdict line says what it compares with its two numbers, the change\'s own (P2), and a stored summary without them keeps its words', () => {
  // The summary's change carries the two numbers behind matched (per100_now, per100_start: the cells judged both now and
  // at the start, each cut to the same depth), unrounded; the line shows them to one decimal.
  const change = { matched: -1.8, threshold: 0.7, meaningful: true, direction: 'fewer', model_threshold: 1.5, per100_now: 6.8125, per100_start: 8.6125 };
  const like = likeForLike(change);
  assert.deepEqual([like.direction, like.light, like.d, like.now, like.start], ['fewer', false, '1.8', '6.8', '8.6']);
  assert.equal(verdictLine(like), '▼ Fewer, like for like: 6.8 now, 8.6 at the start');
  assert.equal(verdictLine(likeForLike({ ...change, model_threshold: 2.3 })), '▽ Fewer, like for like: 6.8 now, 8.6 at the start');
  assert.equal(verdictLine(likeForLike({ ...change, matched: 1.8, direction: 'more', per100_now: 10.4, per100_start: 8.6 })), '▲ More, like for like: 10.4 now, 8.6 at the start');
  assert.equal(verdictLine(likeForLike({ ...change, matched: 1.8, direction: 'more', model_threshold: 2.3, per100_now: 10.4, per100_start: 8.6 })), '△ More, like for like: 10.4 now, 8.6 at the start');
  assert.equal(verdictLine(likeForLike({ ...change, matched: -0.3, meaningful: false, direction: 'same', per100_now: 8.3, per100_start: 8.6 })), 'About the same, like for like: 8.3 now, 8.6 at the start');
  // Without both numbers (a summary stored before them, or one of them not a number): today's line, its size alone.
  for (const without of [{ per100_now: undefined, per100_start: undefined }, { per100_now: null }, { per100_start: Number.NaN }]) {
    const old = likeForLike({ ...change, ...without });
    assert.deepEqual([old.now, old.start], [null, null]); assert.equal(verdictLine(old), '▼ 1.8 fewer, like for like');
  }
  assert.equal(verdictLine(likeForLike({ matched: 0.1, threshold: 0.7, meaningful: false, direction: 'same' })), 'Same as at the start, like for like');
  // A word held a few seconds (the engine's Steady keeps the word shown while the change itself has moved on): the same
  // words without numbers, whatever numbers the block carries (the engine's are those the word compared, an earlier
  // result's), and its tooltip's changeHeld says what the change reads now.
  const held = { threshold: 0.7, model_threshold: 1.5, held: true, per100_start: 8.6 };
  for (const [block, line] of [
    [{ matched: -1.0, meaningful: false, direction: 'same', direction_now: 'fewer', per100_now: 7.6 }, 'About the same, like for like'],
    [{ matched: 0.9, meaningful: true, direction: 'fewer', direction_now: 'more', per100_now: 9.5 }, '▽ Fewer, like for like'],
    [{ matched: 0.2, meaningful: true, direction: 'fewer', direction_now: 'same', per100_now: 8.8 }, '▽ Fewer, like for like'],
    [{ matched: -1.2, meaningful: true, direction: 'more', direction_now: 'fewer', per100_now: 7.4 }, '△ More, like for like'],
    [{ matched: -2.4, meaningful: true, direction: 'fewer', direction_now: 'fewer', per100_now: 6.2 }, '▼ Fewer, like for like'],
    [{ matched: 2.4, meaningful: true, direction: 'more', direction_now: 'same', per100_now: 11 }, '▲ More, like for like'],
  ]) {
    const like = likeForLike({ ...held, ...block });
    assert.deepEqual([like.held, like.now, like.start], [true, null, null], line); assert.equal(verdictLine(like), line);
  }
  assert.equal(likeForLike({ ...change, held: false }).now, '6.8', 'a word that is its own keeps its numbers');
  // F8: a held word keeps the size it was decided on, that of the two numbers its word compared (the held block's
  // per100_now and per100_start), never the change's own size now: ▼2.5 and a solid arrow, not ▽0.3 inside the threshold.
  const decidedBlock = { matched: -0.3, threshold: 0.7, model_threshold: 2.3, meaningful: true, direction: 'fewer', direction_now: 'same', held: true,
    per100_now: 8.6, per100_start: 11.1 };
  const decided = likeForLike(decidedBlock);
  assert.deepEqual([decided.direction, decided.d, decided.light, decided.now, decided.start], ['fewer', '2.5', false, null, null]);
  assert.equal(verdictLine(decided), '▼ Fewer, like for like');
  assert.equal(copy('chipFewer', { n: '9', d: decided.d }), 'Mixing 9 in 100 · ▼2.5 like for like');
  // ... decided on a size under the model's own threshold: drawn hollow, whatever the change's own size now.
  const decidedLight = likeForLike({ ...decidedBlock, matched: -2.6, per100_now: 9.9 });
  assert.deepEqual([decidedLight.d, decidedLight.light], ['1.2', true]); assert.equal(verdictLine(decidedLight), '▽ Fewer, like for like');
  // ... a word that is its own, and a held block stored before the numbers came: the change's own size.
  assert.deepEqual([likeForLike({ ...decidedBlock, held: false, direction_now: undefined }).d, likeForLike({ ...decidedBlock, per100_now: undefined }).d], ['0.3', '0.3']);
  for (const key of ['verdictSameHeld', 'verdictMoreHeld', 'verdictFewerHeld', 'verdictMoreLightHeld', 'verdictFewerLightHeld']) {
    const at = QC_TEMPLATES[key.replace(/Held$/, 'At')];
    assert.equal(`${QC_TEXT[key]}: {now} now, {start} at the start`, at, `${key}: the words of its line with numbers`);
    assert.deepEqual(bannedWords(QC_TEXT[key]), [], key);
  }
  // The words exactly, each with "like for like" and what it compares, 11-12 words with numbers in, none banned.
  const words = {
    verdictSameAt: 'About the same, like for like: {now} now, {start} at the start',
    verdictMoreAt: '▲ More, like for like: {now} now, {start} at the start',
    verdictFewerAt: '▼ Fewer, like for like: {now} now, {start} at the start',
    verdictMoreLightAt: '△ More, like for like: {now} now, {start} at the start',
    verdictFewerLightAt: '▽ Fewer, like for like: {now} now, {start} at the start',
  };
  for (const [key, text] of Object.entries(words)) {
    assert.equal(QC_TEMPLATES[key], text, key);
    const line = copy(key, { now: '6.8', start: '8.6' }), n = line.split(' ').length;
    assert.deepEqual(bannedWords(line), [], key); assert.match(line, /like for like/, key); assert.ok(n >= 11 && n <= 12, `${key}: ${n} words`);
  }
  assert.equal(QC_TEXT.likeForLikeNumbers, "Its two numbers: of every 100 cells judged both now and at the start, how many look mixed, each cell cut to the same number of transcripts on both sides. So they can differ from the number above and from iteration 0's.");
  assert.deepEqual(bannedWords(QC_TEXT.likeForLikeNumbers), []);
});

test('coverage over all the run\'s cells and transcript retention, in plain words', () => {
  assert.equal(HEADLINE_MIN_JUDGED, 30); assert.equal(LOW_COVERAGE, 0.5); assert.equal(RETENTION_LOW, 0.9);
  const engine = coverageOf({ judged: 31000, coverage: { cells: 40000, judged: 31000, empty: 100, under20: 2900, under100: 6000, judged_share: 0.775 } });
  assert.deepEqual(engine, { cells: 40000, judged: 31000, share: 0.775, empty: 100, under20: 2900, under100: 6000 });
  // The worker's own counts (a start line: its `cells`); transcripts [0, 0, 200]: one judged cell of three.
  assert.deepEqual(coverageOf({ judged: 1, ncells: 3, empty: 2, under20: 0, under100: 0 }), { cells: 3, judged: 1, share: 1 / 3, empty: 2, under20: 0, under100: 0 });
  assert.equal(coverageOf({ judged: 3200, cells: 3439 }).share, 3200 / 3439);
  assert.equal(coverageOf({ judged: 300 }), null, 'not counted: no coverage claimed'); assert.equal(coverageOf(null), null);
  assert.equal(copy('basisCoverage', { count: QC_TEXT.countAveraged, j: '31,000', n: '40,000' }), "Based on proseg's count (averaged): 31,000 of 40,000 cells (those with 100 or more transcripts)");
  assert.equal(copy('tooFewCount', { j: '1', n: '3' }), 'Too few cells with 100 or more transcripts yet (1 of 3).');
  assert.equal(retentionOf({}), null); assert.deepEqual(retentionOf({ retention: { ratio: 0.92, tx: 920, start_tx: 1000 } }), { ratio: 0.92, tx: 920, start: 1000 });
  // In view (short) only when it moves the number by itself, what it does in its ⓘ; else a sentence for
  // the headline's ⓘ.
  const of = ratio => retentionOf({ retention: { ratio } });
  const line = ratio => retentionLine(of(ratio));
  assert.equal(line(0.92), '', 'within 10 %: not in view'); assert.equal(retentionMoves(of(0.92)), null); assert.equal(retentionTooltip(of(0.92)), '');
  // Either way it ends with what it compares: proseg's count now against the cells it started with, kept the same way.
  assert.equal(retentionSentence(of(0.92)), "Cells hold 92 % of the transcripts they started with. Proseg's count now, against the cells proseg started with, kept the same way.");
  assert.equal(line(0.75), 'Cells hold 75 % of their start RNA'); assert.equal(retentionMoves(of(0.75)), 'down');
  assert.equal(retentionTooltip(of(0.75)), `${QC_TEXT.retentionDownTooltip} ${QC_TEXT.retentionBasis}`); assert.match(QC_TEXT.retentionDownTooltip, /lowers the number above/);
  assert.equal(line(0.899), 'Cells hold 90 % of their start RNA', 'under 90 %, unrounded');
  assert.equal(RETENTION_HIGH, 1.1);
  // The usual case, cells that gained transcripts, explains the raw number too.
  assert.equal(line(1.23), 'Cells hold 123 % of their start RNA'); assert.equal(retentionTooltip(of(1.23)), `${QC_TEXT.retentionUpTooltip} ${QC_TEXT.retentionBasis}`);
  assert.match(QC_TEXT.retentionUpTooltip, /raises the number above/);
  assert.equal(line(1.099), '', 'under 110 %, unrounded: no claim'); assert.equal(line(1.004), '');
  // Under 300 judged cells the basis says the number is rough, and a short line says so in view.
  assert.equal(ROUGH_JUDGED, 300);
  assert.equal(copy('basisCoverageFew', { count: QC_TEXT.countSaved, j: '42', n: '120' }), "Based on proseg's count (as saved): only 42 of 120 cells (those with 100 or more transcripts), so this number is rough");
  assert.equal(copy('roughCells', { j: '42', n: '120' }), 'Rough: based on only 42 of 120 cells');
  assert.equal(retentionLine(retentionOf({ retention: { ratio: null } })), '', 'no start line: no ratio'); assert.equal(retentionLine(null), '');
  assert.equal(retentionSentence(null), '');
});

test('"How is this estimated?" is five short points and how the model was learned, at most 90 words, never naming the user\'s files', () => {
  assert.deepEqual(HOW_KEYS, ['how1', 'how2', 'how3', 'how4', 'how5']);
  assert.equal(QC_TEXT.how1, "A ResolVI model learned this sample's kinds of cells once.");
  assert.match(QC_TEXT.how3, /same-kind neighbours apart; truly overlapping cells look mixed even with good outlines\.$/, 'its blind spots both ways');
  assert.match(QC_TEXT.how4, /aren't judged; changes are compared like for like/);
  assert.equal(QC_TEXT.how5, "It reads proseg's own count of each cell: the transcripts proseg places in it, less those it reads as background.",
    "proseg's own count, never the outline's");
  const words = text => text.split(/\s+/).length, points = words(HOW_KEYS.map(k => QC_TEXT[k]).join(' '));
  // The provenance line: the model's cells, when, how long; a small table's passes (more_epochs_small_table, the
  // number ui.qc_provenance_shown looks for); its table in plain words in its tooltip (never a zarr key on screen).
  const plain = learnedLine({ cells: 42000, table: 'xenium_table' }, null, { date: '21 Oct 2026', t: '7 min' });
  assert.deepEqual(plain, { text: 'Learned from 42,000 cells on 21 Oct 2026, in 7 min.', tooltip: "The cells it learned from: Xenium's cells." });
  const small = learnedLine({ cells: 3455, table: 'xenium_table' }, { table: 'table', priors: 'single', epochs_run: 429, epochs_notebook: 50,
    notebook_equivalent: false, differences: ['more_epochs_small_table', 'single_table_priors'] }, { date: '1 Oct 2026', t: '12 min' });
  assert.equal(small.text, 'Learned in 429 passes over 3,455 cells on 1 Oct 2026, in 12 min.');
  assert.equal(small.tooltip, `The cells it learned from: the sample's other cell table. ${QC_TEXT.howPasses}`, "the provenance's own table wins");
  assert.ok(points + words(plain.text) <= 90 && points + words(small.text) <= 90, `${points} + ${words(small.text)} words`);
  // A model's own (notebook_equivalent) or of unknown provenance: the same plain line, nothing claimed either way.
  assert.equal(learnedLine({ cells: 15000 }, { notebook_equivalent: true, differences: [] }, { date: '17 Oct 2026', t: '2 min' }).text,
    'Learned from 15,000 cells on 17 Oct 2026, in 2 min.');
  assert.equal(learnedLine({ cells: 15000 }, null, {}).text, 'Learned from 15,000 cells on —, in —.');
  assert.equal(learnedLine({ cells: 15000 }, null, {}).tooltip, '', 'no table said: no tooltip');
  assert.equal(learnedLine(null, null), null); assert.equal(learnedLine({ cells: null }, null), null, 'no cell count: no line');
  assert.deepEqual([tableName('xenium_table'), tableName('table'), tableName('cells_2'), tableName(null)], ["Xenium's cells", "the sample's other cell table", "the sample's other cell table", null]);
  for (const text of [...HOW_KEYS.map(k => QC_TEXT[k]), plain.text, plain.tooltip, small.text, small.tooltip]) {
    assert.deepEqual(bannedWords(text), [], text);
    assert.doesNotMatch(text, /notebook|part\s*3|same (settings|model|recipe)/i, text);
  }
});

test('no string says where RNA came from as a fact, claims a setting cannot fix something, or calls a score an error count', () => {
  const all = Object.entries({ ...QC_TEXT, ...QC_TEMPLATES });
  for (const [key, text] of all) {
    assert.doesNotMatch(text, /no setting is likely|unlikely to fix|settings are unlikely|cannot fix|can't fix|won't fix|whatever the settings/i, key);
    assert.doesNotMatch(text, /seems to come from a neighbour|of its RNA seems to come from/i, key);
  }
  // The meaning beside the headline, and its blind spot, in view (qc-panel.js puts it under the number).
  assert.equal(QC_TEXT.meaning, "A model's estimate of how mixed each cell's RNA looks, not a count of boundary errors. Between cells of the same kind it can't tell RNA apart, so it can miss mixing there or show mixing it can't back up.");
  // The cards that offer no setting say what was seen or where to look; their own line wins over the engine's.
  for (const rule of ['R1', 'R6', 'R7']) {
    assert.equal(RULE_COPY[rule].own, true, rule); assert.notEqual(RULE_COPY[rule].confidence, 'unlikely', rule);
    assert.doesNotMatch(QC_TEXT[RULE_COPY[rule].title], /unlikely|probably the tissue/i, rule);
  }
  assert.equal(QC_TEXT.ruleR1Title, 'Mixed from the start: cause not known');
  // (2026-10-22: no area is shown, so the body says "they", the cells, never "the area".) What was seen is the card's
  // why in view; the possible causes, and for R7 that other settings were not tried, open its Why? (ruleR1More,
  // ruleR7More: shorter cards).
  assert.equal(copy('ruleR1Body'), 'These cells already looked mixed at iteration 0, and proseg has not made them more mixed.');
  assert.match(QC_TEXT.ruleR1More, /^The cause isn't known: .* or they may be at the edge of the tissue\./);
  assert.equal(copy('ruleR7Body', { setting: 'tighter shapes' }), 'You tried tighter shapes here and mixing rose rather than dropped.');
  assert.match(QC_TEXT.ruleR7More, /^Other settings were not tried here\. Its cells' RNA may include a neighbour's/);
  for (const rule of ['R1', 'R3', 'R6', 'R7']) assert.equal(RULE_COPY[rule].more, `rule${rule}More`, rule);
  for (const rule of ['U', 'R2', 'R4', 'R5']) assert.equal(RULE_COPY[rule].more, null, rule);
  assert.equal(QC_TEXT.closerLook, 'Worth a closer look');
  assert.equal(copy('trialDepthNote', { x: '8' }), 'Cells also lost 8 % of their transcripts. The rows above compare like for like, so that loss alone does not move them.');
  assert.equal(QC_TEXT.trialColumns, 'expected → now, like for like');
  assert.deepEqual([QC_TEXT.inspect, bannedWords(QC_TEXT.inspectTooltip + QC_TEXT.trialNoMatched)], ['Look at its cells', []]);
});

test('2026-10-22: no string names a flagged area or its letter; the legend is a short title and one row per shade', () => {
  // Nothing of the area list, the chips, the outline key or the notes about areas is left in the deck.
  for (const key of ['areasHeading', 'areasNone', 'areaRow', 'areaChip', 'areaChipLetter', 'areaTooltip', 'areaShare', 'areaRose', 'areaSimilar',
    'areaAlready', 'tagRose', 'tagEdge', 'areasSmall', 'areasSmallTooltip', 'outlineLegend', 'outlineSolid', 'outlineDashed', 'outlineDotted',
    'forArea', 'forAreas', 'forWhole', 'finishedArea', 'finishedAreas', 'stoppedArea', 'stoppedAreas', 'finishedNone', 'stoppedNone',
    'finishedStartArea', 'finishedStartAreas', 'areasFormed', 'single', 'pausedStart', 'threeDNote', 'legendNote', 'overviewEdge'])
    assert.ok(!(key in QC_COPY), `${key} is gone`);
  // The test of a setting names its own area (the region it re-runs, or the square around the cell, and the areas it
  // was measured in), never a flagged one.
  for (const [key, text] of Object.entries({ ...QC_TEXT, ...QC_TEMPLATES })) {
    assert.doesNotMatch(text, /\{(L|letters)\}/, `${key}: never an area's letter`);
    if (!key.startsWith('test')) assert.doesNotMatch(text, /\bareas?\b/i, `${key}: never "area"`);
  }
  // A trial about flagged areas and a card's short why both say "where mixing stands out" (with no area on screen,
  // "here" pointed at nothing); the whole tissue's words are as before.
  assert.equal(QC_TEXT.standsOut, 'where mixing stands out'); assert.equal(QC_TEXT.standsOutHere, undefined);
  assert.equal(copy('ruleR4Body', { where: QC_TEXT.standsOut, g: '18', start: "Xenium's cells" }),
    'Cells where mixing stands out grew about 18 % bigger and seem to reach into their neighbours.');
  assert.equal(copy('ruleR2Body', { v: '0.16', where: QC_TEXT.inTissue, g: '16', start: 'the starting cells' }),
    'Shapes are looser than standard (0.16); cells in the tissue grew about 16 % bigger.');
  // Every card's why in view is one short sentence, its numbers kept (about 12 words; at most 17 with the longest values:
  // U names the knob changed, R7 the change tried).
  const values = { start: "Xenium's cells", where: QC_TEXT.standsOut, g: '11–24', k: '11', v: '0.08', a: '11.0', b: '12.6', change: 'one-piece cells off' };
  for (const key of [...Object.values(RULE_COPY).map(r => r.body), 'ruleR2BodyRose', 'ruleUBodySwitch']) {
    const text = copy(key, { ...values, setting: key.startsWith('ruleU') ? 'Cell compactness' : 'a firmer hold to the starting cells' });
    assert.ok(text.split(/\s+/).length <= 17, `${key}: ${text}`); assert.doesNotMatch(text.slice(0, -1), /[.!?] [A-Z]/, `${key}: one sentence`);
  }
  assert.equal(copy('trialWatching', { k: '62' }), 'Watching where mixing stands out and the whole tissue · 62 iterations so far');
  assert.equal(copy('trialArea', { a: '23.0', b: '21.4' }), 'Where mixing stands out: 23.0 → 21.4 in 100');
  assert.equal(QC_TEXT.trialLessArea, 'Less mixed than expected where mixing stands out, by more than the usual ups and downs.');
  for (const key of ['suggestionFooter', 'suggestionFooterEnded']) assert.ok(QC_TEXT[key].includes('not just where mixing stands out'), key);
  assert.match(QC_TEXT.inspectTooltip, /^Moves the view to where mixing stands out/);
  assert.equal(copy('showStartTooltip', { start: "Xenium's cells" }), "Shows the outlines of Xenium's cells and moves the view to where mixing stands out.");
  // The legend: a short title and one row per shade, a few words each; no note line. The patches' key the same way.
  assert.deepEqual([QC_TEXT.legendTitle, QC_TEXT.legendLittle, QC_TEXT.legendSome, QC_TEXT.legendMixed, QC_TEXT.legendUnjudged],
    ['RNA from neighbours', 'Under 10 %', '10–30 %', '30 % or more (looks mixed)', 'Too few transcripts']);
  assert.equal(QC_TEXT.overviewTitle, 'Cells that look mixed, per patch');
  const patches = [copy('overviewLow', { a: '14' }), copy('overviewMid', { b: '21' }), copy('overviewHigh', { c: '28' })];
  assert.deepEqual(patches, ['Up to 14 in 100 (usual here)', 'About 21 in 100', '28 in 100 or more']);
  for (const text of [QC_TEXT.legendTitle, QC_TEXT.legendLittle, QC_TEXT.legendSome, QC_TEXT.legendMixed, QC_TEXT.legendUnjudged, QC_TEXT.overviewTitle, ...patches])
    assert.ok(text.split(' ').length <= 8, `a few words: ${text}`);
});

test('the test of a setting: its words as the spec has them, plain, no clash word, "hold" only in the hold\'s name, within their word limits', () => {
  // The answer's parts, the form's lines, the answers and the tooltips, word for word.
  const words = {
    testBackTitle: 'Do they come back?', testBackYes: 'Yes', testBackNo: 'No',
    testBackNot: 'These come and go between runs: proseg already moves them on its own. Nothing to judge.',
    testFewerTitle: 'Fewer than their own RNA?', testFewerTitleAlone: 'Fewer suspects?',
    testFewerYes: 'Clearly fewer', testFewerAloneYes: 'Clearly fewer (own RNA not separated)', testFewerNo: 'Not clear',
    testFewerYesShrink: 'Clearly fewer, mostly by shrinking the cell: their own RNA at the same spot dropped too',
    testFewerWhyYes: 'In every run with it, the suspects drop further than their own RNA at the same spot, beyond what identical runs give.',
    testFewerAloneWhyYes: 'Every run with it is below every run as now, by more than identical runs differ.',
    testFewerWhyApart: 'Not every run with it shows the suspects dropping further than their own RNA at the same spot.',
    testFewerAloneWhyApart: 'Not every run with it is below every run as now.',
    testFewerWhyGap: 'The difference is no wider than identical runs give.',
    testTooFewCell: "This cell's suspects weren't found in every run, so nothing was judged.",
    testNoControls: "Their own RNA at the same spot wasn't found in every run, so nothing was judged.",
    testCostTitle: 'What it cost', testCostOwn: 'Own RNA at the same spot', testCostAll: 'All their own RNA', testCostPerCell: 'Transcripts per cell here',
    testShrink: 'Mostly by shrinking the cell', testWentTitle: 'Where the suspects went',
    testWentStay: 'stay', testWentNeighbour: 'to the neighbour', testWentOther: 'another cell', testWentBackground: 'background',
    testAloneOne: 'This cell alone: 1 suspect: too few to judge one cell.',
    testCalibrationSettings: "Measured with proseg's default settings; this run has others.",
    testNewRun: 'New run with this', testKeep: 'Keep my setting', testRunAgain: 'Run again',
    testRepeatNote: 'Each test is a new chance of a false "clearly fewer"; a repeat is not a stronger answer.',
    testOpenTooltip: 'Opens a corner of the image where fresh runs of this area go to the end of the schedule, as now and with a value you choose (3 of each, or 4 for a start from cells only), to see whether the suspects of this cell and of nearby cells with the same pattern stay in their cells. About a minute; your run waits meanwhile.',
    // The How line (the runs line's tooltip): how the runs are made, never what they may show.
    testHow: "Every run goes afresh from your run's starting cells to the end of the schedule, all at once; those with the value change only this setting.",
    testPoolAsking: 'Finding nearby cells with the same pattern…',
    testPoolFailed: "Couldn't count the nearby cells with the same pattern. The test counts them again when it starts.",
    testFarDrift: 'Far drift', testFarDriftNote: 'Not tried here yet.',
    testFarDriftTip: "Of the drift proseg allows, the share with the far distance, 4 µm by default, from where a transcript was seen; the rest has the short drift distance, 1 µm by default. Proseg's default share is 0.2.",
    // Six more of proseg's own settings: their plain names, each not tried here yet, what each is in its tooltip.
    testNotTriedNote: 'Not tried here yet.', testGrid: 'Starting grid (µm)', testGaps: 'Background gaps', testDrift: 'Drift (on or off)',
    testNearDrift: 'Short drift distance (µm)', testCrowding: 'Crowding levels', testLayers: 'Depth layers', testSwitchOn: 'on', testSwitchOff: 'off',
    testDriftOnWord: 'drift on', testDriftOffWord: 'drift off', testChoose: 'choose one of the values shown.',
    testNearDriftOff: 'Short drift distance does nothing while drift is switched off in this run.',
    testFlat: 'This sample has no depth: proseg makes one layer whatever is set.',
    testRetry: "One run couldn't start, so every run started again without one start-up step.",
    // A press of Run while the engine's refusals are listed: one line (qc-panel.js testNotHere): the run named for a
    // reason about it or its sample, else the cell, else wait.
    testNotHereRun: "This run can't be tested.", testNotHereCant: "This cell can't be tested.", testNotHereFirst: 'Wait until it ends.',
    // A test that waits for what comes first, and its Cancel (only what holds: a run its press resumed is paused again,
    // the mixing check stays on); a reason the page has no words for, by its kind.
    testStatusWaiting: 'Testing a setting: waiting to start',
    testCancelWaitingTooltip: 'Stops the test before its runs start. Your run is paused again if the test resumed it; the mixing check stays on.',
    testReasonUnknown: "The test can't run here", testWaitUnknown: 'Starts as soon as it can', testCaveatUnknown: 'Not measured yet here',
  };
  // Nothing is refused for not being measured: no line says "nothing to change".
  assert.ok(!('testNotHereUnmeasured' in QC_TEXT));
  for (const [key, text] of Object.entries(words)) assert.equal(QC_TEXT[key], text, key);
  const filled = {
    testBack: [{ n: '187', p: '57', runs: '55 · 57 · 59' }, 'As now, these 187 suspects stay in their cells in 57 % of cases (runs 55 · 57 · 59 %).'],
    testFewerText: [{ v: '0.02', a: '57', b: '52', c: '61', d: '58' }, 'With 0.02: suspects 57 → 52 %, their own RNA at the same spot 61 → 58 %.'],
    testFewerTextAlone: [{ v: '0.02', a: '57', b: '52' }, 'With 0.02: suspects 57 → 52 %.'],
    testTooFew: [{ n: '121', k: '160' }, 'Too few suspects were left to judge once the runs were read: 121 of 160.'],
    testAlone: [{ n: '9' }, 'This cell alone: 9 suspects: too few to judge one cell.'],
    testCalibration: [{ fa: '1', fw: '3', areas: '3', drop: '25', from: '57', to: '43', pw: '7', sf: '1' },
      'About 1 in 100 tests of identical runs come out "clearly fewer" (3 in 100 in the worst of 3 areas). A drop of the suspects alone by 25 % (57 % → 43 %), their own RNA at the same spot unchanged, is caught about 7 times in 10. When the suspects and their own RNA drop alike, it still says "clearly fewer" about 1 in 100 times. Measured on one sample.'],
    testRepeat: [{ v: '0.02', word: 'clearly fewer' }, 'Earlier test with 0.02: clearly fewer.'],
    testRepeatNone: [{ v: '0.02' }, 'Earlier test with 0.02: nothing judged.'],
    // What a test needs: the pool line's tooltip, never in the line (a pool short of it is tested all the same).
    testPool: [{ c: '23', n: '187' }, 'Nearby cells with the same pattern: 23 cells, 187 suspects'],
    testPoolOne: [{ n: '9' }, 'Nearby cells with the same pattern: 1 cell, 9 suspects'],
    testPoolTooltip: [{ k: '160' }, 'The test was measured with 160 suspects or more. With fewer it runs all the same, and its answer says so.'],
    testAreaRegion: [{ w: '300', h: '300' }, 'Area: your whole region, 300 × 300 µm'],
    testAreaSquare: [{ w: '500', h: '500' }, 'Area: a 500 × 500 µm square around this cell'],
    testAreaCrop: [{ w: '500', h: '180' }, 'Area: 500 × 180 µm around this cell'],
    testRuns: [{ a: '3', b: '3', v: '0.02', t: '45 s' }, 'Your run waits · 3 runs as now, 3 with 0.02 · about 45 s'],
    testIteration: [{ i: '212', n: '450', phase: QC_TEXT.testPhaseRefining }, 'Iteration 212 of 450 · refining'],
  };
  for (const [key, [values, text]] of Object.entries(filled)) assert.equal(copy(key, values), text, key);
  assert.deepEqual([QC_TEXT.testPhaseSettling, QC_TEXT.testPhaseRefining, QC_TEXT.testPhaseFinal], ['settling', 'refining', 'final settling']);
  // "Nearby cells with the same pattern", verbatim, wherever the pool is named.
  for (const key of ['testPool', 'testPoolOne']) assert.match(QC_TEMPLATES[key], /^Nearby cells with the same pattern: /);
  for (const key of ['testPoolAsking', 'testPoolFailed', 'testOpenTooltip']) assert.match(QC_TEXT[key], /nearby cells with the same pattern/i);
  // Every word of the test: no banned stem, no clash word (each means something else in the app), "hold" only in the hold's
  // own name, no plain "compactness" (the setting's name stands beside a sentence, never in it), no outcome promised by the
  // form, within its word limit: a line in view at most 18 words, a sentence of the answer 22, a tooltip 75.
  const keys = Object.keys({ ...QC_TEXT, ...QC_TEMPLATES }).filter(k => k.startsWith('test'));
  const sample = { n: '187', p: '57', runs: '55 · 57', v: '0.02', a: '57', b: '52', c: '61', d: '58', k: '160', fa: '1', fw: '3', areas: '3', drop: '25',
    from: '57', to: '43', pw: '7', sf: '1', word: 'not clear', w: '500', h: '500', t: '45 s', i: '212', phase: 'refining', min: '0.1', max: '0.3', reason: 'x',
    label: 'stay', need: '160', name: 'Starting grid', old: '2 µm', new: '1 µm' };
  const text = key => Object.hasOwn(QC_TEXT, key) ? QC_TEXT[key] : copy(key, sample);
  const tooltips = ['testOpenTooltip', 'testFarDriftTip', 'testCalibration', 'testProgressTooltip', 'testNewRunTooltip', 'testKeepTooltip', 'testRunAgainTooltip',
    'testCancelTooltip', 'testCancelWaitingTooltip', 'testStatusTooltip', 'testHow', 'testPoolTooltip', 'testCalibrationShort', 'testAveraged', 'testAveragedSaved',
    'testHowHold', 'testHowHoldGiven', 'testHowHoldGivenAll', 'testHowHoldTried', 'testGridTip', 'testGapsTip', 'testDriftTip', 'testNearDriftTip',
    'testCrowdingTip', 'testLayersTip'];
  const answer = ['testBack', 'testBackNot', 'testFewerText', 'testFewerYesShrink', 'testFewerWhyYes', 'testFewerAloneWhyYes', 'testFewerWhyApart',
    'testFewerAloneWhyApart', 'testFewerWhyGap', 'testTooFew', 'testTooFewCell', 'testNoControls', 'testAlone', 'testAloneOne', 'testRepeat', 'testRepeatNote',
    'testPoolFailed', 'testRetry', 'testCalibrationSettings', 'testDriftOff', 'testNearDriftOff', 'testFlat', 'testNotHereRun', 'testNotHereCant', 'testNotHereFirst',
    'testFailed', 'testCancelled', 'testCellMissing'];
  for (const key of keys) {
    const t = text(key), n = t.split(/\s+/).length;
    assert.deepEqual(bannedWords(t), [], key);
    assert.doesNotMatch(t, /look-alike|\bsimilar\b|same kind|like for like/i, `${key}: no clash word`);
    // testHowHold says what the runs are given of the hold itself (P4): its own name there too.
    if (!['testHold', 'testHowHold'].includes(key)) assert.doesNotMatch(t, /\bhold\b/i, `${key}: "hold" only in the hold's own name`);
    assert.doesNotMatch(t, /\bcompactness\b/i, `${key}: no plain "compactness"`);
    assert.ok(n <= (tooltips.includes(key) ? 75 : answer.includes(key) ? 22 : 18), `${key}: ${n} words`);
  }
  // The form says nothing of what a setting may do: no "tighter", "looser", "seen to", "lower mixing" in its words.
  for (const key of ['testFarDriftNote', 'testNotTriedNote', 'testFarDriftTip', 'testGridTip', 'testGapsTip', 'testDriftTip', 'testNearDriftTip', 'testCrowdingTip',
    'testLayersTip', 'testPool', 'testPoolTooltip', 'testAreaRegion', 'testAreaSquare', 'testAreaCrop', 'testRuns', 'testHow', 'testHowHold', 'testOpenTooltip',
    'testNotHereRun', 'testNotHereCant', 'testNotHereFirst', 'testStatusWaiting', 'testCancelWaitingTooltip'])
    assert.doesNotMatch(text(key), /tighter|looser|seen to|lower(?:s|ed)? mixing|helps?\b|better|worse/i, key);
  // The How line says how the runs are made (from your run's starting cells, to the end of the schedule, all at once, one
  // setting changed), never an outcome: no "suspects", "fewer", "stay" or "mixing" in it.
  assert.match(QC_TEXT.testHow, /starting cells/); assert.match(QC_TEXT.testHow, /end of the schedule/);
  assert.doesNotMatch(QC_TEXT.testHow, /suspect|fewer|\bstay|mix/i);
  // P4: how firmly the runs hold, said after it, and the values they were given (in a test of the hold the runs as now's
  // and those with the value's, else every run's): exactly, never an outcome either.
  assert.equal(QC_TEXT.testHowHold, 'Proseg makes the hold stronger where cells hold more transcripts, so each run gets a value as strong as in your run. Only the value given to proseg changes.');
  assert.equal(copy('testHowHoldGiven', { a: '1.1615', v: '0.5' }), 'Here the runs as now are given 1.1615 for your 0.5.');
  assert.equal(copy('testHowHoldGivenAll', { a: '1.1615', v: '0.5' }), 'Here every run is given 1.1615 for your 0.5.');
  assert.equal(copy('testHowHoldTried', { b: '4.64602', t: '2' }), 'Those with the value are given 4.64602 for 2.');
  for (const key of ['testHowHold', 'testHowHoldGiven', 'testHowHoldGivenAll', 'testHowHoldTried']) assert.doesNotMatch(text(key), /suspect|fewer|\bstay|mix/i, key);
  // The How tooltip joins them: within a tooltip's 75 words however many of them it says (qc-test.test.mjs holds the
  // panel's own tooltips to it too).
  const how = [QC_TEXT.testHow, QC_TEXT.testHowHold, copy('testHowHoldGiven', { a: '1.16150', v: '0.5' }), copy('testHowHoldTried', { b: '4.64602', t: '2' })].join(' ');
  assert.ok(how.split(/\s+/).length <= 75, `the How tooltip of a test of the hold: ${how.split(/\s+/).length} words`);
  // The reasons' words (the engine's, the shared fixture's: qc-test.test.mjs compares them) pass the same rules.
  for (const [code, words] of Object.entries(TEST_REASON_WORDS)) {
    const t = testReasonText({ code, max_um: 1200, min_um: 300, have: 74, need: 160 });
    assert.deepEqual(bannedWords(t), [], code); assert.doesNotMatch(t, /look-alike|\bsimilar\b|same kind|like for like|\bhold\b|\bcompactness\b/i, code);
    assert.ok(t.split(/\s+/).length <= 18, code); assert.equal(words.includes('{'), ['tiled_box', 'region_small', 'too_few'].includes(code), code);
  }
  assert.equal(testReasonText({ code: 'unknown' }), "The test can't run here");
  assert.equal(testReasonText({ code: 'unknown' }, QC_TEXT.testCaveatUnknown), 'Not measured yet here');
});

// ---- proseg's own count ------------------------------------------------------------------------------------------------
test("every number about a cell's RNA is proseg's own count, and the words say so: the build's words, exactly", () => {
  // The words of the change to proseg's own count, word for word. The outline is never called a cell's count, and the
  // averaged count is never "expected" (proseg's own expected counts are another number).
  const text = {
    numberTooltip: "A cell looks mixed when the model reads 30 % or more of its RNA as coming from neighbouring cells. Only cells with 100 or more transcripts in proseg's count are judged.",
    startTooltip: "The start: proseg's count at iteration 0. The arrow compares like for like: the cells proseg started with, their RNA kept as proseg keeps it now, so a change shows what the boundaries and drift did.",
    tooFew: "Too few cells with 100 or more transcripts in proseg's count yet.",
    lowDepth: "Most cells here have fewer than 100 transcripts in proseg's count, so only a few can be judged.",
    nucleiCellsStart: "This run started from cells with their nuclei, and proseg's count at iteration 0 holds only the nuclei, so there is no starting number to compare with.",
    how5: "It reads proseg's own count of each cell: the transcripts proseg places in it, less those it reads as background.",
    ruleR3More: 'Drift lets proseg give RNA that drifted a little back to the cell it came from. It starts acting once cells have formed.',
    earlierTooltipPlain: "The proseg cells already saved in this sample, read on proseg's own count.",
    basisTooltip: "Proseg's count at this iteration, averaged over its random background draw. The saved table holds one draw of it, so its check can differ by a cell or two in 100.",
    retentionBasis: "Proseg's count now, against the cells proseg started with, kept the same way.",
    notRemeasured: "Its thresholds are not yet re-measured on proseg's count for runs like this one.",
    unavailableNoCounts: "This run was made before proseg's own count was kept. Open a new run to see clusters and the mixing check.",
    unavailableNoCountsTooltip: 'This run has no count from proseg to score: it was made before proseg-live kept its count. A new run has one.',
    unavailableOldProseg: 'Update proseg-live to see clusters and the mixing check.',
    fixBuild: 'build it again with native/build.sh', fixInstall: "install it as the README's Install section says",
    fixSetting: "point the binary under [proseg] in the app's config.toml at the new one",
    fixSettingNotebook: 'point the binary under [proseg] in ~/.config/sopa-companion/config.toml at the new one and launch again',
  };
  for (const [key, words] of Object.entries(text)) assert.equal(QC_TEXT[key], words, key);
  const templates = {
    earlier: "Proseg's cells saved in this sample: {b} in 100",
    earlierInfo: "Proseg's cells saved in this sample: {b} in 100.",
    earlierTooltip: "The proseg cells already saved in this sample ({table}), read on proseg's own count.",
    // {count}: which count it reads, by name (countAveraged, countSaved: the next test).
    basis: 'Based on {count}: {j} cells with 100 or more transcripts · typical cell {t} ({t0} at the start)',
    basisNoStart: 'Based on {count}: {j} cells with 100 or more transcripts · typical cell {t}',
    basisCoverage: 'Based on {count}: {j} of {n} cells (those with 100 or more transcripts)',
    basisCoverageFew: 'Based on {count}: only {j} of {n} cells (those with 100 or more transcripts), so this number is rough',
    unavailableOldProsegTooltip: "This proseg-live can't keep proseg's own count. To update it, {fix}.",
    snappedBadge: 'At iteration {i}',
    // Each pane's badge names its own result's iteration: the check's and the clusters' are lists of their own.
    snappedTooltipMixing: "The mixing check at iteration {i}: the nearest with proseg's count.",
    snappedTooltipClusters: "Clusters at iteration {i}: the nearest with proseg's count.",
  };
  for (const [key, words] of Object.entries(templates)) assert.equal(QC_TEMPLATES[key], words, key);
  // Unchanged: drift's tested why, the nuclei-only start and the rough confidence.
  assert.equal(QC_TEXT.nucleiStart, 'This run started from nuclei only, so there is no starting number to compare with.');
  assert.equal(QC_TEXT.rough, 'Rough guess');
  for (const [key, words] of Object.entries({ ...QC_TEXT, ...QC_TEMPLATES })) {
    assert.doesNotMatch(words, /expected (count|transcript|RNA)|counts? (?:every transcript )?inside (each|its|the) outline|this view counts/i, `${key}: ${words}`);
    // The averaged count is said in full wherever it is explained; its short name, "(averaged)" (the count's label in the
    // basis line: countAveraged), is explained by the tooltip beside it (basisTooltip, which says it in full), and the
    // saved count's words say it is "not averaged".
    if (/proseg's count/i.test(words) && /(?<!not )averaged/.test(words) && key !== 'countAveraged') assert.match(words, /averaged over its random background draw/, key);
  }
  assert.match(QC_TEXT.basisTooltip, /averaged over its random background draw/);
  // A run that ended with no start line says no start number beside its own.
  assert.equal(copy('finishedNoStart', { i: '450', n: '11' }), 'Finished at iteration 450: 11 in 100 cells looked mixed.');
  assert.equal(copy('stoppedNoStart', { i: '450', n: '11' }), 'Stopped at iteration 450: 11 in 100 cells looked mixed.');
});

test("the saved count, the smoothing and the start, said where they are shown: the round's words, exactly (decision A, H2, H3, H5, H10, H13, E2, E3)", () => {
  const text = {
    countAveraged: "proseg's count (averaged)", countSaved: "proseg's count (as saved)",
    basisTooltip: "Proseg's count at this iteration, averaged over its random background draw. The saved table holds one draw of it, so its check can differ by a cell or two in 100.",
    basisTooltipSaved: "Proseg's count at this iteration as proseg saves it: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.",
    changeSmoothed: 'While the run plays the change is smoothed like the number above.',
    changeNowSame: 'about the same as at the start', changeNowMore: 'more than at the start', changeNowFewer: 'fewer than at the start',
    likeForLikeSaved: "The change since the start reads proseg's count averaged over its random background draw on both sides, like for like, not the count as saved.",
    retentionSaved: "Both sides read proseg's count averaged over its random background draw, not the count as saved.",
    chartStartTooltip: "The dashed line: proseg's count at iteration 0.",
    chipCountAveraged: "Read on proseg's count averaged over its random background draw.",
    chipCountSaved: 'Read on the count proseg saves at this iteration.',
    // ... and with a change beside the number there, each part's count (F3).
    chipCountSavedChange: "The number reads the count proseg saves at this iteration; the change since the start reads proseg's count averaged over its random background draw, on both sides.",
    outlineEraTooltip: "From tests made while the mixing check counted the RNA inside the drawn outlines, before it read proseg's count; not measured again yet.",
    cardAveraged: "Its numbers read proseg's count averaged over its random background draw, smoothed as the run played, not the count as saved.",
    notRemeasuredShort: "Not yet re-measured on proseg's count for runs like this one.",
  };
  for (const [key, words] of Object.entries(text)) assert.equal(QC_TEXT[key], words, key);
  const templates = {
    numberLive: "From iteration {i}, the newest proseg has counted, smoothed with the iterations before it; its whole number changes only once it is 0.3 past the halfway point, so it doesn't flicker. Paused, it shows one iteration's own number.",
    numberPast: 'Iteration {i}, smoothed with the iterations before it, as it was shown live.',
    numberOwn: "Iteration {i}'s own number, not smoothed.",
    changeHeld: "This word is held a few seconds so it doesn't flicker; the change itself now reads {now}.",
    verdictSame: 'Same as at the start, like for like', verdictMore: '▲ {d} more, like for like', verdictFewer: '▼ {d} fewer, like for like',
    verdictMoreLight: '△ {d} more, like for like', verdictFewerLight: '▽ {d} fewer, like for like',
    start: "At the start (proseg's count at iteration 0): {s} in 100", startHeading: 'At the start: iteration 0', startReference: 'Iteration 0: {s}',
    cellCard: 'Mixing: the model reads about {p} % of its RNA as coming from neighbouring cells ({q} % at iteration 0).',
    cellCardFrom: 'Mixing at iteration {i}: the model read about {p} % of its RNA as coming from neighbouring cells ({q} % at iteration 0).',
    finished: 'Finished at iteration {i}: {n} in 100 cells looked mixed ({s} at iteration 0).',
    stoppedAt: 'Stopped at iteration {i}: {n} in 100 cells looked mixed ({s} at iteration 0).',
    ruleR1Body: 'These cells already looked mixed at iteration 0, and proseg has not made them more mixed.',
    chartHoverSmoothed: 'Iteration {i} · {n} in 100 cells look mixed, smoothed (of {j} with 100 or more transcripts)',
    chartHoverSaved: 'Iteration {i} · {n} in 100 cells look mixed, as saved (of {j} with 100 or more transcripts)',
    timelineHoverSmoothed: '{n} in 100 cells look mixed, smoothed', timelineHoverSaved: '{n} in 100 cells look mixed, as saved',
    chipSmoothed: 'From iteration {i}, smoothed while the run plays.',
    chipTooltipMoreLight: '{n} in 100 cells look mixed; {d} more than at the start, like for like. The gap is smaller than another learning of the model could give, so it holds for this model only. Click to open the mixing check.',
    chipTooltipFewerLight: '{n} in 100 cells look mixed; {d} fewer than at the start, like for like. The gap is smaller than another learning of the model could give, so it holds for this model only. Click to open the mixing check.',
    testAveraged: "The test judges the cell and picks its suspects on proseg's count averaged over its random background draw; the panel and the pink dots at iteration {i} show the count proseg saves there, so the two can differ a little.",
    testCalibrationShort: 'Tests with fewer than {need} suspects, like this one, were measured apart, on only {n} test cells of one sample: about {fa} in 100 tests of identical runs came out "clearly fewer" ({fw} in 100 at worst), and a drop of the suspects alone by {drop} % was caught about {pw} times in 10.',
  };
  for (const [key, words] of Object.entries(templates)) assert.equal(QC_TEMPLATES[key], words, key);
  for (const words of [...Object.values(text), ...Object.values(templates)]) {
    assert.deepEqual(bannedWords(words), [], words);
    assert.doesNotMatch(words, /\{start\}/, `${words}: the start is iteration 0's, never named after the cells it started from (H3)`);
  }
  // H3: no key of the start's words names the cells the run started from; only R1's button shows their outlines. The
  // verdict's own two numbers (P2: verdict…At) fill {start} with a number, the like-for-like start's per 100.
  const numbers = ['verdictFewerAt', 'verdictFewerLightAt', 'verdictMoreAt', 'verdictMoreLightAt', 'verdictSameAt'];
  assert.deepEqual(Object.entries(QC_TEMPLATES).filter(([, words]) => words.includes('{start}')).map(([key]) => key).sort(), ['showStartOutlines', 'showStartTooltip', ...numbers]);
  for (const key of numbers) assert.match(QC_TEMPLATES[key], /: \{now\} now, \{start\} at the start$/, key);
  // The words by count: a summary's count "sampled" (an iteration counted for a pause, the hold, a finish or the end)
  // reads "as saved", anything else "averaged"; the hover's words by how its point was made; a held word's clause.
  assert.equal(countWord({ count: 'sampled' }), QC_TEXT.countSaved);
  for (const point of [{ count: 'averaged' }, {}, null]) assert.equal(countWord(point), QC_TEXT.countAveraged);
  assert.deepEqual([{ single: false }, { single: true, count: 'sampled' }, { single: true }, {}].map(p => hoverKey(p)),
    ['chartHoverSmoothed', 'chartHoverSaved', 'chartHover', 'chartHover']);
  assert.equal(hoverKey({ single: false }, 'timelineHover'), 'timelineHoverSmoothed');
  assert.equal(heldClause({ held: true, direction: 'fewer', direction_now: 'same' }),
    "This word is held a few seconds so it doesn't flicker; the change itself now reads about the same as at the start.");
  assert.equal(heldClause({ held: true, direction: 'same', direction_now: 'more' }), copy('changeHeld', { now: QC_TEXT.changeNowMore }));
  assert.equal(heldClause({ held: true, direction: 'same', direction_now: 'fewer' }), copy('changeHeld', { now: QC_TEXT.changeNowFewer }));
  for (const change of [{ held: false, direction_now: 'same' }, { direction: 'fewer' }, { held: true }, { held: true, direction_now: 'x' }, null])
    assert.equal(heldClause(change), '', JSON.stringify(change));
  // H5: the shape cards' figures from the outline-era tests say so; drift's (R3) were measured on proseg's count.
  assert.deepEqual(Object.fromEntries(Object.entries(RULE_COPY).map(([rule, r]) => [rule, r.era])),
    { U: null, R7: null, R1: null, R2: 'outline', R3: 'proseg', R4: 'outline', R5: 'outline', R6: null });
});

test("a run without proseg's own count says why, an old proseg-live how to update it, in the words the check uses", () => {
  assert.equal(reasonText('unavailable', 'no_counts'), QC_TEXT.unavailableNoCounts);
  assert.equal(reasonText('unavailable', 'old_proseg'), QC_TEXT.unavailableOldProseg);
  for (const reason of ['no_counts', 'old_proseg']) {
    const words = reasonText('unavailable', reason);
    assert.deepEqual(bannedWords(words), [], reason); assert.doesNotMatch(words, /:|outline/, `${reason}: one plain line, no colon`);
  }
  // How to update it, by how the app finds the program (GET /health proseg_fix); none for a fix it has no words for.
  assert.deepEqual(PROSEG_FIXES, { build: 'fixBuild', install: 'fixInstall', setting: 'fixSetting' });
  assert.equal(oldProsegTooltip('build'), "This proseg-live can't keep proseg's own count. To update it, build it again with native/build.sh.");
  assert.equal(oldProsegTooltip('install'), "This proseg-live can't keep proseg's own count. To update it, install it as the README's Install section says.");
  assert.equal(oldProsegTooltip('setting'), "This proseg-live can't keep proseg's own count. To update it, point the binary under [proseg] in the app's config.toml at the new one.");
  // A notebook's session reads the user's own settings file at launch (analysis.js SETUP says the same of python).
  assert.equal(oldProsegTooltip('setting', true), "This proseg-live can't keep proseg's own count. To update it, point the binary under [proseg] in ~/.config/sopa-companion/config.toml at the new one and launch again.");
  assert.equal(oldProsegTooltip('install', true), oldProsegTooltip('install'));
  for (const fix of [null, undefined, '', 'toString', 'a new fix']) assert.equal(oldProsegTooltip(fix), '', String(fix));
  for (const fix of Object.keys(PROSEG_FIXES)) for (const notebook of [false, true]) assert.deepEqual(bannedWords(oldProsegTooltip(fix, notebook)), [], fix);
  // Every place the reason shows gives it its tooltip (unavailableTooltip): the panel's status line, the clusters' line
  // under Layers > Cells, Go deeper's window. Plural "counts" nowhere: proseg's count is one count.
  assert.equal(unavailableTooltip('no_counts'), QC_TEXT.unavailableNoCountsTooltip);
  assert.equal(unavailableTooltip('old_proseg', { prosegFix: 'build' }), oldProsegTooltip('build'));
  assert.equal(unavailableTooltip('old_proseg', { prosegFix: 'setting', notebook: true }), oldProsegTooltip('setting', true));
  for (const reason of ['too_few_cells', 'no_transcripts', 'software', undefined]) assert.equal(unavailableTooltip(reason, { prosegFix: 'build' }), '', String(reason));
  assert.deepEqual(bannedWords(QC_TEXT.unavailableNoCountsTooltip), []);
  for (const words of [QC_TEXT.unavailableNoCounts, QC_TEXT.unavailableNoCountsTooltip, TEST_REASON_WORDS.no_counts, oldProsegTooltip('build')])
    assert.doesNotMatch(words, /proseg's own counts/, words);
});

test("the test's reasons are the build's list, in its order and kinds: refusals of a run without proseg's count, its caveats, a glowing cell tested", () => {
  // The engine's REASON_WORDS / KIND (engine/qc_pool.py) and the shared fixture hold exactly this list (qc-test.test.mjs
  // compares the page with the fixture); a cell that does not look mixed but glows is tested with a caveat.
  const spec = [
    ['ending', 'refuse', 'This run is being saved'], ['ended', 'refuse', 'This run ended before its last iteration'],
    ['no_counts', 'refuse', "Made before proseg's own count was kept"], ['old_proseg', 'refuse', 'Update proseg-live to test a setting'],
    ['not_at_end', 'wait', 'Starts once your run reaches its last iteration'], ['past_end', 'caveat', 'Not measured yet for a run that sampled on past its end'],
    ['approx_targets', 'caveat', "Suspects placed approximately: proseg's exact shares at your run's end are no longer available"],
    ['targets_now', 'caveat', 'Suspects taken where your run is now, not at its end'],
    ['whole_sample', 'caveat', 'Not measured yet for runs of 20,000 cells or more'], ['tiled_box', 'caveat', 'Not measured yet for regions over {max_um} µm'],
    ['schedule', 'caveat', 'Not measured yet for a changed schedule'], ['nuclei_start', 'caveat', 'Not measured yet for runs started from nuclei only'],
    ['lasso', 'caveat', 'Not measured yet for lasso regions'], ['area_narrow', 'caveat', 'Not measured yet for regions over 500 µm long and under 500 µm wide'],
    ['square_not_measured', 'caveat', 'Not measured yet for cells-only starts on regions over 500 µm'],
    ['region_small', 'caveat', 'Not measured yet for regions under {min_um} µm'], ['check_unavailable', 'refuse', "The mixing check can't run on this sample"],
    ['check_off', 'wait', 'Switches the mixing check on first'], ['check_wait', 'wait', 'Starts once the mixing check has learned this sample'],
    ['check_scoring', 'wait', "Starts once the mixing check has scored your run's last iteration"],
    ['details', 'wait', "Starts once Go deeper's details are learned for this sample"], ['settings', 'caveat', 'Not measured yet with these mixing check settings'],
    ['count_basis', 'caveat', "Not measured yet with suspects from proseg's own count"], ['busy', 'refuse', 'A test of this run is on its way'],
    ['other_run', 'wait', 'Starts once the other run pauses or ends'], ['edge', 'caveat', "Not measured yet for cells this close to your region's edge"],
    ['not_judged', 'refuse', 'This cell has too few transcripts to be judged'], ['no_suspects', 'refuse', 'No suspects to test in this cell: none glows, or none could be placed'],
    // F1: refused only where nothing glows on either count; where its suspects on the averaged count cannot be tested
    // (none glows, or none can be tested) and its pink dots on the count proseg saves can, it is tested on those, with
    // this caveat.
    ['saved_suspects', 'caveat', "Suspects taken from proseg's count as saved: its averaged count gives none that can be tested"],
    ['low_share', 'caveat', "Doesn't look mixed overall: its pink dots are tested anyway"],
    ['weak_suspects', 'caveat', 'Its suspects are weak: none of its pink dots looks likely'],
    ['same_type', 'refuse', "The neighbours this cell's RNA seems to come from all look like its own kind"],
    ['pattern_none', 'caveat', 'Not measured yet for suspects that sit where no cell started'],
    ['pattern_neighbour', 'caveat', "Not measured yet for suspects that sit inside a neighbour's starting outline"],
    ['pattern_other', 'caveat', "Not measured yet for suspects that sit inside another cell's starting outline"],
    ['pattern_mixed', 'caveat', 'Not measured yet for suspects that sit in different places'],
    ['no_primary', 'refuse', "This cell's suspects all come from neighbours at the area's edge"],
    ['no_controls', 'refuse', 'No own RNA at the same spot to compare its suspects with'],
    // A pool short of k: the row's numbers are those of pools that reached it (no longer "not measured": the
    // calibration on proseg's count, 2026-10-31, measured a start from cells only's short pools, and weaker).
    ['too_few', 'caveat', 'Its numbers are for tests with {need} suspects or more: {have} here'],
  ];
  assert.deepEqual(Object.entries(TEST_REASON_WORDS), spec.map(([code, , words]) => [code, words]));
  assert.deepEqual(Object.entries(TEST_REASON_KINDS), spec.map(([code, kind]) => [code, kind]));
  // A run without proseg's count can't be tested whatever its cells: no cell of it.
  assert.deepEqual(TEST_RUN_REFUSALS, ['ending', 'ended', 'check_unavailable', 'no_counts', 'old_proseg']);
  assert.equal(testReasonText({ code: 'low_share', words: "the engine's own" }), "Doesn't look mixed overall: its pink dots are tested anyway");
});

test('waiting for room on the graphics card says what the panel needs and what is free', () => {
  // GET /samples/{id}/qc room (engine/qc.py _wait_for_gpu): learning waits for the room the sample's panel needs, and a
  // card smaller than that in all can never learn it (gpu_small); the words give both numbers, in GB of 1,024 MB.
  const room = { need_mb: 5000, free_mb: 2100, total_mb: 8192, genes: 5101 };
  assert.equal(reasonText('waiting', 'gpu_memory', room),
    'Waiting for room on the graphics card: learning this sample needs about 4.9 GB free (5,101 genes), and 2.1 GB is free now.');
  assert.equal(copy('failed', { reason: reasonText('failed', 'gpu_small', { ...room, total_mb: 4096 }) }),
    "Couldn't learn this sample: its 5,101 genes need about 4.9 GB free on the graphics card, which has 4.0 GB in all: it needs a graphics card with more memory.");
  // On a Mac (room.mac): its GPU uses the Mac's own memory, so the words name the Mac.
  assert.equal(copy('failed', { reason: reasonText('failed', 'gpu_small', { ...room, total_mb: 4096, mac: true }) }),
    "Couldn't learn this sample: its 5,101 genes and their counts need about 4.9 GB on this Mac's GPU, which may use 4.0 GB in all: it needs a Mac with more memory.");
  // Without the numbers each needs (the pause before a retry, an engine of an earlier version): the plain words.
  assert.equal(reasonText('waiting', 'gpu_memory'), QC_TEXT.waiting);
  assert.equal(reasonText('waiting', 'gpu_memory', { need_mb: 5000, genes: 5101 }), QC_TEXT.waiting);
  assert.equal(reasonText('failed', 'gpu_small'), QC_TEXT.failureSmallCard);
  assert.equal(FAILURE_REASONS.gpu_small, 'failureSmallCard');
  // The numbers belong to these two reasons alone.
  assert.equal(reasonText('waiting', 'lock', room), reasonText('waiting', 'lock'));
  assert.equal(reasonText('failed', 'oom', room), QC_TEXT.failureMemory);
  for (const text of [reasonText('waiting', 'gpu_memory', room), reasonText('failed', 'gpu_small', room), QC_TEXT.failureSmallCard]) {
    assert.deepEqual(bannedWords(text), [], text);
  }
});
