// Go deeper: every word of the per-cell window (the bar's labels are whole numbers, loose RNA reads "< 1 %" below 1 %
// so that its own and from neighbours add up to 100, and a gene chip's first number is rounded down so it never claims
// more than the estimate). On screen a number, a few-word verdict and labels; the explanations, caveats and method sit
// in tooltips (an info icon beside what they explain) or in the collapsed "How is this worked out?", four short points;
// the cell's count is proseg's own (its background in the basis's tooltip), the glow is of the transcripts proseg counts
// for the cell, the line of where the glowing transcripts sat at the start (places only, never an outcome; the start's
// hold in its tooltip), "like for like" kept on the step line. A count inside an outline is never called the cell's. No
// string names files outside the app. Templates use named values, never HTML. The check's banned-word rule (qc-copy.mjs
// BANNED_WORDS) holds for every string; "ResolVI" appears only in deepHow1. Strings the check already has (learning,
// waiting, the failure reasons, Try again, the starting cells' names, "Test a setting here…") are reused from
// qc-copy.mjs unchanged.
import {
  bannedWords, QC_TEXT, copy as qcCopy, number, duration, startName, cellShare, sharePercent, MIXED_CODE, SOME_CODE, NOT_JUDGED,
} from './qc-copy.mjs';

export { bannedWords };

export const DEEP_TEXT = Object.freeze({
  // ---- the mixing check's button and the window's frame
  deepButton: 'Go deeper ›',
  deepButtonOpen: 'Go deeper (open)',
  deepButtonTooltip: "Open a window that shows where this cell's RNA seems to come from: its own, its neighbours', or loose RNA in the tissue, and which genes are which. Key D.",
  // The cell's name's tooltip (no subtitle line).
  deepSubtitle: 'Where its RNA seems to come from (estimate)',
  deepFind: 'Show this cell on the stage',
  deepSmaller: 'Make smaller',
  deepBigger: 'Make bigger',
  deepClose: 'Close (Esc)',
  deepDrag: 'Drag to move · double-click to put it back',
  deepPinned: 'Last cell you clicked',
  deepPinnedTooltip: 'This window keeps the last cell you clicked; click another cell to look at it instead.',
  // ---- the headline and the bar
  // The card's number and meaning: the model's reading of the cell's RNA, from neighbouring cells of any kind, an
  // estimate with its blind spot both ways, in the label's tooltip.
  deepNumberLabel: 'from neighbours',
  deepNumberTooltip: "The share of its RNA the model reads as coming from neighbouring cells: an estimate on proseg's count, not a count of outline errors. Between cells of the same kind it can't tell RNA apart, so it can miss RNA from such a neighbour, or show some it can't back up.",
  deepMixed: 'Looks mixed',
  deepSome: 'Some mixing',
  deepLittle: 'Little or no mixing seen',
  deepStepsTooltip: "Looks mixed: the model reads 30 % or more of its RNA as coming from neighbours. Some mixing: 10–30 %. The same steps as the Mixing colours and the cell's line in the mixing check. The change since the start is like for like: this cell against itself at the start, both at the same number of transcripts.",
  deepSameAsStart: 'same as at the start',
  // The change since the start, like for like (the cell against itself at the start at the same depth; how, in the step
  // line's tooltip): a direction only, never the start's own number beside it, which compares other depths and can read
  // the other way ("8 %" beside "up from 10 %"); that number stays on the small line ("start 10 %").
  deepUpFromStart: 'more than at the start',
  deepDownFromStart: 'less than at the start',
  // ... said with the comparison's own two numbers on the step line (deepStepLikeAt): the cell's share from neighbours
  // now and at the start, both at the matched depth (the worker's start_change now and start), so the line says what it
  // compares where the headline (on the count proseg saves, at a pause, the hold, a finish or the end) and the small
  // line's start read other numbers; what the two are in the steps' tooltip.
  deepLikeMore: '▲ more',
  deepLikeLess: '▼ less',
  deepLikeSame: 'about the same',
  deepLikeNumbersTooltip: "Its two numbers: this cell's share from neighbours now and as proseg started it, both cut to the same number of transcripts. So they can differ from the number above and from the small line's start.",
  deepStartUnjudged: 'Too few transcripts at the start to compare',
  // Most of the RNA from neighbours looks like its own kind: the line under the headline; its tooltip says how much
  // looks like a different kind (lookAlikeLine).
  deepLookAlike: "Mostly its own kind: can't be checked gene by gene",
  deepLookAlikeAll: "All its own kind: can't be checked gene by gene",
  deepFloor: "The model's least: about 1 %",
  deepFloorTooltip: 'The model never says 0 %: about 1 % is the least it gives any cell.',
  deepFew: 'Few transcripts: single genes are rough',
  deepRough: 'Its gene-by-gene split is rough',
  deepRoughTooltip: 'For this cell the gene-by-gene split had to be stretched a lot to match the number above: treat it as rough.',
  deepOutsideTooltip: "Transcripts of genes the model doesn't know are left out of the split.",
  deepSparkLabel: 'This cell over the run: share from neighbours',
  deepOwn: 'Own',
  deepNeighbours: 'Neighbours',
  deepLoose: 'Loose',
  deepOwnTooltip: "RNA the model reads as this cell's own.",
  deepNeighboursTooltip: "RNA the model reads as coming from nearby cells: this cell's outline may reach into a neighbour, or a neighbour's RNA may have spread a little. Between cells of the same kind it can't tell RNA apart.",
  deepLooseTooltip: 'Loose RNA in the tissue: RNA floating free that lands a little on every cell, not from any one neighbour (often called ambient or background RNA).',
  // The basis's tooltip without the frame's cell statistics (its background not known here); on an answer read on the
  // count proseg saves at its iteration (a pause, the hold, a finish or the end), that count's own words.
  deepBasisNoBg: 'The transcripts proseg places in this cell, less those it reads as background, averaged over its random background draw. The saved table holds one draw of this count.',
  deepBasisNoBgSaved: 'The transcripts proseg places in this cell, less those it reads as background, as proseg saves them: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.',
  // The steps' tooltip on such an answer: the change since the start stays like for like on the averaged count.
  deepStartSaved: "Since the start: both sides read proseg's count averaged over its random background draw, like for like, not the count as saved.",
  // A tooltip's last words where the constants behind it were measured on other runs (tiled or whole-sample ones).
  deepNotRemeasured: "Not yet re-measured on proseg's count for runs like this one.",
  // ---- from neighbours: the neighbour genes (chips), the neighbours (rows), Show on the image
  deepSourcesNone: 'Too little from neighbours to list any',
  deepForeignLabel: 'Neighbour genes',
  deepForeignNone: "No gene stands out as a neighbour's.",
  deepSpread: 'The rest of the RNA from neighbours is spread thinly over genes this cell also makes.',
  deepSameKindShareTooltip: "Between cells of the same kind the model can't tell RNA apart, so that part of the RNA from neighbours can't be checked gene by gene.",
  deepSourcesHeading: 'Neighbours',
  deepSourcesGuide: 'Which neighbours it seems to come from: a rough guide.',
  deepSourcePctSmall: '< 1 %',
  // A row's Show: the button as the mockup draws it ([Show]); the words below are its tooltip and name.
  deepSourceShowShort: 'Show',
  deepSourceShownShort: 'Shown',
  deepSourceShow: 'Light up the transcripts that likely come from this neighbour',
  deepSourceShown: 'Lit up: press again for every neighbour',
  // A neighbour of the same kind: its Show explains instead (never greyed out); its share can't be checked.
  deepShowSameKind: 'Same kind as this cell: no gene tells its RNA apart',
  deepSourceSameKind: "same kind: can't be checked",
  deepSourceSameKindTooltip: "Same kind as this cell: no gene tells its RNA apart, so this share can't be checked.",
  deepSourceDeeper: 'Go deeper into this cell',
  deepSourcesSpread: 'Spread thinly: no neighbour stands out',
  // The rows' heading's tooltip with the guide: learned, or not learned from this cell (a rougher guide). The model
  // learned from the sample's own cell table, Xenium's or another's: never "Xenium's cells" on any sample.
  deepLearnedNote: "The model learned which neighbours each of the sample's cells borrows from; proseg keeps the same cells, so this carries that over and adjusts it for distance.",
  deepNotLearnedNote: "The model didn't learn from this cell, so its RNA from neighbours is shared between them by distance and by the genes each makes: a rougher guide.",
  deepGroupUnknown: 'group not known yet',
  deep3d: 'Arrows show in the 2-D view',
  deepZoom: 'Zoom in to see the arrows',
  // ---- its own, loose RNA, all genes
  deepOwnKindTooltip: 'Its own RNA looks like a cell that makes these genes.',
  deepOwnGenesTooltip: "The genes the model reads as mostly this cell's own here, with their transcripts.",
  deepOwnGenesNone: 'No gene is mostly its own here',
  deepOwnGenesNoneTooltip: 'Most genes here are shared with its neighbours.',
  deepOwnTopNone: 'No gene here is mostly its own: most are shared with its neighbours.',
  deepGroupsNote: 'Cell groups are named by their top genes (Clusters & UMAP); they are not cell types.',
  deepColGene: 'Gene',
  deepColTranscripts: 'Transcripts',
  deepColSplit: 'Own · neighbours · loose',
  deepColNeighbours: 'From neighbours',
  deepTagForeign: "a neighbour's gene",
  // The tag rule: the giving neighbour's kind makes the gene at least 4 times as much ("rarely makes"
  // misread for two states of one cell kind).
  deepTagForeignTooltip: "A neighbour of a different kind makes this gene at least 4 times as much as this cell's own kind, and most of its transcripts here seem to come from that neighbour.",
  deepTableNote: "Each gene's transcripts are shared out by how much of that gene each part usually makes, so the shares add up to the number above.",
  // ---- the footer and "How is this worked out?" (collapsed; deepHow1 is the only place naming the method)
  deepEstimate: 'An estimate, not a measurement',
  deepBlindSpot: 'Between cells of the same kind the model can\'t tell RNA apart: it can miss RNA from such a neighbour, or show some it can\'t back up. "Little from neighbours" doesn\'t mean "correct".',
  deepHowTitle: 'How is this worked out?',
  deepHow1: "A ResolVI model splits each cell's RNA into three parts: its own, from neighbours, and loose RNA in the tissue.",
  deepHow2: "Each gene's transcripts are shared out by how much of that gene each part usually makes: a gene only a neighbour's kind makes goes mostly to that neighbour.",
  deepHow3: "Which neighbour: learned for the starting cells, carried over as outlines change and adjusted for distance (a rougher guide for cells the model didn't learn).",
  deepHow4: "The glow places this split by position: a gene's transcripts close to the neighbour it seems to come from count as likelier. A rough guide.",
  // ---- states
  deepOff: "Go deeper uses the mixing check, which is off for this run. Switch it on to see where this cell's RNA seems to come from. The first time on a sample it learns that sample's cell types (a few minutes, once).",
  deepSwitchOn: 'Switch on the mixing check',
  deepLearningNote: "This happens once per sample. This window fills in by itself when it's done.",
  deepNeedsDetailNoTime: "This sample was learned before Go deeper existed, so the details it needs were not kept. Learning it again with them takes a few minutes, once; the mixing check's numbers stay the same.",
  deepLearnDetail: 'Learn the details',
  deepDetailMismatch: "the new learning didn't match the old one exactly, so nothing was changed",
  deepTooFewLater: 'This fills in by itself once proseg counts 100 or more for this cell.',
  deepTooFewLaterStart: 'Later iterations may hold more: look again once the run moves on.',
  deepEmpty: 'Proseg counts no transcripts for this cell in this iteration.',
  deepNotChecked: "This cell isn't checked in this run.",
  // A run paused on its prior: the run button says Start there, Step beside it (main.js renderRun).
  deepNucleiOnlyRun: 'This run started from nuclei only, so the first iteration has nothing to show yet: press Start or Step.',
  // Only a start from cells alone has a start line (proseg's count at iteration 0); cells with their nuclei start
  // from the nuclei (the panel's nucleiCellsStart), and a start line the engine refused for another reason says less.
  deepNucleiCellsRun: "This run started from cells with their nuclei, and proseg's count at iteration 0 holds only the nuclei, so the first iteration has nothing to show yet: press Start or Step.",
  deepNoStartRun: 'The first iteration has nothing to show for this run: press Start or Step.',
  // The same, naming no button, for every other run: one that ended, is saving its results or samples on by itself,
  // and one paused later with its first iteration scrubbed back to (Resume would leave the view there).
  deepNucleiOnlyRunEnded: 'This run started from nuclei only, so the first iteration has nothing to show.',
  deepNucleiCellsRunEnded: "This run started from cells with their nuclei, and proseg's count at iteration 0 holds only the nuclei, so the first iteration has nothing to show.",
  deepNoStartRunEnded: 'The first iteration has nothing to show for this run.',
  deepError: "Couldn't work out this cell.",
  deepRetrying: "Couldn't reach the mixing check. It tries again by itself.",
  // ---- "Show on the image" and "Where they sit": the
  // transcripts that likely come from neighbours glow (the model's per-gene split placed by position: a rough guide),
  // the neighbour genes' other transcripts as dots; the words per gene are first guesses.
  deepWhereShow: 'Show on the image',
  deepWhereShowTooltip: "Light up on the image the transcripts that likely come from neighbours (brighter: likelier), with these genes' other transcripts as dots. Every other cell fades. Press again to go back.",
  deepWhereHeading: 'Where they sit',
  deepWhereIntro: "The glow marks the transcripts that likely come from neighbours: the model splits each gene as a whole, and where each transcript lies (near the neighbour or not) shares that out, so it is a rough guide. The dots are all of these genes' transcripts where they lie.",
  deepWhereWorking: 'Finding them on the image…',
  deepWhereNone: "None of these genes' transcripts lie here.",
  deepWhereFailed: "Couldn't show them on the image for this iteration.",
  deepWhereWaiting: 'Waiting for this iteration to be saved…',
  deepWhere3d: "The genes' coloured dots show in the 2-D view.",
  deepWhereEdge: 'at the shared edge',
  deepWhereSpread: 'spread through the cell',
  deepWhereLayer: 'as common around',
  deepWhereFew: 'too few to tell',
  // The words per gene read proseg's count of the cell (each transcript's share for it, summed), placed by position.
  deepWhereEdgeTooltip: "Of what proseg counts for this cell, far more of this gene lies within 2 µm of this neighbour than the rest of its RNA does: it looks like RNA reaching in across the shared edge, which a better outline could leave out.",
  deepWhereSpreadTooltip: 'What proseg counts of it here is spread through the cell, not gathered at the edge it shares with this neighbour.',
  deepWhereLayerTooltip: "In proseg's count it is about as common in the cells around as in this cell: it looks like the tissue's own RNA at this spot, which no outline removes.",
  deepWhereFewTooltip: 'Proseg counts fewer than 4 of them for this cell: too few to tell where they sit.',
  deepWhereOther: 'From neighbours not listed',
  deepWhereUnder: 'under 1',
  // This cell's own genes inside a neighbour's drawn outline (the outline says only where): how many of them proseg
  // counts for this cell (its shares summed), against what it counts of the other transcripts inside that outline.
  deepWhereIntoTooltip: "Genes the model reads as this cell's own that lie inside this neighbour's outline, and how many of them proseg counts for this cell. Many counted here, close to this cell, mean the drawn outline reaches over RNA proseg gives this cell.",
  // The last words of the glow's, the genes' and the into lines' tooltips on a reply read on the count proseg saves at
  // its iteration (a pause, the hold, a finish or the end).
  deepWhereSaved: 'Read on the count proseg saves at this iteration: one random background draw, not averaged.',
  // The glow's line: who the glowing transcripts likely come from, every neighbour or one row's, and where they sat at
  // the start. The glow is of what proseg counts for the cell: some of it can lie just outside the drawn outline. The
  // dashed white line on the image is the cell's starting outline, said only where the stage draws it (in 2-D, for a
  // run that started from cells).
  deepGlowAll: 'neighbours',
  deepGlowTooltip: "Pink: proseg counts it for this cell, and it likely came from a neighbour. Brighter: likelier. Pink just outside the outline: proseg moved it in.",
  deepGlowStartOutline: "The dashed white line is this cell's starting outline.",
  // Added to the glow's tooltip when proseg's share of each transcript is placed from the frame's layers (approx), or
  // when no share is held here (none: a tiled frame's view window misses the cell, or the frame's counts are gone);
  // nothing glows then, and each gene in Where they sit stands alone, with no word or number, this as its tooltip.
  deepGlowApprox: "Approximate here: go to the newest iteration or pause for proseg's exact share.",
  deepGlowNone: "Proseg's share of these transcripts isn't available for this iteration: go to the newest iteration or pause to see pink dots here.",
  // The same once the run ended (done, stopped or failed) or while it saves its results (finishing): proseg is gone or
  // writing, so it can't be paused or asked for a share (and a whole-sample run's newest iteration may hold none here
  // either).
  deepGlowApproxEnded: "Approximate here: placed from this run's own layers, as a run that ended can't give proseg's exact share.",
  deepGlowNoneEnded: "Proseg's share of these transcripts isn't available for this iteration, and a run that ended can't be asked for it.",
  // The glow's line then: nothing was measured, so no verdict (deepGlowNone says why in its tooltip).
  deepGlowNoneShort: 'No pink dots for this iteration',
  // The glow's line when none of the pink dots lies near the neighbour it would name (that row's kind unknown).
  deepSourceNoOutlinePlain: 'Likely from a cell with no outline here',
  // The hold's own words (the line's tooltip, after the split by level): a start from cells only seeds every spot of
  // a starting cell as a nucleus seeds its own (zarr_bridge/helper.py prepare), so its whole cell is held at that level.
  deepStartHoldCells: 'A start from cells only holds the whole starting cell at the nucleus level.',
  deepGlowSameKind: 'Not lit: RNA from same-kind neighbours',
  deepGlowSameKindTooltip: "Between cells of the same kind the model can't tell RNA apart gene by gene, so that part of its RNA from neighbours has no transcripts to light up.",
  deepGlowUnplacedTooltip: "These genes seem to come from a neighbour that isn't listed, so their transcripts glow alike, at the model's share: where they lie isn't used.",
});

export const DEEP_TEMPLATES = Object.freeze({
  deepWindowLabel: 'Go deeper: {cell}',
  deepBack: 'Back to {cell}',
  deepLive: 'Iteration {i} · live',
  deepPaused: 'Paused · iteration {i}',
  deepNotNewest: 'Iteration {i} (not the newest)',
  deepAtStart: 'At the start ({start})',
  deepFinished: 'Finished · iteration {i}',
  deepStopped: 'Stopped · iteration {i}',
  deepEarlier: 'From iteration {j} · working out iteration {i}…',
  // The frame on screen has no proseg count of its own (between counted frames, or its file gone): the engine answers
  // it as the nearest counted frame, as the mixing check's snapped badge says.
  deepSnapped: "At iteration {i}: the nearest with proseg's count",
  deepNumber: '{p} %',
  deepStepChange: '{step} · {change}',
  // A change or "same as at the start" says it is like for like on the line itself (the start's raw share
  // beside the small line can read the other way, "start 1 %" beside "same").
  deepStepLike: '{step} · {change} · like for like',
  // ... and with its two numbers when the answer has them (startChange's `at`, whole percents); while its word is held
  // (start_change.held: the engine keeps a followed cell's word a few seconds), the same words without them, which are
  // then those the word compared, an earlier answer's than the headline's.
  deepStepLikeAt: '{step} · {change}, like for like: {now} % now, {start} % at the start',
  deepStepLikeHeld: '{step} · {change}, like for like',
  // ... and its tooltip says the word is held, with what this answer reads on its own (startChange's `own`: about the
  // same, more or less than at the start); never at a rest frame, whose word is its own.
  deepLikeHeld: "This word is held from earlier answers so it doesn't flicker; this answer alone reads {now}, like for like.",
  // Without the like-for-like comparison, the start's own number with no direction.
  deepStartWas: '{q} % at the start',
  deepLookAlikeTooltip: "Most of the RNA from neighbours looks like this cell's own kind: only about {q} % of its RNA looks like a different kind of cell, which genes can back up. The rest can't be checked.",
  deepLookAlikeAllTooltip: "All of the RNA from neighbours looks like this cell's own kind, so no gene backs it up: treat the {p} % as uncertain.",
  // The basis beside the small line: proseg's count of the cell, named by which count it is (averaged over its random
  // background draw, or as proseg saves it at that iteration: one draw); its tooltip says what it counts, with what
  // proseg reads as background in the frame on screen (the cell's statistics: assigned less kept).
  deepBasisShort: "Proseg's count (averaged): {n}",
  deepBasisShortSaved: "Proseg's count (as saved): {n}",
  deepBasis: 'The transcripts proseg places in this cell, less {bg} it reads as background, averaged over its random background draw. The saved table holds one draw of this count.',
  deepBasisSaved: 'The transcripts proseg places in this cell, less {bg} it reads as background, as proseg saves them: one random background draw, not averaged. A run that finishes at this iteration saves exactly this count in its table.',
  deepBasisStart: "At the start: {n}, proseg's count at iteration 0.",
  deepOutside: '{n} transcripts of unknown genes left out',
  deepFewTooltip: "With {n} transcripts, single genes hold only one or two each: treat any one gene's split as rough.",
  deepSparkHover: 'Iteration {i} · {p} % from neighbours',
  deepSparkStart: 'start {q} %',
  deepPart: '{part} {p} %',
  deepPartUnder: '{part} < 1 %',
  deepBarLabel: '{own} · {nb} · {loose}',
  // The neighbour genes' label's tooltip (the tagged genes).
  deepForeignHeading: "Genes far more typical of a neighbour's kind, so they probably come from neighbours ({k} here).",
  deepForeignChip: '{gene} {n} of {tx}',
  deepForeignMore: '+{k} more',
  deepSameKindShare: 'About {p} % looks like its own kind',
  // A row: line 1 its name and share, line 2 the genes it seems to give and its distance; the rest in its tooltip.
  deepSourceRow: '{cell} · {group}',
  deepSourceMakes: 'makes {genes}',
  deepSourceKind: 'Its own kind makes {genes}.',
  deepSourceDistance: '{d} µm',
  deepSourcePct: '{p} %',
  deepSourceTooltip: "{head}: about {p} % of this cell's RNA seems to come from here (a rough guide).",
  deepSourceTooltipSmall: "{head}: under 1 % of this cell's RNA seems to come from here (a rough guide).",
  deepSourceGene: '{gene} {n}',
  deepSourcesMore: '+{k} more · about {p} %',
  deepSourcesMoreSmall: '+{k} more under 1 %',
  deepPartlyNote: '{k} of these neighbours are new since the model learned this cell, so their shares are a rougher guide.',
  deepPartlyNoteOne: '1 of these neighbours is new since the model learned this cell, so its share is a rougher guide.',
  // In the Neighbours label's tooltip.
  deepSameKindCount: '{k} of its {n} nearest neighbours look the same kind.',
  deepSameKindCountOne: '1 of its {n} nearest neighbours looks the same kind.',
  deepGroupsFrom: 'Cell groups from iteration {i}',
  deepOwnKind: 'Own kind: {genes}',
  deepOwnGroup: 'In Clusters & UMAP it is in the group {group}.',
  deepOwnGenes: 'Top here: {genes}',
  // Its own top genes in the own kind line's tooltip, with their transcripts here.
  deepOwnTop: 'Its top genes here, mostly its own: {genes}.',
  deepOwnGene: '{gene} {n}',
  deepLooseLine: 'Loose RNA · {p} %',
  deepLooseBody: 'The model puts about {p} % here in every cell of this sample, spread almost evenly over all genes, so no genes are listed.',
  deepTableHeading: 'All {k} genes, split',
  // A very big cell's reply leaves out its genes with the fewest transcripts (the worker's 32 KB cap): said, not hidden,
  // and never claimed to hold one transcript each.
  deepTableRest: 'And {k} more genes holding {tx} transcripts in all, not listed one by one.',
  deepRowTooltip: '{gene}: {tx} transcripts · about {own} its own, {nb} from neighbours{from} (estimate)',
  deepRowFrom: ' (most from {cell})',
  deepRowPercent: '{p} %',
  // The last line of "How is this worked out?": what the sample's model learned from, and when.
  deepLearnedFrom: 'Learned from {cells} cells on {date}, in {t}.',
  deepLearnedFromNoTime: 'Learned from {cells} cells on {date}.',
  deepNeedsDetail: "This sample was learned before Go deeper existed, so the details it needs were not kept. Learning it again with them takes about {t}, once; the mixing check's numbers stay the same.",
  deepDetailLearning: 'Learning the details for this sample… {k} of {of} · about {t} left',
  deepDetailLearningNoTime: 'Learning the details for this sample… {k} of {of}',
  deepDetailFailed: "Couldn't add the details: {reason}.",
  deepWorking: 'Working out {cell}…',
  // {n}: the cell's count in the genes the model knows (the worker's transcripts: the split's own, which the 100 is of),
  // so a cell's readout, of all its genes, may read more.
  deepTooFew: 'Too few transcripts to split: proseg counts {n} for this cell in this iteration, in the genes the model knows, and at least 100 are needed.',
  // The start line's own count: proseg's at iteration 0 (only a start from cells alone has one).
  deepTooFewStart: 'Too few transcripts to split: proseg counts {n} for this cell at iteration 0, in the genes the model knows, and at least 100 are needed.',
  deepTooFewGenes: 'Its genes so far (not split): {genes}',
  // Iteration 0: the cells are the starting ones, and proseg's count of them already sets background aside.
  deepNuclei: "At the first iteration proseg's cells are those {start} drew, and its count already sets part of their RNA aside as background.",
  deepNucleiShort: 'Cells as {start} drew them',
  deepCantRun: "Go deeper can't run for this run: {reason}",
  deepArrowLabel: '{p} %',
  deepArrowTooltip: "{cell} · {group}: about {p} % of {target}'s RNA seems to come from here (estimate)",
  deepSameKindTooltip: "{cell} looks like the same kind as {target}: RNA from it can't be told apart.",
  // "Where they sit": a gene and its word on screen; its numbers in the line's tooltip, all on proseg's count: how much
  // of it in this cell lies within 2 µm of the neighbour against the rest of this cell's RNA (the baseline: most cells
  // hold many transcripts near their edge), its density here, in the neighbour and in the cells around (each per 100 µm²
  // of its own footprint); per neighbour, how many of this cell's own genes inside the neighbour's outline proseg counts
  // for this cell, against what it counts of the other transcripts there.
  deepWhereGene: '{gene} · {word}',
  deepWhereNear: "{n} of the {of} proseg counts for this cell lie within 2 µm of it (the rest of this cell's RNA: {m} of {mof}).",
  deepWhereNearAlone: '{n} of the {of} proseg counts for this cell lie within 2 µm of it.',
  deepWhereFewIn: 'Proseg counts {n} for this cell.',
  // "Too few to tell" for a gene proseg counts 4 or more of when its shares are placed approximately and fewer than 4
  // of them are placed here (the words are judged on what is placed).
  deepWhereFewPlaced: 'Proseg counts {n} for this cell, but fewer than 4 of them could be placed in this view: too few to tell where they sit.',
  deepWhereDensity: "Per 100 µm², in proseg's count: {here} in this cell · {giver} in it · {around} in the cells around.",
  deepWhereDensityAround: "Per 100 µm², in proseg's count: {here} in this cell · {around} in the cells around.",
  deepWhereIntoLine: "Inside the neighbour's outline, proseg counts for this cell: {genes}",
  deepWhereInto: '{gene}: {n}, {near} of them within 2 µm of this cell.',
  deepWhereIntoBase: "Other transcripts inside the neighbour's outline: proseg counts {of} of them for this cell, {n} within 2 µm of this cell.",
  // The dots kept from the iteration proseg counted before the one on screen while this one's shares would only be
  // approximate (following a cell as the run plays on).
  deepWhereSnapped: 'Dots at iteration {i}',
  deepWhereSnappedTooltip: "Proseg's exact shares at iteration {i}, the iteration it counted before this one: here this iteration's shares would only be approximate.",
  deepWhereTruncated: 'Only {n} transcripts are drawn: there are more in this area.',
  // The glow's line (startLine): "12 likely from neighbours · 9 inside its own starting outline · 3 where no cell
  // started" (or from one row's neighbour), "maybe" while none reaches 1 in 2 (every glowing one counted then); the
  // places are the engine's own words for the same classes (the test's reasons pattern_none, pattern_neighbour,
  // pattern_other: engine/qc_pool.py REASON_WORDS), in a fixed order, each only when some sat there; the genes whose
  // neighbour is not listed glow at the model's share alone.
  deepStartLikely: '{n} likely from {who}',
  deepStartMaybe: '{n} maybe from {who}',
  deepStartOwn: '{n} inside its own starting outline',
  deepStartNone: '{n} where no cell started',
  deepStartGiver: "{n} inside a neighbour's starting outline",
  deepStartOther: "{n} inside another cell's starting outline",
  // Not in the start line's reply (outside the box it drew around the cell and its neighbours, or cut short).
  deepStartUnknown: '{n} unknown at the start',
  deepGlowNoneFrom: 'None looks likely from {who}',
  deepGlowUnplaced: 'No edge measured for {genes}',
  // The glow's line when none of its pink dots lies near the neighbour it would name (that row: {kind} its own kind's
  // genes, {cell} its name in the tooltip): their source is likely a cell the outlines miss.
  deepSourceNoOutline: 'Likely from a {kind} cell with no outline here',
  deepSourceNoOutlineTooltip: 'None of these pink dots lies near {cell}: they more likely come from a cell of its kind that has no outline here.',
  // The line's tooltip, after the glow's: how many of the likely ones lie at the shared edge now (the reply's near flags),
  // then the start's hold of those inside its own starting outline, by level (a detail run's start layers: PriorLayers
  // startHold), the ones whose level could not be told said so.
  deepStartNear: 'At the shared edge, within 2 µm of their neighbour: {n} of the likely ones.',
  deepStartHold: 'Inside its own starting outline, the start held {nucleus} at the nucleus level and {cell} at the cell level.',
  deepStartHoldSome: 'Inside its own starting outline, the start held {nucleus} at the nucleus level and {cell} at the cell level; for {unknown} it could not be told.',
});

/** A string of the deck: a plain one, or a template filled with `values` (a missing value throws: braces or
 *  "undefined" never reach the page). */
export function deep(key, values = {}) {
  if (Object.hasOwn(DEEP_TEXT, key)) return DEEP_TEXT[key];
  if (!Object.hasOwn(DEEP_TEMPLATES, key)) throw new RangeError(`Unknown Go deeper copy key: ${key}`);
  return DEEP_TEMPLATES[key].replace(/\{([^}]+)\}/g, (_, name) => {
    if (!Object.hasOwn(values, name) || values[name] == null) throw new TypeError(`Missing Go deeper copy value: ${key}.${name}`);
    return String(values[name]);
  });
}

// ---- numbers as the window shows them -----------------------------------------------------------------------------
const finite = Number.isFinite;
const clamp01 = v => Math.max(0, Math.min(1, v));
/** Whole transcripts (en-GB thousands). */
export const count = n => number(Math.round(finite(n) ? n : 0));
/** A count said beside a floor it must reach (`floor`: "at least 100", "fewer than 4"): rounded once like every count,
 *  but one under the floor that would round up to it reads its first decimal, rounded down (99.7 of "at least 100",
 *  3.6 of "fewer than 4"), so the number never contradicts the floor's words. */
export function underFloor(n, floor) {
  return finite(n) && n < floor && Math.round(n) >= floor ? number(Math.floor(n * 10) / 10, 1) : count(n);
}

/** The headline's percentage for a cell's byte (floor(share x 250); 255 = not judged): exactly the number on the cell's
 *  line in the mixing check (qc-copy.mjs cardLine uses the same sharePercent(cellShare(code))), so the window always
 *  equals that line and the Mixing colours. Null for a byte that is not judged. */
export function headlinePercent(code) {
  const share = cellShare(code);
  return share == null ? null : sharePercent(share);
}
/** The headline's step word, on the steps of the colours and the cell's line (codes 75 and 25: 30 % and 10 %). */
export function stepKey(code) {
  return !Number.isInteger(code) || code >= NOT_JUDGED ? null : code >= MIXED_CODE ? 'deepMixed' : code >= SOME_CODE ? 'deepSome' : 'deepLittle';
}
/** The worker's DEEP_START_BAND (analysis/worker.py, measured on proseg's count): the cell against itself at the start
 *  reads "about the same" within this many points where a reply's start_change has no band. qc-deeper.js normalise fills
 *  a missing band with it, and startChange takes it for a band that is not above 0; engine/qc_scoring.py START_BAND is
 *  its twin. */
export const START_BAND = 5;
/** The change since the start: more, less or the same by the worker's like-for-like comparison (`like` = the result's
 *  start_change {points, band, direction, ...}: the cell against itself at the start at the same depth; more or less
 *  only past its band, START_BAND points where the reply has none: a cell that did not change then says either in about
 *  one answer in ten; the worker's own word wins, kept near the band as last shown so it does not flip from frame to
 *  frame), said with no number (the start byte's own percent compares other depths); else the start's own number
 *  (`q`: the start byte's percent) with no direction (a raw byte against the start's compares different depths); `unjudged`
 *  when the start had too few transcripts (255); null when the start is not known. `like`: the change is the worker's
 *  like-for-like one (the step line says so). With the comparison's own two numbers (`like.now`, `like.start`: the
 *  cell's share from neighbours now and at the start in percent, both at the matched depth) the change is said short
 *  ("▼ less") and `at` holds them as whole percents for the step line (deepStepLikeAt), but for a word held from earlier
 *  answers (`like.held`: the worker's word kept near the band, or the engine's kept a few seconds for a followed cell:
 *  `held` true, no `at`, deepStepLikeHeld: the engine's numbers are then those the word compared, an earlier answer's
 *  than the headline's); an answer without them (kept from before them) says it as before. A held word, with the
 *  numbers or without, carries `own`: this answer's own word on its points ("more", "fewer", "same": the step's
 *  tooltip says the word is held and what this answer alone reads, deepLikeHeld). At a rest frame (`rest`: an answer
 *  on the count proseg saves, at a pause, the hold, a finish or the end) the word is always this iteration's own, on
 *  its own points (never one held from earlier answers), with its two numbers only when they are the ones its points
 *  compare (ownNumbers: else the long words, no numbers), as the panel shows a resting run's own change. */
export function startChange(code, startCode, like = null, { rest = false } = {}) {
  const now = cellShare(code);
  if (now == null || !Number.isInteger(startCode) || startCode < 0) return null;
  if (startCode >= NOT_JUDGED) return { unjudged: true };
  const values = { q: number(Math.floor(cellShare(startCode) * 100 + 1e-9)) };
  if (!like || !finite(like.points)) return { key: 'deepStartWas', text: deep('deepStartWas', values), like: false };
  const band = finite(like.band) && like.band > 0 ? like.band : START_BAND;
  // This answer's own word: its points against its band (the worker's rule, nothing kept from an earlier answer).
  const own = Math.abs(like.points) < band ? 'same' : like.points > 0 ? 'more' : 'fewer';
  const word = rest ? own : ['more', 'fewer', 'same'].includes(like.direction) ? like.direction : own;
  const held = !rest && like.held === true;
  if (finite(like.now) && finite(like.start) && (!rest || ownNumbers(like))) {
    const key = word === 'same' ? 'deepLikeSame' : word === 'more' ? 'deepLikeMore' : 'deepLikeLess';
    if (held) return { key, text: deep(key), like: true, held: true, own };
    return { key, text: deep(key), like: true, at: { now: number(Math.round(like.now)), start: number(Math.round(like.start)) } };
  }
  const key = word === 'same' ? 'deepSameAsStart' : word === 'more' ? 'deepUpFromStart' : 'deepDownFromStart';
  return held ? { key, text: deep(key), like: true, own } : { key, text: deep(key), like: true };
}
/** Whether a start_change's two numbers are its own: the ones its points compare (now - start = points, each sent to 2
 *  decimals: within 0.02). A word the engine held for a followed cell carries an earlier answer's (those it compared),
 *  which a rest frame's own word never stands beside. */
export const ownNumbers = like => finite(like?.now) && finite(like?.start) && finite(like?.points)
  && Math.abs(like.now - like.start - like.points) <= 0.02;
/** The words of a word's direction for deepLikeHeld's {now}: what an answer alone reads while the word shown is held. */
export const LIKE_NOW = Object.freeze({ same: 'deepLikeSame', more: 'deepUpFromStart', fewer: 'deepDownFromStart' });
/** Under the headline when most of the RNA from neighbours looks like the cell's own kind (same_kind_share >= 0.5):
 *  {text, tip}, the tip saying how much of all its RNA looks like a different kind (rounded down: q = 100 x share x (1 -
 *  same_kind_share)), or that nothing backs the headline up; null otherwise. */
export function lookAlikeLine(code, sameShare) {
  const share = cellShare(code);
  if (share == null || !finite(sameShare) || sameShare < 0.5) return null;
  const q = Math.floor(share * (1 - clamp01(sameShare)) * 100 + 1e-9);
  return q >= 1 ? { text: DEEP_TEXT.deepLookAlike, tip: deep('deepLookAlikeTooltip', { q: number(q) }) }
    : { text: DEEP_TEXT.deepLookAlikeAll, tip: deep('deepLookAlikeAllTooltip', { p: headlinePercent(code) }) };
}
/** The bar's three labels (whole numbers that add up to 100): from neighbours = the headline's number, loose RNA "< 1 %"
 *  below 1 % (else its whole number), its own = 100 minus the other two. `parts` [own, neighbours, loose] (the check's
 *  split) gives the widths. Returns {own, nb, loose, looseUnder, texts: {own, nb, loose}, widths: [3 shares]}. */
export function barLabels(code, parts) {
  const nbText = headlinePercent(code);
  const nb = nbText == null ? null : Math.floor(cellShare(code) * 100 + 1e-9);
  const p = Array.isArray(parts) && parts.length === 3 && parts.every(finite) ? parts.map(v => Math.max(0, v)) : [1 - (cellShare(code) ?? 0), cellShare(code) ?? 0, 0];
  const total = p[0] + p[1] + p[2] || 1, widths = p.map(v => v / total);
  const looseRaw = widths[2] * 100, looseUnder = looseRaw < 1;
  const loose = looseUnder ? 0 : Math.max(0, Math.min(Math.round(looseRaw), nb == null ? 100 : 100 - nb));
  const own = nb == null ? null : 100 - nb - loose;
  const texts = nb == null ? null : {
    own: deep('deepPart', { part: DEEP_TEXT.deepOwn, p: number(own) }),
    nb: deep('deepPart', { part: DEEP_TEXT.deepNeighbours, p: number(nb) }),
    loose: looseUnder ? deep('deepPartUnder', { part: DEEP_TEXT.deepLoose }) : deep('deepPart', { part: DEEP_TEXT.deepLoose, p: number(loose) }),
  };
  return { own, nb, loose, looseUnder, texts, widths };
}
/** A tagged gene as a chip: its transcripts from neighbours rounded DOWN (a chip never claims more than the estimate)
 *  "of" its transcripts here. */
export function chipText(gene, nb, tx) {
  return deep('deepForeignChip', { gene, n: number(Math.max(0, Math.floor((finite(nb) ? nb : 0) + 1e-6))), tx: count(tx) });
}
/** Whole numbers for `values` that add up to `total` (largest remainder); `total` defaults to their rounded sum. */
export function largestRemainder(values, total = null) {
  const v = values.map(x => (finite(x) && x > 0 ? x : 0)), sum = v.reduce((a, b) => a + b, 0);
  const target = Number.isInteger(total) ? total : Math.round(sum);
  if (!(sum > 0) || target <= 0) return v.map(() => 0);
  const scaled = v.map(x => x * target / sum), floors = scaled.map(Math.floor);
  let left = target - floors.reduce((a, b) => a + b, 0);
  const order = scaled.map((x, i) => [x - floors[i], i]).sort((a, b) => b[0] - a[0] || a[1] - b[1]);
  for (const [, i] of order) { if (left <= 0) break; floors[i]++; left--; }
  return floors;
}
/** A gene's row tooltip: whole numbers that add up to its transcripts (largest remainder over its own, from neighbours
 *  and loose), and the neighbour that gives most of it when one is known. */
export function rowTooltip(gene, tx, own, nb, fromName = null) {
  const total = Math.round(finite(tx) ? tx : 0), loose = Math.max(0, (finite(tx) ? tx : 0) - (own || 0) - (nb || 0));
  const [o, n] = largestRemainder([own, nb, loose], total);
  return deep('deepRowTooltip', { gene, tx: number(total), own: number(o), nb: number(n), from: fromName ? deep('deepRowFrom', { cell: fromName }) : '' });
}
/** A part of the headline as a whole percent, rounded DOWN and never above `cap` (the headline's own whole number):
 *  rows, folds and arrow labels never add up to more than the headline (one neighbour's "about 34 %" once stood under
 *  a 33 % headline). */
export function partPercent(share, cap = null) {
  const p = Math.floor(clamp01(finite(share) ? share : 0) * 100 + 1e-9);
  return finite(cap) ? Math.min(p, Math.max(0, cap)) : p;
}
/** A row's share as its line says it: "43 %", or "< 1 %" (partPercent). */
export function sharePct(share, cap = null) {
  const p = partPercent(share, cap);
  return p < 1 ? DEEP_TEXT.deepSourcePctSmall : deep('deepSourcePct', { p: number(p) });
}
/** The folded neighbours' line: "+k more · about p %", or "+k more under 1 %" while they hold less than 1 % of the
 *  cell's RNA between them (partPercent). */
export function foldText(k, share, cap = null) {
  const p = partPercent(share, cap);
  return p < 1 ? deep('deepSourcesMoreSmall', { k: number(k) }) : deep('deepSourcesMore', { k: number(k), p: number(p) });
}
/** The same-kind count, a sentence for the Neighbours label's tooltip: "k of its n nearest neighbours look the same
 *  kind." (null for none). */
export function sameKindCountText(k, n) {
  if (!(k > 0)) return null;
  return k === 1 ? deep('deepSameKindCountOne', { n: number(n) }) : deep('deepSameKindCount', { k: number(k), n: number(n) });
}
/** Where the glowing transcripts sat at the start, in the line's order: [the class (qc-deeper.js PLACES), its words]. */
export const START_PARTS = Object.freeze([['own', 'deepStartOwn'], ['none', 'deepStartNone'], ['giver', 'deepStartGiver'],
  ['other', 'deepStartOther'], ['unknown', 'deepStartUnknown']]);
/** Go deeper's line for the glow `glow` ({count, likely, places: {own, none, giver, other, unknown} | null}, as
 *  glowPoints counts them) of `who` (every neighbour: deepGlowAll, or one row's name): how many likely come from there
 *  ("maybe" while none reaches 1 in 2: every glowing one counted then), then where those it counts sat at the start, a
 *  part per place that holds some, in START_PARTS' order ("12 likely from neighbours · 9 inside its own starting outline
 *  · 3 where no cell started"); the head alone while the places are not known; "None looks likely from …" for no glow.
 *  Facts about places only, never an outcome. */
export function startLine(glow, who) {
  if (!(glow?.count > 0)) return deep('deepGlowNoneFrom', { who });
  const head = glow.likely > 0 ? deep('deepStartLikely', { n: count(glow.likely), who }) : deep('deepStartMaybe', { n: count(glow.count), who });
  return [head, ...placeParts(glow.places)].join(' · ');
}
/** The places part of the line: a part per place that holds some, in START_PARTS' order (none while not known). */
const placeParts = p => (p ? START_PARTS.filter(([k]) => p[k] > 0).map(([k, key]) => deep(key, { n: count(p[k]) })) : []);
/** "Likely from a {kind} cell with no outline here" for the source row none of the glow lies near (`row`: {kind}, its
 *  own kind's first three genes; without them the plain words): the head of the line that says so, and of the genes it
 *  seems to give in "Where they sit". */
export function noOutlineHead(row) {
  const kind = genesList((row?.kind ?? []).slice(0, 3), ' · ');
  return kind ? deep('deepSourceNoOutline', { kind }) : DEEP_TEXT.deepSourceNoOutlinePlain;
}
/** The line instead when none of the glowing transcripts lies near the neighbour it would name (`row`: that source
 *  row, {name, kind}): noOutlineHead, then where they sat at the start as startLine says it; {text, tip}, the tip
 *  naming that neighbour before the glow's own tooltip (startTip's, `outline` as there). */
export function noOutlineLine(glow, row, outline = false) {
  return { text: [noOutlineHead(row), ...placeParts(glow?.places)].join(' · '),
    tip: [deep('deepSourceNoOutlineTooltip', { cell: row?.name ?? '' }), startTip(glow, outline)].join(' ') };
}
/** What the glow is (deepGlowTooltip), with `outline` (the stage draws the cell's starting outline dashed: 2-D, a run
 *  from cells) that the dashed line is it. */
export function glowTip(outline = false) { return outline ? `${DEEP_TEXT.deepGlowTooltip} ${DEEP_TEXT.deepGlowStartOutline}` : DEEP_TEXT.deepGlowTooltip; }
/** The line's tooltip: what the glow is (glowTip, `outline` as there), how many of the likely ones lie at the shared
 *  edge now, and, once the start's levels are known (a detail run: `glow.levels` {nucleus, cell, unknown} over those
 *  inside its own starting outline), how the start held those (the hold split), with what a start from cells only means
 *  for it. */
export function startTip(glow, outline = false) {
  const out = [glowTip(outline)];
  if (glow?.likely > 0 && glow.near > 0) out.push(deep('deepStartNear', { n: count(glow.near) }));
  const own = glow?.places?.own, l = glow?.levels;
  if (own > 0 && l && l.nucleus + l.cell > 0) {
    const values = { nucleus: count(l.nucleus), cell: count(l.cell) };
    out.push(l.unknown > 0 ? deep('deepStartHoldSome', { ...values, unknown: count(l.unknown) }) : deep('deepStartHold', values),
      DEEP_TEXT.deepStartHoldCells);
  }
  return out.join(' ');
}
/** The partly-learned note for `k` shown rows that are pairs the model never saw (null for none). */
export function partlyText(k) { return k > 0 ? (k === 1 ? deep('deepPartlyNoteOne', {}) : deep('deepPartlyNote', { k: number(k) })) : null; }
/** Genes as the window lists them: "A, B, C" (comma) or "A · B · C" (dot). */
export const genesList = (genes, sep = ', ') => (genes || []).filter(g => typeof g === 'string' && g).join(sep);

/** A date as the page writes it ("21 Oct 2026"); null when it is not one. */
export function dateText(value) {
  if (value == null || value === '') return null;
  const d = new Date(value);
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' });
}
/** The provenance line: how many cells the sample's model learned from, when and how long it took (`model`:
 *  the check's sampleModel {cells, created, seconds}); null when it does not say. */
export function learnedLine(model) {
  const cells = model?.cells, date = dateText(model?.created);
  if (!finite(cells) || cells <= 0 || !date) return null;
  return finite(model.seconds) && model.seconds > 0 ? deep('deepLearnedFrom', { cells: number(cells), date, t: duration(model.seconds) })
    : deep('deepLearnedFromNoTime', { cells: number(cells), date });
}
/** Every item of "How is this worked out?", in order: four points, then the provenance line when the model says it. */
export function howItems(model = null) {
  const line = learnedLine(model);
  return [DEEP_TEXT.deepHow1, DEEP_TEXT.deepHow2, DEEP_TEXT.deepHow3, DEEP_TEXT.deepHow4, ...(line ? [line] : [])];
}
/** The header's frame tag for the result shown (section 7.3 A). `frame`: {kind: 'start' | 'live' | 'paused' | 'older' |
 *  'finished' | 'stopped' | 'earlier' | 'snapped', iter, asked, startSource}. */
export function frameTag({ kind, iter, asked = null, startSource = 'xenium' } = {}) {
  const i = number(iter);
  switch (kind) {
    case 'start': return deep('deepAtStart', { start: startName(startSource) });
    case 'live': return deep('deepLive', { i });
    case 'paused': return deep('deepPaused', { i });
    case 'older': return deep('deepNotNewest', { i });
    case 'finished': return deep('deepFinished', { i });
    case 'stopped': return deep('deepStopped', { i });
    case 'earlier': return deep('deepEarlier', { j: i, i: number(asked) });
    case 'snapped': return deep('deepSnapped', { i });
    default: return '';
  }
}
/** The engine's reason for a run that cannot be checked, inside deepCantRun ("Go deeper can't run for this run: the
 *  mixing check needs ..."): the check's own sentence, its first letter lowered unless it starts a name. */
export function cantRunText(reasonSentence) {
  const text = String(reasonSentence || QC_TEXT.unavailable).trim().replace(/\.$/, '');
  const lowered = /^[A-Z][a-z]/.test(text) && !/^(Xenium|Proseg|Python)\b/.test(text) ? text[0].toLowerCase() + text.slice(1) : text;
  return deep('deepCantRun', { reason: lowered }) + '.';
}
/** The detail job's failure in plain words (detail_mismatch has its own; the others are the check's failure reasons). */
export function detailFailedText(reason, failureText) {
  return deep('deepDetailFailed', { reason: reason === 'detail_mismatch' ? DEEP_TEXT.deepDetailMismatch : failureText || QC_TEXT.failureUnexpected });
}

/** Every string the window can show, expanded with sample values (the node test runs them through the banned-word
 *  rule; the page's qcTexts() lists the ones on screen), with the check's "Test a setting here…" beside Show on the
 *  image. */
export function allTexts() {
  const values = name => ({ cell: 'cell-c-1', i: '1,201', j: '1,180', start: startName('xenium'), p: '46', q: '1', n: '682', d: '25',
    k: '13', of: '50', t: '2 min', tx: '39', gene: 'GENE1', genes: 'GENE1 · GENE2 · GENE3', group: 'makes GENE1 · GENE2 · GENE3', part: DEEP_TEXT.deepOwn,
    own: '8', nb: '31', loose: 'Loose < 1 %', from: deep('deepRowFrom', { cell: 'cell-b-1' }), step: DEEP_TEXT.deepMixed, change: 'up from 1 % at the start',
    reason: QC_TEXT.failureMemory, target: 'cell-a-1', m: '50', word: DEEP_TEXT.deepWhereEdge, around: DEEP_TEXT.deepWhereUnder, mof: '80',
    near: '3', here: '9', giver: '14', who: DEEP_TEXT.deepGlowAll, cells: '40,000', date: '21 Oct 2026',
    head: 'cell-b-1 · makes GENE1 · GENE2 · GENE3', text: '12 likely from neighbours', nucleus: '5', unknown: '1', bg: '165',
    kind: 'GENE1 · GENE2 · GENE3' }[name] ?? '7');
  const templates = Object.entries(DEEP_TEMPLATES).map(([key, pattern]) =>
    deep(key, Object.fromEntries([...pattern.matchAll(/\{([^}]+)\}/g)].map(m => [m[1], values(m[1])]))));
  // The line and its tooltip as the window puts them together: every place, the edge and the hold split; and the
  // line that names no neighbour (none of the pink dots lies near the one it would name).
  const glow = { count: 14, likely: 12, near: 7, places: { own: 9, none: 1, giver: 1, other: 0, unknown: 1 }, levels: { nucleus: 5, cell: 3, unknown: 1 } };
  const away = noOutlineLine(glow, { name: 'cell-b-1', kind: ['GENE1', 'GENE2', 'GENE3', 'GENE4'] });
  return [...Object.values(DEEP_TEXT), ...templates, ...howItems({ cells: 40000, created: '2026-10-21T10:00:00Z', seconds: 420 }),
    cantRunText(QC_TEXT.unavailableTable), detailFailedText('detail_mismatch'), detailFailedText('oom', QC_TEXT.failureMemory),
    startLine(glow, DEEP_TEXT.deepGlowAll), startTip(glow), startTip(glow, true), away.text, away.tip, QC_TEXT.testOpen, QC_TEXT.testOpenTooltip];
}
export { qcCopy, QC_TEXT };
