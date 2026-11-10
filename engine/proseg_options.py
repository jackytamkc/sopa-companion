"""Proseg 3.2.0 options for Xenium parquet crops.

Defaults are effective Xenium defaults; None means derived or unset. Geometry and
model dimensions, initialization, filtering, density maps and hyperparameters are
startup-only. Live diffusion changes rebuild every cached distance/proposal table.
Temperature is the live patch's additional CLI override (None follows the schedule).
Bounds intentionally avoid degenerate distributions and unbounded live table sizes.
"""
import math

# The genes every run leaves out unless the user sets Excluded genes or Include neg ctrls (input_args): proseg's own
# --xenium default (Xenium's controls), plus the control probes of the other platforms sopa reads, as proseg's own
# presets name them: CosMx's NegPrb and SystemControl (with FalseCode and Negative), MERSCOPE's Blank (these
# were modelled as genes). Compare leaves out the same genes (zarr_bridge/helper.py NOT_GENES).
DEFAULT_EXCLUDED_GENES = '^(Deprecated|NegControl|Unassigned|Intergenic|FalseCode|SystemControl|NegPrb|Negative|Blank)'
# A text option holds one short pattern: Compare reads the excluded genes with Python's re as well, one gene name at a
# time, so a long or nested pattern there could take minutes (the helper's PATTERN_MAX is the same number).
MAX_PATTERN = 200

# Exclusions: input presets, columns and alternate input sources cannot apply;
# output paths/names and live transport flags belong to the app; monitor-cell-polygons,
# monitor-cell-polygons-freq and cells-per-chunk are dead upstream flags.
# use-cell-initialization clears compartment columns required by Xenium parquet 3.2.0.
# unmodeled-fixed-cells requires the excluded fixed-mask input.
# help/version are informational CLI actions, not segmentation settings.
# Excluded: transcript-csv — app supplies the Xenium parquet input
# Excluded: xenium — app supplies the Xenium parquet input
# Excluded: cosmx — alternate input preset, source, or input column mapping
# Excluded: cosmx-micron — alternate input preset, source, or input column mapping
# Excluded: merscope — alternate input preset, source, or input column mapping
# Excluded: merfish — alternate input preset, source, or input column mapping
# Excluded: anndata — alternate input preset, source, or input column mapping
# Excluded: zarr — alternate input preset, source, or input column mapping
# Excluded: spatialdata-table — alternate input preset, source, or input column mapping
# Excluded: anndata-coordinate-key — alternate input preset, source, or input column mapping
# Excluded: coordinate-scale — Xenium coordinates are already micrometres; scaling them would misregister the
#   segmentation against the morphology image and the drawn region
# Excluded: use-cell-initialization — incompatible with Xenium parquet compartment requirements
# Excluded: visiumhd — alternate input preset, source, or input column mapping
# Excluded: gene-column — alternate input preset, source, or input column mapping
# Excluded: transcript-id-column — alternate input preset, source, or input column mapping
# Excluded: cellpose-masks — alternate input preset, source, or input column mapping
# Excluded: cellpose-masks-fixed — alternate input preset, source, or input column mapping
# Excluded: unmodeled-fixed-cells — requires fixed Cellpose masks
# Excluded: spaceranger-barcode-mappings — alternate input preset, source, or input column mapping
# Excluded: cellpose-cellprobs — alternate input preset, source, or input column mapping
# Excluded: cellpose-x-transform — alternate input preset, source, or input column mapping
# Excluded: cellpose-y-transform — alternate input preset, source, or input column mapping
# Excluded: cellpose-scale — alternate input preset, source, or input column mapping
# Excluded: cellpose-cellprob-discount — alternate input preset, source, or input column mapping
# Excluded: x-column — alternate input preset, source, or input column mapping
# Excluded: y-column — alternate input preset, source, or input column mapping
# Excluded: z-column — alternate input preset, source, or input column mapping
# Excluded: compartment-column — alternate input preset, source, or input column mapping
# Excluded: compartment-nuclear — alternate input preset, source, or input column mapping
# Excluded: fov-column — alternate input preset, source, or input column mapping
# Excluded: cell-assignment-column — alternate input preset, source, or input column mapping
# Excluded: cell-assignment-unassigned — alternate input preset, source, or input column mapping
# Excluded: cell-id-column — alternate input preset, source, or input column mapping
# Excluded: cell-id-unassigned — alternate input preset, source, or input column mapping
# Excluded: qv-column — alternate input preset, source, or input column mapping
# Excluded: parquet-fmt — app supplies the Xenium parquet input
# Excluded: zarr-shape — alternate input preset, source, or input column mapping
# Excluded: zarr-shape-geometry-column — alternate input preset, source, or input column mapping
# Excluded: zarr-shape-cell-id-column — alternate input preset, source, or input column mapping
# Excluded: cells-per-chunk — dead upstream option
# Excluded: output-path — app controls output paths and files
# Excluded: output-spatialdata — app controls output paths and files
# Excluded: overwrite — app controls output paths and files
# Excluded: output-counts — app controls output paths and files
# Excluded: output-expected-counts — app controls output paths and files
# Excluded: output-rates — app controls output paths and files
# Excluded: output-cell-metadata — app controls output paths and files
# Excluded: output-transcript-metadata — app controls output paths and files
# Excluded: output-gene-metadata — app controls output paths and files
# Excluded: output-metagene-rates — app controls output paths and files
# Excluded: output-metagene-loadings — app controls output paths and files
# Excluded: output-cell-polygons — app controls output paths and files
# Excluded: output-union-cell-polygons — app controls output paths and files
# Excluded: output-cell-polygon-layers — app controls output paths and files
# Excluded: output-voxel-counts — app controls output paths and files
# Excluded: monitor-cell-polygons — dead upstream option
# Excluded: monitor-cell-polygons-freq — dead upstream option
# Excluded: live-dir — app controls live transport and run lifetime
# Excluded: live-control — app controls live transport and run lifetime
# Excluded: live-hold — app controls live transport and run lifetime
# Excluded: live-tiled — app controls live transport and run lifetime
# Excluded: live-counts — app controls live transport and run lifetime
# Excluded: live-counts-dir — app controls live transport and run lifetime
# Excluded: live-counts-start — app controls live transport and run lifetime

# Biology-facing guidance checked against the local 3.2.0 source and research A2-A6.
# Suggested trial ranges are conservative comparisons around defaults, not validated
# biological optima. Effects on shapes are conditional on the data and other settings.
OPTIONS = [
    # Evidence (native/proseg/src): main.rs:1149-1198,1548
    {'flag': 'burnin-samples',
     'label': 'Burnin samples',
     'group': 'Schedule',
     'type': 'int',
     'default': 200,
     'help': 'Sets how long cells adjust on the coarse grid before finer boundaries and transcript '
             'drift are allowed.',
     'advanced': False,
     'live': False,
     'min': 0,
     'what': 'Sets how long cells adjust on the coarse grid before finer boundaries and transcript '
             'drift are allowed.',
     'up': 'Cells have more time to change their starting outlines before fine detail appears.',
     'down': 'Fine boundaries appear sooner, but cells may still be close to their starting shapes.',
     'use_when': 'Cells are still changing greatly when the view switches to the finer grid.',
     'typical': 'Try 100 to 300 iterations; proseg default 200.',
     'caution': 'Longer runs cost time. Use at least one early coarse-grid phase iteration if automatic hill-climb '
                'cooling is enabled.'},
    # Evidence (native/proseg/src): main.rs:1239-1250,1298-1310
    {'flag': 'samples',
     'label': 'Samples',
     'group': 'Schedule',
     'type': 'int',
     'default': 200,
     'help': 'Sets how long cells explore different fine boundaries before the final settling phase; the '
             'last Recorded samples of them are averaged into the counts.',
     'advanced': False,
     'live': False,
     'min': 0,
     'what': 'Sets how long cells explore different fine boundaries before the final settling phase; the '
             'last Recorded samples of them are averaged into the counts.',
     'up': 'Cells explore boundaries for longer before the run settles and pauses at the end of the '
           'standard run.',
     'down': 'Sampling ends sooner, so the settling phase and the end of the standard run come sooner.',
     'use_when': 'The standard run ends while cell outlines are still changing substantially (Resume also '
                 'samples on past its end).',
     'typical': 'Try 100 to 400 iterations; proseg default 200.',
     'caution': 'Must be at least Recorded samples.'},
    # Evidence (native/proseg/src): main.rs:1241-1249,1310,1373-1385; sampler/paramsampler.rs:52-61
    {'flag': 'recorded-samples',
     'label': 'Recorded samples',
     'group': 'Schedule',
     'type': 'int',
     'default': 100,
     'help': 'Sets how many final sampling iterations contribute to average transcript counts. It does '
             'not choose which viewer frames are saved.',
     'advanced': False,
     'live': False,
     'min': 0,
     'what': 'Sets how many final sampling iterations contribute to average transcript counts. It does '
             'not choose which viewer frames are saved.',
     'up': 'More of the run contributes to average counts; outlines do not directly become larger or '
           'smoother.',
     'down': 'Fewer iterations contribute to average counts, so those averages may vary more between '
             'runs.',
     'use_when': 'You need more averaging of transcript counts after boundaries have had time to '
                 'adjust.',
     'typical': 'Try 50 to 200 iterations, no more than Samples; proseg default 100.',
     'caution': 'Iterations sampled on past the end of the standard run (Resume) are averaged in as well.'},
    # Evidence (native/proseg/src): main.rs:1236,1295-1326; sampler/voxelsampler.rs:122-132
    {'flag': 'hillclimb',
     'label': 'Hillclimb',
     'group': 'Schedule',
     'type': 'int',
     'default': 50,
     'help': 'Adds a final settling phase to the standard run, making worse-fitting boundary changes '
             'less likely.',
     'advanced': False,
     'live': False,
     'min': 0,
     'what': 'Adds a final settling phase to the standard run, making worse-fitting boundary changes '
             'less likely.',
     'up': 'Cells have more iterations to settle, usually with less boundary flicker.',
     'down': 'The standard run ends sooner with more of the normal boundary movement still present.',
     'use_when': 'The standard run ends with outlines that still fluctuate and you want a more settled '
                 'final view.',
     'typical': 'Try 0 to 100 iterations; proseg default 50.',
     'caution': 'Finish & save saves the state on screen; it never runs this phase afterwards. Resuming '
                'past the end of the standard run samples on at full temperature, so outlines move again: '
                'lower Temperature first to keep them settled. A Temperature override replaces automatic '
                'cooling.'},
    # Evidence (native/proseg/src): main.rs:1542-1544; sampler/voxelsampler.rs:98-113
    {'flag': 'morphology-steps-per-iter',
     'label': 'Boundary attempts per iteration (higher = faster change)',
     'group': 'Schedule',
     'type': 'int',
     'default': 4000,
     'help': 'Controls how many boundary-change attempts cells get in each iteration.',
     'advanced': False,
     'live': True,
     'min': 1,
     'max': 1000000,
     'what': 'Controls how many boundary-change attempts cells get in each iteration.',
     'up': 'Boundaries have more chances to move between displayed iterations, but each iteration '
           'takes longer.',
     'down': 'Changes are more gradual per iteration and each iteration spends less time on '
             'boundaries.',
     'use_when': 'Cells barely move between iterations, or you want smaller steps while inspecting '
                 'changes.',
     'typical': 'Try 1024 to 8192 attempts per iteration; default 4000.',
     'caution': 'Attempts are grouped in batches of 64, rounded down with at least one batch, and '
                'spread across processing blocks. Attempts are not guaranteed accepted changes.'},
    # Evidence (native/proseg/src): sampler/voxelsampler.rs:122-132; sampler/transcriptrepo.rs:138; sampler/paramsampler.rs:31; main.rs:1302-1326
    {'flag': 'temperature',
     'label': 'Temperature (lower = boundaries settle)',
     'group': 'Schedule',
     'type': 'float',
     'default': None,
     'help': 'Controls how readily boundaries explore changes that fit the data and cell-shape '
             'preferences less well. Lower values help outlines settle.',
     'advanced': True,
     'live': True,
     'min': 0.01,
     'max': 1.0,
     'what': 'Controls how readily boundaries explore changes that fit the data and cell-shape '
             'preferences less well. Lower values help outlines settle.',
     'up': 'Towards 1, boundaries explore more freely and may flicker more.',
     'down': 'Below 1, worse-fitting changes become less likely; near 0.01, boundaries strongly favour '
             'improvements.',
     'use_when': 'Outlines keep flickering after cells have had time to develop, especially before '
                 'Finish & save.',
     'typical': 'Try 1 for exploration, 0.1 for settling, or 0.01 for stronger settling; no units. '
                'Default is automatic: 1 during sampling, then cooling in the final settling phase.',
     'caution': 'Below 1 does not strictly forbid worse-fitting moves. It does not cool transcript '
                'repositioning or expression updates. Resume or Step to let the change act before '
                'saving; Pause is the way to freeze the view.'},
    # Evidence (native/proseg/src): main.rs:38-39,731-742,919-925; sampler/voxelcheckerboard.rs:1360-1423
    {'flag': 'burnin-voxel-size',
     'label': 'Burnin voxel size',
     'group': 'Voxels & geometry',
     'type': 'float',
     'default': 2.0,
     'help': 'Sets the width of the small grid blocks used to build the starting cells and adjust '
             'their early outlines.',
     'advanced': False,
     'live': False,
     'min': 0.01,
     'what': 'Sets the width of the small grid blocks used to build the starting cells and adjust '
             'their early outlines.',
     'up': 'Starting cells look more blocky; small neighbouring nuclei can compete for the same block.',
     'down': 'Starting cells can represent finer details and separate nearby nuclei more finely.',
     'use_when': 'The starting view loses small nuclei or looks too coarse for tightly packed cells.',
     'typical': 'Try 1 to 4 µm; default 2 µm.',
     'caution': 'Smaller blocks cost memory and time. This width must be a whole-number multiple of '
                'the final Voxel size. A finer grid cannot create missing Xenium cell identities.'},
    # Evidence (native/proseg/src): main.rs:38-39,731-742,1200-1227; sampler/voxelcheckerboard.rs:2919-2975
    {'flag': 'voxel-size',
     'label': 'Voxel size',
     'group': 'Voxels & geometry',
     'type': 'float',
     'default': 1.0,
     'help': 'Sets the width of the small grid blocks that form the final cell boundaries.',
     'advanced': False,
     'live': False,
     'min': 0.01,
     'what': 'Sets the width of the small grid blocks that form the final cell boundaries.',
     'up': 'Edges look coarser and tiny protrusions or gaps cannot be represented as finely.',
     'down': 'Edges can follow finer detail, including narrow protrusions and small gaps.',
     'use_when': 'Grid-shaped steps in boundaries hide the level of detail you need to inspect.',
     'typical': 'Try 0.5, 1 or 2 µm; default 1 µm.',
     'caution': 'Finer grids increase memory and work. Burnin voxel size must divide by this width to '
                'a whole number. This changes the segmentation model, not just its display.'},
    # Evidence (native/proseg/src): main.rs:240-242,812-816; sampler/voxelcheckerboard.rs:1293-1301; sampler/voxelsampler.rs:415-453
    {'flag': 'voxel-layers',
     'label': 'Voxel layers',
     'group': 'Voxels & geometry',
     'type': 'int',
     'default': 4,
     'help': 'Sets how many depth slices can have different cell outlines through the tissue.',
     'advanced': True,
     'live': False,
     'min': 1,
     'max': 256,
     'what': 'Sets how many depth slices can have different cell outlines through the tissue.',
     'up': 'Cells can have more detail through depth, and cells above one another have more room to '
           'differ.',
     'down': 'Cells have less depth detail; one layer makes all transcripts compete in a single flat '
             'segmentation.',
     'use_when': 'Cells overlap in the flat view and you need to inspect whether they separate through '
                 'depth.',
     'typical': 'Try 1 to 8 layers; default 4.',
     'caution': 'More layers cost memory and change how shape preferences act. Layers are not '
                'microscope acquisition planes; flat input forces one layer.'},
    # Evidence (native/proseg/src): sampler/voxelcheckerboard.rs:1282-1301; sampler/voxelsampler.rs:98-113
    {'flag': 'quad-size',
     'label': 'Quad size',
     'group': 'Voxels & geometry',
     'type': 'float',
     'default': 150.0,
     'help': 'Sets the width of tissue blocks processed together. It mainly controls how work is '
             'divided across the computer.',
     'advanced': True,
     'live': False,
     'min': 0.01,
     'what': 'Sets the width of tissue blocks processed together. It mainly controls how work is '
             'divided across the computer.',
     'up': 'Larger regions share a processing block; there is no predictable change in cell size or '
           'roundness.',
     'down': 'More, smaller processing blocks are used; there is no predictable change in cell size or '
             'roundness.',
     'use_when': 'You are comparing processing speed on a fixed crop, rather than correcting a '
                 'biological boundary.',
     'typical': 'Usually keep 150 µm, the default; exploratory speed comparisons might use 100 to 200 '
                'µm.',
     'caution': 'This also changes how boundary attempts are distributed. Smaller blocks are not '
                'guaranteed faster or equivalent.'},
    # Evidence (native/proseg/src): sampler/transcripts.rs:1353; main.rs:812-816
    {'flag': 'ignore-z-coord',
     'label': 'Ignore z coord',
     'group': 'Voxels & geometry',
     'type': 'bool',
     'default': False,
     'help': 'Treats all transcripts as lying in one flat plane instead of using their measured depth.',
     'advanced': True,
     'live': False,
     'what': 'Treats all transcripts as lying in one flat plane instead of using their measured depth.',
     'up': 'On: cells become flat, so transcripts from cells above one another compete in the same '
           'layer.',
     'down': 'Off: cells can differ through depth when the input has usable depth measurements.',
     'use_when': 'Depth measurements are unusable and you deliberately want a two-dimensional '
                 'comparison.',
     'typical': 'On or off; default off.',
     'caution': 'Turning this on can mix evidence from overlapping cells. It is not the same as '
                'choosing the 2-D display.'},
    # Evidence (native/proseg/src): main.rs:995; sampler/voxelsampler.rs:21-29,415-507; sampler/math.rs:93-95
    {'flag': 'cell-compactness',
     'label': 'Cell compactness (lower = rounder cells)',
     'group': 'Cell shape & priors',
     'type': 'float',
     'default': 0.04,
     'help': 'Controls how strongly irregular cell outlines are discouraged in each depth layer. '
             'Smaller values apply a stronger preference for compact shapes.',
     'advanced': False,
     'live': True,
     'min': 0.001,
     'max': 10.0,
     'what': 'Controls how strongly irregular cell outlines are discouraged in each depth layer. '
             'Smaller values apply a stronger preference for compact shapes.',
     'up': 'Cells can become more elongated or follow transcripts into thin protrusions.',
     'down': 'Cells tend to look rounder and smoother, with fewer thin protrusions.',
     'use_when': 'Lower it for spiky outlines; raise it when elongated cell types are being rounded '
                 'off.',
     'typical': 'Try 0.02 to 0.08 in small steps; default 0.04, with no units.',
     'caution': 'This is a shape preference, not a cell-size target. Strong smoothing can erase real '
                'extensions, and weaker smoothing can admit noise.'},
    # Evidence (native/proseg/src): main.rs:932-940,1618; sampler/voxelsampler.rs:302-348
    # Its label's hint says what it does for every start: a run started from Cellpose cells holds
    # those, not Xenium's.
    {'flag': 'prior-weight',
     'label': 'Prior weight (higher = the starting cells hold their transcripts harder)',
     'group': 'Cell shape & priors',
     'type': 'float',
     'default': 1.0,
     'help': 'Controls how strongly each grid block keeps the starting cell suggested by its original '
             'transcripts: a Xenium cell, or a cell of the segmentation the run started from. Proseg '
             'automatically increases this strength when there are more transcripts per cell.',
     'advanced': True,
     'live': True,
     'min': 0.0,
     'max': 1000.0,
     'what': 'Controls how strongly each grid block keeps the starting cell suggested by its original '
             'transcripts: a Xenium cell, or a cell of the segmentation the run started from. Proseg '
             'automatically increases this strength when there are more transcripts per cell.',
     'up': 'With the usual settings, cells stay closer to Xenium nuclear assignments; cell-body '
           'assignments also pull them back if their reassignment probability is below 0.5. In a run '
           'started from cells only (no nuclei), the whole starting outline holds more firmly, '
           'including any neighbouring RNA already inside it.',
     'down': 'Boundaries have more freedom to depart from Xenium assignments and follow transcript '
             'evidence. In a run started from cells only, this is what lets proseg move the starting '
             'outlines.',
     'use_when': 'Raise it if cells cross reliable Xenium nuclei; lower it if a visibly wrong starting '
                 'assignment holds boundaries in place.',
     'typical': 'Try 0.5 to 2 as a multiplier; default 1. Zero removes this preference. Started from '
                'cells only, try the value that brings the scaled strength to about 1 (0.07 at 1,400 '
                'transcripts per cell).',
     'caution': 'The scaled strength (this value times the transcripts per cell over 100, when that is '
                'above 1) is shown as prior weight in the run log when above 1. At a '
                'reassignment probability of 0.5 that region is neutral; above 0.5, more weight '
                'strengthens avoidance of its original cell. Started from cells only, nothing is '
                'neutral: every transcript inside a starting cell holds its spot to that cell, as a '
                'nucleus would. This does not trace the displayed Xenium polygon exactly.'},
    # Evidence (native/proseg/src): main.rs:923; sampler/voxelcheckerboard.rs:1378-1416; sampler/voxelsampler.rs:334-348
    {'flag': 'nuclear-reassignment-prob',
     'label': 'Nuclear reassignment prob',
     'group': 'Cell shape & priors',
     'type': 'float',
     'default': 0.2,
     'help': 'Sets how much the model distrusts Xenium nuclear assignments when building the starting '
             'grid. At 0.2, nuclear grid blocks start with an 80 percent preference for their Xenium '
             'cell.',
     'advanced': True,
     'live': False,
     'min': 1e-06,
     'max': 0.999999,
     'what': 'Sets how much the model distrusts Xenium nuclear assignments when building the starting '
             'grid. At 0.2, nuclear grid blocks start with an 80 percent preference for their Xenium '
             'cell.',
     'up': 'Towards 0.5, boundaries can more easily depart from the original nucleus; above 0.5 they '
           'are encouraged to avoid that original cell.',
     'down': 'Cells hold more tightly to their original Xenium nuclei.',
     'use_when': 'The nuclear assignment itself is visibly wrong, or reliable nuclei are not being '
                 'respected.',
     'typical': 'Try probabilities 0.1 to 0.4; default 0.2. A value of 0.5 is neutral.',
     'caution': 'Requires a new run. This cannot split one Xenium cell identity into two or merge two '
                'identities into one.'},
    # Evidence (native/proseg/src): main.rs:924; sampler/voxelcheckerboard.rs:1440-1473; sampler/voxelsampler.rs:334-348
    {'flag': 'prior-seg-reassignment-prob',
     'label': 'Prior seg reassignment prob',
     'group': 'Cell shape & priors',
     'type': 'float',
     'default': 0.5,
     'help': 'Sets how much cells should follow Xenium transcript assignments outside nuclei. The '
             'default 0.5 gives no preference for keeping or leaving that Xenium cell.',
     'advanced': True,
     'live': False,
     'min': 1e-06,
     'max': 0.999999,
     'what': 'Sets how much cells should follow Xenium transcript assignments outside nuclei. The '
             'default 0.5 gives no preference for keeping or leaving that Xenium cell.',
     'up': 'Towards 0.5, cells are less tied to the original cell body; above 0.5, they are encouraged '
           'to leave their original assignment.',
     'down': 'Below 0.5, cells follow the original Xenium cell-body assignments more closely.',
     'use_when': 'Lower it when Xenium cell bodies look reliable and you want proseg to preserve them '
                 'more closely.',
     'typical': 'Try probabilities 0.2 to 0.5; default 0.5.',
     'caution': 'Requires a new run. Values above 0.5 actively oppose the original assignment; they do '
                'not merely remove its influence.'},
    # Evidence (native/proseg/src): sampler/voxelcheckerboard.rs:1975-1976,2781-2815
    {'flag': 'expand-initialized-cells',
     'label': 'Expand initialized cells',
     'group': 'Cell shape & priors',
     'type': 'int',
     'default': 0,
     'help': 'Grows the starting cells into empty neighbouring grid blocks before sampling begins.',
     'advanced': True,
     'live': False,
     'min': 0,
     'what': 'Grows the starting cells into empty neighbouring grid blocks before sampling begins.',
     'up': 'Starting cells cover more tissue around their nuclei, before evidence has refined their '
           'outlines.',
     'down': 'Starting cells remain closer to their nuclear starting regions; zero adds no outward '
             'expansion.',
     'use_when': 'Cells start very small and you want to compare a broader starting point in a new '
                 'run.',
     'typical': 'Try 0 to 2 grid-block expansion steps; default 0. Each step uses the starting grid '
                'width in µm.',
     'caution': 'This changes the starting point, not the final target size. Expansion fills empty '
                'blocks and does not split or merge identities.'},
    # Evidence (native/proseg/src): sampler/voxelsampler.rs:151-179,232-260
    {'flag': 'ab-nihlo-bubble-prob',
     'label': 'Background gap proposals (higher = more gaps)',
     'group': 'Cell shape & priors',
     'type': 'float',
     'default': 0.05,
     'help': 'Sets the chance that a proposed boundary change assigns a grid block to background, '
             'leaving a gap rather than giving it to a neighbouring cell.',
     'advanced': True,
     'live': True,
     'min': 0.0,
     'max': 1.0,
     'what': 'Sets the chance that a proposed boundary change assigns a grid block to background, '
             'leaving a gap rather than giving it to a neighbouring cell.',
     'up': 'More attempts create background gaps along cell interfaces; accepted gaps still depend on '
           'the data and shape preferences.',
     'down': 'Fewer attempts create these gaps, which may reduce newly appearing pockets of '
             'background.',
     'use_when': 'Try lowering it when gaps repeatedly open inside cells or along their interfaces.',
     'typical': 'Try probabilities 0.01 to 0.1; default 0.05.',
     'caution': 'It never creates a new cell. It changes which moves are tried, not a target gap '
                'fraction, and does not directly fill existing holes. Avoid the extremes for routine '
                'use.'},
    # Evidence (native/proseg/src): sampler/voxelsampler.rs:181-207; sampler/connectivity.rs:8-30,50-110
    {'flag': 'enforce-connectivity',
     'label': 'Keep each cell in one piece (enforce connectivity)',
     'group': 'Cell shape & priors',
     'type': 'bool',
     'default': False,
     'help': 'Checks proposed boundary changes and blocks those that would break a cell into '
             'disconnected pieces nearby.',
     'advanced': True,
     'live': True,
     'what': 'Checks proposed boundary changes and blocks those that would break a cell into '
             'disconnected pieces nearby.',
     'up': 'On: cells are less able to develop new disconnected pieces as their boundaries move.',
     'down': 'Off: cells can change more freely, including changes that separate pieces.',
     'use_when': 'A cell is starting to break apart as the run progresses.',
     'typical': 'On or off; default off.',
     'caution': 'Does not repair existing pieces. A cell connected through depth can look disconnected '
                'in one layer or in the flat view. The extra checks take time.'},
    # Evidence (native/proseg/src): main.rs:1548-1551,1619; sampler/transcriptrepo.rs:211-284
    {'flag': 'no-diffusion',
     'label': 'Switch off the transcript drift model (no diffusion)',
     'group': 'Transcript diffusion',
     'type': 'bool',
     'default': False,
     'help': 'Stops further transcript repositioning, the process that lets a transcript count for a '
             'nearby cell without its boundary stretching to the measured position.',
     'advanced': True,
     'live': True,
     'what': 'Stops further transcript repositioning, the process that lets a transcript count for a '
             'nearby cell without its boundary stretching to the measured position.',
     'up': 'On: transcript assignments stop exploring nearby positions; boundaries may need to reach '
           'transcripts or leave them as background.',
     'down': 'Off: after the early coarse-grid phase, nearby cells can count drifted transcripts while keeping more '
             'compact outlines.',
     'use_when': 'Leave it off when cells grow thin extensions towards scattered transcripts; turn it '
                 'on to compare results without further repositioning.',
     'typical': 'On or off; default off, so repositioning is enabled after the early coarse-grid phase.',
     'caution': 'Turning it on during a run freezes existing transcript offsets; it does not move '
                'transcripts back. Start a new run with it on to remove repositioning entirely. '
                'Displayed measured transcript positions do not move.'},
    # Evidence (native/proseg/src): sampler/transcriptrepo.rs:353-358; main.rs:1548-1551
    {'flag': 'diffusion-probability',
     'label': 'Share of far-drifted transcripts (diffusion probability)',
     'group': 'Transcript diffusion',
     'type': 'float',
     'default': 0.2,
     'help': 'Sets the share of transcripts expected to have drifted farther from their cell, rather '
             'than only a short distance.',
     'advanced': True,
     'live': True,
     'min': 0.0,
     'max': 1.0,
     'what': 'Sets the share of transcripts expected to have drifted farther from their cell, rather '
             'than only a short distance.',
     'up': 'When the far distance is larger than the near distance, more distant transcripts can count '
           'for nearby cells without long cell protrusions.',
     'down': 'Counts stay more local, so distant transcripts may remain background or need boundaries '
             'to reach them.',
     'use_when': 'Cells stretch towards scattered transcripts that appear to have drifted away from an '
                 'otherwise compact cell.',
     'typical': 'Try probabilities 0.1 to 0.3; default 0.2, meaning a 20 percent far-drift share.',
     'caution': 'Zero removes only the far component; short-distance repositioning continues. Has no '
                'effect during the early coarse-grid phase or with No diffusion on. Large values can mix neighbouring '
                'cells.'},
    # Evidence (native/proseg/src): sampler/transcriptrepo.rs:53-55,353-375
    {'flag': 'diffusion-sigma-near',
     'label': 'Short drift distance, µm (diffusion sigma near)',
     'group': 'Transcript diffusion',
     'type': 'float',
     'default': 1.0,
     'help': 'Sets the typical short sideways drift distance allowed when counting transcripts for '
             'nearby cells.',
     'advanced': True,
     'live': True,
     'min': 0.01,
     'max': 100.0,
     'what': 'Sets the typical short sideways drift distance allowed when counting transcripts for '
             'nearby cells.',
     'up': 'More transcripts just outside a boundary can count for that cell without expanding its '
           'outline.',
     'down': 'Short-distance assignments stay closer to the measured transcript positions.',
     'use_when': 'Small offsets between transcripts and plausible cell edges appear to cause '
                 'unnecessary boundary extensions.',
     'typical': 'Try 0.5 to 2 µm; default 1 µm.',
     'caution': 'This is a typical distance along each sideways axis, not a maximum reach. Normally '
                'keep it below the far distance. Inactive during the early coarse-grid phase or with No diffusion on.'},
    # Evidence (native/proseg/src): sampler/transcriptrepo.rs:53-55,353-375
    {'flag': 'diffusion-sigma-far',
     'label': 'Long drift distance, µm (diffusion sigma far)',
     'group': 'Transcript diffusion',
     'type': 'float',
     'default': 4.0,
     'help': 'Sets the typical sideways drift distance for the transcripts expected to have moved '
             'farther from their cell.',
     'advanced': True,
     'live': True,
     'min': 0.01,
     'max': 100.0,
     'what': 'Sets the typical sideways drift distance for the transcripts expected to have moved '
             'farther from their cell.',
     'up': 'Cells can count more distant scattered transcripts without stretching their outlines to '
           'reach them.',
     'down': 'Far-drift assignments stay closer to the cell, so distant transcripts may remain '
             'background.',
     'use_when': 'Scattered transcripts several micrometres from plausible boundaries appear to be '
                 'pulling out long extensions.',
     'typical': 'Try 2 to 6 µm; default 4 µm.',
     'caution': 'A typical distance along each sideways axis, not a maximum. Large distances can mix '
                'neighbouring cells. No effect when the far-drift probability is zero, during the early coarse-grid phase, '
                'or with No diffusion on.'},
    # Evidence (native/proseg/src): sampler/transcriptrepo.rs:55,353-358; sampler/transcripts.rs:150-185
    {'flag': 'diffusion-sigma-z',
     'label': 'Depth drift distance (diffusion sigma z)',
     'group': 'Transcript diffusion',
     'type': 'float',
     'default': 0.1,
     'help': 'Sets how far transcripts may drift through tissue depth when deciding which cell counts '
             'them.',
     'advanced': True,
     'live': True,
     'min': 0.001,
     'max': 2.0,
     'what': 'Sets how far transcripts may drift through tissue depth when deciding which cell counts '
             'them.',
     'up': 'Transcripts can more readily count for cells in nearby depth layers.',
     'down': 'Assignments stay closer to the measured depth, keeping cells above and below one another '
             'more distinct.',
     'use_when': 'You suspect depth measurement uncertainty is affecting assignments between '
                 'overlapping cells.',
     'typical': 'Try 0.05 to 0.2 of the normalized tissue depth; default 0.1, about one tenth of that '
                'depth.',
     'caution': 'These are fractions of the depth after tilt correction and clipping, not µm. Little '
                'use with one layer; inactive during the early coarse-grid phase or with No diffusion on.'},
    # Evidence (native/proseg/src): sampler/transcriptrepo.rs:28-49,216-284
    {'flag': 'diffusion-proposal-sigma',
     'label': 'Drift search step, µm (diffusion proposal sigma)',
     'group': 'Transcript diffusion',
     'type': 'float',
     'default': 4.0,
     'help': 'Sets the typical sideways jump tried when looking for a better transcript assignment. It '
             'controls the search steps, rather than the expected biological drift distance.',
     'advanced': True,
     'live': True,
     'min': 0.01,
     'max': 100.0,
     'what': 'Sets the typical sideways jump tried when looking for a better transcript assignment. It '
             'controls the search steps, rather than the expected biological drift distance.',
     'up': 'Assignments try longer jumps per pass; more jumps may be rejected, so cells need not '
           'change faster.',
     'down': 'Assignments try shorter jumps and explore nearby cells more locally.',
     'use_when': 'You are comparing how quickly transcript assignments explore nearby cells after '
                 'setting plausible drift distances.',
     'typical': 'Usually keep 4 µm, the default; try 2 to 6 µm for search comparisons.',
     'caution': 'Mainly a sampling-speed setting, with no reliable direction for final cell shapes. '
                'Inactive during the early coarse-grid phase or with No diffusion on.'},
    # Evidence (native/proseg/src): sampler/transcriptrepo.rs:40-50,216-284; sampler/transcripts.rs:150-185
    {'flag': 'diffusion-proposal-sigma-z',
     'label': 'Depth drift search step (diffusion proposal sigma z)',
     'group': 'Transcript diffusion',
     'type': 'float',
     'default': 0.2,
     'help': 'Sets the typical depth jump tried when searching for a different transcript assignment.',
     'advanced': True,
     'live': True,
     'min': 0.001,
     'max': 2.0,
     'what': 'Sets the typical depth jump tried when searching for a different transcript assignment.',
     'up': 'Assignments try crossing more depth layers at once, but more attempts may be rejected.',
     'down': 'Assignments try smaller depth steps; there is no guaranteed change in final outlines.',
     'use_when': 'You are comparing search speed for assignments through depth, rather than choosing a '
                 'cell shape.',
     'typical': 'Usually keep 0.2 of normalized tissue depth, the default; try 0.1 to 0.3 for search '
                'comparisons.',
     'caution': 'Not µm and not the expected biological drift. Little use with one layer; inactive '
                'during the early coarse-grid phase or with No diffusion on.'},
    # Evidence (native/proseg/src): sampler/paramsampler.rs:277-325,67-123; main.rs:234
    {'flag': 'ncomponents',
     'label': 'Expression groups (Ncomponents)',
     'group': 'Expression model',
     'type': 'int',
     'default': 10,
     'help': 'Sets how many groups of cells can have different expected gene patterns and typical '
             'volumes in the model.',
     'advanced': True,
     'live': False,
     'min': 1,
     'what': 'Sets how many groups of cells can have different expected gene patterns and typical '
             'volumes in the model.',
     'up': 'The model can describe more kinds of cells, but it does not create more cell identities or '
           'predictably enlarge boundaries.',
     'down': 'More cells share the same gene-pattern and volume assumptions, with no single direction '
             'for their outlines.',
     'use_when': 'You are comparing expression models for tissue with different cell diversity, rather '
                 'than fixing one visible boundary.',
     'typical': 'Try 5 to 20 groups when the crop contains enough cells; default 10.',
     'caution': 'These are internal groups, not the displayed analysis clusters or a requested cell '
                'count. Small crops may not support many groups.'},
    # Evidence (native/proseg/src): sampler.rs:415-419; sampler/paramsampler.rs:468-519; main.rs:237-238
    {'flag': 'nhidden',
     'label': 'Shared gene patterns (Nhidden)',
     'group': 'Expression model',
     'type': 'int',
     'default': 200,
     'help': 'Sets how many shared gene patterns describe genes that are not modelled individually.',
     'advanced': True,
     'live': False,
     'min': 1,
     'what': 'Sets how many shared gene patterns describe genes that are not modelled individually.',
     'up': 'The model can represent more expression detail; cell outlines have no predictable size or '
           'smoothness change.',
     'down': 'The model uses fewer shared gene patterns, which can simplify expression differences '
             'between cells.',
     'use_when': 'You are comparing model detail and speed for a large gene panel.',
     'typical': 'Try 100 to 300 shared patterns; default 200, in addition to the individually modelled '
                'genes.',
     'caution': 'More patterns cost memory and computation. Ignored when No factorization is on; not a '
                'knob for directly rounding cells.'},
    # Evidence (native/proseg/src): sampler.rs:415-419; main.rs:808-824; sampler/featureselection.rs:249-310
    {'flag': 'nunfactored',
     'label': 'Individual genes (Nunfactored)',
     'group': 'Expression model',
     'type': 'int',
     'default': 200,
     'help': 'Sets how many selected genes are modelled individually, while the remaining genes share '
             'learned expression patterns.',
     'advanced': True,
     'live': False,
     'min': 0,
     'what': 'Sets how many selected genes are modelled individually, while the remaining genes share '
             'learned expression patterns.',
     'up': 'More genes have their own expression estimates; the model may distinguish cells '
           'differently, without a fixed boundary direction.',
     'down': 'More genes are described through shared patterns; genes are still included, not '
             'discarded.',
     'use_when': 'You want to compare individual gene detail against a more shared expression model.',
     'typical': 'Try 100 to 300 genes for large panels; default 200.',
     'caution': 'Proseg chooses the genes. At or above the number of retained genes, factorization is '
                'disabled. This does not set how many genes appear in the viewer.'},
    # Evidence (native/proseg/src): main.rs:808-824,970; sampler.rs:415-419
    {'flag': 'no-factorization',
     'label': 'No factorization',
     'group': 'Expression model',
     'type': 'bool',
     'default': False,
     'help': 'Models every gene individually instead of describing some genes through shared '
             'expression patterns.',
     'advanced': True,
     'live': False,
     'what': 'Models every gene individually instead of describing some genes through shared '
             'expression patterns.',
     'up': 'On: cell assignments use individual gene estimates; there is no reliable change towards '
           'larger, smaller or smoother cells.',
     'down': 'Off: for large panels, some genes share learned patterns that help describe expression '
             'across cells.',
     'use_when': 'You want a model comparison, or a small crop has trouble with the gene-selection '
                 'startup step.',
     'typical': 'On or off; default off, but automatically on when the retained gene count is no '
                'larger than Nunfactored.',
     'caution': 'Skips gene selection but may cost more during sampling; it is not guaranteed faster. '
                'The app can retry startup with this on after a failed initialization.'},
    # Evidence (native/proseg/src): main.rs:960; sampler/paramsampler.rs:264-274,530-561; sampler/voxelsampler.rs:433-436,478-481
    {'flag': 'use-scaled-cells',
     'label': 'Use scaled cells',
     'group': 'Expression model',
     'type': 'bool',
     'default': False,
     'help': 'Lets each cell have an extra overall expression multiplier, so transcript abundance is '
             'not explained by cell volume alone.',
     'advanced': True,
     'live': False,
     'what': 'Lets each cell have an extra overall expression multiplier, so transcript abundance is '
             'not explained by cell volume alone.',
     'up': 'On: equally sized cells can differ more in overall transcript abundance without relying on '
           'volume changes alone.',
     'down': 'Off: the extra multiplier stays at one, leaving volume and gene-specific rates to '
             'explain the counts.',
     'use_when': 'You are testing whether overall expression differences between cells affect the '
                 'model.',
     'typical': 'On or off; default off.',
     'caution': 'Mainly an expression-model choice. The code does not support a reliable prediction '
                'that cells will grow or shrink.'},
    # Evidence (native/proseg/src): main.rs:953-957; sampler/paramsampler.rs:248-256
    {'flag': 'variable-burnin-dispersion',
     'label': 'Variable burnin dispersion',
     'group': 'Expression model',
     'type': 'bool',
     'default': False,
     'help': 'Lets proseg learn how much gene counts differ between similar cells during the early '
             'coarse-grid phase.',
     'advanced': True,
     'live': False,
     'what': 'Lets proseg learn how much gene counts differ between similar cells during the early '
             'coarse-grid phase.',
     'up': 'On: early expression variability adapts to the data; there is no fixed direction for cell '
           'outlines.',
     'down': 'Off: early expression variability uses Burnin dispersion until fine sampling starts.',
     'use_when': 'You want to compare a learned early expression model with the simpler fixed starting '
                 'assumption.',
     'typical': 'On or off; default off.',
     'caution': 'An explicit Dispersion value overrides both modes. Usually leave this alone when '
                'diagnosing a visible shape problem.'},
    # Evidence (native/proseg/src): sampler/paramsampler.rs:248-256,306-321,510-519; sampler/math.rs:58-69
    {'flag': 'burnin-dispersion',
     'label': 'Burnin dispersion',
     'group': 'Expression model',
     'type': 'float',
     'default': 1.0,
     'help': 'Sets early variation in gene counts among similar cells. For the same average count, '
             'larger values allow less extra variation between cells.',
     'advanced': True,
     'live': False,
     'min': 0.001,
     'max': 1000.0,
     'what': 'Sets early variation in gene counts among similar cells. For the same average count, '
             'larger values allow less extra variation between cells.',
     'up': 'Early cells are judged against more similar gene-count patterns; there is no guaranteed '
           'direction for boundary size.',
     'down': 'Early cells can differ more in gene counts within a group; boundaries do not necessarily '
             'become larger or smaller.',
     'use_when': 'You are comparing early expression assumptions, rather than directly controlling '
                 'cell shape.',
     'typical': 'Try 0.5 to 2 for model comparisons; default 1, with no units.',
     'caution': 'Ignored when Variable burnin dispersion is on or Dispersion is fixed. Changing it '
                'also affects expected expression while the model readjusts.'},
    # Evidence (native/proseg/src): sampler/paramsampler.rs:248-256,306-321,510-519; sampler/math.rs:58-69; main.rs:1628
    {'flag': 'dispersion',
     'label': 'Expression variability (dispersion)',
     'group': 'Expression model',
     'type': 'float',
     'default': None,
     'help': 'Fixes how much gene counts may differ between similar cells instead of learning that '
             'variation. For the same average count, larger values allow less extra variation.',
     'advanced': True,
     'live': True,
     'min': 0.001,
     'max': 1000.0,
     'what': 'Fixes how much gene counts may differ between similar cells instead of learning that '
             'variation. For the same average count, larger values allow less extra variation.',
     'up': 'The expression model expects more similar gene counts within a group, with no reliable '
           'larger-or-smaller cell effect.',
     'down': 'The model allows greater gene-count differences within a group, with no reliable '
             'smoothing effect.',
     'use_when': 'You deliberately want to compare expression variability assumptions; use shape knobs '
                 'for spiky outlines.',
     'typical': 'Default is learned automatically. If fixing it for comparison, try 0.5 to 2, with no '
                'units.',
     'caution': 'Overrides Burnin dispersion too. The expression average also readjusts, so this is '
                'not a direct size knob. After setting it live, returning to automatic learning '
                'requires a new run with this field unset.'},
    # Evidence (native/proseg/src): main.rs:986; sampler/paramsampler.rs:616-628
    {'flag': 'hyperparam-e-phi',
     'label': 'Hyperparam e phi',
     'group': 'Expression model',
     'type': 'float',
     'default': 1.0,
     'help': 'Sets part of the starting preference for learned expression variability: how similar the '
             'gene counts of related cells should be before the data dominate.',
     'advanced': True,
     'live': False,
     'min': 1e-06,
     'what': 'Sets part of the starting preference for learned expression variability: how similar the '
             'gene counts of related cells should be before the data dominate.',
     'up': 'Favours higher dispersion values, meaning less extra variation at the same average gene '
           'count; cell outlines have no predictable direction.',
     'down': 'Favours lower dispersion values, allowing more extra variation at the same average gene '
             'count.',
     'use_when': 'You are making a planned comparison of expression-model assumptions, not correcting '
                 'a visible outline.',
     'typical': 'Usually keep 1, the default; exploratory comparisons may use 0.5 to 2, with no units.',
     'caution': 'Has no effect while dispersion is fixed. The fitted expression averages also change, '
                'so there is no guaranteed boundary or size outcome.'},
    # Evidence (native/proseg/src): main.rs:987; sampler/paramsampler.rs:616-628
    {'flag': 'hyperparam-f-phi',
     'label': 'Hyperparam f phi',
     'group': 'Expression model',
     'type': 'float',
     'default': 1.0,
     'help': 'Sets another part of the starting preference for learned expression variability, '
             'controlling how large learned dispersion values tend to be.',
     'advanced': True,
     'live': False,
     'min': 1e-06,
     'what': 'Sets another part of the starting preference for learned expression variability, '
             'controlling how large learned dispersion values tend to be.',
     'up': 'Favours higher dispersion values, meaning less extra variation at the same average gene '
           'count; no fixed visual boundary change follows.',
     'down': 'Favours lower dispersion values, allowing more extra variation at the same average gene '
             'count.',
     'use_when': 'You are testing expression assumptions with a planned comparison, rather than trying '
                 'to resize cells.',
     'typical': 'Usually keep 1, the default; exploratory comparisons may use 0.5 to 2, with no units.',
     'caution': 'Has no effect while dispersion is fixed. This is a scale, not its reciprocal; raising '
                'it favours higher dispersion values.'},
    # Evidence (native/proseg/src): main.rs:989; sampler/paramsampler.rs:708-724
    {'flag': 'hyperparam-neg-mu-phi',
     'label': 'Hyperparam neg mu phi',
     'group': 'Expression model',
     'type': 'float',
     'default': 1.0,
     'help': 'Sets the preferred expression scale, a multiplier of expected gene activity, for '
             'groups of cells before their observed counts are considered.',
     'advanced': True,
     'live': False,
     'what': 'Sets the preferred expression scale, a multiplier of expected gene activity, for '
             'groups of cells before their observed counts are considered.',
     'up': 'Favours lower expression scales; there is no reliable prediction of bigger or smaller cell '
           'outlines after the model readjusts.',
     'down': 'Favours higher expression scales, without a reliable direction for cell outlines.',
     'use_when': 'You are testing a different starting assumption about expression levels.',
     'typical': 'Usually keep 1, the default; exploratory comparisons may use 0.5 to 2, with no units.',
     'caution': 'The negative sign in the name matters: larger values favour lower scales. This acts '
                'on the expression model, not directly on cell size.'},
    # Evidence (native/proseg/src): main.rs:990; sampler/paramsampler.rs:680-724
    {'flag': 'hyperparam-tau-phi',
     'label': 'Hyperparam tau phi',
     'group': 'Expression model',
     'type': 'float',
     'default': 0.1,
     'help': 'Sets how tightly expression scales, the multipliers of expected gene activity in '
             'groups of cells, are held near their preferred starting level.',
     'advanced': True,
     'live': False,
     'min': 1e-06,
     'what': 'Sets how tightly expression scales, the multipliers of expected gene activity in '
             'groups of cells, are held near their preferred starting level.',
     'up': 'Expression scales are pulled more tightly towards the preferred level; outlines have no '
           'fixed direction of change.',
     'down': 'Expression scales have more freedom to follow the observed counts; outlines need not '
             'become larger or smaller.',
     'use_when': 'You are comparing the strength of expression-model assumptions.',
     'typical': 'Usually keep 0.1, the default; exploratory comparisons may use 0.05 to 0.2, with no '
                'units.',
     'caution': 'Whether fitted scales rise or fall depends on which side of the preferred level they '
                'were already on. Not a shape-smoothing setting.'},
    # Evidence (native/proseg/src): sampler/voxelcheckerboard.rs:2245-2284,3051-3054; sampler/paramsampler.rs:728-739
    {'flag': 'density-bandwidth',
     'label': 'Density bandwidth',
     'group': 'Expression model',
     'type': 'float',
     'default': 0.5,
     'help': 'Sets the distance over which nearby transcripts are pooled to estimate local crowding. '
             'That estimate helps the model distinguish cell signal from background.',
     'advanced': True,
     'live': False,
     'min': 1e-06,
     'what': 'Sets the distance over which nearby transcripts are pooled to estimate local crowding. '
             'That estimate helps the model distinguish cell signal from background.',
     'up': 'The background model uses a more spatially smoothed estimate of crowding; it does not '
           'directly smooth cell edges.',
     'down': 'The background model responds to more local changes in transcript crowding.',
     'use_when': 'You are comparing how uneven tissue density affects background assignments.',
     'typical': 'Try 0.25 to 1 µm for model comparisons; default 0.5 µm.',
     'caution': 'No reliable direction for the number of background transcripts or cell size. This map '
                'is built at startup and rebuilt at grid refinement.'},
    # Evidence (native/proseg/src): sampler/voxelcheckerboard.rs:2287-2335; sampler/paramsampler.rs:728-739; sampler.rs:408-412
    {'flag': 'density-bins',
     'label': 'Density bins',
     'group': 'Expression model',
     'type': 'int',
     'default': 5,
     'help': 'Sets how many levels of local transcript crowding get separate background-expression '
             'estimates.',
     'advanced': True,
     'live': False,
     'min': 1,
     'max': 16,
     'what': 'Sets how many levels of local transcript crowding get separate background-expression '
             'estimates.',
     'up': 'The model can treat more density levels differently; cells do not predictably expand or '
           'shrink.',
     'down': 'Background estimates are shared across more of the tissue; one bin shares them across '
             'all crowding levels.',
     'use_when': 'You want to compare background modelling in tissue with dense and sparse regions.',
     'typical': 'Try 3 to 8 groups; default 5. One group is a useful comparison; the supported maximum '
                'is 16.',
     'caution': 'More groups leave fewer observations per estimate. This is not a desired fraction of '
                'transcripts to keep as background.'},
    # Evidence (native/proseg/src): main.rs:519; sampler/transcripts.rs:1320-1322
    {'flag': 'min-qv',
     'label': 'Min qv',
     'group': 'Input filtering',
     'type': 'float',
     'default': 20.0,
     'help': 'Discards transcripts below a minimum Xenium quality score before cells are built or '
             'fitted.',
     'advanced': True,
     'live': False,
     'min': 0.0,
     'what': 'Discards transcripts below a minimum Xenium quality score before cells are built or '
             'fitted.',
     'up': 'Fewer transcripts are shown or counted, especially uncertain detections; weakly supported '
           'cell extensions may lose evidence.',
     'down': 'More uncertain detections enter the model and viewer, potentially adding noise around '
             'cells.',
     'use_when': 'You want to compare how uncertain detections affect boundaries or transcript counts.',
     'typical': 'Try quality scores 20 to 30; Xenium default 20.',
     'caution': 'Requires a new run. Removed transcripts are absent, not grey background points. '
                'Raising this may also remove nuclear evidence.'},
    # Evidence (native/proseg/src): main.rs:520-523,744; sampler/transcripts.rs:1324-1328
    {'flag': 'excluded-genes',
     'label': 'Excluded genes',
     'group': 'Input filtering',
     'type': 'str',
     'default': DEFAULT_EXCLUDED_GENES,
     'help': 'Removes transcripts whose gene names match a text pattern before cells are fitted. By default '
             'the control probes of every platform: Xenium\'s, CosMx\'s NegPrb and SystemControl, and '
             'MERSCOPE\'s Blank.',
     'advanced': True,
     'live': False,
     'what': 'Removes transcripts whose gene names match a text pattern before cells are fitted. By default '
             'the control probes of every platform: Xenium\'s, CosMx\'s NegPrb and SystemControl, and '
             'MERSCOPE\'s Blank.',
     'up': 'A broader matching pattern removes more genes and their transcripts from the model and '
           'viewer.',
     'down': 'A narrower matching pattern keeps more genes and transcripts, potentially including '
             'controls.',
     'use_when': 'You need to exclude known control probes or a clearly identified problematic gene.',
     'typical': f'Default pattern: {DEFAULT_EXCLUDED_GENES}. There is no numeric range.',
     'caution': 'The pattern is a regular expression, a rule for matching names, of at most '
                f'{MAX_PATTERN} characters. An empty pattern matches every gene and removes everything; use '
                'a deliberate non-matching pattern such as a^ to exclude none. Compare leaves out the same '
                'genes.'},
    # Evidence (native/proseg/src): main.rs:520-523; sampler/transcripts.rs:1324-1328
    {'flag': 'include-neg-ctrls',
     'label': 'Include neg ctrls',
     'group': 'Input filtering',
     'type': 'bool',
     'default': False,
     'help': 'Allows normally excluded control and other default-filtered transcript names into the '
             'input unless your explicit exclusion pattern still removes them.',
     'advanced': True,
     'live': False,
     'what': 'Allows normally excluded control and other default-filtered transcript names into the '
             'input unless your explicit exclusion pattern still removes them.',
     'up': 'On: more control detections can appear and participate in the expression model.',
     'down': 'Off: the default control-name filter is used unless you supplied a different exclusion '
             'pattern.',
     'use_when': 'You are deliberately comparing the effect of control probes on the model.',
     'typical': 'On or off; default off.',
     'caution': 'Controls are not automatically treated as a special noise reference. An explicit '
                'Excluded genes pattern takes precedence; this is not a way to improve cell coverage.'},
    # Evidence (native/proseg/src): sampler/transcripts.rs:76-99,102-145; main.rs:803-805
    {'flag': 'max-transcript-nucleus-distance',
     'label': 'Max transcript nucleus distance',
     'group': 'Input filtering',
     'type': 'float',
     'default': 60.0,
     'help': 'Drops transcripts too far from every starting cell centre. These centres are calculated '
             'from Xenium-assigned transcripts, not distances to the nuclear edge.',
     'advanced': True,
     'live': False,
     'min': 0.0,
     'what': 'Drops transcripts too far from every starting cell centre. These centres are calculated '
             'from Xenium-assigned transcripts, not distances to the nuclear edge.',
     'up': 'More distant transcripts remain available to cells or background, including evidence in '
           'extended cytoplasm.',
     'down': 'Distant transcripts disappear from the input, reducing evidence far from starting cells.',
     'use_when': 'Long cell extensions appear to lack transcripts because a distance filter removed '
                 'them.',
     'typical': 'Try 40 to 80 µm; default 60 µm.',
     'caution': 'Requires a new run. Applies even to assigned transcripts. It is not a maximum cell '
                'radius and cannot recover transcripts outside the crop.'},
    # Evidence (native/proseg/src): sampler/transcripts.rs:1360-1370
    {'flag': 'non-unique-cell-ids',
     'label': 'Non unique cell ids',
     'group': 'Input filtering',
     'type': 'bool',
     'default': False,
     'help': 'Treats the same cell identifier in different microscope fields as separate starting '
             'cells.',
     'advanced': True,
     'live': False,
     'what': 'Treats the same cell identifier in different microscope fields as separate starting '
             'cells.',
     'up': 'On: repeated identifiers in different fields are separated rather than combined.',
     'down': 'Off: the same identifier across fields represents one cell.',
     'use_when': 'The input really reuses cell identifiers for different cells in different fields.',
     'typical': 'On or off; default off for Xenium.',
     'caution': 'Not a remedy for merged biological nuclei. Turning it on for globally unique Xenium '
                'identifiers that span fields can wrongly split their evidence.'},
    # Evidence (native/proseg/src): main.rs:628-635; sampler/voxelsampler.rs:98-113
    {'flag': 'nthreads',
     'label': 'CPU threads',
     'group': 'Performance',
     'type': 'int',
     'default': None,
     'help': 'Sets how many computer workers can process the run at once. It changes speed rather than '
             'the intended cell model.',
     'advanced': False,
     'live': False,
     'min': 1,
     'what': 'Sets how many computer workers can process the run at once. It changes speed rather than '
             'the intended cell model.',
     'up': 'Iterations may arrive faster if enough independent work is available; cell shape has no '
           'intended direction of change.',
     'down': 'The run uses fewer processor resources, often with slower iterations.',
     'use_when': 'You need to leave computing capacity for other work or compare run speed.',
     'typical': 'Try 1 to 8 threads, within available cores. Proseg defaults to all cores; the app '
                'supplies its configured thread count.',
     'caution': 'More threads are not always faster. Random sampling means repeated runs can differ '
                'even with the same thread count.'},
    # Evidence (native/proseg/src): main.rs:1581-1587
    {'flag': 'check-consistency',
     'label': 'Check consistency',
     'group': 'Debug',
     'type': 'bool',
     'default': False,
     'help': 'Runs extra internal checks that transcript counts and cell geometry remain consistent.',
     'advanced': True,
     'live': False,
     'what': 'Runs extra internal checks that transcript counts and cell geometry remain consistent.',
     'up': 'On: iterations are slower and internal errors may stop the run; no intended change in '
           'cells.',
     'down': 'Off: skips these extra checks, with no intended change in cell shapes or counts.',
     'use_when': 'You are investigating a reproducible software problem.',
     'typical': 'On or off; default off.',
     'caution': 'A debugging setting, not a biological quality score or accuracy improvement.'},
    # Evidence (native/proseg/src): main.rs:1519; spatialdata_output.rs:111-123
    {'flag': 'exclude-spatialdata-transcripts',
     'label': 'Exclude spatialdata transcripts',
     'group': 'Debug',
     'type': 'bool',
     'default': False,
     'help': 'Omits individual transcript points from the saved SpatialData file while keeping the '
             'segmentation and cell table.',
     'advanced': True,
     'live': False,
     'what': 'Omits individual transcript points from the saved SpatialData file while keeping the '
             'segmentation and cell table.',
     'up': 'On: saved output contains no transcript point layer; cells in the live view are unchanged.',
     'down': 'Off: the SpatialData output includes transcript points.',
     'use_when': 'You want a smaller final export and do not need its transcript point layer.',
     'typical': 'On or off; default off.',
     'caution': 'Only affects export, not fitting, live transcript display, or the separate '
                'transcript-metadata file.'},
    # Evidence (native/proseg/src): main.rs:380,1381-1385; output.rs:114-136 (commented out)
    {'flag': 'output-counts-fmt',
     'label': 'Output counts fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Has no effect in this proseg version: cell counts are always saved in the existing '
             'count-file format.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Has no effect in this proseg version: cell counts are always saved in the existing '
             'count-file format.',
     'up': 'Choosing another format does not change cells, counts, or the saved count file.',
     'down': 'Keeping infer produces the same output as the other selections.',
     'use_when': 'Leave this unchanged; it cannot alter segmentation or choose the count-file format '
                 'here.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'This setting is accepted but not used when proseg saves counts.'},
    # Evidence (native/proseg/src): main.rs:388-391,1372-1522; output.rs:296-323 (writer not called)
    {'flag': 'output-rates-fmt',
     'label': 'Output rates fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Has no effect in this app: this proseg version does not save a separate table of '
             'estimated gene activity through this setting.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Has no effect in this app: this proseg version does not save a separate table of '
             'estimated gene activity through this setting.',
     'up': 'Choosing another format does not change cells or produce an expression-rate table.',
     'down': 'Keeping infer has the same result.',
     'use_when': 'Leave this unchanged; it does not change which transcripts count for cells.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'Changing it neither produces that table nor changes the segmentation.'},
    # Evidence (native/proseg/src): main.rs:1398-1405; output.rs:485-492
    {'flag': 'output-cell-metadata-fmt',
     'label': 'Output cell metadata fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Chooses the file format for the cell information table when it is exported. It does not '
             'change cells or their transcript assignments.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Chooses the file format for the cell information table when it is exported. It does not '
             'change cells or their transcript assignments.',
     'up': 'There is no higher setting; choosing another format only changes how the table is stored.',
     'down': 'There is no lower setting; infer chooses a format from the output filename.',
     'use_when': 'Your downstream software needs a particular table format.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'Keep infer so the format matches the filename chosen by the app.'},
    # Evidence (native/proseg/src): main.rs:1431-1439; output.rs:513-516
    {'flag': 'output-transcript-metadata-fmt',
     'label': 'Output transcript metadata fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Chooses the file format for the transcript information table when it is exported. It '
             'does not change cells or their transcript assignments.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Chooses the file format for the transcript information table when it is exported. It '
             'does not change cells or their transcript assignments.',
     'up': 'There is no higher setting; choosing another format only changes how the table is stored.',
     'down': 'There is no lower setting; infer chooses a format from the output filename.',
     'use_when': 'Your downstream software needs a particular table format.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'Keep infer so the format matches the filename chosen by the app.'},
    # Evidence (native/proseg/src): main.rs:1409-1418; output.rs:783-790
    {'flag': 'output-gene-metadata-fmt',
     'label': 'Output gene metadata fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Chooses the file format for the gene information table when it is exported. It does not '
             'change cells or their transcript assignments.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Chooses the file format for the gene information table when it is exported. It does not '
             'change cells or their transcript assignments.',
     'up': 'There is no higher setting; choosing another format only changes how the table is stored.',
     'down': 'There is no lower setting; infer chooses a format from the output filename.',
     'use_when': 'Your downstream software needs a particular table format.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'Only matters when this optional table is requested; the app does not request it '
                'separately.'},
    # Evidence (native/proseg/src): main.rs:1422-1427; output.rs:218-249
    {'flag': 'output-metagene-rates-fmt',
     'label': 'Output metagene rates fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Chooses the file format for the shared gene-pattern activity table when it is exported. '
             'It does not change cells or their transcript assignments.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Chooses the file format for the shared gene-pattern activity table when it is exported. '
             'It does not change cells or their transcript assignments.',
     'up': 'There is no higher setting; choosing another format only changes how the table is stored.',
     'down': 'There is no lower setting; infer chooses a format from the output filename.',
     'use_when': 'Your downstream software needs a particular table format.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'Only matters when this optional table is requested; the app does not request it '
                'separately.'},
    # Evidence (native/proseg/src): main.rs:1444-1450; output.rs:250-293
    {'flag': 'output-metagene-loadings-fmt',
     'label': 'Output metagene loadings fmt',
     'group': 'Debug',
     'type': 'str',
     'default': 'infer',
     'help': 'Chooses the file format for the gene contributions to shared expression patterns table '
             'when it is exported. It does not change cells or their transcript assignments.',
     'choices': ['infer', 'csv', 'csv-gz', 'parquet'],
     'advanced': True,
     'live': False,
     'what': 'Chooses the file format for the gene contributions to shared expression patterns table '
             'when it is exported. It does not change cells or their transcript assignments.',
     'up': 'There is no higher setting; choosing another format only changes how the table is stored.',
     'down': 'There is no lower setting; infer chooses a format from the output filename.',
     'use_when': 'Your downstream software needs a particular table format.',
     'typical': 'Choices: infer, csv, csv-gz, parquet; default infer. No numeric range.',
     'caution': 'Only matters when this optional table is requested; the app does not request it '
                'separately.'},
]

# Recommendations are conditional trials, not promises about final cell size.
# Evidence is beside the corresponding options above. No exposed parameter is a
# target cell area, and the sampler cannot create or merge cell identities.
GUIDE = [
    {
        'symptom': 'Cells are too big or swallowing neighbours',
        'explanation': 'There is no direct target-size knob. Compare the nuclear image and Xenium '
                       'outlines first. Try one change at a time: protect reliable nuclei, or '
                       'discourage thin extensions if those account for the excess area.',
        'actions': [
            {'flag': 'prior-weight', 'direction': 'raise', 'value': 2.0, 'live': True,
             'why': 'Helps cells respect reliable Xenium nuclear assignments with the default '
                    'reassignment settings. It does not impose a maximum cell size. In a run started '
                    'from cells only it holds the starting outlines instead, with any neighbouring RNA '
                    'already inside them.'},
            {'flag': 'cell-compactness', 'direction': 'lower', 'value': 0.02, 'live': True,
             'why': 'Discourages irregular extensions reaching into neighbours; it is not a '
                    'general shrink command.'},
            {'flag': 'no-diffusion', 'direction': 'off', 'value': False, 'live': True,
             'why': 'After the early coarse-grid phase, allows drifted transcripts to count for nearby cells without '
                    'requiring boundaries to reach them.'},
        ],
    },
    {
        'symptom': 'Cells are too small or do not cover the cytoplasm',
        'explanation': 'Check that the missing cytoplasm contains retained transcripts. If outlines '
                       'are rounded off, relax the shape preference. If they cling to an incorrect '
                       'starting assignment, reduce its influence. Neither change guarantees growth.',
        'actions': [
            {'flag': 'cell-compactness', 'direction': 'raise', 'value': 0.08, 'live': True,
             'why': 'Allows transcript-supported elongated cytoplasm and narrow extensions.'},
            {'flag': 'prior-weight', 'direction': 'lower', 'value': 0.5, 'live': True,
             'why': 'Gives transcript evidence more freedom where a strong Xenium assignment is '
                    'holding a wrong boundary in place. In a run started from cells only, this is what '
                    'lets proseg move the starting outlines (in one test on a sample started from cells '
                    'only, a lower weight cleared a neighbour\'s genes from borders).'},
            {'flag': 'morphology-steps-per-iter', 'direction': 'raise', 'value': 8192, 'live': True,
             'why': 'Gives slowly developing cells more boundary attempts each iteration; '
                    'iterations take longer and cells can also shrink.'},
            {'flag': 'max-transcript-nucleus-distance', 'direction': 'raise', 'value': 80.0,
             'live': False,
             'why': 'Only if the distance filter removed distant cytoplasmic transcripts: retain '
                    'them within 80 µm of a starting cell centre in a new run.'},
        ],
    },
    {
        'symptom': 'Ragged or spiky boundaries and thin protrusions',
        'explanation': 'Decide whether the extensions match the biology or just chase scattered '
                       'transcripts. A stronger shape preference discourages spikes; transcript '
                       'drift can explain nearby scattered signal without stretching the cell.',
        'actions': [
            {'flag': 'cell-compactness', 'direction': 'lower', 'value': 0.02, 'live': True,
             'why': 'Smaller values penalise irregular outlines more strongly in each depth layer.'},
            {'flag': 'no-diffusion', 'direction': 'off', 'value': False, 'live': True,
             'why': 'Enables transcript repositioning after the early coarse-grid phase, so a cell can count nearby '
                    'transcripts beyond its measured outline.'},
            {'flag': 'diffusion-probability', 'direction': 'raise', 'value': 0.3, 'live': True,
             'why': 'If far drift is wider than near drift and repositioning is enabled, this '
                    'allows more far-drifted transcripts. Check for mixing between neighbours.'},
        ],
    },
    {
        'symptom': 'Cells are too round for elongated cell types',
        'explanation': 'Use the tissue image and transcript pattern to check the expected shape. '
                       'A weaker preference for compact outlines allows real extensions to develop.',
        'actions': [
            {'flag': 'cell-compactness', 'direction': 'raise', 'value': 0.08, 'live': True,
             'why': 'Larger values allow less round outlines, including thin protrusions. '
                    'Inspect neighbouring cells for newly admitted noise.'},
        ],
    },
    {
        'symptom': 'A cell appears in several pieces',
        'explanation': 'Inspect individual depth layers and the 3-D view: pieces may connect through '
                       'depth. Connectivity checks can prevent further breaks, but cannot repair '
                       'existing pieces. For prevention from the start, use the setting in a new run '
                       'and inspect the starting cells too.',
        'actions': [
            {'flag': 'enforce-connectivity', 'direction': 'on', 'value': True, 'live': True,
             'why': 'Blocks local changes that would disconnect cells. Existing fragments stay '
                    'until other accepted boundary changes happen to connect them.'},
        ],
    },
    {
        'symptom': 'Holes or background gaps appear inside cells',
        'explanation': 'Inspect depth layers to distinguish real gaps from the flat projection. '
                       'Gap proposals are one way background pockets arise. Reducing them changes '
                       'which moves are tried; it does not directly fill existing holes.',
        'actions': [
            {'flag': 'ab-nihlo-bubble-prob', 'direction': 'lower', 'value': 0.01, 'live': True,
             'why': 'Tries fewer moves that turn a block into background. Accepted gaps still '
                    'depend on transcript evidence and shape preferences.'},
        ],
    },
    {
        'symptom': 'Many transcripts are left as background',
        'explanation': 'Some background is expected; assigning every transcript is not the goal. '
                       'If grey points sit just outside plausible cell edges, test drift or allow '
                       'real extensions. A transcript inside a boundary can still be modelled as '
                       'background signal. Filtered-out transcripts are absent, not grey.',
        'actions': [
            {'flag': 'no-diffusion', 'direction': 'off', 'value': False, 'live': True,
             'why': 'Lets nearby cells count drifted transcripts after the early coarse-grid phase without expanding '
                    'their outlines to every measured point.'},
            {'flag': 'diffusion-sigma-far', 'direction': 'raise', 'value': 6.0, 'live': True,
             'why': 'If repositioning is enabled and the far-drift share is above zero, allows '
                    'a wider typical drift distance of 6 µm. Check that neighbours stay distinct.'},
            {'flag': 'cell-compactness', 'direction': 'raise', 'value': 0.08, 'live': True,
             'why': 'If the points mark real elongated cytoplasm, allows boundaries to follow '
                    'that shape. It will not force noisy transcripts into cells.'},
        ],
    },
    {
        'symptom': 'Boundaries keep flickering; I want to settle before Finish & save',
        'explanation': 'Movement during ordinary sampling is expected. Once the overall cells look '
                       'plausible, lower Temperature, then Resume or Step through more iterations. '
                       'Watch several updates until outlines look stable, Pause and inspect, then '
                       'Finish & save. Finish does not perform extra settling. Cooling does not '
                       'stop transcript repositioning or expression updates, so some movement can remain.',
        'actions': [
            {'flag': 'temperature', 'direction': 'lower', 'value': 0.1, 'live': True,
             'why': 'Makes worse-fitting boundary changes less likely, as in the hill-climb '
                    'phase. It does not strictly restrict changes to improvements.'},
            {'flag': 'temperature', 'direction': 'lower', 'value': 0.01, 'live': True,
             'why': 'A stronger settling trial if 0.1 still leaves distracting movement. '
                    'Let further iterations run before saving.'},
        ],
    },
    {
        'symptom': 'Xenium merged nuclei or split one nucleus into several cells',
        'explanation': 'No knob can create a missing cell identity or merge two starting identities. '
                       'Correct the Xenium starting segmentation outside this app, then start a '
                       'new run using the corrected input. If the identities are correct but their '
                       'boundaries are misplaced, reducing Prior weight can relax those boundaries; '
                       'it does not fix the number of cells.',
        'actions': [],
    },
    {
        'symptom': 'I want cells to stay close to the Xenium segmentation',
        'explanation': 'Proseg uses Xenium transcript assignments to build preferences for small '
                       'grid blocks; it does not trace the displayed polygons directly. With '
                       'defaults, nuclei are favoured but assignments outside nuclei are neutral. '
                       'Following the cell bodies more closely therefore needs a new run. A run '
                       'started from cells only (no nuclei) is different: every transcript inside a '
                       'starting cell counts as nuclear, so the grid block holding it is held to that '
                       'cell and nothing is neutral.',
        'actions': [
            {'flag': 'prior-weight', 'direction': 'raise', 'value': 2.0, 'live': True,
             'why': 'Strengthens existing non-neutral preferences, including the default nuclear '
                    'preference. It cannot strengthen neutral cell-body preferences; in a run started '
                    'from cells only, where none is neutral, it holds the whole starting outline more '
                    'firmly.'},
            {'flag': 'prior-seg-reassignment-prob', 'direction': 'lower', 'value': 0.2,
             'live': False,
             'why': 'In a new run, gives non-nuclear grid blocks an 80 percent preference for '
                    'their original Xenium cell, instead of the neutral default.'},
            {'flag': 'nuclear-reassignment-prob', 'direction': 'lower', 'value': 0.1,
             'live': False,
             'why': 'In a new run, gives nuclear grid blocks a 90 percent preference for their '
                    'original Xenium cell. Use only if that assignment looks reliable.'},
        ],
    },
    {
        'symptom': 'I want each iteration to move boundaries more',
        'explanation': 'Boundary attempts control how much work occurs between viewer updates. '
                       'More attempts give more opportunities for movement, but cannot force '
                       'changes that the model rejects.',
        'actions': [
            {'flag': 'morphology-steps-per-iter', 'direction': 'raise', 'value': 8192, 'live': True,
             'why': 'Tries more boundary changes each iteration. Each iteration takes longer; '
                    'this does not make the computer finish an iteration faster.'},
        ],
    },
    {
        'symptom': 'I want each iteration to move boundaries less',
        'explanation': 'Fewer boundary attempts make changes more gradual between displayed '
                       'iterations. This does not settle the model in the way cooling does.',
        'actions': [
            {'flag': 'morphology-steps-per-iter', 'direction': 'lower', 'value': 1024, 'live': True,
             'why': 'Tries fewer boundary changes each iteration, making individual steps easier '
                    'to inspect. More iterations may be needed overall.'},
        ],
    },
]

_BY_FLAG = {o['flag']: o for o in OPTIONS}

def _validate(values, live=False):
    if not isinstance(values, dict):
        raise ValueError("options must be a dict")
    for flag, value in values.items():
        if flag not in _BY_FLAG:
            reason = " (incompatible with Xenium parquet in proseg 3.2.0)" if flag == 'use-cell-initialization' else ''
            raise ValueError(f"Unknown flag {flag}{reason}")
        option = _BY_FLAG[flag]
        if live and not option['live']:
            raise ValueError(f"{flag} is startup-only")
        typ = option['type']
        valid = (type(value) is bool if typ == 'bool' else
                 type(value) is int if typ == 'int' else
                 type(value) in (int, float) if typ == 'float' else isinstance(value, str))
        if not valid:
            raise ValueError(f"{flag} must be {typ}")
        if typ in ('int', 'float'):
            if abs(value) > (3.4028234e38 if typ == 'float' else 2**63-1) or not math.isfinite(value):
                raise ValueError(f"{flag} must be finite and representable")
            for bound, op in [('min', lambda a,b: a < b), ('max', lambda a,b: a > b)]:
                if bound in option and op(value, option[bound]):
                    raise ValueError(f"{flag} must satisfy {bound}={option[bound]}")
        if 'choices' in option and value not in option['choices']:
            raise ValueError(f"{flag} must be one of {option['choices']}")
        if typ == 'str' and any(c in value for c in '\n\r\0'):
            raise ValueError(f"{flag} must be a single line")
        if typ == 'str' and len(value) > MAX_PATTERN:
            raise ValueError(f"{flag} must be at most {MAX_PATTERN} characters")
        if flag == 'excluded-genes':
            # Compare reads it with Python's re (zarr_bridge/helper.py), proseg with Rust's regex: a pattern Compare
            # refuses (not one Python reads, one that reads differently there, one that could take minutes over a gene
            # name) would let the run start, then fail every comparison of it. Refused here by the same rule, the
            # helper's own (_excluded), in its words.
            from ..zarr_bridge.helper import HelperError, _excluded
            try:
                _excluded(value)
            except HelperError as exc:
                raise ValueError(str(exc)) from None
    if not live:
        merged = {o['flag']: o['default'] for o in OPTIONS} | values
        if merged['recorded-samples'] > merged['samples']:
            raise ValueError('recorded-samples must be <= samples')
        ratio = merged['burnin-voxel-size'] / merged['voxel-size']
        if ratio < 1 or abs(ratio - round(ratio)) > 1e-5:
            raise ValueError('voxel-size must be <= burnin-voxel-size with an integer ratio')

def to_args(values: dict) -> list[str]:
    """Validate explicit values and emit arguments in schema order; omit unset defaults."""
    _validate(values)
    result = []
    for option in OPTIONS:
        flag = option['flag']
        if flag not in values:
            continue
        value = values[flag]
        if option['type'] == 'bool':
            if value:
                result.append('--' + flag)
        else:
            result.extend(['--' + flag, str(value)])
    return result

def input_args(values: dict) -> list[str]:
    """What every run passes after the user's options (engine/runs.py _command): the app's default excluded genes
    (DEFAULT_EXCLUDED_GENES), unless the user set Excluded genes or let the controls in (Include neg ctrls), so the
    options' own default is what proseg leaves out (proseg's --xenium default leaves out Xenium's controls only)."""
    if 'excluded-genes' in values or values.get('include-neg-ctrls'):
        return []
    return ['--excluded-genes', DEFAULT_EXCLUDED_GENES]

def live_lines(values: dict) -> list[str]:
    """Validate a partial live update and serialize control-file set lines."""
    _validate(values, live=True)
    return [f"set {o['flag']} {str(values[o['flag']]).lower()}"
            for o in OPTIONS if o['flag'] in values]
