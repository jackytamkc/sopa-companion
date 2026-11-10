# SoPA companion: architecture

How the app is built, as it is now: its processes, how data flows between them, the HTTP API, the files it writes and
the page's modules. It is used from a notebook's session (API mode: `sopa_companion.launch(sdata)` starts a private
engine for one SpatialData, see [A notebook's session](#a-notebooks-session-api-mode)); the same engine also runs on
its own (one engine per user, in an app window of its own), as the tests and the development harness drive it
([DEVELOPING.md](DEVELOPING.md)). The README says what the app does and how to install it.

## Components

```
 browser page (frontend/)  ──HTTP + WebSocket, loopback──▶  engine (python -m sopa_companion serve: FastAPI, the Python it was started with)
                                                             │
         ┌───────────────────────────────┬───────────────────┼─────────────────────┬────────────────────────────┐
         ▼                               ▼                   ▼                     ▼                            ▼
 proseg-live (Rust)              sopa helper            analysis worker      mixing-check trainer       stain helpers
 proseg-live package; native/bin zarr_bridge/helper.py  analysis/worker.py   analysis/qc_trainer.py     zarr_bridge/stain*.py
 one process per active run      one call per command   one long-lived       one process per fit        one per stain run;
 snapshots + control file        (sopa environment)     (GPU with RAPIDS,    (sopa env, GPU 0); its     patch helper; InstanSeg
                                                        else CPU)            probe at engine start
```

| Process | Interpreter | Started by | Lives | Talks through |
|---|---|---|---|---|
| Engine | the Python `sopa-companion` is installed in (this workstation: the app's `.venv`) | `cli.py` (`serve`, or `launch` in the background; the `sopa-companion` command is the same) | until the app window closes or `stop` | HTTP, WebSocket |
| Engine of a notebook's session | the kernel's Python, or `[api] engine_python` | `api/handle.py` (`serve --config <session.toml>`) | until the session closes or the kernel ends | HTTP, WebSocket; JSON lines with the kernel |
| proseg-live | native program: the proseg-live package's (`native/bin/` in a source folder) | `engine/runs.py` | one active run | its run folder: `live/` files and `live/control`; the run's folder in the counts spool |
| sopa helper | `[zarr] python` (`auto`: the engine's own) | `engine/zarr_samples.py` | one command | argv JSON in, progress lines on stderr, one JSON result on stdout |
| Analysis worker (GPU with RAPIDS, else CPU) | `[analysis] python` (`auto`: the engine's own when it has scipy, scikit-learn, igraph, umap-learn and numba, as `pip install sopa` brings) | `engine/analysis.py` | started with the first active run (again after a failure; never again in a session whose Python cannot run it) | JSON lines on stdin and stdout |
| Trainer | `[qc] python` (`auto`: the engine's own when it has scvi-tools, scviva-tools and torch) | `engine/qc.py` | one fit | a JSON job on stdin, JSON events on stdout |
| Mixing check's probe | `[qc] python` (the engine's own when `auto` found it lacking) | `engine/qc.py` | about a second, at the engine's start; again after no answer or at a press | one JSON line on stdout |
| Stain helper | `[stain] python` (`auto`: the engine's own when it has Cellpose) | `engine/stain.py` | one stain run (paused and resumed) | `stain/job.json`, JSON events on stdout |
| InstanSeg worker (dormant: only while the internal switch `[stain] instanseg_enabled` is on) | `[stain] instanseg_python` | the stain helper | one stain run | one tile at a time through `.npy` files |
| Patch helper | `[stain] patch_python`, else `[zarr] python` (`auto`: the engine's own) | `engine/stain_worker.py` | long-lived, CPU only; ends when its stdin closes | JSON lines |
| Shape queries | the engine's own Python | `engine/shape_query.py` | two worker processes | JSON lines |
| Cellpose GUI | `[stain] python` (`auto`: the engine's own when it has Cellpose) | `engine/stain_patches.py` | detached, outlives the engine | the patch folder chosen at export; it starts in its cache folder |

Every helper but the Cellpose GUI runs at nice 19 with a few threads and without a display, sees only its own GPU
(`CUDA_VISIBLE_DEVICES`, PCI bus order) and ends with the engine (`setpriv --pdeathsig` on Linux; on macOS and
Windows the engine's reaper, a small process that ends every child it was told about once the engine is gone,
`portable.tie`; or its closed stdin). Each runs in the engine's own Python unless a setting names another (`locate.py`: `"auto"`). An
installed copy's helpers in its own environment need nothing on `PYTHONPATH`; helpers in other environments, and
every helper of a source folder, import the app as `sopa_companion` from `PYTHONPATH`: the folder holding the app
when it is called `sopa_companion`, else a folder with one link to it (`engine/config.py` `helper_path`, under
`~/.cache/sopa-companion/pythonpath/`), never an environment's site-packages.

### The engine (Python)

| Module | Role |
|---|---|
| `cli.py` | commands (`sopa-companion`, the command an install puts on the path, or `python -m sopa_companion`; with no command, `launch`; `--version`); the default port 8777 falls back to the next free one up to 8876 |
| `dirs.py` | per-user folders: today's on Linux, platformdirs elsewhere; source tree or installed copy |
| `locate.py` | each helper's Python (`"auto"`, `"off"` or a path) and the proseg-live program |
| `server/app.py` | routes, WebSocket, lifespan (start-up sweep, shutdown order), telemetry log, closing answer |
| `server/api_serve.py` | a notebook's session: its lifespan, `/health`'s `session`, closing, the one sample it may open |
| `server/presence.py` | the pages signed in and how the last one went (a goodbye: closed; none: away), and the watcher that shuts the engine down or pauses its runs |
| `server/security.py` | Host and Origin checks, the page key (a header, or the socket's handshake), bearer token, one-use launch codes and their sign-in, the answers' security headers and the page's strict Content-Security-Policy |
| `server/runtime_file.py` | the private runtime file (pid, port, token), one engine per state folder, the app's `serve` command (`-m` or the console script), and the data folder's engine lock |
| `engine/config.py` | settings (an installed copy: `config.example.toml`, your `~/.config/sopa-companion/config.toml`, `--config`; a source folder: `config.toml` else `config.example.toml`, `config.local.toml`, `--config`); `"auto"` resolved through `locate.py` and `dirs.py` |
| `engine/session.py` | a notebook's session: `session.toml`, the kernel's control lines and the events pipe, its answer to closing |
| `engine/children.py` | how helpers start: the GPU each sees, its environment, `PYTHONPATH` (none for an installed copy's own environment), working folder, `setpriv` |
| `portable.py` | what differs between Linux, macOS and Windows: children that end with their starter (setpriv there, a reaper here), process probes and groups, flock and LockFileEx, the events pipe's handle, select on pipes, files replaced while another process reads them, owners without uids, folder links (junctions), console windows, priority |
| `engine/crashlog.py` | crash reports of sessions that ended unexpectedly, and the page logs kept for them |
| `engine/zarr_samples.py` | SpatialData samples: open (helper `inspect`), detail and notes, exports, priors, counts, saves |
| `engine/samples.py` | images and crops of Xenium output bundles (runs made from bundles by older versions) |
| `engine/runs.py` | runs: create, prepare, launch proseg-live, index frames and proseg's counted frames, control (counts and where tokens too), finish, save to the zarr, leave (save or discard), close; runs from earlier versions |
| `engine/counts_spool.py` | the spool of proseg's own count: the folder per engine and per run where the counted frames the engine asks for wait, its caps, pruning and sweep |
| `engine/tiles.py`, `transcript_index.py`, `vectors.py` | whole-sample tile containers, a spatial index of transcripts, prior vectors |
| `engine/shape_query.py` | box queries over a sample's exported shapes in worker processes |
| `engine/proseg_options.py` | proseg 3.2.0's option schema: labels, help, bounds, live knobs, guide |
| `engine/analysis.py` | the analysis worker's scheduler: clusters and UMAP, the mixing check's passes, Go deeper, Compare |
| `engine/qc.py` | mixing check: `QcModels` (learning each sample) and `QcManager` (each run's results and summaries) |
| `engine/qc_rules.py`, `qc_areas.py`, `qc_constants.py` | suggestion rules, trials, steady verdicts, thresholds; flagged areas; the thresholds' values for each kind of run |
| `engine/qc_test.py`, `qc_pool.py`, `qc_scoring.py`, `qc_readout.py`, `qc_rule.py` | Test a setting here: its reasons (refusals, waits, caveats) and the cell's pool (Go deeper's scoring in Python), fresh runs as now and with the value, each run's readout, the measured rule |
| `engine/deeper.py`, `deeper_where.py`, `held.py` | Go deeper: answers and their cache; where a cell's transcripts sit; proseg's share of each transcript for a cell (its where exports) |
| `engine/compare.py` | Compare: the area all columns cover, counting, scoring, results |
| `engine/stain.py`, `stain_patches.py`, `patch_folders.py`, `stain_worker.py`, `stain_exclusions.py`, `stain_rules.py`, `stain_intensity.py`, `stain_levels.py`, `cellpose_settings.py` | stain runs, training patches, the folders they are exported into and the GUI hand-off, removed cells and their rules, brightness, pyramid levels |
| `engine/runlog.py` | the Log button's text |
| `engine/leftovers.py` | removes what deleted runs leave behind once nothing holds it |
| `engine/shipping.py` | the file lists a code hash is taken over (the Log names the code that runs) |
| `desktop/` | `launcher.py` (launch, stop, status), `browser.py` (where the app opens: the default browser as the desktop names it, or the Chrome app window; the private opening page; shared with a notebook's session), `chrome.py` (which Chrome, its command line), `window.py` (the Chrome window's profile, and whether it runs), `processes.py`, `logs.py`, `install.py`, `migrate.py` (the previous version's state folder and window profile, moved once) |
| `private.py` | folders and files of this user's alone (0700, 0600, never through a link), for the launcher and a notebook's kernel |

### proseg-live (Rust)

`native/proseg-live.patch` adds `src/live.rs` and hooks to proseg 3.2.0: after every sampler iteration, on the main
thread while no Rayon task runs, it snapshots the state (a writer thread compresses and commits it), applies the control
file, and honours pause, step, speed, finish, continue and live settings between iterations. `--live-hold` makes proseg
hold at the end of its schedule instead of finishing; `--live-tiled` writes tile containers for regions too large for
dense rasters. Snapshots never touch the sampler's random state. `--live-counts` (with `--live-counts-dir` and
`--live-counts-start`) adds proseg's own count of each cell, gene by gene, for the frames the engine asks for and the
states proseg stops at, and where exports of a box (which cell holds each pooled transcript there), written by a
second writer thread: see [Proseg's own count](#prosegs-own-count). Counting only reads the model: seeded runs with and
without it save byte-identical results.

### The page

Plain ES modules and raw WebGL2, served by the engine from `frontend/`; no build step and no package manager. See
[The page's modules](#the-pages-modules).

## Coordinates and labels

- **Working frame**: the intrinsic coordinate system of the sample's chosen transcripts (points) element, x right and y
  down (micrometres for Xenium). Shapes and images are mapped into it with spatialdata's transformations, as sopa's
  `to_intrinsic` does. A sample without transcripts works in one of its own coordinate systems instead. Every box,
  polygon and image placement in the API is in this frame.
- **Rasters**: a frame covers `[x0, x0 + nx·voxel) × [y0, y0 + ny·voxel)`; column i, row j is voxel (i, j). The voxel is
  the burn-in size (2 µm) until the "upscale" frame, then 1 µm.
- **z**: proseg normalises each run's transcript z (tilt removed, 1–99 % quantiles mapped to 0..1). A transcript's layer
  is `clamp(floor((z − z.grid_origin) / z.grid_spacing), 0, nz − 1)`.
- **Labels**: label = internal cell index + 1, 0 = background; a run's internal ids never change. `cells.txt` (and
  `GET /runs/{id}/cells`) gives each label's original cell id (line k names label k + 1). A cell's colour is a function of
  its label alone.

## Data flow

### Opening a sample

`POST /samples/open {path}` runs the helper's `inspect`: elements, their transformations into the working frame, the
candidate priors, the frame's unit (micrometres or pixels), what z holds, images left out and why, the gene column and
every points element. The engine keeps the result (`cache/zarr/<sample>/inspect.json`, made again when the zarr
changes) and answers the sample detail: transcripts (null without any), extent, overview, images and channels (contrast
defaults from a coarse level), shapes, tables, `prior_options`, `suggested_prior`, `limits.detail_um`, `tools`
(`["proseg", "stain"]`, `["stain"]` for images only, `["proseg"]` without an image it can show) and `notes` (`{code,
text, detail}`: `pixel_frame`, `several_points`, `skipped_images`, `z_index`, `image_only`, `default_image`,
`skipped_names`, `image_unreadable`, and the helper's own, such as `unassigned_token`; the page adds `image_only` or
`no_image` when the engine sends none for a sample offering one tool). A sample that cannot be opened answers one plain
line and is not kept in Recent. Recent (`samples.json`) lists the 30 samples opened last and keeps 300 known: a run's
sample opened again for the run is known without being listed. Without transcripts the frame is a coordinate system
the main image reaches (through any element: shapes, images and labels, read alike by every helper command). Tissue
found by detection is cached with the way it was found (`zarr_bridge/tissue.py` `DETECTION_VERSION`).

### A Proseg Live run

1. `POST /runs {sample, region, prior, options, source?}` (region `box`, `polygon` or `full`; prior `{cells, nuclei}`,
   each a shapes key or a transcript column) creates the run: `mode` "detail" when its bounding box fits
   `limits.detail_um` (1,200 µm), else "tiled". A proseg run on a sample without transcripts is refused (409).
2. **Preparing** (helper `prepare`): the region's transcripts with the prior assigned (sopa's point-in-polygon join for
   shapes), written as a Xenium-schema parquet; a prior without nuclei seeds every transcript inside a cell.
3. **Loading**: proseg-live reads its input, writes `info.json`, its voxel prior and frame 0, then holds (paused) until
   Resume or Step. The run detail carries `phase` (preparing, loading, sampling, paused, finishing, done, failed,
   stopped) and `eta_s` while it prepares and loads.
4. **Sampling**: each snapshot's file is written, then its line is appended to `frames.jsonl` (the commit point). The
   engine indexes new lines and pushes them to the page watching the run; the page fetches what it shows (a dense frame
   whole; a tiled frame only over the view, `/frames/{seq}/view`). The frames proseg counts (the ones the engine asks
   for, and frame 0, the upscale, pauses, the hold, finish and the end) each get a line in `counts.jsonl`, merged into
   their frame's record: [Proseg's own count](#prosegs-own-count).
5. **Control**: `POST /runs/{id}/control {action}` (`pause`, `resume`, `step`, `speed`, `set` live values, `finish`,
   `abort`) rewrites `live/control`; proseg-live acknowledges in `live/ack` (`paused` and `finishing` only once every
   queued frame is committed, so the acknowledged frame is exactly what a finish saves). The file is replaced whole; a
   reader that cannot read it keeps the lines it read last (on NTFS through ntfs-3g a replaced file is missing for a
   moment), as the stain helper does with its `control.json`.
6. **The end of the schedule**: proseg holds and writes `schedule_done`. *Finish & save* writes proseg's outputs from that
   state; Resume or Step writes `continue 1` and samples on, open-ended (phase `continued`), until Finish & save.
7. **Save to the sample** (`POST /runs/{id}/zarr`): see [Saving into a sample](#saving-into-a-sample).

Run states: `preparing` → `running` ⇄ `paused` → `finishing` → `done` | `failed` | `stopped`. An engine holds one run
at a time (see [Leaving a run](#leaving-a-run)). A run pauses itself when the data disk has under 15 GB free; a
test's own run is aborted instead, saying "only N GB free on the data disk", and the test fails with those words.

### Proseg's own count

Every number the app shows about a cell's RNA (the mixing check, Go deeper and its pink dots, the clusters and the
UMAP, the cell readouts, the test of a setting, Compare's run column) is proseg's own count of each cell, gene by gene;
the drawn outline (the consensus plane, the layers, the starting outlines) is for drawing and positions only:
footprints, centres, near flags, start places, bins and areas. Three counts per cell and gene: **n, assigned** (the
transcripts whose destination after drift the cell owns), **f, sampled** (proseg's draw of what it keeps as the cell's
RNA at that iteration; the saved table is f at the last iteration) and **e, averaged** (f averaged over proseg's random
background draw, with the frame's parameters). Every number the engine works out about a cell's RNA reads e
(`analysis/counts_file.py` `LIVE_COUNT`). On an iteration proseg did not count, the page's cell readouts show its one
draw (f, cells.zst `foreground`), and the tooltip says so; Compare's like for like cuts from f. e is never called
"expected", which is another of proseg's outputs.

- **Launch** (`engine/runs.py`): a proseg-live whose `--help` lists `--live-counts` gets `--live-counts request
  --live-counts-dir <the run's spool folder>`, and `--live-counts-start` for a user's run whose cells start without
  nuclei (its start block: proseg's iteration-0 assignment, counted with each frame's parameters). An older proseg-live
  gets none: its runs segment as before, without clusters, mixing check, Go deeper, test or Compare column
  (`old_proseg`); a run whose info.json has no `counts` was made before (`no_counts`).
- **Tokens.** The engine asks proseg to count a frame with a `counts N` line in `live/control`, and for a where export
  of a box (proseg's pooled transcripts there, each with the cell holding it now) with `where N x0 y0 x1 y1`; proseg
  serves a token with its next frame (forced past history thinning and the tiled cadence), at once from the paused state
  while paused or held. It counts frame 0, the upscale, pauses, the hold, finish and the end by itself. Each counted
  frame's files are in place before its line is appended to `counts.jsonl` (the commit point); a paused or finishing
  ack comes only after that state's line.
- **Where the files go.** Request and pause counts go to the run's folder in the counts spool
  (`engine/counts_spool.py`: `<RAM disk or temporary folder>/sopa-companion-counts-<engine pid>-*/<run>/`, at most
  `[analysis] counts_spool_mb` a run and twice that for the engine; made when proseg starts the run, removed with the
  run, swept at the next start after a crash); the durable ones (frame 0, the upscale, the hold, finish, the end) and
  every per-cell sidecar go to `live/frames`.
- **The scheduler's pace** (`engine/analysis.py`): live passes take each run's newest counted frame; one counts token is
  on its way at a time, asked as a live pass starts, no sooner than proseg's own counting time over `[analysis]
  counts_extract_share` (10 % of its time) after the newest count, only while the run samples, and past iteration
  2,000 at most once per thinning step (5, 25, then 100 iterations, as proseg-live keeps frames there). A frame asked
  for (scrubbing, stepping, Go deeper, Compare) is taken as the counted frame it snaps to: the newest at or before it
  whose file is there or whose results are kept, else the oldest after it. Analysed files are pruned from the spool,
  keeping the newest and an evenly spaced reserve for a check switched on later.
- **What pages get**: frame records in `GET /runs/{id}/frames` and the `frames` push carry `counted_file` when the
  frame is counted, never the engine's `counts` line or `counts_file`, and so does the run's `last`, its `info` without
  the spool folder (`engine/runs.py` `page_frame`, `page_info`: the engine keeps both whole); the run its
  `proseg_counts {reason, newest, recent}`; a watch from a frame on first names the counted
  frames before it (`{ev: "counted"}`); `GET /runs/{id}/counted/{seq}` serves the frame's per-cell sidecar.
- **Results on this count** live in folders of their own (`analysis/pc1/`, `qc/<model8>.pc1/`): what earlier versions
  made on outline counts is never read.

The files' layouts are under [Proseg's count files](#prosegs-count-files-plc1-version-1).

### Clusters and UMAP

The scheduler (`engine/analysis.py`) sends every active run's newest counted frame to the analysis worker as soon as
the previous pass ends (`[analysis] min_interval_s` can space passes out); frames asked for by the page (`POST
/runs/{id}/analysis {seq}`, answered with the counted frame they snap to, `requested`) go first. A pass reads the
frame's counts file (proseg's averaged count of each cell, gene by gene: the same for dense and tiled runs, never an
outline count), normalises, runs PCA, kNN and Leiden (clusters matched to the previous result so ids and colours stay),
top genes, and a UMAP warm-started from the previous layout. Results are stored in the run's `analysis/pc1/` folder
and announced on the WebSocket. A run with too few cells to cluster says so (`unavailable: too_few_cells`); one
without proseg's own count gets no pass (`old_proseg`, `no_counts`).

Each run's settings (resolution, neighbours, spacing, fewest transcripts, genes used: the UMAP pane's Advanced settings,
`POST /runs/{id}/analysis {params}`) default to the worker's own overlaid with `[analysis]`'s. A change starts a new
**settings generation** of the run's results: what was computed with the earlier settings is dropped from memory,
pushes and downloads (a pass of them still running is thrown away when it ends), the frame on screen is computed first,
and the scheduler removes the earlier generations' files between passes. Results of generation n are
`analysis/pc1/<seq>-s<n>.*` (0, a run that never changed them: `<seq>.*`); the run's settings and generation are kept in
`analysis/settings.json`, and new runs start with the settings last applied (`<data_dir>/analysis-settings.json`). The
worker lays the map out afresh (then rotates it onto the one on screen) when neighbours, spacing, fewest transcripts or
genes changed; a new resolution only regroups the cells on the same map. Compare's shared layout keeps its fixed
recipe.

**The worker's backend.** The worker chooses at its start (`analysis/worker.py` `pick_backend`, `[analysis] backend`:
`auto`, `gpu` or `cpu`): the GPU when cupy, cuML, cuGraph and cuDF are installed, CUDA shows a device and cupy can run
code on it; else the CPU when scipy, scikit-learn, igraph, umap-learn and numba are there (`analysis/backend.py`), which
`pip install sopa` brings. Its hello says which and why (`backend`, `why`, `threads`). The CPU recipe (`analysis/cpu.py`)
writes exactly the GPU's files: PCA from the covariance of the top genes, built in row blocks, and its top vectors
(ARPACK); exact neighbours (scikit-learn); igraph's Leiden started from the last clusters (2 iterations, so colours stay
steadier); umap-learn's layout warm-started from the last one. Genes used stops at 5,000 there (the engine sends that
as the field's limit). The CPU worker runs at nice 10 with at most 4 threads, keeps 2 runs (4 on the GPU), and spaces a
run's passes by `"auto"` (its cells / 15,000 s, 0.5-6 s) when `min_interval_s` is 0. It loads its libraries before its
hello, so one that cannot load is named at once, then warms up while idle (its layouts compile: ~3.5 s). The worker
starts with the first active run, while it prepares, on either backend, so its warm-up is done by the first frame. A
Python that cannot run Clusters & UMAP at all (packages missing, or its libraries failing to load: a lasting hello)
turns them off for the session: one line in the log names what is missing, nothing is started again, each request is
refused saying why, and the page names the packages under Layers > Cells. A GPU stack that is installed but missed its
device at the start (busy, out of memory) is never that, even where the CPU's packages are missing: the device may come
back, and the engine asks again.

### The mixing check

- **Learning** (`QcModels`): when the check is on for a run, its sample is learned once with ResolVI by the trainer, on
  GPU 0 only (on a Mac without an NVIDIA GPU, Apple's GPU where it passed the probe's test, kept with the sample's key:
  its fit given MPS_TIME_FACTOR times the time limit, learned at once with no wait for free memory (only a Mac too small
  in all is refused, the dense counts counted), its model keyed apart from a CUDA one: `device_of`, `apple_memory`,
  `model_key`), one fit at a time (a lock per data folder and one per user), as soon as GPU 0 has the free memory a fit
  of that table needs: `room_needed_mb` from its panel (about 0.9 GB plus 0.6 MB a gene, the cells adding little, with a
  20 % margin, from the trainer's measured peaks), or `[qc] gpu_min_free_mb` when it is a number. While it waits, the
  sample's state (`GET /samples/{id}/qc`) carries `room` `{need_mb, free_mb, total_mb, genes}`, which the page puts into
  words; a card with less than that in all fails at once with reason `gpu_small`, nothing read. The trainer reads the
  sample's cell table read-only while holding the sample's save lock shared, and exports a frozen per-cell calculator
  (`encoder.npz`), Go deeper's details (`deep.npz`), the learning cells and `model.json` into a staging folder that the
  engine checks (self-tests) before renaming it into place. The cache key covers the zarr, the table and its
  fingerprint, the recipe and the environment's package versions. The table is the first of `[qc] tables`, then any
  other, that holds whole counts and whose cells can be placed on the transcripts: its own centroids when they agree
  with its outlines' centres in the frame, else the outlines' (centroids alone must lie on the transcripts; centres the
  outlines placed need only touch them). A table refused for itself (`not_counts`, `positions`) is remembered by its
  fingerprint and the trainer's rules (`tables.json`), and the next one is tried; one of under 200 cells, and a fit
  whose calculator sees no mixing in 80 % or more of its cells, fail as `too_few_cells`. The sample's reason reaches the
  page with the switch off too (`sample_reason`).
- **Can it learn here?** A probe (`analysis/qc_probe.py`, about a second in `[qc] python`, at the engine's start)
  asks as the trainer asks: the NVIDIA driver's library (NVML) for the cards and the CUDA the driver supports, torch's
  own CUDA check for the cards it sees, without one Apple's GPU through a short test of the learning's operations in a
  process of its own (`zarr_bridge/devices.py MPS_LEARNING_TEST`), and whether scvi-tools and scviva-tools are
  installed. A definite no (`no_gpu`, `apple_gpu`, `torch_cpu`, `driver`, `software`, in that order) makes every sample `unavailable` with that reason, and a model
  already learned is not served either; a probe that gives no answer leaves the check offered (its trainer says what
  fails) and is asked again after 60, 300 and 900 s; a press of the switch or Try again asks again where the check is
  not offered. `GET /health` says `qc.state` and `qc.reason`.
- **Scoring** (`QcManager`): each analysed frame also gets the check's pass in the analysis worker (op `qc`) on
  proseg's own count of the frame, and a run whose cells start without nuclei gets its start line once (`qc_start`:
  proseg's count at iteration 0, frame 0's counts file, with the prior's footprints for positions); a start with
  nuclei (every Xenium bundle) has no start line, as proseg's iteration 0 holds only the nuclei, and its start routes
  answer 404 `no_start` with the run's `start_kind`. The engine turns each result into a summary: the share of cells
  that look mixed (at least `judge_min_counts` transcripts of proseg's count, share at least `mixed_at`), a headline
  smoothed over the run clock, coverage and transcript retention, the like-for-like change since the start (each frame
  against its own start block: proseg's iteration-0 assignment kept as proseg keeps RNA at that frame; each cell's two
  averaged profiles scaled to the shallower depth and scored once, with no draws), steady verdicts against size-aware
  thresholds, flagged areas (bins of 50 µm, 100 µm in whole-sample runs), one suggestion card at a time, and trials of a
  changed setting against a flat band (drift too: proseg's count sees it). The thresholds are read for the run's kind
  (`engine/qc_constants.py`: a dense region, a tiled region or a whole sample), and every summary says which count it
  read and whether its kind's thresholds were re-measured on proseg's count (`basis`, `count`, `constants`): a number
  shown with thresholds not re-measured carries a caveat in its tooltip. Results are in `qc/<model8>.pc1/`.
- The switch is per run (`<run>/qc/settings.json`), remembered for new runs, off as shipped.

### Go deeper, test a setting, Compare

- **Go deeper** (`POST /runs/{id}/qc/deeper {seq, cell, follow?}`): the worker's op `deeper` explains one cell with the
  model's details, on proseg's count of the frame (a frame without its own count is answered as the counted frame it
  snaps to): the split of its RNA into its own, from neighbours and loose RNA, per gene, and the neighbours it seems to
  come from. Answers are cached (memory and `qc/<model8>.pc1/deeper/`); `follow` is a lease the page renews.
  `GET /runs/{id}/qc/deeper/where` (reply version 3) gives every transcript of the genes asked in the box around the
  cell and its sources, with proseg's share of it for the cell (`w`, `engine/held.py`): exact from proseg's where export
  of the box (asked for at once while the run is paused or held, and with each counted frame while a window follows a
  playing run), else approximate from the frame's per-layer labels times the cell's keep (e / n), else none (a past
  tiled frame outside its window, or a frame whose counts are gone); never from the flat 2-D column (`held` says
  which). Each point's owner, from the run's labels, stays for drawing and the near flags. The cell's total and each
  gene's in-cell and near sums are sums of w; each gene also carries proseg's own count of it (`count`). "Too few to
  tell" reads that count (and, with approximate shares, also what was placed); the other words (within 2 µm of a shared
  edge, spread through the cell, as common around) read the sums of w. A point glows pink at w × the model's near/far
  value ≥ 0.2, whatever outline it lies in (pink just outside: proseg moved it in); when none of the glow lies near the
  neighbour Go deeper would name, the window says the source looks like a cell with no outline there and never points
  at that neighbour.
- **Test a setting here** (`GET /runs/{id}/qc/test/pool`, `POST /runs/{id}/qc/test`): runs on any proseg run that
  reaches the end of its schedule and keeps proseg's own count, whole-sample and tiled runs included (their last frame's
  positions read in its column basis, `engine/qc_pool.py` `_TiledContext`; proseg's share of each transcript never from
  that column); the area is the region as it is when it fits in 500 × 500 µm, else a 500 µm square around the cell,
  shifted to stay inside the region and cut to it where the region is narrower (its rows of the run's own input). The
  engine's reasons come in three kinds (`engine/qc_pool.py` `KIND`): refusals, where there is nothing to test or the
  test can't run (a cell with too few transcripts, nothing that glows, or no own RNA to compare its suspects with; a run
  being saved, one that ended before its end, or one without proseg's own count); waits, which the app meets itself (a
  press before the run's end waits for it, resuming the run if it was paused; one with the mixing check off switches it
  on and waits until it has learned and scored; one while another run samples waits for it to pause), the test then in
  state `waiting`; and caveats: where the measurements did not cover a situation, the test runs all the same, judged
  with the nearest measured row (`constants.start`, `constants.situation`), and its answer carries a caveat (`caveats`).
  The cell's suspects, and those of nearby cells with the same pattern, are fixed before any run (`engine/qc_pool.py`:
  Go deeper's own recipe at the run's last iteration, its scoring the page's, ported in `engine/qc_scoring.py`, on
  proseg's count there: the glowing points, with proseg's share of each from its where export of the area while the run
  is held at its end, else approximate, said in a caveat; a cell that does not look mixed but glows is tested, saying
  so); then 3 + 3 fresh runs of the area (4 + 4 for a start from cells only), as now and with the value, all from one
  shared input, all at once, to the end of the schedule. Each run's own per-transcript file is read
  (`engine/qc_readout.py`) and the measured rule (`engine/qc_rule.py`) says whether the suspects drop clearly further
  than their own RNA at the same spot, what it cost and where they went. The run itself waits paused at its end (one
  sampling past it is paused when the runs start); the test runs are ordinary runs while they exist and are deleted when
  the test ends.
- **Compare** (`POST /runs/{id}/compare {columns, seq?}`): the run's counted frame and up to three of the sample's
  segmentations, each read by its own count: the run's column proseg's count of the frame; a saved proseg result its
  table's; Xenium's cells Xenium's cell table; a segmentation with no count of its own (Cellpose, a stain) the
  transcripts inside its outlines, by the run's own input rule (quality and excluded genes: the pattern is checked when
  the run starts by the same rule Compare reads it with, and Compare works the gene names out under a time limit); each
  column says which (`count`), and proseg's count, which sets background aside, reads less mixed for that alone. Cut to
  the area all of them cover (100 µm tiles holding at least 3 cells of each, the area clipped to the sample and the
  run), scored with the sample's model (mixed per 100, like for like at
  100 transcripts, a histogram) and laid out on one shared UMAP coloured by the model's kinds. One comparison per run;
  a finished one is kept beside the run.

### Stain runs and training patches

- `POST /runs {kind: "stain", sample, region, stain}` creates a paused run with its settings checked (image, channels,
  method, model from the discovered list, Cellpose settings, tissue only); no model loads before Start.
- The stain helper plans tiles over the region (only those touching the tissue with *Tissue only*), first measuring a
  few central tiles' cells to set the overlap, then segments tile by tile. Each tile commits its masks, polygons and
  drawing arrays before `tiles.json` names it; pause and stop act between tiles, and Resume continues from the committed
  tiles. Each tile keeps the cells whose centre lies in its own part; the others are spares that come back when their
  owning tile has run and does not cover them; sopa's `solve_conflicts` joins the rest at the end.
- The page reads committed outlines as prior vectors (`GET /runs/{id}/stain/vectors?box=`). Removed areas, the size
  range and the brightness range (`stain/exclusions.json`) leave cells out of the outlines, counts, patches and saves
  without changing the model's output; the ranges' histograms read every cell's values (`GET /runs/{id}/stain/cells`:
  `PLCV` v1, `engine/vectors.py` `encode_cells`, each cell's area, whether a removed area holds it, its brightness per
  channel).
- Training patches (`kind: "patches"` items, or a paused or finished stain run): proposals, a versioned box draft
  (`patches.json`), a watch for corrected masks, and the Cellpose GUI started detached on the export's folder.
- Export patches asks every time where the export goes: a folder dialog in the page (`patch-folder.mjs`) lists folders
  through `GET /runs/{id}/stain/patches/folders?path=`, which also says what a folder holds and why it may not take
  patches, and opens on a suggestion: the folder this sample's patches at this resolution and channels went to last,
  else `<sample>-scale<k>` in the default place (`~/sopa-companion/training-patches`; in a notebook given `patches_dir`,
  there). A chosen folder is a training folder (the Cellpose GUI trains one model on every patch in its folder): the
  patches go straight into it, beside earlier ones, as `<sample>-<YYYYMMDD-HHMMSS>-NN.tif` and `-NN_seg.npy`, never
  replacing a file, and the app's records go into its hidden `.sopa-companion/` (one manifest per export in `sets/`,
  written last, the GUI's record and log). `engine/patch_folders.py` checks a typed path when it is listed and again at
  export (links resolved; zarr stores, hidden folders, the app's own folders and environments, notebook sessions'
  folders, `/`, the home and temporary folders themselves, folders of the earlier layout and folders that cannot be
  written to are refused), and writes an export in: the helper writes into the app's `stain-staging/`, the engine
  copies the files in under hidden part names (created exclusively, never through a link, through folder handles
  opened without following links), then, under the run's lock, renames them to names it found free and writes the
  manifest. The folders exported into are listed in `patch-folders.json` (a notebook's goes with its session); the
  models' "learned at" labels and *Use settings from Cellpose* read the sets found there, in the default place and in
  `patches_dir`, and the sets of the earlier layout (`<data_dir>/stain-patches/<run>/<set>/`, `<patches_dir>/<run>/<set>/`),
  which are read as before. The GUI starts in its cache folder with `PYTHONSAFEPATH`, so nothing in a patch folder is
  imported or read as a cache; but the Cellpose GUI itself loads the folder's `_seg.npy` masks with pickle when it
  opens a patch and when it trains, so a mask put there by someone who can write to the folder runs as the user (the
  README says to keep training folders where only the user, or people they trust, can write). The app's own readers
  of those masks use a restricted unpickler; they, and the opening of the GUI's log, never follow a link or wait on a
  FIFO.

### Saving into a sample

The helper's `save` writes proseg's results (shapes plus their table) or a stain run's shapes into the sample zarr with
sopa's own functions, as one recoverable transaction: a lock per zarr shared by every process, held in each of
`~/.cache/proseg_live/zarr-locks/` and `~/.cache/sopa-companion/zarr-locks/` in that order (the first is the only one the
previous version of the app, the package `proseg_live`, takes: an engine of either version keeps the other out; with
`SOPA_COMPANION_ZARR_LOCK_DIR`, that folder alone), names checked with SpatialData's rules (no clash ignoring case;
shapes another table annotates are never replaced), elements being replaced moved aside into
`<zarr folder>/.proseg-live-backups/` (the previous version's name, the only folder it rolls back from; recovery also
reads `.sopa-companion-backups/`) with a write-ahead journal naming the writer, no-clobber writes, and a rollback on
any failure. Each step is journalled (fsynced) before it is taken. An interrupted save is rolled back by the next save
into that zarr: its own backup folders only (the whole folder name, and the zarr's path in the journal), once the
process that wrote it has died (its pid, computer, boot and start time; a journal of an earlier version names only its
folder's pid and stamp: a process started after that stamp, or a thread, is not its writer); a live writer refuses the
save; a folder whose rollback left anything is kept, marked `rolled_back`, and never rolled back again. A name clash
answers 409 `{error, clash, existing}`, whether the engine finds it or the helper does, so the page can offer Replace;
save defaults are worked out against the sample's current names whenever the dialog opens. A stain run with no cells
left is refused (409). Provenance: the table's `uns["sopa_companion"]` (run,
iteration, prior, region, options) for proseg results; the shapes group's `attrs["sopa_companion"]` (method, model and its
hash, parameters, runtime, region, progress, removed cells) for stain results. Every reader also takes the previous
version's `proseg_live` record (names.py `PROVENANCE_KEYS`): a sample keeps what either version saved into it.

### Leaving a run

An engine holds one run at a time: `POST /runs` (proseg, stain or patches) answers 409 "Another run is open: save it to
the sample or discard it first." while it holds a run that is not a setting's test's (those come and go with their
test). A page opens the run there is, and a page on the first screen or a picker follows a run started in another
window. Every way off a run (the brand, New run, Open sample…, Proceed to proseg, Close) is one engine operation,
`POST /runs/{id}/leave {answer: "add" | "discard" | "stay", revision?, page?}`, decided by one rule shared with the
page (`engine/runs.py` `leave_kind` and `frontend/src/shell.mjs` `leaveKind`, both held to `tests/data/leave_cases.json`):

- kind `none` (nothing to lose: not started, a stain run that found no cells, a training-patches item, a setting's
  test run) and `saved` (its result is in its sample and current): it goes without a question;
- kind `ask`: the page asks Save to the sample, Discard or Stay (only Stay or Discard when a stain run's filters leave
  no cells). `add` finishes a proseg run, or stops a stain run after its current tile and joins its finished tiles,
  saves the result under the run's next free names (one retry after a clash: nothing is ever replaced) and deletes
  the run;
- kind `saving` (a save under way): the page waits for it, then goes as for a saved run.

`discard` stops the run if it is active, waits for a save or a patch export under way, then deletes the run and what was
made from it (its folder, index, tile cache, its folder in the counts spool, analysis, mixing-check state, Go deeper,
Compare, its test), never the
sample, learned models, the sample's caches or exported patch sets. `stay` withdraws a leave under way, and takes a
*Finish & save* back (the Save dialog's *Not now* sends it), so that closing asks about that run like any other.
`revision`, with the window's `page`, keeps one window's answers in their order: an answer overtaken by a later one of
the same window changes nothing. The run then goes from the registry, and the page watching it gets `gone {reason:
"left"}`.

Nothing survives a session: a run is saved or discarded as it is left or as the engine closes. At start every run
folder this version made (`run.json` `session_model` 2) is removed and reported once (`health.leftovers`), except one a
save of the last session may still use, which is left as it is until that save ends. Runs made by earlier versions are
kept apart from the registry: those whose result was never saved are listed for an answer (`GET /runs/older`), Save to
the sample (never as the sample's segmentation) or Discard, carried out one at a time in the engine (`POST
/runs/older/{id}`; `POST /runs/older` for Save all, Discard all, or Later, which sets the rest aside until the next
start); the others are removed and counted. A test engine (`[proseg] keep_unsaved`, set in a test's own settings)
keeps every run folder, loads them again at start, allows several runs (`[proseg] max_active_runs` active at once) and
never writes into a sample as it closes.

### Closing

The last page that leaves decides (`server/presence.py`), whatever browser showed it. A page counts while its socket is
open (`/api/v1/ws?page=<its id>&visit=<its showing>`); one that goes for good says goodbye on its way: its socket's
`{op: "bye"}`, its log's last upload (`POST /page-log` with `pagehide`) or its answer to closing (`POST /closing` with
`bye`). Each names the page's showing (1, then one more each time the browser brings the page back from its cache): a
goodbye, or an answer to closing, said before the page came back and reaching the engine after it changes nothing. The
last page gone with a goodbye is a real close: once none came back within `[desktop] close_grace_s`
(`close_grace_pending_s` while a run waits for an answer, so a reload comes back in time) the engine shuts down, and a
run that page did not answer for is discarded without the dialog. The last page gone without one (a crashed or killed
tab, a tab the browser put away, a lost tunnel, a laptop asleep) is away: each run still sampling pauses as its Pause
would (`Runs.pause_for_away`, marked `paused_away` until it is resumed, finished or stopped), the engine waits, and the
next page that opens the run says why it is paused. An engine left away stops by itself only with nothing waiting for an
answer, after `away_stop_s` (0: never). Engines with `exit_when_closed = false` (development and test engines) do
neither. With a run not saved yet the page asks first (the leave prompt, in Close's words) and posts the answer (`POST
/closing {answer: "add" | "discard" | "withdraw", runs?, revision?, page?, bye?, visit?}`, also as the page goes: the
answer given, else `discard`, for the run that page showed); Keep working, or a page connecting again, withdraws it. At
shutdown a setting's test is cancelled and the mixing check stops first. Then every run leaves (`Runs.close`): a decided
run (a leave or save under way, or a *Finish & save* not declined) is completed whatever the answer; the others go as
the page answered within the last 60 s for the runs it showed, else as the desktop dialog answers (zenity: *Save to the
sample* adds; closing it, or no answer within `save_dialog_timeout_s`, discards), and are discarded when there is no
display or no dialog tool to ask on; after the last page's goodbye the dialog never asks. `sopa_companion stop --force`
(`POST /shutdown`, bearer token only) asks nobody: it discards the runs nobody answered for, still completes decided
runs and a page's answer from the last 60 s, and answers `{discarding, saving}`; a plain `stop` refuses while a run is
not saved to its sample. Every run folder then goes, except a run whose save failed while closing (`closed: "save
failed"`) and one whose save helper still uses it after an hour (`closed: "save running"`): the next start removes both
with the other leftovers. SIGHUP closes the engine as SIGTERM does (the outputs that were its terminal go to /dev/null
first), unless it was started with SIGHUP ignored (nohup).

## A notebook's session (API mode)

`sopa_companion.launch(sdata)` (the package's `__init__.py`, which imports `api/` on first use) runs in the kernel and
imports none of the engine, server or desktop code. It checks the call (Linux, macOS or Windows, the main thread, a SpatialData written to a local zarr,
which points element holds the transcripts), reads the settings (the app's settings files, then
`~/.config/sopa-companion/config.toml`), removes the sessions that dead kernels left, makes the session's folders and
`session.toml`, then starts the engine as the kernel's child: `setpriv --pdeathsig TERM <python> -m sopa_companion serve
--config <state>/session.toml` on Linux (the same without `setpriv` on macOS and Windows, where the events pipe reaches
it as an inherited handle), which ends when the kernel does: on its death signal, or as everywhere when its control
pipe ends. That interpreter is first asked, in the state folder and
without `PYTHONPATH`, whether it imports this very copy of the app (`locate.imports_this_copy`): then the engine gets
no `PYTHONPATH` (the kernel's own never reaches it), else the session's `pylib`.

| Kernel side | Role |
|---|---|
| `api/launch.py` | `launch()`: its checks, the settings it takes, the session's folders and `session.toml` |
| `api/handle.py` | `CompanionApp`: the engine child and its pipes (one I/O thread), the browser or the link, `open()`, `close()` |
| `api/folders.py` | the private session folders, their owner records, and the sweep of sessions a crash left |
| `api/window.py` | opening the app where `[desktop] browser` says (`desktop/browser.py`), one-use links shown in the notebook and replaced once used, the session's Chrome window closed |
| `api/deliver.py`, `api/mainthread.py` | adding a delivered result to the object on its own thread, between cells |
| `api/protocol.py` | the session's settings tables and JSON-line messages, checked strictly |

- **Folders.** The state folder (0700) is `$XDG_RUNTIME_DIR/sopa-companion/<id>/`, else
  `~/.cache/sopa-companion/sessions/<id>/`: `session.toml` (0600: it holds the session's token), `owner.json` (the
  kernel's and the engine's process identities), `logs/`, `pylib/sopa_companion` (a link to the package, made only when
  an interpreter needs it: the engine's, or a helper's in another environment), the Chrome window's `chrome-profile/`
  (`browser = "chrome"`) and, for a minute, the opening page (a snap browser's is in its own
  `~/snap/<name>/common/sopa-companion`). The data folder,
  `<scratch_dir>/sopa-companion-<id>/`, is the engine's `data_dir` (runs, caches, deliveries; its marker names the
  session, its state folder and this computer). Both go when the session ends; the sweep at the next `launch` removes a
  session only when its kernel is gone, its engine is not running and the data folder's engine lock is free. The
  scratch folders sessions used are kept in `~/.cache/sopa-companion/scratch-roots.json`, so that a data folder whose
  state folder went (a reboot or a logout) is removed too: only this computer's, with its lock free. Training
  patches go where the export dialog's user chooses, always outside the session's folders (they are refused there);
  `patches_dir` is the base of the dialog's suggestion, and its sets are found again. InstanSeg's weights (while its
  switch is on) go into `~/.cache/sopa-companion/instanseg-models`, also outside the session's folders. The
  Clusters & UMAP settings new runs start with (`analysis-settings.json`) stay in the data folder, for the session
  only.
- **`session.toml`** is merged over the app's settings files (`--config`): `[server]` (port 0, any free one; the token),
  `[paths]` (the two folders), `[desktop]` (the pages' watcher only for a window; never `browser`, which only the
  kernel reads), `[zarr]` (the kernel's Python),
  `[proseg] binary` (the proseg-live `launch` checked), `[analysis]`, `[stain]` and `[qc]` (each environment's python,
  a full path or `"off"`, never `"auto"`: `launch` resolved it in the kernel, and the engine, which may run in another
  interpreter, never looks again; and each GPU; `[qc] models_dir`, the model cache; `[stain] instanseg_enabled`, the dormant
  InstanSeg's switch as `launch` read it) and `[session]` (`mode = "api"`, its id, tool, how it opens, `patches_dir`,
  the link's 600 s, the kernel's pid).
- **Pipes.** The kernel writes JSON lines to the engine's stdin: `hello`, `bind` (the sample), `object` (the object's
  element names, its zarr and sopa's auto-save, again whenever they change), `notebook` (a cell is running),
  `received`, `receive_failed` (a result it could not read), `applied`, `close`. The engine answers on a pipe of its own
  (`SOPA_COMPANION_EVENTS_FD`, taken out of its environment as it starts): `ready` (its port), `session` (its state),
  `deliver` (whose record leaves out outline points: the staged shapes keep them), `closing`, `ended`, `link` (a link
  signed a browser in: the kernel replaces the link it showed). Each event is one
  whole line of at most 1 MiB, written as the kernel reads (for at most 30 s); one cut off part-way ends the channel.
  The end of stdin means the kernel is gone.
- **One sample.** The engine opens the sample it was given; `GET /samples` lists it alone, `POST /samples/open` opens
  only its path, `GET /browse` and `DELETE /samples/{id}` answer 403. `GET /health` carries `session` (the sample and
  its state, where results go, whether the notebook is busy, whether it is ending), and a `session` push says when that
  changes. The page shows no Open sample…, browser or Recent, and says *Add to your SpatialData* for *Save to the
  sample*.
- **A save** runs as on the desktop (the helper's `save`, with its lock and journal), into the object's zarr when it
  has one and sopa's auto-save is on. Its result is staged in `<data_dir>/deliveries/` and named by a `deliver` event;
  the kernel reads it on its I/O thread, adds the elements to the object on the main thread between cells (never while
  one runs), then answers `received` and `applied` (`added` or `failed`, with any name clash). The run's
  `zarr_save.state` follows: `saving`, `queued` (waiting for the notebook), `saved`, `failed`.
- **Closing.** The page's Close posts `POST /closing {answer, end: true}`, and `app.close(pending=...)` sends `close`;
  the last page's goodbye without an answer (its tab closed), the kernel ending, or nobody answering, discard. A page
  gone without a goodbye pauses its runs, and the session waits for `app.open()` or `app.close()` (never stopping by
  itself). Every run then leaves with that
  answer (`Runs.close_session`: a decided run completes unless the kernel has gone, and an added run is delivered and
  waited for), the window is closed and, when the kernel is gone, the engine removes the session's folders itself;
  `ended` lists what was added and what was not.
- **Links.** A browser the session opens itself (the default browser, or its Chrome window) takes a launch code that
  lives seconds, through an opening page in the state folder, never on a command line (`app.url` then holds that code,
  used by the browser). A link for a person (no display, SSH, `open="link"` or `"none"`, no browser that can read the
  opening page, or none signed in within 20 s of opening one) lives 10 minutes and is shown and kept in `app.url`; in a
  notebook its output is replaced once it has been used, or once it is no longer needed (a newer link shown, or the
  browser the session opened signing in after all: `presence.windows`), so the notebook holds one live code at most.
  Every link names the session (`api=<id>`), whose tab keeps its storage for it. `app.open()` opens the app again with a
  fresh code. Each works once: the page trades it for the session engine's own page key (opening the link spends
  nothing).

## HTTP API

All routes are under `/api/v1`, on loopback only. Every request needs the page key (the header `X-SoPA-Page-Key`; the
page trades a one-use launch code for it, `POST /session/sign-in`, and keeps it in its tab) or the bearer token from
the private runtime file (`Authorization: Bearer`), never a cookie (none is set, and every `Cookie` header is dropped
before the app); requests that change something (and the WebSocket) also need the app's own Origin, and every request
a loopback Host on the engine's port. The app's own files (`GET` or `HEAD` outside `/api/`: `index.html`, `src/`,
`styles/`) are public, the Host check included; `index.html`, under whatever address it is asked for (and any other page
the engine serves, its development pages aside), carries a strict Content-Security-Policy, and no answer may be framed
or read by another site. Errors are problem JSON with a plain `detail`. Bodies are size-limited; non-finite numbers are
refused. In a row, every path takes every method named, and a path after the first leaves out the start it shares with
the first (`/runs/{id}/frames`, `/frames/{seq}`).

| Method | Path | What |
|---|---|---|
| GET | `/health` | version, protocol, whether proseg-live and each environment are there (`proseg_missing`: what makes a missing proseg-live, `build`, `install` or `setting`; `proseg_fix`: what updates the one found, for a run whose proseg-live cannot keep proseg's own count), limits, speeds, `one_run`, `leftovers`, `older`, `session`, `presence` (the pages signed in: `pages`, `state`, `since_s`, `connects`, and `windows`: the pages of browsers the app opened, which the launcher and a notebook wait for) |
| GET | `/options`, `/options/guide` | proseg's option schema (`flag, label, group, type, default, help, min?, max?, choices?, advanced, live`) and the knobs' guide |
| GET | `/samples` | recent samples, newest first |
| GET | `/browse?path=` | folders and SpatialData stores under a folder, for the open dialog |
| POST | `/samples/open` | open a `.zarr` → the sample detail |
| GET, DELETE | `/samples/{id}` | the sample detail; forget it from Recent (its files are never touched) |
| GET | `/samples/{id}/overview.png` | a thumbnail of the whole sample |
| GET | `/samples/{id}/image/{key}/c{k}.u16?box=&px=` | one channel over a box at the pyramid level that fits (raw uint16) |
| GET | `/samples/{id}/shapes/{key}?box=&limit=` | the picker's prior outlines (JSON) |
| GET | `/samples/{id}/segmentations` | every cell-like shapes element, named by what made it, with its outlines' URL |
| GET | `/samples/{id}/segmentations/{key}/outlines?v=` | its outlines as prior vectors (cells layer only) |
| GET | `/samples/{id}/tissue?image=` | the tissue: `region_of_interest`, else detected on the image |
| GET | `/samples/{id}/cellpose-settings` | the settings the Cellpose GUI saved with the newest corrected patch |
| POST | `/samples/{id}/patches/preview` | the boxes Training patches would start with (nothing is created) |
| GET, POST | `/samples/{id}/qc` | the sample's mixing-check model; `{learn: true}` or `{detail: true}` to learn |
| GET, POST | `/runs` | the run list; create a run (`kind`: proseg, stain or patches; 409 while another is open) |
| GET, DELETE | `/runs/{id}` | the run detail; delete a run that is not active (the page leaves runs instead) |
| POST | `/runs/{id}/control` | pause, resume, step, speed, set, finish, abort |
| POST | `/runs/{id}/zarr` | save into the sample |
| POST | `/runs/{id}/leave` | leave the run: `add` (save it to the sample, then delete it), `discard` or `stay` |
| GET, POST | `/runs/older` | the runs earlier versions made and never saved; Save all, Discard all or Later |
| POST | `/runs/older/{id}` | one of them: Save to the sample or Discard |
| GET | `/runs/{id}/frames?since=`, `/frames/{seq}` | frame records (`counted_file` on a counted frame); a dense frame's rasters |
| GET | `/runs/{id}/frames/{seq}/view?box=&stride=&max_px=&planes=`, `/frames/{seq}/window` | a frame's labels over a box; its per-layer window |
| GET | `/runs/{id}/prior`, `/prior/view` | proseg's voxel prior (dense; tiled over a box) |
| POST | `/runs/{id}/view` | the box tiled frames carry per-layer labels for |
| GET | `/runs/{id}/transcripts`, `/transcripts/view?box=&limit=` | the modelled transcripts (all; inside a box) |
| GET | `/runs/{id}/image/{name}`, `/image/view/c{k}.u16?box=&px=` | the run's image channels (detail runs; tiled over a box) |
| GET | `/runs/{id}/cells`, `/cellstats/{seq}`, `/cellhistory?cell=&points=` | original cell ids; per-cell statistics; one cell over the run |
| GET | `/runs/{id}/counted/{seq}` | proseg's count of each cell in a counted frame (its sidecar: f32 per cell; 404 for a frame without it) |
| GET | `/runs/{id}/xenium?format=bin&v=1` | the run's prior shapes as prior vectors (any other query: 404) |
| GET | `/runs/{id}/xenium/masks`, `/xenium/masks.bin` | Xenium label masks over the run (Xenium bundles) |
| GET, POST | `/runs/{id}/analysis` | clusters and UMAP: status and settings; ask for a frame (answered with the counted frame it snaps to, `requested`), change the settings, switch automatic passes |
| GET | `/runs/{id}/analysis/{seq}?generation=`, `/analysis/{seq}/cells?generation=` | a result's summary; per-cell cluster and UMAP position (of that settings generation only) |
| GET, POST | `/runs/{id}/qc` | the run's mixing check (`start_kind`: how its cells started); switch it, or ask for a frame (answered with the counted frame it snaps to, `requested`) |
| GET | `/runs/{id}/qc/{seq}`, `/qc/{seq}/cells?model=`, `/qc/{seq}/bins?model=` | a result's summary; one byte per cell; the per-bin layer |
| GET | `/runs/{id}/qc/start`, `/qc/start/cells`, `/qc/start/bins` | the start line (404 `no_start`, with `start_kind`, for a run whose start has nuclei) |
| GET | `/runs/{id}/qc/series?points=` | the run's line |
| POST | `/runs/{id}/qc/trial`, `/qc/trial/end` | watch a changed setting; keep or undo it |
| GET | `/runs/{id}/qc/test/pool?cell=&wait=` | the cell's pool a test would judge, and its reasons: what refuses it, what a press waits for, what was not measured |
| POST, GET, DELETE | `/runs/{id}/qc/test` | start, follow, cancel or dismiss a test of a setting |
| GET | `/runs/{id}/qc/test/view` | two of the test's runs (one as now, one with the value) over a crop around the cell |
| POST, DELETE | `/runs/{id}/qc/deeper` | explain a cell (optionally follow it); stop following |
| GET | `/runs/{id}/qc/deeper/{seq}/{cell}?model=`, `/qc/deeper/history`, `/qc/deeper/where` | a kept answer (immutable only with the model named and when it is that frame's own); the cell's line; where its transcripts sit, with proseg's share of each |
| POST, GET | `/runs/{id}/compare` | start (or find) a comparison; its state |
| GET | `/runs/{id}/compare/points/{index}.bin` | a column's UMAP points (`?own=1`: on its own map) |
| GET | `/runs/{id}/compare/clusters/{index}.bin` | a column's own clusters (Each its own) |
| GET | `/stain/models?method=` | Cellpose's models, InstanSeg's only while its switch is on (discovered, nothing loaded or downloaded) |
| GET | `/runs/{id}/stain/vectors?box=` | a stain run's outlines as prior vectors |
| GET, POST | `/runs/{id}/stain/exclusions` | removed cells: the areas and both ranges; remove the cells in an area |
| POST | `/runs/{id}/stain/exclusions/size`, `/exclusions/brightness` | the size range; the brightness range in a channel |
| DELETE | `/runs/{id}/stain/exclusions/{area}?version=` | bring an area's cells back |
| GET | `/runs/{id}/stain/cells?set=&channel=` | every cell's area, whether an area removed it, its brightness (`PLCV` v1) |
| GET | `/runs/{id}/stain/brightness?channel=` | cells' brightness in a channel (percentiles) |
| GET, POST | `/runs/{id}/stain/patches` | training patches: their state; the box draft |
| POST | `/runs/{id}/stain/patches/propose`, `/patches/export`, `/patches/open` | propose boxes; export a set into the folder chosen; open the Cellpose GUI on its folder |
| GET | `/runs/{id}/stain/patches/folders?path=` | the export dialog: a folder's subfolders, what it holds, the suggestion |
| GET | `/log`, `/runs/{id}/log` | the Log button's text |
| POST | `/telemetry` | the page's performance report (at most 16 KB) |
| POST | `/page-log` | the page's own log, kept for a crash report (at most 2 MB); with `pagehide`, that page's goodbye (for the showing `visit` names) |
| GET | `/crash`, `/crash/{id}.txt` | the newest crash report not offered yet; a report as text |
| POST | `/crash/page`, `/crash/{id}/seen` | a page that ended without unloading (its report); a report offered |
| GET, POST | `/closing` | the page's answer to closing with a run not saved: `add`, `discard` or `withdraw` (a notebook's session: `end`); with `bye`, that page's goodbye (for the showing `visit` names) |
| POST | `/shutdown` | `stop --force`: discard the runs nobody answered for, complete the decided ones, exit (bearer token only) |
| POST | `/session/launch-code` | a one-use code for the window's opening page, or with `?link=1` a link for a person (bearer token only) |
| POST | `/session/sign-in` | a launch code (header `X-SoPA-Launch-Code`) traded once for the page's key `{key}` (no credential; the app's own Origin; never cached) |

**WebSocket** `/api/v1/ws`. A page opens `/api/v1/ws?page=<its id>&visit=<its showing>` offering the subprotocols
`sopa-companion` and `key.<its key>`, checked in the handshake; the engine answers `sopa-companion` (never the key), and
closes a page's socket whose key is not this engine's with 4401. Tools use the bearer header (a tool's socket is no
page). The page sends `{op: "watch", run, since}`, `{op: "ping"}` and, as it goes for good, `{op: "bye"}`; the engine
pushes `runs` (the list, on every lifecycle change and frame counts at most every 1.5 s), `run` (the watched run's
detail, with `proseg_counts`), `frames` (new frame records), `counted` (once per watch from a frame on: the counted
frames before it, whose counts may have come after their records), `gone` (the watched run went; `reason: "left"` after
a leave), `analysis` (each new result, and progress; `state: "settings"` first when the run's Clusters & UMAP settings
changed), `qc` (sample learning, the run's start line, results, errors, the switch, trials), `deeper` (answers, the
details being learned), `older` (the runs from earlier versions, as their answers go), `session` (a notebook's session,
as it changes) and `pong`.

## Files

### Folders

| Where | What |
|---|---|
| `<state_dir>` (`auto`: `~/.local/state/sopa-companion`) | `run/server.json` (the runtime file, 0600, of one engine at a time: a second engine of another data folder is refused while it runs), `run/engine.alive` (while the engine is up), `logs/` (server.log, analysis.log, qc-trainer.log, telemetry.jsonl, page-log.txt, `page-logs/`), `crash-reports/`, `chrome-profile/` (the app window's). The default instance's first start moves the previous version's `~/.local/state/proseg_live` and its window profile `~/.local/share/proseg_live/chrome-profile` into it (`desktop/migrate.py`) |
| `<data_dir>` (`auto`: installed, `~/.local/share/sopa-companion`; a source folder, `var/`) | `.engine.lock`, `samples.json` (Recent), `patch-folders.json` (the training patches' folders exported into, newest first, with the sample, level and channels of their patches), `runs/`, `cache/`, `qc-models/`, `stain-cache/` (the helpers' caches; the Cellpose GUI's working folder), `stain-staging/` (an export before it is copied into its folder; emptied at start), `stain-patches/` (sets exported before the export dialog), `instanseg-models/`, `analysis-settings.json` (the Clusters & UMAP settings new runs start with) |
| a training patches folder (chosen at export) | the patches, `<sample>-<YYYYMMDD-HHMMSS>-NN.tif` and `-NN_seg.npy`, beside whatever else is there; `.sopa-companion/sets/<set>.json` (each export's manifest), `.sopa-companion/gui.json` and `cellpose.log` (the Cellpose GUI's record and output) |
| `<data_dir>/cache/zarr/<sample>/` | `inspect.json`, overview and image statistics, `shapes/` exports (`<key>.bin` + `.json` + `.stamp`), `segmentations.json` and outlines, `counts/`, tissue outlines |
| `<data_dir>/qc-models/<sample>/<key>/` | a learned model: `encoder.npz`, `deep.npz`, `kinds.npz`, `model.json`, `cells.parquet`, `batch.parquet`, `train.log` |
| `~/.cache/sopa-companion/` | `zarr-locks/` (save locks shared by every engine), `qc-fit.lock`, `pythonpath/` (the link to the app for a helper in another environment, or for every helper of a source folder not called `sopa_companion`) |
| `~/.cache/proseg_live/` | the previous version's `zarr-locks/` and `qc-fit.lock`, which this version takes too (first), so engines of both versions keep each other out |
| a notebook's session | its state folder and `<scratch_dir>/sopa-companion-<id>/` (see [A notebook's session](#a-notebooks-session-api-mode)) |
| the counts spool (the RAM disk `/dev/shm` when it has room, else the system's temporary folder) | `sopa-companion-counts-<engine pid>-<random>/<run id>/NNNNNN.counts.zst`: the counted frames the engine asked proseg for, and pauses' (made with the engine, removed with each run and with the engine; what a crashed engine left is swept at the next start) |

### A run folder (`<data_dir>/runs/<id>/`)

| File | What |
|---|---|
| `run.json` | what was asked for and the run's state (`session_model` 2: made by this version, which keeps no run between sessions) |
| `input/transcripts.parquet` | the region's transcripts with the prior, in the Xenium schema proseg reads (a test's runs have none: they read the test's one shared input in place, `<data_dir>/qc-tests/<test id>/transcripts.parquet` or the parent's own) |
| `live/info.json` | written once by proseg-live: schedule and plan, counts, voxel sizes, z geometry, prior, file layout; `counts` (how proseg's own count is kept: mode, format, the start block, the spool) |
| `live/frames.jsonl` | one line per snapshot: `seq` (0-based, no gaps), `iter`, `phase` (init, burnin, upscale, sampling, recorded, hillclimb, continued), `file`, `cells_file`, raster geometry (`nx, ny, nz, x0, y0, voxel`), `temperature`, `loglik`, counts (`assigned, foreground, transcripts, cells, visible, changed`), `t`, step times `ms`, effective live values `live`; tiled frames add `format: "tiles"`, `tile`, `window`, `view` |
| `live/frames/NNNNNN.bin.zst` | a dense snapshot: zstd of little-endian u32 `consensus[ny·nx]` (2-D projection), `last_change[ny·nx]` (iteration the column's label last changed), `layers[nz·ny·nx]` (layer-major) |
| `live/frames/NNNNNN.tiles` | a tiled snapshot: `"PLT1"`, u32 header length, a JSON header `{nx, ny, nz, tile, tiles_x, tiles_y, planes, window}`, an index of (u64 offset, u32 length) entries per tile and plane plus one for the window, then zstd blobs of u32 (length 0 = all zero) |
| `live/frames/NNNNNN.cells.zst` | per-cell statistics: `area`, `voxels`, `assigned`, `foreground`, `changed` (u32), centroid `x`, `y` (f32 µm) |
| `live/counts.jsonl` | one line per counted frame, appended once its files are in place: `seq`, `iter`, `file`, `counted_file`, `reason` (init, upscale, request, every, pause, hold, finish, end), `token`, `where`, sums, `has_f`, `has_start`, `saturated`, `bytes`, `ms` |
| `live/frames/NNNNNN.counts.zst` | proseg's own count of a durable counted frame (frame 0, the upscale, the hold, finish, the end; the others are in the counts spool): `PLC1`, below |
| `live/frames/NNNNNN.counted.zst` | every counted frame's per-cell sidecar: f32 per cell, proseg's count of the cell |
| `live/where.json`, `where-N.bin.zst` | proseg's newest where exports (it keeps 4): which cell holds each pooled transcript in a box |
| `live/prior.bin.zst`, `live/prior.tiles` | proseg's voxel prior: `label` then `permille` (prior probability × 1000) |
| `live/transcripts.bin.gz` | every modelled transcript: little-endian `x f32, y f32, z f32 (normalised), gene u32` |
| `live/genes.txt`, `live/cells.txt` | gene names by index; original cell ids by label − 1 |
| `live/control` | control lines: `paused 0/1`, `step N`, `min_ms M`, `finish 0/1`, `continue 0/1`, `set <flag> <value>`, `view x0 y0 x1 y1`, `counts N`, `where N x0 y0 x1 y1` |
| `live/ack` | `{state: running, paused or finishing, iter, seq}` |
| `live/schedule_done`, `live/sampling_done` | the schedule's end reached (proseg holds); sampling over (outputs being written) |
| `live/window.json`, `window-N.bin.zst` | a paused tiled run's per-layer labels for a moved view |
| `output/` | proseg's own results: `proseg-output.zarr`, cell polygons, cell metadata (and counts, transcript metadata for regions) |
| `xenium/vectors.v1.bin.zst` | the run's prior vectors, built once |
| `analysis/pc1/<seq>[-s<n>].json`, `<seq>[-s<n>].cells.bin` | a clusters and UMAP result of settings generation n (none: 0), on proseg's own count: summary; `int16 cluster[ncells]` then `float32 umap[ncells·3]` (results earlier versions stored in `analysis/` itself, on outline counts, are never read) |
| `analysis/settings.json` | the run's Clusters & UMAP settings `{v, settings, generation}` |
| `qc/settings.json` | the run's mixing-check switch `{v, enabled, n}` |
| `qc/<model8>.pc1/` | results on proseg's own count: `<seq>.qc.bin` (one byte per cell: `floor(share × 250)`, 255 = not judged), `<seq>.qc.json`, `<seq>.qc.grid`, `start.*`, `series.jsonl`, `shapes.jsonl`, `drawn.jsonl`, `trials.jsonl`, `state.json`, `deeper/` (`qc/<model8>/`, outline counts of earlier versions, is never read) |
| `compare/<job>/` | a finished comparison: `result.json`, per-column `points-<i>.bin` (f32 x, f32 y, u8 kind, u8 share code); while it works, `run-<token>.counts.zst`, its copy of the run column's count |
| `stain/` | a stain run: `job.json`, `plan.json`, `tiling.json`, `tiles.json`, `tile-N.masks.npz`, `tile-N.geojson`, `tile-N.npz`, spares, `resolved-N/`, `exclusions.json`, `brightness/`, `model.json`, `stain.log` |
| `patches.json` | the training patches' box draft |
| `proseg.log` | proseg's console output |

### Prior vectors (`PLPV`, version 1)

One little-endian binary for a set of outlines (a run's prior shapes, a segmentation layer, a stain run's cells), mapped
by the page into typed arrays without parsing (`frontend/src/vectors-decode.mjs`). Every section starts on a multiple of
4 bytes.

| Offset | Type | Field |
|---|---|---|
| 0 | u8[4] | magic `PLPV` |
| 4 | u32 | version: 1 |
| 8 | u32 | header bytes H (a multiple of 4): where the first section starts |
| 12 | u32 | layer count L: 2 (cells, then nuclei) |
| 16 | L × u32[4] | per layer: flags (bit 0 present, bit 1 numeric ids), polygons n, vertices V, names bytes N |
| 16 + 16L | u32 | meta bytes M |
| 20 + 16L | u8[M] | meta: UTF-8 JSON merged into the decoded value (e.g. `source: {cells?, nuclei?}`); zeros up to H |

Then, for each present layer, five sections: `offsets` u32[n + 1] (polygon i is vertices offsets[i] … offsets[i+1] − 1),
`xy` f32[2V] (µm, the working frame), `bbox` f32[4n], `ids` u32[n] (numeric ids, or an index into names), and `names`
u8[N] zero-padded (a JSON array of the ids as text, when bit 1 is clear). The file ends exactly after the last section;
anything that does not add up is refused.

### Proseg's count files (`PLC1`, version 1)

Written by proseg-live's counts writer, read by `analysis/counts_file.py` (numpy and the zstd codec only: the workers,
the engine and the tests' fake proseg). `NNNNNN.counts.zst` (NNNNNN the frame's seq) is one zstd frame of
little-endian fields with no padding:

| Field | Type | What |
|---|---|---|
| magic | u8[4] | `PLC1` |
| version | u32 | 1 |
| ncells, ngenes, nnz | u32 × 3 | rows (cell = label − 1), columns (`genes.txt` order), entries (per row the genes with n > 0) |
| flags | u32 | bit 0: f present; bit 1: genes as u32 (exactly when ngenes > 65,535); bit 2: the start block |
| e_scale | u32 | 32 |
| iteration | u32 | the frame's iteration |
| saturated | u32 | how many stored values were clipped at 65,535 |
| indptr | u32[ncells + 1] | CSR rows |
| gene | u16[nnz] (u32 with bit 1) | per row, ascending: its first gene, then each the step from the one before |
| n | u16[nnz] | the assigned count |
| e | u16[nnz] | the averaged count × 32, rounded half up (steps of 1/32 of a transcript) |
| f | u16[nnz], with bit 0 | the sampled count (f ≤ n) |
| start_nnz, start_e | u32, u16[start_nnz], with bit 2 | the start block: proseg's iteration-0 assignment averaged with this frame's background split, on frame 0's rows and genes, stored as e |

`frames/NNNNNN.counted.zst` is one zstd frame of f32 per cell: the sum of its row's stored e / 32 (`GET
/runs/{id}/counted/{seq}` serves it as it is). A where export, `where-N.bin.zst` with its record `where.json`, is one
zstd frame of `u32 count`, then `u32 i`, `u32 j` (the observed voxel's column and row on the frame's raster), `u8 k`
(its layer), `u32 gene`, `u32 dest` (the label of the cell holding it now, 0 none) and `u32 n`, each `count` long: one
record per pooled entry observed in the box's span (proseg scans the box grown by max(20 µm, 6 × the run's
`diffusion-sigma-far`), 24 µm by default); a bucket's size (its gene at its observed voxel) is counted from the run's
own transcripts, never read from an export.

### Settings

Which files are merged, in order (`engine/config.py` `sources`), depends on the copy and the route. A source folder (a
clone, this computer's production: `pyproject.toml` beside the code) merges `config.toml` (else `config.example.toml`),
then `config.local.toml`, then `--config`; it never reads your own settings file. An installed copy (a wheel) merges its
`config.example.toml`, then your own `~/.config/sopa-companion/config.toml` (under `$XDG_CONFIG_HOME` when that is set),
then `--config`. A notebook session's engine (its `--config` is a `session.toml`) never reads your file, at either of
its loads: `launch` read it already. `config.local.toml` is each copy's own and never ships with a deploy. `config.example.toml` lists every
setting. `"auto"` (an empty value or none means the same) finds things: `[paths] data_dir` and `state_dir` by the copy's
own rule (`dirs.py`; an installed copy's data folder is `~/.local/share/sopa-companion` whatever `$XDG_DATA_HOME`
says, as its state folder), `[proseg] binary` (a source folder's `native/bin/proseg-live`, this Python's scripts folder,
then only a program whose `--version` says `proseg 3.2.0`, never a script, in the user scheme's scripts folder, which a
virtual environment's Python does not read, and on `PATH`; missing, the engine keeps the place its fix puts it; whether
the one found keeps proseg's own count is its `--help` listing `--live-counts`, never its version line) and
each helper's Python (this Python
when it has the feature's packages, else off; `"off"` turns a feature off) (`locate.py`). The engine's log starts with
where each helper runs and which proseg-live it found. Sections: `[server]` (loopback host, port, token), `[desktop]`
(browser, Chrome flags, closing), `[paths]` (data and state folders), `[proseg]` (binary, threads, region limits),
`[analysis]`, `[zarr]`, `[qc]` and `[stain]` (each environment's python and GPU, the Clusters & UMAP defaults, proseg's
count's spool and pace (`counts_spool_mb`, `counts_extract_share`, `counts_mb_s`: none an option on screen), and
the mixing check's thresholds and learning), and `[api]` (a notebook's session only). A notebook's `launch` reads the
same files and then `~/.config/sopa-companion/config.toml`, and hands the engine what it takes from them in
`session.toml` (its `--config`), each helper's Python as a full path or `"off"`, resolved in the kernel. Test engines
also read `[proseg] keep_unsaved` and `max_active_runs`, which the example leaves out.

## The page's modules

| Module | Role |
|---|---|
| `index.html`, `styles/` | the page; design tokens and the app's styles |
| `main.js` | wiring: state, the picker and the run page, WebSocket handling, paint batches, keyboard, leaving and closing |
| `api.js` | every engine request and the WebSocket |
| `shell.mjs` | the first screen's two tools, the header, the one prompt for leaving a run (Close among the ways out) |
| `older-runs.mjs` | the runs earlier versions made and never saved, listed at start |
| `session.mjs`, `storage-boot.mjs` | the page in a notebook's session: its sample, its words, its browser memory |
| `crash-report.mjs` | a session that ended unexpectedly: its report, offered once |
| `proceed.mjs` | Proceed to proseg from a saved stain run |
| `range-chooser.mjs`, `stain-cells.mjs` | the stain size and brightness ranges over their histograms; the cells' values (`PLCV`) |
| `sample-browser.js`, `sample-notes.mjs` | the open dialog; what an open found, in plain words |
| `stage.js`, `scene3d.js`, `transcripts.js`, `volume.js`, `volume-sanity.mjs`, `cluster-gpu.js`, `xenium-masks.js`, `xenium-helpers.mjs` | the WebGL2 stage: 2-D and 3-D (the volume's ray march checked on the CPU), cells, transcripts, image layers, masks, focus, glow |
| `tiled-loader.mjs`, `tiled-view.mjs` | whole-sample views: what to fetch for the view, bounded caches |
| `image-layers.mjs`, `layer-panel.js` | image channels (contrast, colormaps) and the Layers panel |
| `polygons.mjs`, `vectors-decode.mjs`, `prior-layers.js`, `prior-paths.mjs`, `prior-raster.mjs`, `prior-raster-worker.mjs`, `seg-layers.mjs` | outlines: decoding, grid index, painting in a worker, the prior's and the sample's segmentation layers |
| `region.mjs`, `lasso.mjs` | rectangles and lassos |
| `options.js`, `option-guide.mjs`, `prior-options.mjs`, `robustness.mjs` | the picker's options, Live knobs, the starting-cells choice |
| `timeline.js` | the iteration timeline |
| `cellstats.mjs` | per-cell statistics, proseg's count of each cell on a counted frame (`decodeCounted`), the cell readout, most active cells |
| `analysis.js`, `analysis-helpers.mjs`, `analysis-settings.mjs`, `umap.js`, `clusters.js` | clusters and the UMAP pane, its Advanced settings |
| `qc-client.mjs`, `qc-panel.js`, `qc-chip.mjs`, `qc-chart.mjs`, `qc-overlay.mjs`, `qc-areas.mjs`, `qc-copy.mjs` | the mixing check: its state, panel, chip, chart, overlay and words |
| `qc-deeper.js`, `qc-deeper-copy.mjs`, `glow.mjs` | Go deeper's window, words and glow |
| `qc-test-corner.mjs`, `qc-test-corner-copy.mjs` | the test of a setting in the stage's corner |
| `compare.mjs`, `compare-copy.mjs` | Compare's view |
| `stain.mjs`, `stain-settings.mjs`, `stain-exclusions.mjs`, `stain-patches.mjs`, `patch-folder.mjs` | stain segmentation: picker, results, removed cells, training patches and their export dialog |
| `zarr-save.js`, `zarr-save-state.mjs` | the Save to the sample dialog |
| `explain.mjs` | never-greyed-out buttons: what a press still needs |
| `paint-batch.mjs` | one paint per change: a snapshot's panels are written in the frame that shows it |
| `log-export.mjs` | the Log button (no imports: it works when the rest of the page failed to start) |
| `telemetry.mjs` | the performance report |
| `harness-hook.mjs` | read-only counters and a snapshot for a development harness (`?harness=1`) |

Rules the page keeps: every on-screen string comes from a copy deck and passes the banned-words tests (plain words, no
jargon); nothing animates by itself (motion is change-driven, at most 12 paints a second for the two glows, none with
reduced or calm motion); buttons are never greyed out for something still to choose; no synchronous GL query outside a
context's start; a lost GPU context is rebuilt from what the stage holds.

## Tests

- `tests/` (pytest): the engine against a fake proseg (`fake_proseg.py`, which writes proseg's count files, honours
  counts and where tokens, and as a "3.2.0" without `--live-counts` drives the older proseg-live's case), fake analysis
  worker, trainer, probe, stain
  helpers and Cellpose GUI, synthetic Xenium and SpatialData samples (`make_spatialdata.py`), and the real helpers where
  the sopa environment is found (else skipped with a reason). The CPU worker's tests (`test_worker_cpu.py`) run the real
  worker and its cases (`cpu_cases.py`) in a Python that has its libraries: the API environment's, else the sopa
  environment's. `conftest.py` says where those environments come from.
  `tests/api/` is the notebook's side, run with a sopa environment's Python: `launch`, the handle's process and pipes,
  adding results to a SpatialData, a real kernel.
- `frontend/test/` (`node --test`): the page's modules against a small DOM fixture, the copy decks' banned words, the
  shaders' arithmetic.
- `native/proseg`: `cargo test` after `native/build.sh` (the patch's own tests: live control, snapshots, and proseg's
  count: its options and control lines, the PLC1 round trip, forced frames, once per frame, the line before the ack,
  tokens while paused, where exports, the writer's commit order).
