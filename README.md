# SoPA companion

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24116670.svg)](https://doi.org/10.5281/zenodo.24116670)

A companion to [sopa](https://github.com/prism-oncology/sopa) for your Jupyter notebook: `sopa_companion.launch(sdata)`
opens a [SpatialData](https://spatialdata.scverse.org) sample written by sopa in your browser, and what you keep is
added back to that same object, as sopa's own functions add it. Currently only the sopa API is supported (the app
opens from a notebook, on the object you pass), and two tools are supported:

- **Stain segmentation**: find cells in the sample's images with Cellpose, tile by tile over a rectangle, a lasso or
  the whole sample; remove cells by area, size or stain brightness; make training patches for the Cellpose GUI; add the
  result to your SpatialData as a shapes layer, the way sopa does.
- **Proseg Live**: watch [proseg](https://github.com/dcjones/proseg) segment cells from the transcripts, one sampler
  iteration at a time. Pause, step, scrub back to any earlier iteration, change proseg's settings while it runs, then
  *Finish & save* proseg's own results from exactly the state on screen and add them to your SpatialData (shapes plus
  their table, with sopa's own functions).

Around Proseg Live: live clusters and a UMAP of every iteration (on the CPU, or on an NVIDIA GPU with RAPIDS), a
**mixing check** that estimates how much of each cell's RNA looks like a neighbour's (a ResolVI model learned once per
sample on an NVIDIA GPU, or on a Mac's own GPU), **Go deeper** for one cell, a quick **test of a setting** on a small
square, other segmentations of the sample drawn on the tissue, and **Compare**, which reads several segmentations side
by side with the same model, each on its own UMAP and clusters (lined up with the run's) or all on one shared map.

How it is built: [docs/ARCHITECTURE.md](https://github.com/jackytamkc/sopa-companion/blob/main/docs/ARCHITECTURE.md). Working on the app itself:
[docs/DEVELOPING.md](https://github.com/jackytamkc/sopa-companion/blob/main/docs/DEVELOPING.md).

## Quick start

After [Install](#install), in a notebook of that environment:

```python
import sopa
import sopa_companion

sdata = sopa.io.toy_dataset(length=1000)   # or your own sample: sdata = spatialdata.read_zarr("sample.zarr")
sdata.write("toy.zarr")                    # the app works on a SpatialData saved to a local .zarr
app = sopa_companion.launch(sdata)         # opens the app in your browser; the notebook stays free
```

The app opens in a new tab of your browser. On a remote computer (SSH, VS Code's Remote-SSH included) it prints a link
instead, with the `ssh -N -L` command for a tunnel; VS Code forwards the port by itself (keep the same port number on
your computer). **Close** in the app, or `app.close()`, ends the session. The rest is under
[Use it from a notebook](#use-it-from-a-notebook).

## Requirements

- **Linux**, 64-bit (x86-64 or ARM64), with glibc 2.28 or newer (Ubuntu 20.04, Debian 10, RHEL 8 or later);
  **macOS** 11 or newer on Apple silicon; or **Windows** 10 or 11, 64-bit (x64). macOS and Windows are new in this alpha
  and are tested on GitHub's machines, which have no GPU. WSL2 on Windows takes the Linux install.
- **Python 3.12 or 3.13**, in the environment where you installed sopa
  ([sopa's guide](https://prism-oncology.github.io/sopa/getting_started/)), or a new one: the install brings sopa.
- **A modern web browser with WebGL2** (Firefox 133 or newer, Chrome, Edge), on the computer you look at.
- On Linux, `setpriv` from util-linux, which every common Linux has: the app's helpers end with its session (macOS and
  Windows end them in their own ways).
- **The mixing check requires a GPU: an NVIDIA GPU, or on a Mac with Apple silicon its own GPU.** It learns each sample
  once on GPU 0 (PyTorch with CUDA, driver 580 or newer for the CUDA 13 packages; on Windows with PyTorch's CUDA build,
  not tested yet), or on a Mac without one on Apple's GPU, where it passes a short test of PyTorch's operations first
  (PyTorch 2.15 needs macOS 15 or later there; on macOS 14 the install brings PyTorch 2.14, which passes it). On an M2
  Ultra it learns about 3.5 times as slowly as on a recent NVIDIA workstation GPU and gives equivalent results (not the
  same cells' numbers: another random stream, as another seed gives), and a model learned there is kept apart from an
  NVIDIA GPU's. The free memory a fit takes grows with the gene panel, hardly with the number of cells (on a Mac the
  cells' counts too, in its own memory):

  | Gene panel | Free GPU memory needed |
  |---|---|
  | 300 genes | about 1.4 GB |
  | 1,000 genes | about 2.0 GB |
  | 5,000 genes (Xenium Prime 5K) | about 4.8 GB |
  | 10,000 genes | about 8.4 GB |

  For larger panels, use a GPU with more memory. Its scores, Go deeper and Compare run on the GPU with RAPIDS, else on
  the CPU (on a Mac, the CPU).
- **Everything else runs without a GPU.** Clusters and the UMAP run on the CPU with what sopa installs (about twice a
  second for a region of a few thousand cells, every 3-5 s for a 40,000-cell sample), and faster on the GPU with RAPIDS
  (Linux only). Stain segmentation uses CUDA, or Apple's GPU on a Mac, when it can and falls back to the CPU (Apple's
  GPU with PyTorch 2.15 needs macOS 15 or later; on macOS 14 the install brings PyTorch 2.14).

## Install

What every computer needs:

- **conda**, if you have none yet: [Miniforge](https://conda-forge.org/download/) (conda with conda-forge's packages).
  Any environment with Python 3.12 or 3.13 does too, sopa's own included.

Then follow the steps for your computer. Each section is complete. To update later, in the same environment:
`pip install --upgrade "sopa-companion[all]"`.

### macOS (Apple silicon: M1 or later)

`uname -m` in Terminal says `arm64` on Apple silicon. On an Intel Mac it says `x86_64`: read
[Another computer](#another-computer) before step 3.

1. Miniforge, if you have no conda (in Terminal; answer yes when it offers to initialise conda, then open a new
   Terminal window):

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
   bash Miniforge3-MacOSX-arm64.sh
   ```

2. An environment with Python and JupyterLab:

   ```bash
   conda create --name sopa -c conda-forge python=3.12 jupyterlab
   conda activate sopa
   ```

3. The app, from PyPI (with sopa, Cellpose, PyTorch and the rest; a few minutes):

   ```bash
   pip install "sopa-companion[all]"
   ```

4. Check it: `sopa-companion --version` says the app's version and where its proseg-live is.
5. Try it: `jupyter lab`, open a new notebook, and run the [Quick start](#quick-start).

On a Mac, Cellpose and the mixing check use Apple's GPU where it passes a short test of PyTorch's operations; where it
does not, Cellpose runs on the CPU, the mixing check's panel says why, and the run's Log has the details. PyTorch 2.15
needs macOS 15 (Sequoia) or later for Apple's GPU (on macOS 14 every linear layer would stop the process), so step 3
installs PyTorch 2.14 on macOS 14, in place of a newer one already in the environment, and the newest PyTorch on
macOS 15 or later. On an M2 Ultra, Cellpose ran about 8 times as fast on Apple's GPU as on the CPU.

### Windows (64-bit)

1. Miniforge, if you have no conda: run the installer
   [Miniforge3-Windows-x86_64.exe](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe).
   Then open **Miniforge Prompt** from the Start menu, run `conda init powershell`, and close it. The steps below run
   in **PowerShell**; if it says running scripts is disabled, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` there once and open a new PowerShell window.
2. An environment with Python and JupyterLab:

   ```powershell
   conda create --name sopa -c conda-forge python=3.12 jupyterlab
   conda activate sopa
   ```

3. The app, from PyPI (a few minutes):

   ```powershell
   pip install "sopa-companion[all]"
   ```

4. Check it: `sopa-companion --version`.
5. Try it: `jupyter lab`, a new notebook, the [Quick start](#quick-start). On Windows the notebook shows the app's link
   to click: the app does not open a browser by itself there yet.

On Windows, keep samples in folders with short paths (Windows refuses paths longer than 260 characters unless long paths
are switched on). Exporting training patches is not available on Windows yet. For the mixing check on an NVIDIA GPU,
PyTorch's CUDA build replaces the CPU one pip installs on Windows (`pip install torch torchvision --index-url
https://download.pytorch.org/whl/cu130`); the mixing check on a Windows GPU is not tested yet.

### Linux (64-bit: x86-64 or ARM64)

1. Miniforge, if you have no conda (open a new terminal after it):

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

2. An environment with Python and JupyterLab (or your sopa environment):

   ```bash
   conda create --name sopa -c conda-forge python=3.12 jupyterlab
   conda activate sopa
   ```

3. The app, from PyPI:

   ```bash
   pip install "sopa-companion[all]"
   ```

4. Check it: `sopa-companion --version`.
5. Try it: `jupyter lab` (or your own Jupyter, VS Code or PyCharm with this environment's Python), then the
   [Quick start](#quick-start).

### What the install line does

It installs two packages from [PyPI](https://pypi.org/project/sopa-companion/): the app, and `proseg-live` (proseg
3.2.0 with live mode and proseg's own count of each cell, GPL v3) built for this computer, with everything they need:
sopa and spatialdata, Cellpose, PyTorch, and scvi-tools with scviva-tools for the mixing check. The same files, with
their checksums, are on each [release](https://github.com/jackytamkc/sopa-companion/releases) here.

### Another computer

PyPI has proseg-live wheels for Linux x86-64 and ARM64 with glibc 2.28 or newer, macOS 11 or newer on Apple silicon,
and Windows x64. Elsewhere (an Intel Mac, say) pip builds proseg-live from its source, which needs Rust 1.88 or newer
([rustup.rs](https://rustup.rs)) and a C compiler (on a Mac, `xcode-select --install`): install those first, then
step 3's line is the same.

### Optional extras

**Optional: Clusters & UMAP on the GPU.** They need nothing more on the CPU. On Linux with an NVIDIA GPU they run
faster with RAPIDS, which for now goes into an environment of its own:

```bash
conda create --name rapids python=3.12
conda activate rapids
pip install 'rapids-singlecell-cu13[rapids]' --extra-index-url=https://pypi.nvidia.com
python -c 'import sys; print(sys.executable)'     # this environment's Python, for your settings file
conda activate sopa
```

Then name that Python in your settings file. Your own settings file is `~/.config/sopa-companion/config.toml` (under
`$XDG_CONFIG_HOME` when that is set; on macOS `~/Library/Application Support/sopa-companion/config.toml`, on Windows
`%LOCALAPPDATA%\sopa-companion\config.toml`); put there only the lines you change ([Settings](#settings)):

```toml
[analysis]
python = "~/miniforge3/envs/rapids/bin/python"
```

With an older driver or GPU, use `rapids-singlecell-cu12[rapids]` together with PyTorch's CUDA 12 build (`pip
install torch --index-url https://download.pytorch.org/whl/cu126`) in the sopa environment.

**Optional: Cellpose's own window**, to train a model on exported patches where you have a desktop:
`pip install 'cellpose[gui]'` (sopa installs Cellpose without it).

## Use it from a notebook

```python
import sopa_companion
app = sopa_companion.launch(sdata)     # sdata: a SpatialData written to a local .zarr
```

- **What it does.** `launch` starts a private engine for this kernel alone, on the object's zarr (elements only in
  memory are left out, and a warning names them), and opens the app where `[desktop] browser` says (a new tab of your
  default browser, or with `"chrome"` a window of its own), or prints a link when there is no display or you are on
  SSH. `app.open()` opens it again when its tab or window went away (a crash, a laptop closed); the session goes on
  meanwhile. In a notebook it returns at once and you keep working; a script waits until the app closes. The sample
  is read and saved with the kernel's own Python (its sopa and spatialdata); on Linux the app needs `setpriv`, and
  everywhere a web browser (without one it prints the link). On Windows the notebook shows the link to open for now,
  and the app's own Chrome window (`"chrome"`) is Linux's alone.
- **Its options.** `tool="proseg"` or `"stain"` opens that tool; `points_key` chooses the transcripts among several
  points elements; `open` is `"auto"`, `"window"`, `"link"` or `"none"`; `gpu` is one GPU number or `{"analysis": …,
  "stain": …, "qc": …}`; `mixing_check=True` switches the mixing check on; `timeout` limits the start (120 s).
- **Results go into your object.** *Add to your SpatialData* adds the result to the object you passed, under new
  names, between notebook cells (never while one runs), as sopa's own functions add it.
  With sopa's auto-save on it is written to the zarr too. `app.added` lists what was added.
- **The link** (`SoPA companion: http://127.0.0.1:<port>/?code=…`, printed when the app opens no window, and then in
  `app.url`; on SSH with the `ssh -N -L` command for a tunnel) signs one browser in, once, within 10 minutes. Until it
  is used, anyone who can read it can use it: keep it private (a saved notebook keeps the cell's output, so clear the
  outputs before sharing a notebook within those 10 minutes). Once it has been used the notebook's output is replaced
  by a line saying when, so a notebook saved afterwards keeps no code; so is a link no longer needed (a newer one shown
  by `app.open()`, or the browser the app opened signing in after all), which says why. A browser the app opens takes a
  code of its own that lives seconds (then in `app.url`, already used), never on a command line; one that shows no app
  within 20 seconds gets the link printed too. A tunnel must keep the port's number: the app answers only on its own.
- **Closing.** The app's **Close** ends the session, asking first about a run not added yet (*Add to your SpatialData
  and close*, *Discard and close*, *Keep working*). `app.close()` closes it from the notebook and discards a run not
  added; `app.close(pending="add")` adds it first. A tab or window closed (its page says goodbye) without an answer, or
  a kernel that stops, discards it too. A tab that crashed or lost its connection pauses its run instead, and the
  session waits for `app.open()` or `app.close()`.
- **Its folders.** The session's private folder (`session.toml`, logs, the Chrome window's own profile) is
  `$XDG_RUNTIME_DIR/sopa-companion/<id>/`, else `~/.cache/sopa-companion/sessions/<id>/` (on macOS
  `~/Library/Caches/sopa-companion/sessions/<id>/`, on Windows `%LOCALAPPDATA%\sopa-companion\Cache\sessions\<id>\`);
  its runs and caches go into
  `<scratch_dir>/sopa-companion-<id>/`. Both are removed when the session ends, and a session left by a crash is removed
  at the next `launch`. `model_cache=` keeps the mixing check's learned models for later sessions. Training patches
  are exported into a folder you choose at each export, always outside the session's folders, so they stay; with
  `patches_dir=` that choice starts there (a folder per sample and resolution in it). The Clusters & UMAP settings
  that new runs start with last for the session only.
- **From sopa.** A sopa with `packaging/sopa-fork/sopa.patch` (sopa's optional `companion` extra, proposed) opens it
  as `sopa.companionapp(sdata)` once the companion is installed.

### Settings

`launch` reads the app's own settings (an installed copy's own `config.example.toml`), then your own
`~/.config/sopa-companion/config.toml` (under `$XDG_CONFIG_HOME` when that is set; macOS and Windows: as
[Install](#install) says). An installed copy's own files are
inside the Python environment, so your own file is the place for your settings. `launch` takes these from them, and
its own arguments win:

| Setting | What |
|---|---|
| `[api] engine_python` | the engine's Python; empty: the kernel's own, which then needs the engine's packages (the wheel brings them) |
| `[api] scratch_dir` | where each session's runs and caches go, removed at close; empty: the system's temporary folder |
| `[api] model_cache` | a folder that keeps the mixing check's learned models between sessions; empty: they go with the session |
| `[desktop] browser` | where the app opens: `"default"`, your default browser; `"chrome"`, a window of its own; or a browser's `.desktop` file |
| `[desktop] chrome_flags` | flags for the session's Chrome window (on a remote desktop, `"--use-angle=vulkan"` and `"--ignore-gpu-blocklist"` let it use the GPU) |
| `[proseg] binary` | the proseg-live program: `auto` (the default) finds the one installed with the app; or a path |
| `[analysis] python`, `gpu` | Clusters & UMAP's Python: `auto` (or empty), the kernel's own when it has scipy, scikit-learn, igraph, umap-learn and numba (sopa brings them; on the CPU); `off`; or another environment's Python (with RAPIDS: on the GPU) |
| `[stain] python`, `gpu` | Cellpose's Python: `auto` (or empty), the kernel's own when it has Cellpose; `off`; or another environment's. It also runs the Cellpose GUI, which needs `cellpose[gui]` there |
| `[qc] python`, `gpu` | the mixing check's Python: `auto` (or empty), the kernel's own when it has scvi-tools, scviva-tools and torch; `off`; or another environment's |

Other settings in your own file do not reach the session's engine, which reads the app's own files and what `launch`
hands it. GPU numbers count the GPUs the kernel sees (its `CUDA_VISIBLE_DEVICES`, else nvidia-smi's numbering); the
mixing check learns on GPU 0 only (a Mac's own GPU where it has no NVIDIA one).

## Use it

### The first screen

The app opens on your SpatialData: choose **Stain segmentation** or **Proseg Live**. The header names the sample on
every screen; **New run** stays in the tool, and a click on the app's name goes back to the two cards. Both leave the
open run first, asking about it when it is not added yet ([One run at a time](#one-run-at-a-time)). While the app
prepares a large SpatialData, the first screen shows how far it is, and a card pressed meanwhile opens by itself when
the sample is ready.

When the sample opens, the app says in one line each what it found that matters (details in an ⓘ): transcripts that
seem to be in pixels rather than micrometres, several transcript elements of which one is used, images left out, z
given as plane numbers, element names the app cannot use. A sample with images but no transcripts opens for Stain
segmentation only, and one with transcripts but no image it can show for Proseg Live only.

### Stain segmentation

Choose the image and its nucleus channel; a cell body or membrane channel is optional, and Cellpose's model (`cpsam` by
default, or a model you trained under **Your models**). A built-in model not on the computer yet is downloaded by its
first run (`cpsam`: about 1.15 GB), once, into Cellpose's own model folder (`~/.cellpose/models`). **Resolution** is one of the image's own pyramid levels by its
SpatialData name: `scale0` is the image as stored, `scale1`, `scale2`, … the smaller copies the sample already holds.
**Auto** keeps the image at its own size; Cellpose's settings go by Cellpose's own names (`diameter`, in pixels of
`scale0`; `flow_threshold`; `cellprob_threshold`), each with its plain meaning in its tooltip; more Cellpose settings
fold under the method. *Tissue only* (on by default) segments only tiles that touch the tissue (the sample's
`region_of_interest`, else a tissue outline found on the image). Choose a rectangle, a lasso or the whole sample and press
**Start**: nothing runs before that. Images are segmented in tiles (`[stain] tile_pixels`, 1024 by default, overlapping
by at least `overlap_pixels`); only 2-D images are supported (extra axes of size one are fine), with the channel names
and coordinates SpatialData gives them.

Finished tiles appear as outlines over the image; the panel shows tile progress, cells and their sizes. Pause and Stop
act after the current tile; Resume continues from the finished tiles. Tiles are joined without seams: each tile keeps
the cells whose centre lies in its own part, and a cell only one tile found is never lost. **Remove cells** works on
the result without changing it: draw areas (click one on the image and press Delete to bring its cells back; Undo and
Redo go back and forth), or keep only the cells within a size range and a brightness range in a channel. **Add as
shapes** adds the result to your SpatialData (`cellpose_boundaries_1`, then the next free number; the dialog says when a
name would replace an element); the added layer then appears among Proseg Live's starting cells, and *Proceed to proseg*
opens the proseg setup on it, over the same region.

**Training patches.** From a paused or finished run, or straight from the sample (the *Training patches* section at the
top of the picker), the app proposes varied boxes with cells (8 boxes of 256 × 256 pixels by default; move, resize, drop
or add them). **Export patches** asks every time where to save: it suggests the folder this sample's patches at this
resolution went to last, else a new one named after the sample and its resolution under
`~/sopa-companion/training-patches` (`<sample>-scale0`, ...); **New folder** names one inside the folder shown. The image
crops and Cellpose masks go straight into the folder you choose, beside what is already there, under names of their own
(`<sample>-<date>-<time>-01.tif`, `…-01_seg.npy`), and never replace a file. Keep one folder per model: the Cellpose GUI
trains one model on every patch in the folder it has open, so a sample that needs a model of its own gets a folder of its
own (the dialog says when a folder already holds patches cut at another resolution or of other channels). The app's
records of each export are in the folder's hidden `.sopa-companion`. A folder in a zarr store, a hidden folder, the
app's own folders and Python environments, a notebook session's folders, `/`, your home folder itself and folders you
cannot write to are refused, in plain words. The Cellpose GUI loads the `_seg.npy` masks with Python's pickle, which
can run code: a mask someone else puts in a folder they can write to runs as you when the GUI opens it or trains on it.
Keep training folders where only you, or people you trust, can write. **Open in Cellpose GUI** opens the first crop
with its outlines where the notebook's computer has a desktop (elsewhere the panel gives the command to start it by
hand); correct them, save with Ctrl+S and train a model in the GUI. The panel notices corrected patches; a GUI-trained
model appears under **Your models** for new runs, with the resolution its patches were cut at.

### Proseg Live

1. **Start from**: the sample's cells and nuclei (the suggested pairing, as proseg uses Xenium's), or any one shapes
   layer or cell-assignment column of the transcripts on its own (Cellpose cells, an earlier proseg result, ...). Shapes
   are assigned to transcripts with sopa's point-in-polygon join.
2. **Region**: a rectangle, a lasso, or the full sample. Regions up to 1.2 mm across run in full detail; anything larger
   runs in **whole-sample mode**, where the view streams only the part on screen.
3. Every run follows proseg's own schedule (burn-in at 2 µm voxels, sampling at 1 µm, then hill-climbing) and its
   defaults; every option is under *Advanced proseg options*, and only values you change are passed.
4. **Start segmenting** prepares the run and stops on proseg's starting point (frame 0). Press **Start** (or **Step**).
   Pause (Space), Step (S) and the speed (Max, Fast, Slow) control proseg between iterations. **Live knobs**
   (temperature, compactness, prior weight, diffusion, ...) apply from the next iteration and are marked on the
   timeline. The timeline shows proseg's phases and log-likelihood: click or drag to look at any iteration; ←/→ step
   through them, **Live** (L) follows new ones.
5. At the end of proseg's standard run the run pauses and **Finish & save** is the main button: it writes proseg's own
   results from that state into the run's `output/` folder. **Resume** samples on, open-ended, until you finish.
6. **Add to your SpatialData** adds the result to your object: a shapes layer and its AnnData table (sopa's
   `_read_proseg` and `add_parsed_table`, optionally `set_boundaries_attrs`, *Make it the default segmentation*). The
   dialog's names are checked against your SpatialData's elements when it opens and says when one would be replaced.
   The table's `uns["sopa_companion"]` records the run, iteration, starting cells, region and options.

**Display.** 2-D or 3-D (V); filled cells or outlines (1/2); transcripts (X) coloured by the cell or cluster they sit in;
changed voxels flashing (T); napari-style image layers (contrast, gamma, colormap, opacity, blending); proseg's own
voxel prior; hover a cell for its statistics (its count in proseg's count and what proseg sets aside as background),
click to follow it over the run.

**Clusters and UMAP.** Every run is clustered as it goes on the GPU, or on the CPU (its clusters start from the last
ones, so colours change little): proseg's own count of each cell (the iterations proseg counts as the run goes),
normalisation, PCA, kNN, Leiden, top genes, and a UMAP warm-started from the previous one so its dots fly to their new
places. A region with too few cells says so
instead. **Advanced settings**, under the UMAP pane's title: cluster resolution, neighbours per cell, dot spacing, the
fewest transcripts a cell needs and the genes used (how many of the panel's most variable genes; a smaller panel uses
all of its genes). **Apply** groups the run's cells again: the clusters shown stay until the new ones are ready, then
the dots fly to their new places. **Reset** goes back to the defaults, and new runs start with the settings last
applied. Compare keeps its own fixed recipe. On the CPU, genes used stops at 5,000 (10,000 on the GPU). Where they
cannot run (`[analysis] python` off, or a Python that lacks what they need) the pane stays shut and Layers > Cells says
what is missing, or where to set it.

**Mixing check** (off until you switch it on). For every cell, a model's estimate of how much of its RNA (proseg's own
count of it, averaged over proseg's random background draw) looks like it came from neighbouring cells: "18 in 100 cells
look mixed", the change since the start like for like (the cells proseg started with, their RNA kept as proseg keeps it
now, each cell against itself at the same depth), and a verdict only when the change is larger than repeat runs give. A
run started from nuclei, with their cells or without (Xenium's cells and nuclei, say), has no start to compare with:
proseg's first iteration holds only the nuclei. It is an estimate, not a count of boundary errors, and it cannot tell
RNA apart between cells of the same kind. The first time it is on for a sample it learns that sample once (a ResolVI
model on GPU 0, or a Mac's own GPU, a few minutes); later runs of the sample start at once. Learning starts once the
GPU has the free memory the sample's panel needs ([Requirements](#requirements)): until then the panel says how much it
needs and how much is free, and a GPU with less than that in all is told so at once (a Mac never waits: it learns at
once, unless it has too little memory in all). It may suggest a setting to try
(*Try it* watches the change; *New run with this setting* is the clear test).
`python -m sopa_companion.engine.qc learn <sample.zarr>...` learns samples ahead, with the app closed.

**Go deeper** (D, with a cell selected): how the model splits that cell's RNA (proseg's count of it) into its own, from
neighbours and loose RNA in the tissue, which genes, and which neighbours it seems to come from. The cells involved
stay lit while the rest greys out; **Show on the image** makes the transcripts proseg counts for the cell that likely
came from a neighbour glow pink (one just outside the outline is RNA proseg moved in), a rough guide (the model splits
RNA per gene, not per transcript). Paused, or following a cell while the run plays, the dots are proseg's exact share;
on an earlier iteration they are placed approximately, or not at all, and the tooltip says so.

**Test a setting here** (in Go deeper, or under the mixing check's row, with a cell selected) asks whether a value of
one of proseg's own settings (cell compactness, the hold to the starting cells, far drift, the starting grid,
background gaps, drift on or off, the short drift distance, crowding levels or depth layers) keeps the suspects (the
transcripts proseg counts for the cell that likely came from a neighbour: its pink dots) out of this cell and of nearby
cells with the same pattern. The "?" beside the setting plays what it does to a made-up cell, one panel per value.
Fresh runs of your region (or, when it is
over 500 µm, of a 500 µm square around the cell, cut to your region where it is narrower) go to the end of proseg's
schedule, all at once: 3 as now and 3 with your value (4 and 4 for a run started from cells only), about a minute
while your run waits at its end. A corner of the image shows one of them as now and one with your value, side by side.
The answer says whether proseg keeps the suspects in their cells, whether they are clearly fewer with your value than
their own RNA at the same spot, what it cost and where they went; then *New run with this*, *Keep my setting* or *Run
again*. It works on any run that reaches its last iteration, a whole sample too. A press before your run reaches its
last iteration waits for it, and one with the mixing check off switches it on first. Where the test was not measured,
it runs all the same and its answer says so in its tooltip. A cell that doesn't look mixed overall is tested on its
pink dots all the same. The form says why when a cell can't be tested (too few transcripts, no pink dots), or a run
(being saved, ended before its last iteration, or without proseg's own count). Every proseg
setting is global: the test shows what one value does to the area, not to one cell.

**Other masks and Compare.** *Layers > Load Other Mask* draws any other segmentation (mask) the sample holds beside the
run's cells (Xenium's, an imported one, a stain segmentation or proseg result added earlier). **Compare** reads the run's
frame and up to three of the sample's segmentations with the sample's mixing model, side by side: their UMAPs on one
shared layout, how mixed their cells look, at a common depth too, on the area all of them cover. Each column reads its
own count and says which: proseg's for the run and a saved proseg result, Xenium's for Xenium's cells, the transcripts
inside the outlines for a segmentation with no count of its own (Cellpose, a stain). Proseg sets background aside, so
its columns read less mixed for that reason too. It sets no verdict between columns.

**Log.** The Log button saves a plain-text log of the session and the open run (settings, schedule, what you did, the
numbers behind the mixing check's words, what went wrong). Nothing is saved without your click; the session's token is
never in it. When the app closed unexpectedly (the engine stopped, or its page crashed), the next start offers that
session's log once: *Save a copy…* or *Not now*.

### One run at a time

The app holds one run (a proseg run, a stain run or a set of training patches) and keeps no runs between sessions.
Leaving it (the app's name, **New run**, *Proceed to proseg*, **Close**) asks first when it holds a result that is not
in your SpatialData yet: **Add to your SpatialData** finishes the run if it still samples (a stain run stops after its
current tile), adds its cells under new names and then removes the run; **Discard** removes it; **Stay** keeps it open.
A run with nothing to lose (not started, or added already) goes without a question.

### What you are seeing

Proseg models cells as sets of voxels (2 µm during burn-in, then 1 µm; 4 layers in z by default) and samples boundary
changes with Metropolis–Hastings. "Voxels changed" counts 2-D columns whose label changed; the log-likelihood is
proseg's expression likelihood (not the full posterior). Every frame on screen is proseg's own state at that iteration, read
from the snapshots it writes; nothing is re-simulated in the page. The outlines are drawn from proseg's voxels, for
drawing and places only: every number about a cell's RNA (its readout, the clusters and the UMAP, the mixing check, Go
deeper, the test of a setting, Compare's run column) is proseg's own count of the cell, gene by gene, averaged over
proseg's random background draw, never a count of the transcripts inside its outline. Proseg counts the iterations
the app asks for as the run goes (and its first iteration, pauses and the end). On an iteration it did not count, the
clusters and the check are the newest counted iteration's before it, and say which; a cell's readout there shows that
iteration's own draw of proseg's count, its tooltip saying so.

## Guarantees and limits

- **What you save is what you see.** Pause and *Finish & save* wait for proseg's own acknowledgement, written only once
  every queued snapshot is on disk, so the confirmed iteration is exactly the saved state.
- **Bounded history.** A snapshot for every iteration up to 2,000, then every 5th to 10,000, every 25th to 50,000 and
  every 100th after; the state at any pause, finish or end is always kept. Past iteration 2,000 the iterations the app
  asks proseg to count are kept too, at most one between two kept snapshots, so at most as many again. A run pauses
  itself when the data disk has less than 15 GB free (a test's runs stop instead, and the test says why).
- **Proseg's own count, bounded too.** The counts the app asks proseg for wait in a folder on the RAM disk (or the
  temporary folder), at most `[analysis] counts_spool_mb` (1 GB) a run and twice that in all, and go with the run; those
  of the first iteration, the switch to 1 µm voxels and the end stay in the run's folder. The app asks for them so that
  counting takes about a tenth of proseg's time at most, and counting never changes what proseg samples (seeded, it
  saves byte for byte what proseg saves without it).
- **Writing into the sample's zarr is all-or-nothing.** Names follow SpatialData's rules and never silently replace
  anything; a replaced element is moved aside first and put back if anything fails, also after a crash. Every step is
  written to a journal before it is taken; the next write into that zarr rolls back that zarr's own interrupted ones
  only (never another zarr's whose name begins alike), and only once the process that wrote them has ended; a write that
  finds another session's write into the same zarr under way refuses, touching nothing. A folder kept for a write that
  could not be rolled back whole is left in `.proseg-live-backups` beside the zarr, never deleted. Writes into one zarr
  never overlap, and the app waits for a running one before it closes. Apart from what you add, the app only reads
  your sample.
- **Graphics memory.** If the GPU cannot hold an upload the page rebuilds its view; a second loss within a minute halves
  the size of the images in use.
- **Diagnostics stay on this computer.** While a run is open the page sends the engine a performance report every 10 s
  (how often it draws, long tasks, memory, the display and GPU renderer, what it downloaded; no cell, run or file name).
  They are kept in `<state_dir>/logs/telemetry.jsonl` (at most 5 MB plus one older file). The page also keeps its own
  log with the engine (`<state_dir>/logs/page-logs/`, the newest 6) for the crash reports in
  `<state_dir>/crash-reports/` (the newest 10, with the session's secrets blanked).
- **Network use.** The engine and page talk over loopback only. Downloads happen only when needed: a Cellpose built-in
  model missing from Cellpose's cache.

## Licence

- **proseg** is Copyright (C) 2024 Daniel C. Jones and licensed under the GNU General Public License v3.
  `proseg-live` is proseg 3.2.0 with this repository's `native/proseg-live.patch`; it is derived from proseg and
  covered by the GPL v3, and whoever passes it on must do so under its terms, with the source. It is a package of its
  own (`native/package/`): every release carries its complete source, `proseg_live-<version>.tar.gz` (proseg's crate
  with the patch applied, its `Cargo.lock`, the patch itself and proseg's licence), beside its wheels, which carry the
  licence too.
- The app runs `proseg-live` as a separate program and talks to it through files and a control file; the
  `sopa-companion` package holds no GPL code. Its dependencies (FastAPI, uvicorn, numpy, zarr, sopa, spatialdata,
  Cellpose, scvi-tools, RAPIDS and others) are installed separately, each under its own licence.
- **The app's own code** (everything outside `native/`) is under the BSD 3-Clause licence: see
  [LICENSE](https://github.com/jackytamkc/sopa-companion/blob/main/LICENSE).

To cite the app, see [CITATION.cff](https://github.com/jackytamkc/sopa-companion/blob/main/CITATION.cff) (GitHub's
"Cite this repository"). Every release is archived on
Zenodo: [10.5281/zenodo.24116670](https://doi.org/10.5281/zenodo.24116670) names all versions and leads to each
version's own DOI.
