# Developing SoPA companion

The supported way to use the app is from a notebook, `sopa_companion.launch(sdata)`: see the [README](../README.md).
This page is for working on the app itself: the standalone engine that its tests and its development harness drive, a
source folder, the patched proseg, the tests and the layout. How it is built: [ARCHITECTURE.md](ARCHITECTURE.md).

## The standalone engine

The same engine runs on its own, on samples it opens (`.zarr` folders), with no notebook:

```bash
sopa-companion                      # start the engine (or reuse it) and open the app in your browser
sopa-companion launch --print-url   # a link for an SSH tunnel: it signs one browser in, once, within 10 minutes; keep it private
sopa-companion status               # --json for a machine-readable answer; never shows credentials
sopa-companion stop                 # refuses while a run is not saved to its sample yet
sopa-companion stop --force         # no questions: a run nobody saved or answered for is discarded
sopa-companion serve                # the engine alone, in the foreground
sopa-companion --version            # the app's version and the proseg-live it runs
```

`python -m sopa_companion` does the same. `sopa-companion install-desktop` adds a launcher to the applications menu
and the desktop.

The engine listens on `127.0.0.1` only, port 8777 (the next free one up to 8876 when it is taken; another port with
`--port` or `[server] port`). Every request needs the engine's token or the page's own key, and changes must come from
the app's own page. A page trades its one-use code for its key once and keeps it in its tab: the key travels in a
header, never in a cookie, so no other site or service on this computer is ever sent it. One engine runs per data
folder. `launch` opens the app where `[desktop] browser` says: `"default"` (what ships) is your default web browser, as
the desktop names it (GNOME's, which `xdg-open` uses, then `xdg-settings` and `xdg-mime`); `"chrome"` (and `"auto"`, the
previous versions' value) is an app window of its own in Google Chrome or a Chromium that is not a snap, with its own
profile, one per engine, and `[desktop] chrome_flags` (without Chrome the default browser opens instead, and `launch`
says so); on Linux the name of a browser's `.desktop` file picks that browser. `launch --browser default|chrome` chooses
for one launch, and `status` says where the app would open. Either way the browser opens a page in the engine's private
folder, which sends it on with a one-use code, so the code is never on a command line (any account on the computer can
read those). A snap browser gets that page in a folder of its own (`~/snap/<name>/common/sopa-companion`), a flatpak
through its file forwarding; when no browser can read it, or there is none, `launch` says why and prints a one-use link
to open by hand instead. A browser that took the page but shows no app within `[desktop] open_wait_s` (20 seconds) gets
that link printed too. A link is printed to a terminal only: started from its icon, `launch` prints the address and
how to ask for a link (`launch --print-url`), never a live code where nobody sees it.

Tools it uses when they are there: `setpriv` from util-linux (helpers end with the engine), `zenity` (questions when the
app closes), `gio`, `xdg-settings` or `xdg-mime` (which browser is the default), `xdg-open`, and for the desktop entry
`desktop-file-validate` and `xdg-user-dir`.

**Settings.** An installed copy's engine reads your own settings file, `~/.config/sopa-companion/config.toml` (under
`$XDG_CONFIG_HOME` when that is set): put there only the lines you change from the app's defaults
(`config.example.toml` in the installed package lists every setting). Runs and caches go to
`~/.local/share/sopa-companion`, the engine's private state (runtime file, logs, crash reports) to
`~/.local/state/sopa-companion`. Each helper runs in the app's own Python when that Python has what it needs (`"auto"`,
the default); `"off"` turns a feature off, and a path names another environment's Python. Worth a look:

- `[proseg] threads` and `tiled_threads`: `"auto"` takes half the cores (at most 32) for a run on a region and three
  quarters for a whole-sample run.
- `[analysis] gpu`, `[stain] gpu`: which GPU (nvidia-smi's numbering). `[qc] gpu` must be 0: the mixing check learns
  on GPU 0 only, at the lowest priority, once it has `gpu_min_free_mb` free.
- `[analysis] python`: this environment (`auto`), or another's Python: RAPIDS (cupy, cuML, cuGraph, cuDF) runs Clusters
  & UMAP on the GPU, else the CPU (scipy, scikit-learn, igraph, umap-learn, numba). `[analysis] backend`: `auto` (the
  GPU when RAPIDS and a CUDA GPU are there), `gpu` or `cpu`.

**Desktop entry.** `sopa-companion install-desktop` installs a **SoPA companion** launcher in the applications menu and
on the desktop, with the app's icon; its context menu has *Stop engine* and *Show engine log*. `--uninstall` removes
what it installed, `--no-desktop-icon` leaves the desktop alone and `--dry-run` shows what it would do. Run it again
after moving the app's folder.

**The first screen.** Choose **Stain segmentation** or **Proseg Live**, then **Open sample…** (a `.zarr` folder) or one
of the **Recent** samples (shared by both tools; the 30 opened last). A sample that cannot be opened says why in one
line (the whole of it in its tooltip) and is not added to Recent. Results are saved into the sample's zarr: **Save as
shapes** (stain) and **Save to the sample** (proseg), where a notebook's session says *Add to your SpatialData*; an
existing element is replaced only when you choose *Replace*.

**One run at a time.** The app holds one run (a proseg run, a stain run or a set of training patches) and keeps no
runs between sessions. Leaving it (the app's name, **New run**, **Open sample…**, *Proceed to proseg*, **Close**) asks
first when it holds a result that is not in its sample yet: **Save to the sample** finishes the run if it still
samples (a stain run stops after its current tile), writes its cells into the sample under new names (nothing is
replaced) and then removes the run; **Discard** removes it; **Stay** keeps it open. A run with nothing to lose (not
started, or saved already) goes without a question. A run whose *Finish & save* was pressed is saved before the app
closes, whatever the answer (*Not now* in the Save dialog that follows takes that back). Runs made by an earlier
version of the app are listed at its first start, each with *Save to the sample* or *Discard* (**Later** asks again at
the next start).

**Closing.** The app's page says goodbye as it goes: closing its last tab or window (✕, Ctrl+W, **Close**, quitting the
browser) stops the engine once no page came back within `[desktop] close_grace_s` (6 seconds; 20,
`close_grace_pending_s`, while a run waits for an answer, so a reload comes back in time). With a run not saved yet,
**Close** and the window's ✕ ask first: *Save and close*, *Discard and close* or *Keep working*. Closing the window
anyway (the browser's own *Leave*) carries out the answer given, or with none discards the run, without a desktop
dialog. **Close** in a browser that does not let a page close its own tab says that the app has closed and that the tab
can be closed (or, with the app still open in another tab, that only this tab is closed). A page that went without a
goodbye (its tab crashed, the browser was killed or put the tab away, an SSH tunnel dropped, a laptop went to sleep) is
away: its run pauses, the engine keeps it and waits, and opening the app again finds it, saying why it is paused. An
engine left away stops by itself only with nothing waiting for an answer, after `[desktop] away_stop_s` (30 minutes; 0:
never). If the engine is stopped any other way (a signal, a logout), it asks with a desktop dialog: *Save to the sample*
saves the run; closing the dialog, or no answer within `[desktop] save_dialog_timeout_s`, discards it, and so does an
engine with no desktop or no `zenity` to ask on. `stop --force` asks nobody: it discards the run nobody answered for,
and finishes a save already under way (or a page's answer from the last minute) first. A `serve` whose terminal hangs up
(an SSH session that dropped) closes as at any other close, so an unanswered run is discarded; one started under `nohup`
keeps running. After a crash, the next start removes what the last session left and says so. A page that cannot reach
its engine at start (no answer within 20 seconds, or a failure) says so on the first screen, with **Reload**.

## From a source folder

The app also runs from a clone, as its developers do. The folder is the `sopa_companion` package itself:

```bash
git clone <this repository> sopa_companion
cd sopa_companion
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install --no-deps -e .      # the app itself; its dependencies are requirements.txt's
cp config.example.toml config.toml        # then name your environments' Pythons in it (below)
native/build.sh                           # builds native/bin/proseg-live: Rust 1.88+, patch, tar, sha256sum, strip, curl or wget
.venv/bin/python -m sopa_companion launch
```

The engine's own environment (`requirements.txt`) holds none of sopa, Cellpose, scvi-tools or RAPIDS, so `"auto"`
finds nothing there and those features stay off. Name the environments that have them in `config.toml`: your sopa
environment's Python as `[zarr] python`, `[stain] python` and `[analysis] python` (and `[qc] python` once it has
scvi-tools and scviva-tools), or a RAPIDS environment's as `[analysis] python` to run Clusters & UMAP on the GPU.
`--no-deps` matters: the package's own dependencies
(sopa, proseg-live) belong to an installed copy and are on no index this folder uses.

A source folder reads its own `config.toml` (else `config.example.toml`), then `config.local.toml`, never your own
settings file, and keeps its runs in `var/` inside it. A notebook's `launch` from a source folder reads the same files
(`config.toml`, else `config.example.toml`, then `config.local.toml`), then your own settings file. Helpers in other
environments import the app through `PYTHONPATH`: the folder holding the app when it is called `sopa_companion`, else
a folder with one link to it under `~/.cache/sopa-companion/pythonpath`.

**Updating from the previous version** (whose package was called `proseg_live`). Install the desktop entry again
(`install-desktop`, above: it replaces the old entry and icon, and the dock's favourite must be pinned again). The first
start moves the previous version's state folder, `~/.local/state/proseg_live`, and its window profile,
`~/.local/share/proseg_live/chrome-profile`, into `~/.local/state/sopa-companion`, whole (never copied), so the page
keeps what it remembered; it refuses, saying why, while anything of the previous version still uses them. Samples keep
everything either version saved into them, and an engine of each can run beside the other: their saves into one sample
never overlap.

## The patched proseg

`native/proseg-live.patch` is the only source of the changes to proseg 3.2.0 (per-iteration snapshots, pause, step,
finish and live settings, and proseg's own count of each cell with the frames the app asks for). `native/build.sh`
downloads the crate, checks crates.io's checksum, applies the patch into `native/proseg/` (made here, never committed),
builds with the patch's `Cargo.lock` (`cargo --locked`) and installs `native/bin/proseg-live`. `native/build.sh --fresh`
makes `native/proseg/` again from the crate and the patch (a tree with edits the patch lacks is kept aside, never
deleted); after editing `native/proseg/`, `native/build.sh --save-patch` writes the changes back into the patch. Its
first run downloads proseg's crate and dependencies.

## Releases

A tag `v<version>` (the app's version, `__init__.py`) runs `.github/workflows/release.yml`: it builds both packages
(proseg-live for Linux x86-64 and ARM64, macOS on Apple silicon and Windows x64), installs them on Linux, macOS and
Windows exactly as the README's Install says for each (on Linux beside a squatter's index that both names must never
come from), runs the notebook route end to end on each (`ci/launch_smoke.py`: a real kernel, proseg, Cellpose, saves,
close, a killed kernel), runs the app's tests (with `tests/test_portable.py` on macOS and Windows too), and publishes
the release. Its text is what the version changed, then the install and the files (`ci/release_text.py` puts the
version's section of `CHANGELOG.md` into `ci/release-notes.md`). Every version adds its section at the top of
`CHANGELOG.md` (`## <version> (<date>)`, then its changes as a list, in a user's words): `ci/check_companion.py` and
the tests stop a version without one. Started by hand (`gh workflow run release.yml --ref <branch>`) it
does the same and publishes nothing: the way to try a change on macOS and Windows. A push's own checks stay on Linux.
Before the release here, the same run publishes both packages on PyPI (`pypi-proseg-live`, then `pypi-companion`) by
trusted publishing: PyPI knows each project's publisher as this repository's `release.yml` in the environment
`pypi-proseg-live` or `pypi-sopa-companion` (pypi.org, the project's Publishing settings), so no PyPI token exists
anywhere. A file PyPI already has is left alone: proseg-live keeps its version until `native/` changes (then
`native/package/pyproject.toml` takes a new fourth number).
Every file of a release made this way also gets a build-provenance attestation (the workflow run and the commit that
built it, signed), which anyone can check: `gh attestation verify FILE --repo jackytamkc/sopa-companion`. GitHub
makes them for public repositories only (a private one needs GitHub Enterprise Cloud), so a private copy skips the
step. A release put together by hand from the same files (0.5.0 to 0.6.2) has none.

**Apple's GPU for the mixing check.** The mixing check learns on an NVIDIA GPU, or on a Mac without one on Apple's GPU
where it passes the probe's test (0.6.0; PyTorch 2.15 needs macOS 15 for it, so `[stain]` and `[qc]` install PyTorch
2.14 on macOS 14 with Apple silicon: pyproject.toml's marker matches the Darwin version as text, `"23." in
platform_release`, because pip with packaging 22 to 25 stops on a version comparison with a Linux kernel's release).
A new macOS or PyTorch needs no change here: the short test decides at run time. `ci/mps_probe.py`, run on a Mac with
Apple silicon in the app's environment (`python ci/mps_probe.py`), says whether the operations ResolVI uses at every
step run on Apple's GPU (MPS) and agree with the CPU's, and how fast the app's own recipe learns there against the CPU
(a tiny fit first, then the full one; a crash prints Metal's reason and the Python stack where it stopped). GitHub's
macOS runners have no usable Apple GPU, so it needs a real Mac.

## Tests

In one environment with sopa and the app (as CI runs them), from the folder that holds `sopa_companion`:

```bash
SOPA_COMPANION_SOPA_PYTHON=$(which python) SOPA_COMPANION_API_PYTHON=$(which python) python -m pytest -q sopa_companion/tests
python sopa_companion/tests/make_mini_harness.py /tmp/mini-harness   # optional: the crop that SOPA_COMPANION_HARNESS_DIR names
```

From a source folder, with its own virtual environment, from the folder that holds `sopa_companion`:

```bash
PYTHONPATH=$PWD sopa_companion/.venv/bin/python -m pytest -q sopa_companion/tests          # long: run files apart if you like
PYTHONPATH=$PWD <sopa python> -m pytest -q sopa_companion/tests/api                       # the notebook's side
cd sopa_companion && node --test frontend/test/*.test.mjs
cd native/proseg && cargo test --release --locked                                    # after native/build.sh
```

Tests that need more than the app's virtual environment skip with a reason when it is missing: those that run the sopa
environment find it in `SOPA_COMPANION_SOPA_PYTHON` or the settings' `[zarr]`, `[stain]` or `[qc]` python, and a few
that read a development harness's fixtures (not part of this repository) need `SOPA_COMPANION_HARNESS_DIR`
(`tests/make_mini_harness.py` makes a stand-in for its crop). Opt-in checks with real models:
`SOPA_COMPANION_STAIN_SMOKE=1` (Cellpose) and `SOPA_COMPANION_TRAINER_SMOKE=1` (the mixing check's trainer on GPU 0).

## Layout

| Path | What |
|---|---|
| `__init__.py` | `import sopa_companion`: the notebook's entry point (`launch`, imported from `api/` on first use) |
| `cli.py`, `__main__.py` | `python -m sopa_companion` (and the `sopa-companion` command): serve, launch, stop, status, install-desktop |
| `names.py` | the names that cross versions: what the previous version (`proseg_live`) wrote, holds or runs |
| `dirs.py`, `locate.py` | where the app keeps its folders; what it runs (each helper's Python, proseg-live) |
| `private.py` | folders and files of this user's alone (0700, 0600, never through a link; on Windows its profile's access list) |
| `portable.py` | what differs between Linux, macOS and Windows for processes, locks, pipes and shared files, in one place |
| `server/` | FastAPI app: the API, the WebSocket, token, page key and origin checks, the pages present, the runtime file; a notebook's session |
| `engine/` | samples, runs and their lifecycle, stain runs, the mixing check, Go deeper, Compare, the Log, crash reports, settings |
| `zarr_bridge/` | helpers run in the sopa environment: reading samples, assigning priors, Cellpose, saving with sopa |
| `analysis/` | the analysis worker (GPU with RAPIDS, else CPU) and the mixing check's trainer and probe |
| `desktop/` | launcher, the browser (the default one, or the Chrome app window), desktop entry and icon, engine logs, the previous version's state folder moved |
| `api/` | the notebook's side: `sopa_companion.launch(sdata)`, its session, adding results to the object |
| `frontend/` | the page: plain ES modules and WebGL2, no build step |
| `native/` | `proseg-live.patch` and `build.sh` (the patched proseg 3.2.0), proseg's `LICENSE.md`, a `README.md` about them; `package/`: the proseg-live package |
| `pyproject.toml`, `packaging/` | the `sopa-companion` wheel, and sopa's optional `companion` extra |
| `.github/workflows/`, `ci/` | tests, proseg-live's wheels, the release (on Linux, macOS and Windows), a weekly check that sopa's set still resolves |
| `tests/`, `frontend/test/` | pytest and node tests (`tests/api/`: the notebook's side, run in a sopa environment) |
| `docs/ARCHITECTURE.md` | components, data flow, API routes, file formats |
| `docs/DEVELOPING.md` | this page |
