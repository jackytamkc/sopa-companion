# What changed in each version

The changes in each version of SoPA companion, newest first. To update, in the environment the app is installed in:
`pip install --upgrade "sopa-companion[all]"` (on Windows with an NVIDIA GPU, with the second half of the README's
install line). `sopa-companion --version` says which version you have.

## 0.8.1 (2026-11-12)

- **Fixed:** exporting training patches with the model's outlines failed on a computer that had not downloaded the
  Cellpose model yet (0.8.0 showed an error and stopped the download). The export now waits for the download, and its
  dialog says that the model is being downloaded.
- **Cellpose's own window on Linux:** when the window cannot open because the system lacks the libraries a window
  needs (common on WSL2 and on servers), the app says so and gives the `apt` line that installs them. The README has
  the same line under Optional extras.
- **The page of a link that no longer works** says why, and what to run for a new link: `app.open()` in the notebook.
- **README:** Linux, or WSL2 on a Windows computer, is recommended. A new section, "Limits on macOS and Windows",
  lists what is missing on macOS and on native Windows, with notes for WSL2 (no browser window opens there: click the
  link the notebook shows). WSL2 was tested on a Windows computer with an NVIDIA GPU and works as Linux does.
- **README:** a table and a chart of the free GPU memory the ResolVI mixing check needs, for gene panels from 300 to
  20,000 genes.

## 0.8.0 (2026-11-11)

The first beta.

- **Fixed (Windows):** on many Windows computers the page opened and stayed empty (only its frame showed). Windows told
  the app that the page's `.js` files were plain text, and browsers refuse to run those. The app now names each file's
  type itself.
- **Windows with an NVIDIA GPU:** one install line brings PyTorch's build for that GPU:
  `pip install "sopa-companion[all]" --extra-index-url https://download.pytorch.org/whl/cu132`. On Windows the
  install keeps PyTorch to 2.14 or 2.15, the versions that index holds.
- **The mixing check is named "ResolVI mixing check"**, and its "How is this estimated?" texts credit ResolVI's authors
  (Can Ergen and Nir Yosef).
- **README:** clips of the app on a public sample, what runs under each part of the app with the methods' authors and
  the papers to cite, how to keep your own PyTorch, and what each extra installs.
- The status moves from alpha to beta.

## 0.7.0 (2026-11-10)

- **On PyPI:** `pip install "sopa-companion[all]"` is the whole install on Linux, macOS and Windows. It no longer
  needs the GitHub CLI or a GitHub account. Updating is the same line with `--upgrade`.
- **proseg-live 3.2.0.3:** the same program as 3.2.0.2, in a package that PyPI accepts. Where no ready-built
  proseg-live fits the computer, pip builds it from its source.
- The README and `CITATION.cff` give the Zenodo DOI. Every release, from 0.6.2 on, is archived there with a DOI of
  its own (all versions: 10.5281/zenodo.24116670).

## 0.6.2 (2026-11-10)

The first public release.

- The repository is public. The app's own code is under the BSD 3-Clause licence; proseg-live, the patched proseg it
  runs, stays under the GNU General Public License v3.
- `CITATION.cff` says how to cite the app (GitHub's "Cite this repository").

## Before the public release

Versions 0.3.0 to 0.6.1 went to alpha testers privately (October and November 2026).

- **0.6.1:** on macOS 14 with Apple silicon the install brings PyTorch 2.14 (2.15 needs macOS 15 for Apple's GPU). An
  export of training patches with the model's outlines says each patch as it is written. The model list shows a
  spinner while it loads.
- **0.6.0:** the mixing check learns on a Mac's own GPU.
- **0.5.2:** Apple's GPU is used only where a short test of it passes; elsewhere Cellpose runs on the CPU. A built-in
  Cellpose model is downloaded once.
- **0.5.1:** an export of training patches waits long enough for its patches. Cellpose's own window opens on macOS and
  Windows.
- **0.5.0:** macOS on Apple silicon and Windows x64, besides Linux.
- **0.4.2:** the mixing check waits for the GPU memory its gene panel needs (before, a fixed 6 GB), and says how much
  it needs and how much is free.
- **0.4.1:** the first release to alpha testers.
- **0.4.0:** the test of a setting covers nine of proseg's own settings, each with a short demo. Compare shows each
  segmentation on its own map.
- **0.3.0:** proseg-live became a package of its own, and the `sopa-companion` command was added.
