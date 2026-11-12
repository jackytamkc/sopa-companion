SoPA companion {version} (beta): Linux, macOS on Apple silicon and Windows x64.

Install it in the environment your notebooks use (with sopa as its guide installs it,
https://prism-oncology.github.io/sopa/getting_started/, or a new one: the install brings sopa), on Linux, macOS or
Windows:

    pip install "sopa-companion[all]"

On Windows with an NVIDIA GPU, the same line with PyTorch's own index beside PyPI, which holds its build for that GPU
(PyPI has only the CPU build for Windows):

    pip install "sopa-companion[all]" --extra-index-url https://download.pytorch.org/whl/cu132

It installs the app and proseg-live from PyPI, and what they need: sopa, Cellpose, PyTorch, and scvi-tools with
scviva-tools for the ResolVI mixing check. On macOS 14 with Apple silicon that is PyTorch 2.14, in place of a newer one already
in the environment (2.15 needs macOS 15 for Apple's GPU); on macOS 15 or later, the newest. On a computer no
proseg-live wheel fits (an Intel Mac, a Linux older than glibc 2.28), pip builds it from its source: Rust 1.88 or newer
and a C compiler first. To update: `pip install --upgrade "sopa-companion[all]"`.

Then, in a notebook: `app = sopa_companion.launch(sdata)` (the README's Quick start).

Use Linux, or WSL2 on a Windows computer: everything in the app runs there. macOS and native Windows have limits (no
RAPIDS; on native Windows no export of training patches and a link to click in place of a browser window): the README's
"Limits on macOS and Windows" lists them.

Files (the same as on PyPI, with their checksums):
- sopa_companion-{version}-py3-none-any.whl: the app.
- dependencies.txt: the packages the app needs from PyPI.
- proseg_live-3.2.0.3-py3-none-manylinux_2_28_x86_64.whl and proseg_live-3.2.0.3-py3-none-manylinux_2_28_aarch64.whl:
  proseg 3.2.0 with live mode, built for Linux; proseg_live-3.2.0.3-py3-none-macosx_11_0_arm64.whl for macOS on Apple
  silicon; proseg_live-3.2.0.3-py3-none-win_amd64.whl for Windows x64.
- proseg_live-3.2.0.3.tar.gz: proseg-live's complete source (proseg 3.2.0 by Daniel C. Jones, with the app's patch),
  under the GNU General Public License v3. pip builds from it where no wheel fits (Rust 1.88 or newer and a C compiler).
- sopa_companion-{version}.tar.gz: the app's source.
- SHA256SUMS: the files' checksums.
