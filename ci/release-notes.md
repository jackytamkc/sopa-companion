SoPA companion {version}, for alpha testers: Linux, macOS on Apple silicon and Windows x64.

Install it in the environment your notebooks use (with sopa as its guide installs it,
https://prism-oncology.github.io/sopa/getting_started/, or a new one: the install brings sopa). On Linux and macOS:

    d=$(mktemp -d)
    gh release download {tag} --repo jackytamkc/sopa-companion --dir "$d" --pattern 'sopa_companion-*.whl' --pattern "proseg_live-*_$(uname -m).whl"
    pip install "sopa-companion[all]" "$d"/sopa_companion-*.whl "$d"/proseg_live-*.whl

On Windows, in PowerShell (an Anaconda PowerShell Prompt, or PowerShell with the environment activated):

    $d = New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid()))
    gh release download {tag} --repo jackytamkc/sopa-companion --dir $d --pattern 'sopa_companion-*.whl' --pattern 'proseg_live-*_amd64.whl'
    pip install "sopa-companion[all]" (Resolve-Path "$d\sopa_companion-*.whl") (Resolve-Path "$d\proseg_live-*.whl")

The last line installs the app and proseg-live from the two files just downloaded (pip never asks an index for
either), and from PyPI what they need: sopa, Cellpose, PyTorch, and scvi-tools with scviva-tools for the mixing check.
On macOS 14 with Apple silicon that is PyTorch 2.14, in place of a newer one already in the environment (2.15 needs
macOS 15 for Apple's GPU); on macOS 15 or later, the newest.
On a computer none of the wheels below fits (an Intel Mac, a Linux older than glibc 2.28), build proseg-live from its
source here instead (Rust 1.88 or newer and a C compiler):

    gh release download {tag} --repo jackytamkc/sopa-companion --dir "$d" --pattern 'proseg_live-*.tar.gz'
    pip install "sopa-companion[all]" "$d"/sopa_companion-*.whl "$d"/proseg_live-*.tar.gz

Then, in a notebook: `app = sopa_companion.launch(sdata)` (the README's Quick start).

Files:
- sopa_companion-{version}-py3-none-any.whl: the app.
- dependencies.txt: the packages the app needs from PyPI.
- proseg_live-3.2.0.2-py3-none-manylinux_2_28_x86_64.whl and proseg_live-3.2.0.2-py3-none-manylinux_2_28_aarch64.whl:
  proseg 3.2.0 with live mode, built for Linux; proseg_live-3.2.0.2-py3-none-macosx_11_0_arm64.whl for macOS on Apple
  silicon; proseg_live-3.2.0.2-py3-none-win_amd64.whl for Windows x64.
- proseg_live-3.2.0.2.tar.gz: proseg-live's complete source (proseg 3.2.0 by Daniel C. Jones, with the app's patch),
  under the GNU General Public License v3. pip builds from it where no wheel fits (Rust 1.88 or newer and a C compiler).
- sopa_companion-{version}.tar.gz: the app's source.
- SHA256SUMS: the files' checksums.
