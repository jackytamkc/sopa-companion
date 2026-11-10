"""native/build.sh, in a scratch copy with a fake cargo and a fake maturin (no compiling, no network): proseg-live.patch
on the pinned proseg 3.2.0 crate is the one source of native/proseg and of the proseg-live package's source; cargo only
ever runs --locked; --fresh and --save-patch never lose a tree or a patch, and a failure leaves both as they were; the
program names no folder of the builder's computer; --tree makes the tree alone, --package the package's source beside
it, and --wheel a wheel that native/package/check.py has passed (its own cases: test_proseg_live_package.py)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tomllib

import pytest

from sopa_companion.engine.config import ROOT

SCRIPT = (ROOT / "native/build.sh").read_text()
VERSION = re.search(r"^VERSION=(\S+)$", SCRIPT, re.M).group(1)
PINNED = re.search(r"^SHA256=([0-9a-f]{64})$", SCRIPT, re.M).group(1)

# The fake cargo writes the rustflags its build was given to $FAKE_FLAGS when a test names that file.
FAKE_CARGO = """#!/usr/bin/env bash
echo "cargo $*" >> "$FAKE_LOG"
if [[ "$1" == fetch && " $* " == *" --offline "* && -n "${FAKE_OFFLINE_FAIL:-}" ]]; then
  echo "error: attempting to make an HTTP request, but --offline was specified" >&2
  exit 101
fi
if [[ "$1" == build && -n "${FAKE_FLAGS:-}" ]]; then
  printf '%s' "${CARGO_ENCODED_RUSTFLAGS-not set}" > "$FAKE_FLAGS"
fi
if [[ "$1" == build ]]; then
  mkdir -p "$CARGO_TARGET_DIR/release"
  printf '#!/bin/sh\\necho "proseg %s"\\n' "$(sed -n 's/^version = "\\(.*\\)"$/\\1/p' proseg/Cargo.toml | head -1)" \\
    > "$CARGO_TARGET_DIR/release/proseg"
  chmod +x "$CARGO_TARGET_DIR/release/proseg"
fi
"""
FAKE_DOWNLOAD = """#!/usr/bin/env bash
echo "$(basename "$0") $*" >> "$FAKE_LOG"
while [[ $# -gt 0 ]]; do
  if [[ "$1" == -o || "$1" == -O ]]; then echo "not the crate" > "$2"; shift; fi
  shift
done
"""
FAKES = {"cargo": FAKE_CARGO, "rustc": 'echo "rustc 1.0.0"\necho "host: x86_64-unknown-linux-gnu"\n',
         "strip": "exit 0\n", "curl": FAKE_DOWNLOAD, "wget": FAKE_DOWNLOAD,
         "git": 'echo "git $*" >> "$FAKE_LOG"\necho "no network in tests" >&2\nexit 128\n'}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pinned_crate() -> Path | None:
    """This computer's copy of the pinned crate: cargo's, or the one native/build.sh downloaded."""
    places = [Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo") / "registry/cache"]
    hits = [hit for place in places for hit in sorted(place.glob(f"*/proseg-{VERSION}.crate"))]
    hits.append(ROOT / f"native/.cache/proseg-{VERSION}.crate")
    return next((hit for hit in hits if hit.is_file() and sha256(hit) == PINNED), None)


def files(tree: Path) -> dict[str, bytes]:
    return {p.relative_to(tree).as_posix(): p.read_bytes() for p in tree.rglob("*")
            if p.is_file() and "target" not in p.relative_to(tree).parts}


def made(crate: Path, patch: Path, out: Path) -> dict[str, bytes]:
    """The crate plus a patch, made here independently of build.sh."""
    with tarfile.open(crate) as archive:
        archive.extractall(out, filter="data")
    tree = out / f"proseg-{VERSION}"
    for name in (".cargo-ok", ".cargo_vcs_info.json"):
        (tree / name).unlink(missing_ok=True)
    subprocess.run(["patch", "--batch", "--forward", "--no-backup-if-mismatch", "--quiet", "-p1", "-d", str(tree),
                    "-i", str(patch)], check=True, timeout=120)
    return files(tree)


class Native:
    """A scratch app's native/ folder: build.sh, the patch and the proseg-live package's files, a fake toolchain first
    on PATH, its own CARGO_HOME holding the pinned crate, and its own CARGO_TARGET_DIR."""

    def __init__(self, tmp: Path, crate: Path):
        self.tmp, self.crate = tmp, crate
        self.dir = tmp / "app/native"
        self.dir.mkdir(parents=True)
        for name in ("build.sh", "proseg-live.patch"):
            shutil.copy(ROOT / "native" / name, self.dir / name)
        shutil.copytree(ROOT / "native/package", self.dir / "package", ignore=shutil.ignore_patterns("__pycache__"))
        fakes = tmp / "fake-bin"
        fakes.mkdir()
        for name, body in FAKES.items():
            (fakes / name).write_text(body if body.startswith("#!") else "#!/usr/bin/env bash\n" + body)
            (fakes / name).chmod(0o755)
        self.cache = tmp / "cargo-home/registry/cache/index.example"
        self.cache.mkdir(parents=True)
        shutil.copy(crate, self.cache / crate.name)
        self.log = tmp / "calls.log"
        self.env = {"PATH": f"{fakes}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp / "home"),
                    "CARGO_HOME": str(tmp / "cargo-home"), "CARGO_TARGET_DIR": str(tmp / "target"),
                    "FAKE_LOG": str(self.log), "LANG": "C.UTF-8"}

    def __call__(self, *args: str, cwd: Path | None = None, **env: str) -> subprocess.CompletedProcess:
        self.log.write_text("")
        return subprocess.run(["bash", str(self.dir / "build.sh"), *args], cwd=cwd or self.dir, env={**self.env, **env},
                              capture_output=True, text=True, timeout=300)

    @property
    def calls(self) -> list[str]:
        return self.log.read_text().splitlines()

    @property
    def tree(self) -> Path:
        return self.dir / "proseg"

    def expected(self, patch: Path | None = None) -> dict[str, bytes]:
        out = self.tmp / f"expected-{len(list(self.tmp.glob('expected-*')))}"
        return made(self.crate, patch or self.dir / "proseg-live.patch", out)

    def leftovers(self) -> list[str]:
        return sorted(p.name for p in (self.dir / ".cache").glob("work-*"))


@pytest.fixture
def native(tmp_path):
    crate = pinned_crate()
    if crate is None:
        pytest.skip(f"proseg-{VERSION}.crate is not on this computer yet (native/build.sh downloads it once)")
    return Native(tmp_path, crate)


def test_a_missing_tree_is_made_from_the_pinned_crate_and_the_patch_and_built_locked(native):
    done = native()
    assert done.returncode == 0, done.stderr
    assert files(native.tree) == native.expected()                  # Cargo.lock included: the crate's, as patched
    assert (native.tree / "Cargo.lock").is_file() and not (native.tree / ".cargo_vcs_info.json").exists()
    host = "--target x86_64-unknown-linux-gnu --manifest-path proseg/Cargo.toml"
    assert native.calls == [f"cargo fetch --locked --offline {host}",
                            "cargo build --release --locked --offline --manifest-path proseg/Cargo.toml --bin proseg"]
    assert (native.dir / "bin/proseg-live").stat().st_mode & 0o777 == 0o755
    assert "Installed proseg" in done.stdout and native.leftovers() == []
    # Built again: the tree is used as it is.
    before = files(native.tree)
    assert native().returncode == 0 and files(native.tree) == before


def test_only_a_locked_fetch_ever_goes_online(native):
    """With crates missing from ~/.cargo the locked fetch downloads exactly the locked versions; nothing is ever built
    or resolved without --locked (the old fallback re-resolved every dependency to its newest version)."""
    done = native(FAKE_OFFLINE_FAIL="1")
    assert done.returncode == 0, done.stderr
    host = "--target x86_64-unknown-linux-gnu --manifest-path proseg/Cargo.toml"
    assert native.calls == [f"cargo fetch --locked --offline {host}", f"cargo fetch --locked {host}",
                            "cargo build --release --locked --offline --manifest-path proseg/Cargo.toml --bin proseg"]
    assert all("--locked" in call for call in native.calls if call.startswith("cargo "))


def test_fresh_never_loses_the_tree_when_something_fails(native):
    assert native().returncode == 0
    before = files(native.tree)
    # A patch that does not apply (as the old --fresh met on a git checkout): the tree stays, no prompt, no leftovers.
    good = (native.dir / "proseg-live.patch").read_text()
    (native.dir / "proseg-live.patch").write_text(good + "--- a/src/nothing.rs\n+++ b/src/nothing.rs\n@@ -1 +1 @@\n-x\n+y\n")
    failed = native("--fresh")
    assert failed.returncode != 0 and "does not apply" in failed.stderr
    assert files(native.tree) == before and native.leftovers() == [] and not native.calls
    # No crate with crates.io's checksum (a damaged copy, and a download that is not the crate): refused.
    (native.dir / "proseg-live.patch").write_text(good)
    (native.cache / native.crate.name).write_bytes(b"damaged")
    failed = native("--fresh")
    assert failed.returncode != 0 and "checksum" in failed.stderr
    assert files(native.tree) == before and native.leftovers() == [] and not native.calls[1:]
    assert native.calls[0].startswith("curl ") and native.calls[0].endswith(f"proseg-{VERSION}.crate")


def test_fresh_remakes_the_tree_and_keeps_changes_the_patch_does_not_have(native):
    assert native().returncode == 0
    expected = files(native.tree)
    # Made by build.sh and not edited since: replaced, nothing kept.
    assert native("--fresh").returncode == 0 and files(native.tree) == expected
    assert not list((native.dir / ".cache").glob("proseg-kept-*"))
    # Edited: remade, and the edited tree kept whole (never deleted).
    with open(native.tree / "src/live.rs", "a") as stream:
        stream.write("// an edit not saved in the patch\n")
    done = native("--fresh")
    assert done.returncode == 0 and "kept in" in done.stderr
    assert files(native.tree) == expected
    [kept] = (native.dir / ".cache").glob("proseg-kept-*")
    assert (kept / "src/live.rs").read_text().endswith("// an edit not saved in the patch\n")


def test_save_patch_writes_the_trees_changes_and_gives_the_tree_back_exactly(native):
    assert native().returncode == 0
    with open(native.tree / "src/live.rs", "a") as stream:
        stream.write("// saved in the patch\n")
    (native.tree / "src/extra.rs").write_text("// a new file\n")
    (native.tree / "target/release").mkdir(parents=True)            # a cargo run inside the tree: never in the patch
    (native.tree / "target/release/proseg").write_text("built\n")
    edited = files(native.tree)
    done = native("--save-patch")
    assert done.returncode == 0, done.stderr
    assert not native.calls                                         # saving builds nothing
    patch = native.dir / "proseg-live.patch"
    assert native.expected(patch) == edited                         # the crate plus the new patch is the edited tree
    assert not re.search(r"^diff .* b/target/", patch.read_text(), re.M)
    # Fixed times: a changed file's, and the epoch on the missing side of a file the patch creates.
    assert re.search(r"^--- a/src/main\.rs\t2000-01-01 00:00:00\.000000000 \+0000$", patch.read_text(), re.M)
    assert re.search(r"^--- a/src/extra\.rs\t1970-01-01 00:00:00\.000000000 \+0000$", patch.read_text(), re.M)
    first = patch.read_bytes()
    assert native("--save-patch").returncode == 0 and patch.read_bytes() == first      # the same tree, the same patch
    # Now the tree is what the patch makes: --fresh keeps nothing and loses nothing.
    assert native("--fresh").returncode == 0 and files(native.tree) == edited
    assert not list((native.dir / ".cache").glob("proseg-kept-*"))


def test_save_patch_refuses_what_a_patch_cannot_carry(native):
    assert native().returncode == 0
    patch = (native.dir / "proseg-live.patch").read_bytes()
    (native.tree / "src/blob.bin").write_bytes(bytes(range(256)) * 4)
    done = native("--save-patch")
    assert done.returncode != 0 and "proseg-live.patch is unchanged" in done.stderr
    assert (native.dir / "proseg-live.patch").read_bytes() == patch and native.leftovers() == []


def test_a_changed_patch_remakes_an_unedited_tree_but_never_an_edited_one(native):
    assert native().returncode == 0
    with open(native.tree / "src/live.rs", "a") as stream:
        stream.write("// a newer patch\n")
    assert native("--save-patch").returncode == 0
    newer = (native.dir / "proseg-live.patch").read_bytes()
    # The tree as the older patch made it (a clone before a pull), then the newer patch arrives.
    (native.dir / "proseg-live.patch").write_bytes((ROOT / "native/proseg-live.patch").read_bytes())
    assert native("--fresh").returncode == 0
    (native.dir / "proseg-live.patch").write_bytes(newer)
    done = native()
    assert done.returncode == 0 and "making it again" in done.stderr
    assert (native.tree / "src/live.rs").read_text().endswith("// a newer patch\n")
    # An edited tree is built as it is, whatever the patch says.
    with open(native.tree / "src/live.rs", "a") as stream:
        stream.write("// mine\n")
    (native.dir / "proseg-live.patch").write_bytes((ROOT / "native/proseg-live.patch").read_bytes())
    done = native()
    assert done.returncode == 0 and "--save-patch saves them" in done.stderr
    assert (native.tree / "src/live.rs").read_text().endswith("// mine\n")


def test_a_tree_without_its_lock_file_gets_the_locked_one(native):
    assert native().returncode == 0
    lock = (native.tree / "Cargo.lock").read_bytes()
    (native.dir / ".cache/proseg.made").unlink()                     # a tree from git, not made by build.sh
    (native.tree / "Cargo.lock").unlink()
    done = native()
    assert done.returncode == 0 and "had no Cargo.lock" in done.stderr
    assert (native.tree / "Cargo.lock").read_bytes() == lock
    assert all("--locked" in call for call in native.calls)


def test_build_sh_takes_no_unknown_options(native):
    done = native("--fast")
    assert done.returncode != 0 and "unknown option" in done.stderr and not native.calls
    assert not native.tree.exists()


def test_the_binary_names_no_folder_of_the_builders_computer(native, tmp_path):
    """Panic messages name the source file they come from: built plainly, the binary that the wheel ships carried every
    dependency's path under cargo's registry, and with it the builder's home folder and user name. The build remaps the
    home folder, this folder, the target and cargo's home to neutral names (rustc takes the last that matches: the
    deeper ones come later), after the flags the caller gave, which stay; one flag per field, so that a folder with
    spaces stays one flag."""
    def flags(native, **env) -> list[str]:
        written = native.tmp / "flags"
        done = native(FAKE_FLAGS=str(written), **env)
        assert done.returncode == 0, done.stderr
        return written.read_text().split("\x1f")

    assert flags(native) == remaps(native)
    assert flags(native, RUSTFLAGS=" -C  target-cpu=x86-64-v2") == ["-C", "target-cpu=x86-64-v2", *remaps(native)]
    # CARGO_ENCODED_RUSTFLAGS is what cargo reads when both are set: its flags stay, RUSTFLAGS' are not added.
    assert flags(native, CARGO_ENCODED_RUSTFLAGS="-Cdebuginfo=0\x1f-Copt-level=3", RUSTFLAGS="-Cbad") == \
        ["-Cdebuginfo=0", "-Copt-level=3", *remaps(native)]
    # A relative target is this folder's; a home folder of "/" (some containers) is left out, else every path would go.
    assert flags(native, CARGO_TARGET_DIR="target", HOME="/")[:2] == [f"--remap-path-prefix={native.dir}=/proseg-live",
                                                                         f"--remap-path-prefix={native.dir}/target=/target"]
    spaced = Native(tmp_path / "my apps", native.crate)
    assert flags(spaced) == remaps(spaced) and " " in str(spaced.dir)


def remaps(native: Native) -> list[str]:
    """The remaps build.sh appends for this scratch copy: its home folder, its native/ folder, its target, cargo's home."""
    env = native.env
    return [f"--remap-path-prefix={env['HOME']}=~", f"--remap-path-prefix={native.dir}=/proseg-live",
            f"--remap-path-prefix={env['CARGO_TARGET_DIR']}=/target", f"--remap-path-prefix={env['CARGO_HOME']}=/cargo"]


def package_files(native: Native) -> dict[str, bytes]:
    """What --package must give: the crate and the patch (without cargo's own Cargo.toml.orig, which cargo refuses in a
    package's source), with the package's pyproject.toml and page and the patch itself beside them."""
    expected = native.expected()
    expected.pop("Cargo.toml.orig")
    for name in ("pyproject.toml", "PROSEG-LIVE.md"):
        expected[name] = (ROOT / "native/package" / name).read_bytes()
    expected["proseg-live.patch"] = (native.dir / "proseg-live.patch").read_bytes()
    return expected


def test_tree_makes_native_proseg_and_builds_nothing(native):
    """--tree (CI's cargo tests): native/proseg as the build makes it, and nothing built or fetched."""
    done = native("--tree")
    assert done.returncode == 0, done.stderr
    assert "native/proseg is ready" in done.stdout
    assert files(native.tree) == native.expected() and native.calls == []
    assert not (native.dir / "bin").exists() and native.leftovers() == []
    # Made already: kept as it is; an edit the patch lacks too (as the build keeps it).
    with open(native.tree / "src/live.rs", "a") as stream:
        stream.write("// an edit\n")
    edited = files(native.tree)
    done = native("--tree")
    assert done.returncode == 0 and files(native.tree) == edited and native.calls == []


def test_package_makes_the_proseg_live_source_tree(native, tmp_path):
    """--package DIR: DIR/proseg-live-<VERSION>, a fresh tree from the crate and the patch with the package's
    pyproject.toml, its page and the patch itself (maturin sdist and maturin build take it), whose Cargo.toml has the
    proseg-live program; native/proseg is neither made nor touched, nothing is built, and a folder already there is
    refused. A relative DIR is the caller's, wherever build.sh is. The tree, as maturin's sdist lays it out, passes
    native/package/check.py."""
    caller = tmp_path / "elsewhere"
    caller.mkdir()
    done = native("--package", "made", cwd=caller)
    assert done.returncode == 0, done.stderr
    made = caller / "made" / f"proseg-live-{VERSION}"
    assert files(made) == package_files(native)
    bins = {b["name"]: b["path"] for b in tomllib.loads((made / "Cargo.toml").read_text())["bin"]}
    assert bins == {"proseg": "src/main.rs", "proseg-live": "src/main.rs", "proseg-to-baysor": "src/to_baysor.rs"}
    assert not native.tree.exists() and native.calls == [] and native.leftovers() == []
    # Once more into the same folder: refused, the tree as it was.
    before = files(made)
    again = native("--package", str(caller / "made"))
    assert again.returncode != 0 and "is there already" in again.stderr
    assert files(made) == before and native.leftovers() == []
    # Next to a tree being edited: that tree is untouched.
    assert native("--tree").returncode == 0
    with open(native.tree / "src/live.rs", "a") as stream:
        stream.write("// an edit not saved in the patch\n")
    edited = files(native.tree)
    assert native("--package", str(tmp_path / "second")).returncode == 0
    assert files(native.tree) == edited and files(tmp_path / "second" / f"proseg-live-{VERSION}") == package_files(native)
    sdist = tmp_path / f"proseg_live-{VERSION}.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(made, arcname=f"proseg_live-{VERSION}")
    checked = subprocess.run([sys.executable, str(ROOT / "native/package/check.py"), "sdist", str(sdist)],
                             capture_output=True, text=True, timeout=120)
    assert checked.returncode == 0, checked.stderr


# A stand-in for maturin: it records how it was called, then writes the wheel maturin would (the program an ELF header
# for x86-64 followed by $FAKE_PROGRAM_TEXT, proseg's licence from the tree), or nothing with FAKE_MATURIN_FAIL.
FAKE_MATURIN = r"""#!PYTHON
import json, os, sys, zipfile
from pathlib import Path
args = sys.argv[1:]
tree = Path(args[args.index("--manifest-path") + 1]).parent
Path(os.environ["FAKE_MATURIN"]).write_text(json.dumps({
    "args": args, "flags": os.environ.get("CARGO_ENCODED_RUSTFLAGS"), "target": os.environ.get("CARGO_TARGET_DIR"),
    "tree": sorted(path.name for path in tree.iterdir())}))
if os.environ.get("FAKE_MATURIN_FAIL"):
    sys.exit("maturin: the build failed")
out = Path(args[args.index("--out") + 1])
out.mkdir(parents=True, exist_ok=True)
header = bytearray(64)
header[:7] = b"\x7fELF\x02\x01\x01"
header[18:20] = (62).to_bytes(2, "little")
with zipfile.ZipFile(out / "proseg_live-3.2.0-py3-none-manylinux_2_28_x86_64.whl", "w") as wheel:
    program = zipfile.ZipInfo("proseg_live-3.2.0.data/scripts/proseg-live")
    program.external_attr = 0o100755 << 16
    wheel.writestr(program, bytes(header) + os.environ.get("FAKE_PROGRAM_TEXT", "").encode())
    wheel.writestr("proseg_live-3.2.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: proseg-live\nVersion: 3.2.0\n")
    wheel.writestr("proseg_live-3.2.0.dist-info/licenses/LICENSE.md", (tree / "LICENSE.md").read_text())
"""


def test_wheel_needs_maturin_and_checks_the_result(native, tmp_path):
    """--wheel DIR [-- MATURIN ARGS]: without maturin on PATH it says how to get it and does nothing. With it: maturin
    builds the package's source (made in native/.cache, inside native/, so this folder's remap covers it) --release
    --locked, with build.sh's remaps after the caller's flags and the caller's target, and by default --compatibility
    linux (the arguments after -- replace that); native/package/check.py then checks the wheel before it reaches DIR
    (one that names the builder's computer never does), and nothing is left in native/.cache."""
    fakes = tmp_path / "fake-bin"
    no_maturin = f"{fakes}:/usr/bin:/bin"
    done = native("--wheel", str(tmp_path / "wheels"), PATH=no_maturin)
    assert done.returncode != 0 and "--wheel needs maturin on PATH" in done.stderr
    assert not (tmp_path / "wheels").exists() and native.calls == [] and native.leftovers() == []

    (fakes / "maturin").write_text(FAKE_MATURIN.replace("PYTHON", sys.executable, 1))
    (fakes / "maturin").chmod(0o755)
    record = tmp_path / "maturin.json"
    caller = tmp_path / "elsewhere"
    caller.mkdir()
    done = native("--wheel", "wheels", cwd=caller, FAKE_MATURIN=str(record),
                  CARGO_ENCODED_RUSTFLAGS="-Cdebuginfo=0")
    assert done.returncode == 0, done.stderr
    called = json.loads(record.read_text())
    work = Path(called["args"][called["args"].index("--manifest-path") + 1]).parents[1]
    assert work.parent == native.dir / ".cache" and work.name.startswith("work-")
    assert called["args"] == ["build", "--release", "--locked", "--manifest-path",
                              str(work / f"proseg-live-{VERSION}/Cargo.toml"), "--out", str(work / "wheels"),
                              "--compatibility", "linux"]
    assert called["flags"].split("\x1f") == ["-Cdebuginfo=0", *remaps(native)]
    assert called["target"] == native.env["CARGO_TARGET_DIR"]
    assert {"pyproject.toml", "PROSEG-LIVE.md", "proseg-live.patch", "Cargo.lock", "LICENSE.md"} <= set(called["tree"])
    assert "Cargo.toml.orig" not in called["tree"]
    built = caller / "wheels" / "proseg_live-3.2.0-py3-none-manylinux_2_28_x86_64.whl"
    assert built.is_file() and f"Built {built}" in done.stdout and ": ok" in done.stdout
    assert native.calls == [] and native.leftovers() == [] and not native.tree.exists()

    # maturin's own arguments after --: a release-grade wheel's.
    done = native("--wheel", str(tmp_path / "release"), "--", "--zig", "--compatibility", "manylinux_2_28",
                  FAKE_MATURIN=str(record))
    assert done.returncode == 0, done.stderr
    assert json.loads(record.read_text())["args"][-3:] == ["--zig", "--compatibility", "manylinux_2_28"]
    # A program that names the builder's home folder (a build without the remaps): refused, nothing reaches DIR.
    registry = "registry/src/index.crates.io-1949cf8c6b5b557f/num-integer-0.1.46/src/roots.rs"
    done = native("--wheel", str(tmp_path / "named"), FAKE_MATURIN=str(record),
                  FAKE_PROGRAM_TEXT=f"{native.env['HOME']}/.cargo/{registry}")
    assert done.returncode != 0 and f"its home folder ({native.env['HOME']}/)" in done.stderr
    assert not list((tmp_path / "named").glob("*.whl")) and native.leftovers() == []
    # maturin failing: build.sh fails with it, and leaves nothing behind.
    done = native("--wheel", str(tmp_path / "failed"), FAKE_MATURIN=str(record), FAKE_MATURIN_FAIL="1")
    assert done.returncode != 0 and "the build failed" in done.stderr
    assert not (tmp_path / "failed").exists() and native.leftovers() == []


def test_the_package_and_wheel_modes_take_one_folder(native):
    for args in (["--package"], ["--package", ""], ["--package", "a", "b"], ["--wheel"], ["--wheel", "a", "b"],
                 ["--tree", "a"]):
        done = native(*args)
        assert done.returncode != 0 and "native/build.sh:" in done.stderr, args
    assert native.calls == [] and not native.tree.exists()


def test_the_patch_takes_out_prosegs_debug_dumps():
    """proseg 3.2.0 wrote membership_debug.txt and rebalanced_membership_debug.txt into the folder it ran in (the app
    runs proseg in each run's folder): the patch takes both writes out, and adds none back."""
    lines = (ROOT / "native/proseg-live.patch").read_text().splitlines()
    removed = [line for line in lines if line.startswith("-") and "File::create(" in line]
    assert any('"membership_debug.txt"' in line for line in removed)
    assert any('"rebalanced_membership_debug.txt"' in line for line in removed)
    assert not [line for line in lines if line.startswith("+") and "membership_debug" in line]
