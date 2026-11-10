"""native/package/check.py, which proseg-live's wheels and source distribution pass before anyone gets them (imported by
its path: native/ is no Python package). The wheels and sdists are made here, small: a program that is an ELF header
followed by the strings a build may leave in it."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import tarfile
import zipfile

import pytest

from sopa_companion.engine.config import ROOT

SPEC = importlib.util.spec_from_file_location("proseg_live_check", ROOT / "native/package/check.py")
check = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check)

LICENCE = (ROOT / "native/LICENSE.md").read_text()
X86_64 = "proseg_live-3.2.0-py3-none-manylinux_2_28_x86_64.whl"
# A panic message's source file in cargo's registry, as a plain build writes it.
REGISTRY = "registry/src/index.crates.io-1949cf8c6b5b557f/num-integer-0.1.46/src/roots.rs"


def program(*texts: str, machine: int = 62) -> bytes:
    """An x86-64 ELF program (or `machine`'s), 64-bit and little-endian, holding these strings."""
    header = bytearray(64)
    header[:7] = b"\x7fELF\x02\x01\x01"
    header[18:20] = machine.to_bytes(2, "little")
    return bytes(header) + b"\0".join(text.encode() for text in texts)


def wheel(folder: Path, data: bytes = b"", *, name: str = X86_64, programs=("proseg-live",), licence: str = LICENCE,
          mode: int = 0o755, extra: dict | None = None) -> Path:
    """A wheel laid out as maturin lays out proseg-live's: its programs in .data/scripts, its metadata and licence in
    .dist-info; `extra` adds files {name: text}."""
    path = folder / name
    data = data or program()
    with zipfile.ZipFile(path, "w") as archive:
        for called in programs:
            info = zipfile.ZipInfo(f"proseg_live-3.2.0.data/scripts/{called}")
            info.external_attr = (0o100000 | mode) << 16
            archive.writestr(info, data)
        archive.writestr("proseg_live-3.2.0.dist-info/METADATA", "Metadata-Version: 2.4\nName: proseg-live\nVersion: 3.2.0\n")
        archive.writestr("proseg_live-3.2.0.dist-info/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n")
        if licence:
            archive.writestr("proseg_live-3.2.0.dist-info/licenses/LICENSE.md", licence)
        for member, text in (extra or {}).items():
            archive.writestr(member, text)
    return path


@pytest.fixture
def builder(monkeypatch, tmp_path):
    """The builder's computer as check.py sees it: a home folder and user name of the test's own, no cargo settings,
    not CI. Returns a function that makes a wheel of a program holding these strings and gives check.py's problems."""
    home = tmp_path / "home" / "someone"

    def problems(*texts: str, user: str = "someone", at: Path | str = home, ci: bool = False) -> list[str]:
        monkeypatch.setenv("HOME", str(at))
        monkeypatch.setenv("USER", user)
        monkeypatch.setenv("LOGNAME", user)
        for name in ("CARGO_TARGET_DIR", "CARGO_HOME", "CI"):
            monkeypatch.delenv(name, raising=False)
        if ci:
            monkeypatch.setenv("CI", "true")
        folder = tmp_path / f"wheel-{len(list(tmp_path.glob('wheel-*')))}"
        folder.mkdir()
        return check.wheel_problems(wheel(folder, program(*texts)))

    problems.home = home
    return problems


def test_check_refuses_a_program_that_names_the_builders_computer(builder):
    """A program built without native/build.sh's remaps hands the builder's home folder, user name and build folders to
    everyone who installs the wheel: refused, saying how to build it. Only whole folders count: the name inside a longer
    word, "/root" in a source file named roots.rs (in every build, as a container's root user builds it), or a user
    name that is also a folder every Rust program names, never refuses a wheel. On CI the build machines' own folders
    are refused too."""
    home = builder.home
    refused = builder(f"{home}/.cargo/{REGISTRY}")                         # a plain cargo build's panic location
    assert any(f"its home folder ({home}/)" in problem and "native/build.sh" in problem for problem in refused)
    refused = builder("/scratch/someone/target/release/build/out/bindings.rs")   # the user name, outside the home folder
    assert any("its user name (/someone/)" in problem for problem in refused)
    # A folder native/build.sh renames, outside the home folder: this app's folder (and the target in it).
    refused = builder(f"{ROOT}/native/proseg/src/main.rs", at="/nonexistent-home")
    assert any(f"the folder {ROOT}/" in problem for problem in refused)
    # A user name that is also a folder every Rust program names (/dev/urandom, the standard library's /rust/deps/,
    # the remapped /cargo/) is no path of this computer.
    for user in ("dev", "rust", "cargo"):
        assert builder("/dev/urandom", "/rust/deps/x-0.1.0/src/lib.rs", f"/cargo/{REGISTRY}", user=user,
                       at=f"/home/{user}") == []
    # As native/build.sh writes them: neutral paths. The name inside a word, and "/root" of roots.rs for the root user.
    for user, at in (("someone", home), ("root", "/root")):
        assert builder(f"/cargo/{REGISTRY}", "~/elsewhere.rs", "handsomeone", "/proseg-live/proseg/src/live.rs",
                       user=user, at=at) == []
    # The CI's build folders: a container's cargo home, the runner's home and its workspace.
    runner = "/ho" + "me/runner/work/sopa-companion/src/lib.rs"
    for named, folder in ((f"/root/.cargo/{REGISTRY}", "/root/"), (runner, "/ho" + "me/runner/"),
                          ("/github/workspace/src/main.rs", "/github/")):
        assert builder(named) == []
        assert any(f"the folder {folder}" in problem for problem in builder(named, ci=True)), named


def test_check_refuses_any_file_of_the_wheel_that_names_the_builders_computer(tmp_path, monkeypatch):
    """Not the program alone: a software bill of materials, as maturin writes one by default, names the folder the wheel
    was built in."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CI", raising=False)
    sbom = {"proseg_live-3.2.0.dist-info/sboms/proseg.cyclonedx.json":
            f'{{"bom-ref": "path+file://{ROOT}/native/.cache/work-x/proseg-live-3.2.0#proseg@3.2.0"}}'}
    problems = check.wheel_problems(wheel(tmp_path, extra=sbom))
    assert any(problem.startswith("proseg_live-3.2.0.dist-info/sboms/") and f"the folder {ROOT}/" in problem
               for problem in problems)


def sdist(folder: Path, *, leave_out=(), replace: dict | None = None) -> Path:
    """An sdist as maturin's: proseg_live-3.2.0/ with the package's files; `leave_out` drops some, `replace` changes
    some {name: text}."""
    files = {"pyproject.toml": (ROOT / "native/package/pyproject.toml").read_text(),
             "Cargo.toml": '[package]\nname = "proseg"\n\n[[bin]]\nname = "proseg"\npath = "src/main.rs"\n\n'
                           '[[bin]]\nname = "proseg-live"\npath = "src/main.rs"\n',
             "Cargo.lock": "version = 4\n", "src/main.rs": "fn main() {}\n", "src/live.rs": "// live\n",
             "LICENSE.md": LICENCE, "README.md": "# Proseg\n", "PROSEG-LIVE.md": "# proseg-live\n",
             "proseg-live.patch": "diff -ruN a/Cargo.toml b/Cargo.toml\n", **(replace or {})}
    path = folder / "proseg_live-3.2.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, text in files.items():
            if name in leave_out:
                continue
            data = text.encode()
            info = tarfile.TarInfo(f"proseg_live-3.2.0/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def test_check_reads_the_sdist_and_the_wheels_layout(tmp_path, monkeypatch):
    """The sdist is the complete source (the GPL's, and pip's cargo fallback): every file of check.SOURCE, a Cargo.toml
    with the proseg-live program, proseg's own licence and README beside the package's page and the patch. A wheel holds
    one program, proseg-live (proseg-live.exe on Windows, where no mode is kept), runnable, an ELF of its tag's
    architecture on Linux, proseg's licence in its .dist-info, and no Python module."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CI", raising=False)

    def folder(name: str) -> Path:
        (tmp_path / name).mkdir()
        return tmp_path / name

    assert check.sdist_problems(sdist(folder("good"))) == []
    for missing in ("Cargo.lock", "LICENSE.md", "proseg-live.patch", "src/live.rs", "PROSEG-LIVE.md"):
        assert any(missing in problem for problem in check.sdist_problems(sdist(folder(f"no-{missing.replace('/', '-')}"),
                                                                                leave_out=[missing]))), missing
    no_program = sdist(folder("no-program"), replace={"Cargo.toml": '[[bin]]\nname = "proseg"\npath = "src/main.rs"\n'})
    assert any("[[bin]] proseg-live" in problem for problem in check.sdist_problems(no_program))
    other_licence = sdist(folder("other-licence"), replace={"LICENSE.md": "MIT License\n"})
    assert any("not proseg's" in problem for problem in check.sdist_problems(other_licence))
    page_as_readme = sdist(folder("page-as-readme"), replace={"README.md": "# proseg-live\n"})
    assert any("README.md" in problem for problem in check.sdist_problems(page_as_readme))

    assert check.wheel_problems(wheel(folder("one"))) == []
    for name, layout in (("two", dict(programs=("proseg-live", "proseg"))), ("none", dict(programs=())),
                         ("renamed", dict(programs=("proseg",)))):
        assert any("one program" in problem for problem in check.wheel_problems(wheel(folder(name), **layout))), name
    assert any("licence" in problem for problem in check.wheel_problems(wheel(folder("unlicensed"), licence="")))
    assert any("mode 755" in problem for problem in check.wheel_problems(wheel(folder("not-runnable"), mode=0o644)))
    module = wheel(folder("module"), extra={"proseg_live/__init__.py": ""})
    assert any("more than the program" in problem for problem in check.wheel_problems(module))
    arm = wheel(folder("arm"), name="proseg_live-3.2.0-py3-none-manylinux_2_28_aarch64.whl")       # an x86-64 program
    assert any("ELF program for aarch64" in problem for problem in check.wheel_problems(arm))
    assert check.wheel_problems(wheel(folder("arm-ok"), program(machine=183),
                                      name="proseg_live-3.2.0-py3-none-manylinux_2_28_aarch64.whl")) == []
    windows = wheel(folder("windows"), b"MZ", name="proseg_live-3.2.0-py3-none-win_amd64.whl", programs=("proseg-live.exe",),
                    mode=0o644)
    assert check.wheel_problems(windows) == []
    assert check.wheel_problems(tmp_path / "proseg_live-3.2.0.zip") != []                        # not a wheel's name


def test_check_says_what_it_found_and_exits_with_it(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CI", raising=False)
    good = wheel(tmp_path)
    assert check.main(["wheel", str(good)]) == 0 and ": ok" in capsys.readouterr().out
    (tmp_path / "bad").mkdir()
    bad = wheel(tmp_path / "bad", licence="")
    assert check.main(["wheel", str(good), str(bad)]) == 1
    assert "licence" in capsys.readouterr().err
    assert check.main(["sdist", str(tmp_path / "nothing.tar.gz")]) == 1
    assert check.main(["sdist"]) == 2 and check.main(["tarball", str(good)]) == 2


def test_the_package_is_proseg_live_3_2_0_2_the_patch_that_exports_prosegs_own_count():
    """proseg-live 3.2.0.2: proseg 3.2.0 (the crate native/build.sh pins) with the patch that adds proseg's own count to
    live mode (3.2.0.1) and counts a pause on a frame a token counted again with f (3.2.0.2), so pip replaces an
    installed 3.2.0. The patch adds the three options the engine passes, and --help lists them (the engine's capability
    check reads it)."""
    import tomllib

    package = tomllib.loads((ROOT / "native/package/pyproject.toml").read_text())["project"]
    crate = next(line.split("=", 1)[1] for line in (ROOT / "native/build.sh").read_text().splitlines()
                 if line.startswith("VERSION="))
    assert package["version"] == "3.2.0.2" and package["version"].startswith(crate + ".")
    patch = (ROOT / "native/proseg-live.patch").read_text()
    main = patch.split("diff -ruN a/src/main.rs b/src/main.rs\n", 1)[1].split("\ndiff ", 1)[0]
    for field in ("live_counts: Option<String>,", "live_counts_dir: Option<String>,", "live_counts_start: Option<u64>,"):
        assert f"+    {field}" in main.splitlines(), field


def smoke_module():
    pytest.importorskip("numpy")
    pytest.importorskip("pyarrow")
    spec = importlib.util.spec_from_file_location("proseg_live_smoke", ROOT / "native/package/smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE = '''#!{python}
"""A stand-in proseg-live for smoke.py: says its version, its options, and writes a live folder ({counts})."""
import json, pathlib, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("proseg 3.2.0")
    sys.exit(0)
if args == ["--help"]:
    print("--live-dir --live-control --live-hold --live-tiled" + ("{counts}" != "old" and " --live-counts --live-counts-dir --live-counts-start" or ""))
    sys.exit(0)
live = pathlib.Path(args[args.index("--live-dir") + 1])
(live / "frames").mkdir(parents=True)
(live / "frames" / "000000.bin.zst").write_bytes(b"")
counted = "{counts}" in ("counted", "late", "lost") and "--live-counts" in args
info = {{"schema": "proseg-live-1"}}
if counted:
    info["counts"] = {{"mode": "every", "format": "PLC1"}}
    if "{counts}" != "lost":
        (live / "frames" / "000000.counts.zst").write_bytes(b"")
    line = {{"seq": 0, "reason": "every" if "{counts}" == "late" else "init", "file": "frames/000000.counts.zst"}}
    (live / "counts.jsonl").write_text(json.dumps(line) + "\\n")
else:
    (live / "counts.jsonl").write_text("")
(live / "info.json").write_text(json.dumps(info))
'''


def test_the_smoke_test_runs_proseg_live_counting_every_frame(tmp_path, monkeypatch):
    """native/package/smoke.py (the proseg-live workflow's smoke jobs) passes an installed proseg-live only when its
    --help offers the counts options and its short live run, counting every frame, writes a counts line with its file
    for each frame, the first at the init frame: an older proseg-live, one that does not count, one whose first count
    is not the init frame's, or one whose line names no file, fails it saying why."""
    import sys

    smoke = smoke_module()
    monkeypatch.setattr(smoke, "version", lambda name: "3.2.0.1")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    for kind, problem in (("counted", None), ("silent", "the live run's counts"), ("late", "the live run's counts"),
                          ("lost", "the live run's counts"),
                          ("old", "does not offer --live-dir, --live-control, --live-hold, --live-tiled, --live-counts")):
        fake = tmp_path / f"proseg-live-{kind}"
        fake.write_text(FAKE.format(python=sys.executable, counts=kind))
        fake.chmod(0o755)
        monkeypatch.setattr(smoke, "program", lambda fake=fake: fake)
        if problem is None:
            assert smoke.main() == 0
        else:
            with pytest.raises(SystemExit, match=problem):
                smoke.main()
