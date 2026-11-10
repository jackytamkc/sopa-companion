"""The CI's helpers (ci/) and workflows (.github/workflows/), without CI: the release's dependencies.txt, the check of the
companion's release files, the squatter's index the release job installs beside, the weekly resolver's sets (with a
stand-in uv), and workflows that pin every action to a commit and call only files that are here."""
from __future__ import annotations

import base64
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile

from sopa_companion import __version__
from sopa_companion.engine.config import ROOT

CI = ROOT / "ci"
WORKFLOWS = ROOT / ".github/workflows"
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]


def run(*args: str, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run([*args], capture_output=True, text=True, timeout=120, **kwargs)


def requirements_txt(*extras: str) -> list[str]:
    done = run(sys.executable, str(CI / "requirements.py"), *extras)
    assert done.returncode == 0, done.stderr
    return done.stdout.splitlines()


def test_dependencies_txt_is_the_apps_list_without_the_release_packages():
    """dependencies.txt (the release's first pip line) is pyproject.toml's dependencies without proseg-live, headed by
    what it is; with extras (CI's test environment) their lists follow. No line ever names the app or proseg-live: an
    extra that names the app stands for its own extras' lists."""
    lines = requirements_txt()
    assert lines[0] == (f"# SoPA companion {__version__}: the packages it needs from PyPI (pip install -r this file). "
                        "The app and proseg-live are on PyPI too, and in this release's files.")
    assert lines[1:] == [text for text in PROJECT["dependencies"] if not text.startswith("proseg-live")]
    extras = PROJECT["optional-dependencies"]
    assert requirements_txt("test") == lines[1:] + extras["test"]
    everything = requirements_txt("all")
    assert set(everything) == set(lines[1:]) | {*extras["sopa"], *extras["stain"], *extras["qc"]}
    assert len(everything) == len(set(everything))                      # sopa>=2.2.10 is in the base and in [sopa]
    assert not [line for line in everything if re.match(r"(?i)(sopa[-_]companion|proseg[-_]live)\b", line)]
    unknown = run(sys.executable, str(CI / "requirements.py"), "nothing")
    assert unknown.returncode == 1 and "no extra nothing" in unknown.stderr


def release(folder: Path, *, version: str = __version__, metadata_extra: str = "", leave_out=(), add=(),
            entry_points: str = "[console_scripts]\nsopa-companion = sopa_companion.cli:main\n", sdist: bool = True,
            dependencies: list[str] | None = None) -> Path:
    """A release folder as python -m build and ci/requirements.py leave it: the companion's wheel (every module and
    package-data file this tree declares, METADATA from pyproject.toml), its sdist and dependencies.txt."""
    sys.path.insert(0, str(CI))
    try:
        import check_companion
    finally:
        sys.path.remove(str(CI))
    folder.mkdir(parents=True, exist_ok=True)
    info = f"sopa_companion-{version}.dist-info"
    requires = "".join(f"Requires-Dist: {text}\n" for text in PROJECT["dependencies"])
    requires += "".join(f'Requires-Dist: {text}; extra == "{extra}"\n'
                        for extra, texts in PROJECT["optional-dependencies"].items() for text in texts)
    with zipfile.ZipFile(folder / f"sopa_companion-{version}-py3-none-any.whl", "w") as wheel:
        for name in sorted(check_companion.expected_files() - set(leave_out)) + list(add):
            wheel.writestr(name, "")
        wheel.writestr(f"{info}/METADATA", f"Metadata-Version: 2.4\nName: sopa-companion\nVersion: {version}\n"
                                           f"{requires}{metadata_extra}")
        wheel.writestr(f"{info}/WHEEL", "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n")
        wheel.writestr(f"{info}/entry_points.txt", entry_points)
    if sdist:
        (folder / f"sopa_companion-{version}.tar.gz").write_bytes(b"")
    lines = requirements_txt() if dependencies is None else dependencies
    (folder / "dependencies.txt").write_text("\n".join(lines) + "\n")
    return folder


def check_companion(folder: Path, *version: str) -> subprocess.CompletedProcess:
    return run(sys.executable, str(CI / "check_companion.py"), str(folder), *version)


def test_check_companion_passes_the_release_files_and_names_what_differs(tmp_path):
    """The companion's release check: a pure Python wheel of this version (a tag's version must be the code's), with
    the command, every module and the page, the proseg-live pin of proseg-live's package, sopa and platformdirs, and
    nothing of native/; its sdist; dependencies.txt the wheel's own requirements without proseg-live."""
    good = check_companion(release(tmp_path / "good"))
    assert good.returncode == 0, good.stderr
    assert check_companion(release(tmp_path / "tag"), __version__).returncode == 0
    cases = {
        "a tag that is not the code's version": (dict(), ("9.9.9",)),
        "no sdist": (dict(sdist=False), ()),
        "a module missing": (dict(leave_out=["sopa_companion/cli.py"]), ()),
        "proseg-live inside the app's wheel": (dict(add=["sopa_companion/native/bin/proseg-live"]), ()),
        "no command": (dict(entry_points="[console_scripts]\n"), ()),
        "dependencies.txt naming proseg-live": (dict(dependencies=requirements_txt() + ["proseg-live==3.2.0.*"]), ()),
        "dependencies.txt without one": (dict(dependencies=requirements_txt()[:-1]), ()),
        "dependencies.txt with one more": (dict(dependencies=requirements_txt() + ["requests>=2"]), ()),
    }
    for case, (layout, version) in cases.items():
        done = check_companion(release(tmp_path / re.sub(r"\W+", "-", case), **layout), *version)
        assert done.returncode == 1 and "check_companion.py:" in done.stderr, case
    # The pin follows proseg-live's own version: its proseg's first three numbers, and the oldest patch the app needs
    # (NEEDS) as the floor.
    package = tomllib.loads((ROOT / "native/package/pyproject.toml").read_text())["project"]["version"]
    assert f'"{checker().proseg_live_pin(package)}"' in (ROOT / "pyproject.toml").read_text()


def checker():
    """ci/check_companion.py as a module (standard library only)."""
    sys.path.insert(0, str(CI))
    try:
        import check_companion
    finally:
        sys.path.remove(str(CI))
    return check_companion


def test_the_pin_has_proseg_live_s_patch_as_its_floor():
    """ci/check_companion.py's rule: proseg-live's version pins its proseg (==3.2.0.*), and the floor is the oldest patch
    the companion needs (NEEDS: 3.2.0.1, the first patch that keeps proseg's own count), so pip replaces an installed
    3.2.0 and keeps a 3.2.0.1 while the package is 3.2.0.2. Without NEEDS, or for another proseg, the package's own
    fourth number is the floor, and a version without one pins that proseg alone. The release check reads a build's own
    order of the two (==3.2.0.*,>=3.2.0.1) as the same."""
    pin, normal = checker().proseg_live_pin, checker().normal
    assert checker().NEEDS == "3.2.0.1"
    assert pin("3.2.0.2") == pin("3.2.0.1") == "proseg-live>=3.2.0.1,==3.2.0.*"
    assert pin("3.2.0", needs=None) == "proseg-live==3.2.0.*"
    assert pin("3.2.0.2", needs=None) == "proseg-live>=3.2.0.2,==3.2.0.*"
    assert pin("3.2.1.4") == "proseg-live>=3.2.1.4,==3.2.1.*"
    assert normal("proseg-live>=3.2.0.1,==3.2.0.*") == normal("proseg_live == 3.2.0.*, >=3.2.0.1")
    assert normal("proseg-live>=3.2.0.1,==3.2.0.*") != normal("proseg-live==3.2.0.*")
    assert normal("Uvicorn[standard] >= 0.54.0") == "uvicorn[standard]>=0.54.0"
    assert normal('scvi-tools>=1.5; extra == "qc"') == 'scvi-tools>=1.5;extra=="qc"'
    assert normal("sopa[cellpose,stain]<3,>=2") == normal("sopa[cellpose,stain]>=2,<3")


def test_the_squatters_index_holds_both_names_above_the_release(tmp_path):
    """The release job installs beside a squatter's simple index: both of the app's names, at versions an index would
    prefer (proseg-live's within the companion's ==3.2.0.* pin too), each a wheel that installs only its metadata,
    listed with its sha256 as pip reads a simple index."""
    done = run(sys.executable, str(CI / "squat_index.py"), str(tmp_path / "squat"))
    assert done.returncode == 0, done.stderr
    root = tmp_path / "squat"
    assert re.findall(r'href="([^"]+)/"', (root / "index.html").read_text()) == ["sopa-companion", "proseg-live"]
    found = {}
    for name in ("sopa-companion", "proseg-live"):
        for href in re.findall(r'href="([^"]+)"', (root / name / "index.html").read_text()):
            wheel_name, digest = href.split("#sha256=")
            data = (root / name / wheel_name).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
            with zipfile.ZipFile(root / name / wheel_name) as wheel:
                assert all(member.split("/")[0].endswith(".dist-info") for member in wheel.namelist())
                metadata = wheel.read(next(m for m in wheel.namelist() if m.endswith("/METADATA"))).decode()
                record = wheel.read(next(m for m in wheel.namelist() if m.endswith("/RECORD"))).decode()
                for line in record.splitlines():
                    member, hashed, _ = line.split(",")
                    if hashed:
                        expected = base64.urlsafe_b64encode(hashlib.sha256(wheel.read(member)).digest()).rstrip(b"=")
                        assert hashed == "sha256=" + expected.decode()
            found.setdefault(name, []).append(re.search(r"^Version: (.+)$", metadata, re.M).group(1))
    assert found == {"sopa-companion": ["99.0"], "proseg-live": ["99.0", "3.2.0.99"]}


FAKE_UV = """#!/usr/bin/env bash
echo "$*" >> "$FAKE_UV_LOG"
out=""
while [[ $# -gt 0 ]]; do [[ "$1" == -o ]] && out="$2"; shift; done
if [[ -n "${FAKE_UV_FAIL:-}" && "$out" == $FAKE_UV_FAIL ]]; then echo "No solution found" >&2; exit 1; fi
printf 'sopa==2.2.11\\n    # via -r base.in\\n' > "$out"
"""


def test_the_weekly_resolver_resolves_each_set_and_fails_only_on_linux_base_and_mixing(tmp_path):
    """ci/resolve.sh writes the sets (the app's list without its comment, sopa[cellpose] unpinned, ResolVI's packages,
    RAPIDS for CUDA 13) and the overrides beside them, resolves each set where it must (with --no-build, NVIDIA's index
    for RAPIDS, macOS 15 for macOS), and fails only when a base or mixing set on Linux does not resolve; RAPIDS and the
    other systems warn. Its accepted answers are in ci/resolve/expected/."""
    fakes = tmp_path / "bin"
    fakes.mkdir()
    (fakes / "uv").write_text(FAKE_UV)
    (fakes / "uv").chmod(0o755)
    log = tmp_path / "uv.log"
    env = {"PATH": f"{fakes}:/usr/bin:/bin", "FAKE_UV_LOG": str(log), "HOME": str(tmp_path)}
    done = run("bash", str(CI / "resolve.sh"), str(tmp_path / "out"), env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    out = tmp_path / "out"
    assert (out / "companion.in").read_text().splitlines() == requirements_txt()[1:]
    assert (out / "base.in").read_text().splitlines() == ["sopa[cellpose]", "-r companion.in"]
    assert (out / "mixing.in").read_text().splitlines() == ["-r base.in", "scvi-tools>=1.5", "scviva-tools>=0.1.7"]
    assert (out / "gpu-cu13.in").read_text().splitlines() == ["-r mixing.in", "rapids-singlecell-cu13[rapids]"]
    assert (out / "overrides.txt").read_text() == (CI / "resolve/overrides.txt").read_text()
    calls = log.read_text().splitlines()
    names = sorted(call.split(" -o ")[1] for call in calls)
    linux = [f"{s}-{p}-{a}-manylinux_2_28.txt" for s in ("base", "mixing") for p in ("3.12", "3.13")
             for a in ("x86_64", "aarch64")]
    others = [f"{s}-{p}-{a}.txt" for p in ("3.12", "3.13") for s, a in (("gpu-cu13", "x86_64-manylinux_2_28"),
                                                                      ("base", "aarch64-apple-darwin"),
                                                                      ("base", "x86_64-pc-windows-msvc"))]
    assert names == sorted(linux + others) and sorted(path.name for path in out.glob("*.txt")) == sorted(
        [*names, "overrides.txt"])
    assert all("--no-build" in call and "--override overrides.txt" in call for call in calls)
    assert all(("--extra-index-url https://pypi.nvidia.com" in call) == ("gpu-cu13.in" in call) for call in calls)
    # The answers' notes name the inputs by their names alone: the same wherever the output folder is.
    assert "    # via -r base.in" in (out / "base-3.12-x86_64-manylinux_2_28.txt").read_text()
    warned = run("bash", str(CI / "resolve.sh"), str(tmp_path / "warned"),
                 env={**env, "FAKE_UV_FAIL": "*-apple-darwin.txt"})
    assert warned.returncode == 0 and "::warning::base-3.12-aarch64-apple-darwin does not resolve" in warned.stdout
    assert (tmp_path / "warned/base-3.12-aarch64-apple-darwin.err").is_file()
    failed = run("bash", str(CI / "resolve.sh"), str(tmp_path / "failed"),
                 env={**env, "FAKE_UV_FAIL": "mixing-3.13-aarch64-manylinux_2_28.txt"})
    assert failed.returncode == 1 and "::error::mixing-3.13-aarch64-manylinux_2_28 does not resolve" in failed.stdout
    # The accepted answers: every set the resolver writes, from the inputs it writes today.
    expected = CI / "resolve/expected"
    assert sorted(path.name for path in expected.glob("*.txt")) == sorted([*names, "overrides.txt"])
    for name in ("companion.in", "base.in", "mixing.in", "gpu-cu13.in"):
        assert (expected / name).read_text() == (out / name).read_text(), name


def uses() -> list[tuple[str, str]]:
    """(workflow, the uses: value) of every step and job in the workflows."""
    return [(path.name, match.group(1)) for path in sorted(WORKFLOWS.glob("*.yml"))
            for match in re.finditer(r"^\s*(?:-\s+)?uses:\s*(\S+)", path.read_text(), re.M)]


def test_the_workflows_pin_every_action_to_a_commit_and_tools_to_versions():
    """Every action is pinned to a commit (40 hex digits) with the release it is beside it, or is this repository's
    own workflow; uv, maturin and Rust are pinned to versions."""
    found = uses()
    assert {name for name, _ in found} == {"tests.yml", "proseg-live.yml", "release.yml", "resolve.yml"}
    for name, value in found:
        assert value in ("./.github/workflows/proseg-live.yml", "./.github/workflows/tests.yml") or \
            re.fullmatch(r"[\w.-]+/[\w.-]+@[0-9a-f]{40}", value), (name, value)
    for path in WORKFLOWS.glob("*.yml"):
        for line in path.read_text().splitlines():
            if re.search(r"uses:\s*[\w.-]+/[\w.-]+@[0-9a-f]{40}", line):
                assert re.search(r"#\s*(v\d+(\.\d+)*|master)\s*$", line), (path.name, line)
    text = "\n".join(path.read_text() for path in WORKFLOWS.glob("*.yml"))
    assert 'version: "0.11.28"' in text and 'MATURIN_VERSION: "1.15.0"' in text and 'toolchain: "1.94.1"' in text


def test_the_workflows_call_files_that_are_here():
    """Each helper, script and text the workflows run or read is in the repository (a rename leaves no workflow calling
    a file that is gone), and the shell scripts are runnable and parse."""
    text = "\n".join(path.read_text() for path in WORKFLOWS.glob("*.yml"))
    named = set(re.findall(r"(?:sopa_companion/)?((?:ci|native|tests)/[\w./-]+\.(?:py|sh|md))", text))
    assert {"ci/requirements.py", "ci/check_companion.py", "ci/squat_index.py", "ci/install_smoke.sh", "ci/resolve.sh",
            "ci/release-notes.md", "native/build.sh", "native/package/check.py", "native/package/smoke.py",
            "tests/make_mini_harness.py"} <= named
    for path in named:
        assert (ROOT / path).is_file(), path
    scripts = ("ci/install_smoke.sh", "ci/resolve.sh", "native/build.sh")
    for script in scripts:
        assert (ROOT / script).stat().st_mode & 0o111, script
        parsed = run("bash", "-n", str(ROOT / script))
        assert parsed.returncode == 0, parsed.stderr
    # The workflows run them directly, so a checkout must make them runnable: git records them so (a repository on a
    # disk that keeps no file modes would otherwise record what it reads, not runnable). Only in the app's own
    # repository: an installed copy may keep an empty .git, or sit inside another repository that does not track it.
    top = run("git", "-C", str(ROOT), "rev-parse", "--show-toplevel") if shutil.which("git") else None
    if top is not None and top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT.resolve():
        recorded = run("git", "-C", str(ROOT), "ls-files", "-s", "--", *scripts)
        assert recorded.returncode == 0, recorded.stderr
        modes = {line.split("\t")[1]: line.split()[0] for line in recorded.stdout.splitlines()}
        assert modes == dict.fromkeys(scripts, "100755"), modes


def jobs(text: str) -> dict[str, str]:
    """A workflow's jobs, each as its own lines (the block under `  <name>:` in `jobs:`)."""
    found, name = {}, None
    for line in text.split("\njobs:\n", 1)[1].splitlines():
        if match := re.fullmatch(r"  ([\w-]+):\s*", line):
            name = match.group(1)
            found[name] = ""
        elif name is not None:
            found[name] += line + "\n"
    return found


def test_only_the_job_that_publishes_can_write_and_only_after_the_tests():
    """Every workflow's token reads the repository only; the release's publish job alone may write (it creates the
    release), and only after the app's tests (tests.yml, called by release.yml), the builds and the testers' install have
    passed: the other jobs run code from PyPI (build backends, sopa's packages). No checkout keeps the token for the
    steps after it. PyPI takes both packages by trusted publishing, after the same checks and before the release here:
    two jobs that can prove who they are (id-token) and nothing else, each in the environment PyPI knows its project's
    publisher by, proseg-live before the app that needs it; no PyPI token is kept in the repository."""
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text()
        assert re.search(r"^permissions: \{contents: read\}$", text, re.M), path.name
        checkouts = re.findall(r"^( *)- uses: actions/checkout@\S+.*\n\1  with: \{([^}]*)\}", text, re.M)
        assert len(checkouts) == text.count("uses: actions/checkout@") and all(
            "persist-credentials: false" in options for _, options in checkouts), path.name
        writes = [name for name, block in jobs(text).items() if "contents: write" in block]
        assert writes == (["publish"] if path.name == "release.yml" else []), path.name
    release = jobs((WORKFLOWS / "release.yml").read_text())
    # The publish job alone may also sign: its files' build provenance (OIDC and attestations), made only where GitHub
    # makes attestations, a public repository (skipped while this one is private).
    assert "permissions: {contents: write, id-token: write, attestations: write}" in release["publish"]
    assert re.search(r"if: \$\{\{ !github\.event\.repository\.private \}\}\n\s+uses: actions/attest-build-provenance@[0-9a-f]{40}",
                     release["publish"])
    for path in sorted(WORKFLOWS.glob("*.yml")):
        signers = [name for name, block in jobs(path.read_text()).items() if "id-token: write" in block]
        assert signers == (["pypi-proseg-live", "pypi-companion", "publish"] if path.name == "release.yml" else []), path.name
    for name, needs, files in (("pypi-proseg-live", "[tests, install]", "proseg_live-*"),
                               ("pypi-companion", "[pypi-proseg-live]", "sopa_companion-*")):
        job = release[name]
        environment = "pypi-sopa-companion" if name == "pypi-companion" else name
        assert re.search(rf"^    needs: {re.escape(needs)}$", job, re.M) and f"\n    environment: {environment}\n" in job
        assert "if: github.event_name == 'push' && github.ref_type == 'tag'" in job
        assert "permissions: {id-token: write}" in job and "actions/checkout@" not in job
        assert f"run: mkdir upload && cp release/{files} upload/" in job
        assert re.search(r"uses: pypa/gh-action-pypi-publish@[0-9a-f]{40}\s+# v[\d.]+\n\s+with: \{packages-dir: upload, skip-existing: true\}", job)
    text = (WORKFLOWS / "release.yml").read_text()
    assert "PYPI_API_TOKEN" not in text and "password:" not in text and "secrets." not in text
    # The release asks both for macOS and Windows too: their tests (tests.yml systems) and proseg-live's wheels.
    assert release["tests"].strip() == "uses: ./.github/workflows/tests.yml\n    with: {other_systems: true}"
    assert release["proseg-live"].strip() == "uses: ./.github/workflows/proseg-live.yml\n    with: {other_systems: true}"
    assert re.search(r"^    needs: \[tests, install, pypi-companion\]$", release["publish"], re.M)
    assert "if: github.event_name == 'push' && github.ref_type == 'tag'" in release["publish"]
    tests = (WORKFLOWS / "tests.yml").read_text()
    assert re.search(r"^  workflow_call:\n    inputs:\n      other_systems: \{type: boolean, default: false\}$", tests, re.M)
    # Called by the release, its runs keep apart from a push's own (github.workflow is the caller's name there).
    assert 'group: "tests-${{ github.workflow }}-${{ github.ref }}"' in tests


def test_the_release_job_installs_its_own_files_beside_a_squatters_index_and_find_links():
    """Before it publishes, the release job installs the files PyPI will serve: the run's own two wheels in a new folder,
    given to pip as files, with a squatter's index beside PyPI and its wheels as find-links folders, in the environment
    variables and the environment's own pip settings: the pip line must leave them all out (its versions are checked
    after, by install_smoke.sh)."""
    from test_config_docs import INSTALL
    install = jobs((WORKFLOWS / "release.yml").read_text())["install"]
    lines = [line.strip() for line in install.splitlines()]
    pip_line = INSTALL[2].replace("pip install", "env/bin/pip install", 1)
    copy = 'cp sopa-companion-release/sopa_companion-*.whl sopa-companion-release/proseg_live-*_$(uname -m).whl "$d"'
    assert lines.index(copy) == lines.index(INSTALL[0]) + 1 and lines.index(pip_line) == lines.index(copy) + 1
    assert "--pattern 'sopa_companion-*.whl'" in INSTALL[1] and '--pattern "proseg_live-*_$(uname -m).whl"' in INSTALL[1]
    assert 'PIP_EXTRA_INDEX_URL: "file://${{ github.workspace }}/squat"' in lines
    assert ('PIP_FIND_LINKS: "${{ github.workspace }}/squat/sopa-companion ${{ github.workspace }}/squat/proseg-live"'
            in lines)
    assert "find-links = %s/squat/sopa-companion %s/squat/proseg-live" in install and "> env/pip.conf" in install
    assert lines.index("- run: ci/install_smoke.sh env/bin sopa-companion-release") > lines.index(pip_line)


def test_the_release_installs_and_runs_the_notebook_route_on_linux_macos_and_windows():
    """The install job runs on Linux, macOS (Apple silicon) and Windows (x64), each with the release job's lines for it:
    Linux's and macOS's are the same three (uname -m names the wheel), Windows' the PowerShell three in the environment
    made active, with the run's own wheels in the new folder. Each then runs the notebook route end to end
    (ci/launch_smoke.py: a kernel, proseg, Cellpose, saves, close, a killed kernel); the desktop's install_smoke.sh stays
    Linux's. The release publishes every system's wheels."""
    from test_config_docs import INSTALL, WINDOWS_INSTALL
    release = (WORKFLOWS / "release.yml").read_text()
    install = jobs(release)["install"]
    assert "os: [ubuntu-24.04, macos-15, windows-2022]" in install and "fail-fast: false" in install
    lines = [line.strip() for line in install.splitlines()]
    mac = lines.index("if: runner.os == 'macOS'")
    copy = 'cp sopa-companion-release/sopa_companion-*.whl sopa-companion-release/proseg_live-*_$(uname -m).whl "$d"'
    assert lines[mac + 2:mac + 5] == [INSTALL[0], copy, INSTALL[2].replace("pip install", "env/bin/pip install", 1)]
    windows = lines.index("if: runner.os == 'Windows'")
    assert lines[windows + 1:windows + 7] == [
        "shell: pwsh", "run: |", ". ./env/Scripts/Activate.ps1", WINDOWS_INSTALL[0],
        "Copy-Item sopa-companion-release/sopa_companion-*.whl, sopa-companion-release/proseg_live-*_amd64.whl $d",
        WINDOWS_INSTALL[2]]
    assert lines[lines.index("- run: ci/install_smoke.sh env/bin sopa-companion-release") + 1] == "if: runner.os == 'Linux'"
    assert '"$ENV_PY" ci/launch_smoke.py "$RUNNER_TEMP/launch-smoke"' in lines
    publish = jobs(release)["publish"]
    assert "macosx" not in publish and "win_amd64" not in publish            # nothing is left out of the release
    proseg = (WORKFLOWS / "proseg-live.yml").read_text()
    assert "continue-on-error" not in proseg and "aarch64-apple-darwin" in proseg and "x86_64-pc-windows-msvc" in proseg
