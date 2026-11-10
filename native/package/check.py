"""proseg-live's source distribution and wheels, checked before anyone gets them (native/build.sh --wheel and the
proseg-live workflow run this):

    python3 native/package/check.py sdist FILE
    python3 native/package/check.py wheel FILE...

sdist: the complete source the GPL asks for, released beside the wheels: the package's pyproject.toml, the patched
crate (its Cargo.toml with the proseg-live program, Cargo.lock, src/main.rs, src/live.rs), proseg's own LICENSE.md and
README.md, the package's page PROSEG-LIVE.md and proseg-live.patch itself.
wheel: one program, proseg-live (proseg-live.exe on Windows), runnable, and no Python module; proseg's licence inside its
.dist-info; on Linux an ELF of the wheel's architecture; and nothing of the computer that built it, in any of its files.
A plain cargo build writes the builder's home folder, user name and build folders into the program's panic messages
(native/build.sh writes neutral names in their place), and maturin's software bill of materials names the folder a
wheel was built in: the wheel would give them to everyone who installs it.
Standard library only: it runs in any Python 3.11 or newer.
"""
from __future__ import annotations

import getpass
import os
from pathlib import Path
import re
import sys
import tarfile
import tomllib
import zipfile

PROGRAM = "proseg-live"
LICENCE = "Proseg is Copyright (C) 2024 by Daniel C. Jones"
APP = Path(__file__).resolve().parents[2]          # the app's folder: native/package/check.py is inside it
# A Linux wheel's architecture (the end of its platform tag) and the ELF machine number its program must have.
ELF_MACHINES = {"x86_64": 62, "aarch64": 183}
# The places the CI's computers build in (CI=true): a program built there must not name them either.
CI_FOLDERS = ("/root/", "/home/runner/", "/github/")
# What the source distribution holds, inside its top folder.
SOURCE = ("pyproject.toml", "Cargo.toml", "Cargo.lock", "src/main.rs", "src/live.rs", "LICENSE.md", "README.md",
          "PROSEG-LIVE.md", "proseg-live.patch")


def names_of_this_computer(data: bytes, home: str, user: str, folders=()) -> list[str]:
    """What of the builder's computer the program names: a path inside the home folder or inside a folder that
    native/build.sh gives a neutral name (`folders`: this app's folder, the cargo target and cargo's home), or the user
    name as the folder of a per-user root (/home/<name>/, /scratch/<name>/, ...). Only whole folders count: "/root" in
    ".../src/roots.rs" is no path of this computer, nor is a user name that is also a folder every Rust program names
    (/dev/urandom, the standard library's /rust/deps/) without a per-user root."""
    home = home.rstrip("/")
    found = [f"its home folder ({home}/)"] if home and (home + "/").encode() in data else []
    for folder in dict.fromkeys(folder.rstrip("/") for folder in folders):
        inside_home = home and (folder + "/").startswith(home + "/")
        if folder and not inside_home and (folder + "/").encode() in data:
            found.append(f"the folder {folder}/")
    if user and re.search(rb"/(?:home|Users|scratch|data|work)/" + re.escape(user.encode()) + rb"/", data):
        found.append(f"its user name (/{user}/)")
    return found


def given_and_resolved(path: str) -> list[str]:
    """`path` as it is given and with its links resolved: a build can see either."""
    if not path:
        return []
    try:
        return [path, str(Path(path).resolve())]
    except (OSError, RuntimeError):
        return [path]


def this_computer() -> tuple[str, str, list[str]]:
    """(home folder, user name, the folders native/build.sh renames): this app's folder, the target (CARGO_TARGET_DIR,
    a relative one from native/; else native/target) and cargo's home (CARGO_HOME, else ~/.cargo)."""
    try:
        home = str(Path.home())
    except (RuntimeError, KeyError):           # no home folder known: nothing of it to look for
        home = ""
    try:
        user = getpass.getuser()
    except Exception:                          # no user name known (OSError from Python 3.13, KeyError before)
        user = ""
    target = os.environ.get("CARGO_TARGET_DIR") or "target"
    target = target if os.path.isabs(target) else str(APP / "native" / target)
    cargo_home = os.environ.get("CARGO_HOME") or (home + "/.cargo" if home else "")
    folders = [*given_and_resolved(str(APP)), *given_and_resolved(target), *given_and_resolved(cargo_home)]
    if os.environ.get("CI") == "true":
        folders += CI_FOLDERS
    return home, user, folders


def wheel_problems(path: Path) -> list[str]:
    match = re.fullmatch(r"proseg_live-([^-]+)-py3-none-([^-]+)\.whl", path.name)
    if not match:
        return ["not a proseg-live wheel's name (proseg_live-<version>-py3-none-<platform>.whl)"]
    version, platforms = match.groups()
    windows = platforms.startswith("win")
    program = PROGRAM + (".exe" if windows else "")
    problems = []
    with zipfile.ZipFile(path) as wheel:
        files = [info for info in wheel.infolist() if not info.is_dir()]
        info_folder = f"proseg_live-{version}.dist-info/"
        scripts = f"proseg_live-{version}.data/scripts/"
        programs = [info for info in files if info.filename.startswith(scripts)]
        others = sorted(info.filename for info in files if not info.filename.startswith((info_folder, scripts)))
        if [info.filename for info in programs] != [scripts + program]:
            problems.append(f"it must hold one program, {scripts}{program}; it holds "
                            f"{[info.filename for info in programs] or 'none'}")
        if others:
            problems.append(f"it holds more than the program and its .dist-info: {others[:5]}")
        try:
            metadata = wheel.read(info_folder + "METADATA").decode("utf-8", "replace")
        except KeyError:
            metadata = ""
        if f"\nName: {PROGRAM}\n" not in "\n" + metadata or f"\nVersion: {version}\n" not in "\n" + metadata:
            problems.append(f"its METADATA does not say Name: {PROGRAM} and Version: {version}")
        licences = [info.filename for info in files if info.filename.startswith(info_folder)
                    and info.filename != info_folder + "METADATA"
                    and wheel.read(info).decode("utf-8", "replace").lstrip().startswith(LICENCE)]
        if not licences:
            problems.append(f"proseg's licence (LICENSE.md, starting \"{LICENCE}\") is not in {info_folder}")
        computer = this_computer()
        for info in files:
            named = names_of_this_computer(wheel.read(info), *computer)
            if named:
                problems.append(f"{info.filename} names the computer that built it: {' and '.join(named)}. Build it "
                                "again with native/build.sh --wheel, which writes neutral names in their place")
        for info in programs[:1]:
            data = wheel.read(info)
            if not windows and (info.external_attr >> 16) & 0o777 != 0o755:
                problems.append(f"{info.filename} is not mode 755 (it is {oct((info.external_attr >> 16) & 0o777)})")
            arch = next((arch for arch in ELF_MACHINES for tag in platforms.split(".") if tag.endswith("_" + arch)
                         and "linux" in tag), None)
            if "linux" in platforms:
                machine = int.from_bytes(data[18:20], "little") if len(data) >= 20 else None
                if data[:4] != b"\x7fELF" or data[4:6] != b"\x02\x01" or arch is None or machine != ELF_MACHINES[arch]:
                    problems.append(f"{info.filename} is not a 64-bit little-endian ELF program for {arch or platforms}")
    return problems


def sdist_problems(path: Path) -> list[str]:
    match = re.fullmatch(r"proseg_live-([^-]+)\.tar\.gz", path.name)
    if not match:
        return ["not a proseg-live source distribution's name (proseg_live-<version>.tar.gz)"]
    top = f"proseg_live-{match.group(1)}/"
    with tarfile.open(path) as archive:
        members = {member.name: member for member in archive.getmembers() if member.isfile()}

        def text(name: str) -> str:
            stream = archive.extractfile(members[top + name])
            return stream.read().decode("utf-8", "replace") if stream else ""

        missing = [name for name in SOURCE if top + name not in members]
        problems = [f"it lacks {', '.join(missing)} in {top}"] if missing else []
        if "Cargo.toml" not in missing:
            try:
                bins = {entry.get("name"): entry.get("path") for entry in tomllib.loads(text("Cargo.toml")).get("bin", [])}
            except tomllib.TOMLDecodeError as error:
                bins = {}
                problems.append(f"its Cargo.toml does not read: {error}")
            if bins.get(PROGRAM) != "src/main.rs":
                problems.append(f"its Cargo.toml has no [[bin]] {PROGRAM} on src/main.rs (proseg-live.patch adds it)")
        if "pyproject.toml" not in missing:
            project = tomllib.loads(text("pyproject.toml"))
            if (project.get("project") or {}).get("name") != PROGRAM:
                problems.append(f"its pyproject.toml is not {PROGRAM}'s")
        if "LICENSE.md" not in missing and not text("LICENSE.md").lstrip().startswith(LICENCE):
            problems.append(f"its LICENSE.md is not proseg's (it starts \"{LICENCE}\")")
        if "README.md" not in missing and "PROSEG-LIVE.md" not in missing and text("README.md") == text("PROSEG-LIVE.md"):
            problems.append("its README.md is the package's page, not proseg's own")
        if "proseg-live.patch" not in missing and not text("proseg-live.patch").startswith("diff "):
            problems.append("its proseg-live.patch is not the patch")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("sdist", "wheel") or (argv[0] == "sdist" and len(argv) != 2):
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    check = sdist_problems if argv[0] == "sdist" else wheel_problems
    failed = False
    for name in argv[1:]:
        path = Path(name)
        try:
            problems = check(path)
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as error:
            problems = [f"cannot be read: {error}"]
        for problem in problems:
            print(f"check.py: {path}: {problem}", file=sys.stderr)
        if not problems:
            print(f"check.py: {path}: ok")
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
