"""The companion's release files, checked before anyone gets them (release.yml runs this after python -m build and
ci/requirements.py):

    python ci/check_companion.py DIST [VERSION]

VERSION is __init__.py's __version__ unless one is given: a tag passes its own, so a tag that is not the code's version
fails. DIST must hold
- the wheel, sopa_companion-VERSION-py3-none-any.whl: pure Python (no proseg-live inside: that is a package of its own,
  native/package/), its METADATA saying Version: VERSION and requiring proseg-live as proseg-live's version pins it
  (proseg_live_pin), sopa>=2.2.10 and platformdirs>=4; the sopa-companion command (sopa_companion.cli:main); every
  module of the packages pyproject.toml declares and its package data (the page, config.example.toml, the icon), and
  nothing else of this folder: nothing of native/, tests/, harness/ or ci/;
- the source distribution, sopa_companion-VERSION.tar.gz;
- dependencies.txt (ci/requirements.py): the wheel's own requirements without an extra and without proseg-live, the same
  set, so the release's first pip line installs exactly what the second one needs.
Standard library only.
"""
from __future__ import annotations

import configparser
from email.parser import Parser
from pathlib import Path
import re
import sys
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "sopa_companion"
# What the release's lines take from the release folder, never from an index (README, Install).
FROM_THE_RELEASE = ("sopa-companion", "proseg-live")
# Besides proseg-live's pin, every wheel requires sopa itself and platformdirs (the per-user folders off Linux).
REQUIRED = ("sopa>=2.2.10", "platformdirs>=4")
# Folders of this repository that are tools, never the app: nothing of them is in the wheel.
NOT_THE_APP = ("native", "tests", "harness", "ci", "docs", "packaging", ".github")


def code_version() -> str:
    text = (ROOT / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M).group(1)


# The oldest proseg-live the companion needs: 3.2.0.1, the first patch that keeps proseg's own count. A later patch on
# the same proseg is taken, never required (3.2.0.2 counts a pause again with f; with 3.2.0.1 such a pause reads the
# averaged count, and the page says so).
NEEDS = "3.2.0.1"


def proseg_live_pin(version: str | None = None, needs: str | None = NEEDS) -> str:
    """The companion's pin on proseg-live, from its package's version (`version`: native/package's, unless given): the
    proseg it patches, ==<its first three numbers>.* (3.2.1 is another proseg), and a floor: `needs` when it is a patch
    of that proseg, else the package's own fourth number when it has one. So 3.2.0.2 gives proseg-live>=3.2.0.1,==3.2.0.*
    (pip replaces an installed 3.2.0 and keeps a 3.2.0.1), and 3.2.0 without `needs` gives proseg-live==3.2.0.*."""
    if version is None:
        package = tomllib.loads((ROOT / "native/package/pyproject.toml").read_text(encoding="utf-8"))
        version = package["project"]["version"]
    parts, need = version.split("."), (needs or "").split(".")
    floor = needs if len(need) > 3 and need[:3] == parts[:3] else version if len(parts) > 3 else None
    return "proseg-live" + (f">={floor}," if floor else "") + "==" + ".".join(parts[:3]) + ".*"


def normal(requirement: str) -> str:
    """A requirement as text that compares equal however it was written: no spaces, its name in lower case with runs of
    -, _ and . as one -, and its version specifiers in one order (a build writes proseg-live>=3.2.0.1,==3.2.0.* as
    proseg-live==3.2.0.*,>=3.2.0.1)."""
    text = re.sub(r"\s+", "", requirement)
    found = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[^\]]*\])?([^;]*)(.*)", text)
    if not found or not found.group(1):
        return text
    name, extras, specifiers, marker = found.groups()
    specifiers = ",".join(sorted(part for part in specifiers.split(",") if part))
    return re.sub(r"[-_.]+", "-", name).lower() + (extras or "") + specifiers + marker


def expected_files() -> set[str]:
    """What the wheel must hold of this folder, by pyproject.toml's setuptools settings: each declared package's modules
    and the package data's files."""
    settings = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"]["setuptools"]
    folders = {name: ROOT / settings["package-dir"][PACKAGE] / Path(*name.split(".")[1:]) for name in settings["packages"]}
    files = {f"{name.replace('.', '/')}/{path.name}" for name, folder in folders.items() for path in folder.glob("*.py")}
    for name, patterns in settings.get("package-data", {}).items():
        files |= {f"{name.replace('.', '/')}/{path.relative_to(folders[name]).as_posix()}"
                  for pattern in patterns for path in folders[name].glob(pattern) if path.is_file()}
    return files


def wheel_problems(wheel_path: Path, version: str) -> tuple[list[str], set[str]]:
    """The wheel's problems, and its requirements without an extra (normalised) for dependencies.txt's check."""
    problems: list[str] = []
    info = f"{PACKAGE}-{version}.dist-info/"
    with zipfile.ZipFile(wheel_path) as wheel:
        names = {name for name in wheel.namelist() if not name.endswith("/")}

        def read(name: str) -> str:
            try:
                return wheel.read(info + name).decode("utf-8")
            except KeyError:
                problems.append(f"{wheel_path.name} has no {info}{name}")
                return ""

        metadata = Parser().parsestr(read("METADATA"))
        tags = Parser().parsestr(read("WHEEL"))
        entry_points = read("entry_points.txt")
    if metadata["Name"] != "sopa-companion" or metadata["Version"] != version:
        problems.append(f"its METADATA says {metadata['Name']} {metadata['Version']}, not sopa-companion {version}")
    if tags.get_all("Tag") != ["py3-none-any"] or tags["Root-Is-Purelib"] != "true":
        problems.append(f"it is not a pure Python wheel (WHEEL: {tags.get_all('Tag')}, Root-Is-Purelib "
                        f"{tags['Root-Is-Purelib']})")
    requires = metadata.get_all("Requires-Dist") or []
    base = {normal(r) for r in requires if not re.search(r";.*\bextra\s*==", r)}
    for wanted in (proseg_live_pin(), *REQUIRED):
        if normal(wanted) not in base:
            problems.append(f"it does not require {wanted} (it requires {sorted(base)})")
    commands = configparser.ConfigParser()
    commands.read_string(entry_points)
    if commands.get("console_scripts", "sopa-companion", fallback="").strip() != "sopa_companion.cli:main":
        problems.append("its entry_points.txt has no command sopa-companion = sopa_companion.cli:main")
    missing = sorted(expected_files() - names)
    if missing:
        problems.append(f"it lacks {', '.join(missing[:10])}{' ...' if len(missing) > 10 else ''}")
    stray = sorted(name for name in names if name.split("/")[0] == PACKAGE and len(name.split("/")) > 2
                   and name.split("/")[1] in NOT_THE_APP)
    if stray:
        problems.append(f"it holds what is no part of the app: {', '.join(stray[:10])}")
    return problems, {r for r in base if not r.startswith(FROM_THE_RELEASE)}


def dependencies_problems(path: Path, base: set[str]) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return [f"no {path.name}: {error}"]
    listed = {normal(line.split("#", 1)[0]) for line in lines if line.split("#", 1)[0].strip()}
    named = sorted(line for line in listed if line.startswith(FROM_THE_RELEASE))
    problems = [f"{path.name} names {', '.join(named)}, which come from the release folder alone"] if named else []
    if listed != base:
        problems.append(f"{path.name} is not the wheel's requirements: only in {path.name}: {sorted(listed - base)}; "
                        f"only in the wheel: {sorted(base - listed)}")
    return problems


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    dist = Path(argv[0])
    version = argv[1] if len(argv) == 2 else code_version()
    problems = [] if version == code_version() else [f"the version given, {version}, is not the code's, {code_version()}"]
    wheel = dist / f"{PACKAGE}-{version}-py3-none-any.whl"
    base: set[str] = set()
    if wheel.is_file():
        found, base = wheel_problems(wheel, version)
        problems += found
    else:
        problems.append(f"there is no {wheel.name} in {dist}")
    if not (dist / f"{PACKAGE}-{version}.tar.gz").is_file():
        problems.append(f"there is no {PACKAGE}-{version}.tar.gz in {dist}")
    if wheel.is_file():
        problems += dependencies_problems(dist / "dependencies.txt", base)
    for problem in problems:
        print(f"check_companion.py: {problem}", file=sys.stderr)
    if not problems:
        print(f"check_companion.py: {dist}: sopa-companion {version}, its source and dependencies.txt are ok")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
