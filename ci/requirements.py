"""The app's dependencies from pyproject.toml, one per line, as pip reads them: the companion's own list without
proseg-live, plus the extras named on the command line.

    python ci/requirements.py              the release's dependencies.txt, headed by a comment that says what it is
    python ci/requirements.py test [...]   the same list and these extras' (CI's test environment)

proseg-live is the release's own package, as the app is: no line names it, nor the app itself.
An extra that names the app (all = sopa-companion[sopa,stain,qc]) stands for those extras' own lists, so pip is never
asked to look for sopa-companion either. Standard library only.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
FROM_THE_RELEASE = ("sopa-companion", "proseg-live")
HEADER = ("# SoPA companion {version}: the packages it needs from PyPI (pip install -r this file). The app and "
          "proseg-live are on PyPI too, and in this release's files.")


def name(requirement: str) -> str:
    """A requirement's project name as an index compares it (lower case; runs of -, _ and . are one -)."""
    return re.sub(r"[-_.]+", "-", re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement).group(1)).lower()


def extras_named(requirement: str) -> list[str]:
    """The extras a requirement asks for: "sopa-companion[sopa,stain,qc]>=0.1" -> ["sopa", "stain", "qc"]."""
    found = re.match(r"\s*[A-Za-z0-9][A-Za-z0-9._-]*\s*\[([^\]]*)\]", requirement)
    return [extra.strip() for extra in found.group(1).split(",") if extra.strip()] if found else []


def version() -> str:
    """The app's version, __init__.py's __version__ (pyproject.toml takes it from there)."""
    text = (ROOT / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M).group(1)


def requirements(extras=()) -> list[str]:
    """The base list and these extras' lists, in pyproject.toml's order, each requirement once, never one of
    FROM_THE_RELEASE. A requirement of the app itself brings the extras it names."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    optional = project.get("optional-dependencies") or {}
    unknown = [extra for extra in extras if extra not in optional]
    if unknown:
        raise SystemExit(f"ci/requirements.py: pyproject.toml has no extra {', '.join(unknown)} "
                         f"(it has {', '.join(optional)})")
    own = name(project["name"])
    found: list[str] = []

    def add(group: list[str], read: frozenset) -> None:
        for requirement in group:
            if name(requirement) == own:
                for extra in extras_named(requirement):
                    if extra in optional and extra not in read:
                        add(optional[extra], read | {extra})
            elif name(requirement) not in FROM_THE_RELEASE and requirement not in found:
                found.append(requirement)

    add(project["dependencies"], frozenset())
    for extra in extras:
        add(optional[extra], frozenset({extra}))
    return found


def main(extras: list[str]) -> int:
    lines = requirements(extras)
    print("\n".join(lines if extras else [HEADER.format(version=version()), *lines]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
