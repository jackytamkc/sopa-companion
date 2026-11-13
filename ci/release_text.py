#!/usr/bin/env python3
"""The text of a release: what changed in this version (its section of CHANGELOG.md), then how to install it
(ci/release-notes.md).

    python ci/release_text.py VERSION CHANGELOG NOTES > notes.md

NOTES is the release's text with {version} where the version goes and a line {changes} where this version's changes go.
A version without a section in CHANGELOG stops here (exit 1), so no release goes out without saying what it changed;
ci/check_companion.py asks the same of a tag before anything is built on. Standard library only.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

# A version's heading in CHANGELOG.md: "## 0.8.1 (2026-11-12)". Its section runs to the next "## " heading.
HEADING = re.compile(r"^## (\d+(?:\.\d+)+) \((\d{4}-\d{2}-\d{2})\)$", re.M)


def sections(changelog: str) -> dict[str, tuple[str, str]]:
    """Each version's date and text, in the changelog's own order (newest first)."""
    found = list(HEADING.finditer(changelog))
    ends = [match.start() for match in found[1:]] + [len(changelog)]
    out: dict[str, tuple[str, str]] = {}
    for match, end in zip(found, ends):
        body = changelog[match.end():end]
        later = re.search(r"^## ", body, re.M)          # a heading that is no version's ends the section too
        out[match.group(1)] = (match.group(2), (body[:later.start()] if later else body).strip())
    return out


def changes(version: str, changelog: str) -> str:
    """What `version` changed, or ValueError when the changelog has no section for it or the section lists nothing."""
    section = sections(changelog).get(version)
    if section is None:
        raise ValueError(f"CHANGELOG.md has no section for {version} (a heading \"## {version} (YYYY-MM-DD)\")")
    if not re.search(r"^- \S", section[1], re.M):
        raise ValueError(f"CHANGELOG.md's section for {version} lists no change (lines starting with \"- \")")
    return section[1]


def unwrapped(text: str) -> str:
    """GitHub shows a release's text with every line break kept, so a paragraph or a list item that is wrapped in the
    file becomes one line. Headings, the start of each list item, blank lines and indented code stay as they are."""
    out: list[str] = []
    for line in text.splitlines():
        starts = not line.strip() or line.startswith("    ") or re.match(r"#|- |\d+\. ", line)
        if out and not starts and out[-1].strip() and not out[-1].startswith(("    ", "#")):
            out[-1] += " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def release_text(version: str, changelog: str, notes: str) -> str:
    if notes.count("\n{changes}\n") != 1:
        raise ValueError("the release's text needs one line {changes}, where this version's changes go")
    text = notes.replace("\n{changes}\n", "\n" + changes(version, changelog) + "\n").replace("{version}", version)
    return unwrapped(text)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    version, changelog, notes = argv[0], Path(argv[1]), Path(argv[2])
    try:
        text = release_text(version, changelog.read_text(encoding="utf-8"), notes.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        print(f"release_text.py: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
