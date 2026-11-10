"""The app's Python code and its tests say what the code does and why, never who asked for it: no comment or docstring
dates a request to the person who made it ("owner, <date>: ...", "the owner's rule, <date>"), which an upload would
carry as it stands. The page's own files have their check in frontend/test/qc-copy.test.mjs. Nothing that ships points
at a review's findings or a design note's sections either: the upload holds neither, so a reader could not follow it."""
from __future__ import annotations

import re

from sopa_companion.engine.config import ROOT
from test_tests_portable import SHIPPED, TESTS, lines_matching

# The app's code and its tests (the harness is the measuring tool's own, never part of the app).
FOLDERS = ("analysis", "api", "desktop", "engine", "server", "tests", "zarr_bridge")
# A request dated to whoever made it, also where a comment's line break splits it; or what they said or asked.
ATTRIBUTED = re.compile(r"(?i)\bowner(?:'s \w+)?, 20\d\d-\d\d-\d\d|\bowner (?:said|asked)\b")


def test_no_comment_dates_a_request_to_a_person():
    found = []
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            text = re.sub(r"\s*\n\s*(?:#\s*)?", " ", path.read_text(errors="replace"))     # lines joined, as read
            found += [f"{path.relative_to(ROOT)}: {match.group(0)}" for match in ATTRIBUTED.finditer(text)]
    assert not found, found


# Built from parts, so this file never matches itself: a review's finding ids (a review thread's, a sceptic's, a lens
# with a number), a design note by its file name, and a design decision by the number of the owner's answer.
NOT_IN_THE_UPLOAD = [re.compile(r"\brev-(?:engine|page|samples)" + "#" + r"\d|\bsceptic-g" + r"\d"),
                     re.compile(r"\b(?:streamline|security|bugs|infrastructure|privacy)" + "-" + r"\d+\b"),
                     re.compile(r"\b[A-Z0-9]+_" + "DESIGN" + r"\b|\bCONTRACT" + r"\.md\b"),
                     re.compile(r"(?i)\bowner " + "answer" + r"\b")]


def test_nothing_that_ships_points_at_a_review_or_a_design_note():
    assert lines_matching(NOT_IN_THE_UPLOAD, (*SHIPPED, *TESTS)) == []
