"""The app's tests run on any computer and name nothing of one: none names a folder in someone's home, a conda
environment's python or a real sample; what only some computers have (the sopa environment, a measuring harness's
folder) comes from conftest.py, and a test that needs it is skipped, saying what to set, where it is not. Nothing else
that ships names a real sample either: the code, the page, the docs, the settings, the build and packaging files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

import pytest

import conftest

APP = Path(__file__).resolve().parents[1]
# Built from parts, so this file never matches itself: a folder in someone's home (/home/<name>/, not a made-up
# one-letter /home/x/), a python in a conda installation's environments, and the packages of one python version inside
# an environment (lib/python3.x/site-packages: another computer's environment may have another version).
FOLDERS_OF_ONE_COMPUTER = [re.compile("/ho" + r"me/\w{2,}/"),
                           re.compile(r"(?:mini|ana|mamba)?(?:forge|conda)3?/" + "en" + "vs/"),
                           re.compile(r"lib/python3\.\d+/" + "site-" + "packages")]
# A real sample's code, as the samples the app was measured on are named: fixtures name made-up samples (sample-1) and
# comments say "a whole sample".
REAL_SAMPLES = [re.compile(r"(?i)\b" + "d" + r"u\d{2}\b")]
# The app's tests: pytest's and node's, with their helpers and data makers.
TESTS = ("tests/**/*.py", "frontend/test/*.mjs")
# Every other text file that ships (the repository, and the page and settings the wheel carries; proseg-live's package,
# the CI's helpers and workflows): a real sample's code in any of them reaches every reader, as a page's comment or a
# docstring's example. The settings and the docs may give an example conda environment under the home folder, so only
# the real samples are looked for there. "ci/**/*", not "ci/**": on Python 3.12 that pattern finds folders only.
SHIPPED = ("README.md", "config.example.toml", "pyproject.toml", "requirements.txt", "cli.py",
           "__init__.py", "__main__.py", ".gitignore", "docs/ARCHITECTURE.md", "frontend/index.html", "frontend/*.md",
           "frontend/src/*", "frontend/styles/*", "native/build.sh", "native/proseg-live.patch", "native/package/*",
           "packaging/**/*", "names.py", "dirs.py", "locate.py", "portable.py", "private.py", "analysis/**/*.py", "api/**/*.py", "desktop/**/*.py",
           "desktop/*.svg", "engine/**/*.py", "server/**/*.py", "zarr_bridge/**/*.py", "tests/**/*.json",
           "frontend/test/fixtures/*.json", "ci/**/*", ".github/workflows/*")


def files(globs) -> list[Path]:
    """The files these patterns find in the app's folder. Each must find one at least: a folder that moved would
    otherwise leave the check without a word."""
    found = set()
    for pattern in globs:
        matched = {path for path in APP.glob(pattern) if path.is_file() and "__pycache__" not in path.parts}
        assert matched, f"{pattern} finds no file"
        found |= matched
    return sorted(found)


def lines_matching(patterns: list[re.Pattern], globs=TESTS) -> list[str]:
    """Every line of these files (the tests unless said) that one of the patterns finds, as file:line: text (lines as
    an editor counts them: a test's text may hold a U+2028)."""
    paths = files(globs)
    assert len(paths) > 100
    return [f"{path.relative_to(APP)}:{n}: {line.strip()[:100]}" for path in paths
            for n, line in enumerate(path.read_text(errors="replace").split("\n"), 1)
            if any(pattern.search(line) for pattern in patterns)]


def test_no_test_names_a_folder_of_one_computer():
    assert lines_matching(FOLDERS_OF_ONE_COMPUTER) == []


def test_no_test_names_a_real_sample():
    assert lines_matching(REAL_SAMPLES) == []


def test_nothing_that_ships_names_a_real_sample():
    assert lines_matching(REAL_SAMPLES, SHIPPED) == []


def test_what_a_computer_lacks_is_skipped_and_said(monkeypatch, tmp_path):
    from sopa_companion.engine import config as C
    python = tmp_path / "env" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    # This computer's settings: the first sopa python that is there ([zarr] empty, [stain]'s gone, [qc]'s there).
    monkeypatch.setattr(C, "load_config", lambda: {"zarr": {"python": ""}, "stain": {"python": str(tmp_path / "gone")},
                                                   "qc": {"python": str(python)}})
    assert conftest.configured(("zarr", "python"), ("stain", "python"), ("qc", "python")) == str(python)

    def no_settings():
        raise FileNotFoundError("No settings")
    monkeypatch.setattr(C, "load_config", no_settings)                           # settings that do not load: none
    assert conftest.configured(("zarr", "python")) == ""
    # No sopa python, no harness folder: None where a check is optional, else the test is skipped, saying what to set.
    monkeypatch.setattr(conftest, "SOPA", str(tmp_path / "gone"))
    monkeypatch.setattr(conftest, "HARNESS", "")
    assert conftest.sopa_python(skip=False) is None and conftest.harness_path("fixtures", skip=False) is None
    assert conftest.harness_crop(skip=False) is None
    with pytest.raises(pytest.skip.Exception, match="sopa environment"):
        conftest.sopa_python()
    with pytest.raises(pytest.skip.Exception, match=conftest.HARNESS_VAR):
        conftest.harness_crop()
    monkeypatch.setattr(conftest, "SOPA", str(python))
    assert conftest.sopa_python() == str(python)
    # A harness folder anywhere: its crop is found by the name its fixtures.json records, once the harness has made it.
    monkeypatch.setattr(conftest, "HARNESS", str(tmp_path / "harness"))
    with pytest.raises(pytest.skip.Exception, match="fixtures/fixtures.json"):
        conftest.harness_crop()
    (tmp_path / "harness" / "fixtures").mkdir(parents=True)
    (tmp_path / "harness" / "fixtures" / "fixtures.json").write_text('{"crop": {"path": "/elsewhere/a-crop.zarr"}}')
    with pytest.raises(pytest.skip.Exception, match="fixtures/a-crop.zarr"):
        conftest.harness_crop()                                                  # recorded, not made yet
    (tmp_path / "harness" / "fixtures" / "a-crop.zarr").mkdir()
    assert conftest.harness_crop() == tmp_path / "harness" / "fixtures" / "a-crop.zarr"
    assert conftest.harness_path("fixtures", "fixtures.json") == tmp_path / "harness" / "fixtures" / "fixtures.json"


@conftest.needs_sopa
def test_the_mini_harness_is_a_harness_folder_the_tests_read(monkeypatch, tmp_path):
    """tests/make_mini_harness.py, the harness folder's stand-in on a computer without the harness (CI): its crop is
    found by the name its fixtures.json records, and the engine's inspection of the crop is where the crop's tests read
    it (test_tissue.py), under the crop's own path. It makes a folder of its own only: a folder that holds something is
    refused."""
    root = tmp_path / "harness"
    command = [conftest.sopa_python(), str(APP / "tests/make_mini_harness.py"), str(root)]
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"} | {"MPLCONFIGDIR": str(tmp_path / "mpl")}
    made = subprocess.run(command, capture_output=True, text=True, env=env, timeout=600)
    assert made.returncode == 0, made.stdout + made.stderr[-3000:]
    monkeypatch.setattr(conftest, "HARNESS", str(root.resolve()))
    crop = conftest.harness_crop()
    assert crop == root.resolve() / "fixtures" / "mini-crop.zarr" and (crop / "zarr.json").is_file()
    record = json.loads(conftest.harness_path("fixtures", "fixtures.json").read_text())["crop"]
    assert record["points_key"] == "transcripts" and record["transcripts"] == 20000
    inspected = [json.loads(path.read_text()) for path in conftest.harness_path("var", "cache", "zarr").glob("*/inspect.json")]
    assert [found.get("path") for found in inspected] == [str(crop)]
    assert "region_of_interest" in {shape["key"] for shape in inspected[0]["shapes"]}
    again = subprocess.run(command, capture_output=True, text=True, env=env, timeout=600)
    assert again.returncode != 0 and "must be an empty folder" in again.stderr
