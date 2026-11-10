"""The docs that ship against the code they describe: docs/ARCHITECTURE.md lists every route the engine answers and no
route it does not; README.md, docs/DEVELOPING.md and docs/ARCHITECTURE.md say what leaving a run and closing do (a run
nobody answered for is discarded, `stop --force` included, and the app keeps no runs between sessions); README.md says
how the app is used from a notebook and which settings that route reads, and installs the app and proseg-live from the
release's two files alone (as the release's own text does); docs/DEVELOPING.md says how a source folder is set up; no
text sends anyone to an index for the two; the docs say what a training folder others can write to risks through the
Cellpose GUI; README.md says which computers and which Python the app runs on, and sopa's links point where sopa lives
now."""
from __future__ import annotations

import re
import shlex
import tomllib

from fastapi.routing import APIRoute, APIWebSocketRoute

from sopa_companion.engine.config import ROOT
from test_tests_portable import SHIPPED, TESTS, lines_matching

README = (ROOT / "README.md").read_text()
DEVELOPING = (ROOT / "docs/DEVELOPING.md").read_text()
ARCHITECTURE = (ROOT / "docs/ARCHITECTURE.md").read_text()
RELEASE_NOTES = (ROOT / "ci/release-notes.md").read_text()
# The README's install line, on every system: both packages are on PyPI, and the app's pin brings proseg-live.
PYPI_INSTALL = 'pip install "sopa-companion[all]"'
# The release job's lines (release.yml, install), before PyPI has the files it is about to publish: a new folder, the
# run's own two wheels into it (the app, and this computer's proseg-live; a release's download by the same patterns),
# then one pip line that installs both from those files and what they need from PyPI. pip takes a name given as a file
# from that file alone, so neither name is asked of an index (nor of pip's own settings, which may name indexes or
# find-links folders).
INSTALL = ("d=$(mktemp -d)",
           "gh release download --repo jackytamkc/sopa-companion --dir \"$d\" --pattern 'sopa_companion-*.whl' "
           "--pattern \"proseg_live-*_$(uname -m).whl\"",
           "pip install \"sopa-companion[all]\" \"$d\"/sopa_companion-*.whl \"$d\"/proseg_live-*.whl")
# Windows' lines (PowerShell), the same three: a new folder, the two wheels (Windows' proseg-live is the amd64 one),
# both given to pip as files.
WINDOWS_INSTALL = ("$d = New-Item -ItemType Directory -Path (Join-Path ([IO.Path]::GetTempPath()) ([guid]::NewGuid()))",
                   "gh release download --repo jackytamkc/sopa-companion --dir $d --pattern 'sopa_companion-*.whl' "
                   "--pattern 'proseg_live-*_amd64.whl'",
                   'pip install "sopa-companion[all]" (Resolve-Path "$d\\sopa_companion-*.whl") '
                   '(Resolve-Path "$d\\proseg_live-*.whl")')
# On a computer no proseg-live wheel fits: its source distribution from the same release, built by pip in its place.
SOURCE_BUILD = ("gh release download --repo jackytamkc/sopa-companion --dir \"$d\" --pattern 'proseg_live-*.tar.gz'",
                "pip install \"sopa-companion[all]\" \"$d\"/sopa_companion-*.whl \"$d\"/proseg_live-*.tar.gz")
BASE = "/api/v1"
# Answered by the security middleware itself (server/security.py), before any route: the launcher's one-use codes, and
# the page's trade of one for its key.
MIDDLEWARE = {("POST", "/session/launch-code"), ("POST", "/session/sign-in")}


def section(text: str, heading: str) -> str:
    """The text under a heading, up to the next heading of the same or a higher level."""
    level = len(heading.split(" ", 1)[0])
    start = text.index("\n" + heading + "\n") + len(heading) + 2
    following = re.search(rf"^#{{1,{level}}} ", text[start:], re.M)
    return text[start:start + following.start()] if following else text[start:]


def words(text: str) -> str:
    """Prose as one line: markdown's line breaks inside a paragraph are spaces."""
    return re.sub(r"\s+", " ", text)


def normal(path: str) -> str:
    """A route's path with every parameter alike ({run_id}, {id}, {seq:int}: {}) and no query."""
    return re.sub(r"\{[^}]*\}", "{}", path.split("?", 1)[0])


def documented() -> list[tuple[str, str, set[str]]]:
    """(method, path as shown, the paths it can stand for) for every path of ARCHITECTURE.md's route table. Each path
    takes each method of its row; a path after the first may leave out the start it shares with the first
    (`/runs/{id}/frames`, `/frames/{seq}`), so it stands for itself after any start of the first path."""
    entries = []
    for line in section(ARCHITECTURE, "## HTTP API").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not line.startswith("| ") or len(cells) < 3 or "`" not in cells[1]:
            continue                                        # the header row and the prose around the table
        shown = re.findall(r"`([^`]+)`", cells[1])
        assert all(path.startswith("/") for path in shown), line
        paths = [normal(path) for path in shown]
        first = paths[0].split("/")
        starts = {"/".join(first[:k]) for k in range(1, len(first))}         # "", "/runs", "/runs/{}", ...
        for text, path in zip(shown, paths):
            stands_for = {path} if path == paths[0] else {start + path for start in starts}
            entries += [(method, text, stands_for) for method in cells[0].split(", ")]
    return entries


def test_the_architecture_lists_every_route_the_engine_answers(config):
    """Every route the engine registers is in the table, and every path in the table is one the engine registers: a
    route the engine dropped (the previous run's readout) or gained (leaving a run, the runs of earlier versions, crash
    reports, the page's log, a stain run's cell values) cannot go unsaid. The WebSocket has its own paragraph."""
    from sopa_companion.server import create_app
    app = create_app(config, token="t" * 40)
    routes = {(method, normal(route.path[len(BASE):])) for route in app.routes if isinstance(route, APIRoute)
              and route.path.startswith(BASE + "/") for method in route.methods - {"HEAD"}} | MIDDLEWARE
    entries = documented()
    missing = sorted(f"{method} {path}" for method, path in routes
                     if not any(method == listed and path in stands_for for listed, _, stands_for in entries))
    assert missing == [], "answered by the engine, not in docs/ARCHITECTURE.md"
    unknown = sorted(f"{method} {shown}" for method, shown, stands_for in entries
                     if not any((method, path) in routes for path in stands_for))
    assert unknown == [], "in docs/ARCHITECTURE.md, not answered by the engine"
    sockets = [route.path for route in app.routes if isinstance(route, APIWebSocketRoute)]
    assert sockets == [BASE + "/ws"] and f"**WebSocket** `{BASE}/ws`" in ARCHITECTURE


def test_the_docs_say_what_leaving_and_closing_do():
    """The app keeps no runs between sessions, every way out asks Save to the sample, Discard or Stay, and nobody
    answering discards: the docs say so, never that an unanswered run, a closed dialog, no desktop or `stop --force`
    saves it (server/app.py decide_closing and POST /shutdown discard), nor that a Runs list, a session key or the
    previous run's readout is there, nor that a run outlives the app. The README says it in a notebook's words (Add to
    your SpatialData), docs/DEVELOPING.md in the standalone engine's."""
    notebook = words(section(README, "### One run at a time"))
    assert "keeps no runs between sessions" in notebook
    assert "**Add to your SpatialData**" in notebook and "**Discard**" in notebook and "**Stay**" in notebook
    force = next(line for line in DEVELOPING.splitlines() if line.startswith("sopa-companion stop --force"))
    said = force.split("#", 1)[1]
    assert "discarded" in said and "saves" not in said
    run = words(section(DEVELOPING, "## The standalone engine"))
    assert "keeps no runs between sessions" in run
    assert "**Save to the sample**" in run and "**Discard**" in run and "**Stay**" in run
    for unanswered in ("or no answer within `[desktop] save_dialog_timeout_s`, discards it, and so does an engine with no "
                       "desktop or no `zenity` to ask on", "`stop --force` asks nobody: it discards the run nobody answered for",
                       "closes as at any other close, so an unanswered run is discarded"):
        assert unanswered in run, unanswered
    closing = words(section(ARCHITECTURE, "### Closing"))
    assert "`sopa_companion stop --force` (`POST /shutdown`, bearer token only) asks nobody: it discards" in closing
    assert "discards), and are discarded when there is no display or no dialog tool to ask on" in closing
    # The last page decides: a goodbye closes and discards what it did not answer for; none (away) pauses the runs.
    assert ("The last page gone with a goodbye is a real close" in closing and
            "a run that page did not answer for is discarded without the dialog" in closing)
    assert "is away: each run still sampling pauses as its Pause would" in closing
    assert "A page that went without a goodbye" in run and "is away: its run pauses, the engine keeps it and waits" in run
    leaving = words(section(ARCHITECTURE, "### Leaving a run"))
    assert '`POST /runs/{id}/leave {answer: "add" | "discard" | "stay", revision?, page?}`' in leaving
    assert "Nothing survives a session" in leaving
    assert "| POST | `/shutdown` | `stop --force`: discard the runs nobody answered for" in ARCHITECTURE
    for name, text in (("README.md", README), ("docs/DEVELOPING.md", DEVELOPING), ("docs/ARCHITECTURE.md", ARCHITECTURE)):
        stale = [phrase for phrase in ("saves them", "saving its runs", "save every active run", "saves every active run",
                                       "session key", "**Runs** lists", "qc/previous", "also after the app was closed",
                                       "also after an engine restart", "max_active_runs` hold", "max_active_runs`: runs")
                 if phrase in words(text)]
        assert stale == [], name


def test_the_readme_says_how_a_notebook_uses_the_app_and_what_it_reads():
    """From a notebook: launch(sdata), where results go, the printed link (one use, within 10 minutes), closing, the
    session's folders, and the settings that route reads: the user's own ~/.config/sopa-companion/config.toml after the
    app's files (an installed copy's example; a source folder's, which docs/DEVELOPING.md gives), the tables launch()
    takes, and that "auto" (or an empty value) is the kernel's own Python when it has the feature's packages, which
    proseg-live is found for."""
    notebook = section(README, "## Use it from a notebook")
    prose = words(notebook)
    for said in ("sopa_companion.launch(sdata)", "*Add to your SpatialData*", "`app.added`", "once, within 10 minutes",
                 "`app.close()`", '`app.close(pending="add")`', "`$XDG_RUNTIME_DIR/sopa-companion/<id>/`",
                 "`<scratch_dir>/sopa-companion-<id>/`", "`~/.config/sopa-companion/config.toml`", "`$XDG_CONFIG_HOME`",
                 "an installed copy's own `config.example.toml`", "`setpriv`"):
        assert said in prose, said
    assert ("A notebook's `launch` from a source folder reads the same files (`config.toml`, else `config.example.toml`, "
            "then `config.local.toml`), then your own settings file") in words(section(DEVELOPING, "## From a source folder"))
    rows = {line.strip("|").split(" | ")[0].strip() for line in notebook.splitlines() if line.startswith("| `[")}
    assert rows == {"`[api] engine_python`", "`[api] scratch_dir`", "`[api] model_cache`", "`[desktop] browser`",
                    "`[desktop] chrome_flags`", "`[proseg] binary`", "`[analysis] python`, `gpu`",
                    "`[stain] python`, `gpu`", "`[qc] python`, `gpu`"}
    assert "Other settings in your own file do not reach the session's engine" in prose
    assert "`auto` (the default) finds the one installed with the app" in prose
    assert ("Clusters & UMAP's Python: `auto` (or empty), the kernel's own when it has scipy, scikit-learn, igraph, "
            "umap-learn and numba (sopa brings them; on the CPU)") in prose
    assert "Cellpose's Python: `auto` (or empty), the kernel's own when it has Cellpose; `off`" in prose
    assert ("the mixing check's Python: `auto` (or empty), the kernel's own when it has scvi-tools, scviva-tools and "
            "torch; `off`") in prose
    # The Install section names your own settings file, and the architecture describes the session.
    assert "Your own settings file is `~/.config/sopa-companion/config.toml`" in words(section(README, "## Install"))
    assert "\n## A notebook's session (API mode)\n" in ARCHITECTURE and "**`session.toml`**" in ARCHITECTURE


def test_the_readme_says_nothing_of_the_dormant_instanseg():
    """InstanSeg is dormant (an internal switch, off unless set: engine/stain.py StainRuns.methods): the README, which
    users read, names it nowhere: not as a feature, a setting, a save name, a smoke test, a download or a dependency."""
    assert "instanseg" not in README.lower()


def test_the_readme_installs_from_pypi_with_one_line():
    """The README's Install gives one line on each system (macOS, Windows, Linux), the same: the app with its extras
    from PyPI, which brings proseg-live; updating is that line with --upgrade. Nothing asks for the GitHub CLI any
    more. The release's own text gives the same line."""
    install = section(README, "## Install")
    lines = [line.strip() for line in install.splitlines()]
    assert lines.count(PYPI_INSTALL) == 3
    assert '`pip install --upgrade "sopa-companion[all]"`' in install
    assert "gh release download" not in README and "gh auth login" not in README and "GitHub CLI" not in README
    assert "](https://prism-oncology.github.io/sopa/getting_started/)" in section(README, "## Requirements")
    notes = [line.strip() for line in RELEASE_NOTES.splitlines()]
    assert PYPI_INSTALL in notes and "gh release download" not in RELEASE_NOTES and "{version}" in RELEASE_NOTES


def pip_installs(text: str) -> list[str]:
    """Every pip install command a text gives: on one line, or in an inline code span across a paragraph's lines (a
    fenced block's lines count one by one); each up to the end of its line, its span, a comment or the next command."""
    prose = re.sub(r"^ *```.*?^ *```", "", text, flags=re.M | re.S)
    spans = [match.group(1) for match in re.finditer(r"`([^`\n]*(?:\n[^`\n]*)*?)`", prose)]
    found = []
    for piece in [*text.splitlines(), *(" ".join(span.split()) for span in spans)]:
        for match in re.finditer(r"\bpip3? install\b([^`#;|&\n]*)", piece):
            found.append("pip install" + re.split(r"\bpip3? install\b", match.group(1))[0])
    return found


def asked_of_an_index(command: str) -> list[str]:
    """What of sopa-companion, proseg-live or sopa[companion] a pip install command asks an index for: none when it has
    --no-index, or when it gives both as files (the app's wheel, and proseg-live's wheel or source distribution): pip
    takes a name given as a file from that file alone, a requirement of the same name with extras, and the app's own pin
    on proseg-live, included. Else each name it gives, and a wheel of the app given alone (pip asks an index for its
    proseg-live); proseg-live's source distribution alone asks for nothing."""
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if "--no-index" in parts:
        return []
    given = [word.strip("'\".,:;()") for word in parts[2:]]
    files = [re.split(r"[\\/]", word)[-1] for word in given if re.search(r"[\\/]", word)]   # POSIX or Windows paths
    if any(re.fullmatch(r"sopa_companion-[^/]*\.whl", name) for name in files) and \
            any(re.fullmatch(r"proseg_live-[^/]*\.(whl|tar\.gz)", name) for name in files):
        return []
    asked = []
    for word in given:
        name = re.split(r"[\\/]", word)[-1]
        if word.startswith("-") or (re.search(r"[\\/]", word) and re.fullmatch(r"proseg_live-[^/]*\.tar\.gz", name)):
            continue
        if re.fullmatch(r"(?i)(sopa[-_]companion|proseg[-_]live)(\[[^\]]*\])?([<>=!~].*)?", word) or \
                re.match(r"(?i)sopa\[[^\]]*\bcompanion\b", word) or re.match(r"(?i)(sopa_companion|proseg_live)-.*\.whl$", name):
            asked.append(word)
    return asked


def test_the_release_job_s_lines_ask_no_index_for_the_two_packages():
    """Both packages are on PyPI under the project's own names, so the README's line asks PyPI for them. The release
    job installs the files it is about to publish before PyPI has them: its lines give both to pip as files (the app's
    wheel with proseg-live's wheel or source distribution), which asks no index for either name, whatever an index or a
    find-links folder beside PyPI holds (a local wheel of the app alone would ask one for proseg-live)."""
    assert asked_of_an_index(pip_installs(PYPI_INSTALL)[0]) == ['"sopa-companion[all]"'.strip('"')]
    for asking in ("pip install sopa-companion", "pip install --upgrade 'sopa[companion]'", "pip install proseg-live==3.2.0",
                   "pip install sopa_companion-0.2.0-py3-none-linux_x86_64.whl",
                   "pip install --find-links sopa-companion-release sopa-companion proseg-live",
                   'pip install "sopa-companion[all]" "$d"/sopa_companion-*.whl',
                   "pip install sopa-companion-release/sopa_companion-0.4.1-py3-none-any.whl proseg-live"):
        assert asked_of_an_index(pip_installs(asking)[0]), asking
    for fine in (INSTALL[2], WINDOWS_INSTALL[2], SOURCE_BUILD[1], "pip install 'sopa[cellpose]'", "pip install scvi-tools scviva-tools",
                 "pip install 'cellpose[gui]'", ".venv/bin/pip install --no-deps -e .",
                 "pip install --no-deps sopa-companion-release/proseg_live-3.2.0.2.tar.gz"):
        assert not asked_of_an_index(pip_installs(fine)[0]), fine
    assert pip_installs("the `pip\n  install sopa-companion` line") == ["pip install sopa-companion"]


def test_a_source_folder_installs_without_dependencies_and_names_its_environments():
    """A clone's own environment installs the app with --no-deps (its dependencies, sopa and proseg-live among them,
    are an installed copy's, and on no index the folder uses), and its config.toml names the environments that have sopa,
    Cellpose and what Clusters & UMAP need (or RAPIDS): "auto" finds none of them in the engine's own environment.
    requirements.txt says the same. docs/DEVELOPING.md says it: the README's users install the release."""
    source = section(DEVELOPING, "## From a source folder")
    assert ".venv/bin/pip install --no-deps -e ." in source and "native/build.sh" in source
    prose = words(source)
    assert ("your sopa environment's Python as `[zarr] python`, `[stain] python` and `[analysis] python` (and "
            "`[qc] python` once it has scvi-tools and scviva-tools), or a RAPIDS environment's as `[analysis] python`") \
        in prose
    header = [line for line in (ROOT / "requirements.txt").read_text().splitlines() if line.startswith("#")]
    assert any("pip install --no-deps -e ." in line for line in header)


def test_the_docs_say_a_training_folder_others_can_write_to_can_run_code_through_the_cellpose_gui():
    """The Cellpose GUI loads a folder's `_seg.npy` masks with pickle (when it opens a patch and when it trains), so a
    mask someone else put in a training folder they can write to runs as the user: the README says so where it tells
    where patches go, the architecture says so, and neither says that nothing in a patch folder runs."""
    patches = words(README[README.index("**Training patches.**"):README.index("### Proseg Live")])
    assert ("The Cellpose GUI loads the `_seg.npy` masks with Python's pickle, which can run code: a mask someone else "
            "puts in a folder they can write to runs as you when the GUI opens it or trains on it.") in patches
    assert "Keep training folders where only you, or people you trust, can write." in patches
    assert ("the Cellpose GUI itself loads the folder's `_seg.npy` masks with pickle when it opens a patch and when it "
            "trains, so a mask put there by someone who can write to the folder runs as the user") in words(ARCHITECTURE)
    for name, text in (("README.md", README), ("docs/DEVELOPING.md", DEVELOPING), ("docs/ARCHITECTURE.md", ARCHITECTURE)):
        assert "nothing in a patch folder runs" not in words(text).lower(), name


def test_the_readme_says_which_computers_and_which_python_the_app_runs_on():
    """What holds today, said plainly: Linux (the release's proseg-live is built for x86-64 and ARM64), macOS on Apple
    silicon and Windows x64, the last two new in the alpha and tested without a GPU; and the Python that pyproject.toml
    asks for (`requires-python`), where the Requirements list it and where a source folder makes its environment; a
    notebook's install says what else the app needs."""
    floor = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"].removeprefix(">=")
    requirements = words(section(README, "## Requirements"))
    assert "- **Linux**, 64-bit (x86-64 or ARM64)" in requirements
    assert "**macOS** 11 or newer on Apple silicon; or **Windows** 10 or 11, 64-bit (x64)" in requirements
    assert "tested on GitHub's machines, which have no GPU" in requirements
    assert f"- **Python {floor} or " in requirements
    assert "on Linux the app needs `setpriv`" in words(section(README, "## Use it from a notebook"))
    assert f"python{floor} -m venv .venv" in section(DEVELOPING, "## From a source folder")


def test_sopas_links_point_where_sopa_lives_now():
    """sopa moved to prism-oncology on GitHub: the README links it and its documentation there, and nothing that ships,
    no doc and no test points at its old home."""
    assert "[sopa](https://github.com/prism-oncology/sopa)" in README
    assert "](https://prism-oncology.github.io/sopa/getting_started/)" in README
    old_home = [re.compile(r"(?i)gustave" + "roussy")]      # built from parts, so this file never matches itself
    assert lines_matching(old_home, (*SHIPPED, *TESTS, "docs/**/*.md")) == []
