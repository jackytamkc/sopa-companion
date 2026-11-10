"""Folders and files of this user's alone (private.py), the one place the launcher's and a notebook's opening pages are
written through: a 0700 folder of this user's, never reached through a link; a 0600 file made new, never over another
file or through a link."""
from __future__ import annotations

import os
from pathlib import Path
import stat

import pytest

from sopa_companion.private import is_mine, private_folder, write_private


def test_a_private_folder_is_this_users_alone_and_never_a_link(tmp_path):
    mine = tmp_path / "mine"
    mine.mkdir()
    mine.chmod(0o700)
    assert private_folder(mine) == mine
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o755)
    (tmp_path / "link").symlink_to(mine, target_is_directory=True)
    (tmp_path / "file").write_text("")
    for place in (shared, tmp_path / "link", tmp_path / "file", tmp_path / "missing"):
        with pytest.raises(ValueError, match="not a private folder of yours"):
            private_folder(place)


def test_a_private_folder_is_made_0700_whatever_the_umask_and_an_odd_one_is_refused(tmp_path):
    before = os.umask(0)
    try:
        made = private_folder(tmp_path / "a" / "b", create=True)
    finally:
        os.umask(before)
    assert made == tmp_path / "a" / "b" and stat.S_IMODE(made.stat().st_mode) == 0o700
    assert private_folder(made, create=True) == made                     # there already: kept
    open_one = tmp_path / "open"
    open_one.mkdir()
    open_one.chmod(0o755)
    with pytest.raises(ValueError):
        private_folder(open_one, create=True)                            # never made private behind its owner's back
    assert stat.S_IMODE(open_one.stat().st_mode) == 0o755
    (tmp_path / "elsewhere").mkdir(mode=0o700)
    (tmp_path / "pointer").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError):
        private_folder(tmp_path / "pointer", create=True)


def test_a_private_file_is_made_new_0600_and_never_through_a_link(tmp_path):
    path = write_private(tmp_path / "page.html", "<p>hello</p>")
    assert path.read_text() == "<p>hello</p>" and stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        write_private(path, "again")                                     # never over another file
    assert path.read_text() == "<p>hello</p>"
    target = tmp_path / "target.txt"
    target.write_text("theirs")
    (tmp_path / "trap.html").symlink_to(target)
    with pytest.raises(FileExistsError):
        write_private(tmp_path / "trap.html", "mine")                    # nor through a link
    assert target.read_text() == "theirs"


def test_a_private_file_that_could_not_be_written_is_removed(tmp_path):
    path = tmp_path / "page.html"
    with pytest.raises(TypeError):
        write_private(path, b"bytes are no text")                         # the write fails: nothing is left
    assert not path.exists()


def test_is_mine_looks_at_the_path_itself(tmp_path):
    (tmp_path / "file").write_text("")
    assert is_mine(tmp_path / "file") and is_mine(tmp_path)
    (tmp_path / "dangling").symlink_to(tmp_path / "nowhere")
    assert is_mine(tmp_path / "dangling")                                # the link itself, never what it leads to
    assert not is_mine(tmp_path / "missing")
    assert not is_mine(Path("/"))                                        # root's
