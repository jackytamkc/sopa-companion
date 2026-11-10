"""API mode's runs: a save goes to the notebook's object too. The
helper writes the object's zarr when it has one (and the engine's view when that is not it), stages the built elements
for the notebook, and the engine hands them over (deliver): queued until the notebook's main thread applied them.
Names never clash with the object's. Closing follows the session's answer: decided runs are completed, a Finish & save
is dropped once the kernel has gone, and what was not added is noted for the final summary. The session is a fake with
the session interface; the object's view is a double with a save's arguments."""
from pathlib import Path
import threading

import pytest

from conftest import wait_for
from test_runs_save import ended_run, adopt
from sopa_companion.api import protocol
from sopa_companion.engine.runs import NO_CELLS, RunError, Runs, leave_kind
from sopa_companion.engine.samples import Samples
from sopa_companion.engine.zarr_samples import ZarrSamples


class FakeSession:
    """Session as engine/session.py gives it, the notebook's acknowledgements played by the test (receive, apply)."""
    mode = "api"

    def __init__(self, view_path, target=None, names=(), view="in_place"):
        self.bound, self.view_path, self.notebook_busy = "zs", str(view_path), False
        self._target, self._names, self.view = target, set(names), view
        self.kernel_gone = False
        self.events, self.not_added = [], []
        self.acks, self.applies = {}, {}
        self.condition = threading.Condition()

    def public(self):
        return {"mode": "api", "sample": {"view": self.view}}

    def target(self):
        return self._target

    def names(self):
        return set(self._names)

    def deliver(self, event):
        protocol.validate_event(event)
        self.events.append(event)

    def received(self, id, timeout):
        with self.condition:
            self.condition.wait_for(lambda: id in self.acks or self.kernel_gone, timeout)
            return self.acks.get(id, False)

    def applied(self, id):
        return self.applies.get(id)

    def closing_policy(self):
        return None

    def note_not_added(self, run, reason, message):
        protocol.build_event("ended", added=[], not_added=[{"run": run, "reason": reason, "message": message}])
        self.not_added.append((run, reason, message))

    # the notebook's side
    def receive(self, id, ok=True):
        with self.condition:
            self.acks[id] = ok
            self.condition.notify_all()

    def apply(self, id, state="added", clash=(), message=None):
        self.applies[id] = {"op": "applied", "id": id, "state": state, "keys": [], "clash": list(clash),
                            "message": message}

    def gone(self):
        with self.condition:
            self.kernel_gone = True
            self.condition.notify_all()


class Target:
    """The engine's view of the notebook's object (sid "zs"), as Runs asks it: its element names, which saves add to, and
    its saves (kept here, never written), which take API arguments and stage what they built where they are told."""
    save_defaults = ZarrSamples.save_defaults

    def __init__(self, root):
        self.root = str(root)
        Path(root).mkdir(exist_ok=True)
        self.existing, self.saved, self.delivers = ["transcripts"], [], []
        self.view_error = None                       # the helper's words when the view's write failed

    def path(self, sid):
        return self.root

    def known(self, sid):
        return sid == "zs"

    def open(self, path, recent=True):
        return {"id": "zs"}

    def existing_elements(self, sid):
        return sorted(self.existing)

    def save(self, sid, output, shapes_key, table_key, set_default, overwrite, provenance, deliver=None, **kwargs):
        self.saved.append({"shapes_key": shapes_key, "table_key": table_key, "set_default": set_default,
                           "provenance": provenance, **kwargs})
        self.delivers.append(deliver)
        result = {"cells": 6}
        if deliver is not None:
            Path(deliver["stage_to"]).mkdir(parents=True)
            result.update(staged=deliver["stage_to"], written_to=deliver["path"], genes=3, replaced=[],
                          view_written=deliver["view"] is not None and self.view_error is None, view_error=self.view_error)
            if deliver["path"] == str(Path(self.root).resolve()) or deliver["view"]:
                self.existing += [key for key in (shapes_key, table_key) if key and key not in self.existing]
        return result


def target(tmp_path) -> Target:
    return Target(tmp_path / "view.zarr")


def api_runs(config, tmp_path, **session):
    zarr = target(tmp_path)
    fake = FakeSession(zarr.root, **session)
    return Runs(config, Samples(config), zarr, session=fake), zarr, fake


def delivered(fake, count=1):
    wait_for(lambda: len(fake.events) >= count)
    return fake.events[count - 1]


def test_an_add_in_place_writes_the_objects_zarr_and_hands_over_the_staged_result(config, tmp_path):
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="in_place")
    fake._target = zarr.root                                      # in place: the object's zarr is the view
    run = adopt(runs, ended_run(config, "added"))
    runs.save_to_zarr("added", "proseg_boundaries", "proseg_table", set_default=True)
    event = delivered(fake)
    where = zarr.delivers[-1]
    assert where["path"] == str(Path(zarr.root).resolve()) and where["view"] is None
    assert where["frame_from"] == str(Path(zarr.root).resolve()) and where["consolidate"] is True
    staged = Path(where["stage_to"])
    assert staged.parent.parent == runs.deliveries and staged.name == "staged.zarr" and staged.exists()
    assert (event["run"], event["kind"], event["shapes_key"], event["table_key"]) == ("added", "proseg", "proseg_boundaries",
                                                                                       "proseg_table")
    assert event["set_default"] is True and event["cells"] == 6 and event["genes"] == 3
    assert event["staged"] == str(staged) and event["written_to"] == where["path"]
    assert event["provenance"]["run"] == "added" and event["provenance"]["session"] == "api"
    wait_for(lambda: run.meta["zarr_save"]["state"] == "queued")
    assert run.meta["zarr_save"]["written_to"] == where["path"]
    assert leave_kind(run.public())["kind"] == "saved"           # in a zarr: leaving needs no wait for the notebook
    fake.receive(event["id"])
    wait_for(lambda: not staged.parent.exists())
    fake.apply(event["id"])
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    assert run.meta["zarr_save"]["added"] is True and run.meta["zarr_save"]["message"] == \
        "Added 6 cells as proseg_boundaries + proseg_table"
    protocol.validate_zarr_save(run.meta["zarr_save"])
    runs.close("discard")


def test_a_session_that_cannot_say_where_its_sample_lies_never_has_that_store_written(config, tmp_path):
    """Writing off (no target) and the session unable to say how its sample is opened: the engine's zarr may be the
    object's own store opened in place, so it is never handed to the helper as a view to write (a session
    whose description failed sent the user's own zarr as the view)."""
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="in_place")
    where = runs._delivery_where(fake.bound, tmp_path / "staged")
    assert where["path"] is None and where["view"] is None

    def broken():
        raise RuntimeError("the session could not describe itself")
    fake.public = broken
    where = runs._delivery_where(fake.bound, tmp_path / "staged")
    assert where["path"] is None and where["view"] is None
    assert where["frame_from"] == str(Path(zarr.root).resolve())          # still framed by it, never written
    fake.public = lambda: {"mode": "api", "sample": {"view": "overlay"}}  # an overlay's view is the engine's own
    assert runs._delivery_where(fake.bound, tmp_path / "staged")["view"] == str(Path(zarr.root).resolve())
    runs.close("discard")


def test_an_overlay_writes_the_target_and_the_view_and_its_names_never_clash(config, tmp_path):
    """The names a save must not take are the view's, the object's (as the handle pushed them) and its
    zarr's; an overlay's view is written beside the target, unconsolidated by the helper."""
    other = tmp_path / "object.zarr"
    (other / "tables" / "proseg_table_2").mkdir(parents=True)
    runs, zarr, fake = api_runs(config, tmp_path, target=str(other), names=["proseg_boundaries"], view="overlay")
    run = adopt(runs, ended_run(config, "named"))
    offered = runs.current_save_defaults("named")
    assert (offered["shapes_key"], offered["table_key"]) == ("proseg_boundaries_3", "proseg_table_3")
    assert {"proseg_boundaries", "proseg_table_2"} <= set(offered["existing"])
    with pytest.raises(RunError) as clash:
        runs.save_to_zarr("named", "proseg_boundaries", "t")
    assert clash.value.status == 409 and clash.value.extra["clash"] == ["proseg_boundaries"]
    runs.save_to_zarr("named", offered["shapes_key"], offered["table_key"])
    delivered(fake)
    where = zarr.delivers[-1]
    assert where["path"] == str(other.resolve()) and where["view"] == str(Path(zarr.root).resolve())
    wait_for(lambda: run.meta["zarr_save"]["state"] == "queued")
    assert run.meta["zarr_save"]["view_written"] is True and run.meta["zarr_save"]["view_error"] is None
    runs.close("discard")


def test_a_view_that_could_not_take_the_result_is_said_and_the_add_goes_on(config, tmp_path):
    """A failure in the view's transaction alone does not fail the save: the result is in the object's zarr and goes
    to the object; the run says the app's layers do not show it (view_written false, with the helper's words)."""
    other = tmp_path / "object.zarr"
    other.mkdir()
    runs, zarr, fake = api_runs(config, tmp_path, target=str(other), view="overlay")
    zarr.view_error = "No space left on device"
    run = adopt(runs, ended_run(config, "unseen"))
    runs.save_to_zarr("unseen", "proseg_boundaries", "proseg_table")
    event = delivered(fake)
    wait_for(lambda: run.meta["zarr_save"]["state"] == "queued")
    assert run.meta["zarr_save"]["view_written"] is False
    assert run.meta["zarr_save"]["view_error"] == "No space left on device"
    fake.receive(event["id"])
    fake.apply(event["id"])
    wait_for(lambda: run.meta["zarr_save"]["state"] == "saved")
    assert run.meta["zarr_save"]["view_written"] is False                  # kept with the saved state
    runs.close("discard")


def test_a_memory_only_add_leaves_once_applied_and_a_clash_there_keeps_the_run(config, tmp_path):
    """No target (an unbacked object, or auto-save off): the result reaches memory only, so leaving with Add waits until
    the notebook applied it. A name made in the notebook after the add (a clash at apply) is never retried: the run
    stays, to be asked again."""
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="copy")
    run = adopt(runs, ended_run(config, "memory"))
    runs.leave("memory", "add")
    first = delivered(fake)
    assert zarr.delivers[-1]["path"] is None and first["written_to"] is None
    wait_for(lambda: (run.meta["zarr_save"] or {}).get("state") == "queued")
    assert leave_kind(run.public())["kind"] == "saving" and "memory" in runs.runs
    fake.receive(first["id"])
    fake.apply(first["id"], "failed", clash=["proseg_boundaries"])
    wait_for(lambda: run.meta.get("leaving") is None and run.meta["zarr_save"]["state"] == "failed")
    assert run.meta["zarr_save"]["clash"] == ["proseg_boundaries"] and len(fake.events) == 1
    assert "memory" in runs.runs
    fake._names = {"proseg_boundaries"}
    runs.leave("memory", "add")
    second = delivered(fake, 2)
    assert second["shapes_key"] == "proseg_boundaries_2"
    fake.receive(second["id"])
    fake.apply(second["id"])
    wait_for(lambda: "memory" not in runs.runs)
    runs.close("discard")


def acknowledge(fake, stop):
    """The notebook's io thread and main thread: every delivery received and applied as it comes."""
    seen = set()
    while not stop.is_set():
        for event in list(fake.events):
            if event["id"] not in seen:
                seen.add(event["id"])
                fake.receive(event["id"])
                fake.apply(event["id"])
        stop.wait(0.02)


def test_closing_the_session_completes_decided_runs_and_notes_the_rest(config, tmp_path):
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="copy")
    adopt(runs, ended_run(config, "unanswered"))
    adopt(runs, ended_run(config, "finish-and-save", finish_requested=True))
    adopt(runs, ended_run(config, "nothing", state="stopped"))
    stop = threading.Event()
    notebook = threading.Thread(target=acknowledge, args=(fake, stop))
    notebook.start()
    try:
        runs.close_session("discard")
    finally:
        stop.set()
        notebook.join(5)
    assert [e["run"] for e in fake.events] == ["finish-and-save"]
    assert fake.not_added == [("unanswered", "discarded", "Discarded as the session closed")]
    runs.close(policy="discard")
    assert not runs.runs


def test_closing_the_session_with_add_adds_and_waits_for_the_notebook_to_receive(config, tmp_path):
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="copy")
    run = adopt(runs, ended_run(config, "to-add"))
    stop = threading.Event()
    notebook = threading.Thread(target=acknowledge, args=(fake, stop))
    notebook.start()
    try:
        outcomes = runs.close_session("add")
    finally:
        stop.set()
        notebook.join(5)
    assert [e["run"] for e in fake.events] == ["to-add"] and fake.not_added == []
    assert outcomes[0]["answer"] == "add" and outcomes[0]["left"] is True
    assert run.meta["zarr_save"].get("received") is True and not runs.runs
    runs.close(policy="discard")


def leaving_while_followed(config, tmp_path, monkeypatch):
    """A run whose add is in a zarr and received by the notebook, its delivery's follower waiting for the notebook to
    apply it. While its folder is being removed (`removing` names it once the removal started; the test plays
    `_remove`), rmtree takes away what is written there: such a write fails (FileNotFoundError, the folder still
    there) and is noted in `written`; an unhandled exception in a thread lands in `failed`."""
    from sopa_companion.engine import runs as R
    other = tmp_path / "object.zarr"
    other.mkdir()
    runs, zarr, fake = api_runs(config, tmp_path, target=str(other), view="overlay")
    run = adopt(runs, ended_run(config, "left"))
    runs.save_to_zarr("left", "proseg_boundaries", "proseg_table")
    event = delivered(fake)
    fake.receive(event["id"])
    wait_for(lambda: run.meta["zarr_save"].get("received") is True)
    removing, written, failed = [], [], []
    write = R.atomic_bytes

    def emptied(path, data, durable=False):                   # rmtree takes what is written there away meanwhile
        if Path(path).parent == run.dir and removing:
            written.append(Path(path).name)
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return write(path, data, durable)
    monkeypatch.setattr(R, "atomic_bytes", emptied)
    monkeypatch.setattr(threading, "excepthook", failed.append)
    return runs, fake, run, event, removing, written, failed


def test_a_run_that_left_while_its_delivery_was_followed_ends_the_follower_quietly(config, tmp_path, monkeypatch):
    """A run leaves at queued once its add is in a zarr (and, while the session closes, once the notebook received it)
    while its delivery's follower still waits for the notebook to apply it. Its folder is removed after the engine let
    go of it, by rmtree, which takes away a run.json being written while the folder is still there: the follower's
    save failed with FileNotFoundError, raised again as the folder was there, an unhandled exception in its thread (the
    session-closing test above logged it). It writes nothing more there and ends quietly."""
    runs, fake, run, event, removing, written, failed = leaving_while_followed(config, tmp_path, monkeypatch)
    monkeypatch.setattr(runs, "_remove", removing.append)     # its folder's removal under way, not over
    runs.delete("left")
    assert removing == [run.dir] and "left" not in runs.runs
    fake.apply(event["id"])
    run.follower.join(5)
    assert not run.follower.is_alive() and written == [] and failed == []
    runs.close("discard")


def test_a_deleted_run_is_marked_so_before_its_folder_goes(config, tmp_path, monkeypatch):
    """The notebook applies the delivery while the run's folder is being removed: the follower, waiting for that, finds
    the run marked deleted already (_delete_run marks it before the removal starts) and writes nothing there."""
    runs, fake, run, event, removing, written, failed = leaving_while_followed(config, tmp_path, monkeypatch)

    def remove(folder):                                       # the notebook applies while the folder is removed
        removing.append(folder)
        fake.apply(event["id"])
        run.follower.join(5)
    monkeypatch.setattr(runs, "_remove", remove)
    runs.delete("left")
    assert removing == [run.dir] and not run.follower.is_alive()
    assert written == [] and failed == [], [f"{a.exc_type.__name__}: {a.exc_value}" for a in failed]
    runs.close("discard")


def test_a_save_s_update_looks_at_the_mark_before_it_writes(config, tmp_path, monkeypatch):
    """The run is deleted just after its follower looked at it (it read the notebook's answer, its update still to
    write): the update looks at the mark before it writes, and writes nothing into the folder being removed."""
    runs, fake, run, event, removing, written, failed = leaving_while_followed(config, tmp_path, monkeypatch)
    monkeypatch.setattr(runs, "_remove", removing.append)
    applied, left = fake.applied, []

    def applying(id):                                         # the run leaves as the follower reads the answer
        answer = applied(id)
        if answer is not None and not left:
            left.append(id)
            runs.delete("left")
        return answer
    fake.applied = applying
    fake.apply(event["id"])
    run.follower.join(5)
    assert left and removing == [run.dir] and not run.follower.is_alive()
    assert written == [] and failed == [], [f"{a.exc_type.__name__}: {a.exc_value}" for a in failed]
    runs.close("discard")


def test_closing_the_session_with_add_discards_a_run_with_nothing_to_save_and_says_why(config, tmp_path):
    """An add of a run that cannot be saved (its filters leave no cells) is carried out as a discard;
    the final summary says it was not added, and why."""
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="copy")
    adopt(runs, ended_run(config, "filtered", "stain", summary={"cells": 0}, summary_model={"cells": 4}))
    outcomes = runs.close_session("add")
    assert fake.events == [] and zarr.saved == []
    assert fake.not_added == [("filtered", "failed", NO_CELLS)]
    assert outcomes[0]["answer"] == "add" and outcomes[0]["left"] is True and not runs.runs
    runs.close(policy="discard")


def test_a_save_under_way_that_fails_as_the_session_closes_is_noted_as_failed(config, tmp_path, monkeypatch):
    """A save pressed before the session closed is decided: waited for. When it fails, the final summary says so (the
    run held a result it was "saving", not one it was asked about)."""
    runs, zarr, fake = api_runs(config, tmp_path, target=None, view="copy")
    adopt(runs, ended_run(config, "failing"))
    entered, go = threading.Event(), threading.Event()
    def save(*args, **kwargs):
        entered.set()
        assert go.wait(5)
        raise OSError("disk full")
    monkeypatch.setattr(zarr, "save", save)
    runs.save_to_zarr("failing", "proseg_boundaries", "proseg_table")
    assert entered.wait(5)
    closer = threading.Thread(target=runs.close_session, args=("discard",))
    closer.start()
    wait_for(lambda: runs.closing)
    assert closer.is_alive()                                      # the save under way is waited for
    go.set()
    closer.join(10)
    assert not closer.is_alive() and fake.events == []
    assert fake.not_added == [("failing", "failed", "disk full")]
    runs.close(policy="discard")
    assert not runs.runs


def test_once_the_kernel_has_gone_a_finish_and_save_is_dropped_and_adds_reach_no_memory(config, tmp_path):
    """The kernel gone: a Finish & save is dropped (nothing can reach the notebook); an add already under way
    completes on disk and is said not to have reached it."""
    runs, zarr, fake = api_runs(config, tmp_path, target=str(tmp_path / "object.zarr"), view="overlay")
    (tmp_path / "object.zarr").mkdir()
    adopt(runs, ended_run(config, "finish-and-save", finish_requested=True))
    added = adopt(runs, ended_run(config, "added-before"))
    runs.save_to_zarr("added-before", "proseg_boundaries", "proseg_table")
    delivered(fake)
    fake.gone()
    wait_for(lambda: added.meta["zarr_save"]["state"] == "failed")
    assert added.meta["zarr_save"]["message"] == "Your notebook did not receive it: it is in object.zarr only"
    assert [r["decided"] for r in runs.pending()] == [False, False]
    runs.close_session("discard")
    assert len(fake.events) == 1               # the delivery made before is the session's to report (session.finish)
    assert fake.not_added == [("finish-and-save", "discarded", "Discarded as the session closed")]
    runs.close(policy="discard")
    assert not runs.runs
