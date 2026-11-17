"""Protect partial cleanup and bounded resource use without Docker."""

from pathlib import Path
from threading import Event

from harnyx_sandbox.sandbox.workspace import QueryWorkspaces


def test_partial_delete_continues_without_following_links(tmp_path, monkeypatch):
    import os

    root = tmp_path / "queries"
    root.mkdir()
    workspaces = QueryWorkspaces(root)
    path = Path(workspaces.prepare())
    retained = path / "busy"
    retained.write_text("keep")
    (path / "removable").write_text("remove")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "preserved").write_text("keep")
    (path / "link").symlink_to(outside, target_is_directory=True)
    original = os.unlink

    def fail_one(name, *args, **kwargs):
        if name == "busy":
            raise PermissionError("synthetic deletion failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_one)
    try:
        workspaces._remove(str(path))
        assert retained.exists()
        assert not (path / "removable").exists()
        assert not (path / "link").exists()
        assert (outside / "preserved").read_text() == "keep"
    finally:
        workspaces.close()


def test_cleanup_queue_does_not_grow_when_deletion_blocks(tmp_path, monkeypatch):
    workspaces = QueryWorkspaces(tmp_path)
    started, release = Event(), Event()
    processed = []

    def blocked(path):
        processed.append(path)
        started.set()
        release.wait(3)

    monkeypatch.setattr(workspaces, "_remove", blocked)
    try:
        workspaces.remove_later("first")
        assert started.wait(1)
        for index in range(30):
            workspaces.remove_later(str(index))
        # One running job and nine queued jobs; later submissions remain on disk.
        assert workspaces._executor._work_queue.qsize() == 9
        workspaces.close()
        assert processed == ["first"]
    finally:
        release.set()
        workspaces.close()
