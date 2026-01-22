from __future__ import annotations

import pytest

from caster_commons.context.snapshot import ContextSnapshot


def test_context_snapshot_immutable_view() -> None:
    snapshot = ContextSnapshot({"foo": "bar"})
    assert snapshot["foo"] == "bar"
    assert dict(snapshot) == {"foo": "bar"}

    with pytest.raises(KeyError):
        _ = snapshot["missing"]


def test_context_snapshot_to_dict_returns_copy() -> None:
    data = {"nested": {"value": 42}}
    snapshot = ContextSnapshot(data)
    clone = snapshot.to_dict()
    assert clone == data
    assert clone is not data
