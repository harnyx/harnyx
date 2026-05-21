from __future__ import annotations

from uuid import uuid4

from harnyx_validator.infrastructure.state.restore_inbox import InMemoryRestoreInbox


def test_restore_inbox_does_not_block_ready_items_behind_delayed_retry() -> None:
    inbox = InMemoryRestoreInbox()
    delayed_batch_id = uuid4()
    ready_batch_id = uuid4()

    assert inbox.release_for_retry(delayed_batch_id, after_seconds=60.0) is True
    assert inbox.put_once(ready_batch_id) is True

    assert inbox.get(timeout=0.01) == ready_batch_id
