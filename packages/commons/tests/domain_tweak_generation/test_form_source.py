from uuid import UUID

import pytest

from harnyx_commons.domain_tweak_generation import DeepSearchQAFormSource


def test_form_source_loads_every_problem_without_answer_fields() -> None:
    """Future failure: production selection must not regress to a hand-picked form cohort."""
    source = DeepSearchQAFormSource()

    assert len(source.forms) == 900
    assert tuple(item.source_index for item in source.forms) == tuple(range(900))
    assert all(set(item.model_dump()) == {"form_identity", "source_index", "form"} for item in source.forms)


def test_cursor_is_batch_deterministic_and_selects_without_replacement() -> None:
    """Future failure: one request must not finalize duplicate forms or use snapshot order as policy."""
    source = DeepSearchQAFormSource()
    batch_id = UUID("00000000-0000-0000-0000-000000000123")

    first = source.cursor(batch_id, target_count=10).next_inputs(10)
    repeated = source.cursor(batch_id, target_count=10).next_inputs(10)
    other = source.cursor(UUID(int=456), target_count=10).next_inputs(10)

    assert first == repeated
    assert len({item.form_identity for item in first}) == 10
    assert tuple(item.source_index for item in first) != tuple(range(10))
    assert first != other


def test_cursor_revisits_failed_forms_only_after_unseen_pool_is_exhausted() -> None:
    """Future failure: refill must explore unseen forms before retrying failed ones."""
    source = DeepSearchQAFormSource()
    cursor = source.cursor(UUID(int=789), target_count=2)

    first = cursor.next_inputs(1)[0]
    rest = cursor.next_inputs(899)
    retried = cursor.next_inputs(1)[0]

    assert first.form_identity not in {item.form_identity for item in rest}
    assert len({item.form_identity for item in rest}) == 899
    assert retried.form_identity in {item.form_identity for item in (*rest, first)}


def test_cursor_never_returns_finalized_forms() -> None:
    source = DeepSearchQAFormSource()
    cursor = source.cursor(UUID(int=987), target_count=2)
    finalized = cursor.next_inputs(1)[0]

    subsequent = cursor.next_inputs(899, excluding=frozenset({finalized.form_identity}))

    assert finalized.form_identity not in {item.form_identity for item in subsequent}


def test_source_rejects_target_larger_than_snapshot_before_generation() -> None:
    source = DeepSearchQAFormSource()

    with pytest.raises(ValueError, match="eligible DeepSearchQA form pool"):
        source.cursor(UUID(int=0), target_count=901)
