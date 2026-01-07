from __future__ import annotations

from datetime import UTC, datetime

from caster_validator.domain.agent import AgentRegistry, AgentStatus, SandboxSpec, ToolDescriptor


def test_agent_registry_mark_synced_updates_metadata() -> None:
    original = AgentRegistry(
        uid=7,
        gist_id="abc123",
        gist_file="main.py",
        gist_commit_sha="deadbeef",
        runtime_image="caster/validator:0.1.0",
        last_synced_block=100,
        last_synced_at=datetime(2025, 10, 10, tzinfo=UTC),
        sandbox=SandboxSpec(
            runtime_image="caster/sandbox:0.1.0",
            sdk_version="0.1.0",
                tools=(ToolDescriptor(name="search_web", description="Search"),),
        ),
    )

    updated = original.mark_synced(
        gist_commit_sha="feedface",
        last_synced_block=101,
        last_synced_at=datetime(2025, 10, 12, tzinfo=UTC),
    )

    assert updated.status is AgentStatus.ACTIVE
    assert updated.sync_error is None
    assert updated.gist_commit_sha == "feedface"
    assert updated.last_synced_block == 101
    assert updated.last_synced_at == datetime(2025, 10, 12, tzinfo=UTC)
    assert updated.sandbox == original.sandbox


def test_agent_registry_mark_error_sets_status() -> None:
    record = AgentRegistry(
        uid=42,
        gist_id="gist",
        gist_file="agent.py",
        gist_commit_sha="deadbeef",
        runtime_image="caster/validator:0.1.0",
        last_synced_block=5,
        last_synced_at=datetime(2025, 10, 11, tzinfo=UTC),
    )

    errored = record.mark_error("failed checksum", at=datetime(2025, 10, 13, tzinfo=UTC))

    assert errored.status is AgentStatus.ERRORED
    assert errored.sync_error == "failed checksum"
    assert errored.last_synced_at == datetime(2025, 10, 13, tzinfo=UTC)


def test_agent_registry_disable_preserves_sync_metadata() -> None:
    record = AgentRegistry(
        uid=1,
        gist_id="gist",
        gist_file="agent.py",
        gist_commit_sha="abc",
        runtime_image="caster/validator:0.1.0",
        last_synced_block=5,
        last_synced_at=datetime(2025, 10, 11, tzinfo=UTC),
        sync_error=None,
    )

    disabled = record.disable(
        reason="operator request",
        at=datetime(2025, 10, 14, tzinfo=UTC),
    )

    assert disabled.status is AgentStatus.DISABLED
    assert disabled.sync_error == "operator request"
    assert disabled.last_synced_at == datetime(2025, 10, 14, tzinfo=UTC)
