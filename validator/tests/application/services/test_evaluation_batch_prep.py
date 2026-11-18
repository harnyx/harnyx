from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from harnyx_commons.sandbox.options import SandboxOptions
from harnyx_validator.application.dto.evaluation import ScriptArtifactSpec
from harnyx_validator.application.services.evaluation_batch_prep import (
    BatchExecutionPlanner,
    EvaluationBatchConfig,
    RunContext,
)


def test_validator_sandbox_mounts_only_selected_artifact_subpath(
    monkeypatch,
) -> None:
    state_dir = Path("/workspace/.harnyx_state")
    artifact_id = uuid4()
    staged = SimpleNamespace(
        host_path=state_dir / "evaluation_agents" / artifact_id.hex / "agent.py",
        container_path=f"/workspace/.harnyx_state/evaluation_agents/{artifact_id.hex}/agent.py",
    )
    planner = object.__new__(BatchExecutionPlanner)
    planner._agent_resolver = lambda *_args: staged
    monkeypatch.setenv("STATE_VOLUME_NAME", "validator-state")
    run_ctx = RunContext(
        batch_id=uuid4(),
        tasks=(),
        config=EvaluationBatchConfig(state_dir=str(state_dir)),
        base_options=SandboxOptions(
            image="sandbox",
            container_name="base",
            volumes=(("/trusted", "/trusted", "ro"),),
        ),
        base_env={},
        base_volumes=(("/trusted", "/trusted", "ro"),),
        state_dir=state_dir,
    )
    artifact = ScriptArtifactSpec(uid=7, artifact_id=artifact_id, content_hash="hash", size_bytes=1)

    options = planner._build_sandbox_options_factory(run_ctx)(artifact)

    assert options.volumes == (("/trusted", "/trusted", "ro"),)
    assert len(options.volume_subpath_mounts) == 1
    mount = options.volume_subpath_mounts[0]
    assert mount.source == "validator-state"
    assert mount.subpath == f"evaluation_agents/{artifact_id.hex}"
    assert mount.target == f"/workspace/.harnyx_state/evaluation_agents/{artifact_id.hex}"
