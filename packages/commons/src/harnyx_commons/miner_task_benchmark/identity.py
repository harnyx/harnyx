from __future__ import annotations

from uuid import UUID, uuid5

_BENCHMARK_RUN_NAMESPACE = UUID("50b0cf93-f838-4336-b8df-559d24daf1d0")
_BENCHMARK_BACKING_BATCH_NAMESPACE = UUID("2427dcaf-ef8c-477f-822d-d8dcf0ebec2a")
_BENCHMARK_TASK_NAMESPACE = UUID("65cec579-c4a4-4966-8d3d-d3d85bc047a8")


def benchmark_run_id_for_source_batch(
    *,
    suite_slug: str,
    source_batch_id: UUID,
    dataset_version: str,
    scoring_version: str,
    champion_artifact_id: UUID | None = None,
) -> UUID:
    identity = f"{suite_slug}:{source_batch_id}:{dataset_version}:{scoring_version}"
    if champion_artifact_id is not None:
        identity += f":{champion_artifact_id}"
    return uuid5(
        _BENCHMARK_RUN_NAMESPACE,
        identity,
    )


def benchmark_backing_batch_id_for_run(*, suite_slug: str, run_id: UUID) -> UUID:
    return uuid5(_BENCHMARK_BACKING_BATCH_NAMESPACE, f"{suite_slug}:{run_id}")


def benchmark_task_id_for_item(*, suite_slug: str, run_id: UUID, item_index: int) -> UUID:
    return uuid5(_BENCHMARK_TASK_NAMESPACE, f"{suite_slug}:{run_id}:{item_index}")


__all__ = [
    "benchmark_backing_batch_id_for_run",
    "benchmark_run_id_for_source_batch",
    "benchmark_task_id_for_item",
]
