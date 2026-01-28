"""Background worker for processing miner-task batches."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING

from caster_validator.application.services.evaluation_batch import EvaluationBatchConfig, MinerTaskBatchService
from caster_validator.application.status import StatusProvider
from caster_validator.infrastructure.state.batch_inbox import InMemoryBatchInbox
from caster_validator.runtime.agent_artifact import create_platform_agent_resolver

if TYPE_CHECKING:
    from caster_validator.runtime.bootstrap import RuntimeContext

logger = logging.getLogger("caster_validator.evaluation_worker")


# Re-export config for convenience
EVALUATION_CONFIG = EvaluationBatchConfig()


def estimate_cycle_duration_seconds(
    *,
    uid_count: int,
    claim_count: int,
    http_timeout_floor: float = 30.0,
    bootstrap_padding_seconds: float = 2.0,
    sandbox_startup_delay_seconds: float = 2.0,
    sandbox_wait_for_healthz: bool = True,
) -> float:
    """Return a conservative duration estimate for a full evaluation cycle."""
    if uid_count <= 0 or claim_count <= 0:
        return 0.0

    startup = max(0.0, sandbox_startup_delay_seconds)
    if sandbox_wait_for_healthz:
        startup += 15.0
    startup += bootstrap_padding_seconds

    per_evaluation = http_timeout_floor

    return uid_count * startup + uid_count * claim_count * per_evaluation


class EvaluationWorker:
    """Async evaluation worker that drains the inbox and processes batches.

    This worker runs as an asyncio task and is intended to execute on the same
    event loop as the validator server so shared async clients (LLM providers,
    HTTP pools, semaphores) remain single-loop.
    """

    worker_name = "validator-evaluation-worker"

    def __init__(
        self,
        *,
        batch_service: MinerTaskBatchService,
        batch_inbox: InMemoryBatchInbox,
        status_provider: StatusProvider | None = None,
    ) -> None:
        self._batch_service = batch_service
        self._inbox = batch_inbox
        self._status = status_provider
        self._stop = threading.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Start the evaluation task (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name=self.worker_name)

    async def stop(self, *, timeout: float = 5.0) -> None:
        """Request stop and wait for termination."""
        task = self._task
        if task is None:
            return
        self._stop.set()
        self._inbox.wake()
        try:
            await asyncio.wait_for(task, timeout=timeout)
        finally:
            self._task = None

    @property
    def running(self) -> bool:
        task = self._task
        return bool(task is not None and not task.done())

    async def _run(self) -> None:
        while not self._stop.is_set():
            batch = await asyncio.to_thread(self._inbox.get, stop_event=self._stop)
            if batch is None:
                continue

            if self._status is not None:
                self._status.state.queued_batches = len(self._inbox)

            try:
                await self._batch_service.process_async(batch)
            except Exception as exc:
                logger.exception(
                    "batch processing failed",
                    extra={"batch_id": str(batch.batch_id)},
                )
                if self._status is not None:
                    self._status.state.last_error = str(exc)
                    self._status.state.running = False


def create_evaluation_worker(
    *,
    batch_service: MinerTaskBatchService,
    batch_inbox: InMemoryBatchInbox,
    status_provider: StatusProvider | None = None,
) -> EvaluationWorker:
    """Factory function to create an EvaluationWorker with injected dependencies."""
    return EvaluationWorker(
        batch_service=batch_service,
        batch_inbox=batch_inbox,
        status_provider=status_provider,
    )


def create_evaluation_worker_from_context(context: RuntimeContext) -> EvaluationWorker:
    """Create an EvaluationWorker wired up from a RuntimeContext.

    This is a convenience factory for the common case where all dependencies
    come from a single RuntimeContext.
    """
    if context.platform_client is None:
        raise RuntimeError("platform client is not configured")

    agent_resolver = create_platform_agent_resolver(context.platform_client)
    batch_service = MinerTaskBatchService(
        platform_client=context.platform_client,
        subtensor_client=context.subtensor_client,
        sandbox_manager=context.sandbox_manager,
        session_manager=context.session_manager,
        evaluation_records=context.evaluation_records,
        orchestrator_factory=context.create_evaluation_orchestrator,
        sandbox_options_factory=context.build_sandbox_options,
        agent_resolver=agent_resolver,
        status_provider=context.status_provider,
        progress=context.progress_tracker,
    )
    return EvaluationWorker(
        batch_service=batch_service,
        batch_inbox=context.batch_inbox,
        status_provider=context.status_provider,
    )


__all__ = [
    "EVALUATION_CONFIG",
    "EvaluationWorker",
    "create_evaluation_worker",
    "create_evaluation_worker_from_context",
    "estimate_cycle_duration_seconds",
]
