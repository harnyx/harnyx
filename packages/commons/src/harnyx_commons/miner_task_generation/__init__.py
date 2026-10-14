"""Canonical miner task generation entrypoint."""

from .contracts import (
    BatchGenerationResult,
    BatchTerminalGenerationError,
    CandidateStageError,
    FinalizedTask,
    FinalizedTaskCallback,
    MinerTaskDatasetRequest,
)
from .dataset_builder import MinerTaskDatasetBuilder, UnderfilledDatasetError

__all__ = [
    "MinerTaskDatasetRequest",
    "MinerTaskDatasetBuilder",
    "UnderfilledDatasetError",
    "BatchGenerationResult",
    "FinalizedTask",
    "FinalizedTaskCallback",
    "BatchTerminalGenerationError",
    "CandidateStageError",
]
