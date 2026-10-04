"""Canonical miner task generation entrypoint."""

from .contracts import (
    AgentToolSet,
    AuditResult,
    BatchGenerationResult,
    BatchTerminalGenerationError,
    CandidateStageError,
    DossierAnswer,
    FinalizedTask,
    FinalizedTaskCallback,
    MinerTaskDatasetRequest,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    StageRunResult,
)
from .dataset_builder import MinerTaskDatasetBuilder, UnderfilledDatasetError
from .source_fetch import PublicSourceFetcher, SourceFetchError
from .source_workspace import SourceDocument, SourceWorkspace

__all__ = [
    "MinerTaskDatasetRequest",
    "MinerTaskDatasetBuilder",
    "UnderfilledDatasetError",
    "BatchGenerationResult",
    "FinalizedTask",
    "FinalizedTaskCallback",
    "AuditResult",
    "BatchTerminalGenerationError",
    "CandidateStageError",
    "ProofStep",
    "ReferenceAnswerSelection",
    "ReferenceProof",
    "SourceDocument",
    "SourceWorkspace",
    "DossierAnswer",
    "SourceFetchError",
    "PublicSourceFetcher",
    "StageRunResult",
    "AgentToolSet",
]
