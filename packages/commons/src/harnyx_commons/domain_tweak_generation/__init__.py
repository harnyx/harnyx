"""Source-dossier task generation."""

from harnyx_commons.domain_tweak_generation.agent_runner import (
    EFFORT,
    MODEL,
    DomainTweakAgentRunner,
)
from harnyx_commons.domain_tweak_generation.candidate_pipeline import CandidatePipeline
from harnyx_commons.domain_tweak_generation.contracts import (
    AcceptedRouteContext,
    AgentToolSet,
    AuditResult,
    BatchTerminalGenerationError,
    CandidateFailure,
    CandidateStageError,
    DomainTweakBatchGenerationResult,
    DomainTweakFinalizedTask,
    DomainTweakFinalizedTaskCallback,
    DomainTweakStageSummary,
    DossierAnswer,
    DossierFact,
    DossierRequirement,
    GroundedQuestionDossier,
    PortfolioAllocation,
    PortfolioCallCallback,
    PortfolioCallEvent,
    PortfolioPacket,
    ProofStep,
    ReferenceAnswerSelection,
    ReferenceProof,
    SlotAttemptCallback,
    SlotAttemptEvent,
    StageRunResult,
)
from harnyx_commons.domain_tweak_generation.dataset_builder import (
    DomainTweakMinerTaskDatasetBuilder,
    finalized_tasks_from_domain_tweak_result,
)
from harnyx_commons.domain_tweak_generation.refill_pipeline import ShortfallRefillPipeline
from harnyx_commons.domain_tweak_generation.source_fetch import PublicSourceFetcher, SourceFetchError
from harnyx_commons.domain_tweak_generation.source_workspace import SourceDocument, SourceWorkspace

__all__ = [
    "AgentToolSet",
    "AcceptedRouteContext",
    "AuditResult",
    "BatchTerminalGenerationError",
    "CandidateFailure",
    "CandidatePipeline",
    "CandidateStageError",
    "DossierAnswer",
    "DossierFact",
    "DossierRequirement",
    "DomainTweakAgentRunner",
    "DomainTweakBatchGenerationResult",
    "DomainTweakFinalizedTask",
    "DomainTweakFinalizedTaskCallback",
    "DomainTweakMinerTaskDatasetBuilder",
    "DomainTweakStageSummary",
    "EFFORT",
    "GroundedQuestionDossier",
    "MODEL",
    "PortfolioAllocation",
    "PortfolioCallCallback",
    "PortfolioCallEvent",
    "PortfolioPacket",
    "ProofStep",
    "PublicSourceFetcher",
    "ReferenceAnswerSelection",
    "ReferenceProof",
    "ShortfallRefillPipeline",
    "SlotAttemptCallback",
    "SlotAttemptEvent",
    "SourceDocument",
    "SourceFetchError",
    "SourceWorkspace",
    "StageRunResult",
    "finalized_tasks_from_domain_tweak_result",
]
