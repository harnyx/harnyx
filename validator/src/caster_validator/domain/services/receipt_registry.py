"""Receipt bookkeeping for miner citations."""

from __future__ import annotations

from collections.abc import Iterable

from caster_commons.domain.tool_call import (
    ToolCall,
    ToolResultPolicy,
)
from caster_commons.tools.types import is_citation_source
from caster_validator.domain.evaluation import MinerCitation
from caster_validator.domain.exceptions import InvalidCitationError


class ReceiptRegistry:
    """In-memory registry of tool call receipts."""

    def __init__(self) -> None:
        self._receipts: dict[str, ToolCall] = {}

    def record(self, receipt: ToolCall) -> None:
        """Insert or update a receipt."""
        self._receipts[receipt.receipt_id] = receipt

    def lookup(self, receipt_id: str) -> ToolCall | None:
        """Return the stored receipt for the identifier, if any."""
        return self._receipts.get(receipt_id)

    def validate_citations(self, citations: Iterable[MinerCitation]) -> None:
        """Ensure every citation maps to a successful receipt."""
        invalid: list[str] = []
        for citation in citations:
            receipt = self._receipts.get(citation.receipt_id)
            if receipt is None or not receipt.is_successful():
                invalid.append(citation.receipt_id)
                continue
            if not is_citation_source(receipt.tool):
                invalid.append(citation.receipt_id)
                continue
            if receipt.metadata.result_policy is not ToolResultPolicy.REFERENCEABLE:
                invalid.append(citation.receipt_id)
                continue
            result = next(
                (res for res in receipt.metadata.results if res.result_id == citation.result_id),
                None,
            )
            if result is None:
                invalid.append(citation.receipt_id)
        if invalid:
            raise InvalidCitationError(f"unknown or failed receipt references: {invalid}")

    def values(self) -> tuple[ToolCall, ...]:
        """Return a snapshot of tracked receipts."""
        return tuple(self._receipts.values())
