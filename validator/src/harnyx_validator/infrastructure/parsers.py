"""Shared parsing helpers for validator infrastructure."""

from __future__ import annotations

import json
from collections.abc import Mapping

from pydantic import TypeAdapter

from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec

_BATCH_PAYLOAD_ADAPTER = TypeAdapter(MinerTaskBatchSpec)
_PLATFORM_RESPONSE_ONLY_BATCH_KEYS = frozenset({"champion_artifact_id", "completed_at", "failed_at"})


def parse_batch(payload: Mapping[str, object]) -> MinerTaskBatchSpec:
    """Normalize raw batch payloads into MinerTaskBatchSpec."""
    normalized_payload = {
        key: value for key, value in payload.items() if key not in _PLATFORM_RESPONSE_ONLY_BATCH_KEYS
    }
    return _BATCH_PAYLOAD_ADAPTER.validate_json(json.dumps(normalized_payload), strict=True)


__all__ = ["parse_batch"]
