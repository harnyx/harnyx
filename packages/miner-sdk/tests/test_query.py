from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.query import Query, Response


def test_query_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Query.model_validate({"text": "   "})


def test_query_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Query.model_validate({"text": "hello", "other": "nope"})


def test_response_requires_non_empty_text() -> None:
    with pytest.raises(ValidationError):
        Response.model_validate({"text": ""})
