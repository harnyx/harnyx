from __future__ import annotations

import json

from harnyx_miner.dev import _serialize_response
from harnyx_miner_sdk.query import Response


def test_serialize_response_emits_json_object() -> None:
    payload = json.loads(_serialize_response(Response(text="hello")))

    assert payload == {"text": "hello"}
