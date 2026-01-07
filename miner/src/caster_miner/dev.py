from __future__ import annotations

import argparse
import asyncio
import json
import runpy
from collections.abc import Sequence
from pathlib import Path

from caster_commons.tools.local_dev_host import create_local_tool_host
from caster_miner_sdk._internal.tool_invoker import bind_tool_invoker
from caster_miner_sdk.criterion_evaluation import CriterionEvaluationRequest
from caster_miner_sdk.decorators import clear_entrypoints, entrypoint_exists, get_entrypoint

_DEFAULT_RUBRIC_TITLE = "Accuracy"
_DEFAULT_RUBRIC_DESCRIPTION = "Judge whether the claim is factually correct."
_DEFAULT_CLAIM_TEXT = "Caster Subnet validators manage sandboxed miners."


def _load_agent(agent_path: Path) -> None:
    clear_entrypoints()
    runpy.run_path(str(agent_path))
    if not entrypoint_exists("evaluate_criterion"):
        raise RuntimeError("agent did not register entrypoint 'evaluate_criterion'")


def _build_request(*, request_path: Path | None, claim_text: str | None) -> dict[str, object]:
    if request_path is not None:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
        payload = CriterionEvaluationRequest.model_validate(raw)
        return payload.model_dump()

    raw = {
        "claim_text": claim_text or _DEFAULT_CLAIM_TEXT,
        "rubric_title": _DEFAULT_RUBRIC_TITLE,
        "rubric_description": _DEFAULT_RUBRIC_DESCRIPTION,
        "verdict_options": [
            {"value": -1, "description": "Fail"},
            {"value": 1, "description": "Pass"},
        ],
    }
    payload = CriterionEvaluationRequest.model_validate(raw)
    return payload.model_dump()


async def _amain(argv: Sequence[str] | None) -> None:
    parser = argparse.ArgumentParser(description="Run a miner agent locally with real tool calls.")
    parser.add_argument("--agent-path", required=True, help="Path to the miner agent file.")
    parser.add_argument("--request-json", help="Path to a CriterionEvaluationRequest JSON file.")
    parser.add_argument("--claim-text", help="Override claim text for the default request payload.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    agent_path = Path(args.agent_path)
    if not agent_path.exists():
        raise FileNotFoundError(f"agent path not found: {agent_path}")
    request_path = Path(args.request_json) if args.request_json else None
    if request_path is not None and not request_path.exists():
        raise FileNotFoundError(f"request json not found: {request_path}")

    _load_agent(agent_path)
    request_payload = _build_request(request_path=request_path, claim_text=args.claim_text)
    invoker = create_local_tool_host()
    entrypoint = get_entrypoint("evaluate_criterion")

    try:
        with bind_tool_invoker(invoker):
            result = await entrypoint(request_payload)
    finally:
        await invoker.aclose()

    print(json.dumps(result))


def main(argv: Sequence[str] | None = None) -> None:
    try:
        asyncio.run(_amain(argv))
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc


__all__ = ["main"]
