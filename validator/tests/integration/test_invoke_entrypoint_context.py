from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from harnyx_commons.application.dto.session import SessionTokenRequest
from harnyx_commons.application.session_manager import SessionManager
from harnyx_commons.domain.miner_task import Query
from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from harnyx_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from harnyx_commons.sandbox.manager import SandboxDeployment
from harnyx_validator.application.dto.evaluation import EntrypointInvocationRequest
from harnyx_validator.application.invoke_entrypoint import EntrypointInvoker

pytestmark = [pytest.mark.anyio("asyncio"), pytest.mark.integration]


async def test_invoker_delivers_cost_and_time_context_through_real_sandbox(
    sandbox_launcher: Callable[[str], SandboxDeployment],
) -> None:
    deployment = sandbox_launcher("validator.tests.integration.sandbox.context_agent")
    sessions = InMemorySessionRegistry()
    tokens = InMemoryTokenRegistry()
    receipts = InMemoryReceiptLog()
    manager = SessionManager(sessions, tokens)
    session_id = uuid4()
    token = uuid4().hex
    issued_at = datetime.now(UTC)
    manager.issue(
        SessionTokenRequest(
            session_id=session_id,
            uid=42,
            task_id=uuid4(),
            issued_at=issued_at,
            expires_at=issued_at + timedelta(minutes=5),
            budget_usd=0.75,
            token=token,
        )
    )
    invoker = EntrypointInvoker(
        session_registry=sessions,
        sandbox_client=deployment.client,
        token_registry=tokens,
        receipt_log=receipts,
    )

    result = await invoker.invoke(
        EntrypointInvocationRequest(
            session_id=session_id,
            token=token,
            uid=42,
            execution_time_limit_seconds=12.5,
            query=Query(text="inspect context"),
        )
    )

    assert result.response.text is not None
    assert json.loads(result.response.text) == {
        "cost_budget": {
            "session_budget_usd": 0.75,
            "session_hard_limit_usd": 0.75,
            "session_used_budget_usd": 0.0,
            "session_remaining_budget_usd": 0.75,
        },
        "time_budget": {"limit_seconds": 12.5},
    }
