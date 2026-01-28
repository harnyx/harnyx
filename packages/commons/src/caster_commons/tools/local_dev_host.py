"""Local in-process tool host used by miner development utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from caster_commons.application.dto.session import SessionTokenRequest
from caster_commons.application.session_manager import SessionManager
from caster_commons.clients import CHUTES, DESEARCH
from caster_commons.config.llm import LlmSettings
from caster_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from caster_commons.infrastructure.state.session_registry import InMemorySessionRegistry
from caster_commons.infrastructure.state.token_registry import InMemoryTokenRegistry
from caster_commons.llm.providers.chutes import ChutesLlmProvider
from caster_commons.tools.desearch import DeSearchClient
from caster_commons.tools.dto import ToolInvocationRequest
from caster_commons.tools.executor import ToolExecutor
from caster_commons.tools.http_serialization import serialize_tool_execute_response
from caster_commons.tools.runtime_invoker import ALLOWED_TOOL_MODELS, RuntimeToolInvoker
from caster_commons.tools.types import parse_tool_name
from caster_commons.tools.usage_tracker import UsageTracker


@dataclass(frozen=True, slots=True)
class LocalToolHost:
    """In-process tool host that matches the miner SDK tool invoker protocol."""

    session_id: UUID
    token: str
    _tool_executor: ToolExecutor
    _search_client: DeSearchClient
    _llm_provider: ChutesLlmProvider

    async def invoke(
        self,
        method: str,
        *,
        args: Sequence[Any] | None = None,
        kwargs: Mapping[str, Any] | None = None,
    ) -> Any:
        tool = parse_tool_name(method)
        result = await self._tool_executor.execute(
            ToolInvocationRequest(
                session_id=self.session_id,
                token=self.token,
                tool=tool,
                args=tuple(args or ()),
                kwargs=dict(kwargs or {}),
            )
        )
        return serialize_tool_execute_response(result).model_dump(mode="json")

    async def aclose(self) -> None:
        await self._search_client.aclose()
        await self._llm_provider.aclose()


def create_local_tool_host(*, uid: int = 1, session_ttl_minutes: int = 30) -> LocalToolHost:
    llm_settings = LlmSettings()
    if not llm_settings.desearch_api_key_value:
        raise RuntimeError("DESEARCH_API_KEY must be set to run local tool host")
    if not llm_settings.chutes_api_key_value:
        raise RuntimeError("CHUTES_API_KEY must be set to run local tool host")

    usage_tracker = UsageTracker()

    sessions = InMemorySessionRegistry()
    tokens = InMemoryTokenRegistry()
    receipts = InMemoryReceiptLog()
    session_manager = SessionManager(sessions, tokens)

    session_id = uuid4()
    token = uuid4().hex
    now = datetime.now(UTC)
    session_manager.issue(
        SessionTokenRequest(
            session_id=session_id,
            uid=uid,
            claim_id=uuid4(),
            issued_at=now,
            expires_at=now + timedelta(minutes=session_ttl_minutes),
            budget_usd=0.05,
            token=token,
        )
    )

    desearch = DeSearchClient(
        base_url=DESEARCH.base_url,
        api_key=llm_settings.desearch_api_key_value,
        timeout=DESEARCH.timeout_seconds,
        max_concurrent=llm_settings.desearch_max_concurrent,
    )
    llm_provider = ChutesLlmProvider(
        base_url=CHUTES.base_url,
        api_key=llm_settings.chutes_api_key_value,
        timeout=CHUTES.timeout_seconds,
        max_concurrent=llm_settings.chutes_max_concurrent,
    )
    tool_invoker = RuntimeToolInvoker(
        receipts,
        search_client=desearch,
        llm_provider=llm_provider,
        llm_provider_name="chutes",
        allowed_models=ALLOWED_TOOL_MODELS,
    )
    tool_executor = ToolExecutor(
        sessions,
        receipts,
        usage_tracker,
        tool_invoker,
        token_registry=tokens,
        clock=lambda: datetime.now(UTC),
    )

    return LocalToolHost(
        session_id=session_id,
        token=token,
        _tool_executor=tool_executor,
        _search_client=desearch,
        _llm_provider=llm_provider,
    )


__all__ = ["LocalToolHost", "create_local_tool_host"]
