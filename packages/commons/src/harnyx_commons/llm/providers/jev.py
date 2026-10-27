"""TypeSafe JEV decision provider shared by hosted tools and benchmark grading."""

from __future__ import annotations

import asyncio
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.llm.pricing import price_decision
from harnyx_commons.tools.decision_models import (
    BooleanAnswer,
    ChoiceAnswer,
    DecisionContext,
    DecisionQueryRequest,
    DecisionQueryResponse,
    DecisionUsage,
    ScoreAnswer,
    validate_decision_answers,
)
from harnyx_commons.tools.ports import DecisionNativeResponse, DecisionProviderResult
from harnyx_commons.tools.provider_billing import ProviderBillingMetadata
from harnyx_miner_sdk.tools.decision_models import Probability

DECISIONS_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


class _JevChoice(ChoiceAnswer):
    probabilities: dict[str, Probability]
    confidence: Probability


class _JevScore(ScoreAnswer):
    probabilities: dict[str, Probability]
    confidence: Probability
    legend: dict[str, DecisionContext]


class _JevNoul(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    type: Literal["noul"]
    noul: Probability


class _JevUsage(DecisionUsage):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class _JevBilling(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    usage: _JevUsage


class _JevResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    model: str = Field(min_length=1)
    answers: dict[str, Annotated[_JevChoice | _JevScore | _JevNoul, Field(discriminator="type")]]
    usage: _JevUsage


class JevDecisionError(ToolProviderError):
    """A failed decision retains its private native receipt when one was received."""

    def __init__(
        self,
        message: str,
        *,
        native_response: DecisionNativeResponse | None = None,
        billing: ProviderBillingMetadata | None = None,
    ) -> None:
        status = None if native_response is None else native_response.status_code
        super().__init__(
            message,
            provider="jev",
            billing=billing,
            http_status=status,
            failure_code=ToolProviderFailureCode.AUTHENTICATION_FAILED
            if status in {401, 403}
            else ToolProviderFailureCode.PROVIDER_FAILED,
        )
        self.native_response = native_response


class JevDecisionProvider:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        timeout_seconds: float = 300,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.get_secret_value().strip():
            raise ValueError("TypeSafe API key must be provided for JEV")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None
        self._semaphore = asyncio.Semaphore(4)

    async def query(self, request: DecisionQueryRequest) -> DecisionProviderResult:
        if request.provider != "jev":
            raise ValueError("JEV decision adapter requires jev")
        questions = {
            key: question.model_dump(mode="json", exclude_none=True) for key, question in request.questions.items()
        }
        for question in questions.values():
            if question["type"] == "boolean":
                question["type"] = "noul"
        async with self._semaphore:
            try:
                response = await self._client.post(
                    DECISIONS_ENDPOINT,
                    headers={"Authorization": f"Bearer {self._api_key.get_secret_value().strip()}"},
                    json={"model": request.model, "state": request.state, "questions": questions},
                    timeout=request.timeout if request.timeout is not None else self._timeout_seconds,
                )
            except httpx.HTTPError as exc:
                raise JevDecisionError(f"{type(exc).__name__}: {exc}") from exc
        native_response = DecisionNativeResponse(body=response.text, status_code=response.status_code)
        billing = self._billing(request, response.text)
        if not response.is_success:
            raise JevDecisionError(f"JEV HTTP {response.status_code}", native_response=native_response, billing=billing)
        try:
            native = _JevResponse.model_validate_json(response.text)
            if native.model != request.model:
                raise ValueError("JEV returned a different model")
            normalized = DecisionQueryResponse(
                model=native.model,
                provider="jev",
                answers={
                    key: BooleanAnswer(type="boolean", probability=answer.noul)
                    if isinstance(answer, _JevNoul)
                    else answer
                    for key, answer in native.answers.items()
                },
                usage=native.usage,
            )
            validate_decision_answers(request, normalized)
        except (ValidationError, ValueError) as exc:
            raise JevDecisionError(
                "invalid JEV response: schema or criterion validation failed",
                native_response=native_response,
                billing=billing,
            ) from exc
        assert billing is not None
        return DecisionProviderResult(
            response=normalized,
            actual_cost_usd=billing.actual_cost_usd,
            actual_cost_provider="jev",
            actual_cost_evidence=billing.provider_usage or {},
            reference_cost_usd=billing.reference_cost_usd,
            native_response=native_response,
        )

    @staticmethod
    def _billing(request: DecisionQueryRequest, raw_body: str) -> ProviderBillingMetadata | None:
        try:
            usage = _JevBilling.model_validate_json(raw_body).usage
        except ValidationError:
            return None
        cost = price_decision("jev", request.model, usage)
        return ProviderBillingMetadata(
            actual_cost_provider="jev",
            actual_cost_usd=cost,
            reference_cost_usd=cost,
            source="response_body",
            usage=usage,
            provider_usage={
                "provider": "jev",
                "model": request.model,
                "settlement_source": "static_pricing",
                "pricing_origin": "typesafe_published_rates",
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
