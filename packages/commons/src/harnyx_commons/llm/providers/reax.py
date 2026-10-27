"""REAX typed decisions with key-default routing and independent receipt accounting."""

from __future__ import annotations

from math import isfinite
from typing import Annotated, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from harnyx_commons.errors import ToolProviderError, ToolProviderFailureCode
from harnyx_commons.json_types import JsonObject
from harnyx_commons.llm.pricing import price_reax_decision
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

DECISIONS_ENDPOINT = "https://api.decisionmodels.io/v1/systemone"


class _Choice(ChoiceAnswer):
    confidence: Probability
    probabilities: dict[str, Probability] = Field(min_length=2)


class _Score(ScoreAnswer):
    confidence: Probability
    probabilities: dict[str, Probability] = Field(min_length=2)
    legend: dict[str, DecisionContext] = Field(min_length=2)


class _Noul(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["noul"]
    noul: Probability


class _Usage(DecisionUsage):
    input_tokens: int = Field(ge=1)
    output_tokens: int = Field(ge=0)
    decisions: int = Field(ge=1)


class _UsageEnvelope(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    usage: _Usage


class _Routing(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    model: str
    tier: Literal["eu", "global"]
    model_version: str | None = None


class _Response(_Routing):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    answers: dict[str, Annotated[_Choice | _Score | _Noul, Field(discriminator="type")]] = Field(min_length=1)
    usage: _Usage


class _Charge(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    charge: str = Field(pattern=r"^[0-9]+$")
    currency: Literal["USD", "EUR"]


class _ApiErrorDetail(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    code: str
    message: str
    request_id: str
    retryable: bool
    details: list[JsonObject] | None = None


class _ApiErrorEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    error: _ApiErrorDetail


def _http_error_message(request: DecisionQueryRequest, response: httpx.Response) -> str:
    message = f"REAX HTTP {response.status_code}"
    try:
        error = _ApiErrorEnvelope.model_validate_json(response.text).error
    except ValidationError:
        return message
    if error.code not in {"invalid_request", "request_limit_exceeded", "payload_too_large"}:
        return message
    # Vendor text/details may echo submitted content. Expose only known codes and
    # documented limits; token counting remains owned by the provider.
    limits = (
        "1 question, at most 16 options and 8,000 input tokens"
        if request.model == "s1-fast-2"
        else "up to 3 questions and 8,192 input tokens per question including state"
    )
    guidance = "Reduce serialized request/state size. " if error.code == "payload_too_large" else ""
    return f"{message} ({error.code}): {guidance}Model limits: {limits}."


def _effective_routing(request: DecisionQueryRequest, response: httpx.Response) -> _Routing:
    routing = _Routing.model_validate_json(response.text)
    if routing.model != request.model or routing.model_version not in {None, request.model}:
        raise ValueError("REAX returned a different model version")
    for header, expected in (
        ("S1-Region", routing.tier),
        ("S1-Model", request.model),
        ("S1-Model-Version", request.model),
    ):
        if header in response.headers and response.headers[header] != expected:
            raise ValueError("REAX routing evidence is inconsistent")
    return routing


def _billing(request: DecisionQueryRequest, response: httpx.Response) -> ProviderBillingMetadata:
    """Charge headers remain usable when answer, routing or token validation fails."""
    evidence: JsonObject = {
        "provider": "reax",
        "model": request.model,
        "settlement_source": "unavailable",
        "headers": {key: value for key, value in response.headers.items() if key.lower().startswith("s1-")},
    }
    usage = None
    reference = None
    actual = None
    try:
        native_usage = _UsageEnvelope.model_validate_json(response.text).usage
        usage = DecisionUsage(input_tokens=native_usage.input_tokens, output_tokens=native_usage.output_tokens)
    except ValidationError:
        pass  # Invalid usage must not erase a separately valid charge.
    try:
        routing = _effective_routing(request, response)
        evidence["tier"] = routing.tier
        if usage is not None:
            priced = price_reax_decision(request.model, routing.tier, usage)
            if priced is not None and isfinite(priced):
                reference = priced
    except (ValidationError, ValueError, OverflowError):
        pass  # Reference pricing requires coherent native routing evidence.
    try:
        charge = _Charge.model_validate(
            {
                "charge": response.headers.get("S1-Charge-Nanos"),
                "currency": response.headers.get("S1-Currency"),
            }
        )
        nanos = int(charge.charge)
        evidence.update({"charge_nanos": nanos, "currency": charge.currency})
        if nanos == 0 or charge.currency == "USD":
            amount = nanos / 1_000_000_000
            if isfinite(amount):
                actual = amount
                evidence["settlement_source"] = "provider_returned"
        elif reference is not None:
            actual = reference
            evidence["settlement_source"] = "published_usd_usage_rate"
            evidence["pricing_origin"] = "reax_published_usd_rates"
    except (ValidationError, ValueError, OverflowError):
        pass  # Retain the original headers, but never infer a charge from invalid settlement.
    return ProviderBillingMetadata(
        actual_cost_provider="reax",
        actual_cost_usd=actual,
        reference_cost_usd=reference,
        source="response_headers",
        usage=usage,
        provider_usage=evidence,
        reference_cost_unavailable=reference is None,
    )


class ReaxDecisionProvider:
    def __init__(
        self,
        *,
        api_key: SecretStr,
        timeout_seconds: float = 300,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.get_secret_value().strip():
            raise ValueError("REAX API key must be provided")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        self._owns_client = client is None

    async def query(self, request: DecisionQueryRequest) -> DecisionProviderResult:
        if request.provider != "reax":
            raise ValueError("REAX decision adapter requires reax")
        questions = {
            key: question.model_dump(mode="json", exclude_none=True) for key, question in request.questions.items()
        }
        for question in questions.values():
            if question["type"] == "boolean":
                question["type"] = "noul"
        try:
            response = await self._client.post(
                DECISIONS_ENDPOINT,
                headers={"Authorization": f"Bearer {self._api_key.get_secret_value().strip()}"},
                json={"model": request.model, "state": request.state, "questions": questions},
                timeout=request.timeout if request.timeout is not None else self._timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise ToolProviderError("REAX transport failed", provider="reax") from exc
        billing = _billing(request, response)
        if not response.is_success:
            raise ToolProviderError(
                _http_error_message(request, response),
                provider="reax",
                billing=billing,
                http_status=response.status_code,
                failure_code=ToolProviderFailureCode.AUTHENTICATION_FAILED
                if response.status_code in {401, 403}
                else ToolProviderFailureCode.PROVIDER_FAILED,
            )
        try:
            native = _Response.model_validate_json(response.text)
            _effective_routing(request, response)
            if native.usage.decisions != len(request.questions):
                raise ValueError("REAX decision count differs from requested questions")
            normalized = DecisionQueryResponse(
                model=native.model,
                id=native.id,
                provider="reax",
                answers={
                    key: BooleanAnswer(type="boolean", probability=answer.noul) if isinstance(answer, _Noul) else answer
                    for key, answer in native.answers.items()
                },
                usage=DecisionUsage(input_tokens=native.usage.input_tokens, output_tokens=native.usage.output_tokens),
                provider_metadata={
                    "model_version": native.model_version,
                    "tier": native.tier,
                    "execution_region": response.headers.get("S1-Execution-Region"),
                    "billing": billing.provider_usage,
                },
            )
            validate_decision_answers(request, normalized)
        except (ValidationError, ValueError) as exc:
            raise ToolProviderError("Invalid REAX decision response", provider="reax", billing=billing) from exc
        if billing.actual_cost_usd is None:
            raise ToolProviderError("REAX response missing settled cost", provider="reax", billing=billing)
        return DecisionProviderResult(
            response=normalized,
            actual_cost_usd=billing.actual_cost_usd,
            actual_cost_provider="reax",
            actual_cost_evidence=billing.provider_usage or {},
            reference_cost_usd=billing.reference_cost_usd,
            native_response=DecisionNativeResponse(body=response.text, status_code=response.status_code),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
