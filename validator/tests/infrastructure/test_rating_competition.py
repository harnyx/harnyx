from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import Mock
from uuid import uuid4

import bittensor as bt
import pytest
from pydantic import ValidationError

from harnyx_commons.bittensor import VerificationError, build_canonical_request
from harnyx_commons.domain.judge_usage import JudgeUsageSummary
from harnyx_commons.domain.miner_task import Query
from harnyx_commons.rating_competition import (
    RatingAnswer,
    RatingJudgment,
    RatingQualityEvidence,
    RatingReceipt,
    RatingWork,
    RatingWorkPage,
)
from harnyx_miner_sdk.endpoint_protocol import EndpointCallback, query_digest
from harnyx_miner_sdk.query import Response
from harnyx_validator.application.rating_competition import verified_rating_answer
from harnyx_validator.runtime.rating_competition_worker import RatingCompetitionWorker


def _answer(query: Query) -> RatingAnswer:
    key = bt.Keypair.create_from_uri("//Alice")
    assignment_id = uuid4()
    body = EndpointCallback(
        assignment_id=assignment_id,
        query_digest=query_digest(query),
        nonce="a" * 32,
        expires_at=datetime(2026, 1, 1, tzinfo=UTC),
        response=Response(text="Answer"),
    ).model_dump_json()
    path = f"/old/base/v1/endpoint-assignments/{assignment_id}/callback"
    signature = key.sign(build_canonical_request("POST", path, body.encode())).hex()
    return RatingAnswer(
        assignment_id=assignment_id,
        expected_hotkey=key.ss58_address,
        callback_body_utf8=body,
        signature_hex=signature,
        signed_callback_path=path,
        receipt_logs=(),
    )


def _evidence() -> RatingQualityEvidence:
    return RatingQualityEvidence(
        first_order_preference="first",
        second_order_preference="second",
        reasoning=None,
        judge_usage=JudgeUsageSummary(
            call_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            reasoning_tokens=0,
            actual_cost_usd=0.0,
            models=(),
        ),
    )


def test_original_signed_path_survives_delayed_judging_and_binding_is_checked() -> None:
    query = Query(text="Question")
    answer = _answer(query)
    assert verified_rating_answer(answer, query).text == "Answer"
    with pytest.raises(ValueError):
        verified_rating_answer(answer, Query(text="Different question"))
    with pytest.raises(VerificationError):
        verified_rating_answer(answer.model_copy(update={"signed_callback_path": "/new/base/callback"}), query)
    with pytest.raises(ValueError):
        verified_rating_answer(answer.model_copy(update={"assignment_id": uuid4()}), query)


def test_quality_result_must_agree_with_both_order_preferences() -> None:
    with pytest.raises(ValidationError):
        RatingJudgment(comparison_id=uuid4(), quality_result=-1, two_order_evidence=_evidence())


@pytest.mark.anyio
async def test_worker_retries_completed_result_without_rejudging_and_keeps_other_work_independent() -> None:
    query = Query(text="Question")
    work = RatingWork(comparison_id=uuid4(), query=query, first=_answer(query), second=_answer(query))
    slow = work.model_copy(update={"comparison_id": uuid4()})
    gate = asyncio.Event()
    calls = []

    class Service:
        async def judge(self, item):
            calls.append(item.comparison_id)
            if item.comparison_id == slow.comparison_id:
                await gate.wait()
            return RatingJudgment(comparison_id=item.comparison_id, quality_result=1, two_order_evidence=_evidence())

    class Platform:
        failures = 1
        accepted = []

        async def poll_rating_comparisons(self, **kwargs):
            return RatingWorkPage(items=tuple(item for item in (work, slow) if item.comparison_id not in self.accepted))

        async def submit_rating_judgment(self, result):
            if self.failures:
                self.failures -= 1
                raise ConnectionError("lost delivery")
            self.accepted.append(result.comparison_id)

    platform = Platform()
    worker = RatingCompetitionWorker(platform, Service())
    for _ in range(5):
        await worker.tick()
        await asyncio.sleep(0)
    assert platform.accepted == [work.comparison_id]
    assert calls.count(work.comparison_id) == 1
    gate.set()
    for _ in range(3):
        await asyncio.sleep(0)
        await worker.tick()
    assert platform.accepted == [work.comparison_id, slow.comparison_id]


@pytest.mark.anyio
async def test_signed_client_round_trips_strict_work_and_completed_judgment() -> None:
    import httpx

    from harnyx_commons.bittensor import verify_signed_request
    from harnyx_validator.infrastructure.tools.platform_client import HttpPlatformClient

    query = Query(text="Question")
    work = RatingWork(comparison_id=uuid4(), query=query, first=_answer(query), second=_answer(query))
    result = RatingJudgment(comparison_id=work.comparison_id, quality_result=1, two_order_evidence=_evidence())
    key = bt.Keypair.create_from_uri("//Bob")
    received = []

    def handler(request: httpx.Request) -> httpx.Response:
        verify_signed_request(
            method=request.method,
            path_qs=request.url.raw_path.decode(),
            body=request.content,
            authorization_header=request.headers["Authorization"],
            allowed_ss58=(key.ss58_address,),
        )
        if request.method == "GET":
            return httpx.Response(200, content=RatingWorkPage(items=(work,)).model_dump_json())
        received.append(RatingJudgment.model_validate_json(request.content, strict=True))
        return httpx.Response(204)

    client = HttpPlatformClient(base_url="https://platform.test", hotkey=key, transport=httpx.MockTransport(handler))
    page = await client.poll_rating_comparisons()
    assert page.items == (work,)
    await client.submit_rating_judgment(result)
    assert received == [result]


def test_signed_citation_hydrates_only_from_its_assignment_receipts() -> None:
    from harnyx_commons.domain.tool_call import (
        SearchToolResult,
        ToolCall,
        ToolCallDetails,
        ToolCallOutcome,
        ToolResultPolicy,
    )
    from harnyx_miner_sdk.query import CitationRef

    query = Query(text="Question")
    answer = _answer(query)
    callback = EndpointCallback.model_validate_json(answer.callback_body_utf8)
    callback = callback.model_copy(
        update={
            "response": Response(text="Answer [[1]]", citations=[CitationRef(receipt_id="r1", result_id="result1")])
        }
    )
    body = callback.model_dump_json()
    signature = (
        bt.Keypair.create_from_uri("//Alice")
        .sign(build_canonical_request("POST", answer.signed_callback_path, body.encode()))
        .hex()
    )
    receipt = ToolCall(
        receipt_id="r1",
        session_id=answer.assignment_id,
        uid=0,
        tool="search_web",
        issued_at=datetime(2026, 1, 1, tzinfo=UTC),
        outcome=ToolCallOutcome.OK,
        details=ToolCallDetails(
            request_hash="request",
            result_policy=ToolResultPolicy.REFERENCEABLE,
            results=(
                SearchToolResult(
                    index=0,
                    result_id="result1",
                    url="https://example.com/source",
                    note="Source evidence",
                    title="Source",
                ),
            ),
        ),
    )
    answer = answer.model_copy(
        update={
            "callback_body_utf8": body,
            "signature_hex": signature,
            "receipt_logs": (RatingReceipt.from_tool_call(receipt),),
        }
    )
    round_trip = RatingAnswer.model_validate_json(answer.model_dump_json(), strict=True)
    response = verified_rating_answer(round_trip, query)
    assert [excerpt.model_dump() for excerpt in response.citations[0].excerpts] == [
        {"start": 0, "end": 15, "text": "Source evidence"}
    ]
    with pytest.raises(ValueError):
        verified_rating_answer(answer.model_copy(update={"receipt_logs": ()}), query)


@pytest.mark.anyio
async def test_rating_status_is_ready_only_while_worker_task_is_running() -> None:
    from harnyx_validator.application.status import StatusProvider

    class Platform:
        async def poll_rating_comparisons(self, **kwargs):
            return RatingWorkPage(items=())

    status = StatusProvider()
    assert status.snapshot()["rating_worker_ready"] is False
    worker = RatingCompetitionWorker(Platform(), object())
    status.rating_worker_readiness = worker.is_running
    assert status.snapshot()["rating_worker_ready"] is False
    worker.start()
    assert status.snapshot()["rating_worker_ready"] is True
    await worker.stop()
    assert status.snapshot()["rating_worker_ready"] is False


@pytest.mark.anyio
@pytest.mark.parametrize(
    "status,error_code,terminal",
    [
        (413, None, True),
        (409, "rating_judgment_already_accepted", True),
        (409, None, False),
        (409, "some_other_conflict", False),
        (503, None, False),
        (429, None, False),
    ],
)
async def test_worker_stops_only_terminal_delivery_and_does_not_rejudge(
    status, error_code, terminal, caplog, monkeypatch
):
    capture = Mock()
    monkeypatch.setattr("harnyx_commons.observability.sentry._capture_exception", capture)
    import httpx

    from harnyx_validator.infrastructure.tools.platform_client import HttpPlatformClient

    query = Query(text="Question")
    work = RatingWork(comparison_id=uuid4(), query=query, first=_answer(query), second=_answer(query))
    result = RatingJudgment(comparison_id=work.comparison_id, quality_result=1, two_order_evidence=_evidence())
    submissions = []
    judged = []
    accepted = False

    def handler(request):
        nonlocal accepted
        if request.method == "GET":
            return httpx.Response(200, content=RatingWorkPage(items=() if accepted else (work,)).model_dump_json())
        submissions.append(request.content)
        if len(submissions) == 1:
            return httpx.Response(status, json={} if error_code is None else {"error_code": error_code})
        accepted = True
        return httpx.Response(204)

    class Service:
        async def judge(self, item):
            judged.append(item.comparison_id)
            return result

    client = HttpPlatformClient(
        "https://platform.test", bt.Keypair.create_from_uri("//Bob"), transport=httpx.MockTransport(handler)
    )
    worker = RatingCompetitionWorker(client, Service())
    for _ in range(12):
        await worker.tick()
        await asyncio.sleep(0)
    assert judged == [work.comparison_id]
    capture.assert_called_once()
    assert len(submissions) == (1 if terminal else 2)
    if terminal:
        assert "automatic judging and delivery stopped" in caplog.text
    else:
        assert submissions[0] == submissions[1]


@pytest.mark.anyio
async def test_backlog_capacity_includes_blocked_judging_and_delivery_and_skips_held_work():
    from uuid import UUID

    from harnyx_validator.application.ports.platform import RatingJudgmentDeliveryRejectedError

    query = Query(text="Question")
    template = RatingWork(comparison_id=uuid4(), query=query, first=_answer(query), second=_answer(query))
    items = [template.model_copy(update={"comparison_id": UUID(int=i + 1)}) for i in range(45)]
    judge_gates = {item.comparison_id: asyncio.Event() for item in items}
    delivery_gates = {item.comparison_id: asyncio.Event() for item in items}
    judged, accepted, rejected, attempts = [], set(), set(), []
    polled = []

    class Service:
        async def judge(self, item):
            judged.append(item.comparison_id)
            await judge_gates[item.comparison_id].wait()
            return RatingJudgment(comparison_id=item.comparison_id, quality_result=1, two_order_evidence=_evidence())

    class Platform:
        async def poll_rating_comparisons(self, *, after=None, limit=100):
            polled.append((after, limit))
            available = [
                item
                for item in items
                if (after is None or item.comparison_id > after) and item.comparison_id not in accepted
            ]
            page = available[:limit]
            return RatingWorkPage(
                items=tuple(page), next_after=page[-1].comparison_id if len(available) > limit else None
            )

        async def submit_rating_judgment(self, result):
            identity = result.comparison_id
            attempts.append(identity)
            await delivery_gates[identity].wait()
            if identity == items[0].comparison_id:
                rejected.add(identity)
                raise RatingJudgmentDeliveryRejectedError("oversized")
            if identity == items[1].comparison_id and attempts.count(identity) == 1:
                raise ConnectionError("temporary delivery failure")
            accepted.add(identity)

    worker = RatingCompetitionWorker(Platform(), Service(), poll_interval_seconds=3600)
    worker.start()

    async def pump():
        for _ in range(5):
            await asyncio.sleep(0)
            await worker.tick()

    try:
        await pump()
        assert len(judged) == 10
        assert all(limit <= 10 for _, limit in polled)
        # Completed results still occupy every slot while delivery is blocked.
        for item in items[:10]:
            judge_gates[item.comparison_id].set()
        await pump()
        assert len(judged) == 10
        assert len(attempts) == 10
        # A rejection frees one slot even though Platform keeps returning that ID.
        delivery_gates[items[0].comparison_id].set()
        await pump()
        assert len(judged) == 11
        assert judged[-1] == items[10].comparison_id
        assert attempts.count(items[0].comparison_id) == 1
        # Temporary retry retains its result; eventual success independently frees another slot.
        delivery_gates[items[1].comparison_id].set()
        await pump()
        assert attempts.count(items[1].comparison_id) == 2
        assert len(judged) == 12
        assert judged.count(items[1].comparison_id) == 1
        assert len(set(judged) - accepted - rejected) == 10
        # Drain past the held/rejected first pages and reach the whole backlog.
        for gate in (*judge_gates.values(), *delivery_gates.values()):
            gate.set()
        for _ in range(35):
            await pump()
            assert len(set(judged) - accepted - rejected) <= 10
        assert len(set(judged)) == len(items)
        assert len(judged) == len(items)
        assert len(accepted) == len(items) - 1
    finally:
        await worker.stop()


@pytest.mark.anyio
async def test_failed_judging_waits_doubles_caps_and_resets_after_success(monkeypatch):
    capture = Mock()
    monkeypatch.setattr("harnyx_commons.observability.sentry._capture_exception", capture)
    real_sleep = asyncio.sleep
    delays, gates, attempts = [], [], []
    failures_left = 7
    delivered = False
    query = Query(text="Question")
    work = RatingWork(comparison_id=uuid4(), query=query, first=_answer(query), second=_answer(query))

    async def retry_sleep(delay):
        delays.append(delay)
        gate = asyncio.Event()
        gates.append(gate)
        await gate.wait()

    monkeypatch.setattr(asyncio, "sleep", retry_sleep)

    class Service:
        async def judge(self, item):
            nonlocal failures_left
            attempts.append(item.comparison_id)
            if failures_left:
                failures_left -= 1
                raise ConnectionError("whole judging attempt exhausted")
            return RatingJudgment(comparison_id=item.comparison_id, quality_result=1, two_order_evidence=_evidence())

    class Platform:
        async def poll_rating_comparisons(self, **kwargs):
            return RatingWorkPage(items=() if delivered else (work,))

        async def submit_rating_judgment(self, result):
            nonlocal delivered
            delivered = True

    worker = RatingCompetitionWorker(Platform(), Service())

    async def pump():
        for _ in range(4):
            await worker.tick()
            await real_sleep(0)

    try:
        await pump()
        for index, expected in enumerate((30, 60, 120, 240, 300, 300, 300)):
            assert delays[index] == expected
            assert len(attempts) == index + 1
            # Repeated polling cannot bypass the awaited timer.
            await pump()
            assert len(attempts) == index + 1
            gates[index].set()
            await pump()
        assert delivered
        assert len(attempts) == 8
        # A fresh attempt after successful completion starts at the initial delay.
        failures_left, delivered = 1, False
        await pump()
        assert delays[-1] == 30
        assert len(attempts) == 9
        gates[-1].set()
        await pump()
        assert delivered
        assert len(attempts) == 10
        assert capture.call_count == 8
        assert len({id(call.args[0]) for call in capture.call_args_list}) == 8
    finally:
        tasks = (*worker._active.values(), *worker._submissions.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_judging_backoff_keeps_capacity_while_other_judging_and_delivery_continue(monkeypatch):
    from uuid import UUID

    real_sleep = asyncio.sleep
    retry_gates, judged, submissions, accepted = [], [], [], set()
    query = Query(text="Question")
    template = RatingWork(comparison_id=uuid4(), query=query, first=_answer(query), second=_answer(query))
    items = [template.model_copy(update={"comparison_id": UUID(int=i + 1)}) for i in range(12)]
    failing = {item.comparison_id for item in items[:9]}
    judge_gate, delivery_gate = asyncio.Event(), asyncio.Event()

    async def retry_sleep(delay):
        assert delay == 30
        gate = asyncio.Event()
        retry_gates.append(gate)
        await gate.wait()

    monkeypatch.setattr(asyncio, "sleep", retry_sleep)

    class Service:
        async def judge(self, item):
            judged.append(item.comparison_id)
            if item.comparison_id in failing:
                raise ConnectionError("judging exhausted")
            if item.comparison_id == items[9].comparison_id:
                await judge_gate.wait()
            return RatingJudgment(comparison_id=item.comparison_id, quality_result=1, two_order_evidence=_evidence())

    class Platform:
        async def poll_rating_comparisons(self, *, after=None, limit=100):
            available = [
                item
                for item in items
                if item.comparison_id not in accepted and (after is None or item.comparison_id > after)
            ]
            page = available[:limit]
            return RatingWorkPage(
                items=tuple(page), next_after=page[-1].comparison_id if len(available) > limit else None
            )

        async def submit_rating_judgment(self, result):
            submissions.append(result.comparison_id)
            await delivery_gate.wait()
            if len(submissions) == 1:
                raise ConnectionError("temporary delivery failure")
            accepted.add(result.comparison_id)

    worker = RatingCompetitionWorker(Platform(), Service())

    async def pump():
        for _ in range(6):
            await worker.tick()
            await real_sleep(0)
            assert len(worker._active) + len(worker._completed) <= 10

    try:
        await pump()
        assert len(retry_gates) == 9
        assert len(judged) == 10
        judge_gate.set()
        await pump()
        assert submissions == [items[9].comparison_id]
        assert len(judged) == 10
        delivery_gate.set()
        await pump()
        await pump()
        assert submissions.count(items[9].comparison_id) == 2
        assert accepted == {item.comparison_id for item in items[9:]}
        assert len(judged) == 12
        assert len(worker._active) == 9
    finally:
        tasks = (*worker._active.values(), *worker._submissions.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.anyio
async def test_polling_failure_is_reported_once_without_stopping_worker_early(monkeypatch):
    """A failed poll must be visible to Sentry while the worker handles shutdown normally."""
    capture = Mock()
    monkeypatch.setattr("harnyx_commons.observability.sentry._capture_exception", capture)
    failure = ConnectionError("Platform unavailable")

    class Platform:
        async def poll_rating_comparisons(self, **kwargs):
            worker._stop.set()
            raise failure

    worker = RatingCompetitionWorker(Platform(), object())
    await worker._run()
    capture.assert_called_once_with(failure)
