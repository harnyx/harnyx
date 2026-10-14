"""Author citations render against the frozen key without fetching public sources."""

import socket

import pytest
from public.packages.commons.tests.miner_task_generation.test_candidate_pipeline import author_reference, draft

from harnyx_commons.miner_task_generation.contracts import ReferenceValidationError
from harnyx_commons.miner_task_generation.reference_generation import render_author_reference
from harnyx_commons.miner_task_generation.source_urls import public_addresses

pytestmark = pytest.mark.anyio("asyncio")


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(
        "harnyx_commons.miner_task_generation.reference_generation.public_addresses",
        lambda host, port: ("93.184.216.34",),
    )


@pytest.mark.parametrize("structured", [False, True])
async def test_author_prose_and_frozen_structured_answer_render_without_host_source_fetch(structured):
    result = await render_author_reference(draft(structured), author_reference(structured))
    answer = result.reference_answer
    assert answer is not None and answer.text == ('{"item":"item"}' if structured else "item [[1]]")
    assert answer.citations[0].excerpts[0].text == "item"
    assert "Required final-answer components" in answer.note


async def test_bibliographic_citation_does_not_discard_a_tested_reference():
    packet = author_reference()
    citation = packet.citations[0]
    packet = packet.model_copy(
        update={
            "citations": [
                citation,
                citation.model_copy(update={"url": "https://example.org/bibliography", "excerpts": ()}),
            ]
        }
    )
    result = await render_author_reference(draft(), packet)
    assert result.reference_answer.text == "item [[1]]"
    assert result.reference_answer.citations == tuple(packet.citations)


async def test_missing_citation_ranges_do_not_trigger_host_hydration():
    packet = author_reference()
    citation = packet.citations[0]
    packet = packet.model_copy(
        update={
            "citations": [
                citation.model_copy(
                    update={"excerpts": (citation.excerpts[0].model_copy(update={"start": None, "end": None}),)}
                )
            ]
        }
    )
    result = await render_author_reference(draft(), packet)
    assert result.reference_answer.citations[0].excerpts[0].start is None


async def test_plain_reference_cannot_silently_fall_back_to_uncited_key():
    packet = author_reference().model_copy(update={"answer_text": None})
    with pytest.raises(ValueError, match="complete answer"):
        await render_author_reference(draft(), packet)


async def test_plain_reference_report_cannot_be_misplaced_in_supplementary_note():
    packet = author_reference().model_copy(update={"note": "The report was misplaced here [[1]]"})
    with pytest.raises(ReferenceValidationError, match="answer_text and note=null"):
        await render_author_reference(draft(), packet)


async def test_broken_frozen_structured_contract_is_not_a_repairable_reference_error(monkeypatch):
    def broken(*args):
        raise ValueError("Broken frozen contract")

    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.validate_structured_draft", broken)
    with pytest.raises(ValueError) as failure:
        await render_author_reference(draft(True), author_reference(True))
    assert not isinstance(failure.value, ReferenceValidationError)


@pytest.mark.parametrize("answer_text,note", [("misplaced report [[1]]", None), (None, None), (None, " ")])
async def test_structured_report_cannot_be_silently_lost(answer_text, note):
    packet = author_reference(True).model_copy(update={"answer_text": answer_text, "note": note})
    with pytest.raises(ReferenceValidationError, match="complete research report in note"):
        await render_author_reference(draft(True), packet)


@pytest.mark.parametrize("count", [400, 401])
async def test_reference_preserves_shared_materialized_segment_boundary(count):
    packet = author_reference()
    citation = packet.citations[0]
    packet = packet.model_copy(
        update={"citations": [citation.model_copy(update={"excerpts": citation.excerpts * count})]}
    )
    if count == 401:
        with pytest.raises(ReferenceValidationError, match="400-evidence-segment"):
            await render_author_reference(draft(), packet)
    else:
        result = await render_author_reference(draft(), packet)
        assert len(result.reference_answer.citations[0].excerpts) == 400


async def test_bibliographic_packet_does_not_introduce_a_mandatory_excerpt_gate():
    packet = author_reference()
    packet = packet.model_copy(update={"citations": [packet.citations[0].model_copy(update={"excerpts": ()})]})
    result = await render_author_reference(draft(), packet)
    assert result.reference_answer.citations[0].excerpts == ()


@pytest.mark.parametrize(
    "url",
    [
        "/workspace/source.txt",
        "file:///workspace/source.txt",
        "https://user:password@example.org/report",
        "https://example.org:0/report",
        "https://example.org:65536/report",
        "https://example.org:invalid/report",
    ],
)
async def test_reference_rejects_local_or_credential_bearing_source_urls(url):
    packet = author_reference()
    packet = packet.model_copy(update={"citations": [packet.citations[0].model_copy(update={"url": url})]})
    with pytest.raises(ValueError, match="HTTP|credentials|[Pp]ort"):
        await render_author_reference(draft(), packet)


@pytest.mark.parametrize(
    "url,addresses",
    [
        ("http://localhost/report", ["127.0.0.1"]),
        ("http://127.0.0.1/report", ["127.0.0.1"]),
        ("http://10.0.0.1/report", ["10.0.0.1"]),
        ("http://169.254.169.254/report", ["169.254.169.254"]),
        ("http://[::1]/report", ["::1"]),
        ("http://[fd00::1]/report", ["fd00::1"]),
        ("https://example.org/report", ["93.184.216.34", "10.0.0.1"]),
    ],
)
async def test_reference_cannot_publish_a_nonpublic_or_mixed_dns_authority(monkeypatch, url, addresses):
    """A public-looking citation must not bypass the previous public-address boundary."""
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.public_addresses", public_addresses)
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443)) for address in addresses
        ],
    )
    packet = author_reference()
    packet = packet.model_copy(update={"citations": [packet.citations[0].model_copy(update={"url": url})]})
    with pytest.raises(ReferenceValidationError, match="non-public"):
        await render_author_reference(draft(), packet)


@pytest.mark.parametrize("resolution", ["empty", "failure", "invalid", "public"])
async def test_reference_dns_validation_preserves_failures_and_accepts_public_addresses(monkeypatch, resolution):
    """Resolution failures must not silently permit unvalidated citation authorities."""
    monkeypatch.setattr("harnyx_commons.miner_task_generation.reference_generation.public_addresses", public_addresses)

    def resolve(*args, **kwargs):
        if resolution == "failure":
            raise socket.gaierror("lookup failed")
        if resolution == "empty":
            return []
        if resolution == "invalid":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("not-an-ip", 443))]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    if resolution == "public":
        assert (await render_author_reference(draft(), author_reference())).reference_answer is not None
    else:
        with pytest.raises(ReferenceValidationError, match="DNS"):
            await render_author_reference(draft(), author_reference())
