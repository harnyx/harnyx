"""Public-only, DNS-pinned, complete-source acquisition."""

from __future__ import annotations

import asyncio
import http.client
import ipaddress
import re
import socket
import ssl
import sys
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urljoin, urlsplit

from pydantic import BaseModel, Field

from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain_tweak_generation.source_workspace import SourceDocument, SourceLink
from harnyx_commons.source_extractor_worker import (
    MAX_ADDRESS_SPACE_BYTES,
    MAX_CPU_SECONDS,
    MAX_EXTRACTED_CHARACTERS,
    MAX_PDF_PAGES,
    ExtractionRejectedError,
    extract_source,
)

MAX_RESPONSE_BYTES = 64 * 1024 * 1024
MAX_WALL_SECONDS = 60.0
MAX_REDIRECTS = 5
MAX_FETCH_WALL_SECONDS = (MAX_REDIRECTS + 1) * MAX_WALL_SECONDS
_MAX_CONCURRENT_SOURCE_WORKERS = 5
DocumentKind = Literal["html", "text", "pdf", "xlsx", "static_json"]
_SENSITIVE_QUERY_KEYS = frozenset(
    {"key", "api_key", "apikey", "token", "access_token", "auth", "authorization", "signature", "sig"}
)
_API_PATH_SEGMENTS = frozenset({"api", "graphql", "rest"})
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_PERCENT_ESCAPE = re.compile(r"%[0-9A-Fa-f]{2}")
_XLSX_MEDIA_TYPES = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel.sheet.macroenabled.12",
    }
)


class SourceFetchError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class _FetchedBody:
    final_url: str
    media_type: str
    content_encoding: str
    body: bytes


class _FetchedHeader(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    final_url: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    content_encoding: str
    body_length: int = Field(ge=0, le=MAX_RESPONSE_BYTES)


class _WorkerError(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2_000)


class _ExtractedLinkPayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    url: str = Field(min_length=1)
    text: str


class _ExtractedSourcePayload(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    content: str = Field(min_length=1, max_length=MAX_EXTRACTED_CHARACTERS)
    links: tuple[_ExtractedLinkPayload, ...] = ()


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, pinned_ip: str, timeout: float) -> None:
        super().__init__(host, port=port, timeout=timeout)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_ip, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, port: int, pinned_ip: str, timeout: float) -> None:
        self._ssl_context = ssl.create_default_context()
        super().__init__(host, port=port, timeout=timeout, context=self._ssl_context)
        self._pinned_ip = pinned_ip

    def connect(self) -> None:
        raw = socket.create_connection((self._pinned_ip, self.port), self.timeout)
        try:
            self.sock = self._ssl_context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


class PublicSourceFetcher:
    """Fetches and fully extracts one source without accepting a partial prefix."""

    def __init__(self) -> None:
        self._worker_slots = asyncio.Semaphore(_MAX_CONCURRENT_SOURCE_WORKERS)

    async def fetch(self, url: str, *, document_kind: DocumentKind) -> SourceDocument:
        _validated_document_url(url, document_kind)
        async with self._worker_slots:
            fetched = await _fetch_isolated(url, document_kind)
            _validate_document_response(fetched, document_kind)
            extracted = await _extract_isolated(
                fetched.body,
                fetched.media_type,
                fetched.content_encoding,
                fetched.final_url,
            )
        return SourceDocument(
            requested_url=url,
            final_url=fetched.final_url,
            media_type=fetched.media_type,
            content=extracted.content,
            fetched_bytes=len(fetched.body),
            links=tuple(SourceLink(url=link.url, text=link.text) for link in extracted.links),
        )


def _fetch_complete_body(initial_url: str, document_kind: DocumentKind) -> _FetchedBody:
    current = initial_url
    for redirect_index in range(MAX_REDIRECTS + 1):
        parsed = _validated_document_url(current, document_kind)
        host = parsed.hostname
        assert host is not None
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        addresses = _public_addresses(host, port)
        path = parsed.path or "/"
        if parsed.query:
            path += "?" + parsed.query
        connection, response = _request_from_public_address(
            scheme=parsed.scheme,
            host=host,
            port=port,
            addresses=addresses,
            path=path,
            host_header=parsed.netloc,
        )
        try:
            if _public_addresses(host, port) != addresses:
                raise SourceFetchError("source_fetch_rejected", "DNS resolution changed during the request")
            peer = connection.sock.getpeername()[0] if connection.sock is not None else None
            if peer not in addresses:
                raise SourceFetchError("source_fetch_rejected", "connected peer was not in the validated DNS set")
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise SourceFetchError("source_unavailable", "redirect response omitted Location")
                if redirect_index >= MAX_REDIRECTS:
                    raise SourceFetchError("source_unavailable", "too many redirects")
                current = urljoin(current, location)
                continue
            if response.status < 200 or response.status >= 300:
                raise SourceFetchError("source_unavailable", f"source returned HTTP {response.status}")
            length = response.getheader("Content-Length")
            if length is not None and int(length) > MAX_RESPONSE_BYTES:
                raise SourceFetchError("source_extraction_limit", "source exceeds the 64 MiB response limit")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = response.read(min(1024 * 1024, MAX_RESPONSE_BYTES + 1 - total))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_RESPONSE_BYTES:
                    raise SourceFetchError("source_extraction_limit", "source exceeds the 64 MiB response limit")
                chunks.append(chunk)
            media_type = (response.getheader("Content-Type") or "application/octet-stream").split(";", 1)[0]
            return _FetchedBody(
                final_url=current,
                media_type=media_type.casefold(),
                content_encoding=(response.getheader("Content-Encoding") or "").casefold(),
                body=b"".join(chunks),
            )
        except SourceFetchError:
            raise
        except (OSError, http.client.HTTPException, ValueError) as exc:
            raise SourceFetchError("source_unavailable", f"source request failed: {exc}") from exc
        finally:
            connection.close()
    raise AssertionError("redirect loop must return or raise")


def _request_from_public_address(
    *,
    scheme: str,
    host: str,
    port: int,
    addresses: tuple[str, ...],
    path: str,
    host_header: str,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    last_connection_error: OSError | None = None
    for address in addresses:
        connection: http.client.HTTPConnection
        if scheme == "https":
            connection = _PinnedHTTPSConnection(host, port, address, timeout=60.0)
        else:
            connection = _PinnedHTTPConnection(host, port, address, timeout=60.0)
        try:
            connection.connect()
        except ssl.SSLError as exc:
            connection.close()
            raise SourceFetchError("source_unavailable", f"source request failed: {exc}") from exc
        except OSError as exc:
            last_connection_error = exc
            connection.close()
            continue
        except BaseException:
            connection.close()
            raise
        try:
            connection.request(
                "GET",
                path,
                headers={
                    "Accept": (
                        "text/html,application/pdf,text/plain,application/json,"
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*;q=0.5"
                    ),
                    "Accept-Encoding": "gzip, deflate",
                    "Host": host_header,
                    "User-Agent": "Harnyx-source-acquisition/1.0",
                },
            )
            return connection, connection.getresponse()
        except (OSError, http.client.HTTPException, ValueError) as exc:
            connection.close()
            raise SourceFetchError("source_unavailable", f"source request failed: {exc}") from exc
        except BaseException:
            connection.close()
            raise
    assert last_connection_error is not None
    raise SourceFetchError(
        "source_unavailable",
        f"source request failed: {last_connection_error}",
    ) from last_connection_error


def _validated_url(url: str) -> Any:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        raise SourceFetchError("source_fetch_rejected", "source URL must use HTTP(S)")
    if not parsed.hostname or parsed.username is not None or parsed.password is not None:
        raise SourceFetchError("source_fetch_rejected", "source URL authority is invalid or contains credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise SourceFetchError("source_fetch_rejected", "source URL port is invalid") from exc
    if port is not None and not 1 <= port <= 65535:
        raise SourceFetchError("source_fetch_rejected", "source URL port is invalid")
    return parsed


def _validated_document_url(url: str, document_kind: DocumentKind) -> Any:
    parsed = _validated_url(url)
    host = parsed.hostname
    assert host is not None
    if host.split(".", 1)[0].casefold() == "api":
        raise SourceFetchError("source_fetch_rejected", "API hostnames are not public-document sources")
    _validate_percent_encoding(parsed.path)
    canonical_path = _fully_unquote_url_component(parsed.path)
    segments = {segment.casefold() for segment in canonical_path.split("/") if segment}
    if segments & _API_PATH_SEGMENTS:
        raise SourceFetchError("source_fetch_rejected", "API, GraphQL, and REST paths are not document sources")
    _validate_percent_encoding(parsed.query)
    try:
        query_items = parse_qsl(parsed.query, keep_blank_values=True, errors="strict")
    except UnicodeDecodeError as exc:
        raise SourceFetchError("source_fetch_rejected", "source URL contains invalid percent encoding") from exc
    query_keys = {_fully_unquote_url_component(key).casefold() for key, _ in query_items}
    if query_keys & _SENSITIVE_QUERY_KEYS:
        raise SourceFetchError("source_fetch_rejected", "source URL contains credential-shaped query material")
    if document_kind == "static_json" and (parsed.query or not parsed.path.casefold().endswith(".json")):
        raise SourceFetchError(
            "source_fetch_rejected",
            "static JSON documents require a query-free .json URL",
        )
    return parsed


def _validate_percent_encoding(value: str) -> None:
    if _INVALID_PERCENT_ESCAPE.search(value):
        raise SourceFetchError("source_fetch_rejected", "source URL contains malformed percent encoding")


def _fully_unquote_url_component(value: str) -> str:
    """Resolve nested URL encoding so validation sees the path/query an origin may see."""
    decoded = value
    for _ in range(len(value) + 1):
        if not _PERCENT_ESCAPE.search(decoded):
            return decoded
        try:
            next_decoded = unquote(decoded, errors="strict")
        except UnicodeDecodeError as exc:
            raise SourceFetchError("source_fetch_rejected", "source URL contains invalid percent encoding") from exc
        decoded = next_decoded
    raise SourceFetchError("source_fetch_rejected", "source URL contains excessive nested percent encoding")


def _validate_document_response(fetched: _FetchedBody, document_kind: DocumentKind) -> None:
    media_type = fetched.media_type
    path = urlsplit(fetched.final_url).path.casefold()
    valid = {
        "html": "html" in media_type,
        "text": media_type.startswith("text/"),
        "pdf": "pdf" in media_type or path.endswith(".pdf"),
        "xlsx": media_type in _XLSX_MEDIA_TYPES or path.endswith(".xlsx"),
        "static_json": "json" in media_type and path.endswith(".json"),
    }[document_kind]
    if not valid:
        raise SourceFetchError("source_unavailable", "source response does not match the declared document kind")


def _public_addresses(host: str, port: int) -> tuple[str, ...]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise SourceFetchError("source_unavailable", f"DNS resolution failed: {exc}") from exc
    addresses = tuple(sorted({str(item[4][0]) for item in infos}))
    if not addresses:
        raise SourceFetchError("source_unavailable", "DNS resolution returned no addresses")
    for address in addresses:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError as exc:
            raise SourceFetchError("source_fetch_rejected", "DNS returned an invalid address") from exc
        if not parsed.is_global:
            raise SourceFetchError("source_fetch_rejected", f"DNS returned non-public address {address}")
    return addresses


async def _fetch_isolated(url: str, document_kind: DocumentKind) -> _FetchedBody:
    output = await _run_worker(
        (sys.executable, "-m", "harnyx_commons.source_fetch_worker", url, document_kind),
        input_payload=None,
        timeout_seconds=MAX_FETCH_WALL_SECONDS,
        timeout_code="source_unavailable",
        timeout_message="source fetch exceeded its absolute wall-time limit",
    )
    if output[:1] == b"E":
        _raise_worker_error(output)
    if output[:1] != b"O" or len(output) < 5:
        raise SourceFetchError("source_unavailable", "isolated source fetcher returned an invalid response")
    header_length = int.from_bytes(output[1:5], "big")
    header_end = 5 + header_length
    if header_length <= 0 or header_end > len(output):
        raise SourceFetchError("source_unavailable", "isolated source fetcher returned an invalid header")
    header = _FetchedHeader.model_validate_json(output[5:header_end])
    body = output[header_end:]
    if len(body) != header.body_length:
        raise SourceFetchError("source_unavailable", "isolated source fetcher returned an incomplete body")
    return _FetchedBody(
        final_url=header.final_url,
        media_type=header.media_type,
        content_encoding=header.content_encoding,
        body=body,
    )


async def _extract_isolated(body: bytes, media_type: str, encoding: str, url: str) -> _ExtractedSourcePayload:
    output = await _run_worker(
        (sys.executable, "-m", "harnyx_commons.source_extractor_worker", media_type, encoding, url),
        input_payload=body,
        timeout_seconds=MAX_WALL_SECONDS,
        timeout_code="source_extraction_limit",
        timeout_message="source extraction exceeded 60 wall seconds",
    )
    if output[:1] == b"O":
        try:
            return _ExtractedSourcePayload.model_validate_json(output[1:])
        except Exception as exc:
            raise SourceFetchError("source_unavailable", "isolated source extractor returned invalid metadata") from exc
    if output[:1] == b"E":
        _raise_worker_error(output)
    raise SourceFetchError("source_unavailable", "isolated source extractor returned an invalid response")


async def _run_worker(
    command: tuple[str, ...],
    *,
    input_payload: bytes | None,
    timeout_seconds: float,
    timeout_code: str,
    timeout_message: str,
) -> bytes:
    process = await asyncio.create_subprocess_exec(
        *command,
        stdin=asyncio.subprocess.PIPE if input_payload is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    communication = asyncio.create_task(process.communicate(input_payload))
    try:
        async with asyncio.timeout(timeout_seconds):
            output, _ = await asyncio.shield(communication)
    except TimeoutError as exc:
        await _kill_and_drain(process, communication)
        raise SourceFetchError(timeout_code, timeout_message) from exc
    except BaseException:
        await _kill_and_drain(process, communication)
        raise
    if process.returncode != 0 or not output:
        raise SourceFetchError(timeout_code, "isolated source worker terminated")
    return output


async def _kill_and_drain(
    process: asyncio.subprocess.Process,
    communication: asyncio.Task[tuple[bytes, bytes]],
) -> None:
    if process.returncode is None:
        process.kill()
    # Cleanup must preserve the timeout or cancellation that triggered it.
    await asyncio.gather(communication, return_exceptions=True)


def _raise_worker_error(output: bytes) -> None:
    error = _WorkerError.model_validate_json(output[1:])
    raise SourceFetchError(error.code, error.message)


def _extract_content(body: bytes, media_type: str, encoding: str, url: str) -> str:
    try:
        return extract_source(body, media_type, encoding, url).content
    except ExtractionRejectedError as exc:
        raise SourceFetchError(exc.code, str(exc)) from exc


__all__ = [
    "MAX_ADDRESS_SPACE_BYTES",
    "MAX_CPU_SECONDS",
    "MAX_EXTRACTED_CHARACTERS",
    "MAX_PDF_PAGES",
    "MAX_RESPONSE_BYTES",
    "MAX_WALL_SECONDS",
    "PublicSourceFetcher",
    "SourceFetchError",
]
