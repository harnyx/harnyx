import asyncio
import io
import json
import socket
import ssl
import zipfile
from types import SimpleNamespace

import fitz
import pytest
from openpyxl import Workbook

from harnyx_commons import source_extractor_worker
from harnyx_commons.domain_tweak_generation import source_fetch
from harnyx_commons.domain_tweak_generation.source_fetch import (
    PublicSourceFetcher,
    SourceFetchError,
    _extract_content,
    _extract_isolated,
    _fetch_complete_body,
    _public_addresses,
    _validated_document_url,
    _validated_url,
)


def test_source_url_and_dns_reject_credentials_and_private_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: direct fetch must not become an SSRF path."""
    with pytest.raises(SourceFetchError, match="credentials"):
        _validated_url("https://user:secret@example.com/report")

    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))],
    )
    with pytest.raises(SourceFetchError, match="non-public"):
        _public_addresses("example.com", 443)


@pytest.mark.parametrize(
    "url,kind",
    [
        ("https://api.example.com/report", "html"),
        ("https://example.com/api/report", "html"),
        ("https://example.com/graphql/report", "html"),
        ("https://example.com/rest/report", "html"),
        ("https://example.com/report?token=secret", "html"),
        ("https://example.com/report.json?version=1", "static_json"),
        ("https://example.com/report", "static_json"),
    ],
)
def test_document_url_rejects_exact_api_and_credential_shapes(url: str, kind: str) -> None:
    """Future failure: model declarations must not bypass the host's pre-connect document boundary."""
    with pytest.raises(SourceFetchError) as captured:
        _validated_document_url(url, kind)  # type: ignore[arg-type]

    assert captured.value.code == "source_fetch_rejected"


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/api%2Freport",
        "https://example.com/%2Fapi%2Freport",
        "https://example.com/%61pi/report",
        "https://example.com/%2561pi/report",
        "https://example.com/%67raphql/report",
        "https://example.com/re%73t/report",
        "https://example.com/report%",
        "https://example.com/report?%74oken=secret",
        "https://example.com/report?%2574oken=secret",
    ],
)
def test_document_url_rejects_encoded_api_and_credential_shapes(url: str) -> None:
    """Future failure: URL encoding must not hide a prohibited source shape from pre-connect validation."""
    with pytest.raises(SourceFetchError) as captured:
        _validated_document_url(url, "html")

    assert captured.value.code == "source_fetch_rejected"


def test_redirect_revalidates_encoded_api_path_before_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: a public redirect must not connect to an encoded API path."""
    connections: list[_RedirectConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: ("93.184.216.34",))
    monkeypatch.setattr(
        source_fetch,
        "_PinnedHTTPSConnection",
        lambda *_args, **_kwargs: connections.append(_RedirectConnection()) or connections[-1],
    )

    with pytest.raises(SourceFetchError) as captured:
        _fetch_complete_body("https://example.com/report", "html")

    assert captured.value.code == "source_fetch_rejected"
    assert len(connections) == 1


class _RedirectSocket:
    def getpeername(self) -> tuple[str, int]:
        return ("93.184.216.34", 443)


class _RedirectResponse:
    status = 302

    def getheader(self, name: str) -> str | None:
        return "/%252Fapi%252Freport" if name == "Location" else None


class _RedirectConnection:
    def __init__(self) -> None:
        self.sock = _RedirectSocket()

    def connect(self) -> None:
        return None

    def request(self, *_args: object, **_kwargs: object) -> None:
        return None

    def getresponse(self) -> _RedirectResponse:
        return _RedirectResponse()

    def close(self) -> None:
        return None


class _FetchedSocket:
    def __init__(self, address: str) -> None:
        self._address = address

    def getpeername(self) -> tuple[str, int]:
        return (self._address, 443)


class _FetchedResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        location: str | None = None,
        body_error: OSError | None = None,
    ) -> None:
        self.status = status
        self._location = location
        self._body = io.BytesIO(b"<html>reachable</html>")
        self._body_error = body_error

    def getheader(self, name: str) -> str | None:
        if name == "Location":
            return self._location
        return "text/html" if name == "Content-Type" else None

    def read(self, size: int) -> bytes:
        if self._body_error is not None:
            raise self._body_error
        return self._body.read(size)


class _AddressConnection:
    def __init__(
        self,
        address: str,
        *,
        connection_error: BaseException | None = None,
        request_error: OSError | None = None,
        response_error: OSError | None = None,
        body_error: OSError | None = None,
        response_status: int = 200,
        response_location: str | None = None,
    ) -> None:
        self.sock = _FetchedSocket(address)
        self._connection_error = connection_error
        self._request_error = request_error
        self._response_error = response_error
        self._body_error = body_error
        self._response_status = response_status
        self._response_location = response_location
        self.closed = False

    def connect(self) -> None:
        if self._connection_error is not None:
            raise self._connection_error

    def request(self, *_args: object, **_kwargs: object) -> None:
        if self._request_error is not None:
            raise self._request_error

    def getresponse(self) -> _FetchedResponse:
        if self._response_error is not None:
            raise self._response_error
        return _FetchedResponse(
            status=self._response_status,
            location=self._response_location,
            body_error=self._body_error,
        )

    def close(self) -> None:
        self.closed = True


def test_fetch_tries_next_validated_public_address_after_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: one unreachable address must not hide a reachable address from the same safe DNS set."""
    addresses = ("2600:1406:bc00:53::b81e:94ce", "93.184.216.34")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(
            address,
            connection_error=OSError(101, "Network is unreachable") if address == addresses[0] else None,
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    fetched = _fetch_complete_body("https://example.com/report", "html")

    assert fetched.body == b"<html>reachable</html>"
    assert [connection.sock.getpeername()[0] for connection in connections] == list(addresses)
    assert all(connection.closed for connection in connections)


def test_fetch_reports_unavailable_only_after_every_validated_address_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: the final source error must represent exhaustion of the safe DNS set."""
    addresses = ("2600:1406:bc00:53::b81e:94ce", "93.184.216.34")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(address, connection_error=OSError(101, f"unreachable {address}"))
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match=f"unreachable {addresses[-1]}") as captured:
        _fetch_complete_body("https://example.com/report", "html")

    assert captured.value.code == "source_unavailable"
    assert [connection.sock.getpeername()[0] for connection in connections] == list(addresses)
    assert all(connection.closed for connection in connections)


def test_fetch_does_not_fallback_after_dns_set_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: DNS rebinding detection must stop instead of connecting to another validated address."""
    addresses = ("93.184.216.34", "93.184.216.35")
    resolutions = iter((addresses, ("93.184.216.35",)))
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: next(resolutions))

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(address)
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match="DNS resolution changed") as captured:
        _fetch_complete_body("https://example.com/report", "html")

    assert captured.value.code == "source_fetch_rejected"
    assert len(connections) == 1
    assert connections[0].closed


def test_fetch_does_not_fallback_after_connected_peer_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: a connected peer outside the validated DNS set must stop acquisition."""
    addresses = ("93.184.216.34", "93.184.216.35")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, _address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection("93.184.216.99")
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match="connected peer") as captured:
        _fetch_complete_body("https://example.com/report", "html")

    assert captured.value.code == "source_fetch_rejected"
    assert len(connections) == 1
    assert connections[0].closed


def test_fetch_rejects_mixed_public_and_private_dns_before_connect(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: one public answer must not make a mixed unsafe DNS set connectable."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    connection_attempted = False

    def create_connection(*_args: object, **_kwargs: object) -> _AddressConnection:
        nonlocal connection_attempted
        connection_attempted = True
        return _AddressConnection("93.184.216.34")

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match="non-public") as captured:
        _fetch_complete_body("https://example.com/report", "html")

    assert captured.value.code == "source_fetch_rejected"
    assert not connection_attempted


def test_cross_host_redirect_uses_the_new_hosts_validated_dns_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: redirect acquisition must not reuse the origin host's pinned addresses."""
    address_sets = {
        "example.com": ("93.184.216.34",),
        "reports.example.org": ("93.184.216.35",),
    }
    resolutions: list[tuple[str, int]] = []
    connections: list[tuple[str, str, _AddressConnection]] = []

    def resolve(host: str, port: int) -> tuple[str, ...]:
        resolutions.append((host, port))
        return address_sets[host]

    def create_connection(host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(
            address,
            response_status=302 if host == "example.com" else 200,
            response_location="https://reports.example.org/final" if host == "example.com" else None,
        )
        connections.append((host, address, connection))
        return connection

    monkeypatch.setattr(source_fetch, "_public_addresses", resolve)
    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    fetched = _fetch_complete_body("https://example.com/report", "html")

    assert fetched.final_url == "https://reports.example.org/final"
    assert resolutions == [
        ("example.com", 443),
        ("example.com", 443),
        ("reports.example.org", 443),
        ("reports.example.org", 443),
    ]
    assert [(host, address) for host, address, _connection in connections] == [
        ("example.com", "93.184.216.34"),
        ("reports.example.org", "93.184.216.35"),
    ]
    assert all(connection.closed for _host, _address, connection in connections)


def test_fetch_does_not_try_another_address_after_http_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: address fallback must not retry an application-level source failure."""
    addresses = ("93.184.216.34", "93.184.216.35")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(address, response_status=503)
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match="HTTP 503"):
        _fetch_complete_body("https://example.com/report", "html")

    assert len(connections) == 1
    assert connections[0].closed


def test_fetch_does_not_try_another_address_after_tls_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: address fallback must not bypass a TLS security failure."""
    addresses = ("93.184.216.34", "93.184.216.35")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(address, connection_error=ssl.SSLError("TLS validation failed"))
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match="TLS validation failed"):
        _fetch_complete_body("https://example.com/report", "html")

    assert len(connections) == 1
    assert connections[0].closed


@pytest.mark.parametrize("failure_point", ("request", "response", "body"))
def test_fetch_does_not_try_another_address_after_connected_io_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    """Future failure: fallback must not replay a request after connection establishment succeeded."""
    addresses = ("93.184.216.34", "93.184.216.35")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        error = OSError(f"{failure_point} failed")
        connection = _AddressConnection(
            address,
            request_error=error if failure_point == "request" else None,
            response_error=error if failure_point == "response" else None,
            body_error=error if failure_point == "body" else None,
        )
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(SourceFetchError, match=f"{failure_point} failed") as captured:
        _fetch_complete_body("https://example.com/report", "html")

    assert captured.value.code == "source_unavailable"
    assert len(connections) == 1
    assert connections[0].closed


def test_fetch_closes_and_propagates_non_os_connect_failure_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Future failure: unexpected connect failures must close once without becoming address fallback."""
    addresses = ("93.184.216.34", "93.184.216.35")
    connections: list[_AddressConnection] = []

    monkeypatch.setattr(source_fetch, "_public_addresses", lambda _host, _port: addresses)

    def create_connection(_host: str, _port: int, address: str, *, timeout: float) -> _AddressConnection:
        del timeout
        connection = _AddressConnection(address, connection_error=KeyboardInterrupt("connect interrupted"))
        connections.append(connection)
        return connection

    monkeypatch.setattr(source_fetch, "_PinnedHTTPSConnection", create_connection)

    with pytest.raises(KeyboardInterrupt, match="connect interrupted"):
        _fetch_complete_body("https://example.com/report", "html")

    assert len(connections) == 1
    assert connections[0].closed


def test_https_connection_closes_raw_socket_when_tls_wrapping_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: a TLS error before self.sock assignment must not leak the raw socket."""

    class RawSocket:
        closed = False

        def close(self) -> None:
            self.closed = True

    class FailingTlsContext:
        def wrap_socket(self, raw: RawSocket, *, server_hostname: str) -> None:
            del raw, server_hostname
            raise ssl.SSLError("TLS validation failed")

    raw = RawSocket()
    monkeypatch.setattr(source_fetch.socket, "create_connection", lambda *_args, **_kwargs: raw)
    connection = source_fetch._PinnedHTTPSConnection("example.com", 443, "93.184.216.34", timeout=60.0)
    connection._ssl_context = FailingTlsContext()  # type: ignore[assignment]

    with pytest.raises(ssl.SSLError, match="TLS validation failed"):
        connection.connect()

    assert raw.closed


@pytest.mark.parametrize(
    "url,kind",
    [
        ("https://example.com/capitals/report.html", "html"),
        ("https://example.com/download?id=42", "pdf"),
        ("https://example.com/files/report.xlsx", "xlsx"),
        ("https://example.com/files/report.json", "static_json"),
        ("https://example.com/rapid/report", "html"),
        ("https://example.com/100%25-complete.pdf", "pdf"),
        ("https://example.com/report?note=100%25%20complete", "html"),
        ("https://example.com/report?name=%26token=secret", "html"),
    ],
)
def test_document_url_accepts_ordinary_public_document_shapes(url: str, kind: str) -> None:
    _validated_document_url(url, kind)  # type: ignore[arg-type]


def test_html_extraction_projects_table_headers_and_rows() -> None:
    """Future failure: selected rows need visible column bindings in reference evidence."""
    content = _extract_content(
        b"<html><table><tr><th>Name</th><th>Value</th></tr><tr><td>A</td><td>7</td></tr></table></html>",
        "text/html",
        "",
        "https://example.com",
    )
    assert "HEADER\tName\tValue" in content
    assert "ROW\tA\t7" in content


def test_html_extraction_retains_complete_normalized_navigation_links() -> None:
    """Future failure: dossier inspection must not lose non-anchor or relative navigation targets."""
    extracted = source_extractor_worker.extract_source(
        b"""
        <html><head>
          <link href="/annual.css"><script src="scripts/annual.js"></script>
        </head><body>
          <a href="reports/2025.html"> Annual   report </a>
          <a href="reports/2025.html">Annual report</a>
          <form action="/annual/search"><button>Search records</button></form>
        </body></html>
        """,
        "text/html",
        "",
        "https://example.com/archive/index.html",
    )

    assert tuple((link.url, link.text) for link in extracted.links) == (
        ("https://example.com/archive/reports/2025.html", "Annual report"),
        ("https://example.com/annual.css", ""),
        ("https://example.com/archive/scripts/annual.js", ""),
        ("https://example.com/annual/search", "Search records"),
    )


@pytest.mark.parametrize(
    ("base_href", "expected_url"),
    [
        ("../published/", "https://example.com/published/report.html"),
        ("https://cdn.example.net/records/", "https://cdn.example.net/records/report.html"),
    ],
)
def test_html_extraction_resolves_navigation_against_the_document_base(
    base_href: str,
    expected_url: str,
) -> None:
    """Future failure: retained navigation must resolve as the HTML document resolves it."""
    extracted = source_extractor_worker.extract_source(
        f"""
        <html><head>
          <base target="_blank"><base href="{base_href}"><base href="https://ignored.example/">
        </head><body>
          <a href="report.html">Annual report</a>
        </body></html>
        """.encode(),
        "text/html",
        "",
        "https://example.com/archive/index.html",
    )

    assert tuple((link.url, link.text) for link in extracted.links) == ((expected_url, "Annual report"),)


@pytest.mark.parametrize(
    ("limit_name", "html", "message"),
    [
        (
            "MAX_EXTRACTED_LINK_URL_CHARACTERS",
            b'<html><body><a href="/path-that-is-too-long">report</a></body></html>',
            "source link URL exceeds extraction limit",
        ),
        (
            "MAX_EXTRACTED_LINK_TEXT_CHARACTERS",
            b'<html><body><a href="/report">text that is too long</a></body></html>',
            "source link text exceeds extraction limit",
        ),
    ],
)
def test_html_extraction_rejects_oversized_individual_link_metadata(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    html: bytes,
    message: str,
) -> None:
    """Future failure: one hostile link must not escape the bounded source-extraction envelope."""
    monkeypatch.setattr(source_extractor_worker, "MAX_EXTRACTED_LINK_URL_CHARACTERS", 10_000)
    monkeypatch.setattr(source_extractor_worker, "MAX_EXTRACTED_LINK_TEXT_CHARACTERS", 10_000)
    monkeypatch.setattr(source_extractor_worker, limit_name, 20)

    with pytest.raises(source_extractor_worker.ExtractionRejectedError, match=message):
        source_extractor_worker.extract_source(html, "text/html", "", "https://example.com")


def test_extraction_envelope_rejects_link_metadata_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: link navigation must fail visibly instead of silently dropping later targets."""
    written: list[bytes] = []
    monkeypatch.setattr(source_extractor_worker, "_set_resource_limits", lambda: None)
    monkeypatch.setattr(source_extractor_worker, "MAX_EXTRACTED_ENVELOPE_CHARACTERS", 100)
    monkeypatch.setattr(
        source_extractor_worker,
        "extract_source",
        lambda *_args: source_extractor_worker.ExtractedSource(
            content="body",
            links=(source_extractor_worker.ExtractedLink(url="https://example.com", text="x" * 200),),
        ),
    )
    monkeypatch.setattr(source_extractor_worker, "_write", written.append)
    monkeypatch.setattr(source_extractor_worker.sys, "argv", ["worker", "text/html", "", "https://example.com"])
    monkeypatch.setattr(source_extractor_worker.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"body")))

    source_extractor_worker.main()

    assert len(written) == 1 and written[0].startswith(b"E")
    assert json.loads(written[0][1:])["code"] == "source_extraction_limit"


@pytest.mark.anyio
async def test_isolated_pdf_extractor_has_enough_headroom_inside_its_512_mib_limit() -> None:
    """Future failure: the worker baseline must not consume its whole address-space allowance before parsing."""
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Alpha 1200")
    body = document.tobytes()
    document.close()

    extracted = await _extract_isolated(body, "application/pdf", "", "https://example.com/report.pdf")

    assert "Alpha 1200" in extracted.content


@pytest.mark.anyio
async def test_extracted_character_limit_rejects_the_whole_source_without_a_prefix() -> None:
    """Future failure: an oversized source must never become apparently valid truncated evidence."""
    with pytest.raises(SourceFetchError) as captured:
        await _extract_isolated(
            b"x" * (source_extractor_worker.MAX_EXTRACTED_CHARACTERS + 1),
            "text/plain",
            "",
            "https://example.com/report.txt",
        )

    assert captured.value.code == "source_extraction_limit"


def test_pdf_page_limit_rejects_the_complete_document(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: the PDF guard must reject rather than extract a permitted prefix."""
    monkeypatch.setattr(source_extractor_worker, "MAX_PDF_PAGES", 1)
    document = fitz.open()
    document.new_page().insert_text((72, 72), "first")
    document.new_page().insert_text((72, 72), "second")
    body = document.tobytes()
    document.close()

    with pytest.raises(SourceFetchError) as captured:
        _extract_content(body, "application/pdf", "", "https://example.com/report.pdf")

    assert captured.value.code == "source_extraction_limit"


def test_malformed_declared_json_is_an_explicit_source_failure() -> None:
    """Future failure: an extractor parse failure must not silently degrade into raw evidence."""
    with pytest.raises(SourceFetchError) as captured:
        _extract_content(b'{"broken":', "application/json", "", "https://example.com/report.json")

    assert captured.value.code == "source_unavailable"


def test_xlsx_extraction_preserves_sheet_row_and_cell_identity() -> None:
    """Future failure: spreadsheet evidence must remain addressable without evaluating formulas."""
    workbook = Workbook()
    active = workbook.active
    active.title = "Published table"
    active.append(("Name", "Value"))
    active.append(("Alpha", 1200))
    active["C2"] = "=B2*2"
    hidden = workbook.create_sheet("Notes")
    hidden.sheet_state = "hidden"
    hidden.append(("Source", "Annual report"))
    payload = io.BytesIO()
    workbook.save(payload)
    workbook.close()

    content = _extract_content(
        payload.getvalue(),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "",
        "https://example.com/report.xlsx",
    )

    assert 'XLSX_WORKSHEET\ttitle="Published table"\tstate=visible' in content
    assert 'XLSX_ROW\tworksheet="Published table"\trow=2\tA2=Alpha\tB2=1200' in content
    assert "C2=" not in content
    assert 'XLSX_WORKSHEET\ttitle="Notes"\tstate=hidden' in content


def test_xlsx_preflight_rejects_macro_payload() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("xl/vbaProject.bin", b"macro")

    with pytest.raises(SourceFetchError) as captured:
        _extract_content(
            payload.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "",
            "https://example.com/report.xlsx",
        )

    assert captured.value.code == "source_unavailable"


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "entries", "message"),
    [
        ("MAX_XLSX_ZIP_ENTRIES", 1, (("one.xml", b"1"), ("two.xml", b"2")), "too many entries"),
        ("MAX_XLSX_UNCOMPRESSED_BYTES", 3, (("large.xml", b"1234"),), "128 MiB"),
    ],
)
def test_xlsx_preflight_rejects_entry_and_expansion_limits(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    entries: tuple[tuple[str, bytes], ...],
    message: str,
) -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        for name, content in entries:
            archive.writestr(name, content)
    monkeypatch.setattr(source_extractor_worker, limit_name, limit_value)

    with pytest.raises(source_extractor_worker.ExtractionRejectedError, match=message) as captured:
        source_extractor_worker._preflight_xlsx_zip(payload.getvalue())

    assert captured.value.code == "source_extraction_limit"


def test_xlsx_preflight_rejects_unsafe_compression_ratio() -> None:
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("large.xml", b"0" * 20_000)

    with pytest.raises(source_extractor_worker.ExtractionRejectedError, match="compression ratio") as captured:
        source_extractor_worker._preflight_xlsx_zip(payload.getvalue())

    assert captured.value.code == "source_extraction_limit"


def test_xlsx_preflight_rejects_encrypted_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _EncryptedEntry:
        flag_bits = 0x1
        filename = "xl/workbook.xml"
        file_size = 1
        compress_size = 1

    class _EncryptedArchive:
        def __enter__(self) -> "_EncryptedArchive":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def infolist(self) -> list[_EncryptedEntry]:
            return [_EncryptedEntry()]

    monkeypatch.setattr(source_extractor_worker.zipfile, "ZipFile", lambda _body: _EncryptedArchive())

    with pytest.raises(source_extractor_worker.ExtractionRejectedError, match="encrypted") as captured:
        source_extractor_worker._preflight_xlsx_zip(b"encrypted")

    assert captured.value.code == "source_unavailable"


def test_xlsx_preflight_rejects_malformed_archive() -> None:
    with pytest.raises(SourceFetchError) as captured:
        _extract_content(
            b"not-a-zip",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "",
            "https://example.com/report.xlsx",
        )

    assert captured.value.code == "source_unavailable"


class _BlockingProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.started = asyncio.Event()
        self.released = asyncio.Event()
        self.killed = False

    async def communicate(self, _input: bytes | None = None) -> tuple[bytes, bytes]:
        self.started.set()
        await self.released.wait()
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9
        self.released.set()


@pytest.mark.anyio
async def test_fetch_cancellation_kills_and_drains_the_source_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Future failure: cancelled candidates must not leave source threads or sockets alive."""
    process = _BlockingProcess()

    async def create_process(*_args: object, **_kwargs: object) -> _BlockingProcess:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    task = asyncio.create_task(PublicSourceFetcher().fetch("https://example.com/report", document_kind="html"))
    await process.started.wait()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed
