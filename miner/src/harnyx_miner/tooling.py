"""Keep miner-owned provider credentials on the configured tooling endpoint."""

from uuid import UUID

import httpx


def require_tooling_url(search_url: str, assignment_id: UUID, platform_base_url: str | None) -> None:
    if platform_base_url is None:
        raise ValueError("PLATFORM_BASE_URL is required for credential-bearing tooling")
    for value in (platform_base_url, search_url):
        parsed = httpx.URL(value)
        if (
            parsed.scheme != "https"
            or not parsed.host
            or parsed.userinfo
            or any(char in value for char in "?#\\")
            or any(ord(char) < 33 or ord(char) > 126 for char in value)
        ):
            raise ValueError("tooling URL must be an unambiguous HTTPS URL")
    expected = httpx.URL(platform_base_url.rstrip("/") + f"/v1/endpoint-assignments/{assignment_id}/search")
    if httpx.URL(search_url) != expected:
        raise ValueError("search URL does not match the configured assignment tooling route")
