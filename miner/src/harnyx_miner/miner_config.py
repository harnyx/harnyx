from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import cast

import bittensor as bt
import httpx

from harnyx_miner.submit import _authorization_header, _platform_base_url

_CONFIG_PATH = "/v1/miner-config"
_SUPPORTED_PROVIDERS = frozenset({"chutes", "openrouter", "desearch", "parallel"})
_MIN_TASK_RETRY_COUNT = 0
_MAX_TASK_RETRY_COUNT = 3


def get_config(*, wallet_name: str, hotkey_name: str) -> dict[str, object]:
    return _request_json(
        method="GET",
        payload=None,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
    )


def put_provider_credential(
    *,
    provider: str,
    api_key: str,
    wallet_name: str,
    hotkey_name: str,
) -> dict[str, object]:
    normalized_provider = _normalize_provider(provider)
    normalized_api_key = api_key.strip()
    if not normalized_api_key:
        raise ValueError("provider API key must be non-empty")
    return _request_json(
        method="PUT",
        payload={
            "key": f"provider_credentials.{normalized_provider}",
            "value": normalized_api_key,
        },
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
    )


def delete_provider_credential(
    *,
    provider: str,
    wallet_name: str,
    hotkey_name: str,
) -> dict[str, object]:
    normalized_provider = _normalize_provider(provider)
    return _request_json(
        method="DELETE",
        payload={"key": f"provider_credentials.{normalized_provider}"},
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
    )


def put_task_retry_count(
    *,
    task_retry_count: int,
    wallet_name: str,
    hotkey_name: str,
) -> dict[str, object]:
    if task_retry_count < _MIN_TASK_RETRY_COUNT or task_retry_count > _MAX_TASK_RETRY_COUNT:
        raise ValueError("task retry count must be between 0 and 3")
    return _request_json(
        method="PUT",
        payload={"key": "task_retry_count", "value": str(task_retry_count)},
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
    )


def _request_json(
    *,
    method: str,
    payload: dict[str, str] | None,
    wallet_name: str,
    hotkey_name: str,
) -> dict[str, object]:
    body = b"" if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    headers = {
        "Authorization": _authorization_header(wallet, method, _CONFIG_PATH, body),
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    with httpx.Client(base_url=_platform_base_url(), timeout=10) as client:
        response = client.request(method, _CONFIG_PATH, headers=headers, content=body)
    if response.status_code != 200:
        raise RuntimeError(_failure_message(response))
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("platform response must be a JSON object")
    return cast(dict[str, object], data)


def _normalize_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in _SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(_SUPPORTED_PROVIDERS))
        raise ValueError(f"unsupported provider '{provider}'; expected one of: {supported}")
    return normalized


def _failure_message(response: httpx.Response) -> str:
    detail = _safe_error_code(response)
    if detail is None:
        return f"miner config request failed ({response.status_code})"
    return f"miner config request failed ({response.status_code}): {detail}"


def _safe_error_code(response: httpx.Response) -> str | None:
    try:
        data = response.json()
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    error_code = data.get("error_code")
    if not isinstance(error_code, str) or not error_code:
        return None
    return error_code


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Manage miner provider credentials (Bittensor-signed).")
    parser.add_argument("--wallet-name", required=True, help="Bittensor wallet name (directory under ~/.bittensor).")
    parser.add_argument("--hotkey-name", required=True, help="Bittensor hotkey name (file under wallet hotkeys).")
    parser.add_argument("--get", action="store_true", help="Read redacted miner config.")
    parser.add_argument("--provider", help="Provider to configure.")
    parser.add_argument("--api-key", help="Provider API key to upload.")
    parser.add_argument("--delete-provider", help="Provider credential to delete.")
    parser.add_argument("--task-retry-count", type=int, help="Miner task retry count, 0 through 3.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = _dispatch(args)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result))


def _dispatch(args: argparse.Namespace) -> dict[str, object]:
    wallet_name = str(args.wallet_name)
    hotkey_name = str(args.hotkey_name)
    requested_operations = sum(
        1
        for enabled in (
            bool(args.get),
            bool(args.provider and args.api_key),
            bool(args.delete_provider),
            args.task_retry_count is not None,
        )
        if enabled
    )
    if requested_operations != 1:
        raise ValueError(
            "choose exactly one of --get, --provider with --api-key, --delete-provider, or --task-retry-count"
        )
    if args.get:
        return get_config(wallet_name=wallet_name, hotkey_name=hotkey_name)
    if args.task_retry_count is not None:
        return put_task_retry_count(
            task_retry_count=int(args.task_retry_count),
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
        )
    if args.delete_provider:
        return delete_provider_credential(
            provider=str(args.delete_provider),
            wallet_name=wallet_name,
            hotkey_name=hotkey_name,
        )
    return put_provider_credential(
        provider=str(args.provider),
        api_key=str(args.api_key),
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
    )


__all__ = [
    "delete_provider_credential",
    "get_config",
    "main",
    "put_provider_credential",
    "put_task_retry_count",
]
