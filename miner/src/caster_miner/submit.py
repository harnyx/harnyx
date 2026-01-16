from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import bittensor as bt
import httpx
from dotenv import load_dotenv

_UPLOAD_PATH = "/v1/miners/scripts"
_MAX_AGENT_BYTES = 256_000


def _build_canonical_request(method: str, path_qs: str, body: bytes) -> bytes:
    normalized_method = (method or "GET").upper()
    normalized_path = path_qs or "/"
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = "\n".join((normalized_method, normalized_path, body_hash))
    return canonical.encode("utf-8")


def _load_agent_bytes(agent_path: Path) -> bytes:
    if not agent_path.exists():
        raise FileNotFoundError(f"agent path not found: {agent_path}")
    content = agent_path.read_bytes()
    if len(content) > _MAX_AGENT_BYTES:
        raise ValueError(f"agent script exceeds {_MAX_AGENT_BYTES} bytes ({len(content)} bytes)")
    try:
        source = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("agent script must be UTF-8 encoded") from exc
    try:
        compile(source, str(agent_path), "exec")
    except SyntaxError as exc:
        raise ValueError("agent script failed bytecode compilation") from exc
    return content


def _platform_base_url() -> str:
    load_dotenv(dotenv_path=Path(".env"), override=False)
    base_url = (os.getenv("PLATFORM_BASE_URL") or "").strip()
    if not base_url:
        raise RuntimeError(
            "PLATFORM_BASE_URL must be set (for example: http://localhost:8200)"
        )
    return base_url.rstrip("/")


def _authorization_header(wallet: bt.wallet, method: str, path_qs: str, body: bytes) -> str:
    canonical = _build_canonical_request(method, path_qs, body)
    signature = wallet.hotkey.sign(canonical)
    return f'Bittensor ss58="{wallet.hotkey.ss58_address}",sig="{signature.hex()}"'


def _summarize_response_text(response: httpx.Response, *, limit: int = 500) -> str:
    text = (response.text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _upload_agent(*, agent_path: Path, wallet_name: str, hotkey_name: str) -> dict[str, object]:
    content = _load_agent_bytes(agent_path)
    digest = hashlib.sha256(content).hexdigest()
    payload = {
        "script_b64": base64.b64encode(content).decode(),
        "sha256": digest,
    }
    body = json.dumps(payload).encode()

    path = _UPLOAD_PATH
    wallet = bt.wallet(name=wallet_name, hotkey=hotkey_name)
    authorization = _authorization_header(wallet, "POST", path, body)
    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json",
    }
    with httpx.Client(base_url=_platform_base_url(), timeout=10) as client:
        response = client.post(path, headers=headers, content=body)
    if response.status_code != 200:
        detail = _summarize_response_text(response)
        raise RuntimeError(f"script upload failed ({response.status_code}): {detail}")
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("platform response must be a JSON object")
    return cast(dict[str, object], data)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Upload a miner agent script to the platform (Bittensor-signed).")
    parser.add_argument("--agent-path", required=True, help="Path to the miner agent Python file.")
    parser.add_argument("--wallet-name", required=True, help="Bittensor wallet name (directory under ~/.bittensor).")
    parser.add_argument("--hotkey-name", required=True, help="Bittensor hotkey name (file under wallet hotkeys).")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        result = _upload_agent(
            agent_path=Path(args.agent_path),
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
        )
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(result))


__all__ = ["main"]
