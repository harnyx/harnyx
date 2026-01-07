from __future__ import annotations

import os

os.environ.setdefault("EXTERNAL_CLIENT_RETRY_ATTEMPTS", "1")
os.environ.setdefault("EXTERNAL_CLIENT_RETRY_INITIAL_MS", "0")
os.environ.setdefault("EXTERNAL_CLIENT_RETRY_MAX_MS", "0")
os.environ.setdefault("EXTERNAL_CLIENT_RETRY_JITTER", "0")

