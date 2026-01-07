"""Agent used to validate sandbox seccomp enforcement."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any

from caster_commons.tools.decorators import entrypoint


@entrypoint("spawn_thread")
async def spawn_thread(request: Mapping[str, Any]) -> dict[str, Any]:
    """Attempt to start a thread; should be blocked by seccomp inside the worker."""

    del request
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    return {"thread": "started"}
