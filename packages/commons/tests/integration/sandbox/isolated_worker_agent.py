"""Synthetic concurrent miner workload, with no paid or external tool calls."""

import asyncio
import os

from harnyx_miner_sdk.decorators import entrypoint

loads = 1


@entrypoint("isolated")
async def shared(request: dict[str, object]) -> dict[str, int]:
    if request.get("block"):
        while True:
            pass
    await asyncio.sleep(float(request.get("delay", 0.03)))
    return {"pid": os.getpid(), "loads": loads}


# The workspace must already be active before any miner module code executes.
if os.getcwd().startswith("/sandbox/queries/query-"):
    import tempfile
    from pathlib import Path

    Path("module.txt").write_text("initialized")
    with tempfile.NamedTemporaryFile(delete=False) as temporary:
        module_temp = temporary.name


@entrypoint("workspace")
async def workspace(request: dict[str, object]) -> dict[str, object]:
    import tempfile
    from pathlib import Path

    Path("nested").mkdir()
    Path("nested/input").write_text("abcdef")
    Path("nested/input").rename("nested/output")
    with open("nested/output", "r+") as stream:
        stream.truncate(3)
    contents = Path("nested/output").read_text()
    listed = os.listdir("nested")
    Path("nested/output").unlink()
    Path("nested").rmdir()
    with tempfile.NamedTemporaryFile() as temporary:
        temp_parent = str(Path(temporary.name).parent)
    try:
        Path("/workspace/forbidden").write_text("no")
        root_read_only = False
    except OSError:
        root_read_only = True
    if request.get("fail"):
        raise ValueError("expected failure")
    await asyncio.sleep(float(request.get("delay", 0.05)))
    return {"cwd": os.getcwd(), "temp": temp_parent, "module_temp": str(Path(module_temp).parent),
            "module": Path("module.txt").read_text(), "contents": contents, "listed": listed,
            "root_read_only": root_read_only, "uid": os.getuid()}
