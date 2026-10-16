"""Read positive OOM evidence from this sandbox's own cgroup v2."""

from __future__ import annotations

import re
from pathlib import Path

from harnyx_miner_sdk.sandbox_protocol import SandboxResourceStatus


def _mount_path(value: str) -> Path:
    return Path(re.sub(r"\\([0-7]{3})", lambda match: chr(int(match[1], 8)), value))


class EvaluationMemoryState:
    """Latch OOM for the container lifetime, independent of compiler recovery."""

    def __init__(self, counter: Path | None = None) -> None:
        self.counter = counter or self._own_counter()
        self.oom_kill_count = 0
        self.draining = False
        self.refresh()

    @staticmethod
    def _own_counter() -> Path:
        membership = next(
            (line[3:] for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::")),
            None,
        )
        if membership is None:
            raise RuntimeError("evaluation requires cgroup v2 memory counters")
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            left, separator, right = line.partition(" - ")
            if not separator or right.split()[0] != "cgroup2":
                continue
            fields = left.split()
            mount_root, mount_point = _mount_path(fields[3]), _mount_path(fields[4])
            try:
                relative = Path(membership).relative_to(mount_root)
            except ValueError:
                continue
            candidate = mount_point / relative / "memory.events.local"
            if candidate.is_file():
                return candidate
        raise RuntimeError("own cgroup v2 memory.events.local is not visible")

    def refresh(self) -> SandboxResourceStatus:
        counts = dict(line.split() for line in self.counter.read_text().splitlines())
        count = int(counts["oom_kill"])
        if count < self.oom_kill_count:
            raise RuntimeError("sandbox OOM counter moved backwards")
        self.oom_kill_count = count
        self.draining |= count > 0
        return SandboxResourceStatus(counter_status="ready", oom_kill_count=count, draining=self.draining)
