from __future__ import annotations

from pathlib import Path

from harnyx_validator.runtime import resource_usage
from harnyx_validator.runtime.resource_usage import ValidatorResourceUsageProvider


def test_resource_usage_snapshot_reads_process_memory_cpu_and_disk(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cpu_snapshots = iter(((12.0, 120.0), (15.0, 125.0)))
    monkeypatch.setattr(resource_usage, "_read_cpu_snapshot", lambda: next(cpu_snapshots))
    monkeypatch.setattr(resource_usage, "_read_process_rss_bytes", lambda: 512 * 1024 * 1024)
    monkeypatch.setattr(resource_usage, "_read_total_memory_bytes", lambda: 2 * 1024 * 1024 * 1024)
    monkeypatch.setattr(resource_usage, "_read_cpu_capacity_cores", lambda: 4.0)
    monkeypatch.setattr(
        resource_usage,
        "_read_disk_usage",
        lambda path: (
            100 * 1024 * 1024 * 1024,
            25 * 1024 * 1024 * 1024,
            25.0,
        ),
    )

    provider = ValidatorResourceUsageProvider(disk_usage_path=tmp_path)

    snapshot = provider.snapshot()

    assert snapshot.cpu_percent == 60.0
    assert snapshot.cpu_capacity_cores == 4.0
    assert snapshot.memory_used_bytes == 512 * 1024 * 1024
    assert snapshot.memory_total_bytes == 2 * 1024 * 1024 * 1024
    assert snapshot.memory_percent == 25.0
    assert snapshot.disk_used_bytes == 25 * 1024 * 1024 * 1024
    assert snapshot.disk_total_bytes == 100 * 1024 * 1024 * 1024
    assert snapshot.disk_percent == 25.0
    assert snapshot.captured_at.tzinfo is not None


def test_read_total_memory_bytes_prefers_cgroup_v2_limit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text(str(2 * 1024 * 1024 * 1024), encoding="utf-8")
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_MEMORY_MAX_PATH", cgroup_v2)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_MEMORY_LIMIT_PATH", cgroup_v1)
    monkeypatch.setattr(resource_usage, "_read_host_total_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    assert resource_usage._read_total_memory_bytes() == 2 * 1024 * 1024 * 1024


def test_read_total_memory_bytes_prefers_cgroup_v1_limit_when_v2_is_unbounded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text("max", encoding="utf-8")
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_MEMORY_MAX_PATH", cgroup_v2)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_MEMORY_LIMIT_PATH", cgroup_v1)
    monkeypatch.setattr(resource_usage, "_read_host_total_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    assert resource_usage._read_total_memory_bytes() == 3 * 1024 * 1024 * 1024


def test_read_total_memory_bytes_falls_back_to_host_when_cgroup_limits_are_unbounded(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_v2 = tmp_path / "memory.max"
    cgroup_v2.write_text("max", encoding="utf-8")
    cgroup_v1 = tmp_path / "memory.limit_in_bytes"
    cgroup_v1.write_text(str(16 * 1024 * 1024 * 1024), encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_MEMORY_MAX_PATH", cgroup_v2)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_MEMORY_LIMIT_PATH", cgroup_v1)
    monkeypatch.setattr(resource_usage, "_read_host_total_memory_bytes", lambda: 16 * 1024 * 1024 * 1024)

    assert resource_usage._read_total_memory_bytes() == 16 * 1024 * 1024 * 1024


def test_read_cpu_capacity_cores_preserves_sub_one_core_cgroup_v2_quota(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cpu_max = tmp_path / "cpu.max"
    cpu_max.write_text("50000 100000", encoding="utf-8")
    _disable_proc_cgroup_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_CPU_MAX_PATH", cpu_max)
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_QUOTA_PATH", tmp_path / "missing-quota")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_PERIOD_PATH", tmp_path / "missing-period")
    monkeypatch.setattr(resource_usage.os, "sched_getaffinity", lambda _pid: set(range(8)))

    assert resource_usage._read_cpu_capacity_cores() == 0.5


def test_read_cpu_capacity_cores_matches_cgroup_v1_membership_to_mount_controller(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cpu_mount = tmp_path / "sys" / "fs" / "cgroup" / "cpu"
    cpuacct_mount = tmp_path / "sys" / "fs" / "cgroup" / "cpuacct"
    cpu_group = cpu_mount / "validator"
    cpuacct_group = cpuacct_mount / "unrelated"
    cpu_group.mkdir(parents=True)
    cpuacct_group.mkdir(parents=True)
    (cpu_group / "cpu.cfs_quota_us").write_text("400000", encoding="utf-8")
    (cpu_group / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
    (cpuacct_group / "cpu.cfs_quota_us").write_text("100000", encoding="utf-8")
    (cpuacct_group / "cpu.cfs_period_us").write_text("100000", encoding="utf-8")
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("2:cpu:/validator\n3:cpuacct:/unrelated\n", encoding="utf-8")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "\n".join(
            (
                f"1 0 0:1 / {cpu_mount} rw - cgroup cgroup rw,cpu",
                f"2 0 0:2 / {cpuacct_mount} rw - cgroup cgroup rw,cpuacct",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(resource_usage, "_PROC_SELF_CGROUP_PATH", proc_cgroup)
    monkeypatch.setattr(resource_usage, "_PROC_SELF_MOUNTINFO_PATH", mountinfo)
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_CPU_MAX_PATH", tmp_path / "missing-cpu.max")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_QUOTA_PATH", tmp_path / "missing-quota")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_PERIOD_PATH", tmp_path / "missing-period")
    monkeypatch.setattr(resource_usage.os, "sched_getaffinity", lambda _pid: set(range(8)))

    assert resource_usage._read_cpu_capacity_cores() == 4.0


def test_read_cpu_capacity_cores_reads_nested_cgroup_v2_cpu_max(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_mount = tmp_path / "sys" / "fs" / "cgroup"
    nested_cgroup = cgroup_mount / "kubepods.slice" / "pod.slice"
    nested_cgroup.mkdir(parents=True)
    (nested_cgroup / "cpu.max").write_text("250000 100000", encoding="utf-8")
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/kubepods.slice/pod.slice\n", encoding="utf-8")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 0 0:1 / {cgroup_mount} rw - cgroup2 cgroup rw\n", encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_PROC_SELF_CGROUP_PATH", proc_cgroup)
    monkeypatch.setattr(resource_usage, "_PROC_SELF_MOUNTINFO_PATH", mountinfo)
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_CPU_MAX_PATH", tmp_path / "missing-cpu.max")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_QUOTA_PATH", tmp_path / "missing-quota")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_PERIOD_PATH", tmp_path / "missing-period")
    monkeypatch.setattr(resource_usage.os, "sched_getaffinity", lambda _pid: set(range(8)))

    assert resource_usage._read_cpu_capacity_cores() == 2.5


def test_read_cpu_capacity_cores_uses_tightest_ancestor_cgroup_v2_quota(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cgroup_mount = tmp_path / "sys" / "fs" / "cgroup"
    parent_cgroup = cgroup_mount / "kubepods.slice"
    child_cgroup = parent_cgroup / "pod.slice"
    child_cgroup.mkdir(parents=True)
    (parent_cgroup / "cpu.max").write_text("100000 100000", encoding="utf-8")
    (child_cgroup / "cpu.max").write_text("max 100000", encoding="utf-8")
    proc_cgroup = tmp_path / "cgroup"
    proc_cgroup.write_text("0::/kubepods.slice/pod.slice\n", encoding="utf-8")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(f"1 0 0:1 / {cgroup_mount} rw - cgroup2 cgroup rw\n", encoding="utf-8")
    monkeypatch.setattr(resource_usage, "_PROC_SELF_CGROUP_PATH", proc_cgroup)
    monkeypatch.setattr(resource_usage, "_PROC_SELF_MOUNTINFO_PATH", mountinfo)
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_CPU_MAX_PATH", tmp_path / "missing-cpu.max")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_QUOTA_PATH", tmp_path / "missing-quota")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_PERIOD_PATH", tmp_path / "missing-period")
    monkeypatch.setattr(resource_usage.os, "sched_getaffinity", lambda _pid: set(range(8)))

    assert resource_usage._read_cpu_capacity_cores() == 1.0


def test_read_cpu_capacity_cores_falls_back_to_os_cpu_count(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _disable_proc_cgroup_discovery(monkeypatch, tmp_path)
    monkeypatch.setattr(resource_usage, "_CGROUP_V2_CPU_MAX_PATH", tmp_path / "missing-cpu.max")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_QUOTA_PATH", tmp_path / "missing-quota")
    monkeypatch.setattr(resource_usage, "_CGROUP_V1_CPU_PERIOD_PATH", tmp_path / "missing-period")
    monkeypatch.delattr(resource_usage.os, "sched_getaffinity")
    monkeypatch.setattr(resource_usage.os, "cpu_count", lambda: 16)

    assert resource_usage._read_cpu_capacity_cores() == 16.0


def _disable_proc_cgroup_discovery(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(resource_usage, "_PROC_SELF_CGROUP_PATH", tmp_path / "missing-cgroup")
    monkeypatch.setattr(resource_usage, "_PROC_SELF_MOUNTINFO_PATH", tmp_path / "missing-mountinfo")
