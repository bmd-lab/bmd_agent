from __future__ import annotations

import json
import subprocess

import pytest

from bmd_agent import cli
from bmd_agent.profiling import PerformanceProfiler, profiled_runner


class SequenceClock:
    def __init__(self, values: list[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return next(self._values)


class StepClock:
    def __init__(self, step: float = 0.25) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def test_profile_uses_monotonic_clock_and_keeps_nested_timings_inclusive() -> None:
    profiler = PerformanceProfiler(clock=SequenceClock([0, 1, 2, 5, 8, 9]))

    with profiler.activate():
        with profiler.phase("outer"):
            with profiler.phase("inner"):
                pass

    profile = profiler.snapshot()
    phases = {phase.name: phase for phase in profile.phases}

    assert profile.clock == "time.perf_counter"
    assert profile.total_elapsed_seconds == 9
    assert phases["outer"].elapsed_seconds == 7
    assert phases["inner"].elapsed_seconds == 3
    assert sum(phase.elapsed_seconds for phase in profile.phases) > profile.total_elapsed_seconds
    assert "inclusive" in profile.timing_semantics


def test_profiled_runners_count_existing_operations_bytes_and_wait_time() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        remote = command[-1]
        returncode = 1 if remote.startswith("test -f ") else 0
        stdout: bytes | str = b"payload" if not kwargs.get("text") else "payload"
        stderr: bytes | str = b"" if not kwargs.get("text") else ""
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    profiler = PerformanceProfiler(clock=StepClock())
    with profiler.activate():
        scheduler = profiled_runner(runner, role="scheduler")
        remote = profiled_runner(runner, role="remote")
        producer = profiled_runner(runner, role="producer")
        bmdex = profiled_runner(runner, role="bmdex")
        scheduler(["ssh", "-o", "ConnectTimeout=10", "host", "sacct -j 1"])
        remote(["ssh", "host", "cat -- /allowed/OUTCAR"])
        remote(["ssh", "host", "tail -c 128 -- /allowed/std_err.txt"])
        remote(["ssh", "host", "test -f /allowed/error.1.tar.gz"])
        remote(["ssh", "host", "test -d /allowed/stage_1"])
        remote(["ssh", "host", "stat -c %s -- /allowed/vasprun.xml"])
        remote(["ssh", "host", "awk 'fixed program' /allowed/OUTCAR"])
        remote(["ssh", "host", "ls /allowed"])
        producer(["python", "-m", "backend.calculations.capabilities"])
        with profiler.phase("bmdex_acquisition"):
            bmdex(["python", "-m", "bmdex.context"])

    profile = profiler.snapshot()
    counts = profile.operations.counts
    waits = profile.operations.elapsed_seconds

    assert len(calls) == 10
    assert counts["subprocess_invocations"] == 10
    assert counts["ssh_invocations"] == 8
    assert counts["ssh_connections"] == 8
    assert counts["ssh_exec_channels"] == 8
    assert counts["ssh_control_operations"] == 0
    assert counts["remote_commands"] == 8
    assert counts["remote_file_reads"] == 2
    assert counts["bounded_remote_file_reads"] == 1
    assert counts["existence_probes"] == 2
    assert counts["directory_probes"] == 1
    assert counts["stat_probes"] == 1
    assert counts["remote_extractor_operations"] == 1
    assert counts["directory_listing_operations"] == 1
    assert counts["archive_probes"] == 1
    assert counts["archive_probe_batches"] == 0
    assert counts["scheduler_operations"] == 1
    assert counts["producer_operations"] == 1
    assert counts["bmdex_operations"] == 1
    assert waits["ssh_wait"] > 0
    assert waits["archive_probe_wait"] > 0
    assert waits["bounded_remote_file_read_wait"] > 0
    assert profile.operations.bytes_transferred == 70
    assert profile.operations.ssh_bytes_transferred == 56
    assert profile.operations.failure_count == 0
    assert any(phase.name == "bmdex_acquisition" for phase in profile.phases)


def test_profiling_preserves_failures_and_records_timeout_without_masking() -> None:
    command = ["ssh", "host", "cat -- /allowed/OSZICAR"]

    def timeout_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        raise subprocess.TimeoutExpired(command, timeout=20, output=b"partial")

    profiler = PerformanceProfiler(clock=StepClock())
    with profiler.activate():
        runner = profiled_runner(timeout_runner, role="remote")
        with pytest.raises(subprocess.TimeoutExpired):
            runner(command, capture_output=True, check=True, timeout=20)

    profile = profiler.snapshot()
    assert profile.operations.failure_count == 1
    assert profile.operations.timeout_count == 1
    assert profile.operations.counts["remote_file_reads"] == 1
    assert profile.operations.bytes_transferred == len(b"partial")


def test_profile_model_is_json_safe() -> None:
    profiler = PerformanceProfiler(clock=SequenceClock([10, 12]))
    with profiler.activate():
        pass

    encoded = json.dumps(profiler.snapshot().to_dict(), allow_nan=False)
    payload = json.loads(encoded)

    assert payload["schema_version"] == 3
    assert payload["evidence_type"] == "agent_performance_telemetry"
    assert payload["total_elapsed_seconds"] == 2
    assert payload["operations"]["counts"]["ssh_invocations"] == 0
    assert payload["operations"]["counts"]["ssh_connections"] == 0
    assert "ssh_exec_channels" in payload["operations"]["count_semantics"]


def test_profile_output_is_opt_in_and_follows_unchanged_normal_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_show_job(
        job_id: str,
        registry: object,
        *,
        trajectory_json: bool,
        profiling: bool = False,
    ) -> int:
        assert job_id == "21906221"
        assert trajectory_json is False
        print("ordinary scientific conclusion")
        return 0

    monkeypatch.setattr(cli, "_show_job", fake_show_job)

    assert cli.show_job("21906221") == 0
    normal_output = capsys.readouterr().out
    assert "Performance profile" not in normal_output

    assert cli.show_job("21906221", profile=True) == 0
    profiled_output = capsys.readouterr().out
    assert profiled_output.startswith(normal_output + "\nPerformance profile")
    assert "developer telemetry" in profiled_output


def test_runner_is_unwrapped_when_no_profile_is_active() -> None:
    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(command, 0, stdout=b"", stderr=b"")

    assert profiled_runner(runner, role="remote") is runner
