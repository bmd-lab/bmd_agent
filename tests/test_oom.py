import json

import pytest

from bmd_agent import cli
from bmd_agent.resources.oom import (
    INSUFFICIENT_OOM_EVIDENCE,
    NO_OOM_EVIDENCE,
    OOM_ESTABLISHED,
    OOM_POSSIBLE,
    assess_oom_evidence,
    serialize_oom_evidence,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord, SlurmStepAccountingRecord


def accounting(**overrides: object) -> SlurmAccountingRecord:
    values: dict[str, object] = {
        "job_id": "21853598",
        "name": "vasp",
        "state": "FAILED",
        "elapsed": "1-07:24:17",
        "start": "2026-09-01T00:00:00",
        "end": "2026-09-02T07:24:17",
        "partition": "leeburton-pool",
        "exit_code": "1:0",
        "node_count": 1,
        "allocated_cpus": 24,
        "task_count": 24,
        "req_mem": "128Gn",
        "req_tres": "cpu=24,mem=128G,node=1",
        "alloc_tres": "cpu=24,mem=128G,node=1",
    }
    values.update(overrides)
    return SlurmAccountingRecord(**values)  # type: ignore[arg-type]


def step(**overrides: object) -> SlurmStepAccountingRecord:
    values: dict[str, object] = {
        "job_id_raw": "21853598.batch",
        "name": "batch",
        "state": "FAILED",
        "exit_code": "1:0",
        "node_count": 1,
        "allocated_cpus": 1,
        "task_count": 1,
        "req_mem": "128Gn",
        "req_tres": "cpu=1,mem=128G,node=1",
        "alloc_tres": "cpu=1,mem=128G,node=1",
    }
    values.update(overrides)
    return SlurmStepAccountingRecord(**values)  # type: ignore[arg-type]


def test_explicit_slurm_oom_state_establishes_oom() -> None:
    evidence = assess_oom_evidence(accounting(state="OUT_OF_MEMORY", reason="OutOfMemory"))

    assert evidence.assessment == OOM_ESTABLISHED
    assert evidence.scheduler_oom_state is True
    assert any(marker.kind == "scheduler_oom_state" for marker in evidence.explicit_evidence)


def test_explicit_batch_step_oom_state_is_considered() -> None:
    scheduler = accounting(steps=(step(state="OUT_OF_MEMORY", reason="OutOfMemory"),))

    evidence = assess_oom_evidence(scheduler)

    assert evidence.assessment == OOM_ESTABLISHED
    assert any(".batch" in marker.source for marker in evidence.explicit_evidence)


def test_explicit_cgroup_oom_kill_log_establishes_oom() -> None:
    evidence = assess_oom_evidence(
        accounting(),
        log_observations=(("slurm stderr", "Detected 1 oom-kill event(s) in step"),),
    )

    assert evidence.assessment == OOM_ESTABLISHED
    assert evidence.explicit_evidence[0].kind == "explicit_oom_marker"


@pytest.mark.parametrize(
    "message",
    (
        "malloc: Cannot allocate memory",
        "terminate called after throwing std::bad_alloc",
        "allocation failed for FFT workspace",
        "insufficient memory for operation",
    ),
)
def test_allocation_failure_is_suggestive_not_definitive(message: str) -> None:
    evidence = assess_oom_evidence(
        accounting(),
        log_observations=(("runtime", message),),
    )

    assert evidence.assessment == OOM_POSSIBLE
    assert evidence.explicit_evidence == ()
    assert evidence.suggestive_evidence[0].kind == "allocation_failure"


@pytest.mark.parametrize(
    "message",
    (
        "forrtl: error (78): process killed (SIGTERM)",
        "application terminated by SIGKILL",
    ),
)
def test_signal_alone_does_not_establish_oom(message: str) -> None:
    evidence = assess_oom_evidence(
        accounting(),
        log_observations=(("runtime", message),),
        inspected_log_sources=("runtime",),
    )

    assert evidence.assessment == NO_OOM_EVIDENCE
    assert evidence.explicit_evidence == ()
    assert evidence.suggestive_evidence == ()


def test_failed_exit_alone_does_not_establish_oom() -> None:
    evidence = assess_oom_evidence(accounting(state="FAILED", exit_code="1:0"))

    assert evidence.assessment == INSUFFICIENT_OOM_EVIDENCE
    assert evidence.scheduler_oom_state is False


def test_high_compatible_max_rss_is_memory_pressure_not_established_oom() -> None:
    scheduler = accounting(
        allocated_cpus=1,
        task_count=1,
        req_tres="cpu=1,mem=128G,node=1",
        alloc_tres="cpu=1,mem=128G,node=1",
        max_rss="124G",
    )

    evidence = assess_oom_evidence(scheduler)

    assert evidence.assessment == OOM_POSSIBLE
    assert evidence.memory_utilization_percent == pytest.approx(96.875)
    assert evidence.explicit_evidence == ()


def test_low_max_rss_has_no_positive_oom_evidence() -> None:
    scheduler = accounting(
        allocated_cpus=1,
        task_count=1,
        req_tres="cpu=1,mem=128G,node=1",
        alloc_tres="cpu=1,mem=128G,node=1",
        max_rss="32G",
    )

    evidence = assess_oom_evidence(scheduler)

    assert evidence.assessment == NO_OOM_EVIDENCE
    assert evidence.memory_utilization_percent == pytest.approx(25.0)


def test_missing_max_rss_is_explicitly_limited() -> None:
    evidence = assess_oom_evidence(accounting(max_rss=None))

    assert evidence.assessment == INSUFFICIENT_OOM_EVIDENCE
    assert evidence.maximum_rss is None
    assert "maximum RSS accounting was unavailable" in evidence.limitations


@pytest.mark.parametrize(
    ("req_mem", "allocated_cpus", "task_count", "node_count", "expected_scope"),
    (
        ("4Gc", 24, 24, 1, "per_cpu"),
        ("128Gn", 24, 1, 1, "per_node"),
        ("128G", 24, 24, 1, "unknown"),
    ),
)
def test_reqmem_scope_is_preserved(
    req_mem: str,
    allocated_cpus: int,
    task_count: int,
    node_count: int,
    expected_scope: str,
) -> None:
    scheduler = accounting(
        req_mem=req_mem,
        req_tres=None,
        alloc_tres=None,
        allocated_cpus=allocated_cpus,
        task_count=task_count,
        node_count=node_count,
        max_rss="1G",
    )

    evidence = assess_oom_evidence(scheduler)

    assert evidence.requested_memory is not None
    assert evidence.requested_memory.scope == expected_scope


def test_job_step_max_rss_is_preserved_without_incompatible_percentage() -> None:
    scheduler = accounting(steps=(step(max_rss="120G", task_count=24, allocated_cpus=24),))

    evidence = assess_oom_evidence(scheduler)

    assert evidence.maximum_rss is not None
    assert evidence.maximum_rss.raw_value == "120G"
    assert ".batch" in evidence.maximum_rss.source
    assert evidence.memory_utilization_percent is None
    assert any("not compared" in item for item in evidence.limitations)


def test_low_numeric_job_step_max_rss_supports_no_positive_oom_evidence() -> None:
    scheduler = accounting(
        steps=(
            step(
                job_id_raw="21853598.0",
                name="vasp_std",
                max_rss="28.80G",
                task_count=24,
                allocated_cpus=24,
            ),
        )
    )

    evidence = assess_oom_evidence(scheduler)

    assert evidence.assessment == NO_OOM_EVIDENCE
    assert evidence.maximum_rss is not None
    assert evidence.maximum_rss.raw_value == "28.80G"
    assert evidence.maximum_rss.source == "SLURM step 21853598.0 maximum RSS"
    assert evidence.memory_utilization_percent is None


def test_accounting_acquisition_failure_is_insufficient_evidence() -> None:
    evidence = assess_oom_evidence(
        None,
        source_limitations=("scheduler accounting timed out after 60 seconds",),
    )

    assert evidence.assessment == INSUFFICIENT_OOM_EVIDENCE
    assert evidence.sufficiency == "insufficient_evidence"
    assert "scheduler accounting was unavailable" in evidence.limitations
    assert "scheduler accounting timed out after 60 seconds" in evidence.limitations


def test_oom_evidence_is_json_safe_and_does_not_prescribe_memory_value() -> None:
    evidence = assess_oom_evidence(
        accounting(),
        log_observations=(("stderr", "Detected oom-kill event"),),
    )

    payload = serialize_oom_evidence(evidence)

    assert json.loads(json.dumps(payload))["assessment"] == OOM_ESTABLISHED
    assert not any("GB" in guidance or "GiB" in guidance for guidance in evidence.guidance)


@pytest.mark.parametrize(
    ("evidence", "expected"),
    (
        (
            assess_oom_evidence(
                accounting(state="OUT_OF_MEMORY"),
                inspected_log_sources=("stderr",),
            ),
            "explicit evidence:",
        ),
        (
            assess_oom_evidence(
                accounting(
                    allocated_cpus=1,
                    task_count=1,
                    max_rss="124G",
                    alloc_tres="cpu=1,mem=128G,node=1",
                ),
                inspected_log_sources=("stderr",),
            ),
            "scheduler-derived memory utilization: 96.9%",
        ),
        (
            assess_oom_evidence(accounting(), inspected_log_sources=("stderr",)),
            "positive OOM markers: none observed",
        ),
    ),
)
def test_cli_reports_established_possible_and_no_evidence_outcomes(
    evidence: object,
    expected: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_oom_evidence(evidence)  # type: ignore[arg-type]

    output = capsys.readouterr().out
    assert evidence.assessment in output  # type: ignore[attr-defined]
    assert expected in output
