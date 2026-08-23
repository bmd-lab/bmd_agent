import subprocess

import pytest

from bmd_agent.resources.slurm import (
    build_sacct_command,
    build_squeue_command,
    get_job_accounting,
    get_queue,
    normalize_job_id,
    parse_sacct_output,
    parse_squeue_output,
)


SQUEUE_OUTPUT = """\
123|alice|relax_NaCl|R|00:12|None
124|bob|static_Si|PD|00:00|Priority
"""


def test_parse_squeue_output() -> None:
    jobs = parse_squeue_output(SQUEUE_OUTPUT)

    assert len(jobs) == 2
    assert jobs[0].job_id == "123"
    assert jobs[0].user == "alice"
    assert jobs[0].state == "R"
    assert jobs[1].reason == "Priority"


def test_parse_squeue_output_ignores_malformed_lines() -> None:
    jobs = parse_squeue_output("bad line\n" + SQUEUE_OUTPUT)

    assert [job.job_id for job in jobs] == ["123", "124"]


def test_build_squeue_command_rejects_unsafe_partition() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        build_squeue_command("pool;scancel 1")


def test_get_queue_uses_mocked_ssh_transport() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(command, 0, stdout=SQUEUE_OUTPUT, stderr="")

    jobs = get_queue("powerslurm-bmdguest", "leeburton-pool", runner=runner)

    assert [job.job_id for job in jobs] == ["123", "124"]
    assert calls == [
        [
            "ssh",
            "powerslurm-bmdguest",
            "squeue -p leeburton-pool --noheader '--format=%i|%u|%j|%t|%M|%R'",
        ]
    ]


def test_job_id_validation_strips_safe_slurm_suffixes() -> None:
    assert normalize_job_id("20893681") == "20893681"
    assert normalize_job_id("20893681.batch") == "20893681"

    with pytest.raises(ValueError, match="unsafe"):
        normalize_job_id("20893681;scancel 1")


def test_build_sacct_command_is_fixed_and_read_only() -> None:
    command = build_sacct_command("20893681")

    assert command == (
        "sacct -X -P -n -j 20893681 "
        "--format=JobIDRaw,JobName%30,State,Elapsed,Start,End,Partition%20,ExitCode"
    )
    assert "sbatch" not in command
    assert "scancel" not in command


def test_parse_sacct_output_prefers_primary_job_row() -> None:
    output = (
        "20893681.batch|batch|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0\n"
        "20893681|run|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0\n"
    )

    record = parse_sacct_output("20893681.batch", output)

    assert record is not None
    assert record.job_id == "20893681"
    assert record.state == "COMPLETED"
    assert record.elapsed == "02:48:37"
    assert record.partition == "leeburton-pool"
    assert record.exit_code == "0:0"


def test_get_job_accounting_uses_mocked_ssh_transport() -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["check"] is True
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "20893681|run|COMPLETED|02:48:37|2026-08-21T12:14:10|"
                "2026-08-21T15:02:47|leeburton-pool|0:0\n"
            ),
            stderr="",
        )

    record = get_job_accounting("powerslurm-bmdguest", "20893681", runner=runner)

    assert record is not None
    assert record.state == "COMPLETED"
    assert calls == [
        [
            "ssh",
            "powerslurm-bmdguest",
            (
                "sacct -X -P -n -j 20893681 "
                "--format=JobIDRaw,JobName%30,State,Elapsed,Start,End,Partition%20,ExitCode"
            ),
        ]
    ]
