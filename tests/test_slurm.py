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


SACCT_FORMAT = (
    "JobIDRaw,JobName%30,State,Elapsed,Start,End,Partition%20,ExitCode,Timelimit%20,"
    "NodeList%80,NNodes,NCPUS,AllocCPUS,TotalCPU,CPUTimeRAW,AllocTRES%120,ReqTRES%120,MaxRSS"
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
        "sacct -P -n -j 20893681 "
        f"--format={SACCT_FORMAT}"
    )
    assert "sbatch" not in command
    assert "scancel" not in command
    assert "scontrol" not in command


def test_parse_sacct_output_prefers_primary_job_row() -> None:
    output = (
        "20893681.batch|batch|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00|node-a|1|24|24|"
        "00:01:00|607020|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|100M\n"
        "20893681|run|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00|node-a|1|24|24|"
        "2-15:00:00|607020|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|\n"
        "20893681.0|vasp|COMPLETED|02:47:00|2026-08-21T12:15:00|"
        "2026-08-21T15:02:00|leeburton-pool|0:0|72:00:00|node-a|1|24|24|"
        "2-14:50:00|601200|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|4G\n"
    )

    record = parse_sacct_output("20893681.batch", output)

    assert record is not None
    assert record.job_id == "20893681"
    assert record.state == "COMPLETED"
    assert record.elapsed == "02:48:37"
    assert record.partition == "leeburton-pool"
    assert record.exit_code == "0:0"
    assert record.timelimit == "72:00:00"
    assert record.node_list == "node-a"
    assert record.node_count == 1
    assert record.cpu_count == 24
    assert record.allocated_cpus == 24
    assert record.total_cpu == "2-15:00:00"
    assert record.total_cpu_seconds == 226800
    assert record.cpu_time_raw == 607020
    assert record.alloc_tres == "billing=24,cpu=24,mem=128G,node=1"
    assert record.req_tres == "billing=24,cpu=24,mem=128G,node=1"
    assert record.max_rss == "4G"
    assert record.max_rss_bytes == 4 * 1024 ** 3
    assert record.max_rss_source == "20893681.0"
    assert record.cpu_efficiency == pytest.approx(226800 / (10117 * 24))
    assert [step.job_id for step in record.steps] == ["20893681.batch", "20893681.0"]
    assert record.steps[1].max_rss == "4G"


def test_parse_sacct_output_preserves_timeout_accounting() -> None:
    output = (
        "21101222|run|TIMEOUT|01:00:20|2026-08-27T22:25:18|"
        "2026-08-27T23:25:38|leeburton-pool|0:0|01:00:00|node-b|1|24|24|"
        "20:00:00|86976|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|\n"
    )

    record = parse_sacct_output("21101222", output)

    assert record is not None
    assert record.state == "TIMEOUT"
    assert record.elapsed == "01:00:20"
    assert record.timelimit == "01:00:00"
    assert record.node_list == "node-b"
    assert record.cpu_efficiency == pytest.approx(72000 / (3620 * 24))


def test_parse_sacct_output_handles_missing_optional_accounting_fields() -> None:
    output = (
        "20893681|run|COMPLETED|02:48:37|2026-08-21T12:14:10|"
        "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00\n"
    )

    record = parse_sacct_output("20893681", output)

    assert record is not None
    assert record.node_list is None
    assert record.allocated_cpus is None
    assert record.total_cpu is None
    assert record.cpu_time_raw is None
    assert record.alloc_tres is None
    assert record.req_tres is None
    assert record.max_rss is None
    assert record.cpu_efficiency is None


def test_parse_sacct_output_can_derive_allocation_from_tres() -> None:
    output = (
        "20893681|run|COMPLETED|01:00:00|2026-08-21T12:14:10|"
        "2026-08-21T13:14:10|leeburton-pool|0:0|02:00:00|node-c|||"
        "12:00:00|43200|billing=24,cpu=24,mem=128G,node=1|"
        "billing=24,cpu=24,mem=128G,node=1|\n"
    )

    record = parse_sacct_output("20893681", output)

    assert record is not None
    assert record.node_count == 1
    assert record.allocated_cpus == 24
    assert record.cpu_efficiency == pytest.approx(43200 / (3600 * 24))


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
                "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00\n"
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
                "sacct -P -n -j 20893681 "
                f"--format={SACCT_FORMAT}"
            ),
        ]
    ]
