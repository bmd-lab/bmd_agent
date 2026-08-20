import subprocess

import pytest

from bmd_agent.resources.slurm import build_squeue_command, get_queue, parse_squeue_output


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
