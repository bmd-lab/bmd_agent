from dataclasses import dataclass
import re
import shlex
import subprocess
from typing import Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]

_PARTITION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SQUEUE_FORMAT = "%i|%u|%j|%t|%M|%R"


@dataclass
class SlurmJob:
    job_id: str
    user: str
    name: str
    state: str
    elapsed: str
    reason: str


def get_queue(
    ssh_host: str,
    partition: str,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> list[SlurmJob]:
    """Return jobs visible in a configured SLURM partition."""

    command = [
        "ssh",
        ssh_host,
        build_squeue_command(partition),
    ]

    result = runner(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    return parse_squeue_output(result.stdout)


def build_squeue_command(partition: str) -> str:
    """Build a shell-quoted read-only remote squeue command."""

    if not _PARTITION_RE.fullmatch(partition):
        raise ValueError("partition contains unsafe characters")

    return " ".join(
        [
            "squeue",
            "-p",
            shlex.quote(partition),
            "--noheader",
            shlex.quote(f"--format={_SQUEUE_FORMAT}"),
        ]
    )


def parse_squeue_output(output: str) -> list[SlurmJob]:
    """Parse pipe-delimited squeue output."""

    jobs: list[SlurmJob] = []

    for line in output.splitlines():
        if not line.strip():
            continue

        parts = line.split("|", maxsplit=5)

        if len(parts) != 6:
            continue

        jobs.append(
            SlurmJob(
                job_id=parts[0].strip(),
                user=parts[1].strip(),
                name=parts[2].strip(),
                state=parts[3].strip(),
                elapsed=parts[4].strip(),
                reason=parts[5].strip(),
            )
        )

    return jobs
