from dataclasses import dataclass
import subprocess


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
) -> list[SlurmJob]:
    """Return jobs visible in a configured SLURM partition."""

    remote_command = (
        "squeue "
        f"-p {partition} "
        "--noheader "
        "'--format=%i|%u|%j|%t|%M|%R'"
    )

    command = [
        "ssh",
        ssh_host,
        remote_command,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )

    jobs: list[SlurmJob] = []

    for line in result.stdout.splitlines():
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
