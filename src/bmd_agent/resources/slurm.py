from dataclasses import dataclass
import subprocess


SSH_HOST = "powerslurm-bmdguest"
PARTITION = "leeburton-pool"


@dataclass
class SlurmJob:
    job_id: str
    user: str
    name: str
    state: str
    elapsed: str
    reason: str


def get_queue() -> list[SlurmJob]:
    """Return jobs visible in the BMD PowerSLURM partition."""

    remote_command = (
        "squeue "
        f"-p {PARTITION} "
        "--noheader "
        "'--format=%i|%u|%j|%t|%M|%R'"
    )

    command = [
        "ssh",
        SSH_HOST,
        remote_command,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )

    jobs = []

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
