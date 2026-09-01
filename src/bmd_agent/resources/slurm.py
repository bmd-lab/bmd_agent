from dataclasses import dataclass
import re
import shlex
import subprocess
from typing import Callable


Runner = Callable[..., subprocess.CompletedProcess[str]]

_PARTITION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_JOB_ID_RE = re.compile(r"^\d+(?:_\d+)?(?:\.(?:batch|extern))?$")
_SQUEUE_FORMAT = "%i|%u|%j|%t|%M|%R"
_SACCT_FIELDS = (
    "JobIDRaw",
    "JobName%30",
    "User%20",
    "Account%30",
    "State",
    "ExitCode",
    "Elapsed",
    "ElapsedRaw",
    "Start",
    "End",
    "Partition%20",
    "Timelimit%20",
    "NodeList%80",
    "NNodes",
    "AllocCPUS",
    "NTasks",
    "ReqMem",
    "ReqTRES%120",
    "AllocTRES%120",
    "TotalCPU",
    "CPUTimeRAW",
    "WorkDir%160",
)
_SACCT_FORMAT = ",".join(_SACCT_FIELDS)


@dataclass
class SlurmJob:
    job_id: str
    user: str
    name: str
    state: str
    elapsed: str
    reason: str


@dataclass
class SlurmAccountingRecord:
    job_id: str
    name: str
    state: str
    elapsed: str
    start: str
    end: str
    partition: str
    exit_code: str
    timelimit: str | None = None
    user: str | None = None
    account: str | None = None
    elapsed_raw: int | None = None
    node_list: str | None = None
    node_count: int | None = None
    allocated_cpus: int | None = None
    task_count: int | None = None
    req_mem: str | None = None
    req_tres: str | None = None
    alloc_tres: str | None = None
    total_cpu: str | None = None
    cpu_time_raw: int | None = None
    work_dir: str | None = None


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


def get_job_accounting(
    ssh_host: str,
    job_id: str,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> SlurmAccountingRecord | None:
    """Return completed-job accounting visible for one SLURM job ID."""

    command = [
        "ssh",
        ssh_host,
        build_sacct_command(job_id),
    ]

    result = runner(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )

    return parse_sacct_output(job_id, result.stdout)


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


def build_sacct_command(job_id: str) -> str:
    """Build a shell-quoted read-only remote sacct command."""

    normalized_job_id = normalize_job_id(job_id)

    return " ".join(
        [
            "sacct",
            "-X",
            "-P",
            "-n",
            "-j",
            shlex.quote(normalized_job_id),
            shlex.quote(f"--format={_SACCT_FORMAT}"),
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


def parse_sacct_output(job_id: str, output: str) -> SlurmAccountingRecord | None:
    """Parse pipe-delimited sacct output for one requested job."""

    normalized_job_id = normalize_job_id(job_id)
    wanted = {
        normalized_job_id,
        f"{normalized_job_id}.batch",
        f"{normalized_job_id}.extern",
    }
    fallback: SlurmAccountingRecord | None = None

    for line in output.splitlines():
        if not line.strip():
            continue

        parts = line.split("|")
        if len(parts) < 8 or parts[0].strip() not in wanted:
            continue

        record = _parse_sacct_record(parts)

        if parts[0].strip() == normalized_job_id:
            return record

        fallback = fallback or record

    return fallback


def _parse_sacct_record(parts: list[str]) -> SlurmAccountingRecord:
    if len(parts) >= len(_SACCT_FIELDS):
        return SlurmAccountingRecord(
            job_id=normalize_job_id(parts[0].strip()),
            name=_optional_part(parts, 1) or "",
            user=_optional_part(parts, 2),
            account=_optional_part(parts, 3),
            state=_optional_part(parts, 4) or "",
            exit_code=_optional_part(parts, 5) or "",
            elapsed=_optional_part(parts, 6) or "",
            elapsed_raw=_optional_int(parts, 7),
            start=_optional_part(parts, 8) or "",
            end=_optional_part(parts, 9) or "",
            partition=_optional_part(parts, 10) or "",
            timelimit=_optional_part(parts, 11),
            node_list=_optional_part(parts, 12),
            node_count=_optional_int(parts, 13),
            allocated_cpus=_optional_int(parts, 14),
            task_count=_optional_int(parts, 15),
            req_mem=_optional_part(parts, 16),
            req_tres=_optional_part(parts, 17),
            alloc_tres=_optional_part(parts, 18),
            total_cpu=_optional_part(parts, 19),
            cpu_time_raw=_optional_int(parts, 20),
            work_dir=_optional_part(parts, 21),
        )

    return SlurmAccountingRecord(
        job_id=normalize_job_id(parts[0].strip()),
        name=parts[1].strip(),
        state=parts[2].strip(),
        elapsed=parts[3].strip(),
        start=parts[4].strip(),
        end=parts[5].strip(),
        partition=parts[6].strip(),
        exit_code=parts[7].strip(),
        timelimit=parts[8].strip() if len(parts) > 8 and parts[8].strip() else None,
    )


def _optional_part(parts: list[str], index: int) -> str | None:
    if index >= len(parts):
        return None
    value = parts[index].strip()
    return value or None


def _optional_int(parts: list[str], index: int) -> int | None:
    value = _optional_part(parts, index)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def normalize_job_id(job_id: str) -> str:
    """Validate a SLURM job ID and strip non-primary job-step suffixes."""

    candidate = str(job_id or "").strip()

    if not _JOB_ID_RE.fullmatch(candidate):
        raise ValueError("SLURM job ID contains unsafe characters")

    return re.sub(r"\.(?:batch|extern)$", "", candidate)
