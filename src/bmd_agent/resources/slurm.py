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
    "State",
    "Elapsed",
    "Start",
    "End",
    "Partition%20",
    "ExitCode",
    "Timelimit%20",
    "NodeList%80",
    "NNodes",
    "NCPUS",
    "AllocCPUS",
    "TotalCPU",
    "CPUTimeRAW",
    "AllocTRES%120",
    "ReqTRES%120",
    "MaxRSS",
)
_SACCT_FORMAT = ",".join(_SACCT_FIELDS)
_MEMORY_RE = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)(?P<unit>[KMGTP]?)\s*$", re.IGNORECASE)


@dataclass
class SlurmJob:
    job_id: str
    user: str
    name: str
    state: str
    elapsed: str
    reason: str


@dataclass
class SlurmStepAccountingRecord:
    job_id: str
    name: str
    state: str
    elapsed: str
    node_list: str | None = None
    node_count: int | None = None
    cpu_count: int | None = None
    allocated_cpus: int | None = None
    total_cpu: str | None = None
    total_cpu_seconds: float | None = None
    cpu_time_raw: int | None = None
    alloc_tres: str | None = None
    req_tres: str | None = None
    max_rss: str | None = None
    max_rss_bytes: int | None = None


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
    node_list: str | None = None
    node_count: int | None = None
    cpu_count: int | None = None
    allocated_cpus: int | None = None
    total_cpu: str | None = None
    total_cpu_seconds: float | None = None
    cpu_time_raw: int | None = None
    alloc_tres: str | None = None
    req_tres: str | None = None
    max_rss: str | None = None
    max_rss_bytes: int | None = None
    max_rss_source: str | None = None
    cpu_efficiency: float | None = None
    steps: tuple[SlurmStepAccountingRecord, ...] = ()


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
    fallback: SlurmAccountingRecord | None = None
    primary: SlurmAccountingRecord | None = None
    steps: list[SlurmStepAccountingRecord] = []
    max_rss_candidates: list[tuple[str, str, int]] = []

    for line in output.splitlines():
        if not line.strip():
            continue

        parts = line.split("|")
        row_job_id = _optional_part(parts, 0)
        if len(parts) < 8 or row_job_id is None:
            continue
        if not _sacct_row_belongs_to_job(row_job_id, normalized_job_id):
            continue

        max_rss = _optional_part(parts, 17)
        max_rss_bytes = _parse_memory_bytes(max_rss)
        if max_rss is not None and max_rss_bytes is not None:
            max_rss_candidates.append((row_job_id, max_rss, max_rss_bytes))

        if row_job_id == normalized_job_id:
            primary = _parse_primary_record(row_job_id, parts)
            continue

        step = _parse_step_record(row_job_id, parts)
        steps.append(step)
        if fallback is None:
            fallback = _parse_primary_record(row_job_id, parts, primary_job_id=normalized_job_id)

    record = primary or fallback
    if record is None:
        return None

    if record.max_rss_bytes is None and max_rss_candidates:
        source, max_rss, max_rss_bytes = max(max_rss_candidates, key=lambda item: item[2])
        record.max_rss = max_rss
        record.max_rss_bytes = max_rss_bytes
        record.max_rss_source = source
    elif record.max_rss is not None:
        record.max_rss_source = record.job_id

    record.steps = tuple(steps)
    record.cpu_efficiency = _derive_cpu_efficiency(record)
    return record


def _sacct_row_belongs_to_job(row_job_id: str, normalized_job_id: str) -> bool:
    return row_job_id == normalized_job_id or row_job_id.startswith(f"{normalized_job_id}.")


def _parse_primary_record(
    row_job_id: str,
    parts: list[str],
    *,
    primary_job_id: str | None = None,
) -> SlurmAccountingRecord:
    alloc_tres = _optional_part(parts, 15)
    ncpus = _optional_int(_optional_part(parts, 11))
    allocated_cpus = (
        _optional_int(_optional_part(parts, 12))
        or _optional_int(_tres_value(alloc_tres, "cpu"))
        or ncpus
    )
    node_count = (
        _optional_int(_optional_part(parts, 10))
        or _optional_int(_tres_value(alloc_tres, "node"))
    )
    total_cpu = _optional_part(parts, 13)
    max_rss = _optional_part(parts, 17)
    max_rss_bytes = _parse_memory_bytes(max_rss)
    return SlurmAccountingRecord(
        job_id=primary_job_id or normalize_job_id(row_job_id),
        name=_required_part(parts, 1),
        state=_required_part(parts, 2),
        elapsed=_required_part(parts, 3),
        start=_required_part(parts, 4),
        end=_required_part(parts, 5),
        partition=_required_part(parts, 6),
        exit_code=_required_part(parts, 7),
        timelimit=_optional_part(parts, 8),
        node_list=_optional_part(parts, 9),
        node_count=node_count,
        cpu_count=ncpus,
        allocated_cpus=allocated_cpus,
        total_cpu=total_cpu,
        total_cpu_seconds=_parse_slurm_duration_seconds(total_cpu),
        cpu_time_raw=_optional_int(_optional_part(parts, 14)),
        alloc_tres=alloc_tres,
        req_tres=_optional_part(parts, 16),
        max_rss=max_rss,
        max_rss_bytes=max_rss_bytes,
        max_rss_source=row_job_id if max_rss_bytes is not None else None,
    )


def _parse_step_record(row_job_id: str, parts: list[str]) -> SlurmStepAccountingRecord:
    alloc_tres = _optional_part(parts, 15)
    ncpus = _optional_int(_optional_part(parts, 11))
    allocated_cpus = (
        _optional_int(_optional_part(parts, 12))
        or _optional_int(_tres_value(alloc_tres, "cpu"))
        or ncpus
    )
    total_cpu = _optional_part(parts, 13)
    max_rss = _optional_part(parts, 17)
    return SlurmStepAccountingRecord(
        job_id=row_job_id,
        name=_required_part(parts, 1),
        state=_required_part(parts, 2),
        elapsed=_required_part(parts, 3),
        node_list=_optional_part(parts, 9),
        node_count=(
            _optional_int(_optional_part(parts, 10))
            or _optional_int(_tres_value(alloc_tres, "node"))
        ),
        cpu_count=ncpus,
        allocated_cpus=allocated_cpus,
        total_cpu=total_cpu,
        total_cpu_seconds=_parse_slurm_duration_seconds(total_cpu),
        cpu_time_raw=_optional_int(_optional_part(parts, 14)),
        alloc_tres=alloc_tres,
        req_tres=_optional_part(parts, 16),
        max_rss=max_rss,
        max_rss_bytes=_parse_memory_bytes(max_rss),
    )


def _required_part(parts: list[str], index: int) -> str:
    return parts[index].strip() if len(parts) > index else ""


def _optional_part(parts: list[str], index: int) -> str | None:
    if len(parts) <= index:
        return None
    value = parts[index].strip()
    return value or None


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _tres_value(tres: str | None, key: str) -> str | None:
    if tres is None:
        return None
    for item in tres.split(","):
        if "=" not in item:
            continue
        item_key, item_value = item.split("=", maxsplit=1)
        if item_key.strip() == key:
            return item_value.strip()
    return None


def _parse_slurm_duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.upper() in {"UNKNOWN", "UNLIMITED", "NOT_SET"}:
        return None

    days = 0
    if "-" in text:
        day_text, text = text.split("-", maxsplit=1)
        try:
            days = int(day_text)
        except ValueError:
            return None

    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (float(part) for part in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (float(part) for part in parts)
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = float(parts[0])
        else:
            return None
    except ValueError:
        return None
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _parse_memory_bytes(value: str | None) -> int | None:
    if value is None:
        return None
    match = _MEMORY_RE.fullmatch(value)
    if not match:
        return None
    number = float(match.group("value"))
    unit = match.group("unit").upper()
    multiplier = {
        "": 1,
        "K": 1024,
        "M": 1024 ** 2,
        "G": 1024 ** 3,
        "T": 1024 ** 4,
        "P": 1024 ** 5,
    }[unit]
    return int(number * multiplier)


def _derive_cpu_efficiency(record: SlurmAccountingRecord) -> float | None:
    total_cpu_seconds = record.total_cpu_seconds
    elapsed_seconds = _parse_slurm_duration_seconds(record.elapsed)
    allocated_cpus = record.allocated_cpus
    if total_cpu_seconds is None or elapsed_seconds in (None, 0) or not allocated_cpus:
        return None
    return total_cpu_seconds / (elapsed_seconds * allocated_cpus)


def normalize_job_id(job_id: str) -> str:
    """Validate a SLURM job ID and strip non-primary job-step suffixes."""

    candidate = str(job_id or "").strip()

    if not _JOB_ID_RE.fullmatch(candidate):
        raise ValueError("SLURM job ID contains unsafe characters")

    return re.sub(r"\.(?:batch|extern)$", "", candidate)
