from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping, Sequence

from bmd_agent.resources.slurm import SlurmAccountingRecord, SlurmStepAccountingRecord


OOM_ESTABLISHED = "OOM ESTABLISHED"
OOM_POSSIBLE = "OOM POSSIBLE / MEMORY PRESSURE"
NO_OOM_EVIDENCE = "NO OOM EVIDENCE FOUND"
INSUFFICIENT_OOM_EVIDENCE = "INSUFFICIENT EVIDENCE"
OOM_DIAGNOSTIC_EVIDENCE = "oom_diagnostic_evidence"

_MEMORY_PRESSURE_RATIO = 0.90
_MEMORY_VALUE_RE = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[KMGTPE]?)"
    r"(?:i?[Bb])?(?P<scope>[cn]?)\s*$",
    re.IGNORECASE,
)
_TRES_MEMORY_RE = re.compile(r"(?:^|,)mem=(?P<value>[^,]+)", re.IGNORECASE)
_EXPLICIT_OOM_PATTERNS = (
    re.compile(r"\bout[\s_-]*of[\s_-]*memory\b", re.IGNORECASE),
    re.compile(r"\boutofmemory\b", re.IGNORECASE),
    re.compile(r"\boom[\s_-]*kill(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bexceeded(?: the)? memory limit\b", re.IGNORECASE),
    re.compile(r"\bmemory limit exceeded\b", re.IGNORECASE),
    re.compile(r"\bcgroup\b.*\bmemory\b.*\b(?:oom|kill)", re.IGNORECASE),
    re.compile(r"\bkilled process\b.*\b(?:oom|out of memory)\b", re.IGNORECASE),
)
_ALLOCATION_FAILURE_PATTERNS = (
    re.compile(r"\bcannot allocate memory\b", re.IGNORECASE),
    re.compile(r"\bstd::bad_alloc\b", re.IGNORECASE),
    re.compile(r"\ballocation failed\b", re.IGNORECASE),
    re.compile(r"\binsufficient memory\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class OomEvidenceMarker:
    source_type: str
    source: str
    kind: str
    text: str


@dataclass(frozen=True)
class MemoryObservation:
    source: str
    raw_value: str
    bytes_value: int | None
    scope: str


@dataclass(frozen=True)
class OomDiagnosticEvidence:
    assessment: str
    evidence_type: str = OOM_DIAGNOSTIC_EVIDENCE
    sufficiency: str = "limited"
    scheduler_oom_state: bool = False
    explicit_evidence: tuple[OomEvidenceMarker, ...] = ()
    suggestive_evidence: tuple[OomEvidenceMarker, ...] = ()
    requested_memory: MemoryObservation | None = None
    allocated_memory: MemoryObservation | None = None
    maximum_rss: MemoryObservation | None = None
    maximum_vm_size: MemoryObservation | None = None
    average_rss: MemoryObservation | None = None
    memory_utilization_percent: float | None = None
    inspected_sources: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    guidance: tuple[str, ...] = ()


def assess_oom_evidence(
    scheduler: SlurmAccountingRecord | None,
    *,
    log_observations: Iterable[tuple[str, str]] = (),
    inspected_log_sources: Iterable[str] = (),
    source_limitations: Iterable[str] = (),
) -> OomDiagnosticEvidence:
    """Describe positive OOM evidence without changing lifecycle semantics."""

    explicit: list[OomEvidenceMarker] = []
    suggestive: list[OomEvidenceMarker] = []
    inspected: list[str] = []
    limitations: list[str] = list(source_limitations)
    rows = _scheduler_rows(scheduler)

    if scheduler is None:
        limitations.append("scheduler accounting was unavailable")
    else:
        inspected.append("SLURM parent accounting")
        if scheduler.steps:
            inspected.append("SLURM job-step accounting")
        else:
            limitations.append("SLURM job-step accounting was unavailable")
        for source, row in rows:
            _add_scheduler_markers(source, row, explicit)

    inspected_logs = list(inspected_log_sources)
    inspected.extend(inspected_logs)
    log_count = 0
    for source, text in log_observations:
        log_count += 1
        inspected.append(source)
        _add_text_markers(source, text, explicit, suggestive)
    if log_count == 0 and not inspected_logs:
        limitations.append("no readable diagnostic log evidence was available")

    requested = _requested_memory(scheduler)
    allocated = _allocated_memory(scheduler)
    maximum_rss = _largest_memory_observation(
        rows,
        "max_rss",
        "maximum RSS",
        scope="task_max",
    )
    maximum_vm = _largest_memory_observation(
        rows,
        "max_vm_size",
        "maximum VM size",
        scope="task_max",
    )
    average_rss = _largest_memory_observation(
        rows,
        "ave_rss",
        "average RSS",
        scope="task_average",
    )
    utilization, utilization_limitation = _compatible_memory_utilization(
        scheduler,
        maximum_rss,
        requested,
        allocated,
    )
    if utilization_limitation:
        limitations.append(utilization_limitation)
    if maximum_rss is None:
        limitations.append("maximum RSS accounting was unavailable")

    scheduler_oom_state = any(marker.kind == "scheduler_oom_state" for marker in explicit)
    has_negative_evidence_basis = (
        scheduler is not None
        and (
            (maximum_rss is not None and maximum_rss.bytes_value is not None)
            or bool(inspected_logs)
            or log_count > 0
        )
    )
    if explicit:
        assessment = OOM_ESTABLISHED
        sufficiency = "positive_explicit_evidence"
        guidance = (
            "The execution evidence identifies memory exhaustion.",
            "Review the memory allocation and memory-intensive calculation settings before rerunning.",
        )
    elif suggestive or (
        utilization is not None and utilization >= _MEMORY_PRESSURE_RATIO * 100
    ):
        if utilization is not None and utilization >= _MEMORY_PRESSURE_RATIO * 100:
            suggestive.append(
                OomEvidenceMarker(
                    source_type="scheduler_observation",
                    source=maximum_rss.source if maximum_rss else "SLURM accounting",
                    kind="near_memory_allocation",
                    text=(
                        f"maximum RSS was {utilization:.1f}% of a semantically compatible "
                        "memory allocation"
                    ),
                )
            )
        assessment = OOM_POSSIBLE
        sufficiency = "suggestive_evidence"
        guidance = (
            "Memory pressure is possible, but the available evidence does not establish OOM termination.",
        )
    elif not has_negative_evidence_basis:
        assessment = INSUFFICIENT_OOM_EVIDENCE
        sufficiency = "insufficient_evidence"
        guidance = (
            "Available scheduler and execution evidence was insufficient to assess OOM.",
        )
    else:
        assessment = NO_OOM_EVIDENCE
        sufficiency = "no_positive_evidence_in_inspected_sources"
        guidance = (
            "No positive OOM evidence was found; this does not prove that OOM did not occur.",
        )

    return OomDiagnosticEvidence(
        assessment=assessment,
        sufficiency=sufficiency,
        scheduler_oom_state=scheduler_oom_state,
        explicit_evidence=tuple(_deduplicate_markers(explicit)),
        suggestive_evidence=tuple(_deduplicate_markers(suggestive)),
        requested_memory=requested,
        allocated_memory=allocated,
        maximum_rss=maximum_rss,
        maximum_vm_size=maximum_vm,
        average_rss=average_rss,
        memory_utilization_percent=utilization,
        inspected_sources=tuple(dict.fromkeys(inspected)),
        limitations=tuple(dict.fromkeys(limitations)),
        guidance=guidance,
    )


def serialize_oom_evidence(evidence: OomDiagnosticEvidence | None) -> Mapping[str, object] | None:
    """Serialize already-derived OOM evidence without acquiring new evidence."""

    return asdict(evidence) if evidence is not None else None


def oom_candidate_lines(text: str, *, limit: int = 8) -> tuple[str, ...]:
    """Return bounded lines relevant to the fixed OOM evidence vocabulary."""

    matches: list[str] = []
    for line in str(text).splitlines():
        compact = " ".join(line.strip().split())
        if not compact:
            continue
        if not (
            _matches_any(compact, _EXPLICIT_OOM_PATTERNS)
            or _matches_any(compact, _ALLOCATION_FAILURE_PATTERNS)
        ):
            continue
        if compact not in matches:
            matches.append(compact[:500])
        if len(matches) >= limit:
            break
    return tuple(matches)


def _scheduler_rows(
    scheduler: SlurmAccountingRecord | None,
) -> tuple[tuple[str, SlurmAccountingRecord | SlurmStepAccountingRecord], ...]:
    if scheduler is None:
        return ()
    return (
        (f"SLURM job {scheduler.job_id}", scheduler),
        *tuple((f"SLURM step {step.job_id_raw}", step) for step in scheduler.steps),
    )


def _add_scheduler_markers(
    source: str,
    row: SlurmAccountingRecord | SlurmStepAccountingRecord,
    explicit: list[OomEvidenceMarker],
) -> None:
    state = str(row.state or "").strip()
    if _normalized_scheduler_state(state) == "OUT_OF_MEMORY":
        explicit.append(
            OomEvidenceMarker("scheduler_observation", source, "scheduler_oom_state", state)
        )
    reason = str(row.reason or "").strip()
    if reason and _matches_any(reason, _EXPLICIT_OOM_PATTERNS):
        explicit.append(
            OomEvidenceMarker("scheduler_observation", source, "scheduler_oom_reason", reason)
        )


def _add_text_markers(
    source: str,
    text: str,
    explicit: list[OomEvidenceMarker],
    suggestive: list[OomEvidenceMarker],
) -> None:
    for compact in oom_candidate_lines(text):
        if _matches_any(compact, _EXPLICIT_OOM_PATTERNS):
            explicit.append(
                OomEvidenceMarker("log_observation", source, "explicit_oom_marker", compact)
            )
        elif _matches_any(compact, _ALLOCATION_FAILURE_PATTERNS):
            suggestive.append(
                OomEvidenceMarker("log_observation", source, "allocation_failure", compact)
            )


def _requested_memory(scheduler: SlurmAccountingRecord | None) -> MemoryObservation | None:
    if scheduler is None:
        return None
    tres = _tres_memory(scheduler.req_tres)
    if tres is not None:
        return MemoryObservation("SLURM ReqTRES", tres, _memory_bytes(tres), "job_total")
    if not scheduler.req_mem:
        return None
    value, scope = _parse_req_mem(scheduler.req_mem)
    return MemoryObservation("SLURM ReqMem", scheduler.req_mem, value, scope)


def _allocated_memory(scheduler: SlurmAccountingRecord | None) -> MemoryObservation | None:
    if scheduler is None:
        return None
    tres = _tres_memory(scheduler.alloc_tres)
    if tres is None:
        return None
    return MemoryObservation("SLURM AllocTRES", tres, _memory_bytes(tres), "job_total")


def _largest_memory_observation(
    rows: Sequence[tuple[str, SlurmAccountingRecord | SlurmStepAccountingRecord]],
    attribute: str,
    label: str,
    *,
    scope: str,
) -> MemoryObservation | None:
    observations: list[MemoryObservation] = []
    for source, row in rows:
        raw = getattr(row, attribute, None)
        if not raw:
            continue
        observations.append(
            MemoryObservation(f"{source} {label}", str(raw), _memory_bytes(str(raw)), scope)
        )
    comparable = [item for item in observations if item.bytes_value is not None]
    if comparable:
        return max(comparable, key=lambda item: item.bytes_value or 0)
    return observations[0] if observations else None


def _compatible_memory_utilization(
    scheduler: SlurmAccountingRecord | None,
    maximum_rss: MemoryObservation | None,
    requested: MemoryObservation | None,
    allocated: MemoryObservation | None,
) -> tuple[float | None, str | None]:
    if scheduler is None or maximum_rss is None or maximum_rss.bytes_value is None:
        return None, None
    denominator = allocated or requested
    if denominator is None or denominator.bytes_value is None:
        return None, "requested/allocated memory could not be converted to a compatible quantity"

    row = _memory_source_row(scheduler, maximum_rss.source)
    task_count = getattr(row, "task_count", None) if row is not None else None
    allocated_cpus = getattr(row, "allocated_cpus", None) if row is not None else None
    if denominator.scope == "per_cpu":
        if task_count is None or allocated_cpus is None or task_count != allocated_cpus:
            return None, "MaxRSS and per-CPU ReqMem were not semantically comparable"
    elif denominator.scope == "job_total":
        if task_count != 1:
            return None, "task-level MaxRSS was not compared with total job/node memory"
    elif denominator.scope == "per_node":
        node_count = getattr(row, "node_count", None) if row is not None else None
        if task_count != 1 or node_count != 1:
            return None, "task-level MaxRSS was not compared with total job/node memory"
    else:
        return None, "ReqMem scope was unavailable; memory utilization was not calculated"

    return 100.0 * maximum_rss.bytes_value / denominator.bytes_value, None


def _memory_source_row(
    scheduler: SlurmAccountingRecord,
    source: str,
) -> SlurmAccountingRecord | SlurmStepAccountingRecord | None:
    if f"job {scheduler.job_id}" in source:
        return scheduler
    for step in scheduler.steps:
        if step.job_id_raw in source:
            return step
    return None


def _parse_req_mem(raw: str) -> tuple[int | None, str]:
    match = _MEMORY_VALUE_RE.fullmatch(raw)
    if match is None:
        return None, "unknown"
    scope = {"c": "per_cpu", "n": "per_node"}.get(
        match.group("scope").lower(),
        "unknown",
    )
    return _memory_bytes(raw[:-1] if match.group("scope") else raw), scope


def _tres_memory(value: str | None) -> str | None:
    if not value:
        return None
    match = _TRES_MEMORY_RE.search(value)
    return match.group("value").strip() if match else None


def _memory_bytes(raw: str) -> int | None:
    match = _MEMORY_VALUE_RE.fullmatch(raw)
    if match is None:
        return None
    value = float(match.group("value"))
    unit = match.group("unit").upper()
    if not unit:
        return None
    exponent = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5, "E": 6}[unit]
    return int(value * (1024**exponent))


def _normalized_scheduler_state(state: str) -> str:
    return state.strip().upper().rstrip("+").replace(" ", "_")


def _matches_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _deduplicate_markers(markers: Iterable[OomEvidenceMarker]) -> list[OomEvidenceMarker]:
    deduplicated: list[OomEvidenceMarker] = []
    seen: set[tuple[str, str, str, str]] = set()
    for marker in markers:
        key = (marker.source_type, marker.source, marker.kind, marker.text)
        if key not in seen:
            seen.add(key)
            deduplicated.append(marker)
    return deduplicated
