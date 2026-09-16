from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
import json
from pathlib import Path
from typing import Any
import warnings

from bmd_agent.resources.run import (
    ScientificResult,
    parse_incar_contents,
    parse_vasp_output_files,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord, normalize_job_id
from bmd_agent.resources.vasp import StructureInfo, parse_poscar


class LifecycleState(str, Enum):
    PRE_RUN = "PRE_RUN"
    RUNNING = "RUNNING"
    INCOMPLETE = "INCOMPLETE"
    COMPLETED = "COMPLETED"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class LocalFileEvidence:
    name: str
    path: Path
    present: bool
    size: int | None = None

    @property
    def non_empty(self) -> bool:
        return self.present and self.size is not None and self.size > 0


@dataclass(frozen=True)
class LocalStageBinding:
    label: str
    path: Path
    stage_index: int | None = None
    producer_path: str | None = None


@dataclass(frozen=True)
class BmdWorkflowDiscovery:
    workflow_root: Path
    submission_path: Path
    submission: Mapping[str, Any]
    workflow_stages: tuple[Mapping[str, Any], ...]
    stage_bindings: tuple[LocalStageBinding, ...]
    current_stage: LocalStageBinding | None = None
    producer_root: str | None = None
    relocated: bool = False
    job_id: str | None = None
    attempt_state_path: Path | None = None
    attempt_state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class LifecycleAnalysis:
    state: LifecycleState
    directory: Path
    calculation_kind: str
    message: str
    input_files: Mapping[str, LocalFileEvidence] = field(default_factory=dict)
    output_files: Mapping[str, LocalFileEvidence] = field(default_factory=dict)
    bmd_workflow: BmdWorkflowDiscovery | None = None
    scheduler: SlurmAccountingRecord | None = None
    scheduler_error: str | None = None
    normal_completion: bool | None = None
    structure: StructureInfo | None = None
    incar_settings: Mapping[str, Any] = field(default_factory=dict)
    scientific: ScientificResult | None = None
    evidence_gaps: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()


SchedulerLookup = Callable[[str], SlurmAccountingRecord | None]

_INPUT_FILENAMES = ("POSCAR", "INCAR", "KPOINTS")
_OUTPUT_FILENAMES = ("OUTCAR", "OSZICAR", "vasprun.xml", "CONTCAR", "DOSCAR", "XDATCAR")
_BMD_LOG_KEYS = ("log_out", "log_err", "slurm_out", "slurm_err")
_ACTIVE_SCHEDULER_STATES = {
    "BOOT_FAIL_REQUEUE_FED",
    "CONFIGURING",
    "COMPLETING",
    "PENDING",
    "REQUEUED",
    "RESIZING",
    "RUNNING",
    "STAGE_OUT",
    "SUSPENDED",
}
_INACTIVE_UNSUCCESSFUL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "TIMEOUT",
}
_NORMAL_COMPLETION_MARKERS = (
    "General timing and accounting informations for this job",
    "Voluntary context switches",
)


def analyze_calculation_directory(
    directory: Path | str,
    *,
    scheduler_lookup: SchedulerLookup | None = None,
    max_ancestor_levels: int = 4,
) -> LifecycleAnalysis:
    """Classify a calculation directory from local, read-only evidence."""

    current = Path(directory).resolve()
    workflow = _discover_bmd_workflow(current, max_ancestor_levels=max_ancestor_levels)
    if workflow is not None:
        return _analyze_bmd_workflow(
            current,
            workflow,
            scheduler_lookup=scheduler_lookup,
        )
    return _analyze_direct_vasp_directory(current)


def _analyze_bmd_workflow(
    current: Path,
    workflow: BmdWorkflowDiscovery,
    *,
    scheduler_lookup: SchedulerLookup | None,
) -> LifecycleAnalysis:
    target = workflow.current_stage.path if workflow.current_stage else workflow.workflow_root
    input_files = _observe_files(target, _INPUT_FILENAMES)
    output_files = _observe_files(target, _OUTPUT_FILENAMES)
    log_files = _observe_bmd_logs(workflow)
    meaningful_execution = _has_meaningful_files(output_files) or _has_meaningful_files(log_files)
    if workflow.relocated:
        scheduler, scheduler_error = (
            None,
            "scheduler evidence belongs to the original producer location and was not used for relocated local snapshot",
        )
    else:
        scheduler, scheduler_error = _lookup_scheduler(workflow.job_id, scheduler_lookup)
    normal_completion = _detect_normal_completion(target)
    producer_success = _producer_success(workflow.attempt_state)
    scientific = _derive_local_scientific(target, workflow.submission)
    structure = _structure_from_inputs(input_files)
    incar_settings = _incar_settings(input_files)
    gaps = _input_gaps(input_files)
    limitations: list[str] = []

    if scheduler is not None and _is_active_scheduler_state(scheduler.state):
        return LifecycleAnalysis(
            state=LifecycleState.RUNNING,
            directory=current,
            calculation_kind="BMD Compute",
            message="Active scheduler evidence indicates the calculation is running.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if workflow.relocated and normal_completion:
        return LifecycleAnalysis(
            state=LifecycleState.COMPLETED,
            directory=current,
            calculation_kind="BMD Compute",
            message="Relocated BMD Compute snapshot has durable local VASP normal-completion evidence.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            scheduler_error=scheduler_error,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if _scheduler_success(scheduler) and (normal_completion or producer_success):
        return LifecycleAnalysis(
            state=LifecycleState.COMPLETED,
            directory=current,
            calculation_kind="BMD Compute",
            message="BMD Compute and scheduler evidence indicate successful completion.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if scheduler is not None and _is_inactive_unsuccessful_scheduler_state(scheduler):
        return LifecycleAnalysis(
            state=LifecycleState.INCOMPLETE,
            directory=current,
            calculation_kind="BMD Compute",
            message="Scheduler evidence indicates execution stopped before successful completion.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if scheduler is not None and _scheduler_nonzero_exit(scheduler) and not normal_completion:
        return LifecycleAnalysis(
            state=LifecycleState.INCOMPLETE,
            directory=current,
            calculation_kind="BMD Compute",
            message="Scheduler exit status is non-zero and successful VASP completion is absent.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if _has_required_inputs(input_files) and not meaningful_execution:
        return LifecycleAnalysis(
            state=LifecycleState.PRE_RUN,
            directory=current,
            calculation_kind="BMD Compute",
            message="Inputs are present and no meaningful execution output was observed.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            scheduler_error=scheduler_error,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if meaningful_execution:
        if workflow.job_id and scheduler is None:
            limitations.append(scheduler_error or "scheduler accounting was unavailable")
        limitations.append(
            "partial execution evidence is present, but no reliable active/inactive execution state was established"
        )
        return LifecycleAnalysis(
            state=LifecycleState.UNKNOWN,
            directory=current,
            calculation_kind="BMD Compute",
            message="BMD Compute evidence was found, but lifecycle state is not deterministic.",
            input_files=input_files,
            output_files=output_files | log_files,
            bmd_workflow=workflow,
            scheduler=scheduler,
            scheduler_error=scheduler_error,
            normal_completion=normal_completion,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
            limitations=tuple(limitations),
        )

    return LifecycleAnalysis(
        state=LifecycleState.UNKNOWN,
        directory=current,
        calculation_kind="BMD Compute",
        message="BMD Compute provenance was found, but calculation inputs or execution evidence are incomplete.",
        input_files=input_files,
        output_files=output_files | log_files,
        bmd_workflow=workflow,
        scheduler=scheduler,
        scheduler_error=scheduler_error,
        normal_completion=normal_completion,
        structure=structure,
        incar_settings=incar_settings,
        scientific=scientific,
        evidence_gaps=tuple(gaps),
    )


def _analyze_direct_vasp_directory(directory: Path) -> LifecycleAnalysis:
    input_files = _observe_files(directory, _INPUT_FILENAMES)
    output_files = _observe_files(directory, _OUTPUT_FILENAMES)
    meaningful_execution = _has_meaningful_files(output_files)
    normal_completion = _detect_normal_completion(directory)
    structure = _structure_from_inputs(input_files)
    incar_settings = _incar_settings(input_files)
    scientific = _derive_local_scientific(directory, None)
    gaps = _input_gaps(input_files)

    if normal_completion:
        return LifecycleAnalysis(
            state=LifecycleState.COMPLETED,
            directory=directory,
            calculation_kind="direct VASP",
            message="Durable VASP normal-completion evidence was found.",
            input_files=input_files,
            output_files=output_files,
            normal_completion=True,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if _has_required_inputs(input_files) and not meaningful_execution:
        return LifecycleAnalysis(
            state=LifecycleState.PRE_RUN,
            directory=directory,
            calculation_kind="direct VASP",
            message="Inputs are present and no meaningful execution output was observed.",
            input_files=input_files,
            output_files=output_files,
            normal_completion=False,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    if meaningful_execution:
        return LifecycleAnalysis(
            state=LifecycleState.UNKNOWN,
            directory=directory,
            calculation_kind="direct VASP",
            message=(
                "Partial VASP execution evidence is present, but no scheduler/provenance "
                "or durable normal-completion evidence establishes whether it is active or stopped."
            ),
            input_files=input_files,
            output_files=output_files,
            normal_completion=False,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
            limitations=(
                "manual VASP directory has no scheduler/provenance evidence for lifecycle status",
            ),
        )

    if any(observation.present for observation in input_files.values()):
        return LifecycleAnalysis(
            state=LifecycleState.UNKNOWN,
            directory=directory,
            calculation_kind="direct VASP",
            message="Some VASP input evidence is present, but required inputs are incomplete.",
            input_files=input_files,
            output_files=output_files,
            normal_completion=False,
            structure=structure,
            incar_settings=incar_settings,
            scientific=scientific,
            evidence_gaps=tuple(gaps),
        )

    return LifecycleAnalysis(
        state=LifecycleState.UNKNOWN,
        directory=directory,
        calculation_kind="none",
        message="No recognizable BMD Compute or VASP calculation found in this directory.",
        input_files=input_files,
        output_files=output_files,
        evidence_gaps=("no recognizable BMD Compute or VASP calculation evidence was found",),
    )


def _discover_bmd_workflow(
    current: Path,
    *,
    max_ancestor_levels: int,
) -> BmdWorkflowDiscovery | None:
    for root in _bounded_ancestors(current, max_ancestor_levels=max_ancestor_levels):
        submission_path = root / "submission.json"
        if not submission_path.is_file():
            continue
        try:
            submission = _read_json(submission_path)
            workflow = _workflow_from_submission(root, submission_path, submission, current)
        except Exception:
            continue
        if workflow is not None:
            return workflow
    return None


def _workflow_from_submission(
    root: Path,
    submission_path: Path,
    submission: Mapping[str, Any],
    current: Path,
) -> BmdWorkflowDiscovery | None:
    flow_spec = _mapping(submission.get("flow_spec"))
    workflow_spec = _mapping(flow_spec.get("workflow_spec"))
    stages = tuple(
        item for item in workflow_spec.get("stages", ())
        if isinstance(item, Mapping)
    )
    if not stages:
        return None
    paths = _mapping(submission.get("paths"))
    stage_bindings = _stage_bindings(root, paths, stage_count=len(stages))
    relocated = _is_relocated_local_snapshot(
        root,
        current,
        stage_bindings,
    )
    if relocated:
        stage_bindings = _relocated_stage_bindings(
            root,
            paths,
            stage_count=len(stages),
        )
    related = current == root or any(_is_relative_to(current, binding.path) for binding in stage_bindings)
    if not related:
        return None
    current_stage = _current_stage_binding(current, root, stage_bindings)
    attempt_state_path = _optional_local_path(
        _mapping(submission.get("submission")).get("attempt_state")
        or paths.get("submission_attempt_state"),
        root=root,
    )
    attempt_state = _read_json(attempt_state_path) if attempt_state_path and attempt_state_path.is_file() else None
    return BmdWorkflowDiscovery(
        workflow_root=root,
        submission_path=submission_path,
        submission=submission,
        workflow_stages=stages,
        stage_bindings=stage_bindings,
        current_stage=current_stage,
        producer_root=_producer_root_text(paths),
        relocated=relocated,
        job_id=_find_job_id(submission, attempt_state),
        attempt_state_path=attempt_state_path,
        attempt_state=attempt_state,
    )


def _stage_bindings(
    root: Path,
    paths: Mapping[str, Any],
    *,
    stage_count: int,
) -> tuple[LocalStageBinding, ...]:
    bindings: list[LocalStageBinding] = []
    seen: set[Path] = set()
    stage_dirs = paths.get("stage_dirs")
    if isinstance(stage_dirs, Mapping):
        for index, (label, value) in enumerate(stage_dirs.items(), start=1):
            path = _optional_local_path(value, root=root)
            if path is None:
                continue
            bindings.append(LocalStageBinding(str(label), path, index, _path_text(value)))
            seen.add(path)
    result_dir = _optional_local_path(paths.get("result_dir"), root=root)
    if result_dir is not None and result_dir not in seen:
        stage_index = 1 if not bindings and stage_count == 1 else stage_count
        bindings.append(LocalStageBinding("result_dir", result_dir, stage_index, _path_text(paths.get("result_dir"))))
    return tuple(bindings)


def _relocated_stage_bindings(
    root: Path,
    paths: Mapping[str, Any],
    *,
    stage_count: int,
) -> tuple[LocalStageBinding, ...]:
    stage_index = 1 if stage_count == 1 else stage_count
    producer_path = _path_text(paths.get("result_dir"))
    return (LocalStageBinding("result_dir", root, stage_index, producer_path),)


def _is_relocated_local_snapshot(
    root: Path,
    current: Path,
    bindings: Sequence[LocalStageBinding],
) -> bool:
    if current != root:
        return False
    if not _has_recognizable_local_vasp_evidence(root):
        return False
    if any(_has_recognizable_local_vasp_evidence(binding.path) for binding in bindings):
        return False
    return not any(_is_relative_to(binding.path, root) for binding in bindings)


def _has_recognizable_local_vasp_evidence(directory: Path) -> bool:
    return _has_meaningful_files(_observe_files(directory, _INPUT_FILENAMES + _OUTPUT_FILENAMES))


def _current_stage_binding(
    current: Path,
    root: Path,
    bindings: Sequence[LocalStageBinding],
) -> LocalStageBinding | None:
    matches = [binding for binding in bindings if _is_relative_to(current, binding.path)]
    if matches:
        return max(matches, key=lambda binding: len(binding.path.parts))
    if current == root:
        with_outputs = [
            binding for binding in bindings
            if _has_meaningful_files(_observe_files(binding.path, _OUTPUT_FILENAMES))
        ]
        if with_outputs:
            return with_outputs[-1]
        return bindings[0] if bindings else None
    return None


def _bounded_ancestors(current: Path, *, max_ancestor_levels: int) -> tuple[Path, ...]:
    roots = [current]
    roots.extend(current.parents[:max_ancestor_levels])
    return tuple(roots)


def _observe_files(directory: Path, filenames: Sequence[str]) -> dict[str, LocalFileEvidence]:
    observations: dict[str, LocalFileEvidence] = {}
    for name in filenames:
        path = directory / name
        if not path.is_file():
            observations[name] = LocalFileEvidence(name, path, False)
            continue
        try:
            observations[name] = LocalFileEvidence(name, path, True, path.stat().st_size)
        except OSError:
            observations[name] = LocalFileEvidence(name, path, True, None)
    return observations


def _observe_bmd_logs(workflow: BmdWorkflowDiscovery) -> dict[str, LocalFileEvidence]:
    paths = _mapping(workflow.submission.get("paths"))
    observations: dict[str, LocalFileEvidence] = {}
    for key in _BMD_LOG_KEYS:
        path = _optional_local_path(paths.get(key), root=workflow.workflow_root)
        if path is None:
            continue
        if not path.is_file():
            observations[key] = LocalFileEvidence(key, path, False)
            continue
        try:
            observations[key] = LocalFileEvidence(key, path, True, path.stat().st_size)
        except OSError:
            observations[key] = LocalFileEvidence(key, path, True, None)
    return observations


def _has_required_inputs(inputs: Mapping[str, LocalFileEvidence]) -> bool:
    return all(inputs.get(name) is not None and inputs[name].non_empty for name in _INPUT_FILENAMES)


def _input_gaps(inputs: Mapping[str, LocalFileEvidence]) -> list[str]:
    gaps: list[str] = []
    for name in _INPUT_FILENAMES:
        observation = inputs.get(name)
        if observation is None or not observation.present:
            gaps.append(f"{name} is missing")
        elif observation.size == 0:
            gaps.append(f"{name} is empty")
        elif observation.size is None:
            gaps.append(f"{name} size could not be established")
    return gaps


def _has_meaningful_files(files: Mapping[str, LocalFileEvidence]) -> bool:
    return any(observation.non_empty for observation in files.values())


def _detect_normal_completion(directory: Path) -> bool:
    outcar = directory / "OUTCAR"
    try:
        if not outcar.is_file() or outcar.stat().st_size <= 0:
            return False
        text = _read_file_tail(outcar).decode("utf-8", "replace")
    except OSError:
        return False
    return any(marker in text for marker in _NORMAL_COMPLETION_MARKERS)


def _read_file_tail(path: Path, *, limit: int = 256_000) -> bytes:
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > limit:
            handle.seek(size - limit)
        return handle.read(limit)


def _structure_from_inputs(inputs: Mapping[str, LocalFileEvidence]) -> StructureInfo | None:
    poscar = inputs.get("POSCAR")
    if poscar is None or not poscar.non_empty:
        return None
    try:
        return parse_poscar(poscar.path.read_bytes(), source=str(poscar.path))
    except Exception:
        return None


def _incar_settings(inputs: Mapping[str, LocalFileEvidence]) -> Mapping[str, Any]:
    incar = inputs.get("INCAR")
    if incar is None or not incar.non_empty:
        return {}
    try:
        settings, error = parse_incar_contents(incar.path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return {}
    return {} if error else dict(settings)


def _derive_local_scientific(directory: Path, submission: Mapping[str, Any] | None) -> ScientificResult | None:
    keys = {
        "contcar": directory / "CONTCAR",
        "vasprun": directory / "vasprun.xml",
        "kpoints": directory / "KPOINTS",
    }
    local_paths: dict[str, Path] = {}
    for key, path in keys.items():
        try:
            if path.is_file() and path.stat().st_size > 0:
                local_paths[key] = path
        except OSError:
            continue
    if not any(key in local_paths for key in ("contcar", "vasprun")):
        return None
    display_paths = {key: str(path) for key, path in local_paths.items()}
    workflow_spec = _workflow_spec_from_submission(submission)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            scientific = parse_vasp_output_files(local_paths, display_paths, workflow_spec)
    except Exception as exc:
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            error=str(exc),
        )
    if _has_malformed_xml_warning(caught):
        reason = "vasprun.xml could not be parsed completely"
        unavailable = tuple(
            item for item in scientific.unavailable
            if "xml is malformed" not in item.lower()
        )
        if reason not in unavailable:
            unavailable = unavailable + (reason,)
        return replace(scientific, unavailable=unavailable)
    return scientific


def _has_malformed_xml_warning(caught_warnings: Sequence[warnings.WarningMessage]) -> bool:
    return any("xml is malformed" in str(item.message).lower() for item in caught_warnings)


def _workflow_spec_from_submission(submission: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if submission is None:
        return {
            "stages": [
                {
                    "stage_type": "direct_vasp",
                    "theory": "unknown",
                    "modifiers": [],
                    "label": None,
                    "options": {},
                }
            ]
        }
    return _mapping(_mapping(submission.get("flow_spec")).get("workflow_spec"))


def _lookup_scheduler(
    job_id: str | None,
    scheduler_lookup: SchedulerLookup | None,
) -> tuple[SlurmAccountingRecord | None, str | None]:
    if job_id is None:
        return None, "No valid SLURM job ID was found in inspected producer artifacts."
    if scheduler_lookup is None:
        return None, "scheduler lookup is unavailable"
    try:
        return scheduler_lookup(job_id), None
    except Exception as exc:
        return None, str(exc)


def _is_active_scheduler_state(state: str | None) -> bool:
    if not state:
        return False
    return str(state).upper().split()[0] in _ACTIVE_SCHEDULER_STATES


def _is_inactive_unsuccessful_scheduler_state(scheduler: SlurmAccountingRecord) -> bool:
    state = str(scheduler.state or "").upper().split()[0]
    return state in _INACTIVE_UNSUCCESSFUL_STATES


def _scheduler_success(scheduler: SlurmAccountingRecord | None) -> bool:
    if scheduler is None:
        return False
    return str(scheduler.state).upper().split()[0] == "COMPLETED" and scheduler.exit_code == "0:0"


def _scheduler_nonzero_exit(scheduler: SlurmAccountingRecord | None) -> bool:
    if scheduler is None:
        return False
    return bool(scheduler.exit_code and scheduler.exit_code != "0:0")


def _producer_success(payload: Mapping[str, Any] | None) -> bool:
    if payload is None:
        return False
    status = str(payload.get("status") or payload.get("state") or "").lower()
    if status in {"success", "succeeded", "completed", "complete"}:
        return True
    return any(isinstance(payload.get(key), Mapping) for key in ("result", "results", "results_summary", "completed_result"))


def _find_job_id(
    submission: Mapping[str, Any],
    attempt_payload: Mapping[str, Any] | None,
) -> str | None:
    candidates: list[Any] = [
        submission.get("job_id"),
        _mapping(submission.get("submission")).get("job_id"),
    ]
    if attempt_payload is not None:
        candidates.extend(
            [
                attempt_payload.get("job_id"),
                _mapping(attempt_payload.get("job_record")).get("job_id"),
            ]
        )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return normalize_job_id(str(candidate))
        except ValueError:
            continue
    return None


def _read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path_text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _producer_root_text(paths: Mapping[str, Any]) -> str | None:
    result_dir = _path_text(paths.get("result_dir"))
    if result_dir is not None:
        return result_dir
    stage_dirs = paths.get("stage_dirs")
    if isinstance(stage_dirs, Mapping):
        for value in reversed(tuple(stage_dirs.values())):
            path = _path_text(value)
            if path is not None:
                return path
    return None


def _optional_local_path(value: Any, *, root: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
