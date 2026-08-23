from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any

from bmd_agent.config import SlurmClusterResource
from bmd_agent.resources.slurm import (
    SlurmAccountingRecord,
    get_job_accounting,
    normalize_job_id,
)
from bmd_agent.resources.vasp import (
    RemotePathError,
    authorize_remote_path,
    build_remote_file_path,
    remote_directory_exists,
    remote_file_exists,
    retrieve_remote_file,
)


RemoteRunner = Callable[..., subprocess.CompletedProcess[bytes]]
SlurmRunner = Callable[..., subprocess.CompletedProcess[str]]
ScientificParser = Callable[
    [Mapping[str, Path], Mapping[str, str], Mapping[str, Any]],
    "ScientificResult",
]

SUBMISSION_FILENAME = "submission.json"
PRODUCER_PROVENANCE = "producer_provenance"
SCHEDULER_OBSERVATION = "scheduler_observation"
LOG_OBSERVATION = "log_observation"
ARTIFACT_OBSERVATION = "artifact_observation"
PYMATGEN_DERIVED = "pymatgen_derived"

_ARTIFACT_FILENAMES = {
    "contcar": "CONTCAR",
    "outcar": "OUTCAR",
    "vasprun": "vasprun.xml",
    "kpoints": "KPOINTS",
    "doscar": "DOSCAR",
}
_SCIENTIFIC_READ_KEYS = ("contcar", "vasprun", "kpoints")
_PACKAGE_RE = re.compile(r"^\[runner\]\s+(\w+)\s+version:\s*(.+)$")
_PYTHON_RE = re.compile(r"^\[runner\]\s+python:\s*(.+)$")
_ENV_RE = re.compile(r"\b(PMG_VASP_PSP_DIR)=([^\s]+)")
_STARTING_JOB_RE = re.compile(
    r"\bStarting job\s*-\s*(?P<label>[^()\r\n]+?)\s*"
    r"\((?P<uuid>[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\)",
    re.IGNORECASE,
)


class RunInspectionError(RuntimeError):
    """Raised when run inspection cannot safely continue."""


@dataclass(frozen=True)
class WorkflowStage:
    index: int
    stage_type: str
    theory: str
    modifiers: tuple[str, ...]
    label: str | None


@dataclass(frozen=True)
class PathObservation:
    label: str
    path: str
    kind: str
    present: bool
    evidence_type: str


@dataclass(frozen=True)
class AttemptStateObservation:
    path: str | None
    present: bool
    evidence_type: str = PRODUCER_PROVENANCE
    error: str | None = None


@dataclass(frozen=True)
class LogRuntimeObservation:
    evidence_type: str
    sources: tuple[str, ...]
    python: str | None
    packages: Mapping[str, str]
    environment: Mapping[str, str]
    stage_uuids: Mapping[str, str]


@dataclass(frozen=True)
class ScientificResult:
    source_paths: tuple[str, ...]
    evidence_type: str = PYMATGEN_DERIVED
    final_formula: str | None = None
    final_energy_ev: float | None = None
    energy_per_atom_ev: float | None = None
    electronic_convergence: bool | None = None
    band_gap_ev: float | None = None
    band_kpoints: int | None = None
    bands: int | None = None
    unavailable: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class ComparisonObservation:
    status: str
    evidence_type: str
    checked_fields: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
    reason: str | None = None


@dataclass(frozen=True)
class RunInspection:
    flow_root: str
    submission_path: str
    workflow_stages: tuple[WorkflowStage, ...]
    stage_directories: tuple[PathObservation, ...]
    result_directory: PathObservation
    log_paths: tuple[PathObservation, ...]
    final_artifacts: tuple[PathObservation, ...]
    producer_git: Mapping[str, Any]
    cluster_request: Mapping[str, Any]
    resources_request: Mapping[str, Any]
    environment_policy: Mapping[str, Any]
    attempt_state: AttemptStateObservation
    job_id: str | None
    scheduler: SlurmAccountingRecord | None
    scheduler_error: str | None
    runtime: LogRuntimeObservation
    scientific: ScientificResult
    comparison: ComparisonObservation


def inspect_remote_run(
    cluster: SlurmClusterResource,
    flow_root: str,
    *,
    remote_runner: RemoteRunner = subprocess.run,
    slurm_runner: SlurmRunner = subprocess.run,
    scientific_parser: ScientificParser | None = None,
    timeout: float = 20,
) -> RunInspection:
    """Inspect a BMD Compute run through configured read-only resources."""

    submission_path = build_remote_file_path(
        flow_root,
        SUBMISSION_FILENAME,
        allowed_roots=cluster.allowed_remote_roots,
    )
    submission = _read_json_file(
        cluster.ssh_host,
        submission_path,
        runner=remote_runner,
        timeout=timeout,
    )
    producer = _parse_submission(submission, allowed_roots=cluster.allowed_remote_roots)

    stage_directories = tuple(
        _observe_remote_path(
            cluster.ssh_host,
            label=label,
            path=path,
            kind="directory",
            runner=remote_runner,
            timeout=timeout,
        )
        for label, path in producer["stage_dirs"].items()
    )
    result_directory = _observe_remote_path(
        cluster.ssh_host,
        label="result_dir",
        path=producer["result_dir"],
        kind="directory",
        runner=remote_runner,
        timeout=timeout,
    )
    log_paths = tuple(
        _observe_remote_path(
            cluster.ssh_host,
            label=label,
            path=path,
            kind="file",
            runner=remote_runner,
            timeout=timeout,
        )
        for label, path in producer["log_paths"].items()
    )
    final_artifacts = tuple(
        _observe_remote_path(
            cluster.ssh_host,
            label=key,
            path=build_remote_file_path(
                producer["result_dir"],
                filename,
                allowed_roots=cluster.allowed_remote_roots,
            ),
            kind="file",
            runner=remote_runner,
            timeout=timeout,
        )
        for key, filename in _ARTIFACT_FILENAMES.items()
    )

    attempt_state, attempt_payload = _read_attempt_state(
        cluster.ssh_host,
        producer["attempt_state_path"],
        runner=remote_runner,
        timeout=timeout,
    )
    job_id = _find_job_id(submission, attempt_payload)
    scheduler, scheduler_error = _inspect_scheduler(
        cluster.ssh_host,
        job_id,
        runner=slurm_runner,
        timeout=timeout,
    )
    runtime = _parse_runtime_logs(
        cluster.ssh_host,
        log_paths,
        runner=remote_runner,
        timeout=timeout,
    )
    scientific = _derive_scientific_result(
        cluster.ssh_host,
        final_artifacts,
        submission["flow_spec"]["workflow_spec"],
        runner=remote_runner,
        parser=scientific_parser or parse_vasp_output_files,
        timeout=timeout,
    )
    comparison = compare_with_producer_result(attempt_payload, scientific)

    return RunInspection(
        flow_root=str(authorize_remote_path(flow_root, allowed_roots=cluster.allowed_remote_roots)),
        submission_path=str(submission_path),
        workflow_stages=producer["workflow_stages"],
        stage_directories=stage_directories,
        result_directory=result_directory,
        log_paths=log_paths,
        final_artifacts=final_artifacts,
        producer_git=producer["producer_git"],
        cluster_request=producer["cluster_request"],
        resources_request=producer["resources_request"],
        environment_policy=producer["environment_policy"],
        attempt_state=attempt_state,
        job_id=job_id,
        scheduler=scheduler,
        scheduler_error=scheduler_error,
        runtime=runtime,
        scientific=scientific,
        comparison=comparison,
    )


def parse_vasp_output_files(
    local_paths: Mapping[str, Path],
    display_paths: Mapping[str, str],
    workflow_spec: Mapping[str, Any],
) -> ScientificResult:
    """Derive compact scientific observations from local temporary VASP files."""

    if "vasprun" not in local_paths:
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            unavailable=("vasprun.xml is unavailable",),
        )

    source_paths = tuple(display_paths[key] for key in local_paths)
    try:
        from pymatgen.core import Structure
        from pymatgen.io.vasp.outputs import Vasprun
    except Exception as exc:
        return ScientificResult(
            source_paths=source_paths,
            error=str(exc),
        )

    unavailable: list[str] = []
    parse_eigenvalues = (
        "kpoints" in local_paths
        or _workflow_has_stage(workflow_spec, "band_structure")
    )
    vasprun = None
    try:
        vasprun = _load_vasprun(
            Vasprun,
            local_paths["vasprun"],
            parse_eigenvalues=parse_eigenvalues,
        )
    except Exception as exc:
        unavailable.append(f"vasprun.xml could not be parsed: {exc}")

    final_structure = None
    if vasprun is not None:
        final_structure = getattr(vasprun, "final_structure", None)
    if final_structure is None and "contcar" in local_paths:
        try:
            final_structure = Structure.from_file(str(local_paths["contcar"]))
        except Exception as exc:
            unavailable.append(f"CONTCAR could not be parsed: {exc}")

    final_formula = None
    natoms = None
    if final_structure is not None:
        try:
            final_formula = final_structure.composition.reduced_formula
            natoms = len(final_structure)
        except Exception as exc:
            unavailable.append(f"final structure summary could not be derived: {exc}")
    else:
        unavailable.append("final formula unavailable: final structure could not be derived")

    final_energy = _float_or_none(getattr(vasprun, "final_energy", None))
    if final_energy is None:
        unavailable.append("final energy unavailable: vasprun.xml did not provide final_energy")

    energy_per_atom = (
        final_energy / natoms
        if final_energy is not None and natoms
        else None
    )
    if energy_per_atom is None:
        unavailable.append("energy/atom unavailable: final energy or atom count unavailable")

    electronic_convergence = getattr(vasprun, "converged_electronic", None)
    if electronic_convergence is None:
        electronic_convergence = getattr(vasprun, "converged", None)
    if electronic_convergence is None:
        unavailable.append("electronic convergence unavailable: vasprun.xml did not provide convergence status")

    band_gap = None
    band_kpoints = None
    bands = None
    band_requested = "kpoints" in local_paths or _workflow_has_stage(workflow_spec, "band_structure")
    if band_requested:
        band_structure = None
        if vasprun is None:
            unavailable.append("band structure unavailable: vasprun.xml could not be parsed")
        else:
            try:
                band_structure = _band_structure_from_vasprun(vasprun, local_paths)
            except Exception as exc:
                unavailable.append(f"band structure could not be derived: {exc}")

        if band_structure is not None:
            band_data = getattr(band_structure, "bands", None) or {}
            band_kpoints = _band_kpoint_count(band_structure, band_data)
            bands = _band_count(band_data)
            if band_kpoints is None:
                unavailable.append("band k-points unavailable: band structure did not provide k-points")
            if bands is None:
                unavailable.append("bands unavailable: band structure did not provide eigenvalue bands")
            try:
                gap = band_structure.get_band_gap()
                band_gap = _float_or_none(gap.get("energy"))
            except Exception as exc:
                unavailable.append(f"band gap could not be derived: {exc}")
        elif vasprun is not None and not any(
            item.startswith("band structure could not be derived")
            for item in unavailable
        ):
            unavailable.append("band structure could not be derived")

    return ScientificResult(
        source_paths=source_paths,
        final_formula=final_formula,
        final_energy_ev=_round_float(final_energy),
        energy_per_atom_ev=_round_float(energy_per_atom),
        electronic_convergence=_bool_or_none(electronic_convergence),
        band_gap_ev=_round_float(band_gap),
        band_kpoints=band_kpoints,
        bands=bands,
        unavailable=tuple(unavailable),
    )


def compare_with_producer_result(
    attempt_payload: Mapping[str, Any] | None,
    scientific: ScientificResult,
) -> ComparisonObservation:
    """Compare with durable producer result fields when such fields exist."""

    producer_result = _producer_result_payload(attempt_payload)
    if producer_result is None:
        return ComparisonObservation(
            status="unavailable",
            evidence_type=PRODUCER_PROVENANCE,
            reason="No durable BMD Compute result payload was found in inspected producer artifacts.",
        )

    checks = {
        "final_formula": scientific.final_formula,
        "final_energy_ev": scientific.final_energy_ev,
        "energy_per_atom_ev": scientific.energy_per_atom_ev,
        "electronic_convergence": scientific.electronic_convergence,
    }
    band_result = producer_result.get("band_structure")
    if isinstance(band_result, Mapping):
        checks["band_gap_ev"] = scientific.band_gap_ev
        checks["kpoints_count"] = scientific.band_kpoints
        checks["bands"] = scientific.bands

    checked: list[str] = []
    mismatches: list[str] = []
    for key, observed in checks.items():
        expected = _nested_result_value(producer_result, key)
        if expected is None or observed is None:
            continue
        checked.append(key)
        if not _values_match(expected, observed):
            mismatches.append(key)

    if not checked:
        return ComparisonObservation(
            status="unavailable",
            evidence_type=PRODUCER_PROVENANCE,
            reason="Producer result payload did not contain comparable fields.",
        )

    return ComparisonObservation(
        status="mismatch" if mismatches else "match",
        evidence_type=PRODUCER_PROVENANCE,
        checked_fields=tuple(checked),
        mismatches=tuple(mismatches),
    )


def _read_json_file(
    ssh_host: str,
    path: PurePosixPath,
    *,
    runner: RemoteRunner,
    timeout: float,
) -> dict[str, Any]:
    contents = retrieve_remote_file(
        ssh_host,
        path,
        runner=runner,
        timeout=timeout,
    )

    try:
        payload = json.loads(contents.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RunInspectionError(f"Malformed JSON in {path}") from exc

    if not isinstance(payload, dict):
        raise RunInspectionError(f"{path} must contain a JSON object")

    return payload


def _parse_submission(
    submission: Mapping[str, Any],
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> dict[str, Any]:
    flow_spec = _required_mapping(submission, "flow_spec")
    workflow_spec = _required_mapping(flow_spec, "workflow_spec")
    paths = _required_mapping(submission, "paths")
    provenance = _required_mapping(submission, "provenance")

    return {
        "workflow_stages": _parse_workflow_stages(workflow_spec),
        "stage_dirs": _parse_stage_dirs(paths, allowed_roots=allowed_roots),
        "result_dir": _required_authorized_path(
            paths,
            "result_dir",
            allowed_roots=allowed_roots,
        ),
        "log_paths": _parse_log_paths(paths, allowed_roots=allowed_roots),
        "producer_git": _producer_git_provenance(provenance),
        "cluster_request": dict(_optional_mapping(submission, "cluster")),
        "resources_request": dict(_optional_mapping(submission, "resources")),
        "environment_policy": dict(_optional_mapping(submission, "environment")),
        "attempt_state_path": _attempt_state_path(
            submission,
            paths,
            allowed_roots=allowed_roots,
        ),
    }


def _parse_workflow_stages(workflow_spec: Mapping[str, Any]) -> tuple[WorkflowStage, ...]:
    stages = workflow_spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise RunInspectionError("submission flow_spec.workflow_spec.stages must be a non-empty list")

    parsed: list[WorkflowStage] = []
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, Mapping):
            raise RunInspectionError("workflow stage records must be JSON objects")
        stage_type = _required_str(stage, "stage_type")
        theory = _required_str(stage, "theory")
        modifiers = stage.get("modifiers") or []
        if not isinstance(modifiers, list) or not all(isinstance(item, str) for item in modifiers):
            raise RunInspectionError("workflow stage modifiers must be a list of strings")
        label = stage.get("label")
        if label is not None and not isinstance(label, str):
            raise RunInspectionError("workflow stage label must be a string or null")
        parsed.append(
            WorkflowStage(
                index=index,
                stage_type=stage_type,
                theory=theory,
                modifiers=tuple(modifiers),
                label=label,
            )
        )

    return tuple(parsed)


def _parse_stage_dirs(
    paths: Mapping[str, Any],
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> dict[str, PurePosixPath]:
    stage_dirs = _required_mapping(paths, "stage_dirs")
    parsed: dict[str, PurePosixPath] = {}
    for label, path in stage_dirs.items():
        if not isinstance(label, str) or not label:
            raise RunInspectionError("stage directory labels must be non-empty strings")
        parsed[label] = _authorize_path_value(
            path,
            f"paths.stage_dirs.{label}",
            allowed_roots=allowed_roots,
        )
    return parsed


def _parse_log_paths(
    paths: Mapping[str, Any],
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> dict[str, PurePosixPath]:
    parsed: dict[str, PurePosixPath] = {}
    for key in ("log_out", "log_err", "slurm_out", "slurm_err"):
        value = paths.get(key)
        if value is None:
            continue
        parsed[key] = _authorize_path_value(
            value,
            f"paths.{key}",
            allowed_roots=allowed_roots,
        )
    return parsed


def _attempt_state_path(
    submission: Mapping[str, Any],
    paths: Mapping[str, Any],
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath | None:
    submission_block = _optional_mapping(submission, "submission")
    value = submission_block.get("attempt_state") or paths.get("submission_attempt_state")
    if value is None:
        return None
    return _authorize_path_value(
        value,
        "submission.attempt_state",
        allowed_roots=allowed_roots,
    )


def _read_attempt_state(
    ssh_host: str,
    path: PurePosixPath | None,
    *,
    runner: RemoteRunner,
    timeout: float,
) -> tuple[AttemptStateObservation, Mapping[str, Any] | None]:
    if path is None:
        return AttemptStateObservation(path=None, present=False), None

    if not remote_file_exists(ssh_host, path, runner=runner, timeout=timeout):
        return AttemptStateObservation(path=str(path), present=False), None

    try:
        return (
            AttemptStateObservation(path=str(path), present=True),
            _read_json_file(ssh_host, path, runner=runner, timeout=timeout),
        )
    except RunInspectionError as exc:
        return AttemptStateObservation(path=str(path), present=True, error=str(exc)), None


def _find_job_id(
    submission: Mapping[str, Any],
    attempt_payload: Mapping[str, Any] | None,
) -> str | None:
    candidates = [
        submission.get("job_id"),
        _optional_mapping(submission, "submission").get("job_id"),
    ]
    if attempt_payload is not None:
        candidates.extend([
            attempt_payload.get("job_id"),
            _optional_mapping(attempt_payload, "job_record").get("job_id"),
        ])

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return normalize_job_id(str(candidate))
        except ValueError:
            continue
    return None


def _inspect_scheduler(
    ssh_host: str,
    job_id: str | None,
    *,
    runner: SlurmRunner,
    timeout: float,
) -> tuple[SlurmAccountingRecord | None, str | None]:
    if job_id is None:
        return None, "No valid SLURM job ID was found in inspected producer artifacts."
    try:
        return get_job_accounting(
            ssh_host,
            job_id,
            runner=runner,
            timeout=timeout,
        ), None
    except (ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)


def _parse_runtime_logs(
    ssh_host: str,
    paths: Iterable[PathObservation],
    *,
    runner: RemoteRunner,
    timeout: float,
) -> LogRuntimeObservation:
    sources: list[str] = []
    packages: dict[str, str] = {}
    environment: dict[str, str] = {}
    stage_uuids: dict[str, str] = {}
    python: str | None = None

    for observation in paths:
        if not observation.present or not observation.label.startswith("log_"):
            continue
        path = PurePosixPath(observation.path)
        try:
            text = retrieve_remote_file(
                ssh_host,
                path,
                runner=runner,
                timeout=timeout,
            ).decode("utf-8", "replace")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            continue
        sources.append(observation.path)
        for line in text.splitlines():
            python_match = _PYTHON_RE.match(line.strip())
            if python_match:
                python = python_match.group(1).strip()
                continue
            package_match = _PACKAGE_RE.match(line.strip())
            if package_match:
                packages[package_match.group(1)] = package_match.group(2).strip()
            for env_match in _ENV_RE.finditer(line):
                environment[env_match.group(1)] = env_match.group(2)
            for uuid_match in _STARTING_JOB_RE.finditer(line):
                label = uuid_match.group("label").strip()
                if label:
                    stage_uuids.setdefault(label, uuid_match.group("uuid").lower())

    return LogRuntimeObservation(
        evidence_type=LOG_OBSERVATION,
        sources=tuple(sources),
        python=python,
        packages=packages,
        environment=environment,
        stage_uuids=stage_uuids,
    )


def _derive_scientific_result(
    ssh_host: str,
    artifacts: Iterable[PathObservation],
    workflow_spec: Mapping[str, Any],
    *,
    runner: RemoteRunner,
    parser: ScientificParser,
    timeout: float,
) -> ScientificResult:
    present = {
        observation.label: observation
        for observation in artifacts
        if observation.present
    }
    if "vasprun" not in present:
        return ScientificResult(
            source_paths=tuple(observation.path for observation in present.values()),
            unavailable=("vasprun.xml is unavailable",),
        )

    with tempfile.TemporaryDirectory(prefix="bmd-agent-run-") as tmpdir:
        root = Path(tmpdir)
        local_paths: dict[str, Path] = {}
        display_paths: dict[str, str] = {}
        for key in _SCIENTIFIC_READ_KEYS:
            observation = present.get(key)
            if observation is None:
                continue
            filename = _ARTIFACT_FILENAMES.get(key)
            if filename is None:
                continue
            local_path = root / filename
            contents = retrieve_remote_file(
                ssh_host,
                PurePosixPath(observation.path),
                runner=runner,
                timeout=timeout,
            )
            local_path.write_bytes(contents)
            local_paths[key] = local_path
            display_paths[key] = observation.path
        return parser(local_paths, display_paths, workflow_spec)


def _observe_remote_path(
    ssh_host: str,
    *,
    label: str,
    path: PurePosixPath,
    kind: str,
    runner: RemoteRunner,
    timeout: float,
) -> PathObservation:
    if kind == "directory":
        present = remote_directory_exists(ssh_host, path, runner=runner, timeout=timeout)
    elif kind == "file":
        present = remote_file_exists(ssh_host, path, runner=runner, timeout=timeout)
    else:
        raise ValueError(f"unsupported remote path observation kind: {kind}")
    return PathObservation(
        label=label,
        path=str(path),
        kind=kind,
        present=present,
        evidence_type=ARTIFACT_OBSERVATION,
    )


def _producer_git_provenance(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    bmd_compute = _optional_mapping(provenance, "bmd_compute")
    return dict(_optional_mapping(bmd_compute, "source"))


def _required_authorized_path(
    mapping: Mapping[str, Any],
    key: str,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath:
    return _authorize_path_value(mapping.get(key), f"paths.{key}", allowed_roots=allowed_roots)


def _authorize_path_value(
    value: Any,
    label: str,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise RunInspectionError(f"submission {label} must be a non-empty string")
    try:
        return authorize_remote_path(value, allowed_roots=allowed_roots)
    except RemotePathError as exc:
        raise RemotePathError(f"submission {label}: {exc}") from exc


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, Mapping):
        raise RunInspectionError(f"submission missing object field: {key}")
    return value


def _optional_mapping(mapping: Mapping[str, Any] | None, key: str) -> Mapping[str, Any]:
    if mapping is None:
        return {}
    value = mapping.get(key)
    return value if isinstance(value, Mapping) else {}


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise RunInspectionError(f"submission field must be a non-empty string: {key}")
    return value


def _load_vasprun(vasprun_cls: Any, path: Path, *, parse_eigenvalues: bool):
    common = {
        "exception_on_bad_xml": False,
        "parse_potcar_file": False,
    }
    if parse_eigenvalues:
        return vasprun_cls(str(path), **common)
    scalar_only = {
        "parse_dos": False,
        **common,
    }
    try:
        return vasprun_cls(
            str(path),
            parse_eigenvalues=False,
            **scalar_only,
        )
    except TypeError:
        return vasprun_cls(
            str(path),
            parse_eigen=False,
            **scalar_only,
        )


def _band_structure_from_vasprun(vasprun: Any, local_paths: Mapping[str, Path]):
    existing = getattr(vasprun, "band_structure", None)
    if existing is not None:
        return existing

    get_band_structure = getattr(vasprun, "get_band_structure", None)
    if not callable(get_band_structure):
        return None

    if "kpoints" in local_paths:
        return get_band_structure(
            kpoints_filename=str(local_paths["kpoints"]),
            line_mode=True,
        )

    return get_band_structure(line_mode=True)


def _workflow_has_stage(workflow_spec: Mapping[str, Any], stage_type: str) -> bool:
    stages = workflow_spec.get("stages")
    if not isinstance(stages, list):
        return False
    return any(
        isinstance(stage, Mapping) and stage.get("stage_type") == stage_type
        for stage in stages
    )


def _band_kpoint_count(band_structure: Any, bands: Mapping[Any, Any]) -> int | None:
    kpoints = getattr(band_structure, "kpoints", None)
    if kpoints:
        return len(kpoints)
    for matrix in bands.values():
        rows = _matrix_rows(matrix)
        if rows:
            return len(_matrix_rows(rows[0]))
    return None


def _band_count(bands: Mapping[Any, Any]) -> int | None:
    counts = [len(_matrix_rows(matrix)) for matrix in bands.values()]
    return max(counts) if counts else None


def _matrix_rows(matrix: Any) -> list[Any]:
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    try:
        return list(matrix)
    except TypeError:
        return []


def _producer_result_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    for key in ("result", "results", "results_summary", "completed_result"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return value
    job_record = _optional_mapping(payload, "job_record")
    submission_result = job_record.get("result")
    return submission_result if isinstance(submission_result, Mapping) else None


def _nested_result_value(result: Mapping[str, Any], key: str) -> Any:
    if key in result:
        return result[key]
    band = result.get("band_structure")
    if isinstance(band, Mapping):
        return band.get(key)
    return None


def _values_match(expected: Any, observed: Any) -> bool:
    if isinstance(expected, bool) or isinstance(observed, bool):
        return expected is observed
    expected_float = _float_or_none(expected)
    observed_float = _float_or_none(observed)
    if expected_float is not None and observed_float is not None:
        return abs(expected_float - observed_float) <= 1e-4
    return str(expected) == str(observed)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)
