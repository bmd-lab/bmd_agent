from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
import warnings
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
    remote_file_size,
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
PRODUCER_REQUESTED = "producer_requested"
EXECUTED_INPUT = "executed_input"
AGENT_COMPARISON = "agent_comparison"
TERMINATION_OBSERVATION = "termination_observation"
TRAJECTORY_OBSERVATION = "trajectory_observation"
CONVERGENCE_PROGRESS_ASSESSMENT = "convergence_progress_assessment"

CONVERGED = "CONVERGED"
EVIDENCE_OF_PROGRESS = "EVIDENCE OF PROGRESS"
NO_CLEAR_EVIDENCE_OF_PROGRESS = "NO CLEAR EVIDENCE OF PROGRESS"
INSUFFICIENT_EVIDENCE = "INSUFFICIENT EVIDENCE"

_ARTIFACT_FILENAMES = {
    "contcar": "CONTCAR",
    "outcar": "OUTCAR",
    "vasprun": "vasprun.xml",
    "kpoints": "KPOINTS",
    "doscar": "DOSCAR",
}
_SCIENTIFIC_READ_KEYS = ("contcar", "vasprun", "kpoints")
_DIAGNOSE_ARTIFACT_FILENAMES = {
    "oszicar": "OSZICAR",
    "vasprun": "vasprun.xml",
}
_DIAGNOSE_VASPRUN_MAX_BYTES = 50_000_000
_DIAGNOSE_RECENT_WINDOW = 5
_DIAGNOSE_CRITERIA_KEYS = ("NELM", "EDIFF", "NSW", "EDIFFG")
_INCOMPLETE_VASPRUN_TRAJECTORY_REASON = "file could not be parsed completely"
_UNREADABLE_VASPRUN_TRAJECTORY_REASON = "file could not be read"
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
    options: Mapping[str, Any] = field(default_factory=dict)
    evidence_type: str = PRODUCER_REQUESTED


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
class InitialStructureObservation:
    status: str
    evidence_type: str = PRODUCER_PROVENANCE
    representation_type: str | None = None
    representation_hash: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class StructureObservation:
    source_path: str | None
    evidence_type: str = PYMATGEN_DERIVED
    formula: str | None = None
    reduced_formula: str | None = None
    site_count: int | None = None
    lattice_a: float | None = None
    lattice_b: float | None = None
    lattice_c: float | None = None
    alpha: float | None = None
    beta: float | None = None
    gamma: float | None = None
    volume: float | None = None
    density: float | None = None
    c_over_a: float | None = None
    unavailable: tuple[str, ...] = ()
    error: str | None = None


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
    structure: StructureObservation | None = None
    executed_parameters: tuple["IncarObservation", ...] = ()


@dataclass(frozen=True)
class IncarObservation:
    label: str
    path: str
    present: bool
    stage_index: int | None
    source_type: str = "retained_incar"
    evidence_type: str = EXECUTED_INPUT
    values: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class InputExpectationObservation:
    stage_label: str
    stage_index: int | None
    option_path: str
    requested_value: Any
    input_key: str
    expected_value: Any
    observed_value: Any
    status: str
    source_values: Mapping[str, Any] = field(default_factory=dict)
    evidence_type: str = AGENT_COMPARISON
    reason: str | None = None


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
    initial_structure: InitialStructureObservation = field(
        default_factory=lambda: InitialStructureObservation(
            status="unavailable",
            reason="Submitted structure provenance was not inspected.",
        )
    )
    executed_inputs: tuple[IncarObservation, ...] = ()
    input_expectations: tuple[InputExpectationObservation, ...] = ()


@dataclass(frozen=True)
class QuantityComparison:
    run_label: str
    flow_root: str
    quantity: str
    label: str
    unit: str | None
    baseline_value: float | None
    comparison_value: float | None
    delta: float | None
    percent_delta: float | None
    status: str
    evidence_type: str = AGENT_COMPARISON
    reason: str | None = None


@dataclass(frozen=True)
class InitialStructureComparison:
    status: str
    evidence_type: str = AGENT_COMPARISON
    reason: str | None = None


@dataclass(frozen=True)
class RunComparison:
    inspections: tuple[RunInspection, ...]
    labels: Mapping[str, str]
    quantities: tuple[QuantityComparison, ...]
    initial_structure: InitialStructureComparison
    energy_warning: str | None = None


@dataclass(frozen=True)
class TerminationObservation:
    evidence_type: str = TERMINATION_OBSERVATION
    scheduler_state: str | None = None
    scheduler_exit_code: str | None = None
    scheduler_elapsed: str | None = None
    scheduler_timelimit: str | None = None
    scheduler_reports_timeout: bool | None = None
    vasp_completed_normally: bool | None = None
    custodian_events: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class ElectronicIterationObservation:
    iteration: int | None
    algorithm: str | None = None
    energy: float | None = None
    dE: float | None = None
    deps: float | None = None
    rms: float | None = None
    rms_c: float | None = None


@dataclass(frozen=True)
class ElectronicCycleObservation:
    cycle_index: int
    completed_ionic_step: bool
    iterations: int
    final_iteration: ElectronicIterationObservation | None = None


@dataclass(frozen=True)
class IonicStepObservation:
    step_index: int
    electronic_iterations: int | None = None
    free_energy: float | None = None
    energy_zero: float | None = None
    dE: float | None = None
    max_force: float | None = None


@dataclass(frozen=True)
class StageTrajectoryObservation:
    stage_index: int | None
    stage_label: str
    stage_type: str | None
    theory: str | None
    directory: str
    evidence_type: str = TRAJECTORY_OBSERVATION
    oszicar_path: str | None = None
    oszicar_present: bool = False
    oszicar_error: str | None = None
    vasprun_path: str | None = None
    vasprun_present: bool = False
    vasprun_error: str | None = None
    vasprun_skipped_reason: str | None = None
    criteria: Mapping[str, Any] = field(default_factory=dict)
    criteria_source_values: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    criteria_discrepancies: tuple[str, ...] = ()
    ionic_steps_observed: int | None = None
    electronic_iterations_by_ionic_step: tuple[int, ...] = ()
    electronic_cycles: tuple[ElectronicCycleObservation, ...] = ()
    final_electronic_iteration_count: int | None = None
    recent_electronic_iterations: tuple[ElectronicIterationObservation, ...] = ()
    completed_ionic_steps: int | None = None
    electronic_iterations_by_completed_ionic_step: tuple[int, ...] = ()
    incomplete_electronic_iteration_count: int | None = None
    recent_incomplete_electronic_iterations: tuple[ElectronicIterationObservation, ...] = ()
    ionic_steps: tuple[IonicStepObservation, ...] = ()
    recent_ionic_steps: tuple[IonicStepObservation, ...] = ()
    vasprun_ionic_steps: int | None = None
    converged_electronic: bool | None = None
    converged_ionic: bool | None = None
    unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConvergenceProgressAssessment:
    label: str
    stage_index: int | None
    stage_label: str
    scope: str
    sufficiency: str
    evidence_type: str = CONVERGENCE_PROGRESS_ASSESSMENT
    basis: tuple[str, ...] = ()
    counter_evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    features: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunDiagnosis:
    inspection: RunInspection
    termination: TerminationObservation
    trajectories: tuple[StageTrajectoryObservation, ...]
    assessments: tuple[ConvergenceProgressAssessment, ...] = ()


@dataclass(frozen=True)
class _ComparableQuantity:
    key: str
    label: str
    unit: str | None
    value: Callable[[RunInspection], float | None]


@dataclass(frozen=True)
class _BoundStageDirectory:
    label: str
    directory: PurePosixPath
    stage_index: int | None


@dataclass(frozen=True)
class _OszicarTrajectory:
    ionic_steps_observed: int | None
    electronic_iterations_by_ionic_step: tuple[int, ...]
    electronic_cycles: tuple[ElectronicCycleObservation, ...]
    final_electronic_iteration_count: int | None
    recent_electronic_iterations: tuple[ElectronicIterationObservation, ...]
    ionic_steps: tuple[IonicStepObservation, ...]
    recent_ionic_steps: tuple[IonicStepObservation, ...]
    completed_ionic_steps: int | None = None
    electronic_iterations_by_completed_ionic_step: tuple[int, ...] = ()
    incomplete_electronic_iteration_count: int | None = None
    recent_incomplete_electronic_iterations: tuple[ElectronicIterationObservation, ...] = ()


@dataclass(frozen=True)
class _VasprunTrajectory:
    present: bool
    path: str | None = None
    skipped_reason: str | None = None
    error: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    ionic_steps: int | None = None
    converged_electronic: bool | None = None
    converged_ionic: bool | None = None
    max_forces: Mapping[int, float] = field(default_factory=dict)


def inspect_remote_run(
    cluster: SlurmClusterResource,
    flow_root: str,
    *,
    remote_runner: RemoteRunner = subprocess.run,
    slurm_runner: SlurmRunner = subprocess.run,
    scientific_parser: ScientificParser | None = None,
    derive_scientific: bool = True,
    modifier_policies: Iterable[Mapping[str, Any]] = (),
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
    executed_inputs = _observe_executed_inputs(
        cluster.ssh_host,
        producer["stage_dirs"],
        producer["result_dir"],
        producer["workflow_stages"],
        allowed_roots=cluster.allowed_remote_roots,
        runner=remote_runner,
        timeout=timeout,
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
    if derive_scientific:
        scientific = _derive_scientific_result(
            cluster.ssh_host,
            final_artifacts,
            submission["flow_spec"]["workflow_spec"],
            runner=remote_runner,
            parser=scientific_parser or parse_vasp_output_files,
            timeout=timeout,
        )
    else:
        scientific = ScientificResult(
            source_paths=(),
            unavailable=("scientific artifact parsing skipped by diagnose-run v1",),
        )
    executed_inputs = executed_inputs + scientific.executed_parameters
    input_expectations = compare_requested_options_to_executed_inputs(
        producer["workflow_stages"],
        executed_inputs,
        modifier_policies,
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
        initial_structure=producer["initial_structure"],
        executed_inputs=executed_inputs,
        input_expectations=input_expectations,
    )


def compare_remote_runs(
    cluster: SlurmClusterResource,
    flow_roots: Sequence[str],
    *,
    remote_runner: RemoteRunner = subprocess.run,
    slurm_runner: SlurmRunner = subprocess.run,
    scientific_parser: ScientificParser | None = None,
    modifier_policies: Iterable[Mapping[str, Any]] = (),
    timeout: float = 20,
) -> RunComparison:
    """Inspect and compare multiple remote runs through the read-only boundary."""

    if len(flow_roots) < 2:
        raise RunInspectionError("compare-runs requires at least two remote flow roots")

    policy_tuple = tuple(modifier_policies)
    inspections = tuple(
        inspect_remote_run(
            cluster,
            flow_root,
            remote_runner=remote_runner,
            slurm_runner=slurm_runner,
            scientific_parser=scientific_parser,
            modifier_policies=policy_tuple,
            timeout=timeout,
        )
        for flow_root in flow_roots
    )
    return build_run_comparison(inspections, modifier_policies=policy_tuple)


def diagnose_remote_run(
    cluster: SlurmClusterResource,
    flow_root: str,
    *,
    remote_runner: RemoteRunner = subprocess.run,
    slurm_runner: SlurmRunner = subprocess.run,
    modifier_policies: Iterable[Mapping[str, Any]] = (),
    timeout: float = 20,
    max_vasprun_bytes: int = _DIAGNOSE_VASPRUN_MAX_BYTES,
) -> RunDiagnosis:
    """Describe termination and convergence trajectory evidence for one run."""

    inspection = inspect_remote_run(
        cluster,
        flow_root,
        remote_runner=remote_runner,
        slurm_runner=slurm_runner,
        derive_scientific=False,
        modifier_policies=modifier_policies,
        timeout=timeout,
    )
    trajectories = _observe_stage_trajectories(
        cluster.ssh_host,
        inspection,
        allowed_roots=cluster.allowed_remote_roots,
        runner=remote_runner,
        timeout=timeout,
        max_vasprun_bytes=max_vasprun_bytes,
    )
    return RunDiagnosis(
        inspection=inspection,
        termination=_termination_observation(inspection),
        trajectories=trajectories,
        assessments=assess_convergence_progress(trajectories),
    )


def assess_convergence_progress(
    trajectories: Sequence[StageTrajectoryObservation],
) -> tuple[ConvergenceProgressAssessment, ...]:
    """Build conservative progress assessments from observed trajectory evidence."""

    assessments: list[ConvergenceProgressAssessment] = []
    for trajectory in trajectories:
        electronic = _assess_electronic_progress(trajectory)
        assessments.append(electronic)
        ionic = (
            _assess_ionic_progress(trajectory)
            if _stage_uses_ionic_progress(trajectory)
            else None
        )
        if ionic is not None:
            assessments.append(ionic)
        assessments.append(_assess_stage_progress(trajectory, electronic, ionic))
    return tuple(assessments)


def _assess_electronic_progress(
    trajectory: StageTrajectoryObservation,
) -> ConvergenceProgressAssessment:
    basis: list[str] = []
    counter: list[str] = []
    limitations: list[str] = []
    features = _assessment_features(trajectory)
    explicit = trajectory.converged_electronic

    if explicit is True:
        basis.append("trajectory_observation: vasprun converged_electronic=True")
        return _assessment(
            trajectory,
            "electronic",
            CONVERGED,
            basis=basis,
            features=features,
        )
    if explicit is False:
        counter.append("trajectory_observation: vasprun converged_electronic=False")

    if trajectory.oszicar_error:
        limitations.append(f"OSZICAR parsing unavailable: {trajectory.oszicar_error}")
    if not trajectory.oszicar_present and explicit is None:
        limitations.append("OSZICAR trajectory unavailable")

    ediff = _positive_float(trajectory.criteria.get("EDIFF"))
    if ediff is None and explicit is None:
        limitations.append("EDIFF criterion unavailable")

    final_cycle = _final_completed_electronic_cycle(trajectory)
    final_iteration = final_cycle.final_iteration if final_cycle else None
    reaches_ediff = _electronic_iteration_reaches_ediff(final_iteration, ediff)
    if reaches_ediff is True:
        basis.append(
            "trajectory_observation: final completed electronic cycle reached EDIFF"
        )
    elif reaches_ediff is False:
        counter.append(
            "trajectory_observation: final completed electronic cycle has not reached EDIFF"
        )

    if basis and counter:
        limitations.append("electronic convergence evidence is contradictory")
        return _assessment(
            trajectory,
            "electronic",
            INSUFFICIENT_EVIDENCE,
            sufficiency="contradictory",
            basis=basis,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )
    if basis:
        return _assessment(
            trajectory,
            "electronic",
            CONVERGED,
            basis=basis,
            limitations=limitations,
            features=features,
        )

    incomplete_first_cycle = (
        trajectory.completed_ionic_steps == 0
        and trajectory.incomplete_electronic_iteration_count is not None
    )
    if incomplete_first_cycle:
        limitations.append("only an incomplete first electronic cycle was observed")
        return _assessment(
            trajectory,
            "electronic",
            INSUFFICIENT_EVIDENCE,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )

    if ediff is not None and _completed_electronic_cycle_below_nelm(trajectory):
        basis.append(
            "trajectory_observation: a completed electronic cycle ended before NELM with EDIFF configured"
        )
        return _assessment(
            trajectory,
            "electronic",
            EVIDENCE_OF_PROGRESS,
            basis=basis,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )

    if counter and final_cycle is not None and ediff is not None:
        return _assessment(
            trajectory,
            "electronic",
            NO_CLEAR_EVIDENCE_OF_PROGRESS,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )

    if final_cycle is None and explicit is None:
        limitations.append("completed electronic cycle unavailable")

    return _assessment(
        trajectory,
        "electronic",
        INSUFFICIENT_EVIDENCE,
        counter_evidence=counter,
        limitations=limitations,
        features=features,
    )


def _assess_ionic_progress(
    trajectory: StageTrajectoryObservation,
) -> ConvergenceProgressAssessment:
    basis: list[str] = []
    counter: list[str] = []
    limitations: list[str] = []
    features = _assessment_features(trajectory)
    explicit = trajectory.converged_ionic
    completed_steps = trajectory.completed_ionic_steps

    if explicit is True:
        basis.append("trajectory_observation: vasprun converged_ionic=True")
    elif explicit is False:
        counter.append("trajectory_observation: vasprun converged_ionic=False")

    if completed_steps is None:
        limitations.append("completed ionic step count unavailable")
    elif completed_steps == 0:
        limitations.append("zero completed ionic steps observed")

    ediffg = _float_or_none(trajectory.criteria.get("EDIFFG"))
    force_criterion = abs(ediffg) if ediffg is not None and ediffg < 0 else None
    if ediffg is None:
        limitations.append("EDIFFG criterion unavailable")
    elif ediffg >= 0:
        limitations.append("positive or zero EDIFFG criterion is not interpreted in v1")

    final_step = _final_ionic_step(trajectory)
    if final_step is None:
        limitations.append("completed ionic trajectory unavailable")
    final_force = _float_or_none(getattr(final_step, "max_force", None))
    if force_criterion is not None and final_force is None:
        limitations.append("maximum force evidence unavailable")
    elif force_criterion is not None and final_force is not None:
        if final_force <= force_criterion:
            basis.append("trajectory_observation: final maximum force reached EDIFFG")
        else:
            counter.append(
                "trajectory_observation: final maximum force has not reached EDIFFG"
            )

    if basis and counter:
        limitations.append("ionic convergence evidence is contradictory")
        return _assessment(
            trajectory,
            "ionic",
            INSUFFICIENT_EVIDENCE,
            sufficiency="contradictory",
            basis=basis,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )
    if basis:
        return _assessment(
            trajectory,
            "ionic",
            CONVERGED,
            basis=basis,
            limitations=limitations,
            features=features,
        )
    if completed_steps == 0:
        return _assessment(
            trajectory,
            "ionic",
            INSUFFICIENT_EVIDENCE,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )
    if counter and completed_steps is not None and completed_steps > 0 and force_criterion is not None:
        limitations.append("trend-based ionic progress assessment is not implemented in v1")
        return _assessment(
            trajectory,
            "ionic",
            NO_CLEAR_EVIDENCE_OF_PROGRESS,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )
    return _assessment(
        trajectory,
        "ionic",
        INSUFFICIENT_EVIDENCE,
        counter_evidence=counter,
        limitations=limitations,
        features=features,
    )


def _assess_stage_progress(
    trajectory: StageTrajectoryObservation,
    electronic: ConvergenceProgressAssessment,
    ionic: ConvergenceProgressAssessment | None,
) -> ConvergenceProgressAssessment:
    basis: list[str] = []
    counter: list[str] = []
    limitations: list[str] = []
    features = _assessment_features(trajectory)

    if ionic is None:
        if electronic.label == CONVERGED:
            basis.append("electronic scope is converged")
            return _assessment(trajectory, "stage", CONVERGED, basis=basis, features=features)
        if electronic.label == EVIDENCE_OF_PROGRESS:
            basis.extend(electronic.basis)
            limitations.extend(electronic.limitations)
            return _assessment(
                trajectory,
                "stage",
                EVIDENCE_OF_PROGRESS,
                basis=basis,
                counter_evidence=tuple(electronic.counter_evidence),
                limitations=limitations,
                features=features,
            )
        return _assessment(
            trajectory,
            "stage",
            electronic.label,
            sufficiency=electronic.sufficiency,
            counter_evidence=tuple(electronic.counter_evidence),
            limitations=tuple(electronic.limitations),
            features=features,
        )

    if trajectory.completed_ionic_steps == 0:
        limitations.append("zero completed ionic steps observed")
        limitations.append("stage progress cannot be assessed from an incomplete first SCF cycle alone")
        return _assessment(
            trajectory,
            "stage",
            INSUFFICIENT_EVIDENCE,
            limitations=limitations,
            features=features,
        )

    if electronic.sufficiency == "contradictory" or ionic.sufficiency == "contradictory":
        limitations.append("component convergence evidence is contradictory")
        return _assessment(
            trajectory,
            "stage",
            INSUFFICIENT_EVIDENCE,
            sufficiency="contradictory",
            basis=tuple(electronic.basis) + tuple(ionic.basis),
            counter_evidence=tuple(electronic.counter_evidence) + tuple(ionic.counter_evidence),
            limitations=limitations,
            features=features,
        )

    if electronic.label == CONVERGED and ionic.label == CONVERGED:
        basis.append("electronic and ionic scopes are converged")
        return _assessment(trajectory, "stage", CONVERGED, basis=basis, features=features)

    if electronic.label in (CONVERGED, EVIDENCE_OF_PROGRESS) or ionic.label == EVIDENCE_OF_PROGRESS:
        if electronic.label in (CONVERGED, EVIDENCE_OF_PROGRESS):
            basis.extend(electronic.basis)
        if ionic.label == EVIDENCE_OF_PROGRESS:
            basis.extend(ionic.basis)
        limitations.append("not all required stage scopes are converged")
        return _assessment(
            trajectory,
            "stage",
            EVIDENCE_OF_PROGRESS,
            basis=basis,
            counter_evidence=tuple(electronic.counter_evidence) + tuple(ionic.counter_evidence),
            limitations=limitations,
            features=features,
        )

    if electronic.label == NO_CLEAR_EVIDENCE_OF_PROGRESS or ionic.label == NO_CLEAR_EVIDENCE_OF_PROGRESS:
        counter.extend(electronic.counter_evidence)
        counter.extend(ionic.counter_evidence)
        limitations.append("no v1 criterion-based progress evidence was established")
        return _assessment(
            trajectory,
            "stage",
            NO_CLEAR_EVIDENCE_OF_PROGRESS,
            counter_evidence=counter,
            limitations=limitations,
            features=features,
        )

    limitations.extend(electronic.limitations)
    limitations.extend(ionic.limitations)
    return _assessment(
        trajectory,
        "stage",
        INSUFFICIENT_EVIDENCE,
        limitations=tuple(dict.fromkeys(limitations)),
        features=features,
    )


def _assessment(
    trajectory: StageTrajectoryObservation,
    scope: str,
    label: str,
    *,
    sufficiency: str | None = None,
    basis: Sequence[str] = (),
    counter_evidence: Sequence[str] = (),
    limitations: Sequence[str] = (),
    features: Mapping[str, Any] | None = None,
) -> ConvergenceProgressAssessment:
    return ConvergenceProgressAssessment(
        label=label,
        stage_index=trajectory.stage_index,
        stage_label=trajectory.stage_label,
        scope=scope,
        sufficiency=sufficiency or (
            "insufficient" if label == INSUFFICIENT_EVIDENCE else "sufficient"
        ),
        basis=tuple(dict.fromkeys(basis)),
        counter_evidence=tuple(dict.fromkeys(counter_evidence)),
        limitations=tuple(dict.fromkeys(limitations)),
        features=dict(features or {}),
    )


def _assessment_features(trajectory: StageTrajectoryObservation) -> Mapping[str, Any]:
    final_iteration = _final_observed_electronic_iteration(trajectory)
    final_step = _final_ionic_step(trajectory)
    features: dict[str, Any] = {
        "stage_type": trajectory.stage_type,
        "theory": trajectory.theory,
        "completed_ionic_steps": trajectory.completed_ionic_steps,
        "incomplete_electronic_iteration_count": trajectory.incomplete_electronic_iteration_count,
        "final_electronic_iteration_count": trajectory.final_electronic_iteration_count,
        "converged_electronic": trajectory.converged_electronic,
        "converged_ionic": trajectory.converged_ionic,
    }
    for key in _DIAGNOSE_CRITERIA_KEYS:
        if key in trajectory.criteria:
            features[key] = trajectory.criteria[key]
    if final_iteration is not None:
        for key, attribute in (
            ("final_dE", "dE"),
            ("final_deps", "deps"),
            ("final_rms", "rms"),
            ("final_rms_c", "rms_c"),
        ):
            value = getattr(final_iteration, attribute)
            if value is not None:
                features[key] = value
    if final_step is not None and final_step.max_force is not None:
        features["final_max_force"] = final_step.max_force
    return {
        key: _json_safe_value(value)
        for key, value in features.items()
        if value is not None
    }


def _stage_uses_ionic_progress(trajectory: StageTrajectoryObservation) -> bool:
    stage_type = (trajectory.stage_type or "").lower()
    if any(term in stage_type for term in ("relax", "optimisation", "optimization")):
        return True
    nsw = _int_or_none(trajectory.criteria.get("NSW"))
    return nsw is not None and nsw > 0


def _final_completed_electronic_cycle(
    trajectory: StageTrajectoryObservation,
) -> ElectronicCycleObservation | None:
    completed = [cycle for cycle in trajectory.electronic_cycles if cycle.completed_ionic_step]
    if completed:
        return completed[-1]
    if (
        trajectory.incomplete_electronic_iteration_count is None
        and trajectory.recent_electronic_iterations
        and trajectory.final_electronic_iteration_count is not None
    ):
        return ElectronicCycleObservation(
            cycle_index=max(1, trajectory.completed_ionic_steps or 1),
            completed_ionic_step=True,
            iterations=trajectory.final_electronic_iteration_count,
            final_iteration=trajectory.recent_electronic_iterations[-1],
        )
    return None


def _final_observed_electronic_iteration(
    trajectory: StageTrajectoryObservation,
) -> ElectronicIterationObservation | None:
    if trajectory.incomplete_electronic_iteration_count is not None:
        recent = trajectory.recent_incomplete_electronic_iterations
    else:
        recent = trajectory.recent_electronic_iterations
    return recent[-1] if recent else None


def _electronic_iteration_reaches_ediff(
    iteration: ElectronicIterationObservation | None,
    ediff: float | None,
) -> bool | None:
    if iteration is None or ediff is None:
        return None
    values = [
        abs(value)
        for value in (iteration.dE, iteration.deps)
        if value is not None
    ]
    if not values:
        return None
    return any(value <= ediff for value in values)


def _completed_electronic_cycle_below_nelm(
    trajectory: StageTrajectoryObservation,
) -> bool:
    nelm = _positive_int(trajectory.criteria.get("NELM"))
    if nelm is None:
        return False
    completed_cycles = [cycle for cycle in trajectory.electronic_cycles if cycle.completed_ionic_step]
    if completed_cycles:
        return any(cycle.iterations < nelm for cycle in completed_cycles)
    counts = trajectory.electronic_iterations_by_completed_ionic_step
    return any(count < nelm for count in counts)


def _final_ionic_step(trajectory: StageTrajectoryObservation) -> IonicStepObservation | None:
    if trajectory.ionic_steps:
        return trajectory.ionic_steps[-1]
    if trajectory.recent_ionic_steps:
        return trajectory.recent_ionic_steps[-1]
    return None


def _positive_float(value: Any) -> float | None:
    numeric = _float_or_none(value)
    return numeric if numeric is not None and numeric > 0 else None


def _positive_int(value: Any) -> int | None:
    numeric = _int_or_none(value)
    return numeric if numeric is not None and numeric > 0 else None


def build_run_comparison(
    inspections: Sequence[RunInspection],
    *,
    modifier_policies: Iterable[Mapping[str, Any]] = (),
) -> RunComparison:
    """Build numeric comparisons from already inspected run evidence."""

    if len(inspections) < 2:
        raise RunInspectionError("compare-runs requires at least two inspected runs")

    policy_tuple = tuple(modifier_policies)
    labels = {
        inspection.flow_root: run_label_from_provenance(
            inspection,
            modifier_policies=policy_tuple,
        )
        for inspection in inspections
    }
    baseline = inspections[0]
    quantities: list[QuantityComparison] = []
    for inspection in inspections[1:]:
        run_label = labels[inspection.flow_root]
        for spec in _COMPARABLE_QUANTITIES:
            baseline_value = spec.value(baseline)
            comparison_value = spec.value(inspection)
            status = "available"
            reason = None
            delta = None
            percent_delta = None
            if baseline_value is None:
                status = "unavailable"
                reason = "baseline value unavailable"
            elif comparison_value is None:
                status = "unavailable"
                reason = "comparison value unavailable"
            else:
                delta = comparison_value - baseline_value
                if baseline_value != 0:
                    percent_delta = (delta / abs(baseline_value)) * 100
            quantities.append(
                QuantityComparison(
                    run_label=run_label,
                    flow_root=inspection.flow_root,
                    quantity=spec.key,
                    label=spec.label,
                    unit=spec.unit,
                    baseline_value=_round_float(baseline_value),
                    comparison_value=_round_float(comparison_value),
                    delta=_round_float(delta),
                    percent_delta=_round_float(percent_delta),
                    status=status,
                    reason=reason,
                )
            )

    return RunComparison(
        inspections=tuple(inspections),
        labels=labels,
        quantities=tuple(quantities),
        initial_structure=_compare_initial_structures(inspections),
        energy_warning=_energy_warning(inspections),
    )


def parse_vasp_output_files(
    local_paths: Mapping[str, Path],
    display_paths: Mapping[str, str],
    workflow_spec: Mapping[str, Any],
) -> ScientificResult:
    """Derive compact scientific observations from local temporary VASP files."""

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
    executed_parameters: tuple[IncarObservation, ...] = ()
    if "vasprun" in local_paths:
        try:
            vasprun = _load_vasprun(
                Vasprun,
                local_paths["vasprun"],
                parse_eigenvalues=parse_eigenvalues,
            )
            executed_parameters = vasp_reported_parameter_observations(
                vasprun,
                source_path=display_paths.get("vasprun"),
                stage_index=_workflow_final_stage_index(workflow_spec),
            )
        except Exception as exc:
            unavailable.append(f"vasprun.xml could not be parsed: {exc}")
    else:
        unavailable.append("vasprun.xml is unavailable")

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
    structure_observation = None
    if final_structure is not None:
        try:
            final_formula = final_structure.composition.reduced_formula
            natoms = len(final_structure)
            structure_source = (
                display_paths.get("vasprun")
                if vasprun is not None and getattr(vasprun, "final_structure", None) is final_structure
                else display_paths.get("contcar")
            )
            structure_observation = structure_observation_from_structure(
                final_structure,
                source_path=structure_source,
            )
        except Exception as exc:
            unavailable.append(f"final structure summary could not be derived: {exc}")
    else:
        unavailable.append("final formula unavailable: final structure could not be derived")
        structure_observation = StructureObservation(
            source_path=None,
            unavailable=("final structure unavailable: final structure could not be derived",),
        )

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
        structure=structure_observation,
        executed_parameters=executed_parameters,
    )


def vasp_reported_parameter_observations(
    vasprun: Any,
    *,
    source_path: str | None,
    stage_index: int | None,
) -> tuple[IncarObservation, ...]:
    """Expose VASP-reported executed parameters parsed by pymatgen."""

    path = source_path or "vasprun.xml"
    observations: list[IncarObservation] = []
    for attribute, source_type in (
        ("incar", "vasprun_xml.incar"),
        ("parameters", "vasprun_xml.parameters"),
    ):
        values = _vasp_parameter_values(getattr(vasprun, attribute, None))
        if values:
            observations.append(
                IncarObservation(
                    label=source_type,
                    path=path,
                    present=True,
                    stage_index=stage_index,
                    source_type=source_type,
                    values=values,
                )
            )
    return tuple(observations)


def _termination_observation(inspection: RunInspection) -> TerminationObservation:
    unavailable: list[str] = []
    scheduler = inspection.scheduler
    if scheduler is None:
        unavailable.append(inspection.scheduler_error or "scheduler accounting was unavailable")
        scheduler_timeout = None
    else:
        scheduler_timeout = scheduler.state.upper() == "TIMEOUT"
        if scheduler.timelimit is None:
            unavailable.append("scheduler timelimit was unavailable")

    unavailable.append("VASP normal-completion marker is unavailable in diagnose-run v1")
    unavailable.append("No producer-declared custodian event artifact is inspected in diagnose-run v1")

    return TerminationObservation(
        scheduler_state=scheduler.state if scheduler else None,
        scheduler_exit_code=scheduler.exit_code if scheduler else None,
        scheduler_elapsed=scheduler.elapsed if scheduler else None,
        scheduler_timelimit=scheduler.timelimit if scheduler else None,
        scheduler_reports_timeout=scheduler_timeout,
        vasp_completed_normally=None,
        custodian_events=(),
        unavailable=tuple(unavailable),
    )


def _observe_stage_trajectories(
    ssh_host: str,
    inspection: RunInspection,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
    runner: RemoteRunner,
    timeout: float,
    max_vasprun_bytes: int,
) -> tuple[StageTrajectoryObservation, ...]:
    stage_dirs = {
        observation.label: PurePosixPath(observation.path)
        for observation in inspection.stage_directories
    }
    bindings = _bound_stage_directories(
        stage_dirs,
        PurePosixPath(inspection.result_directory.path),
        inspection.workflow_stages,
    )
    stages = {stage.index: stage for stage in inspection.workflow_stages}
    return tuple(
        _observe_stage_trajectory(
            ssh_host,
            binding,
            stages.get(binding.stage_index) if binding.stage_index is not None else None,
            inspection.executed_inputs,
            allowed_roots=allowed_roots,
            runner=runner,
            timeout=timeout,
            max_vasprun_bytes=max_vasprun_bytes,
        )
        for binding in bindings
    )


def _observe_stage_trajectory(
    ssh_host: str,
    binding: _BoundStageDirectory,
    stage: WorkflowStage | None,
    executed_inputs: Sequence[IncarObservation],
    *,
    allowed_roots: Iterable[PurePosixPath | str],
    runner: RemoteRunner,
    timeout: float,
    max_vasprun_bytes: int,
) -> StageTrajectoryObservation:
    unavailable: list[str] = []
    oszicar_path = build_remote_file_path(
        binding.directory,
        _DIAGNOSE_ARTIFACT_FILENAMES["oszicar"],
        allowed_roots=allowed_roots,
    )
    oszicar_present = remote_file_exists(ssh_host, oszicar_path, runner=runner, timeout=timeout)
    oszicar_error = None
    oszicar_trajectory = _OszicarTrajectory(None, (), (), None, (), (), ())
    if oszicar_present:
        try:
            oszicar_trajectory = parse_oszicar_trajectory(
                retrieve_remote_file(ssh_host, oszicar_path, runner=runner, timeout=timeout)
            )
        except Exception as exc:
            oszicar_error = str(exc)
            unavailable.append(f"OSZICAR could not be parsed: {exc}")
    else:
        unavailable.append("OSZICAR is unavailable")

    vasprun = _observe_vasprun_trajectory(
        ssh_host,
        binding.directory,
        allowed_roots=allowed_roots,
        runner=runner,
        timeout=timeout,
        max_vasprun_bytes=max_vasprun_bytes,
    )
    if vasprun.skipped_reason:
        unavailable.append(vasprun.skipped_reason)
    if vasprun.error:
        unavailable.append(f"vasprun trajectory enrichment unavailable: {vasprun.error}")
    if not vasprun.present:
        unavailable.append("vasprun.xml is unavailable")

    ionic_steps = _merge_max_forces(
        oszicar_trajectory.ionic_steps,
        vasprun.max_forces,
    )
    recent_ionic_steps = tuple(ionic_steps[-_DIAGNOSE_RECENT_WINDOW:])
    criteria, criteria_sources, criteria_discrepancies = _trajectory_criteria(
        binding.stage_index,
        executed_inputs,
        vasprun.parameters,
    )

    return StageTrajectoryObservation(
        stage_index=binding.stage_index,
        stage_label=binding.label,
        stage_type=stage.stage_type if stage else None,
        theory=stage.theory if stage else None,
        directory=str(binding.directory),
        oszicar_path=str(oszicar_path),
        oszicar_present=oszicar_present,
        oszicar_error=oszicar_error,
        vasprun_path=vasprun.path,
        vasprun_present=vasprun.present,
        vasprun_error=vasprun.error,
        vasprun_skipped_reason=vasprun.skipped_reason,
        criteria=criteria,
        criteria_source_values=criteria_sources,
        criteria_discrepancies=criteria_discrepancies,
        ionic_steps_observed=oszicar_trajectory.ionic_steps_observed,
        electronic_iterations_by_ionic_step=oszicar_trajectory.electronic_iterations_by_ionic_step,
        electronic_cycles=oszicar_trajectory.electronic_cycles,
        final_electronic_iteration_count=oszicar_trajectory.final_electronic_iteration_count,
        recent_electronic_iterations=oszicar_trajectory.recent_electronic_iterations,
        completed_ionic_steps=oszicar_trajectory.completed_ionic_steps,
        electronic_iterations_by_completed_ionic_step=(
            oszicar_trajectory.electronic_iterations_by_completed_ionic_step
        ),
        incomplete_electronic_iteration_count=(
            oszicar_trajectory.incomplete_electronic_iteration_count
        ),
        recent_incomplete_electronic_iterations=(
            oszicar_trajectory.recent_incomplete_electronic_iterations
        ),
        ionic_steps=ionic_steps,
        recent_ionic_steps=recent_ionic_steps,
        vasprun_ionic_steps=vasprun.ionic_steps,
        converged_electronic=vasprun.converged_electronic,
        converged_ionic=vasprun.converged_ionic,
        unavailable=tuple(unavailable),
    )


def parse_oszicar_trajectory(contents: bytes | str) -> _OszicarTrajectory:
    """Parse compact electronic/ionic trajectory evidence from OSZICAR."""

    text = contents.decode("utf-8", "replace") if isinstance(contents, bytes) else contents
    try:
        from pymatgen.io.vasp.outputs import Oszicar
    except Exception as exc:
        raise RunInspectionError(str(exc)) from exc

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        temporary_path = handle.name
    try:
        oszicar = Oszicar(temporary_path)
    finally:
        Path(temporary_path).unlink(missing_ok=True)

    electronic_steps = tuple(getattr(oszicar, "electronic_steps", ()) or ())
    ionic_steps = tuple(getattr(oszicar, "ionic_steps", ()) or ())
    algorithm_steps = _oszicar_algorithms(text)
    completed_ionic_steps = len(ionic_steps)
    completed_electronic_steps = electronic_steps[:completed_ionic_steps]
    incomplete_electronic_steps = electronic_steps[completed_ionic_steps:]
    incomplete_algorithm_steps = algorithm_steps[completed_ionic_steps:]
    final_electronic_steps = tuple(electronic_steps[-1]) if electronic_steps else ()
    electronic_cycles = _electronic_cycle_observations(
        electronic_steps,
        algorithm_steps,
        completed_ionic_steps,
    )
    recent_electronic = _recent_electronic_iterations(
        final_electronic_steps,
        algorithm_steps[-1] if algorithm_steps else (),
    )
    incomplete_cycle = tuple(incomplete_electronic_steps[-1]) if incomplete_electronic_steps else ()
    recent_incomplete = _recent_electronic_iterations(
        incomplete_cycle,
        incomplete_algorithm_steps[-1] if incomplete_algorithm_steps else (),
    )
    ionic_step_observations = _ionic_step_observations(ionic_steps, electronic_steps)
    recent_ionic = tuple(ionic_step_observations[-_DIAGNOSE_RECENT_WINDOW:])

    return _OszicarTrajectory(
        ionic_steps_observed=len(ionic_steps),
        electronic_iterations_by_ionic_step=tuple(len(step) for step in electronic_steps),
        electronic_cycles=electronic_cycles,
        final_electronic_iteration_count=len(final_electronic_steps) if final_electronic_steps else None,
        recent_electronic_iterations=recent_electronic,
        ionic_steps=ionic_step_observations,
        recent_ionic_steps=recent_ionic,
        completed_ionic_steps=completed_ionic_steps,
        electronic_iterations_by_completed_ionic_step=tuple(
            len(step) for step in completed_electronic_steps
        ),
        incomplete_electronic_iteration_count=(
            len(incomplete_cycle) if incomplete_cycle else None
        ),
        recent_incomplete_electronic_iterations=recent_incomplete,
    )


def _oszicar_algorithms(text: str) -> tuple[tuple[str | None, ...], ...]:
    groups: list[list[str | None]] = []
    current: list[str | None] = []
    pattern = re.compile(r"^\s*(?P<algorithm>[A-Za-z]+)\s*:\s*(?P<body>.*)$")
    for line in text.splitlines():
        match = pattern.match(line)
        if not match:
            continue
        tokens = match.group("body").split()
        if tokens and tokens[0] == "1" and current:
            groups.append(current)
            current = []
        current.append(match.group("algorithm").upper())
    if current:
        groups.append(current)
    return tuple(tuple(group) for group in groups)


def _electronic_cycle_observations(
    electronic_steps: Sequence[Sequence[Mapping[str, Any]]],
    algorithm_steps: Sequence[Sequence[str | None]],
    completed_ionic_steps: int,
) -> tuple[ElectronicCycleObservation, ...]:
    observations: list[ElectronicCycleObservation] = []
    for index, steps in enumerate(electronic_steps, start=1):
        step_tuple = tuple(steps)
        algorithms = algorithm_steps[index - 1] if index - 1 < len(algorithm_steps) else ()
        final_iteration = None
        if step_tuple:
            recent = _recent_electronic_iterations(step_tuple[-1:], algorithms[-1:] if algorithms else ())
            final_iteration = recent[-1] if recent else None
        observations.append(
            ElectronicCycleObservation(
                cycle_index=index,
                completed_ionic_step=index <= completed_ionic_steps,
                iterations=len(step_tuple),
                final_iteration=final_iteration,
            )
        )
    return tuple(observations)


def _recent_electronic_iterations(
    electronic_steps: Sequence[Mapping[str, Any]],
    algorithms: Sequence[str | None],
) -> tuple[ElectronicIterationObservation, ...]:
    offset = max(0, len(electronic_steps) - _DIAGNOSE_RECENT_WINDOW)
    observations: list[ElectronicIterationObservation] = []
    for local_index, step in enumerate(electronic_steps[offset:], start=offset):
        observations.append(
            ElectronicIterationObservation(
                iteration=_int_or_none(step.get("N")),
                algorithm=algorithms[local_index] if local_index < len(algorithms) else None,
                energy=_round_float(_float_or_none(step.get("E"))),
                dE=_round_float(_float_or_none(step.get("dE"))),
                deps=_round_float(_float_or_none(step.get("deps"))),
                rms=_round_float(_float_or_none(step.get("rms"))),
                rms_c=_round_float(_float_or_none(step.get("rms(c)"))),
            )
        )
    return tuple(observations)


def _ionic_step_observations(
    ionic_steps: Sequence[Mapping[str, Any]],
    electronic_steps: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[IonicStepObservation, ...]:
    observations: list[IonicStepObservation] = []
    for index, step in enumerate(ionic_steps, start=1):
        electronic_index = index - 1
        observations.append(
            IonicStepObservation(
                step_index=index,
                electronic_iterations=(
                    len(electronic_steps[electronic_index])
                    if electronic_index < len(electronic_steps)
                    else None
                ),
                free_energy=_round_float(_float_or_none(step.get("F"))),
                energy_zero=_round_float(_float_or_none(step.get("E0"))),
                dE=_round_float(_float_or_none(step.get("dE"))),
            )
        )
    return tuple(observations)


def _observe_vasprun_trajectory(
    ssh_host: str,
    directory: PurePosixPath,
    *,
    allowed_roots: Iterable[PurePosixPath | str],
    runner: RemoteRunner,
    timeout: float,
    max_vasprun_bytes: int,
) -> _VasprunTrajectory:
    path = build_remote_file_path(
        directory,
        _DIAGNOSE_ARTIFACT_FILENAMES["vasprun"],
        allowed_roots=allowed_roots,
    )
    if not remote_file_exists(ssh_host, path, runner=runner, timeout=timeout):
        return _VasprunTrajectory(present=False, path=str(path))
    try:
        size = remote_file_size(ssh_host, path, runner=runner, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as exc:
        return _VasprunTrajectory(present=True, path=str(path), error=str(exc))
    if size > max_vasprun_bytes:
        return _VasprunTrajectory(
            present=True,
            path=str(path),
            skipped_reason=f"vasprun.xml skipped because size {size} bytes exceeds limit {max_vasprun_bytes}",
        )

    try:
        from pymatgen.io.vasp.outputs import Vasprun
    except Exception as exc:
        return _VasprunTrajectory(present=True, path=str(path), error=str(exc))

    try:
        contents = retrieve_remote_file(ssh_host, path, runner=runner, timeout=timeout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return _VasprunTrajectory(
            present=True,
            path=str(path),
            error=_UNREADABLE_VASPRUN_TRAJECTORY_REASON,
        )

    try:
        with tempfile.NamedTemporaryFile("wb", delete=False) as handle:
            handle.write(contents)
            temporary_path = Path(handle.name)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", UserWarning)
                vasprun = _load_vasprun(Vasprun, temporary_path, parse_eigenvalues=False)
                if _has_malformed_xml_warning(caught):
                    return _VasprunTrajectory(
                        present=True,
                        path=str(path),
                        error=_INCOMPLETE_VASPRUN_TRAJECTORY_REASON,
                    )
                ionic_steps = tuple(getattr(vasprun, "ionic_steps", ()) or ())
                parameters = _vasp_parameter_values(getattr(vasprun, "parameters", None))
                converged_electronic = _bool_or_none(
                    getattr(vasprun, "converged_electronic", None)
                )
                converged_ionic = _bool_or_none(getattr(vasprun, "converged_ionic", None))
                max_forces = _max_forces_by_step(ionic_steps)
        finally:
            temporary_path.unlink(missing_ok=True)
    except Exception:
        return _VasprunTrajectory(
            present=True,
            path=str(path),
            error=_INCOMPLETE_VASPRUN_TRAJECTORY_REASON,
        )

    return _VasprunTrajectory(
        present=True,
        path=str(path),
        parameters=parameters,
        ionic_steps=len(ionic_steps),
        converged_electronic=converged_electronic,
        converged_ionic=converged_ionic,
        max_forces=max_forces,
    )


def _has_malformed_xml_warning(caught_warnings: Sequence[warnings.WarningMessage]) -> bool:
    return any(
        "xml is malformed" in str(item.message).lower()
        for item in caught_warnings
    )


def _max_forces_by_step(ionic_steps: Sequence[Mapping[str, Any]]) -> Mapping[int, float]:
    values: dict[int, float] = {}
    for step_index, step in enumerate(ionic_steps, start=1):
        max_force = _max_force(step.get("forces"))
        if max_force is not None:
            values[step_index] = _round_float(max_force) or max_force
    return values


def _max_force(forces: Any) -> float | None:
    rows = _matrix_rows(forces)
    maxima: list[float] = []
    for row in rows:
        components = [_float_or_none(component) for component in _matrix_rows(row)]
        if not components or any(component is None for component in components):
            continue
        maxima.append(sum(component * component for component in components if component is not None) ** 0.5)
    return max(maxima) if maxima else None


def _merge_max_forces(
    ionic_steps: Sequence[IonicStepObservation],
    max_forces: Mapping[int, float],
) -> tuple[IonicStepObservation, ...]:
    if not max_forces:
        return tuple(ionic_steps)
    merged: list[IonicStepObservation] = []
    seen: set[int] = set()
    for step in ionic_steps:
        seen.add(step.step_index)
        merged.append(
            IonicStepObservation(
                step_index=step.step_index,
                electronic_iterations=step.electronic_iterations,
                free_energy=step.free_energy,
                energy_zero=step.energy_zero,
                dE=step.dE,
                max_force=max_forces.get(step.step_index, step.max_force),
            )
        )
    for step_index, max_force in max_forces.items():
        if step_index not in seen:
            merged.append(IonicStepObservation(step_index=step_index, max_force=max_force))
    return tuple(sorted(merged, key=lambda item: item.step_index))


def _trajectory_criteria(
    stage_index: int | None,
    executed_inputs: Sequence[IncarObservation],
    vasprun_parameters: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]], tuple[str, ...]]:
    sources: list[tuple[str, Mapping[str, Any]]] = []
    for observation in executed_inputs:
        if observation.stage_index != stage_index or not observation.present or observation.error:
            continue
        sources.append((_executed_source_label(observation), observation.values))
    if vasprun_parameters:
        sources.append(("diagnose_vasprun.parameters", vasprun_parameters))

    criteria: dict[str, Any] = {}
    source_values: dict[str, Mapping[str, Any]] = {}
    discrepancies: list[str] = []
    for key in _DIAGNOSE_CRITERIA_KEYS:
        values = {
            source: values[key]
            for source, values in sources
            if key in values
        }
        if not values:
            continue
        source_values[key] = values
        observed = list(values.values())
        criteria[key] = observed[0]
        if not all(_values_match(observed[0], value) for value in observed[1:]):
            discrepancies.append(key)
    return criteria, source_values, tuple(discrepancies)


def _vasp_parameter_values(parameters: Any) -> Mapping[str, Any]:
    if parameters is None:
        return {}
    try:
        items = dict(parameters).items()
    except Exception:
        try:
            items = parameters.items()
        except Exception:
            return {}
    return {
        str(key).upper(): _json_safe_value(value)
        for key, value in items
    }


def structure_observation_from_structure(
    structure: Any,
    *,
    source_path: str | None,
) -> StructureObservation:
    """Derive generic final-structure observations from a pymatgen Structure."""

    unavailable: list[str] = []
    composition = getattr(structure, "composition", None)
    lattice = getattr(structure, "lattice", None)
    formula = getattr(composition, "formula", None)
    reduced_formula = getattr(composition, "reduced_formula", None)
    try:
        site_count = len(structure)
    except Exception:
        site_count = None
        unavailable.append("site count unavailable: final structure length could not be read")

    lattice_a = _float_or_none(getattr(lattice, "a", None))
    lattice_b = _float_or_none(getattr(lattice, "b", None))
    lattice_c = _float_or_none(getattr(lattice, "c", None))
    alpha = _float_or_none(getattr(lattice, "alpha", None))
    beta = _float_or_none(getattr(lattice, "beta", None))
    gamma = _float_or_none(getattr(lattice, "gamma", None))
    volume = _float_or_none(getattr(structure, "volume", None))
    density = _float_or_none(getattr(structure, "density", None))

    if lattice_a is None or lattice_b is None or lattice_c is None:
        unavailable.append("lattice lengths unavailable: final structure lattice is incomplete")
    if alpha is None or beta is None or gamma is None:
        unavailable.append("lattice angles unavailable: final structure lattice is incomplete")
    if volume is None:
        unavailable.append("volume unavailable: final structure volume could not be read")
    if density is None:
        unavailable.append("density unavailable: final structure density could not be read")

    c_over_a = lattice_c / lattice_a if lattice_a not in (None, 0) and lattice_c is not None else None
    if c_over_a is None:
        unavailable.append("c/a unavailable: lattice a or c is unavailable")

    return StructureObservation(
        source_path=source_path,
        formula=str(formula) if formula is not None else None,
        reduced_formula=str(reduced_formula) if reduced_formula is not None else None,
        site_count=site_count,
        lattice_a=_round_float(lattice_a),
        lattice_b=_round_float(lattice_b),
        lattice_c=_round_float(lattice_c),
        alpha=_round_float(alpha),
        beta=_round_float(beta),
        gamma=_round_float(gamma),
        volume=_round_float(volume),
        density=_round_float(density),
        c_over_a=_round_float(c_over_a),
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
    workflow_stages = _parse_workflow_stages(workflow_spec)
    provenance = _optional_mapping(submission, "provenance")

    return {
        "workflow_stages": workflow_stages,
        "initial_structure": _initial_structure_observation(flow_spec),
        "stage_dirs": _parse_stage_dirs(
            paths,
            stage_count=len(workflow_stages),
            allowed_roots=allowed_roots,
        ),
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
        options = stage.get("options") or {}
        if not isinstance(options, Mapping):
            raise RunInspectionError("workflow stage options must be a JSON object")
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
                options=_json_safe_value(options),
            )
        )

    return tuple(parsed)


def _parse_stage_dirs(
    paths: Mapping[str, Any],
    *,
    stage_count: int,
    allowed_roots: Iterable[PurePosixPath | str],
) -> dict[str, PurePosixPath]:
    value = paths.get("stage_dirs")
    if value is None:
        if stage_count == 1:
            return {}
        raise RunInspectionError("submission paths.stage_dirs is required for multi-stage runs")
    if not isinstance(value, Mapping):
        raise RunInspectionError("submission paths.stage_dirs must be a JSON object")

    parsed: dict[str, PurePosixPath] = {}
    for label, path in value.items():
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


def _observe_executed_inputs(
    ssh_host: str,
    stage_dirs: Mapping[str, PurePosixPath],
    result_dir: PurePosixPath,
    workflow_stages: Sequence[WorkflowStage],
    *,
    allowed_roots: Iterable[PurePosixPath | str],
    runner: RemoteRunner,
    timeout: float,
) -> tuple[IncarObservation, ...]:
    observations: list[IncarObservation] = []
    seen_paths: set[str] = set()

    for binding in _bound_stage_directories(stage_dirs, result_dir, workflow_stages):
        observation = _observe_incar(
            ssh_host,
            label=binding.label,
            directory=binding.directory,
            stage_index=binding.stage_index,
            allowed_roots=allowed_roots,
            runner=runner,
            timeout=timeout,
        )
        if observation.path in seen_paths:
            continue
        observations.append(observation)
        seen_paths.add(observation.path)

    return tuple(observations)


def _bound_stage_directories(
    stage_dirs: Mapping[str, PurePosixPath],
    result_dir: PurePosixPath,
    workflow_stages: Sequence[WorkflowStage],
) -> tuple[_BoundStageDirectory, ...]:
    bindings: list[_BoundStageDirectory] = []
    seen_directories: set[str] = set()
    stage_count = len(workflow_stages)

    for index, (label, directory) in enumerate(stage_dirs.items(), start=1):
        bindings.append(_BoundStageDirectory(label, directory, index))
        seen_directories.add(str(directory))

    result_stage_index = None
    if not stage_dirs and stage_count == 1:
        result_stage_index = 1
    elif stage_dirs and stage_count:
        result_stage_index = stage_count

    if str(result_dir) not in seen_directories:
        bindings.append(_BoundStageDirectory("result_dir", result_dir, result_stage_index))

    return tuple(bindings)


def _observe_incar(
    ssh_host: str,
    *,
    label: str,
    directory: PurePosixPath,
    stage_index: int | None,
    allowed_roots: Iterable[PurePosixPath | str],
    runner: RemoteRunner,
    timeout: float,
) -> IncarObservation:
    path = build_remote_file_path(
        directory,
        "INCAR",
        allowed_roots=allowed_roots,
    )
    present = remote_file_exists(ssh_host, path, runner=runner, timeout=timeout)
    if not present:
        return IncarObservation(
            label=label,
            path=str(path),
            present=False,
            stage_index=stage_index,
        )

    try:
        contents = retrieve_remote_file(
            ssh_host,
            path,
            runner=runner,
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return IncarObservation(
            label=label,
            path=str(path),
            present=True,
            stage_index=stage_index,
            error=str(exc),
        )

    values, error = parse_incar_contents(contents)
    return IncarObservation(
        label=label,
        path=str(path),
        present=True,
        stage_index=stage_index,
        values=values,
        error=error,
    )


def parse_incar_contents(contents: bytes | str) -> tuple[Mapping[str, Any], str | None]:
    """Parse VASP INCAR content with pymatgen into JSON-safe key/value evidence."""

    text = contents.decode("utf-8") if isinstance(contents, bytes) else contents
    try:
        from pymatgen.io.vasp.inputs import Incar
    except Exception as exc:
        return {}, str(exc)

    try:
        if hasattr(Incar, "from_str"):
            incar = Incar.from_str(text)
        else:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                handle.write(text)
                temporary_path = handle.name
            try:
                incar = Incar.from_file(temporary_path)
            finally:
                Path(temporary_path).unlink(missing_ok=True)
    except Exception as exc:
        return {}, str(exc)

    return {
        str(key).upper(): _json_safe_value(value)
        for key, value in dict(incar).items()
    }, None


def compare_requested_options_to_executed_inputs(
    stages: Sequence[WorkflowStage],
    executed_inputs: Sequence[IncarObservation],
    modifier_policies: Iterable[Mapping[str, Any]],
) -> tuple[InputExpectationObservation, ...]:
    """Compare producer-requested option effects with observed VASP inputs."""

    policies = tuple(_input_expectation_policies(modifier_policies))
    if not policies:
        return ()

    inputs_by_index: dict[int, list[IncarObservation]] = {}
    for observation in executed_inputs:
        if observation.stage_index is not None:
            inputs_by_index.setdefault(observation.stage_index, []).append(observation)

    observations: list[InputExpectationObservation] = []
    for stage in stages:
        executed = tuple(inputs_by_index.get(stage.index, ()))
        for policy in policies:
            if policy["modifier"] not in stage.modifiers:
                continue
            requested_value = _nested_option_value(
                stage.options,
                policy["option_key"],
                policy["method_key"],
            )
            if requested_value is None:
                continue
            effect = policy["effects"].get(str(requested_value))
            if not isinstance(effect, Mapping):
                continue
            for input_key, expected_value in effect.items():
                status, observed_value, source_values, reason = _evaluate_executed_parameter(
                    executed,
                    str(input_key).upper(),
                    expected_value,
                )
                observations.append(
                    InputExpectationObservation(
                        stage_label=_stage_evidence_label(stage, executed),
                        stage_index=stage.index,
                        option_path=f"{policy['option_key']}.{policy['method_key']}",
                        requested_value=requested_value,
                        input_key=str(input_key).upper(),
                        expected_value=expected_value,
                        observed_value=observed_value,
                        status=status,
                        source_values=source_values,
                        reason=reason,
                    )
                )
    return tuple(observations)


def _evaluate_executed_parameter(
    executed: Sequence[IncarObservation],
    input_key: str,
    expected_value: Any,
) -> tuple[str, Any, Mapping[str, Any], str | None]:
    readable = tuple(
        observation
        for observation in executed
        if observation.present and observation.error is None
    )
    if not readable:
        reason = "no readable executed-input evidence was bound to this stage"
        errors = [
            f"{observation.label}: {observation.error}"
            for observation in executed
            if observation.error
        ]
        if errors:
            reason = "; ".join(errors)
        return "unavailable", None, {}, reason

    source_values = {
        _executed_source_label(observation): observation.values.get(input_key, None)
        for observation in readable
    }
    present_values = [
        value
        for value in source_values.values()
        if value is not None
    ]
    if not present_values:
        return (
            "absent",
            None,
            source_values,
            f"{input_key} was absent from readable executed-input evidence",
        )

    if len(present_values) != len(source_values):
        return (
            "discrepancy",
            _consensus_value(present_values),
            source_values,
            "executed-input sources disagree about parameter presence",
        )

    if not all(_values_match(present_values[0], value) for value in present_values[1:]):
        return (
            "discrepancy",
            _consensus_value(present_values),
            source_values,
            "executed-input sources disagree about parameter value",
        )

    observed_value = present_values[0]
    if not _values_match(expected_value, observed_value):
        return (
            "discrepancy",
            observed_value,
            source_values,
            "executed value differs from producer-requested option effect",
        )

    return "supported", observed_value, source_values, None


def _stage_evidence_label(
    stage: WorkflowStage,
    executed: Sequence[IncarObservation],
) -> str:
    retained = [observation.label for observation in executed if observation.source_type == "retained_incar"]
    if retained:
        return retained[0]
    return stage.label or f"stage_{stage.index}"


def _executed_source_label(observation: IncarObservation) -> str:
    return f"{observation.source_type}:{observation.label}"


def _workflow_final_stage_index(workflow_spec: Mapping[str, Any]) -> int | None:
    stages = workflow_spec.get("stages")
    if not isinstance(stages, list) or not stages:
        return None
    return len(stages)


def _consensus_value(values: Sequence[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    if all(_values_match(first, value) for value in values[1:]):
        return first
    return tuple(values)


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
    if not any(key in present for key in _SCIENTIFIC_READ_KEYS):
        return ScientificResult(
            source_paths=tuple(observation.path for observation in present.values()),
            unavailable=("final VASP structure/result artifacts are unavailable",),
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


def _initial_structure_observation(flow_spec: Mapping[str, Any]) -> InitialStructureObservation:
    structure = flow_spec.get("structure")
    if not isinstance(structure, Mapping):
        return InitialStructureObservation(
            status="unavailable",
            reason="flow_spec.structure was not preserved as a JSON object.",
        )

    structure_type = structure.get("type")
    representation_type = str(structure_type) if structure_type is not None else "unknown"
    text = structure.get("text")
    if isinstance(text, str) and text:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        structure_format = structure.get("format")
        if isinstance(structure_format, str) and structure_format:
            representation_type = f"{representation_type}:{structure_format}"
        return InitialStructureObservation(
            status="available",
            representation_type=representation_type,
            representation_hash=digest,
        )

    return InitialStructureObservation(
        status="unavailable",
        representation_type=representation_type,
        reason=(
            "Submitted structure content was not preserved in a stable exact "
            "representation; path-only or parsed provenance cannot establish identity."
        ),
    )


def run_label_from_provenance(
    inspection: RunInspection,
    *,
    modifier_policies: Iterable[Mapping[str, Any]] = (),
) -> str:
    """Create a concise display label from producer-requested workflow provenance."""

    theories = _ordered_unique(stage.theory.upper() for stage in inspection.workflow_stages)
    base = " -> ".join(theories) if theories else inspection.flow_root
    modifiers: list[str] = []
    for stage in inspection.workflow_stages:
        for modifier in stage.modifiers:
            label = _modifier_option_label(stage, modifier, modifier_policies)
            if label not in modifiers:
                modifiers.append(label)
    if modifiers:
        return f"{base} + {' + '.join(modifiers)}"
    return base


def _modifier_option_label(
    stage: WorkflowStage,
    modifier: str,
    modifier_policies: Iterable[Mapping[str, Any]],
) -> str:
    for policy in modifier_policies:
        if policy.get("modifier") != modifier:
            continue
        option_key = policy.get("option_key")
        method_key = policy.get("method_key")
        if not isinstance(option_key, str) or not isinstance(method_key, str):
            continue
        requested_value = _nested_option_value(stage.options, option_key, method_key)
        if requested_value is None:
            continue
        for method in policy.get("methods") or []:
            if not isinstance(method, Mapping):
                continue
            if method.get("value") == requested_value and isinstance(method.get("label"), str):
                return str(method["label"])
        return f"{modifier}:{requested_value}"
    return str(modifier).replace("_", " ")


def _input_expectation_policies(
    modifier_policies: Iterable[Mapping[str, Any]],
) -> Iterable[Mapping[str, Any]]:
    for policy in modifier_policies:
        modifier = policy.get("modifier")
        option_key = policy.get("option_key")
        method_key = policy.get("method_key")
        methods = policy.get("methods")
        if not all(isinstance(value, str) and value for value in (modifier, option_key, method_key)):
            continue
        if not isinstance(methods, list):
            continue
        effects: dict[str, Mapping[str, Any]] = {}
        for method in methods:
            if not isinstance(method, Mapping):
                continue
            value = method.get("value")
            effect = method.get("incar_effect")
            if isinstance(value, str) and isinstance(effect, Mapping):
                effects[value] = {
                    str(key).upper(): _json_safe_value(item)
                    for key, item in effect.items()
                }
        if effects:
            yield {
                "modifier": modifier,
                "option_key": option_key,
                "method_key": method_key,
                "effects": effects,
            }


def _nested_option_value(
    options: Mapping[str, Any],
    option_key: str,
    method_key: str,
) -> Any:
    option_value = options.get(option_key)
    if isinstance(option_value, Mapping):
        return option_value.get(method_key)
    return option_value


def _compare_initial_structures(
    inspections: Sequence[RunInspection],
) -> InitialStructureComparison:
    observations = [inspection.initial_structure for inspection in inspections]
    if not all(observation.status == "available" and observation.representation_hash for observation in observations):
        return InitialStructureComparison(
            status="unavailable",
            reason="Not every run preserved an exact submitted structure representation.",
        )
    hashes = {observation.representation_hash for observation in observations}
    if len(hashes) == 1:
        return InitialStructureComparison(
            status="match",
            reason="Exact submitted structure representations have identical hashes.",
        )
    return InitialStructureComparison(
        status="mismatch",
        reason="Exact submitted structure representation hashes differ.",
    )


def _energy_warning(inspections: Sequence[RunInspection]) -> str | None:
    baseline = _requested_configuration_signature(inspections[0])
    if any(_requested_configuration_signature(inspection) != baseline for inspection in inspections[1:]):
        return (
            "Requested theory/modifier configurations differ; total energy and "
            "energy/atom are observations, not a ranking of method quality."
        )
    return None


def _requested_configuration_signature(inspection: RunInspection) -> tuple[Any, ...]:
    return tuple(
        (
            stage.stage_type,
            stage.theory,
            tuple(stage.modifiers),
            json.dumps(stage.options, sort_keys=True, separators=(",", ":")),
        )
        for stage in inspection.workflow_stages
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def _structure_value(inspection: RunInspection, field_name: str) -> float | None:
    structure = inspection.scientific.structure
    if structure is None:
        return None
    return _float_or_none(getattr(structure, field_name, None))


def _scientific_value(inspection: RunInspection, field_name: str) -> float | None:
    return _float_or_none(getattr(inspection.scientific, field_name, None))


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


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _round_float(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _bool_or_none(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


_COMPARABLE_QUANTITIES = (
    _ComparableQuantity(
        "lattice_a",
        "lattice a",
        "A",
        lambda inspection: _structure_value(inspection, "lattice_a"),
    ),
    _ComparableQuantity(
        "lattice_b",
        "lattice b",
        "A",
        lambda inspection: _structure_value(inspection, "lattice_b"),
    ),
    _ComparableQuantity(
        "lattice_c",
        "lattice c",
        "A",
        lambda inspection: _structure_value(inspection, "lattice_c"),
    ),
    _ComparableQuantity(
        "alpha",
        "alpha",
        "deg",
        lambda inspection: _structure_value(inspection, "alpha"),
    ),
    _ComparableQuantity(
        "beta",
        "beta",
        "deg",
        lambda inspection: _structure_value(inspection, "beta"),
    ),
    _ComparableQuantity(
        "gamma",
        "gamma",
        "deg",
        lambda inspection: _structure_value(inspection, "gamma"),
    ),
    _ComparableQuantity(
        "volume",
        "volume",
        "A^3",
        lambda inspection: _structure_value(inspection, "volume"),
    ),
    _ComparableQuantity(
        "density",
        "density",
        "g/cm^3",
        lambda inspection: _structure_value(inspection, "density"),
    ),
    _ComparableQuantity(
        "c_over_a",
        "c/a cell-axis ratio",
        None,
        lambda inspection: _structure_value(inspection, "c_over_a"),
    ),
    _ComparableQuantity(
        "energy_per_atom_ev",
        "energy/atom",
        "eV/atom",
        lambda inspection: _scientific_value(inspection, "energy_per_atom_ev"),
    ),
    _ComparableQuantity(
        "band_gap_ev",
        "band gap",
        "eV",
        lambda inspection: _scientific_value(inspection, "band_gap_ev"),
    ),
)
