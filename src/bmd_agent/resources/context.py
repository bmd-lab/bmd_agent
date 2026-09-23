from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bmd_agent.resources.custodian import CUSTODIAN_INTERVENTION_EVIDENCE
from bmd_agent.resources.run import (
    AGENT_COMPARISON,
    ARTIFACT_OBSERVATION,
    CONVERGENCE_PROGRESS_ASSESSMENT,
    EXECUTED_INPUT,
    LOG_OBSERVATION,
    PRODUCER_PROVENANCE,
    PRODUCER_REQUESTED,
    PYMATGEN_DERIVED,
    SCHEDULER_OBSERVATION,
    TRAJECTORY_OBSERVATION,
    TRAJECTORY_PROGRESS_EVIDENCE,
    DirectVaspInspection,
    IncarObservation,
    JobInspection,
    RunDiagnosis,
    RunInspection,
    ScientificResult,
    StageTrajectoryObservation,
    WorkflowStage,
    derive_trajectory_progress_evidence,
    run_label_from_provenance,
)


@dataclass(frozen=True)
class EvidenceGap:
    """Unavailable evidence already established during inspection."""

    evidence_type: str
    scope: str
    reason: str
    source: str | None = None


@dataclass(frozen=True)
class ScientificIdentitySummary:
    """Compact, source-safe identity derived from existing observations."""

    job_id: str
    calculation_type: str
    formula: str | None = None
    site_count: int | None = None
    structure_evidence_type: str | None = None
    producer_workflow: str | None = None
    producer_stage_types: tuple[str, ...] = ()
    producer_theories: tuple[str, ...] = ()
    producer_modifiers: tuple[str, ...] = ()
    producer_evidence_type: str | None = None
    executed_input_indicators: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    executed_input_evidence_type: str | None = None


@dataclass(frozen=True)
class ScientificContext:
    """Reasoning-facing aggregate over native BMD Agent evidence.

    This object is the intended future boundary between infrastructure/evidence
    acquisition and later reasoning or enrichment. It is not itself a reasoning
    engine, and it must not acquire new evidence.
    """

    job: JobInspection
    identity: ScientificIdentitySummary
    evidence_gaps: tuple[EvidenceGap, ...] = ()


def build_scientific_context(inspection: JobInspection) -> ScientificContext:
    """Build a pure context facade around an existing job inspection."""

    return ScientificContext(
        job=inspection,
        identity=_identity_summary(inspection),
        evidence_gaps=_evidence_gaps(inspection),
    )


def _identity_summary(inspection: JobInspection) -> ScientificIdentitySummary:
    scientific = _scientific_result(inspection)
    formula, site_count, structure_evidence_type = _structure_identity(scientific)
    stages = _producer_stages(inspection)
    executed_indicators = _executed_input_indicators(_executed_inputs(inspection))

    return ScientificIdentitySummary(
        job_id=inspection.job_id,
        calculation_type=inspection.calculation_type,
        formula=formula,
        site_count=site_count,
        structure_evidence_type=structure_evidence_type,
        producer_workflow=_producer_workflow(inspection),
        producer_stage_types=_ordered_unique(stage.stage_type for stage in stages),
        producer_theories=_ordered_unique(stage.theory for stage in stages),
        producer_modifiers=_ordered_unique(
            modifier for stage in stages for modifier in stage.modifiers
        ),
        producer_evidence_type=PRODUCER_REQUESTED if stages else None,
        executed_input_indicators=executed_indicators,
        executed_input_evidence_type=EXECUTED_INPUT if executed_indicators else None,
    )


def _structure_identity(
    scientific: ScientificResult | None,
) -> tuple[str | None, int | None, str | None]:
    if scientific is None:
        return None, None, None

    structure = scientific.structure
    if structure is not None:
        formula = structure.reduced_formula or structure.formula
        if formula is not None or structure.site_count is not None:
            return formula, structure.site_count, structure.evidence_type

    if scientific.final_formula is not None:
        return scientific.final_formula, None, scientific.evidence_type

    return None, None, None


def _producer_workflow(inspection: JobInspection) -> str | None:
    run = _bmd_run_inspection(inspection)
    if run is None or not run.workflow_stages:
        return None
    return run_label_from_provenance(run)


def _producer_stages(inspection: JobInspection) -> tuple[WorkflowStage, ...]:
    run = _bmd_run_inspection(inspection)
    if run is None:
        return ()
    return run.workflow_stages


def _executed_input_indicators(
    inputs: Sequence[IncarObservation],
) -> Mapping[str, Mapping[str, Any]]:
    indicators: dict[str, Mapping[str, Any]] = {}

    spin = _indicator_from_parameter(inputs, "ISPIN", _spin_polarized_value)
    if spin is not None:
        indicators["spin_polarized"] = spin

    dft_u = _indicator_from_parameter(inputs, "LDAU", _bool_value)
    if dft_u is not None:
        indicators["dft_u_enabled"] = dft_u

    soc = _indicator_from_parameter(inputs, "LSORBIT", _bool_value)
    if soc is not None:
        indicators["soc_enabled"] = soc

    hybrid = _indicator_from_parameter(inputs, "LHFCALC", _bool_value)
    if hybrid is not None:
        indicators["hybrid_enabled"] = hybrid

    dispersion = _indicator_from_parameter(inputs, "IVDW", lambda value: value)
    if dispersion is not None:
        indicators["dispersion_indicator"] = dispersion

    return indicators


def _indicator_from_parameter(
    inputs: Sequence[IncarObservation],
    parameter: str,
    transform,
) -> Mapping[str, Any] | None:
    source_values = {
        _executed_source_label(observation): observation.values[parameter]
        for observation in inputs
        if (
            observation.present
            and observation.error is None
            and parameter in observation.values
        )
    }
    if not source_values:
        return None

    values = list(source_values.values())
    first = values[0]
    status = "available" if all(value == first for value in values[1:]) else "discrepant"
    observed = transform(first) if status == "available" else None
    return {
        "evidence_type": EXECUTED_INPUT,
        "parameter": parameter,
        "value": observed,
        "status": status,
        "source_values": source_values,
    }


def _evidence_gaps(inspection: JobInspection) -> tuple[EvidenceGap, ...]:
    gaps: list[EvidenceGap] = []

    if inspection.scheduler is None:
        _add_gap(
            gaps,
            SCHEDULER_OBSERVATION,
            "scheduler",
            inspection.scheduler_error or "scheduler accounting was unavailable",
        )
    if inspection.scheduler_work_dir is None:
        _add_gap(
            gaps,
            SCHEDULER_OBSERVATION,
            "scheduler_work_dir",
            inspection.calculation_reason or "scheduler WorkDir was unavailable",
        )
    if inspection.calculation_type == "unknown" and inspection.calculation_reason:
        _add_gap(
            gaps,
            ARTIFACT_OBSERVATION,
            "calculation_identity",
            inspection.calculation_reason,
        )

    if inspection.bmd_compute is not None:
        _bmd_compute_gaps(gaps, inspection.bmd_compute)
    elif inspection.direct_vasp is not None:
        _direct_vasp_gaps(gaps, inspection.direct_vasp)
    else:
        _add_gap(
            gaps,
            PRODUCER_PROVENANCE,
            "producer",
            inspection.calculation_reason or "no supported calculation evidence was identified",
        )

    return _deduplicated_gaps(gaps)


def _bmd_compute_gaps(gaps: list[EvidenceGap], diagnosis: RunDiagnosis) -> None:
    run = diagnosis.inspection
    if not run.producer_git:
        _add_gap(
            gaps,
            PRODUCER_PROVENANCE,
            "producer",
            "producer git provenance unavailable",
            run.submission_path,
        )
    if not run.custodian_policy.available:
        _add_gap(
            gaps,
            PRODUCER_PROVENANCE,
            "execution_policy.custodian",
            run.custodian_policy.reason or "Custodian execution-policy provenance unavailable",
            run.submission_path,
        )
    _custodian_gaps(gaps, run.custodian_evidence)

    if run.runtime and not run.runtime.sources:
        _add_gap(
            gaps,
            LOG_OBSERVATION,
            "runtime",
            "no runner log content was readable",
        )

    _initial_structure_gap(gaps, run)
    _path_gaps(gaps, run.final_artifacts)
    _scientific_gaps(gaps, run.scientific)
    if run.comparison.status == "unavailable":
        _add_gap(
            gaps,
            AGENT_COMPARISON,
            "producer_result_comparison",
            run.comparison.reason or "producer result comparison unavailable",
        )
    _trajectory_gaps(gaps, diagnosis.trajectories)
    _assessment_gaps(gaps, diagnosis.assessments)


def _direct_vasp_gaps(gaps: list[EvidenceGap], direct: DirectVaspInspection) -> None:
    _add_gap(
        gaps,
        PRODUCER_PROVENANCE,
        "producer",
        direct.producer_reason,
    )
    _path_gaps(gaps, direct.artifacts)
    _scientific_gaps(gaps, direct.scientific)
    _trajectory_gaps(gaps, (direct.trajectory,))
    _assessment_gaps(gaps, direct.assessments)
    _custodian_gaps(gaps, direct.custodian_evidence)


def _custodian_gaps(gaps: list[EvidenceGap], evidence_items: Sequence[Any]) -> None:
    for evidence in evidence_items:
        if evidence.error:
            _add_gap(
                gaps,
                CUSTODIAN_INTERVENTION_EVIDENCE,
                "custodian_intervention_history",
                evidence.error,
                evidence.source_path,
            )


def _initial_structure_gap(gaps: list[EvidenceGap], run: RunInspection) -> None:
    initial = run.initial_structure
    if initial.status == "unavailable":
        _add_gap(
            gaps,
            initial.evidence_type,
            "initial_structure",
            initial.reason or "initial structure provenance unavailable",
        )


def _path_gaps(gaps: list[EvidenceGap], paths: Sequence[Any]) -> None:
    for observation in paths:
        if getattr(observation, "present", True):
            continue
        label = getattr(observation, "label", "path")
        kind = getattr(observation, "kind", "path")
        _add_gap(
            gaps,
            getattr(observation, "evidence_type", ARTIFACT_OBSERVATION),
            str(label),
            f"{kind} was not present",
            getattr(observation, "path", None),
        )


def _scientific_gaps(
    gaps: list[EvidenceGap],
    scientific: ScientificResult | None,
) -> None:
    if scientific is None:
        _add_gap(
            gaps,
            PYMATGEN_DERIVED,
            "scientific_result",
            "scientific result evidence unavailable",
        )
        return

    if scientific.error:
        _add_gap(
            gaps,
            scientific.evidence_type,
            "scientific_result",
            scientific.error,
        )
    for reason in scientific.unavailable:
        _add_gap(
            gaps,
            scientific.evidence_type,
            "scientific_result",
            reason,
        )

    structure = scientific.structure
    if structure is None:
        if scientific.final_formula is None:
            _add_gap(
                gaps,
                scientific.evidence_type,
                "structure",
                "structure-derived identity unavailable",
            )
        return

    if structure.error:
        _add_gap(gaps, structure.evidence_type, "structure", structure.error)
    for reason in structure.unavailable:
        _add_gap(gaps, structure.evidence_type, "structure", reason, structure.source_path)


def _trajectory_gaps(
    gaps: list[EvidenceGap],
    trajectories: Sequence[StageTrajectoryObservation],
) -> None:
    for trajectory in trajectories:
        scope = _trajectory_scope(trajectory)
        if not trajectory.oszicar_present:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.OSZICAR",
                "OSZICAR trajectory unavailable",
                trajectory.oszicar_path,
            )
        if trajectory.oszicar_error:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.OSZICAR",
                f"OSZICAR parsing unavailable: {trajectory.oszicar_error}",
                trajectory.oszicar_path,
            )
        if trajectory.vasprun_skipped_reason:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.vasprun",
                trajectory.vasprun_skipped_reason,
                trajectory.vasprun_path,
            )
        elif trajectory.vasprun_error:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.vasprun",
                trajectory.vasprun_error,
                trajectory.vasprun_path,
            )
        elif not trajectory.vasprun_present:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.vasprun",
                "vasprun.xml trajectory unavailable",
                trajectory.vasprun_path,
            )

        if not trajectory.outcar_present:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.OUTCAR",
                "OUTCAR force evidence unavailable",
                trajectory.outcar_path,
            )
        if trajectory.outcar_error:
            _add_gap(
                gaps,
                TRAJECTORY_OBSERVATION,
                f"{scope}.OUTCAR",
                f"OUTCAR force evidence unavailable: {trajectory.outcar_error}",
                trajectory.outcar_path,
            )
        for reason in trajectory.unavailable:
            _add_gap(gaps, trajectory.evidence_type, scope, reason)

        progress = derive_trajectory_progress_evidence(trajectory)
        for reason in progress.limitations:
            _add_gap(gaps, progress.evidence_type, scope, reason)


def _assessment_gaps(gaps: list[EvidenceGap], assessments: Sequence[Any]) -> None:
    for assessment in assessments:
        scope = f"{assessment.stage_label}.{assessment.scope}"
        for reason in assessment.limitations:
            _add_gap(gaps, assessment.evidence_type, scope, reason)


def _bmd_run_inspection(inspection: JobInspection) -> RunInspection | None:
    if inspection.bmd_compute is None:
        return None
    return inspection.bmd_compute.inspection


def _scientific_result(inspection: JobInspection) -> ScientificResult | None:
    if inspection.direct_vasp is not None:
        return inspection.direct_vasp.scientific
    run = _bmd_run_inspection(inspection)
    return run.scientific if run is not None else None


def _executed_inputs(inspection: JobInspection) -> tuple[IncarObservation, ...]:
    if inspection.direct_vasp is not None:
        return inspection.direct_vasp.executed_inputs
    run = _bmd_run_inspection(inspection)
    return run.executed_inputs if run is not None else ()


def _add_gap(
    gaps: list[EvidenceGap],
    evidence_type: str,
    scope: str,
    reason: str | None,
    source: str | None = None,
) -> None:
    if reason:
        gaps.append(EvidenceGap(evidence_type, scope, reason, source))


def _deduplicated_gaps(gaps: Sequence[EvidenceGap]) -> tuple[EvidenceGap, ...]:
    deduplicated: list[EvidenceGap] = []
    seen: set[tuple[str, str, str, str | None]] = set()
    for gap in gaps:
        key = (gap.evidence_type, gap.scope, gap.reason, gap.source)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(gap)
    return tuple(deduplicated)


def _trajectory_scope(trajectory: StageTrajectoryObservation) -> str:
    if trajectory.stage_index is not None:
        return f"stage_{trajectory.stage_index}"
    return trajectory.stage_label


def _executed_source_label(observation: IncarObservation) -> str:
    return f"{observation.source_type}:{observation.label}"


def _ordered_unique(values) -> tuple[str, ...]:
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text not in unique:
            unique.append(text)
    return tuple(unique)


def _spin_polarized_value(value: Any) -> bool | None:
    number = _int_or_none(value)
    if number is None:
        return None
    return number == 2


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", ".true.", "t", "1", "yes"}:
            return True
        if normalized in {"false", ".false.", "f", "0", "no"}:
            return False
    number = _int_or_none(value)
    if number is not None:
        return number != 0
    return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
