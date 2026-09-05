from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from bmd_agent.resources.bmdex import BmdexCompositionEvidence, EnrichedScientificContext
from bmd_agent.resources.context import EvidenceGap
from bmd_agent.resources.run import (
    ARTIFACT_OBSERVATION,
    CONVERGENCE_PROGRESS_ASSESSMENT,
    EXECUTED_INPUT,
    PYMATGEN_DERIVED,
    PRODUCER_PROVENANCE,
    PRODUCER_REQUESTED,
    SCHEDULER_OBSERVATION,
    TRAJECTORY_OBSERVATION,
    TRAJECTORY_PROGRESS_EVIDENCE,
    ConvergenceProgressAssessment,
    JobInspection,
    ScientificResult,
    StageTrajectoryObservation,
    TrajectoryProgressEvidence,
    WorkflowStage,
    derive_trajectory_progress_evidence,
)

CATEGORY_OBSERVATION = "observation"
CATEGORY_DERIVED_OBSERVATION = "derived_observation"
CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE = "contextual_reference_evidence"
CATEGORY_ASSESSMENT = "assessment"
CATEGORY_EVIDENCE_GAP = "evidence_gap"
CATEGORY_LIMITATION = "limitation"


@dataclass(frozen=True)
class EvidenceSourceRef:
    """Reference to the native observation that supports a summary item."""

    source_evidence_type: str
    source_scope: str
    native_path: str
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScientificEvidenceItem:
    """One compact, source-labelled evidence statement."""

    category: str
    subject: str
    predicate: str
    value: Any = None
    unit: str | None = None
    status: str | None = None
    source: EvidenceSourceRef | None = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScientificEvidenceSummary:
    """Pure synthesis view over an EnrichedScientificContext.

    The chain is intentionally one-way and read-only:
    EnrichedScientificContext -> build_scientific_evidence_summary() ->
    ScientificEvidenceSummary. This module does not inspect files, run
    subprocesses, query schedulers, call BMDex, or alter the scientific meaning
    of the underlying observations.
    """

    source_context: EnrichedScientificContext
    items: tuple[ScientificEvidenceItem, ...] = ()

    def by_category(self, category: str) -> tuple[ScientificEvidenceItem, ...]:
        return tuple(item for item in self.items if item.category == category)


def build_scientific_evidence_summary(
    context: EnrichedScientificContext,
) -> ScientificEvidenceSummary:
    """Build a compact reasoning-facing summary from existing evidence only."""

    items: list[ScientificEvidenceItem] = []
    _add_job_items(items, context)
    _add_identity_items(items, context)
    _add_producer_items(items, context)
    _add_scientific_result_items(items, context)
    _add_executed_input_items(items, context)
    _add_trajectory_items(items, context)
    _add_assessment_items(items, context)
    _add_bmdex_items(items, context)
    _add_context_gap_items(items, context)
    return ScientificEvidenceSummary(source_context=context, items=tuple(items))


def _add_job_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    job = context.base.job
    record = job.scheduler
    if record is None:
        return

    source = EvidenceSourceRef(
        SCHEDULER_OBSERVATION,
        f"job:{job.job_id}",
        "source_context.base.job.scheduler",
    )
    _append(
        items,
        CATEGORY_OBSERVATION,
        "scheduler",
        "state",
        getattr(record, "state", None),
        source=_source_with_native_path(source, "source_context.base.job.scheduler.state"),
    )
    _append(
        items,
        CATEGORY_OBSERVATION,
        "scheduler",
        "elapsed",
        getattr(record, "elapsed", None),
        source=_source_with_native_path(source, "source_context.base.job.scheduler.elapsed"),
    )
    _append(
        items,
        CATEGORY_OBSERVATION,
        "scheduler",
        "allocated_cpus",
        getattr(record, "allocated_cpus", None),
        source=_source_with_native_path(
            source,
            "source_context.base.job.scheduler.allocated_cpus",
        ),
    )
    _append(
        items,
        CATEGORY_OBSERVATION,
        "scheduler",
        "node_list",
        getattr(record, "node_list", None),
        source=_source_with_native_path(
            source,
            "source_context.base.job.scheduler.node_list",
        ),
    )


def _add_identity_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    identity = context.base.identity
    _append(
        items,
        CATEGORY_OBSERVATION,
        "calculation",
        "calculation_type",
        identity.calculation_type,
        source=EvidenceSourceRef(
            ARTIFACT_OBSERVATION,
            f"job:{identity.job_id}",
            "source_context.base.identity.calculation_type",
        ),
    )
    if identity.formula is not None:
        _append(
            items,
            CATEGORY_DERIVED_OBSERVATION,
            "structure",
            "formula",
            identity.formula,
            source=EvidenceSourceRef(
                identity.structure_evidence_type or PYMATGEN_DERIVED,
                "structure_identity",
                "source_context.base.identity.formula",
            ),
        )
    if identity.site_count is not None:
        _append(
            items,
            CATEGORY_DERIVED_OBSERVATION,
            "structure",
            "site_count",
            identity.site_count,
            source=EvidenceSourceRef(
                identity.structure_evidence_type or PYMATGEN_DERIVED,
                "structure_identity",
                "source_context.base.identity.site_count",
            ),
        )


def _add_producer_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    identity = context.base.identity
    if identity.producer_workflow is not None:
        _append(
            items,
            CATEGORY_OBSERVATION,
            "producer_workflow",
            "label",
            identity.producer_workflow,
            source=EvidenceSourceRef(
                identity.producer_evidence_type or PRODUCER_REQUESTED,
                "BMD Compute workflow",
                "source_context.base.identity.producer_workflow",
            ),
        )

    for index, stage in enumerate(_producer_stages(context.base.job)):
        _append(
            items,
            CATEGORY_OBSERVATION,
            "producer_workflow",
            "stage",
            _stage_value(stage),
            source=EvidenceSourceRef(
                PRODUCER_REQUESTED,
                f"stage:{stage.index}",
                f"source_context.base.job.bmd_compute.inspection.workflow_stages[{index}]",
            ),
        )

    run = (
        context.base.job.bmd_compute.inspection
        if context.base.job.bmd_compute is not None
        else None
    )
    if run is None or not run.producer_git:
        return

    _append(
        items,
        CATEGORY_OBSERVATION,
        "producer_provenance",
        "git_commit",
        run.producer_git.get("git_commit"),
        source=EvidenceSourceRef(
            PRODUCER_PROVENANCE,
            "BMD Compute submission",
            "source_context.base.job.bmd_compute.inspection.producer_git.git_commit",
            {"producer_git": dict(run.producer_git)},
        ),
    )
    _append(
        items,
        CATEGORY_OBSERVATION,
        "producer_provenance",
        "git_state",
        run.producer_git.get("state"),
        source=EvidenceSourceRef(
            PRODUCER_PROVENANCE,
            "BMD Compute submission",
            "source_context.base.job.bmd_compute.inspection.producer_git.state",
            {"producer_git": dict(run.producer_git)},
        ),
    )


def _add_scientific_result_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    scientific = _scientific_result(context.base.job)
    if scientific is None:
        return

    base_path = _scientific_result_native_path(context.base.job)
    source = EvidenceSourceRef(
        scientific.evidence_type,
        "scientific_result",
        base_path,
        {"source_paths": tuple(scientific.source_paths)},
    )
    fields = (
        ("final_energy_ev", scientific.final_energy_ev, "eV"),
        ("energy_per_atom_ev", scientific.energy_per_atom_ev, "eV/atom"),
        ("electronic_convergence", scientific.electronic_convergence, None),
        ("band_gap_ev", scientific.band_gap_ev, "eV"),
        ("band_kpoints", scientific.band_kpoints, None),
        ("bands", scientific.bands, None),
    )
    for predicate, value, unit in fields:
        if value is None:
            continue
        _append(
            items,
            CATEGORY_DERIVED_OBSERVATION,
            "scientific_result",
            predicate,
            value,
            unit=unit,
            source=_source_with_native_path(source, f"{base_path}.{predicate}"),
        )


def _add_executed_input_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    identity = context.base.identity
    for indicator, observed in identity.executed_input_indicators.items():
        _append(
            items,
            CATEGORY_OBSERVATION,
            "executed_input",
            indicator,
            observed.get("value"),
            status=str(observed.get("status")) if observed.get("status") is not None else None,
            source=EvidenceSourceRef(
                identity.executed_input_evidence_type or EXECUTED_INPUT,
                "executed_input_indicators",
                f"source_context.base.identity.executed_input_indicators.{indicator}",
                {
                    "parameter": observed.get("parameter"),
                    "source_values": dict(observed.get("source_values", {})),
                },
            ),
        )


def _add_trajectory_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    for index, trajectory in enumerate(_job_trajectories(context.base.job)):
        base_path = _trajectory_native_path(context.base.job, index)
        source = EvidenceSourceRef(
            trajectory.evidence_type,
            _trajectory_scope(trajectory),
            base_path,
        )
        _append(
            items,
            CATEGORY_OBSERVATION,
            trajectory.stage_label,
            "completed_ionic_steps",
            trajectory.completed_ionic_steps,
            source=_source_with_native_path(source, f"{base_path}.completed_ionic_steps"),
        )
        _append(
            items,
            CATEGORY_OBSERVATION,
            trajectory.stage_label,
            "converged_electronic",
            trajectory.converged_electronic,
            source=_source_with_native_path(source, f"{base_path}.converged_electronic"),
        )
        _append(
            items,
            CATEGORY_OBSERVATION,
            trajectory.stage_label,
            "converged_ionic",
            trajectory.converged_ionic,
            source=_source_with_native_path(source, f"{base_path}.converged_ionic"),
        )
        _add_criteria_items(items, trajectory, base_path)
        _add_trajectory_progress_items(
            items,
            derive_trajectory_progress_evidence(trajectory),
            base_path,
        )


def _add_criteria_items(
    items: list[ScientificEvidenceItem],
    trajectory: StageTrajectoryObservation,
    base_path: str,
) -> None:
    for key in ("NELM", "EDIFF", "NSW", "EDIFFG", "ISIF"):
        if key not in trajectory.criteria:
            continue
        _append(
            items,
            CATEGORY_OBSERVATION,
            trajectory.stage_label,
            f"criterion.{key}",
            trajectory.criteria.get(key),
            source=EvidenceSourceRef(
                EXECUTED_INPUT,
                _trajectory_scope(trajectory),
                f"{base_path}.criteria.{key}",
                {"source_values": dict(trajectory.criteria_source_values.get(key, {}))},
            ),
        )


def _add_trajectory_progress_items(
    items: list[ScientificEvidenceItem],
    progress: TrajectoryProgressEvidence,
    base_path: str,
) -> None:
    source = EvidenceSourceRef(
        TRAJECTORY_PROGRESS_EVIDENCE,
        _trajectory_scope(progress),
        f"{base_path} -> derive_trajectory_progress_evidence()",
    )
    fields = (
        ("force_source", progress.force_source, None),
        ("force_observation_count", progress.force_observation_count, None),
        ("force_criterion_magnitude_eV_A", progress.force_criterion_magnitude_eV_A, "eV/A"),
        ("initial_max_force_eV_A", progress.initial_max_force_eV_A, "eV/A"),
        ("current_max_force_eV_A", progress.current_max_force_eV_A, "eV/A"),
        ("best_max_force_eV_A", progress.best_max_force_eV_A, "eV/A"),
        ("best_force_step", progress.best_force_step, None),
        ("initial_force_over_abs_EDIFFG", progress.initial_force_over_abs_EDIFFG, None),
        ("current_force_over_abs_EDIFFG", progress.current_force_over_abs_EDIFFG, None),
        ("best_force_over_abs_EDIFFG", progress.best_force_over_abs_EDIFFG, None),
        ("min_electronic_iterations", progress.min_electronic_iterations, None),
        ("median_electronic_iterations", progress.median_electronic_iterations, None),
        ("max_electronic_iterations", progress.max_electronic_iterations, None),
    )
    for predicate, value, unit in fields:
        if value is None:
            continue
        _append(
            items,
            CATEGORY_DERIVED_OBSERVATION,
            progress.stage_label,
            predicate,
            value,
            unit=unit,
            status=(
                progress.atomic_force_status
                if predicate.endswith("force_eV_A")
                or predicate.endswith("EDIFFG")
                or predicate in {"force_source", "force_observation_count", "best_force_step"}
                else progress.electronic_iteration_status
            ),
            source=_source_with_native_path(
                source,
                f"{base_path} -> derive_trajectory_progress_evidence().{predicate}",
            ),
        )
    for index, limitation in enumerate(progress.limitations):
        _append(
            items,
            CATEGORY_LIMITATION,
            progress.stage_label,
            "trajectory_progress_limitation",
            limitation,
            source=EvidenceSourceRef(
                progress.evidence_type,
                _trajectory_scope(progress),
                f"{base_path} -> derive_trajectory_progress_evidence().limitations[{index}]",
            ),
        )


def _add_assessment_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    for index, assessment in enumerate(_job_assessments(context.base.job)):
        base_path = _assessment_native_path(context.base.job, index)
        _append(
            items,
            CATEGORY_ASSESSMENT,
            assessment.stage_label,
            assessment.scope,
            assessment.label,
            status=assessment.sufficiency,
            source=EvidenceSourceRef(
                assessment.evidence_type,
                f"{assessment.stage_label}.{assessment.scope}",
                base_path,
                {
                    "basis": tuple(assessment.basis),
                    "counter_evidence": tuple(assessment.counter_evidence),
                    "features": dict(assessment.features),
                },
            ),
            limitations=tuple(assessment.limitations),
        )
        for limitation_index, limitation in enumerate(assessment.limitations):
            _append(
                items,
                CATEGORY_LIMITATION,
                assessment.stage_label,
                f"{assessment.scope}_assessment_limitation",
                limitation,
                source=EvidenceSourceRef(
                    CONVERGENCE_PROGRESS_ASSESSMENT,
                    f"{assessment.stage_label}.{assessment.scope}",
                    f"{base_path}.limitations[{limitation_index}]",
                ),
            )


def _add_bmdex_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    evidence = context.composition_context
    if evidence is None:
        return

    producer = dict(evidence.producer)
    _append(
        items,
        CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE,
        "BMDex composition_context",
        "status",
        evidence.status,
        status=evidence.status,
        source=_bmdex_source(evidence, "source_context.composition_context.payload.status"),
    )
    for dataset_name, dataset in evidence.datasets.items():
        if not isinstance(dataset, Mapping):
            continue
        records = dataset.get("records")
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            continue
        for record_index, record in enumerate(records):
            if not isinstance(record, Mapping):
                continue
            _add_bmdex_record_items(
                items,
                evidence,
                dataset_name,
                dataset,
                record,
                record_index,
                producer,
            )

    for index, missing in enumerate(evidence.missing_evidence):
        _append(
            items,
            CATEGORY_EVIDENCE_GAP,
            _bmdex_missing_subject(missing),
            "missing_evidence",
            dict(missing),
            status="unavailable",
            source=_bmdex_source(
                evidence,
                f"source_context.composition_context.payload.missing_evidence[{index}]",
                {"missing_evidence": dict(missing)},
            ),
        )

    for index, limitation in enumerate(evidence.limitations):
        _append(
            items,
            CATEGORY_LIMITATION,
            "BMDex composition_context",
            str(limitation.get("code", "limitation")),
            dict(limitation),
            source=_bmdex_source(
                evidence,
                f"source_context.composition_context.payload.limitations[{index}]",
                {"limitation": dict(limitation)},
            ),
        )


def _add_bmdex_record_items(
    items: list[ScientificEvidenceItem],
    evidence: BmdexCompositionEvidence,
    dataset_name: str,
    dataset: Mapping[str, Any],
    record: Mapping[str, Any],
    record_index: int,
    producer: Mapping[str, Any],
) -> None:
    element = str(record.get("element") or "unknown_element")
    status = str(record.get("status")) if record.get("status") is not None else None
    native_path = (
        "source_context.composition_context.payload.datasets."
        f"{dataset_name}.records[{record_index}]"
    )
    provenance = {
        "producer": producer,
        "dataset_name": dataset_name,
        "dataset_id": dataset.get("dataset_id"),
        "path": dataset.get("path"),
        "quantity": dataset.get("quantity") or dataset.get("term"),
        "source_reference_status": dataset.get("source_reference_status"),
    }

    if "abundance" in record:
        _append(
            items,
            CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE,
            element,
            "element_abundance",
            record.get("abundance"),
            unit=record.get("units") or dataset.get("units"),
            status=status,
            source=_bmdex_source(evidence, native_path, provenance),
        )
    if "representative_oxidation_states" in record:
        _append(
            items,
            CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE,
            element,
            "representative_oxidation_states",
            record.get("representative_oxidation_states"),
            status=status,
            source=_bmdex_source(evidence, native_path, provenance),
        )


def _add_context_gap_items(
    items: list[ScientificEvidenceItem],
    context: EnrichedScientificContext,
) -> None:
    for index, gap in enumerate(context.base.evidence_gaps):
        _add_gap_item(items, gap, f"source_context.base.evidence_gaps[{index}]")
    for index, gap in enumerate(context.evidence_gaps):
        _add_gap_item(items, gap, f"source_context.evidence_gaps[{index}]")


def _add_gap_item(
    items: list[ScientificEvidenceItem],
    gap: EvidenceGap,
    native_path: str,
) -> None:
    _append(
        items,
        CATEGORY_EVIDENCE_GAP,
        gap.scope,
        "unavailable",
        gap.reason,
        status="unavailable",
        source=EvidenceSourceRef(
            gap.evidence_type,
            gap.scope,
            native_path,
            {"source": gap.source} if gap.source else {},
        ),
    )


def _append(
    items: list[ScientificEvidenceItem],
    category: str,
    subject: str,
    predicate: str,
    value: Any,
    *,
    unit: str | None = None,
    status: str | None = None,
    source: EvidenceSourceRef | None = None,
    limitations: tuple[str, ...] = (),
) -> None:
    items.append(
        ScientificEvidenceItem(
            category=category,
            subject=subject,
            predicate=predicate,
            value=value,
            unit=unit,
            status=status,
            source=source,
            limitations=limitations,
        )
    )


def _source_with_native_path(
    source: EvidenceSourceRef,
    native_path: str,
) -> EvidenceSourceRef:
    return EvidenceSourceRef(
        source.source_evidence_type,
        source.source_scope,
        native_path,
        source.provenance,
    )


def _bmdex_source(
    evidence: BmdexCompositionEvidence,
    native_path: str,
    provenance: Mapping[str, Any] | None = None,
) -> EvidenceSourceRef:
    merged = {"producer": dict(evidence.producer)}
    if provenance:
        merged.update(provenance)
    return EvidenceSourceRef(
        evidence.evidence_type,
        "BMDex composition_context",
        native_path,
        merged,
    )


def _bmdex_missing_subject(missing: Mapping[str, Any]) -> str:
    dataset = missing.get("dataset")
    element = missing.get("element")
    if dataset and element:
        return f"BMDex {dataset}.{element}"
    if dataset:
        return f"BMDex {dataset}"
    return "BMDex missing_evidence"


def _scientific_result(job: JobInspection) -> ScientificResult | None:
    if job.direct_vasp is not None:
        return job.direct_vasp.scientific
    if job.bmd_compute is not None:
        return job.bmd_compute.inspection.scientific
    return None


def _scientific_result_native_path(job: JobInspection) -> str:
    if job.direct_vasp is not None:
        return "source_context.base.job.direct_vasp.scientific"
    if job.bmd_compute is not None:
        return "source_context.base.job.bmd_compute.inspection.scientific"
    return "source_context.base.job"


def _producer_stages(job: JobInspection) -> tuple[WorkflowStage, ...]:
    if job.bmd_compute is None:
        return ()
    return job.bmd_compute.inspection.workflow_stages


def _stage_value(stage: WorkflowStage) -> Mapping[str, Any]:
    return {
        "index": stage.index,
        "label": stage.label,
        "stage_type": stage.stage_type,
        "theory": stage.theory,
        "modifiers": tuple(stage.modifiers),
        "options": dict(stage.options),
    }


def _job_trajectories(
    job: JobInspection,
) -> tuple[StageTrajectoryObservation, ...]:
    if job.bmd_compute is not None:
        return job.bmd_compute.trajectories
    if job.direct_vasp is not None:
        return (job.direct_vasp.trajectory,)
    return ()


def _job_assessments(
    job: JobInspection,
) -> tuple[ConvergenceProgressAssessment, ...]:
    if job.bmd_compute is not None:
        return job.bmd_compute.assessments
    if job.direct_vasp is not None:
        return job.direct_vasp.assessments
    return ()


def _trajectory_native_path(job: JobInspection, index: int) -> str:
    if job.direct_vasp is not None:
        return "source_context.base.job.direct_vasp.trajectory"
    return f"source_context.base.job.bmd_compute.trajectories[{index}]"


def _assessment_native_path(job: JobInspection, index: int) -> str:
    if job.direct_vasp is not None:
        return f"source_context.base.job.direct_vasp.assessments[{index}]"
    return f"source_context.base.job.bmd_compute.assessments[{index}]"


def _trajectory_scope(
    trajectory: StageTrajectoryObservation | TrajectoryProgressEvidence,
) -> str:
    if trajectory.stage_index is not None:
        return f"stage_{trajectory.stage_index}"
    return trajectory.stage_label
