from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from bmd_agent.resources.bmdex import BmdexDomainContextEnrichment
from bmd_agent.resources.custodian import (
    CustodianInterventionEvidence,
    CustodianPolicyEvidence,
    TerminationEvidenceAssessment,
)
from bmd_agent.resources.lifecycle import LifecycleAnalysis, LifecycleState
from bmd_agent.resources.oom import (
    INSUFFICIENT_OOM_EVIDENCE,
    NO_OOM_EVIDENCE,
    OOM_ESTABLISHED,
    OOM_POSSIBLE,
    MemoryObservation,
    OomDiagnosticEvidence,
)
from bmd_agent.resources.run import (
    JobInspection,
    ScientificResult,
    StageTrajectoryObservation,
    WorkflowStage,
)


@dataclass(frozen=True)
class ConciseSection:
    """One student-facing group of already synthesized statements."""

    title: str
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ConciseStageSummary:
    """Compact workflow-stage state derived from existing typed observations."""

    index: int
    label: str
    status: str


@dataclass(frozen=True)
class ConciseDiagnosticSummary:
    """Pure presentation model over an existing BMD Agent analysis."""

    calculation_label: str
    status: str
    identifier_label: str | None = None
    identifier_value: str | None = None
    stages: tuple[ConciseStageSummary, ...] = ()
    sections: tuple[ConciseSection, ...] = ()
    detailed_evidence_command: str | None = None


def build_job_concise_summary(
    inspection: JobInspection,
    *,
    contextual_enrichment: BmdexDomainContextEnrichment | None = None,
    detailed_evidence_command: str | None = None,
) -> ConciseDiagnosticSummary:
    """Build a student-facing summary without acquiring or reparsing evidence."""

    status = _job_status(inspection)
    stages = () if status == "PENDING" else _job_stage_summaries(inspection, status)
    sections: list[ConciseSection] = []
    unsuccessful: list[ConciseSection] = []

    if status == "PENDING":
        reason = _useful_scheduler_reason(inspection)
        lines = ["The job is waiting in the SLURM queue."]
        if reason:
            lines.append(f"Queue reason: {reason}.")
        sections.append(ConciseSection("What happened", tuple(lines)))
    elif status == "RUNNING":
        sections.append(
            ConciseSection(
                "What happened",
                ("The calculation is currently running.", "No execution error has been established."),
            )
        )
    elif status == "COMPLETED":
        sections.extend(_completed_sections(_job_scientific(inspection)))
        sections.append(
            ConciseSection("Execution", ("No execution problems were detected.",))
        )
    else:
        unsuccessful = _unsuccessful_job_sections(inspection, contextual_enrichment)
        sections.extend(section for section in unsuccessful if section.title != "Assessment")

    if status not in {"PENDING", "COMPLETED"}:
        progress = _progress_lines(
            _job_focus_trajectory(inspection),
            scientific=_job_scientific(inspection),
        )
        if progress:
            sections.append(ConciseSection("Progress", progress))
        if status not in {"RUNNING"}:
            sections.extend(
                section for section in unsuccessful if section.title == "Assessment"
            )

    return ConciseDiagnosticSummary(
        calculation_label=_job_calculation_label(inspection),
        status=status,
        identifier_label="SLURM job",
        identifier_value=inspection.job_id,
        stages=stages,
        sections=tuple(sections),
        detailed_evidence_command=detailed_evidence_command,
    )


def build_lifecycle_concise_summary(
    analysis: LifecycleAnalysis,
    *,
    contextual_enrichment: BmdexDomainContextEnrichment | None = None,
    detailed_evidence_command: str | None = None,
) -> ConciseDiagnosticSummary:
    """Build a concise local/path summary from the existing lifecycle analysis."""

    status = _lifecycle_status(analysis)
    sections: list[ConciseSection] = []
    unsuccessful: list[ConciseSection] = []
    relocation = _relocation_section(analysis)
    if relocation is not None:
        sections.append(relocation)
    if analysis.calculation_kind == "none":
        sections.append(
            ConciseSection(
                "Assessment",
                ("No recognizable BMD Compute or VASP calculation was found.",),
            )
        )
    elif status == "PENDING":
        lines = ["The job is waiting in the SLURM queue."]
        reason = _lifecycle_scheduler_reason(analysis)
        if reason:
            lines.append(f"Queue reason: {reason}.")
        sections.append(ConciseSection("What happened", tuple(lines)))
    elif status == "RUNNING":
        sections.append(
            ConciseSection(
                "What happened",
                ("The calculation is currently running.", "No execution error has been established."),
            )
        )
        progress = _progress_lines(_lifecycle_focus_trajectory(analysis), scientific=analysis.scientific)
        if progress:
            sections.append(ConciseSection("Progress", progress))
    elif status == "COMPLETED":
        sections.extend(_completed_sections(analysis.scientific))
        sections.append(
            ConciseSection("Execution", ("No execution problems were detected.",))
        )
    elif status == "PRE_RUN":
        sections.append(
            ConciseSection(
                "What happened",
                ("The required calculation inputs are present, but execution has not started.",),
            )
        )
    else:
        unsuccessful = _unsuccessful_lifecycle_sections(analysis, contextual_enrichment)
        sections.extend(section for section in unsuccessful if section.title != "Assessment")
        progress = _progress_lines(_lifecycle_focus_trajectory(analysis), scientific=analysis.scientific)
        if progress:
            sections.append(ConciseSection("Progress", progress))
        sections.extend(section for section in unsuccessful if section.title == "Assessment")

    return ConciseDiagnosticSummary(
        calculation_label=_lifecycle_calculation_label(analysis),
        status=status,
        identifier_label="Calculation directory",
        identifier_value=str(analysis.directory),
        stages=(
            ()
            if status == "PENDING"
            else _lifecycle_stage_summaries(analysis)
        ),
        sections=tuple(sections),
        detailed_evidence_command=detailed_evidence_command,
    )


def render_concise_summary(summary: ConciseDiagnosticSummary) -> str:
    """Render a concise summary without inspecting or reinterpreting evidence."""

    lines = ["BMD Agent", "=========", "", summary.calculation_label, f"Status: {summary.status}"]
    if summary.identifier_label and summary.identifier_value:
        lines.append(f"{summary.identifier_label}: {summary.identifier_value}")

    if summary.stages:
        lines.extend(("", "Workflow", "--------"))
        for stage in summary.stages:
            lines.append(f"Stage {stage.index}  {stage.status:<10}  {stage.label}")

    for section in summary.sections:
        if not section.lines:
            continue
        lines.extend(("", section.title, "-" * len(section.title), *section.lines))

    if summary.detailed_evidence_command:
        lines.extend(
            (
                "",
                "Detailed evidence:",
                f"  {summary.detailed_evidence_command}",
            )
        )
    return "\n".join(lines) + "\n"


def _unsuccessful_job_sections(
    inspection: JobInspection,
    enrichment: BmdexDomainContextEnrichment | None,
) -> list[ConciseSection]:
    termination = _job_termination(inspection)
    custodian = _job_custodian_evidence(inspection)
    sections = [
        ConciseSection(
            "What happened",
            _termination_lines(_scheduler_state(inspection), termination, custodian),
        )
    ]

    context_lines = _context_lines(enrichment)
    if context_lines:
        sections.append(ConciseSection("Why this may have happened", context_lines))

    memory_lines = _memory_lines(inspection.oom)
    if memory_lines:
        sections.append(ConciseSection("Memory", memory_lines))

    assessment_lines = _assessment_lines(
        termination,
        policy=_job_custodian_policy(inspection),
        has_frozen_intervention=_has_frozen_intervention(custodian),
    )
    assessment_lines = tuple(
        _ordered_unique((*assessment_lines, *_job_evidence_uncertainties(inspection)))
    )
    if assessment_lines:
        sections.append(ConciseSection("Assessment", assessment_lines))
    return sections


def _unsuccessful_lifecycle_sections(
    analysis: LifecycleAnalysis,
    enrichment: BmdexDomainContextEnrichment | None,
) -> list[ConciseSection]:
    diagnostics = analysis.diagnostics
    termination = diagnostics.termination if diagnostics is not None else None
    custodian = (
        (diagnostics.custodian,)
        if diagnostics is not None and diagnostics.custodian is not None
        else ()
    )
    lines = _lifecycle_termination_lines(analysis, termination, custodian)
    sections = [ConciseSection("What happened", lines)]

    context_lines = _context_lines(enrichment)
    if context_lines:
        sections.append(ConciseSection("Why this may have happened", context_lines))

    memory_lines = _memory_lines(diagnostics.oom if diagnostics is not None else None)
    if memory_lines:
        sections.append(ConciseSection("Memory", memory_lines))

    policy = analysis.bmd_workflow.custodian_policy if analysis.bmd_workflow else None
    assessment = _assessment_lines(
        termination,
        policy=policy,
        has_frozen_intervention=_has_frozen_intervention(custodian),
    )
    assessment = tuple(
        _ordered_unique((*assessment, *_lifecycle_evidence_uncertainties(analysis)))
    )
    if not assessment and analysis.state == LifecycleState.UNKNOWN:
        assessment = ("Agent could not determine the calculation state from the available evidence.",)
    if assessment:
        sections.append(ConciseSection("Assessment", assessment))
    return sections


def _termination_lines(
    scheduler_state: str,
    termination: TerminationEvidenceAssessment | None,
    evidence: Sequence[CustodianInterventionEvidence],
) -> tuple[str, ...]:
    if _supported_classification(termination, "custodian_triggered_process_termination"):
        frozen = _first_frozen_intervention(evidence)
        if frozen is not None:
            timeout = _duration(frozen.timeout_seconds)
            first = f"Automatic frozen-job protection intervened {frozen.count} times"
            if timeout:
                first += f" after {timeout} passed without new VASP output"
            lines = [first + ".", "The run stopped after the final intervention."]
            correction = _plain_correction(frozen.action_summaries)
            if correction:
                lines.append(correction)
            return tuple(lines)
        return ("Automatic calculation protection stopped the VASP process after repeated interventions.",)
    if _supported_classification(termination, "slurm_timeout"):
        return ("SLURM stopped the job when its time limit was reached.",)
    if _supported_classification(termination, "slurm_out_of_memory"):
        return ("SLURM reported that the job stopped because it ran out of memory.",)
    if _supported_classification(termination, "vasp_nonzero_exit_observed"):
        return ("VASP exited with a nonzero status before successful completion.",)

    if scheduler_state:
        return (f"SLURM recorded the job as {scheduler_state}.",)
    return ("Agent could not establish whether the calculation completed.",)


def _lifecycle_termination_lines(
    analysis: LifecycleAnalysis,
    termination: TerminationEvidenceAssessment | None,
    evidence: Sequence[CustodianInterventionEvidence],
) -> tuple[str, ...]:
    scheduler_state = (
        str(analysis.scheduler.state or "").upper().split()[0]
        if analysis.scheduler is not None
        else ""
    )
    lines = _termination_lines(scheduler_state, termination, evidence)
    if not scheduler_state and analysis.state == LifecycleState.UNKNOWN:
        return ("The available files show partial execution, but do not establish whether the calculation is still active or how it stopped.",)
    return lines


def _assessment_lines(
    termination: TerminationEvidenceAssessment | None,
    *,
    policy: CustodianPolicyEvidence | None,
    has_frozen_intervention: bool,
) -> tuple[str, ...]:
    lines: list[str] = []
    if termination is not None and termination.status == "supported":
        labels = {
            "custodian_triggered_process_termination": (
                "The evidence supports automatic frozen-job protection as the immediate cause of termination."
            ),
            "slurm_timeout": "The evidence supports the scheduler time limit as the immediate cause of termination.",
            "slurm_out_of_memory": "The evidence supports memory exhaustion as the immediate cause of termination.",
            "vasp_nonzero_exit_observed": "The evidence establishes a nonzero VASP exit, but not its underlying cause.",
        }
        label = labels.get(termination.classification)
        if label:
            lines.append(label)
    elif termination is None or termination.status != "supported":
        lines.append("Agent could not determine the cause from the available evidence.")

    if termination is not None and any("eventually converge" in item for item in termination.limitations):
        lines.append("This does not establish whether the calculation would eventually have converged.")
    if has_frozen_intervention and policy is not None and not policy.available:
        lines.append(
            "This historical submission does not record the execution policy used, so Agent cannot compare it with the current policy."
        )
    return tuple(_ordered_unique(lines))


def _context_lines(
    enrichment: BmdexDomainContextEnrichment | None,
) -> tuple[str, ...]:
    if enrichment is None or enrichment.evidence is None or not enrichment.evidence.records:
        return ()
    lines = [record.contextual_statement for record in enrichment.evidence.records]
    lines.append(
        "This is contextual scientific evidence; it may help interpret the run but does not establish its termination cause."
    )
    return tuple(_ordered_unique(lines))


def _job_evidence_uncertainties(inspection: JobInspection) -> tuple[str, ...]:
    trajectories = _job_trajectories(inspection)
    scientific = _job_scientific(inspection)
    lines: list[str] = []
    if any(item.vasprun_error or item.vasprun_skipped_reason for item in trajectories):
        lines.append(
            "Some result and trajectory evidence is unavailable because vasprun.xml could not be parsed completely."
        )
    elif scientific is not None and scientific.error:
        lines.append("Final scientific results could not be reconstructed from the available artifacts.")
    return tuple(lines)


def _lifecycle_evidence_uncertainties(
    analysis: LifecycleAnalysis,
) -> tuple[str, ...]:
    diagnostics = analysis.diagnostics
    if diagnostics is not None and any(
        item.vasprun_error or item.vasprun_skipped_reason
        for item in diagnostics.trajectories
    ):
        return (
            "Some result and trajectory evidence is unavailable because vasprun.xml could not be parsed completely.",
        )
    if analysis.scientific is not None and analysis.scientific.error:
        return ("Final scientific results could not be reconstructed from the available artifacts.",)
    return ()


def _memory_lines(evidence: OomDiagnosticEvidence | None) -> tuple[str, ...]:
    if evidence is None:
        return ()
    if evidence.assessment == OOM_ESTABLISHED:
        lines = ["Evidence indicates an out-of-memory failure."]
    elif evidence.assessment == OOM_POSSIBLE:
        lines = ["Memory pressure is possible, but the available evidence does not establish an out-of-memory failure."]
    elif evidence.assessment == NO_OOM_EVIDENCE:
        lines = ["No evidence of an out-of-memory failure was found."]
    elif evidence.assessment == INSUFFICIENT_OOM_EVIDENCE:
        lines = ["Available evidence was insufficient to assess memory exhaustion."]
    else:
        return ()

    maximum = _memory_amount(evidence.maximum_rss)
    allocation = _memory_amount(evidence.allocated_memory or evidence.requested_memory)
    if maximum and allocation:
        lines.append(f"Peak observed use was about {maximum} of {allocation} allocated.")
    elif maximum:
        lines.append(f"Peak observed use was about {maximum}; a compatible allocation was unavailable.")
    return tuple(lines)


def _completed_sections(scientific: ScientificResult | None) -> list[ConciseSection]:
    if scientific is None:
        return []
    lines: list[str] = []
    structure = scientific.structure
    formula = scientific.final_formula
    if formula is None and structure is not None:
        formula = structure.reduced_formula or structure.formula
    if formula:
        lines.append(str(formula))
    if structure is not None and structure.site_count is not None:
        lines.append(f"{structure.site_count} atoms")
    if scientific.electronic_convergence is True:
        lines.append("Electronic convergence: reached")
    elif scientific.electronic_convergence is False:
        lines.append("Electronic convergence: not reached")
    if scientific.final_energy_ev is not None:
        lines.append(f"Final energy: {scientific.final_energy_ev:.6f} eV")
    if scientific.band_gap_ev is not None:
        lines.append(f"Band gap: {scientific.band_gap_ev:.3f} eV")
    return [ConciseSection("Results", tuple(lines))] if lines else []


def _progress_lines(
    trajectory: StageTrajectoryObservation | None,
    *,
    scientific: ScientificResult | None,
) -> tuple[str, ...]:
    if trajectory is None:
        if scientific is not None and scientific.electronic_convergence is False:
            return ("Electronic convergence was not reached.",)
        return ()

    lines: list[str] = []
    if trajectory.completed_ionic_steps is not None and trajectory.completed_ionic_steps > 0:
        lines.append(f"{trajectory.completed_ionic_steps} ionic steps were completed.")
    if trajectory.incomplete_electronic_iteration_count is not None:
        count = trajectory.incomplete_electronic_iteration_count
        noun = "iteration" if count == 1 else "iterations"
        lines.append(f"An incomplete electronic cycle contains {count} observed {noun}.")

    electronic = trajectory.converged_electronic
    if electronic is None and scientific is not None:
        electronic = scientific.electronic_convergence
    if electronic is True:
        lines.append("Electronic convergence was reached.")
    elif electronic is False:
        lines.append("Electronic convergence was not reached.")

    if _is_relaxation_trajectory(trajectory):
        if trajectory.converged_ionic is True:
            lines.append("Ionic convergence was reached.")
        elif trajectory.converged_ionic is False:
            lines.append("Ionic convergence was not reached.")
        elif trajectory.completed_ionic_steps:
            lines.append("Ionic convergence could not be established from the available evidence.")

        current_force = next(
            (step.max_force for step in reversed(trajectory.ionic_steps) if step.max_force is not None),
            None,
        )
        criterion = _negative_force_criterion(trajectory.criteria.get("EDIFFG"))
        if current_force is not None and criterion is not None:
            lines.append(
                f"Most recent maximum atomic force: {current_force:.4f} eV/A; convergence criterion: {criterion:.4f} eV/A."
            )
    return tuple(_ordered_unique(lines))


def _job_stage_summaries(
    inspection: JobInspection,
    job_status: str,
) -> tuple[ConciseStageSummary, ...]:
    diagnosis = inspection.bmd_compute
    if diagnosis is None or len(diagnosis.inspection.workflow_stages) < 2:
        return ()
    trajectories = {item.stage_index: item for item in diagnosis.trajectories}
    focus = _job_focus_trajectory(inspection)
    summaries: list[ConciseStageSummary] = []
    for stage in diagnosis.inspection.workflow_stages:
        trajectory = trajectories.get(stage.index)
        status = _stage_status(stage, trajectory)
        if trajectory is focus and status != "COMPLETED" and job_status in {"FAILED", "RUNNING"}:
            status = job_status
        summaries.append(ConciseStageSummary(stage.index, _stage_label(stage), status))
    return tuple(summaries)


def _lifecycle_stage_summaries(
    analysis: LifecycleAnalysis,
) -> tuple[ConciseStageSummary, ...]:
    workflow = analysis.bmd_workflow
    if workflow is None or len(workflow.workflow_stages) < 2:
        return ()
    evidence_by_index = {item.stage_index: item for item in workflow.stage_evidence}
    summaries: list[ConciseStageSummary] = []
    for index, raw in enumerate(workflow.workflow_stages, start=1):
        stage_index = _int_value(raw.get("index")) or index
        evidence = evidence_by_index.get(stage_index)
        if evidence is None:
            status = "UNKNOWN"
        elif evidence.normal_completion:
            status = "COMPLETED"
        elif evidence.has_meaningful_execution:
            status = "INCOMPLETE"
        elif evidence.has_required_inputs:
            status = "NOT STARTED"
        else:
            status = "UNKNOWN"
        summaries.append(
            ConciseStageSummary(stage_index, _mapping_stage_label(raw), status)
        )
    return tuple(summaries)


def _relocation_section(analysis: LifecycleAnalysis) -> ConciseSection | None:
    workflow = analysis.bmd_workflow
    if workflow is None or not workflow.relocated:
        return None
    lines = [f"Current acquisition directory: {workflow.workflow_root}"]
    if workflow.producer_root:
        lines.append(f"Original BMD Compute run directory: {workflow.producer_root}")
    return ConciseSection("Location", tuple(lines))


def _stage_status(
    stage: WorkflowStage,
    trajectory: StageTrajectoryObservation | None,
) -> str:
    if trajectory is None or not _trajectory_has_observations(trajectory):
        return "NOT STARTED"
    if _is_relaxation_stage(stage.stage_type):
        return "COMPLETED" if trajectory.converged_ionic is True else "INCOMPLETE"
    return "COMPLETED" if trajectory.converged_electronic is True else "INCOMPLETE"


def _job_calculation_label(inspection: JobInspection) -> str:
    if inspection.bmd_compute is not None:
        stages = inspection.bmd_compute.inspection.workflow_stages
        if stages:
            return " -> ".join(_stage_label(stage) for stage in stages)
    if inspection.direct_vasp is not None:
        return "Direct VASP calculation"
    return "Calculation"


def _lifecycle_calculation_label(analysis: LifecycleAnalysis) -> str:
    workflow = analysis.bmd_workflow
    if workflow is not None and workflow.workflow_stages:
        return " -> ".join(_mapping_stage_label(stage) for stage in workflow.workflow_stages)
    if analysis.calculation_kind == "direct VASP":
        return "Direct VASP calculation"
    if analysis.calculation_kind == "BMD Compute":
        return "BMD Compute calculation"
    return "Calculation directory"


def _stage_label(stage: WorkflowStage) -> str:
    return _calculation_label(stage.theory, stage.stage_type, stage.modifiers)


def _mapping_stage_label(stage: Mapping[str, Any]) -> str:
    modifiers = stage.get("modifiers")
    return _calculation_label(
        stage.get("theory"),
        stage.get("stage_type"),
        modifiers if isinstance(modifiers, list) else (),
    )


def _calculation_label(theory: Any, stage_type: Any, modifiers: Iterable[Any]) -> str:
    theory_label = str(theory).upper() if theory else "Unknown"
    stage_labels = {
        "band_structure": "Band Structure",
        "dos": "DOS",
        "relax": "Geometry Optimisation",
        "static": "Static Energy",
    }
    raw_stage = str(stage_type or "calculation")
    label = f"{theory_label} {stage_labels.get(raw_stage, raw_stage.replace('_', ' ').title())}"
    modifier_labels = [_modifier_label(item) for item in modifiers]
    if modifier_labels:
        label += " + " + " + ".join(modifier_labels)
    return label


def _modifier_label(value: Any) -> str:
    text = str(value)
    labels = {"soc": "SOC", "dft_u": "DFT+U"}
    return labels.get(text.lower(), text.replace("_", " ").upper())


def _job_status(inspection: JobInspection) -> str:
    state = _scheduler_state(inspection)
    if state in {"PENDING", "CONFIGURING"}:
        return "PENDING"
    if state in {"RUNNING", "COMPLETING", "RESIZING", "SUSPENDED", "STAGE_OUT"}:
        return "RUNNING"
    if state == "COMPLETED" and (
        inspection.scheduler is None or inspection.scheduler.exit_code in {None, "0:0"}
    ):
        return "COMPLETED"
    if state:
        return "FAILED"
    return "UNKNOWN"


def _lifecycle_status(analysis: LifecycleAnalysis) -> str:
    scheduler_state = str(analysis.scheduler.state or "").upper().split()[0] if analysis.scheduler else ""
    if scheduler_state in {"PENDING", "CONFIGURING"}:
        return "PENDING"
    if scheduler_state in {"RUNNING", "COMPLETING", "RESIZING", "SUSPENDED", "STAGE_OUT"}:
        return "RUNNING"
    if analysis.state == LifecycleState.INCOMPLETE:
        return "FAILED"
    return analysis.state.value


def _job_focus_trajectory(inspection: JobInspection) -> StageTrajectoryObservation | None:
    trajectories = _job_trajectories(inspection)
    observed = [item for item in trajectories if _trajectory_has_observations(item)]
    if observed:
        return observed[-1]
    return trajectories[-1] if trajectories else None


def _lifecycle_focus_trajectory(analysis: LifecycleAnalysis) -> StageTrajectoryObservation | None:
    diagnostics = analysis.diagnostics
    if diagnostics is None or not diagnostics.trajectories:
        return None
    observed = [item for item in diagnostics.trajectories if _trajectory_has_observations(item)]
    return observed[-1] if observed else diagnostics.trajectories[-1]


def _trajectory_has_observations(trajectory: StageTrajectoryObservation) -> bool:
    return bool(
        trajectory.oszicar_present
        or trajectory.vasprun_present
        or trajectory.outcar_present
        or trajectory.completed_ionic_steps is not None
        or trajectory.incomplete_electronic_iteration_count is not None
        or trajectory.electronic_cycles
        or trajectory.ionic_steps
    )


def _job_trajectories(inspection: JobInspection) -> tuple[StageTrajectoryObservation, ...]:
    if inspection.bmd_compute is not None:
        return inspection.bmd_compute.trajectories
    if inspection.direct_vasp is not None:
        return (inspection.direct_vasp.trajectory,)
    return ()


def _job_scientific(inspection: JobInspection) -> ScientificResult | None:
    if inspection.bmd_compute is not None:
        return inspection.bmd_compute.inspection.scientific
    if inspection.direct_vasp is not None:
        return inspection.direct_vasp.scientific
    return None


def _job_termination(inspection: JobInspection) -> TerminationEvidenceAssessment | None:
    if inspection.bmd_compute is not None:
        return inspection.bmd_compute.termination.assessment
    if inspection.direct_vasp is not None:
        return inspection.direct_vasp.termination_assessment
    return None


def _job_custodian_evidence(
    inspection: JobInspection,
) -> tuple[CustodianInterventionEvidence, ...]:
    if inspection.bmd_compute is not None:
        return inspection.bmd_compute.inspection.custodian_evidence
    if inspection.direct_vasp is not None:
        return inspection.direct_vasp.custodian_evidence
    return ()


def _job_custodian_policy(inspection: JobInspection) -> CustodianPolicyEvidence | None:
    if inspection.bmd_compute is None:
        return None
    return inspection.bmd_compute.inspection.custodian_policy


def _scheduler_state(inspection: JobInspection) -> str:
    if inspection.scheduler is None:
        return ""
    return str(inspection.scheduler.state or "").upper().split()[0]


def _useful_scheduler_reason(inspection: JobInspection) -> str | None:
    reason = inspection.scheduler.reason if inspection.scheduler is not None else None
    return _useful_reason(reason)


def _lifecycle_scheduler_reason(analysis: LifecycleAnalysis) -> str | None:
    reason = analysis.scheduler.reason if analysis.scheduler is not None else None
    return _useful_reason(reason)


def _useful_reason(reason: Any) -> str | None:
    text = str(reason or "").strip()
    if not text or text.lower() in {"none", "n/a", "unknown"}:
        return None
    return text.rstrip(".")


def _supported_classification(
    assessment: TerminationEvidenceAssessment | None,
    classification: str,
) -> bool:
    return bool(
        assessment is not None
        and assessment.status == "supported"
        and assessment.classification == classification
    )


def _first_frozen_intervention(
    evidence: Sequence[CustodianInterventionEvidence],
):
    return next(
        (
            item
            for source in evidence
            for item in source.repeated_interventions
            if item.handler.rsplit(".", 1)[-1] == "FrozenJobErrorHandler"
        ),
        None,
    )


def _has_frozen_intervention(evidence: Sequence[CustodianInterventionEvidence]) -> bool:
    return _first_frozen_intervention(evidence) is not None


def _plain_correction(summaries: Sequence[str]) -> str | None:
    if not summaries:
        return None
    if len(summaries) != 1:
        return "Automatic corrections were applied during the run."
    summary = summaries[0]
    if " -> " not in summary:
        return "An automatic correction was applied during the run."
    target, value = summary.split(" -> ", 1)
    if target.startswith("INCAR."):
        target = target.removeprefix("INCAR.")
    return f"An automatic correction changed {target} to {value}."


def _duration(seconds: Any) -> str | None:
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds <= 0:
        return None
    if seconds % 3600 == 0:
        hours = int(seconds // 3600)
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    if seconds % 60 == 0:
        minutes = int(seconds // 60)
        return f"{minutes} minutes"
    return f"{seconds:g} seconds"


def _memory_amount(observation: MemoryObservation | None) -> str | None:
    if observation is None:
        return None
    if observation.bytes_value is None:
        return observation.raw_value or None
    gib = observation.bytes_value / (1024**3)
    if gib >= 10:
        return f"{gib:.0f} GB"
    if gib >= 1:
        return f"{gib:.1f} GB"
    mib = observation.bytes_value / (1024**2)
    return f"{mib:.0f} MB"


def _is_relaxation_stage(stage_type: str | None) -> bool:
    text = str(stage_type or "").lower()
    return any(item in text for item in ("relax", "optimisation", "optimization"))


def _is_relaxation_trajectory(trajectory: StageTrajectoryObservation) -> bool:
    return _is_relaxation_stage(trajectory.stage_type) or _int_value(
        trajectory.criteria.get("NSW")
    ) not in {None, 0}


def _negative_force_criterion(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return abs(number) if number < 0 else None


def _int_value(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
