from bmd_agent import cli
import bmd_agent.resources.run as run_resource
from bmd_agent.resources.context import (
    EvidenceGap,
    ScientificContext,
    build_scientific_context,
)
from bmd_agent.resources.run import (
    AGENT_COMPARISON,
    ARTIFACT_OBSERVATION,
    CONVERGENCE_PROGRESS_ASSESSMENT,
    EXECUTED_INPUT,
    PRODUCER_PROVENANCE,
    PRODUCER_REQUESTED,
    PYMATGEN_DERIVED,
    TRAJECTORY_OBSERVATION,
    TRAJECTORY_PROGRESS_EVIDENCE,
    AttemptStateObservation,
    ComparisonObservation,
    DirectVaspInspection,
    IncarObservation,
    InitialStructureObservation,
    IonicStepObservation,
    JobInspection,
    LogRuntimeObservation,
    PathObservation,
    RunDiagnosis,
    RunInspection,
    ScientificResult,
    StageTrajectoryObservation,
    StructureObservation,
    TerminationObservation,
    WorkflowStage,
    assess_convergence_progress,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord


DIRECT_DIR = "/bmd-db/guest/flows/direct-vasp"
FLOW_ROOT = "/bmd-db/guest/flows/bmd-compute"


def scheduler(
    *,
    job_id: str = "21153721",
    state: str = "TIMEOUT",
    work_dir: str | None = DIRECT_DIR,
) -> SlurmAccountingRecord:
    return SlurmAccountingRecord(
        job_id=job_id,
        name="vasp",
        state=state,
        elapsed="01:00:20",
        start="2026-08-30T00:00:00",
        end="2026-08-30T01:00:20",
        partition="leeburton-pool",
        exit_code="0:0",
        timelimit="01:00:00",
        node_list="compute-0-269",
        allocated_cpus=24,
        work_dir=work_dir,
    )


def direct_scientific(
    *,
    formula: str | None = "MnCu5",
    site_count: int | None = 24,
    unavailable: tuple[str, ...] = (),
) -> ScientificResult:
    structure = None
    if formula is not None or site_count is not None:
        structure = StructureObservation(
            source_path=f"{DIRECT_DIR}/CONTCAR",
            reduced_formula=formula,
            site_count=site_count,
        )
    return ScientificResult(
        source_paths=(f"{DIRECT_DIR}/CONTCAR", f"{DIRECT_DIR}/vasprun.xml"),
        final_formula=formula,
        structure=structure,
        unavailable=unavailable,
    )


def direct_trajectory(
    *,
    completed_steps: int | None = 50,
    vasprun_error: str | None = None,
    oszicar_present: bool = True,
    outcar_present: bool = True,
    outcar_error: str | None = None,
    unavailable: tuple[str, ...] = (),
) -> StageTrajectoryObservation:
    ionic_steps = (
        IonicStepObservation(
            step_index=1,
            electronic_iterations=8,
            max_force=0.2,
            max_force_source="OUTCAR",
        ),
        IonicStepObservation(
            step_index=completed_steps or 1,
            electronic_iterations=7,
            max_force=0.071891,
            max_force_source="OUTCAR",
        ),
    ) if completed_steps else ()
    return StageTrajectoryObservation(
        stage_index=1,
        stage_label="work_dir",
        stage_type="direct_vasp",
        theory="unknown",
        directory=DIRECT_DIR,
        oszicar_path=f"{DIRECT_DIR}/OSZICAR",
        oszicar_present=oszicar_present,
        vasprun_path=f"{DIRECT_DIR}/vasprun.xml",
        vasprun_present=vasprun_error is None,
        vasprun_error=vasprun_error,
        outcar_path=f"{DIRECT_DIR}/OUTCAR",
        outcar_present=outcar_present,
        outcar_error=outcar_error,
        outcar_force_alignment_status="aligned" if outcar_error is None else "unavailable",
        criteria={"EDIFF": 1e-6, "EDIFFG": -0.01, "ISIF": 3},
        completed_ionic_steps=completed_steps,
        electronic_iterations_by_completed_ionic_step=(8, 7) if completed_steps else (),
        ionic_steps=ionic_steps,
        recent_ionic_steps=ionic_steps[-2:],
        converged_electronic=True if completed_steps else None,
        converged_ionic=False if completed_steps else None,
        incomplete_electronic_iteration_count=None if completed_steps else 25,
        unavailable=unavailable,
    )


def direct_job(
    *,
    state: str = "TIMEOUT",
    scientific: ScientificResult | None = None,
    trajectory: StageTrajectoryObservation | None = None,
    executed_values=None,
) -> JobInspection:
    trajectory = trajectory or direct_trajectory()
    executed_inputs = (
        IncarObservation(
            "work_dir",
            f"{DIRECT_DIR}/INCAR",
            True,
            1,
            values=executed_values
            or {
                "ISPIN": 2,
                "LDAU": True,
                "LSORBIT": True,
                "LHFCALC": False,
                "IVDW": 12,
            },
        ),
    )
    direct = DirectVaspInspection(
        directory=DIRECT_DIR,
        artifacts=(
            PathObservation("incar", f"{DIRECT_DIR}/INCAR", "file", True, ARTIFACT_OBSERVATION),
            PathObservation("oszicar", f"{DIRECT_DIR}/OSZICAR", "file", trajectory.oszicar_present, ARTIFACT_OBSERVATION),
            PathObservation("outcar", f"{DIRECT_DIR}/OUTCAR", "file", trajectory.outcar_present, ARTIFACT_OBSERVATION),
        ),
        executed_inputs=executed_inputs,
        scientific=scientific or direct_scientific(),
        trajectory=trajectory,
        assessments=assess_convergence_progress((trajectory,)),
    )
    return JobInspection(
        job_id="21153721",
        scheduler=scheduler(state=state),
        scheduler_error=None,
        scheduler_work_dir=DIRECT_DIR,
        calculation_directory=DIRECT_DIR,
        calculation_type="direct VASP",
        calculation_reason=None,
        direct_vasp=direct,
    )


def bmd_compute_job() -> JobInspection:
    stages = (
        WorkflowStage(1, "relax", "pbe", (), None),
        WorkflowStage(2, "static", "hse06", ("soc",), "soc_static"),
    )
    scientific = ScientificResult(
        source_paths=(),
        unavailable=("scientific artifact parsing skipped by diagnose-run v1",),
    )
    run = RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=stages,
        stage_directories=(),
        result_directory=PathObservation("result_dir", f"{FLOW_ROOT}/stage_02", "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={"git_commit": "0396e5eabcdef", "state": "clean", "dirty": False},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(None, False, error="attempt state unavailable"),
        job_id="21153722",
        scheduler=scheduler(job_id="21153722", state="COMPLETED", work_dir=FLOW_ROOT),
        scheduler_error=None,
        runtime=LogRuntimeObservation(PRODUCER_PROVENANCE, (), None, {}, {}, {}),
        scientific=scientific,
        comparison=ComparisonObservation(
            status="unavailable",
            evidence_type=PRODUCER_PROVENANCE,
            reason="No durable BMD Compute result payload was found.",
        ),
        initial_structure=InitialStructureObservation(
            status="unavailable",
            reason="Submitted structure provenance was not inspected.",
        ),
        executed_inputs=(
            IncarObservation("stage_02", f"{FLOW_ROOT}/stage_02/INCAR", True, 2, values={"LSORBIT": True}),
        ),
    )
    trajectory = StageTrajectoryObservation(
        stage_index=2,
        stage_label="stage_02",
        stage_type="static",
        theory="hse06",
        directory=f"{FLOW_ROOT}/stage_02",
        oszicar_present=False,
        vasprun_present=False,
        outcar_present=False,
        unavailable=("trajectory evidence unavailable in fixture",),
    )
    return JobInspection(
        job_id="21153722",
        scheduler=scheduler(job_id="21153722", state="COMPLETED", work_dir=FLOW_ROOT),
        scheduler_error=None,
        scheduler_work_dir=FLOW_ROOT,
        calculation_directory=FLOW_ROOT,
        calculation_type="BMD Compute",
        calculation_reason=None,
        bmd_compute=RunDiagnosis(
            inspection=run,
            termination=TerminationObservation(scheduler_state="COMPLETED"),
            trajectories=(trajectory,),
            assessments=assess_convergence_progress((trajectory,)),
        ),
    )


def test_direct_vasp_timeout_context_keeps_native_job_and_identity() -> None:
    inspection = direct_job()

    context = build_scientific_context(inspection)

    assert isinstance(context, ScientificContext)
    assert context.job is inspection
    assert context.identity.job_id == "21153721"
    assert context.identity.calculation_type == "direct VASP"
    assert context.identity.formula == "MnCu5"
    assert context.identity.site_count == 24
    assert context.identity.structure_evidence_type == PYMATGEN_DERIVED


def test_direct_vasp_does_not_infer_producer_workflow_or_theory() -> None:
    context = build_scientific_context(direct_job())

    assert context.identity.producer_workflow is None
    assert context.identity.producer_stage_types == ()
    assert context.identity.producer_theories == ()
    assert context.identity.producer_evidence_type is None


def test_executed_input_indicators_are_factual_and_source_safe() -> None:
    context = build_scientific_context(direct_job())

    indicators = context.identity.executed_input_indicators
    assert context.identity.executed_input_evidence_type == EXECUTED_INPUT
    assert indicators["spin_polarized"]["value"] is True
    assert indicators["spin_polarized"]["parameter"] == "ISPIN"
    assert indicators["dft_u_enabled"]["value"] is True
    assert indicators["soc_enabled"]["value"] is True
    assert indicators["hybrid_enabled"]["value"] is False
    assert indicators["dispersion_indicator"]["value"] == 12
    assert indicators["soc_enabled"]["evidence_type"] == EXECUTED_INPUT
    assert indicators["soc_enabled"]["source_values"] == {"retained_incar:work_dir": True}


def test_direct_vasp_completed_context_preserves_completed_scientific_evidence() -> None:
    context = build_scientific_context(direct_job(state="COMPLETED"))

    assert context.identity.formula == "MnCu5"
    assert context.identity.site_count == 24
    assert not [
        gap for gap in context.evidence_gaps
        if gap.scope == "scientific_result" and "skipped" in gap.reason
    ]


def test_formula_and_site_count_only_appear_when_supported() -> None:
    context = build_scientific_context(
        direct_job(scientific=direct_scientific(formula=None, site_count=None))
    )

    assert context.identity.formula is None
    assert context.identity.site_count is None
    assert context.identity.structure_evidence_type is None
    assert EvidenceGap(PYMATGEN_DERIVED, "structure", "structure-derived identity unavailable") in context.evidence_gaps


def test_direct_vasp_unavailable_producer_provenance_becomes_gap() -> None:
    context = build_scientific_context(direct_job())

    assert EvidenceGap(
        PRODUCER_PROVENANCE,
        "producer",
        "no BMD Compute producer record found",
    ) in context.evidence_gaps


def test_malformed_vasprun_gap_survives_context() -> None:
    trajectory = direct_trajectory(vasprun_error="file could not be parsed completely")

    context = build_scientific_context(direct_job(trajectory=trajectory))

    assert EvidenceGap(
        TRAJECTORY_OBSERVATION,
        "stage_1.vasprun",
        "file could not be parsed completely",
        f"{DIRECT_DIR}/vasprun.xml",
    ) in context.evidence_gaps


def test_missing_oszicar_and_outcar_evidence_become_gaps() -> None:
    trajectory = direct_trajectory(oszicar_present=False, outcar_present=False)

    context = build_scientific_context(direct_job(trajectory=trajectory))

    assert EvidenceGap(
        TRAJECTORY_OBSERVATION,
        "stage_1.OSZICAR",
        "OSZICAR trajectory unavailable",
        f"{DIRECT_DIR}/OSZICAR",
    ) in context.evidence_gaps
    assert EvidenceGap(
        TRAJECTORY_OBSERVATION,
        "stage_1.OUTCAR",
        "OUTCAR force evidence unavailable",
        f"{DIRECT_DIR}/OUTCAR",
    ) in context.evidence_gaps


def test_bmd_compute_context_preserves_producer_workflow_and_theory() -> None:
    context = build_scientific_context(bmd_compute_job())

    assert context.identity.calculation_type == "BMD Compute"
    assert context.identity.producer_workflow == "PBE -> HSE06 + soc"
    assert context.identity.producer_stage_types == ("relax", "static")
    assert context.identity.producer_theories == ("pbe", "hse06")
    assert context.identity.producer_modifiers == ("soc",)
    assert context.identity.producer_evidence_type == PRODUCER_REQUESTED
    assert context.identity.executed_input_indicators["soc_enabled"]["value"] is True


def test_bmd_compute_skipped_scientific_parsing_is_represented_as_gap() -> None:
    context = build_scientific_context(bmd_compute_job())

    assert EvidenceGap(
        PYMATGEN_DERIVED,
        "scientific_result",
        "scientific artifact parsing skipped by diagnose-run v1",
    ) in context.evidence_gaps


def test_sparse_unknown_job_context_records_scheduler_and_identity_gaps() -> None:
    inspection = JobInspection(
        job_id="1",
        scheduler=None,
        scheduler_error="scheduler accounting was unavailable",
        scheduler_work_dir=None,
        calculation_directory=None,
        calculation_type="unknown",
        calculation_reason="scheduler accounting was unavailable",
    )

    context = build_scientific_context(inspection)

    assert context.identity.calculation_type == "unknown"
    assert EvidenceGap(
        run_resource.SCHEDULER_OBSERVATION,
        "scheduler",
        "scheduler accounting was unavailable",
    ) in context.evidence_gaps
    assert EvidenceGap(
        ARTIFACT_OBSERVATION,
        "calculation_identity",
        "scheduler accounting was unavailable",
    ) in context.evidence_gaps


def test_trajectory_progress_limitations_contribute_gaps() -> None:
    context = build_scientific_context(direct_job())

    assert EvidenceGap(
        TRAJECTORY_PROGRESS_EVIDENCE,
        "stage_1",
        "atomic-force evidence only; variable-cell convergence also requires broader cell/stress evidence",
    ) in context.evidence_gaps


def test_assessment_limitations_contribute_gaps_without_changing_assessment() -> None:
    trajectory = direct_trajectory(completed_steps=0)
    inspection = direct_job(trajectory=trajectory)
    expected = inspection.direct_vasp.assessments

    context = build_scientific_context(inspection)

    assert context.job.direct_vasp.assessments == expected
    assert any(
        gap.evidence_type == CONVERGENCE_PROGRESS_ASSESSMENT
        and "incomplete first" in gap.reason
        for gap in context.evidence_gaps
    )


def test_duplicate_representations_are_deduplicated_conservatively() -> None:
    scientific = direct_scientific(
        unavailable=(
            "vasprun.xml could not be parsed completely",
            "vasprun.xml could not be parsed completely",
        )
    )

    context = build_scientific_context(direct_job(scientific=scientific))

    duplicates = [
        gap for gap in context.evidence_gaps
        if gap == EvidenceGap(
            PYMATGEN_DERIVED,
            "scientific_result",
            "vasprun.xml could not be parsed completely",
        )
    ]
    assert len(duplicates) == 1


def test_builder_causes_no_remote_reads_or_subprocess_calls(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise AssertionError("ScientificContext builder must not acquire evidence")

    monkeypatch.setattr(run_resource, "retrieve_remote_file", fail)
    monkeypatch.setattr(run_resource, "remote_file_exists", fail)
    monkeypatch.setattr(run_resource, "remote_directory_exists", fail)
    monkeypatch.setattr(run_resource, "get_job_accounting", fail)

    context = build_scientific_context(direct_job())

    assert context.job.job_id == "21153721"


def test_no_new_cli_context_command_is_exposed(capsys) -> None:
    exit_code = cli.main(["job", "21153721", "--context-json"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent job <SLURM_JOB_ID> [--trajectory-json]" in captured.out
