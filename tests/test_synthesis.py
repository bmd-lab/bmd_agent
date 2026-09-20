from __future__ import annotations

from dataclasses import replace
import subprocess

from bmd_agent import cli
from bmd_agent.resources import run as run_resource
from bmd_agent.resources.bmdex import (
    BMDEX_COMPOSITION_CONTEXT,
    BmdexCompositionEvidence,
    EnrichedScientificContext,
)
from bmd_agent.resources.context import (
    EvidenceGap,
    ScientificContext,
    ScientificIdentitySummary,
    build_scientific_context,
)
from bmd_agent.resources.run import (
    CONVERGENCE_PROGRESS_ASSESSMENT,
    EXECUTED_INPUT,
    PRODUCER_REQUESTED,
    TRAJECTORY_PROGRESS_EVIDENCE,
    ConvergenceProgressAssessment,
    JobInspection,
    WorkflowStage,
)
from bmd_agent.resources.synthesis import (
    CATEGORY_ASSESSMENT,
    CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE,
    CATEGORY_DERIVED_OBSERVATION,
    CATEGORY_EVIDENCE_GAP,
    CATEGORY_LIMITATION,
    CATEGORY_OBSERVATION,
    ScientificEvidenceSummary,
    build_scientific_evidence_summary,
)
from test_bmdex import he_payload, mncu5_payload
from test_context import bmd_compute_job, direct_job, direct_trajectory


def enriched_direct_context(
    *,
    with_bmdex: bool = True,
) -> EnrichedScientificContext:
    return EnrichedScientificContext(
        base=build_scientific_context(direct_job()),
        composition_context=(
            BmdexCompositionEvidence(mncu5_payload()) if with_bmdex else None
        ),
    )


def item_values(summary: ScientificEvidenceSummary, predicate: str) -> list:
    return [item.value for item in summary.items if item.predicate == predicate]


def test_summary_keeps_original_enriched_context_identity() -> None:
    context = enriched_direct_context()

    summary = build_scientific_evidence_summary(context)

    assert summary.source_context is context


def test_direct_vasp_summary_has_native_observations_without_workflow_inference() -> None:
    summary = build_scientific_evidence_summary(enriched_direct_context(with_bmdex=False))

    assert "direct VASP" in item_values(summary, "calculation_type")
    assert "MnCu5" in item_values(summary, "formula")
    assert 24 in item_values(summary, "site_count")
    assert not [
        item
        for item in summary.items
        if item.subject == "producer_workflow"
    ]


def test_direct_vasp_summary_includes_executed_input_indicators() -> None:
    summary = build_scientific_evidence_summary(enriched_direct_context(with_bmdex=False))
    spin = next(item for item in summary.items if item.predicate == "spin_polarized")
    dispersion = next(
        item for item in summary.items if item.predicate == "dispersion_indicator"
    )

    assert spin.category == CATEGORY_OBSERVATION
    assert spin.value is True
    assert spin.status == "available"
    assert spin.source is not None
    assert spin.source.source_evidence_type == EXECUTED_INPUT
    assert spin.source.provenance["parameter"] == "ISPIN"
    assert dispersion.value == 12


def test_trajectory_progress_evidence_is_summarized_from_existing_derivation() -> None:
    summary = build_scientific_evidence_summary(enriched_direct_context(with_bmdex=False))

    assert 50 in item_values(summary, "completed_ionic_steps")
    assert 0.2 in item_values(summary, "initial_max_force_eV_A")
    assert 0.071891 in item_values(summary, "current_max_force_eV_A")
    assert 0.01 in item_values(summary, "force_criterion_magnitude_eV_A")
    progress_item = next(
        item for item in summary.items
        if item.predicate == "current_force_over_abs_EDIFFG"
    )
    assert progress_item.category == CATEGORY_DERIVED_OBSERVATION
    assert progress_item.source is not None
    assert progress_item.source.source_evidence_type == TRAJECTORY_PROGRESS_EVIDENCE
    assert "derive_trajectory_progress_evidence" in progress_item.source.native_path


def test_bmd_compute_summary_preserves_workflow_stage_options() -> None:
    inspection = bmd_compute_job()
    run = inspection.bmd_compute.inspection
    stage = WorkflowStage(
        1,
        "relax",
        "pbe",
        ("dispersion",),
        "relax_with_modifier",
        options={"modifier": {"dispersion": {"method": "dftd3-bj"}}},
    )
    updated = replace(
        inspection,
        bmd_compute=replace(
            inspection.bmd_compute,
            inspection=replace(run, workflow_stages=(stage,)),
        ),
    )
    summary = build_scientific_evidence_summary(
        EnrichedScientificContext(base=build_scientific_context(updated))
    )

    workflow_items = [
        item for item in summary.items
        if item.subject == "producer_workflow" and item.predicate == "stage"
    ]
    assert len(workflow_items) == 1
    assert workflow_items[0].source is not None
    assert workflow_items[0].source.source_evidence_type == PRODUCER_REQUESTED
    assert workflow_items[0].value["options"] == {
        "modifier": {"dispersion": {"method": "dftd3-bj"}}
    }


def test_bmdex_contextual_reference_evidence_and_provenance_are_preserved() -> None:
    context = EnrichedScientificContext(
        base=build_scientific_context(direct_job()),
        composition_context=BmdexCompositionEvidence(mncu5_payload()),
    )

    summary = build_scientific_evidence_summary(context)

    abundance = next(
        item for item in summary.items
        if item.subject == "Mn" and item.predicate == "element_abundance"
    )
    charges = next(
        item for item in summary.items
        if item.subject == "Cu"
        and item.predicate == "representative_oxidation_states"
    )
    assert abundance.category == CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE
    assert abundance.value == 950.0
    assert abundance.unit == "mg/kg"
    assert abundance.source is not None
    assert abundance.source.provenance["producer"]["name"] == "BMDex"
    assert abundance.source.provenance["dataset_id"] == (
        "bmdex.datasets.element_abundances.earth_abundance"
    )
    assert charges.value == [1, 2]


def test_bmdex_missing_evidence_becomes_source_specific_gap() -> None:
    context = EnrichedScientificContext(
        base=build_scientific_context(direct_job()),
        composition_context=BmdexCompositionEvidence(he_payload()),
    )

    summary = build_scientific_evidence_summary(context)

    gap = next(
        item for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP
        and item.subject == "BMDex element_charges.He"
    )
    assert gap.status == "unavailable"
    assert gap.source is not None
    assert gap.source.source_evidence_type == "composition_context"
    assert gap.value == {
        "dataset": "element_charges",
        "element": "He",
        "reason": "element_not_present_in_dataset",
    }


def test_context_and_enrichment_gaps_are_explicit_only() -> None:
    base = ScientificContext(
        job=JobInspection(
            job_id="1",
            scheduler=None,
            scheduler_error=None,
            scheduler_work_dir=None,
            calculation_directory=None,
            calculation_type="unknown",
            calculation_reason=None,
        ),
        identity=ScientificIdentitySummary(
            job_id="1",
            calculation_type="unknown",
        ),
        evidence_gaps=(
            EvidenceGap("test_evidence", "explicit_scope", "explicit reason"),
        ),
    )
    context = EnrichedScientificContext(
        base=base,
        evidence_gaps=(
            EvidenceGap(
                BMDEX_COMPOSITION_CONTEXT,
                "missing_bmdex_repository",
                "BMDex repository configuration is unavailable.",
            ),
        ),
    )

    summary = build_scientific_evidence_summary(context)

    gap_values = [
        item.value for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP
    ]
    assert "explicit reason" in gap_values
    assert "BMDex repository configuration is unavailable." in gap_values
    assert not [
        item for item in summary.items
        if "literature" in item.subject.lower()
        or "stability" in item.subject.lower()
        or "synthesis" in item.subject.lower()
    ]


def test_convergence_assessments_are_preserved_without_upgrade() -> None:
    trajectory = direct_trajectory(completed_steps=0)
    inspection = direct_job(trajectory=trajectory)
    summary = build_scientific_evidence_summary(
        EnrichedScientificContext(base=build_scientific_context(inspection))
    )

    assessment_items = [
        item for item in summary.items
        if item.category == CATEGORY_ASSESSMENT
    ]
    expected = inspection.direct_vasp.assessments
    assert [item.value for item in assessment_items] == [
        assessment.label for assessment in expected
    ]
    assert [item.status for item in assessment_items] == [
        assessment.sufficiency for assessment in expected
    ]
    assert all(
        item.source is not None
        and item.source.source_evidence_type == CONVERGENCE_PROGRESS_ASSESSMENT
        for item in assessment_items
    )


def test_limitations_remain_source_separated() -> None:
    summary = build_scientific_evidence_summary(enriched_direct_context())

    limitation_items = [
        item for item in summary.items
        if item.category == CATEGORY_LIMITATION
    ]
    assert any(
        item.source is not None
        and item.source.source_evidence_type == TRAJECTORY_PROGRESS_EVIDENCE
        for item in limitation_items
    )
    assert any(
        item.source is not None
        and item.source.source_evidence_type == "composition_context"
        and item.subject == "BMDex composition_context"
        for item in limitation_items
    )


def test_variable_cell_force_scope_limitation_is_not_reemitted_as_gap() -> None:
    limitation = (
        "atomic-force evidence only; variable-cell convergence also requires "
        "broader cell/stress evidence"
    )

    summary = build_scientific_evidence_summary(enriched_direct_context(with_bmdex=False))

    limitation_items = [
        item for item in summary.items
        if item.category == CATEGORY_LIMITATION and item.value == limitation
    ]
    mirrored_gap_items = [
        item for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP and item.value == limitation
    ]
    assert len(limitation_items) == 1
    assert limitation_items[0].source is not None
    assert limitation_items[0].source.source_evidence_type == TRAJECTORY_PROGRESS_EVIDENCE
    assert mirrored_gap_items == []


def test_assessment_limitations_are_not_reemitted_as_gaps() -> None:
    trend_limitation = "trend-based ionic progress assessment is not implemented in v1"
    stage_limitation = (
        "ionic progress evidence is insufficient for force-based stage progress assessment"
    )
    expected_limitations = {trend_limitation, stage_limitation}
    inspection = direct_job()
    assessments = (
        ConvergenceProgressAssessment(
            label="INSUFFICIENT EVIDENCE",
            stage_index=1,
            stage_label="work_dir",
            scope="ionic",
            sufficiency="insufficient",
            limitations=(trend_limitation,),
        ),
        ConvergenceProgressAssessment(
            label="INSUFFICIENT EVIDENCE",
            stage_index=1,
            stage_label="work_dir",
            scope="stage",
            sufficiency="insufficient",
            limitations=(stage_limitation,),
        ),
    )
    updated = replace(
        inspection,
        direct_vasp=replace(inspection.direct_vasp, assessments=assessments),
    )

    summary = build_scientific_evidence_summary(
        EnrichedScientificContext(base=build_scientific_context(updated))
    )

    limitation_items = [
        item for item in summary.items
        if item.category == CATEGORY_LIMITATION and item.value in expected_limitations
    ]
    mirrored_gap_items = [
        item for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP and item.value in expected_limitations
    ]
    assert {item.value for item in limitation_items} == expected_limitations
    assert {
        item.source.source_scope
        for item in limitation_items
        if item.source is not None
    } == {"work_dir.ionic", "work_dir.stage"}
    assert mirrored_gap_items == []


def test_same_limitation_text_survives_at_distinct_assessment_scopes() -> None:
    shared_limitation = "shared native limitation"
    inspection = direct_job()
    assessments = (
        ConvergenceProgressAssessment(
            label="INSUFFICIENT EVIDENCE",
            stage_index=1,
            stage_label="work_dir",
            scope="ionic",
            sufficiency="insufficient",
            limitations=(shared_limitation,),
        ),
        ConvergenceProgressAssessment(
            label="INSUFFICIENT EVIDENCE",
            stage_index=1,
            stage_label="work_dir",
            scope="stage",
            sufficiency="insufficient",
            limitations=(shared_limitation,),
        ),
    )
    updated = replace(
        inspection,
        direct_vasp=replace(inspection.direct_vasp, assessments=assessments),
    )

    summary = build_scientific_evidence_summary(
        EnrichedScientificContext(base=build_scientific_context(updated))
    )

    limitation_items = [
        item for item in summary.items
        if item.category == CATEGORY_LIMITATION and item.value == shared_limitation
    ]
    mirrored_gap_items = [
        item for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP and item.value == shared_limitation
    ]
    assert {
        item.source.source_scope
        for item in limitation_items
        if item.source is not None
    } == {"work_dir.ionic", "work_dir.stage"}
    assert mirrored_gap_items == []


def test_genuine_context_gaps_remain_evidence_gaps_after_limitation_filter() -> None:
    trajectory = direct_trajectory(vasprun_error="file could not be parsed completely")

    summary = build_scientific_evidence_summary(
        EnrichedScientificContext(
            base=build_scientific_context(direct_job(trajectory=trajectory))
        )
    )

    gap_values = [
        item.value for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP
    ]
    assert "no BMD Compute producer record found" in gap_values
    assert "file could not be parsed completely" in gap_values


def test_bmdex_scientific_limitations_are_not_evidence_gaps() -> None:
    payload = mncu5_payload()
    abundance_limitation = payload["limitations"][0]
    context = EnrichedScientificContext(
        base=build_scientific_context(direct_job()),
        composition_context=BmdexCompositionEvidence(payload),
    )

    summary = build_scientific_evidence_summary(context)

    assert any(
        item.category == CATEGORY_LIMITATION
        and item.value == abundance_limitation
        and item.source is not None
        and item.source.source_evidence_type == "composition_context"
        for item in summary.items
    )
    assert not [
        item for item in summary.items
        if item.category == CATEGORY_EVIDENCE_GAP and item.value == abundance_limitation
    ]


def test_upstream_context_and_assessment_objects_are_not_changed_by_filter() -> None:
    context = build_scientific_context(direct_job())
    expected_gaps = context.evidence_gaps
    expected_assessments = context.job.direct_vasp.assessments

    summary = build_scientific_evidence_summary(
        EnrichedScientificContext(base=context)
    )

    assert context.evidence_gaps == expected_gaps
    assert context.job.direct_vasp.assessments == expected_assessments
    assert [
        item.value for item in summary.items
        if item.category == CATEGORY_ASSESSMENT
    ] == [assessment.label for assessment in expected_assessments]


def test_categories_are_explicit_and_bounded() -> None:
    summary = build_scientific_evidence_summary(enriched_direct_context())

    categories = {item.category for item in summary.items}
    assert {
        CATEGORY_OBSERVATION,
        CATEGORY_DERIVED_OBSERVATION,
        CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE,
        CATEGORY_ASSESSMENT,
        CATEGORY_EVIDENCE_GAP,
        CATEGORY_LIMITATION,
    }.issubset(categories)
    assert categories <= {
        CATEGORY_OBSERVATION,
        CATEGORY_DERIVED_OBSERVATION,
        CATEGORY_CONTEXTUAL_REFERENCE_EVIDENCE,
        CATEGORY_ASSESSMENT,
        CATEGORY_EVIDENCE_GAP,
        CATEGORY_LIMITATION,
    }


def test_native_paths_are_meaningful_and_source_context_relative() -> None:
    summary = build_scientific_evidence_summary(enriched_direct_context())

    sourced = [item for item in summary.items if item.source is not None]
    assert sourced
    assert all(item.source.native_path.startswith("source_context.") for item in sourced)
    assert any(
        item.source.native_path
        == "source_context.base.job.scheduler.state"
        for item in sourced
    )
    assert any(
        item.source.native_path
        == (
            "source_context.composition_context.payload.datasets."
            "element_abundances.records[0]"
        )
        for item in sourced
    )


def test_builder_performs_no_remote_reads_subprocess_network_or_producer_calls(
    monkeypatch,
) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("ScientificEvidenceSummary builder must not acquire evidence")

    monkeypatch.setattr(subprocess, "run", fail)
    monkeypatch.setattr(run_resource, "retrieve_remote_file", fail)
    monkeypatch.setattr(run_resource, "remote_file_exists", fail)
    monkeypatch.setattr(run_resource, "remote_directory_exists", fail)
    monkeypatch.setattr(run_resource, "get_job_accounting", fail)

    summary = build_scientific_evidence_summary(enriched_direct_context())

    assert summary.source_context.base.identity.job_id == "21153721"


def test_no_cli_synthesis_surface_is_exposed(capsys) -> None:
    exit_code = cli.main(["evidence-summary"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Target was not recognized as a SLURM job ID" in captured.out
