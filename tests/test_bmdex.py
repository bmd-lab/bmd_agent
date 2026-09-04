from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from bmd_agent import cli
from bmd_agent.config import GitRepositoryResource, ResourceRegistry, parse_resources
from bmd_agent.resources import run as run_resource
from bmd_agent.resources.bmdex import (
    BMDEX_COMPOSITION_CONTEXT,
    EVIDENCE_TYPE,
    PRODUCER_MODULE,
    BmdexCompositionError,
    BmdexCompositionEvidence,
    EnrichedScientificContext,
    bmdex_repository,
    enrich_scientific_context_with_bmdex,
    inspect_bmdex_composition_context,
    parse_bmdex_composition_payload,
)
from bmd_agent.resources.context import (
    EvidenceGap,
    ScientificContext,
    ScientificIdentitySummary,
    build_scientific_context,
)
from bmd_agent.resources.run import JobInspection


def base_context(
    *,
    formula: str | None = "MnCu5",
    gaps: tuple[EvidenceGap, ...] = (),
) -> ScientificContext:
    return ScientificContext(
        job=JobInspection(
            job_id="21153721",
            scheduler=None,
            scheduler_error=None,
            scheduler_work_dir=None,
            calculation_directory=None,
            calculation_type="direct VASP",
            calculation_reason=None,
        ),
        identity=ScientificIdentitySummary(
            job_id="21153721",
            calculation_type="direct VASP",
            formula=formula,
            site_count=24 if formula else None,
            structure_evidence_type=run_resource.PYMATGEN_DERIVED if formula else None,
            executed_input_indicators={
                "spin_polarized": {
                    "evidence_type": run_resource.EXECUTED_INPUT,
                    "parameter": "ISPIN",
                    "value": True,
                    "status": "available",
                    "source_values": {"retained_incar:work_dir": 2},
                },
            } if formula else {},
            executed_input_evidence_type=run_resource.EXECUTED_INPUT if formula else None,
        ),
        evidence_gaps=gaps,
    )


def repository(
    tmp_path: Path,
    *,
    include_checkout: bool = True,
    include_python: bool = True,
    capability_python: bool = True,
) -> GitRepositoryResource:
    checkout = tmp_path / "BMDex"
    python = tmp_path / "env" / "bin" / "python"
    if include_checkout:
        checkout.mkdir()
    if include_python:
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("", encoding="utf-8")
    return GitRepositoryResource(
        key="bmdex",
        name="BMDex",
        path=checkout,
        role="curated_knowledge",
        access="read_only",
        protected=True,
        live=False,
        capability_python=python if capability_python else None,
    )


def mncu5_payload() -> dict:
    return {
        "schema_version": 1,
        "status": "ok",
        "evidence_type": "composition_context",
        "producer": {
            "name": "BMDex",
            "contract_module": "tools.composition.context_producer",
            "contract_command": "python -B -m tools.composition.context_producer",
            "git": {
                "commit": "a" * 40,
                "dirty": False,
                "state": "clean",
            },
        },
        "composition": {
            "supplied_formula": "MnCu5",
            "reduced_formula": "MnCu5",
            "elements": ["Mn", "Cu"],
            "stoichiometric_amounts": [
                {"element": "Mn", "amount": 1},
                {"element": "Cu", "amount": 5},
            ],
        },
        "datasets": {
            "element_abundances": {
                "dataset_id": "bmdex.datasets.element_abundances.earth_abundance",
                "path": "datasets/element_abundances/earth-abundance.yaml",
                "quantity": "crustal abundance",
                "units": "mg/kg",
                "source_reference_status": "not_machine_readable",
                "records": [
                    {
                        "element": "Mn",
                        "status": "present",
                        "abundance": 950.0,
                        "units": "mg/kg",
                    },
                    {
                        "element": "Cu",
                        "status": "present",
                        "abundance": 60.0,
                        "units": "mg/kg",
                    },
                ],
            },
            "element_charges": {
                "dataset_id": "bmdex.datasets.element_charges.oxidation_states_84",
                "path": "datasets/element_charges/oxidation_states_84.yaml",
                "term": "representative oxidation states",
                "source_reference_status": "not_machine_readable",
                "records": [
                    {
                        "element": "Mn",
                        "status": "present",
                        "representative_oxidation_states": [2],
                    },
                    {
                        "element": "Cu",
                        "status": "present",
                        "representative_oxidation_states": [1, 2],
                    },
                ],
            },
        },
        "missing_evidence": [],
        "limitations": [
            {
                "code": "abundance_element_level_only",
                "text": (
                    "Element abundance is contextual element-level evidence and "
                    "does not establish compound viability or sustainability."
                ),
            },
            {
                "code": "oxidation_states_element_level_only",
                "text": (
                    "Representative oxidation-state entries are element-level "
                    "reference data and do not establish oxidation states, charge "
                    "balance, stability, or existence of the supplied compound."
                ),
            },
        ],
    }


def he_payload() -> dict:
    payload = mncu5_payload()
    payload["composition"] = {
        "supplied_formula": "He",
        "reduced_formula": "He",
        "elements": ["He"],
        "stoichiometric_amounts": [{"element": "He", "amount": 1}],
    }
    payload["datasets"]["element_abundances"]["records"] = [
        {
            "element": "He",
            "status": "present",
            "abundance": 0.008,
            "units": "mg/kg",
        }
    ]
    payload["datasets"]["element_charges"]["records"] = [
        {
            "element": "He",
            "status": "missing",
            "representative_oxidation_states": None,
        }
    ]
    payload["missing_evidence"] = [
        {
            "dataset": "element_charges",
            "element": "He",
            "reason": "element_not_present_in_dataset",
        }
    ]
    return payload


def structured_error_payload() -> dict:
    return {
        "schema_version": 1,
        "status": "error",
        "evidence_type": "composition_context",
        "producer": {
            "name": "BMDex",
            "contract_module": "tools.composition.context_producer",
            "contract_command": "python -B -m tools.composition.context_producer",
            "git": {
                "commit": "a" * 40,
                "dirty": False,
                "state": "clean",
            },
        },
        "error": {
            "code": "malformed_formula",
            "message": "Formula syntax is unsupported or malformed.",
        },
    }


def completed(payload: dict, *, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["python"],
        returncode=returncode,
        stdout=json.dumps(payload),
        stderr=stderr,
    )


def test_configured_bmdex_repository_and_capability_python_is_parsed() -> None:
    registry = parse_resources(
        {
            "repositories": {
                "bmdex": {
                    "name": "BMDex",
                    "path": "/home/example/projects/BMDex",
                    "role": "curated_knowledge",
                    "access": "read_only",
                    "protected": True,
                    "live": False,
                    "capability_python": "/home/example/micromamba/envs/bmdex/bin/python",
                }
            },
            "clusters": {
                "powerslurm": {
                    "name": "PowerSLURM",
                    "ssh_host": "powerslurm-bmdguest",
                    "partition": "leeburton-pool",
                    "access": "observational",
                    "allowed_remote_roots": ["/bmd-db/guest/flows"],
                }
            },
        }
    )

    repo = bmdex_repository(registry)

    assert repo is not None
    assert repo.capability_python is not None
    assert repo.capability_python.parts[-3:] == ("bmdex", "bin", "python")


def test_exact_fixed_producer_command_cwd_json_stdin_and_shell_false(tmp_path: Path) -> None:
    repo = repository(tmp_path)
    seen = {}

    def runner(command, **kwargs):
        seen["command"] = command
        seen.update(kwargs)
        return completed(mncu5_payload())

    evidence = inspect_bmdex_composition_context(repo, "MnCu5", runner=runner, timeout=7)

    assert evidence.status == "ok"
    assert seen["command"] == [str(repo.capability_python), "-B", "-m", PRODUCER_MODULE]
    assert seen["cwd"] == repo.path
    assert json.loads(seen["input"]) == {"formula": "MnCu5"}
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["check"] is False
    assert seen["timeout"] == 7
    assert seen["shell"] is False


def test_valid_mncu5_schema_v1_response_is_preserved(tmp_path: Path) -> None:
    payload = mncu5_payload()

    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(payload),
    )

    assert isinstance(enriched, EnrichedScientificContext)
    assert isinstance(enriched.composition_context, BmdexCompositionEvidence)
    assert enriched.composition_context.payload == payload
    assert enriched.composition_context.schema_version == 1
    assert enriched.composition_context.evidence_type == EVIDENCE_TYPE
    assert enriched.composition_context.datasets["element_abundances"]["records"][0]["abundance"] == 950.0
    assert enriched.evidence_gaps == ()


def test_he_status_ok_with_missing_evidence_remains_successful_enrichment(tmp_path: Path) -> None:
    context = base_context(formula="He")
    payload = he_payload()

    enriched = enrich_scientific_context_with_bmdex(
        context,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(payload),
    )

    assert enriched.composition_context is not None
    assert enriched.composition_context.status == "ok"
    assert enriched.composition_context.missing_evidence == (
        {
            "dataset": "element_charges",
            "element": "He",
            "reason": "element_not_present_in_dataset",
        },
    )
    assert enriched.evidence_gaps == ()


def test_formula_unavailable_causes_no_subprocess_call(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        raise AssertionError("BMDex must not be invoked without a safe formula")

    enriched = enrich_scientific_context_with_bmdex(
        base_context(formula=None),
        repository(tmp_path),
        runner=runner,
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps == (
        EvidenceGap(
            BMDEX_COMPOSITION_CONTEXT,
            "formula_unavailable",
            "ScientificContext identity formula is unavailable.",
        ),
    )


def test_missing_repository_is_an_enrichment_gap() -> None:
    enriched = enrich_scientific_context_with_bmdex(base_context(), None)

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_bmdex_repository"


def test_missing_checkout_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path, include_checkout=False),
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_checkout"


def test_missing_interpreter_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path, include_python=False),
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_interpreter"


def test_missing_capability_python_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path, capability_python=False),
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "missing_interpreter"


def test_nonzero_producer_exit_without_contract_response_is_an_enrichment_gap(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["python"],
            returncode=1,
            stdout="not json",
            stderr="Traceback hidden\nValueError: boom\n" + ("x" * 500),
        )

    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=runner,
    )

    assert enriched.composition_context is None
    gap = enriched.evidence_gaps[0]
    assert gap.scope == "producer_failed"
    assert "exit 1" in gap.reason
    assert "ValueError: boom" in gap.reason
    assert "Traceback" not in gap.reason
    assert len(gap.reason) <= 240


def test_structured_producer_error_is_an_enrichment_gap(tmp_path: Path) -> None:
    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(structured_error_payload(), returncode=2),
    )

    assert enriched.composition_context is None
    gap = enriched.evidence_gaps[0]
    assert gap.scope == "structured_producer_error"
    assert "malformed_formula" in gap.reason


def test_malformed_json_is_reported() -> None:
    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload("{not-json")

    assert exc_info.value.kind == "malformed_json"


def test_malformed_payload_is_reported_for_missing_required_fields() -> None:
    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(
            json.dumps({"schema_version": 1, "status": "ok", "evidence_type": "composition_context"})
        )

    assert exc_info.value.kind == "malformed_payload"


def test_malformed_payload_is_reported_for_formula_mismatch() -> None:
    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(
            json.dumps(he_payload()),
            requested_formula="MnCu5",
        )

    assert exc_info.value.kind == "malformed_payload"


def test_malformed_payload_is_reported_for_unexpected_evidence_type() -> None:
    payload = mncu5_payload()
    payload["evidence_type"] = "other_context"

    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(json.dumps(payload))

    assert exc_info.value.kind == "malformed_payload"


def test_malformed_payload_is_reported_for_unexpected_status() -> None:
    payload = mncu5_payload()
    payload["status"] = "partial"

    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(json.dumps(payload))

    assert exc_info.value.kind == "malformed_payload"


def test_unsupported_schema_version_is_reported() -> None:
    payload = mncu5_payload()
    payload["schema_version"] = 2

    with pytest.raises(BmdexCompositionError) as exc_info:
        parse_bmdex_composition_payload(json.dumps(payload))

    assert exc_info.value.kind == "unsupported_schema"


def test_timeout_is_an_enrichment_gap(tmp_path: Path) -> None:
    def runner(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["python"], timeout=20)

    enriched = enrich_scientific_context_with_bmdex(
        base_context(),
        repository(tmp_path),
        runner=runner,
    )

    assert enriched.composition_context is None
    assert enriched.evidence_gaps[0].scope == "timeout"


def test_bmdex_provenance_and_limitations_are_preserved_independently(tmp_path: Path) -> None:
    payload = mncu5_payload()
    context = base_context(
        gaps=(EvidenceGap(run_resource.PRODUCER_PROVENANCE, "producer", "no BMD Compute producer record found"),)
    )

    enriched = enrich_scientific_context_with_bmdex(
        context,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(payload),
    )

    assert enriched.base is context
    assert enriched.base.evidence_gaps == context.evidence_gaps
    assert enriched.composition_context is not None
    assert enriched.composition_context.producer["git"]["commit"] == "a" * 40
    assert enriched.composition_context.producer["git"]["state"] == "clean"
    assert enriched.composition_context.limitations == tuple(payload["limitations"])
    assert enriched.evidence_gaps == ()


def test_existing_direct_vasp_context_evidence_remains_unchanged(tmp_path: Path) -> None:
    context = base_context(
        gaps=(EvidenceGap(run_resource.TRAJECTORY_OBSERVATION, "stage_1.vasprun", "file could not be parsed completely"),)
    )

    enriched = enrich_scientific_context_with_bmdex(
        context,
        repository(tmp_path),
        runner=lambda *_args, **_kwargs: completed(mncu5_payload()),
    )

    assert enriched.base is context
    assert enriched.base.identity.calculation_type == "direct VASP"
    assert enriched.base.identity.producer_workflow is None
    assert enriched.base.identity.executed_input_indicators["spin_polarized"]["value"] is True
    assert enriched.base.evidence_gaps == context.evidence_gaps


def test_build_scientific_context_remains_zero_io(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("ScientificContext builder must remain pure")

    monkeypatch.setattr(run_resource, "retrieve_remote_file", fail)
    monkeypatch.setattr(run_resource, "remote_file_exists", fail)
    monkeypatch.setattr(run_resource, "remote_directory_exists", fail)
    monkeypatch.setattr(run_resource, "get_job_accounting", fail)

    context = build_scientific_context(
        JobInspection(
            job_id="21153721",
            scheduler=None,
            scheduler_error="scheduler unavailable",
            scheduler_work_dir=None,
            calculation_directory=None,
            calculation_type="unknown",
            calculation_reason="scheduler unavailable",
        )
    )

    assert context.identity.job_id == "21153721"


def test_no_cli_command_was_added(capsys) -> None:
    exit_code = cli.main(["bmdex"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unknown command: bmdex" in captured.out


def test_agent_adapter_does_not_import_bmdex_or_parse_datasets() -> None:
    source = Path("src/bmd_agent/resources/bmdex.py").read_text(encoding="utf-8")

    assert "from tools" not in source
    assert "import tools" not in source
    assert ".yaml" not in source.lower()


def test_agent_adapter_has_no_network_scheduler_or_write_paths() -> None:
    source = Path("src/bmd_agent/resources/bmdex.py").read_text(encoding="utf-8")

    forbidden = (
        "urllib",
        "requests",
        "socket",
        "sacct",
        "squeue",
        "ssh",
        "open(",
        ".open(",
        "write_text",
        "mkdir",
    )
    assert not [token for token in forbidden if token in source]
