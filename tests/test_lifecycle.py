from __future__ import annotations

import json
from pathlib import Path

from bmd_agent import cli
from bmd_agent.config import ConfigurationError
from bmd_agent.resources.lifecycle import (
    LifecycleState,
    analyze_calculation_directory,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord


POSCAR = """\
Si
1.0
5.43 0.00 0.00
0.00 5.43 0.00
0.00 0.00 5.43
Si
2
Direct
0.00 0.00 0.00
0.25 0.25 0.25
"""

INCAR = """\
ENCUT = 520
EDIFF = 1E-6
ISIF = 3
"""

KPOINTS = """\
Automatic mesh
0
Gamma
4 4 4
0 0 0
"""

NORMAL_OUTCAR = """\
 some vasp output
 General timing and accounting informations for this job:
 Voluntary context switches
"""


def write_inputs(directory: Path, *, missing: str | None = None) -> None:
    values = {"POSCAR": POSCAR, "INCAR": INCAR, "KPOINTS": KPOINTS}
    for name, contents in values.items():
        if name != missing:
            (directory / name).write_text(contents, encoding="utf-8")


def scheduler_record(
    *,
    job_id: str = "21153721",
    state: str = "RUNNING",
    exit_code: str = "0:0",
    work_dir: str = "/unused",
) -> SlurmAccountingRecord:
    return SlurmAccountingRecord(
        job_id=job_id,
        name="vasp",
        state=state,
        elapsed="00:10:00",
        start="2026-09-16T10:00:00",
        end="",
        partition="leeburton-pool",
        exit_code=exit_code,
        timelimit="01:00:00",
        node_list="compute-0-1",
        allocated_cpus=24,
        work_dir=work_dir,
    )


def write_submission(
    root: Path,
    *,
    job_id: str = "21153721",
    stage_dir: Path | None = None,
    result_dir: Path | None = None,
    attempt_state: Path | None = None,
) -> None:
    stage_dir = stage_dir or root / "stage_01"
    result_dir = result_dir or stage_dir
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {
                        "stage_type": "relax",
                        "theory": "pbe",
                        "modifiers": [],
                        "label": "relax",
                        "options": {},
                    }
                ]
            }
        },
        "submission": {
            "job_id": job_id,
            **({"attempt_state": str(attempt_state)} if attempt_state else {}),
        },
        "paths": {
            "stage_dirs": {"stage_01": str(stage_dir)},
            "result_dir": str(result_dir),
        },
    }
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")


def test_empty_directory_is_unknown_no_calculation(tmp_path: Path) -> None:
    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.calculation_kind == "none"
    assert "No recognizable" in analysis.message


def test_inputs_without_outputs_are_pre_run(tmp_path: Path) -> None:
    write_inputs(tmp_path)

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.PRE_RUN
    assert analysis.calculation_kind == "direct VASP"
    assert analysis.structure is not None
    assert analysis.incar_settings["ENCUT"] == 520


def test_missing_input_does_not_crash_and_reports_gap(tmp_path: Path) -> None:
    write_inputs(tmp_path, missing="KPOINTS")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert "KPOINTS is missing" in analysis.evidence_gaps


def test_non_empty_partial_output_is_not_pre_run(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "OSZICAR").write_text(" 1 F= -.1 E0= -.1 d E =0\n", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.state != LifecycleState.PRE_RUN
    assert "scheduler/provenance" in analysis.message


def test_manual_completed_vasp_without_scheduler_is_completed(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.COMPLETED
    assert analysis.normal_completion is True


def test_truncated_vasprun_does_not_imply_completed(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "vasprun.xml").write_text("<modeling>", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    assert analysis.state == LifecycleState.UNKNOWN
    assert analysis.state != LifecycleState.COMPLETED


def test_active_bmd_scheduler_provenance_is_running(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.RUNNING
    assert analysis.calculation_kind == "BMD Compute"
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_01"


def test_scheduler_running_before_outputs_is_running(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        stage,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="RUNNING"),
    )

    assert analysis.state == LifecycleState.RUNNING
    assert not any(item.non_empty for item in analysis.output_files.values())


def test_failed_scheduler_with_partial_output_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    (stage / "OUTCAR").write_text("partial", encoding="utf-8")
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        stage,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="TIMEOUT",
            exit_code="0:1",
        ),
    )

    assert analysis.state == LifecycleState.INCOMPLETE


def test_nonzero_exit_with_incomplete_output_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    (stage / "OUTCAR").write_text("partial", encoding="utf-8")
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="COMPLETED",
            exit_code="1:0",
        ),
    )

    assert analysis.state == LifecycleState.INCOMPLETE


def test_successful_scheduler_and_normal_vasp_evidence_is_completed(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    (stage / "OUTCAR").write_text(NORMAL_OUTCAR, encoding="utf-8")
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(
        stage,
        scheduler_lookup=lambda job_id: scheduler_record(
            job_id=job_id,
            state="COMPLETED",
            exit_code="0:0",
        ),
    )

    assert analysis.state == LifecycleState.COMPLETED


def test_workflow_root_invocation_uses_submission_provenance(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(root)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.workflow_root == root.resolve()
    assert analysis.calculation_kind == "BMD Compute"


def test_stage_directory_invocation_binds_to_workflow_root(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage = root / "stage_01"
    stage.mkdir(parents=True)
    write_inputs(stage)
    write_submission(root, stage_dir=stage)

    analysis = analyze_calculation_directory(stage)

    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.workflow_root == root.resolve()
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.path == stage.resolve()


def test_arbitrary_stage_named_directory_without_provenance_is_manual(tmp_path: Path) -> None:
    stage = tmp_path / "stage_01"
    stage.mkdir()
    write_inputs(stage)

    analysis = analyze_calculation_directory(stage)

    assert analysis.calculation_kind == "direct VASP"
    assert analysis.bmd_workflow is None


def test_early_stage_failure_with_later_stage_absent_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "flow"
    stage_1 = root / "stage_01"
    stage_2 = root / "stage_02"
    stage_1.mkdir(parents=True)
    write_inputs(stage_1)
    (stage_1 / "OUTCAR").write_text("partial", encoding="utf-8")
    payload = {
        "flow_spec": {
            "workflow_spec": {
                "stages": [
                    {"stage_type": "relax", "theory": "pbe", "modifiers": [], "options": {}},
                    {"stage_type": "static", "theory": "hse06", "modifiers": [], "options": {}},
                ]
            }
        },
        "submission": {"job_id": "21153721"},
        "paths": {
            "stage_dirs": {"stage_01": str(stage_1), "stage_02": str(stage_2)},
            "result_dir": str(stage_2),
        },
    }
    (root / "submission.json").write_text(json.dumps(payload), encoding="utf-8")

    analysis = analyze_calculation_directory(
        root,
        scheduler_lookup=lambda job_id: scheduler_record(job_id=job_id, state="FAILED", exit_code="1:0"),
    )

    assert analysis.state == LifecycleState.INCOMPLETE
    assert analysis.bmd_workflow is not None
    assert analysis.bmd_workflow.current_stage is not None
    assert analysis.bmd_workflow.current_stage.label == "stage_01"


def test_potcar_contents_are_not_read(tmp_path: Path) -> None:
    write_inputs(tmp_path)
    (tmp_path / "POTCAR").write_text("secret-potcar", encoding="utf-8")

    analysis = analyze_calculation_directory(tmp_path)

    observed_paths = [item.path.name for item in analysis.input_files.values()]
    observed_paths.extend(item.path.name for item in analysis.output_files.values())
    assert "POTCAR" not in observed_paths


def test_bare_cli_routes_to_cwd_lifecycle_analysis(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    write_inputs(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        cli,
        "load_resources",
        lambda: (_ for _ in ()).throw(ConfigurationError("missing config")),
    )

    exit_code = cli.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Calculation state: PRE_RUN" in captured.out
    assert "BMD Agent" in captured.out


def test_existing_explicit_cli_commands_remain_available(capsys) -> None:
    exit_code = cli.main(["job"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent job <SLURM_JOB_ID>" in captured.out
