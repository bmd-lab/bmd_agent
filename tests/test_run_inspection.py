import json
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import sys
import types
import warnings

import pytest

from bmd_agent import cli
import bmd_agent.resources.run as run_resource
from bmd_agent.config import ResourceRegistry, SlurmClusterResource
from bmd_agent.resources.run import (
    AGENT_COMPARISON,
    ARTIFACT_OBSERVATION,
    EXECUTED_INPUT,
    LOG_OBSERVATION,
    PYMATGEN_DERIVED,
    AttemptStateObservation,
    ComparisonObservation,
    IncarObservation,
    InitialStructureObservation,
    InputExpectationObservation,
    LogRuntimeObservation,
    PathObservation,
    RunDiagnosis,
    RunInspection,
    RunInspectionError,
    ScientificResult,
    StageTrajectoryObservation,
    TerminationObservation,
    StructureObservation,
    TRAJECTORY_OBSERVATION,
    WorkflowStage,
    build_run_comparison,
    compare_remote_runs,
    compare_requested_options_to_executed_inputs,
    diagnose_remote_run,
    inspect_remote_run,
    parse_incar_contents,
    parse_oszicar_trajectory,
    parse_vasp_output_files,
    run_label_from_provenance,
    vasp_reported_parameter_observations,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord
from bmd_agent.resources.vasp import RemotePathError


FLOW_ROOT = "/bmd-db/guest/flows/validation-run"
LOG_ROOT = "/bmd-db/guest/logs"
RESULT_DIR = f"{FLOW_ROOT}/producer-delta"
OSZICAR_TWO_STEP = b"""\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E+00   -2.00000E-01   12   1.000E-01   2.000E-02
   1 F= -.11000000E+02 E0= -.10950000E+02  d E =-.110000E+02
       N       E                     dE             d eps       ncg     rms          rms(c)
RMM:   1   -1.200000000000E+01   -1.00000E+00   -1.00000E-01   10   5.000E-02   1.000E-02
RMM:   2   -1.210000000000E+01   -1.00000E-01   -1.00000E-02   12   4.000E-02   8.000E-03
   2 F= -.12100000E+02 E0= -.12050000E+02  d E =-.110000E+01
"""
OSZICAR_NELM_LIMIT = b"""\
       N       E                     dE             d eps       ncg     rms          rms(c)
DAV:   1   -1.000000000000E+01   -1.00000E+01   -1.00000E+01   10   1.000E+00   2.000E-01
DAV:   2   -1.100000000000E+01   -1.00000E+00   -2.00000E-01   12   1.000E-01   2.000E-02
DAV:   3   -1.110000000000E+01   -1.00000E-01   -2.00000E-02   12   9.000E-02   1.000E-02
   1 F= -.11100000E+02 E0= -.11050000E+02  d E =-.111000E+02
"""


def oszicar_incomplete_cycle(iterations: int = 25) -> bytes:
    lines = [
        "       N       E                     dE             d eps       ncg     rms          rms(c)"
    ]
    for index in range(1, iterations + 1):
        lines.append(
            f"DAV: {index:3d}   {-10 - index / 100:.12E}   -1.00000E-02   "
            "-1.00000E-03   10   1.000E-01   2.000E-02"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


class RemoteFixture:
    def __init__(self, *, files: dict[str, bytes], directories: set[str]) -> None:
        self.files = files
        self.directories = directories
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.commands.append(command)
        assert command[:2] == ["ssh", "powerslurm-bmdguest"]
        assert kwargs["capture_output"] is True
        assert kwargs["timeout"] == 20

        remote_command = command[2]
        parts = shlex.split(remote_command)

        if parts[:2] == ["cat", "--"]:
            assert kwargs["check"] is True
            path = parts[2]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(command, 0, stdout=self.files[path], stderr=b"")

        if parts[:2] == ["test", "-f"]:
            assert kwargs["check"] is False
            return subprocess.CompletedProcess(
                command,
                0 if parts[2] in self.files else 1,
                stdout=b"",
                stderr=b"",
            )

        if parts[:2] == ["test", "-d"]:
            assert kwargs["check"] is False
            return subprocess.CompletedProcess(
                command,
                0 if parts[2] in self.directories else 1,
                stdout=b"",
                stderr=b"",
            )

        if parts[:4] == ["stat", "-c", "%s", "--"]:
            assert kwargs["check"] is True
            path = parts[4]
            if path not in self.files:
                raise subprocess.CalledProcessError(1, command, stderr=b"missing")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{len(self.files[path])}\n".encode("utf-8"),
                stderr=b"",
            )

        raise AssertionError(f"unexpected remote command: {remote_command}")


def cluster() -> SlurmClusterResource:
    return SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(PurePosixPath("/bmd-db/guest"),),
    )


def submission_payload(*, outside_path: bool = False) -> dict:
    stage_dirs = {
        "producer-alpha": f"{FLOW_ROOT}/producer-alpha",
        "producer-beta": f"{FLOW_ROOT}/producer-beta",
        "producer-gamma": f"{FLOW_ROOT}/producer-gamma",
        "producer-delta": RESULT_DIR,
    }
    if outside_path:
        stage_dirs["producer-beta"] = "/etc/not-authorized"

    workflow_spec = {
        "stages": [
            {"stage_type": "relax", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
            {
                "stage_type": "static",
                "theory": "r2scan",
                "modifiers": ["custom_modifier"],
                "label": None,
                "options": {"custom_modifier": {"nested": "kept"}},
            },
            {"stage_type": "dos", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
            {
                "stage_type": "band_structure",
                "theory": "pbe",
                "modifiers": ["dispersion"],
                "label": None,
                "options": {"dispersion": {"method": "dftd3"}},
            },
        ],
        "label": None,
        "recipe": "custom",
    }
    return {
        "flow_spec": {
            "workflow": "custom_workflow",
            "workflow_spec": workflow_spec,
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": "Example\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\ndirect\n0 0 0\n",
            },
        },
        "paths": {
            "run_dir": FLOW_ROOT,
            "logs_dir": LOG_ROOT,
            "stage_dirs": stage_dirs,
            "result_dir": RESULT_DIR,
            "log_out": f"{LOG_ROOT}/validation-run.out",
            "log_err": f"{LOG_ROOT}/validation-run.err",
            "slurm_out": f"{LOG_ROOT}/validation-run.slurm.out",
            "slurm_err": f"{LOG_ROOT}/validation-run.slurm.err",
        },
        "cluster": {"partition": "leeburton-pool", "account": "account-name"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 160, "walltime": "04:00:00"},
        "environment": {
            "VASP_CMD": "srun --mpi=pmi2 -n $SLURM_NTASKS vasp_std",
            "PMG_VASP_PSP_DIR": "/bmd-db/potcars",
        },
        "submission": {
            "attempt_state": f"{LOG_ROOT}/submission_attempts/attempt.json",
        },
        "provenance": {
            "bmd_compute": {
                "source": {
                    "git_commit": "abcdef0123456789",
                    "state": "clean",
                    "dirty": False,
                }
            },
            "execution": {"workflow_spec": workflow_spec},
        },
    }


def default_files(*, include_vasprun: bool = True, invalid_job_id: bool = False) -> dict[str, bytes]:
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(submission_payload()).encode("utf-8"),
        f"{LOG_ROOT}/submission_attempts/attempt.json": json.dumps(
            {
                "job_id": "20893681;scancel 1" if invalid_job_id else "20893681",
                "job_record": {
                    "job_id": "20893681;scancel 1" if invalid_job_id else "20893681"
                },
            }
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.out": (
            "[runner] python: 3.12.13 (main)\n"
            "[runner] atomate2 version: 0.1.5\n"
            "[runner] jobflow version: 0.1.19\n"
            "[runner] pymatgen version: 2026.8.13\n"
            "[runner] custodian version: 2025.12.14\n"
            "PMG_VASP_PSP_DIR=/bmd-db/potcars\n"
            "2026-08-21 12:00:00 Starting job - stage_01 (64210872-5626-40c7-a7eb-79f7e49272ba)\n"
            "2026-08-21 12:10:00 Starting job - custom.second (328290de-b493-4d39-a4c7-238eb9055720)\n"
            "2026-08-21 12:20:00 Starting job - final stage (48F52369-D3B9-40B5-9A9F-A87BAE6D007F)\n"
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.err": b"",
        f"{LOG_ROOT}/validation-run.slurm.out": b"",
        f"{LOG_ROOT}/validation-run.slurm.err": b"",
        f"{RESULT_DIR}/CONTCAR": b"contcar",
        f"{RESULT_DIR}/OUTCAR": b"outcar",
        f"{RESULT_DIR}/KPOINTS": b"kpoints",
        f"{FLOW_ROOT}/producer-alpha/INCAR": b"ENCUT = 520\nIVDW = 11\n",
        f"{FLOW_ROOT}/producer-delta/INCAR": b"ENCUT = 600\nIVDW = 11\n",
    }
    if include_vasprun:
        files[f"{RESULT_DIR}/vasprun.xml"] = b"vasprun"
    return files


def default_directories() -> set[str]:
    return {
        FLOW_ROOT,
        f"{FLOW_ROOT}/producer-alpha",
        f"{FLOW_ROOT}/producer-beta",
        f"{FLOW_ROOT}/producer-gamma",
        RESULT_DIR,
    }


def single_stage_submission_payload(*, custom: bool = False) -> dict:
    workflow_spec = {
        "stages": [
            {
                "stage_type": "relax",
                "theory": "pbe",
                "modifiers": ["dispersion"],
                "label": None,
                "options": {"dispersion": {"method": "dftd3"}},
            },
        ],
        "label": None,
        "recipe": "custom" if custom else "relax",
    }
    return {
        "flow_spec": {
            "workflow": "custom_workflow" if custom else "relax",
            "workflow_spec": workflow_spec,
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": "Example\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\ndirect\n0 0 0\n",
            },
        },
        "paths": {
            "run_dir": FLOW_ROOT,
            "logs_dir": LOG_ROOT,
            "stage_dirs": {},
            "result_dir": FLOW_ROOT,
            "log_out": f"{LOG_ROOT}/validation-run.out",
            "log_err": f"{LOG_ROOT}/validation-run.err",
        },
        "cluster": {"partition": "leeburton-pool", "account": "account-name"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 160, "walltime": "04:00:00"},
        "environment": {"VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std"},
        "provenance": {
            "bmd_compute": {
                "source": {
                    "git_commit": "abcdef0123456789",
                    "state": "clean",
                    "dirty": False,
                }
            },
            "execution": {"workflow_spec": workflow_spec},
        },
    }


def single_stage_files(*, custom: bool = False) -> dict[str, bytes]:
    return {
        f"{FLOW_ROOT}/submission.json": json.dumps(
            single_stage_submission_payload(custom=custom)
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.out": b"",
        f"{LOG_ROOT}/validation-run.err": b"",
        f"{FLOW_ROOT}/CONTCAR": b"contcar",
        f"{FLOW_ROOT}/OUTCAR": b"outcar",
        f"{FLOW_ROOT}/vasprun.xml": b"vasprun",
        f"{FLOW_ROOT}/INCAR": b"IVDW = 11\n",
    }


def legacy_static_submission_payload() -> dict:
    workflow_spec = {
        "stages": [
            {
                "stage_type": "static",
                "theory": "pbe",
                "modifiers": [],
                "label": None,
                "options": {},
            },
        ],
        "label": None,
        "recipe": "static",
    }
    return {
        "flow_spec": {
            "workflow": "static",
            "workflow_spec": workflow_spec,
            "structure": {
                "type": "pasted_text",
                "format": "poscar",
                "text": "Example\n1\n1 0 0\n0 1 0\n0 0 1\nX\n1\ndirect\n0 0 0\n",
            },
        },
        "paths": {
            "run_dir": FLOW_ROOT,
            "logs_dir": LOG_ROOT,
            "result_dir": FLOW_ROOT,
            "log_out": f"{LOG_ROOT}/legacy-static.out",
            "log_err": f"{LOG_ROOT}/legacy-static.err",
        },
        "cluster": {"partition": "leeburton-pool", "account": "account-name"},
        "resources": {"nodes": 1, "ntasks": 24, "mem_gb": 128, "walltime": "72:00:00"},
        "environment": {"VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std"},
    }


def legacy_static_files() -> dict[str, bytes]:
    return {
        f"{FLOW_ROOT}/submission.json": json.dumps(legacy_static_submission_payload()).encode("utf-8"),
        f"{LOG_ROOT}/legacy-static.out": b"[runner] python: 3.12.13 (main)\n",
        f"{LOG_ROOT}/legacy-static.err": b"",
        f"{FLOW_ROOT}/CONTCAR": b"contcar",
        f"{FLOW_ROOT}/OUTCAR": b"outcar",
        f"{FLOW_ROOT}/vasprun.xml": b"vasprun",
        f"{FLOW_ROOT}/INCAR": b"ENCUT = 520\n",
    }


def slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    assert command == [
        "ssh",
        "powerslurm-bmdguest",
        (
            "sacct -P -n -j 20893681 "
            "--format=JobIDRaw,JobName%30,State,Elapsed,Start,End,Partition%20,ExitCode,Timelimit%20,"
            "NodeList%80,NNodes,NCPUS,AllocCPUS,TotalCPU,CPUTimeRAW,"
            "AllocTRES%120,ReqTRES%120,MaxRSS"
        ),
    ]
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    assert kwargs["check"] is True
    return subprocess.CompletedProcess(
        command,
        0,
        stdout=(
            "20893681|validation|COMPLETED|02:48:37|2026-08-21T12:14:10|"
            "2026-08-21T15:02:47|leeburton-pool|0:0|72:00:00|node-a|1|24|24|"
            "2-15:00:00|607020|billing=24,cpu=24,mem=128G,node=1|"
            "billing=24,cpu=24,mem=128G,node=1|\n"
            "20893681.0|vasp|COMPLETED|02:47:00|2026-08-21T12:15:00|"
            "2026-08-21T15:02:00|leeburton-pool|0:0|72:00:00|node-a|1|24|24|"
            "2-14:50:00|601200|billing=24,cpu=24,mem=128G,node=1|"
            "billing=24,cpu=24,mem=128G,node=1|4G\n"
        ),
        stderr="",
    )


def fake_scientific_parser(
    local_paths: dict[str, Path],
    display_paths: dict[str, str],
    workflow_spec: dict,
) -> ScientificResult:
    assert set(local_paths) == {"contcar", "vasprun", "kpoints"}
    assert workflow_spec["stages"][3]["stage_type"] == "band_structure"
    return ScientificResult(
        source_paths=tuple(display_paths.values()),
        final_formula="Example2",
        final_energy_ev=-12.5,
        energy_per_atom_ev=-6.25,
        electronic_convergence=True,
        band_gap_ev=1.2,
        band_kpoints=310,
        bands=24,
    )


def test_inspect_run_uses_submission_stage_dirs_and_preserves_sources() -> None:
    remote = RemoteFixture(files=default_files(), directories=default_directories())

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=fake_scientific_parser,
        modifier_policies=(modifier_policy(),),
    )

    assert [stage.stage_type for stage in inspection.workflow_stages] == [
        "relax",
        "static",
        "dos",
        "band_structure",
    ]
    assert inspection.workflow_stages[1].options == {"custom_modifier": {"nested": "kept"}}
    assert inspection.workflow_stages[3].options == {"dispersion": {"method": "dftd3"}}
    assert inspection.initial_structure.status == "available"
    assert inspection.initial_structure.representation_hash is not None
    assert [item.label for item in inspection.stage_directories] == [
        "producer-alpha",
        "producer-beta",
        "producer-gamma",
        "producer-delta",
    ]
    assert all("stage_01" not in " ".join(command) for command in remote.commands)
    assert inspection.producer_git["git_commit"] == "abcdef0123456789"
    assert inspection.scheduler is not None
    assert inspection.scheduler.state == "COMPLETED"
    assert inspection.scheduler.node_list == "node-a"
    assert inspection.scheduler.node_count == 1
    assert inspection.scheduler.allocated_cpus == 24
    assert inspection.scheduler.total_cpu == "2-15:00:00"
    assert inspection.scheduler.cpu_time_raw == 607020
    assert inspection.scheduler.alloc_tres == "billing=24,cpu=24,mem=128G,node=1"
    assert inspection.scheduler.req_tres == "billing=24,cpu=24,mem=128G,node=1"
    assert inspection.scheduler.max_rss == "4G"
    assert inspection.scheduler.max_rss_source == "20893681.0"
    assert inspection.scheduler.cpu_efficiency == pytest.approx(226800 / (10117 * 24))
    assert inspection.runtime.evidence_type == LOG_OBSERVATION
    assert inspection.runtime.packages["pymatgen"] == "2026.8.13"
    assert inspection.runtime.environment["PMG_VASP_PSP_DIR"] == "/bmd-db/potcars"
    assert inspection.runtime.stage_uuids == {
        "stage_01": "64210872-5626-40c7-a7eb-79f7e49272ba",
        "custom.second": "328290de-b493-4d39-a4c7-238eb9055720",
        "final stage": "48f52369-d3b9-40b5-9a9f-a87bae6d007f",
    }
    assert inspection.scientific.evidence_type == PYMATGEN_DERIVED
    assert inspection.scientific.final_formula == "Example2"
    assert all(item.evidence_type == ARTIFACT_OBSERVATION for item in inspection.final_artifacts)
    incar_inputs = {item.label: item for item in inspection.executed_inputs}
    assert incar_inputs["producer-alpha"].evidence_type == EXECUTED_INPUT
    assert incar_inputs["producer-alpha"].stage_index == 1
    assert incar_inputs["producer-alpha"].values["IVDW"] == 11
    assert incar_inputs["producer-delta"].stage_index == 4
    assert incar_inputs["producer-beta"].present is False
    assert "result_dir" not in incar_inputs
    assert all(item.stage_index is not None for item in inspection.executed_inputs)
    assert inspection.input_expectations[0].status == "supported"


@pytest.mark.parametrize("custom", [False, True])
def test_single_stage_runs_bind_result_dir_to_stage_one(custom: bool) -> None:
    remote = RemoteFixture(files=single_stage_files(custom=custom), directories={FLOW_ROOT})

    def parser(
        local_paths: dict[str, Path],
        display_paths: dict[str, str],
        workflow_spec: dict,
    ) -> ScientificResult:
        assert set(local_paths) == {"contcar", "vasprun"}
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            final_formula="Example",
        )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        scientific_parser=parser,
        modifier_policies=(modifier_policy(),),
    )

    assert inspection.workflow_stages[0].options == {"dispersion": {"method": "dftd3"}}
    assert inspection.executed_inputs == (
        IncarObservation(
            "result_dir",
            f"{FLOW_ROOT}/INCAR",
            True,
            1,
            values={"IVDW": 11},
        ),
    )
    assert inspection.input_expectations[0].status == "supported"
    assert inspection.input_expectations[0].observed_value == 11


def test_legacy_submission_without_provenance_remains_inspectable(
    capsys: pytest.CaptureFixture[str],
) -> None:
    remote = RemoteFixture(files=legacy_static_files(), directories={FLOW_ROOT})

    def parser(
        local_paths: dict[str, Path],
        display_paths: dict[str, str],
        workflow_spec: dict,
    ) -> ScientificResult:
        assert set(local_paths) == {"contcar", "vasprun"}
        assert workflow_spec["stages"][0]["stage_type"] == "static"
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            final_formula="MgO",
            executed_parameters=(
                IncarObservation(
                    "vasprun_xml.parameters",
                    f"{FLOW_ROOT}/vasprun.xml",
                    True,
                    1,
                    source_type="vasprun_xml.parameters",
                    values={"ENCUT": 520},
                ),
            ),
        )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        scientific_parser=parser,
    )

    assert inspection.producer_git == {}
    assert inspection.workflow_stages == (
        WorkflowStage(
            1,
            "static",
            "pbe",
            (),
            None,
            {},
        ),
    )
    assert inspection.result_directory.present is True
    assert inspection.executed_inputs[0] == IncarObservation(
        "result_dir",
        f"{FLOW_ROOT}/INCAR",
        True,
        1,
        values={"ENCUT": 520},
    )
    assert inspection.executed_inputs[1].source_type == "vasprun_xml.parameters"
    assert inspection.executed_inputs[1].values["ENCUT"] == 520
    assert inspection.input_expectations == ()

    cli.print_run_inspection(inspection)
    captured = capsys.readouterr()
    assert "git commit:  unavailable" in captured.out
    assert "git state:   unavailable" in captured.out


def test_parse_oszicar_trajectory_preserves_electronic_and_ionic_evidence() -> None:
    trajectory = parse_oszicar_trajectory(OSZICAR_TWO_STEP)

    assert trajectory.ionic_steps_observed == 2
    assert trajectory.completed_ionic_steps == 2
    assert trajectory.electronic_iterations_by_ionic_step == (2, 2)
    assert trajectory.electronic_iterations_by_completed_ionic_step == (2, 2)
    assert trajectory.incomplete_electronic_iteration_count is None
    assert trajectory.final_electronic_iteration_count == 2
    assert trajectory.recent_electronic_iterations[-1].algorithm == "RMM"
    assert trajectory.recent_electronic_iterations[-1].iteration == 2
    assert trajectory.recent_electronic_iterations[-1].energy == -12.1
    assert trajectory.recent_ionic_steps[-1].step_index == 2
    assert trajectory.recent_ionic_steps[-1].free_energy == -12.1


def test_parse_oszicar_distinguishes_incomplete_first_electronic_cycle() -> None:
    trajectory = parse_oszicar_trajectory(oszicar_incomplete_cycle(25))

    assert trajectory.ionic_steps_observed == 0
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.electronic_iterations_by_ionic_step == (25,)
    assert trajectory.electronic_iterations_by_completed_ionic_step == ()
    assert trajectory.incomplete_electronic_iteration_count == 25
    assert trajectory.final_electronic_iteration_count == 25
    assert trajectory.recent_incomplete_electronic_iterations[-1].iteration == 25
    assert trajectory.recent_ionic_steps == ()


def test_diagnose_run_uses_producer_stage_dirs_for_oszicar_evidence() -> None:
    files = default_files(include_vasprun=False)
    files[f"{FLOW_ROOT}/producer-alpha/INCAR"] = b"NELM = 3\nEDIFF = 1E-6\nNSW = 1\nEDIFFG = -0.02\n"
    files[f"{FLOW_ROOT}/producer-alpha/OSZICAR"] = OSZICAR_NELM_LIMIT
    files[f"{RESULT_DIR}/OSZICAR"] = OSZICAR_TWO_STEP
    remote = RemoteFixture(files=files, directories=default_directories())

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
    )

    assert diagnosis.termination.scheduler_state == "COMPLETED"
    assert diagnosis.termination.scheduler_timelimit == "72:00:00"
    assert diagnosis.termination.scheduler_reports_timeout is False
    assert [trajectory.stage_label for trajectory in diagnosis.trajectories] == [
        "producer-alpha",
        "producer-beta",
        "producer-gamma",
        "producer-delta",
    ]
    assert "result_dir" not in [trajectory.stage_label for trajectory in diagnosis.trajectories]

    first_stage = diagnosis.trajectories[0]
    assert first_stage.evidence_type == TRAJECTORY_OBSERVATION
    assert first_stage.stage_index == 1
    assert first_stage.oszicar_present is True
    assert first_stage.criteria["NELM"] == 3
    assert first_stage.criteria["EDIFF"] == 1e-06
    assert first_stage.criteria["NSW"] == 1
    assert first_stage.criteria["EDIFFG"] == -0.02
    assert first_stage.final_electronic_iteration_count == 3
    assert first_stage.completed_ionic_steps == 1
    assert first_stage.incomplete_electronic_iteration_count is None
    assert first_stage.vasprun_present is False

    final_stage = diagnosis.trajectories[-1]
    assert final_stage.stage_index == 4
    assert final_stage.ionic_steps_observed == 2
    assert final_stage.completed_ionic_steps == 2
    assert final_stage.electronic_iterations_by_ionic_step == (2, 2)


@pytest.mark.parametrize("custom", [False, True])
def test_diagnose_single_stage_uses_result_dir_binding(custom: bool) -> None:
    files = single_stage_files(custom=custom)
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files.pop(f"{FLOW_ROOT}/vasprun.xml")
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    assert len(diagnosis.trajectories) == 1
    trajectory = diagnosis.trajectories[0]
    assert trajectory.stage_label == "result_dir"
    assert trajectory.stage_index == 1
    assert trajectory.criteria["NSW"] == 2
    assert trajectory.ionic_steps_observed == 2
    assert trajectory.completed_ionic_steps == 2


def test_diagnose_vasprun_size_cap_skips_large_transfer() -> None:
    files = single_stage_files()
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/vasprun.xml"] = b"x" * 20
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
        max_vasprun_bytes=5,
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_error is None
    assert trajectory.vasprun_skipped_reason is not None
    assert "exceeds limit 5" in trajectory.vasprun_skipped_reason
    assert all(
        "cat -- /bmd-db/guest/flows/validation-run/vasprun.xml" not in command
        for command in (" ".join(item) for item in remote.commands)
    )


def test_diagnose_vasprun_enriches_force_and_convergence_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrajectoryVasprun:
        parameters = {"NELM": 60, "EDIFF": 1e-06, "NSW": 2, "EDIFFG": -0.01}
        converged_electronic = True
        converged_ionic = False
        ionic_steps = (
            {"forces": ((3.0, 4.0, 0.0), (0.0, 0.0, 1.0))},
            {"forces": ((0.0, 0.3, 0.4),)},
        )

    monkeypatch.setattr(run_resource, "_load_vasprun", lambda *args, **kwargs: FakeTrajectoryVasprun())

    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 60\nEDIFF = 1E-6\nNSW = 2\nEDIFFG = -0.01\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = OSZICAR_TWO_STEP
    files[f"{FLOW_ROOT}/vasprun.xml"] = b"<modeling/>"
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_ionic_steps == 2
    assert trajectory.converged_electronic is True
    assert trajectory.converged_ionic is False
    assert trajectory.recent_ionic_steps[-1].max_force == 0.5
    assert trajectory.criteria["NELM"] == 60
    assert trajectory.criteria_discrepancies == ()


def test_diagnose_malformed_vasprun_keeps_oszicar_evidence_and_quiet_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def malformed_loader(*args: object, **kwargs: object) -> object:
        warnings.warn(
            "XML is malformed. Parsing has stopped but partial data is available.",
            UserWarning,
        )
        raise IndexError("list index out of range")

    monkeypatch.setattr(run_resource, "_load_vasprun", malformed_loader)

    files = single_stage_files()
    files[f"{FLOW_ROOT}/INCAR"] = b"NELM = 200\nEDIFF = 1E-6\n"
    files[f"{FLOW_ROOT}/OSZICAR"] = oszicar_incomplete_cycle(25)
    files[f"{FLOW_ROOT}/vasprun.xml"] = b"<modeling><calculation>"
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )

    trajectory = diagnosis.trajectories[0]
    assert trajectory.oszicar_present is True
    assert trajectory.completed_ionic_steps == 0
    assert trajectory.electronic_iterations_by_completed_ionic_step == ()
    assert trajectory.incomplete_electronic_iteration_count == 25
    assert trajectory.recent_incomplete_electronic_iterations[-1].iteration == 25
    assert trajectory.vasprun_present is True
    assert trajectory.vasprun_error == "file could not be parsed completely"
    assert "list index out of range" not in " ".join(trajectory.unavailable)
    assert "XML is malformed" not in " ".join(trajectory.unavailable)

    cli.print_run_diagnosis(diagnosis)

    captured = capsys.readouterr()
    assert captured.err == ""
    assert "completed ionic steps: 0" in captured.out
    assert "electronic iterations for completed ionic steps: none" in captured.out
    assert "incomplete electronic cycle: 25 iterations observed / NELM 200" in captured.out
    assert "vasprun trajectory enrichment unavailable: file could not be parsed completely" in captured.out
    assert "list index out of range" not in captured.out
    assert "XML is malformed" not in captured.out


def test_diagnose_scheduler_timeout_is_evidence_not_prediction() -> None:
    files = default_files(include_vasprun=False)
    remote = RemoteFixture(files=files, directories=default_directories())

    def timeout_slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "20893681|validation|TIMEOUT|72:00:00|2026-08-21T12:14:10|"
                "2026-08-24T12:14:10|leeburton-pool|0:0|72:00:00\n"
            ),
            stderr="",
        )

    diagnosis = diagnose_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=timeout_slurm_runner,
    )

    assert diagnosis.termination.scheduler_state == "TIMEOUT"
    assert diagnosis.termination.scheduler_reports_timeout is True


def test_diagnose_rejects_producer_paths_outside_allowed_roots() -> None:
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(
            submission_payload(outside_path=True)
        ).encode("utf-8")
    }
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    with pytest.raises(RemotePathError, match="paths.stage_dirs.producer-beta"):
        diagnose_remote_run(
            cluster(),
            FLOW_ROOT,
            remote_runner=remote,
            slurm_runner=slurm_runner,
        )


def test_producer_supplied_paths_outside_allowed_roots_are_rejected() -> None:
    files = {
        f"{FLOW_ROOT}/submission.json": json.dumps(
            submission_payload(outside_path=True)
        ).encode("utf-8")
    }
    remote = RemoteFixture(files=files, directories={FLOW_ROOT})

    with pytest.raises(RemotePathError, match="paths.stage_dirs.producer-beta"):
        inspect_remote_run(
            cluster(),
            FLOW_ROOT,
            remote_runner=remote,
            slurm_runner=slurm_runner,
            scientific_parser=fake_scientific_parser,
        )

    assert len(remote.commands) == 1


def test_missing_artifacts_are_reported_not_invented() -> None:
    remote = RemoteFixture(
        files=default_files(include_vasprun=False),
        directories=default_directories(),
    )

    def parser_reports_missing_vasprun(
        local_paths: dict[str, Path],
        display_paths: dict[str, str],
        workflow_spec: dict,
    ) -> ScientificResult:
        assert set(local_paths) == {"contcar", "kpoints"}
        return ScientificResult(
            source_paths=tuple(display_paths.values()),
            final_formula="Example2",
            unavailable=("vasprun.xml is unavailable",),
        )

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=parser_reports_missing_vasprun,
    )

    artifacts = {item.label: item for item in inspection.final_artifacts}
    assert artifacts["vasprun"].present is False
    assert inspection.scientific.final_formula == "Example2"
    assert "vasprun.xml is unavailable" in inspection.scientific.unavailable


def test_invalid_job_id_is_not_sent_to_scheduler() -> None:
    remote = RemoteFixture(
        files=default_files(invalid_job_id=True),
        directories=default_directories(),
    )
    slurm_calls = 0

    def unused_slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal slurm_calls
        slurm_calls += 1
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=unused_slurm_runner,
        scientific_parser=fake_scientific_parser,
    )

    assert slurm_calls == 0
    assert inspection.job_id is None
    assert inspection.scheduler_error == "No valid SLURM job ID was found in inspected producer artifacts."


def test_run_inspector_source_has_no_validation_run_specific_logic() -> None:
    source = Path("src/bmd_agent/resources/run.py").read_text(encoding="utf-8").lower()

    assert "hse06" not in source
    assert "sns2" not in source
    assert "dftd3" not in source
    assert "stage_03" not in source


class FakeComposition:
    formula = "X2"
    reduced_formula = "X2"


class FakeLattice:
    a = 3.0
    b = 4.0
    c = 6.0
    alpha = 90.0
    beta = 91.0
    gamma = 120.0


class FakeStructure:
    composition = FakeComposition()
    lattice = FakeLattice()
    volume = 62.5
    density = 4.2

    def __len__(self) -> int:
        return 2


class FakeBandStructure:
    kpoints = [object(), object(), object(), object()]
    bands = {
        "up": [
            [-2.0, -1.9, -1.8, -1.7],
            [0.4, 0.5, 0.6, 0.7],
        ]
    }

    def get_band_gap(self) -> dict[str, float]:
        return {"energy": 1.25}


class FakeBandVasprun:
    calls: list[dict[str, object]] = []
    band_calls: list[dict[str, object]] = []

    def __init__(self, path: str, **kwargs: object) -> None:
        self.calls.append({"path": path, "kwargs": kwargs})
        self.final_structure = FakeStructure()
        self.final_energy = -4.0
        self.converged_electronic = True
        self.converged = True
        self.incar = {"IVDW": 11}
        self.parameters = {"ENCUT": 520}
        self.efermi = None if kwargs.get("parse_dos") is False else 0.3

    def get_band_structure(self, **kwargs: object) -> FakeBandStructure:
        self.band_calls.append(kwargs)
        if self.efermi is None:
            raise ValueError("e_fermi is None.")
        return FakeBandStructure()


class FakeFailingBandVasprun(FakeBandVasprun):
    def get_band_structure(self, **kwargs: object) -> FakeBandStructure:
        self.band_calls.append(kwargs)
        raise ValueError("e_fermi is None.")


@contextmanager
def fake_pymatgen_modules(vasprun_cls: type[FakeBandVasprun]):
    modules = {
        "pymatgen": types.ModuleType("pymatgen"),
        "pymatgen.core": types.ModuleType("pymatgen.core"),
        "pymatgen.io": types.ModuleType("pymatgen.io"),
        "pymatgen.io.vasp": types.ModuleType("pymatgen.io.vasp"),
        "pymatgen.io.vasp.outputs": types.ModuleType("pymatgen.io.vasp.outputs"),
    }
    modules["pymatgen.core"].Structure = types.SimpleNamespace(
        from_file=lambda path: FakeStructure(),
    )
    modules["pymatgen.io.vasp.outputs"].Vasprun = vasprun_cls

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def fake_vasp_paths(tmp_path: Path) -> tuple[dict[str, Path], dict[str, str]]:
    local_paths = {
        "contcar": tmp_path / "CONTCAR",
        "vasprun": tmp_path / "vasprun.xml",
        "kpoints": tmp_path / "KPOINTS",
    }
    display_paths = {
        "contcar": "/remote/result/CONTCAR",
        "vasprun": "/remote/result/vasprun.xml",
        "kpoints": "/remote/result/KPOINTS",
    }
    return local_paths, display_paths


def arbitrary_band_workflow_spec() -> dict[str, object]:
    return {
        "stages": [
            {"stage_type": "prepare", "theory": "example"},
            {"stage_type": "screen", "theory": "example"},
            {"stage_type": "static", "theory": "example"},
            {"stage_type": "checkpoint", "theory": "example"},
            {"stage_type": "band_structure", "theory": "example"},
        ]
    }


def test_band_parsing_keeps_final_vasprun_fermi_reference(tmp_path: Path) -> None:
    FakeBandVasprun.calls = []
    FakeBandVasprun.band_calls = []
    local_paths, display_paths = fake_vasp_paths(tmp_path)

    with fake_pymatgen_modules(FakeBandVasprun):
        result = parse_vasp_output_files(
            local_paths,
            display_paths,
            arbitrary_band_workflow_spec(),
        )

    assert FakeBandVasprun.calls
    vasprun_kwargs = FakeBandVasprun.calls[-1]["kwargs"]
    assert "parse_dos" not in vasprun_kwargs
    assert "parse_eigenvalues" not in vasprun_kwargs
    assert FakeBandVasprun.band_calls[-1]["line_mode"] is True
    assert FakeBandVasprun.band_calls[-1]["kpoints_filename"].endswith("KPOINTS")
    assert result.error is None
    assert result.final_formula == "X2"
    assert result.final_energy_ev == -4.0
    assert result.energy_per_atom_ev == -2.0
    assert result.electronic_convergence is True
    assert result.band_gap_ev == 1.25
    assert result.band_kpoints == 4
    assert result.bands == 2
    assert result.structure is not None
    assert result.structure.formula == "X2"
    assert result.structure.reduced_formula == "X2"
    assert result.structure.site_count == 2
    assert result.structure.lattice_a == 3.0
    assert result.structure.lattice_b == 4.0
    assert result.structure.lattice_c == 6.0
    assert result.structure.alpha == 90.0
    assert result.structure.beta == 91.0
    assert result.structure.gamma == 120.0
    assert result.structure.volume == 62.5
    assert result.structure.density == 4.2
    assert result.structure.c_over_a == 2.0
    assert result.executed_parameters[0].source_type == "vasprun_xml.incar"
    assert result.executed_parameters[0].stage_index == 5
    assert result.executed_parameters[0].values["IVDW"] == 11
    assert result.executed_parameters[1].source_type == "vasprun_xml.parameters"
    assert result.executed_parameters[1].values["ENCUT"] == 520


def test_scalar_observations_survive_band_structure_failure(tmp_path: Path) -> None:
    FakeFailingBandVasprun.calls = []
    FakeFailingBandVasprun.band_calls = []
    local_paths, display_paths = fake_vasp_paths(tmp_path)

    with fake_pymatgen_modules(FakeFailingBandVasprun):
        result = parse_vasp_output_files(
            local_paths,
            display_paths,
            arbitrary_band_workflow_spec(),
        )

    assert result.error is None
    assert result.final_formula == "X2"
    assert result.final_energy_ev == -4.0
    assert result.energy_per_atom_ev == -2.0
    assert result.electronic_convergence is True
    assert result.band_gap_ev is None
    assert result.band_kpoints is None
    assert result.bands is None
    assert "band structure could not be derived: e_fermi is None." in result.unavailable


def modifier_policy() -> dict:
    return {
        "modifier": "dispersion",
        "option_key": "dispersion",
        "method_key": "method",
        "methods": [
            {"value": "dftd3", "label": "DFT-D3", "incar_effect": {"IVDW": 11}},
            {"value": "dftd3-bj", "label": "DFT-D3(BJ)", "incar_effect": {"IVDW": 12}},
        ],
    }


def test_parse_incar_observes_absent_and_selected_ivdw_values() -> None:
    no_dispersion, error = parse_incar_contents("ENCUT = 520\n")
    assert error is None
    assert "IVDW" not in no_dispersion

    d3, error = parse_incar_contents("IVDW = 11\n")
    assert error is None
    assert d3["IVDW"] == 11

    d3bj, error = parse_incar_contents("IVDW = 12\n")
    assert error is None
    assert d3bj["IVDW"] == 12


def test_requested_options_are_compared_to_executed_input_by_policy() -> None:
    stages = (
        WorkflowStage(
            1,
            "relax",
            "pbe",
            ("dispersion",),
            None,
            {"dispersion": {"method": "dftd3"}},
        ),
    )
    executed = (
        IncarObservation(
            "producer-alpha",
            f"{FLOW_ROOT}/producer-alpha/INCAR",
            True,
            1,
            values={"IVDW": 12},
        ),
    )

    observations = compare_requested_options_to_executed_inputs(
        stages,
        executed,
        (modifier_policy(),),
    )

    assert observations == (
        InputExpectationObservation(
            stage_label="producer-alpha",
            stage_index=1,
            option_path="dispersion.method",
            requested_value="dftd3",
            input_key="IVDW",
            expected_value=11,
            observed_value=12,
            status="discrepancy",
            source_values={"retained_incar:producer-alpha": 12},
            reason="executed value differs from producer-requested option effect",
        ),
    )


def test_requested_check_statuses_distinguish_unavailable_absent_supported_and_disagreement() -> None:
    stages = (
        WorkflowStage(
            1,
            "relax",
            "pbe",
            ("dispersion",),
            None,
            {"dispersion": {"method": "dftd3"}},
        ),
    )

    unavailable = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                False,
                1,
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert unavailable.status == "unavailable"
    assert unavailable.reason == "no readable executed-input evidence was bound to this stage"

    absent = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                True,
                1,
                values={"ENCUT": 520},
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert absent.status == "absent"
    assert absent.source_values == {"retained_incar:result_dir": None}

    supported = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                False,
                1,
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/vasprun.xml",
                True,
                1,
                source_type="vasprun_xml.incar",
                values={"IVDW": 11},
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert supported.status == "supported"
    assert supported.observed_value == 11
    assert supported.source_values == {"vasprun_xml.incar:vasprun_xml.incar": 11}

    disagreement = compare_requested_options_to_executed_inputs(
        stages,
        (
            IncarObservation(
                "result_dir",
                f"{FLOW_ROOT}/INCAR",
                True,
                1,
                values={"IVDW": 11},
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/vasprun.xml",
                True,
                1,
                source_type="vasprun_xml.incar",
                values={"IVDW": 12},
            ),
        ),
        (modifier_policy(),),
    )[0]
    assert disagreement.status == "discrepancy"
    assert disagreement.source_values == {
        "retained_incar:result_dir": 11,
        "vasprun_xml.incar:vasprun_xml.incar": 12,
    }
    assert disagreement.reason == "executed-input sources disagree about parameter value"


def test_vasp_reported_parameter_observations_are_generic() -> None:
    vasprun = types.SimpleNamespace(
        incar={"IVDW": 11},
        parameters={"ENCUT": 520},
    )

    observations = vasp_reported_parameter_observations(
        vasprun,
        source_path="/remote/result/vasprun.xml",
        stage_index=1,
    )

    assert [observation.source_type for observation in observations] == [
        "vasprun_xml.incar",
        "vasprun_xml.parameters",
    ]
    assert observations[0].values["IVDW"] == 11
    assert observations[1].values["ENCUT"] == 520


def comparison_inspection(
    flow_root: str,
    *,
    a: float | None,
    b: float | None,
    c: float | None,
    volume: float | None,
    density: float | None,
    energy_per_atom: float | None,
    band_gap: float | None,
    modifiers: tuple[str, ...] = (),
    options: dict | None = None,
) -> RunInspection:
    structure = None
    if a is not None:
        structure = StructureObservation(
            source_path=f"{flow_root}/result/CONTCAR",
            formula="X2",
            reduced_formula="X2",
            site_count=2,
            lattice_a=a,
            lattice_b=b,
            lattice_c=c,
            alpha=90.0,
            beta=90.0,
            gamma=120.0,
            volume=volume,
            density=density,
            c_over_a=c / a if a and c is not None else None,
        )

    return RunInspection(
        flow_root=flow_root,
        submission_path=f"{flow_root}/submission.json",
        workflow_stages=(
            WorkflowStage(1, "relax", "pbe", modifiers, None, options or {}),
        ),
        stage_directories=(),
        result_directory=PathObservation("result_dir", f"{flow_root}/result", "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id="1",
        scheduler=SlurmAccountingRecord(
            job_id="1",
            name="comparison",
            state="COMPLETED",
            elapsed="00:01:00",
            start="2026-08-25T00:00:00",
            end="2026-08-25T00:01:00",
            partition="leeburton-pool",
            exit_code="0:0",
        ),
        scheduler_error=None,
        runtime=LogRuntimeObservation(LOG_OBSERVATION, (), None, {}, {}, {}),
        scientific=ScientificResult(
            source_paths=(),
            final_formula="X2",
            energy_per_atom_ev=energy_per_atom,
            band_gap_ev=band_gap,
            structure=structure,
        ),
        comparison=ComparisonObservation("unavailable", "producer_provenance"),
        initial_structure=InitialStructureObservation(
            status="available",
            representation_type="pasted_text:poscar",
            representation_hash="same-hash",
        ),
    )


def test_run_labels_are_derived_from_provenance_and_modifier_policy() -> None:
    control = comparison_inspection("/flow/control", a=3, b=3, c=6, volume=54, density=4, energy_per_atom=-1, band_gap=1)
    corrected = comparison_inspection(
        "/flow/corrected",
        a=3,
        b=3,
        c=6,
        volume=54,
        density=4,
        energy_per_atom=-1,
        band_gap=1,
        modifiers=("dispersion",),
        options={"dispersion": {"method": "dftd3-bj"}},
    )

    assert run_label_from_provenance(control, modifier_policies=(modifier_policy(),)) == "PBE"
    assert run_label_from_provenance(corrected, modifier_policies=(modifier_policy(),)) == "PBE + DFT-D3(BJ)"


def test_build_run_comparison_uses_first_run_as_baseline_and_handles_three_runs() -> None:
    baseline = comparison_inspection("/flow/baseline", a=3.0, b=4.0, c=6.0, volume=72.0, density=4.0, energy_per_atom=-4.0, band_gap=2.0)
    second = comparison_inspection("/flow/second", a=3.0, b=4.1, c=5.4, volume=66.42, density=4.2, energy_per_atom=-4.2, band_gap=1.8)
    third = comparison_inspection("/flow/third", a=2.9, b=4.0, c=5.1, volume=59.16, density=4.4, energy_per_atom=-4.5, band_gap=1.5)

    comparison = build_run_comparison(
        (baseline, second, third),
        modifier_policies=(modifier_policy(),),
    )

    assert comparison.initial_structure.status == "match"
    assert len({quantity.flow_root for quantity in comparison.quantities}) == 2
    c_quantity = next(
        quantity
        for quantity in comparison.quantities
        if quantity.flow_root == "/flow/second" and quantity.quantity == "lattice_c"
    )
    assert c_quantity.baseline_value == 6.0
    assert c_quantity.comparison_value == 5.4
    assert c_quantity.delta == -0.6
    assert c_quantity.percent_delta == -10.0


def test_comparison_reports_missing_values_and_energy_warning_for_different_configs() -> None:
    baseline = comparison_inspection("/flow/baseline", a=3.0, b=4.0, c=6.0, volume=72.0, density=4.0, energy_per_atom=-4.0, band_gap=2.0)
    comparison_run = comparison_inspection(
        "/flow/missing",
        a=None,
        b=None,
        c=None,
        volume=None,
        density=None,
        energy_per_atom=-4.2,
        band_gap=None,
        modifiers=("dispersion",),
        options={"dispersion": {"method": "dftd3"}},
    )

    comparison = build_run_comparison(
        (baseline, comparison_run),
        modifier_policies=(modifier_policy(),),
    )

    lattice_a = next(quantity for quantity in comparison.quantities if quantity.quantity == "lattice_a")
    band_gap = next(quantity for quantity in comparison.quantities if quantity.quantity == "band_gap_ev")
    assert lattice_a.status == "unavailable"
    assert lattice_a.reason == "comparison value unavailable"
    assert band_gap.status == "unavailable"
    assert comparison.energy_warning is not None
    assert "not a ranking of method quality" in comparison.energy_warning


def test_compare_remote_runs_requires_at_least_two_roots() -> None:
    with pytest.raises(RunInspectionError, match="at least two"):
        compare_remote_runs(cluster(), [FLOW_ROOT])


def cli_run_inspection(
    *,
    workflow_stages: tuple[WorkflowStage, ...],
    executed_inputs: tuple[IncarObservation, ...],
    input_expectations: tuple[InputExpectationObservation, ...] = (),
    scheduler: SlurmAccountingRecord | None = None,
    scheduler_error: str | None = None,
    cluster_request: dict[str, object] | None = None,
    resources_request: dict[str, object] | None = None,
    environment_policy: dict[str, object] | None = None,
) -> RunInspection:
    return RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=workflow_stages,
        stage_directories=(),
        result_directory=PathObservation("result_dir", RESULT_DIR, "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={},
        cluster_request=cluster_request or {},
        resources_request=resources_request or {},
        environment_policy=environment_policy or {},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id=scheduler.job_id if scheduler else None,
        scheduler=scheduler,
        scheduler_error=scheduler_error,
        runtime=LogRuntimeObservation(LOG_OBSERVATION, (), None, {}, {}, {}),
        scientific=ScientificResult(source_paths=()),
        comparison=ComparisonObservation("unavailable", "producer_provenance"),
        executed_inputs=executed_inputs,
        input_expectations=input_expectations,
    )


def test_cli_executed_input_summary_reports_present_settings_without_optional_absences(
    capsys: pytest.CaptureFixture[str],
) -> None:
    magmom = [1.0, 0.0, 0.0] * 30
    inspection = cli_run_inspection(
        workflow_stages=(
            WorkflowStage(1, "static", "pbe", ("hubbard",), None),
            WorkflowStage(2, "static", "pbe", ("hubbard", "soc"), None),
        ),
        executed_inputs=(
            IncarObservation(
                "stage_01",
                f"{FLOW_ROOT}/stage_01/INCAR",
                True,
                1,
                values={
                    "ENCUT": 520,
                    "LDAU": True,
                    "LDAUTYPE": 2,
                    "LDAUL": [2, -1],
                    "LDAUU": [5.3, 0],
                    "LMAXMIX": 4,
                },
            ),
            IncarObservation(
                "stage_02",
                f"{FLOW_ROOT}/stage_02/INCAR",
                True,
                2,
                values={
                    "ENCUT": 520,
                    "LDAU": True,
                    "LDAUU": [5.3, 0],
                    "LSORBIT": True,
                    "LNONCOLLINEAR": True,
                    "SAXIS": [0, 0, 1],
                    "MAGMOM": magmom,
                },
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/stage_02/vasprun.xml",
                True,
                2,
                source_type="vasprun_xml.incar",
                values={
                    "ENCUT": 520,
                    "LSORBIT": True,
                    "LNONCOLLINEAR": True,
                    "SAXIS": [0, 0, 1],
                    "MAGMOM": magmom,
                },
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "stage 1: PBE Static Energy (stage_01)" in captured.out
    assert "stage 2: PBE Static Energy (stage_02)" in captured.out
    assert "LDAU=true" in captured.out
    assert "LDAUU=[5.3, 0]" in captured.out
    assert "LSORBIT=true" in captured.out
    assert "SAXIS=[0, 0, 1]" in captured.out
    assert "MAGMOM=present, 30 sites / 90 noncollinear components" in captured.out
    assert captured.out.count("LSORBIT=true") == 1
    assert "IVDW absent" not in captured.out


def test_cli_executed_input_summary_reports_source_disagreement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inspection = cli_run_inspection(
        workflow_stages=(WorkflowStage(1, "static", "pbe", (), None),),
        executed_inputs=(
            IncarObservation(
                "stage_01",
                f"{FLOW_ROOT}/stage_01/INCAR",
                True,
                1,
                values={"ENCUT": 520},
            ),
            IncarObservation(
                "vasprun_xml.incar",
                f"{FLOW_ROOT}/stage_01/vasprun.xml",
                True,
                1,
                source_type="vasprun_xml.incar",
                values={"ENCUT": 400},
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "discrepancy: ENCUT" in captured.out
    assert "retained_incar:stage_01=520" in captured.out
    assert "vasprun_xml.incar:vasprun_xml.incar=400" in captured.out


def test_cli_inspect_run_summary_is_evidence_oriented(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster()})
    inspection = RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=(
            WorkflowStage(1, "relax", "pbe", (), None),
            WorkflowStage(2, "static", "r2scan", (), None),
        ),
        stage_directories=(
            PathObservation("producer-alpha", f"{FLOW_ROOT}/producer-alpha", "directory", True, ARTIFACT_OBSERVATION),
        ),
        result_directory=PathObservation("result_dir", RESULT_DIR, "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(
            PathObservation("log_out", f"{LOG_ROOT}/validation-run.out", "file", True, ARTIFACT_OBSERVATION),
        ),
        final_artifacts=(
            PathObservation("vasprun", f"{RESULT_DIR}/vasprun.xml", "file", False, ARTIFACT_OBSERVATION),
        ),
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        cluster_request={"partition": "leeburton-pool"},
        resources_request={"ntasks": 24},
        environment_policy={"PMG_VASP_PSP_DIR": "/bmd-db/potcars"},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id="20893681",
        scheduler=SlurmAccountingRecord(
            job_id="20893681",
            name="validation",
            state="COMPLETED",
            elapsed="02:48:37",
            start="2026-08-21T12:14:10",
            end="2026-08-21T15:02:47",
            partition="leeburton-pool",
            exit_code="0:0",
        ),
        scheduler_error=None,
        runtime=LogRuntimeObservation(
            evidence_type=LOG_OBSERVATION,
            sources=(f"{LOG_ROOT}/validation-run.out",),
            python="3.12.13",
            packages={"pymatgen": "2026.8.13"},
            environment={},
            stage_uuids={},
        ),
        scientific=ScientificResult(
            source_paths=(f"{RESULT_DIR}/vasprun.xml",),
            unavailable=("vasprun.xml is unavailable",),
        ),
        comparison=ComparisonObservation(
            status="unavailable",
            evidence_type="producer_provenance",
            reason="No durable producer result.",
        ),
    )

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "inspect_remote_run", lambda cluster, flow_root, **kwargs: inspection)

    exit_code = cli.main(["inspect-run", FLOW_ROOT])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Producer provenance (producer_provenance)" in captured.out
    assert "Scheduler (scheduler_observation)" in captured.out
    assert "Logs (log_observation)" in captured.out
    assert "Evidence paths (artifact_observation)" in captured.out
    assert "Independent parsing (pymatgen_derived)" in captured.out


def test_cli_diagnose_run_summary_is_descriptive_not_predictive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    scheduler = SlurmAccountingRecord(
        job_id="21101222",
        name="validation",
        state="TIMEOUT",
        elapsed="01:00:20",
        start="2026-08-27T22:25:18",
        end="2026-08-27T23:25:38",
        partition="leeburton-pool",
        exit_code="0:0",
        timelimit="01:00:00",
        node_list="node-b",
        node_count=1,
        allocated_cpus=24,
        total_cpu="20:00:00",
        total_cpu_seconds=72000,
        cpu_time_raw=86976,
        alloc_tres="billing=24,cpu=24,mem=128G,node=1",
        req_tres="billing=24,cpu=24,mem=128G,node=1",
        max_rss="4G",
        max_rss_source="21101222.0",
        cpu_efficiency=72000 / (3620 * 24),
    )
    inspection = cli_run_inspection(
        workflow_stages=(WorkflowStage(1, "relax", "pbe", (), None),),
        executed_inputs=(),
        scheduler=scheduler,
        cluster_request={"partition": "leeburton-pool"},
        resources_request={"nodes": 1, "ntasks": 24, "walltime": "01:00:00"},
        environment_policy={"VASP_CMD": "mpirun -n $SLURM_NTASKS vasp_std"},
    )
    diagnosis = RunDiagnosis(
        inspection=inspection,
        termination=TerminationObservation(
            scheduler_state="TIMEOUT",
            scheduler_exit_code="0:0",
            scheduler_elapsed="72:00:00",
            scheduler_timelimit="72:00:00",
            scheduler_reports_timeout=True,
            unavailable=("VASP normal-completion marker is unavailable in diagnose-run v1",),
        ),
        trajectories=(
            StageTrajectoryObservation(
                stage_index=1,
                stage_label="result_dir",
                stage_type="relax",
                theory="pbe",
                directory=FLOW_ROOT,
                oszicar_path=f"{FLOW_ROOT}/OSZICAR",
                oszicar_present=True,
                vasprun_path=f"{FLOW_ROOT}/vasprun.xml",
                vasprun_present=True,
                criteria={"NELM": 60, "EDIFFG": -0.01},
                ionic_steps_observed=2,
                electronic_iterations_by_ionic_step=(4, 60),
                final_electronic_iteration_count=60,
                recent_electronic_iterations=(
                    run_resource.ElectronicIterationObservation(
                        iteration=60,
                        algorithm="RMM",
                        energy=-12.1,
                        dE=-0.001,
                        deps=-0.0001,
                        rms=0.02,
                        rms_c=0.01,
                    ),
                ),
                recent_ionic_steps=(
                    run_resource.IonicStepObservation(
                        step_index=2,
                        electronic_iterations=60,
                        free_energy=-12.1,
                        energy_zero=-12.05,
                        dE=-0.1,
                        max_force=0.5,
                    ),
                ),
                vasprun_ionic_steps=2,
                converged_electronic=False,
                converged_ionic=False,
            ),
        ),
    )

    cli.print_run_diagnosis(diagnosis)

    captured = capsys.readouterr()
    assert "Requested execution (producer_provenance)" in captured.out
    assert "walltime: 01:00:00" in captured.out
    assert "Scheduler execution (scheduler_observation)" in captured.out
    assert "node list: node-b" in captured.out
    assert "allocated CPUs: 24" in captured.out
    assert "total CPU: 20:00:00" in captured.out
    assert "CPUTimeRAW: 86976" in captured.out
    assert "CPU efficiency: 82.9% (scheduler-derived utilization, not scientific efficiency)" in captured.out
    assert "MaxRSS: 4G (21101222.0)" in captured.out
    assert "AllocTRES: billing=24,cpu=24,mem=128G,node=1" in captured.out
    assert "ReqTRES: billing=24,cpu=24,mem=128G,node=1" in captured.out
    assert "Termination evidence (termination_observation)" in captured.out
    assert "scheduler reports timeout: True" in captured.out
    assert "Trajectory evidence (trajectory_observation)" in captured.out
    assert "EDIFFG: -0.01 eV/A force criterion" in captured.out
    assert "RMM N=60 E=-12.1 dE=-0.001 deps=-0.0001 rms=0.02 rms(c)=0.01" in captured.out
    assert "step 2 electronic_iterations=60 F=-12.1 E0=-12.05 dE=-0.1 max_force=0.5" in captured.out
    forbidden = ("likely to benefit", "more walltime alone", "stalled", "oscillating", "diverging")
    assert all(term not in captured.out.lower() for term in forbidden)


def test_cli_executed_input_wording_distinguishes_unavailable_and_absent(
    capsys: pytest.CaptureFixture[str],
) -> None:
    inspection = RunInspection(
        flow_root=FLOW_ROOT,
        submission_path=f"{FLOW_ROOT}/submission.json",
        workflow_stages=(
            WorkflowStage(1, "relax", "pbe", ("dispersion",), None, {"dispersion": {"method": "dftd3"}}),
        ),
        stage_directories=(),
        result_directory=PathObservation("result_dir", FLOW_ROOT, "directory", True, ARTIFACT_OBSERVATION),
        log_paths=(),
        final_artifacts=(),
        producer_git={},
        cluster_request={},
        resources_request={},
        environment_policy={},
        attempt_state=AttemptStateObservation(path=None, present=False),
        job_id=None,
        scheduler=None,
        scheduler_error=None,
        runtime=LogRuntimeObservation(LOG_OBSERVATION, (), None, {}, {}, {}),
        scientific=ScientificResult(source_paths=()),
        comparison=ComparisonObservation("unavailable", "producer_provenance"),
        executed_inputs=(
            IncarObservation("missing_result", f"{FLOW_ROOT}/INCAR", False, 1),
            IncarObservation("parsed_result", f"{FLOW_ROOT}/INCAR", True, 1, values={"ENCUT": 520}),
        ),
        input_expectations=(
            InputExpectationObservation(
                stage_label="missing_result",
                stage_index=1,
                option_path="dispersion.method",
                requested_value="dftd3",
                input_key="IVDW",
                expected_value=11,
                observed_value=None,
                status="unavailable",
                reason="no readable executed-input evidence was bound to this stage",
            ),
            InputExpectationObservation(
                stage_label="parsed_result",
                stage_index=1,
                option_path="dispersion.method",
                requested_value="dftd3",
                input_key="IVDW",
                expected_value=11,
                observed_value=None,
                status="absent",
                source_values={"retained_incar:parsed_result": None},
                reason="IVDW was absent from readable executed-input evidence",
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "IVDW unavailable, expected 11: unavailable" in captured.out
    assert "IVDW absent, expected 11: absent" in captured.out
    assert "observed absent" not in captured.out
    assert "present (retained_incar, /bmd-db/guest/flows/validation-run/INCAR); IVDW absent" not in captured.out


def test_cli_requested_executed_magmom_discrepancy_remains_visible(
    capsys: pytest.CaptureFixture[str],
) -> None:
    expected_magmom = [1.0, 0.0, 0.0] * 30
    observed_magmom = [0.0, 1.0, 0.0] * 30
    inspection = cli_run_inspection(
        workflow_stages=(WorkflowStage(1, "static", "pbe", ("soc",), None),),
        executed_inputs=(
            IncarObservation(
                "stage_01",
                f"{FLOW_ROOT}/stage_01/INCAR",
                True,
                1,
                values={
                    "LSORBIT": True,
                    "LNONCOLLINEAR": True,
                    "MAGMOM": observed_magmom,
                },
            ),
        ),
        input_expectations=(
            InputExpectationObservation(
                stage_label="stage_01",
                stage_index=1,
                option_path="magnetism.magmom",
                requested_value="noncollinear",
                input_key="MAGMOM",
                expected_value=expected_magmom,
                observed_value=observed_magmom,
                status="discrepancy",
                source_values={"retained_incar:stage_01": observed_magmom},
                reason="executed value differs from producer-requested option effect",
            ),
        ),
    )

    cli.print_run_inspection(inspection)

    captured = capsys.readouterr()
    assert "MAGMOM=present, 30 sites / 90 noncollinear components" in captured.out
    assert "MAGMOM observed present, 90 sites" in captured.out
    assert "expected present, 90 sites" in captured.out
    assert "discrepancy" in captured.out
    assert "retained_incar:stage_01: present, 90 sites, fingerprint" in captured.out
