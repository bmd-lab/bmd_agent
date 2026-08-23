import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess

import pytest

from bmd_agent import cli
from bmd_agent.config import ResourceRegistry, SlurmClusterResource
from bmd_agent.resources.run import (
    ARTIFACT_OBSERVATION,
    LOG_OBSERVATION,
    PYMATGEN_DERIVED,
    AttemptStateObservation,
    ComparisonObservation,
    LogRuntimeObservation,
    PathObservation,
    RunInspection,
    ScientificResult,
    WorkflowStage,
    inspect_remote_run,
)
from bmd_agent.resources.slurm import SlurmAccountingRecord
from bmd_agent.resources.vasp import RemotePathError


FLOW_ROOT = "/bmd-db/guest/flows/validation-run"
LOG_ROOT = "/bmd-db/guest/logs"
RESULT_DIR = f"{FLOW_ROOT}/producer-delta"


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
            {"stage_type": "static", "theory": "r2scan", "modifiers": [], "label": None, "options": {}},
            {"stage_type": "dos", "theory": "pbe", "modifiers": [], "label": None, "options": {}},
            {
                "stage_type": "band_structure",
                "theory": "pbe",
                "modifiers": ["spin_polarized"],
                "label": None,
                "options": {},
            },
        ],
        "label": None,
        "recipe": "custom",
    }
    return {
        "flow_spec": {
            "workflow": "custom_workflow",
            "workflow_spec": workflow_spec,
            "structure": {"type": "parsed"},
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
            "producer-alpha completed 64210872-5626-40c7-a7eb-79f7e49272ba\n"
        ).encode("utf-8"),
        f"{LOG_ROOT}/validation-run.err": b"",
        f"{LOG_ROOT}/validation-run.slurm.out": b"",
        f"{LOG_ROOT}/validation-run.slurm.err": b"",
        f"{RESULT_DIR}/CONTCAR": b"contcar",
        f"{RESULT_DIR}/OUTCAR": b"outcar",
        f"{RESULT_DIR}/KPOINTS": b"kpoints",
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


def slurm_runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
    assert command == [
        "ssh",
        "powerslurm-bmdguest",
        (
            "sacct -X -P -n -j 20893681 "
            "--format=JobIDRaw,JobName%30,State,Elapsed,Start,End,Partition%20,ExitCode"
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
            "2026-08-21T15:02:47|leeburton-pool|0:0\n"
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
    )

    assert [stage.stage_type for stage in inspection.workflow_stages] == [
        "relax",
        "static",
        "dos",
        "band_structure",
    ]
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
    assert inspection.runtime.evidence_type == LOG_OBSERVATION
    assert inspection.runtime.packages["pymatgen"] == "2026.8.13"
    assert inspection.runtime.environment["PMG_VASP_PSP_DIR"] == "/bmd-db/potcars"
    assert inspection.scientific.evidence_type == PYMATGEN_DERIVED
    assert inspection.scientific.final_formula == "Example2"
    assert all(item.evidence_type == ARTIFACT_OBSERVATION for item in inspection.final_artifacts)


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

    def parser_should_not_run(*args: object, **kwargs: object) -> ScientificResult:
        raise AssertionError("missing vasprun.xml should prevent scientific parsing")

    inspection = inspect_remote_run(
        cluster(),
        FLOW_ROOT,
        remote_runner=remote,
        slurm_runner=slurm_runner,
        scientific_parser=parser_should_not_run,
    )

    artifacts = {item.label: item for item in inspection.final_artifacts}
    assert artifacts["vasprun"].present is False
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
    assert "stage_03" not in source


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
    monkeypatch.setattr(cli, "inspect_remote_run", lambda cluster, flow_root: inspection)

    exit_code = cli.main(["inspect-run", FLOW_ROOT])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Producer provenance (producer_provenance)" in captured.out
    assert "Scheduler (scheduler_observation)" in captured.out
    assert "Logs (log_observation)" in captured.out
    assert "Evidence paths (artifact_observation)" in captured.out
    assert "Independent parsing (pymatgen_derived)" in captured.out
