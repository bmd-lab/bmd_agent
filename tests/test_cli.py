from pathlib import Path
from types import SimpleNamespace

import pytest

from bmd_agent import cli
from bmd_agent.config import ConfigurationError, ResourceRegistry, SlurmClusterResource
from bmd_agent.resources.run import (
    InitialStructureComparison,
    QuantityComparison,
    RunComparison,
)


def test_cli_nonexistent_nonnumeric_target_is_clean_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = cli.main(["frobnicate"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert (
        "Target was not recognized as a SLURM job ID or existing calculation path."
        in captured.out
    )
    assert "Traceback" not in captured.out


def test_bare_numeric_target_and_explicit_job_use_same_job_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool]] = []

    def fake_show_job(
        job_id: str,
        registry: ResourceRegistry | None = None,
        *,
        trajectory_json: bool = False,
    ) -> int:
        calls.append((job_id, trajectory_json))
        return 0

    monkeypatch.setattr(cli, "show_job", fake_show_job)

    assert cli.main(["21853598"]) == 0
    assert cli.main(["job", "21853598"]) == 0
    assert calls == [("21853598", False), ("21853598", False)]


def test_bare_numeric_target_wins_over_same_named_local_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "21853598").mkdir()
    monkeypatch.chdir(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(cli, "show_job", lambda job_id, **kwargs: calls.append(job_id) or 0)
    monkeypatch.setattr(
        cli,
        "show_current_directory",
        lambda *args, **kwargs: pytest.fail("bare numeric target must not dispatch as a path"),
    )

    assert cli.main(["21853598"]) == 0
    assert calls == ["21853598"]


@pytest.mark.parametrize("target_kind", ("relative", "absolute", "numeric_relative"))
def test_existing_path_target_uses_lifecycle_analysis(
    target_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "21853598" if target_kind == "numeric_relative" else "calculation"
    calculation = tmp_path / name
    calculation.mkdir()
    monkeypatch.chdir(tmp_path)
    targets = {
        "relative": "./calculation",
        "absolute": str(calculation.resolve()),
        "numeric_relative": "./21853598",
    }
    analyzed: list[Path] = []

    def fake_show_current_directory(
        directory: Path | None = None,
        registry: ResourceRegistry | None = None,
    ) -> int:
        assert directory is not None
        analyzed.append(directory.resolve())
        return 0

    monkeypatch.setattr(cli, "show_current_directory", fake_show_current_directory)
    monkeypatch.setattr(
        cli,
        "show_job",
        lambda *args, **kwargs: pytest.fail("explicit path syntax must not dispatch as a job"),
    )

    assert cli.main([targets[target_kind]]) == 0
    assert analyzed == [calculation.resolve()]


def test_cli_structure_requires_directory(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["structure"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent structure <remote-directory>" in captured.out


def test_cli_compare_runs_requires_two_roots(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["compare-runs", "/one"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent compare-runs <flow-a> <flow-b>" in captured.out


def test_cli_diagnose_run_requires_root(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["diagnose-run"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent diagnose-run <remote-flow-root>" in captured.out


def test_cli_job_requires_id(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["job"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Usage: bmd-agent job <SLURM_JOB_ID>" in captured.out


def test_cli_compare_runs_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(),
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster})
    baseline = SimpleNamespace(
        flow_root="/flow/a",
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        scheduler=SimpleNamespace(state="COMPLETED"),
        input_expectations=(),
    )
    comparison_run = SimpleNamespace(
        flow_root="/flow/b",
        producer_git={"git_commit": "abcdef012345", "state": "clean"},
        scheduler=SimpleNamespace(state="COMPLETED"),
        input_expectations=(),
    )
    comparison = RunComparison(
        inspections=(baseline, comparison_run),
        labels={"/flow/a": "PBE", "/flow/b": "PBE + modifier"},
        quantities=(
            QuantityComparison(
                run_label="PBE + modifier",
                flow_root="/flow/b",
                quantity="lattice_c",
                label="lattice c",
                unit="A",
                baseline_value=6.0,
                comparison_value=5.4,
                delta=-0.6,
                percent_delta=-10.0,
                status="available",
            ),
        ),
        initial_structure=InitialStructureComparison("match"),
        energy_warning="energy values are observations, not a ranking of method quality.",
    )

    monkeypatch.setattr(cli, "load_resources", lambda: registry)
    monkeypatch.setattr(cli, "modifier_policies_from_compute", lambda registry: ((), None))
    monkeypatch.setattr(cli, "compare_remote_runs", lambda cluster, flow_roots, **kwargs: comparison)

    exit_code = cli.main(["compare-runs", "/flow/a", "/flow/b"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "BMD Compute Run Comparison" in captured.out
    assert "baseline: PBE" in captured.out
    assert "lattice c: 6.0 -> 5.4 A, delta -0.600000 A, -10.000%" in captured.out
    assert "not a ranking of method quality" in captured.out


def test_cli_reports_missing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_load_resources() -> None:
        raise ConfigurationError("No BMD Agent resource configuration found.")

    monkeypatch.setattr(cli, "load_resources", fail_load_resources)

    exit_code = cli.main(["status"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "No BMD Agent resource configuration found." in captured.err


def test_status_exposes_resolved_profile_as_context_not_live_observation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(),
        deployment_profile="power",
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster})

    assert cli.show_status(registry) == 0

    output = capsys.readouterr().out
    assert "Deployment context:" in output
    assert "profile: power (TAU POWER)" in output
    assert "profile schema: bmd_agent.deployment_profile v1" in output
    assert "expected scheduler: slurm" in output
    assert "live" not in output.lower()


def test_queue_uses_deployment_local_operational_timeouts(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cluster = SlurmClusterResource(
        key="powerslurm",
        name="PowerSLURM",
        ssh_host="powerslurm-bmdguest",
        partition="leeburton-pool",
        access="observational",
        allowed_remote_roots=(),
        ssh_connect_timeout_seconds=13,
        remote_command_timeout_seconds=31,
        scheduler_accounting_timeout_seconds=91,
    )
    registry = ResourceRegistry(repositories={}, clusters={"powerslurm": cluster})
    observed: dict[str, object] = {}

    def fake_queue(**kwargs: object) -> list:
        observed.update(kwargs)
        return []

    monkeypatch.setattr(cli, "get_queue", fake_queue)

    assert cli.show_queue(registry) == 0

    assert observed == {
        "ssh_host": "powerslurm-bmdguest",
        "partition": "leeburton-pool",
        "timeout": 31,
        "ssh_connect_timeout": 13,
    }
    assert "No jobs in leeburton-pool." in capsys.readouterr().out
