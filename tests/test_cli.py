import pytest
from types import SimpleNamespace

from bmd_agent import cli
from bmd_agent.config import ConfigurationError, ResourceRegistry, SlurmClusterResource
from bmd_agent.resources.run import (
    InitialStructureComparison,
    QuantityComparison,
    RunComparison,
)


def test_cli_unknown_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = cli.main(["frobnicate"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Unknown command: frobnicate" in captured.out


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
