import pytest

from bmd_agent import cli
from bmd_agent.config import ConfigurationError


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
