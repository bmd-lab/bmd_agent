from pathlib import Path
import subprocess

from bmd_agent.config import GitRepositoryResource
from bmd_agent.resources.git import inspect_repository


FORBIDDEN_GIT_ARGS = {
    "fetch",
    "pull",
    "checkout",
    "reset",
    "clean",
    "commit",
    "push",
    "merge",
}


def repository(path: Path) -> GitRepositoryResource:
    return GitRepositoryResource(
        key="bmd_compute",
        name="BMD Compute",
        path=path,
        role="compute",
        access="read_only",
        protected=True,
        live=True,
    )


def test_git_inspection_uses_mocked_read_only_commands(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    calls: list[list[str]] = []
    env_values: list[str] = []

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        env = kwargs["env"]
        assert isinstance(env, dict)
        env_values.append(env["GIT_OPTIONAL_LOCKS"])

        if "branch" in command:
            return subprocess.CompletedProcess(command, 0, stdout="main\n", stderr="")

        if "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout="abc123def456\n", stderr="")

        if "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout=" M src/app.py\n", stderr="")

        raise AssertionError(f"unexpected command: {command}")

    inspection = inspect_repository(repository(repo_path), runner=runner)

    assert inspection.exists is True
    assert inspection.branch == "main"
    assert inspection.commit == "abc123def456"
    assert inspection.status_lines == (" M src/app.py",)
    assert env_values == ["0", "0", "0"]
    assert not any(arg in FORBIDDEN_GIT_ARGS for command in calls for arg in command)


def test_git_inspection_reports_missing_repository(tmp_path: Path) -> None:
    inspection = inspect_repository(repository(tmp_path / "missing"))

    assert inspection.exists is False
    assert inspection.error is None


def test_git_inspection_reports_safe_directory_failure(tmp_path: Path) -> None:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()

    def runner(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(
            128,
            command,
            stderr=(
                "fatal: detected dubious ownership in repository\n"
                "git config --global --add safe.directory ..."
            ),
        )

    inspection = inspect_repository(repository(repo_path), runner=runner)

    assert inspection.exists is True
    assert inspection.error == "dubious ownership / safe.directory policy prevented inspection"
