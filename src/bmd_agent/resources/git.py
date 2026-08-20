from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Callable, Sequence

from bmd_agent.config import GitRepositoryResource


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class GitInspection:
    name: str
    path: Path
    exists: bool
    branch: str | None = None
    commit: str | None = None
    status_lines: tuple[str, ...] = ()
    error: str | None = None

    @property
    def modified(self) -> bool:
        return bool(self.status_lines)


def inspect_repository(
    repository: GitRepositoryResource,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 10,
) -> GitInspection:
    """Return read-only Git information for a configured repository."""

    if not repository.path.exists():
        return GitInspection(
            name=repository.name,
            path=repository.path,
            exists=False,
        )

    try:
        branch = run_git(repository.path, ["branch", "--show-current"], runner=runner, timeout=timeout)
        commit = run_git(repository.path, ["rev-parse", "HEAD"], runner=runner, timeout=timeout)
        status = run_git(
            repository.path,
            ["status", "--porcelain=v1", "--untracked-files=all", "--no-renames"],
            runner=runner,
            timeout=timeout,
        )

    except subprocess.CalledProcessError as exc:
        return GitInspection(
            name=repository.name,
            path=repository.path,
            exists=True,
            error=_git_error_message(exc),
        )

    except subprocess.TimeoutExpired:
        return GitInspection(
            name=repository.name,
            path=repository.path,
            exists=True,
            error="git inspection timed out",
        )

    return GitInspection(
        name=repository.name,
        path=repository.path,
        exists=True,
        branch=branch,
        commit=commit,
        status_lines=tuple(line for line in status.splitlines() if line),
    )


def run_git(
    path: Path,
    args: Sequence[str],
    *,
    runner: Runner = subprocess.run,
    timeout: float = 10,
) -> str:
    """Run one read-only Git query with optional locking disabled."""

    command = ["git", "-c", "gc.auto=0", "-C", str(path), *args]
    env = os.environ.copy()
    env["GIT_OPTIONAL_LOCKS"] = "0"

    result = runner(
        command,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
        env=env,
    )

    return result.stdout.rstrip("\r\n")


def _git_error_message(exc: subprocess.CalledProcessError) -> str:
    details = _error_text(exc)

    if "dubious ownership" in details or "safe.directory" in details:
        return "dubious ownership / safe.directory policy prevented inspection"

    first_line = details.splitlines()[0] if details.splitlines() else ""

    return first_line or "git command failed"


def _error_text(exc: subprocess.CalledProcessError) -> str:
    stderr = exc.stderr or ""
    stdout = exc.stdout or ""

    return f"{stderr}\n{stdout}".strip()
