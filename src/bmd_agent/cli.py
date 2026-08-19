from pathlib import Path
import subprocess
import sys

from bmd_agent.resources.slurm import get_queue


def run_git(path: Path, *args: str) -> str:
    """Run a read-only Git query in a repository."""

    result = subprocess.run(
        ["git", "-C", str(path), *args],
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def inspect_repository(name: str, path: Path) -> None:
    print(name)
    print(f"  path:   {path}")

    if not path.exists():
        print("  exists: no")
        print()
        return

    print("  exists: yes")

    try:
        branch = run_git(path, "branch", "--show-current")
        commit = run_git(path, "rev-parse", "HEAD")
        status = run_git(path, "status", "--porcelain")

        print(f"  branch: {branch}")
        print(f"  commit: {commit[:12]}")

        if status:
            print("  state:  modified")
            print("  changes:")

            for line in status.splitlines():
                print(f"    {line}")
        else:
            print("  state:  clean")

    except subprocess.CalledProcessError:
        print("  git:    inspection failed")

    print()


def show_status() -> None:
    home = Path.home()

    resources = {
        "BMD Compute": home / "projects" / "bmd_compute",
        "BMDex": home / "projects" / "BMDex",
    }

    print("BMD Agent")
    print("=========")
    print()

    for name, path in resources.items():
        inspect_repository(name, path)


def show_queue() -> None:
    print("BMD PowerSLURM Queue")
    print("====================")
    print()

    try:
        jobs = get_queue()
    except subprocess.TimeoutExpired:
        print("PowerSLURM connection timed out.")
        return
    except subprocess.CalledProcessError as exc:
        print("Unable to inspect PowerSLURM.")
        if exc.stderr:
            print(exc.stderr.strip())
        return

    if not jobs:
        print("No jobs in leeburton-pool.")
        return

    states: dict[str, int] = {}
    users: dict[str, int] = {}

    for job in jobs:
        states[job.state] = states.get(job.state, 0) + 1
        users[job.user] = users.get(job.user, 0) + 1

    print(f"Total jobs: {len(jobs)}")
    print()

    print("States:")
    for state, count in sorted(states.items()):
        print(f"  {state}: {count}")

    print()

    print("Users:")
    for user, count in sorted(users.items()):
        print(f"  {user}: {count}")

def main() -> None:
    command = sys.argv[1] if len(sys.argv) > 1 else "status"

    if command == "status":
        show_status()
    elif command == "queue":
        show_queue()
    else:
        print(f"Unknown command: {command}")
        print()
        print("Available commands:")
        print("  status")
        print("  queue")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
