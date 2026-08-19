from pathlib import Path
import subprocess


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


def main() -> None:
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


if __name__ == "__main__":
    main()
