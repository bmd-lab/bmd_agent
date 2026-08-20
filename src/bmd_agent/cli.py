import subprocess
import sys

from bmd_agent.config import (
    ConfigurationError,
    GitRepositoryResource,
    ResourceRegistry,
    SlurmClusterResource,
    load_resources,
)
from bmd_agent.resources.compute import (
    ComputeCapabilities,
    ComputeCapabilityError,
    inspect_compute_capabilities,
)
from bmd_agent.resources.git import GitInspection, inspect_repository
from bmd_agent.resources.slurm import get_queue
from bmd_agent.resources.vasp import RemotePathError, read_remote_structure


def show_repository(inspection: GitInspection) -> None:
    """Display read-only information about a Git repository inspection."""

    print(inspection.name)
    print(f"  path:   {inspection.path}")

    if not inspection.exists:
        print("  exists: no")
        print()
        return

    print("  exists: yes")

    if inspection.error:
        print(f"  git:    inspection failed: {inspection.error}")
        print()
        return

    print(f"  branch: {inspection.branch}")
    print(f"  commit: {inspection.commit[:12] if inspection.commit else ''}")

    if inspection.modified:
        print("  state:  modified")
        print("  changes:")

        for line in inspection.status_lines:
            print(f"    {line}")
    else:
        print("  state:  clean")

    print()


def show_status(registry: ResourceRegistry | None = None) -> int:
    """Display the state of configured BMD repositories."""

    registry = registry or load_resources()

    print("BMD Agent")
    print("=========")
    print()

    for repository in registry.repositories.values():
        show_repository(inspect_repository(repository))

    return 0


def show_queue(registry: ResourceRegistry | None = None) -> int:
    """Display a summary of the configured BMD SLURM queue."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)

    print("BMD PowerSLURM Queue")
    print("====================")
    print()

    try:
        jobs = get_queue(
            ssh_host=cluster.ssh_host,
            partition=cluster.partition,
        )

    except subprocess.TimeoutExpired:
        print("PowerSLURM connection timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to inspect PowerSLURM.")

        if exc.stderr:
            print(exc.stderr.strip())

        return 1

    if not jobs:
        print(f"No jobs in {cluster.partition}.")
        return 0

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

    return 0


def show_compute(registry: ResourceRegistry | None = None) -> int:
    """Display BMD Compute's advertised executable capabilities."""

    registry = registry or load_resources()
    repository = bmd_compute_repository(registry)

    try:
        capabilities = inspect_compute_capabilities(repository)

    except ComputeCapabilityError as exc:
        print(f"Unable to inspect BMD Compute capabilities: {exc}")
        return 1

    print_compute_capabilities(capabilities)
    return 0


def print_compute_capabilities(capabilities: ComputeCapabilities) -> None:
    """Print a concise user-facing BMD Compute capability summary."""

    source = capabilities.source

    print("BMD Compute Capabilities")
    print("========================")
    print()

    print("Source:")
    print(f"  repository: {source.get('repository', 'unknown')}")
    print(f"  commit:     {_display_commit(source.get('commit'))}")
    print(f"  state:      {_display_dirty_state(source.get('dirty'))}")
    print()

    print("Scope:")
    print(f"  {capabilities.scope}")
    print()

    print("Supported capabilities:")
    for capability in capabilities.capabilities:
        theory = _display_theory(capability["theory"])
        stage = _display_stage(capability["stage_type"])
        print(f"  {theory:<6} {stage}")


def show_structure(directory: str, registry: ResourceRegistry | None = None) -> int:
    """Display structural information from a remote VASP POSCAR."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)

    print("BMD VASP Structure")
    print("==================")
    print()

    try:
        info = read_remote_structure(
            ssh_host=cluster.ssh_host,
            directory=directory,
            allowed_roots=cluster.allowed_remote_roots,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.CalledProcessError as exc:
        print("Unable to read structure from PowerSLURM.")

        if exc.stderr:
            print(exc.stderr.decode(errors="replace").strip())

        return 1

    except Exception as exc:
        print(f"Unable to parse structure: {exc}")
        return 1

    print(f"Source:          {info.source}")
    print(f"Formula:         {info.formula}")
    print(f"Reduced formula: {info.reduced_formula}")
    print(f"Sites:           {info.sites}")
    print(f"Volume:          {info.volume:.3f} A^3")
    print()

    print("Lattice:")
    print(f"  a: {info.a:.6f} A")
    print(f"  b: {info.b:.6f} A")
    print(f"  c: {info.c:.6f} A")

    return 0


def bmd_compute_repository(registry: ResourceRegistry) -> GitRepositoryResource:
    """Return the configured BMD Compute repository resource."""

    try:
        return registry.repositories["bmd_compute"]

    except KeyError:
        pass

    for repository in registry.repositories.values():
        if repository.role == "compute":
            return repository

    raise ConfigurationError(
        "Resource configuration must include a BMD Compute repository resource."
    )


def powerslurm_cluster(registry: ResourceRegistry) -> SlurmClusterResource:
    """Return the configured PowerSLURM resource."""

    try:
        return registry.clusters["powerslurm"]

    except KeyError as exc:
        raise ConfigurationError(
            "Resource configuration must include [clusters.powerslurm]."
        ) from exc


def main(argv: list[str] | None = None) -> int:
    """BMD Agent command-line entry point."""

    argv = list(sys.argv[1:] if argv is None else argv)
    command = argv[0] if argv else "status"

    try:
        if command == "status":
            return show_status()

        if command == "queue":
            return show_queue()

        if command == "compute":
            return show_compute()

        if command == "structure":
            if len(argv) < 2:
                print("Usage: bmd-agent structure <remote-directory>")
                return 2

            return show_structure(argv[1])

    except ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Unknown command: {command}")
    print()
    print("Available commands:")
    print("  status")
    print("  queue")
    print("  compute")
    print("  structure <remote-directory>")
    return 2


def _display_commit(value: object) -> str:
    if isinstance(value, str) and value:
        return value[:12]

    return "unavailable"


def _display_dirty_state(value: object) -> str:
    if value is True:
        return "dirty"

    if value is False:
        return "clean"

    return "unavailable"


def _display_theory(value: object) -> str:
    return str(value).upper()


def _display_stage(value: object) -> str:
    text = str(value)
    labels = {
        "band_structure": "Band Structure",
        "dos": "DOS",
        "relax": "Geometry Optimisation",
        "static": "Static Energy",
    }

    if text in labels:
        return labels[text]

    return text.replace("_", " ").title()


if __name__ == "__main__":
    raise SystemExit(main())
