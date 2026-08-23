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
from bmd_agent.resources.run import (
    RunInspection,
    RunInspectionError,
    inspect_remote_run,
)
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


def show_inspect_run(flow_root: str, registry: ResourceRegistry | None = None) -> int:
    """Display evidence gathered for a BMD Compute run."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)

    print("BMD Compute Run Inspection")
    print("==========================")
    print()

    try:
        inspection = inspect_remote_run(cluster, flow_root)

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.TimeoutExpired:
        print("Run inspection timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to inspect run through PowerSLURM.")
        if exc.stderr:
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
            print(stderr.strip())
        return 1

    except RunInspectionError as exc:
        print(f"Unable to inspect run: {exc}")
        return 1

    print_run_inspection(inspection)
    return 0


def print_run_inspection(inspection: RunInspection) -> None:
    """Print a concise evidence-oriented run inspection summary."""

    print("Producer provenance (producer_provenance):")
    print(f"  flow root:   {inspection.flow_root}")
    print(f"  submission:  {inspection.submission_path}")
    print(f"  git commit:  {_display_commit(inspection.producer_git.get('git_commit'))}")
    print(f"  git state:   {inspection.producer_git.get('state', 'unavailable')}")
    print()

    print("Requested workflow (producer_provenance):")
    for stage in inspection.workflow_stages:
        modifiers = f" [{', '.join(stage.modifiers)}]" if stage.modifiers else ""
        print(
            f"  {stage.index}. {_display_theory(stage.theory):<6} "
            f"{_display_stage(stage.stage_type)}{modifiers}"
        )
    print()

    print("Requested execution (producer_provenance):")
    _print_mapping_values(
        inspection.cluster_request,
        ("partition", "account"),
    )
    _print_mapping_values(
        inspection.resources_request,
        ("nodes", "ntasks", "mem_gb", "walltime"),
    )
    _print_mapping_values(
        inspection.environment_policy,
        ("VASP_CMD", "JOBFLOW_CONFIG_FILE", "PMG_VASP_PSP_DIR"),
    )
    print()

    print("Scheduler (scheduler_observation):")
    if inspection.scheduler is None:
        print(f"  unavailable: {inspection.scheduler_error or 'no accounting record found'}")
    else:
        record = inspection.scheduler
        print(f"  job id:    {record.job_id}")
        print(f"  state:     {record.state}")
        print(f"  exit:      {record.exit_code}")
        print(f"  elapsed:   {record.elapsed}")
        print(f"  start:     {record.start}")
        print(f"  end:       {record.end}")
        print(f"  partition: {record.partition}")
    print()

    print("Logs (log_observation):")
    if inspection.runtime.sources:
        if inspection.runtime.python:
            print(f"  python: {inspection.runtime.python}")
        for package, version in sorted(inspection.runtime.packages.items()):
            print(f"  {package}: {version}")
        for key, value in sorted(inspection.runtime.environment.items()):
            print(f"  {key}: {value}")
        if inspection.runtime.stage_uuids:
            print("  stage UUIDs:")
            for label, uuid in sorted(inspection.runtime.stage_uuids.items()):
                print(f"    {label}: {uuid}")
    else:
        print("  unavailable: no runner log content was readable")
    print()

    print("Evidence paths (artifact_observation):")
    print(f"  result_dir: {_present_text(inspection.result_directory)}")
    if inspection.stage_directories:
        print("  stage directories:")
        for observation in inspection.stage_directories:
            print(f"    {observation.label}: {_present_text(observation)}")
    if inspection.log_paths:
        print("  logs:")
        for observation in inspection.log_paths:
            print(f"    {observation.label}: {_present_text(observation)}")
    print("  final artifacts:")
    for observation in inspection.final_artifacts:
        print(f"    {observation.label}: {_present_text(observation)}")
    print()

    print("Independent parsing (pymatgen_derived):")
    scientific = inspection.scientific
    if scientific.error:
        print(f"  unavailable: {scientific.error}")
    else:
        _print_optional_value("final formula", scientific.final_formula)
        _print_optional_value("final energy eV", scientific.final_energy_ev)
        _print_optional_value("energy/atom eV", scientific.energy_per_atom_ev)
        _print_optional_value("electronic convergence", scientific.electronic_convergence)
        _print_optional_value("band gap eV", scientific.band_gap_ev)
        _print_optional_value("band k-points", scientific.band_kpoints)
        _print_optional_value("bands", scientific.bands)
        for item in scientific.unavailable:
            print(f"  unavailable: {item}")
    print()

    print("Comparison with durable producer result:")
    print(f"  {inspection.comparison.status}")
    if inspection.comparison.reason:
        print(f"  reason: {inspection.comparison.reason}")
    if inspection.comparison.mismatches:
        print(f"  mismatches: {', '.join(inspection.comparison.mismatches)}")


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

        if command == "inspect-run":
            if len(argv) < 2:
                print("Usage: bmd-agent inspect-run <remote-flow-root>")
                return 2

            return show_inspect_run(argv[1])

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
    print("  inspect-run <remote-flow-root>")
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


def _print_mapping_values(mapping: object, keys: tuple[str, ...]) -> None:
    if not isinstance(mapping, dict):
        return
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            print(f"  {key}: {value}")


def _present_text(observation: object) -> str:
    path = getattr(observation, "path")
    state = "present" if getattr(observation, "present") else "absent"
    return f"{state} ({path})"


def _print_optional_value(label: str, value: object) -> None:
    if value is not None:
        print(f"  {label}: {value}")


if __name__ == "__main__":
    raise SystemExit(main())
