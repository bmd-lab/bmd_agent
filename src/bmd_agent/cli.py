from collections.abc import Iterable, Mapping
import hashlib
import subprocess
import sys
from typing import Any

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
    PRODUCER_REQUESTED,
    RunDiagnosis,
    RunInspection,
    RunComparison,
    RunInspectionError,
    compare_remote_runs,
    diagnose_remote_run,
    inspect_remote_run,
)
from bmd_agent.resources.slurm import get_queue
from bmd_agent.resources.vasp import RemotePathError, read_remote_structure


_EXECUTED_INPUT_DISPLAY_KEYS = (
    "ENCUT",
    "EDIFF",
    "EDIFFG",
    "NSW",
    "ISIF",
    "IBRION",
    "ISMEAR",
    "SIGMA",
    "ISPIN",
    "MAGMOM",
    "LDAU",
    "LDAUTYPE",
    "LDAUL",
    "LDAUU",
    "LDAUJ",
    "LMAXMIX",
    "LHFCALC",
    "HFSCREEN",
    "AEXX",
    "ALGO",
    "PRECFOCK",
    "LSORBIT",
    "LNONCOLLINEAR",
    "SAXIS",
    "ISYM",
    "GGA_COMPAT",
    "IVDW",
    "NCORE",
    "NBANDS",
    "LWAVE",
    "LCHARG",
)


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
    modifier_policies, _ = modifier_policies_from_compute(registry)

    print("BMD Compute Run Inspection")
    print("==========================")
    print()

    try:
        inspection = inspect_remote_run(
            cluster,
            flow_root,
            modifier_policies=modifier_policies,
        )

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


def show_compare_runs(flow_roots: list[str], registry: ResourceRegistry | None = None) -> int:
    """Display baseline-relative evidence comparisons for completed runs."""

    if len(flow_roots) < 2:
        print("Usage: bmd-agent compare-runs <flow-a> <flow-b> [<flow-c> ...]")
        return 2

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)
    modifier_policies, policy_warning = modifier_policies_from_compute(registry)

    print("BMD Compute Run Comparison")
    print("==========================")
    print()

    try:
        comparison = compare_remote_runs(
            cluster,
            flow_roots,
            modifier_policies=modifier_policies,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.TimeoutExpired:
        print("Run comparison timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to compare runs through PowerSLURM.")
        if exc.stderr:
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
            print(stderr.strip())
        return 1

    except RunInspectionError as exc:
        print(f"Unable to compare runs: {exc}")
        return 1

    print_run_comparison(comparison, policy_warning=policy_warning)
    return 0


def show_diagnose_run(flow_root: str, registry: ResourceRegistry | None = None) -> int:
    """Display descriptive convergence and termination evidence for one run."""

    registry = registry or load_resources()
    cluster = powerslurm_cluster(registry)
    modifier_policies, _ = modifier_policies_from_compute(registry)

    print("BMD Compute Run Diagnosis")
    print("=========================")
    print()

    try:
        diagnosis = diagnose_remote_run(
            cluster,
            flow_root,
            modifier_policies=modifier_policies,
        )

    except RemotePathError as exc:
        print(f"Refusing remote read: {exc}")
        return 2

    except subprocess.TimeoutExpired:
        print("Run diagnosis timed out.")
        return 1

    except subprocess.CalledProcessError as exc:
        print("Unable to diagnose run through PowerSLURM.")
        if exc.stderr:
            stderr = (
                exc.stderr.decode(errors="replace")
                if isinstance(exc.stderr, bytes)
                else str(exc.stderr)
            )
            print(stderr.strip())
        return 1

    except RunInspectionError as exc:
        print(f"Unable to diagnose run: {exc}")
        return 1

    print_run_diagnosis(diagnosis)
    return 0


def print_run_inspection(inspection: RunInspection) -> None:
    """Print a concise evidence-oriented run inspection summary."""

    print("Producer provenance (producer_provenance):")
    print(f"  flow root:   {inspection.flow_root}")
    print(f"  submission:  {inspection.submission_path}")
    print(f"  git commit:  {_display_commit(inspection.producer_git.get('git_commit'))}")
    print(f"  git state:   {inspection.producer_git.get('state', 'unavailable')}")
    print()

    print(f"Requested workflow ({PRODUCER_REQUESTED}):")
    for stage in inspection.workflow_stages:
        modifiers = f" [{', '.join(stage.modifiers)}]" if stage.modifiers else ""
        print(
            f"  {stage.index}. {_display_theory(stage.theory):<6} "
            f"{_display_stage(stage.stage_type)}{modifiers}"
        )
        options = _format_options(stage.options)
        if options:
            print(f"     options: {options}")
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

    print("Executed VASP inputs (executed_input):")
    _print_executed_input_summary(inspection)
    if inspection.input_expectations:
        print("  requested/executed checks (agent_comparison):")
        for expectation in inspection.input_expectations:
            print(f"    {expectation.stage_label}: {expectation.option_path}={expectation.requested_value} -> "
                  f"{expectation.input_key} {_expectation_observed_text(expectation)}, "
                  f"expected {_expectation_expected_text(expectation)}: "
                  f"{expectation.status}")
            for source, value in sorted(expectation.source_values.items()):
                print(f"      {source}: {_expectation_source_value(expectation, value)}")
            if expectation.reason:
                print(f"      reason: {expectation.reason}")
    print()

    print("Independent parsing (pymatgen_derived):")
    scientific = inspection.scientific
    if scientific.error:
        print(f"  unavailable: {scientific.error}")
    else:
        _print_optional_value("final formula", scientific.final_formula)
        if scientific.structure:
            _print_structure_observation(scientific.structure)
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


def print_run_diagnosis(diagnosis: RunDiagnosis) -> None:
    """Print descriptive termination and trajectory evidence."""

    inspection = diagnosis.inspection
    print("Producer provenance (producer_provenance):")
    print(f"  flow root:   {inspection.flow_root}")
    print(f"  git commit:  {_display_commit(inspection.producer_git.get('git_commit'))}")
    print(f"  git state:   {inspection.producer_git.get('state', 'unavailable')}")
    print()

    print(f"Requested workflow ({PRODUCER_REQUESTED}):")
    for stage in inspection.workflow_stages:
        modifiers = f" [{', '.join(stage.modifiers)}]" if stage.modifiers else ""
        print(
            f"  {stage.index}. {_display_theory(stage.theory):<6} "
            f"{_display_stage(stage.stage_type)}{modifiers}"
        )
        options = _format_options(stage.options)
        if options:
            print(f"     options: {options}")
    print()

    termination = diagnosis.termination
    print(f"Termination evidence ({termination.evidence_type}):")
    _print_optional_value("scheduler state", termination.scheduler_state)
    _print_optional_value("scheduler exit", termination.scheduler_exit_code)
    _print_optional_value("elapsed", termination.scheduler_elapsed)
    _print_optional_value("time limit", termination.scheduler_timelimit)
    _print_optional_value("scheduler reports timeout", termination.scheduler_reports_timeout)
    _print_optional_value("VASP normal completion", termination.vasp_completed_normally)
    if termination.custodian_events:
        print("  custodian events:")
        for event in termination.custodian_events:
            print(f"    {event}")
    for item in termination.unavailable:
        print(f"  unavailable: {item}")
    print()

    print("Trajectory evidence (trajectory_observation):")
    if not diagnosis.trajectories:
        print("  unavailable: no producer-bound VASP stage directories were available")
        return

    for trajectory in diagnosis.trajectories:
        print(f"  {_trajectory_heading(trajectory)}:")
        print(f"    directory: {trajectory.directory}")
        print(
            f"    OSZICAR: "
            f"{'present' if trajectory.oszicar_present else 'unavailable'} "
            f"({trajectory.oszicar_path})"
        )
        if trajectory.oszicar_error:
            print(f"      error: {trajectory.oszicar_error}")
        _print_vasprun_trajectory_source(trajectory)
        _print_trajectory_criteria(trajectory)
        _print_electronic_trajectory(trajectory)
        _print_ionic_trajectory(trajectory)
        print("    convergence flags:")
        print(f"      electronic: {_diagnosis_value(trajectory.converged_electronic)}")
        print(f"      ionic: {_diagnosis_value(trajectory.converged_ionic)}")
        for item in trajectory.unavailable:
            print(f"    unavailable: {item}")


def print_run_comparison(
    comparison: RunComparison,
    *,
    policy_warning: str | None = None,
) -> None:
    """Print a concise baseline-relative run comparison."""

    baseline = comparison.inspections[0]
    baseline_label = comparison.labels[baseline.flow_root]

    print("Runs:")
    for index, inspection in enumerate(comparison.inspections):
        role = "baseline" if index == 0 else "comparison"
        label = comparison.labels[inspection.flow_root]
        scheduler_state = inspection.scheduler.state if inspection.scheduler else "unavailable"
        print(f"  {index + 1}. {label} ({role})")
        print(f"     flow root: {inspection.flow_root}")
        print(f"     git: {_display_commit(inspection.producer_git.get('git_commit'))} "
              f"{inspection.producer_git.get('state', 'unavailable')}")
        print(f"     scheduler: {scheduler_state}")
    print()

    print("Initial structure (agent_comparison):")
    print(f"  {comparison.initial_structure.status}")
    if comparison.initial_structure.reason:
        print(f"  reason: {comparison.initial_structure.reason}")
    print()

    print("Executed-input checks:")
    any_checks = False
    for inspection in comparison.inspections:
        label = comparison.labels[inspection.flow_root]
        for expectation in inspection.input_expectations:
            any_checks = True
            print(f"  {label} {expectation.stage_label}: "
                  f"{expectation.input_key} {_expectation_observed_text(expectation)}, "
                  f"expected {_expectation_expected_text(expectation)}: "
                  f"{expectation.status}")
            for source, value in sorted(expectation.source_values.items()):
                print(f"    {source}: {_expectation_source_value(expectation, value)}")
            if expectation.reason:
                print(f"    reason: {expectation.reason}")
    if not any_checks:
        print("  unavailable: no producer option/input-effect checks were available")
    print()

    print("Final structure/result differences (agent_comparison):")
    print(f"  baseline: {baseline_label}")
    current_flow = None
    for quantity in comparison.quantities:
        if quantity.flow_root != current_flow:
            current_flow = quantity.flow_root
            print(f"  {quantity.run_label}:")
        if quantity.status != "available":
            print(f"    {quantity.label}: unavailable ({quantity.reason})")
            continue
        unit = f" {quantity.unit}" if quantity.unit else ""
        percent = (
            f", {quantity.percent_delta:+.3f}%"
            if quantity.percent_delta is not None
            else ""
        )
        print(
            f"    {quantity.label}: {quantity.baseline_value} -> "
            f"{quantity.comparison_value}{unit}, delta {quantity.delta:+.6f}{unit}{percent}"
        )
    if comparison.energy_warning:
        print()
        print(f"Warning: {comparison.energy_warning}")
    if policy_warning:
        print()
        print(f"Modifier policy warning: {policy_warning}")


def _trajectory_heading(trajectory: object) -> str:
    stage_index = getattr(trajectory, "stage_index", None)
    if stage_index is None:
        prefix = "unindexed stage"
    else:
        prefix = f"stage {stage_index}"
    theory = getattr(trajectory, "theory", None)
    stage_type = getattr(trajectory, "stage_type", None)
    if theory and stage_type:
        prefix = f"{prefix}: {_display_theory(theory)} {_display_stage(stage_type)}"
    label = getattr(trajectory, "stage_label", None)
    return f"{prefix} ({label})" if label else prefix


def _print_vasprun_trajectory_source(trajectory: object) -> None:
    path = getattr(trajectory, "vasprun_path", None)
    if getattr(trajectory, "vasprun_present", False):
        print(f"    vasprun.xml: present ({path})")
    else:
        print(f"    vasprun.xml: unavailable ({path})")
    if getattr(trajectory, "vasprun_skipped_reason", None):
        print(f"      skipped: {getattr(trajectory, 'vasprun_skipped_reason')}")
    if getattr(trajectory, "vasprun_error", None):
        print(
            "      vasprun trajectory enrichment unavailable: "
            f"{getattr(trajectory, 'vasprun_error')}"
        )


def _print_trajectory_criteria(trajectory: object) -> None:
    criteria = getattr(trajectory, "criteria", {})
    print("    criteria:")
    if not criteria:
        print("      unavailable")
    for key in ("NELM", "EDIFF", "NSW", "EDIFFG"):
        if key in criteria:
            print(f"      {key}: {_format_trajectory_criterion(key, criteria[key])}")
    discrepancies = getattr(trajectory, "criteria_discrepancies", ())
    if discrepancies:
        print(f"      discrepancies: {', '.join(discrepancies)}")
        source_values = getattr(trajectory, "criteria_source_values", {})
        for key in discrepancies:
            values = source_values.get(key, {})
            for source, value in sorted(values.items()):
                print(f"        {key} {source}: {_format_executed_value(key, value)}")


def _format_trajectory_criterion(key: str, value: object) -> str:
    if key != "EDIFFG":
        return _format_executed_value(key, value)
    numeric = _float_or_none(value)
    if numeric is None:
        return _format_executed_value(key, value)
    if numeric < 0:
        return f"{value} eV/A force criterion"
    if numeric > 0:
        return f"{value} eV energy-change criterion"
    return "0 (no generic convergence interpretation inferred)"


def _print_electronic_trajectory(trajectory: object) -> None:
    print("    electronic trajectory:")
    completed_ionic_steps = getattr(
        trajectory,
        "completed_ionic_steps",
        getattr(trajectory, "ionic_steps_observed", None),
    )
    print(f"      completed ionic steps: {_diagnosis_value(completed_ionic_steps)}")

    counts = getattr(
        trajectory,
        "electronic_iterations_by_completed_ionic_step",
        getattr(trajectory, "electronic_iterations_by_ionic_step", ()),
    )
    if counts:
        print(
            "      electronic iterations for completed ionic steps: "
            f"{_format_count_sequence(counts)}"
        )
    else:
        print("      electronic iterations for completed ionic steps: none")

    incomplete_count = getattr(trajectory, "incomplete_electronic_iteration_count", None)
    if incomplete_count is not None:
        print(
            "      incomplete electronic cycle: "
            f"{incomplete_count} iterations observed{_nelm_suffix(trajectory)}"
        )
        recent = getattr(trajectory, "recent_incomplete_electronic_iterations", ())
        recent_label = "recent incomplete-cycle iterations"
    else:
        final_count = getattr(trajectory, "final_electronic_iteration_count", None)
        if final_count is None:
            print("      final electronic cycle: unavailable")
        else:
            print(
                "      final electronic cycle: "
                f"{final_count} iterations observed{_nelm_suffix(trajectory)}"
            )
        recent = getattr(trajectory, "recent_electronic_iterations", ())
        recent_label = "recent final-cycle iterations"

    if recent:
        print(f"      {recent_label}:")
        for iteration in recent:
            print(f"        {_format_electronic_iteration(iteration)}")
    else:
        print(f"      {recent_label}: unavailable")


def _print_ionic_trajectory(trajectory: object) -> None:
    print("    ionic trajectory:")
    completed_ionic_steps = getattr(
        trajectory,
        "completed_ionic_steps",
        getattr(trajectory, "ionic_steps_observed", None),
    )
    print(f"      completed ionic steps: {_diagnosis_value(completed_ionic_steps)}")
    vasprun_steps = getattr(trajectory, "vasprun_ionic_steps", None)
    if vasprun_steps is not None:
        print(f"      vasprun ionic steps: {vasprun_steps}")
    recent = getattr(trajectory, "recent_ionic_steps", ())
    if recent:
        print("      recent completed ionic steps:")
        for step in recent:
            print(f"        {_format_ionic_step(step)}")
    else:
        print("      recent completed ionic steps: unavailable")


def _nelm_suffix(trajectory: object) -> str:
    criteria = getattr(trajectory, "criteria", {})
    if isinstance(criteria, Mapping) and "NELM" in criteria:
        return f" / NELM {criteria['NELM']}"
    return ""


def _format_electronic_iteration(iteration: object) -> str:
    parts = [
        f"N={_diagnosis_value(getattr(iteration, 'iteration', None))}",
    ]
    algorithm = getattr(iteration, "algorithm", None)
    if algorithm:
        parts.insert(0, str(algorithm))
    for label, attribute in (
        ("E", "energy"),
        ("dE", "dE"),
        ("deps", "deps"),
        ("rms", "rms"),
        ("rms(c)", "rms_c"),
    ):
        value = getattr(iteration, attribute, None)
        if value is not None:
            parts.append(f"{label}={value}")
    return " ".join(parts)


def _format_ionic_step(step: object) -> str:
    parts = [f"step {getattr(step, 'step_index')}"]
    for label, attribute in (
        ("electronic_iterations", "electronic_iterations"),
        ("F", "free_energy"),
        ("E0", "energy_zero"),
        ("dE", "dE"),
        ("max_force", "max_force"),
    ):
        value = getattr(step, attribute, None)
        if value is not None:
            parts.append(f"{label}={value}")
    return " ".join(parts)


def _format_count_sequence(values: tuple[int, ...]) -> str:
    if len(values) <= 8:
        return "[" + ", ".join(str(value) for value in values) + "]"
    recent = ", ".join(str(value) for value in values[-5:])
    return f"{len(values)} values, last five [{recent}]"


def _diagnosis_value(value: object) -> str:
    return "unavailable" if value is None else str(value)


def _float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

        if command == "compare-runs":
            return show_compare_runs(argv[1:])

        if command == "diagnose-run":
            if len(argv) < 2:
                print("Usage: bmd-agent diagnose-run <remote-flow-root>")
                return 2

            return show_diagnose_run(argv[1])

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
    print("  compare-runs <flow-a> <flow-b> [<flow-c> ...]")
    print("  diagnose-run <remote-flow-root>")
    return 2


def modifier_policies_from_compute(
    registry: ResourceRegistry,
) -> tuple[tuple[Mapping[str, Any], ...], str | None]:
    """Return configured producer modifier policies when the capability adapter can read them."""

    try:
        repository = bmd_compute_repository(registry)
    except ConfigurationError:
        return (), None

    try:
        capabilities = inspect_compute_capabilities(repository)
    except ComputeCapabilityError as exc:
        return (), str(exc)

    policies = capabilities.payload.get("modifier_policies")
    if not isinstance(policies, list):
        return (), None
    return tuple(policy for policy in policies if isinstance(policy, Mapping)), None


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


def _print_structure_observation(observation: object) -> None:
    print("  final structure:")
    _print_optional_nested_value("formula", getattr(observation, "formula", None))
    _print_optional_nested_value("reduced formula", getattr(observation, "reduced_formula", None))
    _print_optional_nested_value("site count", getattr(observation, "site_count", None))
    _print_optional_nested_value("lattice a A", getattr(observation, "lattice_a", None))
    _print_optional_nested_value("lattice b A", getattr(observation, "lattice_b", None))
    _print_optional_nested_value("lattice c A", getattr(observation, "lattice_c", None))
    _print_optional_nested_value("alpha deg", getattr(observation, "alpha", None))
    _print_optional_nested_value("beta deg", getattr(observation, "beta", None))
    _print_optional_nested_value("gamma deg", getattr(observation, "gamma", None))
    _print_optional_nested_value("volume A^3", getattr(observation, "volume", None))
    _print_optional_nested_value("density g/cm^3", getattr(observation, "density", None))
    _print_optional_nested_value("c/a cell-axis ratio", getattr(observation, "c_over_a", None))
    for item in getattr(observation, "unavailable", ()):
        print(f"    unavailable: {item}")


def _print_optional_nested_value(label: str, value: object) -> None:
    if value is not None:
        print(f"    {label}: {value}")


def _format_options(options: Mapping[str, Any]) -> str:
    flattened = []
    for key, value in _flatten_options(options):
        flattened.append(f"{key}={value}")
    return ", ".join(flattened)


def _flatten_options(options: Mapping[str, Any], prefix: str = "") -> list[tuple[str, object]]:
    flattened: list[tuple[str, object]] = []
    for key, value in sorted(options.items()):
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.extend(_flatten_options(value, path))
        else:
            flattened.append((path, value))
    return flattened


def _print_executed_input_summary(inspection: RunInspection) -> None:
    groups = _executed_input_groups(inspection.executed_inputs)
    if not groups:
        print("  unavailable: no executed-input observations were gathered")
        return

    stages = {stage.index: stage for stage in inspection.workflow_stages}
    for stage_index, observations in groups:
        print(f"  {_executed_stage_heading(stage_index, observations, stages)}:")
        for observation in observations:
            print(f"    source {_executed_source_label(observation)}: {_executed_source_state(observation)}")

        settings, discrepancies = _reconciled_executed_settings(observations)
        if settings:
            print(f"    settings: {', '.join(settings)}")
        elif any(_readable_executed_source(observation) for observation in observations):
            print("    settings: no displayed INCAR settings present")
        else:
            print("    settings: unavailable")

        for key, source_values in discrepancies:
            values = "; ".join(
                f"{source}={_format_executed_value(key, value, fingerprint=True)}"
                for source, value in source_values
            )
            print(f"    discrepancy: {key}: {values}")


def _executed_input_groups(
    observations: tuple[object, ...],
) -> list[tuple[int | None, tuple[object, ...]]]:
    grouped: dict[int | None, list[object]] = {}
    ordered_keys: list[int | None] = []
    for observation in observations:
        key = getattr(observation, "stage_index", None)
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append(observation)

    return [
        (key, tuple(grouped[key]))
        for key in sorted(
            ordered_keys,
            key=lambda value: (value is None, value if isinstance(value, int) else 0),
        )
    ]


def _executed_stage_heading(
    stage_index: int | None,
    observations: tuple[object, ...],
    stages: Mapping[int, object],
) -> str:
    if stage_index is None:
        base = "unindexed"
    else:
        stage = stages.get(stage_index)
        if stage is None:
            base = f"stage {stage_index}"
        else:
            base = (
                f"stage {stage_index}: "
                f"{_display_theory(getattr(stage, 'theory'))} "
                f"{_display_stage(getattr(stage, 'stage_type'))}"
            )

    labels = _ordered_unique(
        str(getattr(observation, "label"))
        for observation in observations
        if getattr(observation, "source_type", None) == "retained_incar"
    )
    if labels:
        return f"{base} ({', '.join(labels)})"
    return base


def _executed_source_label(observation: object) -> str:
    source_type = getattr(observation, "source_type")
    label = getattr(observation, "label")
    return f"{source_type}:{label}"


def _executed_source_state(observation: object) -> str:
    path = getattr(observation, "path")
    if not getattr(observation, "present"):
        return f"unavailable ({path})"
    error = getattr(observation, "error")
    if error:
        return f"present but unparsed ({path}): {error}"
    return f"present ({path})"


def _reconciled_executed_settings(
    observations: tuple[object, ...],
) -> tuple[list[str], list[tuple[str, list[tuple[str, object]]]]]:
    readable = [
        observation
        for observation in observations
        if _readable_executed_source(observation)
    ]
    settings: list[str] = []
    discrepancies: list[tuple[str, list[tuple[str, object]]]] = []

    for key in _EXECUTED_INPUT_DISPLAY_KEYS:
        source_values = [
            (_executed_source_label(observation), getattr(observation, "values").get(key))
            for observation in readable
            if key in getattr(observation, "values")
        ]
        if not source_values:
            continue

        values = [value for _, value in source_values]
        if _all_values_match(values):
            context = _first_mapping_with_key(readable, key)
            settings.append(f"{key}={_format_executed_value(key, values[0], context)}")
        else:
            discrepancies.append((key, source_values))

    return settings, discrepancies


def _readable_executed_source(observation: object) -> bool:
    return (
        bool(getattr(observation, "present"))
        and getattr(observation, "error") is None
        and isinstance(getattr(observation, "values"), Mapping)
    )


def _first_mapping_with_key(observations: list[object], key: str) -> Mapping[str, object]:
    for observation in observations:
        values = getattr(observation, "values")
        if isinstance(values, Mapping) and key in values:
            return values
    return {}


def _all_values_match(values: list[object]) -> bool:
    if not values:
        return True
    first = values[0]
    return all(_display_values_match(first, value) for value in values[1:])


def _display_values_match(left: object, right: object) -> bool:
    return _format_canonical_value(left) == _format_canonical_value(right)


def _format_canonical_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_format_canonical_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_format_canonical_value(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _format_canonical_value(item)) for key, item in value.items())
        )
    return value


def _expectation_observed_text(expectation: object) -> str:
    status = getattr(expectation, "status")
    value = getattr(expectation, "observed_value")
    if status == "unavailable":
        return "unavailable"
    if status == "absent":
        return "absent"
    formatted = _format_executed_value(
        getattr(expectation, 'input_key'),
        value,
        fingerprint=status == 'discrepancy',
    )
    return f"observed {formatted}"


def _expectation_expected_text(expectation: object) -> str:
    return _format_executed_value(
        getattr(expectation, "input_key"),
        getattr(expectation, "expected_value"),
        fingerprint=getattr(expectation, "status") == "discrepancy",
    )


def _expectation_source_value(expectation: object, value: object) -> str:
    return _format_executed_value(
        getattr(expectation, "input_key"),
        value,
        fingerprint=getattr(expectation, "status") == "discrepancy",
    )


def _format_executed_value(
    key: object,
    value: object,
    context: Mapping[str, object] | None = None,
    *,
    fingerprint: bool = False,
) -> str:
    if value is None:
        return "absent"
    if str(key).upper() == "MAGMOM":
        return _format_magmom(value, context or {}, fingerprint=fingerprint)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list | tuple):
        return _format_sequence(value)
    if isinstance(value, Mapping):
        return "{" + ", ".join(
            f"{item_key}: {_format_executed_value(item_key, item_value)}"
            for item_key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        ) + "}"
    return str(value)


def _format_sequence(value: list[object] | tuple[object, ...]) -> str:
    if len(value) <= 8:
        return "[" + ", ".join(_format_executed_value("", item) for item in value) + "]"
    return f"present, {len(value)} values"


def _format_magmom(
    value: object,
    context: Mapping[str, object],
    *,
    fingerprint: bool = False,
) -> str:
    flattened = _flatten_numeric_values(value)
    if flattened is None:
        return "present"
    count = len(flattened)
    if count == 0:
        return "present, 0 values"
    if count <= 8:
        return "[" + ", ".join(_format_executed_value("", item) for item in flattened) + "]"

    noncollinear = (
        _truthy_vasp_value(context.get("LNONCOLLINEAR"))
        or _truthy_vasp_value(context.get("LSORBIT"))
    )
    if noncollinear and count % 3 == 0:
        summary = f"present, {count // 3} sites / {count} noncollinear components"
    else:
        summary = f"present, {count} sites"
    if fingerprint:
        summary = f"{summary}, fingerprint {_value_fingerprint(value)}"
    return summary


def _value_fingerprint(value: object) -> str:
    canonical = repr(_format_canonical_value(value)).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()[:12]


def _flatten_numeric_values(value: object) -> list[object] | None:
    if isinstance(value, int | float):
        return [value]
    if not isinstance(value, list | tuple):
        return None
    flattened: list[object] = []
    for item in value:
        nested = _flatten_numeric_values(item)
        if nested is None:
            return None
        flattened.extend(nested)
    return flattened


def _truthy_vasp_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().strip(".").upper() in {"T", "TRUE"}
    return bool(value)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


if __name__ == "__main__":
    raise SystemExit(main())
