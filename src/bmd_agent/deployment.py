from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import date
from importlib import resources
from pathlib import PurePosixPath
import re
import tomllib
from typing import Any

from bmd_agent.config import ConfigurationError, ResourceRegistry, SlurmClusterResource


DEPLOYMENT_PROFILE_SCHEMA = "bmd_agent.deployment_profile"
DEPLOYMENT_PROFILE_SCHEMA_VERSION = 1

_IDENTIFIER_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_RESOURCE_VALUE_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SACCT_FIELD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]*$")


class DeploymentProfileError(ConfigurationError):
    """Raised when a deployment profile is missing or invalid."""


@dataclass(frozen=True)
class DeploymentProfileProvenance:
    verified_date: str
    verification: str


@dataclass(frozen=True)
class DeploymentExpectations:
    scheduler: str
    partition: str
    account: str


@dataclass(frozen=True)
class RecordedHostFacts:
    os_name: str
    os_family: str
    os_compatibility_family: str
    os_version: str
    os_variant: str


@dataclass(frozen=True)
class ReportedMemory:
    value: int
    unit: str
    approximate: bool
    source_semantics: str


@dataclass(frozen=True)
class RecordedSlurmFacts:
    default_time: str
    max_time: str
    over_time_limit: str
    preempt_mode: str
    sacct_fields: tuple[str, ...]
    step_accounting_supported: bool


@dataclass(frozen=True)
class RecordedSchedulerFacts:
    version: str
    node_count: int
    total_cpus: int
    cpus_per_node: int
    memory_per_node: ReportedMemory
    slurm: RecordedSlurmFacts | None = None


@dataclass(frozen=True)
class RecordedDeploymentPaths:
    flows_root: PurePosixPath
    logs_root: PurePosixPath
    potcar_root: PurePosixPath
    remote_python: PurePosixPath
    scheduler_launch_directory: PurePosixPath


@dataclass(frozen=True)
class RecordedSoftwareModule:
    name: str
    role: str


@dataclass(frozen=True)
class RecordedSoftwareFacts:
    vasp_version: str
    modules: tuple[RecordedSoftwareModule, ...]
    module_build_label_note: str


@dataclass(frozen=True)
class DeploymentProfile:
    schema: str
    schema_version: int
    deployment_id: str
    name: str
    provenance: DeploymentProfileProvenance
    expected: DeploymentExpectations
    host: RecordedHostFacts
    scheduler: RecordedSchedulerFacts
    paths: RecordedDeploymentPaths
    software: RecordedSoftwareFacts


@dataclass(frozen=True)
class DeploymentLiveObservation:
    """Optional live facts kept separate from recorded profile knowledge."""

    scheduler: str | None = None
    partition: str | None = None
    account: str | None = None
    os_family: str | None = None
    os_version: str | None = None
    scheduler_version: str | None = None


@dataclass(frozen=True)
class DeploymentDrift:
    field: str
    reference_kind: str
    reference_value: str
    observed_value: str
    status: str = "deployment_drift_observed"


@dataclass(frozen=True)
class DeploymentContext:
    """Resolved profile knowledge plus deployment-local acquisition settings."""

    profile: DeploymentProfile | None
    cluster: SlurmClusterResource | None
    live_observation: DeploymentLiveObservation | None = None
    drift: tuple[DeploymentDrift, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def deployment_id(self) -> str | None:
        return self.profile.deployment_id if self.profile is not None else None


ProfileLoader = Callable[[str], DeploymentProfile]


def load_deployment_profile(deployment_id: str) -> DeploymentProfile:
    """Load one shipped deployment profile without modifying external state."""

    if not _IDENTIFIER_RE.fullmatch(deployment_id):
        raise DeploymentProfileError("deployment profile ID contains unsafe characters")

    profile_file = resources.files("bmd_agent").joinpath(
        "deployment_profiles",
        f"{deployment_id}.toml",
    )
    try:
        contents = profile_file.read_bytes()
    except FileNotFoundError as exc:
        raise DeploymentProfileError(
            f"Unknown BMD Agent deployment profile: {deployment_id}"
        ) from exc

    try:
        raw = tomllib.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise DeploymentProfileError(
            f"Deployment profile {deployment_id!r} is not valid UTF-8 TOML: {exc}"
        ) from exc

    return parse_deployment_profile(raw, source=str(profile_file))


def parse_deployment_profile(
    payload: Mapping[str, Any],
    *,
    source: str = "<memory>",
) -> DeploymentProfile:
    """Validate deployment-profile schema v1 into narrow typed records."""

    schema = _required_str(payload, "schema", "profile", source)
    if schema != DEPLOYMENT_PROFILE_SCHEMA:
        raise DeploymentProfileError(
            f"{source}: unsupported deployment profile schema {schema!r}"
        )

    schema_version = _required_int(payload, "schema_version", "profile", source)
    if schema_version != DEPLOYMENT_PROFILE_SCHEMA_VERSION:
        raise DeploymentProfileError(
            f"{source}: unsupported deployment profile schema_version {schema_version}"
        )

    deployment_id = _required_str(payload, "deployment_id", "profile", source)
    if not _IDENTIFIER_RE.fullmatch(deployment_id):
        raise DeploymentProfileError(f"{source}: deployment_id is invalid")

    provenance = _required_mapping(payload, "provenance", "profile", source)
    expected = _required_mapping(payload, "expected", "profile", source)
    recorded = _required_mapping(payload, "recorded", "profile", source)
    host = _required_mapping(recorded, "host", "recorded", source)
    scheduler = _required_mapping(recorded, "scheduler", "recorded", source)
    memory = _required_mapping(scheduler, "memory_per_node", "recorded.scheduler", source)
    paths = _required_mapping(recorded, "paths", "recorded", source)
    software = _required_mapping(recorded, "software", "recorded", source)

    verified_date = _required_str(provenance, "verified_date", "provenance", source)
    try:
        date.fromisoformat(verified_date)
    except ValueError as exc:
        raise DeploymentProfileError(
            f"{source}: provenance.verified_date must use YYYY-MM-DD"
        ) from exc

    scheduler_kind = _required_str(expected, "scheduler", "expected", source)
    if not _IDENTIFIER_RE.fullmatch(scheduler_kind):
        raise DeploymentProfileError(f"{source}: expected.scheduler is invalid")

    partition = _safe_resource_value(expected, "partition", "expected", source)
    account = _safe_resource_value(expected, "account", "expected", source)
    modules_raw = _required_list(software, "modules", "recorded.software", source)
    modules = tuple(
        RecordedSoftwareModule(
            name=_required_str(
                _as_mapping(value, f"recorded.software.modules[{index}]", source),
                "name",
                f"recorded.software.modules[{index}]",
                source,
            ),
            role=_required_str(
                _as_mapping(value, f"recorded.software.modules[{index}]", source),
                "role",
                f"recorded.software.modules[{index}]",
                source,
            ),
        )
        for index, value in enumerate(modules_raw)
    )
    if not modules:
        raise DeploymentProfileError(f"{source}: recorded.software.modules must not be empty")

    node_count = _positive_int(scheduler, "node_count", "recorded.scheduler", source)
    total_cpus = _positive_int(scheduler, "total_cpus", "recorded.scheduler", source)
    cpus_per_node = _positive_int(
        scheduler,
        "cpus_per_node",
        "recorded.scheduler",
        source,
    )
    if node_count * cpus_per_node != total_cpus:
        raise DeploymentProfileError(
            f"{source}: recorded scheduler CPU topology is internally inconsistent"
        )

    slurm_raw = scheduler.get("slurm")
    if scheduler_kind == "slurm" and slurm_raw is None:
        raise DeploymentProfileError(
            f"{source}: recorded.scheduler.slurm is required for a SLURM deployment"
        )
    slurm = (
        _parse_recorded_slurm(
            _as_mapping(slurm_raw, "recorded.scheduler.slurm", source),
            source,
        )
        if slurm_raw is not None
        else None
    )

    return DeploymentProfile(
        schema=schema,
        schema_version=schema_version,
        deployment_id=deployment_id,
        name=_required_str(payload, "name", "profile", source),
        provenance=DeploymentProfileProvenance(
            verified_date=verified_date,
            verification=_required_str(provenance, "verification", "provenance", source),
        ),
        expected=DeploymentExpectations(
            scheduler=scheduler_kind,
            partition=partition,
            account=account,
        ),
        host=RecordedHostFacts(
            os_name=_required_str(host, "os_name", "recorded.host", source),
            os_family=_required_str(host, "os_family", "recorded.host", source),
            os_compatibility_family=_required_str(
                host,
                "os_compatibility_family",
                "recorded.host",
                source,
            ),
            os_version=_required_str(host, "os_version", "recorded.host", source),
            os_variant=_required_str(host, "os_variant", "recorded.host", source),
        ),
        scheduler=RecordedSchedulerFacts(
            version=_required_str(scheduler, "version", "recorded.scheduler", source),
            node_count=node_count,
            total_cpus=total_cpus,
            cpus_per_node=cpus_per_node,
            memory_per_node=ReportedMemory(
                value=_positive_int(memory, "value", "memory_per_node", source),
                unit=_required_str(memory, "unit", "memory_per_node", source),
                approximate=_required_bool(memory, "approximate", "memory_per_node", source),
                source_semantics=_required_str(
                    memory,
                    "source_semantics",
                    "memory_per_node",
                    source,
                ),
            ),
            slurm=slurm,
        ),
        paths=RecordedDeploymentPaths(
            flows_root=_absolute_posix_path(paths, "flows_root", "recorded.paths", source),
            logs_root=_absolute_posix_path(paths, "logs_root", "recorded.paths", source),
            potcar_root=_absolute_posix_path(paths, "potcar_root", "recorded.paths", source),
            remote_python=_absolute_posix_path(
                paths,
                "remote_python",
                "recorded.paths",
                source,
            ),
            scheduler_launch_directory=_absolute_posix_path(
                paths,
                "scheduler_launch_directory",
                "recorded.paths",
                source,
            ),
        ),
        software=RecordedSoftwareFacts(
            vasp_version=_required_str(
                software,
                "vasp_version",
                "recorded.software",
                source,
            ),
            modules=modules,
            module_build_label_note=_required_str(
                software,
                "module_build_label_note",
                "recorded.software",
                source,
            ),
        ),
    )


def resolve_deployment_context(
    registry: ResourceRegistry,
    *,
    cluster_key: str | None = None,
    live_observation: DeploymentLiveObservation | None = None,
    profile_loader: ProfileLoader = load_deployment_profile,
) -> DeploymentContext:
    """Compose shipped profile knowledge with deployment-local acquisition settings."""

    cluster, limitation = _resolve_cluster(registry, cluster_key)
    if cluster is None:
        return DeploymentContext(
            profile=None,
            cluster=None,
            live_observation=live_observation,
            limitations=(limitation,) if limitation else (),
        )

    if cluster.deployment_profile is None:
        return DeploymentContext(
            profile=None,
            cluster=cluster,
            live_observation=live_observation,
            limitations=("no known deployment profile is configured",),
        )

    profile = profile_loader(cluster.deployment_profile)
    return DeploymentContext(
        profile=profile,
        cluster=cluster,
        live_observation=live_observation,
        drift=(
            compare_deployment_observation(profile, live_observation)
            if live_observation is not None
            else ()
        ),
    )


def compare_deployment_observation(
    profile: DeploymentProfile,
    observation: DeploymentLiveObservation,
) -> tuple[DeploymentDrift, ...]:
    """Describe mismatches as drift without treating the deployment as broken."""

    comparisons = (
        ("scheduler", "expected", profile.expected.scheduler, observation.scheduler),
        ("partition", "expected", profile.expected.partition, observation.partition),
        ("account", "expected", profile.expected.account, observation.account),
        ("os_family", "recorded", profile.host.os_family, observation.os_family),
        ("os_version", "recorded", profile.host.os_version, observation.os_version),
        (
            "scheduler_version",
            "recorded",
            profile.scheduler.version,
            observation.scheduler_version,
        ),
    )
    return tuple(
        DeploymentDrift(
            field=field,
            reference_kind=reference_kind,
            reference_value=reference_value,
            observed_value=observed_value,
        )
        for field, reference_kind, reference_value, observed_value in comparisons
        if observed_value is not None and observed_value != reference_value
    )


def serialize_deployment_context(context: DeploymentContext) -> dict[str, object]:
    """Return JSON-safe deployment knowledge without probing live infrastructure."""

    cluster = context.cluster
    observation = context.live_observation
    return {
        "evidence_type": "deployment_context",
        "profile": (
            serialize_deployment_profile(context.profile) if context.profile is not None else None
        ),
        "acquisition": (
            {
                "cluster_key": cluster.key,
                "cluster_name": cluster.name,
                "ssh_host": cluster.ssh_host,
                "partition": cluster.partition,
                "access": cluster.access,
                "allowed_remote_roots": [str(path) for path in cluster.allowed_remote_roots],
                "timeouts_seconds": {
                    "ssh_connect": cluster.ssh_connect_timeout_seconds,
                    "remote_command": cluster.remote_command_timeout_seconds,
                    "scheduler_accounting": cluster.scheduler_accounting_timeout_seconds,
                },
            }
            if cluster is not None
            else None
        ),
        "live_observation": (
            {
                "evidence_type": "live_observation",
                "scheduler": observation.scheduler,
                "partition": observation.partition,
                "account": observation.account,
                "os_family": observation.os_family,
                "os_version": observation.os_version,
                "scheduler_version": observation.scheduler_version,
            }
            if observation is not None
            else None
        ),
        "drift": [
            {
                "field": item.field,
                "reference_kind": item.reference_kind,
                "reference_value": item.reference_value,
                "observed_value": item.observed_value,
                "status": item.status,
            }
            for item in context.drift
        ],
        "limitations": list(context.limitations),
    }


def serialize_deployment_profile(profile: DeploymentProfile) -> dict[str, object]:
    """Return a JSON-safe representation preserving expected/recorded semantics."""

    fields = _json_safe(asdict(profile))
    recorded = {
        key: fields.pop(key)
        for key in ("host", "scheduler", "paths", "software")
    }
    return {
        "evidence_type": "deployment_profile",
        **fields,
        "recorded": recorded,
    }


def _resolve_cluster(
    registry: ResourceRegistry,
    cluster_key: str | None,
) -> tuple[SlurmClusterResource | None, str | None]:
    if cluster_key is not None:
        try:
            return registry.clusters[cluster_key], None
        except KeyError as exc:
            raise DeploymentProfileError(
                f"Resource configuration has no cluster named {cluster_key!r}"
            ) from exc
    if len(registry.clusters) == 1:
        return next(iter(registry.clusters.values())), None
    if not registry.clusters:
        return None, "no cluster resource is configured"
    return None, "multiple clusters are configured; select a cluster to resolve deployment context"


def _parse_recorded_slurm(
    table: Mapping[str, Any],
    source: str,
) -> RecordedSlurmFacts:
    sacct_fields = _required_str_tuple(
        table,
        "sacct_fields",
        "recorded.scheduler.slurm",
        source,
    )
    if not sacct_fields or any(not _SACCT_FIELD_RE.fullmatch(item) for item in sacct_fields):
        raise DeploymentProfileError(
            f"{source}: recorded.scheduler.slurm.sacct_fields contains invalid field names"
        )
    if len(set(sacct_fields)) != len(sacct_fields):
        raise DeploymentProfileError(
            f"{source}: recorded.scheduler.slurm.sacct_fields must not contain duplicates"
        )
    return RecordedSlurmFacts(
        default_time=_required_str(
            table,
            "default_time",
            "recorded.scheduler.slurm",
            source,
        ),
        max_time=_required_str(
            table,
            "max_time",
            "recorded.scheduler.slurm",
            source,
        ),
        over_time_limit=_required_str(
            table,
            "over_time_limit",
            "recorded.scheduler.slurm",
            source,
        ),
        preempt_mode=_required_str(
            table,
            "preempt_mode",
            "recorded.scheduler.slurm",
            source,
        ),
        sacct_fields=sacct_fields,
        step_accounting_supported=_required_bool(
            table,
            "step_accounting_supported",
            "recorded.scheduler.slurm",
            source,
        ),
    )


def _required_mapping(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> Mapping[str, Any]:
    value = table.get(key)
    return _as_mapping(value, f"{field}.{key}", source)


def _as_mapping(value: Any, field: str, source: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DeploymentProfileError(f"{source}: {field} must be a table")
    return value


def _required_str(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DeploymentProfileError(f"{source}: {field}.{key} must be a non-empty string")
    return value.strip()


def _required_int(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DeploymentProfileError(f"{source}: {field}.{key} must be an integer")
    return value


def _positive_int(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> int:
    value = _required_int(table, key, field, source)
    if value <= 0:
        raise DeploymentProfileError(f"{source}: {field}.{key} must be positive")
    return value


def _required_bool(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise DeploymentProfileError(f"{source}: {field}.{key} must be true or false")
    return value


def _required_list(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> list[Any]:
    value = table.get(key)
    if not isinstance(value, list):
        raise DeploymentProfileError(f"{source}: {field}.{key} must be a list")
    return value


def _required_str_tuple(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> tuple[str, ...]:
    values = _required_list(table, key, field, source)
    if any(not isinstance(value, str) or not value for value in values):
        raise DeploymentProfileError(
            f"{source}: {field}.{key} entries must be non-empty strings"
        )
    return tuple(values)


def _safe_resource_value(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> str:
    value = _required_str(table, key, field, source)
    if not _RESOURCE_VALUE_RE.fullmatch(value):
        raise DeploymentProfileError(f"{source}: {field}.{key} contains unsafe characters")
    return value


def _absolute_posix_path(
    table: Mapping[str, Any],
    key: str,
    field: str,
    source: str,
) -> PurePosixPath:
    value = PurePosixPath(_required_str(table, key, field, source))
    if not value.is_absolute() or ".." in value.parts:
        raise DeploymentProfileError(
            f"{source}: {field}.{key} must be an absolute normalized POSIX path"
        )
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, PurePosixPath):
        return str(value)
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
