from copy import deepcopy
from importlib import resources
import json
import tomllib

import pytest

from bmd_agent.config import parse_resources
from bmd_agent.deployment import (
    DEPLOYMENT_PROFILE_SCHEMA,
    DEPLOYMENT_PROFILE_SCHEMA_VERSION,
    DeploymentLiveObservation,
    DeploymentProfileError,
    compare_deployment_observation,
    load_deployment_profile,
    parse_deployment_profile,
    resolve_deployment_context,
    serialize_deployment_context,
)


def power_payload() -> dict:
    profile_file = resources.files("bmd_agent").joinpath(
        "deployment_profiles",
        "power.toml",
    )
    return tomllib.loads(profile_file.read_text(encoding="utf-8"))


def registry_config(*, deployment_profile: str | None = "power") -> dict:
    cluster = {
        "name": "PowerSLURM",
        "ssh_host": "powerslurm-bmdguest",
        "partition": "leeburton-pool",
        "access": "observational",
        "allowed_remote_roots": ["/bmd-db/guest/flows", "/bmd-db/guest/logs"],
        "ssh_connect_timeout_seconds": 10,
        "remote_command_timeout_seconds": 20,
        "scheduler_accounting_timeout_seconds": 60,
    }
    if deployment_profile is not None:
        cluster["deployment_profile"] = deployment_profile
    return {"repositories": {}, "clusters": {"powerslurm": cluster}}


def test_power_profile_validates_and_preserves_identity_and_provenance() -> None:
    profile = load_deployment_profile("power")

    assert profile.schema == DEPLOYMENT_PROFILE_SCHEMA
    assert profile.schema_version == DEPLOYMENT_PROFILE_SCHEMA_VERSION
    assert profile.deployment_id == "power"
    assert profile.name == "TAU POWER"
    assert profile.provenance.verified_date == "2026-09-21"
    assert "direct OS" in profile.provenance.verification


def test_power_profile_preserves_expected_and_recorded_infrastructure_facts() -> None:
    profile = load_deployment_profile("power")

    assert profile.expected.scheduler == "slurm"
    assert profile.expected.partition == "leeburton-pool"
    assert profile.expected.account == "power-leeburton-users_v2"
    assert profile.host.os_name == "Rocky Linux"
    assert profile.host.os_family == "rocky"
    assert profile.host.os_compatibility_family == "rhel-like"
    assert profile.host.os_version == "9.8"
    assert profile.host.os_variant == "Blue Onyx"
    assert profile.scheduler.version == "25.11.6"
    assert profile.scheduler.node_count == 8
    assert profile.scheduler.total_cpus == 1536
    assert profile.scheduler.cpus_per_node == 192
    assert profile.scheduler.memory_per_node.value == 1_030_957
    assert profile.scheduler.memory_per_node.unit == "MB"
    assert profile.scheduler.memory_per_node.approximate is True
    assert profile.scheduler.memory_per_node.source_semantics == (
        "approximate value reported by sinfo"
    )
    assert profile.scheduler.slurm is not None
    assert profile.scheduler.slurm.default_time == "12:00:00"
    assert profile.scheduler.slurm.max_time == "UNLIMITED"
    assert profile.scheduler.slurm.over_time_limit == "NONE"
    assert profile.scheduler.slurm.preempt_mode == "REQUEUE"


def test_power_profile_preserves_paths_software_and_accounting_capabilities() -> None:
    profile = load_deployment_profile("power")

    assert str(profile.paths.flows_root) == "/bmd-db/guest/flows"
    assert str(profile.paths.logs_root) == "/bmd-db/guest/logs"
    assert str(profile.paths.potcar_root) == "/bmd-db/potcars"
    assert str(profile.paths.remote_python) == (
        "/bmd/bmdguest/envs/atomate2_remote/bin/python"
    )
    assert str(profile.paths.scheduler_launch_directory) == (
        "/a/home/cc/tree/taucc/enginer/bmdguest"
    )
    assert profile.software.vasp_version == "6.4.1"
    assert [module.name for module in profile.software.modules] == [
        "intel/rocky8-oneAPI-2023",
        "vasp/rocky8-intel-6.4.1",
    ]
    assert "do not identify the current host OS" in profile.software.module_build_label_note
    assert profile.host.os_version == "9.8"
    assert profile.scheduler.slurm is not None
    assert profile.scheduler.slurm.step_accounting_supported is True
    assert {
        "JobID",
        "JobIDRaw",
        "State",
        "ExitCode",
        "Elapsed",
        "ReqMem",
        "MaxRSS",
        "MaxVMSize",
        "AveRSS",
        "AllocTRES",
        "ReqTRES",
        "NCPUS",
        "NNodes",
        "NodeList",
        "Reason",
    } == set(profile.scheduler.slurm.sacct_fields)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("schema", "other.schema", "unsupported deployment profile schema"),
        ("schema_version", 2, "unsupported deployment profile schema_version"),
        ("deployment_id", "power;id", "deployment_id is invalid"),
    ),
)
def test_profile_schema_and_identity_validation(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = power_payload()
    payload[field] = value

    with pytest.raises(DeploymentProfileError, match=message):
        parse_deployment_profile(payload)


def test_profile_schema_does_not_require_slurm_capabilities_for_other_schedulers() -> None:
    payload = power_payload()
    payload["deployment_id"] = "future"
    payload["expected"]["scheduler"] = "pbs"
    payload["recorded"]["scheduler"].pop("slurm")

    profile = parse_deployment_profile(payload)

    assert profile.expected.scheduler == "pbs"
    assert profile.scheduler.slurm is None


def test_profile_loading_is_read_only() -> None:
    profile_file = resources.files("bmd_agent").joinpath(
        "deployment_profiles",
        "power.toml",
    )
    before = profile_file.read_bytes()

    load_deployment_profile("power")

    assert profile_file.read_bytes() == before


def test_power_profile_contains_no_secret_material() -> None:
    text = resources.files("bmd_agent").joinpath(
        "deployment_profiles",
        "power.toml",
    ).read_text(encoding="utf-8")
    lowered = text.lower()

    assert "begin openssh private key" not in lowered
    assert "begin rsa private key" not in lowered
    assert "password" not in lowered
    assert "api_key" not in lowered
    assert "access_token" not in lowered


def test_resolved_context_composes_profile_with_external_acquisition_config() -> None:
    registry = parse_resources(registry_config())

    context = resolve_deployment_context(registry, cluster_key="powerslurm")

    assert context.profile is not None
    assert context.deployment_id == "power"
    assert context.cluster is registry.clusters["powerslurm"]
    assert context.cluster.ssh_host == "powerslurm-bmdguest"
    assert context.cluster.allowed_remote_roots[0].as_posix() == "/bmd-db/guest/flows"
    assert context.profile.paths.potcar_root.as_posix() == "/bmd-db/potcars"
    assert context.profile.paths.potcar_root not in context.cluster.allowed_remote_roots


def test_unknown_deployment_remains_generic_and_supported() -> None:
    registry = parse_resources(registry_config(deployment_profile=None))

    context = resolve_deployment_context(registry)

    assert context.profile is None
    assert context.cluster is not None
    assert context.deployment_id is None
    assert context.limitations == ("no known deployment profile is configured",)


def test_live_mismatch_is_drift_not_profile_mutation_or_failure() -> None:
    profile = load_deployment_profile("power")
    observation = DeploymentLiveObservation(
        scheduler="slurm",
        os_family="rocky",
        os_version="9.9",
        scheduler_version="26.0.0",
    )

    drift = compare_deployment_observation(profile, observation)

    assert [(item.field, item.reference_kind) for item in drift] == [
        ("os_version", "recorded"),
        ("scheduler_version", "recorded"),
    ]
    assert all(item.status == "deployment_drift_observed" for item in drift)
    assert profile.host.os_version == "9.8"


def test_deployment_context_serialization_is_json_safe_and_separates_evidence() -> None:
    registry = parse_resources(registry_config())
    context = resolve_deployment_context(
        registry,
        live_observation=DeploymentLiveObservation(scheduler_version="26.0.0"),
    )

    payload = serialize_deployment_context(context)
    encoded = json.dumps(payload, allow_nan=False)

    assert encoded
    assert payload["profile"]["evidence_type"] == "deployment_profile"
    assert payload["live_observation"]["evidence_type"] == "live_observation"
    assert payload["acquisition"]["ssh_host"] == "powerslurm-bmdguest"
    assert payload["drift"][0]["status"] == "deployment_drift_observed"


def test_parser_does_not_mutate_input_payload() -> None:
    payload = power_payload()
    before = deepcopy(payload)

    parse_deployment_profile(payload)

    assert payload == before
