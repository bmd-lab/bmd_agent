from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Callable, Mapping

from bmd_agent.config import GitRepositoryResource


SCHEMA_VERSION = 1
PRODUCER_MODULE = "backend.calculations.capabilities"
CONTRACT_SCOPE = "BMD Compute executable implementation, not a methodology authority"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class ComputeCapabilityError(RuntimeError):
    """Raised when BMD Compute capability introspection cannot be completed."""


@dataclass(frozen=True)
class ComputeCapabilities:
    """Validated BMD Compute schema-v1 capability payload."""

    payload: dict[str, Any]

    @property
    def source(self) -> Mapping[str, Any]:
        return self.payload["source"]

    @property
    def scope(self) -> str:
        return self.payload["scope"]

    @property
    def capabilities(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["capabilities"])

    @property
    def base_stage_definitions(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["base_stage_definitions"])


def inspect_compute_capabilities(
    repository: GitRepositoryResource,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> ComputeCapabilities:
    """Invoke BMD Compute's configured read-only capability producer."""

    if not repository.path.is_dir():
        raise ComputeCapabilityError(
            f"Configured BMD Compute checkout does not exist: {repository.path}"
        )

    if repository.capability_python is None:
        raise ComputeCapabilityError(
            "BMD Compute repository configuration is missing capability_python."
        )

    if not repository.capability_python.is_file():
        raise ComputeCapabilityError(
            f"Configured BMD Compute capability Python does not exist: {repository.capability_python}"
        )

    try:
        completed = runner(
            [str(repository.capability_python), "-m", PRODUCER_MODULE],
            cwd=repository.path,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )

    except subprocess.TimeoutExpired as exc:
        raise ComputeCapabilityError("BMD Compute capability producer timed out.") from exc

    except subprocess.CalledProcessError as exc:
        details = _first_error_line(exc.stderr)
        message = "BMD Compute capability producer failed."
        if details:
            message = f"{message} {details}"
        raise ComputeCapabilityError(message) from exc

    except OSError as exc:
        raise ComputeCapabilityError(
            f"Could not execute BMD Compute capability producer: {exc}"
        ) from exc

    return parse_capability_payload(completed.stdout)


def parse_capability_payload(output: str) -> ComputeCapabilities:
    """Parse and validate BMD Compute capability schema v1."""

    try:
        raw_payload = json.loads(output)

    except json.JSONDecodeError as exc:
        raise ComputeCapabilityError("BMD Compute capability producer emitted malformed JSON.") from exc

    if not isinstance(raw_payload, dict):
        raise ComputeCapabilityError("BMD Compute capability payload must be a JSON object.")

    payload = dict(raw_payload)
    _validate_payload(payload)
    return ComputeCapabilities(payload=payload)


def supported_capability_pairs(capabilities: ComputeCapabilities) -> tuple[tuple[str, str], ...]:
    """Return producer-declared supported stage/theory pairs."""

    return tuple(
        (str(record["stage_type"]), str(record["theory"]))
        for record in capabilities.capabilities
    )


def _validate_payload(payload: Mapping[str, Any]) -> None:
    schema_version = payload.get("schema_version")

    if schema_version != SCHEMA_VERSION:
        raise ComputeCapabilityError(
            f"Unsupported BMD Compute capability schema_version: {schema_version!r}"
        )

    scope = payload.get("scope")
    if scope != CONTRACT_SCOPE:
        raise ComputeCapabilityError("BMD Compute capability payload has an unexpected scope.")

    _validate_source(_required_mapping(payload, "source"))
    _validate_contract(_required_mapping(payload, "contract"))
    _validate_stage_definitions(_required_list(payload, "base_stage_definitions"))
    _validate_capabilities(_required_list(payload, "capabilities"))


def _validate_source(source: Mapping[str, Any]) -> None:
    _required_str(source, "repository")

    commit = source.get("commit")
    if commit is not None and not isinstance(commit, str):
        raise ComputeCapabilityError("BMD Compute source.commit must be a string or null.")

    dirty = source.get("dirty")
    if dirty is not None and not isinstance(dirty, bool):
        raise ComputeCapabilityError("BMD Compute source.dirty must be true, false, or null.")

    if not isinstance(source.get("provenance_available"), bool):
        raise ComputeCapabilityError("BMD Compute source.provenance_available must be boolean.")

    reason = source.get("unavailable_reason")
    if reason is not None and not isinstance(reason, str):
        raise ComputeCapabilityError(
            "BMD Compute source.unavailable_reason must be a string or null."
        )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    _required_str(contract, "base_stage_definitions")
    _required_str(contract, "capabilities")


def _validate_stage_definitions(stage_definitions: list[Any]) -> None:
    seen: set[str] = set()

    for index, record in enumerate(stage_definitions):
        mapping = _as_mapping(record, f"base_stage_definitions[{index}]")
        stage_type = _required_str(mapping, "stage_type")

        if stage_type in seen:
            raise ComputeCapabilityError(
                f"Duplicate BMD Compute base stage definition: {stage_type!r}"
            )

        seen.add(stage_type)


def _validate_capabilities(capabilities: list[Any]) -> None:
    seen: set[tuple[str, str]] = set()

    for index, record in enumerate(capabilities):
        mapping = _as_mapping(record, f"capabilities[{index}]")
        stage_type = _required_str(mapping, "stage_type")
        theory = _required_str(mapping, "theory")
        supported = mapping.get("theory_supported_for_stage")

        if supported is not True:
            raise ComputeCapabilityError(
                f"BMD Compute capabilities[{index}] is not an explicitly supported capability."
            )

        key = (stage_type, theory)
        if key in seen:
            raise ComputeCapabilityError(
                f"Duplicate BMD Compute capability record: {stage_type!r}/{theory!r}"
            )

        seen.add(key)


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)

    if not isinstance(value, Mapping):
        raise ComputeCapabilityError(f"BMD Compute capability payload missing object field: {key}")

    return value


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)

    if not isinstance(value, list):
        raise ComputeCapabilityError(f"BMD Compute capability payload missing list field: {key}")

    return value


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ComputeCapabilityError(f"BMD Compute {label} must be an object.")

    return value


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)

    if not isinstance(value, str) or not value:
        raise ComputeCapabilityError(f"BMD Compute field must be a non-empty string: {key}")

    return value


def _first_error_line(stderr: str | None) -> str:
    for line in (stderr or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped

    return ""
