from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

from bmd_agent.config import GitRepositoryResource, SlurmClusterResource
from bmd_agent.resources.input_reference import (
    InputReferenceError,
    InputReferenceResponse,
    generate_input_reference,
)
from bmd_agent.resources.run import parse_incar_contents
from bmd_agent.resources.vasp import (
    authorize_remote_path,
    build_remote_file_path,
    parse_poscar,
    remote_file_exists,
    retrieve_remote_file,
)


COMPLIANT = "COMPLIANT"
SUPPORTED_BUT_NONSTANDARD = "SUPPORTED BUT NONSTANDARD"
UNSUPPORTED = "UNSUPPORTED"
INSUFFICIENT_INFORMATION = "INSUFFICIENT INFORMATION"

SUPPLIED_INPUT = "supplied_input"
BMD_COMPUTE_REFERENCE = "bmd_compute_reference"
AGENT_COMPARISON = "agent_comparison"
UNAVAILABLE = "unavailable"

MATCH = "match"
DIFFERS_FROM_BMD_REFERENCE = "differs_from_bmd_reference"
SUPPLIED_EXTRA = "supplied_extra"
REFERENCE_MISSING_FROM_SUPPLIED = "reference_missing_from_supplied"

RunnerBytes = Callable[..., subprocess.CompletedProcess[bytes]]
RunnerText = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class InputFileObservation:
    label: str
    path: str
    present: bool
    evidence_type: str = SUPPLIED_INPUT
    error: str | None = None


@dataclass(frozen=True)
class ProposedInputObservation:
    directory: str
    files: Mapping[str, InputFileObservation]
    evidence_type: str = SUPPLIED_INPUT
    formula: str | None = None
    reduced_formula: str | None = None
    site_count: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class InputReferenceObservation:
    status: str
    evidence_type: str = BMD_COMPUTE_REFERENCE
    schema_version: int | None = None
    reference_phase: str | None = None
    producer_repository: str | None = None
    producer_commit: str | None = None
    producer_dirty: bool | None = None
    workflow_label: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    producer_stderr_summary: str | None = None


@dataclass(frozen=True)
class InputSettingComparison:
    setting: str
    supplied_value: Any
    reference_value: Any
    status: str
    evidence_type: str = AGENT_COMPARISON


@dataclass(frozen=True)
class KpointsComparison:
    status: str
    evidence_type: str = AGENT_COMPARISON
    supplied_summary: str | None = None
    reference_summary: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class InputCheckObservation:
    overall_status: str
    remote_directory: str
    stage_type: str
    theory: str
    resources: Mapping[str, int]
    proposed: ProposedInputObservation
    reference: InputReferenceObservation
    incar_comparisons: tuple[InputSettingComparison, ...] = ()
    kpoints_comparison: KpointsComparison | None = None
    limitations: tuple[str, ...] = ()
    evidence_type: str = AGENT_COMPARISON

    @property
    def matching_incar_settings(self) -> int:
        return sum(1 for item in self.incar_comparisons if item.status == MATCH)

    @property
    def nonmatching_incar_settings(self) -> tuple[InputSettingComparison, ...]:
        return tuple(item for item in self.incar_comparisons if item.status != MATCH)


@dataclass(frozen=True)
class _ReadInput:
    observations: Mapping[str, InputFileObservation]
    contents: Mapping[str, bytes]


@dataclass(frozen=True)
class _ParsedKpoints:
    summary: str | None
    comparable: Mapping[str, Any] | None
    error: str | None = None


def check_remote_input_directory(
    cluster: SlurmClusterResource,
    repository: GitRepositoryResource,
    remote_directory: str,
    *,
    stage: str,
    theory: str,
    nodes: int,
    ntasks: int,
    mem_gb: int,
    remote_runner: RunnerBytes = subprocess.run,
    producer_runner: RunnerText = subprocess.run,
    timeout: float = 20,
) -> InputCheckObservation:
    """Compare proposed remote VASP inputs with BMD Compute's generated reference."""

    resources = {
        "nodes": _positive_int(nodes, "nodes"),
        "ntasks": _positive_int(ntasks, "ntasks"),
        "mem_gb": _positive_int(mem_gb, "mem_gb"),
    }
    stage = str(stage).strip()
    theory = str(theory).strip()
    directory = authorize_remote_path(
        remote_directory,
        allowed_roots=cluster.allowed_remote_roots,
    )
    read_inputs = _read_named_inputs(
        cluster.ssh_host,
        str(directory),
        allowed_roots=cluster.allowed_remote_roots,
        runner=remote_runner,
        timeout=timeout,
    )
    proposed = _proposed_observation(str(directory), read_inputs)

    missing_or_unreadable = [
        f"{name}: {observation.error or 'missing'}"
        for name, observation in read_inputs.observations.items()
        if not observation.present or observation.error
    ]
    if missing_or_unreadable:
        return _insufficient(
            str(directory),
            stage,
            theory,
            resources,
            proposed,
            "Required supplied input evidence is unavailable: "
            + "; ".join(missing_or_unreadable),
        )

    supplied_incar, incar_error = parse_incar_contents(read_inputs.contents["INCAR"])
    if incar_error:
        return _insufficient(
            str(directory),
            stage,
            theory,
            resources,
            proposed,
            f"Supplied INCAR could not be parsed: {incar_error}",
        )

    supplied_kpoints = parse_kpoints_contents(read_inputs.contents["KPOINTS"])
    if supplied_kpoints.error:
        return _insufficient(
            str(directory),
            stage,
            theory,
            resources,
            proposed,
            f"Supplied KPOINTS could not be parsed: {supplied_kpoints.error}",
        )

    if proposed.error:
        return _insufficient(
            str(directory),
            stage,
            theory,
            resources,
            proposed,
            f"Supplied POSCAR could not be parsed: {proposed.error}",
        )

    request = build_input_reference_request(
        read_inputs.contents["POSCAR"].decode("utf-8", "replace"),
        stage=stage,
        theory=theory,
        resources=resources,
    )

    try:
        response = generate_input_reference(
            repository,
            request,
            runner=producer_runner,
            timeout=timeout,
        )
    except InputReferenceError as exc:
        return _insufficient(
            str(directory),
            stage,
            theory,
            resources,
            proposed,
            str(exc),
            reference=InputReferenceObservation(
                status=UNAVAILABLE,
                error_code=exc.kind,
                error_message=str(exc),
                producer_stderr_summary=exc.stderr_summary,
            ),
        )

    reference = _reference_observation(response)
    if response.status == "unsupported":
        return InputCheckObservation(
            overall_status=UNSUPPORTED,
            remote_directory=str(directory),
            stage_type=stage,
            theory=theory,
            resources=resources,
            proposed=proposed,
            reference=reference,
            limitations=_error_limitations(reference),
        )

    if response.status == "error":
        return InputCheckObservation(
            overall_status=INSUFFICIENT_INFORMATION,
            remote_directory=str(directory),
            stage_type=stage,
            theory=theory,
            resources=resources,
            proposed=proposed,
            reference=reference,
            limitations=_error_limitations(reference),
        )

    stages = response.stages
    if len(stages) != 1:
        return _insufficient(
            str(directory),
            stage,
            theory,
            resources,
            proposed,
            "BMD Compute input-reference v1 comparison expected exactly one reference stage.",
            reference=reference,
        )

    reference_stage = stages[0]
    reference_incar = _reference_incar_settings(reference_stage)
    reference_kpoints = _reference_kpoints(reference_stage)
    incar_comparisons = compare_incar_settings(
        supplied_incar,
        reference_incar,
    )
    kpoints_comparison = compare_kpoints(supplied_kpoints, reference_kpoints)

    if kpoints_comparison.status == "unavailable":
        overall = INSUFFICIENT_INFORMATION
    elif (
        any(item.status != MATCH for item in incar_comparisons)
        or kpoints_comparison.status != MATCH
    ):
        overall = SUPPORTED_BUT_NONSTANDARD
    else:
        overall = COMPLIANT

    return InputCheckObservation(
        overall_status=overall,
        remote_directory=str(directory),
        stage_type=stage,
        theory=theory,
        resources=resources,
        proposed=proposed,
        reference=reference,
        incar_comparisons=incar_comparisons,
        kpoints_comparison=kpoints_comparison,
    )


def build_input_reference_request(
    poscar_text: str,
    *,
    stage: str,
    theory: str,
    resources: Mapping[str, int],
) -> dict[str, Any]:
    """Build the narrow single-stage BMD Compute input-reference request."""

    return {
        "structure": {
            "type": "pasted_text",
            "format": "poscar",
            "text": poscar_text,
        },
        "workflow_spec": {
            "stages": [
                {
                    "stage_type": stage,
                    "theory": theory,
                    "modifiers": [],
                    "label": None,
                    "options": {},
                }
            ],
            "label": None,
            "recipe": None,
        },
        "resources": dict(resources),
        "potcar_functional": "PBE_64",
    }


def compare_incar_settings(
    supplied: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[InputSettingComparison, ...]:
    """Compare supplied and BMD-reference INCAR settings semantically."""

    supplied_by_key = {str(key).upper(): value for key, value in supplied.items()}
    reference_by_key = {str(key).upper(): value for key, value in reference.items()}
    comparisons: list[InputSettingComparison] = []

    for setting in sorted(set(supplied_by_key) | set(reference_by_key)):
        supplied_present = setting in supplied_by_key
        reference_present = setting in reference_by_key
        supplied_value = supplied_by_key.get(setting)
        reference_value = reference_by_key.get(setting)

        if supplied_present and reference_present:
            status = (
                MATCH
                if values_match(supplied_value, reference_value)
                else DIFFERS_FROM_BMD_REFERENCE
            )
        elif supplied_present:
            status = SUPPLIED_EXTRA
        else:
            status = REFERENCE_MISSING_FROM_SUPPLIED

        comparisons.append(
            InputSettingComparison(
                setting=setting,
                supplied_value=_json_safe_value(supplied_value),
                reference_value=_json_safe_value(reference_value),
                status=status,
            )
        )

    return tuple(comparisons)


def parse_kpoints_contents(contents: bytes | str) -> _ParsedKpoints:
    """Parse VASP KPOINTS content into a compact comparable representation."""

    text = contents.decode("utf-8") if isinstance(contents, bytes) else contents
    try:
        from pymatgen.io.vasp.inputs import Kpoints
    except Exception as exc:
        return _ParsedKpoints(None, None, str(exc))

    try:
        if hasattr(Kpoints, "from_str"):
            kpoints = Kpoints.from_str(text)
        else:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
                handle.write(text)
                temporary_path = handle.name
            try:
                kpoints = Kpoints.from_file(temporary_path)
            finally:
                Path(temporary_path).unlink(missing_ok=True)
        payload = _json_safe_value(kpoints.as_dict())
    except Exception as exc:
        return _ParsedKpoints(None, None, str(exc))

    return _parsed_kpoints_from_dict(payload)


def compare_kpoints(
    supplied: _ParsedKpoints,
    reference: _ParsedKpoints,
) -> KpointsComparison:
    """Compare supplied and BMD-reference KPOINTS by supported structured fields."""

    if supplied.error:
        return KpointsComparison(
            "unavailable",
            supplied_summary=supplied.summary,
            reference_summary=reference.summary,
            reason=f"supplied KPOINTS could not be parsed: {supplied.error}",
        )
    if reference.error:
        return KpointsComparison(
            "unavailable",
            supplied_summary=supplied.summary,
            reference_summary=reference.summary,
            reason=f"BMD reference KPOINTS could not be parsed: {reference.error}",
        )
    if supplied.comparable is None or reference.comparable is None:
        return KpointsComparison(
            "unavailable",
            supplied_summary=supplied.summary,
            reference_summary=reference.summary,
            reason="KPOINTS mode is not safely comparable in check-input v1.",
        )

    status = MATCH if supplied.comparable == reference.comparable else DIFFERS_FROM_BMD_REFERENCE
    return KpointsComparison(
        status,
        supplied_summary=supplied.summary,
        reference_summary=reference.summary,
    )


def values_match(supplied: Any, reference: Any) -> bool:
    """Return whether two parsed VASP setting values are semantically equivalent."""

    supplied = _expand_compact_numeric_sequence(supplied)
    reference = _expand_compact_numeric_sequence(reference)

    if isinstance(supplied, bool) or isinstance(reference, bool):
        return supplied is reference

    if _is_sequence(supplied) or _is_sequence(reference):
        if not (_is_sequence(supplied) and _is_sequence(reference)):
            return False
        supplied_values = list(supplied)
        reference_values = list(reference)
        if len(supplied_values) != len(reference_values):
            return False
        return all(values_match(left, right) for left, right in zip(supplied_values, reference_values))

    supplied_float = _float_or_none(supplied)
    reference_float = _float_or_none(reference)
    if supplied_float is not None and reference_float is not None:
        return math.isclose(supplied_float, reference_float, rel_tol=0.0, abs_tol=1e-8)

    return str(supplied) == str(reference)


def _read_named_inputs(
    ssh_host: str,
    directory: str,
    *,
    allowed_roots: Sequence[Any],
    runner: RunnerBytes,
    timeout: float,
) -> _ReadInput:
    observations: dict[str, InputFileObservation] = {}
    contents: dict[str, bytes] = {}

    for filename in ("INCAR", "POSCAR", "KPOINTS"):
        path = build_remote_file_path(
            directory,
            filename,
            allowed_roots=allowed_roots,
        )
        try:
            present = remote_file_exists(
                ssh_host,
                path,
                runner=runner,
                timeout=timeout,
            )
            if not present:
                observations[filename] = InputFileObservation(
                    filename,
                    str(path),
                    False,
                    error="missing",
                )
                continue
            contents[filename] = retrieve_remote_file(
                ssh_host,
                path,
                runner=runner,
                timeout=timeout,
            )
            observations[filename] = InputFileObservation(filename, str(path), True)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            observations[filename] = InputFileObservation(
                filename,
                str(path),
                True,
                error=f"could not be read: {exc}",
            )

    return _ReadInput(observations=observations, contents=contents)


def _proposed_observation(
    directory: str,
    read_inputs: _ReadInput,
) -> ProposedInputObservation:
    try:
        structure = parse_poscar(read_inputs.contents.get("POSCAR", b""), source=f"{directory}/POSCAR")
    except Exception as exc:
        return ProposedInputObservation(
            directory=directory,
            files=read_inputs.observations,
            error=str(exc),
        )

    return ProposedInputObservation(
        directory=directory,
        files=read_inputs.observations,
        formula=structure.formula,
        reduced_formula=structure.reduced_formula,
        site_count=structure.sites,
    )


def _insufficient(
    remote_directory: str,
    stage: str,
    theory: str,
    resources: Mapping[str, int],
    proposed: ProposedInputObservation,
    reason: str,
    *,
    reference: InputReferenceObservation | None = None,
) -> InputCheckObservation:
    return InputCheckObservation(
        overall_status=INSUFFICIENT_INFORMATION,
        remote_directory=remote_directory,
        stage_type=stage,
        theory=theory,
        resources=resources,
        proposed=proposed,
        reference=reference or InputReferenceObservation(status=UNAVAILABLE),
        limitations=(reason,),
    )


def _reference_observation(response: InputReferenceResponse) -> InputReferenceObservation:
    producer = response.producer
    source = producer.get("source")
    source = source if isinstance(source, Mapping) else {}
    workflow = response.workflow
    error = response.error
    return InputReferenceObservation(
        status=response.status,
        schema_version=response.payload.get("schema_version"),
        reference_phase=response.reference_phase,
        producer_repository=str(producer.get("repository") or ""),
        producer_commit=_optional_str(source.get("commit")),
        producer_dirty=source.get("dirty") if isinstance(source.get("dirty"), bool) else None,
        workflow_label=_optional_str(workflow.get("label")),
        error_code=_optional_str(error.get("code")),
        error_message=_optional_str(error.get("message")),
    )


def _reference_incar_settings(stage: Mapping[str, Any]) -> Mapping[str, Any]:
    incar = stage.get("incar")
    if not isinstance(incar, Mapping):
        return {}
    settings = incar.get("settings")
    return {
        str(key).upper(): _json_safe_value(value)
        for key, value in settings.items()
    } if isinstance(settings, Mapping) else {}


def _reference_kpoints(stage: Mapping[str, Any]) -> _ParsedKpoints:
    kpoints = stage.get("kpoints")
    if not isinstance(kpoints, Mapping):
        return _ParsedKpoints(None, None, "reference KPOINTS object is unavailable")
    as_dict = kpoints.get("as_dict")
    if isinstance(as_dict, Mapping):
        return _parsed_kpoints_from_dict(_json_safe_value(as_dict))
    text = kpoints.get("text")
    if isinstance(text, str) and text.strip():
        return parse_kpoints_contents(text)
    return _ParsedKpoints(None, None, "reference KPOINTS data is unavailable")


def _parsed_kpoints_from_dict(payload: Any) -> _ParsedKpoints:
    if not isinstance(payload, Mapping):
        return _ParsedKpoints(None, None, "KPOINTS payload is unavailable")
    comparable = _mesh_kpoints_comparable(payload)
    return _ParsedKpoints(
        summary=_kpoints_summary(payload),
        comparable=comparable,
    )


def _mesh_kpoints_comparable(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    style = _kpoints_style(payload)
    if style not in {"gamma", "monkhorst"}:
        return None

    mesh = _first_kpoint_triplet(payload)
    if mesh is None:
        return None

    return {
        "style": style,
        "mesh": mesh,
        "shift": _kpoints_shift(payload),
    }


def _kpoints_summary(payload: Mapping[str, Any]) -> str:
    style = str(payload.get("generation_style") or payload.get("style") or "unknown")
    mesh = _first_kpoint_triplet(payload)
    if mesh is not None:
        return f"{style} {mesh[0]}x{mesh[1]}x{mesh[2]}"
    points = payload.get("kpoints")
    if isinstance(points, list):
        return f"{style} {len(points)} k-points"
    return style


def _kpoints_style(payload: Mapping[str, Any]) -> str:
    return str(payload.get("generation_style") or payload.get("style") or "").lower()


def _first_kpoint_triplet(payload: Mapping[str, Any]) -> tuple[int, int, int] | None:
    kpoints = payload.get("kpoints")
    if not isinstance(kpoints, list) or not kpoints:
        return None
    first = kpoints[0]
    if not isinstance(first, (list, tuple)) or len(first) != 3:
        return None
    try:
        values = tuple(int(float(value)) for value in first)
    except (TypeError, ValueError):
        return None
    if any(value <= 0 for value in values):
        return None
    return values


def _kpoints_shift(payload: Mapping[str, Any]) -> tuple[float, float, float]:
    for key in ("usershift", "kpts_shift", "shift"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                return tuple(float(item) for item in value)
            except (TypeError, ValueError):
                pass
    return (0.0, 0.0, 0.0)


def _error_limitations(reference: InputReferenceObservation) -> tuple[str, ...]:
    message = reference.error_message or reference.error_code
    return (message,) if message else ()


def _positive_int(value: int, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return parsed


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes))


def _expand_compact_numeric_sequence(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    tokens = value.split()
    if not tokens:
        return value
    expanded: list[float] = []
    for token in tokens:
        repeat = 1
        raw_number = token
        if "*" in token:
            pieces = token.split("*", 1)
            if len(pieces) != 2:
                return value
            try:
                repeat = int(pieces[0])
            except ValueError:
                return value
            raw_number = pieces[1]
        number = _float_or_none(raw_number)
        if number is None or repeat < 1:
            return value
        expanded.extend([number] * repeat)
    return expanded
