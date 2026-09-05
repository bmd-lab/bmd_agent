from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Callable, Mapping

from bmd_agent.config import GitRepositoryResource, ResourceRegistry
from bmd_agent.resources.context import EvidenceGap, ScientificContext


SCHEMA_VERSION = 1
EVIDENCE_TYPE = "composition_context"
PRODUCER_NAME = "BMDex"
PRODUCER_MODULE = "tools.composition.context_producer"
BMDEX_COMPOSITION_CONTEXT = "bmdex_composition_context"

Runner = Callable[..., subprocess.CompletedProcess[str]]


class BmdexCompositionError(RuntimeError):
    """Raised when BMDex composition-context evidence is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "producer_failed",
        returncode: int | None = None,
        stderr_summary: str | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.returncode = returncode
        self.stderr_summary = stderr_summary


@dataclass(frozen=True)
class BmdexCompositionEvidence:
    """Validated BMDex composition-context schema-v1 payload."""

    payload: dict[str, Any]

    @property
    def schema_version(self) -> int:
        return int(self.payload["schema_version"])

    @property
    def status(self) -> str:
        return str(self.payload["status"])

    @property
    def evidence_type(self) -> str:
        return str(self.payload["evidence_type"])

    @property
    def producer(self) -> Mapping[str, Any]:
        return self.payload["producer"]

    @property
    def composition(self) -> Mapping[str, Any]:
        return self.payload["composition"]

    @property
    def datasets(self) -> Mapping[str, Any]:
        return self.payload["datasets"]

    @property
    def missing_evidence(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["missing_evidence"])

    @property
    def limitations(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["limitations"])


@dataclass(frozen=True)
class EnrichedScientificContext:
    """ScientificContext plus optional external BMDex composition evidence."""

    base: ScientificContext
    composition_context: BmdexCompositionEvidence | None = None
    evidence_gaps: tuple[EvidenceGap, ...] = ()


def bmdex_repository(registry: ResourceRegistry) -> GitRepositoryResource | None:
    """Return the explicitly configured BMDex repository, if present."""

    return registry.repositories.get("bmdex")


def enrich_scientific_context_with_bmdex(
    context: ScientificContext,
    repository: GitRepositoryResource | None,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> EnrichedScientificContext:
    """Explicitly enrich an existing ScientificContext with BMDex evidence."""

    formula = context.identity.formula
    if not formula:
        return EnrichedScientificContext(
            base=context,
            composition_context=None,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_COMPOSITION_CONTEXT,
                    "formula_unavailable",
                    "ScientificContext identity formula is unavailable.",
                ),
            ),
        )

    if repository is None:
        return EnrichedScientificContext(
            base=context,
            composition_context=None,
            evidence_gaps=(
                EvidenceGap(
                    BMDEX_COMPOSITION_CONTEXT,
                    "missing_bmdex_repository",
                    "BMDex repository configuration is unavailable.",
                ),
            ),
        )

    try:
        evidence = inspect_bmdex_composition_context(
            repository,
            formula,
            runner=runner,
            timeout=timeout,
        )
    except BmdexCompositionError as exc:
        return EnrichedScientificContext(
            base=context,
            composition_context=None,
            evidence_gaps=(_gap_from_error(exc),),
        )

    return EnrichedScientificContext(base=context, composition_context=evidence)


def inspect_bmdex_composition_context(
    repository: GitRepositoryResource,
    formula: str,
    *,
    runner: Runner = subprocess.run,
    timeout: float = 20,
) -> BmdexCompositionEvidence:
    """Invoke BMDex's configured read-only composition-context producer."""

    if not repository.path.is_dir():
        raise BmdexCompositionError(
            f"Configured BMDex checkout does not exist: {repository.path}",
            kind="missing_checkout",
        )

    if repository.capability_python is None:
        raise BmdexCompositionError(
            "BMDex repository configuration is missing capability_python.",
            kind="missing_interpreter",
        )

    if not repository.capability_python.is_file():
        raise BmdexCompositionError(
            f"Configured BMDex capability Python does not exist: {repository.capability_python}",
            kind="missing_interpreter",
        )

    command = [str(repository.capability_python), "-B", "-m", PRODUCER_MODULE]
    request_text = json.dumps({"formula": formula}, sort_keys=True)

    try:
        completed = runner(
            command,
            cwd=repository.path,
            input=request_text,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BmdexCompositionError(
            "BMDex composition-context producer timed out.",
            kind="timeout",
        ) from exc
    except OSError as exc:
        raise BmdexCompositionError(
            f"Could not execute BMDex composition-context producer: {exc}",
            kind="producer_failed",
        ) from exc

    try:
        evidence = parse_bmdex_composition_payload(
            completed.stdout,
            requested_formula=formula,
        )
    except BmdexCompositionError as exc:
        if completed.returncode != 0 and exc.kind in {"malformed_json", "malformed_payload"}:
            raise BmdexCompositionError(
                "BMDex composition-context producer failed without a valid contract response.",
                kind="producer_failed",
                returncode=completed.returncode,
                stderr_summary=_stderr_summary(completed.stderr),
            ) from exc
        raise

    if completed.returncode != 0:
        raise BmdexCompositionError(
            "BMDex composition-context producer returned a successful payload with a nonzero exit.",
            kind="producer_failed",
            returncode=completed.returncode,
            stderr_summary=_stderr_summary(completed.stderr),
        )

    return evidence


def parse_bmdex_composition_payload(
    output: str,
    *,
    requested_formula: str | None = None,
) -> BmdexCompositionEvidence:
    """Parse and validate BMDex composition-context schema v1."""

    try:
        raw_payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise BmdexCompositionError(
            "BMDex composition-context producer emitted malformed JSON.",
            kind="malformed_json",
        ) from exc

    if not isinstance(raw_payload, dict):
        raise BmdexCompositionError(
            "BMDex composition-context payload must be a JSON object.",
            kind="malformed_payload",
        )

    payload = dict(raw_payload)
    _validate_payload(payload, requested_formula=requested_formula)
    if payload["status"] == "error":
        raise _structured_error(payload)
    return BmdexCompositionEvidence(payload=payload)


def _validate_payload(
    payload: Mapping[str, Any],
    *,
    requested_formula: str | None,
) -> None:
    schema_version = payload.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise BmdexCompositionError(
            f"Unsupported BMDex composition-context schema_version: {schema_version!r}",
            kind="unsupported_schema",
        )

    if payload.get("evidence_type") != EVIDENCE_TYPE:
        raise BmdexCompositionError(
            "BMDex composition-context payload has an unexpected evidence_type.",
            kind="malformed_payload",
        )

    status = payload.get("status")
    if status not in {"ok", "error"}:
        raise BmdexCompositionError(
            f"BMDex composition-context payload has unexpected status: {status!r}",
            kind="malformed_payload",
        )

    _validate_producer(_required_mapping(payload, "producer"))

    if status == "error":
        _required_mapping(payload, "error")
        return

    composition = _required_mapping(payload, "composition")
    _required_mapping(payload, "datasets")
    _required_list(payload, "missing_evidence")
    _required_list(payload, "limitations")

    supplied_formula = composition.get("supplied_formula")
    if not isinstance(supplied_formula, str) or not supplied_formula:
        raise BmdexCompositionError(
            "BMDex composition-context composition.supplied_formula must be a non-empty string.",
            kind="malformed_payload",
        )

    if requested_formula is not None and supplied_formula != requested_formula:
        raise BmdexCompositionError(
            "BMDex composition-context payload formula does not match the request.",
            kind="malformed_payload",
        )


def _validate_producer(producer: Mapping[str, Any]) -> None:
    name = producer.get("name")
    if name != PRODUCER_NAME:
        raise BmdexCompositionError(
            "BMDex composition-context producer.name was not BMDex.",
            kind="malformed_payload",
        )


def _structured_error(payload: Mapping[str, Any]) -> BmdexCompositionError:
    error = payload.get("error")
    code = None
    message = None
    if isinstance(error, Mapping):
        code = error.get("code")
        message = error.get("message")

    details = []
    if isinstance(code, str) and code:
        details.append(code)
    if isinstance(message, str) and message:
        details.append(_bounded_text(message))

    suffix = f": {' - '.join(details)}" if details else "."
    return BmdexCompositionError(
        f"BMDex composition-context producer returned a structured error{suffix}",
        kind="structured_producer_error",
    )


def _required_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise BmdexCompositionError(
            f"BMDex composition-context payload missing object field: {key}",
            kind="malformed_payload",
        )
    return value


def _required_list(payload: Mapping[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise BmdexCompositionError(
            f"BMDex composition-context payload missing list field: {key}",
            kind="malformed_payload",
        )
    return value


def _gap_from_error(error: BmdexCompositionError) -> EvidenceGap:
    reason = str(error)
    if error.returncode is not None:
        reason = f"{reason} exit {error.returncode}."
    if error.stderr_summary:
        reason = f"{reason} {error.stderr_summary}"
    return EvidenceGap(BMDEX_COMPOSITION_CONTEXT, error.kind, _bounded_text(reason))


def _stderr_summary(stderr: object) -> str | None:
    if stderr is None:
        return None
    text = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith(("traceback", "file \"", "raise ")):
            continue
        return _bounded_text(stripped)
    return None


def _bounded_text(text: str, limit: int = 240) -> str:
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."
