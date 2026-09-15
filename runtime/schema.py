"""Stable, dependency-free contracts shared by the scheduler and workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class JobState(str, Enum):
    QUEUED = "queued"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


_ENGINE_POLICIES = frozenset({"warm_only", "allow_build", "cold"})


@dataclass(frozen=True)
class JobSpec:
    """Validated job input; ``trainer_argv`` remains the trainer's source of truth."""

    job_id: str
    trainer_argv: tuple[str, ...]
    schema_version: int = 1
    priority: int = 0
    engine_policy: str = "allow_build"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "priority": self.priority,
            "trainer_argv": list(self.trainer_argv),
            "engine_policy": self.engine_policy,
            "metadata": dict(self.metadata),
        }


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def validate_job_spec(raw: Mapping[str, Any]) -> JobSpec:
    """Validate an external JSON-like mapping at the scheduler boundary."""

    if not isinstance(raw, Mapping):
        raise ValueError("job spec must be an object")

    schema_version = _require_int(raw.get("schema_version", 1), "schema_version")
    if schema_version != 1:
        raise ValueError(f"unsupported schema_version: {schema_version}")

    job_id = raw.get("job_id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("job_id must be a non-empty string")

    trainer_argv = raw.get("trainer_argv")
    if not isinstance(trainer_argv, list) or not trainer_argv or not all(isinstance(arg, str) for arg in trainer_argv):
        raise ValueError("trainer_argv must be a non-empty list of strings")

    priority = _require_int(raw.get("priority", 0), "priority")
    engine_policy = raw.get("engine_policy", "allow_build")
    if engine_policy not in _ENGINE_POLICIES:
        raise ValueError(f"engine_policy must be one of {sorted(_ENGINE_POLICIES)}")

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be an object")

    return JobSpec(
        schema_version=schema_version,
        job_id=job_id.strip(),
        priority=priority,
        trainer_argv=tuple(trainer_argv),
        engine_policy=engine_policy,
        metadata=dict(metadata),
    )
