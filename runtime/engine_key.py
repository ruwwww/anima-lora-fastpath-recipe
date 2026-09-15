"""Deterministic compatibility keys for reusable compiled engine profiles."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


# These values identify a job, not the compiled model topology.
JOB_ONLY_KEYS = frozenset(
    {
        "job_id",
        "priority",
        "trainer_argv",
        "engine_policy",
        "metadata",
        "seed",
        "output_dir",
        "output_name",
        "resume",
        "network_weights",
        "max_train_steps",
        "max_train_epochs",
        "learning_rate",
        "unet_lr",
        "text_encoder_lr",
        "lr_scheduler",
        "lr_scheduler_args",
        "optimizer_type",
        "optimizer_args",
        "save_every_n_steps",
        "save_every_n_epochs",
        "save_state",
        "save_state_on_train_end",
        "logging_dir",
        "wandb_run_name",
    }
)


@dataclass(frozen=True)
class EngineSignature:
    key: str
    manifest: dict[str, Any]


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"engine signature value is not JSON serializable: {type(value).__name__}")


def build_engine_signature(config: Mapping[str, Any]) -> EngineSignature:
    """Build a stable key while excluding values that vary per training job."""

    if not isinstance(config, Mapping):
        raise TypeError("engine signature config must be a mapping")

    manifest = {
        str(key): _canonical(value)
        for key, value in config.items()
        if str(key) not in JOB_ONLY_KEYS
    }
    manifest = _canonical(manifest)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    key = hashlib.sha256(encoded).hexdigest()[:32]
    return EngineSignature(key=key, manifest=manifest)
