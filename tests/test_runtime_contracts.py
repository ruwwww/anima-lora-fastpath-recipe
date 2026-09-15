import json

import pytest

from runtime.engine_key import build_engine_signature
from runtime.schema import JobSpec, JobState, validate_job_spec


def test_job_spec_requires_job_id_and_trainer_argv():
    spec = validate_job_spec(
        {
            "schema_version": 1,
            "job_id": "job-a",
            "priority": 10,
            "trainer_argv": ["--network_dim=16"],
        }
    )

    assert isinstance(spec, JobSpec)
    assert spec.job_id == "job-a"
    assert spec.priority == 10
    assert spec.engine_policy == "allow_build"


def test_job_spec_rejects_shell_string_and_unknown_engine_policy():
    with pytest.raises(ValueError, match="trainer_argv"):
        validate_job_spec({"job_id": "job-a", "trainer_argv": "--network_dim=16"})

    with pytest.raises(ValueError, match="engine_policy"):
        validate_job_spec(
            {
                "job_id": "job-a",
                "trainer_argv": ["--network_dim=16"],
                "engine_policy": "magic",
            }
        )


def test_engine_signature_ignores_job_only_values_and_is_stable():
    base = {
        "model_sha256": "model-hash",
        "engine_commit": "abc123",
        "torch_version": "2.13.0+cu130",
        "cuda_version": "13.0",
        "triton_version": "3.7.1",
        "gpu": {"name": "RTX 5060 Ti", "capability": "12.0"},
        "dtype": "bf16",
        "network_module": "networks.lora_anima",
        "network_dim": 16,
        "include_patterns": [],
        "exclude_patterns": [],
        "fused_lora": True,
        "fused_mlp": True,
        "fused_mlp_storage": "fp8",
        "fused_mlp_fp8_backend": "triton",
        "fused_mlp_fp8_direct_backward": True,
        "checkpoint_indices": [0],
        "compile": True,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "shape_profile": {"batch_size": 1, "buckets": [[832, 1216]]},
    }
    a = build_engine_signature(base)
    b = build_engine_signature({**base, "job_id": "b", "seed": 99, "output_dir": "/tmp/b"})

    assert a.key == b.key
    assert a.manifest == b.manifest
    json.dumps(a.manifest, sort_keys=True)


def test_engine_signature_changes_for_runtime_relevant_values():
    base = {"model_sha256": "model-hash", "network_dim": 16, "checkpoint_indices": [0]}
    a = build_engine_signature(base)
    b = build_engine_signature({**base, "network_dim": 8})
    c = build_engine_signature({**base, "checkpoint_indices": [0, 4]})

    assert a.key != b.key
    assert a.key != c.key


def test_job_state_values_are_serializable_and_terminal_states_are_explicit():
    assert JobState.QUEUED.value == "queued"
    assert JobState.RUNNING.value == "running"
    assert JobState.SUCCEEDED.value == "succeeded"
    assert JobState.FAILED.value == "failed"
    assert JobState.CANCELLED.value == "cancelled"
    assert {state.value for state in JobState}
