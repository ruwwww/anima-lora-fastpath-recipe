# Local warm training runtime

This directory contains the first runtime layer around the existing
`sd-scripts/anima_train_network.py` CLI:

```text
JSON jobs → SQLite queue → one GPU worker → persistent Python engine
                                      └→ persistent TorchInductor/Triton cache
```

The warm engine is intentionally conservative. It reuses a Python process and
its imported Torch/Inductor state for jobs with the same `engine_key`; the
trainer still owns model construction, LoRA setup, optimizer setup, and the
training loop. This means the feature is compatible with the current trainer
without pretending that model/optimizer state is already safely reusable.

## Job format

`trainer_argv` is passed as an argument vector, never through a shell. Put all
graph/topology-affecting values in `metadata.engine_config`; the CLI derives a
stable 32-character engine key when `--engine-key` is omitted.

```json
{
  "schema_version": 1,
  "job_id": "portrait-001",
  "priority": 10,
  "engine_policy": "allow_build",
  "trainer_argv": [
    "--pretrained_model_name_or_path=/models/anima.safetensors",
    "--dataset_config=/jobs/portrait.toml",
    "--output_dir=/jobs/out/portrait-001",
    "--network_dim=16",
    "--max_train_steps=800"
  ],
  "metadata": {
    "engine_config": {
      "trainer": "anima",
      "model": "/models/anima.safetensors",
      "resolution": [832, 1216],
      "network_dim": 16,
      "mixed_precision": "bf16",
      "compile": true,
      "compile_mode": "default",
      "fused_lora": true,
      "fused_mlp": true,
      "fused_mlp_storage": "fp8",
      "fused_mlp_fp8_backend": "triton",
      "fused_mlp_fp8_direct_backward": true,
      "checkpoint_blocks": 1
    }
  }
}
```

Do not put per-job values such as output paths, seed, optimizer settings, or
step count in `engine_config`. They are excluded by
`runtime.engine_key.JOB_ONLY_KEYS`; changing model, resolution, rank, dtype,
kernel recipe, or compile settings must change the key.

Policies:

- `allow_build`: use a matching warm engine when available; compile/cache as
  needed.
- `warm_only`: fail fast unless a non-empty engine key is supplied.
- `cold`: restart the persistent engine after the job.

## Queue commands

From this repository:

```bash
export PYTHONPATH="$PWD"

python -m runtime --database=/jobs/runtime.sqlite3 submit portrait-001.json --preflight
python -m runtime --database=/jobs/runtime.sqlite3 list

python -m runtime --database=/jobs/runtime.sqlite3 run \
  --trainer=/workspace/sd-scripts/anima_train_network.py \
  --workdir=/workspace/sd-scripts \
  --engine-cache-root=/jobs/engine-cache \
  --warm
```

Use `--once` for an integration check. `runtime/worker.py` sends SIGINT when a
running job is cancelled; the current job is then marked cancelled and its
engine is discarded because CUDA/Python state after an interrupted trainer is
not considered reusable.

## What this version does and does not optimize

It removes repeated interpreter startup and gives compatible jobs one stable
owner for TorchInductor/Triton cache directories. It does not yet keep the
Anima DiT, LoRA modules, optimizer, or dataloader resident between jobs. That
second phase requires splitting `NetworkTrainer.train()` into a static model
phase and a job-scoped optimizer/data phase, with explicit in-place adapter
reset/load semantics. Until that contract is proven, the runtime deliberately
uses the existing trainer lifecycle and keeps a cold fallback.

The next safe optimization is to add a trainer-native session API that reuses
only the frozen model and compiled block graph, while rebuilding optimizer,
scheduler, dataloader, and job hooks per request. The queue protocol already
provides the engine-key boundary needed for that change.
