# Anima DiT Persistent Runtime Daemon & Session Architecture

This document describes the production daemon and in-VRAM persistent model session architecture for Anima Cosmos 2 DiT LoRA training on the NVIDIA GeForce RTX 5060 Ti 16GB.

---

## 1. Architecture Overview

```text
┌────────────────────────────────────────────────────────────────────────┐
│ GPU VRAM (RTX 5060 Ti 16GB)                                            │
│                                                                        │
│  [STATIC MODEL PLANE - Resides Permanently (~7.5 GiB)]                 │
│  ├── Frozen Anima DiT Weights (bfloat16, 28 transformer blocks)        │
│  ├── Qwen3-0.6B Base Text Encoder (bfloat16, eval mode)                │
│  ├── Qwen-Image 2D VAE (bfloat16, eval mode)                           │
│  └── Pre-compiled TorchInductor Graph Artifacts (28 blocks, fused FP8) │
│                                                                        │
│  [DYNAMIC JOB PLANE - Created & Destroyed Per-Job (~4.5 GiB)]          │
│  ├── LoRA Adapter Weights (rank=16, ~44 MB, reset in-place)            │
│  ├── Prodigy Optimizer State (first/second moments, D-adaptation)      │
│  ├── Micro-batch Activations & Triton FP8 Cache buffers                │
│  └── DataLoader & Bucketed Latent Tensors                              │
└────────────────────────────────────────────────────────────────────────┘
```

### Key Performance Benefits
1. **Zero Cold-Boot Overhead on Subsequent Jobs**: The 4GB DiT weights, text encoder, and VAE are loaded into VRAM only once on job 1.
2. **Zero JIT Recompilation**: In-place parameter re-initialization (`lora_down` via Kaiming uniform, `lora_up` via zeros) preserves Dynamo module identity and tensor strides, avoiding the 44-second PyTorch Inductor tracing tax.
3. **Execution Latency**: 2-step smoke/bluff jobs complete in **~2.5 seconds total**, compared to ~1.5 minutes with cold starts.

---

## 2. Managing the Background Daemon

### A. Environment Paths & Prerequisites
* **Python Virtualenv**: `/home/kuroko/.conda/envs/ai/bin/python`
* **sd-scripts Path**: `/mnt/data/finetune-anima/sd-scripts`
* **Recipe Path**: `/mnt/data/finetune-anima/anima-lora-fastpath-recipe`
* **Default Database**: `/mnt/data/finetune-anima/runtime_daemon.sqlite3`
* **Engine Cache Root**: `/mnt/data/finetune-anima/engine_cache`

### B. Launching the Daemon Worker
To run the persistent worker (background or systemd service):

```bash
PYTHONPATH="/mnt/data/finetune-anima/sd-scripts:/mnt/data/finetune-anima/anima-lora-fastpath-recipe" \
/home/kuroko/.conda/envs/ai/bin/python -m runtime \
  --database=/mnt/data/finetune-anima/runtime_daemon.sqlite3 \
  run \
  --trainer=/mnt/data/finetune-anima/sd-scripts/anima_train_network.py \
  --workdir=/mnt/data/finetune-anima/sd-scripts \
  --engine-cache-root=/mnt/data/finetune-anima/engine_cache \
  --warm \
  --poll-seconds=1.0
```

---

## 3. Submitting Training Jobs

Jobs are submitted via JSON specifications.

### A. Job Specification Template (`job.json`)
```json
{
  "schema_version": 1,
  "job_id": "character-001",
  "priority": 10,
  "engine_policy": "allow_build",
  "trainer_argv": [
    "--pretrained_model_name_or_path=/home/kuroko/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
    "--train_data_dir=/mnt/data/finetune-anima/data_root_character/10_character",
    "--output_dir=/mnt/data/finetune-anima/output_lora/character_001",
    "--output_name=character_001_lora",
    "--network_module=networks.lora_anima",
    "--network_dim=16",
    "--network_alpha=16.0",
    "--network_train_unet_only",
    "--selective_checkpointing=count",
    "--checkpoint_blocks=1",
    "--compile",
    "--compile_mode=default",
    "--compile_cache_size_limit=32",
    "--fused_lora",
    "--fused_mlp",
    "--fused_mlp_storage=fp8",
    "--fused_mlp_fp8_backend=triton",
    "--fused_mlp_fp8_direct_backward",
    "--attn_mode=torch",
    "--optimizer_type=Prodigy",
    "--learning_rate=1.0",
    "--lr_scheduler=cosine",
    "--lr_warmup_steps=10",
    "--optimizer_args", "weight_decay=0.1", "decouple=True", "use_bias_correction=True", "d_coef=1.0",
    "--max_train_epochs=5",
    "--save_every_n_epochs=1",
    "--mixed_precision=bf16",
    "--save_precision=bf16",
    "--seed=2026",
    "--cache_latents",
    "--cache_latents_to_disk",
    "--vae=/home/kuroko/ComfyUI/models/vae/qwen_image_vae.safetensors",
    "--qwen_image_vae_2d",
    "--qwen3=/mnt/data/models/text_encoders/qwen_3_06b_base.safetensors",
    "--resolution=832,1216",
    "--enable_bucket",
    "--min_bucket_reso=512",
    "--max_bucket_reso=1536",
    "--bucket_reso_steps=32",
    "--caption_extension=.txt",
    "--caption_tag_dropout_rate=0.05",
    "--caption_dropout_rate=0.1",
    "--shuffle_caption",
    "--keep_tokens=1",
    "--persistent_session"
  ],
  "metadata": {
    "engine_config": {
      "trainer": "anima",
      "model": "/home/kuroko/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors",
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

### B. CLI Operations
```bash
# Submit job with preflight validation
PYTHONPATH="/mnt/data/finetune-anima/anima-lora-fastpath-recipe" \
/home/kuroko/.conda/envs/ai/bin/python -m runtime \
  --database=/mnt/data/finetune-anima/runtime_daemon.sqlite3 \
  submit job.json --preflight

# List all queued, running, and finished jobs
python -m runtime --database=... list

# Check detailed status or errors for a job
python -m runtime --database=... status <job_id>

# Cancel a queued or currently executing job
python -m runtime --database=... cancel <job_id>
```

---

## 4. Post-Processing & Verification Workflow

After the daemon marks a job as `succeeded`:

### A. Convert LoRA to ComfyUI Cosmos DiT Format
```bash
/home/kuroko/.conda/envs/ai/bin/python /mnt/data/finetune-anima/sd-scripts/networks/convert_anima_lora_to_comfy.py \
  /mnt/data/finetune-anima/output_lora/character_001/character_001_lora.safetensors \
  /home/kuroko/ComfyUI/models/loras/character_001_comfy.safetensors
```

### B. Generate Verification Render
```bash
# Ensure ComfyUI service is active
systemctl --user start comfyui.service

/home/kuroko/.conda/envs/ai/bin/python /mnt/data/finetune-anima/auto_generate.py \
  --lora "character_001_comfy.safetensors" \
  --prompt "<trigger>, @ratatatat74, 1girl, solo, portrait, looking at viewer, masterpiece, best quality" \
  --prefix "character_001_verify" \
  --steps 30 \
  --strength 1.0 \
  --seed 42
```
Outputs are saved to `/home/kuroko/ComfyUI/output/character_001_verify_00001_.png`.
