# Anima DiT Fast LoRA Training Recipe (RTX 5060 Ti 16GB)

A production-validated, hardware-optimized training recipe for **Anima Base v1.0** (Cosmos 2 DiT architecture) LoRA finetuning on consumer **16GB VRAM GPUs (NVIDIA GeForce RTX 5060 Ti / RTX 4080 / RTX 4070 Ti Super)**.

By combining the [sd-scripts fork](https://github.com/ruwwww/sd-scripts) features for **explicit-VJP LoRA/MLP kernels**, **row-wise Triton FP8 activation storage**, **direct FP8 MLP backward**, selective checkpointing, and per-block `torch.compile`, the fixed-bucket benchmark reaches **~1,134 ms/step** on the target RTX 5060 Ti. That is about **27–28% faster** than the uncompiled direct-FP8 benchmark at count 8 while using **~12.70 GiB peak allocation** and retaining **~3.07 GiB measured headroom** to the physical GPU limit.

The timings below are steady-state measurements after JIT warmup on one native-resolution bucket. The first pass through each resolution bucket is slower because Inductor compiles the block. A real multi-bucket dataset includes those compilation passes and bucket-shape overhead: the 15-image Shemira validation run measured **~1.41 s/step overall** and spent **~44 s on the first compile-heavy step**. Do not use 1.134 s/step as an end-to-end ETA for a new multi-bucket dataset.

## Important: invalidate pre-fix direct-FP8 outputs

An early revision of `sd-scripts` had a correctness bug in the direct-FP8 MLP backward path: the second MLP LoRA down-gradient used the forward rank activation instead of its upstream gradient. Outputs trained with that revision can load successfully in ComfyUI while learning almost nothing. The regression is fixed in the current fork and covered by `tests/test_anima_fused_paths.py`; retrain adapters made before the fix.

The fixed 20-step Shemira A/B check produced mean effective LoRA delta **0.0005286** versus **0.0005362** for the unfused reference (about **1.4%** difference), while the pre-fix fast path produced only **0.0000067**.

---

## The Current Fast Path: Compile First, Then Spend the VRAM Budget

Profiling the Anima DiT graph ($d=2048$, 28 blocks, 3,952 patch tokens at native 832x1216) on the RTX 5060 Ti reveals that the old count-12-only recipe left useful performance on the table:

1. **Uncompiled FP8 direct backward, count 8/28:**
   - Stable at **~1,566 ms/step** and **~12.94 GiB peak allocation**.
   - Direct backward avoids materializing the full BF16 GELU activation, but remaining transformer checkpointing still costs time.
2. **Per-block `torch.compile`, count 1/28:**
   - Stable at **~1,134 ms/step** and **~12.70 GiB peak allocation**.
   - Only one transformer block is recomputed; compiled blocks reuse optimized graphs.
3. **Compiled count 0/28:**
   - Faster at **~1,088 ms/step**, but leaves only **~2.31 GiB actual headroom**.
   - It is rejected for unattended training because allocator and bucket variation can trigger OOM.

### The Measured Sweet Spots

Because each block in the Anima model (`library/anima_models.py`) maintains an independent checkpointing gate, the recipe can trade speed for memory deterministically:

- **Fastest safe:** per-block compile + direct FP8 + **1/28 checkpoint**.
- **Conservative compiled:** per-block compile + direct FP8 + **4/28 checkpoints**.
- **No compile fallback:** direct FP8 + **8/28 checkpoints**.
- **Research only:** compiled count 0/28; do not use as the unattended default.

---

## Empirical Benchmark & Ablation (RTX 5060 Ti 16GB)

Measurements captured on NVIDIA RTX 5060 Ti (36 SMs, sm_120, PyTorch 2.13.0+cu130, CUDA 13.0, bf16, rank 16 LoRA, 3,952 patch tokens):

| Configuration | Checkpointed Blocks | Backward Latency | Step Latency | Throughput | Peak VRAM | Actual Headroom | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Compiled count 0 | 0 / 28 | 639 ms | **1,088 ms** | 0.92 it/s | 13.48 GiB | 2.31 GiB | Reject |
| **★ Compiled count 1** | **1 / 28** | **679 ms** | **1,134 ms** | **0.88 it/s** | **12.70 GiB** | **3.07 GiB** | **Fastest Safe** |
| Compiled count 4 | 4 / 28 | 756 ms | 1,223 ms | 0.82 it/s | 11.83 GiB | 3.93 GiB | Conservative |
| Compiled count 8 | 8 / 28 | 868 ms | 1,354 ms | 0.74 it/s | 10.67 GiB | 5.06 GiB | Maximum Margin |
| Uncompiled direct FP8 count 8 | 8 / 28 | 997 ms | 1,566 ms | 0.64 it/s | 12.94 GiB | 2.84 GiB | Fallback |

*Stability Note:* Compiled count-1 passed a 10-step/5-warmup soak with **0.0 MiB allocated/reserved growth**, finite gradients, and no NaNs. Compiled count-1 versus count-8 had maximum loss difference `0.00258` and maximum relative gradient-norm difference `0.15%` over the same short trajectory.

## Recommended Fast Recipe

Use this for a fixed target GPU/resolution after validating one short run:

```bash
python sd-scripts/anima_train_network.py \
  ... \
  --selective_checkpointing=count \
  --checkpoint_blocks=1 \
  --compile \
  --compile_mode=default \
  --compile_cache_size_limit=32 \
  --fused_lora \
  --fused_mlp \
  --fused_mlp_storage=fp8 \
  --fused_mlp_fp8_backend=triton \
  --fused_mlp_fp8_direct_backward
```

For multi-resolution bucket training or a smaller GPU margin, change only:

```bash
--checkpoint_blocks=4
```

If `torch.compile` fails, remove the three compile flags and use the fallback
recipe with `--checkpoint_blocks=8`. Do not enable `--fused_mlp_fp8_input` in
the default recipe; it saves only about 100–130 MiB and adds reduced precision
to the LoRA input-gradient path.

---

## Requirements & Fork Setup

Clone the optimized fork containing the fused FP8/direct-backward and compile-compatible training controls:

```bash
git clone https://github.com/ruwwww/sd-scripts.git sd-scripts
cd sd-scripts
pip install -r requirements.txt
```

## Queueing multiple trainings with a warm Python engine

For high GPU occupancy, the repository also includes a conservative SQLite
queue and persistent Python worker in [`runtime/`](runtime/README.md). It keeps
the trainer CLI unchanged while reusing the Python/TorchInductor process for
jobs with the same deterministic `engine_key`; jobs with different topology
profiles are isolated automatically.

```bash
export PYTHONPATH="$PWD"
python -m runtime --database=/jobs/runtime.sqlite3 submit job.json --preflight
python -m runtime --database=/jobs/runtime.sqlite3 run \
  --trainer=/workspace/sd-scripts/anima_train_network.py \
  --workdir=/workspace/sd-scripts \
  --engine-cache-root=/jobs/engine-cache \
  --warm
```

The warm layer currently reuses interpreter/import/compiler state, not the
model or optimizer objects. This is intentional: the existing one-shot
trainer still owns those lifecycles, while the runtime provides the stable
queue and compatibility boundary for the next model-session phase.

---

## Recommended Training Recipes

### Recipe A: Adaptive Prodigy (Zero-Guesswork Learning Rate)
Recommended for character and style LoRAs where manual learning rate tuning is undesirable.

```bash
python sd-scripts/anima_train_network.py \
  --pretrained_model_name_or_path="/path/to/anima-base-v1.0.safetensors" \
  --train_data_dir="/path/to/data_root" \
  --output_dir="./output_lora" \
  --output_name="anima_lora_prodigy" \
  --network_module="networks.lora_anima" \
  --network_dim=16 \
  --network_alpha=16.0 \
  --network_train_unet_only \
  --selective_checkpointing="count" \
  --checkpoint_blocks=1 \
  --compile \
  --compile_mode="default" \
  --compile_cache_size_limit=32 \
  --fused_lora \
  --fused_mlp \
  --fused_mlp_storage="fp8" \
  --fused_mlp_fp8_backend="triton" \
  --fused_mlp_fp8_direct_backward \
  --attn_mode="torch" \
  --optimizer_type="Prodigy" \
  --learning_rate=1.0 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=10 \
  --optimizer_args weight_decay=0.1 decouple=True use_bias_correction=True d_coef=1.0 \
  --max_train_epochs=5 \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --cache_latents \
  --cache_latents_to_disk \
  --vae="/path/to/qwen_image_vae.safetensors" \
  --qwen_image_vae_2d \
  --text_encoder="/path/to/qwen_3_06b_base.safetensors" \
  --qwen3 \
  --resolution="832,1216" \
  --enable_bucket \
  --caption_extension=".txt" \
  --caption_tag_dropout_rate=0.7 \
  --keep_tokens=1
```

### Recipe B: Legacy Lightweight AdamW8bit (Lowest-Risk Fallback)
Keep this conservative variant for minimal adapter size (~13.8 MB with `dim=8`) or
when the compiled fast path is unavailable. It intentionally retains 12/28
checkpoints and excludes MLP LoRA; use Recipe A for the measured fastest path.

```bash
python sd-scripts/anima_train_network.py \
  ... \
  --network_dim=8 \
  --network_alpha=1.0 \
  --network_args "exclude_patterns=['.*mlp.*']" \
  --selective_checkpointing="count" \
  --checkpoint_blocks=12 \
  --optimizer_type="AdamW8bit" \
  --learning_rate=4e-5 \
  --lr_scheduler="cosine" \
  ...
```

---

## Critical Training Guidelines for Anima DiT

1. **Avoid Subword Collision in Trigger Words:**
   Anima uses the Qwen3 text encoder. If you train a variant of a famous character (e.g., Sasha Calle's Supergirl), do **not** include canonical names in the trigger (e.g. `supergirlflsh` breaks into `['sup', 'erg', 'irl', ...]`, injecting strong blonde/skirt base priors). Use synthetic, collision-free tokens like `sashakara`.

2. **Decoupled 2-Token Architecture (Character vs. Attire):**
   Split character identity from outfit:
   - Character Trigger: `sashakara` (locks facial features & hair).
   - Outfit Trigger: `kryptonsuit` (locks the specific suit/cape).
   - Caption non-suit images with `sashakara, casual clothes...` without the suit token. This allows generating alternative outfits (maid, gym, evening gown) without suit leakage.

3. **Counter-Tagging:**
   Explicitly tag contrasting canonical traits (`short dark brown hair, brown eyes`) in the training captions so the adapter actively suppresses conflicting base model priors.

4. **Always Train UNet Only (`--network_train_unet_only`):**
   Training Qwen3 text encoder weights on small datasets causes catastrophic language forgetting and semantic collapse.

---

## Companion Projects

- [ComfyUI-Anima-BaryCache](https://github.com/ruwwww/ComfyUI-Anima-BaryCache): Inference acceleration node using stepwise barycentric extrapolation (2.24x speedup, down to ~8.4s per generation).
- [anima-lora-fastpath-recipe](https://github.com/ruwwww/anima-lora-fastpath-recipe): This training recipe repository.
- [sd-scripts](https://github.com/ruwwww/sd-scripts): Optimized training scripts fork featuring selective gradient checkpointing.
