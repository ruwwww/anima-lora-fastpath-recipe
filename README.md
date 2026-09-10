# Anima DiT Fast LoRA Training Recipe (RTX 5060 Ti 16GB)

A production-validated, hardware-optimized training recipe for **Anima Base v1.0** (Cosmos 2 DiT architecture) LoRA finetuning on consumer **16GB VRAM GPUs (NVIDIA GeForce RTX 5060 Ti / RTX 4080 / RTX 4070 Ti Super)**.

By leveraging **Selective Block Checkpointing** implemented in our [sd-scripts fork](https://github.com/ruwwww/sd-scripts), this setup cuts training step latency from **2,134 ms down to 1,715 ms (~1.24x speedup)** by recovering **~327 ms of wasted recomputation time**, utilizing **~12.46 GiB VRAM** while leaving a safe **3.34 GiB headroom** to avoid allocator fragmentation.

---

## The Core Bottleneck: Recomputation vs. Idle VRAM

Profiling the Anima DiT graph ($d=2048$, 28 blocks, 3,952 patch tokens at native 832x1216) on the RTX 5060 Ti reveals a stark trade-off:

1. **Full Gradient Checkpointing (28/28 blocks):**
   - Peak VRAM is only **~5.19 GiB**.
   - Over **10.5 GiB of GDDR7 VRAM sits idle and wasted**.
   - The backward pass takes **1,457 ms (71.5% of total step time)** because every single transformer block forward pass must be recomputed from scratch during backward.
2. **Zero Gradient Checkpointing (0/28 blocks):**
   - Forward activation tensors for all 28 blocks accumulate simultaneously.
   - Instantly crashes with **CUDA Out of Memory (OOM)** at **~14.99 GiB** on 16GB devices.

### The Measured Sweet Spot: 12 of 28 Blocks Checkpointed

Because each block in the Anima model (`library/anima_models.py`) maintains an independent checkpointing gate, we can selectively checkpoint a deterministic subset of blocks:
- **12 blocks checkpointed** (`[0, 2, 4, 7, 9, 11, 14, 16, 18, 21, 23, 25]`).
- **16 blocks retain activations in VRAM** (zero backward recomputation for these blocks).

---

## Empirical Benchmark & Ablation (RTX 5060 Ti 16GB)

Measurements captured on NVIDIA RTX 5060 Ti (36 SMs, sm_120, PyTorch 2.13.0+cu130, CUDA 13.0, bf16, rank 16 LoRA, 3,952 patch tokens):

| Configuration | Checkpointed Blocks | Backward Latency | Step Latency | Throughput | Speedup | Peak VRAM | Safety Margin (to 15.8 GiB) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero Checkpointing** | 0 / 28 | — | — | — | — | >15.5 GiB | Negative | **CUDA OOM** |
| **Threshold Minimum** | 7 / 28 | 1,029 ms | 1,612 ms | 0.62 it/s | 1.26x | 14.90 GiB | 0.90 GiB (Critical) | High OOM Risk |
| **Even Count** | 10 / 28 | 1,091 ms | 1,674 ms | 0.60 it/s | 1.22x | 13.43 GiB | 2.37 GiB | Borderline |
| **★ Optimal Sweet Spot** | **12 / 28** | **1,132 ms** | **1,715 ms** | **0.58 it/s** | **1.24x** | **12.46 GiB** | **3.34 GiB** | **Safe & Stable** |
| **Interleaved 50%** | 14 / 28 | 1,173 ms | 1,756 ms | 0.57 it/s | 1.21x | 11.48 GiB | 4.32 GiB | Highly Conservative |
| **Baseline Kohya** | 28 / 28 | 1,457 ms | 2,038 - 2,134 ms | 0.49 it/s | 1.00x | 5.19 GiB | 10.6 GiB (Wasted) | Slow Baseline |

*Stability Note:* In a 15-step steady-state run, the 12-block configuration showed **0.0 MiB memory growth** and zero NaNs, proving no progressive buffer leakage.

---

## Requirements & Fork Setup

Clone the optimized fork containing native selective checkpointing controls:

```bash
git clone -b feat/anima-selective-checkpointing https://github.com/ruwwww/sd-scripts.git sd-scripts
cd sd-scripts
pip install -r requirements.txt
```

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
  --checkpoint_blocks=12 \
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

### Recipe B: Lightweight AdamW8bit (Fast & Low VRAM)
Recommended for fast iterations and minimal adapter size (~13.8 MB with `dim=8`).

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
- [anima-fastpath-recipe](https://github.com/ruwwww/anima-fastpath-recipe): Baseline eager & TorchCompile inference optimization recipes for Anima DiT.
- [sd-scripts](https://github.com/ruwwww/sd-scripts): Optimized training scripts fork featuring selective gradient checkpointing.
