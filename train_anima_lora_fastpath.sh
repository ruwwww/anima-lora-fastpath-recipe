#!/bin/bash
set -e

# ==============================================================================
# Fast-Path Anima DiT LoRA Training Script (RTX 5060 Ti 16GB Sweet Spot)
# Powered by sd-scripts fork: https://github.com/ruwwww/sd-scripts-fourtune
# ==============================================================================

PYTHON_ENV="/home/kuroko/.conda/envs/ai/bin/python"
SD_SCRIPTS_DIR="./sd-scripts"

# Model paths
DIT_MODEL="/home/kuroko/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors"
QWEN3_ENCODER="/mnt/data/models/text_encoders/qwen_3_06b_base.safetensors"
QWEN_VAE="/home/kuroko/ComfyUI/models/vae/qwen_image_vae.safetensors"

# Dataset and output configuration
DATA_ROOT="./data_root_character"
OUTPUT_DIR="./output_lora"
OUTPUT_NAME="anima_character_fastpath_lora"
COMFY_LORA_DIR="/home/kuroko/ComfyUI/models/loras"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COMFY_LORA_DIR}"

echo "======================================================================"
echo "[Anima DiT Fast-Path] Launching Optimized LoRA Training"
echo "Target Hardware: RTX 5060 Ti 16GB (36 SMs, sm_120)"
echo "Checkpointing: 12 / 28 Blocks (Sweet Spot, ~1.24x Speedup)"
echo "======================================================================"

# Run Accelerated Training
"${PYTHON_ENV}" "${SD_SCRIPTS_DIR}/anima_train_network.py" \
  --pretrained_model_name_or_path="${DIT_MODEL}" \
  --train_data_dir="${DATA_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --output_name="${OUTPUT_NAME}" \
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
  --vae="${QWEN_VAE}" \
  --qwen_image_vae_2d \
  --text_encoder="${QWEN3_ENCODER}" \
  --qwen3 \
  --resolution="832,1216" \
  --enable_bucket \
  --min_bucket_reso=512 \
  --max_bucket_reso=1536 \
  --bucket_reso_steps=32 \
  --seed=2026 \
  --caption_extension=".txt" \
  --caption_tag_dropout_rate=0.7 \
  --caption_dropout_rate=0.1 \
  --keep_tokens=1

echo "Training completed successfully!"
