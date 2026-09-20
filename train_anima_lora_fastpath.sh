#!/usr/bin/env bash
# ==============================================================================
# Fast-Path Anima DiT LoRA Training Script (RTX 5060 Ti 16GB Sweet Spot)
# Powered by sd-scripts fork: https://github.com/ruwwww/sd-scripts
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Environment & Executable
PYTHON_ENV="/home/kuroko/.conda/envs/ai/bin/python"
SD_SCRIPTS_DIR="/mnt/data/finetune-anima/sd-scripts"
TRAINER_SCRIPT="${SD_SCRIPTS_DIR}/anima_train_network.py"

# Model Paths
DIT_MODEL="/home/kuroko/ComfyUI/models/diffusion_models/anima-base-v1.0.safetensors"
QWEN3_ENCODER="/mnt/data/models/text_encoders/qwen_3_06b_base.safetensors"
QWEN_VAE="/home/kuroko/ComfyUI/models/vae/qwen_image_vae.safetensors"

# Dataset Paths & Staging Link
SOURCE_DATASET="${SOURCE_DATASET:-/mnt/data/finetune-anima/staging_images/shemira}"
RUNTIME_PARENT="${RUNTIME_PARENT:-/mnt/data/finetune-anima/runtime_dataset/shemira}"
STAGING_LINK="${STAGING_LINK:-${RUNTIME_PARENT}/10_${CONCEPT_NAME:-shemira}}"
TRIGGER="${TRIGGER:-shemira}"
EPOCHS="${EPOCHS:-5}"
CONCEPT_NAME="${CONCEPT_NAME:-shemira}"
CURATED_ROOT="${CURATED_ROOT:-/mnt/data/finetune-anima/data_root_${CONCEPT_NAME}_fastpath}"
CURATED_DIR="${CURATED_DIR:-${CURATED_ROOT}/10_${CONCEPT_NAME}}"

# Caption policy. A no-caption dataset makes every sample use the same class
# token, which can produce a numerically non-empty but practically ineffective
# adapter. Auto-tagging is enabled by default; set AUTO_TAG=0 only when every
# image already has a curated .txt sidecar.
AUTO_TAG="${AUTO_TAG:-1}"
WD14_REPO_ID="${WD14_REPO_ID:-SmilingWolf/wd-swinv2-tagger-v3}"
WD14_MODEL_DIR="${WD14_MODEL_DIR:-/mnt/data/finetune-anima/wd14_tagger_model}"

# Output Configuration
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/data/finetune-anima/output_lora/shemira_fastpath_fixed}"
OUTPUT_NAME="${OUTPUT_NAME:-shemira_anima_fastpath_fixed}"
COMFY_LORA_DIR="/home/kuroko/ComfyUI/models/loras"

restore_comfyui() {
  echo "[Anima Fastpath] Restoring ComfyUI service..."
  systemctl --user start comfyui.service 2>/dev/null || true
}
trap restore_comfyui EXIT

# Validation
echo "======================================================================"
echo "[Anima Fastpath] Preflight Validation"
echo "======================================================================"

for p in "${PYTHON_ENV}" "${TRAINER_SCRIPT}" "${DIT_MODEL}" "${QWEN3_ENCODER}" "${QWEN_VAE}" "${SOURCE_DATASET}"; do
  if [[ ! -e "${p}" ]]; then
    echo "Error: Required path does not exist: ${p}" >&2
    exit 1
  fi
done

# Ensure staging symlink exists non-destructively
mkdir -p "${RUNTIME_PARENT}"
if [[ -L "${STAGING_LINK}" ]]; then
  CURRENT_TARGET="$(readlink -f "${STAGING_LINK}")"
  SOURCE_RESOLVED="$(readlink -f "${SOURCE_DATASET}")"
  if [[ "${CURRENT_TARGET}" != "${SOURCE_RESOLVED}" ]]; then
    echo "Updating existing symlink target from ${CURRENT_TARGET} to ${SOURCE_RESOLVED}..."
    ln -sfn "${SOURCE_DATASET}" "${STAGING_LINK}"
  fi
elif [[ -e "${STAGING_LINK}" ]]; then
  echo "Error: ${STAGING_LINK} exists and is not a symlink. Aborting to protect data." >&2
  exit 1
else
  echo "Creating non-destructive staging link: ${STAGING_LINK} -> ${SOURCE_DATASET}"
  ln -s "${SOURCE_DATASET}" "${STAGING_LINK}"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${COMFY_LORA_DIR}"

# Check images and caption situation
IMG_COUNT=$(find -L "${STAGING_LINK}" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.webp" \) | wc -l)
TXT_COUNT=$(find -L "${STAGING_LINK}" -maxdepth 1 -type f -iname "*.txt" | wc -l)

echo "Dataset source:      ${SOURCE_DATASET}"
echo "Staging symlink:     ${STAGING_LINK} (images: ${IMG_COUNT}, captions: ${TXT_COUNT})"
if [[ "${TXT_COUNT}" -eq 0 ]]; then
  if [[ "${AUTO_TAG}" == "1" || "${AUTO_TAG,,}" == "true" ]]; then
    echo "Caption situation:   No captions; running WD14 once before training."
    systemctl --user stop comfyui.service comfyui-cudagraph.service 2>/dev/null || true
    "${PYTHON_ENV}" "${SD_SCRIPTS_DIR}/finetune/tag_images_by_wd14_tagger.py" \
      "${SOURCE_DATASET}" \
      --repo_id="${WD14_REPO_ID}" \
      --model_dir="${WD14_MODEL_DIR}" \
      --batch_size=8 \
      --thresh=0.35 \
      --character_threshold=0.85 \
      --onnx \
      --caption_extension=".txt"
    TXT_COUNT=$(find -L "${STAGING_LINK}" -maxdepth 1 -type f -iname "*.txt" | wc -l)
  fi
fi
if [[ "${TXT_COUNT}" -eq 0 ]]; then
  echo "Caption situation:   ERROR: no captions available after caption preparation." >&2
  echo "Set AUTO_TAG=1, or add one .txt sidecar for every image before training." >&2
  exit 1
else
  echo "Caption situation:   Found ${TXT_COUNT} caption files."
fi

if [[ "${IMG_COUNT}" -eq 0 ]]; then
  echo "Error: No images found in dataset staging link!" >&2
  exit 1
fi

echo "Curating dataset with trigger '${TRIGGER}'..."
"${PYTHON_ENV}" /mnt/data/finetune-anima/auto_curate.py \
  --staging_dir="${SOURCE_DATASET}" \
  --dest_dir="${CURATED_DIR}" \
  --trigger="${TRIGGER}" \
  --concept_name="${CONCEPT_NAME}" \
  --sparse

CURATED_IMG_COUNT=$(find "${CURATED_DIR}" -maxdepth 1 -type f -iname "*.png" | wc -l)
CURATED_TXT_COUNT=$(find "${CURATED_DIR}" -maxdepth 1 -type f -iname "*.txt" | wc -l)
if [[ "${CURATED_IMG_COUNT}" -eq 0 || "${CURATED_IMG_COUNT}" -ne "${CURATED_TXT_COUNT}" ]]; then
  echo "Error: curated dataset is incomplete (images=${CURATED_IMG_COUNT}, captions=${CURATED_TXT_COUNT})." >&2
  exit 1
fi
echo "Curated dataset:    ${CURATED_DIR} (images: ${CURATED_IMG_COUNT}, captions: ${CURATED_TXT_COUNT})"

echo "Stopping ComfyUI before GPU training..."
systemctl --user stop comfyui.service comfyui-cudagraph.service 2>/dev/null || true

echo "======================================================================"
echo "[Anima DiT Fast-Path] Launching Optimized LoRA Training"
echo "  Target Hardware:      RTX 5060 Ti 16GB (36 SMs, sm_120)"
echo "  Trainer Entry:        ${TRAINER_SCRIPT}"
echo "  DiT Model:            ${DIT_MODEL}"
echo "  Qwen3 Text Encoder:   ${QWEN3_ENCODER}"
echo "  Qwen Image VAE:       ${QWEN_VAE}"
echo "  Train Data Dir:       ${CURATED_ROOT}"
echo "  Output Dir:           ${OUTPUT_DIR}"
echo "  Output Name:          ${OUTPUT_NAME}"
echo "  Kernel Configuration:"
echo "    - selective_checkpointing=count (checkpoint_blocks=1)"
echo "    - compile=default (cache_size_limit=32)"
echo "    - fused_lora, fused_mlp (storage=fp8, backend=triton, direct_backward)"
echo "    - mixed_precision=bf16, save_precision=bf16, attn_mode=torch"
echo "  Training Dynamics:"
echo "    - network_module=networks.lora_anima (dim=16, alpha=16.0, unet_only)"
echo "    - optimizer=Prodigy (lr=1.0, cosine, warmup=10, weight_decay=0.1, decouple=True, d_coef=1.0)"
echo "    - max_train_epochs=${EPOCHS}, seed=2026, cache_latents_to_disk=True"
echo "======================================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_ENV}" "${TRAINER_SCRIPT}" \
--pretrained_model_name_or_path="${DIT_MODEL}" \
--train_data_dir="${CURATED_ROOT}" \
--output_dir="${OUTPUT_DIR}" \
--output_name="${OUTPUT_NAME}" \
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
--max_train_epochs="${EPOCHS}" \
  --save_every_n_epochs=1 \
  --mixed_precision="bf16" \
  --save_precision="bf16" \
  --seed=2026 \
  --cache_latents \
  --cache_latents_to_disk \
  --vae="${QWEN_VAE}" \
  --qwen_image_vae_2d \
  --qwen3="${QWEN3_ENCODER}" \
  --resolution="832,1216" \
  --enable_bucket \
  --min_bucket_reso=512 \
  --max_bucket_reso=1536 \
  --bucket_reso_steps=32 \
  --caption_extension=".txt" \
  --caption_tag_dropout_rate=0.05 \
  --caption_dropout_rate=0.1 \
  --shuffle_caption \
  --keep_tokens=1

echo "[Anima Fastpath] Training completed successfully!"
