#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
CACHE_ROOT="${CACHE_ROOT:-$REPO_ROOT/.cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CACHE_ROOT/xdg}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"

PYTHON_BIN="${PYTHON_BIN:-python}"
IMAGES_ROOT="${IMAGES_ROOT:-$DATA_ROOT/flickr8k/images}"
AUDIO_ROOT="${AUDIO_ROOT:-$DATA_ROOT/flickr8k/audio}"
CAPTIONS_PATH="${CAPTIONS_PATH:-$DATA_ROOT/flickr8k/captions/captions.txt}"
PREPARED_ROOT="${PREPARED_ROOT:-$DATA_ROOT/flickr8k/prepared}"
FULL_MANIFEST_PATH="${FULL_MANIFEST_PATH:-$PREPARED_ROOT/manifest.jsonl}"
WHISPER_DOWNLOAD_ROOT="${WHISPER_DOWNLOAD_ROOT:-$DATA_ROOT/models/whisper}"
OFFLINE="${OFFLINE:-0}"

ROWS_PER_IMAGE="${ROWS_PER_IMAGE:-2}"
SELECTION_MODE="${SELECTION_MODE:-random}"
SELECTION_SEED="${SELECTION_SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TEST_RATIO="${TEST_RATIO:-0.2}"
TEST_RATIO_TAG="${TEST_RATIO_TAG:-20}"
DROP_IMAGES_WITH_FEWER_ROWS="${DROP_IMAGES_WITH_FEWER_ROWS:-0}"

WHISPER_MODEL="${WHISPER_MODEL:-medium.en}"
EPOCHS="5"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-2}"
NUM_WORKERS="${NUM_WORKERS:-0}"
LOG_EVERY="${LOG_EVERY:-10}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-20}"
SAVE_EVERY="${SAVE_EVERY:-1}"
SAVE_EVERY_BATCHES="${SAVE_EVERY_BATCHES:-100}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
NUM_GMLP_LAYERS="${NUM_GMLP_LAYERS:-1}"
P_SPEECH="${P_SPEECH:-0.5}"
DIM_SPEECH_INTER="${DIM_SPEECH_INTER:-128}"
DIM_VISUAL_INTER="${DIM_VISUAL_INTER:-128}"
CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
ATTN_NUM_HEADS="${ATTN_NUM_HEADS:-8}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.1}"
ATTN_GATE_INIT="${ATTN_GATE_INIT:--4.0}"
ATTN_NUM_QUERIES="${ATTN_NUM_QUERIES:-8}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
ENABLE_SPECAUG="${ENABLE_SPECAUG:-0}"
ALLOW_RELOCATED_PATHS="${ALLOW_RELOCATED_PATHS:-0}"

SUBSET_ROOT_DEFAULT="$PREPARED_ROOT/subsets/${ROWS_PER_IMAGE}_per_image_${SELECTION_MODE}_sel${SELECTION_SEED}_split${SPLIT_SEED}_test${TEST_RATIO_TAG}"
SUBSET_ROOT="${SUBSET_ROOT:-$SUBSET_ROOT_DEFAULT}"
SUBSET_MANIFEST="${SUBSET_MANIFEST:-$SUBSET_ROOT/subset_manifest.jsonl}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$SUBSET_ROOT/train_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$SUBSET_ROOT/test_manifest.jsonl}"

SUITE_TAG_DEFAULT="flickr8k_subset${ROWS_PER_IMAGE}_seed${SELECTION_SEED}_split${SPLIT_SEED}_test${TEST_RATIO_TAG}_ep${EPOCHS}"
SUITE_TAG="${SUITE_TAG:-$SUITE_TAG_DEFAULT}"
SUITE_ROOT="${SUITE_ROOT:-$REPO_ROOT/outputs/$SUITE_TAG}"
SUMMARY_TSV="$SUITE_ROOT/summary.tsv"

mkdir -p "$SUITE_ROOT"
mkdir -p "$PREPARED_ROOT" "$WHISPER_DOWNLOAD_ROOT" "$CACHE_ROOT"

run_python() {
  if [[ "$OFFLINE" == "1" ]]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

if [[ ! -f "$FULL_MANIFEST_PATH" ]]; then
  echo "[INFO] Prepared Flickr8k manifest missing. Building it first."
  run_python "$REPO_ROOT/scripts/prepare_flickr8k_for_custom_whisper.py" \
    --images-root "$IMAGES_ROOT" \
    --audio-root "$AUDIO_ROOT" \
    --captions-path "$CAPTIONS_PATH" \
    --output-root "$PREPARED_ROOT"
fi

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$TEST_MANIFEST" || ! -f "$SUBSET_MANIFEST" ]]; then
  echo "[INFO] Building Flickr8k subset: rows_per_image=$ROWS_PER_IMAGE selection_mode=$SELECTION_MODE selection_seed=$SELECTION_SEED"
  subset_args=(
    --manifest-path "$FULL_MANIFEST_PATH"
    --output-root "$SUBSET_ROOT"
    --rows-per-image "$ROWS_PER_IMAGE"
    --selection-mode "$SELECTION_MODE"
    --selection-seed "$SELECTION_SEED"
    --test-ratio "$TEST_RATIO"
    --split-seed "$SPLIT_SEED"
  )
  if [[ "$DROP_IMAGES_WITH_FEWER_ROWS" == "1" ]]; then
    subset_args+=(--drop-images-with-fewer-rows)
  fi
  run_python "$REPO_ROOT/scripts/select_flickr8k_subset_and_split.py" "${subset_args[@]}"
fi

cat > "$SUMMARY_TSV" <<'EOF'
encoder	fuser	experiment_root	checkpoint	train_manifest	test_manifest
EOF

run_one_combo() {
  local encoder="$1"
  local fuser="$2"
  local resnet_depth="${3:-18}"
  local combo_name="${encoder}__${fuser}"
  local experiment_root="$SUITE_ROOT/$combo_name"
  local checkpoint_path="$experiment_root/model/checkpoints/best_train_loss.pt"

  echo "[RUN] encoder=$encoder fuser=$fuser resnet_depth=$resnet_depth"
  IMAGES_ROOT="$IMAGES_ROOT" \
  AUDIO_ROOT="$AUDIO_ROOT" \
  CAPTIONS_PATH="$CAPTIONS_PATH" \
  PREPARED_ROOT="$PREPARED_ROOT" \
  MANIFEST_PATH="$SUBSET_MANIFEST" \
  TRAIN_MANIFEST="$TRAIN_MANIFEST" \
  TEST_MANIFEST="$TEST_MANIFEST" \
  SPLIT_SEED="$SPLIT_SEED" \
  TEST_RATIO="$TEST_RATIO" \
  TEST_RATIO_TAG="$TEST_RATIO_TAG" \
  WHISPER_MODEL="$WHISPER_MODEL" \
  VISUAL_ENCODER="$encoder" \
  VISUAL_FUSER="$fuser" \
  EPOCHS="$EPOCHS" \
  BATCH_SIZE="$BATCH_SIZE" \
  LR="$LR" \
  WEIGHT_DECAY="$WEIGHT_DECAY" \
  NUM_WORKERS="$NUM_WORKERS" \
  LOG_EVERY="$LOG_EVERY" \
  EVAL_LOG_EVERY="$EVAL_LOG_EVERY" \
  SAVE_EVERY="$SAVE_EVERY" \
  SAVE_EVERY_BATCHES="$SAVE_EVERY_BATCHES" \
  IMAGE_SIZE="$IMAGE_SIZE" \
  NUM_GMLP_LAYERS="$NUM_GMLP_LAYERS" \
  NUM_RESNET_LAYERS="$resnet_depth" \
  P_SPEECH="$P_SPEECH" \
  DIM_SPEECH_INTER="$DIM_SPEECH_INTER" \
  DIM_VISUAL_INTER="$DIM_VISUAL_INTER" \
  CLIP_MODEL_NAME="$CLIP_MODEL_NAME" \
  WHISPER_DOWNLOAD_ROOT="$WHISPER_DOWNLOAD_ROOT" \
  ATTN_NUM_HEADS="$ATTN_NUM_HEADS" \
  ATTN_DROPOUT="$ATTN_DROPOUT" \
  ATTN_GATE_INIT="$ATTN_GATE_INIT" \
  ATTN_NUM_QUERIES="$ATTN_NUM_QUERIES" \
  FORCE_RETRAIN="$FORCE_RETRAIN" \
  ENABLE_SPECAUG="$ENABLE_SPECAUG" \
  ALLOW_RELOCATED_PATHS="$ALLOW_RELOCATED_PATHS" \
  OFFLINE="$OFFLINE" \
  EXPERIMENT_ROOT="$experiment_root" \
  bash "$REPO_ROOT/scripts/run_flickr8k_custom_whisper_fuser.sh"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$encoder" \
    "$fuser" \
    "$experiment_root" \
    "$checkpoint_path" \
    "$TRAIN_MANIFEST" \
    "$TEST_MANIFEST" >> "$SUMMARY_TSV"
}

# Requested non-attention baselines
run_one_combo resnet50 proj_concat_proj 50
run_one_combo resnet_gmlp concat_temp 18

# Two attention-based fusers previously added
run_one_combo clip cross_attn_gate 18
run_one_combo clip attn_prefix 18

# New gated sequence concat
run_one_combo clip gated_seq_concat 18

echo "[DONE] subset_manifest=$SUBSET_MANIFEST"
echo "[DONE] train_manifest=$TRAIN_MANIFEST"
echo "[DONE] test_manifest=$TEST_MANIFEST"
echo "[DONE] summary=$SUMMARY_TSV"
