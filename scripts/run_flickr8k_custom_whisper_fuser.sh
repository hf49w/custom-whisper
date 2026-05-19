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
MANIFEST_PATH="${MANIFEST_PATH:-$PREPARED_ROOT/manifest.jsonl}"
WHISPER_DOWNLOAD_ROOT="${WHISPER_DOWNLOAD_ROOT:-$DATA_ROOT/models/whisper}"
OFFLINE="${OFFLINE:-0}"

SPLIT_SEED="${SPLIT_SEED:-42}"
TEST_RATIO="${TEST_RATIO:-0.2}"
TEST_RATIO_TAG="${TEST_RATIO_TAG:-20}"
SPLIT_ROOT="${SPLIT_ROOT:-$PREPARED_ROOT/splits/by_image_id_seed${SPLIT_SEED}_test${TEST_RATIO_TAG}}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$SPLIT_ROOT/train_manifest.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$SPLIT_ROOT/test_manifest.jsonl}"

WHISPER_MODEL="${WHISPER_MODEL:-medium.en}"
VISUAL_ENCODER="${VISUAL_ENCODER:-resnet18}"
VISUAL_FUSER="${VISUAL_FUSER:-concat_proj}"
EPOCHS="${EPOCHS:-5}"
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
NUM_RESNET_LAYERS="${NUM_RESNET_LAYERS:-18}"
P_SPEECH="${P_SPEECH:-0.5}"
DIM_SPEECH_INTER="${DIM_SPEECH_INTER:-128}"
DIM_VISUAL_INTER="${DIM_VISUAL_INTER:-128}"
CKPT_NAME="${CKPT_NAME:-best_train_loss.pt}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
ALLOW_RELOCATED_PATHS="${ALLOW_RELOCATED_PATHS:-0}"

ENABLE_SPECAUG="${ENABLE_SPECAUG:-0}"
SPECAUG_TIME_WARP="${SPECAUG_TIME_WARP:-1}"
SPECAUG_TIME_WARP_WINDOW="${SPECAUG_TIME_WARP_WINDOW:-5}"
SPECAUG_TIME_WARP_MODE="${SPECAUG_TIME_WARP_MODE:-bicubic}"
SPECAUG_FREQ_MASK="${SPECAUG_FREQ_MASK:-1}"
SPECAUG_FREQ_MASK_MIN="${SPECAUG_FREQ_MASK_MIN:-0}"
SPECAUG_FREQ_MASK_MAX="${SPECAUG_FREQ_MASK_MAX:-30}"
SPECAUG_NUM_FREQ_MASK="${SPECAUG_NUM_FREQ_MASK:-2}"
SPECAUG_TIME_MASK="${SPECAUG_TIME_MASK:-1}"
SPECAUG_TIME_MASK_MIN="${SPECAUG_TIME_MASK_MIN:-0}"
SPECAUG_TIME_MASK_MAX="${SPECAUG_TIME_MASK_MAX:-40}"
SPECAUG_NUM_TIME_MASK="${SPECAUG_NUM_TIME_MASK:-2}"

CLIP_MODEL_NAME="${CLIP_MODEL_NAME:-openai/clip-vit-base-patch32}"
CLIP_RETURN_SEQUENCE="${CLIP_RETURN_SEQUENCE:-0}"
ATTN_NUM_HEADS="${ATTN_NUM_HEADS:-8}"
ATTN_DROPOUT="${ATTN_DROPOUT:-0.1}"
ATTN_GATE_INIT="${ATTN_GATE_INIT:--4.0}"
ATTN_NUM_QUERIES="${ATTN_NUM_QUERIES:-8}"
SPECAUG_TAG=""
if [[ "$ENABLE_SPECAUG" == "1" ]]; then
  SPECAUG_TAG="_specaug"
fi
RUN_TAG_DEFAULT="flickr8k_${VISUAL_ENCODER}_${VISUAL_FUSER}_seed${SPLIT_SEED}_test${TEST_RATIO_TAG}_ep${EPOCHS}_bs${BATCH_SIZE}_lr${LR}${SPECAUG_TAG}"
RUN_TAG="${RUN_TAG:-$RUN_TAG_DEFAULT}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$PREPARED_ROOT/$RUN_TAG}"
MODEL_DIR="$EXPERIMENT_ROOT/model"
TRAIN_EVAL_DIR="$EXPERIMENT_ROOT/eval_train"
TEST_EVAL_DIR="$EXPERIMENT_ROOT/eval_test"
SUMMARY_TXT="$EXPERIMENT_ROOT/run_summary.txt"

mkdir -p "$EXPERIMENT_ROOT"
mkdir -p "$PREPARED_ROOT" "$WHISPER_DOWNLOAD_ROOT" "$CACHE_ROOT"

run_python() {
  if [[ "$OFFLINE" == "1" ]]; then
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PYTHON_BIN" "$@"
  else
    "$PYTHON_BIN" "$@"
  fi
}

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "[INFO] Prepared Flickr8k manifest missing. Building it first."
  run_python "$REPO_ROOT/scripts/prepare_flickr8k_for_custom_whisper.py" \
    --images-root "$IMAGES_ROOT" \
    --audio-root "$AUDIO_ROOT" \
    --captions-path "$CAPTIONS_PATH" \
    --output-root "$PREPARED_ROOT"
fi

if [[ ! -f "$TRAIN_MANIFEST" || ! -f "$TEST_MANIFEST" ]]; then
  echo "[INFO] Building Flickr8k train/test split by image_id."
  run_python "$REPO_ROOT/scripts/split_visspeech_custom_whisper_dataset.py" \
    --manifest-path "$MANIFEST_PATH" \
    --output-root "$SPLIT_ROOT" \
    --test-ratio "$TEST_RATIO" \
    --seed "$SPLIT_SEED" \
    --group-by-field image_id
fi

mkdir -p "$MODEL_DIR" "$TRAIN_EVAL_DIR" "$TEST_EVAL_DIR"

train_args=(
  --train-manifest "$TRAIN_MANIFEST"
  --output-root "$MODEL_DIR"
  --whisper-model "$WHISPER_MODEL"
  --visual-encoder "$VISUAL_ENCODER"
  --visual-fuser "$VISUAL_FUSER"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --weight-decay "$WEIGHT_DECAY"
  --num-workers "$NUM_WORKERS"
  --seed "$SPLIT_SEED"
  --log-every "$LOG_EVERY"
  --save-every "$SAVE_EVERY"
  --save-every-batches "$SAVE_EVERY_BATCHES"
  --whisper-download-root "$WHISPER_DOWNLOAD_ROOT"
  --image-size "$IMAGE_SIZE"
  --clip-model-name "$CLIP_MODEL_NAME"
  --num-gmlp-layers "$NUM_GMLP_LAYERS"
  --num-resnet-layers "$NUM_RESNET_LAYERS"
  --p-speech "$P_SPEECH"
  --dim-speech-inter "$DIM_SPEECH_INTER"
  --dim-visual-inter "$DIM_VISUAL_INTER"
  --attn-num-heads "$ATTN_NUM_HEADS"
  --attn-dropout "$ATTN_DROPOUT"
  --attn-gate-init "$ATTN_GATE_INIT"
  --attn-num-queries "$ATTN_NUM_QUERIES"
  --visual-pretrained
)

if [[ "$CLIP_RETURN_SEQUENCE" == "1" || ( "$VISUAL_ENCODER" == "clip" && ( "$VISUAL_FUSER" == "cross_attn_gate" || "$VISUAL_FUSER" == "attn_prefix" || "$VISUAL_FUSER" == "gated_seq_concat" ) ) ]]; then
  train_args+=(--clip-return-sequence)
fi

LAST_CHECKPOINT="$MODEL_DIR/checkpoints/last.pt"
BEST_CHECKPOINT="$MODEL_DIR/checkpoints/$CKPT_NAME"

if [[ "$FORCE_RETRAIN" == "1" ]]; then
  echo "[TRAIN] force_retrain=1"
  train_args+=(--force-retrain)
elif [[ -f "$LAST_CHECKPOINT" && "$TRAIN_MANIFEST" -nt "$LAST_CHECKPOINT" ]]; then
  echo "[TRAIN] train_manifest_newer_than_checkpoint force_retrain=1"
  train_args+=(--force-retrain)
elif [[ -f "$LAST_CHECKPOINT" ]]; then
  echo "[TRAIN] resume_from=$LAST_CHECKPOINT"
  train_args+=(--resume-from "$LAST_CHECKPOINT")
  if [[ "$ALLOW_RELOCATED_PATHS" == "1" ]]; then
    train_args+=(--allow-relocated-paths)
  fi
else
  echo "[TRAIN] fresh_start"
fi

if [[ "$ENABLE_SPECAUG" == "1" ]]; then
  train_args+=(
    --enable-specaug
    --specaug-time-warp-window "$SPECAUG_TIME_WARP_WINDOW"
    --specaug-time-warp-mode "$SPECAUG_TIME_WARP_MODE"
    --specaug-freq-mask-width-range "$SPECAUG_FREQ_MASK_MIN" "$SPECAUG_FREQ_MASK_MAX"
    --specaug-num-freq-mask "$SPECAUG_NUM_FREQ_MASK"
    --specaug-time-mask-width-range "$SPECAUG_TIME_MASK_MIN" "$SPECAUG_TIME_MASK_MAX"
    --specaug-num-time-mask "$SPECAUG_NUM_TIME_MASK"
  )
  if [[ "$SPECAUG_TIME_WARP" != "1" ]]; then
    train_args+=(--disable-specaug-time-warp)
  fi
  if [[ "$SPECAUG_FREQ_MASK" != "1" ]]; then
    train_args+=(--disable-specaug-freq-mask)
  fi
  if [[ "$SPECAUG_TIME_MASK" != "1" ]]; then
    train_args+=(--disable-specaug-time-mask)
  fi
fi

run_python "$REPO_ROOT/scripts/train_visspeech_custom_whisper_fuser.py" "${train_args[@]}"

echo "[EVAL train]"
run_python "$REPO_ROOT/scripts/eval_visspeech_custom_whisper_fuser.py" \
  --checkpoint-path "$BEST_CHECKPOINT" \
  --manifest-path "$TRAIN_MANIFEST" \
  --output-root "$TRAIN_EVAL_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --resume-from-predictions \
  --skip-if-exists \
  --log-every "$EVAL_LOG_EVERY"

echo "[EVAL test]"
run_python "$REPO_ROOT/scripts/eval_visspeech_custom_whisper_fuser.py" \
  --checkpoint-path "$BEST_CHECKPOINT" \
  --manifest-path "$TEST_MANIFEST" \
  --output-root "$TEST_EVAL_DIR" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --resume-from-predictions \
  --skip-if-exists \
  --log-every "$EVAL_LOG_EVERY"

cat > "$SUMMARY_TXT" <<EOF
manifest=$MANIFEST_PATH
train_manifest=$TRAIN_MANIFEST
test_manifest=$TEST_MANIFEST
model_dir=$MODEL_DIR
checkpoint=$BEST_CHECKPOINT
train_eval_dir=$TRAIN_EVAL_DIR
test_eval_dir=$TEST_EVAL_DIR
EOF

echo "[DONE] summary=$SUMMARY_TXT"
