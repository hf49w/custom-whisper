from __future__ import annotations

import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from custom_whisper.audio import N_SAMPLES, log_mel_spectrogram, pad_or_trim
from custom_whisper.tokenizer import Tokenizer, get_tokenizer
from espnet_specaug_vendor import SpecAug as VendoredSpecAug


WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
NON_EVAL_CHARS_RE = re.compile(r"[^a-z0-9'\s]+")
MULTISPACE_RE = re.compile(r"\s+")


def resolve_cross_platform_path(path: str) -> Path:
    text = os.path.expandvars(os.path.expanduser(str(path)))
    match = WINDOWS_DRIVE_RE.match(text)
    if os.name != "nt" and match:
        drive = match.group("drive").lower()
        rest = match.group("rest").replace("\\", "/")
        text = f"/mnt/{drive}/{rest}"
    return Path(os.path.abspath(text))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".csv":
        rows = read_csv_rows(path)
    else:
        raise ValueError(f"Unsupported manifest format: {path}")

    normalized_rows: List[Dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        for key in ("wav_path", "image_path"):
            if normalized.get(key):
                normalized[key] = str(resolve_cross_platform_path(normalized[key]))
        if "annotation" not in normalized:
            annotation = normalized.get("text", "")
            normalized["annotation"] = annotation
        normalized_rows.append(normalized)
    return normalized_rows


def build_ordered_fieldnames(rows: Sequence[Dict[str, Any]]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
    return ordered


def normalize_eval_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = NON_EVAL_CHARS_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def build_prediction_identity(row: Dict[str, Any]) -> str:
    wav_path = str(row.get("wav_path", ""))
    image_path = str(row.get("image_path", ""))
    key = str(row.get("key") or row.get("utt_id") or Path(wav_path).stem)
    return "\t".join((key, wav_path, image_path))


def _edit_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    if not seq_a:
        return len(seq_b)
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, token_a in enumerate(seq_a, start=1):
        current = [i]
        for j, token_b in enumerate(seq_b, start=1):
            cost = 0 if token_a == token_b else 1
            current.append(
                min(
                    prev[j] + 1,
                    current[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = current
    return prev[-1]


def compute_wer(ref_texts: Sequence[str], pred_texts: Sequence[str]) -> float:
    total_words = 0
    total_edits = 0
    for ref_text, pred_text in zip(ref_texts, pred_texts):
        ref_words = normalize_eval_text(ref_text).split()
        pred_words = normalize_eval_text(pred_text).split()
        total_words += max(1, len(ref_words))
        total_edits += _edit_distance(ref_words, pred_words)
    return total_edits / total_words if total_words else 0.0


def compute_cer(ref_texts: Sequence[str], pred_texts: Sequence[str]) -> float:
    total_chars = 0
    total_edits = 0
    for ref_text, pred_text in zip(ref_texts, pred_texts):
        ref_chars = list(normalize_eval_text(ref_text))
        pred_chars = list(normalize_eval_text(pred_text))
        total_chars += max(1, len(ref_chars))
        total_edits += _edit_distance(ref_chars, pred_chars)
    return total_edits / total_chars if total_chars else 0.0


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_clip_model_name() -> str:
    local_clip_dir = REPO_ROOT / "data" / "models" / "clip" / "clip-vit-base-patch32"
    if local_clip_dir.is_dir():
        return str(local_clip_dir)
    return "openai/clip-vit-base-patch32"


def build_tokenizer_and_prefix(model: custom_whisper.Whisper) -> Tuple[Tokenizer, List[int]]:
    tokenizer = get_tokenizer(
        model.is_multilingual,
        num_languages=model.num_languages,
        language="en" if model.is_multilingual else None,
        task="transcribe" if model.is_multilingual else None,
    )
    if tokenizer.no_timestamps is not None:
        prefix = list(tokenizer.sot_sequence_including_notimestamps)
    else:
        prefix = list(tokenizer.sot_sequence)
    return tokenizer, prefix


def encode_supervised_example(
    text: str,
    *,
    tokenizer: Tokenizer,
    prefix_tokens: Sequence[int],
    max_text_ctx: int,
    ignore_prefix_loss: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    normalized_text = normalize_eval_text(text)
    text_tokens = tokenizer.encode(" " + normalized_text) if normalized_text else []
    max_text_tokens = max(1, max_text_ctx - len(prefix_tokens) - 1)
    text_tokens = text_tokens[:max_text_tokens]
    full_tokens = list(prefix_tokens) + text_tokens + [tokenizer.eot]
    input_tokens = torch.tensor(full_tokens[:-1], dtype=torch.long)
    labels = torch.tensor(full_tokens[1:], dtype=torch.long)
    if ignore_prefix_loss and len(prefix_tokens) > 1:
        labels[: len(prefix_tokens) - 1] = -100
    return input_tokens, labels


class VisSpeechPreparedDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


@dataclass
class BatchEncodingConfig:
    n_mels: int
    max_text_ctx: int
    pad_token_id: int
    prefix_tokens: Sequence[int]
    tokenizer: Tokenizer


@dataclass(frozen=True)
class SpecAugmentConfig:
    apply_time_warp: bool = True
    time_warp_window: int = 5
    time_warp_mode: str = "bicubic"
    apply_freq_mask: bool = True
    freq_mask_width_range: Tuple[int, int] = (0, 30)
    num_freq_mask: int = 2
    apply_time_mask: bool = True
    time_mask_width_range: Tuple[int, int] = (0, 40)
    num_time_mask: int = 2

    def __post_init__(self) -> None:
        if not (self.apply_time_warp or self.apply_freq_mask or self.apply_time_mask):
            raise ValueError("SpecAugment requires at least one enabled operation.")
        if self.apply_time_warp and self.time_warp_window <= 0:
            raise ValueError("time_warp_window must be > 0 when time warp is enabled.")
        if not self.apply_time_warp and self.time_warp_window < 0:
            raise ValueError("time_warp_window must be >= 0.")
        if self.num_freq_mask < 0:
            raise ValueError("num_freq_mask must be >= 0.")
        if self.num_time_mask < 0:
            raise ValueError("num_time_mask must be >= 0.")
        if self.time_warp_mode not in {"bilinear", "bicubic"}:
            raise ValueError("time_warp_mode must be 'bilinear' or 'bicubic'.")
        _validate_mask_range(self.freq_mask_width_range, name="freq_mask_width_range")
        _validate_mask_range(self.time_mask_width_range, name="time_mask_width_range")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "apply_time_warp": self.apply_time_warp,
            "time_warp_window": self.time_warp_window,
            "time_warp_mode": self.time_warp_mode,
            "apply_freq_mask": self.apply_freq_mask,
            "freq_mask_width_range": list(self.freq_mask_width_range),
            "num_freq_mask": self.num_freq_mask,
            "apply_time_mask": self.apply_time_mask,
            "time_mask_width_range": list(self.time_mask_width_range),
            "num_time_mask": self.num_time_mask,
        }


def _validate_mask_range(mask_width_range: Tuple[int, int], *, name: str) -> None:
    if len(mask_width_range) != 2:
        raise ValueError(f"{name} must contain exactly two integers.")
    low, high = int(mask_width_range[0]), int(mask_width_range[1])
    if low < 0 or high < 0:
        raise ValueError(f"{name} values must be >= 0.")
    if low >= high:
        raise ValueError(f"{name} max must be greater than min.")


def build_specaug_module(config: SpecAugmentConfig) -> torch.nn.Module:
    return VendoredSpecAug(
        apply_time_warp=config.apply_time_warp,
        time_warp_window=config.time_warp_window,
        time_warp_mode=config.time_warp_mode,
        apply_freq_mask=config.apply_freq_mask,
        freq_mask_width_range=config.freq_mask_width_range,
        num_freq_mask=config.num_freq_mask,
        apply_time_mask=config.apply_time_mask,
        time_mask_width_range=config.time_mask_width_range,
        num_time_mask=config.num_time_mask,
    )


def collate_supervised_batch(
    rows: Sequence[Dict[str, Any]],
    *,
    config: BatchEncodingConfig,
) -> Dict[str, Any]:
    mels: List[torch.Tensor] = []
    input_tokens: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    refs: List[str] = []
    keys: List[str] = []
    wav_paths: List[str] = []
    image_paths: List[str] = []

    for row in rows:
        wav_path = str(row["wav_path"])
        image_path = str(row["image_path"])
        text = str(row.get("annotation", ""))

        audio = custom_whisper.load_audio(wav_path)
        audio = pad_or_trim(audio, length=N_SAMPLES)
        mel = log_mel_spectrogram(audio, n_mels=config.n_mels)
        token_ids, token_labels = encode_supervised_example(
            text,
            tokenizer=config.tokenizer,
            prefix_tokens=config.prefix_tokens,
            max_text_ctx=config.max_text_ctx,
        )

        mels.append(mel)
        input_tokens.append(token_ids)
        labels.append(token_labels)
        refs.append(text)
        keys.append(str(row.get("key") or row.get("utt_id") or Path(wav_path).stem))
        wav_paths.append(wav_path)
        image_paths.append(image_path)

    mel_batch = torch.stack(mels, dim=0)
    max_token_len = max(t.shape[0] for t in input_tokens)
    tokens_batch = torch.full(
        (len(rows), max_token_len),
        fill_value=config.pad_token_id,
        dtype=torch.long,
    )
    labels_batch = torch.full(
        (len(rows), max_token_len),
        fill_value=-100,
        dtype=torch.long,
    )

    for row_index, (token_ids, token_labels) in enumerate(zip(input_tokens, labels)):
        tokens_batch[row_index, : token_ids.shape[0]] = token_ids
        labels_batch[row_index, : token_labels.shape[0]] = token_labels

    return {
        "mel": mel_batch,
        "input_tokens": tokens_batch,
        "labels": labels_batch,
        "refs": refs,
        "keys": keys,
        "wav_paths": wav_paths,
        "image_paths": image_paths,
    }


def freeze_all_but_feature_fuser(model: custom_whisper.AudioImageWhisper) -> Dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if not hasattr(model, "feature_fuser"):
        raise ValueError("Model does not expose feature_fuser; expected AudioImageWhisper.")

    trainable_params = 0
    total_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
    for parameter in model.feature_fuser.parameters():
        parameter.requires_grad = True
        trainable_params += parameter.numel()

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": total_params - trainable_params,
    }


def set_fuser_training_mode(model: custom_whisper.AudioImageWhisper) -> None:
    model.train()
    model.encoder.eval()
    model.decoder.eval()
    model.encoder_visual.eval()
    model.feature_fuser.train()


def set_full_eval_mode(model: custom_whisper.AudioImageWhisper) -> None:
    model.eval()


def forward_fuser_only_loss(
    model: custom_whisper.AudioImageWhisper,
    batch: Dict[str, Any],
    *,
    device: torch.device,
    use_images: bool = True,
    specaug_module: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    mel = batch["mel"].to(device)
    input_tokens = batch["input_tokens"].to(device)
    labels = batch["labels"].to(device)

    with torch.no_grad():
        if specaug_module is not None:
            mel_time_major = mel.transpose(1, 2).contiguous()
            mel_time_major, _ = specaug_module(mel_time_major, None)
            mel = mel_time_major.transpose(1, 2).contiguous()
        audio_features = model.encoder(mel)
        image_features = None
        if use_images:
            image_features = model.encode_image(batch["image_paths"])
            image_features = model._expand_visual_features(
                image_features,
                batch_size=audio_features.shape[0],
            )

    fused_audio = model.fuse_audio_image_features(audio_features, image_features=image_features)
    logits = model.decoder(input_tokens, fused_audio)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


def transcribe_manifest_rows(
    model: custom_whisper.Whisper,
    rows: Sequence[Dict[str, Any]],
    *,
    use_images: bool = True,
    fp16: bool = False,
    transcribe_kwargs: Optional[Dict[str, Any]] = None,
    existing_predictions: Optional[Sequence[Dict[str, Any]]] = None,
    output_path: Optional[Path] = None,
    log_prefix: str = "TRANSCRIBE",
    log_every: int = 20,
) -> List[Dict[str, Any]]:
    prediction_by_id: Dict[str, Dict[str, Any]] = {}
    for prediction in existing_predictions or []:
        prediction_by_id[build_prediction_identity(prediction)] = dict(prediction)

    row_ids = [build_prediction_identity(row) for row in rows]
    row_id_set = set(row_ids)
    original_existing_count = len(prediction_by_id)
    prediction_by_id = {
        row_id: prediction
        for row_id, prediction in prediction_by_id.items()
        if row_id in row_id_set
    }
    ordered_existing_predictions = [
        prediction_by_id[row_id]
        for row_id in row_ids
        if row_id in prediction_by_id
    ]
    total = len(rows)
    completed_before = len(ordered_existing_predictions)
    ignored_existing = original_existing_count - len(prediction_by_id)
    log_interval = max(1, log_every)
    first_new_completed = completed_before + 1

    if output_path is not None:
        ensure_dir(output_path.parent)
        if ordered_existing_predictions:
            write_jsonl(output_path, ordered_existing_predictions)
        else:
            output_path.write_text("", encoding="utf-8")

    if existing_predictions:
        print(
            f"[{log_prefix}] resume_loaded={len(existing_predictions)} "
            f"matched={completed_before} ignored={max(0, ignored_existing)} remaining={total - completed_before}"
        )

    if completed_before == total:
        return ordered_existing_predictions

    progress_bar = (
        tqdm(
            total=total,
            initial=completed_before,
            desc=log_prefix.lower(),
            dynamic_ncols=True,
            leave=True,
        )
        if tqdm is not None
        else None
    )
    try:
        for row in rows:
            row_id = build_prediction_identity(row)
            if row_id in prediction_by_id:
                continue

            wav_path = str(row["wav_path"])
            image_path = str(row["image_path"])
            ref_text = str(row.get("annotation", ""))
            row_transcribe_kwargs: Dict[str, Any] = {
                "verbose": None,
                "fp16": fp16,
                "language": "en",
                "task": "transcribe",
            }
            if transcribe_kwargs:
                row_transcribe_kwargs.update(transcribe_kwargs)
            if use_images:
                row_transcribe_kwargs["image"] = image_path
            result = model.transcribe(wav_path, **row_transcribe_kwargs)
            pred_text = str(result.get("text", "")).strip()
            prediction = {
                "key": str(row.get("key") or Path(wav_path).stem),
                "wav_path": wav_path,
                "image_path": image_path,
                "ref_text": ref_text,
                "pred_text": pred_text,
                "norm_ref_text": normalize_eval_text(ref_text),
                "norm_pred_text": normalize_eval_text(pred_text),
            }
            prediction_by_id[row_id] = prediction
            if output_path is not None:
                append_jsonl(output_path, [prediction])

            completed = len(prediction_by_id)
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(completed=f"{completed}/{total}")
            if completed == first_new_completed or completed == total or completed % log_interval == 0:
                message = f"[{log_prefix}] row={completed}/{total}"
                if progress_bar is not None:
                    progress_bar.write(message)
                else:
                    print(message)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    return [prediction_by_id[build_prediction_identity(row)] for row in rows]


def summarize_predictions(predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    refs = [row["ref_text"] for row in predictions]
    preds = [row["pred_text"] for row in predictions]
    return {
        "count": len(predictions),
        "wer": compute_wer(refs, preds),
        "cer": compute_cer(refs, preds),
    }
