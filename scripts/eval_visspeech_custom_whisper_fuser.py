from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import custom_whisper
from visspeech_custom_whisper_utils import (
    BatchEncodingConfig,
    VisSpeechPreparedDataset,
    build_tokenizer_and_prefix,
    collate_supervised_batch,
    forward_fuser_only_loss,
    load_manifest,
    read_jsonl,
    resolve_cross_platform_path,
    set_full_eval_mode,
    summarize_predictions,
    transcribe_manifest_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fuser-only fine-tuned AudioImageWhisper checkpoint on a prepared VisSpeech manifest."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to a training checkpoint produced by train_visspeech_custom_whisper_fuser.py.",
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
        help="Manifest JSONL/CSV to evaluate.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory where metrics and predictions will be written.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--resume-from-predictions",
        action="store_true",
        help="Resume transcription from output_root/predictions.jsonl if it exists.",
    )
    parser.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip evaluation when output_root/metrics.json already exists.",
    )
    parser.add_argument("--log-every", type=int, default=20)
    return parser.parse_args()


def resolve_device(raw_device: str) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rebuild_model(checkpoint: Dict[str, Any], device: torch.device) -> custom_whisper.AudioImageWhisper:
    config = checkpoint["train_config"]
    model = custom_whisper.load_audio_image_model(
        config["whisper_model"],
        device=device,
        visual_encoder=config["visual_encoder"],
        feature_fuser=config["visual_fuser"],
        visual_pretrained=config["visual_pretrained"],
        image_size=config["image_size"],
        clip_model_name=config["clip_model_name"],
        clip_return_sequence=config["clip_return_sequence"],
        num_gmlp_layers=config["num_gmlp_layers"],
        num_resnet_layers=config["num_resnet_layers"],
        p_speech=config["p_speech"],
        use_residual=config["use_residual"],
        dim_speech_inter=config["dim_speech_inter"],
        dim_visual_inter=config["dim_visual_inter"],
        use_layer_norm=config["use_layer_norm"],
        attn_num_heads=config.get("attn_num_heads", 8),
        attn_dropout=config.get("attn_dropout", 0.1),
        attn_gate_init=config.get("attn_gate_init", -4.0),
        attn_num_queries=config.get("attn_num_queries", 8),
    )
    model.feature_fuser.load_state_dict(checkpoint["feature_fuser_state_dict"])
    return model


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_cross_platform_path(args.checkpoint_path)
    manifest_path = resolve_cross_platform_path(args.manifest_path)
    output_root = resolve_cross_platform_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    metrics_path = output_root / "metrics.json"

    if args.skip_if_exists and metrics_path.is_file():
        print(f"[SKIP] metrics exists: {metrics_path}")
        return

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = resolve_device(args.device or checkpoint["train_config"].get("device", ""))

    rows = load_manifest(manifest_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"No rows loaded from {manifest_path}")

    model = rebuild_model(checkpoint, device=device)
    tokenizer, prefix_tokens = build_tokenizer_and_prefix(model)
    batch_config = BatchEncodingConfig(
        n_mels=model.dims.n_mels,
        max_text_ctx=model.dims.n_text_ctx,
        pad_token_id=tokenizer.eot,
        prefix_tokens=prefix_tokens,
        tokenizer=tokenizer,
    )
    collate_fn = partial(collate_supervised_batch, config=batch_config)
    data_loader = DataLoader(
        VisSpeechPreparedDataset(rows),
        batch_size=max(1, args.batch_size),
        shuffle=False,
        num_workers=max(0, args.num_workers),
        collate_fn=collate_fn,
    )

    set_full_eval_mode(model)
    loss_values: List[float] = []
    eval_start = time.time()
    loss_progress = (
        tqdm(total=len(data_loader), desc="eval loss", dynamic_ncols=True, leave=True)
        if tqdm is not None
        else None
    )
    try:
        with torch.no_grad():
            for batch in data_loader:
                batch_loss = forward_fuser_only_loss(model, batch, device=device, use_images=True)
                loss_values.append(float(batch_loss.detach().cpu().item()))
                if loss_progress is not None:
                    loss_progress.update(1)
                    loss_progress.set_postfix(loss=f"{loss_values[-1]:.4f}")
    finally:
        if loss_progress is not None:
            loss_progress.close()
    avg_loss = sum(loss_values) / max(1, len(loss_values))

    existing_predictions = read_jsonl(predictions_path) if args.resume_from_predictions and predictions_path.is_file() else []
    predictions = transcribe_manifest_rows(
        model,
        rows,
        use_images=True,
        fp16=(device.type == "cuda"),
        existing_predictions=existing_predictions,
        output_path=predictions_path,
        log_prefix="TRANSCRIBE",
        log_every=args.log_every,
    )
    metric_summary = summarize_predictions(predictions)
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "rows": len(rows),
        "avg_loss": avg_loss,
        "wer": metric_summary["wer"],
        "cer": metric_summary["cer"],
        "seconds": time.time() - eval_start,
    }

    write_jsonl(predictions_path, predictions)
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[DONE] rows={len(rows)} avg_loss={avg_loss:.6f} wer={summary['wer']:.6f} cer={summary['cer']:.6f}")
    print(f"[DONE] predictions={predictions_path}")
    print(f"[DONE] metrics={metrics_path}")


if __name__ == "__main__":
    main()
