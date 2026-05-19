from __future__ import annotations

import argparse
import json
import math
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.optim import AdamW
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
    SpecAugmentConfig,
    VisSpeechPreparedDataset,
    build_tokenizer_and_prefix,
    build_specaug_module,
    collate_supervised_batch,
    default_clip_model_name,
    ensure_dir,
    forward_fuser_only_loss,
    freeze_all_but_feature_fuser,
    load_manifest,
    resolve_cross_platform_path,
    set_fuser_training_mode,
    set_random_seed,
)


RESUME_COMPAT_KEYS = (
    "train_manifest",
    "whisper_model",
    "visual_encoder",
    "visual_fuser",
    "visual_pretrained",
    "image_size",
    "clip_model_name",
    "clip_return_sequence",
    "num_gmlp_layers",
    "num_resnet_layers",
    "p_speech",
    "dim_speech_inter",
    "dim_visual_inter",
    "use_residual",
    "use_layer_norm",
    "attn_num_heads",
    "attn_dropout",
    "attn_gate_init",
    "attn_num_queries",
    "batch_size",
    "seed",
    "max_train_samples",
    "specaug_enabled",
    "specaug_config",
)

LEGACY_RESUME_DEFAULTS = {
    "specaug_enabled": False,
    "specaug_config": None,
    "attn_num_heads": 8,
    "attn_dropout": 0.1,
    "attn_gate_init": -4.0,
    "attn_num_queries": 8,
    "max_train_samples": 0,
}

RELOCATABLE_PATH_KEYS = {
    "train_manifest",
    "clip_model_name",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune only the multimodal Whisper feature_fuser on a prepared VisSpeech manifest. "
            "Audio encoder, text decoder, and visual encoder remain frozen."
        )
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        required=True,
        help="Train manifest JSONL/CSV produced from the prepared VisSpeech dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Directory where checkpoints, config, and logs will be written.",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="medium.en",
        help="Base Whisper checkpoint used to initialize AudioImageWhisper.",
    )
    parser.add_argument(
        "--visual-encoder",
        type=str,
        required=True,
        choices=["resnet18", "resnet50", "resnet_gmlp", "clip"],
    )
    parser.add_argument(
        "--visual-fuser",
        type=str,
        required=True,
        choices=[
            "concat_proj",
            "proj_concat",
            "proj_concat_proj",
            "concat_temp",
            "cross_attn_gate",
            "attn_prefix",
            "gated_seq_concat",
        ],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--save-every-batches",
        type=int,
        default=100,
        help="Overwrite last.pt every N completed batches so interrupted runs can resume within an epoch.",
    )
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--whisper-download-root", type=str, default="")
    parser.add_argument("--strict-whisper-load", action="store_true")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional checkpoint path such as checkpoints/last.pt. Resumes optimizer, history, and epoch counter.",
    )
    parser.add_argument(
        "--allow-relocated-paths",
        action="store_true",
        help=(
            "Allow resume when path-like config values such as train_manifest or clip_model_name moved "
            "to a different machine or directory. Use only when the copied data/model assets are the same."
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore existing checkpoints in output_root and start this run from scratch.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--clip-model-name", type=str, default=default_clip_model_name())
    parser.add_argument("--clip-return-sequence", action="store_true")
    parser.add_argument("--num-gmlp-layers", type=int, default=1)
    parser.add_argument("--num-resnet-layers", type=int, default=18, choices=[18, 50])
    parser.add_argument("--p-speech", type=float, default=0.5)
    parser.add_argument("--dim-speech-inter", type=int, default=128)
    parser.add_argument("--dim-visual-inter", type=int, default=128)
    parser.add_argument("--attn-num-heads", type=int, default=8)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--attn-gate-init", type=float, default=-4.0)
    parser.add_argument("--attn-num-queries", type=int, default=8)
    parser.add_argument("--disable-fuser-residual", action="store_true")
    parser.add_argument("--disable-visual-layer-norm", action="store_true")
    parser.add_argument(
        "--enable-specaug",
        action="store_true",
        help=(
            "Enable SpecAugment during training. Defaults mirror the paper's ESPnet/Vorbis "
            "recipe unless overridden below."
        ),
    )
    parser.add_argument(
        "--disable-specaug-time-warp",
        action="store_true",
        help="Disable the SpecAugment time-warp stage.",
    )
    parser.add_argument(
        "--specaug-time-warp-window",
        type=int,
        default=5,
        help="SpecAugment time-warp window used when time warp is enabled.",
    )
    parser.add_argument(
        "--specaug-time-warp-mode",
        type=str,
        default="bicubic",
        choices=["bilinear", "bicubic"],
        help="Interpolation mode used by the time-warp stage.",
    )
    parser.add_argument(
        "--disable-specaug-freq-mask",
        action="store_true",
        help="Disable the SpecAugment frequency-mask stage.",
    )
    parser.add_argument(
        "--specaug-freq-mask-width-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0, 30),
        help="Min/max width for SpecAugment frequency masks.",
    )
    parser.add_argument(
        "--specaug-num-freq-mask",
        type=int,
        default=2,
        help="Number of SpecAugment frequency masks per sample.",
    )
    parser.add_argument(
        "--disable-specaug-time-mask",
        action="store_true",
        help="Disable the SpecAugment time-mask stage.",
    )
    parser.add_argument(
        "--specaug-time-mask-width-range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0, 40),
        help="Min/max width for SpecAugment time masks.",
    )
    parser.add_argument(
        "--specaug-num-time-mask",
        type=int,
        default=2,
        help="Number of SpecAugment time masks per sample.",
    )
    parser.set_defaults(visual_pretrained=True)
    parser.add_argument(
        "--visual-pretrained",
        dest="visual_pretrained",
        action="store_true",
        help="Use pretrained weights for supported visual encoders.",
    )
    parser.add_argument(
        "--no-visual-pretrained",
        dest="visual_pretrained",
        action="store_false",
        help="Disable pretrained weights for supported visual encoders.",
    )
    return parser.parse_args()


def resolve_device(raw_device: str) -> torch.device:
    if raw_device:
        return torch.device(raw_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_training_artifacts(output_root: Path, checkpoints_dir: Path) -> None:
    if checkpoints_dir.is_dir():
        for checkpoint_path in checkpoints_dir.glob("*.pt"):
            if checkpoint_path.is_file():
                checkpoint_path.unlink()
    for artifact_name in ("train_config.json", "train_history.json", "train_summary.json"):
        artifact_path = output_root / artifact_name
        if artifact_path.is_file():
            artifact_path.unlink()


def move_optimizer_state_to_device(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_specaug_config(args: argparse.Namespace) -> Optional[SpecAugmentConfig]:
    if not args.enable_specaug:
        return None
    return SpecAugmentConfig(
        apply_time_warp=not args.disable_specaug_time_warp,
        time_warp_window=args.specaug_time_warp_window,
        time_warp_mode=args.specaug_time_warp_mode,
        apply_freq_mask=not args.disable_specaug_freq_mask,
        freq_mask_width_range=tuple(args.specaug_freq_mask_width_range),
        num_freq_mask=args.specaug_num_freq_mask,
        apply_time_mask=not args.disable_specaug_time_mask,
        time_mask_width_range=tuple(args.specaug_time_mask_width_range),
        num_time_mask=args.specaug_num_time_mask,
    )


def build_resume_compat_config(
    *,
    args: argparse.Namespace,
    train_manifest_path: Path,
) -> Dict[str, Any]:
    specaug_config = build_specaug_config(args)
    return {
        "train_manifest": str(train_manifest_path),
        "whisper_model": args.whisper_model,
        "visual_encoder": args.visual_encoder,
        "visual_fuser": args.visual_fuser,
        "visual_pretrained": bool(args.visual_pretrained),
        "image_size": args.image_size,
        "clip_model_name": args.clip_model_name,
        "clip_return_sequence": bool(args.clip_return_sequence),
        "num_gmlp_layers": args.num_gmlp_layers,
        "num_resnet_layers": args.num_resnet_layers,
        "p_speech": args.p_speech,
        "dim_speech_inter": args.dim_speech_inter,
        "dim_visual_inter": args.dim_visual_inter,
        "use_residual": not args.disable_fuser_residual,
        "use_layer_norm": not args.disable_visual_layer_norm,
        "attn_num_heads": args.attn_num_heads,
        "attn_dropout": args.attn_dropout,
        "attn_gate_init": args.attn_gate_init,
        "attn_num_queries": args.attn_num_queries,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "specaug_enabled": specaug_config is not None,
        "specaug_config": specaug_config.to_dict() if specaug_config is not None else None,
    }


def validate_resume_checkpoint(
    checkpoint: Dict[str, Any],
    *,
    expected_config: Dict[str, Any],
    checkpoint_path: Path,
    allow_relocated_paths: bool,
) -> None:
    raw_checkpoint_config = checkpoint.get("train_config")
    if not isinstance(raw_checkpoint_config, dict):
        raise ValueError(f"Resume checkpoint is missing train_config: {checkpoint_path}")
    checkpoint_config = dict(raw_checkpoint_config)
    for key, default_value in LEGACY_RESUME_DEFAULTS.items():
        checkpoint_config.setdefault(key, default_value)
    mismatches: List[str] = []
    for key in RESUME_COMPAT_KEYS:
        expected_value = expected_config.get(key)
        actual_value = checkpoint_config.get(key)
        if key == "max_train_samples" and actual_value is None:
            actual_value = 0
        if (
            allow_relocated_paths
            and key in RELOCATABLE_PATH_KEYS
            and actual_value is not None
            and expected_value is not None
            and actual_value != expected_value
        ):
            continue
        if actual_value != expected_value:
            mismatches.append(f"{key}: checkpoint={actual_value!r} current={expected_value!r}")
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            f"Resume checkpoint is incompatible with current training arguments: {mismatch_text}"
        )


def infer_best_loss(history: List[Dict[str, Any]]) -> float:
    best_loss = math.inf
    for record in history:
        raw_loss = record.get("loss")
        if raw_loss is None:
            continue
        best_loss = min(best_loss, float(raw_loss))
    return best_loss


def write_train_summary(
    *,
    output_root: Path,
    checkpoints_dir: Path,
    history: List[Dict[str, Any]],
    best_loss: float,
    target_epochs: int,
    global_step: int,
    resume_from_path: Optional[Path],
) -> Dict[str, Any]:
    checkpoint_last_path = checkpoints_dir / "last.pt"
    checkpoint_best_path = checkpoints_dir / "best_train_loss.pt"

    if not checkpoint_last_path.is_file() and resume_from_path is not None:
        checkpoint_last_path = resume_from_path
    if not checkpoint_best_path.is_file() and resume_from_path is not None:
        fallback_best_path = resume_from_path.parent / "best_train_loss.pt"
        if fallback_best_path.is_file():
            checkpoint_best_path = fallback_best_path

    completed_epochs = int(history[-1]["epoch"]) if history else 0
    final_summary = {
        "best_train_loss": None if math.isinf(best_loss) else best_loss,
        "epochs": target_epochs,
        "completed_epochs": completed_epochs,
        "global_step": global_step,
        "last_epoch_loss": history[-1]["loss"] if history else None,
        "checkpoint_last": str(checkpoint_last_path.resolve()),
        "checkpoint_best_train_loss": str(checkpoint_best_path.resolve()),
        "resumed_from": str(resume_from_path.resolve()) if resume_from_path is not None else "",
    }
    (output_root / "train_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return final_summary


def build_checkpoint_payload(
    *,
    completed_epoch: int,
    resume_epoch: int,
    resume_batch_index: int,
    partial_epoch_loss_sum: float,
    partial_epoch_batches: int,
    global_step: int,
    history: List[Dict[str, Any]],
    run_config: Dict[str, Any],
    model: custom_whisper.AudioImageWhisper,
    optimizer: AdamW,
) -> Dict[str, Any]:
    return {
        "epoch": completed_epoch,
        "resume_epoch": resume_epoch,
        "resume_batch_index": resume_batch_index,
        "partial_epoch_loss_sum": partial_epoch_loss_sum,
        "partial_epoch_batches": partial_epoch_batches,
        "global_step": global_step,
        "train_history": history,
        "train_config": run_config,
        "feature_fuser_state_dict": model.feature_fuser.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }


def save_last_checkpoint(
    *,
    checkpoints_dir: Path,
    completed_epoch: int,
    resume_epoch: int,
    resume_batch_index: int,
    partial_epoch_loss_sum: float,
    partial_epoch_batches: int,
    global_step: int,
    history: List[Dict[str, Any]],
    run_config: Dict[str, Any],
    model: custom_whisper.AudioImageWhisper,
    optimizer: AdamW,
) -> Path:
    checkpoint_payload = build_checkpoint_payload(
        completed_epoch=completed_epoch,
        resume_epoch=resume_epoch,
        resume_batch_index=resume_batch_index,
        partial_epoch_loss_sum=partial_epoch_loss_sum,
        partial_epoch_batches=partial_epoch_batches,
        global_step=global_step,
        history=history,
        run_config=run_config,
        model=model,
        optimizer=optimizer,
    )
    checkpoint_path = checkpoints_dir / "last.pt"
    torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def main() -> None:
    args = parse_args()
    if args.resume_from and args.force_retrain:
        raise ValueError("--resume-from and --force-retrain cannot be used together.")
    set_random_seed(args.seed)

    train_manifest_path = resolve_cross_platform_path(args.train_manifest)
    if not train_manifest_path.is_file():
        raise FileNotFoundError(f"Train manifest not found: {train_manifest_path}")
    output_root = ensure_dir(resolve_cross_platform_path(args.output_root))
    checkpoints_dir = ensure_dir(output_root / "checkpoints")
    resume_from_path = resolve_cross_platform_path(args.resume_from) if args.resume_from else None
    if resume_from_path is not None and not resume_from_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from_path}")
    if args.force_retrain:
        print(f"[INFO] force_retrain=1 clearing previous training artifacts under {output_root}")
        cleanup_training_artifacts(output_root, checkpoints_dir)

    train_rows = load_manifest(train_manifest_path)
    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if not train_rows:
        raise ValueError(f"No train rows loaded from {train_manifest_path}")

    device = resolve_device(args.device)
    specaug_config = build_specaug_config(args)
    specaug_module = build_specaug_module(specaug_config) if specaug_config is not None else None
    resume_compat_config = build_resume_compat_config(args=args, train_manifest_path=train_manifest_path)
    resume_checkpoint: Optional[Dict[str, Any]] = None
    if resume_from_path is not None:
        resume_checkpoint = torch.load(resume_from_path, map_location="cpu")
        validate_resume_checkpoint(
            resume_checkpoint,
            expected_config=resume_compat_config,
            checkpoint_path=resume_from_path,
            allow_relocated_paths=args.allow_relocated_paths,
        )

    model = custom_whisper.load_audio_image_model(
        args.whisper_model,
        device=device,
        download_root=args.whisper_download_root or None,
        strict=args.strict_whisper_load,
        visual_encoder=args.visual_encoder,
        feature_fuser=args.visual_fuser,
        visual_pretrained=args.visual_pretrained,
        image_size=args.image_size,
        clip_model_name=args.clip_model_name,
        clip_return_sequence=args.clip_return_sequence,
        num_gmlp_layers=args.num_gmlp_layers,
        num_resnet_layers=args.num_resnet_layers,
        p_speech=args.p_speech,
        use_residual=not args.disable_fuser_residual,
        dim_speech_inter=args.dim_speech_inter,
        dim_visual_inter=args.dim_visual_inter,
        use_layer_norm=not args.disable_visual_layer_norm,
        attn_num_heads=args.attn_num_heads,
        attn_dropout=args.attn_dropout,
        attn_gate_init=args.attn_gate_init,
        attn_num_queries=args.attn_num_queries,
    )
    freeze_stats = freeze_all_but_feature_fuser(model)
    tokenizer, prefix_tokens = build_tokenizer_and_prefix(model)

    if args.visual_encoder == "resnet_gmlp" and args.num_gmlp_layers > 0:
        print(
            "[WARN] resnet_gmlp uses random gMLP layers unless you have a separately trained visual checkpoint. "
            "This script freezes the visual encoder exactly as requested."
        )
    if (
        args.visual_encoder == "clip"
        and args.visual_fuser in {"cross_attn_gate", "attn_prefix", "gated_seq_concat"}
        and not args.clip_return_sequence
    ):
        print(
            "[WARN] clip with an attention-based fuser is using pooled CLIP output as a single visual token. "
            "Pass --clip-return-sequence to use CLIP patch tokens."
        )

    batch_config = BatchEncodingConfig(
        n_mels=model.dims.n_mels,
        max_text_ctx=model.dims.n_text_ctx,
        pad_token_id=tokenizer.eot,
        prefix_tokens=prefix_tokens,
        tokenizer=tokenizer,
    )
    collate_fn = partial(collate_supervised_batch, config=batch_config)
    train_dataset = VisSpeechPreparedDataset(train_rows)

    trainable_parameters = [parameter for parameter in model.feature_fuser.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found in feature_fuser.")
    optimizer = AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)

    run_config: Dict[str, Any] = {
        "train_manifest": str(train_manifest_path),
        "output_root": str(output_root),
        "whisper_model": args.whisper_model,
        "visual_encoder": args.visual_encoder,
        "visual_fuser": args.visual_fuser,
        "visual_pretrained": bool(args.visual_pretrained),
        "image_size": args.image_size,
        "clip_model_name": args.clip_model_name,
        "clip_return_sequence": bool(args.clip_return_sequence),
        "num_gmlp_layers": args.num_gmlp_layers,
        "num_resnet_layers": args.num_resnet_layers,
        "p_speech": args.p_speech,
        "dim_speech_inter": args.dim_speech_inter,
        "dim_visual_inter": args.dim_visual_inter,
        "use_residual": not args.disable_fuser_residual,
        "use_layer_norm": not args.disable_visual_layer_norm,
        "attn_num_heads": args.attn_num_heads,
        "attn_dropout": args.attn_dropout,
        "attn_gate_init": args.attn_gate_init,
        "attn_num_queries": args.attn_num_queries,
        "max_train_samples": args.max_train_samples,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "save_every_batches": args.save_every_batches,
        "seed": args.seed,
        "device": str(device),
        "freeze_stats": freeze_stats,
        "resume_from": str(resume_from_path) if resume_from_path is not None else "",
        "allow_relocated_paths": bool(args.allow_relocated_paths),
        "specaug_enabled": specaug_config is not None,
        "specaug_config": specaug_config.to_dict() if specaug_config is not None else None,
    }
    (output_root / "train_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    history: List[Dict[str, Any]] = []
    best_loss = math.inf
    global_step = 0
    start_epoch = 1
    resume_batch_index = 0
    partial_epoch_loss_sum = 0.0
    partial_epoch_batches = 0

    if resume_checkpoint is not None:
        model.feature_fuser.load_state_dict(resume_checkpoint["feature_fuser_state_dict"])
        optimizer_state_dict = resume_checkpoint.get("optimizer_state_dict")
        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            move_optimizer_state_to_device(optimizer, device)
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr
                param_group["weight_decay"] = args.weight_decay
        else:
            print(f"[WARN] optimizer_state_dict missing in resume checkpoint: {resume_from_path}")
        history = list(resume_checkpoint.get("train_history") or [])
        best_loss = infer_best_loss(history)
        global_step = int(resume_checkpoint.get("global_step", 0))
        if "resume_epoch" in resume_checkpoint:
            start_epoch = int(resume_checkpoint.get("resume_epoch", 1))
            resume_batch_index = int(resume_checkpoint.get("resume_batch_index", 0))
            partial_epoch_loss_sum = float(resume_checkpoint.get("partial_epoch_loss_sum", 0.0))
            partial_epoch_batches = int(resume_checkpoint.get("partial_epoch_batches", 0))
        else:
            start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1

    print(f"[INFO] device={device}")
    print(f"[INFO] train_rows={len(train_rows)}")
    print(f"[INFO] freeze_stats={freeze_stats}")
    print(f"[INFO] output_root={output_root}")
    if specaug_config is not None:
        print(f"[INFO] specaug={specaug_config.to_dict()}")
    if resume_from_path is not None:
        completed_epochs = start_epoch - 1
        print(f"[INFO] resume_from={resume_from_path}")
        print(f"[INFO] completed_epochs={completed_epochs} target_epochs={args.epochs}")
        if args.allow_relocated_paths:
            print("[INFO] allow_relocated_paths=1")
        if resume_batch_index > 0:
            print(f"[INFO] resume_batch_index={resume_batch_index}")

    if start_epoch > args.epochs:
        (output_root / "train_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final_summary = write_train_summary(
            output_root=output_root,
            checkpoints_dir=checkpoints_dir,
            history=history,
            best_loss=best_loss,
            target_epochs=args.epochs,
            global_step=global_step,
            resume_from_path=resume_from_path,
        )
        print(f"[DONE] checkpoint already reached target epochs ({start_epoch - 1}/{args.epochs}); skipping training")
        print(f"[DONE] last_checkpoint={final_summary['checkpoint_last']}")
        print(f"[DONE] best_train_loss_checkpoint={final_summary['checkpoint_best_train_loss']}")
        print(f"[DONE] summary={output_root / 'train_summary.json'}")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.time()
        set_fuser_training_mode(model)
        epoch_generator = torch.Generator()
        epoch_generator.manual_seed(args.seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, args.batch_size),
            shuffle=True,
            generator=epoch_generator,
            num_workers=max(0, args.num_workers),
            collate_fn=collate_fn,
        )
        epoch_total_batches = len(train_loader)
        completed_batches_before_resume = resume_batch_index if epoch == start_epoch else 0
        running_loss = partial_epoch_loss_sum if epoch == start_epoch else 0.0
        running_batches = partial_epoch_batches if epoch == start_epoch else 0
        last_completed_batch = completed_batches_before_resume
        progress_bar = (
            tqdm(
                total=epoch_total_batches,
                desc=f"train epoch {epoch}/{args.epochs}",
                dynamic_ncols=True,
                leave=True,
            )
            if tqdm is not None
            else None
        )
        if progress_bar is not None and completed_batches_before_resume > 0:
            progress_bar.update(completed_batches_before_resume)
            progress_bar.set_postfix(
                loss="resume",
                elapsed="0s",
                eta="?",
            )

        try:
            for batch_index, batch in enumerate(train_loader, start=1):
                if batch_index <= completed_batches_before_resume:
                    continue
                optimizer.zero_grad(set_to_none=True)
                try:
                    loss = forward_fuser_only_loss(
                        model,
                        batch,
                        device=device,
                        use_images=True,
                        specaug_module=specaug_module,
                    )
                    loss.backward()
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=args.grad_clip_norm)
                    optimizer.step()
                except KeyboardInterrupt:
                    checkpoint_path = save_last_checkpoint(
                        checkpoints_dir=checkpoints_dir,
                        completed_epoch=epoch - 1,
                        resume_epoch=epoch,
                        resume_batch_index=last_completed_batch,
                        partial_epoch_loss_sum=running_loss,
                        partial_epoch_batches=running_batches,
                        global_step=global_step,
                        history=history,
                        run_config=run_config,
                        model=model,
                        optimizer=optimizer,
                    )
                    print(f"[INTERRUPTED] saved_checkpoint={checkpoint_path}")
                    raise

                loss_value = float(loss.detach().cpu().item())
                running_loss += loss_value
                running_batches += 1
                global_step += 1
                last_completed_batch = batch_index

                elapsed_seconds = time.time() - epoch_start_time
                avg_batch_seconds = elapsed_seconds / max(1, running_batches)
                remaining_batches = max(0, epoch_total_batches - batch_index)
                eta_seconds = avg_batch_seconds * remaining_batches

                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        elapsed=f"{elapsed_seconds:.0f}s",
                        eta=f"{eta_seconds:.0f}s",
                    )

                if batch_index == 1 or batch_index % max(1, args.log_every) == 0 or batch_index == epoch_total_batches:
                    message = (
                        f"[TRAIN] epoch={epoch}/{args.epochs} "
                        f"batch={batch_index}/{epoch_total_batches} "
                        f"loss={loss_value:.6f} "
                        f"elapsed={elapsed_seconds:.1f}s "
                        f"eta_epoch={eta_seconds:.1f}s"
                    )
                    if progress_bar is not None:
                        progress_bar.write(message)
                    else:
                        print(message)
                if (
                    args.save_every_batches > 0
                    and batch_index % max(1, args.save_every_batches) == 0
                    and batch_index < epoch_total_batches
                ):
                    checkpoint_path = save_last_checkpoint(
                        checkpoints_dir=checkpoints_dir,
                        completed_epoch=epoch - 1,
                        resume_epoch=epoch,
                        resume_batch_index=batch_index,
                        partial_epoch_loss_sum=running_loss,
                        partial_epoch_batches=running_batches,
                        global_step=global_step,
                        history=history,
                        run_config=run_config,
                        model=model,
                        optimizer=optimizer,
                    )
                    if progress_bar is not None:
                        progress_bar.write(
                            f"[CHECKPOINT] saved={checkpoint_path} batch={batch_index}/{epoch_total_batches}"
                        )
                    else:
                        print(f"[CHECKPOINT] saved={checkpoint_path} batch={batch_index}/{epoch_total_batches}")
        except KeyboardInterrupt:
            checkpoint_path = save_last_checkpoint(
                checkpoints_dir=checkpoints_dir,
                completed_epoch=epoch - 1,
                resume_epoch=epoch,
                resume_batch_index=last_completed_batch,
                partial_epoch_loss_sum=running_loss,
                partial_epoch_batches=running_batches,
                global_step=global_step,
                history=history,
                run_config=run_config,
                model=model,
                optimizer=optimizer,
            )
            print(f"[INTERRUPTED] saved_checkpoint={checkpoint_path}")
            raise
        finally:
            if progress_bar is not None:
                progress_bar.close()

        epoch_loss = running_loss / max(1, running_batches)
        epoch_seconds = time.time() - epoch_start_time
        epoch_record = {
            "epoch": epoch,
            "loss": epoch_loss,
            "seconds": epoch_seconds,
            "global_step": global_step,
        }
        history.append(epoch_record)
        print(
            f"[EPOCH] epoch={epoch} "
            f"loss={epoch_loss:.6f} "
            f"seconds={epoch_seconds:.1f}"
        )

        checkpoint_payload = build_checkpoint_payload(
            completed_epoch=epoch,
            resume_epoch=epoch + 1,
            resume_batch_index=0,
            partial_epoch_loss_sum=0.0,
            partial_epoch_batches=0,
            global_step=global_step,
            history=history,
            run_config=run_config,
            model=model,
            optimizer=optimizer,
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint_payload, checkpoints_dir / "best_train_loss.pt")

        if epoch % max(1, args.save_every) == 0:
            torch.save(checkpoint_payload, checkpoints_dir / f"epoch_{epoch:02d}.pt")

        torch.save(checkpoint_payload, checkpoints_dir / "last.pt")
        (output_root / "train_history.json").write_text(
            json.dumps(history, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        resume_batch_index = 0
        partial_epoch_loss_sum = 0.0
        partial_epoch_batches = 0

    final_summary = write_train_summary(
        output_root=output_root,
        checkpoints_dir=checkpoints_dir,
        history=history,
        best_loss=best_loss,
        target_epochs=args.epochs,
        global_step=global_step,
        resume_from_path=resume_from_path,
    )
    print(f"[DONE] last_checkpoint={checkpoints_dir / 'last.pt'}")
    print(f"[DONE] best_train_loss_checkpoint={checkpoints_dir / 'best_train_loss.pt'}")
    print(f"[DONE] summary={output_root / 'train_summary.json'}")


if __name__ == "__main__":
    main()
