from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
WAV_NAME_RE = re.compile(r"^(?P<image_id>.+)_(?P<caption_index>\d+)$")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_FLICKR8K_ROOT = REPO_ROOT / "data" / "flickr8k"
DEFAULT_PREPARED_ROOT = DEFAULT_FLICKR8K_ROOT / "prepared"


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


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


def read_caption_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def build_caption_index(rows: Sequence[Dict[str, str]]) -> Dict[str, List[str]]:
    captions_by_image: Dict[str, List[str]] = {}
    for row in rows:
        image_name = str(row.get("image") or "").strip()
        caption = str(row.get("caption") or "").strip()
        if not image_name or not caption:
            continue
        image_id = Path(image_name).stem
        captions_by_image.setdefault(image_id, []).append(caption)
    return captions_by_image


def build_image_index(images_root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for image_path in images_root.iterdir():
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        image_index.setdefault(image_path.stem, image_path.resolve())
    return image_index


def iter_audio_paths(audio_root: Path) -> List[Path]:
    return sorted(path.resolve() for path in audio_root.rglob("*.wav") if path.is_file())


def build_manifest_rows(
    *,
    audio_paths: Sequence[Path],
    captions_by_image: Dict[str, List[str]],
    image_index: Dict[str, Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for wav_path in audio_paths:
        match = WAV_NAME_RE.match(wav_path.stem)
        if not match:
            continue

        image_id = match.group("image_id")
        caption_index = int(match.group("caption_index"))
        captions = captions_by_image.get(image_id)
        image_path = image_index.get(image_id)

        if captions is None or caption_index >= len(captions) or image_path is None:
            continue

        rows.append(
            {
                "key": wav_path.stem,
                "image_id": image_id,
                "caption_index": caption_index,
                "annotation": captions[caption_index],
                "wav_path": str(wav_path),
                "image_path": str(image_path),
                "wav_filename": wav_path.name,
                "image_filename": image_path.name,
                "dataset": "flickr8k",
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Flickr8k manifest for the custom multimodal Whisper training pipeline. "
            "Rows are aligned by matching wav names like <image_id>_<caption_index>.wav "
            "to caption rows in captions.txt and images in the image directory."
        )
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default=str(DEFAULT_FLICKR8K_ROOT / "images"),
        help="Directory containing Flickr8k images.",
    )
    parser.add_argument(
        "--audio-root",
        type=str,
        default=str(DEFAULT_FLICKR8K_ROOT / "audio"),
        help="Directory containing Flickr8k wav files.",
    )
    parser.add_argument(
        "--captions-path",
        type=str,
        default=str(DEFAULT_FLICKR8K_ROOT / "captions" / "captions.txt"),
        help="CSV-style caption file with columns image,caption.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_PREPARED_ROOT),
        help="Directory where manifest files and stats will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_root = resolve_cross_platform_path(args.images_root)
    audio_root = resolve_cross_platform_path(args.audio_root)
    captions_path = resolve_cross_platform_path(args.captions_path)
    output_root = ensure_dir(resolve_cross_platform_path(args.output_root))

    if not images_root.is_dir():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Audio root not found: {audio_root}")
    if not captions_path.is_file():
        raise FileNotFoundError(f"Captions file not found: {captions_path}")

    caption_rows = read_caption_rows(captions_path)
    captions_by_image = build_caption_index(caption_rows)
    image_index = build_image_index(images_root)
    audio_paths = iter_audio_paths(audio_root)
    manifest_rows = build_manifest_rows(
        audio_paths=audio_paths,
        captions_by_image=captions_by_image,
        image_index=image_index,
    )

    if not manifest_rows:
        raise ValueError("No aligned Flickr8k rows were produced.")

    fieldnames = build_ordered_fieldnames(manifest_rows)
    manifest_jsonl = output_root / "manifest.jsonl"
    manifest_csv = output_root / "metadata.csv"
    stats_json = output_root / "stats.json"
    command_txt = output_root / "run_command.txt"

    write_jsonl(manifest_jsonl, manifest_rows)
    write_csv_rows(manifest_csv, manifest_rows, fieldnames)

    unique_images = {row["image_id"] for row in manifest_rows}
    stats = {
        "images_root": str(images_root),
        "audio_root": str(audio_root),
        "captions_path": str(captions_path),
        "output_root": str(output_root),
        "caption_rows": len(caption_rows),
        "aligned_rows": len(manifest_rows),
        "aligned_images": len(unique_images),
        "manifest_path": str(manifest_jsonl),
        "metadata_csv": str(manifest_csv),
    }
    stats_json.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    command_txt.write_text(" ".join(map(str, os.sys.argv)), encoding="utf-8")

    print(f"[INFO] images_root={images_root}")
    print(f"[INFO] audio_root={audio_root}")
    print(f"[INFO] captions_path={captions_path}")
    print(f"[DONE] aligned_rows={len(manifest_rows)} aligned_images={len(unique_images)}")
    print(f"[DONE] manifest={manifest_jsonl}")
    print(f"[DONE] metadata={manifest_csv}")
    print(f"[DONE] stats={stats_json}")


if __name__ == "__main__":
    main()
