from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


WINDOWS_DRIVE_RE = re.compile(r"^(?P<drive>[A-Za-z]):[\\/](?P<rest>.*)$")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PREPARED_ROOT = REPO_ROOT / "data" / "flickr8k" / "prepared"


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
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.lower() == ".csv":
        return read_csv_rows(path)
    raise ValueError(f"Unsupported manifest format: {path}")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a Flickr8k subset by keeping a fixed number of audio-caption rows per image, "
            "then split the subset into train/test partitions grouped by image_id."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_PREPARED_ROOT / "manifest.jsonl"),
        help="Prepared Flickr8k manifest created by prepare_flickr8k_for_custom_whisper.py.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Directory where subset/split manifests and stats will be written.",
    )
    parser.add_argument(
        "--rows-per-image",
        type=int,
        default=2,
        help="Maximum number of rows kept for each image_id.",
    )
    parser.add_argument(
        "--selection-mode",
        type=str,
        default="random",
        choices=["random", "first_n"],
        help="How rows are chosen within each image group.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=42,
        help="Random seed used when --selection-mode random.",
    )
    parser.add_argument(
        "--drop-images-with-fewer-rows",
        action="store_true",
        help="Drop images that have fewer rows than --rows-per-image instead of keeping the shorter group.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of selected rows assigned to the test split, grouped by image_id.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rows_per_image <= 0:
        raise ValueError("--rows-per-image must be > 0.")
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be between 0 and 1.")


def stable_row_sort_key(row: Dict[str, Any]) -> Tuple[int, str]:
    raw_caption_index = row.get("caption_index")
    try:
        caption_index = int(raw_caption_index)
    except (TypeError, ValueError):
        caption_index = 10**9
    key = str(row.get("key") or row.get("wav_filename") or row.get("wav_path") or "")
    return caption_index, key


def group_rows_by_image(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        image_id = str(row.get("image_id") or "").strip()
        if not image_id:
            raise ValueError("Every row must contain image_id for Flickr8k subset selection.")
        groups.setdefault(image_id, []).append(row)
    for image_id in groups:
        groups[image_id] = sorted(groups[image_id], key=stable_row_sort_key)
    return groups


def choose_rows_per_image(
    *,
    rows: Sequence[Dict[str, Any]],
    rows_per_image: int,
    selection_mode: str,
    selection_seed: int,
    drop_images_with_fewer_rows: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    groups = group_rows_by_image(rows)
    rng = random.Random(selection_seed)
    selected_keys = set()
    dropped_images_missing_rows = 0
    kept_short_groups = 0

    for image_id, group_rows in groups.items():
        if len(group_rows) < rows_per_image:
            if drop_images_with_fewer_rows:
                dropped_images_missing_rows += 1
                continue
            kept_short_groups += 1

        keep_count = min(rows_per_image, len(group_rows))
        if selection_mode == "first_n":
            chosen_rows = group_rows[:keep_count]
        else:
            chosen_rows = rng.sample(group_rows, k=keep_count)
            chosen_rows = sorted(chosen_rows, key=stable_row_sort_key)

        for row in chosen_rows:
            selected_keys.add(str(row.get("key") or ""))

    selected_rows = [row for row in rows if str(row.get("key") or "") in selected_keys]
    selected_group_sizes: Dict[str, int] = {}
    for row in selected_rows:
        image_id = str(row["image_id"])
        selected_group_sizes[image_id] = selected_group_sizes.get(image_id, 0) + 1

    stats = {
        "source_rows": len(rows),
        "source_images": len(groups),
        "selected_rows": len(selected_rows),
        "selected_images": len(selected_group_sizes),
        "rows_per_image_requested": rows_per_image,
        "selection_mode": selection_mode,
        "selection_seed": selection_seed,
        "drop_images_with_fewer_rows": bool(drop_images_with_fewer_rows),
        "dropped_images_with_fewer_rows": dropped_images_missing_rows,
        "kept_images_with_fewer_rows": kept_short_groups,
        "min_selected_rows_per_image": min(selected_group_sizes.values()) if selected_group_sizes else 0,
        "max_selected_rows_per_image": max(selected_group_sizes.values()) if selected_group_sizes else 0,
    }
    return selected_rows, stats


def split_rows_by_group(
    *,
    rows: Sequence[Dict[str, Any]],
    group_by_field: str,
    test_ratio: float,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        group_key = str(row.get(group_by_field) or row.get("key") or "").strip()
        groups.setdefault(group_key, []).append(row)

    group_items = list(groups.items())
    rng = random.Random(split_seed)
    rng.shuffle(group_items)

    total_rows = len(rows)
    target_test_rows = max(1, round(total_rows * test_ratio))
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    test_group_keys: List[str] = []

    if len(group_items) == 1:
        _, only_group_rows = group_items[0]
        train_rows.extend(only_group_rows)
    else:
        running_test_rows = 0
        total_groups = len(group_items)
        for index, (group_key, group_rows) in enumerate(group_items):
            remaining_groups_after = total_groups - index - 1
            should_assign_test = running_test_rows < target_test_rows and remaining_groups_after > 0
            if should_assign_test:
                test_rows.extend(group_rows)
                test_group_keys.append(group_key)
                running_test_rows += len(group_rows)
            else:
                train_rows.extend(group_rows)

        if not train_rows and test_group_keys:
            moved_group_key = test_group_keys.pop()
            moved_rows = groups[moved_group_key]
            train_rows.extend(moved_rows)
            moved_row_ids = {id(row) for row in moved_rows}
            test_rows = [row for row in test_rows if id(row) not in moved_row_ids]
        elif not test_rows and train_rows:
            moved_group_key, moved_rows = group_items[0]
            test_rows.extend(moved_rows)
            test_group_keys.append(moved_group_key)
            moved_row_ids = {id(row) for row in moved_rows}
            train_rows = [row for row in train_rows if id(row) not in moved_row_ids]

    stats = {
        "group_by_field": group_by_field,
        "test_ratio": test_ratio,
        "split_seed": split_seed,
        "total_rows": total_rows,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "total_groups": len(groups),
        "test_groups": len(test_group_keys),
        "train_groups": len(groups) - len(test_group_keys),
    }
    return train_rows, test_rows, stats


def main() -> None:
    args = parse_args()
    validate_args(args)

    manifest_path = resolve_cross_platform_path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    default_output_root = (
        manifest_path.parent
        / "subsets"
        / (
            f"{args.rows_per_image}_per_image"
            f"_{args.selection_mode}_sel{args.selection_seed}"
            f"_split{args.split_seed}_test{int(args.test_ratio * 100):02d}"
        )
    )
    output_root = ensure_dir(
        resolve_cross_platform_path(args.output_root) if args.output_root else default_output_root
    )

    selected_rows, selection_stats = choose_rows_per_image(
        rows=rows,
        rows_per_image=args.rows_per_image,
        selection_mode=args.selection_mode,
        selection_seed=args.selection_seed,
        drop_images_with_fewer_rows=args.drop_images_with_fewer_rows,
    )
    if not selected_rows:
        raise ValueError("No rows were selected. Check --rows-per-image and input manifest contents.")

    train_rows, test_rows, split_stats = split_rows_by_group(
        rows=selected_rows,
        group_by_field="image_id",
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
    )

    fieldnames = build_ordered_fieldnames(selected_rows)
    subset_manifest_jsonl = output_root / "subset_manifest.jsonl"
    subset_manifest_csv = output_root / "subset_manifest.csv"
    train_manifest_jsonl = output_root / "train_manifest.jsonl"
    train_manifest_csv = output_root / "train_manifest.csv"
    test_manifest_jsonl = output_root / "test_manifest.jsonl"
    test_manifest_csv = output_root / "test_manifest.csv"
    stats_path = output_root / "subset_split_stats.json"

    write_jsonl(subset_manifest_jsonl, selected_rows)
    write_csv_rows(subset_manifest_csv, selected_rows, fieldnames)
    write_jsonl(train_manifest_jsonl, train_rows)
    write_csv_rows(train_manifest_csv, train_rows, fieldnames)
    write_jsonl(test_manifest_jsonl, test_rows)
    write_csv_rows(test_manifest_csv, test_rows, fieldnames)

    stats = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "subset_manifest_jsonl": str(subset_manifest_jsonl),
        "subset_manifest_csv": str(subset_manifest_csv),
        "train_manifest_jsonl": str(train_manifest_jsonl),
        "train_manifest_csv": str(train_manifest_csv),
        "test_manifest_jsonl": str(test_manifest_jsonl),
        "test_manifest_csv": str(test_manifest_csv),
        "selection": selection_stats,
        "split": split_stats,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[DONE] selected_rows={selection_stats['selected_rows']} "
        f"selected_images={selection_stats['selected_images']}"
    )
    print(f"[DONE] train_rows={split_stats['train_rows']} test_rows={split_stats['test_rows']}")
    print(f"[DONE] subset_manifest={subset_manifest_jsonl}")
    print(f"[DONE] train_manifest={train_manifest_jsonl}")
    print(f"[DONE] test_manifest={test_manifest_jsonl}")
    print(f"[DONE] stats={stats_path}")


if __name__ == "__main__":
    main()
