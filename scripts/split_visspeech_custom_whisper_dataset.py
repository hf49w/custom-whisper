from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from visspeech_custom_whisper_utils import (
    build_ordered_fieldnames,
    ensure_dir,
    load_manifest,
    resolve_cross_platform_path,
    write_csv_rows,
    write_jsonl,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_VISSPEECH_ROOT = REPO_ROOT / "data" / "visspeech" / "prepared"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split a prepared VisSpeech custom_whisper manifest into train/test partitions. "
            "By default, rows are grouped by yt_id so segments from the same source video stay together."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=str,
        default=str(DEFAULT_VISSPEECH_ROOT / "manifest.jsonl"),
        help="Prepared manifest created by prepare_visspeech_for_custom_whisper.py.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="",
        help="Directory where train/test manifests and split stats will be written.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Fraction of groups assigned to the test split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--group-by-field",
        type=str,
        default="yt_id",
        help="Field used to keep related rows in the same split. Use an empty string to split by row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = resolve_cross_platform_path(args.manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    output_root = (
        resolve_cross_platform_path(args.output_root)
        if args.output_root
        else manifest_path.parent / f"splits/by_{args.group_by_field or 'row'}_seed{args.seed}_test{int(args.test_ratio * 100):02d}"
    )
    ensure_dir(output_root)

    group_by_field = str(args.group_by_field or "").strip()
    groups: Dict[str, List[Dict[str, Any]]] = {}
    if group_by_field:
        for row in rows:
            group_key = str(row.get(group_by_field) or row.get("key") or "")
            groups.setdefault(group_key, []).append(row)
    else:
        for index, row in enumerate(rows):
            groups[f"row-{index:05d}"] = [row]

    group_items = list(groups.items())
    rng = random.Random(args.seed)
    rng.shuffle(group_items)

    total_rows = len(rows)
    target_test_rows = max(1, round(total_rows * args.test_ratio))
    test_rows: List[Dict[str, Any]] = []
    test_group_keys: List[str] = []
    train_rows: List[Dict[str, Any]] = []

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
            test_rows = [row for row in test_rows if row not in moved_rows]
        elif not test_rows and train_rows:
            moved_group_key, moved_rows = group_items[0]
            test_rows.extend(moved_rows)
            test_group_keys.append(moved_group_key)
            train_rows = [row for row in train_rows if row not in moved_rows]

    train_manifest_path = output_root / "train_manifest.jsonl"
    test_manifest_path = output_root / "test_manifest.jsonl"
    train_csv_path = output_root / "train_manifest.csv"
    test_csv_path = output_root / "test_manifest.csv"
    stats_path = output_root / "split_stats.json"

    write_jsonl(train_manifest_path, train_rows)
    write_jsonl(test_manifest_path, test_rows)
    fieldnames = build_ordered_fieldnames(rows)
    write_csv_rows(train_csv_path, train_rows, fieldnames)
    write_csv_rows(test_csv_path, test_rows, fieldnames)

    stats = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "group_by_field": group_by_field,
        "total_rows": total_rows,
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "total_groups": len(groups),
        "test_groups": len(test_group_keys),
        "train_manifest_path": str(train_manifest_path),
        "test_manifest_path": str(test_manifest_path),
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[DONE] train_rows={len(train_rows)} test_rows={len(test_rows)}")
    print(f"[DONE] train_manifest={train_manifest_path}")
    print(f"[DONE] test_manifest={test_manifest_path}")
    print(f"[DONE] stats={stats_path}")


if __name__ == "__main__":
    main()
