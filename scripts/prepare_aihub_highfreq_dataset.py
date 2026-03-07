#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


DEFAULT_DATASET_ROOT = Path(
    "./143.인터페이스(자판-음성)별 고빈도 오류 교정 데이터/01-1.정식개방데이터"
)
DEFAULT_OUTPUT_DIR = Path("./data/aihub_highfreq")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Hub 인터페이스(자판-음성)별 고빈도 오류 교정 데이터를 학습용 JSONL로 정규화합니다."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="정식개방데이터 루트 디렉토리",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="출력 디렉토리",
    )
    return parser.parse_args()


def iter_label_zip_paths(dataset_root: Path) -> Iterator[Tuple[str, Path]]:
    for split_name, split_dir in (
        ("train", dataset_root / "Training" / "02.라벨링데이터"),
        ("validation", dataset_root / "Validation" / "02.라벨링데이터"),
    ):
        for zip_path in sorted(split_dir.glob("*.zip")):
            yield split_name, zip_path


def derive_error_type(zip_path: Path) -> str:
    stem = zip_path.stem
    if "_" in stem:
        return stem.split("_", 1)[1]
    return stem


def load_json_members(zip_path: Path) -> Iterator[Tuple[str, Dict[str, Any]]]:
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.lower().endswith(".json"):
                continue
            raw = zf.read(info)
            payload = json.loads(raw.decode("utf-8-sig"))
            yield info.filename.lstrip("/"), payload


def normalize_error_details(details: Any) -> Dict[str, Any]:
    if isinstance(details, list):
        return {
            "kind": "tags",
            "tags": details,
        }
    if isinstance(details, dict):
        flat_counts: Dict[str, int] = {}
        for outer_key, inner in details.items():
            if isinstance(inner, dict):
                for inner_key, value in inner.items():
                    flat_counts[f"{outer_key}.{inner_key}"] = int(value)
        return {
            "kind": "counts",
            "counts": flat_counts,
        }
    return {
        "kind": "raw",
        "value": details,
    }


def normalize_record(
    split: str,
    error_type: str,
    member_name: str,
    item: Dict[str, Any],
) -> Dict[str, Any]:
    metadata = item.get("metadata_info", {})
    annotation = item.get("annotation", {})
    errors = annotation.get("errors") or []

    normalized_errors: List[Dict[str, Any]] = []
    for error in errors:
        normalized_errors.append(
            {
                "err_idx": error.get("err_idx"),
                "err_location": error.get("err_location"),
                "err_text": error.get("err_text"),
                "cor_text": error.get("cor_text"),
                "edit_distance": error.get("edit_distance"),
                "err_details": normalize_error_details(error.get("err_details")),
            }
        )

    return {
        "split": split,
        "error_type": error_type,
        "member_name": member_name,
        "id": metadata.get("id"),
        "source": metadata.get("source"),
        "interface": metadata.get("interface"),
        "keyboard": metadata.get("keyboard"),
        "date": metadata.get("date"),
        "gender": metadata.get("gender"),
        "age": metadata.get("age"),
        "reg_date": annotation.get("reg_date"),
        "err_sentence": annotation.get("err_sentence"),
        "cor_sentence": annotation.get("cor_sentence"),
        "err_sentence_spell": annotation.get("err_sentence_spell"),
        "cor_sentence_spell": annotation.get("cor_sentence_spell"),
        "error_count": len(normalized_errors),
        "errors": normalized_errors,
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_records: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    summary: Dict[str, Any] = {
        "dataset_root": str(dataset_root),
        "splits": {},
    }

    for split, zip_path in iter_label_zip_paths(dataset_root):
        error_type = derive_error_type(zip_path)
        member_count = 0
        record_count = 0
        type_counter: Counter[str] = Counter()
        source_counter: Counter[str] = Counter()

        for member_name, payload in load_json_members(zip_path):
            member_count += 1
            for item in payload.get("data", []):
                record = normalize_record(split, error_type, member_name, item)
                grouped_records[split].append(record)
                record_count += 1
                type_counter[error_type] += 1
                if record.get("source"):
                    source_counter[str(record["source"])] += 1

        split_summary = summary["splits"].setdefault(
            split,
            {
                "zip_files": 0,
                "records": 0,
                "error_type_counts": {},
                "top_sources": {},
                "members": 0,
            },
        )
        split_summary["zip_files"] += 1
        split_summary["records"] += record_count
        split_summary["members"] += member_count
        split_summary["error_type_counts"][error_type] = (
            split_summary["error_type_counts"].get(error_type, 0) + type_counter[error_type]
        )
        for source_name, source_count in source_counter.items():
            split_summary["top_sources"][source_name] = (
                split_summary["top_sources"].get(source_name, 0) + source_count
            )

    output_counts: Dict[str, int] = {}
    for split, records in grouped_records.items():
        out_path = output_dir / f"{split}.jsonl"
        output_counts[split] = write_jsonl(out_path, records)

    for split_summary in summary["splits"].values():
        top_sources = Counter(split_summary["top_sources"]).most_common(10)
        split_summary["top_sources"] = [{name: count} for name, count in top_sources]

    summary["output_counts"] = output_counts
    summary["output_dir"] = str(output_dir)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
