#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT_DIR = Path("./data/aihub_highfreq")
DEFAULT_OUTPUT_DIR = Path("./data/edit_tagger_v2_dataset")
LABELS = ["KEEP", "SPACE_FIX", "OPEN_REPLACE"]
LABEL_PRIORITY = {
    "KEEP": 0,
    "SPACE_FIX": 1,
    "OPEN_REPLACE": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="정규화된 AI Hub 데이터를 EditTagger v2 학습용 JSONL로 변환합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-tokens", type=int, default=1)
    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=1.0,
        help="noisy example 수 대비 clean KEEP example 비율",
    )
    return parser.parse_args()


def whitespace_tokens(text: str) -> List[Tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def strip_punct_and_space(text: str) -> str:
    out: List[str] = []
    for char in text:
        if char.isspace():
            continue
        if unicodedata.category(char).startswith("P"):
            continue
        out.append(char)
    return "".join(out)


def punctuation_only_change(err_text: str, cor_text: str) -> bool:
    if err_text == cor_text:
        return False
    return strip_punct_and_space(err_text) == strip_punct_and_space(cor_text)


def spacing_change(err_text: str, cor_text: str) -> bool:
    if err_text == cor_text:
        return False
    return err_text.replace(" ", "") == cor_text.replace(" ", "")


def label_from_error(error: Dict[str, Any]) -> str:
    err_text = error.get("err_text") or ""
    cor_text = error.get("cor_text") or ""
    details = error.get("err_details") or {}
    tags = details.get("tags") if details.get("kind") == "tags" else []
    counts = details.get("counts") or {}

    if (
        "띄어쓰기" in tags
        or spacing_change(err_text, cor_text)
        or counts.get("spacing.insert", 0) > 0
        or counts.get("spacing.delete", 0) > 0
        or counts.get("spacing.replace", 0) > 0
    ):
        return "SPACE_FIX"

    if punctuation_only_change(err_text, cor_text):
        return "KEEP"

    return "OPEN_REPLACE"


def overlaps(left: Tuple[int, int], right: Tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def find_text_span(
    sentence: str,
    target: str,
    claimed: Sequence[Tuple[int, int]],
    search_start: int = 0,
) -> Optional[Tuple[int, int]]:
    if not target:
        return None

    start = sentence.find(target, search_start)
    while start >= 0:
        span = (start, start + len(target))
        if not any(overlaps(span, used) for used in claimed):
            return span
        start = sentence.find(target, start + 1)

    if search_start > 0:
        start = sentence.find(target, 0)
        while start >= 0:
            span = (start, start + len(target))
            if not any(overlaps(span, used) for used in claimed):
                return span
            start = sentence.find(target, start + 1)
    return None


def fallback_span_from_location(
    token_spans: Sequence[Tuple[str, int, int]],
    error: Dict[str, Any],
) -> Optional[Tuple[int, int]]:
    err_location = error.get("err_location")
    if not isinstance(err_location, int):
        return None
    if err_location < 0 or err_location >= len(token_spans):
        return None

    err_text = error.get("err_text") or ""
    err_token_count = max(1, len(err_text.split()))
    end_index = min(len(token_spans) - 1, err_location + err_token_count - 1)
    return (token_spans[err_location][1], token_spans[end_index][2])


def locate_error_spans(
    sentence: str,
    token_spans: Sequence[Tuple[str, int, int]],
    errors: Sequence[Dict[str, Any]],
) -> List[Tuple[Tuple[int, int], Dict[str, Any]]]:
    located: List[Tuple[Tuple[int, int], Dict[str, Any]]] = []
    claimed: List[Tuple[int, int]] = []
    cursor = 0

    def error_sort_key(item: Dict[str, Any]) -> Tuple[int, int]:
        idx = item.get("err_idx")
        loc = item.get("err_location")
        return (
            idx if isinstance(idx, int) else 10**9,
            loc if isinstance(loc, int) else 10**9,
        )

    for error in sorted(errors, key=error_sort_key):
        span = find_text_span(sentence, error.get("err_text") or "", claimed, cursor)
        if span is None:
            span = fallback_span_from_location(token_spans, error)
            if span is not None and any(overlaps(span, used) for used in claimed):
                span = None
        if span is None:
            continue
        claimed.append(span)
        cursor = span[1]
        located.append((span, error))

    return located


def assign_token_labels(
    token_spans: Sequence[Tuple[str, int, int]],
    located_errors: Sequence[Tuple[Tuple[int, int], Dict[str, Any]]],
) -> List[str]:
    labels = ["KEEP"] * len(token_spans)
    for span, error in located_errors:
        new_label = label_from_error(error)
        if new_label == "KEEP":
            continue
        for index, (_, start, end) in enumerate(token_spans):
            if overlaps((start, end), span) and LABEL_PRIORITY[new_label] >= LABEL_PRIORITY[labels[index]]:
                labels[index] = new_label
    return labels


def build_noisy_example(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sentence = record.get("err_sentence") or ""
    corrected = record.get("cor_sentence") or ""
    token_spans = whitespace_tokens(sentence)
    if not token_spans:
        return None
    located_errors = locate_error_spans(sentence, token_spans, record.get("errors") or [])
    labels = assign_token_labels(token_spans, located_errors)
    if all(label == "KEEP" for label in labels):
        return None
    return {
        "id": f"{record.get('id')}:noisy",
        "base_id": record.get("id"),
        "variant": "noisy",
        "split": record.get("split"),
        "error_type": record.get("error_type"),
        "source": record.get("source"),
        "text": sentence,
        "corrected_text": corrected,
        "tokens": [token for token, _, _ in token_spans],
        "labels": labels,
        "token_offsets": [[start, end] for _, start, end in token_spans],
        "error_count": record.get("error_count", len(record.get("errors") or [])),
        "located_error_count": len(located_errors),
        "errors": record.get("errors") or [],
    }


def build_clean_example(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sentence = record.get("cor_sentence") or ""
    token_spans = whitespace_tokens(sentence)
    if not token_spans:
        return None
    return {
        "id": f"{record.get('id')}:clean",
        "base_id": record.get("id"),
        "variant": "clean",
        "split": record.get("split"),
        "error_type": "CLEAN_REFERENCE",
        "source": record.get("source"),
        "text": sentence,
        "corrected_text": sentence,
        "tokens": [token for token, _, _ in token_spans],
        "labels": ["KEEP"] * len(token_spans),
        "token_offsets": [[start, end] for _, start, end in token_spans],
        "error_count": 0,
        "located_error_count": 0,
        "errors": [],
    }


def dedupe_examples(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[Tuple[str, Tuple[str, ...]], Dict[str, Any]] = {}
    for row in rows:
        key = (row["text"], tuple(row["labels"]))
        existing = dedup.get(key)
        if existing is None:
            dedup[key] = row
            continue
        if existing["variant"] == "clean" and row["variant"] == "noisy":
            dedup[key] = row
    return list(dedup.values())


def load_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def process_split(input_path: Path, output_path: Path, min_tokens: int, clean_ratio: float) -> Dict[str, Any]:
    noisy_rows: List[Dict[str, Any]] = []
    clean_rows: List[Dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    variant_counter: Counter[str] = Counter()
    error_type_counter: Counter[str] = Counter()

    for record in load_jsonl(input_path):
        noisy = build_noisy_example(record)
        if noisy is not None and len(noisy["tokens"]) >= min_tokens:
            noisy_rows.append(noisy)
            label_counter.update(noisy["labels"])
            variant_counter["noisy"] += 1
            error_type_counter[str(noisy["error_type"])] += 1

        clean = build_clean_example(record)
        if clean is not None and len(clean["tokens"]) >= min_tokens:
            clean_rows.append(clean)

    clean_limit = int(len(noisy_rows) * clean_ratio)
    selected_clean = clean_rows[:clean_limit] if clean_limit > 0 else []
    rows = dedupe_examples([*noisy_rows, *selected_clean])

    for row in selected_clean:
        label_counter.update(row["labels"])
        variant_counter["clean"] += 1
        error_type_counter[str(row["error_type"])] += 1

    count = write_jsonl(output_path, rows)
    return {
        "examples": count,
        "label_counts": dict(label_counter),
        "variant_counts": dict(variant_counter),
        "error_type_counts": dict(error_type_counter),
        "clean_ratio": clean_ratio,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "labels": LABELS,
        "splits": {},
    }

    for split in ("train", "validation"):
        split_summary = process_split(
            input_dir / f"{split}.jsonl",
            output_dir / f"{split}.jsonl",
            args.min_tokens,
            args.clean_ratio,
        )
        summary["splits"][split] = split_summary

    label_config = {
        "labels": LABELS,
        "label2id": {label: idx for idx, label in enumerate(LABELS)},
        "id2label": {idx: label for idx, label in enumerate(LABELS)},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "label_config.json").write_text(
        json.dumps(label_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
