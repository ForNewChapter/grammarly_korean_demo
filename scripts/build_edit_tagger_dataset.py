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
DEFAULT_OUTPUT_DIR = Path("./data/edit_tagger_dataset")
LABEL_PRIORITY = {
    "KEEP": 0,
    "PUNCT_FIX": 1,
    "SPACE_FIX": 2,
    "OPEN_REPLACE": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="정규화된 AI Hub 고빈도 오류 데이터를 EditTagger 학습용 token-label JSONL로 변환합니다."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=1,
        help="이 토큰 수 미만 예시는 제외합니다.",
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


def label_from_error(error_type: str, error: Dict[str, Any]) -> str:
    err_text = error.get("err_text") or ""
    cor_text = error.get("cor_text") or ""
    details = error.get("err_details") or {}
    tags = []
    if details.get("kind") == "tags":
        tags = details.get("tags") or []
    counts = details.get("counts") or {}

    if "띄어쓰기" in tags or spacing_change(err_text, cor_text) or counts.get("spacing.insert", 0) > 0 or counts.get("spacing.delete", 0) > 0 or counts.get("spacing.replace", 0) > 0:
        return "SPACE_FIX"
    if "문장부호" in tags or punctuation_only_change(err_text, cor_text) or counts.get("mark.insert", 0) > 0 or counts.get("mark.delete", 0) > 0 or counts.get("mark.replace", 0) > 0:
        return "PUNCT_FIX"
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
        target = error.get("err_text") or ""
        span = find_text_span(sentence, target, claimed, cursor)
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
    error_type: str,
) -> List[str]:
    labels = ["KEEP"] * len(token_spans)
    for span, error in located_errors:
        new_label = label_from_error(error_type, error)
        for index, (_, start, end) in enumerate(token_spans):
            if overlaps((start, end), span) and LABEL_PRIORITY[new_label] >= LABEL_PRIORITY[labels[index]]:
                labels[index] = new_label
    return labels


def normalize_example(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    sentence = record.get("err_sentence") or ""
    corrected = record.get("cor_sentence") or ""
    token_spans = whitespace_tokens(sentence)
    if not token_spans:
        return None

    located_errors = locate_error_spans(sentence, token_spans, record.get("errors") or [])
    labels = assign_token_labels(token_spans, located_errors, record.get("error_type") or "")
    tokens = [token for token, _, _ in token_spans]

    return {
        "id": record.get("id"),
        "split": record.get("split"),
        "error_type": record.get("error_type"),
        "source": record.get("source"),
        "text": sentence,
        "corrected_text": corrected,
        "tokens": tokens,
        "labels": labels,
        "token_offsets": [[start, end] for _, start, end in token_spans],
        "error_count": record.get("error_count", len(record.get("errors") or [])),
        "located_error_count": len(located_errors),
        "errors": record.get("errors") or [],
    }


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


def process_split(input_path: Path, output_path: Path, min_tokens: int) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    label_counter: Counter[str] = Counter()
    error_type_counter: Counter[str] = Counter()
    dropped = 0

    for record in load_jsonl(input_path):
        example = normalize_example(record)
        if example is None or len(example["tokens"]) < min_tokens:
            dropped += 1
            continue
        rows.append(example)
        label_counter.update(example["labels"])
        error_type_counter[example["error_type"]] += 1

    count = write_jsonl(output_path, rows)
    return {
        "examples": count,
        "dropped": dropped,
        "label_counts": dict(label_counter),
        "error_type_counts": dict(error_type_counter),
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": {},
    }

    for split in ("train", "validation"):
        split_summary = process_split(
            input_dir / f"{split}.jsonl",
            output_dir / f"{split}.jsonl",
            args.min_tokens,
        )
        summary["splits"][split] = split_summary

    labels = ["KEEP", "OPEN_REPLACE", "SPACE_FIX", "PUNCT_FIX"]
    label_config = {
        "labels": labels,
        "label2id": {label: index for index, label in enumerate(labels)},
        "id2label": {index: label for index, label in enumerate(labels)},
    }

    with (output_dir / "label_config.json").open("w", encoding="utf-8") as fp:
        json.dump(label_config, fp, ensure_ascii=False, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
