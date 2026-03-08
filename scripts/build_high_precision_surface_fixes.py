#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_INPUT = Path("data/aihub_highfreq/train.jsonl")
DEFAULT_OUTPUT = Path("public/assets/rules/high_precision_surface_fixes.json")

MANUAL_FIXES = [
    {"from": "되요", "to": "돼요", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "왠", "to": "웬", "reasonTag": "common_misspelling", "confidence": 0.97},
    {"from": "삿어요", "to": "샀어요", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "잇어요", "to": "있어요", "reasonTag": "common_misspelling", "confidence": 0.96},
    {"from": "됀", "to": "된", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬", "to": "됐", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬다", "to": "됐다", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬고", "to": "됐고", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬네", "to": "됐네", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬어", "to": "됐어", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬는데", "to": "됐는데", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "됬으면", "to": "됐으면", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "좋겟다", "to": "좋겠다", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "좋겟네", "to": "좋겠네", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "좋겟어", "to": "좋겠어", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "모르겟다", "to": "모르겠다", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "모르겟지만", "to": "모르겠지만", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "모르겟습니다", "to": "모르겠습니다", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "해바", "to": "해봐", "reasonTag": "common_misspelling", "confidence": 0.97},
]

EXCLUDED_SOURCES = {
    "걍",
    "쫌",
    "글구",
    "글고",
    "그리구",
    "그리거",
    "근데",
    "저두",
    "어케",
    "혹쉬",
    "넘",
    "너므",
    "넘무",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Hub에서 고정밀 표면형 교정 사전을 생성합니다.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-count", type=int, default=30)
    parser.add_argument("--min-dominance", type=float, default=0.95)
    parser.add_argument("--min-similarity", type=float, default=0.74)
    return parser.parse_args()


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


def has_control_chars(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("C") for ch in text)


def typo_similarity(left: str, right: str) -> float:
    return float(SequenceMatcher(None, left, right).ratio())


def mostly_hangul(text: str) -> bool:
    if not text:
        return False
    hangul = len(re.findall(r"[가-힣]", text))
    return (hangul / len(text)) >= 0.7


def iter_errors(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip():
                continue
            row = json.loads(line)
            for error in row.get("errors", []):
                err_text = str(error.get("err_text") or "").strip()
                cor_text = str(error.get("cor_text") or "").strip()
                yield err_text, cor_text


def build_auto_fixes(args: argparse.Namespace) -> List[Dict[str, Any]]:
    grouped: Dict[str, Counter[str]] = defaultdict(Counter)
    for err_text, cor_text in iter_errors(args.input):
        if not err_text or not cor_text or err_text == cor_text:
            continue
        if " " in err_text or " " in cor_text:
            continue
        if len(err_text) > 12 or len(cor_text) > 12:
            continue
        if not mostly_hangul(err_text + cor_text):
            continue
        if punctuation_only_change(err_text, cor_text):
            continue
        if has_control_chars(err_text) or has_control_chars(cor_text):
            continue
        if re.search(r"[\"'“”‘’`]", err_text + cor_text):
            continue
        grouped[err_text][cor_text] += 1

    auto_fixes: List[Dict[str, Any]] = []
    for err_text, replacements in grouped.items():
        if err_text in EXCLUDED_SOURCES:
            continue
        total = sum(replacements.values())
        if not total:
            continue
        cor_text, count = replacements.most_common(1)[0]
        dominance = count / total
        similarity = typo_similarity(err_text, cor_text)
        if count < args.min_count:
            continue
        if dominance < args.min_dominance:
            continue
        if similarity < args.min_similarity:
            continue

        confidence = min(0.995, round(0.9 + min(0.09, count / 2000) + max(0.0, similarity - 0.8) * 0.2, 4))
        auto_fixes.append(
            {
                "from": err_text,
                "to": cor_text,
                "reasonTag": "high_precision_surface_fix",
                "confidence": confidence,
                "frequency": count,
                "dominance": round(dominance, 4),
                "similarity": round(similarity, 4),
            }
        )
    auto_fixes.sort(key=lambda item: (-int(item["frequency"]), item["from"], item["to"]))
    return auto_fixes


def merge_fixes(manual: List[Dict[str, Any]], auto: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for item in [*auto, *manual]:
        key = item["from"]
        existing = merged.get(key)
        if existing is None or float(item.get("confidence", 0.0)) >= float(existing.get("confidence", 0.0)):
            merged[key] = item
    rows = list(merged.values())
    rows.sort(key=lambda item: (-len(item["from"]), item["from"]))
    return rows


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    auto = build_auto_fixes(args)
    merged = merge_fixes(MANUAL_FIXES, auto)
    payload = {
        "generatedFrom": str(args.input.resolve()),
        "count": len(merged),
        "items": merged,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(merged), "top": merged[:20]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
