# 수정 명령 다루기 도구 모음
# "여기를 이걸로 바꿔라" 하는 수정 명령을 만들고, 적용하고, 정리하는 도구들.

import re
from typing import Any, Dict, List, Tuple

TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")


def create_edit(
    stage: str,
    start: int,
    end: int,
    source_text: str,
    replacement: str,
    edit_type: str,
    confidence: float,
    auto_applicable: bool,
    reason_tag: str,
) -> Dict[str, Any]:
    """수정 명령 하나를 만든다. (어디를, 뭘로, 왜 바꾸는지)"""
    return {
        "stage": stage,
        "range": {"start": start, "end": end},
        "sourceText": source_text,
        "replacement": replacement,
        "editType": edit_type,
        "confidence": round(float(confidence), 4),
        "autoApplicable": auto_applicable,
        "reasonTag": reason_tag,
    }


def apply_edits(text: str, edits: List[Dict[str, Any]]) -> str:
    """수정 명령 목록을 텍스트에 실제로 적용해서 결과물을 돌려준다."""
    if not edits:
        return text
    sorted_edits = sorted(edits, key=lambda e: (e["range"]["start"], e["range"]["end"]))
    out: List[str] = []
    cursor = 0
    for e in sorted_edits:
        st, ed = e["range"]["start"], e["range"]["end"]
        out.append(text[cursor:st])
        out.append(e["replacement"])
        cursor = ed
    out.append(text[cursor:])
    return "".join(out)


def dedupe_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """같은 내용의 수정 명령이 중복되면 하나만 남긴다."""
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for e in edits:
        key = (e["range"]["start"], e["range"]["end"], e["replacement"])
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def non_overlapping_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """수정 범위가 겹치면, 더 중요한 것만 남기고 나머지는 뺀다."""
    ordered = sorted(
        edits,
        key=lambda e: (
            e["range"]["start"],
            -(e["range"]["end"] - e["range"]["start"]),
            -float(e.get("confidence", 0.0)),
        ),
    )
    kept: List[Dict[str, Any]] = []
    claimed: List[Tuple[int, int]] = []
    for edit in ordered:
        span = (edit["range"]["start"], edit["range"]["end"])
        if any(span[0] < end and span[1] > start for start, end in claimed):
            continue
        kept.append(edit)
        claimed.append(span)
    return kept


def tokenize_with_ranges(text: str) -> List[Tuple[str, int, int]]:
    """텍스트를 단어 단위로 쪼개고, 각 단어의 위치(시작~끝)도 함께 알려준다."""
    out = []
    for m in TOKEN_PATTERN.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def normalize_phrase_surface(text: str) -> str:
    """구문의 공백을 통일한다. ('먹고  싶다' → '먹고 싶다')"""
    return " ".join(TOKEN_PATTERN.findall(text)).strip()


def is_range_protected(span: Dict[str, int], protected: List[Dict[str, Any]]) -> bool:
    """이 범위가 보호 영역(URL 등)과 겹치는지 확인한다."""
    return any(
        span["start"] < p["range"]["end"] and span["end"] > p["range"]["start"] for p in protected
    )


def range_overlaps(span: Dict[str, int], other: Dict[str, int]) -> bool:
    """두 범위가 서로 겹치는지 확인한다."""
    return span["start"] < other["end"] and other["start"] < span["end"]
