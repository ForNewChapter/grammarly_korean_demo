# 편집(Edit) 객체의 생성, 적용, 중복 제거, 겹침 해소 유틸리티.

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
    """하나의 편집 객체를 생성한다."""
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
    """텍스트에 편집 목록을 적용한다."""
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
    """동일 범위·교체문의 중복 편집을 제거한다."""
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
    """겹치는 편집 중 우선순위가 높은 것만 남긴다."""
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
    """텍스트를 토큰으로 분리하고 각 토큰의 (문자열, 시작, 끝) 위치를 반환한다."""
    out = []
    for m in TOKEN_PATTERN.finditer(text):
        out.append((m.group(0), m.start(), m.end()))
    return out


def normalize_phrase_surface(text: str) -> str:
    """구문 표면형을 정규화한다 (공백 통일)."""
    return " ".join(TOKEN_PATTERN.findall(text)).strip()


def is_range_protected(span: Dict[str, int], protected: List[Dict[str, Any]]) -> bool:
    """주어진 범위가 보호 영역과 겹치는지 확인한다."""
    return any(
        span["start"] < p["range"]["end"] and span["end"] > p["range"]["start"] for p in protected
    )


def range_overlaps(span: Dict[str, int], other: Dict[str, int]) -> bool:
    """두 범위가 서로 겹치는지 확인한다."""
    return span["start"] < other["end"] and other["start"] < span["end"]
