# [5단계] 띄어쓰기 고치기
# Kiwi 분석기가 제안하는 띄어쓰기와 원문을 비교해서
# 빠진 띄어쓰기를 찾아 넣어준다.

from typing import Any, Dict, List, Tuple

from model_loader import ModelLoader, COMPACT_COMPOUND_TOKEN_PATTERNS
from utils.edit_ops import (
    create_edit,
    is_range_protected,
    normalize_phrase_surface,
    tokenize_with_ranges,
)
from utils.hangul import hangul_ratio


def _has_strong_inline_fix(models: ModelLoader, token: str) -> bool:
    """이 단어에 대해 이미 확실한 교정 규칙이 있는지 확인한다."""
    if token in models.surface_fix_index:
        return True
    # _single_token_phrase_candidates 인라인 구현
    candidates = models.phrase_memory_index.get(token) or []
    for item in candidates:
        source = str(item.get("from") or "").strip()
        if not source:
            continue
        if len(tokenize_with_ranges(source)) != 1:
            continue
        if normalize_phrase_surface(source) != normalize_phrase_surface(token):
            continue
        return True
    return False


def _is_compact_compound_token(token: str) -> bool:
    """'부탁드립니다' 같은 붙여쓰기 복합어인지 확인한다."""
    if hangul_ratio(token) < 0.7 or len(token) < 4:
        return False
    return any(pattern.match(token) for pattern in COMPACT_COMPOUND_TOKEN_PATTERNS)


def _compact_preserve_ranges(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Tuple[int, int]]:
    """띄어쓰기를 넣으면 안 되는 복합어 구간을 찾아낸다."""
    ranges: List[Tuple[int, int]] = []
    for token, start, end in tokenize_with_ranges(text):
        span = {"start": start, "end": end}
        if is_range_protected(span, protected):
            continue
        if hangul_ratio(token) < 0.7 or len(token) < 2 or len(token) > 8:
            continue
        if _has_strong_inline_fix(models, token) or _is_compact_compound_token(token):
            ranges.append((start, end))
    return ranges


def spacing_candidates_from_kiwi(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Kiwi가 제안하는 띄어쓰기와 원문을 비교해서, 빠진 띄어쓰기 위치를 찾아낸다."""
    spaced = models.kiwi.space(text)
    candidates: List[Dict[str, Any]] = []
    preserve_ranges = _compact_preserve_ranges(models, text, protected)
    i, j = 0, 0
    while i < len(text) and j < len(spaced):
        if text[i] == spaced[j]:
            i += 1
            j += 1
            continue
        if spaced[j] == " ":
            probe = {"start": max(0, i - 1), "end": min(len(text), i + 1)}
            if not is_range_protected(probe, protected) and not any(
                start < i < end for start, end in preserve_ranges
            ):
                candidates.append({"index": i, "action": "INSERT_SPACE", "score": 0.96})
            j += 1
            continue
        if text[i] == " ":
            probe = {"start": i, "end": i + 1}
            left_probe = {"start": max(0, i - 1), "end": i + 1}
            if not is_range_protected(probe, protected) and not is_range_protected(
                left_probe, protected
            ):
                # 보수적으로 운영: 기존 공백 삭제는 과교정 위험이 커서 제안에서 제외한다.
                pass
            i += 1
            continue
        i += 1
        j += 1
    return {"spacedText": spaced, "candidates": candidates}


def classify_spacing_boundaries(
    candidates: List[Dict[str, Any]], profile: str
) -> List[Dict[str, Any]]:
    """글 스타일에 따라 띄어쓰기 후보를 걸러낸다. 일반 문장은 더 적극적으로 고친다."""
    out = []
    bonus = 0.03 if profile == "NORMAL" else 0.0
    for c in candidates:
        score = min(0.99, c["score"] + bonus)
        if score >= 0.9:
            out.append({**c, "score": score})
    return out


def spacing_to_edits(boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """확정된 띄어쓰기 위치를 실제 수정 명령 목록으로 바꾼다."""
    edits: List[Dict[str, Any]] = []
    for b in boundaries:
        if b["action"] == "INSERT_SPACE":
            edits.append(
                create_edit(
                    stage="SPACING",
                    start=b["index"],
                    end=b["index"],
                    source_text="",
                    replacement=" ",
                    edit_type="SPACE_INSERT",
                    confidence=b["score"],
                    auto_applicable=True,
                    reason_tag="kiwi_spacing",
                )
            )
        else:
            edits.append(
                create_edit(
                    stage="SPACING",
                    start=b["index"],
                    end=b["index"] + 1,
                    source_text=" ",
                    replacement="",
                    edit_type="SPACE_DELETE",
                    confidence=b["score"],
                    auto_applicable=True,
                    reason_tag="kiwi_spacing",
                )
            )
    return edits
