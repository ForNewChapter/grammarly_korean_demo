# 6단계: 구문 메모리와 형태소 혼동 규칙으로 표면형을 정규화한다.

from typing import Any, Dict, List

from model_loader import ModelLoader, MORPHEME_CONFUSION_RULES
from utils.edit_ops import (
    create_edit,
    dedupe_edits,
    is_range_protected,
    non_overlapping_edits,
    normalize_phrase_surface,
    tokenize_with_ranges,
)
from utils.hangul import hangul_ratio, typo_similarity


def _run_phrase_normalizations(
    models: ModelLoader,
    text: str,
    protected: List[Dict[str, Any]],
    *,
    single_token_only: bool = False,
) -> List[Dict[str, Any]]:
    """구문 메모리 인덱스를 조회하여 표면형 정규화 편집을 생성한다."""
    edits: List[Dict[str, Any]] = []
    token_ranges = tokenize_with_ranges(text)
    for index, (token, start, _) in enumerate(token_ranges):
        candidates = models.phrase_memory_index.get(token) or []
        if not candidates:
            continue
        for item in candidates:
            source = str(item.get("from") or "")
            replacement = str(item.get("to") or "")
            if not source or not replacement or source == replacement:
                continue
            source_parts = tokenize_with_ranges(source)
            if not source_parts:
                continue
            if single_token_only and len(source_parts) != 1:
                continue
            span_len = len(source_parts)
            if index + span_len > len(token_ranges):
                continue
            end = token_ranges[index + span_len - 1][2]
            span = {"start": start, "end": end}
            if is_range_protected(span, protected):
                continue
            source_text = text[start:end]
            if normalize_phrase_surface(source_text) != normalize_phrase_surface(source):
                continue
            edits.append(
                create_edit(
                    stage="PRE_NORMALIZE",
                    start=start,
                    end=end,
                    source_text=source_text,
                    replacement=replacement,
                    edit_type="SPELL",
                    confidence=float(item.get("confidence", 0.96)),
                    auto_applicable=True,
                    reason_tag=str(item.get("reasonTag") or "phrase_memory_auto"),
                )
            )
            break
    return edits


def _run_morpheme_normalizations(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """형태소 혼동 규칙을 적용하여 형태소 수준의 정규화 편집을 생성한다."""
    edits: List[Dict[str, Any]] = []
    token_ranges = tokenize_with_ranges(text)
    for token, start, end in token_ranges:
        span = {"start": start, "end": end}
        if is_range_protected(span, protected):
            continue
        if hangul_ratio(token) < 0.7 or len(token) < 2:
            continue

        analyses = models.kiwi_analyze_cached(token, top_n=1)
        if not analyses:
            continue
        morphs = analyses[0][0]
        if not morphs:
            continue

        changed = False
        min_score = 1.0
        morph_seq: List[tuple] = []
        for morph in morphs:
            normalized = MORPHEME_CONFUSION_RULES.get((morph.raw_form, morph.tag))
            if normalized:
                top = max(normalized, key=lambda item: float(item.get("generatorScore", 0.0)))
                morph_seq.append((top["replacement"], morph.tag))
                changed = True
                min_score = min(min_score, float(top.get("generatorScore", 0.91)))
            else:
                morph_seq.append((morph.form, morph.tag))
        if not changed:
            continue

        try:
            restored = models.kiwi.join(morph_seq, lm_search=True)
        except TypeError:
            try:
                restored = models.kiwi.join(morph_seq)
            except Exception:
                continue
        except Exception:
            continue

        if not restored or restored == token:
            continue
        similarity = typo_similarity(token, restored)
        if similarity < 0.68:
            continue
        edits.append(
            create_edit(
                stage="PRE_NORMALIZE",
                start=start,
                end=end,
                source_text=token,
                replacement=restored,
                edit_type="SPELL",
                confidence=max(0.95, min(0.99, round(min_score + ((similarity - 0.68) * 0.2), 4))),
                auto_applicable=True,
                reason_tag="kiwi_morpheme_normalization",
            )
        )
    return edits


def run_pre_spacing_phrase_normalizer(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """띄어쓰기 교정 전에 단일 토큰 구문 메모리 정규화를 수행한다."""
    return _run_phrase_normalizations(
        models,
        text,
        protected,
        single_token_only=True,
    )


def run_kiwi_pre_normalizer(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """구문 메모리와 형태소 혼동 규칙을 모두 적용하여 정규화 편집 목록을 반환한다."""
    edits = [
        *_run_phrase_normalizations(models, text, protected),
        *_run_morpheme_normalizations(models, text, protected),
    ]
    return non_overlapping_edits(dedupe_edits(edits))
