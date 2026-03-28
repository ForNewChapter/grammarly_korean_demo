# 단어 형태 다루기 도구 모음
# 동사 원형에서 어간 떼기, 불규칙 활용 판별 등 단어 형태 관련 도구들.

from typing import Any, Dict, List, Optional, Tuple


def stem_from_lemma(lemma: str, fallback: str = "") -> str:
    """동사 원형에서 '다'를 떼어낸다. 예: '먹다' → '먹'"""
    normalized = str(lemma or "").strip()
    if normalized.endswith("다"):
        return normalized[:-1]
    return fallback or normalized


def infer_irregular_class_from_lemma(lemma: str) -> Optional[str]:
    """동사 원형을 보고 불규칙 활용 종류를 판별한다. 예: '하다' → 'HA'"""
    normalized = str(lemma or "").strip()
    if normalized.endswith("하다"):
        return "HA"
    if normalized.endswith("르다"):
        return "REU"
    if normalized.endswith("치다"):
        return "CHI"
    if normalized.endswith("추다"):
        return "CHU"
    if normalized.endswith("히다"):
        return "HI"
    if normalized.endswith("키다"):
        return "KI"
    if normalized.endswith("우다"):
        return "U"
    return None


def coarse_predicate_pos(tag: str) -> str:
    """품사 태그에서 동사/형용사 대분류만 가져온다. 예: 'VV-R' → 'VV'"""
    return str(tag or "").split("-", 1)[0]


def variant_suffix_slots(variant: Dict[str, Any]) -> List[str]:
    """후보 단어의 어미 패턴 목록을 정리해서 돌려준다."""
    raw_slots = variant.get("suffixSlots") or variant.get("suffixSlot") or []
    if isinstance(raw_slots, str):
        return [raw_slots]
    return [str(item) for item in raw_slots if item]
