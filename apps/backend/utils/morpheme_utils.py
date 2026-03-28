# Kiwi 형태소 분석기 래퍼, 어간 추출, 재활용(reinflection) 등 형태소 관련 유틸리티.

from typing import Any, Dict, List, Optional, Tuple


def stem_from_lemma(lemma: str, fallback: str = "") -> str:
    """레마에서 어간을 추출한다 ('다' 제거)."""
    normalized = str(lemma or "").strip()
    if normalized.endswith("다"):
        return normalized[:-1]
    return fallback or normalized


def infer_irregular_class_from_lemma(lemma: str) -> Optional[str]:
    """레마로부터 불규칙 활용 클래스를 추론한다."""
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
    """태그에서 대분류 용언 품사(VV/VA/VX)를 추출한다."""
    return str(tag or "").split("-", 1)[0]


def variant_suffix_slots(variant: Dict[str, Any]) -> List[str]:
    """후보의 suffix slot 목록을 정규화하여 반환한다."""
    raw_slots = variant.get("suffixSlots") or variant.get("suffixSlot") or []
    if isinstance(raw_slots, str):
        return [raw_slots]
    return [str(item) for item in raw_slots if item]
