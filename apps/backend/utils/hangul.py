# 한글 자모 분해, 음절 처리, 유사도 계산 등 한글 텍스트 기초 유틸리티.

import re
from typing import Optional, Tuple


def hangul_ratio(text: str) -> float:
    """텍스트 내 한글 비율을 반환한다."""
    if not text:
        return 0.0
    return len(re.findall(r"[가-힣]", text)) / len(text)


def hangul_syllable_parts(char: str) -> Optional[Tuple[int, int, int]]:
    """한글 음절을 (초성, 중성, 종성) 인덱스로 분해한다."""
    if not char or len(char) != 1:
        return None
    code = ord(char)
    if code < 0xAC00 or code > 0xD7A3:
        return None
    offset = code - 0xAC00
    return (
        offset // 588,
        (offset % 588) // 28,
        offset % 28,
    )


def char_similarity(a: str, b: str) -> float:
    """두 한글 음절의 자모 유사도를 반환한다 (0.0~1.0)."""
    if a == b:
        return 1.0
    parts_a = hangul_syllable_parts(a)
    parts_b = hangul_syllable_parts(b)
    if parts_a and parts_b:
        matches = sum(1 for left, right in zip(parts_a, parts_b) if left == right)
        return matches / 3.0
    return 0.0


def char_edit_distance(a: str, b: str) -> int:
    """두 문자열 사이의 편집 거리(레벤슈타인)를 반환한다."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def typo_similarity(original: str, candidate: str) -> float:
    """오타 수준의 표면 유사도를 반환한다 (0.0~1.0)."""
    if not original or not candidate:
        return 0.0
    if original == candidate:
        return 1.0

    prefix = 0
    while prefix < min(len(original), len(candidate)) and original[prefix] == candidate[prefix]:
        prefix += 1

    suffix = 0
    while (
        suffix < min(len(original) - prefix, len(candidate) - prefix)
        and original[len(original) - 1 - suffix] == candidate[len(candidate) - 1 - suffix]
    ):
        suffix += 1

    preserved = (prefix + suffix) / max(len(original), len(candidate))
    core_original = original[prefix : len(original) - suffix if suffix else len(original)]
    core_candidate = candidate[prefix : len(candidate) - suffix if suffix else len(candidate)]

    if not core_original or not core_candidate:
        core_similarity = 0.0
    else:
        pair_count = max(len(core_original), len(core_candidate))
        pair_scores = []
        for idx in range(pair_count):
            left = core_original[idx] if idx < len(core_original) else ""
            right = core_candidate[idx] if idx < len(core_candidate) else ""
            if left and right:
                pair_scores.append(char_similarity(left, right))
            else:
                pair_scores.append(0.0)
        core_similarity = sum(pair_scores) / pair_count

    return round(float((0.55 * preserved) + (0.45 * core_similarity)), 4)


def has_jongseong_l(char: str) -> bool:
    """해당 음절의 종성이 'ㄹ'인지 확인한다."""
    parts = hangul_syllable_parts(char)
    return bool(parts and parts[2] == 8)


def normalize_context_token(token: str) -> str:
    """문맥 토큰에서 조사를 제거하여 어근만 남긴다."""
    normalized = re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", token)
    if not re.fullmatch(r"[가-힣]+", normalized):
        return normalized
    particles = (
        "으로는",
        "에게서",
        "한테서",
        "이라도",
        "처럼은",
        "으로",
        "에게",
        "한테",
        "에서",
        "부터",
        "까지",
        "처럼",
        "보다",
        "이랑",
        "랑",
        "이나",
        "나",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "에",
        "의",
        "와",
        "과",
        "도",
        "만",
        "로",
    )
    for particle in particles:
        if normalized.endswith(particle) and len(normalized) > len(particle) + 1:
            return normalized[: -len(particle)]
    return normalized
