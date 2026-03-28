# 한글 글자 다루기 도구 모음
# 한글 자모 분해, 글자끼리 얼마나 비슷한지 비교 등 기본 도구들.

import re
from typing import Optional, Tuple


def hangul_ratio(text: str) -> float:
    """글자 중에 한글이 몇 %인지 알려준다."""
    if not text:
        return 0.0
    return len(re.findall(r"[가-힣]", text)) / len(text)


def hangul_syllable_parts(char: str) -> Optional[Tuple[int, int, int]]:
    """한글 한 글자를 초성·중성·종성으로 쪼갠다. 예: '한' → (ㅎ, ㅏ, ㄴ)"""
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
    """두 한글 글자가 얼마나 비슷한지 0~1 사이 점수로 알려준다."""
    if a == b:
        return 1.0
    parts_a = hangul_syllable_parts(a)
    parts_b = hangul_syllable_parts(b)
    if parts_a and parts_b:
        matches = sum(1 for left, right in zip(parts_a, parts_b) if left == right)
        return matches / 3.0
    return 0.0


def char_edit_distance(a: str, b: str) -> int:
    """한 글자를 다른 글자로 바꾸려면 몇 번 고쳐야 하는지 알려준다."""
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
    """두 단어가 오타 수준으로 얼마나 비슷한지 0~1 사이 점수로 알려준다."""
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
    """이 글자의 받침이 'ㄹ'인지 확인한다."""
    parts = hangul_syllable_parts(char)
    return bool(parts and parts[2] == 8)


def normalize_context_token(token: str) -> str:
    """단어에서 '은/는/이/가' 같은 조사를 떼어낸다."""
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
