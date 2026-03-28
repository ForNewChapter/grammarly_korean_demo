# 1단계: 입력 텍스트에서 교정 대상 범위를 추출한다.

from typing import Any, Dict


def extract_input_range(full_text: str, cursor: int) -> Dict[str, Any]:
    """커서 위치를 기준으로 교정 대상 절(clause) 범위를 추출한다."""
    safe_cursor = max(0, min(cursor, len(full_text)))
    text_before_cursor = full_text[:safe_cursor]
    separators = ["\n", ".", "?", "!", ",", ";"]
    if safe_cursor > 0 and full_text[safe_cursor - 1] in separators:
        text_before_cursor = full_text[: safe_cursor - 1]
    last_separator_index = max(text_before_cursor.rfind(s) for s in separators)
    start = max(0, last_separator_index + 1)
    end = safe_cursor
    raw = full_text[start:end]
    clause_text = raw[-128:]
    clause_start = end - len(clause_text)
    return {
        "clauseText": clause_text,
        "clauseRange": {"start": clause_start, "end": end},
    }
