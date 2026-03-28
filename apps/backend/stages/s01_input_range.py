# [1단계] 어디를 고칠지 범위 잡기
# 사용자가 입력한 글에서 커서 근처 문장을 잘라내서
# 이후 단계에서 교정할 대상 범위를 정한다.

from typing import Any, Dict


def extract_input_range(full_text: str, cursor: int) -> Dict[str, Any]:
    """커서 앞쪽의 문장을 잘라서 '어디부터 어디까지 고칠지' 범위를 돌려준다."""
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
