# 2단계: URL, 이메일, 전화번호 등 교정에서 보호할 영역을 감지하고 마스킹한다.

from typing import Any, Dict, List

from model_loader import PROTECTED_PATTERNS, DOMAIN_ENTITIES, USER_DICT


def detect_protected_spans(text: str) -> List[Dict[str, Any]]:
    """보호 패턴(URL, 이메일 등)과 도메인 엔티티, 사용자 사전 항목의 위치를 감지한다."""
    spans: List[Dict[str, Any]] = []
    for kind, regex in PROTECTED_PATTERNS:
        for m in regex.finditer(text):
            spans.append(
                {
                    "range": {"start": m.start(), "end": m.end()},
                    "kind": kind,
                    "text": m.group(0),
                }
            )
    for term in [*DOMAIN_ENTITIES, *USER_DICT]:
        at = 0
        while at < len(text):
            idx = text.find(term, at)
            if idx < 0:
                break
            spans.append(
                {
                    "range": {"start": idx, "end": idx + len(term)},
                    "kind": "ENTITY" if term in DOMAIN_ENTITIES else "USER_DICT",
                    "text": term,
                }
            )
            at = idx + len(term)

    spans.sort(key=lambda s: (s["range"]["start"], s["range"]["end"]))
    merged: List[Dict[str, Any]] = []
    for s in spans:
        if not merged:
            merged.append(s)
            continue
        last = merged[-1]
        if s["range"]["start"] >= last["range"]["end"]:
            merged.append(s)
            continue
        if s["range"]["end"] > last["range"]["end"]:
            last["range"]["end"] = s["range"]["end"]
            last["text"] = text[last["range"]["start"] : last["range"]["end"]]

    for i, s in enumerate(merged):
        s["placeholder"] = f"__{s['kind']}_{i}__"
    return merged


def apply_mask(text: str, protected: List[Dict[str, Any]]) -> str:
    """보호 영역을 플레이스홀더로 치환한 텍스트를 반환한다."""
    if not protected:
        return text
    out = []
    cursor = 0
    for s in protected:
        st, ed = s["range"]["start"], s["range"]["end"]
        out.append(text[cursor:st])
        out.append(s["placeholder"])
        cursor = ed
    out.append(text[cursor:])
    return "".join(out)
