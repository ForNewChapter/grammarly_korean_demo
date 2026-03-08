#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server import CorrectionEngine


@dataclass
class Case:
    text: str
    expected_substring: str
    field: str = 'suggestedText'
    forbidden_substrings: List[str] | None = None


CASES: List[Case] = [
    Case('아기를 나았다', '낳았다'),
    Case('아기를 나을 수 있을까?', '낳을 수'),
    Case('감기 낳앗으면 좋겟다', '나았으면'),
    Case('선생님이 답을 가르켰다', '가르쳤다'),
    Case('지금 v2인가? 교정이됀건가? 이 문장을 테스트해바', '교정이 된 건가? 이 문장을 테스트해 봐'),
    Case('내일 회의 내용을 다시한번 확인하자', '다시 한 번'),
    Case('학교를 갔다', '학교를 갔다', field='finalText'),
    Case('문제가 다르다', '문제가 다르다', field='finalText'),
    Case('시간을 맞혀서 오세요', '시간을 맞춰서 오세요', forbidden_substrings=['보세요']),
    Case('이 약을 먹으면 병이 금방 낮는다', '이 약을 먹으면 병이 금방 낫는다', forbidden_substrings=['막으면']),
    Case('병원을 갔다 오는 길이다', '병원을 갔다 오는 길이다', field='suggestedText', forbidden_substrings=['오늘 길이다', '보는 길이다']),
]


def main() -> None:
    engine = CorrectionEngine()
    rows = []
    passed = 0
    for case in CASES:
        result = engine.run_pipeline(case.text, len(case.text), 'inspect')
        actual = result.get(case.field, '')
        ok = case.expected_substring in actual
        violated = []
        for forbidden in case.forbidden_substrings or []:
            if forbidden in actual:
                ok = False
                violated.append(forbidden)
        if ok:
            passed += 1
        rows.append({
            'text': case.text,
            'field': case.field,
            'expected_substring': case.expected_substring,
            'actual': actual,
            'forbidden_substrings': case.forbidden_substrings or [],
            'violated_forbidden_substrings': violated,
            'pass': ok,
        })
    payload = {
        'total': len(rows),
        'passed': passed,
        'failed': len(rows) - passed,
        'rows': rows,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
