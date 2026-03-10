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
    Case('회의 시간에 마춰서 자료를 보내주세요', '맞춰서', forbidden_substrings=['마 춰서']),
    Case('문제를 마추고 집에 갔다', '맞히고', forbidden_substrings=['마 주고', '마 추고']),
    Case('정답을 맞췄다', '정답을 맞혔다'),
    Case('정답을 맞추었다', '정답을 맞혔다'),
    Case('동생이 아파서 약을 먹고 많이 낮아졌다', '나아졌다'),
    Case('기온이 빨리 나아졌으면 좋겠다', '나아졌으면', forbidden_substrings=['낳아졌으면', '나가졌으면', '나어졌으면']),
    Case('기온이 많이 낮아졌다', '기온이 많이 낮아졌다', field='finalText', forbidden_substrings=['나아졌다']),
    Case('시간을 맞혀서 와라', '시간을 맞춰서 와라', forbidden_substrings=['보세요']),
    Case('시간을 맞춰서 와라', '시간을 맞춰서 와라', field='suggestedText', forbidden_substrings=['맞혀서']),
    Case('왠지 오늘은 기분이 좋다', '왠지 오늘은 기분이 좋다', field='finalText', forbidden_substrings=['웬지']),
    Case('정답을 맞혔다', '정답을 맞혔다', field='suggestedText', forbidden_substrings=['맞췄다', '맞추었다']),
    Case('웬일로 이렇게 일찍 왔어', '웬일로 이렇게 일찍 왔어', field='finalText'),
    Case('짐을 바닦에 내려놨다', '짐을 바닥에 내려 놨다'),
    Case('어제 본 사람이 오늘도 같은 자리에 안자 있었다', '앉아 있었다'),
    Case('아기를 나을 수 있을까? 낳다야 낫다야?', '아기를 낳을 수 있을까? 낳다야 낫다야?', forbidden_substrings=['낳다이', '낫다냐']),
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
