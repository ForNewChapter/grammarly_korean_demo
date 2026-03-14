#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from server import CorrectionEngine


@dataclass
class CoverageCase:
    text: str
    token: str
    expected_replacement: str


CASES: List[CoverageCase] = [
    CoverageCase("아기를 나았다", "나았다", "낳았다"),
    CoverageCase("아기를 나을 수 있을까?", "나을", "낳을"),
    CoverageCase("감기 낳앗으면 좋겟다", "낳았으면", "나았으면"),
    CoverageCase("선생님이 답을 가르켰다", "가르켰다", "가르쳤다"),
    CoverageCase("친구가 내 이름을 가르켰다", "가르켰다", "가리켰다"),
    CoverageCase("시간을 맞혀서 오세요", "맞춰서", "맞혀서"),
    CoverageCase("문제를 마추고 집에 갔다", "맞히고", "맞히고"),
    CoverageCase("이 약을 먹으면 병이 금방 낮는다", "낮는다", "낫는다"),
    CoverageCase("동생이 아파서 약을 먹고 많이 낮아졌다", "낮아졌다", "나아졌다"),
]


def _stage_map(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {trace["stageName"]: trace["outputArtifacts"] for trace in traces}


def _extract_group(stage_output: Any, token: str, rerank: bool = False) -> Optional[Dict[str, Any]]:
    if not isinstance(stage_output, list):
        return None
    for item in stage_output:
        if not isinstance(item, dict):
            continue
        original = item.get("original")
        if original == token:
            return item
        if rerank and item.get("original") == token:
            return item
    return None


def _candidate_rank_map(group: Optional[Dict[str, Any]], rerank: bool = False) -> List[str]:
    if not group:
        return []
    if rerank:
        ranked = group.get("ranked") or []
        return [str(item.get("replacement")) for item in ranked if item.get("source") != "ORIGINAL"]
    items = group.get("items") or []
    return [str(item.get("replacement")) for item in items if item.get("source") != "ORIGINAL"]


def _topk_hit(candidates: List[str], expected: str, k: int) -> bool:
    return expected in candidates[:k]


def main() -> None:
    engine = CorrectionEngine()
    rows: List[Dict[str, Any]] = []
    aggregate = {
        "open@1": 0,
        "open@3": 0,
        "open@5": 0,
        "verify@1": 0,
        "verify@3": 0,
        "verify@5": 0,
        "rerank@1": 0,
        "rerank@3": 0,
        "rerank@5": 0,
    }

    for case in CASES:
        result = engine.run_pipeline(case.text, len(case.text), "inspect")
        stages = _stage_map(result["traces"])

        open_group = _extract_group(stages.get("8. Open Candidate Generation"), case.token)
        verify_group = _extract_group(stages.get("8-2. Candidate Verifier"), case.token)
        rerank_group = _extract_group(stages.get("9. Reranker (KoBERT-MLM + KoELECTRA)"), case.token, rerank=True)

        open_candidates = _candidate_rank_map(open_group)
        verify_candidates = _candidate_rank_map(verify_group)
        rerank_candidates = _candidate_rank_map(rerank_group, rerank=True)

        row = {
            "text": case.text,
            "token": case.token,
            "expected": case.expected_replacement,
            "suggestedText": result.get("suggestedText", ""),
            "openCandidates": open_candidates[:5],
            "verifiedCandidates": verify_candidates[:5],
            "rerankedCandidates": rerank_candidates[:5],
            "open": {
                "hit@1": _topk_hit(open_candidates, case.expected_replacement, 1),
                "hit@3": _topk_hit(open_candidates, case.expected_replacement, 3),
                "hit@5": _topk_hit(open_candidates, case.expected_replacement, 5),
            },
            "verify": {
                "hit@1": _topk_hit(verify_candidates, case.expected_replacement, 1),
                "hit@3": _topk_hit(verify_candidates, case.expected_replacement, 3),
                "hit@5": _topk_hit(verify_candidates, case.expected_replacement, 5),
            },
            "rerank": {
                "hit@1": _topk_hit(rerank_candidates, case.expected_replacement, 1),
                "hit@3": _topk_hit(rerank_candidates, case.expected_replacement, 3),
                "hit@5": _topk_hit(rerank_candidates, case.expected_replacement, 5),
            },
        }
        rows.append(row)
        for stage_name, prefix in (("open", "open"), ("verify", "verify"), ("rerank", "rerank")):
            for k in (1, 3, 5):
                if row[stage_name][f"hit@{k}"]:
                    aggregate[f"{prefix}@{k}"] += 1

    total = len(rows)
    summary = {
        "total": total,
        "aggregate": {
            key: {
                "count": value,
                "rate": round(value / total, 4) if total else 0.0,
            }
            for key, value in aggregate.items()
        },
        "rows": rows,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
