# 4단계: 하드코딩된 맞춤법 규칙과 고정밀 표면 치환 규칙으로 확실한 오류를 교정한다.

from typing import Any, Dict, List

from model_loader import ModelLoader
from utils.edit_ops import (
    create_edit,
    dedupe_edits,
    is_range_protected,
    non_overlapping_edits,
)


def _find_surface_fix_edits(
    text: str,
    protected: List[Dict[str, Any]],
    rules: List[Dict[str, Any]],
    stage_name: str,
    reason_prefix: str,
) -> List[Dict[str, Any]]:
    """규칙 목록을 순회하며 표면 치환 편집을 생성한다."""
    edits: List[Dict[str, Any]] = []
    for rule in rules:
        at = 0
        src = rule["from"]
        while at < len(text):
            idx = text.find(src, at)
            if idx < 0:
                break
            span = {"start": idx, "end": idx + len(src)}
            if not is_range_protected(span, protected):
                edits.append(
                    create_edit(
                        stage=stage_name,
                        start=idx,
                        end=idx + len(src),
                        source_text=src,
                        replacement=rule["to"],
                        edit_type="SPELL",
                        confidence=rule["confidence"],
                        auto_applicable=True,
                        reason_tag=f"{reason_prefix}:{rule['reasonTag']}",
                    )
                )
            at = idx + len(src)
    return non_overlapping_edits(dedupe_edits(edits))


def run_rule_corrector(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """하드코딩 규칙과 고정밀 표면 치환 규칙으로 확실한 오류를 교정한다."""
    return _find_surface_fix_edits(
        text=text,
        protected=protected,
        rules=models.rule_misspellings,
        stage_name="RULE",
        reason_prefix="surface_fix",
    )
