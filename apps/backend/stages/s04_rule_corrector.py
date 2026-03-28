# [4단계] 확실한 오타 바로 고치기
# "되요→돼요", "잇어요→있어요" 처럼 100% 확실한 맞춤법 규칙을
# 사전에 등록해두고, 해당 패턴이 보이면 바로 고친다.

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
    """등록된 규칙("되요→돼요" 등)과 일치하는 부분을 찾아서 수정 목록을 만든다."""
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
    """확실한 맞춤법 규칙을 적용해서 틀린 부분을 바로 고친다."""
    return _find_surface_fix_edits(
        text=text,
        protected=protected,
        rules=models.rule_misspellings,
        stage_name="RULE",
        reason_prefix="surface_fix",
    )
