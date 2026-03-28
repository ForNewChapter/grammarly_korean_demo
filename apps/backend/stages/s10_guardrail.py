# [10단계] 안전장치 — 위험한 교정 걸러내기
# 교정이 오히려 글을 망칠 수 있는 경우(영문 포함, 너무 큰 변경 등)를 걸러내고,
# 각 교정에 "자동 적용/제안만/거부" 딱지를 붙인다.

import re
from typing import Any, Dict, List, Optional

from model_loader import ModelLoader
from stages.s07_edit_tagger import _is_curated_candidate_source
from utils.edit_ops import create_edit, is_range_protected


def guardrail(
    original: str,
    replacement: str,
    span: Dict[str, int],
    profile: str,
    protected: List[Dict[str, Any]],
    score: float,
) -> Dict[str, Any]:
    """이 교정이 안전한지 검사한다. 위험하면 거부한다."""
    reason_codes: List[str] = []
    if is_range_protected(span, protected):
        return {"decision": "REJECT", "reasonCodes": ["PROTECTED_SPAN"], "score": 0.01}
    if re.search(r"[A-Za-z0-9]", original) or re.search(r"[A-Za-z0-9]", replacement):
        return {"decision": "REJECT", "reasonCodes": ["ALNUM_RISK"], "score": 0.05}
    edit_ratio = abs(len(replacement) - len(original)) / max(1, len(original))
    if edit_ratio > 0.25:
        reason_codes.append("EDIT_RATIO_HIGH")
    if profile == "CHAT":
        reason_codes.append("CHAT_PROFILE")
    reason_codes.append("LEXICAL_REPLACEMENT")
    return {"decision": "SUGGEST_ONLY", "reasonCodes": reason_codes, "score": score}


def policy(edit_type: str, guardrail_decision: str, confidence: float) -> str:
    """교정 종류와 신뢰도에 따라 '자동 적용/제안만/거부'를 결정한다."""
    if guardrail_decision == "REJECT":
        return "REJECT"
    if edit_type in {"SPACE_INSERT", "SPACE_DELETE", "SPELL"} and confidence >= 0.95:
        return "AUTO_APPLY"
    if edit_type == "OPEN_REPLACE":
        return "SUGGEST_ONLY"
    return guardrail_decision


def _proposal_option_for_candidate(
    models: ModelLoader,
    group: Dict[str, Any],
    candidate: Dict[str, Any],
    profile: str,
    protected: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """하나의 교정 후보에 대해 안전 검사를 하고 점수를 매긴다."""
    original = str(group["original"])
    span = group["span"]
    replacement = str(candidate.get("replacement") or original)
    source = str(candidate.get("source") or "ORIGINAL")
    final_score = float(candidate.get("finalScore", candidate.get("rerankScore", 0.2)))

    if replacement == original or source == "ORIGINAL":
        return {
            "span": span,
            "original": original,
            "replacement": original,
            "source": "ORIGINAL",
            "candidateScore": 0.2,
            "optionScore": 0.2,
            "guardrail": {"decision": "PASS", "reasonCodes": ["KEEP_ORIGINAL"], "score": 0.2},
            "decision": "KEEP",
            "applyEdit": False,
        }

    gr = guardrail(
        original=original,
        replacement=replacement,
        span=span,
        profile=profile,
        protected=protected,
        score=final_score,
    )
    decision = policy("OPEN_REPLACE", gr["decision"], final_score)
    source_penalty = 0.03 if not _is_curated_candidate_source(source) else 0.0
    edit_ratio_penalty = 0.05 if "EDIT_RATIO_HIGH" in gr["reasonCodes"] else 0.0
    chat_penalty = 0.02 if "CHAT_PROFILE" in gr["reasonCodes"] else 0.0
    family_bonus = 0.02 if candidate.get("familyMatch") else 0.0
    option_score = max(
        0.2,
        final_score - 0.08 - source_penalty - edit_ratio_penalty - chat_penalty + family_bonus,
    )
    return {
        "span": span,
        "original": original,
        "replacement": replacement,
        "source": source,
        "candidateScore": round(final_score, 4),
        "optionScore": round(float(option_score), 4),
        "guardrail": gr,
        "decision": decision,
        "applyEdit": decision != "REJECT",
    }


def _build_group_proposal_options(
    models: ModelLoader,
    group: Dict[str, Any],
    profile: str,
    protected: List[Dict[str, Any]],
    candidate_limit: int = 3,
) -> List[Dict[str, Any]]:
    """한 그룹의 교정 후보들에 안전장치를 적용하고, 선택지 목록을 만든다."""
    ranked_items = list(group.get("ranked") or [])
    options: List[Dict[str, Any]] = []
    seen_replacements = set()
    for candidate in ranked_items:
        replacement = str(candidate.get("replacement") or "")
        if not replacement or replacement in seen_replacements:
            continue
        seen_replacements.add(replacement)
        options.append(_proposal_option_for_candidate(models, group, candidate, profile, protected))
        if len([item for item in options if item["decision"] != "KEEP"]) >= candidate_limit:
            break

    if not any(item["decision"] == "KEEP" for item in options):
        options.append(
            {
                "span": group["span"],
                "original": group["original"],
                "replacement": group["original"],
                "source": "ORIGINAL",
                "candidateScore": 0.2,
                "optionScore": 0.2,
                "guardrail": {"decision": "PASS", "reasonCodes": ["KEEP_ORIGINAL"], "score": 0.2},
                "decision": "KEEP",
                "applyEdit": False,
            }
        )

    options.sort(
        key=lambda item: (
            item.get("optionScore", 0.0),
            item.get("candidateScore", 0.0),
            item["decision"] != "KEEP",
        ),
        reverse=True,
    )
    return options[: max(2, candidate_limit + 1)]


def _proposal_edit_from_option(option: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """선택된 교정 옵션을 실제 수정 명령으로 바꾼다."""
    if not option.get("applyEdit"):
        return None
    return create_edit(
        stage="RERANKER",
        start=option["span"]["start"],
        end=option["span"]["end"],
        source_text=option["original"],
        replacement=option["replacement"],
        edit_type="OPEN_REPLACE",
        confidence=float(option.get("candidateScore", 0.0)),
        auto_applicable=option.get("decision") == "AUTO_APPLY",
        reason_tag=",".join(option.get("guardrail", {}).get("reasonCodes", [])),
    )
