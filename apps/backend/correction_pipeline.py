# 맞춤법 교정 전체 흐름 관리
# 1단계부터 12단계까지 순서대로 호출하여 입력 텍스트를 교정한다.
# 각 단계의 실제 로직은 stages/ 폴더에 있고, 여기서는 순서만 관리한다.

import time
from typing import Any, Dict, List

from model_loader import ModelLoader
from utils.edit_ops import apply_edits

from stages.s01_input_range import extract_input_range
from stages.s02_protected_spans import detect_protected_spans, apply_mask
from stages.s03_profile_classifier import classify_profile
from stages.s04_rule_corrector import run_rule_corrector
from stages.s05_spacing import (
    spacing_candidates_from_kiwi,
    classify_spacing_boundaries,
    spacing_to_edits,
)
from stages.s06_morpheme_normalizer import (
    run_pre_spacing_phrase_normalizer,
    run_kiwi_pre_normalizer,
)
from stages.s07_edit_tagger import (
    run_edit_tagger,
    route_tagger_labels,
    apply_high_precision_lock_guard,
    apply_meta_comparison_guard,
    _high_precision_lock_spans,
    _phrase_target_lock_spans,
    _meta_comparison_lock_spans,
)
from stages.s08_candidate_generator import (
    generate_open_candidates,
    verify_candidate_groups,
)
from stages.s09_reranker import rerank
from stages.s10_guardrail import (
    policy,
    _build_group_proposal_options,
    _proposal_edit_from_option,
)
from stages.s11_proposal_decoder import (
    _decode_sentence_proposals,
    _seq2seq_sentence_proposals,
    _merge_sentence_proposals,
)
from stages.s12_output_builder import build_output


class CorrectionPipeline:
    """12단계 맞춤법 교정 파이프라인."""

    def __init__(self, models: ModelLoader) -> None:
        self.models = models

    def run(self, full_text: str, cursor: int, mode: str) -> Dict[str, Any]:
        models = self.models
        traces: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []

        def stage(name: str, input_text: str, fn) -> Any:
            t0 = time.perf_counter()
            out = fn()
            latency = round((time.perf_counter() - t0) * 1000, 2)
            traces.append({
                "stageName": name,
                "inputText": input_text,
                "outputArtifacts": out,
                "latencyMs": latency,
            })
            return out

        # ── 1. 입력 범위 추출 ──
        range_art = stage(
            "1. Input Range Extractor",
            full_text,
            lambda: (
                {
                    "clauseText": full_text,
                    "clauseRange": {"start": 0, "end": len(full_text)},
                    "strategy": "full_text_inspect",
                }
                if mode == "inspect"
                else extract_input_range(full_text, cursor)
            ),
        )
        working_clause = range_art["clauseText"]

        # ── 2. 보호 영역 감지 ──
        protected_art = stage(
            "2. Protected Span Detector",
            working_clause,
            lambda: {
                "protected": detect_protected_spans(working_clause),
                "maskedText": apply_mask(working_clause, detect_protected_spans(working_clause)),
            },
        )
        protected = protected_art["protected"]

        # ── 3. 프로필 분류 ──
        profile_art = stage(
            "3. Profile Classifier (KoBERT+KoELECTRA+Rule)",
            protected_art["maskedText"],
            lambda: classify_profile(models, protected_art["maskedText"]),
        )
        profile = profile_art["profile"]

        # ── 4. 규칙 교정 ──
        rule_edits = stage("4. Rule Corrector", working_clause, lambda: run_rule_corrector(models, working_clause, protected))
        if rule_edits:
            working_clause = apply_edits(working_clause, rule_edits)
            for e in rule_edits:
                decision = policy(e["editType"], "AUTO_APPLY", e["confidence"])
                decisions.append({**e, "decision": decision})

        # ── 4-1. 구문 전처리 ──
        compact_phrase_edits = stage(
            "4-1. Phrase Pre-normalizer",
            working_clause,
            lambda: run_pre_spacing_phrase_normalizer(models, working_clause, detect_protected_spans(working_clause)),
        )
        if compact_phrase_edits:
            working_clause = apply_edits(working_clause, compact_phrase_edits)
            for e in compact_phrase_edits:
                decision = policy(e["editType"], "AUTO_APPLY", e["confidence"])
                decisions.append({**e, "decision": decision})

        # ── 5-6. 띄어쓰기 ──
        spacing_cand_art = stage(
            "5. Spacing Candidate Generator (Kiwi)",
            working_clause,
            lambda: spacing_candidates_from_kiwi(models, working_clause, detect_protected_spans(working_clause)),
        )
        spacing_boundaries = stage(
            "6. Spacing Boundary Classifier",
            working_clause,
            lambda: classify_spacing_boundaries(spacing_cand_art["candidates"], profile),
        )
        spacing_edits = stage("6-2. Spacing Edit Converter", working_clause, lambda: spacing_to_edits(spacing_boundaries))
        if spacing_edits:
            working_clause = apply_edits(working_clause, spacing_edits)
            for e in spacing_edits:
                decision = policy(e["editType"], "AUTO_APPLY", e["confidence"])
                decisions.append({**e, "decision": decision})

        # ── 6-3. 형태소 정규화 ──
        kiwi_pre_normalizer_edits = stage(
            "6-3. Kiwi Pre-normalizer",
            working_clause,
            lambda: run_kiwi_pre_normalizer(models, working_clause, detect_protected_spans(working_clause)),
        )
        if kiwi_pre_normalizer_edits:
            working_clause = apply_edits(working_clause, kiwi_pre_normalizer_edits)
            for e in kiwi_pre_normalizer_edits:
                decision = policy(e["editType"], "AUTO_APPLY", e["confidence"])
                decisions.append({**e, "decision": decision})

        # ── 7. Edit Tagger ──
        tagger_labels = stage(
            "7. Edit Tagger (KoBERT-MLM + KoELECTRA-assisted)",
            working_clause,
            lambda: run_edit_tagger(models, working_clause, detect_protected_spans(working_clause)),
        )
        routed_tagger_labels = stage(
            "7-2. Edit Tagger Routing",
            working_clause,
            lambda: route_tagger_labels(tagger_labels),
        )
        locked_spans = [
            *_high_precision_lock_spans(decisions),
            *_phrase_target_lock_spans(models, working_clause, detect_protected_spans(working_clause)),
        ]
        routed_tagger_labels = stage(
            "7-3. High-precision Lock Guard",
            working_clause,
            lambda: apply_high_precision_lock_guard(routed_tagger_labels, locked_spans),
        )
        meta_locked_spans = _meta_comparison_lock_spans(
            working_clause,
            detect_protected_spans(working_clause),
        )
        routed_tagger_labels = stage(
            "7-4. Meta Comparison Guard",
            working_clause,
            lambda: apply_meta_comparison_guard(routed_tagger_labels, meta_locked_spans),
        )

        # ── 8. 후보 생성 + 검증 ──
        open_candidates = stage(
            "8. Open Candidate Generation",
            working_clause,
            lambda: generate_open_candidates(models, working_clause, routed_tagger_labels),
        )
        verified_candidates = stage(
            "8-2. Candidate Verifier",
            working_clause,
            lambda: verify_candidate_groups(models, open_candidates),
        )

        # ── 9. 재순위 ──
        reranked = stage(
            "9. Reranker (KoBERT-MLM + KoELECTRA)",
            working_clause,
            lambda: rerank(models, working_clause, verified_candidates),
        )

        # ── 10. 가드레일 ──
        proposal_base_clause = working_clause
        guardrail_out = stage(
            "10. Guardrail",
            working_clause,
            lambda: [
                {
                    **group,
                    "proposalOptions": _build_group_proposal_options(
                        models, group, profile, detect_protected_spans(working_clause),
                    ),
                }
                for group in reranked
            ],
        )

        # ── 11. 디코딩 + Seq2Seq ──
        proposal_out = stage(
            "11. Policy Engine + Proposal Decoder",
            working_clause,
            lambda: _decode_sentence_proposals(models, working_clause, guardrail_out),
        )
        seq2seq_proposal_out = stage(
            "11-1. Seq2Seq Proposal Source",
            working_clause,
            lambda: _seq2seq_sentence_proposals(
                models, working_clause, detect_protected_spans(working_clause), routed_tagger_labels,
            ),
        )
        merged_proposal_out = _merge_sentence_proposals(proposal_out, seq2seq_proposal_out)

        # ── 12. 최종 출력 조립 ──
        top_proposal = proposal_out[0] if proposal_out else {
            "finalClause": proposal_base_clause,
            "suggestedClause": proposal_base_clause,
            "selectedOptions": [],
            "edits": [],
        }
        auto_edits: List[Dict[str, Any]] = []
        suggestion_edits: List[Dict[str, Any]] = []
        for option in top_proposal.get("selectedOptions") or []:
            edit = _proposal_edit_from_option(option)
            if edit is None:
                continue
            decisions.append({**edit, "decision": option["decision"]})
            suggestion_edits.append(edit)
            if option["decision"] == "AUTO_APPLY":
                auto_edits.append(edit)

        working_clause = apply_edits(proposal_base_clause, auto_edits)
        suggestion_clause = apply_edits(proposal_base_clause, suggestion_edits)

        return build_output(
            models=models,
            full_text=full_text,
            range_art=range_art,
            working_clause=working_clause,
            suggestion_clause=suggestion_clause,
            merged_proposal_out=merged_proposal_out,
            decisions=decisions,
            traces=traces,
            mode=mode,
        )
