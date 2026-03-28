# [12단계] 최종 결과 만들기
# 모든 단계의 결과를 모아서, 프론트엔드에 보낼 최종 응답을 만든다.

from typing import Any, Dict, List

from model_loader import ModelLoader


def build_output(
    *,
    models: ModelLoader,
    full_text: str,
    range_art: Dict[str, Any],
    working_clause: str,
    suggestion_clause: str,
    merged_proposal_out: List[Dict[str, Any]],
    decisions: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
    mode: str,
) -> Dict[str, Any]:
    """모든 교정 결과를 모아서 프론트엔드에 보낼 응답을 만든다."""

    clause_range = range_art["clauseRange"]

    full_text_proposals = [
        {
            "text": (
                full_text[: clause_range["start"]]
                + proposal["suggestedClause"]
                + full_text[clause_range["end"] :]
            ),
            "finalText": (
                full_text[: clause_range["start"]]
                + proposal["finalClause"]
                + full_text[clause_range["end"] :]
            ),
            "score": proposal["score"],
            "source": proposal.get("source", "HYBRID_DECODER"),
            "editCount": proposal["editCount"],
            "edits": proposal["edits"],
        }
        for proposal in merged_proposal_out
    ]

    # 12단계 trace 추가
    traces.append({
        "stageName": "12. Final Output",
        "inputText": working_clause,
        "outputArtifacts": {
            "mode": mode,
            "finalClause": working_clause,
            "suggestedClause": suggestion_clause,
            "proposalCount": len(full_text_proposals),
            "autoCount": len([d for d in decisions if d["decision"] == "AUTO_APPLY"]),
            "suggestCount": len([d for d in decisions if d["decision"] == "SUGGEST_ONLY"]),
            "rejectCount": len([d for d in decisions if d["decision"] == "REJECT"]),
        },
        "latencyMs": 0.0,
    })

    final_text = (
        full_text[: clause_range["start"]]
        + working_clause
        + full_text[clause_range["end"] :]
    )
    suggested_text = (
        full_text[: clause_range["start"]]
        + suggestion_clause
        + full_text[clause_range["end"] :]
    )

    return {
        "finalText": final_text,
        "suggestedText": suggested_text,
        "proposals": full_text_proposals,
        "decisions": decisions,
        "traces": traces,
        "modelInfo": {
            "spacingEngine": "kiwipiepy",
            "profileModel": "KoBERT + KoELECTRA",
            "editTaggerModel": (
                f"Fine-tuned KoELECTRA token-classifier ({models.edit_tagger_source})"
                if models.edit_tagger_model is not None
                else "KoBERT-MLM pseudo-likelihood + KoELECTRA + Kiwi"
            ),
            "candidateGeneratorModel": "lemma-aware generator + morpheme normalization + KoBERT-MLM backoff",
            "proposalGeneratorModel": (
                f"Local KoBART seq2seq proposal source ({models.proposal_seq2seq_source})"
                if models.proposal_seq2seq_model is not None
                else "disabled"
            ),
            "candidateVerifierModel": "source-aware verifier + typo similarity + generator prior",
            "rerankerModel": "KoBERT-MLM + KoELECTRA + Kiwi + context verifier",
            "kobert": "skt/kobert-base-v1",
            "koelectra": "monologg/koelectra-base-v3-discriminator",
            "kobertMlm": "monologg/kobert-lm",
        },
    }
