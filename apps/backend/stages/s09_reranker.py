# [9단계] 교정 후보 최종 순위 매기기
# 여러 AI 모델의 판단을 종합해서, 어떤 교정 후보가 가장 자연스러운지
# 최종 순위를 매긴다.

from typing import Any, Dict, List

from model_loader import ModelLoader
from stages.s07_edit_tagger import (
    _is_curated_candidate_source,
    _rank_replacement_candidates,
)


def rerank(
    models: ModelLoader, sentence: str, groups: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """여러 AI 모델의 판단을 종합해서 교정 후보의 최종 순위를 매긴다."""
    if not groups:
        return []
    ranked: List[Dict[str, Any]] = []
    baseline = models.sentence_quality_baseline(sentence)
    for g in groups:
        family_prior = g.get("candidateFamily")
        st, ed = g["span"]["start"], g["span"]["end"]
        model_seed_items = [item for item in g["items"] if item["source"] != "ORIGINAL"]
        if g.get("fastPath") and model_seed_items:
            scored = [
                {
                    **item,
                    "candidateSentence": sentence[:st] + item["replacement"] + sentence[ed:],
                    "rerankScore": round(float(item.get("candidateVerifierScore", item.get("prefilterScore", 0.0))), 4),
                    "finalScore": round(float(item.get("candidateVerifierScore", item.get("prefilterScore", 0.0))), 4),
                }
                for item in model_seed_items
            ]
        elif model_seed_items:
            rescored = _rank_replacement_candidates(
                models,
                sentence,
                g["original"],
                st,
                ed,
                model_seed_items,
                baseline=baseline,
            )
            curated_rescored = [
                item for item in rescored if _is_curated_candidate_source(item.get("source", "MLM_TOPK"))
            ]
            if len(curated_rescored) >= 2:
                semantic_leader = max(
                    curated_rescored,
                    key=lambda item: (
                        item.get("contextVerifierScore", 0.0),
                        item.get("kiwiDelta", 0.0),
                        item.get("koelectraDelta", 0.0),
                    ),
                )
                typo_leader = max(
                    curated_rescored,
                    key=lambda item: (
                        item.get("typoSimilarity", 0.0),
                        item.get("prefilterScore", 0.0),
                        item.get("generatorNorm", 0.0),
                    ),
                )
                if semantic_leader is not typo_leader:
                    context_gap = semantic_leader.get("contextVerifierScore", 0.0) - typo_leader.get(
                        "contextVerifierScore", 0.0
                    )
                    kiwi_gap = semantic_leader.get("kiwiDelta", 0.0) - typo_leader.get("kiwiDelta", 0.0)
                    electra_gap = semantic_leader.get("koelectraDelta", 0.0) - typo_leader.get(
                        "koelectraDelta", 0.0
                    )
                    if context_gap >= 0.015 and (kiwi_gap >= 1.5 or electra_gap >= 0.01):
                        for item in rescored:
                            if item is semantic_leader:
                                item["semanticLeaderBonus"] = 0.14
                                item["finalScore"] = round(
                                    float(min(0.99, item["finalScore"] + 0.14)),
                                    4,
                                )
                            elif item is typo_leader:
                                item["semanticLeaderPenalty"] = 0.12
                                item["finalScore"] = round(
                                    float(max(0.01, item["finalScore"] - 0.12)),
                                    4,
                                )
                        rescored.sort(key=lambda x: x["finalScore"], reverse=True)
            scored = [
                {
                    **item,
                    "rerankScore": item["finalScore"],
                }
                for item in rescored
            ]
        else:
            scored = []

        scored.append(
            {
                "replacement": g["original"],
                "source": "ORIGINAL",
                "generatorScore": 0.2,
                "rerankScore": 0.2,
                "finalScore": 0.2,
            }
        )
        scored.sort(key=lambda x: x["finalScore"], reverse=True)
        ranked.append({**g, "ranked": scored, "best": scored[0] if scored else None})
    return ranked
