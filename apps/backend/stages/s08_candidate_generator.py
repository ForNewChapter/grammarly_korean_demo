# [8단계] 교정 후보 묶기 + 검증
# 7단계에서 '틀렸다'고 판별된 단어들에 대해 교정 후보를 모아서 묶고,
# 각 후보가 정말 말이 되는지 점수를 매겨 검증한다.

from typing import Any, Dict, List

from model_loader import (
    CANDIDATE_PREFILTER_LIMIT,
    CHEAP_VERIFIER_LIMIT,
    FAST_PATH_MARGIN_MIN,
    FAST_PATH_VERIFIER_MIN,
    ModelLoader,
    merge_candidates,
)
from stages.s07_edit_tagger import (
    _apply_family_prior,
    _build_candidate_family_prior,
    _candidate_pos_compatible_with_prior,
    _candidate_slot_compatible_with_prior,
    _is_curated_candidate_source,
    _normalize_generator_score,
    _phrase_span_candidate_group,
    _retokenized_pair_candidate_group,
    generate_contextual_candidates_for_token,
)
from utils.hangul import typo_similarity


def generate_open_candidates(
    models: ModelLoader, sentence: str, tag_labels: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """틀렸다고 판별된 단어들의 교정 후보를 모아서 그룹으로 묶는다."""
    out: List[Dict[str, Any]] = []
    index = 0
    while index < len(tag_labels):
        retokenized_pair_group = _retokenized_pair_candidate_group(models, sentence, tag_labels, index)
        if retokenized_pair_group is not None:
            out.append(retokenized_pair_group["group"])
            index += int(retokenized_pair_group["consumeCount"])
            continue

        phrase_group = _phrase_span_candidate_group(models, sentence, tag_labels, index)
        if phrase_group is not None:
            out.append(phrase_group["group"])
            index += int(phrase_group["consumeCount"])
            continue

        l = tag_labels[index]
        if l.get("effectiveLabel", l["label"]) != "OPEN_REPLACE":
            index += 1
            continue
        detection_evidence = l.get("detectionEvidence") or {}
        best_replacement = detection_evidence.get("bestReplacement")
        if not best_replacement and l.get("modelCandidates"):
            best_replacement = l["modelCandidates"][0].get("replacement")
        family_prior = _build_candidate_family_prior(
            models,
            sentence,
            l["token"],
            l["range"]["start"],
            l["range"]["end"],
            best_replacement,
        )
        seeded_candidates = l.get("modelCandidates") or []
        generated_candidates = generate_contextual_candidates_for_token(
            models,
            sentence,
            l["token"],
            l["range"]["start"],
            l["range"]["end"],
            expensive=False,
        )
        base = merge_candidates(seeded_candidates, generated_candidates)
        base = _apply_family_prior(models, base, family_prior)
        items = [
            *[
                {
                    "replacement": item["replacement"],
                    "source": item.get("source", "MODEL"),
                    "generatorScore": item.get("generatorScore", item.get("finalScore", 0.5)),
                    "generatorNorm": item.get("generatorNorm"),
                    "prefilterScore": item.get("prefilterScore"),
                    "finalScore": item.get("finalScore"),
                    "rerankScore": item.get("finalScore"),
                    "typoSimilarity": item.get("typoSimilarity"),
                    "contextHintScore": item.get("contextHintScore"),
                    "interfaceScore": item.get("interfaceScore"),
                    "familyLemmas": item.get("familyLemmas"),
                    "familyMatch": item.get("familyMatch"),
                }
                for item in base
            ],
            {"replacement": l["token"], "source": "ORIGINAL", "generatorScore": 0.2},
        ]
        out.append(
            {
                "span": l["range"],
                "original": l["token"],
                "taggerConfidence": l["confidence"],
                "detectionEvidence": l.get("detectionEvidence"),
                "candidateFamily": family_prior,
                "items": items,
            }
        )
        index += 1
    return out


def verify_candidate_groups(
    models: ModelLoader, groups: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """각 후보 그룹에서 점수를 매겨 진짜 괜찮은 후보만 남긴다."""
    verified: List[Dict[str, Any]] = []
    for group in groups:
        original = group["original"]
        family_prior = group.get("candidateFamily")
        candidate_items = [item for item in group["items"] if item["source"] != "ORIGINAL"]
        curated_items = [
            item for item in candidate_items if _is_curated_candidate_source(item.get("source", "MLM_TOPK"))
        ]
        family_matched_items = [item for item in candidate_items if item.get("familyMatch")]
        curated_best = max(
            (float(item.get("prefilterScore", 0.0)) for item in curated_items),
            default=0.0,
        )
        scored_items: List[Dict[str, Any]] = []
        for item in candidate_items:
            source = item.get("source", "MLM_TOPK")
            prefilter = float(item.get("prefilterScore", 0.0))
            tsim = float(item.get("typoSimilarity", typo_similarity(original, item["replacement"])))
            generator_norm = float(
                item.get(
                    "generatorNorm",
                    _normalize_generator_score(float(item.get("generatorScore", 0.0))),
                )
            )
            context_hint_score = float(item.get("contextHintScore", 0.0))
            interface_score = float(item.get("interfaceScore", 0.86))
            family_match = bool(item.get("familyMatch"))
            pos_compatible = _candidate_pos_compatible_with_prior(item, family_prior)
            slot_compatible = _candidate_slot_compatible_with_prior(item, family_prior)
            if not pos_compatible:
                continue
            if not slot_compatible and not _is_curated_candidate_source(source):
                continue
            if family_prior and family_matched_items and not family_match and not _is_curated_candidate_source(source):
                if context_hint_score < 0.18 and prefilter < curated_best + 0.04:
                    continue
            if curated_items and not _is_curated_candidate_source(source):
                if context_hint_score < 0.12 and prefilter < curated_best + 0.05:
                    continue
            if tsim < 0.56 and not family_match and context_hint_score < 0.2:
                continue

            verifier_score = (
                0.42 * tsim
                + 0.24 * generator_norm
                + 0.2 * prefilter
                + 0.06 * context_hint_score
                + 0.04 * interface_score
            )
            if family_prior and family_match:
                verifier_score += 0.08
            if _is_curated_candidate_source(source):
                verifier_score += 0.05
            if slot_compatible:
                verifier_score += 0.03

            scored_items.append(
                {
                    **item,
                    "candidateVerifierScore": round(float(verifier_score), 4),
                    "familyMatch": family_match,
                    "posCompatible": pos_compatible,
                    "slotCompatible": slot_compatible,
                }
            )

        scored_items.sort(
            key=lambda item: (
                item["candidateVerifierScore"],
                item.get("prefilterScore", 0.0),
                item.get("typoSimilarity", 0.0),
            ),
            reverse=True,
        )

        if curated_items:
            curated_ranked = [
                item for item in scored_items if _is_curated_candidate_source(item.get("source", "MLM_TOPK"))
            ][:2]
            mlm_ranked = [
                item for item in scored_items if not _is_curated_candidate_source(item.get("source", "MLM_TOPK"))
            ]
            kept = list(curated_ranked)
            if mlm_ranked and (
                not curated_ranked
                or mlm_ranked[0]["candidateVerifierScore"] >= curated_ranked[0]["candidateVerifierScore"] + 0.08
            ):
                kept.append(mlm_ranked[0])
        else:
            kept = scored_items[:CHEAP_VERIFIER_LIMIT]

        kept.sort(
            key=lambda item: (
                item.get("candidateVerifierScore", 0.0),
                item.get("contextHintScore", 0.0),
                item.get("prefilterScore", 0.0),
                item.get("typoSimilarity", 0.0),
            ),
            reverse=True,
        )

        fast_path = None
        if kept:
            leader = kept[0]
            runner_score = kept[1].get("candidateVerifierScore", 0.0) if len(kept) > 1 else 0.0
            single_curated = len(kept) == 1 and _is_curated_candidate_source(leader.get("source", "MLM_TOPK"))
            clear_curated_gap = (
                _is_curated_candidate_source(leader.get("source", "MLM_TOPK"))
                and leader.get("candidateVerifierScore", 0.0) >= 0.76
                and (leader.get("candidateVerifierScore", 0.0) - runner_score) >= 0.08
                and (
                    leader.get("contextHintScore", 0.0) >= 0.05
                    or leader.get("prefilterScore", 0.0) >= 0.78
                )
            )
            if (
                (
                    single_curated and leader.get("candidateVerifierScore", 0.0) >= 0.72
                )
                or (
                    _is_curated_candidate_source(leader.get("source", "MLM_TOPK"))
                    and leader.get("candidateVerifierScore", 0.0) >= FAST_PATH_VERIFIER_MIN
                    and (leader.get("candidateVerifierScore", 0.0) - runner_score) >= FAST_PATH_MARGIN_MIN
                    and (
                        leader.get("contextHintScore", 0.0) >= 0.08
                        or leader.get("prefilterScore", 0.0) >= 0.9
                    )
                )
                or clear_curated_gap
            ):
                fast_path = {
                    "replacement": leader["replacement"],
                    "score": leader["candidateVerifierScore"],
                    "source": leader.get("source", "MODEL"),
                }

        verified.append(
            {
                **group,
                "items": [*kept, {"replacement": original, "source": "ORIGINAL", "generatorScore": 0.2}],
                "fastPath": fast_path,
                "candidateFamily": family_prior,
                "candidateVerifier": {
                    "inputCount": len(candidate_items),
                    "keptCount": len(kept),
                    "curatedCount": len(curated_items),
                    "familyMatchedCount": len([item for item in kept if item.get("familyMatch")]),
                    "keptSources": [item.get("source", "MODEL") for item in kept],
                    "fastPath": fast_path is not None,
                },
            }
        )
    return verified
