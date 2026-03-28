# 7단계: 토큰별로 편집 라벨(KEEP/OPEN_REPLACE/JOSA_FIX 등)을 예측하고, 라우팅과 가드를 적용한다.

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from model_loader import (
    CANDIDATE_PREFILTER_LIMIT,
    CURATED_CANDIDATE_SOURCES,
    META_COMPARE_TOKEN_PATTERN,
    MLM_BACKOFF_MIN_CANDIDATES,
    MLM_BACKOFF_PREFILTER_MIN,
    MLM_SURFACE_TOP_K,
    MORPHEME_CONFUSION_RULES,
    ROUTING_PROMOTION_LABELS,
    ROUTING_PROMOTION_MIN_SCORE,
    STEM_CONFUSION_RULES,
    TYPO_SIMILARITY_MIN,
    CHEAP_VERIFIER_LIMIT,
    FAST_PATH_MARGIN_MIN,
    FAST_PATH_VERIFIER_MIN,
    ModelLoader,
    merge_candidates,
)
from utils.edit_ops import (
    is_range_protected,
    normalize_phrase_surface,
    range_overlaps,
    tokenize_with_ranges,
)
from utils.hangul import (
    char_edit_distance,
    hangul_ratio,
    has_jongseong_l,
    normalize_context_token,
    typo_similarity,
)
from utils.morpheme_utils import (
    coarse_predicate_pos,
    infer_irregular_class_from_lemma,
    stem_from_lemma,
    variant_suffix_slots,
)


# ── 형태소 번들 / 용언 상태 ───────────────────────────────────


def _token_morph_bundles(
    models: ModelLoader, sentence: str, start: int, end: int, top_n: int = 3
) -> List[Dict[str, Any]]:
    analyses = models.kiwi_analyze_cached(sentence, top_n=top_n)
    if not analyses:
        return []
    bundles: List[Dict[str, Any]] = []
    seen = set()
    for rank, (morphs, score) in enumerate(analyses):
        overlapped = [m for m in morphs if not (m.end <= start or m.start >= end)]
        if not overlapped:
            continue
        predicate_index = None
        for idx, morph in enumerate(overlapped):
            if morph.tag.startswith(("VV", "VA", "VX")):
                predicate_index = idx
                break
        if predicate_index is None:
            continue
        predicate = overlapped[predicate_index]
        suffix_morphs = overlapped[predicate_index + 1 :]
        suffix_signature = tuple(m.tag for m in suffix_morphs)
        bundle_key = (predicate.lemma, predicate.tag.split("-")[0], suffix_signature)
        if bundle_key in seen:
            continue
        seen.add(bundle_key)
        bundles.append(
            {
                "predicate": predicate,
                "suffixMorphs": suffix_morphs,
                "morphs": overlapped,
                "suffixSlot": "+".join(suffix_signature) if suffix_signature else None,
                "analysisRank": rank,
                "analysisScore": float(score),
            }
        )
    return bundles


def _token_morph_bundle(
    models: ModelLoader, sentence: str, start: int, end: int
) -> Optional[Dict[str, Any]]:
    bundles = _token_morph_bundles(models, sentence, start, end, top_n=1)
    return bundles[0] if bundles else None


def _canonical_predicate_states(
    models: ModelLoader,
    sentence: str,
    start: int,
    end: int,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    bundles = _token_morph_bundles(models, sentence, start, end, top_n=top_n)
    states: List[Dict[str, Any]] = []
    seen = set()
    for bundle in bundles:
        predicate = bundle["predicate"]
        lemma = str(predicate.lemma or "").strip()
        if not lemma:
            continue
        c_pos = coarse_predicate_pos(predicate.tag)
        if c_pos not in {"VV", "VA", "VX"}:
            continue
        state = {
            "surface": sentence[start:end],
            "lemma": lemma,
            "pos": c_pos,
            "tag": predicate.tag,
            "slot": bundle.get("suffixSlot"),
            "suffixMorphs": bundle["suffixMorphs"],
            "analysisRank": int(bundle.get("analysisRank", 0)),
            "analysisScore": float(bundle.get("analysisScore", 0.0)),
            "irregularClass": infer_irregular_class_from_lemma(lemma),
            "stem": stem_from_lemma(lemma, predicate.form),
        }
        key = (
            state["lemma"],
            state["pos"],
            state["slot"],
            state["irregularClass"],
        )
        if key in seen:
            continue
        seen.add(key)
        states.append(state)
    return states


def _predicate_lemma_exists(models: ModelLoader, lemma: str, coarse_pos_val: str) -> bool:
    if not lemma:
        return False
    return lemma in (models.predicate_lemma_index.get(coarse_pos_val) or [])


# ── 재활용 / 후보 생성 ───────────────────────────────────────


def _infer_lemma_tag(models: ModelLoader, lemma: str, fallback_tag: str) -> str:
    tag_base = fallback_tag.split("-")[0]
    try:
        analyses = models.kiwi_analyze_cached(lemma, top_n=2)
    except Exception:
        return tag_base
    for tokens, _ in analyses:
        for token in tokens:
            if token.lemma == lemma and token.tag.startswith(("VV", "VA")):
                return token.tag
    return tag_base


def _predicate_lemmas_from_text(models: ModelLoader, text: str, top_n: int = 2) -> List[str]:
    try:
        analyses = models.kiwi_analyze_cached(text, top_n=top_n)
    except Exception:
        return []
    lemmas: List[str] = []
    for tokens, _ in analyses:
        for token in tokens:
            if token.tag.startswith(("VV", "VA")) and token.lemma not in lemmas:
                lemmas.append(token.lemma)
    return lemmas


def _reinflect_candidate(
    models: ModelLoader,
    state: Dict[str, Any],
    target_lemma: str,
    *,
    source: str,
    generator_score: float,
    token: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    target_tag = _infer_lemma_tag(models, target_lemma, state["tag"])
    target_stem = stem_from_lemma(target_lemma, target_lemma)
    morph_seq = [(target_stem, target_tag), *[(m.form, m.tag) for m in state["suffixMorphs"]]]
    try:
        restored = models.kiwi.join(morph_seq, lm_search=True)
    except TypeError:
        try:
            restored = models.kiwi.join(morph_seq)
        except Exception:
            return None
    except Exception:
        return None
    if not restored or restored == token:
        return None
    payload = {
        "replacement": restored,
        "source": source,
        "generatorScore": max(0.0, float(generator_score)),
        "pos": target_tag,
        "suffixSlot": state.get("slot"),
        "familyLemmas": [state["lemma"], target_lemma],
        "irregularClass": infer_irregular_class_from_lemma(target_lemma) or state.get("irregularClass"),
    }
    if extra:
        payload.update({key: value for key, value in extra.items() if value not in (None, "", [])})
    return payload


def _productive_inflection_rule_candidates(
    models: ModelLoader,
    state: Dict[str, Any],
    token: str,
) -> List[Dict[str, Any]]:
    rules = models.inflection_recovery_rules.get("rules") or []
    if not rules:
        return []
    current_lemma = state["lemma"]
    current_stem = state["stem"]
    c_pos = state["pos"]
    slot = state.get("slot")
    dedup: Dict[str, Dict[str, Any]] = {}
    for rule in rules:
        rule_pos = coarse_predicate_pos(str(rule.get("pos") or ""))
        if rule_pos and rule_pos != c_pos:
            continue
        source_suffix = str(rule.get("sourceSuffix") or "")
        target_suffix = str(rule.get("targetSuffix") or "")
        if not source_suffix or not target_suffix:
            continue
        if not current_stem.endswith(source_suffix):
            continue
        rule_slots = rule.get("suffixSlots") or []
        if slot and rule_slots and not any(
            slot == candidate_slot
            or slot.startswith(f"{candidate_slot}+")
            or str(candidate_slot).startswith(f"{slot}+")
            for candidate_slot in rule_slots
        ):
            continue
        target_stem = f"{current_stem[:-len(source_suffix)]}{target_suffix}"
        target_lemma = f"{target_stem}다"
        if target_lemma == current_lemma:
            continue
        if not _predicate_lemma_exists(models, target_lemma, c_pos):
            continue
        rule_score = float(rule.get("generatorScore", 0.0))
        analysis_penalty = min(0.08, 0.02 * int(state.get("analysisRank", 0)))
        payload = _reinflect_candidate(
            models,
            state,
            target_lemma,
            source="INFLECTION_RULE",
            generator_score=max(0.0, rule_score - analysis_penalty),
            token=token,
            extra={
                "contextHints": rule.get("contextHints"),
                "interfaceStats": rule.get("interfaceStats"),
                "suffixSlots": rule_slots,
                "sourceStats": rule.get("sourceStats"),
                "surfaceExamples": None,
                "irregularClass": (rule.get("irregularClasses") or [None])[0],
            },
        )
        if payload is None:
            continue
        existing = dedup.get(payload["replacement"])
        if existing is None or float(payload["generatorScore"]) > float(existing.get("generatorScore", 0.0)):
            dedup[payload["replacement"]] = payload
    return list(dedup.values())


def _jamo_weighted_fallback_candidates(
    models: ModelLoader,
    state: Dict[str, Any],
    token: str,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    current_lemma = state["lemma"]
    current_stem = state["stem"]
    c_pos = state["pos"]
    if not current_stem or len(current_stem) < 2:
        return []
    scored: List[Tuple[float, str]] = []
    for lemma in models.predicate_lemma_index.get(c_pos, []):
        if lemma == current_lemma:
            continue
        candidate_stem = stem_from_lemma(lemma)
        if abs(len(candidate_stem) - len(current_stem)) > 1:
            continue
        if current_stem[0] != candidate_stem[0]:
            continue
        similarity = typo_similarity(current_stem, candidate_stem)
        if similarity < 0.72:
            continue
        if char_edit_distance(current_stem, candidate_stem) > 2:
            continue
        scored.append((similarity, lemma))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    dedup: Dict[str, Dict[str, Any]] = {}
    for similarity, lemma in scored[:limit]:
        payload = _reinflect_candidate(
            models,
            state,
            lemma,
            source="JAMO_FALLBACK",
            generator_score=0.58 + ((similarity - 0.72) * 0.4),
            token=token,
            extra={
                "contextHints": [],
                "interfaceStats": {"generic": 1},
            },
        )
        if payload is None:
            continue
        payload["generatorScore"] = round(float(payload["generatorScore"]), 4)
        existing = dedup.get(payload["replacement"])
        if existing is None or float(payload["generatorScore"]) > float(existing.get("generatorScore", 0.0)):
            dedup[payload["replacement"]] = payload
    return list(dedup.values())


# ── Family Prior ──────────────────────────────────────────────


def _build_candidate_family_prior(
    models: ModelLoader,
    sentence: str,
    original: str,
    start: int,
    end: int,
    best_replacement: Optional[str],
) -> Optional[Dict[str, Any]]:
    if not best_replacement or best_replacement == original:
        return None

    lemmas: List[str] = []
    slot_hints: List[str] = []
    pos_hints: List[str] = []
    states = _canonical_predicate_states(models, sentence, start, end, top_n=5)
    for state in states:
        original_lemma = str(state.get("lemma") or "").strip()
        if original_lemma and original_lemma not in lemmas:
            lemmas.append(original_lemma)
        slot_hint = str(state.get("slot") or "").strip()
        if slot_hint and slot_hint not in slot_hints:
            slot_hints.append(slot_hint)
        pos_hint = str(state.get("pos") or "").strip()
        if pos_hint and pos_hint not in pos_hints:
            pos_hints.append(pos_hint)

    for lemma in _predicate_lemmas_from_text(models, best_replacement):
        if lemma not in lemmas:
            lemmas.append(lemma)

    expanded_lemmas = list(lemmas)
    for lemma in list(lemmas):
        for variant in models.lemma_confusion_map.get(lemma, [])[:8]:
            replacement = str(variant.get("replacement") or "").strip()
            if replacement and replacement not in expanded_lemmas:
                expanded_lemmas.append(replacement)

    surfaces = [item for item in [original, best_replacement] if item]
    if not expanded_lemmas and len(surfaces) < 2:
        return None

    return {
        "bestReplacement": best_replacement,
        "lemmas": expanded_lemmas,
        "surfaces": surfaces,
        "slotHints": slot_hints,
        "posHints": pos_hints,
    }


def _candidate_family_lemmas(models: ModelLoader, replacement: str) -> List[str]:
    return _predicate_lemmas_from_text(models, replacement, top_n=3)


def _suffix_slot_compatible(bundle_slot: Optional[str], variant: Dict[str, Any]) -> bool:
    v_slots = variant_suffix_slots(variant)
    if not bundle_slot or not v_slots:
        return True
    return any(
        bundle_slot == slot
        or bundle_slot.startswith(f"{slot}+")
        or slot.startswith(f"{bundle_slot}+")
        for slot in v_slots
    )


def _apply_family_prior(
    models: ModelLoader,
    candidates: List[Dict[str, Any]],
    family_prior: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not family_prior:
        return candidates

    allowed_lemmas = set(family_prior.get("lemmas") or [])
    best_replacement = family_prior.get("bestReplacement")
    family_hits: List[Dict[str, Any]] = []
    outsiders: List[Dict[str, Any]] = []

    for item in candidates:
        candidate_lemmas = item.get("familyLemmas")
        if candidate_lemmas is None:
            candidate_lemmas = _candidate_family_lemmas(models, item["replacement"])
        family_match = bool(allowed_lemmas.intersection(candidate_lemmas)) or item["replacement"] == best_replacement
        annotated = {
            **item,
            "familyLemmas": candidate_lemmas,
            "familyMatch": family_match,
        }
        if family_match:
            family_hits.append(annotated)
        else:
            outsiders.append(annotated)

    if not family_hits:
        return [*family_hits, *outsiders]

    family_hits.sort(
        key=lambda item: (
            item.get("prefilterScore", 0.0),
            item.get("typoSimilarity", 0.0),
            item.get("generatorNorm", 0.0),
        ),
        reverse=True,
    )
    outsiders.sort(
        key=lambda item: (
            item.get("prefilterScore", 0.0),
            item.get("typoSimilarity", 0.0),
            item.get("generatorNorm", 0.0),
        ),
        reverse=True,
    )

    kept_outsiders: List[Dict[str, Any]] = []
    if outsiders:
        leader_score = family_hits[0].get("prefilterScore", 0.0)
        outsider = outsiders[0]
        if outsider.get("prefilterScore", 0.0) >= leader_score + 0.08:
            kept_outsiders.append(outsider)

    return [*family_hits, *kept_outsiders]


def _candidate_pos_compatible_with_prior(item: Dict[str, Any], family_prior: Optional[Dict[str, Any]]) -> bool:
    if not family_prior:
        return True
    pos_hints = [str(value) for value in (family_prior.get("posHints") or []) if value]
    if not pos_hints:
        return True
    item_pos = str(item.get("pos") or "")
    if not item_pos:
        return True
    coarse_item_pos = item_pos.split("-", 1)[0]
    return coarse_item_pos in pos_hints


def _candidate_slot_compatible_with_prior(item: Dict[str, Any], family_prior: Optional[Dict[str, Any]]) -> bool:
    if not family_prior:
        return True
    slot_hints = [str(value) for value in (family_prior.get("slotHints") or []) if value]
    if not slot_hints:
        return True
    candidate_slots = variant_suffix_slots(item)
    if not candidate_slots:
        return True
    for bundle_slot in slot_hints:
        for slot in candidate_slots:
            if (
                bundle_slot == slot
                or bundle_slot.startswith(f"{slot}+")
                or slot.startswith(f"{bundle_slot}+")
            ):
                return True
    return False


# ── Join Candidates ───────────────────────────────────────────


def _build_join_candidates(
    models: ModelLoader,
    sentence: str,
    token: str,
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    states = _canonical_predicate_states(models, sentence, start, end, top_n=5)
    if not states:
        return []
    dedup: Dict[str, Dict[str, Any]] = {}

    def keep_best(candidate: Dict[str, Any]) -> None:
        replacement = candidate["replacement"]
        existing = dedup.get(replacement)
        if existing is None or float(candidate.get("generatorScore", 0.0)) > float(existing.get("generatorScore", 0.0)):
            dedup[replacement] = candidate

    bundles = _token_morph_bundles(models, sentence, start, end, top_n=5)
    bundle_by_rank = {int(bundle.get("analysisRank", 0)): bundle for bundle in bundles}

    for state in states:
        predicate_lemma = state["lemma"]
        bundle_slot = state.get("slot")
        bundle_penalty = min(0.08, 0.03 * int(state.get("analysisRank", 0)))
        bundle = bundle_by_rank.get(int(state.get("analysisRank", 0)))

        lemma_variants = models.lemma_confusion_map.get(predicate_lemma, [])
        for variant in lemma_variants:
            if not _suffix_slot_compatible(bundle_slot, variant):
                continue
            target_lemma = variant["replacement"]
            payload = _reinflect_candidate(
                models,
                state,
                target_lemma,
                source=variant.get("source", "LEMMA_CONFUSION"),
                generator_score=max(0.0, float(variant["generatorScore"]) - bundle_penalty),
                token=token,
                extra={
                    "interfaceType": variant.get("interfaceType"),
                    "interfaceStats": variant.get("interfaceStats"),
                    "contextHints": variant.get("contextHints"),
                    "pos": variant.get("pos"),
                    "suffixSlot": variant.get("suffixSlot") or variant.get("suffixSlots"),
                    "surfaceExamples": variant.get("surfaceExamples"),
                },
            )
            if payload is not None:
                keep_best(payload)

            for example in variant.get("surfaceExamples") or []:
                example_from = str(example.get("from") or "").strip()
                example_to = str(example.get("to") or "").strip()
                if not example_from or not example_to or example_to == token:
                    continue
                example_similarity = typo_similarity(token, example_from)
                if example_similarity < 0.46:
                    continue
                example_payload = {
                    "replacement": example_to,
                    "source": "FAMILY_SURFACE_EXAMPLE",
                    "generatorScore": max(
                        0.0,
                        min(0.985, float(variant["generatorScore"]) - bundle_penalty + min(0.08, 0.02 * int(example.get("count", 1))) + max(0.0, example_similarity - 0.5) * 0.25),
                    ),
                    "interfaceType": variant.get("interfaceType"),
                    "interfaceStats": variant.get("interfaceStats"),
                    "contextHints": variant.get("contextHints"),
                    "pos": variant.get("pos"),
                    "suffixSlot": variant.get("suffixSlot") or variant.get("suffixSlots"),
                    "familyLemmas": [predicate_lemma, target_lemma],
                }
                keep_best(example_payload)

        for inflection_candidate in _productive_inflection_rule_candidates(models, state, token):
            keep_best(inflection_candidate)

        for fallback_candidate in _jamo_weighted_fallback_candidates(models, state, token):
            keep_best(fallback_candidate)

        if not bundle:
            continue

        predicate = bundle["predicate"]
        predicate_stem = stem_from_lemma(state["lemma"], predicate.form)
        predicate_tag = state["tag"]
        for idx, morph in enumerate(bundle["morphs"]):
            normalized = MORPHEME_CONFUSION_RULES.get((morph.raw_form, morph.tag))
            if not normalized:
                continue
            for variant in normalized:
                morph_seq: List[Tuple[str, str]] = []
                for inner_idx, current in enumerate(bundle["morphs"]):
                    if current is predicate:
                        morph_seq.append((predicate_stem, predicate_tag))
                        continue
                    if inner_idx == idx:
                        morph_seq.append((variant["replacement"], current.tag))
                    else:
                        morph_seq.append((current.form, current.tag))
                try:
                    restored = models.kiwi.join(morph_seq, lm_search=True)
                except TypeError:
                    try:
                        restored = models.kiwi.join(morph_seq)
                    except Exception:
                        continue
                except Exception:
                    continue
                if not restored or restored == token:
                    continue
                payload = {
                    "replacement": restored,
                    "source": variant["source"],
                    "generatorScore": max(0.0, float(variant["generatorScore"]) - bundle_penalty),
                    "familyLemmas": [state["lemma"]],
                }
                keep_best(payload)

    return list(dedup.values())


# ── Surface Candidate Helpers ─────────────────────────────────


def _single_token_phrase_candidates(models: ModelLoader, token: str) -> List[Dict[str, Any]]:
    candidates = models.phrase_memory_index.get(token) or []
    out: List[Dict[str, Any]] = []
    for item in candidates:
        source = str(item.get("from") or "").strip()
        if not source:
            continue
        if len(tokenize_with_ranges(source)) != 1:
            continue
        if normalize_phrase_surface(source) != normalize_phrase_surface(token):
            continue
        out.append(item)
    return out


def _phrase_candidate_payload(
    item: Dict[str, Any],
    *,
    source: str,
    generator_score: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    replacement = str(item.get("to") or "").strip()
    if not replacement:
        return None
    payload: Dict[str, Any] = {
        "replacement": replacement,
        "source": source,
        "generatorScore": float(item.get("confidence", 0.96))
        if generator_score is None
        else float(generator_score),
    }
    if item.get("contextHints"):
        payload["contextHints"] = item.get("contextHints")
    if item.get("interfaceStats"):
        payload["interfaceStats"] = item.get("interfaceStats")
    if item.get("sourceStats"):
        payload["sourceStats"] = item.get("sourceStats")
    return payload


def _surface_jamo_fallback_candidates(
    models: ModelLoader,
    token: str,
    limit: int = 4,
) -> List[Dict[str, Any]]:
    if hangul_ratio(token) < 0.7 or len(token) < 2:
        return []

    bucket = models.surface_fallback_index.get(token[0]) or {}
    scored: List[Tuple[float, str, List[Dict[str, Any]]]] = []
    for source_surface, candidates in bucket.items():
        if source_surface == token:
            continue
        if abs(len(source_surface) - len(token)) > 1:
            continue
        similarity = typo_similarity(token, source_surface)
        if similarity < 0.74:
            continue
        if char_edit_distance(token, source_surface) > 2:
            continue
        scored.append((similarity, source_surface, candidates))

    scored.sort(key=lambda item: (item[0], -abs(len(item[1]) - len(token)), item[1]), reverse=True)
    dedup: Dict[str, Dict[str, Any]] = {}
    for similarity, source_surface, candidates in scored[:limit]:
        for item in candidates:
            replacement = str(item.get("replacement") or "").strip()
            if not replacement or replacement == token:
                continue
            generator_score = min(
                0.88,
                max(
                    0.58,
                    (0.45 * float(item.get("generatorScore", 0.0))) + (0.45 * similarity) + 0.08,
                ),
            )
            payload = {
                "replacement": replacement,
                "source": "SURFACE_JAMO_FALLBACK",
                "generatorScore": round(float(generator_score), 4),
                "approxSource": source_surface,
                "surfaceSimilarity": round(float(similarity), 4),
            }
            if item.get("contextHints"):
                payload["contextHints"] = item.get("contextHints")
            if item.get("interfaceStats"):
                payload["interfaceStats"] = item.get("interfaceStats")
            existing = dedup.get(replacement)
            if existing is None or float(payload["generatorScore"]) > float(existing.get("generatorScore", 0.0)):
                dedup[replacement] = payload
    return list(dedup.values())


def _common_surface_pattern_candidates(models: ModelLoader, token: str) -> List[Dict[str, Any]]:
    if hangul_ratio(token) < 0.7 or len(token) < 2:
        return []

    dedup: Dict[str, Dict[str, Any]] = {}

    def register(replacement: str, score: float, source: str = "COMMON_SURFACE_PATTERN") -> None:
        if not replacement or replacement == token:
            return
        existing = dedup.get(replacement)
        payload = {
            "replacement": replacement,
            "source": source,
            "generatorScore": round(float(score), 4),
        }
        if existing is None or float(payload["generatorScore"]) > float(existing.get("generatorScore", 0.0)):
            dedup[replacement] = payload

    if token.endswith("께요") and len(token) > 2:
        stem_val = token[:-2]
        if stem_val and (stem_val[-1] == "을" or has_jongseong_l(stem_val[-1])):
            register(f"{stem_val}게요", 0.96)

    if token.endswith("께") and len(token) > 1:
        stem_val = token[:-1]
        if stem_val and (stem_val[-1] == "을" or has_jongseong_l(stem_val[-1])):
            register(f"{stem_val}게", 0.95)

    if token.startswith("않") and len(token) > 2:
        suffix = token[1:]
        if suffix.startswith(("하", "해", "되", "돼", "됐", "계")) or _looks_predicate_like(models, suffix):
            register(f"안 {suffix}", 0.95, source="COMMON_PHRASE_PATTERN")

    if token.startswith("안됐") and len(token) > 2:
        register(f"안 {token[1:]}", 0.96, source="COMMON_PHRASE_PATTERN")

    return list(dedup.values())


def _looks_predicate_like(models: ModelLoader, text: str) -> bool:
    if hangul_ratio(text) < 0.7:
        return False
    analyses = models.kiwi_analyze_cached(text, top_n=2)
    for morphs, _score in analyses:
        for morph in morphs:
            if morph.tag.startswith(("VV", "VA", "VX", "XSV", "XSA")):
                return True
    return False


def _is_valid_surface_candidate(original: str, candidate: str) -> bool:
    if not candidate or candidate == original:
        return False
    if " " in candidate or "\n" in candidate or "\t" in candidate:
        return False
    if candidate.startswith("[") or candidate.startswith("#"):
        return False
    if hangul_ratio(candidate) < 0.5:
        return False
    if abs(len(candidate) - len(original)) > 1:
        return False
    max_edits = 1 if len(original) <= 4 else 2
    if char_edit_distance(original, candidate) > max_edits:
        return False
    return True


def _mlm_surface_candidates(
    models: ModelLoader,
    sentence: str,
    token: str,
    start: int,
    end: int,
    top_k: int = MLM_SURFACE_TOP_K,
) -> List[Dict[str, Any]]:
    original_pieces = models.tokenizer.tokenize(token)
    if not original_pieces or len(original_pieces) > 6:
        return []

    masked_text = sentence[:start] + " " + " ".join([models.tokenizer.mask_token] * len(original_pieces)) + " " + sentence[end:]
    inputs = models.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=96)
    mask_positions = (
        (inputs["input_ids"][0] == models.tokenizer.mask_token_id)
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    if len(mask_positions) != len(original_pieces):
        return []

    with torch.no_grad():
        logits = models.kobert_mlm(**inputs).logits[0]

    dedup: Dict[str, Dict[str, Any]] = {}
    for piece_idx, mask_pos in enumerate(mask_positions):
        values, indices = torch.topk(logits[mask_pos], top_k)
        for logit_value, token_id in zip(values.tolist(), indices.tolist()):
            predicted_piece = models.tokenizer.convert_ids_to_tokens([token_id])[0]
            if predicted_piece in {"[CLS]", "[SEP]", "[PAD]", "[UNK]", "[MASK]"}:
                continue
            if predicted_piece == original_pieces[piece_idx]:
                continue
            mutated_piece_sets: List[List[str]] = []

            direct_pieces = original_pieces.copy()
            direct_pieces[piece_idx] = predicted_piece
            mutated_piece_sets.append(direct_pieces)

            if (
                piece_idx == 0
                and len(original_pieces) >= 2
                and original_pieces[0] == "▁"
                and predicted_piece.startswith("▁")
            ):
                merged_pieces = [predicted_piece, *original_pieces[2:]]
                mutated_piece_sets.append(merged_pieces)

            for mutated_pieces in mutated_piece_sets:
                candidate = models.tokenizer.convert_tokens_to_string(mutated_pieces).strip()
                if not _is_valid_surface_candidate(token, candidate):
                    continue
                existing = dedup.get(candidate)
                payload = {
                    "replacement": candidate,
                    "source": "MLM_TOPK",
                    "generatorScore": round(float(logit_value), 4),
                    "pieceIndex": piece_idx,
                }
                if existing is None or payload["generatorScore"] > existing["generatorScore"]:
                    dedup[candidate] = payload
    return list(dedup.values())


# ── Candidate Scoring / Filtering ─────────────────────────────


def _candidate_source_prior(source: str) -> float:
    if source == "CONFUSION_SET":
        return 1.0
    if source == "DATA_CONFUSION":
        return 0.99
    if source == "PHRASE_MEMORY":
        return 0.99
    if source == "PHRASE_MEMORY_FUZZY":
        return 0.95
    if source == "MORPHEME_CONFUSION":
        return 0.98
    if source == "LEMMA_DATA_CONFUSION":
        return 0.97
    if source == "FAMILY_SEED":
        return 0.97
    if source == "AUTO_FAMILY_SEED":
        return 0.96
    if source == "INFLECTION_RULE":
        return 0.95
    if source == "LEMMA_CONFUSION":
        return 0.95
    if source == "FAMILY_SURFACE_EXAMPLE":
        return 0.95
    if source == "MORPH":
        return 0.9
    if source == "JAMO_FALLBACK":
        return 0.72
    if source == "SURFACE_JAMO_FALLBACK":
        return 0.76
    if source == "JAMO":
        return 0.7
    if source == "MLM_TOPK":
        return 0.35
    return 0.25


def _is_curated_candidate_source(source: str) -> bool:
    return source in CURATED_CANDIDATE_SOURCES


def _normalize_generator_score(score: float) -> float:
    if 0.0 <= score <= 1.0:
        return score
    return 1 / (1 + math.exp(-(score / 4.0)))


def _dedupe_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dedup: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        replacement = item["replacement"]
        existing = dedup.get(replacement)
        if existing is None:
            dedup[replacement] = item
            continue
        current_score = _normalize_generator_score(float(item.get("generatorScore", 0.0)))
        existing_score = _normalize_generator_score(float(existing.get("generatorScore", 0.0)))
        current_prior = _candidate_source_prior(item.get("source", "MLM_TOPK"))
        existing_prior = _candidate_source_prior(existing.get("source", "MLM_TOPK"))
        if (current_score, current_prior) > (existing_score, existing_prior):
            dedup[replacement] = item
    return list(dedup.values())


def _candidate_support_total(item: Dict[str, Any]) -> int:
    stats = item.get("interfaceStats")
    if not isinstance(stats, dict):
        return 0
    total = 0
    for value in stats.values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def _context_window_terms(models: ModelLoader, sentence: str, start: int, end: int, window: int = 3) -> List[str]:
    token_spans = tokenize_with_ranges(sentence)
    if not token_spans:
        return []
    center_index = None
    for idx, (_, st, ed) in enumerate(token_spans):
        if st <= start < ed or st < end <= ed or (st >= start and ed <= end):
            center_index = idx
            break
    if center_index is None:
        return []
    terms: List[str] = []
    for idx in range(max(0, center_index - window), min(len(token_spans), center_index + window + 1)):
        if idx == center_index:
            continue
        token = normalize_context_token(token_spans[idx][0])
        if len(token) < 2 or hangul_ratio(token) < 0.5:
            continue
        terms.append(token)
    return terms


def _hint_lookup(hints: List[Dict[str, Any]]) -> Dict[str, int]:
    lookup: Dict[str, int] = {}
    for item in hints:
        term = item.get("term")
        if not term:
            continue
        normalized = normalize_context_token(str(term))
        if not normalized:
            continue
        lookup[normalized] = lookup.get(normalized, 0) + int(item.get("count", 0))
    return lookup


def _context_hint_score(models: ModelLoader, sentence: str, start: int, end: int, item: Dict[str, Any]) -> float:
    hints = item.get("contextHints") or []
    if not hints:
        return 0.0
    context_terms = _context_window_terms(models, sentence, start, end)
    if not context_terms:
        return 0.0
    hint_lkp = _hint_lookup(hints)
    total = sum(hint_lkp.values()) or 1
    matched = 0
    for term in context_terms:
        matched += hint_lkp.get(term, 0)
        if matched:
            continue
        matched += sum(count for hint, count in hint_lkp.items() if term.startswith(hint) or hint.startswith(term))
    return round(float(min(1.0, matched / total)), 4)


def _candidate_interface_score(item: Dict[str, Any], assumed_channel: str = "generic") -> float:
    stats = item.get("interfaceStats") or {}
    if not stats:
        interface_type = item.get("interfaceType")
        if interface_type == "voice":
            return 0.32 if assumed_channel == "generic" else 0.9
        if interface_type == "keyboard":
            return 0.94 if assumed_channel in {"generic", "keyboard"} else 0.72
        return 0.86

    total = max(1, sum(int(value) for value in stats.values()))
    generic_ratio = int(stats.get("generic", 0)) / total
    keyboard_ratio = int(stats.get("keyboard", 0)) / total
    voice_ratio = int(stats.get("voice", 0)) / total
    if assumed_channel == "voice":
        score = (0.92 * voice_ratio) + (0.55 * generic_ratio) + (0.35 * keyboard_ratio)
    elif assumed_channel == "keyboard":
        score = (0.9 * keyboard_ratio) + (0.7 * generic_ratio) + (0.3 * voice_ratio)
    else:
        score = (0.92 * generic_ratio) + (0.82 * keyboard_ratio) + (0.22 * voice_ratio)
    return round(float(max(0.2, min(1.0, score))), 4)


def _enrich_candidate_context(
    models: ModelLoader,
    sentence: str,
    start: int,
    end: int,
    candidates: List[Dict[str, Any]],
    assumed_channel: str = "generic",
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for item in candidates:
        context_score = _context_hint_score(models, sentence, start, end, item)
        interface_score = _candidate_interface_score(item, assumed_channel=assumed_channel)
        if (
            assumed_channel == "generic"
            and str(item.get("interfaceType") or "") == "voice"
            and context_score < 0.12
        ):
            interface_score = min(interface_score, 0.3)
        enriched.append(
            {
                **item,
                "contextHintScore": context_score,
                "interfaceScore": interface_score,
            }
        )
    return enriched


def _prefilter_candidates(
    models: ModelLoader, token: str, candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    prefiltered: List[Dict[str, Any]] = []
    for item in _dedupe_candidates(candidates):
        replacement = item["replacement"]
        source = str(item.get("source", "MLM_TOPK"))
        generator_norm = _normalize_generator_score(float(item.get("generatorScore", 0.0)))
        source_prior = _candidate_source_prior(source)
        context_score = float(item.get("contextHintScore", 0.0))
        interface_score = float(item.get("interfaceScore", 0.86))
        support_total = _candidate_support_total(item)
        tsim = typo_similarity(token, replacement)
        min_typo_similarity = TYPO_SIMILARITY_MIN
        if _is_curated_candidate_source(source):
            if bool(item.get("familyLemmas")) or context_score >= 0.1 or generator_norm >= 0.72:
                min_typo_similarity = 0.48
        if tsim < min_typo_similarity:
            continue
        if source in {"FAMILY_SEED", "AUTO_FAMILY_SEED"}:
            if context_score < 0.08 and support_total < 14:
                continue
        if source in {"DATA_CONFUSION", "LEMMA_DATA_CONFUSION"}:
            if context_score < 0.08 and support_total < 8 and tsim < 0.62:
                continue
        prefilter_score = round(
            float(
                (0.4 * tsim)
                + (0.22 * generator_norm)
                + (0.12 * source_prior)
                + (0.16 * context_score)
                + (0.1 * interface_score)
            ),
            4,
        )
        prefiltered.append(
            {
                **item,
                "candidateSentence": item.get("candidateSentence"),
                "generatorNorm": round(generator_norm, 4),
                "typoSimilarity": tsim,
                "contextHintScore": round(context_score, 4),
                "interfaceScore": round(interface_score, 4),
                "prefilterScore": prefilter_score,
            }
        )
    prefiltered.sort(
        key=lambda x: (
            x["prefilterScore"],
            x.get("contextHintScore", 0.0),
            x["typoSimilarity"],
            x["generatorNorm"],
        ),
        reverse=True,
    )
    return prefiltered[:CANDIDATE_PREFILTER_LIMIT]


# ── Context Verifier / Ranking ────────────────────────────────


def _context_verifier_score(
    typo_sim: float,
    electra_delta: float,
    kiwi_delta: float,
    retrieval_score: float,
    interface_score: float,
) -> float:
    verifier_signal = (
        (electra_delta * 10.0)
        + (kiwi_delta / 2.8)
        + ((typo_sim - 0.55) * 1.4)
        + (retrieval_score * 1.1)
        + ((interface_score - 0.5) * 0.55)
    )
    return 1 / (1 + math.exp(-verifier_signal))


def _rank_replacement_candidates(
    models: ModelLoader,
    sentence: str,
    token: str,
    start: int,
    end: int,
    candidates: List[Dict[str, Any]],
    baseline: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    original_mlm = models.pseudo_logprob_for_replacement(sentence, start, end, token)
    original_normal = baseline["koelectraScore"] if baseline is not None else models.koelectra_normal_score(sentence)
    original_kiwi = baseline["kiwiScore"] if baseline is not None else models.kiwi_sentence_score(sentence)
    ranked: List[Dict[str, Any]] = []
    for item in candidates:
        replacement = item["replacement"]
        candidate_sentence = sentence[:start] + replacement + sentence[end:]
        mlm_score = models.pseudo_logprob_for_replacement(sentence, start, end, replacement)
        electra_score = models.koelectra_normal_score(candidate_sentence)
        kiwi_score = models.kiwi_sentence_score(candidate_sentence)

        mlm_delta = mlm_score - original_mlm
        electra_delta = electra_score - original_normal
        kiwi_delta = kiwi_score - original_kiwi

        mlm_gain = 1 / (1 + math.exp(-mlm_delta))
        electra_gain = 1 / (1 + math.exp(-(electra_delta * 8.0)))
        kiwi_gain = 1 / (1 + math.exp(-(kiwi_delta / 3.0)))
        source = item.get("source", "MLM_TOPK")
        source_prior = _candidate_source_prior(source)
        edit_penalty = min(0.18, 0.06 * char_edit_distance(token, replacement))
        tsim = typo_similarity(token, replacement)
        typo_bonus = 0.08 * tsim
        source_bonus = 0.04 * source_prior
        retrieval_bonus = 0.04 * item.get("contextHintScore", 0.0)
        interface_bonus = 0.02 * item.get("interfaceScore", 0.86)
        semantic_penalty = 0.16 if tsim < 0.6 else 0.0
        context_verifier = _context_verifier_score(
            typo_sim=tsim,
            electra_delta=electra_delta,
            kiwi_delta=kiwi_delta,
            retrieval_score=float(item.get("contextHintScore", 0.0)),
            interface_score=float(item.get("interfaceScore", 0.86)),
        )
        family_bonus = 0.04 if item.get("familyMatch") else 0.0

        final_score = (
            0.24 * mlm_gain
            + 0.28 * electra_gain
            + 0.12 * kiwi_gain
            + 0.22 * context_verifier
            + 0.1 * (1 / (1 + math.exp(-item["generatorScore"])))
            + typo_bonus
            + source_bonus
            + retrieval_bonus
            + interface_bonus
            + family_bonus
            - edit_penalty
            - semantic_penalty
        )

        ranked.append(
            {
                **item,
                "candidateSentence": candidate_sentence,
                "mlmScore": round(float(mlm_score), 4),
                "mlmDelta": round(float(mlm_delta), 4),
                "koelectraScore": round(float(electra_score), 4),
                "koelectraDelta": round(float(electra_delta), 4),
                "kiwiScore": round(float(kiwi_score), 4),
                "kiwiDelta": round(float(kiwi_delta), 4),
                "typoSimilarity": tsim,
                "contextHintScore": round(float(item.get("contextHintScore", 0.0)), 4),
                "interfaceScore": round(float(item.get("interfaceScore", 0.86)), 4),
                "contextVerifierScore": round(float(context_verifier), 4),
                "familyBonus": round(float(family_bonus), 4),
                "finalScore": round(float(max(0.01, min(0.99, final_score))), 4),
            }
        )

    ranked.sort(key=lambda x: x["finalScore"], reverse=True)
    return ranked


# ── Open Replace Candidates ───────────────────────────────────


def open_replace_candidates_for_token(models: ModelLoader, token: str) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = list(models.surface_confusion_map.get(token, []))

    for item in _single_token_phrase_candidates(models, token):
        payload = _phrase_candidate_payload(item, source="PHRASE_MEMORY")
        if payload is not None:
            candidates.append(payload)

    candidates.extend(_common_surface_pattern_candidates(models, token))

    # 활용형 오타를 포착하기 위한 stem 치환 규칙.
    # 예: 낫아 -> 낳아, 낫어서 -> 낳어서, 낫으면 -> 낳으면
    for rule in STEM_CONFUSION_RULES:
        wrong = rule["wrong_stem"]
        if token.startswith(wrong) and len(token) > len(wrong):
            suffix = token[len(wrong) :]
            replacement = f"{rule['correct_stem']}{suffix}"
            if replacement != token:
                score = rule["baseScore"]
                if suffix.startswith(("아", "어", "았", "었")):
                    score = max(score, 0.9)
                candidates.append(
                    {
                        "replacement": replacement,
                        "source": rule["source"],
                        "generatorScore": score,
                    }
                )

    dedup: Dict[str, Dict[str, Any]] = {}
    for c in candidates:
        rep = c["replacement"]
        if rep not in dedup or c["generatorScore"] > dedup[rep]["generatorScore"]:
            dedup[rep] = c
    return list(dedup.values())


def generate_contextual_candidates_for_token(
    models: ModelLoader,
    sentence: str,
    token: str,
    start: int,
    end: int,
    expensive: bool = True,
    assumed_channel: str = "generic",
) -> List[Dict[str, Any]]:
    curated_candidates = _enrich_candidate_context(
        models,
        sentence,
        start,
        end,
        open_replace_candidates_for_token(models, token),
        assumed_channel=assumed_channel,
    )
    surface_fallback_candidates = _enrich_candidate_context(
        models,
        sentence,
        start,
        end,
        _surface_jamo_fallback_candidates(models, token),
        assumed_channel=assumed_channel,
    )
    join_candidates = _build_join_candidates(models, sentence, token, start, end)
    join_candidates = _enrich_candidate_context(
        models,
        sentence,
        start,
        end,
        join_candidates,
        assumed_channel=assumed_channel,
    )
    curated_pool = _prefilter_candidates(
        models,
        token,
        [*curated_candidates, *surface_fallback_candidates, *join_candidates],
    )
    needs_mlm_backoff = (
        len(curated_pool) < MLM_BACKOFF_MIN_CANDIDATES
        or max((item.get("prefilterScore", 0.0) for item in curated_pool), default=0.0) < MLM_BACKOFF_PREFILTER_MIN
    )
    candidate_pool = curated_pool
    if needs_mlm_backoff:
        mlm_candidates = _enrich_candidate_context(
            models,
            sentence,
            start,
            end,
            _mlm_surface_candidates(models, sentence, token, start, end),
            assumed_channel=assumed_channel,
        )
        candidate_pool = _prefilter_candidates(
            models,
            token,
            [*curated_pool, *mlm_candidates],
        )
    if not candidate_pool:
        return []
    if not expensive:
        return candidate_pool
    baseline = models.sentence_quality_baseline(sentence)
    return _rank_replacement_candidates(
        models,
        sentence,
        token,
        start,
        end,
        candidate_pool,
        baseline=baseline,
    )[:CANDIDATE_PREFILTER_LIMIT]


# ── Typo Filter / Reopen ─────────────────────────────────────


def _filter_typo_like_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for candidate in candidates:
        min_tsim = TYPO_SIMILARITY_MIN
        if _is_curated_candidate_source(str(candidate.get("source", ""))):
            if (
                bool(candidate.get("familyLemmas"))
                or float(candidate.get("contextHintScore", 0.0)) >= 0.1
                or float(candidate.get("generatorNorm", 0.0)) >= 0.72
            ):
                min_tsim = 0.48
        if float(candidate.get("typoSimilarity", 0.0)) >= min_tsim:
            filtered.append(candidate)
    return filtered


def _candidate_is_predicate_like(models: ModelLoader, item: Dict[str, Any]) -> bool:
    pos = str(item.get("pos") or "")
    if pos.startswith(("VV", "VA", "VX")):
        return True
    family_lemmas = item.get("familyLemmas") or []
    if family_lemmas:
        return True
    replacement = str(item.get("replacement") or "")
    if not replacement:
        return False
    return bool(_predicate_lemmas_from_text(models, replacement, top_n=3))


def _maybe_reopen_short_valid_token(
    models: ModelLoader,
    text: str,
    token: str,
    start: int,
    end: int,
    base_confidence: float,
) -> Optional[Dict[str, Any]]:
    if not re.fullmatch(r"[가-힣]+", token):
        return None
    if len(token) < 2 or len(token) > 6:
        return None

    predicate_lemmas = _predicate_lemmas_from_text(models, token, top_n=3)
    if not predicate_lemmas:
        return None
    if not any(models.lemma_confusion_map.get(lemma) for lemma in predicate_lemmas):
        return None

    ranked_candidates = generate_contextual_candidates_for_token(
        models,
        text,
        token,
        start,
        end,
        expensive=False,
    )
    typo_like_candidates = _filter_typo_like_candidates(ranked_candidates)
    if not typo_like_candidates:
        return None

    best = typo_like_candidates[0]
    best_source = str(best.get("source") or "")
    if best_source not in CURATED_CANDIDATE_SOURCES:
        return None
    if not _candidate_is_predicate_like(models, best):
        return None
    token_len = len(token)
    min_prefilter = 0.58 if token_len <= 4 else 0.66
    min_generator_norm = 0.72 if token_len <= 4 else 0.82
    if best.get("prefilterScore", 0.0) < min_prefilter:
        return None
    if best.get("generatorNorm", 0.0) < min_generator_norm:
        return None
    context_hint_score = float(best.get("contextHintScore", 0.0) or 0.0)
    manual_reopen_sources = {"FAMILY_SEED", "PHRASE_MEMORY", "FAMILY_SURFACE_EXAMPLE", "AUTO_FAMILY_SEED"}
    min_context_score = 0.15 if token_len <= 4 else 0.08
    if best_source not in manual_reopen_sources and context_hint_score < max(min_context_score, 0.18):
        return None
    if context_hint_score < min_context_score and best.get("familyMatch") is False:
        return None

    return {
        "token": token,
        "label": "OPEN_REPLACE",
        "confidence": max(base_confidence, round(float(best.get("prefilterScore", 0.0)), 4)),
        "range": {"start": start, "end": end},
        "modelCandidates": typo_like_candidates[:5],
        "detectionEvidence": {
            "bestReplacement": best["replacement"],
            "generatorNorm": best.get("generatorNorm"),
            "prefilterScore": best.get("prefilterScore"),
            "typoSimilarity": best.get("typoSimilarity", 0.0),
            "contextHintScore": context_hint_score,
            "source": "finetuned_edit_tagger_reopen",
            "taggerConfidence": base_confidence,
            "predicateLemmas": predicate_lemmas,
        },
    }


# ── Edit Tagger (Heuristic / Fine-tuned) ─────────────────────


def _run_heuristic_edit_tagger(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    for token, st, ed in tokenize_with_ranges(text):
        span = {"start": st, "end": ed}
        if is_range_protected(span, protected):
            labels.append({"token": token, "label": "KEEP", "confidence": 1.0, "range": span})
            continue
        if hangul_ratio(token) < 0.5 or len(token) < 2:
            labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
            continue

        ranked_candidates = generate_contextual_candidates_for_token(models, text, token, st, ed, expensive=False)
        if not ranked_candidates:
            labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
            continue

        typo_like_candidates = _filter_typo_like_candidates(ranked_candidates)
        if not typo_like_candidates:
            labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
            continue

        best = typo_like_candidates[0]
        suspicious = best["prefilterScore"] >= 0.72
        labels.append(
            {
                "token": token,
                "label": "OPEN_REPLACE" if suspicious else "KEEP",
                "confidence": round(
                    float(best["prefilterScore"] if suspicious else 1 - best["prefilterScore"] * 0.4),
                    4,
                ),
                "range": span,
                "modelCandidates": typo_like_candidates[:5],
                "detectionEvidence": {
                    "bestReplacement": best["replacement"],
                    "generatorNorm": best["generatorNorm"],
                    "prefilterScore": best["prefilterScore"],
                    "typoSimilarity": best.get("typoSimilarity", 0.0),
                    "source": "heuristic",
                },
            }
        )
    return labels


def _run_finetuned_edit_tagger(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if models.edit_tagger_model is None or models.edit_tagger_tokenizer is None:
        return _run_heuristic_edit_tagger(models, text, protected)

    token_ranges = tokenize_with_ranges(text)
    if not token_ranges:
        return []

    words = [token for token, _, _ in token_ranges]
    inputs = models.edit_tagger_tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    with torch.no_grad():
        logits = models.edit_tagger_model(**inputs).logits[0]
        probabilities = torch.softmax(logits, dim=-1)

    labels: List[Dict[str, Any]] = []
    word_ids = inputs.word_ids(batch_index=0)
    seen_word_ids = set()

    for piece_index, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen_word_ids:
            continue
        seen_word_ids.add(word_id)

        token, st, ed = token_ranges[word_id]
        span = {"start": st, "end": ed}
        if is_range_protected(span, protected):
            labels.append({"token": token, "label": "KEEP", "confidence": 1.0, "range": span})
            continue
        if re.search(r"[A-Za-z0-9]", token):
            labels.append({"token": token, "label": "KEEP", "confidence": 0.99, "range": span})
            continue
        if hangul_ratio(token) < 0.5 or len(token) < 2:
            labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
            continue

        pred_id = int(torch.argmax(probabilities[piece_index]).item())
        pred_label = models.edit_tagger_id2label.get(pred_id, "KEEP")
        confidence = round(float(probabilities[piece_index][pred_id].item()), 4)

        if pred_label == "OPEN_REPLACE":
            ranked_candidates = generate_contextual_candidates_for_token(models, text, token, st, ed, expensive=False)
            typo_like_candidates = _filter_typo_like_candidates(ranked_candidates)
            if not typo_like_candidates:
                labels.append(
                    {
                        "token": token,
                        "label": "OPEN_REPLACE",
                        "confidence": confidence,
                        "range": span,
                        "modelCandidates": [],
                        "detectionEvidence": {
                            "source": "finetuned_edit_tagger",
                            "taggerConfidence": confidence,
                            "candidateBackoff": "none",
                        },
                    }
                )
                continue
            best = typo_like_candidates[0]
            labels.append(
                {
                    "token": token,
                    "label": "OPEN_REPLACE",
                    "confidence": confidence,
                    "range": span,
                    "modelCandidates": typo_like_candidates[:5],
                    "detectionEvidence": {
                        "bestReplacement": best["replacement"],
                        "generatorNorm": best["generatorNorm"],
                        "prefilterScore": best["prefilterScore"],
                        "typoSimilarity": best.get("typoSimilarity", 0.0),
                        "source": "finetuned_edit_tagger",
                        "taggerConfidence": confidence,
                    },
                }
            )
            continue

        if pred_label in ROUTING_PROMOTION_LABELS:
            ranked_candidates = generate_contextual_candidates_for_token(models, text, token, st, ed, expensive=False)
            typo_like_candidates = _filter_typo_like_candidates(ranked_candidates)
            if typo_like_candidates:
                best = typo_like_candidates[0]
                labels.append(
                    {
                        "token": token,
                        "label": pred_label,
                        "confidence": confidence,
                        "range": span,
                        "modelCandidates": typo_like_candidates[:5],
                        "detectionEvidence": {
                            "bestReplacement": best["replacement"],
                            "generatorNorm": best["generatorNorm"],
                            "prefilterScore": best["prefilterScore"],
                            "typoSimilarity": best.get("typoSimilarity", 0.0),
                            "source": "finetuned_edit_tagger",
                            "taggerConfidence": confidence,
                        },
                    }
                )
                continue

        if pred_label == "KEEP":
            reopened = _maybe_reopen_short_valid_token(models, text, token, st, ed, confidence)
            if reopened is not None:
                labels.append(reopened)
                continue

        labels.append(
            {
                "token": token,
                "label": pred_label,
                "confidence": confidence,
                "range": span,
                "detectionEvidence": {
                    "source": "finetuned_edit_tagger",
                    "taggerConfidence": confidence,
                },
            }
        )

    return labels


def run_edit_tagger(
    models: ModelLoader, text: str, protected: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    if models.edit_tagger_model is not None:
        return _run_finetuned_edit_tagger(models, text, protected)
    return _run_heuristic_edit_tagger(models, text, protected)


# ── Routing ───────────────────────────────────────────────────


def route_tagger_labels(tag_labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    routed: List[Dict[str, Any]] = []
    for label in tag_labels:
        effective_label = label["label"]
        routing_decision = "PASS_TO_DOWNSTREAM"
        routing_note = "후속 후보 생성 단계로 전달"
        best_candidate = (label.get("modelCandidates") or [None])[0]

        if (
            label["label"] in ROUTING_PROMOTION_LABELS
            and best_candidate is not None
            and best_candidate.get("source") in {*CURATED_CANDIDATE_SOURCES, "LEMMA_CONFUSION", "MORPHEME_CONFUSION"}
            and best_candidate.get("prefilterScore", best_candidate.get("finalScore", 0.0)) >= ROUTING_PROMOTION_MIN_SCORE
        ):
            effective_label = "OPEN_REPLACE"
            routing_decision = "PROMOTE_TO_OPEN_REPLACE"
            routing_note = "모델 후보 점수가 충분해 문맥 치환 단계로 승격"

        if label["label"] == "SPACE_FIX":
            effective_label = "KEEP"
            routing_decision = "DEFER_TO_SPACING"
            routing_note = "띄어쓰기 단계가 먼저 처리하므로 여기서는 후보 생성에서 제외"
        elif label["label"] == "PUNCT_FIX" and effective_label != "OPEN_REPLACE":
            effective_label = "KEEP"
            routing_decision = "DEFER_TO_RULES"
            routing_note = "문장부호는 규칙/정책 단계에서 처리하고 여기서는 후보 생성에서 제외"

        routed.append(
            {
                **label,
                "effectiveLabel": effective_label,
                "routingDecision": routing_decision,
                "routingNote": routing_note,
            }
        )
    return routed


# ── Lock Guards ───────────────────────────────────────────────


def _high_precision_lock_spans(decisions: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    locked: List[Dict[str, int]] = []
    for decision in decisions:
        if decision.get("stage") not in {"RULE", "PRE_NORMALIZE"}:
            continue
        if float(decision.get("confidence", 0.0)) < 0.96:
            continue
        source_text = str(decision.get("sourceText") or "")
        replacement = str(decision.get("replacement") or "")
        if len(source_text.strip()) < 2 or len(replacement.strip()) < 2:
            continue
        locked.append({"start": int(decision["range"]["start"]), "end": int(decision["range"]["end"])})
    return locked


def _phrase_target_lock_spans(
    models: ModelLoader,
    text: str,
    protected: List[Dict[str, Any]],
) -> List[Dict[str, int]]:
    locked: List[Dict[str, int]] = []
    token_ranges = tokenize_with_ranges(text)
    for index, (token, start, _) in enumerate(token_ranges):
        candidates = models.phrase_memory_target_index.get(token) or []
        if not candidates:
            continue
        for item in candidates:
            target = str(item.get("to") or "")
            target_parts = tokenize_with_ranges(target)
            span_len = len(target_parts)
            if not target_parts or index + span_len > len(token_ranges):
                continue
            end = token_ranges[index + span_len - 1][2]
            span = {"start": start, "end": end}
            if is_range_protected(span, protected):
                continue
            source_text = text[start:end]
            if normalize_phrase_surface(source_text) != normalize_phrase_surface(target):
                continue
            locked.append(span)
            break
    return locked


def _meta_comparison_lock_spans(
    text: str,
    protected: List[Dict[str, Any]],
) -> List[Dict[str, int]]:
    if "?" not in text and "？" not in text:
        return []
    token_ranges = tokenize_with_ranges(text)
    matched: List[Dict[str, int]] = []
    for token, start, end in token_ranges:
        span = {"start": start, "end": end}
        if is_range_protected(span, protected):
            continue
        if META_COMPARE_TOKEN_PATTERN.fullmatch(token):
            matched.append(span)
    if len(matched) < 2:
        return []
    return matched


def apply_high_precision_lock_guard(
    labels: List[Dict[str, Any]],
    locked_spans: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    if not locked_spans:
        return labels
    guarded: List[Dict[str, Any]] = []
    for label in labels:
        effective = label.get("effectiveLabel", label["label"])
        span = label.get("range") or {"start": 0, "end": 0}
        if effective != "OPEN_REPLACE":
            guarded.append(label)
            continue
        if not any(range_overlaps(span, locked) for locked in locked_spans):
            guarded.append(label)
            continue
        best_candidate = (label.get("modelCandidates") or [None])[0]
        if best_candidate is not None:
            source = str(best_candidate.get("source", ""))
            context_score = float(best_candidate.get("contextHintScore", 0.0))
            if source in {"FAMILY_SEED", "FAMILY_SURFACE_EXAMPLE"} and context_score >= 0.18:
                guarded.append(label)
                continue
        guarded.append(
            {
                **label,
                "effectiveLabel": "KEEP",
                "routingDecision": "LOCKED_HIGH_PRECISION_SPAN",
                "routingNote": "앞단 고정밀 정규화가 적용된 span이라 약한 후속 치환 제안은 차단",
            }
        )
    return guarded


def apply_meta_comparison_guard(
    labels: List[Dict[str, Any]],
    locked_spans: List[Dict[str, int]],
) -> List[Dict[str, Any]]:
    if not locked_spans:
        return labels
    guarded: List[Dict[str, Any]] = []
    for label in labels:
        effective = label.get("effectiveLabel", label["label"])
        span = label.get("range") or {"start": 0, "end": 0}
        if effective != "OPEN_REPLACE" or not any(range_overlaps(span, locked) for locked in locked_spans):
            guarded.append(label)
            continue
        guarded.append(
            {
                **label,
                "effectiveLabel": "KEEP",
                "routingDecision": "LOCKED_META_COMPARISON_SPAN",
                "routingNote": "비교/설명형 메타 문장 구간이라 후보 생성에서 제외",
            }
        )
    return guarded


# ── Phrase / Retokenized Pair Candidate Groups ────────────────


def _phrase_span_candidate_group(
    models: ModelLoader,
    sentence: str,
    tag_labels: List[Dict[str, Any]],
    start_index: int,
    max_span_len: int = 3,
) -> Optional[Dict[str, Any]]:
    first_label = tag_labels[start_index]
    if first_label.get("effectiveLabel", first_label["label"]) != "OPEN_REPLACE":
        return None

    run_end = start_index
    while run_end < len(tag_labels):
        current = tag_labels[run_end]
        if current.get("effectiveLabel", current["label"]) != "OPEN_REPLACE":
            break
        run_end += 1
        if run_end - start_index >= max_span_len:
            break
    if run_end - start_index < 2:
        return None

    first_token = first_label["token"]
    candidates = models.phrase_memory_index.get(first_token) or []
    if not candidates:
        return None

    group_candidates: Dict[int, Dict[str, Any]] = {}
    for item in candidates:
        source = str(item.get("from") or "").strip()
        replacement = str(item.get("to") or "").strip()
        source_parts = tokenize_with_ranges(source)
        span_len = len(source_parts)
        if span_len < 2 or span_len > run_end - start_index:
            continue
        if not replacement or replacement == source:
            continue

        label_slice = tag_labels[start_index : start_index + span_len]
        span_start = label_slice[0]["range"]["start"]
        span_end = label_slice[-1]["range"]["end"]
        source_text = sentence[span_start:span_end]
        normalized_source = normalize_phrase_surface(source)
        normalized_text = normalize_phrase_surface(source_text)
        if not normalized_source or not normalized_text:
            continue

        source_name = "PHRASE_MEMORY"
        g_score = float(item.get("confidence", 0.96))
        match_similarity = 1.0
        collapsed_source = normalized_source.replace(" ", "")
        collapsed_text = normalized_text.replace(" ", "")
        if normalized_text != normalized_source:
            match_similarity = typo_similarity(collapsed_text, collapsed_source)
            if (
                match_similarity < 0.94
                or char_edit_distance(collapsed_text, collapsed_source) > 1
                or float(item.get("confidence", 0.0)) < 0.98
            ):
                continue
            source_name = "PHRASE_MEMORY_FUZZY"
            g_score = max(
                0.72,
                float(item.get("confidence", 0.96)) - 0.08 + ((match_similarity - 0.86) * 0.4),
            )

        payload = _phrase_candidate_payload(
            item,
            source=source_name,
            generator_score=g_score,
        )
        if payload is None:
            continue

        ctx_hint_score = _context_hint_score(models, sentence, span_start, span_end, payload)
        iface_score = _candidate_interface_score(payload, assumed_channel="generic")
        gen_norm = _normalize_generator_score(float(payload.get("generatorScore", 0.0)))
        phrase_span_score = round(
            float(
                (0.48 * gen_norm)
                + (0.28 * match_similarity)
                + (0.14 * ctx_hint_score)
                + (0.1 * iface_score)
            ),
            4,
        )
        annotated = {
            **payload,
            "generatorNorm": round(gen_norm, 4),
            "typoSimilarity": round(float(match_similarity), 4),
            "contextHintScore": round(float(ctx_hint_score), 4),
            "interfaceScore": round(float(iface_score), 4),
            "prefilterScore": phrase_span_score,
            "phraseSpanScore": phrase_span_score,
            "matchedPattern": source,
        }

        bucket = group_candidates.setdefault(
            span_len,
            {
                "spanStart": span_start,
                "spanEnd": span_end,
                "sourceText": source_text,
                "labelSlice": label_slice,
                "items": [],
            },
        )
        bucket["items"].append(annotated)

    best_group: Optional[Dict[str, Any]] = None
    best_score = -1.0
    for span_len, bucket in group_candidates.items():
        ranked_items = sorted(
            bucket["items"],
            key=lambda item: (
                item.get("phraseSpanScore", 0.0),
                item.get("generatorScore", 0.0),
                item.get("typoSimilarity", 0.0),
            ),
            reverse=True,
        )
        if not ranked_items:
            continue
        top_item = ranked_items[0]
        min_score = 0.86 if top_item["source"] == "PHRASE_MEMORY_FUZZY" else 0.76
        if float(top_item.get("phraseSpanScore", 0.0)) < min_score:
            continue
        if float(top_item.get("typoSimilarity", 0.0)) < 0.97 and float(top_item.get("contextHintScore", 0.0)) < 0.12:
            continue
        if float(top_item.get("phraseSpanScore", 0.0)) > best_score:
            best_group = {
                "spanLen": span_len,
                "spanStart": bucket["spanStart"],
                "spanEnd": bucket["spanEnd"],
                "sourceText": bucket["sourceText"],
                "labelSlice": bucket["labelSlice"],
                "rankedItems": ranked_items[:CANDIDATE_PREFILTER_LIMIT],
                "topItem": top_item,
            }
            best_score = float(top_item.get("phraseSpanScore", 0.0))

    if best_group is None:
        return None

    avg_confidence = sum(float(label.get("confidence", 0.0)) for label in best_group["labelSlice"]) / len(
        best_group["labelSlice"]
    )
    family_prior = _build_candidate_family_prior(
        models,
        sentence,
        best_group["sourceText"],
        best_group["spanStart"],
        best_group["spanEnd"],
        best_group["topItem"]["replacement"],
    )
    return {
        "consumeCount": best_group["spanLen"],
        "group": {
            "span": {"start": best_group["spanStart"], "end": best_group["spanEnd"]},
            "original": best_group["sourceText"],
            "taggerConfidence": round(float(avg_confidence), 4),
            "detectionEvidence": {
                "source": "phrase_memory_span",
                "matchedPattern": best_group["topItem"].get("matchedPattern"),
                "matchSimilarity": best_group["topItem"].get("typoSimilarity"),
                "phraseSpanScore": best_group["topItem"].get("phraseSpanScore"),
            },
            "candidateFamily": family_prior,
            "items": [
                *[
                    {
                        **item,
                        "familyLemmas": None,
                    }
                    for item in best_group["rankedItems"]
                ],
                {
                    "replacement": best_group["sourceText"],
                    "source": "ORIGINAL",
                    "generatorScore": 0.2,
                },
            ],
        },
    }


def _retokenized_pair_candidate_group(
    models: ModelLoader,
    sentence: str,
    tag_labels: List[Dict[str, Any]],
    start_index: int,
) -> Optional[Dict[str, Any]]:
    if start_index + 1 >= len(tag_labels):
        return None

    first = tag_labels[start_index]
    second = tag_labels[start_index + 1]
    first_effective = first.get("effectiveLabel", first["label"])
    second_effective = second.get("effectiveLabel", second["label"])
    if first_effective != "KEEP" or second_effective != "KEEP":
        return None
    if sentence[first["range"]["end"] : second["range"]["start"]].strip():
        return None

    source_text = sentence[first["range"]["start"] : second["range"]["end"]]
    collapsed_source = re.sub(r"\s+", "", source_text)
    if len(collapsed_source) < 2 or hangul_ratio(collapsed_source) < 0.7:
        return None

    candidates: List[Dict[str, Any]] = []
    surface_fix = models.surface_fix_index.get(collapsed_source)
    if surface_fix is not None:
        candidates.append(
            {
                "replacement": str(surface_fix.get("to") or "").strip(),
                "source": "SURFACE_FIX_RETOKENIZED",
                "generatorScore": round(float(surface_fix.get("confidence", 0.96)), 4),
            }
        )
    candidates.extend(_common_surface_pattern_candidates(models, collapsed_source))
    if not candidates:
        return None

    span = {"start": first["range"]["start"], "end": second["range"]["end"]}
    ranked_items: List[Dict[str, Any]] = []
    for item in candidates:
        replacement = str(item.get("replacement") or "").strip()
        if not replacement or replacement == source_text:
            continue
        tsim = typo_similarity(collapsed_source, replacement.replace(" ", ""))
        source_name = str(item.get("source") or "")
        typo_min = 0.62 if source_name in {"COMMON_PHRASE_PATTERN", "SURFACE_FIX_RETOKENIZED"} else 0.72
        if tsim < typo_min:
            continue
        ctx_hint_score = _context_hint_score(models, sentence, span["start"], span["end"], item)
        iface_score = _candidate_interface_score(item, assumed_channel="generic")
        gen_norm = _normalize_generator_score(float(item.get("generatorScore", 0.0)))
        prefilter_score = round(
            float(
                (0.46 * gen_norm)
                + (0.3 * tsim)
                + (0.14 * ctx_hint_score)
                + (0.1 * iface_score)
            ),
            4,
        )
        ranked_items.append(
            {
                "replacement": replacement,
                "source": item.get("source", "RETOKENIZED_PAIR"),
                "generatorScore": item.get("generatorScore", 0.0),
                "generatorNorm": round(gen_norm, 4),
                "prefilterScore": prefilter_score,
                "finalScore": prefilter_score,
                "rerankScore": prefilter_score,
                "typoSimilarity": round(float(tsim), 4),
                "contextHintScore": round(float(ctx_hint_score), 4),
                "interfaceScore": round(float(iface_score), 4),
            }
        )

    ranked_items.sort(
        key=lambda item: (
            item.get("prefilterScore", 0.0),
            item.get("generatorScore", 0.0),
            item.get("typoSimilarity", 0.0),
        ),
        reverse=True,
    )
    if not ranked_items:
        return None
    top_source = str(ranked_items[0].get("source") or "")
    min_prefilter = 0.72 if top_source in {"COMMON_PHRASE_PATTERN", "SURFACE_FIX_RETOKENIZED"} else 0.78
    if float(ranked_items[0].get("prefilterScore", 0.0)) < min_prefilter:
        return None

    avg_confidence = (float(first.get("confidence", 0.0)) + float(second.get("confidence", 0.0))) / 2.0
    family_prior = _build_candidate_family_prior(
        models,
        sentence,
        collapsed_source,
        span["start"],
        span["end"],
        ranked_items[0]["replacement"],
    )
    return {
        "consumeCount": 2,
        "group": {
            "span": span,
            "original": source_text,
            "taggerConfidence": round(avg_confidence * 0.78, 4),
            "detectionEvidence": {
                "source": "retokenized_pair",
                "bestReplacement": ranked_items[0]["replacement"],
                "collapsedSource": collapsed_source,
            },
            "candidateFamily": family_prior,
            "items": [
                *ranked_items[:CANDIDATE_PREFILTER_LIMIT],
                {"replacement": source_text, "source": "ORIGINAL", "generatorScore": 0.2},
            ],
        },
    }
