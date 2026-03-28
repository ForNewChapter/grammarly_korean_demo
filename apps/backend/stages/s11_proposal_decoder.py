# 11단계: 빔 서치로 최적 교정 조합을 디코딩하고, Seq2Seq 모델의 제안을 병합한다.

import difflib
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from model_loader import (
    PROPOSAL_BEAM_WIDTH,
    PROPOSAL_TOP_K,
    SEQ2SEQ_SPAN_CONTEXT_TOKENS,
    SEQ2SEQ_SPAN_MAX_TOKENS,
    ModelLoader,
)
from stages.s10_guardrail import _proposal_edit_from_option
from utils.edit_ops import apply_edits, range_overlaps, tokenize_with_ranges
from utils.hangul import char_edit_distance, typo_similarity


# ── 헬퍼 ──────────────────────────────────────────────────────


def _protected_texts_preserved(
    original: str,
    generated: str,
    protected: List[Dict[str, Any]],
) -> bool:
    for span in protected:
        start = int(span.get("start", 0))
        end = int(span.get("end", 0))
        token = original[start:end]
        if token and token.strip() and token not in generated:
            return False
    return True


def _sentence_edit_ratio(original: str, candidate: str) -> float:
    if not original and not candidate:
        return 0.0
    return char_edit_distance(original, candidate) / max(1, max(len(original), len(candidate)))


def _has_suspicious_generation_artifact(text: str) -> bool:
    return bool(re.search(r"[^가-힣A-Za-z0-9\s.,!?~:;'\"()\[\]{}<>/%+-]", text))


def _project_protected_spans(
    protected: List[Dict[str, Any]],
    window_start: int,
    window_end: int,
) -> List[Dict[str, int]]:
    projected: List[Dict[str, int]] = []
    for span in protected:
        start = int(span["range"]["start"])
        end = int(span["range"]["end"])
        overlap_start = max(window_start, start)
        overlap_end = min(window_end, end)
        if overlap_start >= overlap_end:
            continue
        projected.append(
            {
                "start": overlap_start - window_start,
                "end": overlap_end - window_start,
            }
        )
    return projected


def _tagger_open_replace_windows(
    sentence: str,
    tag_labels: List[Dict[str, Any]],
    context_tokens: int = SEQ2SEQ_SPAN_CONTEXT_TOKENS,
    max_span_tokens: int = SEQ2SEQ_SPAN_MAX_TOKENS,
) -> List[Dict[str, Any]]:
    token_ranges = tokenize_with_ranges(sentence)
    if not token_ranges or not tag_labels:
        return []

    windows: List[Dict[str, Any]] = []
    index = 0
    while index < len(tag_labels):
        label = tag_labels[index]
        if label.get("effectiveLabel", label["label"]) != "OPEN_REPLACE":
            index += 1
            continue

        run_end = index + 1
        while run_end < len(tag_labels):
            current = tag_labels[run_end]
            if current.get("effectiveLabel", current["label"]) != "OPEN_REPLACE":
                break
            run_end += 1

        chunk_start = index
        while chunk_start < run_end:
            chunk_end = min(run_end, chunk_start + max_span_tokens)
            span_start = tag_labels[chunk_start]["range"]["start"]
            span_end = tag_labels[chunk_end - 1]["range"]["end"]
            token_start = max(0, chunk_start - context_tokens)
            token_end = min(len(token_ranges), chunk_end + context_tokens)
            window_start = token_ranges[token_start][1]
            window_end = token_ranges[token_end - 1][2]
            windows.append(
                {
                    "span": {"start": span_start, "end": span_end},
                    "window": {"start": window_start, "end": window_end},
                    "windowText": sentence[window_start:window_end],
                    "localSpan": {
                        "start": span_start - window_start,
                        "end": span_end - window_start,
                    },
                }
            )
            chunk_start = chunk_end
        index = run_end
    return windows


def _window_change_summary(original: str, candidate: str) -> Optional[Dict[str, int]]:
    matcher = difflib.SequenceMatcher(a=original, b=candidate)
    a_start: Optional[int] = None
    a_end = 0
    b_start: Optional[int] = None
    b_end = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if a_start is None or i1 < a_start:
            a_start = i1
        if b_start is None or j1 < b_start:
            b_start = j1
        a_end = max(a_end, i2)
        b_end = max(b_end, j2)
    if a_start is None or b_start is None:
        return None
    return {
        "a_start": a_start,
        "a_end": a_end,
        "b_start": b_start,
        "b_end": b_end,
    }


def _change_overlaps_local_span(
    change: Dict[str, int],
    local_span: Dict[str, int],
    margin: int = 2,
) -> bool:
    change_start = int(change["a_start"])
    change_end = int(change["a_end"])
    span_start = max(0, int(local_span["start"]) - margin)
    span_end = int(local_span["end"]) + margin
    if change_start == change_end:
        return span_start <= change_start <= span_end
    return change_start < span_end and change_end > span_start


# ── Beam Decoder ──────────────────────────────────────────────


def _decode_sentence_proposals(
    models: ModelLoader,
    sentence: str,
    groups: List[Dict[str, Any]],
    top_k: int = PROPOSAL_TOP_K,
    beam_width: int = PROPOSAL_BEAM_WIDTH,
) -> List[Dict[str, Any]]:
    if not groups:
        return [
            {
                "text": sentence,
                "finalClause": sentence,
                "suggestedClause": sentence,
                "score": 0.99,
                "editCount": 0,
                "selectedOptions": [],
                "edits": [],
            }
        ]

    ordered_groups = sorted(groups, key=lambda item: (item["span"]["start"], item["span"]["end"]))
    beams: List[Dict[str, Any]] = [
        {
            "rawScore": 0.0,
            "selectedOptions": [],
            "claimedRanges": [],
        }
    ]

    for group in ordered_groups:
        span = group["span"]
        group_options = [item for item in (group.get("proposalOptions") or []) if item.get("decision") != "REJECT"]
        if not group_options:
            group_options = [
                {
                    "span": span,
                    "original": group["original"],
                    "replacement": group["original"],
                    "source": "ORIGINAL",
                    "candidateScore": 0.2,
                    "optionScore": 0.2,
                    "guardrail": {"decision": "PASS", "reasonCodes": ["KEEP_ORIGINAL"], "score": 0.2},
                    "decision": "KEEP",
                    "applyEdit": False,
                }
            ]

        next_beams: List[Dict[str, Any]] = []
        for beam in beams:
            overlaps_existing = any(
                range_overlaps(span, claimed)
                for claimed in beam.get("claimedRanges") or []
            )
            if overlaps_existing:
                next_beams.append(beam)
                continue

            for option in group_options:
                edit_count = len([item for item in beam["selectedOptions"] if item.get("applyEdit")])
                incremental_penalty = 0.04 if option.get("applyEdit") and edit_count > 0 else 0.0
                next_beams.append(
                    {
                        "rawScore": round(float(beam["rawScore"] + option["optionScore"] - incremental_penalty), 4),
                        "selectedOptions": [*beam["selectedOptions"], option],
                        "claimedRanges": [*beam["claimedRanges"], span],
                    }
                )

        dedup: Dict[Tuple[Tuple[int, str], ...], Dict[str, Any]] = {}
        for beam in next_beams:
            key = tuple(
                sorted(
                    (
                        int(item["span"]["start"]),
                        str(item["replacement"]),
                    )
                    for item in beam["selectedOptions"]
                )
            )
            existing = dedup.get(key)
            if existing is None or float(beam["rawScore"]) > float(existing["rawScore"]):
                dedup[key] = beam

        beams = sorted(
            dedup.values(),
            key=lambda item: (
                item["rawScore"],
                -len([opt for opt in item["selectedOptions"] if opt.get("applyEdit")]),
            ),
            reverse=True,
        )[:beam_width]

    baseline_score = 0.2 * len(ordered_groups)
    ceiling_score = 0.99 * len(ordered_groups)
    proposals: List[Dict[str, Any]] = []
    for beam in beams:
        selected_options = sorted(
            beam["selectedOptions"],
            key=lambda item: (item["span"]["start"], item["span"]["end"]),
        )
        suggestion_edits = [
            edit
            for option in selected_options
            if (edit := _proposal_edit_from_option(option)) is not None
        ]
        auto_edits = [
            edit
            for option in selected_options
            if option.get("decision") == "AUTO_APPLY"
            and (edit := _proposal_edit_from_option(option)) is not None
        ]
        suggested_clause = apply_edits(sentence, suggestion_edits)
        final_clause = apply_edits(sentence, auto_edits)
        normalized_score = 0.15
        if ceiling_score > baseline_score:
            normalized_score = 0.15 + (
                0.84 * max(0.0, min(1.0, (beam["rawScore"] - baseline_score) / (ceiling_score - baseline_score)))
            )
        proposals.append(
            {
                "text": suggested_clause,
                "finalClause": final_clause,
                "suggestedClause": suggested_clause,
                "score": round(float(min(0.99, normalized_score)), 4),
                "rawScore": round(float(beam["rawScore"]), 4),
                "editCount": len(suggestion_edits),
                "selectedOptions": selected_options,
                "edits": suggestion_edits,
            }
        )

    deduped_proposals: Dict[str, Dict[str, Any]] = {}
    for proposal in proposals:
        key = proposal["text"]
        existing = deduped_proposals.get(key)
        if existing is None or float(proposal["rawScore"]) > float(existing["rawScore"]):
            deduped_proposals[key] = proposal

    ranked_proposals = sorted(
        deduped_proposals.values(),
        key=lambda item: (
            item["rawScore"],
            -item["editCount"],
        ),
        reverse=True,
    )
    return ranked_proposals[:top_k]


# ── Seq2Seq Proposals ────────────────────────────────────────


def _seq2seq_sentence_proposals(
    models: ModelLoader,
    sentence: str,
    protected: List[Dict[str, Any]],
    tag_labels: Optional[List[Dict[str, Any]]] = None,
    top_k: int = PROPOSAL_TOP_K,
) -> List[Dict[str, Any]]:
    if models.proposal_seq2seq_model is None or models.proposal_seq2seq_tokenizer is None:
        return []
    dedup: Dict[str, Dict[str, Any]] = {}
    windows = _tagger_open_replace_windows(sentence, tag_labels or [])
    for window_index, window in enumerate(windows):
        window_text = str(window["windowText"])
        inputs = models.proposal_seq2seq_tokenizer(
            window_text,
            return_tensors="pt",
            max_length=min(160, max(48, len(window_text) + 24)),
            truncation=True,
        )
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]

        with torch.no_grad():
            output = models.proposal_seq2seq_model.generate(
                **inputs,
                max_length=min(160, max(48, len(window_text) + 24)),
                num_beams=max(4, top_k + 1),
                num_return_sequences=max(4, top_k + 1),
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                output_scores=True,
                return_dict_in_generate=True,
            )

        sequence_scores = getattr(output, "sequences_scores", None)
        window_protected = _project_protected_spans(
            protected,
            int(window["window"]["start"]),
            int(window["window"]["end"]),
        )

        for index, sequence in enumerate(output.sequences):
            generated_window = models.proposal_seq2seq_tokenizer.decode(sequence, skip_special_tokens=True).strip()
            if not generated_window or generated_window == window_text:
                continue
            if _has_suspicious_generation_artifact(generated_window):
                continue
            if not _protected_texts_preserved(window_text, generated_window, window_protected):
                continue
            change = _window_change_summary(window_text, generated_window)
            if change is None:
                continue
            if not _change_overlaps_local_span(change, window["localSpan"]):
                continue

            generated = (
                sentence[: int(window["window"]["start"])]
                + generated_window
                + sentence[int(window["window"]["end"]) :]
            )
            if generated == sentence:
                continue
            if _has_suspicious_generation_artifact(generated):
                continue
            if not _protected_texts_preserved(sentence, generated, protected):
                continue
            edit_ratio = _sentence_edit_ratio(sentence, generated)
            if edit_ratio > 0.35:
                continue
            tsim = typo_similarity(sentence, generated)
            if tsim < 0.55:
                continue
            raw_score = float(sequence_scores[index]) if sequence_scores is not None else -1.5
            normalized_generator = 1.0 / (1.0 + math.exp(-raw_score))
            score = 0.32 + (0.36 * normalized_generator) + (0.26 * tsim) - (0.2 * edit_ratio)
            payload = {
                "text": generated,
                "finalClause": sentence,
                "suggestedClause": generated,
                "score": round(float(min(0.97, max(0.2, score))), 4),
                "rawScore": round(float(score), 4),
                "editCount": 1,
                "selectedOptions": [],
                "edits": [],
                "source": "SEQ2SEQ_SPAN_CONTEXT",
                "sequenceScore": round(float(raw_score), 4),
                "editRatio": round(float(edit_ratio), 4),
                "typoSimilarity": round(float(tsim), 4),
                "windowIndex": window_index,
                "windowRange": window["window"],
                "spanRange": window["span"],
            }
            existing = dedup.get(generated)
            if existing is None or float(payload["rawScore"]) > float(existing["rawScore"]):
                dedup[generated] = payload

    proposals = sorted(
        dedup.values(),
        key=lambda item: (item["rawScore"], item["score"]),
        reverse=True,
    )
    return proposals[:top_k]


# ── Merge ─────────────────────────────────────────────────────


def _merge_sentence_proposals(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
    top_k: int = PROPOSAL_TOP_K,
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for proposal in [*primary, *secondary]:
        key = str(proposal.get("suggestedClause") or proposal.get("text") or "")
        if not key:
            continue
        existing = merged.get(key)
        if existing is None or float(proposal.get("rawScore", 0.0)) > float(existing.get("rawScore", 0.0)):
            merged[key] = proposal
    ranked = sorted(
        merged.values(),
        key=lambda item: (float(item.get("rawScore", 0.0)), -int(item.get("editCount", 0))),
        reverse=True,
    )
    return ranked[:top_k]
