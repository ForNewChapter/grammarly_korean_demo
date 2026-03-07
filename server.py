import os
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from kiwipiepy import Kiwi
from kobert_transformers import get_kobert_model, get_tokenizer
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BASE_DIR = Path(__file__).resolve().parent
LOCAL_EDIT_TAGGER_DIR = BASE_DIR / "models" / "edit_tagger_v1_best" / "best"

RULE_MISSPELLINGS = [
    {"from": "되요", "to": "돼요", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "왠", "to": "웬", "reasonTag": "common_misspelling", "confidence": 0.97},
    {"from": "삿어요", "to": "샀어요", "reasonTag": "common_misspelling", "confidence": 0.99},
    {"from": "잇어요", "to": "있어요", "reasonTag": "common_misspelling", "confidence": 0.96},
]

OPEN_REPLACE_MAP = {
    "낫다": [
        {"replacement": "낳다", "source": "CONFUSION_SET", "generatorScore": 0.94},
        {"replacement": "놓다", "source": "MORPH", "generatorScore": 0.41},
    ],
    "안되": [
        {"replacement": "안 돼", "source": "CONFUSION_SET", "generatorScore": 0.88},
        {"replacement": "안돼", "source": "MORPH", "generatorScore": 0.35},
    ],
}

STEM_CONFUSION_RULES = [
    {"wrong_stem": "낫", "correct_stem": "낳", "source": "MORPH", "baseScore": 0.82},
]

MORPHEME_CONFUSION_RULES = {
    ("겟", "EP"): [{"replacement": "겠", "source": "MORPHEME_CONFUSION", "generatorScore": 0.93}],
    ("엇", "EP"): [{"replacement": "었", "source": "MORPHEME_CONFUSION", "generatorScore": 0.91}],
    ("앗", "EP"): [{"replacement": "았", "source": "MORPHEME_CONFUSION", "generatorScore": 0.91}],
}

TYPO_SIMILARITY_MIN = 0.55
ROUTING_PROMOTION_MIN_SCORE = 0.55
ROUTING_PROMOTION_LABELS = {"PUNCT_FIX", "JOSA_FIX", "EOMI_FIX"}
CURATED_CANDIDATE_SOURCES = {"CONFUSION_SET", "MORPH"}
MLM_SURFACE_TOP_K = 8
CANDIDATE_PREFILTER_LIMIT = 4
CHEAP_VERIFIER_LIMIT = 3
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")

DOMAIN_ENTITIES = ["AirPods Pro 2", "RTX 4060", "ChatGPT"]
USER_DICT = ["온디바이스", "맞춤법", "교정기"]

PROFILE_PROTOTYPES = {
    "NORMAL": ["오늘 날씨가 정말 좋네요.", "회의 자료를 지금 확인해 주세요."],
    "CHAT": ["오늘 개좋다 ㅋㅋ", "이거 진짜 맞냐ㅠㅠ"],
    "NOISY": ["오늘날씨가너무좋은데산책갈까", "아기를낫다"],
    "MIXED": ["RTX 4060 사면됨?", "AirPods Pro 2를 샀어요"],
    "QUERY": ["맞춤법 검사", "교정기 데모"],
}

PROTECTED_PATTERNS = [
    ("URL", re.compile(r"https?://[^\s]+")),
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"\b\d{2,4}-\d{3,4}-\d{4}\b")),
    ("HASHTAG", re.compile(r"#[\w가-힣]+")),
    ("MENTION", re.compile(r"@[\w가-힣._-]+")),
    ("NUMBER", re.compile(r"\b\d+(?:[.,]\d+)?\b")),
]


class CorrectRequest(BaseModel):
    text: str = Field(min_length=0, max_length=5000)
    cursor: Optional[int] = None
    mode: str = "realtime"


class CorrectionEngine:
    def __init__(self) -> None:
        torch.set_num_threads(1)
        self.kiwi = Kiwi(typos="basic")
        self.tokenizer = get_tokenizer()
        self.kobert = get_kobert_model().eval()
        self.kobert_mlm = AutoModelForMaskedLM.from_pretrained("monologg/kobert-lm").eval()
        self.koelectra_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.koelectra = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator").eval()
        self.edit_tagger_model = None
        self.edit_tagger_tokenizer = None
        self.edit_tagger_id2label: Dict[int, str] = {}
        self.edit_tagger_source = "heuristic"
        self.lemma_confusion_map = self._build_lemma_confusion_map()
        if LOCAL_EDIT_TAGGER_DIR.exists():
            self.edit_tagger_tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_EDIT_TAGGER_DIR), use_fast=True)
            self.edit_tagger_model = AutoModelForTokenClassification.from_pretrained(str(LOCAL_EDIT_TAGGER_DIR)).eval()
            raw_id2label = getattr(self.edit_tagger_model.config, "id2label", {}) or {}
            self.edit_tagger_id2label = {
                int(key): value for key, value in raw_id2label.items()
            }
            self.edit_tagger_source = str(LOCAL_EDIT_TAGGER_DIR)
        self.profile_proto_emb = self._build_profile_prototypes(self._sentence_embedding)
        self.profile_proto_emb_electra = self._build_profile_prototypes(
            self._sentence_embedding_koelectra
        )
        self._warmup_runtime()

    def _build_lemma_confusion_map(self) -> Dict[str, List[Dict[str, Any]]]:
        graph: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def add_edge(left: str, right: str, source: str, score: float) -> None:
            graph.setdefault(left, {})
            existing = graph[left].get(right)
            payload = {"replacement": right, "source": source, "generatorScore": score}
            if existing is None or payload["generatorScore"] > existing["generatorScore"]:
                graph[left][right] = payload

        for surface, replacements in OPEN_REPLACE_MAP.items():
            if not surface.endswith("다"):
                continue
            for replacement in replacements:
                target = replacement["replacement"]
                if not target.endswith("다"):
                    continue
                add_edge(surface, target, replacement["source"], replacement["generatorScore"])
                add_edge(target, surface, replacement["source"], replacement["generatorScore"] * 0.9)

        for rule in STEM_CONFUSION_RULES:
            wrong = f"{rule['wrong_stem']}다"
            correct = f"{rule['correct_stem']}다"
            add_edge(wrong, correct, "LEMMA_CONFUSION", max(0.88, rule["baseScore"]))
            add_edge(correct, wrong, "LEMMA_CONFUSION", max(0.8, rule["baseScore"] - 0.05))

        return {lemma: list(targets.values()) for lemma, targets in graph.items()}

    def _warmup_runtime(self) -> None:
        warm_sentence = "오늘 날씨가 좋다"
        try:
            self.kiwi.space(warm_sentence)
            self.kiwi.analyze(warm_sentence, top_n=1)
        except Exception:
            pass

        try:
            inputs = self.tokenizer("[MASK]", return_tensors="pt", truncation=True, max_length=16)
            with torch.no_grad():
                self.kobert_mlm(**inputs)
        except Exception:
            pass

        if self.edit_tagger_model is not None and self.edit_tagger_tokenizer is not None:
            try:
                inputs = self.edit_tagger_tokenizer(
                    ["테스트"],
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=16,
                )
                with torch.no_grad():
                    self.edit_tagger_model(**inputs)
            except Exception:
                pass

    def _build_profile_prototypes(self, emb_fn) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for profile, samples in PROFILE_PROTOTYPES.items():
            emb = torch.stack([emb_fn(s) for s in samples], dim=0).mean(dim=0)
            out[profile] = F.normalize(emb, dim=0)
        return out

    def _sentence_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )
        with torch.no_grad():
            vec = self.kobert(**inputs).last_hidden_state[:, 0, :].squeeze(0)
        return F.normalize(vec, dim=0)

    def _sentence_embedding_koelectra(self, text: str) -> torch.Tensor:
        inputs = self.koelectra_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )
        with torch.no_grad():
            vec = self.koelectra(**inputs).last_hidden_state[:, 0, :].squeeze(0)
        return F.normalize(vec, dim=0)

    def _token_logit_in_mask(self, masked_sentence: str, token: str) -> float:
        inputs = self.tokenizer(
            masked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=96,
        )
        input_ids = inputs["input_ids"][0]
        mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            return -20.0
        mask_idx = int(mask_positions[0].item())
        with torch.no_grad():
            logits = self.kobert_mlm(**inputs).logits[0, mask_idx]
        token_ids = self.tokenizer.encode(token, add_special_tokens=False)
        if not token_ids:
            return -20.0
        valid_ids = [tid for tid in token_ids[:3] if tid < logits.shape[0]]
        if not valid_ids:
            return -20.0
        return float(torch.mean(logits[valid_ids]).detach().item())

    @staticmethod
    def _char_edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)

        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            curr = [i]
            for j, cb in enumerate(b, start=1):
                cost = 0 if ca == cb else 1
                curr.append(
                    min(
                        prev[j] + 1,
                        curr[j - 1] + 1,
                        prev[j - 1] + cost,
                    )
                )
            prev = curr
        return prev[-1]

    @staticmethod
    def _hangul_ratio(text: str) -> float:
        if not text:
            return 0.0
        return len(re.findall(r"[가-힣]", text)) / len(text)

    @staticmethod
    def _hangul_syllable_parts(char: str) -> Optional[Tuple[int, int, int]]:
        if not char or len(char) != 1:
            return None
        code = ord(char)
        if code < 0xAC00 or code > 0xD7A3:
            return None
        offset = code - 0xAC00
        return (
            offset // 588,
            (offset % 588) // 28,
            offset % 28,
        )

    def _char_similarity(self, a: str, b: str) -> float:
        if a == b:
            return 1.0
        parts_a = self._hangul_syllable_parts(a)
        parts_b = self._hangul_syllable_parts(b)
        if parts_a and parts_b:
            matches = sum(1 for left, right in zip(parts_a, parts_b) if left == right)
            return matches / 3.0
        return 0.0

    def _typo_similarity(self, original: str, candidate: str) -> float:
        if not original or not candidate:
            return 0.0
        if original == candidate:
            return 1.0

        prefix = 0
        while prefix < min(len(original), len(candidate)) and original[prefix] == candidate[prefix]:
            prefix += 1

        suffix = 0
        while (
            suffix < min(len(original) - prefix, len(candidate) - prefix)
            and original[len(original) - 1 - suffix] == candidate[len(candidate) - 1 - suffix]
        ):
            suffix += 1

        preserved = (prefix + suffix) / max(len(original), len(candidate))
        core_original = original[prefix : len(original) - suffix if suffix else len(original)]
        core_candidate = candidate[prefix : len(candidate) - suffix if suffix else len(candidate)]

        if not core_original or not core_candidate:
            core_similarity = 0.0
        else:
            pair_count = max(len(core_original), len(core_candidate))
            pair_scores = []
            for idx in range(pair_count):
                left = core_original[idx] if idx < len(core_original) else ""
                right = core_candidate[idx] if idx < len(core_candidate) else ""
                if left and right:
                    pair_scores.append(self._char_similarity(left, right))
                else:
                    pair_scores.append(0.0)
            core_similarity = sum(pair_scores) / pair_count

        return round(float((0.55 * preserved) + (0.45 * core_similarity)), 4)

    def _kiwi_sentence_score(self, text: str) -> float:
        try:
            analyzed = self.kiwi.analyze(text, top_n=1)
        except Exception:
            return -999.0
        if not analyzed:
            return -999.0
        return float(analyzed[0][1])

    def _pseudo_logprob_for_replacement(
        self,
        sentence: str,
        start: int,
        end: int,
        replacement: str,
    ) -> float:
        prefix_ids = self.tokenizer.encode(sentence[:start], add_special_tokens=False)
        replacement_ids = self.tokenizer.encode(replacement, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(sentence[end:], add_special_tokens=False)
        if not replacement_ids:
            return -999.0

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        mask_id = self.tokenizer.mask_token_id
        full_ids = [cls_id, *prefix_ids, *replacement_ids, *suffix_ids, sep_id]
        rep_start = 1 + len(prefix_ids)

        total = 0.0
        with torch.no_grad():
            for offset, token_id in enumerate(replacement_ids):
                masked_ids = full_ids.copy()
                masked_ids[rep_start + offset] = mask_id
                inputs = {
                    "input_ids": torch.tensor([masked_ids], dtype=torch.long),
                    "attention_mask": torch.ones((1, len(masked_ids)), dtype=torch.long),
                    "token_type_ids": torch.zeros((1, len(masked_ids)), dtype=torch.long),
                }
                logits = self.kobert_mlm(**inputs).logits[0, rep_start + offset]
                total += float(logits[token_id].detach().item())
        return total / len(replacement_ids)

    def _is_valid_surface_candidate(self, original: str, candidate: str) -> bool:
        if not candidate or candidate == original:
            return False
        if " " in candidate or "\n" in candidate or "\t" in candidate:
            return False
        if candidate.startswith("[") or candidate.startswith("#"):
            return False
        if self._hangul_ratio(candidate) < 0.5:
            return False
        if abs(len(candidate) - len(original)) > 1:
            return False
        max_edits = 1 if len(original) <= 4 else 2
        if self._char_edit_distance(original, candidate) > max_edits:
            return False
        return True

    def _mlm_surface_candidates(
        self,
        sentence: str,
        token: str,
        start: int,
        end: int,
        top_k: int = MLM_SURFACE_TOP_K,
    ) -> List[Dict[str, Any]]:
        original_pieces = self.tokenizer.tokenize(token)
        if not original_pieces or len(original_pieces) > 6:
            return []

        masked_text = sentence[:start] + " " + " ".join([self.tokenizer.mask_token] * len(original_pieces)) + " " + sentence[end:]
        inputs = self.tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=96)
        mask_positions = (
            (inputs["input_ids"][0] == self.tokenizer.mask_token_id)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
        )
        if len(mask_positions) != len(original_pieces):
            return []

        with torch.no_grad():
            logits = self.kobert_mlm(**inputs).logits[0]

        dedup: Dict[str, Dict[str, Any]] = {}
        for piece_idx, mask_pos in enumerate(mask_positions):
            values, indices = torch.topk(logits[mask_pos], top_k)
            for logit_value, token_id in zip(values.tolist(), indices.tolist()):
                predicted_piece = self.tokenizer.convert_ids_to_tokens([token_id])[0]
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
                    candidate = self.tokenizer.convert_tokens_to_string(mutated_pieces).strip()
                    if not self._is_valid_surface_candidate(token, candidate):
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

    @staticmethod
    def _candidate_source_prior(source: str) -> float:
        if source == "CONFUSION_SET":
            return 1.0
        if source == "MORPHEME_CONFUSION":
            return 0.98
        if source == "LEMMA_CONFUSION":
            return 0.95
        if source == "MORPH":
            return 0.9
        if source == "JAMO":
            return 0.7
        if source == "MLM_TOPK":
            return 0.35
        return 0.25

    @staticmethod
    def _is_curated_candidate_source(source: str) -> bool:
        return source in {
            "CONFUSION_SET",
            "MORPH",
            "LEMMA_CONFUSION",
            "MORPHEME_CONFUSION",
        }

    @staticmethod
    def _normalize_generator_score(score: float) -> float:
        if 0.0 <= score <= 1.0:
            return score
        return 1 / (1 + math.exp(-(score / 4.0)))

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        dedup: Dict[str, Dict[str, Any]] = {}
        for item in candidates:
            replacement = item["replacement"]
            existing = dedup.get(replacement)
            if existing is None:
                dedup[replacement] = item
                continue
            current_score = self._normalize_generator_score(float(item.get("generatorScore", 0.0)))
            existing_score = self._normalize_generator_score(float(existing.get("generatorScore", 0.0)))
            current_prior = self._candidate_source_prior(item.get("source", "MLM_TOPK"))
            existing_prior = self._candidate_source_prior(existing.get("source", "MLM_TOPK"))
            if (current_score, current_prior) > (existing_score, existing_prior):
                dedup[replacement] = item
        return list(dedup.values())

    def _prefilter_candidates(self, token: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        prefiltered: List[Dict[str, Any]] = []
        for item in self._dedupe_candidates(candidates):
            replacement = item["replacement"]
            typo_similarity = self._typo_similarity(token, replacement)
            if typo_similarity < TYPO_SIMILARITY_MIN:
                continue
            generator_norm = self._normalize_generator_score(float(item.get("generatorScore", 0.0)))
            source_prior = self._candidate_source_prior(item.get("source", "MLM_TOPK"))
            prefilter_score = round(
                float((0.5 * typo_similarity) + (0.35 * generator_norm) + (0.15 * source_prior)),
                4,
            )
            prefiltered.append(
                {
                    **item,
                    "candidateSentence": item.get("candidateSentence"),
                    "generatorNorm": round(generator_norm, 4),
                    "typoSimilarity": typo_similarity,
                    "prefilterScore": prefilter_score,
                }
            )
        prefiltered.sort(
            key=lambda x: (x["prefilterScore"], x["typoSimilarity"], x["generatorNorm"]),
            reverse=True,
        )
        return prefiltered[:CANDIDATE_PREFILTER_LIMIT]

    def _sentence_quality_baseline(self, sentence: str) -> Dict[str, float]:
        return {
            "koelectraScore": self._koelectra_normal_score(sentence),
            "kiwiScore": self._kiwi_sentence_score(sentence),
        }

    def _context_verifier_score(
        self,
        source: str,
        typo_similarity: float,
        electra_delta: float,
        kiwi_delta: float,
    ) -> float:
        source_prior = self._candidate_source_prior(source)
        verifier_signal = (electra_delta * 12.0) + (kiwi_delta / 2.5) + ((typo_similarity - 0.55) * 2.0)
        if self._is_curated_candidate_source(source):
            verifier_signal += 0.65
        return 1 / (1 + math.exp(-verifier_signal))

    def _rank_replacement_candidates(
        self,
        sentence: str,
        token: str,
        start: int,
        end: int,
        candidates: List[Dict[str, Any]],
        baseline: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        original_mlm = self._pseudo_logprob_for_replacement(sentence, start, end, token)
        original_normal = baseline["koelectraScore"] if baseline is not None else self._koelectra_normal_score(sentence)
        original_kiwi = baseline["kiwiScore"] if baseline is not None else self._kiwi_sentence_score(sentence)
        has_curated_anchor = any(
            self._is_curated_candidate_source(item.get("source", "MLM_TOPK")) for item in candidates
        )
        best_curated_prefilter = max(
            (
                float(item.get("prefilterScore", 0.0))
                for item in candidates
                if self._is_curated_candidate_source(item.get("source", "MLM_TOPK"))
            ),
            default=0.0,
        )

        ranked: List[Dict[str, Any]] = []
        for item in candidates:
            replacement = item["replacement"]
            candidate_sentence = sentence[:start] + replacement + sentence[end:]
            mlm_score = self._pseudo_logprob_for_replacement(sentence, start, end, replacement)
            electra_score = self._koelectra_normal_score(candidate_sentence)
            kiwi_score = self._kiwi_sentence_score(candidate_sentence)

            mlm_delta = mlm_score - original_mlm
            electra_delta = electra_score - original_normal
            kiwi_delta = kiwi_score - original_kiwi

            mlm_gain = 1 / (1 + math.exp(-mlm_delta))
            electra_gain = 1 / (1 + math.exp(-(electra_delta * 8.0)))
            kiwi_gain = 1 / (1 + math.exp(-(kiwi_delta / 3.0)))
            source = item.get("source", "MLM_TOPK")
            source_prior = self._candidate_source_prior(source)
            edit_penalty = min(0.18, 0.06 * self._char_edit_distance(token, replacement))
            typo_similarity = self._typo_similarity(token, replacement)
            typo_bonus = 0.22 * typo_similarity
            source_bonus = 0.14 * source_prior
            prefilter_bonus = 0.18 * item.get("prefilterScore", 0.0)
            semantic_penalty = 0.24 if typo_similarity < 0.6 else 0.0
            context_verifier = self._context_verifier_score(
                source=source,
                typo_similarity=typo_similarity,
                electra_delta=electra_delta,
                kiwi_delta=kiwi_delta,
            )
            arbitration_penalty = 0.0
            curated_bonus = 0.0
            if self._is_curated_candidate_source(source):
                if kiwi_delta > 0:
                    curated_bonus += 0.12
                if electra_delta > -0.03:
                    curated_bonus += 0.04
            elif has_curated_anchor:
                arbitration_penalty += 0.1
                if kiwi_delta <= 0:
                    arbitration_penalty += 0.08
                if electra_delta <= 0:
                    arbitration_penalty += 0.05
                if item.get("prefilterScore", 0.0) <= best_curated_prefilter + 0.03:
                    arbitration_penalty += 0.08

            final_score = (
                0.18 * mlm_gain
                + 0.16 * electra_gain
                + 0.08 * kiwi_gain
                + 0.24 * context_verifier
                + 0.1 * (1 / (1 + math.exp(-item["generatorScore"])))
                + typo_bonus
                + source_bonus
                + (0.14 * item.get("prefilterScore", 0.0))
                + curated_bonus
                - edit_penalty
                - semantic_penalty
                - arbitration_penalty
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
                    "typoSimilarity": typo_similarity,
                    "contextVerifierScore": round(float(context_verifier), 4),
                    "curatedBonus": round(float(curated_bonus), 4),
                    "arbitrationPenalty": round(float(arbitration_penalty), 4),
                    "finalScore": round(float(max(0.01, min(0.99, final_score))), 4),
                }
            )

        ranked.sort(key=lambda x: x["finalScore"], reverse=True)
        return ranked

    def extract_input_range(self, full_text: str, cursor: int) -> Dict[str, Any]:
        safe_cursor = max(0, min(cursor, len(full_text)))
        text_before_cursor = full_text[:safe_cursor]
        separators = ["\n", ".", "?", "!", ",", ";"]
        if safe_cursor > 0 and full_text[safe_cursor - 1] in separators:
            text_before_cursor = full_text[: safe_cursor - 1]
        last_separator_index = max(text_before_cursor.rfind(s) for s in separators)
        start = max(0, last_separator_index + 1)
        end = safe_cursor
        raw = full_text[start:end]
        clause_text = raw[-128:]
        clause_start = end - len(clause_text)
        return {
            "clauseText": clause_text,
            "clauseRange": {"start": clause_start, "end": end},
        }

    def detect_protected_spans(self, text: str) -> List[Dict[str, Any]]:
        spans: List[Dict[str, Any]] = []
        for kind, regex in PROTECTED_PATTERNS:
            for m in regex.finditer(text):
                spans.append(
                    {
                        "range": {"start": m.start(), "end": m.end()},
                        "kind": kind,
                        "text": m.group(0),
                    }
                )
        for term in [*DOMAIN_ENTITIES, *USER_DICT]:
            at = 0
            while at < len(text):
                idx = text.find(term, at)
                if idx < 0:
                    break
                spans.append(
                    {
                        "range": {"start": idx, "end": idx + len(term)},
                        "kind": "ENTITY" if term in DOMAIN_ENTITIES else "USER_DICT",
                        "text": term,
                    }
                )
                at = idx + len(term)

        spans.sort(key=lambda s: (s["range"]["start"], s["range"]["end"]))
        merged: List[Dict[str, Any]] = []
        for s in spans:
            if not merged:
                merged.append(s)
                continue
            last = merged[-1]
            if s["range"]["start"] >= last["range"]["end"]:
                merged.append(s)
                continue
            if s["range"]["end"] > last["range"]["end"]:
                last["range"]["end"] = s["range"]["end"]
                last["text"] = text[last["range"]["start"] : last["range"]["end"]]

        for i, s in enumerate(merged):
            s["placeholder"] = f"__{s['kind']}_{i}__"
        return merged

    @staticmethod
    def apply_mask(text: str, protected: List[Dict[str, Any]]) -> str:
        if not protected:
            return text
        out = []
        cursor = 0
        for s in protected:
            st, ed = s["range"]["start"], s["range"]["end"]
            out.append(text[cursor:st])
            out.append(s["placeholder"])
            cursor = ed
        out.append(text[cursor:])
        return "".join(out)

    @staticmethod
    def is_range_protected(span: Dict[str, int], protected: List[Dict[str, Any]]) -> bool:
        return any(
            span["start"] < p["range"]["end"] and span["end"] > p["range"]["start"] for p in protected
        )

    def classify_profile(self, text: str) -> Dict[str, Any]:
        length = max(len(text), 1)
        hangul_ratio = len(re.findall(r"[가-힣]", text)) / length
        latin_ratio = len(re.findall(r"[A-Za-z]", text)) / length
        num_ratio = len(re.findall(r"\d", text)) / length
        has_chat = bool(re.search(r"[ㅋㅎㅠ]{2,}|[!?]{2,}", text))

        if latin_ratio > 0.3 or num_ratio > 0.25:
            rule_profile = "MIXED"
        elif has_chat:
            rule_profile = "CHAT"
        elif hangul_ratio < 0.35:
            rule_profile = "QUERY"
        elif " " not in text and len(text) > 12:
            rule_profile = "NOISY"
        else:
            rule_profile = "NORMAL"

        emb = self._sentence_embedding(text)
        emb_electra = self._sentence_embedding_koelectra(text)
        model_scores_kobert = {
            p: float(torch.dot(emb, proto).detach().item()) for p, proto in self.profile_proto_emb.items()
        }
        model_scores_koelectra = {
            p: float(torch.dot(emb_electra, proto).detach().item())
            for p, proto in self.profile_proto_emb_electra.items()
        }
        model_scores = {
            p: round((model_scores_kobert[p] + model_scores_koelectra[p]) / 2.0, 6)
            for p in PROFILE_PROTOTYPES.keys()
        }
        model_profile = max(model_scores, key=model_scores.get)
        if model_profile != rule_profile and model_scores[model_profile] - model_scores.get(rule_profile, 0) > 0.05:
            final = model_profile
        else:
            final = rule_profile
        return {
            "profile": final,
            "ruleProfile": rule_profile,
            "modelProfile": model_profile,
            "kobertScores": model_scores_kobert,
            "koelectraScores": model_scores_koelectra,
            "blendedScores": model_scores,
        }

    def _koelectra_normal_score(self, text: str) -> float:
        emb = self._sentence_embedding_koelectra(text)
        return float(torch.dot(emb, self.profile_proto_emb_electra["NORMAL"]).detach().item())

    def create_edit(
        self,
        stage: str,
        start: int,
        end: int,
        source_text: str,
        replacement: str,
        edit_type: str,
        confidence: float,
        auto_applicable: bool,
        reason_tag: str,
    ) -> Dict[str, Any]:
        return {
            "stage": stage,
            "range": {"start": start, "end": end},
            "sourceText": source_text,
            "replacement": replacement,
            "editType": edit_type,
            "confidence": round(float(confidence), 4),
            "autoApplicable": auto_applicable,
            "reasonTag": reason_tag,
        }

    @staticmethod
    def apply_edits(text: str, edits: List[Dict[str, Any]]) -> str:
        if not edits:
            return text
        sorted_edits = sorted(edits, key=lambda e: (e["range"]["start"], e["range"]["end"]))
        out: List[str] = []
        cursor = 0
        for e in sorted_edits:
            st, ed = e["range"]["start"], e["range"]["end"]
            out.append(text[cursor:st])
            out.append(e["replacement"])
            cursor = ed
        out.append(text[cursor:])
        return "".join(out)

    def run_rule_corrector(self, text: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        edits: List[Dict[str, Any]] = []
        for rule in RULE_MISSPELLINGS:
            at = 0
            src = rule["from"]
            while at < len(text):
                idx = text.find(src, at)
                if idx < 0:
                    break
                span = {"start": idx, "end": idx + len(src)}
                if not self.is_range_protected(span, protected):
                    edits.append(
                        self.create_edit(
                            stage="RULE",
                            start=idx,
                            end=idx + len(src),
                            source_text=src,
                            replacement=rule["to"],
                            edit_type="SPELL",
                            confidence=rule["confidence"],
                            auto_applicable=True,
                            reason_tag=rule["reasonTag"],
                        )
                    )
                at = idx + len(src)
        return self.dedupe_edits(edits)

    @staticmethod
    def dedupe_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for e in edits:
            key = (e["range"]["start"], e["range"]["end"], e["replacement"])
            if key in seen:
                continue
            seen.add(key)
            out.append(e)
        return out

    def spacing_candidates_from_kiwi(
        self, text: str, protected: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        spaced = self.kiwi.space(text)
        candidates: List[Dict[str, Any]] = []
        i, j = 0, 0
        while i < len(text) and j < len(spaced):
            if text[i] == spaced[j]:
                i += 1
                j += 1
                continue
            if spaced[j] == " ":
                probe = {"start": max(0, i - 1), "end": min(len(text), i + 1)}
                if not self.is_range_protected(probe, protected):
                    candidates.append({"index": i, "action": "INSERT_SPACE", "score": 0.96})
                j += 1
                continue
            if text[i] == " ":
                probe = {"start": i, "end": i + 1}
                left_probe = {"start": max(0, i - 1), "end": i + 1}
                if not self.is_range_protected(probe, protected) and not self.is_range_protected(
                    left_probe, protected
                ):
                    # 보수적으로 운영: 기존 공백 삭제는 과교정 위험이 커서 제안에서 제외한다.
                    pass
                i += 1
                continue
            i += 1
            j += 1
        return {"spacedText": spaced, "candidates": candidates}

    @staticmethod
    def classify_spacing_boundaries(candidates: List[Dict[str, Any]], profile: str) -> List[Dict[str, Any]]:
        out = []
        bonus = 0.03 if profile == "NORMAL" else 0.0
        for c in candidates:
            score = min(0.99, c["score"] + bonus)
            if score >= 0.9:
                out.append({**c, "score": score})
        return out

    def spacing_to_edits(self, boundaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        edits: List[Dict[str, Any]] = []
        for b in boundaries:
            if b["action"] == "INSERT_SPACE":
                edits.append(
                    self.create_edit(
                        stage="SPACING",
                        start=b["index"],
                        end=b["index"],
                        source_text="",
                        replacement=" ",
                        edit_type="SPACE_INSERT",
                        confidence=b["score"],
                        auto_applicable=True,
                        reason_tag="kiwi_spacing",
                    )
                )
            else:
                edits.append(
                    self.create_edit(
                        stage="SPACING",
                        start=b["index"],
                        end=b["index"] + 1,
                        source_text=" ",
                        replacement="",
                        edit_type="SPACE_DELETE",
                        confidence=b["score"],
                        auto_applicable=True,
                        reason_tag="kiwi_spacing",
                    )
                )
        return edits

    @staticmethod
    def tokenize_with_ranges(text: str) -> List[Tuple[str, int, int]]:
        out = []
        for m in TOKEN_PATTERN.finditer(text):
            out.append((m.group(0), m.start(), m.end()))
        return out

    def _token_morph_bundle(
        self, sentence: str, start: int, end: int
    ) -> Optional[Dict[str, Any]]:
        analyses = self.kiwi.analyze(sentence, top_n=1)
        if not analyses:
            return None
        morphs = analyses[0][0]
        overlapped = [m for m in morphs if not (m.end <= start or m.start >= end)]
        if not overlapped:
            return None

        predicate_index = None
        for idx, morph in enumerate(overlapped):
            if morph.tag.startswith(("VV", "VA")):
                predicate_index = idx
                break
        if predicate_index is None:
            return None

        predicate = overlapped[predicate_index]
        suffix_morphs = overlapped[predicate_index + 1 :]
        return {
            "predicate": predicate,
            "suffixMorphs": suffix_morphs,
            "morphs": overlapped,
        }

    def _infer_lemma_tag(self, lemma: str, fallback_tag: str) -> str:
        tag_base = fallback_tag.split("-")[0]
        try:
            analyses = self.kiwi.analyze(lemma, top_n=2)
        except Exception:
            return tag_base
        for tokens, _ in analyses:
            for token in tokens:
                if token.lemma == lemma and token.tag.startswith(("VV", "VA")):
                    return token.tag
        return tag_base

    def _build_join_candidates(
        self,
        sentence: str,
        token: str,
        start: int,
        end: int,
    ) -> List[Dict[str, Any]]:
        bundle = self._token_morph_bundle(sentence, start, end)
        if bundle is None:
            return []

        predicate = bundle["predicate"]
        predicate_lemma = predicate.lemma
        predicate_tag = predicate.tag
        predicate_stem = predicate_lemma[:-1] if predicate_lemma.endswith("다") else predicate.form
        suffix_morphs = bundle["suffixMorphs"]
        candidates: List[Dict[str, Any]] = []

        lemma_variants = self.lemma_confusion_map.get(predicate_lemma, [])
        for variant in lemma_variants:
            target_lemma = variant["replacement"]
            target_stem = target_lemma[:-1] if target_lemma.endswith("다") else target_lemma
            target_tag = self._infer_lemma_tag(target_lemma, predicate_tag)
            morph_seq = [(target_stem, target_tag), *[(m.form, m.tag) for m in suffix_morphs]]
            try:
                restored = self.kiwi.join(morph_seq)
            except Exception:
                continue
            if restored and restored != token:
                candidates.append(
                    {
                        "replacement": restored,
                        "source": "LEMMA_CONFUSION",
                        "generatorScore": variant["generatorScore"],
                    }
                )

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
                    restored = self.kiwi.join(morph_seq)
                except Exception:
                    continue
                if restored and restored != token:
                    candidates.append(
                        {
                            "replacement": restored,
                            "source": variant["source"],
                            "generatorScore": variant["generatorScore"],
                        }
                    )

        return candidates

    def open_replace_candidates_for_token(self, token: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = list(OPEN_REPLACE_MAP.get(token, []))

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
        self,
        sentence: str,
        token: str,
        start: int,
        end: int,
        expensive: bool = True,
    ) -> List[Dict[str, Any]]:
        curated_candidates = self.open_replace_candidates_for_token(token)
        join_candidates = self._build_join_candidates(sentence, token, start, end)
        mlm_candidates = self._mlm_surface_candidates(sentence, token, start, end)
        candidate_pool = self._prefilter_candidates(
            token,
            [*curated_candidates, *join_candidates, *mlm_candidates],
        )
        if not candidate_pool:
            return []
        if not expensive:
            return candidate_pool
        baseline = self._sentence_quality_baseline(sentence)
        return self._rank_replacement_candidates(
            sentence,
            token,
            start,
            end,
            candidate_pool,
            baseline=baseline,
        )[:CANDIDATE_PREFILTER_LIMIT]

    @staticmethod
    def _filter_typo_like_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            candidate for candidate in candidates if candidate.get("typoSimilarity", 0.0) >= TYPO_SIMILARITY_MIN
        ]

    def _run_heuristic_edit_tagger(self, text: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        labels: List[Dict[str, Any]] = []
        for token, st, ed in self.tokenize_with_ranges(text):
            span = {"start": st, "end": ed}
            if self.is_range_protected(span, protected):
                labels.append({"token": token, "label": "KEEP", "confidence": 1.0, "range": span})
                continue
            if self._hangul_ratio(token) < 0.5 or len(token) < 2:
                labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
                continue

            ranked_candidates = self.generate_contextual_candidates_for_token(text, token, st, ed, expensive=False)
            if not ranked_candidates:
                labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
                continue

            typo_like_candidates = self._filter_typo_like_candidates(ranked_candidates)
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

    def _run_finetuned_edit_tagger(self, text: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.edit_tagger_model is None or self.edit_tagger_tokenizer is None:
            return self._run_heuristic_edit_tagger(text, protected)

        token_ranges = self.tokenize_with_ranges(text)
        if not token_ranges:
            return []

        words = [token for token, _, _ in token_ranges]
        inputs = self.edit_tagger_tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )
        with torch.no_grad():
            logits = self.edit_tagger_model(**inputs).logits[0]
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
            if self.is_range_protected(span, protected):
                labels.append({"token": token, "label": "KEEP", "confidence": 1.0, "range": span})
                continue
            if self._hangul_ratio(token) < 0.5 or len(token) < 2:
                labels.append({"token": token, "label": "KEEP", "confidence": 0.98, "range": span})
                continue

            pred_id = int(torch.argmax(probabilities[piece_index]).item())
            pred_label = self.edit_tagger_id2label.get(pred_id, "KEEP")
            confidence = round(float(probabilities[piece_index][pred_id].item()), 4)

            if pred_label == "OPEN_REPLACE":
                ranked_candidates = self.generate_contextual_candidates_for_token(text, token, st, ed, expensive=False)
                typo_like_candidates = self._filter_typo_like_candidates(ranked_candidates)
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
                ranked_candidates = self.generate_contextual_candidates_for_token(text, token, st, ed, expensive=False)
                typo_like_candidates = self._filter_typo_like_candidates(ranked_candidates)
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

    def run_edit_tagger(self, text: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.edit_tagger_model is not None:
            return self._run_finetuned_edit_tagger(text, protected)
        return self._run_heuristic_edit_tagger(text, protected)

    def generate_open_candidates(self, sentence: str, tag_labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for l in tag_labels:
            if l.get("effectiveLabel", l["label"]) != "OPEN_REPLACE":
                continue
            base = l.get("modelCandidates") or self.generate_contextual_candidates_for_token(
                sentence,
                l["token"],
                l["range"]["start"],
                l["range"]["end"],
                expensive=False,
            )
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
                    "items": items,
                }
            )
        return out

    def verify_candidate_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verified: List[Dict[str, Any]] = []
        for group in groups:
            original = group["original"]
            candidate_items = [item for item in group["items"] if item["source"] != "ORIGINAL"]
            curated_items = [
                item for item in candidate_items if self._is_curated_candidate_source(item.get("source", "MLM_TOPK"))
            ]
            curated_best = max(
                (float(item.get("prefilterScore", 0.0)) for item in curated_items),
                default=0.0,
            )
            scored_items: List[Dict[str, Any]] = []
            for item in candidate_items:
                source = item.get("source", "MLM_TOPK")
                prefilter = float(item.get("prefilterScore", 0.0))
                typo_similarity = float(item.get("typoSimilarity", self._typo_similarity(original, item["replacement"])))
                generator_norm = float(
                    item.get(
                        "generatorNorm",
                        self._normalize_generator_score(float(item.get("generatorScore", 0.0))),
                    )
                )
                verifier_score = (
                    0.38 * prefilter
                    + 0.24 * typo_similarity
                    + 0.18 * generator_norm
                    + 0.2 * self._candidate_source_prior(source)
                )
                if self._is_curated_candidate_source(source):
                    verifier_score += 0.06
                elif curated_items:
                    verifier_score -= 0.1
                    if prefilter <= curated_best + 0.03:
                        verifier_score -= 0.08

                scored_items.append(
                    {
                        **item,
                        "candidateVerifierScore": round(float(verifier_score), 4),
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
                    item for item in scored_items if self._is_curated_candidate_source(item.get("source", "MLM_TOPK"))
                ][:2]
                mlm_ranked = [
                    item for item in scored_items if not self._is_curated_candidate_source(item.get("source", "MLM_TOPK"))
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
                    item.get("prefilterScore", 0.0),
                    item.get("typoSimilarity", 0.0),
                ),
                reverse=True,
            )

            verified.append(
                {
                    **group,
                    "items": [*kept, {"replacement": original, "source": "ORIGINAL", "generatorScore": 0.2}],
                    "candidateVerifier": {
                        "inputCount": len(candidate_items),
                        "keptCount": len(kept),
                        "curatedCount": len(curated_items),
                        "keptSources": [item.get("source", "MODEL") for item in kept],
                    },
                }
            )
        return verified

    @staticmethod
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

    def rerank(self, sentence: str, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not groups:
            return []
        ranked: List[Dict[str, Any]] = []
        baseline = self._sentence_quality_baseline(sentence)
        for g in groups:
            st, ed = g["span"]["start"], g["span"]["end"]
            model_seed_items = [item for item in g["items"] if item["source"] != "ORIGINAL"]
            if model_seed_items:
                rescored = self._rank_replacement_candidates(
                    sentence,
                    g["original"],
                    st,
                    ed,
                    model_seed_items,
                    baseline=baseline,
                )
                curated_rescored = [
                    item for item in rescored if self._is_curated_candidate_source(item.get("source", "MLM_TOPK"))
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

    def guardrail(
        self,
        original: str,
        replacement: str,
        span: Dict[str, int],
        profile: str,
        protected: List[Dict[str, Any]],
        score: float,
    ) -> Dict[str, Any]:
        reason_codes: List[str] = []
        if self.is_range_protected(span, protected):
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

    @staticmethod
    def policy(edit_type: str, guardrail_decision: str, confidence: float) -> str:
        if guardrail_decision == "REJECT":
            return "REJECT"
        if edit_type in {"SPACE_INSERT", "SPACE_DELETE", "SPELL"} and confidence >= 0.95:
            return "AUTO_APPLY"
        if edit_type == "OPEN_REPLACE":
            return "SUGGEST_ONLY"
        return guardrail_decision

    def run_pipeline(self, full_text: str, cursor: int, mode: str) -> Dict[str, Any]:
        traces: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []

        def stage(name: str, input_text: str, fn) -> Any:
            t0 = time.perf_counter()
            out = fn()
            latency = round((time.perf_counter() - t0) * 1000, 2)
            traces.append(
                {
                    "stageName": name,
                    "inputText": input_text,
                    "outputArtifacts": out,
                    "latencyMs": latency,
                }
            )
            return out

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
                else self.extract_input_range(full_text, cursor)
            ),
        )
        working_clause = range_art["clauseText"]

        protected_art = stage(
            "2. Protected Span Detector",
            working_clause,
            lambda: {
                "protected": self.detect_protected_spans(working_clause),
                "maskedText": self.apply_mask(working_clause, self.detect_protected_spans(working_clause)),
            },
        )
        protected = protected_art["protected"]

        profile_art = stage(
            "3. Profile Classifier (KoBERT+KoELECTRA+Rule)",
            protected_art["maskedText"],
            lambda: self.classify_profile(protected_art["maskedText"]),
        )
        profile = profile_art["profile"]

        rule_edits = stage("4. Rule Corrector", working_clause, lambda: self.run_rule_corrector(working_clause, protected))
        if rule_edits:
            working_clause = self.apply_edits(working_clause, rule_edits)
            for e in rule_edits:
                decision = self.policy(e["editType"], "AUTO_APPLY", e["confidence"])
                decisions.append({**e, "decision": decision})

        spacing_cand_art = stage(
            "5. Spacing Candidate Generator (Kiwi)",
            working_clause,
            lambda: self.spacing_candidates_from_kiwi(working_clause, self.detect_protected_spans(working_clause)),
        )
        spacing_boundaries = stage(
            "6. Spacing Boundary Classifier",
            working_clause,
            lambda: self.classify_spacing_boundaries(spacing_cand_art["candidates"], profile),
        )
        spacing_edits = stage("6-2. Spacing Edit Converter", working_clause, lambda: self.spacing_to_edits(spacing_boundaries))
        if spacing_edits:
            working_clause = self.apply_edits(working_clause, spacing_edits)
            for e in spacing_edits:
                decision = self.policy(e["editType"], "AUTO_APPLY", e["confidence"])
                decisions.append({**e, "decision": decision})

        tagger_labels = stage(
            "7. Edit Tagger (KoBERT-MLM + KoELECTRA-assisted)",
            working_clause,
            lambda: self.run_edit_tagger(working_clause, self.detect_protected_spans(working_clause)),
        )
        routed_tagger_labels = stage(
            "7-2. Edit Tagger Routing",
            working_clause,
            lambda: self.route_tagger_labels(tagger_labels),
        )
        open_candidates = stage(
            "8. Open Candidate Generation",
            working_clause,
            lambda: self.generate_open_candidates(working_clause, routed_tagger_labels),
        )
        verified_candidates = stage(
            "8-2. Candidate Verifier",
            working_clause,
            lambda: self.verify_candidate_groups(open_candidates),
        )
        reranked = stage(
            "9. Reranker (KoBERT-MLM + KoELECTRA)",
            working_clause,
            lambda: self.rerank(working_clause, verified_candidates),
        )

        guardrail_out = stage(
            "10. Guardrail",
            working_clause,
            lambda: [
                {
                    "span": g["span"],
                    "original": g["original"],
                    "replacement": g["best"]["replacement"],
                    "bestScore": g["best"]["finalScore"],
                    "guardrail": self.guardrail(
                        original=g["original"],
                        replacement=g["best"]["replacement"],
                        span=g["span"],
                        profile=profile,
                        protected=self.detect_protected_spans(working_clause),
                        score=g["best"]["finalScore"],
                    ),
                }
                for g in reranked
            ],
        )

        policy_out = stage(
            "11. Policy Engine",
            working_clause,
            lambda: [
                {
                    **g,
                    "decision": self.policy("OPEN_REPLACE", g["guardrail"]["decision"], g["bestScore"]),
                }
                for g in guardrail_out
            ],
        )

        for p in policy_out:
            edit = self.create_edit(
                stage="RERANKER",
                start=p["span"]["start"],
                end=p["span"]["end"],
                source_text=p["original"],
                replacement=p["replacement"],
                edit_type="OPEN_REPLACE",
                confidence=p["bestScore"],
                auto_applicable=p["decision"] == "AUTO_APPLY",
                reason_tag=",".join(p["guardrail"]["reasonCodes"]),
            )
            decisions.append({**edit, "decision": p["decision"]})
            if p["decision"] == "AUTO_APPLY":
                working_clause = self.apply_edits(working_clause, [edit])

        stage(
            "12. Final Output",
            working_clause,
            lambda: {
                "mode": mode,
                "finalClause": working_clause,
                "autoCount": len([d for d in decisions if d["decision"] == "AUTO_APPLY"]),
                "suggestCount": len([d for d in decisions if d["decision"] == "SUGGEST_ONLY"]),
                "rejectCount": len([d for d in decisions if d["decision"] == "REJECT"]),
            },
        )

        final_text = (
            full_text[: range_art["clauseRange"]["start"]]
            + working_clause
            + full_text[range_art["clauseRange"]["end"] :]
        )
        return {
            "finalText": final_text,
            "decisions": decisions,
            "traces": traces,
            "modelInfo": {
                "spacingEngine": "kiwipiepy",
                "profileModel": "KoBERT + KoELECTRA",
                "editTaggerModel": (
                    f"Fine-tuned KoELECTRA token-classifier ({self.edit_tagger_source})"
                    if self.edit_tagger_model is not None
                    else "KoBERT-MLM pseudo-likelihood + KoELECTRA + Kiwi"
                ),
                "candidateGeneratorModel": "lemma-aware generator + morpheme normalization + KoBERT-MLM backoff",
                "candidateVerifierModel": "source-aware verifier + typo similarity + generator prior",
                "rerankerModel": "KoBERT-MLM + KoELECTRA + Kiwi + context verifier",
                "kobert": "skt/kobert-base-v1",
                "koelectra": "monologg/koelectra-base-v3-discriminator",
                "kobertMlm": "monologg/kobert-lm",
            },
        }


app = FastAPI(title="Korean On-Device Grammarly Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE: Optional[CorrectionEngine] = None


@app.on_event("startup")
def startup() -> None:
    global ENGINE
    ENGINE = CorrectionEngine()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(BASE_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(BASE_DIR / "styles.css", media_type="text/css")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": ENGINE is not None}


@app.post("/api/correct")
def correct(req: CorrectRequest) -> Dict[str, Any]:
    if ENGINE is None:
        raise HTTPException(status_code=503, detail="Model engine is not ready.")
    text = req.text or ""
    cursor = len(text) if req.cursor is None else req.cursor
    return ENGINE.run_pipeline(full_text=text, cursor=cursor, mode=req.mode)
