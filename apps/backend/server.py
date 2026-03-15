import json
import logging
import os
import math
import re
import socket
import threading
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

IS_CLOUD_RUN = bool(os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_SERVICE") or os.getenv("K_REVISION"))
os.environ.setdefault("HF_HUB_OFFLINE", "0" if IS_CLOUD_RUN else "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0" if IS_CLOUD_RUN else "1")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]
LOGGER = logging.getLogger("grammarly_korean")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)
DEFAULT_BACKEND_NAME = os.getenv("BACKEND_NAME") or socket.gethostname()
DEFAULT_BACKEND_BRANCH = os.getenv("BACKEND_BRANCH")
DEFAULT_BACKEND_VERSION = os.getenv("BACKEND_VERSION")
LOCAL_EDIT_TAGGER_CANDIDATE_DIRS = [
    REPO_ROOT / "models" / "edit_tagger_v2_best" / "best",
    REPO_ROOT / "models" / "edit_tagger_v1_best" / "best",
]
TYPED_CONFUSION_GRAPH_PATH = REPO_ROOT / "public" / "assets" / "dict" / "typed_confusion_graph.json"
HIGH_PRECISION_SURFACE_FIXES_PATH = REPO_ROOT / "public" / "assets" / "rules" / "high_precision_surface_fixes.json"
PHRASE_MEMORY_PATH = REPO_ROOT / "public" / "assets" / "dict" / "phrase_memory.json"
PREDICATE_FAMILY_SEEDS_PATH = REPO_ROOT / "public" / "assets" / "dict" / "predicate_family_seeds.json"
INFLECTION_RECOVERY_RULES_PATH = REPO_ROOT / "public" / "assets" / "dict" / "inflection_recovery_rules.json"

RULE_MISSPELLINGS = [
    {"from": "되요", "to": "돼요", "reasonTag": "common_misspelling", "confidence": 0.99},
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
CURATED_CANDIDATE_SOURCES = {
    "CONFUSION_SET",
    "MORPH",
    "LEMMA_CONFUSION",
    "MORPHEME_CONFUSION",
    "DATA_CONFUSION",
    "LEMMA_DATA_CONFUSION",
    "FAMILY_SEED",
    "AUTO_FAMILY_SEED",
    "FAMILY_SURFACE_EXAMPLE",
    "PHRASE_MEMORY",
    "INFLECTION_RULE",
}
RUNTIME_BROAD_LEMMA_TARGETS = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
RUNTIME_BROAD_LEMMA_ALLOWLIST = {
    ("잇다", "있다"),
    ("허다", "하다"),
    ("돼다", "되다"),
}
RUNTIME_BROAD_LEMMA_SOURCES = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
MLM_SURFACE_TOP_K = 8
CANDIDATE_PREFILTER_LIMIT = 4
CHEAP_VERIFIER_LIMIT = 3
MLM_BACKOFF_PREFILTER_MIN = 0.8
MLM_BACKOFF_MIN_CANDIDATES = 2
FAST_PATH_VERIFIER_MIN = 0.82
FAST_PATH_MARGIN_MIN = 0.14
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")
META_COMPARE_TOKEN_PATTERN = re.compile(r"^[가-힣]{1,10}다(?:야|냐)$")

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
        self._kiwi_analysis_cache: Dict[Tuple[str, int], List[Tuple[Any, float]]] = {}
        self.typed_confusion_resource = self._load_typed_confusion_resource()
        self.phrase_memory = self._load_phrase_memory()
        self.predicate_family_seeds = self._load_predicate_family_seeds()
        self.inflection_recovery_rules = self._load_inflection_recovery_rules()
        self.phrase_memory_index = self._build_phrase_memory_index()
        self.phrase_memory_target_index = self._build_phrase_memory_target_index()
        self.rule_misspellings = self._load_surface_fix_rules()
        self.surface_fix_index = self._build_surface_fix_index(self.rule_misspellings)
        self.surface_confusion_map = self._build_surface_confusion_map()
        self.lemma_confusion_map = self._build_lemma_confusion_map()
        self.predicate_lemma_index = self._build_predicate_lemma_index()
        self._load_local_edit_tagger()
        self.profile_proto_emb = self._build_profile_prototypes(self._sentence_embedding)
        self.profile_proto_emb_electra = self._build_profile_prototypes(
            self._sentence_embedding_koelectra
        )
        self._warmup_runtime()

    def _load_local_edit_tagger(self) -> None:
        for model_dir in LOCAL_EDIT_TAGGER_CANDIDATE_DIRS:
            if not model_dir.exists():
                continue
            try:
                self.edit_tagger_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
                self.edit_tagger_model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).eval()
                raw_id2label = getattr(self.edit_tagger_model.config, "id2label", {}) or {}
                self.edit_tagger_id2label = {int(key): value for key, value in raw_id2label.items()}
                self.edit_tagger_source = str(model_dir)
                return
            except Exception:
                self.edit_tagger_model = None
                self.edit_tagger_tokenizer = None
                self.edit_tagger_id2label = {}
                self.edit_tagger_source = "heuristic"

    def _load_typed_confusion_resource(self) -> Dict[str, Any]:
        if not TYPED_CONFUSION_GRAPH_PATH.exists():
            return {"surface": {}, "lemma": {}}
        try:
            return json.loads(TYPED_CONFUSION_GRAPH_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"surface": {}, "lemma": {}}

    def _load_surface_fix_rules(self) -> List[Dict[str, Any]]:
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for item in RULE_MISSPELLINGS:
            merged[(item["from"], item["to"])] = dict(item)
        if HIGH_PRECISION_SURFACE_FIXES_PATH.exists():
            try:
                payload = json.loads(HIGH_PRECISION_SURFACE_FIXES_PATH.read_text(encoding="utf-8"))
                for item in payload.get("items", []):
                    src = item.get("from")
                    dst = item.get("to")
                    if not src or not dst:
                        continue
                    key = (src, dst)
                    existing = merged.get(key)
                    if existing is None or float(item.get("confidence", 0.0)) >= float(existing.get("confidence", 0.0)):
                        merged[key] = {
                            "from": src,
                            "to": dst,
                            "reasonTag": item.get("reasonTag", "high_precision_surface_fix"),
                            "confidence": float(item.get("confidence", 0.96)),
                        }
            except Exception:
                pass
        rows = list(merged.values())
        rows.sort(key=lambda item: (-len(item["from"]), item["from"], item["to"]))
        return rows

    def _load_phrase_memory(self) -> Dict[str, Any]:
        if not PHRASE_MEMORY_PATH.exists():
            return {"items": []}
        try:
            return json.loads(PHRASE_MEMORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}

    def _load_predicate_family_seeds(self) -> Dict[str, Any]:
        if not PREDICATE_FAMILY_SEEDS_PATH.exists():
            return {"lemmas": {}}
        try:
            return json.loads(PREDICATE_FAMILY_SEEDS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"lemmas": {}}

    def _load_inflection_recovery_rules(self) -> Dict[str, Any]:
        if not INFLECTION_RECOVERY_RULES_PATH.exists():
            return {"rules": []}
        try:
            return json.loads(INFLECTION_RECOVERY_RULES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"rules": []}

    def _build_phrase_memory_index(self) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.phrase_memory.get("items", []):
            source = str(item.get("from") or "").strip()
            replacement = str(item.get("to") or "").strip()
            if not source or not replacement or source == replacement:
                continue
            parts = self.tokenize_with_ranges(source)
            if not parts:
                continue
            first = parts[0][0]
            index.setdefault(first, []).append(item)
        for key in list(index):
            index[key].sort(
                key=lambda item: (
                    -len(str(item.get("from") or "")),
                    -float(item.get("confidence", 0.0)),
                    str(item.get("from") or ""),
                )
            )
        return index

    def _build_phrase_memory_target_index(self) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.phrase_memory.get("items", []):
            source = str(item.get("from") or "").strip()
            target = str(item.get("to") or "").strip()
            if not source or not target or source == target:
                continue
            target_parts = self.tokenize_with_ranges(target)
            if len(target_parts) < 2:
                continue
            if float(item.get("confidence", 0.0)) < 0.98:
                continue
            first = target_parts[0][0]
            index.setdefault(first, []).append(item)
        for key in list(index):
            index[key].sort(
                key=lambda item: (
                    -len(str(item.get("to") or "")),
                    -float(item.get("confidence", 0.0)),
                    str(item.get("to") or ""),
                )
            )
        return index

    @staticmethod
    def _build_surface_fix_index(rules: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for item in rules:
            source = str(item.get("from") or "").strip()
            if not source:
                continue
            existing = index.get(source)
            if existing is None or float(item.get("confidence", 0.0)) >= float(existing.get("confidence", 0.0)):
                index[source] = item
        return index

    @staticmethod
    def _merge_candidates(
        left: List[Dict[str, Any]],
        right: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for item in [*left, *right]:
            replacement = item["replacement"]
            existing = merged.get(replacement)
            if existing is None or float(item.get("generatorScore", 0.0)) > float(existing.get("generatorScore", 0.0)):
                merged[replacement] = item
        return list(merged.values())

    def _build_surface_confusion_map(self) -> Dict[str, List[Dict[str, Any]]]:
        surface_graph: Dict[str, List[Dict[str, Any]]] = {}
        for surface, replacements in OPEN_REPLACE_MAP.items():
            surface_graph[surface] = list(replacements)
        for surface, replacements in (self.typed_confusion_resource.get("surface") or {}).items():
            existing = surface_graph.get(surface, [])
            surface_graph[surface] = self._merge_candidates(existing, replacements)
        return surface_graph

    @staticmethod
    def _coarse_predicate_pos(tag: str) -> str:
        return str(tag or "").split("-", 1)[0]

    def _build_lemma_confusion_map(self) -> Dict[str, List[Dict[str, Any]]]:
        graph: Dict[str, Dict[str, Dict[str, Any]]] = {}

        def should_keep_runtime_edge(left: str, right: str, payload: Dict[str, Any]) -> bool:
            if (left, right) in RUNTIME_BROAD_LEMMA_ALLOWLIST:
                return True
            if right in RUNTIME_BROAD_LEMMA_TARGETS:
                return False
            source_name = str(payload.get("source") or "")
            if left in RUNTIME_BROAD_LEMMA_SOURCES and source_name in {"DATA_CONFUSION", "LEMMA_DATA_CONFUSION", "AUTO_FAMILY_SEED"}:
                return False
            if source_name in {"AUTO_FAMILY_SEED", "LEMMA_FAMILY_GRAPH", "LEMMA_DATA_CONFUSION"}:
                similarity = float(payload.get("surfaceSimilarity") or self._typo_similarity(left, right))
                if left and right and left[0] != right[0] and similarity < 0.7:
                    return False
                if similarity < 0.58 and not payload.get("contextHints"):
                    return False
            return True

        def merge_hint_rows(
            left: Optional[List[Dict[str, Any]]],
            right: Optional[List[Dict[str, Any]]],
        ) -> List[Dict[str, Any]]:
            merged: Dict[str, int] = {}
            for item in [*(left or []), *(right or [])]:
                term = str(item.get("term") or "").strip()
                if not term:
                    continue
                merged[term] = merged.get(term, 0) + int(item.get("count") or 0)
            return [
                {"term": term, "count": count}
                for term, count in sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))[:12]
            ]

        def merge_surface_examples(
            left: Optional[List[Dict[str, Any]]],
            right: Optional[List[Dict[str, Any]]],
        ) -> List[Dict[str, Any]]:
            merged: Dict[Tuple[str, str], int] = {}
            for item in [*(left or []), *(right or [])]:
                source = str(item.get("from") or "").strip()
                target = str(item.get("to") or "").strip()
                if not source or not target:
                    continue
                key = (source, target)
                merged[key] = merged.get(key, 0) + int(item.get("count") or 0)
            rows = [
                {"from": source, "to": target, "count": count}
                for (source, target), count in sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))
            ]
            return rows[:12]

        def merge_edge_payload(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            incoming_score = float(incoming.get("generatorScore", 0.0))
            existing_score = float(existing.get("generatorScore", 0.0))
            if incoming_score >= existing_score:
                for key in ("source", "generatorScore", "frequency", "surfaceSimilarity", "interfaceType", "pos"):
                    value = incoming.get(key)
                    if value not in (None, "", []):
                        merged[key] = value
            merged_interface_stats = dict(existing.get("interfaceStats") or {})
            for channel, count in (incoming.get("interfaceStats") or {}).items():
                merged_interface_stats[channel] = merged_interface_stats.get(channel, 0) + int(count or 0)
            if merged_interface_stats:
                merged["interfaceStats"] = merged_interface_stats
            merged_error_types = dict(existing.get("errorTypes") or {})
            for error_type, count in (incoming.get("errorTypes") or {}).items():
                merged_error_types[error_type] = merged_error_types.get(error_type, 0) + int(count or 0)
            if merged_error_types:
                merged["errorTypes"] = merged_error_types
            merged_source_stats = dict(existing.get("sourceStats") or {})
            for source_name, count in (incoming.get("sourceStats") or {}).items():
                merged_source_stats[source_name] = merged_source_stats.get(source_name, 0) + int(count or 0)
            if merged_source_stats:
                merged["sourceStats"] = merged_source_stats
            suffix_slots = sorted(
                {
                    str(slot)
                    for slot in [*(existing.get("suffixSlots") or []), *(incoming.get("suffixSlots") or [])]
                    if slot
                }
            )
            if suffix_slots:
                merged["suffixSlots"] = suffix_slots
            context_hints = merge_hint_rows(existing.get("contextHints"), incoming.get("contextHints"))
            if context_hints:
                merged["contextHints"] = context_hints
            surface_examples = merge_surface_examples(existing.get("surfaceExamples"), incoming.get("surfaceExamples"))
            if surface_examples:
                merged["surfaceExamples"] = surface_examples
            return merged

        def add_edge(left: str, right: str, source: str, score: float, extra: Optional[Dict[str, Any]] = None) -> None:
            graph.setdefault(left, {})
            existing = graph[left].get(right)
            payload = {"replacement": right, "source": source, "generatorScore": score, **(extra or {})}
            if not should_keep_runtime_edge(left, right, payload):
                return
            if existing is None:
                graph[left][right] = payload
                return
            graph[left][right] = merge_edge_payload(existing, payload)

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

        for lemma, replacements in (self.typed_confusion_resource.get("lemma") or {}).items():
            for replacement in replacements:
                target = replacement["replacement"]
                add_edge(
                    lemma,
                    target,
                    replacement.get("source", "LEMMA_DATA_CONFUSION"),
                    float(replacement.get("generatorScore", 0.0)),
                    {
                        "frequency": replacement.get("frequency"),
                        "surfaceSimilarity": replacement.get("surfaceSimilarity"),
                        "interfaceType": replacement.get("interfaceType"),
                        "interfaceStats": replacement.get("interfaceStats"),
                        "errorTypes": replacement.get("errorTypes"),
                        "sourceStats": replacement.get("sourceStats"),
                        "contextHints": replacement.get("contextHints"),
                        "pos": replacement.get("pos"),
                        "suffixSlots": replacement.get("suffixSlots"),
                    },
                )

        for lemma, replacements in (self.predicate_family_seeds.get("lemmas") or {}).items():
            for replacement in replacements:
                target = replacement["replacement"]
                add_edge(
                    lemma,
                    target,
                    replacement.get("source", "FAMILY_SEED"),
                    float(replacement.get("generatorScore", 0.0)),
                    {
                        "frequency": replacement.get("frequency"),
                        "surfaceSimilarity": replacement.get("surfaceSimilarity"),
                        "interfaceType": replacement.get("interfaceType"),
                        "interfaceStats": replacement.get("interfaceStats"),
                        "errorTypes": replacement.get("errorTypes"),
                        "sourceStats": replacement.get("sourceStats"),
                        "contextHints": replacement.get("contextHints"),
                        "pos": replacement.get("pos"),
                        "suffixSlots": replacement.get("suffixSlots"),
                    },
                )

        return {lemma: list(targets.values()) for lemma, targets in graph.items()}

    def _build_predicate_lemma_index(self) -> Dict[str, List[str]]:
        buckets: Dict[str, set[str]] = {}

        def register(lemma: str, pos: Optional[str]) -> None:
            normalized = str(lemma or "").strip()
            coarse_pos = self._coarse_predicate_pos(str(pos or ""))
            if not normalized or not normalized.endswith("다"):
                return
            if coarse_pos not in {"VV", "VA", "VX"}:
                return
            buckets.setdefault(coarse_pos, set()).add(normalized)

        for lemma, replacements in self.lemma_confusion_map.items():
            replacement_list = replacements or []
            if replacement_list:
                register(lemma, replacement_list[0].get("pos"))
            for replacement in replacement_list:
                register(replacement.get("replacement"), replacement.get("pos"))

        return {
            pos: sorted(values, key=lambda item: (len(item), item))
            for pos, values in buckets.items()
        }

    def _warmup_runtime(self) -> None:
        warm_sentence = "오늘 날씨가 좋다"
        try:
            self.kiwi.space(warm_sentence)
            self._kiwi_analyze_cached(warm_sentence, top_n=1)
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

    @staticmethod
    def _normalize_context_token(token: str) -> str:
        normalized = re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", token)
        if not re.fullmatch(r"[가-힣]+", normalized):
            return normalized
        particles = (
            "으로는",
            "에게서",
            "한테서",
            "이라도",
            "처럼은",
            "으로",
            "에게",
            "한테",
            "에서",
            "부터",
            "까지",
            "처럼",
            "보다",
            "이랑",
            "랑",
            "이나",
            "나",
            "은",
            "는",
            "이",
            "가",
            "을",
            "를",
            "에",
            "의",
            "와",
            "과",
            "도",
            "만",
            "로",
        )
        for particle in particles:
            if normalized.endswith(particle) and len(normalized) > len(particle) + 1:
                return normalized[: -len(particle)]
        return normalized

    def _context_window_terms(self, sentence: str, start: int, end: int, window: int = 3) -> List[str]:
        token_spans = self.tokenize_with_ranges(sentence)
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
            token = self._normalize_context_token(token_spans[idx][0])
            if len(token) < 2 or self._hangul_ratio(token) < 0.5:
                continue
            terms.append(token)
        return terms

    @staticmethod
    def _hint_lookup(hints: List[Dict[str, Any]]) -> Dict[str, int]:
        lookup: Dict[str, int] = {}
        for item in hints:
            term = item.get("term")
            if not term:
                continue
            normalized = CorrectionEngine._normalize_context_token(str(term))
            if not normalized:
                continue
            lookup[normalized] = lookup.get(normalized, 0) + int(item.get("count", 0))
        return lookup

    def _context_hint_score(self, sentence: str, start: int, end: int, item: Dict[str, Any]) -> float:
        hints = item.get("contextHints") or []
        if not hints:
            return 0.0
        context_terms = self._context_window_terms(sentence, start, end)
        if not context_terms:
            return 0.0
        hint_lookup = self._hint_lookup(hints)
        total = sum(hint_lookup.values()) or 1
        matched = 0
        for term in context_terms:
            matched += hint_lookup.get(term, 0)
            if matched:
                continue
            matched += sum(count for hint, count in hint_lookup.items() if term.startswith(hint) or hint.startswith(term))
        return round(float(min(1.0, matched / total)), 4)

    @staticmethod
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
        self,
        sentence: str,
        start: int,
        end: int,
        candidates: List[Dict[str, Any]],
        assumed_channel: str = "generic",
    ) -> List[Dict[str, Any]]:
        enriched: List[Dict[str, Any]] = []
        for item in candidates:
            context_score = self._context_hint_score(sentence, start, end, item)
            interface_score = self._candidate_interface_score(item, assumed_channel=assumed_channel)
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

    def _kiwi_sentence_score(self, text: str) -> float:
        try:
            analyzed = self._kiwi_analyze_cached(text, top_n=1)
        except Exception:
            return -999.0
        if not analyzed:
            return -999.0
        return float(analyzed[0][1])

    @staticmethod
    def _normalize_phrase_surface(text: str) -> str:
        return " ".join(TOKEN_PATTERN.findall(text)).strip()

    def _kiwi_analyze_cached(self, text: str, top_n: int = 1) -> List[Tuple[Any, float]]:
        key = (text, top_n)
        cached = self._kiwi_analysis_cache.get(key)
        if cached is not None:
            return cached
        try:
            analyses = self.kiwi.analyze(text, top_n=top_n)
        except Exception:
            analyses = []
        self._kiwi_analysis_cache[key] = analyses
        return analyses

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
        if source == "DATA_CONFUSION":
            return 0.99
        if source == "PHRASE_MEMORY":
            return 0.99
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
        if source == "JAMO":
            return 0.7
        if source == "MLM_TOPK":
            return 0.35
        return 0.25

    @staticmethod
    def _is_curated_candidate_source(source: str) -> bool:
        return source in CURATED_CANDIDATE_SOURCES

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
            source = str(item.get("source", "MLM_TOPK"))
            generator_norm = self._normalize_generator_score(float(item.get("generatorScore", 0.0)))
            source_prior = self._candidate_source_prior(source)
            context_score = float(item.get("contextHintScore", 0.0))
            interface_score = float(item.get("interfaceScore", 0.86))
            support_total = self._candidate_support_total(item)
            typo_similarity = self._typo_similarity(token, replacement)
            min_typo_similarity = TYPO_SIMILARITY_MIN
            if self._is_curated_candidate_source(source):
                if bool(item.get("familyLemmas")) or context_score >= 0.1 or generator_norm >= 0.72:
                    min_typo_similarity = 0.48
            if typo_similarity < min_typo_similarity:
                continue
            if source in {"FAMILY_SEED", "AUTO_FAMILY_SEED"}:
                if context_score < 0.08 and support_total < 14:
                    continue
            if source in {"DATA_CONFUSION", "LEMMA_DATA_CONFUSION"}:
                if context_score < 0.08 and support_total < 8 and typo_similarity < 0.62:
                    continue
            prefilter_score = round(
                float(
                    (0.4 * typo_similarity)
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
                    "typoSimilarity": typo_similarity,
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

    @staticmethod
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

    def _sentence_quality_baseline(self, sentence: str) -> Dict[str, float]:
        return {
            "koelectraScore": self._koelectra_normal_score(sentence),
            "kiwiScore": self._kiwi_sentence_score(sentence),
        }

    def _context_verifier_score(
        self,
        typo_similarity: float,
        electra_delta: float,
        kiwi_delta: float,
        retrieval_score: float,
        interface_score: float,
    ) -> float:
        verifier_signal = (
            (electra_delta * 10.0)
            + (kiwi_delta / 2.8)
            + ((typo_similarity - 0.55) * 1.4)
            + (retrieval_score * 1.1)
            + ((interface_score - 0.5) * 0.55)
        )
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
            typo_bonus = 0.08 * typo_similarity
            source_bonus = 0.04 * source_prior
            retrieval_bonus = 0.04 * item.get("contextHintScore", 0.0)
            interface_bonus = 0.02 * item.get("interfaceScore", 0.86)
            semantic_penalty = 0.16 if typo_similarity < 0.6 else 0.0
            context_verifier = self._context_verifier_score(
                typo_similarity=typo_similarity,
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
                    "typoSimilarity": typo_similarity,
                    "contextHintScore": round(float(item.get("contextHintScore", 0.0)), 4),
                    "interfaceScore": round(float(item.get("interfaceScore", 0.86)), 4),
                    "contextVerifierScore": round(float(context_verifier), 4),
                    "familyBonus": round(float(family_bonus), 4),
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
        return self._find_surface_fix_edits(
            text=text,
            protected=protected,
            rules=self.rule_misspellings,
            stage_name="RULE",
            reason_prefix="surface_fix",
        )

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

    @staticmethod
    def non_overlapping_edits(edits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        ordered = sorted(
            edits,
            key=lambda e: (
                e["range"]["start"],
                -(e["range"]["end"] - e["range"]["start"]),
                -float(e.get("confidence", 0.0)),
            ),
        )
        kept: List[Dict[str, Any]] = []
        claimed: List[Tuple[int, int]] = []
        for edit in ordered:
            span = (edit["range"]["start"], edit["range"]["end"])
            if any(span[0] < end and span[1] > start for start, end in claimed):
                continue
            kept.append(edit)
            claimed.append(span)
        return kept

    def _find_surface_fix_edits(
        self,
        text: str,
        protected: List[Dict[str, Any]],
        rules: List[Dict[str, Any]],
        stage_name: str,
        reason_prefix: str,
    ) -> List[Dict[str, Any]]:
        edits: List[Dict[str, Any]] = []
        for rule in rules:
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
        return self.non_overlapping_edits(self.dedupe_edits(edits))

    def spacing_candidates_from_kiwi(
        self, text: str, protected: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        spaced = self.kiwi.space(text)
        candidates: List[Dict[str, Any]] = []
        preserve_ranges = self._compact_preserve_ranges(text, protected)
        i, j = 0, 0
        while i < len(text) and j < len(spaced):
            if text[i] == spaced[j]:
                i += 1
                j += 1
                continue
            if spaced[j] == " ":
                probe = {"start": max(0, i - 1), "end": min(len(text), i + 1)}
                if not self.is_range_protected(probe, protected) and not any(start < i < end for start, end in preserve_ranges):
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

    def _single_token_phrase_candidates(self, token: str) -> List[Dict[str, Any]]:
        candidates = self.phrase_memory_index.get(token) or []
        out: List[Dict[str, Any]] = []
        for item in candidates:
            source = str(item.get("from") or "").strip()
            if not source:
                continue
            if len(self.tokenize_with_ranges(source)) != 1:
                continue
            if self._normalize_phrase_surface(source) != self._normalize_phrase_surface(token):
                continue
            out.append(item)
        return out

    def _has_strong_inline_fix(self, token: str) -> bool:
        if token in self.surface_fix_index:
            return True
        return bool(self._single_token_phrase_candidates(token))

    def _compact_preserve_ranges(
        self, text: str, protected: List[Dict[str, Any]]
    ) -> List[Tuple[int, int]]:
        ranges: List[Tuple[int, int]] = []
        for token, start, end in self.tokenize_with_ranges(text):
            span = {"start": start, "end": end}
            if self.is_range_protected(span, protected):
                continue
            if self._hangul_ratio(token) < 0.7 or len(token) < 2 or len(token) > 8:
                continue
            if self._has_strong_inline_fix(token):
                ranges.append((start, end))
        return ranges

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

    def _run_phrase_normalizations(
        self,
        text: str,
        protected: List[Dict[str, Any]],
        *,
        single_token_only: bool = False,
    ) -> List[Dict[str, Any]]:
        edits: List[Dict[str, Any]] = []
        token_ranges = self.tokenize_with_ranges(text)
        for index, (token, start, _) in enumerate(token_ranges):
            candidates = self.phrase_memory_index.get(token) or []
            if not candidates:
                continue
            for item in candidates:
                source = str(item.get("from") or "")
                replacement = str(item.get("to") or "")
                if not source or not replacement or source == replacement:
                    continue
                source_parts = self.tokenize_with_ranges(source)
                if not source_parts:
                    continue
                if single_token_only and len(source_parts) != 1:
                    continue
                span_len = len(source_parts)
                if index + span_len > len(token_ranges):
                    continue
                end = token_ranges[index + span_len - 1][2]
                span = {"start": start, "end": end}
                if self.is_range_protected(span, protected):
                    continue
                source_text = text[start:end]
                if self._normalize_phrase_surface(source_text) != self._normalize_phrase_surface(source):
                    continue
                edits.append(
                    self.create_edit(
                        stage="PRE_NORMALIZE",
                        start=start,
                        end=end,
                        source_text=source_text,
                        replacement=replacement,
                        edit_type="SPELL",
                        confidence=float(item.get("confidence", 0.96)),
                        auto_applicable=True,
                        reason_tag=str(item.get("reasonTag") or "phrase_memory_auto"),
                    )
                )
                break
        return edits

    def run_pre_spacing_phrase_normalizer(
        self, text: str, protected: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        return self._run_phrase_normalizations(
            text,
            protected,
            single_token_only=True,
        )

    def _run_morpheme_normalizations(
        self, text: str, protected: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        edits: List[Dict[str, Any]] = []
        token_ranges = self.tokenize_with_ranges(text)
        for token, start, end in token_ranges:
            span = {"start": start, "end": end}
            if self.is_range_protected(span, protected):
                continue
            if self._hangul_ratio(token) < 0.7 or len(token) < 2:
                continue

            analyses = self._kiwi_analyze_cached(token, top_n=1)
            if not analyses:
                continue
            morphs = analyses[0][0]
            if not morphs:
                continue

            changed = False
            min_score = 1.0
            morph_seq: List[Tuple[str, str]] = []
            for morph in morphs:
                normalized = MORPHEME_CONFUSION_RULES.get((morph.raw_form, morph.tag))
                if normalized:
                    top = max(normalized, key=lambda item: float(item.get("generatorScore", 0.0)))
                    morph_seq.append((top["replacement"], morph.tag))
                    changed = True
                    min_score = min(min_score, float(top.get("generatorScore", 0.91)))
                else:
                    morph_seq.append((morph.form, morph.tag))
            if not changed:
                continue

            try:
                restored = self.kiwi.join(morph_seq, lm_search=True)
            except TypeError:
                try:
                    restored = self.kiwi.join(morph_seq)
                except Exception:
                    continue
            except Exception:
                continue

            if not restored or restored == token:
                continue
            similarity = self._typo_similarity(token, restored)
            if similarity < 0.68:
                continue
            edits.append(
                self.create_edit(
                    stage="PRE_NORMALIZE",
                    start=start,
                    end=end,
                    source_text=token,
                    replacement=restored,
                    edit_type="SPELL",
                    confidence=max(0.95, min(0.99, round(min_score + ((similarity - 0.68) * 0.2), 4))),
                    auto_applicable=True,
                    reason_tag="kiwi_morpheme_normalization",
                )
            )
        return edits

    def run_kiwi_pre_normalizer(self, text: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        edits = [
            *self._run_phrase_normalizations(text, protected),
            *self._run_morpheme_normalizations(text, protected),
        ]
        return self.non_overlapping_edits(self.dedupe_edits(edits))

    @staticmethod
    def tokenize_with_ranges(text: str) -> List[Tuple[str, int, int]]:
        out = []
        for m in TOKEN_PATTERN.finditer(text):
            out.append((m.group(0), m.start(), m.end()))
        return out

    def _token_morph_bundles(
        self, sentence: str, start: int, end: int, top_n: int = 3
    ) -> List[Dict[str, Any]]:
        analyses = self._kiwi_analyze_cached(sentence, top_n=top_n)
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
        self, sentence: str, start: int, end: int
    ) -> Optional[Dict[str, Any]]:
        bundles = self._token_morph_bundles(sentence, start, end, top_n=1)
        return bundles[0] if bundles else None

    @staticmethod
    def _stem_from_lemma(lemma: str, fallback: str = "") -> str:
        normalized = str(lemma or "").strip()
        if normalized.endswith("다"):
            return normalized[:-1]
        return fallback or normalized

    @staticmethod
    def _infer_irregular_class_from_lemma(lemma: str) -> Optional[str]:
        normalized = str(lemma or "").strip()
        if normalized.endswith("하다"):
            return "HA"
        if normalized.endswith("르다"):
            return "REU"
        if normalized.endswith("치다"):
            return "CHI"
        if normalized.endswith("추다"):
            return "CHU"
        if normalized.endswith("히다"):
            return "HI"
        if normalized.endswith("키다"):
            return "KI"
        if normalized.endswith("우다"):
            return "U"
        return None

    def _canonical_predicate_states(
        self,
        sentence: str,
        start: int,
        end: int,
        top_n: int = 5,
    ) -> List[Dict[str, Any]]:
        bundles = self._token_morph_bundles(sentence, start, end, top_n=top_n)
        states: List[Dict[str, Any]] = []
        seen = set()
        for bundle in bundles:
            predicate = bundle["predicate"]
            lemma = str(predicate.lemma or "").strip()
            if not lemma:
                continue
            coarse_pos = self._coarse_predicate_pos(predicate.tag)
            if coarse_pos not in {"VV", "VA", "VX"}:
                continue
            state = {
                "surface": sentence[start:end],
                "lemma": lemma,
                "pos": coarse_pos,
                "tag": predicate.tag,
                "slot": bundle.get("suffixSlot"),
                "suffixMorphs": bundle["suffixMorphs"],
                "analysisRank": int(bundle.get("analysisRank", 0)),
                "analysisScore": float(bundle.get("analysisScore", 0.0)),
                "irregularClass": self._infer_irregular_class_from_lemma(lemma),
                "stem": self._stem_from_lemma(lemma, predicate.form),
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

    def _predicate_lemma_exists(self, lemma: str, coarse_pos: str) -> bool:
        if not lemma:
            return False
        return lemma in (self.predicate_lemma_index.get(coarse_pos) or [])

    def _reinflect_candidate(
        self,
        state: Dict[str, Any],
        target_lemma: str,
        *,
        source: str,
        generator_score: float,
        token: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        target_tag = self._infer_lemma_tag(target_lemma, state["tag"])
        target_stem = self._stem_from_lemma(target_lemma, target_lemma)
        morph_seq = [(target_stem, target_tag), *[(m.form, m.tag) for m in state["suffixMorphs"]]]
        try:
            restored = self.kiwi.join(morph_seq, lm_search=True)
        except TypeError:
            try:
                restored = self.kiwi.join(morph_seq)
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
            "irregularClass": self._infer_irregular_class_from_lemma(target_lemma) or state.get("irregularClass"),
        }
        if extra:
            payload.update({key: value for key, value in extra.items() if value not in (None, "", [])})
        return payload

    def _productive_inflection_rule_candidates(
        self,
        state: Dict[str, Any],
        token: str,
    ) -> List[Dict[str, Any]]:
        rules = self.inflection_recovery_rules.get("rules") or []
        if not rules:
            return []
        current_lemma = state["lemma"]
        current_stem = state["stem"]
        coarse_pos = state["pos"]
        slot = state.get("slot")
        dedup: Dict[str, Dict[str, Any]] = {}
        for rule in rules:
            rule_pos = self._coarse_predicate_pos(str(rule.get("pos") or ""))
            if rule_pos and rule_pos != coarse_pos:
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
            if not self._predicate_lemma_exists(target_lemma, coarse_pos):
                continue
            rule_score = float(rule.get("generatorScore", 0.0))
            analysis_penalty = min(0.08, 0.02 * int(state.get("analysisRank", 0)))
            payload = self._reinflect_candidate(
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
        self,
        state: Dict[str, Any],
        token: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        current_lemma = state["lemma"]
        current_stem = state["stem"]
        coarse_pos = state["pos"]
        if not current_stem or len(current_stem) < 2:
            return []
        scored: List[Tuple[float, str]] = []
        for lemma in self.predicate_lemma_index.get(coarse_pos, []):
            if lemma == current_lemma:
                continue
            candidate_stem = self._stem_from_lemma(lemma)
            if abs(len(candidate_stem) - len(current_stem)) > 1:
                continue
            if current_stem[0] != candidate_stem[0]:
                continue
            similarity = self._typo_similarity(current_stem, candidate_stem)
            if similarity < 0.72:
                continue
            if self._char_edit_distance(current_stem, candidate_stem) > 2:
                continue
            scored.append((similarity, lemma))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        dedup: Dict[str, Dict[str, Any]] = {}
        for similarity, lemma in scored[:limit]:
            payload = self._reinflect_candidate(
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

    def _infer_lemma_tag(self, lemma: str, fallback_tag: str) -> str:
        tag_base = fallback_tag.split("-")[0]
        try:
            analyses = self._kiwi_analyze_cached(lemma, top_n=2)
        except Exception:
            return tag_base
        for tokens, _ in analyses:
            for token in tokens:
                if token.lemma == lemma and token.tag.startswith(("VV", "VA")):
                    return token.tag
        return tag_base

    def _predicate_lemmas_from_text(self, text: str, top_n: int = 2) -> List[str]:
        try:
            analyses = self._kiwi_analyze_cached(text, top_n=top_n)
        except Exception:
            return []
        lemmas: List[str] = []
        for tokens, _ in analyses:
            for token in tokens:
                if token.tag.startswith(("VV", "VA")) and token.lemma not in lemmas:
                    lemmas.append(token.lemma)
        return lemmas

    def _build_candidate_family_prior(
        self,
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
        states = self._canonical_predicate_states(sentence, start, end, top_n=5)
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

        for lemma in self._predicate_lemmas_from_text(best_replacement):
            if lemma not in lemmas:
                lemmas.append(lemma)

        expanded_lemmas = list(lemmas)
        for lemma in list(lemmas):
            for variant in self.lemma_confusion_map.get(lemma, [])[:8]:
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

    def _candidate_family_lemmas(self, replacement: str) -> List[str]:
        return self._predicate_lemmas_from_text(replacement, top_n=3)

    @staticmethod
    def _variant_suffix_slots(variant: Dict[str, Any]) -> List[str]:
        raw_slots = variant.get("suffixSlots") or variant.get("suffixSlot") or []
        if isinstance(raw_slots, str):
            return [raw_slots]
        return [str(item) for item in raw_slots if item]

    def _suffix_slot_compatible(self, bundle_slot: Optional[str], variant: Dict[str, Any]) -> bool:
        variant_slots = self._variant_suffix_slots(variant)
        if not bundle_slot or not variant_slots:
            return True
        return any(
            bundle_slot == slot
            or bundle_slot.startswith(f"{slot}+")
            or slot.startswith(f"{bundle_slot}+")
            for slot in variant_slots
        )

    def _apply_family_prior(
        self,
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
                candidate_lemmas = self._candidate_family_lemmas(item["replacement"])
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

    @staticmethod
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

    @staticmethod
    def _candidate_slot_compatible_with_prior(item: Dict[str, Any], family_prior: Optional[Dict[str, Any]]) -> bool:
        if not family_prior:
            return True
        slot_hints = [str(value) for value in (family_prior.get("slotHints") or []) if value]
        if not slot_hints:
            return True
        candidate_slots = CorrectionEngine._variant_suffix_slots(item)
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

    def _build_join_candidates(
        self,
        sentence: str,
        token: str,
        start: int,
        end: int,
    ) -> List[Dict[str, Any]]:
        states = self._canonical_predicate_states(sentence, start, end, top_n=5)
        if not states:
            return []
        dedup: Dict[str, Dict[str, Any]] = {}

        def keep_best(candidate: Dict[str, Any]) -> None:
            replacement = candidate["replacement"]
            existing = dedup.get(replacement)
            if existing is None or float(candidate.get("generatorScore", 0.0)) > float(existing.get("generatorScore", 0.0)):
                dedup[replacement] = candidate

        bundles = self._token_morph_bundles(sentence, start, end, top_n=5)
        bundle_by_rank = {int(bundle.get("analysisRank", 0)): bundle for bundle in bundles}

        for state in states:
            predicate_lemma = state["lemma"]
            bundle_slot = state.get("slot")
            bundle_penalty = min(0.08, 0.03 * int(state.get("analysisRank", 0)))
            bundle = bundle_by_rank.get(int(state.get("analysisRank", 0)))

            lemma_variants = self.lemma_confusion_map.get(predicate_lemma, [])
            for variant in lemma_variants:
                if not self._suffix_slot_compatible(bundle_slot, variant):
                    continue
                target_lemma = variant["replacement"]
                payload = self._reinflect_candidate(
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
                    example_similarity = self._typo_similarity(token, example_from)
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

            for inflection_candidate in self._productive_inflection_rule_candidates(state, token):
                keep_best(inflection_candidate)

            for fallback_candidate in self._jamo_weighted_fallback_candidates(state, token):
                keep_best(fallback_candidate)

            if not bundle:
                continue

            predicate = bundle["predicate"]
            predicate_stem = self._stem_from_lemma(state["lemma"], predicate.form)
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
                        restored = self.kiwi.join(morph_seq, lm_search=True)
                    except TypeError:
                        try:
                            restored = self.kiwi.join(morph_seq)
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

    def open_replace_candidates_for_token(self, token: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = list(self.surface_confusion_map.get(token, []))

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
        assumed_channel: str = "generic",
    ) -> List[Dict[str, Any]]:
        curated_candidates = self._enrich_candidate_context(
            sentence,
            start,
            end,
            self.open_replace_candidates_for_token(token),
            assumed_channel=assumed_channel,
        )
        join_candidates = self._build_join_candidates(sentence, token, start, end)
        join_candidates = self._enrich_candidate_context(
            sentence,
            start,
            end,
            join_candidates,
            assumed_channel=assumed_channel,
        )
        curated_pool = self._prefilter_candidates(
            token,
            [*curated_candidates, *join_candidates],
        )
        needs_mlm_backoff = (
            len(curated_pool) < MLM_BACKOFF_MIN_CANDIDATES
            or max((item.get("prefilterScore", 0.0) for item in curated_pool), default=0.0) < MLM_BACKOFF_PREFILTER_MIN
        )
        candidate_pool = curated_pool
        if needs_mlm_backoff:
            mlm_candidates = self._enrich_candidate_context(
                sentence,
                start,
                end,
                self._mlm_surface_candidates(sentence, token, start, end),
                assumed_channel=assumed_channel,
            )
            candidate_pool = self._prefilter_candidates(
                token,
                [*curated_pool, *mlm_candidates],
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

    def _filter_typo_like_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for candidate in candidates:
            min_typo_similarity = TYPO_SIMILARITY_MIN
            if self._is_curated_candidate_source(str(candidate.get("source", ""))):
                if (
                    bool(candidate.get("familyLemmas"))
                    or float(candidate.get("contextHintScore", 0.0)) >= 0.1
                    or float(candidate.get("generatorNorm", 0.0)) >= 0.72
                ):
                    min_typo_similarity = 0.48
            if float(candidate.get("typoSimilarity", 0.0)) >= min_typo_similarity:
                filtered.append(candidate)
        return filtered

    def _candidate_is_predicate_like(self, item: Dict[str, Any]) -> bool:
        pos = str(item.get("pos") or "")
        if pos.startswith(("VV", "VA", "VX")):
            return True
        family_lemmas = item.get("familyLemmas") or []
        if family_lemmas:
            return True
        replacement = str(item.get("replacement") or "")
        if not replacement:
            return False
        return bool(self._predicate_lemmas_from_text(replacement, top_n=3))

    def _maybe_reopen_short_valid_token(
        self,
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

        predicate_lemmas = self._predicate_lemmas_from_text(token, top_n=3)
        if not predicate_lemmas:
            return None
        if not any(self.lemma_confusion_map.get(lemma) for lemma in predicate_lemmas):
            return None

        ranked_candidates = self.generate_contextual_candidates_for_token(
            text,
            token,
            start,
            end,
            expensive=False,
        )
        typo_like_candidates = self._filter_typo_like_candidates(ranked_candidates)
        if not typo_like_candidates:
            return None

        best = typo_like_candidates[0]
        best_source = str(best.get("source") or "")
        if best_source not in CURATED_CANDIDATE_SOURCES:
            return None
        if not self._candidate_is_predicate_like(best):
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
            if re.search(r"[A-Za-z0-9]", token):
                labels.append({"token": token, "label": "KEEP", "confidence": 0.99, "range": span})
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

            if pred_label == "KEEP":
                reopened = self._maybe_reopen_short_valid_token(text, token, st, ed, confidence)
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

    def run_edit_tagger(self, text: str, protected: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.edit_tagger_model is not None:
            return self._run_finetuned_edit_tagger(text, protected)
        return self._run_heuristic_edit_tagger(text, protected)

    def generate_open_candidates(self, sentence: str, tag_labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for l in tag_labels:
            if l.get("effectiveLabel", l["label"]) != "OPEN_REPLACE":
                continue
            detection_evidence = l.get("detectionEvidence") or {}
            best_replacement = detection_evidence.get("bestReplacement")
            if not best_replacement and l.get("modelCandidates"):
                best_replacement = l["modelCandidates"][0].get("replacement")
            family_prior = self._build_candidate_family_prior(
                sentence,
                l["token"],
                l["range"]["start"],
                l["range"]["end"],
                best_replacement,
            )
            seeded_candidates = l.get("modelCandidates") or []
            generated_candidates = self.generate_contextual_candidates_for_token(
                sentence,
                l["token"],
                l["range"]["start"],
                l["range"]["end"],
                expensive=False,
            )
            base = self._merge_candidates(seeded_candidates, generated_candidates)
            base = self._apply_family_prior(base, family_prior)
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
        return out

    def verify_candidate_groups(self, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        verified: List[Dict[str, Any]] = []
        for group in groups:
            original = group["original"]
            family_prior = group.get("candidateFamily")
            candidate_items = [item for item in group["items"] if item["source"] != "ORIGINAL"]
            curated_items = [
                item for item in candidate_items if self._is_curated_candidate_source(item.get("source", "MLM_TOPK"))
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
                typo_similarity = float(item.get("typoSimilarity", self._typo_similarity(original, item["replacement"])))
                generator_norm = float(
                    item.get(
                        "generatorNorm",
                        self._normalize_generator_score(float(item.get("generatorScore", 0.0))),
                    )
                )
                context_hint_score = float(item.get("contextHintScore", 0.0))
                interface_score = float(item.get("interfaceScore", 0.86))
                family_match = bool(item.get("familyMatch"))
                pos_compatible = self._candidate_pos_compatible_with_prior(item, family_prior)
                slot_compatible = self._candidate_slot_compatible_with_prior(item, family_prior)
                if not pos_compatible:
                    continue
                if not slot_compatible and not self._is_curated_candidate_source(source):
                    continue
                if family_prior and family_matched_items and not family_match and not self._is_curated_candidate_source(source):
                    if context_hint_score < 0.18 and prefilter < curated_best + 0.04:
                        continue
                if curated_items and not self._is_curated_candidate_source(source):
                    if context_hint_score < 0.12 and prefilter < curated_best + 0.05:
                        continue
                if typo_similarity < 0.56 and not family_match and context_hint_score < 0.2:
                    continue

                verifier_score = (
                    0.42 * typo_similarity
                    + 0.24 * generator_norm
                    + 0.2 * prefilter
                    + 0.06 * context_hint_score
                    + 0.04 * interface_score
                )
                if family_prior and family_match:
                    verifier_score += 0.08
                if self._is_curated_candidate_source(source):
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
                single_curated = len(kept) == 1 and self._is_curated_candidate_source(leader.get("source", "MLM_TOPK"))
                clear_curated_gap = (
                    self._is_curated_candidate_source(leader.get("source", "MLM_TOPK"))
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
                        self._is_curated_candidate_source(leader.get("source", "MLM_TOPK"))
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

    def _high_precision_lock_spans(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, int]]:
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
        self,
        text: str,
        protected: List[Dict[str, Any]],
    ) -> List[Dict[str, int]]:
        locked: List[Dict[str, int]] = []
        token_ranges = self.tokenize_with_ranges(text)
        for index, (token, start, _) in enumerate(token_ranges):
            candidates = self.phrase_memory_target_index.get(token) or []
            if not candidates:
                continue
            for item in candidates:
                target = str(item.get("to") or "")
                target_parts = self.tokenize_with_ranges(target)
                span_len = len(target_parts)
                if not target_parts or index + span_len > len(token_ranges):
                    continue
                end = token_ranges[index + span_len - 1][2]
                span = {"start": start, "end": end}
                if self.is_range_protected(span, protected):
                    continue
                source_text = text[start:end]
                if self._normalize_phrase_surface(source_text) != self._normalize_phrase_surface(target):
                    continue
                locked.append(span)
                break
        return locked

    def _meta_comparison_lock_spans(
        self,
        text: str,
        protected: List[Dict[str, Any]],
    ) -> List[Dict[str, int]]:
        if "?" not in text and "？" not in text:
            return []
        token_ranges = self.tokenize_with_ranges(text)
        matched: List[Dict[str, int]] = []
        for token, start, end in token_ranges:
            span = {"start": start, "end": end}
            if self.is_range_protected(span, protected):
                continue
            if META_COMPARE_TOKEN_PATTERN.fullmatch(token):
                matched.append(span)
        if len(matched) < 2:
            return []
        return matched

    @staticmethod
    def _range_overlaps(span: Dict[str, int], other: Dict[str, int]) -> bool:
        return span["start"] < other["end"] and other["start"] < span["end"]

    def apply_high_precision_lock_guard(
        self,
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
            if not any(self._range_overlaps(span, locked) for locked in locked_spans):
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
        self,
        labels: List[Dict[str, Any]],
        locked_spans: List[Dict[str, int]],
    ) -> List[Dict[str, Any]]:
        if not locked_spans:
            return labels
        guarded: List[Dict[str, Any]] = []
        for label in labels:
            effective = label.get("effectiveLabel", label["label"])
            span = label.get("range") or {"start": 0, "end": 0}
            if effective != "OPEN_REPLACE" or not any(self._range_overlaps(span, locked) for locked in locked_spans):
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

    def rerank(self, sentence: str, groups: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not groups:
            return []
        ranked: List[Dict[str, Any]] = []
        baseline = self._sentence_quality_baseline(sentence)
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

        compact_phrase_edits = stage(
            "4-1. Phrase Pre-normalizer",
            working_clause,
            lambda: self.run_pre_spacing_phrase_normalizer(working_clause, self.detect_protected_spans(working_clause)),
        )
        if compact_phrase_edits:
            working_clause = self.apply_edits(working_clause, compact_phrase_edits)
            for e in compact_phrase_edits:
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

        kiwi_pre_normalizer_edits = stage(
            "6-3. Kiwi Pre-normalizer",
            working_clause,
            lambda: self.run_kiwi_pre_normalizer(working_clause, self.detect_protected_spans(working_clause)),
        )
        if kiwi_pre_normalizer_edits:
            working_clause = self.apply_edits(working_clause, kiwi_pre_normalizer_edits)
            for e in kiwi_pre_normalizer_edits:
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
        locked_spans = [
            *self._high_precision_lock_spans(decisions),
            *self._phrase_target_lock_spans(working_clause, self.detect_protected_spans(working_clause)),
        ]
        routed_tagger_labels = stage(
            "7-3. High-precision Lock Guard",
            working_clause,
            lambda: self.apply_high_precision_lock_guard(routed_tagger_labels, locked_spans),
        )
        meta_locked_spans = self._meta_comparison_lock_spans(
            working_clause,
            self.detect_protected_spans(working_clause),
        )
        routed_tagger_labels = stage(
            "7-4. Meta Comparison Guard",
            working_clause,
            lambda: self.apply_meta_comparison_guard(routed_tagger_labels, meta_locked_spans),
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

        suggestion_clause = working_clause
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
            elif p["decision"] == "SUGGEST_ONLY":
                suggestion_clause = self.apply_edits(suggestion_clause, [edit])

        stage(
            "12. Final Output",
            working_clause,
            lambda: {
                "mode": mode,
                "finalClause": working_clause,
                "suggestedClause": suggestion_clause,
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
        suggested_text = (
            full_text[: range_art["clauseRange"]["start"]]
            + suggestion_clause
            + full_text[range_art["clauseRange"]["end"] :]
        )
        return {
            "finalText": final_text,
            "suggestedText": suggested_text,
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
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE: Optional[CorrectionEngine] = None
ENGINE_INIT_LOCK = threading.Lock()
ENGINE_WARMUP_THREAD: Optional[threading.Thread] = None
ENGINE_INITIALIZING = False
ENGINE_INIT_ERROR: Optional[str] = None


def get_engine() -> CorrectionEngine:
    global ENGINE
    global ENGINE_INITIALIZING
    global ENGINE_INIT_ERROR
    if ENGINE is not None:
        return ENGINE
    with ENGINE_INIT_LOCK:
        if ENGINE is not None:
            return ENGINE
        ENGINE_INITIALIZING = True
        ENGINE_INIT_ERROR = None
        try:
            LOGGER.info("Initializing correction engine")
            ENGINE = CorrectionEngine()
            LOGGER.info("Correction engine initialized")
            return ENGINE
        except Exception as exc:
            ENGINE = None
            ENGINE_INIT_ERROR = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Correction engine initialization failed")
            raise
        finally:
            ENGINE_INITIALIZING = False


def _warm_engine_background() -> None:
    try:
        get_engine()
    except Exception:
        # Keep the service alive; /api/health will expose the error and /api/correct can retry.
        pass


def schedule_engine_warmup() -> None:
    global ENGINE_WARMUP_THREAD
    if ENGINE is not None:
        return
    with ENGINE_INIT_LOCK:
        if ENGINE is not None:
            return
        if ENGINE_WARMUP_THREAD is not None and ENGINE_WARMUP_THREAD.is_alive():
            return
        ENGINE_WARMUP_THREAD = threading.Thread(
            target=_warm_engine_background,
            name="engine-warmup",
            daemon=True,
        )
        ENGINE_WARMUP_THREAD.start()


@app.on_event("startup")
def startup() -> None:
    if IS_CLOUD_RUN:
        LOGGER.info("Cloud Run detected; deferring heavy engine startup")
        schedule_engine_warmup()
        return
    get_engine()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/app.js")
def app_js() -> FileResponse:
    return FileResponse(APP_DIR / "app.js", media_type="application/javascript")


@app.get("/styles.css")
def styles() -> FileResponse:
    return FileResponse(APP_DIR / "styles.css", media_type="text/css")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {
        "ok": True,
        "backendName": DEFAULT_BACKEND_NAME,
        "backendBranch": DEFAULT_BACKEND_BRANCH,
        "backendVersion": DEFAULT_BACKEND_VERSION,
        "engineReady": ENGINE is not None,
        "engineInitializing": ENGINE_INITIALIZING,
        "engineError": ENGINE_INIT_ERROR,
        "cloudRun": IS_CLOUD_RUN,
    }


@app.post("/api/correct")
def correct(req: CorrectRequest) -> Dict[str, Any]:
    try:
        engine = get_engine()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model engine is not ready: {exc}") from exc
    text = req.text or ""
    cursor = len(text) if req.cursor is None else req.cursor
    return engine.run_pipeline(full_text=text, cursor=cursor, mode=req.mode)
