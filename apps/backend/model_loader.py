# AI 모델과 사전 데이터 불러오기
# 서버 시작 시 필요한 AI 모델들(Kiwi, KoBERT 등)과
# 맞춤법 사전, 혼동 단어 목록 등을 한 번에 불러와서 보관한다.

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from kiwipiepy import Kiwi
from kobert_transformers import get_kobert_model, get_tokenizer
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BartForConditionalGeneration,
    PreTrainedTokenizerFast,
)

from utils.edit_ops import tokenize_with_ranges
from utils.hangul import hangul_ratio, typo_similarity

LOGGER = logging.getLogger("grammarly_korean")

IS_CLOUD_RUN = bool(os.getenv("K_SERVICE") or os.getenv("CLOUD_RUN_SERVICE") or os.getenv("K_REVISION"))
os.environ.setdefault("HF_HUB_OFFLINE", "0" if IS_CLOUD_RUN else "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0" if IS_CLOUD_RUN else "1")

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[1]

# ── 모델 경로 ──────────────────────────────────────────────
LOCAL_EDIT_TAGGER_CANDIDATE_DIRS = [
    REPO_ROOT / "models" / "edit_tagger_v2_best" / "best",
    REPO_ROOT / "models" / "edit_tagger_v1_best" / "best",
]
LOCAL_PROPOSAL_SEQ2SEQ_CANDIDATE_DIRS = [
    REPO_ROOT / "models" / "proposal_seq2seq_best",
    REPO_ROOT / "models" / "kobart-spell-v2-final",
    REPO_ROOT / "models" / "kobart-spell-final",
    REPO_ROOT / "models" / "kobart-spell-20250815_160551" / "checkpoint-1000",
]

# ── 리소스 경로 ────────────────────────────────────────────
TYPED_CONFUSION_GRAPH_PATH = REPO_ROOT / "public" / "assets" / "dict" / "typed_confusion_graph.json"
HIGH_PRECISION_SURFACE_FIXES_PATH = REPO_ROOT / "public" / "assets" / "rules" / "high_precision_surface_fixes.json"
PHRASE_MEMORY_PATH = REPO_ROOT / "public" / "assets" / "dict" / "phrase_memory.json"
PREDICATE_FAMILY_SEEDS_PATH = REPO_ROOT / "public" / "assets" / "dict" / "predicate_family_seeds.json"
INFLECTION_RECOVERY_RULES_PATH = REPO_ROOT / "public" / "assets" / "dict" / "inflection_recovery_rules.json"

# ── 하드코딩 규칙 ──────────────────────────────────────────
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

# ── 상수 ───────────────────────────────────────────────────
TYPO_SIMILARITY_MIN = 0.55
ROUTING_PROMOTION_MIN_SCORE = 0.55
ROUTING_PROMOTION_LABELS = {"PUNCT_FIX", "JOSA_FIX", "EOMI_FIX"}
CURATED_CANDIDATE_SOURCES = {
    "CONFUSION_SET", "MORPH", "LEMMA_CONFUSION", "MORPHEME_CONFUSION",
    "COMMON_SURFACE_PATTERN", "COMMON_PHRASE_PATTERN", "DATA_CONFUSION",
    "LEMMA_DATA_CONFUSION", "FAMILY_SEED", "AUTO_FAMILY_SEED",
    "FAMILY_SURFACE_EXAMPLE", "PHRASE_MEMORY", "PHRASE_MEMORY_FUZZY",
    "INFLECTION_RULE", "SURFACE_JAMO_FALLBACK",
}
RUNTIME_BROAD_LEMMA_TARGETS = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
RUNTIME_BROAD_LEMMA_ALLOWLIST = {("잇다", "있다"), ("허다", "하다"), ("돼다", "되다")}
RUNTIME_BROAD_LEMMA_SOURCES = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
MLM_SURFACE_TOP_K = 8
CANDIDATE_PREFILTER_LIMIT = 4
CHEAP_VERIFIER_LIMIT = 3
MLM_BACKOFF_PREFILTER_MIN = 0.8
MLM_BACKOFF_MIN_CANDIDATES = 2
FAST_PATH_VERIFIER_MIN = 0.82
FAST_PATH_MARGIN_MIN = 0.14
PROPOSAL_TOP_K = 3
PROPOSAL_BEAM_WIDTH = 8
SEQ2SEQ_SPAN_CONTEXT_TOKENS = 3
SEQ2SEQ_SPAN_MAX_TOKENS = 3
TOKEN_PATTERN = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")
META_COMPARE_TOKEN_PATTERN = re.compile(r"^[가-힣]{1,10}다(?:야|냐)$")
COMPACT_COMPOUND_TOKEN_PATTERNS = [
    re.compile(r"^(?:인사|말씀|부탁|문의|안내|연락|전달|보고|공유|설명|소개|추천|축하|사과|요청|전해|보내|도와|알려)드(?:리|릴|려|렸|립)[가-힣]*$"),
]

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


# ── 헬퍼: 후보 병합 ───────────────────────────────────────
def merge_candidates(
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


# ── 모델 로더 클래스 ──────────────────────────────────────
class ModelLoader:
    """ML 모델과 JSON 리소스를 모두 로딩하여 보관하는 컨테이너."""

    def __init__(self) -> None:
        torch.set_num_threads(1)

        # ── ML 모델 ──
        self.kiwi = Kiwi(typos="basic")
        self.tokenizer = get_tokenizer()
        self.kobert = get_kobert_model().eval()
        self.kobert_mlm = AutoModelForMaskedLM.from_pretrained("monologg/kobert-lm").eval()
        self.koelectra_tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
        self.koelectra = AutoModel.from_pretrained("monologg/koelectra-base-v3-discriminator").eval()

        # ── Edit Tagger ──
        self.edit_tagger_model = None
        self.edit_tagger_tokenizer = None
        self.edit_tagger_id2label: Dict[int, str] = {}
        self.edit_tagger_source = "heuristic"

        # ── Seq2Seq ──
        self.proposal_seq2seq_model = None
        self.proposal_seq2seq_tokenizer = None
        self.proposal_seq2seq_source = None

        # ── 캐시 ──
        self._kiwi_analysis_cache: Dict[Tuple[str, int], List[Tuple[Any, float]]] = {}

        # ── JSON 리소스 ──
        self.typed_confusion_resource = self._load_typed_confusion_resource()
        self.phrase_memory = self._load_phrase_memory()
        self.predicate_family_seeds = self._load_predicate_family_seeds()
        self.inflection_recovery_rules = self._load_inflection_recovery_rules()

        # ── 인덱스 빌드 ──
        self.rule_misspellings = self._load_surface_fix_rules()
        self.surface_fix_index = self._build_surface_fix_index(self.rule_misspellings)
        self.phrase_memory_index = self._build_phrase_memory_index()
        self.phrase_memory_target_index = self._build_phrase_memory_target_index()
        self.surface_confusion_map = self._build_surface_confusion_map()
        self.surface_fallback_index = self._build_surface_fallback_index()
        self.lemma_confusion_map = self._build_lemma_confusion_map()
        self.predicate_lemma_index = self._build_predicate_lemma_index()

        # ── 로컬 모델 로드 ──
        self._load_local_edit_tagger()
        self._load_local_proposal_seq2seq()

        # ── 프로필 프로토타입 ──
        self.profile_proto_emb = self._build_profile_prototypes(self.sentence_embedding)
        self.profile_proto_emb_electra = self._build_profile_prototypes(self.sentence_embedding_koelectra)

        # ── 워밍업 ──
        self._warmup_runtime()

    # ── 모델 로딩 ──────────────────────────────────────────

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

    @staticmethod
    def _has_generation_weights(model_dir: Path) -> bool:
        weight_names = (
            "model.safetensors", "pytorch_model.bin",
            "pytorch_model-00001-of-00002.bin", "pytorch_model-00001-of-00003.bin",
        )
        return any((model_dir / name).exists() for name in weight_names)

    @staticmethod
    def _is_valid_seq2seq_model_dir(model_dir: Path) -> bool:
        config_path = model_dir / "config.json"
        if not config_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        model_type = str(config.get("model_type") or "")
        architectures = [str(item) for item in (config.get("architectures") or [])]
        if bool(config.get("is_encoder_decoder")):
            return True
        if model_type in {"bart", "mbart", "t5"}:
            return True
        return any("ConditionalGeneration" in item or "Seq2Seq" in item for item in architectures)

    def _load_local_proposal_seq2seq(self) -> None:
        for model_dir in LOCAL_PROPOSAL_SEQ2SEQ_CANDIDATE_DIRS:
            if (
                not model_dir.exists()
                or not self._has_generation_weights(model_dir)
                or not self._is_valid_seq2seq_model_dir(model_dir)
            ):
                continue
            try:
                self.proposal_seq2seq_tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir))
                self.proposal_seq2seq_model = BartForConditionalGeneration.from_pretrained(str(model_dir)).eval()
                self.proposal_seq2seq_source = str(model_dir)
                return
            except Exception:
                self.proposal_seq2seq_model = None
                self.proposal_seq2seq_tokenizer = None
                self.proposal_seq2seq_source = None

    # ── JSON 리소스 ────────────────────────────────────────

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
                            "from": src, "to": dst,
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

    # ── 인덱스 빌드 ────────────────────────────────────────

    def _build_phrase_memory_index(self) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for item in self.phrase_memory.get("items", []):
            source = str(item.get("from") or "").strip()
            replacement = str(item.get("to") or "").strip()
            if not source or not replacement or source == replacement:
                continue
            parts = tokenize_with_ranges(source)
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
            target_parts = tokenize_with_ranges(target)
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

    def _build_surface_confusion_map(self) -> Dict[str, List[Dict[str, Any]]]:
        surface_graph: Dict[str, List[Dict[str, Any]]] = {}
        for surface, replacements in OPEN_REPLACE_MAP.items():
            surface_graph[surface] = list(replacements)
        for surface, replacements in (self.typed_confusion_resource.get("surface") or {}).items():
            existing = surface_graph.get(surface, [])
            surface_graph[surface] = merge_candidates(existing, replacements)
        return surface_graph

    def _build_surface_fallback_index(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        index: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        def register(source: str, candidates: List[Dict[str, Any]]) -> None:
            source = str(source or "").strip()
            if not source or " " in source or hangul_ratio(source) < 0.7:
                return
            head = source[0]
            bucket = index.setdefault(head, {})
            bucket[source] = merge_candidates(bucket.get(source, []), candidates)

        for source, candidates in self.surface_confusion_map.items():
            register(source, list(candidates))
        for source, rule in self.surface_fix_index.items():
            register(
                source,
                [{"replacement": rule["to"], "source": "CONFUSION_SET", "generatorScore": float(rule.get("confidence", 0.96))}],
            )
        return index

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
                similarity = float(payload.get("surfaceSimilarity") or typo_similarity(left, right))
                if left and right and left[0] != right[0] and similarity < 0.7:
                    return False
                if similarity < 0.58 and not payload.get("contextHints"):
                    return False
            return True

        def merge_hint_rows(left, right):
            merged: Dict[str, int] = {}
            for item in [*(left or []), *(right or [])]:
                term = str(item.get("term") or "").strip()
                if not term:
                    continue
                merged[term] = merged.get(term, 0) + int(item.get("count") or 0)
            return [{"term": term, "count": count} for term, count in sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))[:12]]

        def merge_surface_examples(left, right):
            merged: Dict[Tuple[str, str], int] = {}
            for item in [*(left or []), *(right or [])]:
                source = str(item.get("from") or "").strip()
                target = str(item.get("to") or "").strip()
                if not source or not target:
                    continue
                key = (source, target)
                merged[key] = merged.get(key, 0) + int(item.get("count") or 0)
            rows = [{"from": s, "to": t, "count": c} for (s, t), c in sorted(merged.items(), key=lambda p: (-p[1], p[0]))]
            return rows[:12]

        def merge_edge_payload(existing, incoming):
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
            suffix_slots = sorted({str(slot) for slot in [*(existing.get("suffixSlots") or []), *(incoming.get("suffixSlots") or [])] if slot})
            if suffix_slots:
                merged["suffixSlots"] = suffix_slots
            context_hints = merge_hint_rows(existing.get("contextHints"), incoming.get("contextHints"))
            if context_hints:
                merged["contextHints"] = context_hints
            surface_examples = merge_surface_examples(existing.get("surfaceExamples"), incoming.get("surfaceExamples"))
            if surface_examples:
                merged["surfaceExamples"] = surface_examples
            return merged

        def add_edge(left, right, source, score, extra=None):
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
                add_edge(lemma, target, replacement.get("source", "LEMMA_DATA_CONFUSION"), float(replacement.get("generatorScore", 0.0)), {
                    "frequency": replacement.get("frequency"), "surfaceSimilarity": replacement.get("surfaceSimilarity"),
                    "interfaceType": replacement.get("interfaceType"), "interfaceStats": replacement.get("interfaceStats"),
                    "errorTypes": replacement.get("errorTypes"), "sourceStats": replacement.get("sourceStats"),
                    "contextHints": replacement.get("contextHints"), "pos": replacement.get("pos"), "suffixSlots": replacement.get("suffixSlots"),
                })

        for lemma, replacements in (self.predicate_family_seeds.get("lemmas") or {}).items():
            for replacement in replacements:
                target = replacement["replacement"]
                add_edge(lemma, target, replacement.get("source", "FAMILY_SEED"), float(replacement.get("generatorScore", 0.0)), {
                    "frequency": replacement.get("frequency"), "surfaceSimilarity": replacement.get("surfaceSimilarity"),
                    "interfaceType": replacement.get("interfaceType"), "interfaceStats": replacement.get("interfaceStats"),
                    "errorTypes": replacement.get("errorTypes"), "sourceStats": replacement.get("sourceStats"),
                    "contextHints": replacement.get("contextHints"), "pos": replacement.get("pos"), "suffixSlots": replacement.get("suffixSlots"),
                })

        return {lemma: list(targets.values()) for lemma, targets in graph.items()}

    def _build_predicate_lemma_index(self) -> Dict[str, List[str]]:
        from utils.morpheme_utils import coarse_predicate_pos
        buckets: Dict[str, set] = {}

        def register(lemma, pos):
            normalized = str(lemma or "").strip()
            cp = coarse_predicate_pos(str(pos or ""))
            if not normalized or not normalized.endswith("다"):
                return
            if cp not in {"VV", "VA", "VX"}:
                return
            buckets.setdefault(cp, set()).add(normalized)

        for lemma, replacements in self.lemma_confusion_map.items():
            replacement_list = replacements or []
            if replacement_list:
                register(lemma, replacement_list[0].get("pos"))
            for replacement in replacement_list:
                register(replacement.get("replacement"), replacement.get("pos"))
        return {pos: sorted(values, key=lambda item: (len(item), item)) for pos, values in buckets.items()}

    # ── 임베딩 / 추론 헬퍼 ─────────────────────────────────

    def sentence_embedding(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        with torch.no_grad():
            vec = self.kobert(**inputs).last_hidden_state[:, 0, :].squeeze(0)
        return F.normalize(vec, dim=0)

    def sentence_embedding_koelectra(self, text: str) -> torch.Tensor:
        inputs = self.koelectra_tokenizer(text, return_tensors="pt", truncation=True, max_length=96)
        with torch.no_grad():
            vec = self.koelectra(**inputs).last_hidden_state[:, 0, :].squeeze(0)
        return F.normalize(vec, dim=0)

    def _build_profile_prototypes(self, emb_fn) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for profile, samples in PROFILE_PROTOTYPES.items():
            emb = torch.stack([emb_fn(s) for s in samples], dim=0).mean(dim=0)
            out[profile] = F.normalize(emb, dim=0)
        return out

    def kiwi_analyze_cached(self, text: str, top_n: int = 1) -> List[Tuple[Any, float]]:
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

    def kiwi_sentence_score(self, text: str) -> float:
        try:
            analyzed = self.kiwi_analyze_cached(text, top_n=1)
        except Exception:
            return -999.0
        if not analyzed:
            return -999.0
        return float(analyzed[0][1])

    def token_logit_in_mask(self, masked_sentence: str, token: str) -> float:
        inputs = self.tokenizer(masked_sentence, return_tensors="pt", truncation=True, max_length=96)
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

    def pseudo_logprob_for_replacement(self, sentence: str, start: int, end: int, replacement: str) -> float:
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

    def koelectra_normal_score(self, text: str) -> float:
        emb = self.sentence_embedding_koelectra(text)
        return float(torch.dot(emb, self.profile_proto_emb_electra["NORMAL"]).detach().item())

    def sentence_quality_baseline(self, sentence: str) -> Dict[str, float]:
        return {
            "koelectraScore": self.koelectra_normal_score(sentence),
            "kiwiScore": self.kiwi_sentence_score(sentence),
        }

    # ── 워밍업 ─────────────────────────────────────────────

    def _warmup_runtime(self) -> None:
        warm_sentence = "오늘 날씨가 좋다"
        try:
            self.kiwi.space(warm_sentence)
            self.kiwi_analyze_cached(warm_sentence, top_n=1)
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
                    ["테스트"], is_split_into_words=True, return_tensors="pt", truncation=True, max_length=16,
                )
                with torch.no_grad():
                    self.edit_tagger_model(**inputs)
            except Exception:
                pass
