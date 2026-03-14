#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from kiwipiepy import Kiwi


DEFAULT_INPUT_DIR = Path("./data/aihub_highfreq")
DEFAULT_OUTPUT = Path("./public/assets/dict/lemma_family_graph.json")

PREDICATE_PREFIXES = ("VV", "VA", "VX")
BROAD_LEMMA_BLACKLIST = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
BROAD_LEMMA_ALLOWLIST = {
    ("잇다", "있다"),
    ("허다", "하다"),
    ("돼다", "되다"),
}


@dataclass
class FamilyEdge:
    src_lemma: str
    dst_lemma: str
    pos: Optional[str]
    freq: int = 0
    interfaces: Counter = field(default_factory=Counter)
    error_types: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    context_terms: Counter = field(default_factory=Counter)
    suffix_slots: Counter = field(default_factory=Counter)
    surfaces: Counter = field(default_factory=Counter)

    def observe(
        self,
        record: Dict[str, Any],
        err_surface: str,
        cor_surface: str,
        interface_type: str,
        error_type: str,
        source_name: str,
        context_terms: List[str],
        suffix_slot: Optional[str],
    ) -> None:
        self.freq += 1
        self.interfaces[interface_type] += 1
        self.error_types[error_type] += 1
        self.sources[source_name] += 1
        self.surfaces[f"{err_surface}\t{cor_surface}"] += 1
        if suffix_slot:
            self.suffix_slots[suffix_slot] += 1
        for term in context_terms:
            self.context_terms[term] += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Hub 오류 데이터에서 lemma family graph를 생성합니다.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-replacements", type=int, default=12)
    parser.add_argument("--max-context-hints", type=int, default=8)
    parser.add_argument("--max-surface-examples", type=int, default=6)
    return parser.parse_args()


def normalize_surface(text: str) -> str:
    return str(text or "").strip()


def strip_token_noise(text: str) -> str:
    import re

    return re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", text)


def has_hangul(text: str) -> bool:
    import re

    return bool(re.search(r"[가-힣]", text))


def iter_jsonl_records(input_dir: Path) -> Iterable[Dict[str, Any]]:
    for name in ("train.jsonl", "validation.jsonl"):
        path = input_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def infer_interface(record: Dict[str, Any]) -> str:
    error_type = str(record.get("error_type") or "")
    source = str(record.get("source") or "")
    interface = str(record.get("interface") or "")
    keyboard = str(record.get("keyboard") or "")
    if "음성" in error_type or source.startswith("STT_ENGINE") or "voice" in interface.lower():
        return "voice"
    if keyboard and keyboard.lower() != "none":
        return "keyboard"
    return "generic"


def top_counts(counter: Counter, limit: int, label: str = "term") -> List[Dict[str, Any]]:
    return [{label: key, "count": count} for key, count in counter.most_common(limit)]


def normalize_context_token(token: str) -> str:
    token = strip_token_noise(token)
    if not token:
        return token
    import re

    if not re.fullmatch(r"[가-힣]+", token):
        return token
    particles = (
        "으로는",
        "에게서",
        "한테서",
        "이라도",
        "으로",
        "에게",
        "한테",
        "에서",
        "부터",
        "까지",
        "처럼",
        "보다",
        "이랑",
        "이나",
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
        if token.endswith(particle) and len(token) > len(particle) + 1:
            return token[: -len(particle)]
    return token


def extract_context_terms(sentence: str, target: str, window: int = 2) -> List[str]:
    import re

    cleaned_target = strip_token_noise(target)
    if not cleaned_target:
        return []
    tokens = re.findall(r"[가-힣A-Za-z0-9]+|[^\s]", sentence)
    cleaned = [normalize_context_token(token) for token in tokens]
    indices = [idx for idx, token in enumerate(cleaned) if token == cleaned_target]
    if not indices:
        indices = [idx for idx, token in enumerate(cleaned) if cleaned_target in token or token in cleaned_target]
    if not indices:
        return []
    center = indices[0]
    out: List[str] = []
    for idx in range(max(0, center - window), min(len(tokens), center + window + 1)):
        if idx == center:
            continue
        token = normalize_context_token(tokens[idx])
        if len(token) < 2 or not has_hangul(token):
            continue
        out.append(token)
    return out


def coarse_pos(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    return tag.split("-")[0]


def surface_similarity(left: str, right: str) -> float:
    left = strip_token_noise(left)
    right = strip_token_noise(right)
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(left) - prefix, len(right) - prefix)
        and left[len(left) - 1 - suffix] == right[len(right) - 1 - suffix]
    ):
        suffix += 1
    preserved = (prefix + suffix) / max(len(left), len(right))
    return round(preserved, 4)


def score_from_stats(freq: int, similarity: float) -> float:
    import math

    return round(float(min(0.985, 0.56 + (math.log1p(freq) / 10.0) + (0.22 * similarity))), 4)


class MorphAnalyzer:
    def __init__(self) -> None:
        self.kiwi = Kiwi(typos="basic")
        self.cache: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {}

    def analyze(self, token: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        cleaned = strip_token_noise(token)
        if cleaned in self.cache:
            return self.cache[cleaned]
        if not cleaned:
            self.cache[cleaned] = (None, None, None)
            return self.cache[cleaned]
        try:
            analyses = self.kiwi.analyze(cleaned, top_n=3)
        except Exception:
            analyses = []
        best: Tuple[Optional[str], Optional[str], Optional[str]] = (cleaned, None, None)
        for morphs, _ in analyses:
            if not morphs:
                continue
            predicate_idx = None
            for idx, morph in enumerate(morphs):
                if morph.tag.startswith(PREDICATE_PREFIXES):
                    predicate_idx = idx
                    break
            if predicate_idx is None:
                continue
            predicate = morphs[predicate_idx]
            suffix_slot = "+".join(m.tag for m in morphs[predicate_idx + 1 :]) or None
            best = (predicate.lemma or cleaned, predicate.tag, suffix_slot)
            break
        self.cache[cleaned] = best
        return best


def build_family_edges(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], FamilyEdge]:
    analyzer = MorphAnalyzer()
    edges: Dict[Tuple[str, str], FamilyEdge] = {}
    for record in records:
        error_type = str(record.get("error_type") or "unknown")
        source_name = str(record.get("source") or "unknown")
        interface_type = infer_interface(record)
        cor_sentence = str(record.get("cor_sentence") or "")
        for error in record.get("errors") or []:
            err_surface = normalize_surface(error.get("err_text"))
            cor_surface = normalize_surface(error.get("cor_text"))
            if not err_surface or not cor_surface or err_surface == cor_surface:
                continue
            if " " in err_surface or " " in cor_surface:
                continue
            if len(err_surface) > 24 or len(cor_surface) > 24:
                continue
            if not has_hangul(err_surface + cor_surface):
                continue
            src_lemma, src_pos, src_slot = analyzer.analyze(err_surface)
            dst_lemma, dst_pos, dst_slot = analyzer.analyze(cor_surface)
            src_coarse = coarse_pos(src_pos)
            dst_coarse = coarse_pos(dst_pos)
            if not src_lemma or not dst_lemma or src_lemma == dst_lemma:
                continue
            if src_coarse not in PREDICATE_PREFIXES or dst_coarse not in PREDICATE_PREFIXES:
                continue
            if src_coarse != dst_coarse:
                continue
            if (
                src_lemma in BROAD_LEMMA_BLACKLIST
                or dst_lemma in BROAD_LEMMA_BLACKLIST
            ) and (src_lemma, dst_lemma) not in BROAD_LEMMA_ALLOWLIST:
                continue
            context_terms = extract_context_terms(cor_sentence, cor_surface)
            key = (src_lemma, dst_lemma)
            item = edges.get(key)
            if item is None:
                item = edges[key] = FamilyEdge(src_lemma=src_lemma, dst_lemma=dst_lemma, pos=dst_coarse)
            item.observe(
                record=record,
                err_surface=err_surface,
                cor_surface=cor_surface,
                interface_type=interface_type,
                error_type=error_type,
                source_name=source_name,
                context_terms=context_terms,
                suffix_slot=dst_slot or src_slot,
            )
    return edges


def finalize(edges: Dict[Tuple[str, str], FamilyEdge], min_freq: int, max_replacements: int, max_context_hints: int, max_surface_examples: int) -> Dict[str, Any]:
    lemma_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for (src_lemma, dst_lemma), edge in edges.items():
        if edge.freq < min_freq:
            continue
        similarity = max(
            [surface_similarity(src_lemma, dst_lemma)]
            + [surface_similarity(pair.split("\t", 1)[0], pair.split("\t", 1)[1]) for pair in edge.surfaces]
        )
        dominant_interface = edge.interfaces.most_common(1)[0][0] if edge.interfaces else "generic"
        if similarity < 0.52:
            continue
        if edge.freq <= 2 and similarity < 0.62:
            continue
        if dominant_interface == "voice" and similarity < 0.72:
            continue
        if src_lemma and dst_lemma and src_lemma[0] != dst_lemma[0] and similarity < 0.7:
            continue
        if similarity < 0.58 and not edge.context_terms:
            continue
        entry = {
            "replacement": dst_lemma,
            "source": "LEMMA_FAMILY_GRAPH",
            "generatorScore": score_from_stats(edge.freq, similarity),
            "frequency": edge.freq,
            "surfaceSimilarity": similarity,
            "interfaceType": dominant_interface,
            "interfaceStats": dict(edge.interfaces),
            "errorTypes": dict(edge.error_types),
            "sourceStats": dict(edge.sources),
            "contextHints": top_counts(edge.context_terms, max_context_hints),
            "pos": edge.pos,
            "suffixSlots": [key for key, _ in edge.suffix_slots.most_common(8)],
            "surfaceExamples": [
                {"from": pair.split("\t", 1)[0], "to": pair.split("\t", 1)[1], "count": count}
                for pair, count in edge.surfaces.most_common(max_surface_examples)
            ],
        }
        lemma_map[src_lemma].append(entry)

    for lemma, replacements in lemma_map.items():
        replacements.sort(
            key=lambda item: (
                item["generatorScore"],
                item.get("frequency", 0),
                item.get("surfaceSimilarity", 0.0),
            ),
            reverse=True,
        )
        lemma_map[lemma] = replacements[:max_replacements]

    return {
        "generatedFrom": "aihub_highfreq_jsonl",
        "lemmaKeyCount": len(lemma_map),
        "lemmas": lemma_map,
    }


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output = args.output.resolve()
    edges = build_family_edges(iter_jsonl_records(input_dir))
    payload = finalize(
        edges,
        min_freq=args.min_freq,
        max_replacements=args.max_replacements,
        max_context_hints=args.max_context_hints,
        max_surface_examples=args.max_surface_examples,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "lemmaKeyCount": payload["lemmaKeyCount"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
