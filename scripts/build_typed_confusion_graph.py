#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from kiwipiepy import Kiwi

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")
BROAD_LEMMA_BLACKLIST = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
BROAD_LEMMA_ALLOWLIST = {
    ("잇다", "있다"),
    ("허다", "하다"),
    ("돼다", "되다"),
}


def normalize_surface(text: str) -> str:
    return text.strip()


def strip_token_noise(text: str) -> str:
    return re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", text)


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def char_edit_distance(a: str, b: str) -> int:
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
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


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
    edit = char_edit_distance(left, right)
    edit_component = max(0.0, 1.0 - (edit / max(len(left), len(right))))
    return round((0.55 * preserved) + (0.45 * edit_component), 4)


def infer_interface(record: Dict[str, object]) -> str:
    error_type = str(record.get("error_type") or "")
    source = str(record.get("source") or "")
    interface = str(record.get("interface") or "")
    keyboard = str(record.get("keyboard") or "")
    if "음성" in error_type or source.startswith("STT_ENGINE") or "voice" in interface.lower():
        return "voice"
    if keyboard and keyboard.lower() != "none":
        return "keyboard"
    return "generic"


def coarse_pos(tag: Optional[str]) -> Optional[str]:
    if not tag:
        return None
    return tag.split("-")[0]


def extract_context_terms(sentence: str, target: str, window: int = 2) -> List[str]:
    cleaned_target = strip_token_noise(target)
    if not cleaned_target:
        return []
    tokens = TOKEN_RE.findall(sentence)
    if not tokens:
        return []
    lowered = [strip_token_noise(token) for token in tokens]
    indices = [idx for idx, token in enumerate(lowered) if token == cleaned_target]
    if not indices:
        indices = [idx for idx, token in enumerate(lowered) if cleaned_target in token or token in cleaned_target]
    if not indices:
        return []
    idx = indices[0]
    out: List[str] = []
    for cursor in range(max(0, idx - window), min(len(tokens), idx + window + 1)):
        if cursor == idx:
            continue
        token = strip_token_noise(tokens[cursor])
        if len(token) < 2 or not has_hangul(token):
            continue
        out.append(token)
    return out


@dataclass
class PairStats:
    err: str
    cor: str
    freq: int = 0
    interfaces: Counter = field(default_factory=Counter)
    error_types: Counter = field(default_factory=Counter)
    sources: Counter = field(default_factory=Counter)
    context_terms: Counter = field(default_factory=Counter)

    def observe(self, record: Dict[str, object], err_text: str, cor_text: str) -> None:
        self.freq += 1
        self.interfaces[infer_interface(record)] += 1
        self.error_types[str(record.get("error_type") or "unknown")] += 1
        self.sources[str(record.get("source") or "unknown")] += 1
        for token in extract_context_terms(str(record.get("cor_sentence") or ""), cor_text):
            self.context_terms[token] += 1


@dataclass
class MorphInfo:
    lemma: Optional[str]
    pos: Optional[str]
    suffix_slot: Optional[str]


def analyze_token(kiwi: Kiwi, token: str, cache: Dict[str, MorphInfo]) -> MorphInfo:
    if token in cache:
        return cache[token]
    cleaned = strip_token_noise(token)
    if not cleaned:
        cache[token] = MorphInfo(None, None, None)
        return cache[token]
    try:
        analyses = kiwi.analyze(cleaned, top_n=3)
    except Exception:
        analyses = []
    if not analyses:
        cache[token] = MorphInfo(cleaned, None, None)
        return cache[token]
    best_predicate: Optional[MorphInfo] = None
    best_fallback: Optional[MorphInfo] = None
    for morphs, _ in analyses:
        if not morphs:
            continue
        predicate_index = None
        for idx, morph in enumerate(morphs):
            if morph.tag.startswith(("VV", "VA", "VX")):
                predicate_index = idx
                break
        if predicate_index is not None:
            predicate = morphs[predicate_index]
            suffix_slot = "+".join(m.tag for m in morphs[predicate_index + 1 :]) or None
            best_predicate = MorphInfo(predicate.lemma or cleaned, predicate.tag, suffix_slot)
            break
        if best_fallback is None:
            head = morphs[0]
            best_fallback = MorphInfo(head.lemma or cleaned, head.tag, None)
    if best_predicate is not None:
        cache[token] = best_predicate
        return cache[token]
    if best_fallback is not None:
        cache[token] = best_fallback
        return cache[token]
    cache[token] = MorphInfo(cleaned, None, None)
    return cache[token]


def score_from_freq(freq: int, similarity: float) -> float:
    return round(float(min(0.985, 0.5 + (math.log1p(freq) / 12.0) + (0.28 * similarity))), 4)


def dominant(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def top_counts(counter: Counter, limit: int) -> List[Dict[str, object]]:
    return [{"term": key, "count": count} for key, count in counter.most_common(limit)]


def build_surface_stats(records: Iterable[Dict[str, object]]) -> Dict[Tuple[str, str], PairStats]:
    pairs: Dict[Tuple[str, str], PairStats] = {}
    for record in records:
        for error in record.get("errors", []):
            err_text = normalize_surface(str(error.get("err_text") or ""))
            cor_text = normalize_surface(str(error.get("cor_text") or ""))
            if not err_text or not cor_text or err_text == cor_text:
                continue
            if " " in err_text or " " in cor_text:
                continue
            if not has_hangul(err_text + cor_text):
                continue
            if len(err_text) > 16 or len(cor_text) > 16:
                continue
            if strip_token_noise(err_text) == strip_token_noise(cor_text):
                continue
            similarity = surface_similarity(err_text, cor_text)
            max_edits = 2 if max(len(err_text), len(cor_text)) <= 6 else 3
            if char_edit_distance(strip_token_noise(err_text), strip_token_noise(cor_text)) > max_edits:
                continue
            if similarity < 0.45:
                continue
            key = (err_text, cor_text)
            item = pairs.get(key)
            if item is None:
                item = pairs[key] = PairStats(err=err_text, cor=cor_text)
            item.observe(record, err_text, cor_text)
    return pairs


def finalize_surface_entries(
    surface_pairs: Dict[Tuple[str, str], PairStats],
    kiwi: Kiwi,
    min_freq: int,
    max_per_key: int,
    max_context_hints: int,
) -> Tuple[Dict[str, List[Dict[str, object]]], Dict[str, List[Dict[str, object]]]]:
    morph_cache: Dict[str, MorphInfo] = {}
    surface_graph: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    lemma_graph_raw: Dict[Tuple[str, str], Dict[str, object]] = {}

    for (err_text, cor_text), stats in surface_pairs.items():
        if stats.freq < min_freq:
            continue
        err_info = analyze_token(kiwi, err_text, morph_cache)
        cor_info = analyze_token(kiwi, cor_text, morph_cache)
        similarity = surface_similarity(err_text, cor_text)
        entry = {
            "replacement": cor_text,
            "source": "DATA_CONFUSION",
            "generatorScore": score_from_freq(stats.freq, similarity),
            "frequency": stats.freq,
            "surfaceSimilarity": similarity,
            "interfaceType": dominant(stats.interfaces) or "generic",
            "interfaceStats": dict(stats.interfaces),
            "errorTypes": dict(stats.error_types),
            "sourceStats": dict(stats.sources),
            "contextHints": top_counts(stats.context_terms, max_context_hints),
            "lemma": err_info.lemma,
            "targetLemma": cor_info.lemma,
            "pos": cor_info.pos or err_info.pos,
            "suffixSlot": cor_info.suffix_slot or err_info.suffix_slot,
        }
        surface_graph[err_text].append(entry)

        if err_info.lemma and cor_info.lemma and err_info.lemma != cor_info.lemma:
            err_pos = coarse_pos(err_info.pos)
            cor_pos = coarse_pos(cor_info.pos)
            if err_pos not in {"VV", "VA", "VX"} or cor_pos not in {"VV", "VA", "VX"}:
                continue
            if err_pos != cor_pos and {err_pos, cor_pos} != {"VA", "VX"}:
                continue
            if (
                err_info.lemma in BROAD_LEMMA_BLACKLIST
                or cor_info.lemma in BROAD_LEMMA_BLACKLIST
            ) and (err_info.lemma, cor_info.lemma) not in BROAD_LEMMA_ALLOWLIST:
                continue
            voice_ratio = stats.interfaces.get("voice", 0) / max(1, sum(stats.interfaces.values()))
            if similarity < 0.56 and voice_ratio < 0.5:
                continue
            if voice_ratio >= 0.5 and stats.freq < max(min_freq + 2, 5):
                continue
            lemma_key = (err_info.lemma, cor_info.lemma)
            aggregated = lemma_graph_raw.get(lemma_key)
            if aggregated is None:
                aggregated = lemma_graph_raw[lemma_key] = {
                    "replacement": cor_info.lemma,
                    "source": "LEMMA_DATA_CONFUSION",
                    "frequency": 0,
                    "interfaceStats": Counter(),
                    "errorTypes": Counter(),
                    "sourceStats": Counter(),
                    "contextHints": Counter(),
                    "pos": cor_info.pos or err_info.pos,
                    "suffixSlots": Counter(),
                    "surfaceSimilarities": [],
                }
            aggregated["frequency"] += stats.freq
            aggregated["interfaceStats"].update(stats.interfaces)
            aggregated["errorTypes"].update(stats.error_types)
            aggregated["sourceStats"].update(stats.sources)
            aggregated["contextHints"].update(stats.context_terms)
            aggregated["surfaceSimilarities"].append(similarity)
            if cor_info.suffix_slot:
                aggregated["suffixSlots"][cor_info.suffix_slot] += stats.freq

    for key in list(surface_graph):
        surface_graph[key].sort(
            key=lambda item: (
                item["frequency"],
                item["generatorScore"],
                item["replacement"],
            ),
            reverse=True,
        )
        surface_graph[key] = surface_graph[key][:max_per_key]

    lemma_graph: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for (err_lemma, _), aggregated in lemma_graph_raw.items():
        freq = int(aggregated["frequency"])
        avg_similarity = round(
            sum(aggregated["surfaceSimilarities"]) / max(1, len(aggregated["surfaceSimilarities"])),
            4,
        )
        lemma_graph[err_lemma].append(
            {
                "replacement": aggregated["replacement"],
                "source": aggregated["source"],
                "generatorScore": score_from_freq(freq, avg_similarity),
                "frequency": freq,
                "surfaceSimilarity": avg_similarity,
                "interfaceType": dominant(aggregated["interfaceStats"]) or "generic",
                "interfaceStats": dict(aggregated["interfaceStats"]),
                "errorTypes": dict(aggregated["errorTypes"]),
                "sourceStats": dict(aggregated["sourceStats"]),
                "contextHints": top_counts(aggregated["contextHints"], max_context_hints),
                "pos": aggregated["pos"],
                "suffixSlots": [item[0] for item in aggregated["suffixSlots"].most_common(6)],
            }
        )

    for key in list(lemma_graph):
        lemma_graph[key].sort(
            key=lambda item: (item["frequency"], item["generatorScore"], item["replacement"]),
            reverse=True,
        )
        lemma_graph[key] = lemma_graph[key][:max_per_key]

    return dict(surface_graph), dict(lemma_graph)


def stream_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact typed confusion graph for candidate generation.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/aihub_highfreq/train.jsonl"),
        help="Normalized AI Hub JSONL file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/assets/dict/typed_confusion_graph.json"),
        help="Output JSON file",
    )
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--max-surface-per-key", type=int, default=8)
    parser.add_argument("--max-lemma-per-key", type=int, default=8)
    parser.add_argument("--max-context-hints", type=int, default=8)
    args = parser.parse_args()

    kiwi = Kiwi(typos="basic")
    surface_pairs = build_surface_stats(stream_jsonl(args.input))
    surface_graph, lemma_graph = finalize_surface_entries(
        surface_pairs=surface_pairs,
        kiwi=kiwi,
        min_freq=args.min_freq,
        max_per_key=max(args.max_surface_per_key, args.max_lemma_per_key),
        max_context_hints=args.max_context_hints,
    )

    payload = {
        "version": 1,
        "generatedFrom": str(args.input),
        "minFreq": args.min_freq,
        "surfaceKeyCount": len(surface_graph),
        "lemmaKeyCount": len(lemma_graph),
        "surface": surface_graph,
        "lemma": lemma_graph,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "surfaceKeys": len(surface_graph), "lemmaKeys": len(lemma_graph)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
