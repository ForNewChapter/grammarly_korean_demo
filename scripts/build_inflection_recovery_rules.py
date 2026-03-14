#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_LEMMA_GRAPH = Path("public/assets/dict/lemma_family_graph.json")
DEFAULT_FAMILY_SEEDS = Path("public/assets/dict/predicate_family_seeds.json")
DEFAULT_OUTPUT = Path("public/assets/dict/inflection_recovery_rules.json")
PREDICATE_PREFIXES = ("VV", "VA", "VX")
BROAD_TARGETS = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
ALLOWLIST = {("잇다", "있다"), ("허다", "하다"), ("돼다", "되다")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build productive inflection recovery rules from family assets.")
    parser.add_argument("--lemma-graph", type=Path, default=DEFAULT_LEMMA_GRAPH)
    parser.add_argument("--family-seeds", type=Path, default=DEFAULT_FAMILY_SEEDS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-rules", type=int, default=160)
    return parser.parse_args()


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def stem(lemma: str) -> str:
    return lemma[:-1] if lemma.endswith("다") else lemma


def infer_irregular_class(lemma: str) -> str | None:
    if lemma.endswith("하다"):
        return "HA"
    if lemma.endswith("르다"):
        return "REU"
    if lemma.endswith("치다"):
        return "CHI"
    if lemma.endswith("추다"):
        return "CHU"
    if lemma.endswith("히다"):
        return "HI"
    if lemma.endswith("키다"):
        return "KI"
    if lemma.endswith("우다"):
        return "U"
    return None


def shared_prefix_length(left: str, right: str) -> int:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index


def top_counts(counter: Counter, limit: int) -> List[Dict[str, Any]]:
    return [{"term": key, "count": count} for key, count in counter.most_common(limit)]


def iter_family_edges(lemma_graph: Dict[str, Any], family_seeds: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for src, variants in (lemma_graph.get("lemmas") or {}).items():
        for variant in variants:
            yield src, dict(variant)
    for src, variants in (family_seeds.get("lemmas") or {}).items():
        for variant in variants:
            cloned = dict(variant)
            if "frequency" not in cloned:
                cloned["frequency"] = 6
            yield src, cloned


def should_keep_rule(src_lemma: str, dst_lemma: str, pos: str | None) -> bool:
    if not src_lemma or not dst_lemma or src_lemma == dst_lemma:
        return False
    if (src_lemma, dst_lemma) in ALLOWLIST:
        return True
    if dst_lemma in BROAD_TARGETS:
        return False
    if pos and not pos.startswith(PREDICATE_PREFIXES):
        return False
    return True


def extract_rule(src_lemma: str, dst_lemma: str, pos: str | None) -> Tuple[str, str] | None:
    src_stem = stem(src_lemma)
    dst_stem = stem(dst_lemma)
    if src_stem == dst_stem:
        return None
    prefix_len = shared_prefix_length(src_stem, dst_stem)
    src_suffix = src_stem[prefix_len:]
    dst_suffix = dst_stem[prefix_len:]
    if not src_suffix or not dst_suffix:
        return None
    if len(src_suffix) > 4 or len(dst_suffix) > 4:
        return None
    return src_suffix, dst_suffix


def build_rules(lemma_graph: Dict[str, Any], family_seeds: Dict[str, Any], min_support: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for src_lemma, variant in iter_family_edges(lemma_graph, family_seeds):
        dst_lemma = str(variant.get("replacement") or "").strip()
        pos = str(variant.get("pos") or "")
        if not should_keep_rule(src_lemma, dst_lemma, pos):
            continue
        extracted = extract_rule(src_lemma, dst_lemma, pos)
        if not extracted:
            continue
        src_suffix, dst_suffix = extracted
        key = (src_suffix, dst_suffix, pos or "")
        entry = grouped.setdefault(
            key,
            {
                "sourceSuffix": src_suffix,
                "targetSuffix": dst_suffix,
                "pos": pos or None,
                "frequency": 0,
                "sourceLemmaExamples": Counter(),
                "targetLemmaExamples": Counter(),
                "suffixSlots": Counter(),
                "contextHints": Counter(),
                "interfaceStats": Counter(),
                "sourceStats": Counter(),
                "irregularClasses": Counter(),
            },
        )
        weight = int(variant.get("frequency") or 1)
        entry["frequency"] += weight
        entry["sourceLemmaExamples"][src_lemma] += weight
        entry["targetLemmaExamples"][dst_lemma] += weight
        for slot in variant.get("suffixSlots") or []:
            entry["suffixSlots"][str(slot)] += weight
        for hint in variant.get("contextHints") or []:
            term = str(hint.get("term") or "").strip()
            if term:
                entry["contextHints"][term] += int(hint.get("count") or 0)
        for channel, count in (variant.get("interfaceStats") or {}).items():
            entry["interfaceStats"][str(channel)] += int(count or 0)
        for source_name, count in (variant.get("sourceStats") or {}).items():
            entry["sourceStats"][str(source_name)] += int(count or 0)
        irregular = infer_irregular_class(dst_lemma) or infer_irregular_class(src_lemma)
        if irregular:
            entry["irregularClasses"][irregular] += weight

    rows: List[Dict[str, Any]] = []
    for entry in grouped.values():
        if entry["frequency"] < min_support:
            continue
        support = entry["frequency"]
        irregular_classes = [name for name, _ in entry["irregularClasses"].most_common(2)]
        rows.append(
            {
                "sourceSuffix": entry["sourceSuffix"],
                "targetSuffix": entry["targetSuffix"],
                "pos": entry["pos"],
                "frequency": support,
                "generatorScore": round(min(0.965, 0.58 + (0.07 * min(5, support)) / 5 + 0.03 * len(irregular_classes)), 4),
                "suffixSlots": [name for name, _ in entry["suffixSlots"].most_common(8)],
                "contextHints": top_counts(entry["contextHints"], 8),
                "interfaceStats": dict(entry["interfaceStats"]),
                "sourceStats": dict(entry["sourceStats"]),
                "irregularClasses": irregular_classes,
                "sourceLemmaExamples": [
                    {"lemma": lemma, "count": count}
                    for lemma, count in entry["sourceLemmaExamples"].most_common(6)
                ],
                "targetLemmaExamples": [
                    {"lemma": lemma, "count": count}
                    for lemma, count in entry["targetLemmaExamples"].most_common(6)
                ],
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["frequency"]),
            -float(row["generatorScore"]),
            row["sourceSuffix"],
            row["targetSuffix"],
        )
    )
    return rows


def main() -> None:
    args = parse_args()
    lemma_graph = load_json(args.lemma_graph, {"lemmas": {}})
    family_seeds = load_json(args.family_seeds, {"lemmas": {}})
    rules = build_rules(lemma_graph, family_seeds, args.min_support)
    payload = {
        "generatedFrom": [str(args.lemma_graph), str(args.family_seeds)],
        "count": min(len(rules), args.max_rules),
        "rules": rules[: args.max_rules],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": payload["count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
