#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List

FAMILY_SEEDS = {
    "낫다": [
        {
            "replacement": "낳다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.95,
            "pos": "VV",
            "contextHints": [
                {"term": "아기", "count": 6},
                {"term": "아이", "count": 4},
                {"term": "출산", "count": 3},
                {"term": "임신", "count": 2},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 10},
            "suffixSlots": ["ETM", "EP+EF", "EP+EC", "EC", "EF", "ETM+JKB"],
        }
    ],
    "낳다": [
        {
            "replacement": "낫다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.95,
            "pos": "VV",
            "contextHints": [
                {"term": "감기", "count": 6},
                {"term": "병", "count": 5},
                {"term": "상처", "count": 4},
                {"term": "통증", "count": 3},
                {"term": "몸", "count": 2},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 10},
            "suffixSlots": ["ETM", "EP+EF", "EP+EC", "EC", "EF", "ETM+JKB"],
        }
    ],
    "가르다": [
        {
            "replacement": "가르치다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.93,
            "pos": "VV",
            "contextHints": [
                {"term": "답", "count": 5},
                {"term": "선생님", "count": 5},
                {"term": "학생", "count": 3},
                {"term": "아이", "count": 3},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 10},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
        }
    ],
    "가르키다": [
        {
            "replacement": "가르치다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.97,
            "pos": "VV",
            "contextHints": [
                {"term": "답", "count": 5},
                {"term": "선생님", "count": 5},
                {"term": "학생", "count": 3},
                {"term": "아이", "count": 3},
            ],
            "interfaceType": "keyboard",
            "interfaceStats": {"generic": 4, "keyboard": 6},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
        },
        {
            "replacement": "가리키다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.94,
            "pos": "VV",
            "contextHints": [
                {"term": "이름", "count": 5},
                {"term": "손가락", "count": 4},
                {"term": "방향", "count": 4},
                {"term": "위치", "count": 3},
                {"term": "곳", "count": 3},
                {"term": "친구", "count": 2},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 7, "keyboard": 3},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
        },
    ],
    "가르치다": [
        {
            "replacement": "가르키다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.72,
            "pos": "VV",
            "contextHints": [],
            "interfaceType": "keyboard",
            "interfaceStats": {"keyboard": 10},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
        }
    ],
    "맞추다": [
        {
            "replacement": "맞히다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.95,
            "pos": "VV",
            "contextHints": [
                {"term": "문제", "count": 6},
                {"term": "정답", "count": 5},
                {"term": "답", "count": 5},
                {"term": "시험", "count": 4},
                {"term": "문항", "count": 3},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 9, "keyboard": 4},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
            "surfaceExamples": [
                {"from": "맞추고", "to": "맞히고", "count": 4},
                {"from": "맞추었다", "to": "맞혔다", "count": 5},
                {"from": "맞췄다", "to": "맞혔다", "count": 6},
            ],
        }
    ],
    "맞히다": [
        {
            "replacement": "맞추다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.93,
            "pos": "VV",
            "contextHints": [
                {"term": "시간", "count": 6},
                {"term": "기준", "count": 4},
                {"term": "온도", "count": 3},
                {"term": "호흡", "count": 3},
                {"term": "간격", "count": 3},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 9, "keyboard": 4},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
            "surfaceExamples": [
                {"from": "맞히고", "to": "맞추고", "count": 4},
                {"from": "맞혀서", "to": "맞춰서", "count": 5},
                {"from": "맞히다", "to": "맞추다", "count": 3},
            ],
        }
    ],
    "낮다": [
        {
            "replacement": "낫다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.95,
            "pos": "VA",
            "contextHints": [
                {"term": "병", "count": 6},
                {"term": "감기", "count": 6},
                {"term": "상처", "count": 4},
                {"term": "통증", "count": 3},
                {"term": "약", "count": 3},
                {"term": "몸", "count": 2},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 10},
            "suffixSlots": ["EP+EF", "EF", "ETM", "EC"],
        }
    ],
    "낮아지다": [
        {
            "replacement": "나아지다",
            "source": "FAMILY_SEED",
            "generatorScore": 0.96,
            "pos": "VV",
            "contextHints": [
                {"term": "몸", "count": 5},
                {"term": "상태", "count": 5},
                {"term": "병", "count": 4},
                {"term": "감기", "count": 4},
                {"term": "통증", "count": 3},
                {"term": "약", "count": 3},
            ],
            "interfaceType": "generic",
            "interfaceStats": {"generic": 10},
            "suffixSlots": ["EP+EF", "EP+EC", "EC", "EF", "ETM"],
            "surfaceExamples": [
                {"from": "낮아졌다", "to": "나아졌다", "count": 6},
                {"from": "낮아졌으면", "to": "나아졌으면", "count": 5},
            ],
        }
    ],
}

LEMMA_FAMILY_GRAPH = Path("public/assets/dict/lemma_family_graph.json")
OUTPUT = Path("public/assets/dict/predicate_family_seeds.json")
PREDICATE_POS_PREFIXES = ("VV", "VA", "VX")
BROAD_VOICE_BLACKLIST = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}
BROAD_TARGET_ALLOWLIST = {
    ("잇다", "있다"),
    ("허다", "하다"),
    ("돼다", "되다"),
}
BROAD_SOURCE_BLACKLIST = {"하다", "되다", "가다", "나다", "나오다", "알다", "있다"}


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


def hangul_parts(char: str):
    if len(char) != 1:
        return None
    code = ord(char)
    if code < 0xAC00 or code > 0xD7A3:
        return None
    offset = code - 0xAC00
    return (offset // 588, (offset % 588) // 28, offset % 28)


def char_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    pa, pb = hangul_parts(a), hangul_parts(b)
    if pa and pb:
        return sum(1 for x, y in zip(pa, pb) if x == y) / 3.0
    return 0.0


def typo_similarity(original: str, candidate: str) -> float:
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
                pair_scores.append(char_similarity(left, right))
            else:
                pair_scores.append(0.0)
        core_similarity = sum(pair_scores) / pair_count
    return round(float((0.55 * preserved) + (0.45 * core_similarity)), 4)


def merge_hint_rows(left: List[Dict], right: List[Dict]) -> List[Dict]:
    merged: Dict[str, int] = {}
    for item in [*left, *right]:
        term = str(item.get("term") or "").strip()
        if not term:
            continue
        merged[term] = merged.get(term, 0) + int(item.get("count") or 0)
    return [
        {"term": term, "count": count}
        for term, count in sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))[:12]
    ]


def merge_surface_examples(left: List[Dict], right: List[Dict]) -> List[Dict]:
    merged: Dict[tuple[str, str], int] = {}
    for item in [*left, *right]:
        source = str(item.get("from") or "").strip()
        target = str(item.get("to") or "").strip()
        if not source or not target:
            continue
        key = (source, target)
        merged[key] = merged.get(key, 0) + int(item.get("count") or 0)
    return [
        {"from": source, "to": target, "count": count}
        for (source, target), count in sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))[:12]
    ]


def merge_rows(left: List[Dict], right: List[Dict]) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for item in [*left, *right]:
        key = item["replacement"]
        existing = merged.get(key)
        if existing is None:
            merged[key] = dict(item)
            continue
        if float(item.get("generatorScore", 0.0)) >= float(existing.get("generatorScore", 0.0)):
            for field in ("source", "generatorScore", "frequency", "surfaceSimilarity", "interfaceType", "pos", "typoSimilarity"):
                value = item.get(field)
                if value not in (None, "", []):
                    existing[field] = value
        interface_stats = dict(existing.get("interfaceStats") or {})
        for channel, count in (item.get("interfaceStats") or {}).items():
            interface_stats[channel] = interface_stats.get(channel, 0) + int(count or 0)
        if interface_stats:
            existing["interfaceStats"] = interface_stats
        suffix_slots = sorted({*map(str, existing.get("suffixSlots") or []), *map(str, item.get("suffixSlots") or [])})
        if suffix_slots:
            existing["suffixSlots"] = suffix_slots
        context_hints = merge_hint_rows(existing.get("contextHints") or [], item.get("contextHints") or [])
        if context_hints:
            existing["contextHints"] = context_hints
        surface_examples = merge_surface_examples(existing.get("surfaceExamples") or [], item.get("surfaceExamples") or [])
        if surface_examples:
            existing["surfaceExamples"] = surface_examples
        merged[key] = existing
    rows = list(merged.values())
    rows.sort(key=lambda item: (-float(item.get("generatorScore", 0.0)), item["replacement"]))
    return rows


def should_keep_auto_edge(src_lemma: str, edge: Dict) -> bool:
    replacement = str(edge.get("replacement") or "")
    if not replacement or replacement == src_lemma:
        return False
    if not has_hangul(src_lemma + replacement):
        return False
    if src_lemma in BROAD_SOURCE_BLACKLIST and (src_lemma, replacement) not in BROAD_TARGET_ALLOWLIST:
        return False
    pos = str(edge.get("pos") or "")
    if not pos.startswith(PREDICATE_POS_PREFIXES):
        return False
    freq = int(edge.get("frequency") or 0)
    if freq < 2:
        return False
    sim = typo_similarity(src_lemma, replacement)
    edit = char_edit_distance(src_lemma, replacement)
    if sim < 0.48:
        return False
    if src_lemma and replacement and src_lemma[0] != replacement[0] and sim < 0.7:
        return False
    if replacement in BROAD_VOICE_BLACKLIST and (src_lemma, replacement) not in BROAD_TARGET_ALLOWLIST:
        return False
    stats = edge.get("interfaceStats") or {}
    generic = int(stats.get("generic", 0))
    keyboard = int(stats.get("keyboard", 0))
    voice = int(stats.get("voice", 0))
    if voice > 0 and generic == 0 and keyboard == 0:
        if sim < 0.72 or freq < 6:
            return False
    if replacement in BROAD_VOICE_BLACKLIST and generic + keyboard < 2:
        return False
    suffix_slots = edge.get("suffixSlots") or []
    surface_examples = edge.get("surfaceExamples") or []
    context_hints = edge.get("contextHints") or []
    if not suffix_slots and not surface_examples:
        return False
    if sim < 0.58 and not context_hints:
        return False
    return True


def auto_seed_score(edge: Dict, sim: float) -> float:
    base = float(edge.get("generatorScore", 0.0))
    freq = min(int(edge.get("frequency") or 0), 10)
    bonus = 0.03 * min(freq, 5) + max(0.0, sim - 0.55) * 0.15
    return round(min(0.985, max(base, base + bonus)), 4)


def build_auto_seeds() -> Dict[str, List[Dict]]:
    if not LEMMA_FAMILY_GRAPH.exists():
        return {}
    payload = json.loads(LEMMA_FAMILY_GRAPH.read_text(encoding="utf-8"))
    out: Dict[str, List[Dict]] = {}
    for lemma, edges in (payload.get("lemmas") or {}).items():
        keep: List[Dict] = []
        for edge in edges:
            if not should_keep_auto_edge(lemma, edge):
                continue
            sim = typo_similarity(lemma, str(edge.get("replacement") or ""))
            keep.append(
                {
                    "replacement": edge["replacement"],
                    "source": "AUTO_FAMILY_SEED",
                    "generatorScore": auto_seed_score(edge, sim),
                    "pos": edge.get("pos"),
                    "contextHints": edge.get("contextHints") or [],
                    "interfaceType": edge.get("interfaceType", "generic"),
                    "interfaceStats": edge.get("interfaceStats") or {},
                    "suffixSlots": edge.get("suffixSlots") or [],
                    "surfaceExamples": edge.get("surfaceExamples") or [],
                    "frequency": edge.get("frequency"),
                    "surfaceSimilarity": edge.get("surfaceSimilarity"),
                    "typoSimilarity": sim,
                }
            )
        if keep:
            keep.sort(
                key=lambda item: (
                    -float(item.get("generatorScore", 0.0)),
                    -int(item.get("frequency") or 0),
                    -float(item.get("typoSimilarity", 0.0)),
                    item["replacement"],
                )
            )
            out[lemma] = keep[:6]
    return out


def main() -> None:
    auto = build_auto_seeds()
    merged = dict(FAMILY_SEEDS)
    for lemma, rows in auto.items():
        merged[lemma] = merge_rows(merged.get(lemma, []), rows)
    payload = {
        "generatedFrom": {
            "manual": "inline_manual_seed_families",
            "auto": str(LEMMA_FAMILY_GRAPH),
        },
        "lemmaKeyCount": len(merged),
        "manualLemmaKeyCount": len(FAMILY_SEEDS),
        "autoLemmaKeyCount": len(auto),
        "lemmas": merged,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "lemmaKeyCount": len(merged),
        "manualLemmaKeyCount": len(FAMILY_SEEDS),
        "autoLemmaKeyCount": len(auto),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
