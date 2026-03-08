#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+|[^\s]")
JAMO_RE = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]")

MANUAL_AUTO_ITEMS = [
    {"from": "할수", "to": "할 수", "reasonTag": "phrase_memory:할수->할 수", "confidence": 0.99},
    {"from": "될수", "to": "될 수", "reasonTag": "phrase_memory:될수->될 수", "confidence": 0.99},
    {"from": "안돼", "to": "안 돼", "reasonTag": "phrase_memory:안돼->안 돼", "confidence": 0.99},
    {"from": "않 와", "to": "안 와", "reasonTag": "phrase_memory:않 와->안 와", "confidence": 0.98},
    {"from": "않 와서", "to": "안 와서", "reasonTag": "phrase_memory:않 와서->안 와서", "confidence": 0.99},
    {"from": "못해", "to": "못 해", "reasonTag": "phrase_memory:못해->못 해", "confidence": 0.97},
    {"from": "해 바", "to": "해봐", "reasonTag": "phrase_memory:해 바->해봐", "confidence": 0.97},
]


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text)


def normalize_phrase(text: str) -> str:
    return " ".join(tokenize(text)).strip()


def has_hangul(text: str) -> bool:
    return bool(re.search(r"[가-힣]", text))


def strip_noise(text: str) -> str:
    return re.sub(r"^[^가-힣A-Za-z0-9]+|[^가-힣A-Za-z0-9]+$", "", text)


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


def phrase_similarity(left: str, right: str) -> float:
    left = normalize_phrase(left).replace(" ", "")
    right = normalize_phrase(right).replace(" ", "")
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    edit = char_edit_distance(left, right)
    return round(max(0.0, 1.0 - (edit / max(len(left), len(right)))), 4)


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


def extract_context_terms(sentence: str, target: str, window: int = 2) -> List[str]:
    tokens = tokenize(sentence)
    target_tokens = tokenize(target)
    if not tokens or not target_tokens:
        return []
    if len(target_tokens) > len(tokens):
        return []
    start_idx = None
    for idx in range(len(tokens) - len(target_tokens) + 1):
        if tokens[idx : idx + len(target_tokens)] == target_tokens:
            start_idx = idx
            break
    if start_idx is None:
        return []
    out: List[str] = []
    end_idx = start_idx + len(target_tokens)
    for idx in range(max(0, start_idx - window), min(len(tokens), end_idx + window)):
        if start_idx <= idx < end_idx:
            continue
        token = strip_noise(tokens[idx])
        if len(token) < 2 or not has_hangul(token):
            continue
        out.append(token)
    return out


def stream_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def build_phrase_stats(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    stats: Dict[str, Dict[str, object]] = {}
    for record in records:
        for error in record.get("errors", []):
            err_text = str(error.get("err_text") or "")
            cor_text = str(error.get("cor_text") or "")
            err_norm = normalize_phrase(err_text)
            cor_norm = normalize_phrase(cor_text)
            if not err_norm or not cor_norm or err_norm == cor_norm:
                continue
            if not has_hangul(err_norm + cor_norm):
                continue
            err_tokens = tokenize(err_norm)
            cor_tokens = tokenize(cor_norm)
            if max(len(err_tokens), len(cor_tokens)) > 3:
                continue
            if max(len(err_norm), len(cor_norm)) > 14:
                continue
            if len(err_tokens) == 1 and len(cor_tokens) == 1:
                continue
            if " " not in err_norm and " " not in cor_norm:
                continue
            if JAMO_RE.search(err_norm) or JAMO_RE.search(cor_norm):
                continue
            if any(not re.fullmatch(r"[가-힣A-Za-z0-9]+", token) for token in [*err_tokens, *cor_tokens]):
                continue
            if any(len(strip_noise(token)) > 5 for token in [*err_tokens, *cor_tokens]):
                continue
            similarity = phrase_similarity(err_norm, cor_norm)
            if similarity < 0.74:
                continue
            entry = stats.setdefault(
                err_norm,
                {
                    "targets": Counter(),
                    "interfaces": Counter(),
                    "errorTypes": Counter(),
                    "contexts": defaultdict(Counter),
                    "similarities": defaultdict(list),
                },
            )
            entry["targets"][cor_norm] += 1
            entry["interfaces"][infer_interface(record)] += 1
            entry["errorTypes"][str(record.get("error_type") or "unknown")] += 1
            for token in extract_context_terms(str(record.get("cor_sentence") or ""), cor_norm):
                entry["contexts"][cor_norm][token] += 1
            entry["similarities"][cor_norm].append(similarity)
    return stats


def top_counts(counter: Counter, limit: int) -> List[Dict[str, object]]:
    return [{"term": key, "count": count} for key, count in counter.most_common(limit)]


def finalize_auto_items(
    phrase_stats: Dict[str, Dict[str, object]],
    min_freq: int,
    min_dominance: float,
    max_items: int,
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for source_phrase, row in phrase_stats.items():
        targets: Counter = row["targets"]
        total = sum(targets.values())
        if total < min_freq:
            continue
        target, freq = targets.most_common(1)[0]
        dominance = freq / total
        if dominance < min_dominance:
            continue
        avg_similarity = round(sum(row["similarities"][target]) / max(1, len(row["similarities"][target])), 4)
        if avg_similarity < 0.82:
            continue
        interfaces = row["interfaces"]
        voice_ratio = interfaces.get("voice", 0) / max(1, sum(interfaces.values()))
        if voice_ratio > 0.5 and freq < max(min_freq + 2, 6):
            continue
        if targets[target] != total and total >= 8:
            continue
        if any(len(strip_noise(token)) > 5 for token in [*tokenize(source_phrase), *tokenize(target)]):
            continue
        if "  " in source_phrase or "  " in target:
            continue
        items.append(
            {
                "from": source_phrase,
                "to": target,
                "reasonTag": "phrase_memory_auto",
                "confidence": round(min(0.995, 0.9 + (0.03 * min(freq, 5)) + (0.05 * avg_similarity)), 4),
                "frequency": freq,
                "dominance": round(dominance, 4),
                "interfaceStats": dict(interfaces),
                "errorTypes": dict(row["errorTypes"]),
                "contextHints": top_counts(row["contexts"][target], 6),
            }
        )

    manual_pairs = {(item["from"], item["to"]) for item in MANUAL_AUTO_ITEMS}
    manual_index = {(item["from"], item["to"]): item for item in items}
    for item in MANUAL_AUTO_ITEMS:
        manual_index[(item["from"], item["to"])] = item

    merged = list(manual_index.values())
    merged.sort(key=lambda item: (-len(item["from"]), -float(item.get("confidence", 0.0)), item["from"]))
    manual_items = [item for item in merged if (item["from"], item["to"]) in manual_pairs]
    non_manual_items = [item for item in merged if (item["from"], item["to"]) not in manual_pairs]
    kept_non_manual = non_manual_items[: max(0, max_items - len(manual_items))]
    final_items = [*manual_items, *kept_non_manual]
    final_items.sort(key=lambda item: (-len(item["from"]), -float(item.get("confidence", 0.0)), item["from"]))
    return final_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Build phrase memory asset from normalized AI Hub data.")
    parser.add_argument("--input", type=Path, default=Path("data/aihub_highfreq/train.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("public/assets/dict/phrase_memory.json"),
    )
    parser.add_argument("--min-freq", type=int, default=3)
    parser.add_argument("--min-dominance", type=float, default=0.86)
    parser.add_argument("--max-items", type=int, default=240)
    args = parser.parse_args()

    phrase_stats = build_phrase_stats(stream_jsonl(args.input))
    auto_items = finalize_auto_items(
        phrase_stats=phrase_stats,
        min_freq=args.min_freq,
        min_dominance=args.min_dominance,
        max_items=args.max_items,
    )

    payload = {
        "generatedFrom": str(args.input),
        "count": len(auto_items),
        "items": auto_items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": len(auto_items)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
