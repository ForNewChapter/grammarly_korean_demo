# 3단계: 텍스트의 성격(NORMAL/CHAT/NOISY 등)을 분류하여 이후 단계의 임계값을 조정한다.

import re
from typing import Any, Dict

import torch

from model_loader import ModelLoader, PROFILE_PROTOTYPES


def classify_profile(models: ModelLoader, text: str) -> Dict[str, Any]:
    """규칙 기반 분류와 임베딩 기반 분류를 결합하여 텍스트 프로필을 결정한다."""
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

    emb = models.sentence_embedding(text)
    emb_electra = models.sentence_embedding_koelectra(text)
    model_scores_kobert = {
        p: float(torch.dot(emb, proto).detach().item()) for p, proto in models.profile_proto_emb.items()
    }
    model_scores_koelectra = {
        p: float(torch.dot(emb_electra, proto).detach().item())
        for p, proto in models.profile_proto_emb_electra.items()
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
