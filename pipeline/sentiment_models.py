"""FinBERT (EN) ve BERTurk-tabanlı TR duygu modelleri."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from pipeline.env import get_hf_token, load_project_env

load_project_env()

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "sentiment.yaml"

POSITIVE_LABELS = frozenset({"positive", "pozitif", "olumlu", "pos"})
NEGATIVE_LABELS = frozenset({"negative", "negatif", "olumsuz", "neg"})
NEUTRAL_LABELS = frozenset({"neutral", "nötr", "notr", "nötral"})


def load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _pipeline_kwargs() -> dict[str, Any]:
    token = get_hf_token()
    if token:
        return {"token": token}
    logger.warning(
        "HF_TOKEN tanımlı değil — indirme yavaş veya kilitlenebilir. "
        ".env dosyasına HF_TOKEN=hf_... ekleyin."
    )
    return {}


@lru_cache(maxsize=1)
def get_en_pipeline():
    from transformers import pipeline

    model = load_config()["models"]["en"]["name"]
    logger.info("İngilizce model yükleniyor: %s", model)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=model,
        device=-1,
        truncation=True,
        max_length=512,
        **_pipeline_kwargs(),
    )


@lru_cache(maxsize=1)
def get_tr_pipeline():
    from transformers import pipeline

    model = load_config()["models"]["tr"]["name"]
    logger.info("Türkçe model yükleniyor: %s", model)
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=model,
        device=-1,
        truncation=True,
        max_length=512,
        **_pipeline_kwargs(),
    )


def _normalize_label(label: str) -> str:
    return label.lower().strip()


def predictions_to_score(predictions: list[dict[str, Any]]) -> tuple[float, str, float, float, float]:
    """
  predictions: top_k çıktısı [{label, score}, ...]
  Döner: (score -1..1, baskın label, pos_prob, neg_prob, neu_prob)
    """
    pos = neg = neu = 0.0
    for p in predictions:
        label = _normalize_label(str(p["label"]))
        prob = float(p["score"])
        if label in POSITIVE_LABELS:
            pos += prob
        elif label in NEGATIVE_LABELS:
            neg += prob
        elif label in NEUTRAL_LABELS:
            neu += prob

    total = pos + neg + neu
    if total > 0 and total < 0.99:
        scale = 1.0 / total
        pos, neg, neu = pos * scale, neg * scale, neu * scale

    score = max(-1.0, min(1.0, pos - neg))
    dominant = max(predictions, key=lambda x: float(x["score"]))
    return round(score, 4), str(dominant["label"]), round(pos, 4), round(neg, 4), round(neu, 4)


def analyze_text(text: str, language: str) -> tuple[float, str, float, float, float, str]:
    """Tek metin için duygu skoru."""
    text = (text or "").strip()
    if not text:
        return 0.0, "neutral", 0.0, 0.0, 1.0, "empty"

    lang = (language or "tr").lower()[:2]
    if lang == "en":
        pipe = get_en_pipeline()
        model_name = load_config()["models"]["en"]["name"]
        top_k = 3
    else:
        pipe = get_tr_pipeline()
        model_name = load_config()["models"]["tr"]["name"]
        top_k = 2

    predictions = pipe(text[:2000], top_k=top_k)
    if isinstance(predictions, dict):
        predictions = [predictions]

    score, label, pos, neg, neu = predictions_to_score(predictions)
    return score, label, pos, neg, neu, model_name


def analyze_batch(
    texts: list[str],
    language: str,
) -> list[tuple[float, str, float, float, float, str]]:
    """Batch inference (aynı dil)."""
    if not texts:
        return []

    lang = (language or "tr").lower()[:2]
    cfg = load_config()["inference"]
    batch_size = int(cfg.get("batch_size", 8))
    max_chars = int(cfg.get("max_chars", 2000))
    clipped = [t[:max_chars] if t and t.strip() else "nötr" for t in texts]

    if lang == "en":
        pipe = get_en_pipeline()
        model_name = load_config()["models"]["en"]["name"]
        top_k = 3
    else:
        pipe = get_tr_pipeline()
        model_name = load_config()["models"]["tr"]["name"]
        top_k = 2

    results: list[tuple[float, str, float, float, float, str]] = []
    for i in range(0, len(clipped), batch_size):
        batch = clipped[i : i + batch_size]
        raw = pipe(batch, top_k=top_k)
        if batch and isinstance(raw, dict):
            raw = [raw]
        for preds in raw:
            if isinstance(preds, dict):
                preds = [preds]
            score, label, pos, neg, neu = predictions_to_score(preds)
            results.append((score, label, pos, neg, neu, model_name))
    return results
