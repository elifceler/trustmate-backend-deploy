from __future__ import annotations

"""
TrustMate – Satıcı Güven Skoru Hesaplama (v2)
================================================
‣ Bu sürümde şu sorunlar düzeltildi:
  • Az yorum + şüpheli realness ile puan şişmesi → agresif ceza eklendi
  • Etiketlendirme artık review_count & realness'i de dikkate alıyor
  • Dinamik ağırlıklar düşük hacimde daha sıkı kırpılıyor
  • Ek ceza fonksiyonları: low_volume_penalty, density_penalty

Kullanım:
    result = evaluate_seller(doc, peers, sentences)
"""

import math
from typing import Any, Dict, List

# ────────────────────────── Yardımcı Fonksiyonlar ────────────────────────── #

def clamp01(x: float | None) -> float | None:
    """None değilse [0,1] aralığına yuvarla."""
    return None if x is None else max(min(round(x, 2), 1), 0)

# ─────────────────────── Trust Level (gelişmiş) ─────────────────────────── #

def trust_level(score: float, review_count: int, realness: float | None) -> tuple[str, str]:
    """Etiketi puan + veri kalitesine göre belirle."""
    safe_real = realness or 0

    # Temel eşik
    if score >= 70:
        # Yorum az veya realness düşükse tam güven vermiyoruz
        if review_count < 5 or safe_real < 0.6:
            return "⚠️ Temkinli Güven", "yellow"
        return "✅ Güvenilir Satıcı", "green"

    if score >= 50:
        return "⚠️ Orta Seviye Güven", "yellow"

    return "❌ Düşük Güven Riski", "red"

# ───────────────────────────── Bonus / Duygu ─────────────────────────────── #

def compute_insight_bonus(sentiment: float, repurchase_intent: float = 1.0, community_effect: float = 1.0) -> float:
    """Sentiment polaritesine göre 0‑1 bonus."""
    b = abs(sentiment - 0.5) * 2
    return round((repurchase_intent + b + community_effect) / 3, 2)

# ─────────────────────────── Aspect Summary ─────────────────────────────── #

def extract_aspect_summary(sentences: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    summary = {k: {"positive": 0, "negative": 0} for k in ["delivery", "quality", "return", "customer_service"]}
    for s in sentences:
        for k in summary:
            v = s.get(k)
            if isinstance(v, (int, float)):
                if v >= 0.7:
                    summary[k]["positive"] += 1
                elif v <= 0.3:
                    summary[k]["negative"] += 1
    return summary

# ─────────────────────────── Profil Metrikleri ──────────────────────────── #

def profile_metrics(doc: Dict[str, Any], peers: List[Dict[str, Any]]) -> Dict[str, float]:
    rating = clamp01((doc.get("rating", 0) - 2) / 3)  # 2‑5 arası → 0‑1

    prices = [p["price"] for p in peers if p["seller"] != doc["seller"] and p.get("price")]
    price_score = None
    if prices and doc.get("price"):
        avg_p = sum(prices) / len(prices)
        diff_pct = abs(doc["price"] - avg_p) / avg_p
        if diff_pct > 0.3:
            price_score = 0.0
        else:
            price_score = clamp01(1 - diff_pct / 0.3) if doc["price"] > avg_p else clamp01(1 + diff_pct / 0.3)
    return {"rating_score": rating, "price_score": price_score}

# ───────────────────────── Ek Ayar / Spam Cezası ─────────────────────────── #

def extra_adjustments(doc: dict, sentences: List[dict]) -> tuple[float, float]:
    bonus_adj = 0.0
    realness_adj = 0.0

    # Yorum tarihi yoğunluğu
    dates: List[str] = doc.get("review_dates", [])
    if dates and len(set(dates)) / len(dates) < 0.3:
        realness_adj -= 0.25

    # Kısa yorum baskısı
    if sentences:
        short_ratio = sum(1 for s in sentences if len(s.get("text", "").split()) <= 6) / len(sentences)
        if short_ratio > 0.6:
            realness_adj -= 0.15

    # Cümle kalıp kontrolü
    def is_cliché(text: str) -> bool:
        clichés = [
            "tavsiye ederim", "mükemmel ürün", "kesinlikle alın", "çok güzel", "bayıldım",
            "harika", "süper", "hızlı kargo", "teşekkürler", "birebir aynısı", "herkese öneririm"
        ]
        return any(p in text.lower() for p in clichés)

    # Cümle benzerliği kontrolü (yüzeysel)
    def similarity_ratio(text1: str, text2: str) -> float:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, text1, text2).ratio()


    # Cliché yorum kontrolü
    cliché_count = sum(1 for s in sentences if is_cliché(s["text"]))
    if cliché_count / len(sentences) > 0.6:
        realness_adj -= 0.2

    # Aşırı benzer yorumlar → spam
    similar_pairs = 0
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if similarity_ratio(sentences[i]["text"], sentences[j]["text"]) > 0.85:
                similar_pairs += 1
    if similar_pairs >= 1:
        realness_adj -= 0.15

    return round(bonus_adj, 2), round(realness_adj, 2)

# ─────────────────────────── Ceza Fonksiyonları ─────────────────────────── #

def realness_penalty(realness: float | None) -> float:
    if realness is None:
        return 0.35  # belirsizlik ciddi ceza
    if realness >= 0.4:
        return 0.0
    return round((0.4 - realness) * (0.45 / 0.4), 3)  # max 0.45


def missing_field_penalty(metrics: dict, review_count: int) -> float:
    missing = sum(1 for f in ("return", "customer_service") if metrics.get(f) is None)
    factor = min(math.log(review_count + 1) / 3, 1.0)
    return round(0.07 * missing * factor, 3)


def low_volume_penalty(review_count: int) -> float:
    if review_count <= 2:
        return 0.30
    if review_count <= 4:
        return 0.20
    if review_count <= 6:
        return 0.10
    return 0.0

# ─────────────────────────── Dinamik Ağırlıklar ─────────────────────────── #

def dynamic_weights(review_count: int) -> Dict[str, float]:
    base = {
        "delivery": 0.10, "quality": 0.15, "return": 0.10, "customer_service": 0.10,
        "realness": 0.15, "sentiment": 0.05, "bonus": 0.05,
        "rating_score": 0.20, "price_score": 0.10,
    }
    if review_count < 5:
        # içerik zayıf → ağırlıkları %50 kırp
        for k in ("delivery", "quality", "return", "customer_service", "sentiment", "bonus"):
            base[k] *= 0.5
    return base

# ─────────────────────── Nihai Skor Hesabı & Etiket ─────────────────────── #

def calculate_trust_score(metrics: Dict[str, float | None], profile: Dict[str, float | None], review_count: int) -> float:
    w = dynamic_weights(review_count)
    score_sum = 0.0
    total_w = 0.0
    for k, wt in w.items():
        v = metrics.get(k) if k in metrics else profile.get(k)
        if v is not None:
            score_sum += v * wt
            total_w += wt
    raw = score_sum / total_w if total_w else 0.0

    # Toplam ceza
    penalty = (
        realness_penalty(metrics.get("realness")) +
        missing_field_penalty(metrics, review_count) +
        low_volume_penalty(review_count)
    )

    final = max(raw - penalty, 0) * 100
    return round(final, 2)

# ────────────────────────── Bayrak & Explanation ────────────────────────── #

def compose_flags(doc: dict, metrics: dict, aspect_summary: dict) -> List[str]:
    flags: List[str] = []
    if metrics.get("realness") is not None and metrics["realness"] < 0.3:
        flags.append("Yorumların gerçekliği düşük olabilir.")
    if metrics.get("return") is None:
        flags.append("İade/değişim süreci hakkında yorum bulunmuyor.")
    if aspect_summary.get("quality", {}).get("negative", 0) >= 2:
        flags.append("Ürün kalitesi hakkında olumsuz yorumlar yoğun.")
    if doc.get("review_count", 0) < 5:
        flags.append("Yorum sayısı düşük, veri güvenilirliği sınırlı.")
    return flags

def merge_explanations(ai_reasoning: str,
                       flags: List[str],
                       metrics: dict | None = None,
                       bonus_reasoning: str = "") -> str:
    explanation = ai_reasoning.strip() if ai_reasoning else ""

    if flags:
        explanation += "\n\n⚠️ Uyarılar:"
        for flag in flags:
            explanation += f"\n- {flag}"

    if metrics:
        explanation += "\n\n📊 Metrikler:"
        for key, val in metrics.items():
            try:
                val_pct = round(val * 100, 1) if isinstance(val, (int, float)) else val
                label = key.replace('_', ' ').capitalize()
                explanation += f"\n- {label}: {val_pct}%"
            except Exception:
                explanation += f"\n- {key}: {val}"
        
            if bonus_reasoning:
                explanation += f"\n\n💡 Bonus: {bonus_reasoning}"


    return explanation.strip()


# ─────────────────────────── Ana Pipeline API ──────────────────────────── #

def evaluate_seller(doc: dict, peers: List[dict], sentences: List[dict], ai_reasoning: str = "") -> dict:
    """End‑to‑end hesaplama: metrikleri güncelle, skorları üret."""

    # 1. Normalize metrikler
    metrics = {
        k: clamp01(doc.get(k)) for k in ("delivery", "quality", "return", "customer_service", "realness", "sentiment")
    }
    metrics["bonus"] = clamp01(compute_insight_bonus(metrics.get("sentiment") or 0))

    # 2. Ek ayar
    bonus_adj, realness_adj = extra_adjustments(doc, sentences)
    if metrics["bonus"] is not None:
        metrics["bonus"] = clamp01(metrics["bonus"] + bonus_adj)
    if metrics["realness"] is not None:
        metrics["realness"] = clamp01(metrics["realness"] + realness_adj)

    # 3. Profil metrikleri
    profile = profile_metrics(doc, peers)

    if not ai_reasoning:
        ai_reasoning = generate_trust_explanation(doc, profile)

    # 4. Güven skoru
    trust = calculate_trust_score(metrics, profile, doc.get("review_count", 0))

    # 5. Aspect summary & flags
    aspect_summary = extract_aspect_summary(sentences)
    flags = compose_flags(doc, metrics, aspect_summary)
    explanation = merge_explanations(ai_reasoning, flags, metrics)

    return {
        "seller": doc["seller"],
        "trust_score": trust,
        "trust_level": trust_level(trust, doc.get("review_count", 0), metrics.get("realness")),
        "scores": metrics,
        "profile_metrics": profile,
        "aspect_summary": aspect_summary,
        "general_reasoning": explanation,
        "sentences": sentences,
        "ai_reasoning": ai_reasoning,
        "bonus_reasoning": doc.get("bonus_reasoning", ""),
    }

from app.prompts import TRUST_REASONING_PROMPT
from app.services.gemini_service import call_gemini

def generate_trust_explanation(doc: dict, profile: dict) -> str:
    prompt = TRUST_REASONING_PROMPT.format(
        seller_name=doc.get("seller", "Belirsiz"),
        review_count=doc.get("review_count", 0),
        delivery=doc.get("delivery"),
        quality=doc.get("quality"),
        return_=doc.get("return"),
        customer_service=doc.get("customer_service"),
        sentiment=doc.get("sentiment"),
        realness=doc.get("realness"),
        bonus=doc.get("bonus"),
        avg_rating=profile.get("avg_rating"),
        same_day_ratio=profile.get("same_day_ratio"),
        price_deviation=profile.get("price_deviation"),
        peer_avg_price=profile.get("peer_avg_price"),
    )
    return call_gemini(prompt)
