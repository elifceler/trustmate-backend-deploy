from fastapi import APIRouter, Query
import json
from app.utils.data_loader import load_mock_data
from app.services.gemini_service import analyze_reviews_with_gemini, avg_from_sent
from app.services.scoring_service import (
    compute_insight_bonus, extract_aspect_summary, extra_adjustments,
    compose_flags, merge_explanations, profile_metrics, calculate_trust_score, trust_level
)
from app.services.profile_analysis import analyze_profile_with_gemini

router = APIRouter()

@router.get("/analyze/{seller_name}")
async def analyze_seller(
    seller_name: str,
    detail: str | None = Query(None,
        description="delivery | quality | return | customer_service detayı"),
):
    data = load_mock_data()
    doc = next((d for d in data if d["seller"].lower() == seller_name.lower()), None)
    if not doc:
        return {"error": f"Satıcı bulunamadı: {seller_name}"}

    result = json.loads(analyze_reviews_with_gemini(doc["reviews"]))
    peers = [d for d in data if d["product_id"] == doc["product_id"]]
    profile_reasoning = analyze_profile_with_gemini(doc, peers)

    if "error" in result: return result
    sentences  : list[dict] = result.get("sentences", [])

    metrics = {k: avg_from_sent(sentences, k) for k in
             ("delivery","quality","return","customer_service","realness","sentiment")}
    metrics["bonus"] = compute_insight_bonus(metrics["sentiment"] or 0)

    bonus_adj, realness_adj = extra_adjustments(doc, sentences)
    metrics["bonus"]    = max(min((metrics["bonus"] or 0)   + bonus_adj  , 1), 0)
    metrics["realness"] = max(min((metrics["realness"] or 0)+ realness_adj, 1), 0)

    # Profil tabanlı ek skorlar
    profile_info = profile_metrics(doc, peers)

    # Yeni nesil güven puanı
    review_count = doc.get("review_count", 0)
    trust = calculate_trust_score({**metrics, **profile_info}, profile_info, review_count)


    aspect_summary  = extract_aspect_summary(sentences)
    explanation_flags = compose_flags(doc, metrics, aspect_summary)

    ai_reasoning    = result.get("ai_reasoning", "")
    general_reasoning = merge_explanations(ai_reasoning, explanation_flags)


    if detail:
        return {
            "seller":   seller_name,
            "detail":   detail,
            "score":    metrics.get(detail),
            "summary":  aspect_summary.get(detail, {}),
            "general_reasoning": general_reasoning,
            "profile_reasoning": profile_reasoning,
        }

    return {
        "seller":            seller_name,
        "trust_score":       trust,
        "trust_level": trust_level(
    trust,                         # 1) hesapladığın skor
    doc.get("review_count", 0),    # 2) yorum sayısı
    metrics.get("realness"),       # 3) realness ortalaması
),
        "scores":            metrics,
        "aspect_summary":    aspect_summary,
        "general_reasoning": general_reasoning,
        "profile_reasoning": profile_reasoning,
        "sentences":         sentences,  # UI’de gizleyebilirsin
    }

