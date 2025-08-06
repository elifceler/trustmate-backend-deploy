# app/services/recommender.py – rev‑3 uses new evaluate_seller pipeline
from __future__ import annotations

import json
from typing import Any, Dict, List

from .gemini_service import analyze_reviews_with_gemini
from .scoring_service import evaluate_seller, merge_explanations  # merged explanation util
from ..utils.data_loader import load_mock_data
import re
import unicodedata

def normalize(text: str) -> str:
    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    # Combining karakterleri sil (örneğin ş gibi harfleri düzleştirir)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Lowercase ve trim
    return text.lower().strip()

def recommend_sellers(product: str) -> List[Dict[str, Any]]:
    """Belirtilen üründeki satıcıları puanlayıp en yüksek güvenden düşüğe sıralar."""
    data = load_mock_data()
    norm_input = normalize(product)

    # 🛠️ DEBUG: Buraya ekle
    print("Kullanıcının girdiği ürün:", repr(product))
    print("Normalize edilmiş ürün adı:", norm_input)
    print("Verideki ürün isimleri:", [normalize(d["product"]) for d in data])

    sellers = [d for d in data if normalize(d["product"]) == norm_input]

    ranked: List[Dict[str, Any]] = []

    for doc in sellers:
        try:
            # 1️⃣ Yorum analizi – Gemini
            llm_json = json.loads(analyze_reviews_with_gemini(doc["reviews"]))
            sentences = llm_json.get("sentences", [])

            # 2️⃣ Toplu değerlendirme (yorum + profil + cezalar)
            peers = [s for s in sellers if s["seller"] != doc["seller"]]
            evaluation = evaluate_seller(doc | llm_json, peers, sentences)

            # 3️⃣ İnsan‑dostu açıklama
            # Değerler eksikse boş olarak al
            flags = evaluation.get("flags", [])
            metrics = evaluation.get("scores", {})  # ✅ Doğru key

            explanation = merge_explanations(
                llm_json.get("ai_reasoning", ""),
                flags,
                metrics,
                llm_json.get("bonus_reasoning", ""),
            )

            ranked.append({
                "seller":       doc["seller"],
                "price":        doc["price"],
                "rating":       doc["rating"],
                "trust_score":  evaluation["trust_score"],
                "trust_level":  evaluation["trust_level"],
                "explanation":  explanation,
                "bonus_reasoning": llm_json.get("bonus_reasoning", ""),
            })
        except Exception as e:
            print(f"HATA: Satıcı {doc['seller']} için değerlendirme çuvalladı -> {e}")
            continue


    return sorted(ranked, key=lambda x: x["trust_score"], reverse=True)
