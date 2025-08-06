from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from app.prompts import TRUST_REASONING_PROMPT

from dotenv import load_dotenv
load_dotenv()
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# ────────────────────────── GEMINI CONFIG ─────────────────────────

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ────────────────────────── HELPER FUNCS ──────────────────────────

def safe_score(val: float | None) -> float | None:
    """Return value in [0,1] rounded to 2 decimals or None."""
    try:
        v = float(val)  # type: ignore[arg-type]
        if 0 <= v <= 1:
            return round(v, 2)
    except (TypeError, ValueError):
        pass
    return None


def avg_from_sent(sentences: List[Dict[str, Any]], key: str) -> float | None:
    """Average non‑null sentence‑level scores for given key (2 decimals)."""
    vals = [s.get(key) for s in sentences if s.get(key) is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def compute_insight_bonus(sentiment: float, repurchase_intent: float = 1.0, community_effect: float = 1.0) -> float:
    """Calculate insight_bonus as specified in the prompt."""
    b = abs(sentiment - 0.5) * 2
    bonus = round((repurchase_intent + b + community_effect) / 3, 2)
    return bonus

# ──────────────────────────── PROMPT V3 ───────────────────────────
PROMPT_V3 = r"""
Sana bir satıcıya ait müşteri yorumlarını vereceğim.
⚠️ Sadece TEK SATIR geçerli JSON çıktısı üret. Markdown, açıklama, ``` kod bloğu YASAK.

────────────────────────────────

1️⃣ Cümlelere ayır:
• Ayırıcılar: . ! ? ; … emojiler  
• >200 karakterlik cümleleri anlam bozulmadan parçala  

2️⃣ Her cümle için aşağıdaki başlıklara 0 / 0.5 / 1 puan ver:  
• delivery          → kargo hızı, paketleme, teslimat  
• quality           → kalite, sağlamlık, tanıma uygunluk  
• return            → iade, değişim, garanti süreci  
• customer_service  → iletişim, destek kalitesi, çözüm hızı  
→ başlıkla alakasızsa: null  

3️⃣ Her cümle için sentiment:  
• pozitif → 1  
• nötr    → 0.5  
• negatif→ 0  

4️⃣ Her cümle için realness (0.00–1.00, 2 ondalık):  
+0.4 ≥20 karakter içeriyorsa  
+0.3 sayı, marka veya tarih içeriyorsa  
+0.3 özgün ifade / bağlam içeriyorsa  

5️⃣ Cümle çıktısı (sentence_json) şöyle olmalı:  
{
  "text": "...",
  "delivery": 1,
  "quality": 0.5,
  "return": null,
  "customer_service": 0,
  "sentiment": 1,
  "realness": 0.93
}

6️⃣ Global veriler:  
• sentiment: tüm cümlelerin sentiment ortalaması (2 ondalık)  
• aspect_summary = {başlık: {positive, negative}}  
 • başlık puanı ≥ 0.7 → positive  
 • başlık puanı ≤ 0.3 → negative  
• realness ortalaması hesaplanmayacak → sadece cümlelerde olacak  
• insight_bonus hesaplanmayacak → istemiyorum  

7️⃣ Nihai alanlar:  
Sadece şu alanlar dönecek:  
- sentiment  
- aspect_summary  
- realness_scores (her cümlenin realness değeri sırasıyla)  
- sentences (her biri sentence_json formatında)  
- ai_reasoning (satıcının güvenilirliği hakkında 1 cümlelik yorum)
- bonus_reasoning (eğer varsa, bonus verildiğini açıklayan metin)

8️⃣ Bonus açıklaması:
• Eğer yorumlar kullanıcı açısından güven verici, detaylı veya topluluk faydasını artıracak nitelikteyse, neden bonus verildiğini anlatan kısa bir açıklama üret: "bonus_reasoning"
• Yorumlar yüzeysel veya yetersizse, "bonus_reasoning" = ""

➤ Çıktıyı AYNEN aşağıdaki JSON şemasına göre TEK SATIRDA ver:

{
  "sentiment": 0.82,
  "aspect_summary": {...},
  "realness_scores": [...],
  "sentences": [...],
  "ai_reasoning": "Satıcı hızlı ve ilgili...",
  "bonus_reasoning": "Yorumlar detaylı ve tekrar satın alma niyeti yüksek."
}

- bonus_reasoning alanı her zaman string olmalı (boş olsa bile "").



────────────────────────────────

Yorumlar (JSON dizi):  
{reviews_json}
"""

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()


def fix_llm_output(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recalculate all global metrics from sentence array and patch the dict."""
    sentences = data.get("sentences", [])

    # Re‑compute primary averages
    for key in [
        "delivery",
        "quality",
        "return",
        "customer_service",
        "sentiment",
        "realness",
    ]:
        data[key] = avg_from_sent(sentences, key)

    # Realness scores list
    data["realness_scores"] = [s.get("realness") for s in sentences]

    # Compute bonus
    data["insight_bonus"] = compute_insight_bonus(data["sentiment"] or 0)

    # BONUS REASONING GELMİŞSE ELDE TUT
    if "bonus_reasoning" not in data:
        data["bonus_reasoning"] = ""

    return data

# ───────────────────── GEMINI CALL WRAPPER ───────────────────────

@retry(
    wait=wait_fixed(10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ResourceExhausted),
)
def analyze_reviews_with_gemini(reviews: List[str]) -> str:
    """Send reviews list to Gemini, return *validated* JSON string (single line)."""
    prompt = PROMPT_V3.replace("{reviews_json}", json.dumps(reviews, ensure_ascii=False))
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content(prompt)

    raw = resp.text or ""
    print("⛏️ RAW OUTPUT:\n", raw[:1000])  # İlk 1000 karakter yeterli
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return json.dumps({"error": "JSON not found in LLM output"})

    try:
        llm_data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return json.dumps({"error": "Invalid JSON from LLM", "details": str(exc)})

    fixed_data = fix_llm_output(llm_data)
    return json.dumps(fixed_data, ensure_ascii=False)


# ──────────────────────────── CLI TEST ───────────────────────────

if __name__ == "__main__":
    sample_reviews = [
        "Kargo çok hızlı geldi, ertesi gün elimdeydi.",
        "Ürün bozuk geldi ama hemen değişim yaptılar, ilgililerdi.",
        "Ses kalitesi harika, dış sesleri iyi engelliyor.",
    ]

    fixed = json.loads(analyze_reviews_with_gemini(sample_reviews))
    print("\n------ GLOBAL SKORLAR ------")
    for key in ["delivery", "quality", "return", "customer_service", "sentiment", "realness", "insight_bonus"]:
        print(f"{key}: {fixed[key]}")
