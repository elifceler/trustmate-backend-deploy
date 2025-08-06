import json
import re
from typing import Any, Dict, List

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# ------------------------------------------------------------------
# Gemini model is configured globally elsewhere (main startup)
# ------------------------------------------------------------------

PROMPT_TEMPLATE = """
Aşağıda bir satıcıya ait yapısal veriler bulunmaktadır.
Bu verilere bakarak kullanıcıya satıcının güvenilirliği hakkında detaylı ama kısa bir açıklama yap. Veri sayısı azsa, değerlendirme için yeterli olmadığını belirt.


• Rakiplerine göre fiyatı düşük mü, yüksek mi?
• Ortalama puanı güven verici mi?
• Yorum tarihleri düzenli mi, hep aynı günlerde mi yazılmış?
• Spam sinyali var mı?
• Yorum sayısı yeterli mi?

Cevap kuralları:
- En az 2–3 cümle.
- Teknik ama akıcı Türkçe kullan.
- Markdown / JSON / başlık YOK — sadece düz metin.

Veri:
{seller_json}
Rakip fiyatları: {peer_prices}
"""

# ------------------------------------------------------------------
# Gemini wrapper
# ------------------------------------------------------------------

@retry(wait=wait_fixed(10), stop=stop_after_attempt(3),
       retry=retry_if_exception_type(ResourceExhausted))
def analyze_profile_with_gemini(
    seller_doc: Dict[str, Any],
    peers: List[Dict[str, Any]],
) -> str:
    """LLM'den satıcı profil yorumu döndürür (düz metin)."""

    peer_prices = [p["price"] for p in peers if p["seller"] != seller_doc["seller"]]

    prompt = PROMPT_TEMPLATE.format(
        seller_json=json.dumps(
            {
                "price": seller_doc.get("price"),
                "rating": seller_doc.get("rating"),
                "review_count": seller_doc.get("review_count"),
                "review_dates": seller_doc.get("review_dates"),
                "seller_location": seller_doc.get("seller_location"),
            },
            ensure_ascii=False,
        ),
        peer_prices=", ".join(map(str, peer_prices[:8])) or "N/A",
    )

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(prompt)
    text = (response.text or "").strip()
    # LLM bazen kod bloğu ekleyebilir; onu temizle
    text = re.sub(r"```.*?```", "", text, flags=re.S).strip()
    return text if text else "Analiz üretilemedi."
