from fastapi import APIRouter
from app.services.recommender import recommend_sellers

router = APIRouter()

@router.get("/recommend/{product_name}")
async def recommend(product_name: str):
    ranked = recommend_sellers(product_name)
    if not ranked:
        return {"error": f"'{product_name}' için satıcı verisi bulunamadı."}

    return {
        "recommended": ranked[0],
        "alternatives": ranked[1:3],
    }
