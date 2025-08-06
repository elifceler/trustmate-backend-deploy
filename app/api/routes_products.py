from fastapi import APIRouter
from app.utils.data_loader import load_mock_data

router = APIRouter()

@router.get("/products")
def get_products():
    data = load_mock_data()
    return {"products": sorted({item["product"] for item in data})}


@router.get("/sellers/{product_name}")
def get_sellers(product_name: str):
    data = load_mock_data()
    sellers = [
        {
            "seller": item["seller"],
            "price": item["price"],
            "rating": item["rating"],
            "review_count": item["review_count"],
        }
        for item in data
        if item["product"].lower() == product_name.lower()
    ]
    if not sellers:
        return {"message": f"No sellers found for product: {product_name}"}
    return {"sellers": sellers}
