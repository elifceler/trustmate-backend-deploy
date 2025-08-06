import json
from pathlib import Path

def load_mock_data() -> list:
    """
    mock_product_data.json dosyasını okuyup Python listesi olarak döndürür.
    """
    data_path = Path(__file__).resolve().parent.parent / "data" / "mock_product_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data
