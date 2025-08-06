from pathlib import Path
import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # ← EKLENDİ

from app.api.routes_root import router as root_router
from app.api.routes_products import router as products_router
from app.api.routes_analysis import router as analysis_router
from app.api.routes_recommend import router as recommend_router

def configure_environment():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    print("ACTIVE API KEY:", os.getenv("GEMINI_API_KEY"))

configure_environment()

# ───────────────────────────── FASTAPI APP ─────────────────────────────

app = FastAPI()

# ✅ CORS ayarı burada:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"],  # sadece frontend localhost'tan gelen istekleri kabul et
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(root_router)
app.include_router(products_router)
app.include_router(analysis_router)
app.include_router(recommend_router)
