"""
Main FastAPI application entrypoint for CADA continuous driving anomaly detection.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import MODELS_DIR
from src.scoring.cada_scorer import CADACompositeScorer
from src.models.trainer import train_cada_models
from api.routes import router, set_scorer, load_or_init_scorer



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown lifecycle: initializes CADA scorer."""
    try:
        load_or_init_scorer()
    except Exception as e:
        print(f"Startup scorer initialization note: {e}")
    yield


app = FastAPI(
    title="CADA - Continuous Driving Anomaly Detection API",
    description="Real-time multi-sensor IMU driving risk assessment, anomaly detection, and continuous safety tiering.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for dashboards & external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

