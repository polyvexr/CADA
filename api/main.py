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
from api.routes import router, set_scorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown lifecycle: loads or trains CADA scorer."""
    model_bundle_path = MODELS_DIR / "cada_model_bundle.joblib"
    
    if model_bundle_path.exists():
        print(f"Loading pre-trained CADA model bundle from: {model_bundle_path}")
        scorer = CADACompositeScorer.load(model_bundle_path)
    else:
        print("Model bundle not found. Training CADA models now...")
        scorer = train_cada_models()

    set_scorer(scorer)
    print("CADA Scoring Engine ready for real-time inference.")
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

