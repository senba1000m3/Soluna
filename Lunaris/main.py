import logging
from typing import Any, Dict, List, Optional

# Import the client we created
from anilist_client import AniListClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lunaris API",
    description="Backend for Ani-Risk Project (AniList Analytics)",
    version="0.1.0",
)

# Configure CORS to allow requests from the frontend
origins = [
    "http://localhost:3000",  # React default
    "http://localhost:5173",  # Vite default
    "http://localhost:80",  # Docker production
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AniList Client
# We initialize it here so it can be reused.
# Note: In a production async app, you might want to manage the httpx client session lifecycle more carefully,
# but the current implementation in AniListClient creates a new client per request which is safe but less efficient.
anilist_client = AniListClient()

# --- Pydantic Models ---


class RecommendRequest(BaseModel):
    season: str = "WINTER"
    year: int = 2024
    username: Optional[str] = None  # Optional: for personalized filtering


class SearchRequest(BaseModel):
    query: str


class DropPredictRequest(BaseModel):
    username: str
    anime_id: int


class PairCompareRequest(BaseModel):
    user1: str
    user2: str


class TimelineRequest(BaseModel):
    username: str
    birth_year: int


# --- Endpoints ---


@app.get("/")
def read_root():
    return {"message": "Welcome to Lunaris Backend"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/search")
async def search_anime(request: SearchRequest):
    """
    Search for anime by title.
    """
    try:
        logger.info(f"Searching for anime: {request.query}")
        results = await anilist_client.search_anime(request.query)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommend")
async def recommend_anime(request: RecommendRequest):
    """
    Get seasonal recommendations.
    Currently fetches popular anime for the specified season and year from AniList.
    """
    try:
        logger.info(f"Fetching recommendations for {request.season} {request.year}")
        anime_list = await anilist_client.get_seasonal_anime(
            season=request.season, year=request.year
        )
        return {"recommendations": anime_list}
    except Exception as e:
        logger.error(f"Error in recommend endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drop_predict")
async def predict_drop(request: DropPredictRequest):
    """
    Predict the probability of a user dropping a specific anime.
    (Placeholder: currently returns mock data)
    """
    logger.info(
        f"Predicting drop for user {request.username} on anime {request.anime_id}"
    )

    # TODO:
    # 1. Fetch user's list history (to build user profile vector)
    # 2. Fetch anime metadata (genres, studio, etc.)
    # 3. Feed into trained ML model

    return {
        "username": request.username,
        "anime_id": request.anime_id,
        "drop_probability": 0.42,  # Mock probability
        "prediction": "Likely to Watch",
        "confidence": "Medium",
    }


@app.post("/pair_compare")
async def compare_users(request: PairCompareRequest):
    """
    Compare two users' tastes and compatibility.
    Fetches basic profiles for now.
    """
    try:
        logger.info(f"Comparing users {request.user1} and {request.user2}")
        user1_profile = await anilist_client.get_user_profile(request.user1)
        user2_profile = await anilist_client.get_user_profile(request.user2)

        if not user1_profile:
            raise HTTPException(
                status_code=404, detail=f"User {request.user1} not found"
            )
        if not user2_profile:
            raise HTTPException(
                status_code=404, detail=f"User {request.user2} not found"
            )

        # TODO:
        # 1. Fetch full anime lists for both users
        # 2. Calculate Jaccard Similarity / Cosine Similarity on genres
        # 3. Find overlapping watched shows

        return {
            "user1": user1_profile,
            "user2": user2_profile,
            "compatibility_score": 78.5,  # Mock score
            "shared_anime_count": 0,  # Placeholder
            "message": "Comparison logic to be implemented",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in pair_compare: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/timeline")
async def generate_timeline(request: TimelineRequest):
    """
    Generate a generational timeline of anime based on user's birth year.
    (Placeholder)
    """
    logger.info(
        f"Generating timeline for {request.username}, born {request.birth_year}"
    )

    return {
        "username": request.username,
        "birth_year": request.birth_year,
        "timeline_data": [],
        "message": "Timeline generation logic to be implemented",
    }
