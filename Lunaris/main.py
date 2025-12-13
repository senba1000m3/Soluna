import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session

# Import the client we created
from anilist_client import AniListClient
from database import get_session, init_db
from models import User
from recommendation_engine import RecommendationEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Lunaris API",
    description="Backend for Ani-Risk Project (AniList Analytics)",
    version="0.1.0",
)


@app.on_event("startup")
def on_startup():
    init_db()


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
rec_engine = RecommendationEngine()

# --- Helper Functions ---


def get_next_season_info():
    """
    Determines the next anime season based on the current date.
    """
    now = datetime.now()
    month = now.month
    year = now.year

    # Winter: 1-3, Spring: 4-6, Summer: 7-9, Fall: 10-12
    if 1 <= month <= 3:
        return "SPRING", year
    elif 4 <= month <= 6:
        return "SUMMER", year
    elif 7 <= month <= 9:
        return "FALL", year
    else:
        return "WINTER", year + 1


def format_season_display(season: str) -> str:
    """
    Formats the season string as requested: 'Season-Month'.
    """
    mapping = {
        "WINTER": "冬-1 月",
        "SPRING": "春-4 月",
        "SUMMER": "夏-7 月",
        "FALL": "秋-10 月",
    }
    return mapping.get(season, season)


# --- Pydantic Models ---


class RecommendRequest(BaseModel):
    season: Optional[str] = None
    year: Optional[int] = None
    username: Optional[str] = None  # Optional: for personalized filtering


class SearchRequest(BaseModel):
    query: str


class CreateUserRequest(BaseModel):
    username: str
    anilist_id: Optional[int] = None


class DropPredictRequest(BaseModel):
    username: str
    anime_id: int


class PairCompareRequest(BaseModel):
    user1: str
    user2: str


class TimelineRequest(BaseModel):
    username: Optional[str] = None
    birth_year: int


class UserInfoRequest(BaseModel):
    username: str


# --- Endpoints ---


@app.get("/")
def read_root():
    return {"message": "Welcome to Lunaris Backend"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/users", response_model=User)
def create_user(user: CreateUserRequest, session: Session = Depends(get_session)):
    """
    Create a new user in the database.
    """
    try:
        db_user = User(username=user.username, anilist_id=user.anilist_id)
        session.add(db_user)
        session.commit()
        session.refresh(db_user)
        return db_user
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=400, detail=str(e))


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
    If username is provided, sorts by compatibility score.
    Defaults to the next season if season/year are not provided.
    """
    try:
        # Determine target season
        if not request.season or not request.year:
            target_season, target_year = get_next_season_info()
        else:
            target_season = request.season.upper()
            target_year = request.year

        display_season = format_season_display(target_season)
        logger.info(
            f"Fetching recommendations for {target_season} {target_year} ({display_season})"
        )

        # 1. Fetch Seasonal Anime
        anime_list = await anilist_client.get_seasonal_anime(
            season=target_season, year=target_year
        )

        # 2. If username provided, calculate personalized scores
        user_profile = {}
        if request.username:
            logger.info(f"Building profile for user: {request.username}")
            user_list = await anilist_client.get_user_anime_list(request.username)
            if user_list:
                user_profile = rec_engine.build_user_profile(user_list)
                anime_list = rec_engine.recommend_seasonal(user_profile, anime_list)
            else:
                logger.warning(
                    f"No list found for {request.username}, returning raw popularity list."
                )

        return {
            "season": target_season,
            "year": target_year,
            "display_season": display_season,
            "recommendations": anime_list,
        }

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

        # Fetch lists for analysis
        user1_list = await anilist_client.get_user_anime_list(request.user1)
        user2_list = await anilist_client.get_user_anime_list(request.user2)

        if not user1_list or not user2_list:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch anime lists. Users might have private lists.",
            )

        # Build profiles
        p1 = rec_engine.build_user_profile(user1_list)
        p2 = rec_engine.build_user_profile(user2_list)

        # Compare
        comparison = rec_engine.compare_users(p1, p2)

        return {
            "user1": user1_profile,
            "user2": user2_profile,
            "compatibility_score": comparison["score"],
            "common_genres": comparison["common_genres"],
            "message": "Comparison successful",
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in pair_compare: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user_info")
async def get_user_info(request: UserInfoRequest):
    """
    Get basic user info including birth date if available.
    """
    try:
        profile = await anilist_client.get_user_profile(request.username)
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        return profile
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error fetching user info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/timeline")
async def generate_timeline(request: TimelineRequest):
    """
    Generate a generational timeline of anime based on user's birth year.
    """
    try:
        logger.info(
            f"Generating timeline for {request.username}, born {request.birth_year}"
        )

        # 1. Prepare Years to Fetch
        current_year = datetime.now().year
        chronological_years = list(range(request.birth_year, current_year + 1))
        milestones = rec_engine.get_timeline_milestones(request.birth_year)

        # Union of all years needed
        milestone_years = [m["year"] for m in milestones]
        all_years = sorted(list(set(chronological_years + milestone_years)))

        # 2. Fetch User List (if username provided)
        user_list = []
        if request.username:
            user_list = await anilist_client.get_user_anime_list(request.username)

        # 3. Fetch popular anime for ALL years in parallel (with rate limiting)
        # AniList has a rate limit. We use a semaphore to limit concurrent requests.
        semaphore = asyncio.Semaphore(2)

        async def fetch_with_semaphore(year):
            for attempt in range(3):
                async with semaphore:
                    # Add a small delay to be safe
                    await asyncio.sleep(0.6)
                    res = await anilist_client.get_top_anime_by_year(year, per_page=5)
                    if res:
                        return res
                    # If failed (empty list), wait before retrying
                    await asyncio.sleep(1.0 * (attempt + 1))
            return []

        tasks = []
        for year in all_years:
            tasks.append(fetch_with_semaphore(year))

        results = await asyncio.gather(*tasks)
        year_anime_map = {year: res for year, res in zip(all_years, results)}

        # 4. Build Chronological Data (Every year)
        chronological_data = []
        for year in chronological_years:
            popular_anime = year_anime_map.get(year, [])
            representative = rec_engine.select_representative_anime(
                year, user_list, popular_anime
            )
            if representative:
                chronological_data.append(
                    {
                        "year": year,
                        "age": year - request.birth_year,
                        "anime": representative,
                    }
                )

        # 5. Build Milestones Data (Specific ages)
        timeline_data = []
        for m in milestones:
            year = m["year"]
            popular_anime = year_anime_map.get(year, [])
            # Use get_milestone_content to get top 5 with watched status
            milestone_content = rec_engine.get_milestone_content(
                year, user_list, popular_anime
            )
            if milestone_content:
                m["anime"] = milestone_content
                timeline_data.append(m)

        return {
            "username": request.username,
            "birth_year": request.birth_year,
            "chronological_data": chronological_data,
            "timeline_data": timeline_data,
        }

    except Exception as e:
        logger.error(f"Error generating timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))
