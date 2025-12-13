import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlmodel import Session, select

# Import the client we created
from anilist_client import AniListClient
from database import engine, get_session, init_db
from drop_analysis_engine import DropAnalysisEngine
from hybrid_recommendation_engine import HybridRecommendationEngine
from ingest_data import fetch_and_store_anime, fetch_and_store_user_data
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


async def run_ingestion():
    """
    Runs data ingestion in the background.
    """
    logger.info("Starting background data ingestion...")
    try:
        with Session(engine) as session:
            # Fetch some initial data
            # Current season and previous season
            await fetch_and_store_anime(session, 2024, "WINTER")
            await fetch_and_store_anime(session, 2023, "FALL")

            # Fetch seed user data
            await fetch_and_store_user_data(session, "senba1000m3")

        logger.info("Background data ingestion complete.")
    except Exception as e:
        logger.error(f"Background ingestion failed: {e}")


@app.on_event("startup")
async def on_startup():
    init_db()
    # Run ingestion in background without blocking startup
    asyncio.create_task(run_ingestion())


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
drop_engine = DropAnalysisEngine()

# Initialize Hybrid Recommendation Engine
# 設置 use_bert=False 暫時不使用 BERT，等模型下載後再啟用
try:
    hybrid_rec_engine = HybridRecommendationEngine(use_bert=False)
    logger.info("Hybrid recommendation engine initialized (content-only mode)")
except Exception as e:
    logger.error(f"Failed to initialize hybrid engine: {e}")
    hybrid_rec_engine = None

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
    birth_month: Optional[int] = None
    birth_day: Optional[int] = None


class UserInfoRequest(BaseModel):
    username: str


class AnalyzeDropsRequest(BaseModel):
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
    Get seasonal recommendations using hybrid recommendation engine.
    If username is provided, uses BERT + content-based filtering.
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

        # 2. If username provided, use hybrid recommendation
        if request.username:
            logger.info(f"Building hybrid profile for user: {request.username}")
            user_list = await anilist_client.get_user_anime_list(request.username)

            if user_list:
                # Use hybrid engine if available, otherwise fall back to content-only
                if hybrid_rec_engine:
                    logger.info("Using hybrid recommendation engine...")
                    anime_list = hybrid_rec_engine.recommend_seasonal(
                        user_list=user_list,
                        seasonal_anime=anime_list,
                        bert_weight=0.6,
                        content_weight=0.4,
                        top_reference_anime=50,
                    )
                else:
                    logger.warning("Hybrid engine not available, using content-only")
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


@app.get("/recommend/status")
def get_recommendation_status():
    """
    Get the status of the recommendation engine (BERT availability, mode, etc.)
    """
    try:
        status = {
            "hybrid_engine_available": hybrid_rec_engine is not None,
            "mode": "content_only",
            "bert_enabled": False,
            "bert_weight": 0.0,
            "content_weight": 1.0,
        }

        if hybrid_rec_engine:
            status["bert_enabled"] = hybrid_rec_engine.use_bert
            status["bert_available"] = (
                hybrid_rec_engine.bert_recommender is not None
                and hybrid_rec_engine.bert_recommender.is_available()
            )

            if status["bert_enabled"] and status.get("bert_available", False):
                status["mode"] = "hybrid"
                status["bert_weight"] = 0.6
                status["content_weight"] = 0.4
            else:
                status["mode"] = "content_only"
                status["bert_weight"] = 0.0
                status["content_weight"] = 1.0

        return status

    except Exception as e:
        logger.error(f"Error getting recommendation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/user_profile")
async def debug_user_profile(request: UserInfoRequest):
    """
    調試端點：檢查用戶 profile 是否正確建立
    """
    try:
        logger.info(f"Building profile for debug: {request.username}")

        # 獲取用戶列表
        user_list = await anilist_client.get_user_anime_list(request.username)

        if not user_list:
            return {
                "error": "No user list found",
                "username": request.username,
            }

        # 建立 profile
        user_profile = rec_engine.build_user_profile(user_list)

        # 統計資訊
        scored_entries = [
            e
            for e in user_list
            if e["status"] in ["COMPLETED", "CURRENT"] and e["score"] > 0
        ]

        # 提取 genre 權重
        genre_weights = {
            k.replace("Genre_", ""): v
            for k, v in user_profile.items()
            if k.startswith("Genre_")
        }

        # 排序
        sorted_genres = sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)

        return {
            "username": request.username,
            "total_entries": len(user_list),
            "scored_entries": len(scored_entries),
            "profile_features": len(user_profile),
            "profile_has_data": len(user_profile) > 0,
            "top_10_genres": dict(sorted_genres[:10]),
            "sample_scored_anime": [
                {
                    "title": e.get("media", {})
                    .get("title", {})
                    .get("romaji", "Unknown"),
                    "score": e.get("score", 0),
                    "genres": e.get("media", {}).get("genres", []),
                }
                for e in scored_entries[:5]
            ],
        }

    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
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
        semaphore = asyncio.Semaphore(3)

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

        # 4. Calculate Stats
        stats = rec_engine.calculate_timeline_stats(user_list)

        # 5. Fetch birthday characters (if birth date provided)
        birthday_chars = []
        if request.birth_month and request.birth_day:
            try:
                birthday_chars = await anilist_client.get_characters_by_birthday(
                    request.birth_month, request.birth_day
                )
                logger.info(
                    f"Found {len(birthday_chars)} birthday characters for {request.birth_month}/{request.birth_day}"
                )
            except Exception as e:
                logger.warning(f"Failed to fetch birthday characters: {e}")
                birthday_chars = []

        # 6. Build Chronological Data (Every year)
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

        # 7. Build Milestones Data (Specific ages)
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
            "stats": stats,
            "birthday_characters": birthday_chars,
        }

    except Exception as e:
        logger.error(f"Error generating timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze_drops")
async def analyze_drops(
    request: AnalyzeDropsRequest, session: Session = Depends(get_session)
):
    """
    Fetches user's dropped list, stores data, trains model, and returns analysis.
    Now includes predictions for watching/planning anime and drop pattern statistics.
    """
    try:
        # Import needed models at function start
        from sqlmodel import select

        from models import Anime as AnimeModel
        from models import User as UserModel
        from models import UserRating

        logger.info(f"Starting drop analysis for user: {request.username}")

        # Check if user exists first
        logger.info(f"Checking if user {request.username} exists on AniList...")
        profile = await anilist_client.get_user_profile(request.username)
        if not profile:
            logger.error(f"User {request.username} not found on AniList")
            raise HTTPException(
                status_code=404,
                detail=f"User '{request.username}' not found on AniList. Please check the username.",
            )

        logger.info(f"User found: {profile.get('name')} (ID: {profile.get('id')})")

        # 1. Fetch and Store User Data (Ingest)
        logger.info(f"Fetching and storing anime list for {request.username}...")
        try:
            await fetch_and_store_user_data(session, request.username)
            logger.info("User data stored successfully")
        except Exception as e:
            logger.error(f"Error fetching/storing user data: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch anime list: {str(e)}"
            )

        # 2. Train personalized model for this user
        logger.info("Training personalized drop prediction model...")
        try:
            # Get user ID for personalized training
            db_user = session.exec(
                select(UserModel).where(UserModel.username == request.username)
            ).first()

            if not db_user:
                raise HTTPException(
                    status_code=404, detail="User not found in database"
                )

            train_result = drop_engine.train_model(session, user_id=db_user.id)
            logger.info(f"Model training complete: {train_result}")
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to train model: {str(e)}"
            )

        # 3. Get User's List and categorize
        logger.info("Fetching and analyzing user's anime list...")
        try:
            user_list = await anilist_client.get_user_anime_list(request.username)
            dropped_list = []
            watching_list = []
            planning_list = []

            for entry in user_list:
                status = entry.get("status", "").upper()
                media = entry.get("media", {})
                anime_id = media.get("id")
                title = media.get("title", {})
                cover = media.get("coverImage") or {}

                base_info = {
                    "id": anime_id,
                    "title": title.get("english")
                    or title.get("romaji")
                    or "Unknown Title",
                    "cover": cover.get("large"),
                    "score": entry.get("score", 0),
                    "progress": entry.get("progress", 0),
                    "total_episodes": media.get("episodes"),
                    "genres": media.get("genres", []),
                }

                if status == "DROPPED":
                    dropped_list.append(base_info)
                elif status in ["CURRENT", "PLANNING"]:
                    # Get anime from DB for prediction
                    anime = session.get(AnimeModel, anime_id)
                    if anime and drop_engine.is_trained:
                        # Get user ID for tolerance-based prediction
                        db_user = session.exec(
                            select(UserModel).where(
                                UserModel.username == request.username
                            )
                        ).first()

                        if db_user:
                            drop_probability, reasons = (
                                drop_engine.predict_drop_probability(
                                    anime, db_user.id, session
                                )
                            )
                            base_info["drop_probability"] = float(drop_probability)
                            base_info["drop_reasons"] = reasons
                        else:
                            base_info["drop_probability"] = None
                            base_info["drop_reasons"] = []
                    else:
                        base_info["drop_probability"] = None
                        base_info["drop_reasons"] = []

                    if status == "CURRENT":
                        watching_list.append(base_info)
                    else:
                        planning_list.append(base_info)

            # Sort by drop probability (highest first)
            watching_list.sort(
                key=lambda x: x.get("drop_probability") or 0, reverse=True
            )
            planning_list.sort(
                key=lambda x: x.get("drop_probability") or 0, reverse=True
            )

            logger.info(
                f"Found {len(dropped_list)} dropped, {len(watching_list)} watching, {len(planning_list)} planning"
            )
        except Exception as e:
            logger.error(f"Error processing anime list: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to process anime list: {str(e)}"
            )

        # 4. Analyze drop patterns
        logger.info("Analyzing drop patterns...")
        try:
            ratings = session.exec(select(UserRating)).all()
            animes = session.exec(select(AnimeModel)).all()
            drop_patterns = drop_engine.analyze_drop_patterns(ratings, animes)
            logger.info("Drop pattern analysis complete")
        except Exception as e:
            logger.error(f"Error analyzing drop patterns: {e}")
            drop_patterns = {
                "top_dropped_tags": [],
                "top_dropped_genres": [],
                "top_dropped_studios": [],
            }

        return {
            "username": request.username,
            "dropped_count": len(dropped_list),
            "dropped_list": dropped_list,
            "watching_list": watching_list,
            "planning_list": planning_list,
            "model_metrics": train_result,
            "drop_patterns": drop_patterns,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_drops: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )
