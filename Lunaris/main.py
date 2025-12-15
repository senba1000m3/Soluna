import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

# Import the client we created
from anilist_client import AniListClient
from database import engine, get_session, init_db
from drop_analysis_engine import DropAnalysisEngine
from hybrid_recommendation_engine import HybridRecommendationEngine
from ingest_data import fetch_and_store_anime, fetch_and_store_user_data
from models import User
from progress_tracker import progress_manager
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
    task_id: Optional[str] = None  # Optional task ID for progress tracking


class RecapRequest(BaseModel):
    username: str
    year: Optional[int] = None  # None means all-time recap


# --- Endpoints ---


@app.get("/")
def read_root():
    return {"message": "Welcome to Lunaris API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/progress/{task_id}")
async def stream_progress(task_id: str):
    """
    SSE endpoint to stream progress updates for a specific task
    """
    tracker = progress_manager.get_task(task_id)
    if not tracker:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        try:
            for update in tracker.get_updates():
                yield f"data: {json.dumps(update)}\n\n"
        except Exception as e:
            logger.error(f"Error in progress stream: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            # Clean up after streaming is done
            progress_manager.remove_task(task_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
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
    Compare two users' tastes and compatibility with detailed analytics.
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

        # Detailed comparison
        comparison = rec_engine.detailed_user_comparison(user1_list, user2_list, p1, p2)

        return {
            "user1": user1_profile,
            "user2": user2_profile,
            "compatibility_score": comparison["score"],
            "common_genres": comparison["common_genres"],
            "common_anime": comparison["common_anime"],
            "common_count": comparison["common_count"],
            "disagreements": comparison["disagreements"],
            "avg_score_difference": comparison["avg_score_difference"],
            "radar_data": comparison["radar_data"],
            "stats": comparison["stats"],
            "recommendations": comparison["recommendations"],
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
    Supports real-time progress tracking via SSE.
    """
    # Create progress tracker
    task_id = request.task_id or str(uuid.uuid4())
    tracker = progress_manager.create_task(task_id)
    tracker.update(progress=0, stage="init", status="running", message="開始分析...")

    try:
        # Import needed models at function start
        from sqlmodel import select

        from models import Anime as AnimeModel
        from models import User as UserModel
        from models import UserRating

        logger.info(f"Starting drop analysis for user: {request.username}")
        tracker.update(progress=2, message=f"開始分析使用者: {request.username}")

        # Check if user exists first
        logger.info(f"Checking if user {request.username} exists on AniList...")
        tracker.update(progress=5, message="檢查使用者是否存在...")
        profile = await anilist_client.get_user_profile(request.username)
        if not profile:
            logger.error(f"User {request.username} not found on AniList")
            tracker.error(f"使用者 {request.username} 不存在")
            raise HTTPException(
                status_code=404,
                detail=f"User '{request.username}' not found on AniList. Please check the username.",
            )

        logger.info(f"User found: {profile.get('name')} (ID: {profile.get('id')})")

        # 1. Fetch and Store User Data (Ingest)
        logger.info(f"Fetching and storing anime list for {request.username}...")
        tracker.update(progress=10, stage="fetch_data", message="正在抓取動漫列表...")
        try:
            await fetch_and_store_user_data(session, request.username)
            logger.info("User data stored successfully")
            tracker.update(progress=30, message="動漫列表已儲存")
        except Exception as e:
            logger.error(f"Error fetching/storing user data: {e}")
            tracker.error(f"抓取資料失敗: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch anime list: {str(e)}"
            )

        # 2. Train personalized model for this user
        logger.info("Training personalized drop prediction model...")
        tracker.update(progress=35, stage="train_model", message="開始訓練模型...")
        try:
            # Get user ID for personalized training
            db_user = session.exec(
                select(UserModel).where(UserModel.username == request.username)
            ).first()

            if not db_user:
                tracker.error("資料庫中找不到使用者")
                raise HTTPException(
                    status_code=404, detail="User not found in database"
                )

            # Create a new drop engine with progress tracker
            drop_engine_with_progress = DropAnalysisEngine(progress_tracker=tracker)
            train_result = drop_engine_with_progress.train_model(
                session, user_id=db_user.id
            )
            logger.info(f"Model training complete: {train_result}")

            # Update the global drop_engine with the trained model
            drop_engine.model = drop_engine_with_progress.model
            drop_engine.feature_columns = drop_engine_with_progress.feature_columns
            drop_engine.mlb_genres = drop_engine_with_progress.mlb_genres
            drop_engine.mlb_tags = drop_engine_with_progress.mlb_tags
            drop_engine.le_studio = drop_engine_with_progress.le_studio
            drop_engine.is_trained = drop_engine_with_progress.is_trained

        except Exception as e:
            logger.error(f"Error training model: {e}")
            tracker.error(f"訓練模型失敗: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Failed to train model: {str(e)}"
            )

        # 3. Get User's List and categorize
        logger.info("Fetching and analyzing user's anime list...")
        tracker.update(progress=90, stage="analyze", message="分析動漫列表...")
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
            logger.info("Drop analysis complete!")
            tracker.complete(message="分析完成！")
        except Exception as e:
            logger.error(f"Error analyzing drop patterns: {e}")
            drop_patterns = {
                "top_dropped_tags": [],
                "top_dropped_genres": [],
                "top_dropped_studios": [],
            }

        return {
            "task_id": task_id,
            "username": request.username,
            "dropped_count": len(dropped_list),
            "dropped_list": dropped_list,
            "watching_list": watching_list,
            "planning_list": planning_list,
            "model_metrics": train_result,
            "drop_patterns": drop_patterns,
        }
    except HTTPException:
        tracker.error("發生錯誤")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in analyze_drops: {e}", exc_info=True)
        tracker.error(f"發生錯誤: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {str(e)}"
        )


@app.post("/recap")
async def get_user_recap(
    request: RecapRequest, session: Session = Depends(get_session)
):
    """
    Generate a recap of user's anime watching activity.
    If year is provided, returns recap for that specific year.
    If year is None, returns all-time recap.
    """
    try:
        print("\n" + "=" * 60)
        print(f"📊 開始生成 Recap: {request.username}")
        print(f"年份: {request.year if request.year else '全部'}")
        print("=" * 60)
        logger.info(f"Generating recap for {request.username}, year={request.year}")

        # Create AniListClient with database session for caching
        client_with_cache = AniListClient(db_session=session)

        # Fetch user's complete anime list
        print("🔄 正在抓取使用者動漫列表...")
        user_list = await client_with_cache.get_user_anime_list(request.username)
        print(f"✅ 成功抓取 {len(user_list) if user_list else 0} 筆資料")

        if not user_list:
            print("❌ 錯誤: 找不到使用者或動漫列表為空")
            raise HTTPException(
                status_code=404, detail="User not found or has no anime list"
            )

        # Filter by year if specified
        print(f"🔍 開始篩選資料 (年份: {request.year if request.year else '全部'})...")
        filtered_list = []
        for entry in user_list:
            if request.year is not None:
                # Check if anime was completed or updated in the specified year
                completed_at = entry.get("completedAt", {})
                updated_at = entry.get("updatedAt")
                started_at = entry.get("startedAt", {})

                # Check completed year
                if completed_at and completed_at.get("year") == request.year:
                    filtered_list.append(entry)
                # Check started year if not completed
                elif started_at and started_at.get("year") == request.year:
                    filtered_list.append(entry)
            else:
                # All-time recap - include all entries
                filtered_list.append(entry)

        print(f"✅ 篩選完成，共 {len(filtered_list)} 筆符合條件的資料")

        if not filtered_list:
            print("⚠️  警告: 篩選後無資料，返回空結果")
            return {
                "username": request.username,
                "year": request.year,
                "is_all_time": request.year is None,
                "total_anime": 0,
                "total_episodes": 0,
                "total_minutes": 0,
                "total_hours": 0,
                "completed_count": 0,
                "watching_count": 0,
                "dropped_count": 0,
                "planned_count": 0,
                "paused_count": 0,
                "top_anime": [],
                "genre_distribution": {},
                "format_distribution": {},
                "average_score": 0,
                "total_scored": 0,
                "achievements": [],
            }

        # Calculate statistics
        total_anime = len(filtered_list)
        total_episodes = 0
        total_minutes = 0
        completed_count = 0
        watching_count = 0
        dropped_count = 0
        planned_count = 0
        paused_count = 0
        repeating_count = 0

        genre_count = {}
        format_count = {}
        tag_count = {}
        studio_count = {}
        studio_details = {}  # Store studio info with images
        voice_actor_count = {}
        voice_actor_details = {}  # Store VA info with images
        anime_ids_for_va = []  # Collect anime IDs for separate VA queries
        season_count = {}
        month_added_count = {}
        month_completed_count = {}
        scores = []
        repeat_anime = []

        # For top anime (sorted by score and episodes watched)
        scored_anime = []

        for entry in filtered_list:
            status = entry.get("status", "")
            progress = entry.get("progress", 0)
            score = entry.get("score", 0)
            media = entry.get("media", {})

            # Count by status
            if status == "COMPLETED":
                completed_count += 1
            elif status == "CURRENT":
                watching_count += 1
            elif status == "DROPPED":
                dropped_count += 1
            elif status == "PLANNING":
                planned_count += 1
            elif status == "PAUSED":
                paused_count += 1
            elif status == "REPEATING":
                repeating_count += 1

            # Calculate episodes and duration
            episodes = media.get("episodes") or progress or 0
            duration = media.get("duration") or 24  # Default 24 min per episode

            if status == "COMPLETED":
                total_episodes += episodes
                total_minutes += episodes * duration
            else:
                total_episodes += progress
                total_minutes += progress * duration

            # Collect scores
            if score > 0:
                scores.append(score)
                scored_anime.append(
                    {
                        "id": media.get("id"),
                        "title": media.get("title", {}).get("romaji", "Unknown"),
                        "title_english": media.get("title", {}).get("english"),
                        "coverImage": media.get("coverImage", {}).get("large", ""),
                        "score": score,
                        "episodes": episodes,
                        "progress": progress,
                        "status": status,
                        "genres": media.get("genres", []),
                        "format": media.get("format", "UNKNOWN"),
                        "averageScore": media.get("averageScore", 0),
                        "year": media.get("seasonYear"),
                    }
                )

            # Count genres
            for genre in media.get("genres", []):
                genre_count[genre] = genre_count.get(genre, 0) + 1

            # Count formats
            format_type = media.get("format", "UNKNOWN")
            format_count[format_type] = format_count.get(format_type, 0) + 1

            # Count tags (top 20)
            for tag in media.get("tags", []):
                tag_name = tag.get("name") if isinstance(tag, dict) else tag
                if tag_name:
                    tag_count[tag_name] = tag_count.get(tag_name, 0) + 1

            # Count studios with details
            studios = media.get("studios", {})
            if isinstance(studios, dict):
                studio_nodes = studios.get("nodes", [])
            else:
                studio_nodes = studios if isinstance(studios, list) else []

            for studio in studio_nodes:
                if isinstance(studio, dict):
                    studio_name = studio.get("name")
                    if studio_name:
                        studio_count[studio_name] = studio_count.get(studio_name, 0) + 1
                        # Store studio details (first occurrence)
                        if studio_name not in studio_details:
                            studio_details[studio_name] = {
                                "id": studio.get("id"),
                                "name": studio_name,
                                "siteUrl": studio.get("siteUrl"),
                                "count": 0,
                            }
                        studio_details[studio_name]["count"] = studio_count[studio_name]
                else:
                    studio_name = studio
                    if studio_name:
                        studio_count[studio_name] = studio_count.get(studio_name, 0) + 1

            # Note: Voice actor data will be fetched separately after initial processing

            # Collect anime IDs for voice actor fetching
            anime_ids_for_va.append(media.get("id"))

            # Count seasons
            season = media.get("season")
            season_year = media.get("seasonYear")
            if season and season_year:
                season_key = f"{season_year} {season}"
                season_count[season_key] = season_count.get(season_key, 0) + 1

            # Count months (when added/completed)
            started_at = entry.get("startedAt", {})
            completed_at = entry.get("completedAt", {})

            if started_at and started_at.get("year") and started_at.get("month"):
                month_key = f"{started_at['year']}-{started_at['month']:02d}"
                month_added_count[month_key] = month_added_count.get(month_key, 0) + 1

            if completed_at and completed_at.get("year") and completed_at.get("month"):
                month_key = f"{completed_at['year']}-{completed_at['month']:02d}"
                month_completed_count[month_key] = (
                    month_completed_count.get(month_key, 0) + 1
                )

            # Track repeat counts
            repeat_count = entry.get("repeat", 0)
            if repeat_count > 0:
                repeat_anime.append(
                    {
                        "id": media.get("id"),
                        "title": media.get("title", {}).get("romaji", "Unknown"),
                        "title_english": media.get("title", {}).get("english"),
                        "coverImage": media.get("coverImage", {}).get("large", ""),
                        "repeat_count": repeat_count,
                        "score": score,
                    }
                )

        # Calculate top anime (by user score, then by episodes)
        scored_anime.sort(key=lambda x: (x["score"], x["episodes"]), reverse=True)
        top_anime = scored_anime[:10]

        # Sort repeat anime
        repeat_anime.sort(key=lambda x: x["repeat_count"], reverse=True)

        # Calculate average score
        average_score = sum(scores) / len(scores) if scores else 0

        # Fetch voice actor data separately with parallel processing
        print(f"\n🎤 開始抓取聲優數據...")
        print(f"  - 需要查詢的動漫數量: {len(anime_ids_for_va)}")

        # Use semaphore to limit concurrent requests (max 5 at a time)
        semaphore = asyncio.Semaphore(5)

        async def fetch_va_with_semaphore(anime_id, idx):
            async with semaphore:
                try:
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.15)

                    if idx % 20 == 0:
                        print(f"  - 進度: {idx}/{len(anime_ids_for_va)}")

                    anime_va_data = await client_with_cache.get_anime_voice_actors(
                        anime_id
                    )
                    return (anime_id, anime_va_data)
                except Exception as e:
                    print(f"  ⚠️ 抓取動漫 {anime_id} 聲優數據失敗: {str(e)}")
                    return (anime_id, None)

        # Fetch all voice actor data in parallel with controlled concurrency
        tasks = [
            fetch_va_with_semaphore(anime_id, idx)
            for idx, anime_id in enumerate(anime_ids_for_va)
        ]
        results = await asyncio.gather(*tasks)

        # Process results
        for anime_id, anime_va_data in results:
            if anime_va_data and "characters" in anime_va_data:
                characters = anime_va_data["characters"]
                if characters and isinstance(characters, dict):
                    edges = characters.get("edges", [])

                    # 先收集本部作品所有聲優的 set
                    unique_va_names = set()
                    va_details_map = {}
                    for edge in edges:
                        if isinstance(edge, dict):
                            voice_actors = edge.get("voiceActors", [])
                            if voice_actors and isinstance(voice_actors, list):
                                for va in voice_actors:
                                    if isinstance(va, dict):
                                        va_name = va.get("name")
                                        if va_name and isinstance(va_name, dict):
                                            va_full_name = va_name.get("full")
                                            if va_full_name:
                                                unique_va_names.add(va_full_name)
                                                # 儲存詳細資料（只存一次）
                                                if va_full_name not in va_details_map:
                                                    va_image = va.get("image", {})
                                                    if isinstance(va_image, dict):
                                                        va_image_url = va_image.get(
                                                            "large"
                                                        ) or va_image.get("medium")
                                                    else:
                                                        va_image_url = None
                                                    va_details_map[va_full_name] = {
                                                        "id": va.get("id"),
                                                        "name": va_full_name,
                                                        "native": va_name.get("native"),
                                                        "image": va_image_url,
                                                        "siteUrl": va.get("siteUrl"),
                                                        "count": 0,
                                                    }
                    # 統一+1
                    for va_full_name in unique_va_names:
                        voice_actor_count[va_full_name] = (
                            voice_actor_count.get(va_full_name, 0) + 1
                        )
                        # Store VA details
                        if va_full_name not in voice_actor_details:
                            voice_actor_details[va_full_name] = va_details_map[
                                va_full_name
                            ]
                        voice_actor_details[va_full_name]["count"] = voice_actor_count[
                            va_full_name
                        ]

        print(f"✅ 聲優數據抓取完成!")
        print(f"  - 找到的聲優總數: {len(voice_actor_count)}")

        print(f"\n📈 統計完成:")
        print(f"  - 總動漫數: {total_anime}")
        print(f"  - 總集數: {total_episodes}")
        print(f"  - 總時長: {round(total_minutes / 60, 1)} 小時")
        print(f"  - 完成數: {completed_count}")
        print(f"  - 平均評分: {round(average_score, 1)}")
        print(f"  - 製作公司數: {len(studio_count)}")
        print(f"  - 聲優數: {len(voice_actor_count)}")

        # Sort additional statistics
        top_tags = dict(
            sorted(tag_count.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        # Top studios with details
        top_studios_list = sorted(
            studio_details.values(), key=lambda x: x["count"], reverse=True
        )[:10]
        top_studios = {s["name"]: s for s in top_studios_list}

        # Top voice actors with details
        top_voice_actors_list = sorted(
            voice_actor_details.values(), key=lambda x: x["count"], reverse=True
        )[:20]
        top_voice_actors = {va["name"]: va for va in top_voice_actors_list}

        top_seasons = dict(
            sorted(season_count.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Monthly representative anime (for year mode)
        monthly_representative = {}
        if request.year is not None:
            for entry in filtered_list:
                completed_at = entry.get("completedAt", {})
                if (
                    completed_at
                    and completed_at.get("year") == request.year
                    and completed_at.get("month")
                ):
                    month = completed_at["month"]
                    score = entry.get("score", 0)
                    media = entry.get("media", {})

                    if score > 0:
                        if (
                            month not in monthly_representative
                            or score > monthly_representative[month]["score"]
                        ):
                            monthly_representative[month] = {
                                "month": month,
                                "id": media.get("id"),
                                "title": media.get("title", {}).get(
                                    "romaji", "Unknown"
                                ),
                                "title_english": media.get("title", {}).get("english"),
                                "coverImage": media.get("coverImage", {}).get(
                                    "large", ""
                                ),
                                "score": score,
                            }

        # Calculate achievements with tiers
        print("🏆 計算成就...")
        achievements = []
        total_hours = total_minutes / 60

        # Achievement: Anime Count (Bronze/Silver/Gold/Diamond)
        if completed_count >= 1000:
            achievements.append(
                {
                    "id": "anime_collector_diamond",
                    "title": "鑽石收藏家",
                    "description": f"完成了 {completed_count} 部動漫",
                    "icon": "💎",
                    "tier": "diamond",
                }
            )
        elif completed_count >= 500:
            achievements.append(
                {
                    "id": "anime_collector_gold",
                    "title": "黃金收藏家",
                    "description": f"完成了 {completed_count} 部動漫",
                    "icon": "🥇",
                    "tier": "gold",
                }
            )
        elif completed_count >= 100:
            achievements.append(
                {
                    "id": "anime_collector_silver",
                    "title": "白銀收藏家",
                    "description": f"完成了 {completed_count} 部動漫",
                    "icon": "🥈",
                    "tier": "silver",
                }
            )
        elif completed_count >= 50:
            achievements.append(
                {
                    "id": "anime_collector_bronze",
                    "title": "青銅收藏家",
                    "description": f"完成了 {completed_count} 部動漫",
                    "icon": "🥉",
                    "tier": "bronze",
                }
            )

        # Achievement: Episodes Watched
        if total_episodes >= 10000:
            achievements.append(
                {
                    "id": "episode_legend",
                    "title": "集數傳說",
                    "description": f"觀看了 {total_episodes} 集動漫",
                    "icon": "💎",
                    "tier": "diamond",
                }
            )
        elif total_episodes >= 5000:
            achievements.append(
                {
                    "id": "episode_master_gold",
                    "title": "集數宗師",
                    "description": f"觀看了 {total_episodes} 集動漫",
                    "icon": "🥇",
                    "tier": "gold",
                }
            )
        elif total_episodes >= 2000:
            achievements.append(
                {
                    "id": "episode_master_silver",
                    "title": "集數大師",
                    "description": f"觀看了 {total_episodes} 集動漫",
                    "icon": "🥈",
                    "tier": "silver",
                }
            )
        elif total_episodes >= 1000:
            achievements.append(
                {
                    "id": "episode_master_bronze",
                    "title": "集數達人",
                    "description": f"觀看了 {total_episodes} 集動漫",
                    "icon": "🥉",
                    "tier": "bronze",
                }
            )

        # Achievement: Time Spent
        if total_hours >= 1000:
            achievements.append(
                {
                    "id": "time_lord",
                    "title": "時空領主",
                    "description": f"花了 {total_hours:.0f} 小時在動漫上",
                    "icon": "💎",
                    "tier": "diamond",
                }
            )
        elif total_hours >= 500:
            achievements.append(
                {
                    "id": "time_traveler_gold",
                    "title": "時空旅者",
                    "description": f"花了 {total_hours:.0f} 小時在動漫上",
                    "icon": "🥇",
                    "tier": "gold",
                }
            )
        elif total_hours >= 300:
            achievements.append(
                {
                    "id": "time_traveler_silver",
                    "title": "時間旅行者",
                    "description": f"花了 {total_hours:.0f} 小時在動漫上",
                    "icon": "🥈",
                    "tier": "silver",
                }
            )
        elif total_hours >= 100:
            achievements.append(
                {
                    "id": "time_traveler_bronze",
                    "title": "時光探索者",
                    "description": f"花了 {total_hours:.0f} 小時在動漫上",
                    "icon": "🥉",
                    "tier": "bronze",
                }
            )

        # Achievement: High Score Average
        if average_score >= 85:
            achievements.append(
                {
                    "id": "generous_critic_gold",
                    "title": "慷慨的評論家",
                    "description": f"平均評分 {average_score:.1f}",
                    "icon": "🥇",
                    "tier": "gold",
                }
            )
        elif average_score >= 80:
            achievements.append(
                {
                    "id": "generous_critic_silver",
                    "title": "友善的評論家",
                    "description": f"平均評分 {average_score:.1f}",
                    "icon": "🥈",
                    "tier": "silver",
                }
            )
        elif average_score >= 75:
            achievements.append(
                {
                    "id": "generous_critic_bronze",
                    "title": "溫和的評論家",
                    "description": f"平均評分 {average_score:.1f}",
                    "icon": "🥉",
                    "tier": "bronze",
                }
            )

        # Achievement: Genre Expert
        if genre_count:
            top_genre = max(genre_count, key=genre_count.get)
            genre_total = genre_count[top_genre]
            if genre_total >= 100:
                achievements.append(
                    {
                        "id": "genre_master",
                        "title": f"{top_genre} 宗師",
                        "description": f"觀看了 {genre_total} 部 {top_genre} 動漫",
                        "icon": "🎭",
                        "tier": "gold",
                    }
                )
            elif genre_total >= 50:
                achievements.append(
                    {
                        "id": "genre_expert",
                        "title": f"{top_genre} 專家",
                        "description": f"觀看了 {genre_total} 部 {top_genre} 動漫",
                        "icon": "🎭",
                        "tier": "silver",
                    }
                )
            elif genre_total >= 20:
                achievements.append(
                    {
                        "id": "genre_fan",
                        "title": f"{top_genre} 愛好者",
                        "description": f"觀看了 {genre_total} 部 {top_genre} 動漫",
                        "icon": "🎭",
                        "tier": "bronze",
                    }
                )

        # Achievement: Perfect Record
        if dropped_count == 0 and completed_count >= 100:
            achievements.append(
                {
                    "id": "never_give_up_gold",
                    "title": "永不放棄！",
                    "description": f"完成 {completed_count} 部，零棄番！",
                    "icon": "💪",
                    "tier": "gold",
                }
            )
        elif dropped_count == 0 and completed_count >= 50:
            achievements.append(
                {
                    "id": "never_give_up_silver",
                    "title": "堅持到底",
                    "description": f"完成 {completed_count} 部，零棄番！",
                    "icon": "💪",
                    "tier": "silver",
                }
            )
        elif dropped_count == 0 and completed_count >= 20:
            achievements.append(
                {
                    "id": "never_give_up_bronze",
                    "title": "不輕言放棄",
                    "description": f"完成 {completed_count} 部，零棄番！",
                    "icon": "💪",
                    "tier": "bronze",
                }
            )

        # Achievement: Rewatcher
        if repeat_anime:
            max_repeat = max(a["repeat_count"] for a in repeat_anime)
            if max_repeat >= 5:
                achievements.append(
                    {
                        "id": "rewatcher_gold",
                        "title": "重溫大師",
                        "description": f"最多重看了 {max_repeat} 次",
                        "icon": "🔄",
                        "tier": "gold",
                    }
                )
            elif max_repeat >= 3:
                achievements.append(
                    {
                        "id": "rewatcher_silver",
                        "title": "重溫愛好者",
                        "description": f"最多重看了 {max_repeat} 次",
                        "icon": "🔄",
                        "tier": "silver",
                    }
                )
            elif max_repeat >= 2:
                achievements.append(
                    {
                        "id": "rewatcher_bronze",
                        "title": "二刷達成",
                        "description": f"重看了動漫 {max_repeat} 次",
                        "icon": "🔄",
                        "tier": "bronze",
                    }
                )

        # Achievement: Diverse Taste
        unique_genres = len(genre_count)
        if unique_genres >= 20:
            achievements.append(
                {
                    "id": "diverse_taste_gold",
                    "title": "全方位愛好者",
                    "description": f"涉獵了 {unique_genres} 種類型",
                    "icon": "🌈",
                    "tier": "gold",
                }
            )
        elif unique_genres >= 15:
            achievements.append(
                {
                    "id": "diverse_taste_silver",
                    "title": "多元品味",
                    "description": f"涉獵了 {unique_genres} 種類型",
                    "icon": "🌈",
                    "tier": "silver",
                }
            )
        elif unique_genres >= 10:
            achievements.append(
                {
                    "id": "diverse_taste_bronze",
                    "title": "廣泛興趣",
                    "description": f"涉獵了 {unique_genres} 種類型",
                    "icon": "🌈",
                    "tier": "bronze",
                }
            )

        # Achievement: Seasonal Binger
        if season_count:
            max_season = max(season_count.values())
            if max_season >= 30:
                achievements.append(
                    {
                        "id": "seasonal_champion",
                        "title": "追番冠軍",
                        "description": f"單季看了 {max_season} 部動漫",
                        "icon": "📅",
                        "tier": "gold",
                    }
                )
            elif max_season >= 20:
                achievements.append(
                    {
                        "id": "seasonal_enthusiast",
                        "title": "追番達人",
                        "description": f"單季看了 {max_season} 部動漫",
                        "icon": "📅",
                        "tier": "silver",
                    }
                )
            elif max_season >= 10:
                achievements.append(
                    {
                        "id": "seasonal_fan",
                        "title": "追番愛好者",
                        "description": f"單季看了 {max_season} 部動漫",
                        "icon": "📅",
                        "tier": "bronze",
                    }
                )

        print(f"✅ 成就計算完成，共 {len(achievements)} 個成就")
        print("=" * 60)
        print("🎉 Recap 生成完成！")
        print("=" * 60 + "\n")

        return {
            "username": request.username,
            "year": request.year,
            "is_all_time": request.year is None,
            "total_anime": total_anime,
            "total_episodes": total_episodes,
            "total_minutes": total_minutes,
            "total_hours": round(total_hours, 1),
            "completed_count": completed_count,
            "watching_count": watching_count,
            "dropped_count": dropped_count,
            "planned_count": planned_count,
            "paused_count": paused_count,
            "repeating_count": repeating_count,
            "top_anime": top_anime,
            "genre_distribution": dict(
                sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
            ),
            "format_distribution": dict(
                sorted(format_count.items(), key=lambda x: x[1], reverse=True)
            ),
            "tag_distribution": top_tags,
            "studio_distribution": top_studios,
            "voice_actor_distribution": top_voice_actors,
            "season_distribution": top_seasons,
            "month_added_distribution": dict(
                sorted(month_added_count.items(), key=lambda x: x[0])
            ),
            "month_completed_distribution": dict(
                sorted(month_completed_count.items(), key=lambda x: x[0])
            ),
            "monthly_representative": dict(
                sorted(monthly_representative.items(), key=lambda x: x[0])
            ),
            "most_rewatched": repeat_anime[:5],
            "average_score": round(average_score, 1),
            "total_scored": len(scores),
            "achievements": achievements,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Recap 生成失敗: {str(e)}")
        logger.error(f"Error generating recap: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate recap: {str(e)}"
        )
