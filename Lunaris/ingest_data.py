import asyncio
import logging
from typing import List

from sqlmodel import Session, select

from anilist_client import AniListClient
from database import get_session, init_db
from models import Anime, User, UserRating

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

anilist_client = AniListClient()


async def fetch_and_store_anime(session: Session, year: int, season: str):
    """
    Fetches seasonal anime and stores them in the database.
    """
    logger.info(f"Fetching anime for {season} {year}...")
    anime_list = await anilist_client.get_seasonal_anime(season, year, per_page=50)

    for anime_data in anime_list:
        anime_id = anime_data["id"]

        # Check if anime already exists
        existing_anime = session.get(Anime, anime_id)
        if existing_anime:
            continue

        # Create new Anime record
        anime = Anime(
            id=anime_id,
            title_romaji=anime_data["title"]["romaji"],
            title_english=anime_data["title"]["english"],
            genres=",".join(anime_data["genres"]),
            average_score=anime_data.get("averageScore"),
            popularity=anime_data.get("popularity"),
            tags=",".join([t["name"] for t in (anime_data.get("tags") or [])]),
            # Note: Some fields might be missing in the basic query,
            # for a full ML model we might need a more detailed query later.
        )
        session.add(anime)

    session.commit()
    logger.info(f"Stored {len(anime_list)} anime records.")


async def fetch_and_store_user_data(
    session: Session, username: str, progress_tracker=None
):
    """
    Fetches a user's list and stores it as ratings.
    """
    logger.info(f"Fetching list for user: {username}...")

    if progress_tracker:
        progress_tracker.update(progress=10, message="正在抓取使用者資料...")

    try:
        # 1. Get or Create User
        statement = select(User).where(User.username == username)
        user = session.exec(statement).first()

        if not user:
            # We need to fetch the AniList ID first
            profile = await anilist_client.get_user_profile(username)
            if not profile:
                logger.error(f"User {username} not found on AniList.")
                raise ValueError(f"User {username} not found on AniList.")

            user = User(username=username, anilist_id=profile["id"])
            session.add(user)
            session.commit()
            session.refresh(user)
    except Exception as e:
        logger.error(f"Error creating/fetching user {username}: {e}")
        raise

    # 2. Fetch Anime List
    try:
        if progress_tracker:
            progress_tracker.update(progress=12, message="正在抓取動漫清單...")

        entries = await anilist_client.get_user_anime_list(username)
        if not entries:
            logger.warning(f"No anime list found for user {username}")
            return

        if progress_tracker:
            progress_tracker.update(
                progress=15, message=f"抓取到 {len(entries)} 筆動漫記錄"
            )
    except Exception as e:
        logger.error(f"Error fetching anime list for {username}: {e}")
        raise

    new_ratings_count = 0
    commit_batch_size = 50  # 每 50 筆 commit 一次
    processed_count = 0
    total_entries = len(entries)

    if progress_tracker:
        progress_tracker.update(progress=18, message="開始儲存動漫資料...")

    for entry in entries:
        try:
            media = entry.get("media")
            if not media:
                logger.warning(f"Entry missing media data, skipping")
                continue

            anime_id = media.get("id")
            if not anime_id:
                logger.warning(f"Media missing ID, skipping")
                continue

            # Ensure Anime exists in DB
            if not session.get(Anime, anime_id):
                title = media.get("title", {})
                anime = Anime(
                    id=anime_id,
                    title_romaji=title.get("romaji", "Unknown"),
                    title_english=title.get("english"),
                    genres=",".join(media.get("genres", [])),
                    average_score=media.get("averageScore"),
                    popularity=media.get("popularity"),
                    episodes=media.get("episodes"),
                    season=media.get("season"),
                    season_year=media.get("seasonYear"),
                    studios=",".join(
                        [
                            s["name"]
                            for s in (media.get("studios") or {}).get("nodes", [])
                        ]
                    ),
                    tags=",".join([t["name"] for t in (media.get("tags") or [])]),
                )
                session.add(anime)

            # Check if rating exists and update or create
            statement = select(UserRating).where(
                UserRating.user_id == user.id, UserRating.anime_id == anime_id
            )
            existing_rating = session.exec(statement).first()

            if existing_rating:
                # Update existing rating with latest data
                existing_rating.score = entry.get("score", 0)
                existing_rating.status = entry.get("status", "UNKNOWN")
                existing_rating.progress = entry.get("progress", 0)
                session.add(existing_rating)
            else:
                # Create new rating
                rating = UserRating(
                    user_id=user.id,
                    anime_id=anime_id,
                    score=entry.get("score", 0),
                    status=entry.get("status", "UNKNOWN"),
                    progress=entry.get("progress", 0),
                )
                session.add(rating)
                new_ratings_count += 1

            processed_count += 1

            # 批次 commit 以提升性能
            if processed_count % commit_batch_size == 0:
                session.commit()
                logger.info(f"已處理 {processed_count}/{total_entries} 筆記錄")

                # 更新進度：18% ~ 28%
                if progress_tracker:
                    progress_percentage = 18 + int(
                        (processed_count / total_entries) * 10
                    )
                    progress_tracker.update(
                        progress=progress_percentage,
                        message=f"儲存中... ({processed_count}/{total_entries})",
                    )

        except Exception as e:
            logger.error(f"Error processing entry for anime {anime_id}: {e}")
            continue

    # 最終 commit 確保所有剩餘數據都被保存
    session.commit()
    logger.info(f"完成處理 {processed_count} 筆記錄，新增 {new_ratings_count} 筆評分")

    if progress_tracker:
        progress_tracker.update(
            progress=30, message=f"資料儲存完成！共 {processed_count} 筆記錄"
        )

    logger.info(
        f"Updated {username}'s list: {new_ratings_count} new ratings, "
        f"{total_entries - new_ratings_count} updated ratings."
    )


async def main():
    init_db()
    session = next(get_session())

    # Example: Fetch some seasonal data to populate Anime table
    await fetch_and_store_anime(session, 2024, "WINTER")
    await fetch_and_store_anime(session, 2023, "FALL")

    # Example: Fetch a user's data
    # You can add more users here to build your dataset
    await fetch_and_store_user_data(session, "senba1000m3")

    session.close()


if __name__ == "__main__":
    asyncio.run(main())
