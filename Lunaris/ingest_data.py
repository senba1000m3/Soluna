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


async def fetch_and_store_user_data(session: Session, username: str):
    """
    Fetches a user's list and stores it as ratings.
    """
    logger.info(f"Fetching list for user: {username}...")

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
        entries = await anilist_client.get_user_anime_list(username)
        if not entries:
            logger.warning(f"No anime list found for user {username}")
            return
    except Exception as e:
        logger.error(f"Error fetching anime list for {username}: {e}")
        raise

    new_ratings_count = 0
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
                # Commit immediately to satisfy foreign key constraint for rating
                session.commit()

            # Check if rating exists
            # In a real app, we might want to update existing ratings
            statement = select(UserRating).where(
                UserRating.user_id == user.id, UserRating.anime_id == anime_id
            )
            existing_rating = session.exec(statement).first()

            if not existing_rating:
                rating = UserRating(
                    user_id=user.id,
                    anime_id=anime_id,
                    score=entry.get("score", 0),
                    status=entry.get("status", "UNKNOWN"),
                    progress=entry.get("progress", 0),
                )
                session.add(rating)
                new_ratings_count += 1
        except Exception as e:
            logger.error(f"Error processing entry for anime {anime_id}: {e}")
            continue

    session.commit()
    logger.info(f"Stored {new_ratings_count} new ratings for {username}.")


async def main():
    init_db()
    session = next(get_session())

    # Example: Fetch some seasonal data to populate Anime table
    await fetch_and_store_anime(session, 2024, "WINTER")
    await fetch_and_store_anime(session, 2023, "FALL")

    # Example: Fetch a user's data
    # You can add more users here to build your dataset
    await fetch_and_store_user_data(session, "Gigguk")

    session.close()


if __name__ == "__main__":
    asyncio.run(main())
