import asyncio
import json
import logging
import os
from typing import Any, Dict, List

from anilist_client import AniListClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
USERS_FILE = os.path.join(DATA_DIR, "reference_users.json")

# A list of seed users to start building our dataset
# These are some popular users or users known to have diverse lists
# You can add more users here to expand the dataset
SEED_USERS = [
    "Gigguk",
]


async def ensure_data_dir():
    """Ensures the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory at {DATA_DIR}")


async def fetch_and_save_users(users: List[str]):
    """
    Fetches anime lists for the provided users and saves them to a JSON file.
    """
    client = AniListClient()
    all_user_data = {}

    # Load existing data if available
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                all_user_data = json.load(f)
            logger.info(f"Loaded {len(all_user_data)} existing users from {USERS_FILE}")
        except Exception as e:
            logger.error(f"Failed to load existing data: {e}")

    for username in users:
        if username in all_user_data:
            logger.info(f"User {username} already exists in dataset. Skipping.")
            continue

        logger.info(f"Fetching data for user: {username}")

        # Fetch user profile for metadata
        profile = await client.get_user_profile(username)
        if not profile:
            logger.warning(f"Could not fetch profile for {username}. Skipping.")
            continue

        # Fetch anime list
        anime_list = await client.get_user_anime_list(username)
        if not anime_list:
            logger.warning(
                f"Could not fetch anime list for {username} or list is empty. Skipping."
            )
            continue

        logger.info(f"Fetched {len(anime_list)} entries for {username}")

        # Structure the data
        user_record = {"profile": profile, "anime_list": anime_list}

        all_user_data[username] = user_record

        # Save incrementally to prevent data loss
        try:
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(all_user_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved data for {username}")
        except Exception as e:
            logger.error(f"Failed to save data for {username}: {e}")

        # Rate limiting - be nice to the API
        await asyncio.sleep(1.0)

    logger.info(f"Data ingestion complete. Total users: {len(all_user_data)}")


async def main():
    await ensure_data_dir()
    await fetch_and_save_users(SEED_USERS)


if __name__ == "__main__":
    asyncio.run(main())
