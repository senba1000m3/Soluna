import logging
from typing import Any, Dict, List, Optional

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ANILIST_API_URL = "https://graphql.anilist.co"


class AniListClient:
    def __init__(self):
        self.url = ANILIST_API_URL

    async def _post_request(
        self, query: str, variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Helper method to send GraphQL requests to AniList.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.url,
                    json={"query": query, "variables": variables},
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                data = response.json()

                if "errors" in data:
                    logger.error(f"AniList API Errors: {data['errors']}")
                    # In a production app, you might want to parse specific error codes
                    # For now, we just raise the first error message
                    raise Exception(
                        f"AniList API Error: {data['errors'][0]['message']}"
                    )

                return data["data"]
            except httpx.HTTPStatusError as e:
                logger.error(
                    f"HTTP Error: {e.response.status_code} - {e.response.text}"
                )
                raise
            except Exception as e:
                logger.error(f"Request failed: {str(e)}")
                raise

    async def get_user_anime_list(self, username: str) -> List[Dict[str, Any]]:
        """
        Fetches a user's anime list with status, scores, and progress.
        Useful for drop prediction and synergy matching.
        """
        query = """
        query ($username: String) {
          MediaListCollection(userName: $username, type: ANIME) {
            lists {
              name
              entries {
                id
                status
                score
                progress
                media {
                  id
                  title {
                    romaji
                    english
                  }
                  genres
                  tags {
                    name
                    rank
                  }
                  averageScore
                  popularity
                  coverImage {
                    large
                  }
                  episodes
                  season
                  seasonYear
                  studios(isMain: true) {
                    nodes {
                      name
                    }
                  }
                }
              }
            }
          }
        }
        """
        variables = {"username": username}

        try:
            data = await self._post_request(query, variables)
            # Flatten the lists into a single list of entries
            all_entries = []
            if (
                data
                and "MediaListCollection" in data
                and "lists" in data["MediaListCollection"]
            ):
                for lst in data["MediaListCollection"]["lists"]:
                    if lst["entries"]:
                        all_entries.extend(lst["entries"])
            return all_entries
        except Exception as e:
            logger.error(f"Failed to fetch anime list for user {username}: {e}")
            return []

    async def search_anime(
        self, search_term: str, page: int = 1, per_page: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Searches for anime by title.
        """
        query = """
        query ($search: String, $page: Int, $perPage: Int) {
          Page(page: $page, perPage: $perPage) {
            media(search: $search, type: ANIME, sort: POPULARITY_DESC) {
              id
              title {
                romaji
                english
              }
              description
              genres
              averageScore
              status
              season
              seasonYear
              coverImage {
                large
              }
            }
          }
        }
        """
        variables = {"search": search_term, "page": page, "perPage": per_page}

        try:
            data = await self._post_request(query, variables)
            return data["Page"]["media"]
        except Exception as e:
            logger.error(f"Failed to search anime '{search_term}': {e}")
            return []

    async def get_anime_details(self, anime_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetches detailed information for a specific anime ID.
        """
        query = """
        query ($id: Int) {
          Media(id: $id, type: ANIME) {
            id
            title {
              romaji
              english
            }
            description
            genres
            tags {
              name
              rank
              category
            }
            averageScore
            meanScore
            popularity
            favourites
            studios(isMain: true) {
              nodes {
                name
              }
            }
            startDate {
              year
              month
              day
            }
            endDate {
              year
              month
              day
            }
            episodes
            duration
            season
            seasonYear
            coverImage {
              extraLarge
            }
            bannerImage
          }
        }
        """
        variables = {"id": anime_id}

        try:
            data = await self._post_request(query, variables)
            return data["Media"]
        except Exception as e:
            logger.error(f"Failed to fetch details for anime ID {anime_id}: {e}")
            return None

    async def get_seasonal_anime(
        self, season: str, year: int, page: int = 1, per_page: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fetches anime for a specific season and year.
        Season: WINTER, SPRING, SUMMER, FALL
        """
        query = """
        query ($season: MediaSeason, $year: Int, $page: Int, $perPage: Int) {
          Page(page: $page, perPage: $perPage) {
            media(season: $season, seasonYear: $year, type: ANIME, sort: POPULARITY_DESC) {
              id
              title {
                romaji
                english
              }
              genres
              tags {
                name
                rank
              }
              description
              averageScore
              popularity
              coverImage {
                large
              }
            }
          }
        }
        """
        variables = {
            "season": season.upper(),
            "year": year,
            "page": page,
            "perPage": per_page,
        }

        try:
            data = await self._post_request(query, variables)
            return data["Page"]["media"]
        except Exception as e:
            logger.error(f"Failed to fetch seasonal anime {season} {year}: {e}")
            return []

    async def get_user_profile(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Fetches basic user profile info.
        """
        query = """
        query ($username: String) {
          User(name: $username) {
            id
            name
            dateOfBirth {
              year
              month
              day
            }
            avatar {
              large
            }
            statistics {
              anime {
                count
                meanScore
                minutesWatched
                episodesWatched
              }
            }
          }
        }
        """
        variables = {"username": username}

        try:
            data = await self._post_request(query, variables)
            return data["User"]
        except Exception as e:
            logger.error(f"Failed to fetch profile for user {username}: {e}")
            return None

    async def get_top_anime_by_year(
        self, year: int, per_page: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fetches top rated/popular anime for a specific year.
        """
        query = """
        query ($year: Int, $perPage: Int) {
          Page(page: 1, perPage: $perPage) {
            media(seasonYear: $year, type: ANIME, sort: POPULARITY_DESC) {
              id
              title {
                romaji
                english
              }
              coverImage {
                large
              }
              averageScore
              popularity
              genres
            }
          }
        }
        """
        variables = {"year": year, "perPage": per_page}

        try:
            data = await self._post_request(query, variables)
            return data["Page"]["media"]
        except Exception as e:
            logger.error(f"Failed to fetch top anime for year {year}: {e}")
            return []
