import asyncio
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
        print(f"🌐 [AniList API] 準備發送請求到 {self.url}")
        print(f"📝 [AniList API] Variables: {variables}")

        async with httpx.AsyncClient() as client:
            try:
                print(f"📤 [AniList API] 發送 POST 請求...")
                response = await client.post(
                    self.url,
                    json={"query": query, "variables": variables},
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    timeout=30.0,
                )
                print(f"📥 [AniList API] 收到回應，狀態碼: {response.status_code}")
                response.raise_for_status()
                data = response.json()

                print(f"✅ [AniList API] JSON 解析成功")

                if "errors" in data:
                    print(f"❌ [AniList API] API 返回錯誤: {data['errors']}")
                    logger.error(f"AniList API Errors: {data['errors']}")
                    # In a production app, you might want to parse specific error codes
                    # For now, we just raise the first error message
                    raise Exception(
                        f"AniList API Error: {data['errors'][0]['message']}"
                    )

                print(f"✅ [AniList API] 請求成功完成")
                return data["data"]
            except httpx.HTTPStatusError as e:
                print(f"❌ [AniList API] HTTP 錯誤: {e.response.status_code}")
                print(f"   回應內容: {e.response.text[:200]}")
                logger.error(
                    f"HTTP Error: {e.response.status_code} - {e.response.text}"
                )
                raise
            except httpx.TimeoutException as e:
                print(f"❌ [AniList API] 請求超時")
                logger.error(f"Request timeout: {str(e)}")
                raise
            except Exception as e:
                print(f"❌ [AniList API] 請求失敗: {str(e)}")
                logger.error(f"Request failed: {str(e)}", exc_info=True)
                raise

    async def get_user_anime_list(self, username: str) -> List[Dict[str, Any]]:
        """
        Fetches a user's anime list with status, scores, and progress.
        Useful for drop prediction and synergy matching.
        """
        print(f"🔍 [AniList Client] 抓取使用者動漫列表: {username}")
        logger.info(f"Fetching anime list for user: {username}")
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
                updatedAt
                startedAt {
                  year
                  month
                  day
                }
                completedAt {
                  year
                  month
                  day
                }
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
                  duration
                  season
                  seasonYear
                  format
                  studios(isMain: true) {
                    nodes {
                      id
                      name
                      siteUrl
                    }
                  }
                  characters(page: 1, perPage: 10, sort: ROLE) {
                    edges {
                      role
                      voiceActors(language: JAPANESE, sort: RELEVANCE) {
                        id
                        name {
                          full
                          native
                        }
                        image {
                          large
                          medium
                        }
                        siteUrl
                      }
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
            print(f"📡 [AniList Client] 發送 GraphQL 請求...")
            data = await self._post_request(query, variables)
            print(f"✅ [AniList Client] 收到回應")
            # Flatten the lists into a single list of entries
            all_entries = []
            if (
                data
                and "MediaListCollection" in data
                and "lists" in data["MediaListCollection"]
            ):
                print(f"📊 [AniList Client] 處理列表資料...")
                for lst in data["MediaListCollection"]["lists"]:
                    if lst["entries"]:
                        print(f"  - 列表 '{lst['name']}': {len(lst['entries'])} 筆")
                        all_entries.extend(lst["entries"])

            print(f"✅ [AniList Client] 成功取得 {len(all_entries)} 筆動漫資料")
            logger.info(
                f"Successfully fetched {len(all_entries)} anime entries for {username}"
            )
            return all_entries
        except Exception as e:
            print(f"❌ [AniList Client] 抓取失敗: {str(e)}")
            logger.error(
                f"Failed to fetch anime list for user {username}: {e}", exc_info=True
            )
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
        Note: dateOfBirth is removed as it's private and causes 400 errors.
        """
        query = """
        query ($username: String) {
          User(name: $username) {
            id
            name
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

    async def get_characters_by_birthday(
        self, month: int, day: int
    ) -> List[Dict[str, Any]]:
        """
        Fetches characters who share the same birthday.
        Uses AniList's isBirthday filter (only works for today's date).
        Falls back to Jikan API for other dates.
        """
        from datetime import datetime

        today = datetime.now()
        is_today = today.month == month and today.day == day

        # If it's today, use AniList's isBirthday filter
        if is_today:
            query = """
            query {
              Page(page: 1, perPage: 10) {
                characters(isBirthday: true, sort: FAVOURITES_DESC) {
                  id
                  name {
                    full
                    native
                  }
                  dateOfBirth {
                    month
                    day
                  }
                  image {
                    large
                  }
                  favourites
                  media(sort: POPULARITY_DESC, perPage: 1) {
                    nodes {
                      title {
                        romaji
                        english
                      }
                    }
                  }
                }
              }
            }
            """

            try:
                data = await self._post_request(query, {})
                return data["Page"]["characters"]
            except Exception as e:
                logger.error(f"Failed to fetch today's birthday characters: {e}")
                return []

        # For other dates, use Jikan API (MyAnimeList)
        try:
            import httpx

            # Fetch top characters and filter by birthday
            jikan_url = f"https://api.jikan.moe/v4/characters"
            params = {"order_by": "favorites", "sort": "desc", "limit": 50}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(jikan_url, params=params)
                response.raise_for_status()
                data = response.json()

                # Filter characters by birthday
                characters = []
                for char in data.get("data", []):
                    if len(characters) >= 10:
                        break

                    birthday = char.get("birthday")
                    if not birthday:
                        continue

                    # Parse birthday string (format: "YYYY-MM-DDTHH:MM:SS+00:00" or null)
                    try:
                        date_part = birthday.split("T")[0]
                        parts = date_part.split("-")
                        if len(parts) >= 3:
                            char_month = int(parts[1])
                            char_day = int(parts[2])

                            if char_month == month and char_day == day:
                                # Fetch detailed character info including anime
                                char_id = char.get("mal_id")
                                detail_url = f"https://api.jikan.moe/v4/characters/{char_id}/full"

                                await asyncio.sleep(0.5)  # Rate limiting
                                detail_response = await client.get(detail_url)
                                if detail_response.status_code == 200:
                                    detail_data = detail_response.json()
                                    char_detail = detail_data.get("data", {})

                                    # Get most popular anime
                                    anime_list = char_detail.get("anime", [])
                                    anime_title = "未知作品"
                                    if anime_list and len(anime_list) > 0:
                                        anime_title = (
                                            anime_list[0]
                                            .get("anime", {})
                                            .get("title", "未知作品")
                                        )

                                    # Convert to AniList format
                                    characters.append(
                                        {
                                            "id": char.get("mal_id"),
                                            "name": {
                                                "full": char.get("name"),
                                                "native": char.get("name_kanji", ""),
                                            },
                                            "image": {
                                                "large": char.get("images", {})
                                                .get("jpg", {})
                                                .get("image_url", "")
                                            },
                                            "favourites": char.get("favorites", 0),
                                            "media": {
                                                "nodes": [
                                                    {
                                                        "title": {
                                                            "romaji": anime_title,
                                                            "english": anime_title,
                                                        }
                                                    }
                                                ]
                                            },
                                            "source": "jikan",
                                        }
                                    )
                    except (ValueError, IndexError) as parse_error:
                        logger.debug(
                            f"Failed to parse birthday {birthday}: {parse_error}"
                        )
                        continue

                logger.info(
                    f"Found {len(characters)} characters from Jikan API for {month}/{day}"
                )
                return characters

        except Exception as e:
            logger.error(
                f"Failed to fetch characters from Jikan API for {month}/{day}: {e}"
            )
            return []

    async def get_anime_voice_actors(self, anime_id: int) -> Dict[str, Any]:
        """
        Fetches voice actor data for a specific anime.
        This is needed because MediaListCollection query doesn't return voice actor info.
        """
        print(f"🎤 [AniList Client] 抓取動漫聲優資料: {anime_id}")
        logger.info(f"Fetching voice actors for anime ID: {anime_id}")

        query = """
        query ($id: Int) {
          Media(id: $id, type: ANIME) {
            id
            characters(page: 1, perPage: 25, sort: ROLE) {
              edges {
                role
                node {
                  id
                  name {
                    full
                    native
                  }
                }
                voiceActors(language: JAPANESE, sort: RELEVANCE) {
                  id
                  name {
                    full
                    native
                  }
                  image {
                    large
                    medium
                  }
                  siteUrl
                }
              }
            }
          }
        }
        """

        variables = {"id": anime_id}

        try:
            print(f"📡 [AniList Client] 發送聲優資料請求...")
            data = await self._post_request(query, variables)

            if data and "Media" in data:
                print(f"✅ [AniList Client] 成功取得動漫 {anime_id} 的聲優資料")
                logger.info(f"Successfully fetched voice actors for anime {anime_id}")
                return data["Media"]
            else:
                print(f"⚠️ [AniList Client] 沒有找到動漫 {anime_id} 的資料")
                logger.warning(f"No data found for anime {anime_id}")
                return {}

        except Exception as e:
            print(f"❌ [AniList Client] 抓取聲優資料失敗: {str(e)}")
            logger.error(
                f"Failed to fetch voice actors for anime {anime_id}: {e}", exc_info=True
            )
            return {}
