"""
準備 BERT 訓練資料集
從 AniList 抓取熱門動畫資料並儲存到資料庫

使用方式:
    python prepare_bert_dataset.py --count 3000
    python prepare_bert_dataset.py --count 5000 --min-popularity 1000
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlmodel import Field, Session, SQLModel, create_engine, select
from tqdm import tqdm

from anilist_client import AniListClient

# Fix Windows encoding issue
if sys.platform == "win32":
    import codecs

    # Check if stdout/stderr have buffer attribute (not in uv run)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("prepare_bert_dataset.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# BERT 專用資料庫
BERT_DB_PATH = "bert.db"
BERT_DB_URL = f"sqlite:///{BERT_DB_PATH}"


# ============================================================================
# 資料庫模型 (專為 BERT 訓練設計)
# ============================================================================


class BERTAnime(SQLModel, table=True):
    """BERT 訓練用的動畫資料"""

    __tablename__ = "bert_anime"

    id: int = Field(primary_key=True)  # AniList ID
    title_romaji: str
    title_english: Optional[str] = None
    title_native: Optional[str] = None

    # 基本資訊
    format: Optional[str] = None  # TV, MOVIE, OVA, etc.
    episodes: Optional[int] = None
    duration: Optional[int] = None  # 分鐘
    status: Optional[str] = None  # FINISHED, RELEASING, etc.

    # 日期
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None
    season: Optional[str] = None  # WINTER, SPRING, SUMMER, FALL
    season_year: Optional[int] = None

    # 評分與人氣
    average_score: Optional[int] = None  # 0-100
    mean_score: Optional[int] = None
    popularity: Optional[int] = None
    favourites: Optional[int] = None
    trending: Optional[int] = None

    # 分類 (JSON 字串)
    genres: str = Field(default="[]")  # JSON array
    tags: str = Field(default="[]")  # JSON array with weights
    studios: str = Field(default="[]")  # JSON array

    # 關聯
    source: Optional[str] = None  # MANGA, LIGHT_NOVEL, etc.
    is_adult: bool = False

    # Metadata
    cover_image: Optional[str] = None
    banner_image: Optional[str] = None
    description: Optional[str] = None

    # 時間戳記
    fetched_at: datetime = Field(default_factory=datetime.utcnow)


class BERTUserAnimeList(SQLModel, table=True):
    """BERT 訓練用的使用者動畫列表"""

    __tablename__ = "bert_user_anime_list"

    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int  # AniList User ID
    username: str
    anime_id: int  # 對應 BERTAnime.id

    # 使用者互動資料
    status: str  # COMPLETED, WATCHING, DROPPED, etc.
    score: int = 0  # 0-100
    progress: int = 0  # 看到第幾集
    repeat: int = 0  # 重看次數
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Metadata
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BERTDatasetMetadata(SQLModel, table=True):
    """資料集元數據"""

    __tablename__ = "bert_dataset_metadata"

    id: Optional[int] = Field(default=None, primary_key=True)
    key: str = Field(unique=True)
    value: str
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# 資料庫初始化
# ============================================================================


def init_bert_db():
    """初始化 BERT 資料庫"""
    engine = create_engine(BERT_DB_URL, echo=False)
    SQLModel.metadata.create_all(engine)
    logger.info(f"✅ BERT 資料庫初始化完成: {BERT_DB_PATH}")
    return engine


# ============================================================================
# 資料抓取器
# ============================================================================


class BERTDatasetPreparer:
    """準備 BERT 訓練資料集"""

    def __init__(self, engine):
        self.engine = engine
        self.client = AniListClient()
        self.stats = {
            "total_anime": 0,
            "new_anime": 0,
            "updated_anime": 0,
            "failed_anime": 0,
        }

    async def fetch_popular_anime(
        self,
        target_count: int = 3000,
        min_popularity: int = 0,
        per_page: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        抓取熱門動畫

        Args:
            target_count: 目標動畫數量
            min_popularity: 最低人氣值
            per_page: 每頁數量

        Returns:
            動畫列表
        """
        print("\n" + "=" * 80)
        print("🎯 開始抓取熱門動畫資料")
        print("=" * 80)
        print(f"  目標數量: {target_count}")
        print(f"  最低人氣: {min_popularity}")
        print("=" * 80)

        query = """
        query ($page: Int, $perPage: Int, $sort: [MediaSort], $minPopularity: Int) {
          Page(page: $page, perPage: $perPage) {
            pageInfo {
              total
              currentPage
              lastPage
              hasNextPage
            }
            media(
              type: ANIME,
              sort: $sort,
              popularity_greater: $minPopularity,
              isAdult: false
            ) {
              id
              title {
                romaji
                english
                native
              }
              format
              episodes
              duration
              status
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
              season
              seasonYear
              averageScore
              meanScore
              popularity
              favourites
              trending
              genres
              tags {
                id
                name
                rank
                isMediaSpoiler
              }
              studios(isMain: true) {
                nodes {
                  id
                  name
                }
              }
              source
              isAdult
              coverImage {
                large
                extraLarge
              }
              bannerImage
              description
            }
          }
        }
        """

        all_anime = []
        page = 1
        has_next_page = True

        with tqdm(total=target_count, desc="抓取動畫", unit="部") as pbar:
            while has_next_page and len(all_anime) < target_count:
                try:
                    variables = {
                        "page": page,
                        "perPage": per_page,
                        "sort": ["POPULARITY_DESC"],
                        "minPopularity": min_popularity,
                    }

                    data = await self.client._post_request(query, variables)

                    if not data or "Page" not in data:
                        logger.warning(f"頁面 {page} 沒有資料")
                        break

                    page_info = data["Page"]["pageInfo"]
                    media_list = data["Page"]["media"]

                    all_anime.extend(media_list)
                    pbar.update(len(media_list))

                    has_next_page = page_info.get("hasNextPage", False)
                    page += 1

                    logger.info(f"已抓取第 {page - 1} 頁，累計 {len(all_anime)} 部動畫")

                    # 避免過度請求
                    await asyncio.sleep(1)

                except Exception as e:
                    logger.error(f"抓取第 {page} 頁時發生錯誤: {e}")
                    self.stats["failed_anime"] += 1
                    break

        print(f"\n✅ 抓取完成！共取得 {len(all_anime)} 部動畫")
        return all_anime[:target_count]

    def _format_date(self, date_dict: Optional[Dict]) -> Optional[str]:
        """格式化日期為 YYYY-MM-DD"""
        if not date_dict:
            return None
        year = date_dict.get("year")
        month = date_dict.get("month")
        day = date_dict.get("day")

        if not year:
            return None

        return f"{year:04d}-{month or 1:02d}-{day or 1:02d}"

    async def store_anime(self, anime_list: List[Dict[str, Any]]) -> None:
        """
        儲存動畫資料到資料庫

        Args:
            anime_list: 動畫列表
        """
        print("\n" + "=" * 80)
        print("💾 儲存動畫資料到資料庫")
        print("=" * 80)

        with Session(self.engine) as session:
            for anime_data in tqdm(anime_list, desc="儲存動畫", unit="部"):
                try:
                    anime_id = anime_data["id"]

                    # 檢查是否已存在
                    existing = session.get(BERTAnime, anime_id)

                    # 處理 tags (過濾掉劇透標籤並提取權重)
                    tags = [
                        {"name": t["name"], "rank": t.get("rank", 0)}
                        for t in anime_data.get("tags", [])
                        if not t.get("isMediaSpoiler", False)
                    ]

                    # 處理 studios
                    studios = [
                        s["name"]
                        for s in anime_data.get("studios", {}).get("nodes", [])
                    ]

                    anime_obj = BERTAnime(
                        id=anime_id,
                        title_romaji=anime_data["title"]["romaji"],
                        title_english=anime_data["title"].get("english"),
                        title_native=anime_data["title"].get("native"),
                        format=anime_data.get("format"),
                        episodes=anime_data.get("episodes"),
                        duration=anime_data.get("duration"),
                        status=anime_data.get("status"),
                        start_date=self._format_date(anime_data.get("startDate")),
                        end_date=self._format_date(anime_data.get("endDate")),
                        season=anime_data.get("season"),
                        season_year=anime_data.get("seasonYear"),
                        average_score=anime_data.get("averageScore"),
                        mean_score=anime_data.get("meanScore"),
                        popularity=anime_data.get("popularity"),
                        favourites=anime_data.get("favourites"),
                        trending=anime_data.get("trending"),
                        genres=json.dumps(
                            anime_data.get("genres", []), ensure_ascii=False
                        ),
                        tags=json.dumps(tags, ensure_ascii=False),
                        studios=json.dumps(studios, ensure_ascii=False),
                        source=anime_data.get("source"),
                        is_adult=anime_data.get("isAdult", False),
                        cover_image=anime_data.get("coverImage", {}).get("extraLarge")
                        or anime_data.get("coverImage", {}).get("large"),
                        banner_image=anime_data.get("bannerImage"),
                        description=anime_data.get("description"),
                        fetched_at=datetime.utcnow(),
                    )

                    if existing:
                        # 更新現有資料
                        for field, value in anime_obj.dict(exclude={"id"}).items():
                            setattr(existing, field, value)
                        self.stats["updated_anime"] += 1
                    else:
                        # 新增資料
                        session.add(anime_obj)
                        self.stats["new_anime"] += 1

                    self.stats["total_anime"] += 1

                    # 每 100 筆 commit 一次
                    if self.stats["total_anime"] % 100 == 0:
                        session.commit()

                except Exception as e:
                    logger.error(f"儲存動畫 {anime_data.get('id')} 時發生錯誤: {e}")
                    self.stats["failed_anime"] += 1
                    continue

            # 最終 commit
            session.commit()

        print(f"\n✅ 儲存完成！")
        print(f"  新增: {self.stats['new_anime']} 部")
        print(f"  更新: {self.stats['updated_anime']} 部")
        print(f"  失敗: {self.stats['failed_anime']} 部")

    async def fetch_user_list(
        self, username: str, user_id: Optional[int] = None
    ) -> None:
        """
        抓取並儲存使用者動畫列表

        Args:
            username: AniList 使用者名稱
            user_id: AniList 使用者 ID (如果已知)
        """
        print(f"\n{'=' * 80}")
        print(f"📝 抓取使用者列表: {username}")
        print(f"{'=' * 80}")

        try:
            # 取得使用者資料
            if not user_id:
                profile = await self.client.get_user_profile(username)
                if not profile:
                    logger.error(f"找不到使用者: {username}")
                    return
                user_id = profile["id"]

            # 取得動畫列表
            user_list = await self.client.get_user_anime_list(username)

            if not user_list:
                logger.warning(f"使用者 {username} 沒有動畫列表")
                return

            print(f"  ✓ 取得 {len(user_list)} 筆動畫記錄")

            # 儲存到資料庫
            with Session(self.engine) as session:
                stored_count = 0
                for entry in tqdm(user_list, desc="儲存列表", unit="筆"):
                    try:
                        media = entry.get("media", {})
                        anime_id = media.get("id")

                        if not anime_id:
                            continue

                        # 確保動畫存在於 bert_anime 表
                        if not session.get(BERTAnime, anime_id):
                            logger.warning(f"動畫 {anime_id} 不在資料庫中，跳過")
                            continue

                        # 檢查是否已存在
                        existing = session.exec(
                            select(BERTUserAnimeList).where(
                                BERTUserAnimeList.user_id == user_id,
                                BERTUserAnimeList.anime_id == anime_id,
                            )
                        ).first()

                        started_at = entry.get("startedAt")
                        completed_at = entry.get("completedAt")

                        user_anime = BERTUserAnimeList(
                            user_id=user_id,
                            username=username,
                            anime_id=anime_id,
                            status=entry.get("status", "UNKNOWN"),
                            score=entry.get("score", 0),
                            progress=entry.get("progress", 0),
                            repeat=entry.get("repeat", 0),
                            started_at=self._format_date(started_at),
                            completed_at=self._format_date(completed_at),
                            updated_at=datetime.utcnow(),
                        )

                        if existing:
                            for field, value in user_anime.dict(exclude={"id"}).items():
                                setattr(existing, field, value)
                        else:
                            session.add(user_anime)

                        stored_count += 1

                    except Exception as e:
                        logger.error(f"儲存使用者列表項目時發生錯誤: {e}")
                        continue

                session.commit()

            print(f"\n✅ 使用者列表儲存完成！共 {stored_count} 筆")

        except Exception as e:
            logger.error(f"抓取使用者列表失敗: {e}")

    def save_metadata(self, key: str, value: str) -> None:
        """儲存資料集元數據"""
        with Session(self.engine) as session:
            existing = session.exec(
                select(BERTDatasetMetadata).where(BERTDatasetMetadata.key == key)
            ).first()

            if existing:
                existing.value = value
                existing.updated_at = datetime.utcnow()
            else:
                metadata = BERTDatasetMetadata(key=key, value=value)
                session.add(metadata)

            session.commit()

    def print_summary(self) -> None:
        """列印資料集摘要"""
        with Session(self.engine) as session:
            anime_count = len(session.exec(select(BERTAnime)).all())
            user_list_count = len(session.exec(select(BERTUserAnimeList)).all())

            print("\n" + "=" * 80)
            print("📊 BERT 資料集摘要")
            print("=" * 80)
            print(f"  動畫總數: {anime_count}")
            print(f"  使用者列表記錄: {user_list_count}")
            print(f"  資料庫位置: {BERT_DB_PATH}")
            print("=" * 80)

    async def close(self):
        """關閉連線"""
        await self.client.close()


# ============================================================================
# 主程式
# ============================================================================


async def main():
    parser = argparse.ArgumentParser(
        description="準備 BERT 訓練資料集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 抓取 3000 部熱門動畫
  python prepare_bert_dataset.py --count 3000

  # 抓取 5000 部動畫，最低人氣 1000
  python prepare_bert_dataset.py --count 5000 --min-popularity 1000

  # 抓取動畫並加入使用者列表
  python prepare_bert_dataset.py --count 3000 --users user1 user2 user3

  # 只抓取使用者列表（假設動畫已存在）
  python prepare_bert_dataset.py --only-users --users user1 user2
        """,
    )

    parser.add_argument(
        "--count", type=int, default=3000, help="要抓取的動畫數量 (預設: 3000)"
    )

    parser.add_argument(
        "--min-popularity",
        type=int,
        default=0,
        help="最低人氣值 (預設: 0)",
    )

    parser.add_argument(
        "--per-page",
        type=int,
        default=50,
        help="每頁抓取數量 (預設: 50)",
    )

    parser.add_argument(
        "--users",
        nargs="+",
        help="要抓取列表的使用者名稱",
    )

    parser.add_argument(
        "--only-users",
        action="store_true",
        help="只抓取使用者列表，不抓取動畫",
    )

    args = parser.parse_args()

    # 初始化資料庫
    engine = init_bert_db()
    preparer = BERTDatasetPreparer(engine)

    try:
        # 抓取動畫資料
        if not args.only_users:
            anime_list = await preparer.fetch_popular_anime(
                target_count=args.count,
                min_popularity=args.min_popularity,
                per_page=args.per_page,
            )

            if anime_list:
                await preparer.store_anime(anime_list)

                # 儲存元數據
                preparer.save_metadata("last_fetch_date", datetime.utcnow().isoformat())
                preparer.save_metadata("anime_count", str(len(anime_list)))

        # 抓取使用者列表
        if args.users:
            print(f"\n{'=' * 80}")
            print(f"📚 開始抓取 {len(args.users)} 個使用者的列表")
            print(f"{'=' * 80}")

            for username in args.users:
                await preparer.fetch_user_list(username)
                # 避免過度請求
                await asyncio.sleep(2)

        # 列印摘要
        preparer.print_summary()

        print("\n✅ 資料準備完成！")
        print("\n📝 下一步:")
        print("  1. 檢查資料: python check_bert_data.py")
        print("  2. 訓練模型: python train_bert_model.py")

    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷執行")
        sys.exit(130)
    except Exception as e:
        logger.error(f"發生錯誤: {e}", exc_info=True)
        sys.exit(1)
    finally:
        await preparer.close()


if __name__ == "__main__":
    asyncio.run(main())
