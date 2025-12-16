from datetime import datetime
from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class GlobalUser(SQLModel, table=True):
    """全局使用者表 - 儲存主 ID 及其常用 ID 列表"""

    __tablename__ = "global_user"

    id: Optional[int] = Field(default=None, primary_key=True)
    anilist_username: str = Field(
        index=True, unique=True, description="主 ID 的 AniList 使用者名稱"
    )
    anilist_id: int = Field(description="主 ID 的 AniList 使用者 ID")
    avatar: str = Field(description="主 ID 的頭像 URL")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime = Field(default_factory=datetime.utcnow)

    # 關聯 - 此使用者的常用 ID 列表
    quick_ids: List["QuickID"] = Relationship(back_populates="owner")


class QuickID(SQLModel, table=True):
    """常用 ID 列表 - 不包含主 ID，只儲存額外的常用 ID"""

    __tablename__ = "quick_id"

    id: Optional[int] = Field(default=None, primary_key=True)
    owner_id: int = Field(
        foreign_key="global_user.id", index=True, description="所屬的主 ID"
    )
    anilist_username: str = Field(description="常用 ID 的 AniList 使用者名稱")
    anilist_id: int = Field(description="常用 ID 的 AniList 使用者 ID")
    avatar: str = Field(description="常用 ID 的頭像 URL")
    nickname: Optional[str] = Field(default=None, description="自訂暱稱")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # 關聯
    owner: GlobalUser = Relationship(back_populates="quick_ids")


class User(SQLModel, table=True):
    """舊版使用者表 - 保留用於其他功能"""

    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    anilist_id: Optional[int] = Field(default=None, index=True)

    ratings: List["UserRating"] = Relationship(back_populates="user")


class Anime(SQLModel, table=True):
    id: int = Field(primary_key=True)  # AniList ID
    title_romaji: str
    title_english: Optional[str] = None
    genres: str  # Comma-separated string
    average_score: Optional[int] = None
    popularity: Optional[int] = None
    episodes: Optional[int] = None
    season: Optional[str] = None
    season_year: Optional[int] = None
    studios: Optional[str] = None  # Comma-separated string
    tags: Optional[str] = None  # Comma-separated string

    ratings: List["UserRating"] = Relationship(back_populates="anime")


class UserRating(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    anime_id: int = Field(foreign_key="anime.id")
    score: int
    status: str  # COMPLETED, DROPPED, etc.
    progress: int

    user: User = Relationship(back_populates="ratings")
    anime: Anime = Relationship(back_populates="ratings")
    anilist_id: Optional[int] = Field(
        default=None, description="The user's ID on AniList"
    )


class AnimeVoiceActorCache(SQLModel, table=True):
    """快取動漫聲優資料，避免重複查詢 AniList API"""

    id: Optional[int] = Field(default=None, primary_key=True)
    anime_id: int = Field(index=True, unique=True)  # AniList 動漫 ID
    voice_actors_data: str  # JSON 字串格式儲存聲優資料
    cached_at: datetime = Field(default_factory=datetime.utcnow)  # 快取時間

    # 可選：如果需要定期更新快取，可以加上過期時間檢查
    # 例如：超過 30 天的快取可以重新抓取


class BERTUserProfile(SQLModel, table=True):
    """快取 BERT 使用者 Profile，避免每次都重新訓練"""

    __tablename__ = "bert_user_profile"

    id: Optional[int] = Field(default=None, primary_key=True)
    anilist_username: str = Field(index=True, unique=True)  # AniList 使用者名稱
    anilist_id: int = Field(index=True)  # AniList 使用者 ID

    # 使用者觀看的動畫 ID 列表（JSON 陣列字串）
    user_anime_ids: str = Field(description="JSON array of anime IDs user has watched")

    # BERT 提取的特徵 Profile（JSON 物件字串）
    bert_features: str = Field(
        description="JSON object of BERT-extracted features (genres, tags, etc.)"
    )

    # Profile 的 hash，用於檢測使用者列表是否有變化
    profile_hash: str = Field(
        index=True, description="Hash of user anime list for change detection"
    )

    # 時間戳記
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # 元數據
    anime_count: int = Field(description="Number of anime in user's list")
    model_version: str = Field(default="v1", description="BERT model version used")


class BERTRecommendationCache(SQLModel, table=True):
    """快取 BERT 推薦結果，避免重複推理"""

    __tablename__ = "bert_recommendation_cache"

    id: Optional[int] = Field(default=None, primary_key=True)
    anilist_username: str = Field(index=True)  # AniList 使用者名稱
    profile_hash: str = Field(
        index=True, description="User profile hash for cache invalidation"
    )

    # BERT 推薦的動畫列表（JSON 陣列，包含 anime_id 和 score）
    recommendations: str = Field(
        description="JSON array of recommended anime with scores"
    )

    # 推薦參數
    top_k: int = Field(default=50, description="Number of recommendations stored")

    # 時間戳記
    created_at: datetime = Field(default_factory=datetime.utcnow)
    cached_at: datetime = Field(default_factory=datetime.utcnow)

    # 元數據
    model_version: str = Field(default="v1", description="BERT model version used")
    cache_hit_count: int = Field(
        default=0, description="Number of times this cache was used"
    )

    # 建立複合索引以加速查詢
    class Config:
        indexes = [{"fields": ["anilist_username", "profile_hash"], "unique": True}]
