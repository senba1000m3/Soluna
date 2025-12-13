from typing import List, Optional

from sqlmodel import Field, Relationship, SQLModel


class User(SQLModel, table=True):
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
