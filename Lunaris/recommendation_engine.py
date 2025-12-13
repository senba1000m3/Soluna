import logging
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class RecommendationEngine:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def _extract_features(self, anime_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Converts a list of anime entries into a feature DataFrame (One-Hot Encoding for Genres).
        """
        if not anime_list:
            return pd.DataFrame()

        data = []
        for entry in anime_list:
            # Handle both raw anime objects and user list entries
            anime = entry.get("media", entry)

            row = {
                "id": anime["id"],
                "title": anime["title"]["romaji"],
                "score": entry.get("score", 0)
                if "score" in entry
                else 0,  # User score or 0
                "averageScore": anime.get("averageScore", 0),
            }

            # Extract Genres
            genres = anime.get("genres", [])
            for genre in genres:
                row[f"Genre_{genre}"] = 1

            data.append(row)

        df = pd.DataFrame(data)
        return df.fillna(0)

    def build_user_profile(self, user_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Builds a user preference vector based on their watched history.

        Logic:
        1. Filter for Completed or Watching (score is optional).
        2. Calculate weighted score for each genre:
           - If has score: Weight = (UserScore - MeanUserScore) + 1.0
           - If no score: Weight = 1.0 (neutral positive)
        3. Normalize the vector.
        """
        if not user_list:
            return {}

        # Filter valid entries (Completed or Watching, score is optional)
        valid_entries = [
            e for e in user_list if e["status"] in ["COMPLETED", "CURRENT"]
        ]

        if not valid_entries:
            logger.warning("User has no completed or current entries to build profile.")
            return {}

        # Convert to DataFrame
        df = self._extract_features(valid_entries)

        if df.empty:
            return {}

        # Check if user has any scores
        has_scores = (df["score"] > 0).any()

        if has_scores:
            # Calculate User's Mean Score to center the ratings
            scored_df = df[df["score"] > 0]
            user_mean_score = scored_df["score"].mean()
        else:
            # No scores available, use default
            user_mean_score = 0
            logger.info("User has no scores, using watch history only")

        # Calculate Weighted Genre Scores
        # We look at columns starting with "Genre_"
        genre_cols = [c for c in df.columns if c.startswith("Genre_")]

        user_profile = defaultdict(float)

        for _, row in df.iterrows():
            # Calculate weight based on whether score exists
            if row["score"] > 0 and has_scores:
                # Centered Score: How much better/worse than average did they like this?
                weight = (row["score"] - user_mean_score) + 1.0
            else:
                # No score: just count as watched (neutral positive)
                weight = 1.0

            # Add to genre weights
            for col in genre_cols:
                if row[col] == 1:
                    user_profile[col] += weight

        # Normalize profile vector to sum to reasonable scale
        if user_profile:
            total_weight = sum(user_profile.values())
            if total_weight > 0:
                # Normalize to maintain relative weights
                max_weight = max(user_profile.values())
                user_profile = {k: v / max_weight for k, v in user_profile.items()}

        return dict(user_profile)

    def recommend_seasonal(
        self, user_profile: Dict[str, float], seasonal_anime: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Ranks seasonal anime based on similarity to user profile.
        Returns anime with match_score and match_reasons.
        """
        if not user_profile or not seasonal_anime:
            return seasonal_anime

        # 1. Prepare User Vector
        # We need to ensure the User Vector and Anime Vectors have the same dimensions (same Genres)

        # Extract all unique genres from both profile and new anime to build the feature space
        profile_genres = set(user_profile.keys())

        # Convert seasonal anime to DataFrame to get their genres
        anime_df = self._extract_features(seasonal_anime)
        if anime_df.empty:
            return seasonal_anime

        anime_genre_cols = set([c for c in anime_df.columns if c.startswith("Genre_")])

        # Union of all known genres in this context
        all_genres = list(profile_genres.union(anime_genre_cols))
        all_genres.sort()  # Ensure consistent order

        # 2. Create Vectors

        # User Vector
        user_vec = np.array([user_profile.get(g, 0) for g in all_genres]).reshape(1, -1)

        # Anime Matrix
        anime_matrix = []
        for _, row in anime_df.iterrows():
            vec = []
            for g in all_genres:
                # If the anime has this genre column and it's 1
                if g in anime_df.columns and row.get(g) == 1:
                    vec.append(1)
                else:
                    vec.append(0)
            anime_matrix.append(vec)

        anime_matrix = np.array(anime_matrix)

        # 3. Calculate Cosine Similarity
        if anime_matrix.shape[0] == 0:
            return seasonal_anime

        # similarity shape: (1, n_anime)
        similarity_scores = cosine_similarity(user_vec, anime_matrix)[0]

        # 4. Attach scores and reasons to anime objects
        scored_anime = []
        for idx, anime in enumerate(seasonal_anime):
            # Create a copy to avoid mutating original cache if we had one
            a = anime.copy()
            # Convert numpy float to native float for JSON serialization
            a["match_score"] = float(similarity_scores[idx]) * 100

            # Generate match reasons
            a["match_reasons"] = self._generate_match_reasons(
                user_profile, anime, anime_df.iloc[idx], all_genres
            )

            scored_anime.append(a)

        # 5. Sort by match score
        scored_anime.sort(key=lambda x: x["match_score"], reverse=True)

        return scored_anime

    def _generate_match_reasons(
        self,
        user_profile: Dict[str, float],
        anime: Dict[str, Any],
        anime_row: pd.Series,
        all_genres: List[str],
    ) -> Dict[str, Any]:
        """
        Generate explanation for why an anime was recommended.
        """
        reasons = {"matched_genres": [], "total_weight": 0.0, "top_reason": ""}

        # Find matching genres and their weights
        genre_matches = []
        for genre in all_genres:
            if genre in anime_row.index and anime_row[genre] == 1:
                weight = user_profile.get(genre, 0)
                if weight > 0:
                    genre_name = genre.replace("Genre_", "")
                    genre_matches.append({"genre": genre_name, "weight": float(weight)})

        # Sort by weight
        genre_matches.sort(key=lambda x: x["weight"], reverse=True)
        reasons["matched_genres"] = genre_matches[:5]  # Top 5
        reasons["total_weight"] = sum(g["weight"] for g in genre_matches)

        # Generate top reason
        if genre_matches:
            top_genres = [g["genre"] for g in genre_matches[:3]]
            if len(top_genres) == 1:
                reasons["top_reason"] = f"你喜歡 {top_genres[0]} 類型"
            elif len(top_genres) == 2:
                reasons["top_reason"] = (
                    f"你喜歡 {top_genres[0]} 和 {top_genres[1]} 類型"
                )
            else:
                reasons["top_reason"] = f"你喜歡 {', '.join(top_genres[:2])} 等類型"
        else:
            reasons["top_reason"] = "基於整體偏好匹配"

        return reasons

    def compare_users(
        self, user1_profile: Dict[str, float], user2_profile: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculates compatibility score between two users based on their genre preferences.
        """
        if not user1_profile or not user2_profile:
            return {
                "score": 0.0,
                "common_genres": [],
                "message": "Insufficient data for comparison",
            }

        # 1. Align Genres
        genres1 = set(user1_profile.keys())
        genres2 = set(user2_profile.keys())
        all_genres = list(genres1.union(genres2))
        all_genres.sort()

        # 2. Create Vectors
        vec1 = np.array([user1_profile.get(g, 0) for g in all_genres]).reshape(1, -1)
        vec2 = np.array([user2_profile.get(g, 0) for g in all_genres]).reshape(1, -1)

        # 3. Calculate Cosine Similarity
        similarity = cosine_similarity(vec1, vec2)[0][0]
        score = float(similarity) * 100

        # 4. Find Common Interests (Top genres for both)
        common_genres = []
        for g in all_genres:
            w1 = user1_profile.get(g, 0)
            w2 = user2_profile.get(g, 0)
            # If both have positive weight (liked it)
            if w1 > 0 and w2 > 0:
                # Geometric mean as a combined score for sorting
                combined_weight = (w1 * w2) ** 0.5
                common_genres.append(
                    {"genre": g.replace("Genre_", ""), "score": combined_weight}
                )

        common_genres.sort(key=lambda x: x["score"], reverse=True)

        return {
            "score": max(0.0, score),  # Ensure non-negative
            "common_genres": common_genres[:5],
        }

    def select_representative_anime(
        self,
        year: int,
        user_list: List[Dict[str, Any]],
        popular_anime: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Selects a representative anime for a given year.
        Priority: User's favorite (Score > Status) > Most Popular.
        """
        representative = None
        reason = ""
        is_watched = False
        user_score = 0

        # 1. Try to find in user list (only COMPLETED and CURRENT)
        if user_list:
            # Filter for anime from this year (only COMPLETED and CURRENT)
            candidates = []
            for entry in user_list:
                if entry.get("status") not in ["COMPLETED", "CURRENT"]:
                    continue
                media = entry.get("media", {})
                if media.get("seasonYear") == year:
                    candidates.append(entry)

            if candidates:
                # Sort by Score (desc) -> Status (Completed first)
                # Status priority: COMPLETED (2), CURRENT (1), others (0)
                def status_priority(s):
                    if s == "COMPLETED":
                        return 2
                    if s == "CURRENT":
                        return 1
                    return 0

                candidates.sort(
                    key=lambda x: (x.get("score", 0), status_priority(x.get("status"))),
                    reverse=True,
                )

                best_entry = candidates[0]
                representative = best_entry["media"].copy()
                is_watched = True
                user_score = best_entry.get("score", 0)
                reason = "你的年度最佳"

        # 2. If not found in user list, use most popular
        if not representative and popular_anime:
            representative = popular_anime[0].copy()
            reason = "年度霸權"

        if representative:
            representative["is_watched"] = is_watched
            representative["user_score"] = user_score
            representative["selection_reason"] = reason
            return representative

        return None

    def get_milestone_content(
        self,
        year: int,
        user_list: List[Dict[str, Any]],
        popular_anime: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Returns top 5 anime for a milestone year.
        Prioritizes user watched anime, then fills with popular anime.
        """
        result_anime = []
        added_ids = set()

        # 1. Add User Watched Anime (only COMPLETED and CURRENT)
        if user_list:
            user_candidates = []
            for entry in user_list:
                if entry.get("status") not in ["COMPLETED", "CURRENT"]:
                    continue
                media = entry.get("media", {})
                if media.get("seasonYear") == year:
                    # Create a standardized anime object
                    anime = media.copy()
                    anime["is_watched"] = True
                    anime["user_score"] = entry.get("score", 0)
                    anime["selection_reason"] = (
                        "你的年度最佳" if entry.get("score", 0) >= 80 else "已觀看"
                    )

                    user_candidates.append(anime)

            # Sort by score descending
            user_candidates.sort(key=lambda x: x["user_score"], reverse=True)

            # Add to results (up to 5)
            for anime in user_candidates[:5]:
                result_anime.append(anime)
                added_ids.add(anime["id"])

        # 2. Fill with Popular Anime if needed
        for anime in popular_anime:
            if len(result_anime) >= 5:
                break

            if anime["id"] not in added_ids:
                a = anime.copy()
                a["is_watched"] = False
                a["user_score"] = 0
                a["selection_reason"] = "年度熱門"
                result_anime.append(a)
                added_ids.add(a["id"])

        return result_anime

    def get_timeline_milestones(self, birth_year: int) -> List[Dict[str, Any]]:
        """
        Generates key milestones based on birth year.
        """
        current_year = datetime.now().year

        # Define key ages for anime viewing
        milestones = [
            {"age": 0, "label": "誕生之年 (0歲)"},
            {"age": 10, "label": "童年啟蒙 (10歲)"},
            {"age": 14, "label": "中學時期 (14歲)"},
            {"age": 17, "label": "高中時期 (17歲)"},
            {"age": 21, "label": "大學/社會 (21歲)"},
        ]

        timeline = []
        for m in milestones:
            target_year = birth_year + m["age"]
            # We include it if it's not too far in the future
            if target_year <= current_year + 2:
                timeline.append(
                    {"age": m["age"], "year": target_year, "label": m["label"]}
                )

        return timeline

    def calculate_timeline_stats(
        self, user_list: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates interesting statistics from user's list.
        Only counts COMPLETED anime (excludes DROPPED).
        """
        if not user_list:
            return {}

        stats = {}

        # Filter to only include COMPLETED and CURRENT anime (exclude DROPPED and PLANNING)
        completed_list = [
            entry
            for entry in user_list
            if entry.get("status") in ["COMPLETED", "CURRENT"]
        ]

        # 1. Year with most watched anime
        year_counts = Counter()
        for entry in completed_list:
            media = entry.get("media", {})
            year = media.get("seasonYear")
            if year:
                year_counts[year] += 1

        if year_counts:
            most_watched_year, count = year_counts.most_common(1)[0]
            stats["most_active_year"] = {
                "year": most_watched_year,
                "count": count,
                "label": "動畫成癮年",
            }

        # 2. Favorite Genre
        genre_counts = Counter()
        for entry in completed_list:
            media = entry.get("media", {})
            genres = media.get("genres", [])
            for genre in genres:
                genre_counts[genre] += 1

        if genre_counts:
            fav_genre, count = genre_counts.most_common(1)[0]
            stats["favorite_genre"] = {
                "genre": fav_genre,
                "count": count,
                "label": "本命類型",
            }

        # 3. Total Watch Time (approximate, only completed anime)
        total_minutes = 0
        for entry in completed_list:
            media = entry.get("media", {})
            episodes = media.get("episodes") or 0
            duration = media.get("duration") or 24  # Default to 24 min
            # Use progress if available, otherwise use total episodes if completed
            progress = entry.get("progress") or (
                episodes if entry.get("status") == "COMPLETED" else 0
            )
            total_minutes += progress * duration

        days = total_minutes / (60 * 24)
        stats["total_watch_time"] = {
            "days": round(days, 1),
            "hours": round(total_minutes / 60, 1),
            "label": "人生獻祭時間",
        }

        return stats
