import logging
from collections import Counter, defaultdict
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
        1. Filter for Completed or Watching.
        2. Calculate weighted score for each genre:
           Weight = (UserScore - MeanUserScore) * GenrePresence
        3. Normalize the vector.
        """
        if not user_list:
            return {}

        # Filter valid entries (Completed or Watching, and has a score)
        valid_entries = [
            e
            for e in user_list
            if e["status"] in ["COMPLETED", "CURRENT"] and e["score"] > 0
        ]

        if not valid_entries:
            logger.warning("User has no scored entries to build profile.")
            return {}

        # Convert to DataFrame
        df = self._extract_features(valid_entries)

        if df.empty:
            return {}

        # Calculate User's Mean Score to center the ratings
        # (e.g. if user rates everything 90, a 70 is actually 'bad')
        user_mean_score = df["score"].mean()

        # Calculate Weighted Genre Scores
        # We look at columns starting with "Genre_"
        genre_cols = [c for c in df.columns if c.startswith("Genre_")]

        user_profile = defaultdict(float)

        for _, row in df.iterrows():
            # Centered Score: How much better/worse than average did they like this?
            # We add a small epsilon or base weight so even average shows contribute slightly to genre preference
            weight = (row["score"] - user_mean_score) + 0.1

            # If weight is negative (they disliked it), it reduces the genre score

            for col in genre_cols:
                if row[col] == 1:
                    user_profile[col] += weight

        # Normalize profile vector
        # Convert to list for normalization
        profile_vector = np.array(list(user_profile.values())).reshape(1, -1)
        if profile_vector.size > 0:
            # We don't strictly need 0-1 normalization for Cosine, but it helps for debugging
            pass

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
