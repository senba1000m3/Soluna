import logging
import pickle
from collections import Counter
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sqlmodel import Session, select

from models import Anime, UserRating

logger = logging.getLogger(__name__)


class DropAnalysisEngine:
    def __init__(self):
        self.model = None
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_tags = MultiLabelBinarizer()
        self.le_studio = LabelEncoder()
        self.is_trained = False
        self.feature_columns = None  # Store feature columns for prediction

    def _prepare_features(
        self, ratings: List[UserRating], animes: List[Anime]
    ) -> pd.DataFrame:
        """
        Converts DB records into a feature DataFrame for ML.
        Features:
        - Genres (One-Hot)
        - Tags (One-Hot, top N)
        - Episodes (Numerical)
        - Season (Categorical -> One-Hot)
        - Studio (Categorical -> Label Encoded)
        - Average Score (Numerical)
        - Popularity (Numerical)
        """
        data = []
        anime_map = {a.id: a for a in animes}

        for r in ratings:
            anime = anime_map.get(r.anime_id)
            if not anime:
                continue

            # Target: 1 if DROPPED, 0 if COMPLETED
            # We ignore watching/paused for training binary classifier
            if r.status == "DROPPED":
                target = 1
            elif r.status == "COMPLETED":
                target = 0
            else:
                continue

            row = {
                "target": target,
                "episodes": anime.episodes or 0,
                "average_score": anime.average_score or 0,
                "popularity": anime.popularity or 0,
                "season": anime.season or "UNKNOWN",
                "genres": (anime.genres or "").split(","),
                "tags": (anime.tags or "").split(","),
                "studios": (anime.studios or "").split(",")[0]
                if anime.studios
                else "Unknown",
            }
            data.append(row)

        df = pd.DataFrame(data)
        if df.empty:
            return df

        # --- Feature Engineering ---

        # 1. Genres (One-Hot)
        genres_encoded = self.mlb_genres.fit_transform(df["genres"])
        genre_df = pd.DataFrame(
            genres_encoded, columns=[f"Genre_{c}" for c in self.mlb_genres.classes_]
        )

        # 2. Tags (One-Hot) - Limit to top 20 most common tags to avoid explosion
        # For simplicity in this prototype, we just take all, but in prod limit it.
        tags_encoded = self.mlb_tags.fit_transform(df["tags"])
        # Optimization: Select only top 20 tags by frequency
        tag_counts = df["tags"].explode().value_counts()
        top_tags = tag_counts.head(20).index.tolist()

        # Re-fit only on top tags for cleaner features
        df["tags_filtered"] = df["tags"].apply(
            lambda x: [t for t in x if t in top_tags]
        )
        tags_encoded = self.mlb_tags.fit_transform(df["tags_filtered"])

        tag_df = pd.DataFrame(
            tags_encoded, columns=[f"Tag_{c}" for c in self.mlb_tags.classes_]
        )

        # 3. Studio (Label Encoding)
        df["studio_code"] = self.le_studio.fit_transform(df["studios"])

        # 4. Season (One-Hot)
        season_df = pd.get_dummies(df["season"], prefix="Season")

        # Combine all
        features = pd.concat(
            [
                df[["episodes", "average_score", "popularity", "studio_code"]],
                genre_df,
                tag_df,
                season_df,
            ],
            axis=1,
        )

        # Add target back for splitting
        features["target"] = df["target"]

        return features

    def train_model(self, session: Session) -> Dict[str, Any]:
        """
        Fetches data from DB, trains XGBoost model, and returns metrics.
        """
        logger.info("Fetching training data from DB...")
        ratings = session.exec(select(UserRating)).all()
        animes = session.exec(select(Anime)).all()

        if not ratings or not animes:
            return {"error": "No data available for training"}

        logger.info(f"Found {len(ratings)} ratings and {len(animes)} anime.")

        df = self._prepare_features(ratings, animes)
        if df.empty:
            return {"error": "No valid Completed/Dropped entries found"}

        X = df.drop("target", axis=1)
        y = df["target"]

        # Check class distribution
        dropped_count = int(y.sum())
        completed_count = int(len(y) - dropped_count)

        if dropped_count < 2 or completed_count < 2:
            return {
                "error": f"Insufficient data balance. Dropped: {dropped_count}, Completed: {completed_count}. Need at least 2 of each.",
                "sample_size": len(df),
                "dropped_count": dropped_count,
                "completed_count": completed_count,
                "top_features": [],
                "accuracy": 0.0,
            }

        # Split Data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Train XGBoost
        logger.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X_train, y_train)

        # Evaluate
        accuracy = self.model.score(X_test, y_test)
        logger.info(f"Model trained. Accuracy: {accuracy:.2f}")

        self.is_trained = True

        # Feature Importance
        importance = dict(zip(X.columns, self.model.feature_importances_))
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]

        # Convert numpy types to native Python types for JSON serialization
        top_features_serializable = [
            (str(name), float(val)) for name, val in top_features
        ]

        return {
            "accuracy": float(accuracy),
            "sample_size": int(len(df)),
            "dropped_count": int(y.sum()),
            "completed_count": int(len(y) - y.sum()),
            "top_features": top_features_serializable,
        }

        # Store feature columns for later prediction
        self.feature_columns = X.columns.tolist()

    def predict_drop_probability(self, anime: Anime) -> float:
        """
        Predicts drop probability for a single anime.
        Returns probability between 0.0 and 1.0.
        """
        if not self.is_trained or not self.model or not self.feature_columns:
            logger.warning("Model not trained yet.")
            return 0.0

        try:
            # Prepare features similar to training
            genres = (anime.genres or "").split(",")
            tags = (anime.tags or "").split(",")
            studios = (
                (anime.studios or "").split(",")[0] if anime.studios else "Unknown"
            )

            # Create a single row dataframe
            row_data = {
                "episodes": anime.episodes or 0,
                "average_score": anime.average_score or 0,
                "popularity": anime.popularity or 0,
                "season": anime.season or "UNKNOWN",
            }

            # Encode genres
            genres_encoded = self.mlb_genres.transform([genres])
            for i, genre in enumerate(self.mlb_genres.classes_):
                row_data[f"Genre_{genre}"] = genres_encoded[0][i]

            # Encode tags
            tags_encoded = self.mlb_tags.transform([tags])
            for i, tag in enumerate(self.mlb_tags.classes_):
                row_data[f"Tag_{tag}"] = tags_encoded[0][i]

            # Encode studio
            try:
                studio_code = self.le_studio.transform([studios])[0]
            except ValueError:
                # Unknown studio, use -1 or mean
                studio_code = -1
            row_data["studio_code"] = studio_code

            # Season one-hot
            for season in ["WINTER", "SPRING", "SUMMER", "FALL", "UNKNOWN"]:
                row_data[f"Season_{season}"] = 1 if anime.season == season else 0

            # Create dataframe with only the features that exist in training
            df = pd.DataFrame([row_data])

            # Add missing columns with 0
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

            # Ensure same column order as training
            df = df[self.feature_columns]

            # Predict probability of dropping (class 1)
            proba = self.model.predict_proba(df)[0][1]
            return float(proba)

        except Exception as e:
            logger.error(f"Error predicting drop probability: {e}")
            return 0.0

    def analyze_drop_patterns(
        self, ratings: List[UserRating], animes: List[Anime]
    ) -> Dict[str, Any]:
        """
        Analyzes patterns in dropped anime to find common factors.
        Returns statistics by tags, studios, and genres.
        """
        anime_map = {a.id: a for a in animes}

        dropped_anime = []
        completed_anime = []

        for r in ratings:
            anime = anime_map.get(r.anime_id)
            if not anime:
                continue

            if r.status == "DROPPED":
                dropped_anime.append(anime)
            elif r.status == "COMPLETED":
                completed_anime.append(anime)

        # Analyze tags
        dropped_tags = []
        completed_tags = []
        for anime in dropped_anime:
            if anime.tags:
                dropped_tags.extend(anime.tags.split(","))
        for anime in completed_anime:
            if anime.tags:
                completed_tags.extend(anime.tags.split(","))

        # Analyze genres
        dropped_genres = []
        completed_genres = []
        for anime in dropped_anime:
            if anime.genres:
                dropped_genres.extend(anime.genres.split(","))
        for anime in completed_anime:
            if anime.genres:
                completed_genres.extend(anime.genres.split(","))

        # Analyze studios
        dropped_studios = []
        completed_studios = []
        for anime in dropped_anime:
            if anime.studios:
                dropped_studios.extend(anime.studios.split(","))
        for anime in completed_anime:
            if anime.studios:
                completed_studios.extend(anime.studios.split(","))

        # Calculate drop rates
        def calculate_drop_rate(item, dropped_list, completed_list):
            dropped_count = dropped_list.count(item)
            completed_count = completed_list.count(item)
            total = dropped_count + completed_count
            if total == 0:
                return 0.0
            return dropped_count / total

        # Top dropped tags
        all_tags = set(dropped_tags + completed_tags)
        tag_stats = []
        for tag in all_tags:
            if not tag.strip():
                continue
            dropped_count = dropped_tags.count(tag)
            completed_count = completed_tags.count(tag)
            total = dropped_count + completed_count
            if total >= 2:  # At least 2 occurrences
                drop_rate = dropped_count / total
                tag_stats.append(
                    {
                        "name": tag,
                        "dropped": dropped_count,
                        "completed": completed_count,
                        "total": total,
                        "drop_rate": drop_rate,
                    }
                )
        tag_stats.sort(key=lambda x: (x["drop_rate"], x["total"]), reverse=True)

        # Top dropped genres
        all_genres = set(dropped_genres + completed_genres)
        genre_stats = []
        for genre in all_genres:
            if not genre.strip():
                continue
            dropped_count = dropped_genres.count(genre)
            completed_count = completed_genres.count(genre)
            total = dropped_count + completed_count
            if total >= 2:
                drop_rate = dropped_count / total
                genre_stats.append(
                    {
                        "name": genre,
                        "dropped": dropped_count,
                        "completed": completed_count,
                        "total": total,
                        "drop_rate": drop_rate,
                    }
                )
        genre_stats.sort(key=lambda x: (x["drop_rate"], x["total"]), reverse=True)

        # Top dropped studios
        all_studios = set(dropped_studios + completed_studios)
        studio_stats = []
        for studio in all_studios:
            if not studio.strip():
                continue
            dropped_count = dropped_studios.count(studio)
            completed_count = completed_studios.count(studio)
            total = dropped_count + completed_count
            if total >= 2:
                drop_rate = dropped_count / total
                studio_stats.append(
                    {
                        "name": studio,
                        "dropped": dropped_count,
                        "completed": completed_count,
                        "total": total,
                        "drop_rate": drop_rate,
                    }
                )
        studio_stats.sort(key=lambda x: (x["drop_rate"], x["total"]), reverse=True)

        return {
            "top_dropped_tags": tag_stats[:10],
            "top_dropped_genres": genre_stats[:10],
            "top_dropped_studios": studio_stats[:10],
        }
