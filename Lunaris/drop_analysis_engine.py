import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sqlmodel import Session
from tqdm import tqdm
from xgboost import XGBClassifier

from models import Anime, UserRating

logger = logging.getLogger(__name__)

# Progress tracker type hint
try:
    from progress_tracker import ProgressTracker
except ImportError:
    ProgressTracker = None


class DropAnalysisEngine:
    def __init__(self, progress_tracker: Optional[Any] = None):
        self.model: XGBClassifier | None = None
        self.feature_columns: list[str] = []
        self.mlb_genres = MultiLabelBinarizer()
        self.mlb_tags = MultiLabelBinarizer()
        self.le_studio = LabelEncoder()
        self.is_trained = False

        # Store user tolerance metrics for inference
        self.user_tolerance_cache = {}

        # Progress tracker for real-time updates
        self.progress_tracker = progress_tracker

    def _calculate_user_tolerance_metrics(
        self, user_id: int, session: Session
    ) -> dict[str, float]:
        """
        Calculate user's historical tolerance metrics.
        Returns metrics like genre drop rates, studio drop rates, etc.
        """
        from sqlmodel import select

        # Get all user ratings
        user_ratings = session.exec(
            select(UserRating).where(UserRating.user_id == user_id)
        ).all()

        if not user_ratings:
            return {}

        metrics = {
            "total_anime": len(user_ratings),
            "total_dropped": sum(1 for r in user_ratings if r.status == "DROPPED"),
            "total_completed": sum(1 for r in user_ratings if r.status == "COMPLETED"),
            "overall_drop_rate": 0.0,
            "avg_completion_ratio": 0.0,
            "genre_drop_rates": {},
            "studio_drop_rates": {},
            "tag_drop_rates": {},
        }

        if metrics["total_anime"] > 0:
            metrics["overall_drop_rate"] = (
                metrics["total_dropped"] / metrics["total_anime"]
            )

        # Calculate completion ratios
        completion_ratios = []
        for rating in user_ratings:
            anime = session.get(Anime, rating.anime_id)
            if anime and anime.episodes and anime.episodes > 0:
                ratio = (rating.progress or 0) / anime.episodes
                completion_ratios.append(min(ratio, 1.0))

        if completion_ratios:
            metrics["avg_completion_ratio"] = np.mean(completion_ratios)

        # Calculate genre-specific drop rates
        genre_counts = {}
        genre_drops = {}
        for rating in user_ratings:
            anime = session.get(Anime, rating.anime_id)
            if anime and anime.genres:
                genres = [g.strip() for g in anime.genres.split(",") if g.strip()]
                for genre in genres:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    if rating.status == "DROPPED":
                        genre_drops[genre] = genre_drops.get(genre, 0) + 1

        for genre, count in genre_counts.items():
            if count >= 2:  # Only consider genres with at least 2 samples
                drops = genre_drops.get(genre, 0)
                metrics["genre_drop_rates"][genre] = drops / count

        # Calculate studio-specific drop rates
        studio_counts = {}
        studio_drops = {}
        for rating in user_ratings:
            anime = session.get(Anime, rating.anime_id)
            if anime and anime.studios:
                studio = anime.studios.split(",")[0].strip()
                if studio:
                    studio_counts[studio] = studio_counts.get(studio, 0) + 1
                    if rating.status == "DROPPED":
                        studio_drops[studio] = studio_drops.get(studio, 0) + 1

        for studio, count in studio_counts.items():
            if count >= 2:
                drops = studio_drops.get(studio, 0)
                metrics["studio_drop_rates"][studio] = drops / count

        # Calculate tag-specific drop rates
        tag_counts = {}
        tag_drops = {}
        for rating in user_ratings:
            anime = session.get(Anime, rating.anime_id)
            if anime and anime.tags:
                tags = [t.strip() for t in anime.tags.split(",") if t.strip()]
                for tag in tags[:10]:  # Consider top 10 tags
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
                    if rating.status == "DROPPED":
                        tag_drops[tag] = tag_drops.get(tag, 0) + 1

        for tag, count in tag_counts.items():
            if count >= 2:
                drops = tag_drops.get(tag, 0)
                metrics["tag_drop_rates"][tag] = drops / count

        return metrics

    def _prepare_features(self, session: Session, user_id: int = None) -> pd.DataFrame:
        """
        Prepare feature matrix from anime metadata.
        For personalized training (user_id provided), uses only that user's ratings.
        """
        from sqlmodel import select

        # Get all ratings (optionally filtered by user)
        if self.progress_tracker:
            self.progress_tracker.update(
                progress=42, stage="prepare_features", message="正在準備特徵數據..."
            )
        logger.info("📊 正在準備特徵數據...")
        if user_id:
            ratings = session.exec(
                select(UserRating).where(UserRating.user_id == user_id)
            ).all()
        else:
            ratings = session.exec(select(UserRating)).all()

        if not ratings:
            logger.warning("No ratings found for feature preparation")
            return pd.DataFrame()

        rows = []
        logger.info(f"正在處理 {len(ratings)} 筆評分記錄...")
        total_ratings = len(ratings)
        processed_count = 0

        for idx, rating in enumerate(ratings):
            # Skip non-terminal states for training
            if rating.status not in ["DROPPED", "COMPLETED"]:
                continue

            anime = session.get(Anime, rating.anime_id)
            if not anime:
                continue

            # Parse metadata
            genres = [g.strip() for g in (anime.genres or "").split(",") if g.strip()]
            tags = [t.strip() for t in (anime.tags or "").split(",") if t.strip()]
            studio = anime.studios.split(",")[0].strip() if anime.studios else "Unknown"

            # Use only anime metadata features (no user tolerance to avoid circular dependency)
            row = {
                "anime_id": anime.id,
                "user_id": rating.user_id,
                "episodes": float(anime.episodes or 0),
                "average_score": float(anime.average_score or 0),
                "popularity": float(anime.popularity or 0),
                "season": anime.season or "UNKNOWN",
                "genres": genres,
                "tags": tags,
                "studio": studio,
                "label": 1 if rating.status == "DROPPED" else 0,
            }

            rows.append(row)
            processed_count += 1

            # Update progress more frequently (every 5 items or every 10%)
            if self.progress_tracker and (
                processed_count % 5 == 0 or idx % max(1, total_ratings // 10) == 0
            ):
                progress = 42 + int((idx / max(total_ratings, 1)) * 8)
                self.progress_tracker.update(
                    progress=min(progress, 49),
                    message=f"特徵提取: {processed_count}/{total_ratings}",
                )

        df = pd.DataFrame(rows)

        if df.empty:
            return df

        # Encode categorical features
        if self.progress_tracker:
            self.progress_tracker.update(progress=50, message="正在編碼類別特徵...")
        logger.info("🔧 正在編碼類別特徵...")

        # Genres (multi-label)
        print("  ├─ 處理類型 (Genres)...")
        genre_lists = df["genres"].tolist()
        genres_encoded = self.mlb_genres.fit_transform(genre_lists)
        for i, genre in enumerate(self.mlb_genres.classes_):
            df[f"Genre_{genre}"] = genres_encoded[:, i]

        # Tags (multi-label, limit to top 30 most common)
        if self.progress_tracker:
            self.progress_tracker.update(progress=52, message="處理標籤 (Tags)...")
        print("  ├─ 處理標籤 (Tags)...")
        tag_lists = df["tags"].tolist()
        tags_encoded = self.mlb_tags.fit_transform(tag_lists)
        # Only use tags that exist (up to 30)
        num_tags = min(len(self.mlb_tags.classes_), 30)
        for i in range(num_tags):
            tag = self.mlb_tags.classes_[i]
            df[f"Tag_{tag}"] = tags_encoded[:, i]

        # Studio (label encoding)
        if self.progress_tracker:
            self.progress_tracker.update(
                progress=54, message="處理製作公司 (Studios)..."
            )
        print("  ├─ 處理製作公司 (Studios)...")
        df["studio_code"] = self.le_studio.fit_transform(df["studio"])

        # Season (one-hot)
        if self.progress_tracker:
            self.progress_tracker.update(progress=56, message="處理季節 (Seasons)...")
        print("  └─ 處理季節 (Seasons)...")
        df = pd.get_dummies(df, columns=["season"], prefix="Season")

        # Drop original text columns
        df = df.drop(columns=["genres", "tags", "studio", "anime_id"])

        if self.progress_tracker:
            self.progress_tracker.update(
                progress=58,
                message=f"特徵準備完成！共 {len(df)} 筆樣本，{len(df.columns) - 2} 個特徵",
            )
        logger.info(
            f"✅ 特徵準備完成！共 {len(df)} 筆樣本，{len(df.columns) - 2} 個特徵"
        )

        return df

    def train_model(self, session: Session, user_id: int = None) -> dict[str, Any]:
        """
        Train XGBoost model on user rating data with tolerance-aware features.
        If user_id is provided, trains only on that user's data (personalized model).
        Otherwise trains on all users (global model).
        """
        print("\n" + "=" * 60)
        print("🚀 開始模型訓練...")
        print("=" * 60)
        if self.progress_tracker:
            self.progress_tracker.update(
                progress=35,
                stage="training",
                status="running",
                message="開始模型訓練...",
            )
        logger.info("Starting model training with user tolerance features...")

        try:
            # Prepare features (optionally filtered by user)
            print("\n📋 階段 1/4: 準備訓練數據")
            if self.progress_tracker:
                self.progress_tracker.update(
                    progress=40, stage="stage_1", message="準備訓練數據"
                )
            df = self._prepare_features(session, user_id=user_id)

            if df.empty or len(df) < 10:
                logger.warning("Insufficient data for training")
                print("❌ 數據不足，無法訓練模型")
                return {
                    "status": "insufficient_data",
                    "samples": len(df),
                    "message": "Need at least 10 samples to train",
                }

            # Separate features and labels
            print("\n📋 階段 2/4: 分離特徵與標籤")
            if self.progress_tracker:
                self.progress_tracker.update(
                    progress=60, stage="stage_2", message="分離特徵與標籤"
                )
            X = df.drop(columns=["label", "user_id"])
            y = df["label"]

            # Store feature columns for inference
            self.feature_columns = X.columns.tolist()

            # Handle class imbalance
            n_dropped = sum(y == 1)
            n_completed = sum(y == 0)
            scale_pos_weight = n_completed / max(n_dropped, 1)

            print(f"  ✓ 訓練樣本數: {len(X)}")
            print(f"  ✓ 棄番數量: {n_dropped}")
            print(f"  ✓ 完成數量: {n_completed}")
            print(f"  ✓ 類別權重: {scale_pos_weight:.2f}")

            logger.info(f"Training with {len(X)} samples")
            logger.info(f"  Dropped: {n_dropped}, Completed: {n_completed}")
            logger.info(f"  Scale pos weight: {scale_pos_weight:.2f}")

            # Train XGBoost optimized for personalized small datasets
            print("\n📋 階段 3/4: 訓練 XGBoost 模型")
            if self.progress_tracker:
                self.progress_tracker.update(
                    progress=65, stage="stage_3", message="訓練 XGBoost 模型"
                )
            print("  模型參數:")
            print("    ├─ 決策樹數量: 200")
            print("    ├─ 最大深度: 3")
            print("    ├─ 學習率: 0.05")
            print("    └─ 訓練中...")

            self.model = XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                eval_metric="logloss",
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=1,
                gamma=0.1,
            )

            # Fit with progress tracking
            with tqdm(total=200, desc="訓練進度", unit="樹") as pbar:
                # XGBoost doesn't support direct progress callback, so we simulate
                if self.progress_tracker:
                    self.progress_tracker.update(progress=70, message="訓練模型中...")
                self.model.fit(X, y)
                pbar.update(200)
                if self.progress_tracker:
                    self.progress_tracker.update(progress=82, message="模型訓練完成")

            print("  ✓ 模型訓練完成！")

            # Calculate metrics
            print("\n📋 階段 4/4: 評估模型性能")
            if self.progress_tracker:
                self.progress_tracker.update(
                    progress=85, stage="stage_4", message="評估模型性能"
                )
            y_pred = self.model.predict(X)
            accuracy = (y_pred == y).mean()

            # Get feature importances
            feature_importance = dict(
                zip(self.feature_columns, self.model.feature_importances_)
            )
            top_features = sorted(
                feature_importance.items(), key=lambda x: x[1], reverse=True
            )[:10]

            self.is_trained = True

            result = {
                "accuracy": float(accuracy),
                "sample_size": len(X),
                "dropped_count": int(n_dropped),
                "completed_count": int(n_completed),
                "top_features": [[str(feat), float(imp)] for feat, imp in top_features],
            }

            print(f"\n🎉 訓練完成！準確率: {accuracy:.2%}")
            print("=" * 60)
            print(f"📊 前 5 重要特徵:")
            for i, (feat, imp) in enumerate(top_features[:5], 1):
                print(f"  {i}. {feat}: {imp:.4f}")
            print("=" * 60 + "\n")

            if self.progress_tracker:
                self.progress_tracker.update(
                    progress=88, message=f"訓練完成！準確率: {accuracy:.2%}"
                )
            logger.info(f"Model training complete: Accuracy={accuracy:.2%}")
            return result

        except Exception as e:
            if self.progress_tracker:
                self.progress_tracker.error(message=f"訓練失敗: {str(e)}")
            logger.error(f"Error training model: {e}", exc_info=True)
            return {
                "accuracy": 0.0,
                "sample_size": 0,
                "dropped_count": 0,
                "completed_count": 0,
                "top_features": [],
                "error": str(e),
            }

    def predict_drop_probability(
        self, anime: Anime, user_id: int, session: Session
    ) -> tuple[float, list[str]]:
        """
        Predict drop probability for a given anime and user.
        Uses trained model with user tolerance features.
        """
        if not self.is_trained or not self.model or not self.feature_columns:
            logger.warning("Model not trained yet")
            return 0.0, ["Model not trained"]

        try:
            from sqlmodel import select

            # Parse anime metadata
            genres = [g.strip() for g in (anime.genres or "").split(",") if g.strip()]
            tags = [t.strip() for t in (anime.tags or "").split(",") if t.strip()]
            studio = anime.studios.split(",")[0].strip() if anime.studios else "Unknown"

            # Calculate user's historical statistics for context
            user_ratings = session.exec(
                select(UserRating).where(UserRating.user_id == user_id)
            ).all()

            # Studio stats
            studio_total = 0
            studio_dropped = 0
            for rating in user_ratings:
                r_anime = session.get(Anime, rating.anime_id)
                if r_anime and r_anime.studios:
                    r_studio = r_anime.studios.split(",")[0].strip()
                    if r_studio == studio:
                        studio_total += 1
                        if rating.status == "DROPPED":
                            studio_dropped += 1

            # Genre stats
            genre_stats = {}
            for genre in genres:
                g_total = 0
                g_dropped = 0
                for rating in user_ratings:
                    r_anime = session.get(Anime, rating.anime_id)
                    if r_anime and r_anime.genres and genre in r_anime.genres:
                        g_total += 1
                        if rating.status == "DROPPED":
                            g_dropped += 1
                if g_total >= 2:
                    genre_stats[genre] = (g_dropped, g_total)

            # Find similar dropped anime
            similar_dropped = []
            for rating in user_ratings:
                if rating.status != "DROPPED":
                    continue
                r_anime = session.get(Anime, rating.anime_id)
                if not r_anime:
                    continue

                r_genres = set((r_anime.genres or "").split(","))
                r_studio = (
                    r_anime.studios.split(",")[0].strip() if r_anime.studios else ""
                )

                common_genres = set(genres) & r_genres
                same_studio = studio == r_studio and studio != "Unknown"

                if len(common_genres) >= 2 or same_studio:
                    similar_dropped.append(
                        (r_anime.title_romaji, len(common_genres), same_studio)
                    )

            # Build feature vector from anime metadata only
            row_data = {
                "episodes": float(anime.episodes or 0),
                "average_score": float(anime.average_score or 0),
                "popularity": float(anime.popularity or 0),
            }

            # Encode genres
            genres_for_encoding = [g for g in genres if g in self.mlb_genres.classes_]
            genres_encoded = self.mlb_genres.transform([genres_for_encoding])
            for i, genre in enumerate(self.mlb_genres.classes_):
                row_data[f"Genre_{genre}"] = int(genres_encoded[0][i])

            # Encode tags
            tags_for_encoding = [t for t in tags if t in self.mlb_tags.classes_]
            tags_encoded = self.mlb_tags.transform([tags_for_encoding])
            # Only use tags that exist (up to 30)
            num_tags = min(len(self.mlb_tags.classes_), 30)
            for i in range(num_tags):
                tag = self.mlb_tags.classes_[i]
                row_data[f"Tag_{tag}"] = int(tags_encoded[0][i])

            # Encode studio
            try:
                studio_code = int(self.le_studio.transform([studio])[0])
            except ValueError:
                studio_code = 0
            row_data["studio_code"] = studio_code

            # Season one-hot
            current_season = anime.season or "UNKNOWN"
            for season in ["WINTER", "SPRING", "SUMMER", "FALL", "UNKNOWN"]:
                row_data[f"Season_{season}"] = 1 if current_season == season else 0

            # Create dataframe
            df = pd.DataFrame([row_data])

            # Add missing columns
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0

            # Ensure same column order
            df = df[self.feature_columns]

            # Predict
            proba = float(self.model.predict_proba(df)[0][1])

            # Build detailed explanation
            reasons = []

            # Risk level description
            if proba >= 0.7:
                reasons.append(f"⚠️ 高風險 {proba:.1%} - 根據你的觀看歷史，很可能棄番")
            elif proba >= 0.4:
                reasons.append(f"⚡ 中風險 {proba:.1%} - 建議謹慎考慮")
            elif proba >= 0.2:
                reasons.append(f"📊 低-中風險 {proba:.1%} - 有一定風險但可嘗試")
            else:
                reasons.append(f"✅ 低風險 {proba:.1%} - 符合你的口味")

            # Studio history
            if studio_total > 0:
                studio_rate = studio_dropped / studio_total
                if studio_rate >= 0.3:
                    reasons.append(
                        f"🏢 你對 {studio} 的棄番率: {studio_rate:.0%} ({studio_dropped}/{studio_total}部)"
                    )
                elif studio_total >= 3:
                    reasons.append(
                        f"🏢 你看過 {studio} 的 {studio_total} 部作品，棄了 {studio_dropped} 部"
                    )

            # Genre history (only show risky ones)
            risky_genres = [
                (g, d, t) for g, (d, t) in genre_stats.items() if d / t >= 0.15
            ]
            if risky_genres:
                risky_genres.sort(key=lambda x: x[1] / x[2], reverse=True)
                g, d, t = risky_genres[0]
                reasons.append(f"🎭 你對 {g} 類型的棄番率: {d / t:.0%} ({d}/{t}部)")

            # Similar dropped anime
            if similar_dropped:
                similar_dropped.sort(key=lambda x: (x[2], x[1]), reverse=True)
                top_sim = similar_dropped[0]
                if top_sim[2]:  # same studio
                    reasons.append(f"⚡ 與你棄番的《{top_sim[0]}》同公司同類型")
                elif top_sim[1] >= 3:
                    reasons.append(
                        f"📌 與你棄番的《{top_sim[0]}》有 {top_sim[1]} 個相同類型"
                    )

            # If low risk, explain why
            if proba < 0.2 and not risky_genres and studio_dropped == 0:
                reasons.append(f"✨ 這類型動畫你通常都會看完")

            # Basic info
            reasons.append(
                f"📺 {', '.join(genres[:3])} | {anime.episodes or '?'} 集 | {studio}"
            )

            logger.debug(
                f"Predicted {proba:.2%} drop probability for {anime.title_romaji}"
            )

            return proba, reasons

        except Exception as e:
            logger.error(
                f"Error predicting for {anime.title_romaji}: {e}", exc_info=True
            )
            return 0.0, [f"Prediction error: {str(e)}"]

    def analyze_drop_patterns(
        self, ratings: list[UserRating], animes: list[Anime]
    ) -> dict[str, Any]:
        """
        Analyze overall drop patterns across all users.
        """
        anime_map = {a.id: a for a in animes}

        tag_stats = {}
        genre_stats = {}
        studio_stats = {}

        for rating in ratings:
            anime = anime_map.get(rating.anime_id)
            if not anime:
                continue

            is_dropped = rating.status == "DROPPED"

            # Analyze genres
            if anime.genres:
                genres = [g.strip() for g in anime.genres.split(",") if g.strip()]
                for genre in genres:
                    if genre not in genre_stats:
                        genre_stats[genre] = {"total": 0, "dropped": 0}
                    genre_stats[genre]["total"] += 1
                    if is_dropped:
                        genre_stats[genre]["dropped"] += 1

            # Analyze tags
            if anime.tags:
                tags = [t.strip() for t in anime.tags.split(",") if t.strip()]
                for tag in tags[:10]:
                    if tag not in tag_stats:
                        tag_stats[tag] = {"total": 0, "dropped": 0}
                    tag_stats[tag]["total"] += 1
                    if is_dropped:
                        tag_stats[tag]["dropped"] += 1

            # Analyze studios
            if anime.studios:
                studio = anime.studios.split(",")[0].strip()
                if studio not in studio_stats:
                    studio_stats[studio] = {"total": 0, "dropped": 0}
                studio_stats[studio]["total"] += 1
                if is_dropped:
                    studio_stats[studio]["dropped"] += 1

        # Calculate drop rates and sort
        def calculate_drop_rate(stats: dict, min_samples: int = 3) -> list[dict]:
            results = []
            for name, data in stats.items():
                if data["total"] >= min_samples:
                    drop_rate = data["dropped"] / data["total"]
                    results.append(
                        {
                            "name": name,
                            "drop_rate": float(drop_rate),
                            "total": data["total"],
                            "dropped": data["dropped"],
                        }
                    )
            return sorted(results, key=lambda x: x["drop_rate"], reverse=True)[:10]

        return {
            "top_dropped_genres": calculate_drop_rate(genre_stats),
            "top_dropped_tags": calculate_drop_rate(tag_stats),
            "top_dropped_studios": calculate_drop_rate(studio_stats),
        }
