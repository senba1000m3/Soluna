"""
Hybrid Recommendation Engine
結合 BERT 序列推薦和內容特徵推薦的混合推薦系統
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sqlmodel import Session
from tqdm import tqdm

from bert_model.bert_recommender_optimized import OptimizedBERTRecommender
from database import engine
from recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """
    混合推薦引擎

    兩階段推薦流程：
    1. 使用 BERT 模型從已知動畫中找出使用者偏好的參考動畫
    2. 分析參考動畫的特徵，用於評分新番動畫
    """

    def __init__(
        self,
        bert_model_path: Optional[str] = "bert_model/trained_models/best_model.pth",
        bert_dataset_path: Optional[
            str
        ] = "bert_model/trained_models/item_mappings.pkl",
        use_bert: bool = True,
    ):
        """
        初始化混合推薦引擎

        Args:
            bert_model_path: BERT 模型路徑
            bert_dataset_path: BERT 映射資料路徑
            use_bert: 是否啟用 BERT 推薦（False 時僅使用內容推薦）
        """
        # 內容推薦引擎（基於 genre/tags）
        self.content_engine = RecommendationEngine()

        # BERT 推薦器
        self.bert_recommender = None
        self.use_bert = use_bert

        if use_bert:
            try:
                # 檢查模型檔案是否存在
                from pathlib import Path

                if not Path(bert_model_path).exists():
                    logger.warning(
                        f"BERT model not found at {bert_model_path}, disabling BERT recommendations"
                    )
                    self.use_bert = False
                    return

                with Session(engine) as session:
                    self.bert_recommender = OptimizedBERTRecommender(
                        model_path=bert_model_path,
                        dataset_path=bert_dataset_path,
                        db_session=session,
                        device="auto",
                    )
                logger.info("✅ BERT recommender loaded successfully")
            except Exception as e:
                logger.error(f"Failed to initialize BERT recommender: {e}")
                self.use_bert = False
                self.bert_recommender = None

    def recommend_seasonal(
        self,
        user_list: List[Dict[str, Any]],
        seasonal_anime: List[Dict[str, Any]],
        bert_weight: float = 0.8,
        content_weight: float = 0.2,
        top_reference_anime: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        混合推薦新番動畫

        Args:
            user_list: 使用者的動畫列表（來自 AniList）
            seasonal_anime: 當季新番列表
            bert_weight: BERT 特徵的權重
            content_weight: 內容特徵的權重
            top_reference_anime: 從 BERT 推薦中取前 K 個作為參考

        Returns:
            評分後的新番列表
        """
        if not seasonal_anime:
            return []

        # 階段 1: 建立使用者 profile
        print("\n" + "=" * 60)
        print("🎯 混合推薦引擎 - 開始推薦")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("HYBRID RECOMMENDATION ENGINE - Starting recommendation")
        logger.info(f"BERT enabled: {self.use_bert}")
        logger.info(
            f"BERT available: {self.bert_recommender is not None if self.use_bert else False}"
        )
        logger.info("=" * 60)
        print(f"  ├─ BERT 啟用: {'是' if self.use_bert else '否'}")
        print(f"  ├─ 使用者動畫數: {len(user_list)}")
        print(f"  └─ 新番數量: {len(seasonal_anime)}")
        print("\n📋 階段 1/3: 建立使用者 Profile")
        logger.info("Building user profile from watch history...")
        logger.info(f"User has {len(user_list)} entries in their list")

        # 內容 profile（基於實際觀看的動畫）
        print("  ├─ 建立內容 Profile...")
        with tqdm(total=100, desc="  │  內容分析", unit="%", leave=False) as pbar:
            content_profile = self.content_engine.build_user_profile(user_list)
            pbar.update(100)
        logger.info(f"Content profile built with {len(content_profile)} features")
        print(f"  │  ✓ 內容 Profile: {len(content_profile)} 個特徵")
        if content_profile:
            # 顯示前 5 個 genre 權重
            genre_items = [
                (k, v) for k, v in content_profile.items() if k.startswith("Genre_")
            ]
            genre_items.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"Top 5 genres in profile:")
            for genre, weight in genre_items[:5]:
                logger.info(f"  {genre}: {weight:.3f}")
        else:
            logger.warning(
                "Content profile is EMPTY! This will result in all 50% scores."
            )
            print("  │  ⚠️  內容 Profile 為空！")

        # BERT-enhanced profile（如果可用）
        bert_profile = None
        if self.use_bert and self.bert_recommender:
            print("  ├─ 建立 BERT Profile...")
            logger.info("Attempting to build BERT-enhanced profile...")
            with tqdm(total=100, desc="  │  BERT 分析", unit="%", leave=False) as pbar:
                bert_profile = self._build_bert_enhanced_profile(
                    user_list, top_k=top_reference_anime
                )
                pbar.update(100)
            if bert_profile:
                logger.info("BERT profile successfully built")
                print(f"  │  ✓ BERT Profile: {len(bert_profile)} 個特徵")
            else:
                logger.warning("BERT profile is None")
                print("  │  ⚠️  BERT Profile 為空")
        else:
            print("  └─ 跳過 BERT Profile (未啟用)")

        # 階段 2: 評分新番
        print(f"\n📋 階段 2/3: 評分 {len(seasonal_anime)} 部新番動畫")
        logger.info(f"Scoring {len(seasonal_anime)} seasonal anime...")

        scored_anime = []
        for anime in tqdm(seasonal_anime, desc="  評分進度", unit="部"):
            anime_copy = anime.copy()

            # 內容分數
            content_score = self._calculate_content_score(anime, content_profile)

            # BERT 分數（如果可用）
            bert_score = 0.0
            if bert_profile:
                bert_score = self._calculate_bert_score(anime, bert_profile)

            # 合併分數
            if self.use_bert and bert_profile:
                final_score = content_score * content_weight + bert_score * bert_weight
            else:
                final_score = content_score

            anime_copy["match_score"] = float(final_score)
            anime_copy["content_score"] = float(content_score)
            anime_copy["bert_score"] = float(bert_score) if bert_profile else None

            # Debug logging for first 3 anime
            if len(scored_anime) < 3:
                logger.info(
                    f"Anime #{len(scored_anime) + 1}: {anime.get('title', {}).get('romaji', 'Unknown')}"
                )
                logger.info(f"  Genres: {anime.get('genres', [])}")
                logger.info(f"  Content score: {content_score:.2f}")
                logger.info(f"  BERT score: {bert_score:.2f}")
                logger.info(f"  Final score: {final_score:.2f}")

            # 生成推薦理由
            anime_copy["match_reasons"] = self._generate_match_reasons(
                anime, content_profile, bert_profile
            )

            scored_anime.append(anime_copy)

        # 排序
        print("\n📋 階段 3/3: 排序推薦結果")
        with tqdm(total=100, desc="  排序中", unit="%") as pbar:
            scored_anime.sort(key=lambda x: x["match_score"], reverse=True)
            pbar.update(100)

        # 最終統計
        if scored_anime:
            print("\n" + "=" * 60)
            print("🎉 推薦完成！")
            print("=" * 60)
            print(f"  ✓ 總共評分: {len(scored_anime)} 部")
            print(
                f"  ✓ 前 3 名分數: {[f'{a["match_score"]:.1f}' for a in scored_anime[:3]]}"
            )
            print(
                f"  ✓ 分數範圍: {scored_anime[-1]['match_score']:.1f} - {scored_anime[0]['match_score']:.1f}"
            )
            print("=" * 60 + "\n")
            logger.info("=" * 60)
            logger.info("HYBRID RECOMMENDATION ENGINE - Results")
            logger.info(f"Total anime scored: {len(scored_anime)}")
            logger.info(
                f"Top 3 scores: {[f'{a["match_score"]:.1f}' for a in scored_anime[:3]]}"
            )
            logger.info(
                f"Score range: {scored_anime[-1]['match_score']:.1f} - {scored_anime[0]['match_score']:.1f}"
            )
            logger.info("=" * 60)

        return scored_anime

    def _build_bert_enhanced_profile(
        self, user_list: List[Dict[str, Any]], top_k: int = 50
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """
        使用 BERT 建立增強的使用者 profile

        Args:
            user_list: 使用者觀看列表
            top_k: 取前 K 個 BERT 推薦作為參考

        Returns:
            增強的特徵 profile (genres/tags 加權)
        """
        if not self.bert_recommender:
            return None

        try:
            # 提取使用者觀看過的動畫 ID
            user_anime_ids = []
            for entry in user_list:
                anime = entry.get("media", entry)
                anime_id = anime.get("id")
                if anime_id:
                    user_anime_ids.append(anime_id)

            if not user_anime_ids:
                logger.warning("No anime IDs found in user list")
                return None

            # 使用 BERT 獲取推薦
            logger.info(
                f"Getting BERT recommendations for {len(user_anime_ids)} anime..."
            )

            with Session(engine) as session:
                self.bert_recommender.db_session = session
                bert_recommendations = self.bert_recommender.get_recommendations(
                    user_anime_ids=user_anime_ids, top_k=top_k, use_anilist_ids=True
                )

            if not bert_recommendations:
                logger.warning("BERT returned no recommendations")
                return None

            logger.info(f"Got {len(bert_recommendations)} BERT recommendations")

            # 從推薦結果建立特徵 profile
            # 使用推薦分數作為權重，累積 genres 和 tags
            genre_weights = defaultdict(float)
            tag_weights = defaultdict(float)

            for rec in bert_recommendations:
                score = rec.get("score", 0.5)

                # 累積 genres
                for genre in rec.get("genres", []):
                    genre_weights[f"Genre_{genre}"] += score

                # 累積 tags (只取高相關的)
                for tag in rec.get("tags", [])[:10]:  # 只取前 10 個 tags
                    if isinstance(tag, dict):
                        tag_name = tag.get("name", "")
                        tag_rank = tag.get("rank", 0)
                        if tag_rank >= 60:  # 只取相關度高的 tag
                            tag_weights[f"Tag_{tag_name}"] += score * (tag_rank / 100)
                    elif isinstance(tag, str):
                        tag_weights[f"Tag_{tag}"] += score

            # 正規化權重
            total_weight = sum(genre_weights.values()) + sum(tag_weights.values())
            if total_weight > 0:
                for k in genre_weights:
                    genre_weights[k] /= total_weight
                for k in tag_weights:
                    tag_weights[k] /= total_weight

            # 合併成一個 profile
            bert_profile = {**genre_weights, **tag_weights}

            logger.info(f"BERT profile built with {len(bert_profile)} features")
            logger.info(
                f"Top 5 genres: {sorted(genre_weights.items(), key=lambda x: x[1], reverse=True)[:5]}"
            )

            return bert_profile if bert_profile else None

        except Exception as e:
            logger.error(f"Error building BERT-enhanced profile: {e}")
            return None

    def _weight_features_by_score(
        self, features: Dict[str, Any], recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, float]]:
        """
        根據推薦分數加權特徵

        Args:
            features: 原始特徵計數
            recommendations: 推薦列表（包含分數）

        Returns:
            加權後的特徵
        """
        # 正規化分數
        scores = [rec.get("score", 0) for rec in recommendations]
        if not scores or max(scores) == 0:
            return features

        max_score = max(scores)
        normalized_scores = [s / max_score for s in scores]

        # 這裡簡化處理，直接返回原始特徵
        # 實際上可以根據推薦位置/分數進行加權
        weighted = {}
        for feature_type, feature_dict in features.items():
            weighted[feature_type] = {k: float(v) for k, v in feature_dict.items()}

        return weighted

    def _calculate_content_score(
        self, anime: Dict[str, Any], content_profile: Dict[str, float]
    ) -> float:
        """
        計算內容相似度分數（基於 genre/tags）
        使用與原始推薦引擎相同的邏輯

        Args:
            anime: 動畫資料
            content_profile: 使用者內容 profile

        Returns:
            相似度分數 (0-100)
        """
        if not content_profile:
            logger.warning("Content profile is empty, returning default score 50.0")
            return 50.0  # 預設中等分數

        # 提取動畫的 genres
        genres = anime.get("genres", [])
        if not genres:
            return 50.0

        # 獲取 profile 中的所有 genre keys
        profile_genre_keys = [
            k for k in content_profile.keys() if k.startswith("Genre_")
        ]

        if not profile_genre_keys:
            return 50.0

        # 建立動畫的 genre keys
        anime_genre_keys = [f"Genre_{genre}" for genre in genres]

        # 合併所有 genres (union)
        all_genres = list(set(profile_genre_keys + anime_genre_keys))
        all_genres.sort()

        # 建立使用者向量
        user_vec = np.array([content_profile.get(g, 0.0) for g in all_genres])

        # 建立動畫向量 (anime has genre = 1, else = 0)
        anime_vec = np.array(
            [1.0 if g in anime_genre_keys else 0.0 for g in all_genres]
        )

        # 確保向量不全為零
        user_norm = np.linalg.norm(user_vec)
        anime_norm = np.linalg.norm(anime_vec)

        if user_norm == 0 or anime_norm == 0:
            return 50.0

        # 計算 cosine similarity
        similarity = cosine_similarity(
            user_vec.reshape(1, -1), anime_vec.reshape(1, -1)
        )[0][0]

        # 轉換為 0-100 分數
        score = float(similarity * 100)

        # 確保分數在合理範圍內
        score = max(0.0, min(100.0, score))

        return score

    def _calculate_bert_score(
        self, anime: Dict[str, Any], bert_profile: Dict[str, Any]
    ) -> float:
        """
        計算 BERT 特徵相似度分數

        Args:
            anime: 動畫資料
            bert_profile: BERT 增強的 profile (扁平化的 Genre_/Tag_ 結構)

        Returns:
            相似度分數 (0-100)
        """
        if not bert_profile:
            return 50.0

        # 建立動畫的特徵向量
        anime_features = {}

        # 提取 genres
        for genre in anime.get("genres", []):
            anime_features[f"Genre_{genre}"] = 1.0

        # 提取 tags
        for tag in anime.get("tags", []):
            if isinstance(tag, dict):
                tag_name = tag.get("name", "")
                tag_rank = tag.get("rank", 0)
                if tag_rank >= 60:  # 只使用高相關的 tag
                    anime_features[f"Tag_{tag_name}"] = tag_rank / 100
            elif isinstance(tag, str):
                anime_features[f"Tag_{tag}"] = 1.0

        # 計算加權相似度
        score = 0.0
        total_weight = 0.0

        for feature, weight in bert_profile.items():
            total_weight += weight
            if feature in anime_features:
                score += weight * anime_features[feature] * 100

        # 正規化分數
        if total_weight > 0:
            score = score / total_weight
        else:
            score = 50.0

        return min(100.0, max(0.0, score))

    def _generate_match_reasons(
        self,
        anime: Dict[str, Any],
        content_profile: Dict[str, float],
        bert_profile: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        生成推薦理由（匹配前端期望的格式）

        Args:
            anime: 動畫資料
            content_profile: 內容 profile
            bert_profile: BERT profile

        Returns:
            包含 matched_genres, total_weight, top_reason 的字典
        """
        reasons = {"matched_genres": [], "total_weight": 0.0, "top_reason": ""}

        # Genre 匹配
        anime_genres = set(anime.get("genres", []))
        genre_matches = []

        if content_profile:
            # 從 content_profile 中找出匹配的 genres 及其權重
            for key, weight in content_profile.items():
                if key.startswith("Genre_"):
                    genre_name = key.replace("Genre_", "")
                    if genre_name in anime_genres and weight > 0:
                        genre_matches.append(
                            {"genre": genre_name, "weight": float(weight)}
                        )

        # 如果有 BERT profile，也考慮其 genre 資訊
        if bert_profile and "genres" in bert_profile:
            bert_genres = bert_profile["genres"]
            for genre_name in anime_genres:
                if genre_name in bert_genres:
                    # 如果已經在 content_profile 中，增加權重
                    existing = next(
                        (g for g in genre_matches if g["genre"] == genre_name), None
                    )
                    if existing:
                        existing["weight"] += bert_genres[genre_name] * 0.5
                    else:
                        genre_matches.append(
                            {
                                "genre": genre_name,
                                "weight": float(bert_genres[genre_name] * 0.5),
                            }
                        )

        # 排序並取前 5
        genre_matches.sort(key=lambda x: x["weight"], reverse=True)
        reasons["matched_genres"] = genre_matches[:5]
        reasons["total_weight"] = sum(g["weight"] for g in genre_matches)

        # 生成 top_reason
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
            # 檢查是否有 studio 匹配
            if bert_profile and "studios" in bert_profile:
                anime_studios = set()
                for studio in anime.get("studios", []):
                    if isinstance(studio, dict):
                        anime_studios.add(studio.get("name", ""))
                    else:
                        anime_studios.add(studio)

                bert_studios = set(bert_profile["studios"].keys())
                matched_studios = anime_studios & bert_studios
                if matched_studios:
                    reasons["top_reason"] = (
                        f"來自你喜歡的製作公司: {', '.join(list(matched_studios)[:2])}"
                    )
                else:
                    reasons["top_reason"] = "基於整體偏好匹配"
            else:
                reasons["top_reason"] = "基於整體偏好匹配"

        return reasons

    def build_user_profile(self, user_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        建立完整的使用者 profile（包含 content 和 BERT）

        Args:
            user_list: 使用者觀看列表

        Returns:
            完整的 profile 字典
        """
        profile = {
            "content": self.content_engine.build_user_profile(user_list),
            "bert": None,
        }

        if self.use_bert and self.bert_recommender:
            profile["bert"] = self._build_bert_enhanced_profile(user_list)

        return profile


# 測試程式碼
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 測試初始化（不使用 BERT）
    engine = HybridRecommendationEngine(use_bert=False)
    print("Hybrid engine initialized (content-only mode)")

    # 測試完整功能需要實際的模型檔案
    # engine = HybridRecommendationEngine(
    #     bert_model_path="path/to/model.pth",
    #     bert_dataset_path="path/to/dataset.pkl",
    #     bert_metadata_path="path/to/animes.json",
    #     use_bert=True
    # )
