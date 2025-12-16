"""
混合推薦引擎 (Hybrid Recommendation Engine)
結合 BERT4Rec 序列推薦 (80%) 和內容特徵推薦 (20%)

設計理念:
- BERT4Rec: 根據使用者的觀看序列，預測最符合使用者口味的動畫
- Content-Based: 根據動畫的靜態特徵 (類型、標籤等)，計算內容相似度
- 權重: BERT 80% (主要) + Content 20% (輔助)

"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlmodel import Session

from bert_model.bert_recommender_optimized import OptimizedBERTRecommender
from recommendation_engine import RecommendationEngine

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """
    混合推薦引擎

    整合兩種推薦方法:
    1. BERT4Rec 序列推薦 (80%): 基於使用者的觀看序列，預測喜歡的動畫模式
    2. Content-Based 特徵推薦 (20%): 基於動畫的特徵 (類型、標籤等)，計算相似度

    推薦邏輯:
    - BERT 分數越高 = 越符合使用者的觀看序列 = 越推薦
    - Content 分數越高 = 越符合使用者的偏好特徵 = 越推薦
    """

    def __init__(
        self,
        bert_model_path: str = "bert_model/trained_models/best_model.pth",
        bert_dataset_path: str = "bert_model/trained_models/item_mappings.pkl",
        bert_db_path: str = "bert_model/bert.db",
        bert_weight: float = 0.8,
        content_weight: float = 0.2,
        use_bert: bool = True,
        progress_tracker: Optional[Any] = None,
    ):
        """
        初始化混合推薦引擎

        Args:
            bert_model_path: BERT 模型路徑
            bert_dataset_path: BERT 映射資料路徑
            bert_db_path: BERT 資料庫路徑
            bert_weight: BERT 推薦的權重 (預設 0.8)
            content_weight: Content 推薦的權重 (預設 0.2)
            use_bert: 是否啟用 BERT (False 時僅使用 Content)
            progress_tracker: 進度追蹤器
        """
        self.bert_weight = bert_weight
        self.content_weight = content_weight
        self.use_bert = use_bert
        self.progress_tracker = progress_tracker

        # 初始化 Content 引擎
        self.content_engine = RecommendationEngine()

        # 初始化 BERT 推薦器
        self.bert_recommender = None
        self.bert_model_path = bert_model_path
        self.bert_dataset_path = bert_dataset_path
        self.bert_db_path = bert_db_path

        if use_bert:
            try:
                from pathlib import Path

                if not Path(bert_model_path).exists():
                    logger.warning(
                        f"BERT model not found at {bert_model_path}, falling back to Content only"
                    )
                    self.use_bert = False
                elif not Path(bert_db_path).exists():
                    logger.warning(
                        f"BERT database not found at {bert_db_path}, falling back to Content only"
                    )
                    self.use_bert = False
                else:
                    logger.info(f"BERT model path validated: {bert_model_path}")
                    logger.info(f"BERT database path validated: {bert_db_path}")
            except Exception as e:
                logger.error(f"Failed to validate BERT paths: {e}")
                self.use_bert = False

        logger.info(
            f"Hybrid Recommendation Engine initialized: BERT {self.bert_weight * 100}% + Content {self.content_weight * 100}%"
        )

    def _initialize_bert_recommender(self, session: Session):
        """
        初始化 BERT 推薦器 (延遲初始化，需要 DB session)

        Args:
            session: 資料庫 session
        """
        if not self.use_bert:
            return

        if self.bert_recommender is None:
            try:
                if self.progress_tracker:
                    self.progress_tracker.update(
                        progress=5,
                        message="初始化 BERT 推薦模型...",
                    )

                logger.info("Initializing BERT recommender...")

                from bert_model.bert_recommender_optimized import (
                    OptimizedBERTRecommender,
                )

                self.bert_recommender = OptimizedBERTRecommender(
                    model_path=self.bert_model_path,
                    dataset_path=self.bert_dataset_path,
                    db_session=session,
                    device="auto",
                )

                if self.progress_tracker:
                    self.progress_tracker.update(
                        progress=10,
                        message="BERT 模型初始化完成",
                    )

                logger.info("✅ BERT recommender initialized successfully")
            except Exception as e:
                logger.error(f"❌ Failed to initialize BERT recommender: {e}")
                logger.exception(e)
                self.use_bert = False
                self.bert_recommender = None

                if self.progress_tracker:
                    self.progress_tracker.update(
                        progress=10,
                        message="BERT 初始化失敗，使用純 Content 模式",
                    )

    def _get_user_sequence(self, user_list: List[Dict[str, Any]]) -> List[int]:
        """
        從使用者動畫列表提取觀看序列

        Args:
            user_list: 使用者的動畫列表

        Returns:
            動畫 ID 列表
        """
        sequence = []
        for entry in user_list:
            # 提取動畫物件
            anime = entry.get("media", entry)
            anime_id = anime.get("id")

            if anime_id:
                # 只包含已完成、正在觀看的動畫
                status = entry.get("status", "")
                if status in ["COMPLETED", "CURRENT", "REPEATING"]:
                    sequence.append(anime_id)

        return sequence

    def _predict_bert_score(
        self,
        anime_id: int,
        user_sequence: List[int],
        session: Session,
        top_k: int = 200,
    ) -> float:
        """
        使用 BERT 預測推薦分數

        邏輯:
        - BERT 推薦分數高 = 符合使用者觀看模式 = 推薦
        - 推薦分數範圍: 0-1

        Args:
            anime_id: 動畫 ID
            user_sequence: 使用者的觀看序列
            session: 資料庫 session
            top_k: 從 BERT 取前 K 個推薦

        Returns:
            推薦分數 (0-1，越高越推薦)
        """
        if not self.bert_recommender or not user_sequence:
            return 0.5  # 無法預測時返回中性值

        try:
            # 使用 BERT 推薦器預測
            # get_recommendations 返回推薦的動畫列表
            import concurrent.futures

            def get_bert_recommendations():
                return self.bert_recommender.get_recommendations(
                    user_anime_ids=user_sequence,
                    top_k=top_k,
                    use_anilist_ids=True,
                    force_refresh=False,
                )

            # 使用 ThreadPoolExecutor 實現跨平台超時
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(get_bert_recommendations)
                try:
                    recommendations = future.result(timeout=15.0)  # 15 秒超時
                except concurrent.futures.TimeoutError:
                    logger.warning(f"BERT prediction timeout for anime {anime_id}")
                    return 0.5  # 超時時返回中性值

            # 檢查該動畫是否在推薦列表中
            bert_score = 0.0
            for rec in recommendations:
                rec_anime_id = rec.get("id") or rec.get("anime_id")
                rec_score = rec.get("score", 0.0)
                if rec_anime_id == anime_id:
                    bert_score = rec_score
                    break

            # 如果不在推薦列表中，根據位置給予較低分數
            if bert_score == 0.0:
                logger.debug(
                    f"Anime {anime_id} not in top {top_k} BERT recommendations"
                )
                # 不在前 K 名 = 較低分數
                bert_score = 0.1

            logger.debug(f"Anime {anime_id}: BERT score={bert_score:.3f}")

            return float(bert_score)

        except Exception as e:
            logger.error(f"Error predicting BERT score: {e}")
            return 0.5  # 錯誤時返回中性值

    def _calculate_content_score(
        self, anime: Dict[str, Any], user_profile: Dict[str, float]
    ) -> float:
        """
        計算內容相似度分數

        Args:
            anime: 動畫資料
            user_profile: 使用者內容 profile

        Returns:
            相似度分數 (0-1)
        """
        if not user_profile:
            return 0.5

        # 提取動畫的 genres
        anime_genres = set(anime.get("genres", []))
        if not anime_genres:
            return 0.5

        # 計算 genre 匹配分數
        total_weight = 0.0
        matched_weight = 0.0

        for genre in anime_genres:
            genre_key = f"Genre_{genre}"
            if genre_key in user_profile:
                weight = user_profile[genre_key]
                matched_weight += weight
                total_weight += weight

        # 如果有匹配的 genre，計算相似度
        if total_weight > 0:
            # 正規化到 0-1
            score = matched_weight / sum(user_profile.values())
            # 轉換到更合理的範圍 (0.3-1.0)
            score = 0.3 + (score * 0.7)
        else:
            score = 0.5

        return min(1.0, max(0.0, score))

    def _generate_match_reasons(
        self,
        anime: Dict[str, Any],
        user_profile: Dict[str, float],
        bert_score: float,
        content_score: float,
    ) -> Dict[str, Any]:
        """
        生成推薦理由

        Args:
            anime: 動畫資料
            user_profile: 使用者 profile
            bert_score: BERT 分數
            content_score: Content 分數

        Returns:
            包含 matched_genres, total_weight, top_reason 的字典
        """
        reasons = {"matched_genres": [], "total_weight": 0.0, "top_reason": ""}

        # Genre 匹配
        anime_genres = set(anime.get("genres", []))
        genre_matches = []

        for genre in anime_genres:
            genre_key = f"Genre_{genre}"
            if genre_key in user_profile:
                weight = user_profile[genre_key]
                if weight > 0:
                    genre_matches.append({"genre": genre, "weight": float(weight)})

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
            reasons["top_reason"] = "基於整體觀看模式推薦"

        # 添加分數資訊
        if self.use_bert and self.bert_recommender:
            reasons["bert_score"] = float(bert_score)
            reasons["content_score"] = float(content_score)

        return reasons

    def recommend_seasonal(
        self,
        user_list: List[Dict[str, Any]],
        seasonal_anime: List[Dict[str, Any]],
        session: Session = None,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        混合推薦新番動畫 (主要方法)

        Args:
            user_list: 使用者的動畫列表（來自 AniList）
            seasonal_anime: 當季新番列表
            session: 資料庫 session (用於 BERT)
            top_k: 返回前 K 個推薦

        Returns:
            評分後的新番列表（按分數排序）
        """
        if not seasonal_anime:
            return []

        logger.info("=" * 60)
        logger.info("HYBRID RECOMMENDATION ENGINE - Starting recommendation")
        logger.info(f"BERT enabled: {self.use_bert}")
        logger.info(
            f"BERT available: {self.bert_recommender is not None if self.use_bert else False}"
        )
        logger.info(f"User anime count: {len(user_list)}")
        logger.info(f"Seasonal anime count: {len(seasonal_anime)}")
        logger.info("=" * 60)

        # 初始化 BERT 推薦器 (如果需要)
        if self.use_bert and self.bert_recommender is None and session:
            self._initialize_bert_recommender(session)

        # 1. 建立使用者 profile
        logger.info("Building user profile from watch history...")
        user_profile = self.content_engine.build_user_profile(user_list)
        logger.info(f"Content profile built with {len(user_profile)} features")

        # 2. 提取使用者序列 (用於 BERT)
        user_sequence = []
        if self.use_bert and self.bert_recommender:
            user_sequence = self._get_user_sequence(user_list)
            logger.info(f"User sequence length: {len(user_sequence)}")

        # 3. 評分每部新番
        logger.info(f"Scoring {len(seasonal_anime)} seasonal anime...")
        scored_anime = []

        for anime in seasonal_anime:
            anime_copy = anime.copy()
            anime_id = anime.get("id")

            # Content 分數
            content_score = self._calculate_content_score(anime, user_profile)

            # BERT 分數
            bert_score = 0.5
            if self.use_bert and self.bert_recommender and user_sequence and session:
                bert_score = self._predict_bert_score(
                    anime_id, user_sequence, session, top_k=200
                )

            # 混合分數
            if self.use_bert and self.bert_recommender and user_sequence:
                # BERT + Content 混合
                final_score = (
                    bert_score * self.bert_weight + content_score * self.content_weight
                )
            else:
                # 僅使用 Content
                final_score = content_score

            # 轉換為 0-100 分數 (匹配前端期望)
            final_score = final_score * 100

            anime_copy["match_score"] = float(final_score)
            anime_copy["content_score"] = float(content_score * 100)
            anime_copy["bert_score"] = (
                float(bert_score * 100) if self.use_bert else None
            )

            # 生成推薦理由
            anime_copy["match_reasons"] = self._generate_match_reasons(
                anime, user_profile, bert_score, content_score
            )

            scored_anime.append(anime_copy)

        # 4. 排序並返回 top K
        scored_anime.sort(key=lambda x: x["match_score"], reverse=True)

        logger.info(f"Recommendation complete. Returning top {top_k} results.")
        logger.info(f"Top 3 scores: {[a['match_score'] for a in scored_anime[:3]]}")

        return scored_anime[:top_k]

    @property
    def is_available(self) -> bool:
        """檢查推薦引擎是否可用"""
        return True  # Content engine 總是可用

    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊

        Returns:
            模型狀態和配置資訊
        """
        return {
            "mode": "hybrid" if self.use_bert else "content_only",
            "bert_enabled": self.use_bert,
            "bert_available": self.bert_recommender is not None,
            "bert_weight": self.bert_weight if self.use_bert else 0.0,
            "content_weight": self.content_weight if self.use_bert else 1.0,
        }
