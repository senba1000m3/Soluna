"""
混合棄番預測引擎 (Hybrid Drop Prediction Engine)
結合 BERT4Rec 序列推薦 (80%) 和 XGBoost 特徵分類 (20%)

設計理念:
- BERT4Rec: 根據使用者的觀看序列，預測不太可能繼續看的動畫 (序列不連貫)
- XGBoost: 根據動畫的靜態特徵 (類型、標籤、製作公司等)，預測棄番風險
- 權重: BERT 80% (主要) + XGBoost 20% (輔助)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlmodel import Session

from bert_model.bert_recommender_optimized import OptimizedBERTRecommender
from drop_analysis_engine import DropAnalysisEngine
from models import Anime, UserRating

logger = logging.getLogger(__name__)


class HybridDropPredictionEngine:
    """
    混合棄番預測引擎

    整合兩種預測方法:
    1. BERT4Rec 序列預測 (80%): 基於使用者的觀看序列，預測哪些動畫不符合使用者的觀看模式
    2. XGBoost 特徵預測 (20%): 基於動畫的特徵 (類型、標籤等)，預測棄番風險

    預測邏輯:
    - BERT 分數越低 = 越不符合使用者的觀看序列 = 越可能棄番
    - XGBoost 分數越高 = 棄番機率越高
    """

    def __init__(
        self,
        bert_model_path: str = "bert_model/trained_models/best_model.pth",
        bert_dataset_path: str = "bert_model/trained_models/item_mappings.pkl",
        bert_weight: float = 0.8,
        xgboost_weight: float = 0.2,
        use_bert: bool = True,
        progress_tracker: Optional[Any] = None,
    ):
        """
        初始化混合棄番預測引擎

        Args:
            bert_model_path: BERT 模型路徑
            bert_dataset_path: BERT 映射資料路徑
            bert_weight: BERT 預測的權重 (預設 0.8)
            xgboost_weight: XGBoost 預測的權重 (預設 0.2)
            use_bert: 是否啟用 BERT (False 時僅使用 XGBoost)
            progress_tracker: 進度追蹤器
        """
        self.bert_weight = bert_weight
        self.xgboost_weight = xgboost_weight
        self.use_bert = use_bert
        self.progress_tracker = progress_tracker

        # 初始化 XGBoost 引擎
        self.xgboost_engine = DropAnalysisEngine(progress_tracker=progress_tracker)

        # 初始化 BERT 推薦器
        self.bert_recommender = None
        if use_bert:
            try:
                from pathlib import Path

                if not Path(bert_model_path).exists():
                    logger.warning(
                        f"BERT model not found at {bert_model_path}, falling back to XGBoost only"
                    )
                    self.use_bert = False
                else:
                    # BERT 推薦器需要 DB session，將在預測時傳入
                    self.bert_model_path = bert_model_path
                    self.bert_dataset_path = bert_dataset_path
                    logger.info("BERT model path validated")
            except Exception as e:
                logger.error(f"Failed to validate BERT model: {e}")
                self.use_bert = False

        logger.info(
            f"Hybrid Drop Prediction Engine initialized: BERT {self.bert_weight * 100}% + XGBoost {self.xgboost_weight * 100}%"
        )

    def train_xgboost_model(self, session: Session, user_id: int) -> Dict[str, Any]:
        """
        訓練 XGBoost 模型

        Args:
            session: 資料庫 session
            user_id: 使用者 ID

        Returns:
            訓練結果 (準確率、樣本數等)
        """
        if self.progress_tracker:
            self.progress_tracker.update(
                progress=35,
                stage="train_xgboost",
                message="訓練 XGBoost 模型 (20% 權重)...",
            )

        logger.info(f"Training XGBoost model for user {user_id}")
        result = self.xgboost_engine.train_model(session, user_id=user_id)

        if self.progress_tracker:
            self.progress_tracker.update(progress=70, message="XGBoost 訓練完成")

        return result

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
                self.bert_recommender = OptimizedBERTRecommender(
                    model_path=self.bert_model_path,
                    dataset_path=self.bert_dataset_path,
                    db_session=session,
                    device="auto",
                )
                logger.info("BERT recommender initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize BERT recommender: {e}")
                self.use_bert = False
                self.bert_recommender = None

    def _get_user_sequence(self, user_id: int, session: Session) -> List[int]:
        """
        取得使用者的動畫觀看序列

        Args:
            user_id: 使用者 ID
            session: 資料庫 session

        Returns:
            動畫 ID 列表 (依時間排序)
        """
        from sqlmodel import select

        # 取得使用者的所有評分記錄
        ratings = session.exec(
            select(UserRating)
            .where(UserRating.user_id == user_id)
            .where(UserRating.status.in_(["COMPLETED", "CURRENT", "DROPPED", "PAUSED"]))
        ).all()

        # 依更新時間排序 (如果有的話)
        if ratings and hasattr(ratings[0], "updated_at"):
            ratings = sorted(ratings, key=lambda r: r.updated_at or r.created_at)

        # 返回動畫 ID 序列
        return [rating.anime_id for rating in ratings]

    def _predict_bert_drop_score(
        self, anime_id: int, user_sequence: List[int], session: Session
    ) -> float:
        """
        使用 BERT 預測棄番分數

        邏輯:
        - BERT 推薦分數高 = 符合使用者觀看模式 = 不太可能棄番
        - 棄番分數 = 1 - BERT 推薦分數 (反轉)

        Args:
            anime_id: 動畫 ID
            user_sequence: 使用者的觀看序列
            session: 資料庫 session

        Returns:
            棄番分數 (0-1，越高越可能棄番)
        """
        if not self.bert_recommender or not user_sequence:
            return 0.5  # 無法預測時返回中性值

        try:
            # 使用 BERT 推薦器預測
            # get_recommendations 返回推薦的動畫及其分數
            recommendations = self.bert_recommender.get_recommendations(
                user_sequence=user_sequence, top_k=100, session=session
            )

            # 檢查該動畫是否在推薦列表中
            bert_score = 0.0
            for rec_anime_id, score in recommendations:
                if rec_anime_id == anime_id:
                    bert_score = score
                    break

            # 如果不在推薦列表中，給予低分 (高棄番風險)
            if bert_score == 0.0:
                # 檢查是否在前 100 名之外
                logger.debug(f"Anime {anime_id} not in top 100 BERT recommendations")
                bert_score = 0.1  # 很低的推薦分數 = 很高的棄番風險

            # 轉換為棄番分數 (反轉)
            # BERT score 範圍通常是 0-1
            # 高 BERT 分數 = 低棄番風險
            drop_score = 1.0 - bert_score

            logger.debug(
                f"Anime {anime_id}: BERT score={bert_score:.3f}, Drop score={drop_score:.3f}"
            )

            return float(drop_score)

        except Exception as e:
            logger.error(f"Error predicting BERT drop score: {e}")
            return 0.5  # 錯誤時返回中性值

    def predict_drop_probability(
        self, anime: Anime, user_id: int, session: Session
    ) -> Tuple[float, List[str]]:
        """
        預測棄番機率 (混合預測)

        Args:
            anime: 動畫物件
            user_id: 使用者 ID
            session: 資料庫 session

        Returns:
            (棄番機率, 預測原因列表)
        """
        # 初始化 BERT 推薦器 (如果需要)
        if self.use_bert and self.bert_recommender is None:
            self._initialize_bert_recommender(session)

        reasons = []

        # 1. XGBoost 預測
        xgboost_prob = 0.0
        xgboost_reasons = []
        if self.xgboost_engine.is_trained:
            xgboost_prob, xgboost_reasons = (
                self.xgboost_engine.predict_drop_probability(anime, user_id, session)
            )
            logger.debug(f"XGBoost prediction for anime {anime.id}: {xgboost_prob:.3f}")
        else:
            logger.warning("XGBoost model not trained, skipping XGBoost prediction")
            xgboost_reasons = ["XGBoost 模型未訓練"]

        # 2. BERT 預測
        bert_drop_score = 0.5  # 預設中性值
        if self.use_bert and self.bert_recommender:
            user_sequence = self._get_user_sequence(user_id, session)
            if user_sequence:
                bert_drop_score = self._predict_bert_drop_score(
                    anime.id, user_sequence, session
                )
                logger.debug(
                    f"BERT prediction for anime {anime.id}: {bert_drop_score:.3f}"
                )
            else:
                logger.warning("User sequence is empty, cannot use BERT prediction")
                reasons.append("⚠️ 觀看記錄不足，無法使用序列預測")
        else:
            logger.info("BERT prediction disabled or unavailable")

        # 3. 混合預測
        if self.use_bert and self.bert_recommender and bert_drop_score != 0.5:
            # BERT + XGBoost 混合
            final_probability = (
                bert_drop_score * self.bert_weight + xgboost_prob * self.xgboost_weight
            )
            reasons.append(
                f"🤖 混合預測: BERT {self.bert_weight * 100:.0f}% + XGBoost {self.xgboost_weight * 100:.0f}%"
            )
            reasons.append(
                f"📊 BERT 序列分數: {bert_drop_score:.1%} | XGBoost 特徵分數: {xgboost_prob:.1%}"
            )
        else:
            # 僅使用 XGBoost
            final_probability = xgboost_prob
            reasons.append("📊 僅使用 XGBoost 特徵預測 (BERT 不可用)")

        logger.info(
            f"Final drop probability for anime {anime.id}: {final_probability:.3f}"
        )

        # 添加風險等級說明
        if final_probability >= 0.7:
            reasons.append(f"🔴 高風險 {final_probability:.1%} - 強烈建議謹慎考慮")
        elif final_probability >= 0.5:
            reasons.append(f"🟡 中高風險 {final_probability:.1%} - 可能不太適合")
        elif final_probability >= 0.3:
            reasons.append(f"🟢 中低風險 {final_probability:.1%} - 可以嘗試")
        else:
            reasons.append(f"✅ 低風險 {final_probability:.1%} - 很適合你的口味")

        # 添加 XGBoost 的詳細原因
        if xgboost_reasons:
            reasons.append("\n📋 詳細分析:")
            reasons.extend(xgboost_reasons[1:])  # 跳過第一個總結性原因

        return float(final_probability), reasons

    def analyze_drop_patterns(
        self, ratings: List[UserRating], animes: List[Anime]
    ) -> Dict[str, Any]:
        """
        分析棄番模式 (委託給 XGBoost 引擎)

        Args:
            ratings: 使用者評分列表
            animes: 動畫列表

        Returns:
            棄番模式統計
        """
        return self.xgboost_engine.analyze_drop_patterns(ratings, animes)

    @property
    def is_trained(self) -> bool:
        """檢查模型是否已訓練"""
        return self.xgboost_engine.is_trained

    def get_model_info(self) -> Dict[str, Any]:
        """
        取得模型資訊

        Returns:
            模型狀態和配置資訊
        """
        return {
            "mode": "hybrid" if self.use_bert else "xgboost_only",
            "bert_enabled": self.use_bert,
            "bert_available": self.bert_recommender is not None,
            "bert_weight": self.bert_weight if self.use_bert else 0.0,
            "xgboost_weight": self.xgboost_weight if self.use_bert else 1.0,
            "xgboost_trained": self.xgboost_engine.is_trained,
        }
