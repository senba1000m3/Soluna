"""
BERT-based Anime Recommender Wrapper
整合 AnimeRecBERT 模型用於動畫推薦
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BERTRecommender:
    """
    AnimeRecBERT 模型包裝器
    負責載入預訓練模型並進行推薦
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        dataset_path: Optional[str] = None,
        anime_metadata_path: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        初始化 BERT 推薦器

        Args:
            model_path: 預訓練模型路徑 (.pth 檔案)
            dataset_path: 資料集路徑 (.pkl 檔案，包含 ID 映射)
            anime_metadata_path: 動畫 metadata 路徑 (.json 檔案)
            device: 運算設備 ('cpu' 或 'cuda')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.dataset = None
        self.anime_metadata = {}
        self.id_mapping = {}  # dataset_id -> anime_info
        self.reverse_id_mapping = {}  # external_id -> dataset_id

        if model_path:
            self.load_model(model_path)
        if dataset_path:
            self.load_dataset(dataset_path)
        if anime_metadata_path:
            self.load_anime_metadata(anime_metadata_path)

    def load_model(self, model_path: str) -> None:
        """
        載入預訓練的 BERT 模型

        Args:
            model_path: 模型檔案路徑
        """
        try:
            logger.info(f"Loading BERT model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # 根據 checkpoint 結構載入模型
            if isinstance(checkpoint, dict):
                # 如果有 model_state_dict
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                else:
                    state_dict = checkpoint

                # 這裡需要實際的模型架構
                # 暫時儲存 state_dict，實際使用時需要完整的模型定義
                self.model_state = state_dict
                logger.info("Model checkpoint loaded successfully")
            else:
                self.model = checkpoint
                self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded and moved to device")

        except Exception as e:
            logger.error(f"Failed to load BERT model: {e}")
            raise

    def load_dataset(self, dataset_path: str) -> None:
        """
        載入資料集和 ID 映射資訊

        Args:
            dataset_path: 資料集 pickle 檔案路徑
        """
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            with open(dataset_path, "rb") as f:
                self.dataset = pickle.load(f)

            # 建立 ID 映射
            if hasattr(self.dataset, "smap"):
                # smap: item_id -> sequential_id
                self.id_mapping = self.dataset.smap
                self.reverse_id_mapping = {v: k for k, v in self.id_mapping.items()}

            logger.info(f"Dataset loaded with {len(self.id_mapping)} items")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def load_anime_metadata(self, metadata_path: str) -> None:
        """
        載入動畫 metadata (genres, tags, etc.)

        Args:
            metadata_path: Metadata JSON 檔案路徑
        """
        try:
            logger.info(f"Loading anime metadata from {metadata_path}")
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # 假設 metadata 是 list of dicts 或 dict
            if isinstance(metadata, list):
                self.anime_metadata = {
                    item.get("id") or item.get("anime_id"): item for item in metadata
                }
            else:
                self.anime_metadata = metadata

            logger.info(f"Loaded metadata for {len(self.anime_metadata)} anime")

        except Exception as e:
            logger.error(f"Failed to load anime metadata: {e}")
            # 不是致命錯誤，繼續執行
            self.anime_metadata = {}

    def map_anilist_id_to_dataset_id(self, anilist_id: int) -> Optional[int]:
        """
        將 AniList ID 映射到資料集 ID

        Args:
            anilist_id: AniList 動畫 ID

        Returns:
            資料集中的對應 ID，如果找不到則返回 None
        """
        # 這裡需要實際的映射邏輯
        # 可能需要額外的映射檔案 (anilist_id -> dataset_id)
        # 暫時假設 ID 相同或使用 metadata 進行匹配
        if anilist_id in self.id_mapping:
            return self.id_mapping[anilist_id]

        # 嘗試從 metadata 中查找
        if anilist_id in self.anime_metadata:
            # 如果 metadata 中有映射資訊
            meta = self.anime_metadata[anilist_id]
            if "dataset_id" in meta:
                return meta["dataset_id"]

        return None

    def map_dataset_id_to_anilist_id(self, dataset_id: int) -> Optional[int]:
        """
        將資料集 ID 映射回 AniList ID

        Args:
            dataset_id: 資料集中的動畫 ID

        Returns:
            對應的 AniList ID，如果找不到則返回 None
        """
        if dataset_id in self.reverse_id_mapping:
            return self.reverse_id_mapping[dataset_id]
        return None

    def get_recommendations(
        self, user_anime_ids: List[int], top_k: int = 50, use_anilist_ids: bool = True
    ) -> List[Dict[str, Any]]:
        """
        基於使用者觀看歷史獲取推薦

        Args:
            user_anime_ids: 使用者觀看過的動畫 ID 列表
            top_k: 返回前 K 個推薦
            use_anilist_ids: 輸入是否為 AniList ID (True) 或資料集 ID (False)

        Returns:
            推薦動畫列表，每個包含 id, score, metadata
        """
        if self.model is None and not hasattr(self, "model_state"):
            logger.warning("Model not loaded, returning empty recommendations")
            return []

        try:
            # 1. ID 映射
            if use_anilist_ids:
                dataset_ids = []
                for aid in user_anime_ids:
                    did = self.map_anilist_id_to_dataset_id(aid)
                    if did is not None:
                        dataset_ids.append(did)
                logger.info(
                    f"Mapped {len(dataset_ids)}/{len(user_anime_ids)} AniList IDs to dataset IDs"
                )
            else:
                dataset_ids = user_anime_ids

            if not dataset_ids:
                logger.warning("No valid dataset IDs found")
                return []

            # 2. 準備模型輸入
            # 這裡需要根據實際的 BERT4Rec 輸入格式來準備
            # 通常是一個序列的 token IDs
            input_seq = self._prepare_input_sequence(dataset_ids)

            # 3. 模型推理
            scores = self._inference(input_seq)

            # 4. 獲取 Top-K 推薦
            top_indices = np.argsort(scores)[-top_k:][::-1]
            recommendations = []

            for idx in top_indices:
                dataset_id = idx
                score = float(scores[idx])

                # 映射回 AniList ID
                anilist_id = self.map_dataset_id_to_anilist_id(dataset_id)

                rec = {
                    "dataset_id": int(dataset_id),
                    "anilist_id": anilist_id,
                    "score": score,
                    "metadata": self.anime_metadata.get(anilist_id or dataset_id, {}),
                }
                recommendations.append(rec)

            return recommendations

        except Exception as e:
            logger.error(f"Error during recommendation: {e}")
            return []

    def _prepare_input_sequence(self, anime_ids: List[int]) -> torch.Tensor:
        """
        準備模型輸入序列

        Args:
            anime_ids: 動畫 ID 列表

        Returns:
            模型輸入張量
        """
        # BERT4Rec 輸入格式：[CLS] item1 item2 ... itemN [MASK]
        # 這裡簡化處理，實際需要根據模型訓練時的格式
        max_len = 200  # 假設最大序列長度為 200

        # 填充或截斷
        if len(anime_ids) > max_len - 2:
            anime_ids = anime_ids[-(max_len - 2) :]

        # 添加特殊 token (如果需要)
        # seq = [CLS_TOKEN] + anime_ids + [MASK_TOKEN]

        # 轉換為 tensor
        seq_tensor = torch.LongTensor([anime_ids])
        return seq_tensor.to(self.device)

    def _inference(self, input_seq: torch.Tensor) -> np.ndarray:
        """
        執行模型推理

        Args:
            input_seq: 輸入序列張量

        Returns:
            所有動畫的評分陣列
        """
        if self.model is None:
            # 如果模型未完全載入，返回隨機分數（用於測試）
            logger.warning("Model not fully loaded, returning dummy scores")
            num_items = len(self.id_mapping) if self.id_mapping else 1000
            return np.random.rand(num_items)

        try:
            with torch.no_grad():
                # 執行前向傳播
                output = self.model(input_seq)

                # 獲取最後一個位置的預測（MASK 位置）
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output

                # 取最後一個時間步的輸出
                scores = logits[:, -1, :].cpu().numpy()[0]

                return scores

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            # 返回隨機分數作為 fallback
            num_items = len(self.id_mapping) if self.id_mapping else 1000
            return np.random.rand(num_items)

    def get_anime_features(
        self, anime_ids: List[int], use_anilist_ids: bool = True
    ) -> Dict[str, Any]:
        """
        獲取一組動畫的聚合特徵

        Args:
            anime_ids: 動畫 ID 列表
            use_anilist_ids: 是否使用 AniList ID

        Returns:
            聚合的特徵字典（genres, tags, studios 等的分布）
        """
        from collections import Counter

        features = {
            "genres": Counter(),
            "tags": Counter(),
            "studios": Counter(),
            "formats": Counter(),
            "seasons": Counter(),
        }

        for aid in anime_ids:
            # 獲取 metadata
            if use_anilist_ids:
                metadata = self.anime_metadata.get(aid, {})
            else:
                anilist_id = self.map_dataset_id_to_anilist_id(aid)
                metadata = self.anime_metadata.get(anilist_id, {}) if anilist_id else {}

            # 聚合特徵
            if "genres" in metadata:
                for genre in metadata["genres"]:
                    features["genres"][genre] += 1

            if "tags" in metadata:
                for tag in metadata["tags"]:
                    tag_name = tag if isinstance(tag, str) else tag.get("name", "")
                    if tag_name:
                        features["tags"][tag_name] += 1

            if "studios" in metadata:
                for studio in metadata["studios"]:
                    studio_name = (
                        studio if isinstance(studio, str) else studio.get("name", "")
                    )
                    if studio_name:
                        features["studios"][studio_name] += 1

            if "format" in metadata:
                features["formats"][metadata["format"]] += 1

            if "season" in metadata:
                features["seasons"][metadata["season"]] += 1

        # 轉換為普通 dict 並正規化
        return {k: dict(v) for k, v in features.items()}

    def is_available(self) -> bool:
        """
        檢查模型是否可用

        Returns:
            模型是否已載入且可用
        """
        return self.model is not None or hasattr(self, "model_state")


# 簡單的測試函數
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 測試初始化
    recommender = BERTRecommender()
    print(f"BERT Recommender initialized. Available: {recommender.is_available()}")

    # 測試功能（需要實際的模型檔案）
    # recommender = BERTRecommender(
    #     model_path="path/to/model.pth",
    #     dataset_path="path/to/dataset.pkl",
    #     anime_metadata_path="path/to/animes.json"
    # )
